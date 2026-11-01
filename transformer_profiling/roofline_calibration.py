"""Portable, pattern-specific cache references from independent measurements."""

import hashlib
import json
import configparser
import math
import re


HARDWARE_COMPONENTS = (
    "cpu_cluster",
    "mem_ctrls",
    "roofline_l1",
    "roofline_l2",
    "clk_domain",
    "membus",
    "voltage_domain",
)
SYSTEM_FIELDS = ("cache_line_size", "sve_vl", "sme_vl", "mem_mode", "mem_ranges")


def hardware_identity(config, frequency):
    """Include complete CPU/cache/monitor/memory objects, excluding host I/O paths."""
    system = config["system"]
    return dict(
        sim_freq=frequency,
        system={key: system[key] for key in HARDWARE_COMPONENTS + SYSTEM_FIELDS},
    )


def identity_hash(identity):
    data = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def ini_identity(path):
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    with open(path) as stream:
        parser.read_file(stream)
    prefixes = tuple("system." + key for key in HARDWARE_COMPONENTS)
    result = {
        name: dict(parser[name])
        for name in parser.sections()
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    }
    result["system"] = {key: parser["system"][key] for key in SYSTEM_FIELDS}
    return result


def references(record, config, config_ini, frequency, revision, compute_peak):
    """Reject incompatible calibration; never estimate a replacement cache roof."""
    if record["schema_version"] != 1 or record["gem5_revision"] != revision:
        raise ValueError("Unsupported calibration schema or gem5 revision")
    identity = hardware_identity(config, frequency)
    if identity_hash(identity) != record["hardware_sha256"]:
        raise ValueError("Calibration hardware/boundary mismatch")
    if record["hardware"] != identity:
        raise ValueError("Calibration hardware record disagrees with its hash")
    if record["hardware_ini"] != ini_identity(config_ini):
        raise ValueError("Calibration INI hardware/boundary mismatch")
    if not record["source_hashes"] or not all(
        re.fullmatch(r"[a-f0-9]{64}", sha) for sha in record["source_hashes"].values()
    ):
        raise ValueError("Calibration source hashes are incomplete")
    result = []
    for level in ("Core–L1D", "L1D–L2"):
        sample = record["references"][level]
        rate = sample["bytes_per_second"]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError("Calibration rate must be finite and positive")
        if not all(
            type(sample[k]) is int and sample[k] >= 0
            for k in ("sim_ticks", "read_bytes", "write_bytes")
        ):
            raise ValueError("Calibration counters must be nonnegative integers")
        elapsed = sample["sim_ticks"] / frequency
        count = sample["read_bytes"] + sample["write_bytes"]
        if not elapsed > 0 or not count > 0 or abs(rate - count / elapsed) > 0.01:
            raise ValueError("Calibration bytes/time do not reproduce its rate")
        if sample["direction"] != "total" or sample["source_type"] != "measured":
            raise ValueError("Calibration must use measured combined traffic")
        result.append(
            dict(
                level=level,
                direction="total",
                bytes_per_second=rate,
                ridge_ops_per_byte=compute_peak / rate,
                source_type="measured",
                formula="Native (read bytes + write bytes) / simulated ROI seconds",
                config_fields={"hardware_sha256": record["hardware_sha256"]},
                calibration_id=record["id"],
                sample=sample,
                source_hashes=record["source_hashes"],
                qualification=sample["qualification"],
            )
        )
    return result
