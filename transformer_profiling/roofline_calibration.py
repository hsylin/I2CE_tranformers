"""Portable, pattern-specific cache references from independent measurements."""

import hashlib
import json
import configparser
import math
import re
import statistics


QUALITY = {
    "last_two_pass_delta_max": 0.05,
    "repeat_range_fraction_max": 0.05,
    "median_empty_overhead_max": 0.01,
}


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


def validate_identity(record, config, config_ini, frequency, revision):
    """Validate both complete hardware representations before accepting rates."""
    if record["schema_version"] not in (1, 2) or record["gem5_revision"] != revision:
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


def stable_rate(sample, frequency, quality, source_hashes):
    """Recompute the two-pass / three-repeat gates from exact saved counters."""
    if quality != QUALITY:
        raise ValueError("Calibration stability thresholds differ")
    passes = sample["passes"]
    if (
        len(passes) != 2
        or any(type(p) is not int or p <= 0 for p in passes)
        or passes[0] >= passes[1]
    ):
        raise ValueError("Calibration needs two increasing pass counts")
    rows = sample["samples"]
    if sorted((r["passes"], r["experiment"]) for r in rows) != [
        (p, ex) for p in passes for ex in range(3)
    ]:
        raise ValueError("Calibration needs exactly three repeats at each pass count")
    rates, overheads = {p: [] for p in passes}, []
    for row in rows:
        for key in ("sim_ticks", "count", "empty_ticks"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("Calibration counters must be nonnegative integers")
        if row["sim_ticks"] <= 0 or row["count"] <= 0 or row["empty_ticks"] <= 0:
            raise ValueError("Calibration counters must be positive")
        raw = "runs/exp%d/stats.txt" % row["experiment"]
        if row["stats_sha256"] != source_hashes[raw]:
            raise ValueError("Calibration sample is not bound to its raw statistics")
        rates[row["passes"]].append(row["count"] * frequency / row["sim_ticks"])
        if row["passes"] == passes[-1]:
            overheads.append(row["empty_ticks"] / row["sim_ticks"])
    a, z = (statistics.median(rates[p]) for p in passes)
    delta = abs(z - a) / max(z, a)
    spread = (max(rates[passes[-1]]) - min(rates[passes[-1]])) / z
    if (
        delta > quality["last_two_pass_delta_max"]
        or spread > quality["repeat_range_fraction_max"]
        or statistics.median(overheads) > quality["median_empty_overhead_max"]
    ):
        raise ValueError("Calibration selected an unstable condition")
    if not math.isclose(sample["rate_per_second"], z, rel_tol=1e-12):
        raise ValueError("Calibration counters do not reproduce its median rate")
    return z


def scan_references(record, config, config_ini, frequency, revision):
    """One empirical compute reference and three boundary-specific slopes."""
    validate_identity(record, config, config_ini, frequency, revision)
    if (
        record["schema_version"] != 2
        or record["pattern"] != "streaming-int8-sve128-sdot"
    ):
        raise ValueError("Unsupported scan calibration pattern")
    quality, hashes = record["quality"], record["source_hashes"]
    compute = stable_rate(record["compute"], frequency, quality, hashes)
    memory = []
    for level in ("Core–L1D", "L1D–L2", "Physical DRAM"):
        sample = record["references"][level]
        if sample["direction"] != "total" or sample["source_type"] != "measured":
            raise ValueError("Calibration must use measured combined traffic")
        rate = stable_rate(sample, frequency, quality, hashes)
        memory.append(
            dict(
                level=level,
                direction="total",
                bytes_per_second=rate,
                ridge_ops_per_byte=compute / rate,
                source_type="measured",
                formula="Median of three native (read + write bytes) / gem5 ROI seconds",
                config_fields={"hardware_sha256": record["hardware_sha256"]},
                calibration_id=record["id"],
                sample=sample,
                source_hashes=hashes,
                qualification="Best stable SDOT scan observation at this boundary; pattern-specific, not a universal maximum.",
            )
        )
    return dict(
        compute_ops_per_second=compute,
        memory=memory,
        compute_source_type="measured",
        compute_sample=record["compute"],
        compute_formula="Median of three integer operation counts / gem5 ROI seconds",
        compute_fields={"hardware_sha256": record["hardware_sha256"]},
        calibration_id=record["id"],
        calibration_source_hashes=hashes,
        qualification="The workload may exceed a pattern-specific reference; this is not an absolute hardware ceiling.",
    )


def references(record, config, config_ini, frequency, revision, compute_peak):
    """Reject incompatible calibration; never estimate a replacement cache roof."""
    validate_identity(record, config, config_ini, frequency, revision)
    if record["schema_version"] != 1:
        raise ValueError("Use scan_references for a complete scan calibration")
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
