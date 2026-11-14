"""One strict parameter contract shared by CLI, TSV and frozen run records."""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

AXES = ("s1", "o1", "k1", "s2", "o2", "k2")
COLUMNS = ("id", "sha", *AXES, "learners", "codebook_size", "l1d_size", "l2_size")
FIXED_COMMIT = "fea8ec97db457c1a126c4bac67cf4064d6f0ff60"
FIXED_TILES = dict(zip(AXES, (16, 32, 256, 128, 128, 512)))
HARDWARE = {
    "cpu": "minor",
    "cores": 1,
    "sve_bits": 128,
    "cpu_frequency": "4GHz",
    "memory_bytes": 2 * 1024**3,
    "memory_type": "DDR3_1600_8x8",
    "memory_channels": 1,
    "l1i_bytes": 48 * 1024,
}


def require(condition, message):
    """Raise a user-facing contract error without implicit fallback."""
    if not condition:
        raise ValueError(message)


def digest(path):
    """Hash a file incrementally without loading large inputs into memory."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save_new(path, record):
    """Write a receipt once; never replace an existing run record."""
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def capacity(value):
    """Parse explicit capacity units and enforce cache-line alignment."""
    match = re.fullmatch(r"([1-9][0-9]*)(B|KiB|MiB|GiB|kB|MB|GB)?", str(value))
    require(match, "Capacity must be positive bytes, KiB/MiB/GiB or kB/MB/GB")
    units = {
        None: 1,
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "kB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }
    result = int(match[1]) * units[match[2]]
    require(
        result <= 2**40 and result % 64 == 0,
        "Cache capacity must be 64-byte aligned and <=1 TiB",
    )
    return result


def positive(value):
    """Parse bounded positive decimal tile and cardinality parameters."""
    require(
        re.fullmatch(r"[1-9][0-9]*", str(value)), "Expected a positive decimal integer"
    )
    result = int(value)
    require(result <= 65535, "Parameter exceeds 65535")
    return result


def normalize(row):
    """Apply the same strict field and type contract to CLI and TSV rows."""
    require(set(row) == set(COLUMNS), "Fields must be exactly: " + ", ".join(COLUMNS))
    require(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", row["id"]),
        "Invalid experiment ID",
    )
    require(
        row["sha"] and not row["sha"].startswith("-"), "Missing or invalid source ref"
    )
    result = dict(row)
    for key in (*AXES, "learners", "codebook_size"):
        result[key] = positive(row[key])
    for key in ("l1d_size", "l2_size"):
        result[key] = capacity(row[key])
    return result


def read_manifest(path):
    """Read a complete TSV batch and reject duplicate IDs or headers."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(
            reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
            "Duplicate or missing TSV header",
        )
        rows = [normalize(row) for row in reader]
    require(
        rows and len({row["id"] for row in rows}) == len(rows),
        "Empty manifest or duplicate IDs",
    )
    return rows


def git(repo, *args):
    """Read a Git object; failures must never fall back to the checkout HEAD."""
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.PIPE
    )


def source_capabilities(repo, ref):
    """Resolve a full commit and read its declared kernel capability."""
    try:
        sha = git(repo, "rev-parse", "--verify", ref + "^{commit}").decode().strip()
    except subprocess.CalledProcessError:
        raise ValueError(
            "Source is missing. Import the local Git bundle with git fetch <bundle> <branch>; "
            "then verify git rev-parse <sha>^{commit}. No HEAD fallback."
        ) from None
    try:
        metadata = git(repo, "show", sha + ":Full_NN/runtime_tiles.json")
    except subprocess.CalledProcessError:
        require(sha == FIXED_COMMIT, "Unknown kernel capabilities at " + sha)
        # This exact revision has enum constants, not configurable tile macros.
        # Its only accepted compile-time geometry is the verified default.
        return sha, {
            "backend": "compile-fixed",
            "learners": [2],
            "codebook_entries": [4],
            "fixed_tiles": FIXED_TILES,
            "evidence_commit": FIXED_COMMIT,
        }
    capability = json.loads(metadata)
    require(
        capability.get("schema") == "i2ce.runtime-tiles.v1"
        and capability.get("backend") == "runtime",
        "Unknown runtime capability schema",
    )
    capability["metadata_sha256"] = hashlib.sha256(metadata).hexdigest()
    return sha, capability


def validate_backend(row, capability):
    """Reject dimensions the resolved source does not implement."""
    require(
        row["learners"] in capability["learners"]
        and row["codebook_size"] in capability["codebook_entries"],
        "Unsupported learner/codebook combination",
    )
    if capability["backend"] == "compile-fixed":
        require(
            all(row[key] == value for key, value in capability["fixed_tiles"].items()),
            "This compile-time kernel supports only 16/32/256/128/128/512; no tile override flags exist",
        )
        return
    for name, rule in capability["tiles"].items():
        value = row[name]
        require(
            rule.get("min", 1) <= value <= rule.get("max", 65535),
            "Out-of-range " + name,
        )
        require("values" not in rule or value in rule["values"], "Unsupported " + name)
        require(
            not rule.get("power_of_two") or value & (value - 1) == 0,
            name + " must be a power of two",
        )
        require(
            "multiple_of" not in rule or value % row[rule["multiple_of"]] == 0,
            "Invalid nesting for " + name,
        )
    require(row["k1"] <= row["k2"], "K1 exceeds K2")
    require(
        min(row["k1"], row["k2"]) >= 128 or row["k1"] == row["k2"],
        "Compact depths require K1=K2",
    )


def resolve(repo, row):
    """Bind normalized parameters, capability and invocation to an exact source."""
    sha, capability = source_capabilities(repo, row["sha"])
    validate_backend(row, capability)
    runtime = [item for key in AXES for item in ("--" + key, str(row[key]))]
    return {
        "schema": "i2ce.experiment.v1",
        "parameters": dict(row, sha=sha),
        "requested_ref": row["sha"],
        "capability": capability,
        "hardware": HARDWARE,
        "runtime_argv": runtime if capability["backend"] == "runtime" else [],
        "performance_status": "unmeasured",
        "policy": {"L1D.writeback_clean": False},
    }


def reject_policy_overrides(environment):
    """Keep writeback_clean=False invariant across configuration aliases."""
    for name in environment:
        compact = re.sub("[^a-z]", "", name.lower())
        require(
            "writebackclean" not in compact and "cleanwriteback" not in compact,
            "Cache writeback policy is fixed; remove override " + name,
        )
