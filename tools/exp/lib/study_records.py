"""Immutable records and verified inputs for one finite study."""

import hashlib
import json
import os
from pathlib import Path
import time


def require(condition, message):
    """Keep scientific gates active under every Python optimization mode."""
    if not condition:
        raise ValueError(str(message))


def study_budget(registration):
    limit = registration.get("maximum_formal_runs")
    rows = registration.get("rows")
    require(
        type(limit) is int and limit > 0, "Formal budget must be a positive integer"
    )
    require(
        isinstance(rows, list) and len(rows) == limit,
        "Every authorized slot must have exactly one registered row",
    )
    require(
        registration.get("automatic_retries") == 0, "Automatic retries are forbidden"
    )
    identifiers = [row["id"] for row in rows]
    require(
        identifiers == ["E%02d" % i for i in range(1, limit + 1)],
        "Stage IDs must be ordered and unique",
    )
    for key in ("state", "output"):
        paths = [str(Path(row[key]).resolve()) for row in rows]
        require(len(set(paths)) == limit, "Stage " + key + " paths must be distinct")
    return limit


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    temporary = path.with_name(
        "." + path.name + "." + str(os.getpid()) + "." + str(time.time_ns())
    )
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def replace_status(path, data):
    """Replace only a bounded live status; scientific records still use save."""
    path = Path(path)
    require(path.name.endswith("-current.json"), "Only live status may be replaced")
    temporary = path.with_name("." + path.name + "." + str(os.getpid()))
    with temporary.open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def process(pid):
    directory = Path("/proc") / str(pid)
    if directory.stat().st_uid != os.getuid():
        raise PermissionError("Process is not owned by this account")
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    status = dict(
        line.split(":", 1)
        for line in (directory / "status").read_text().splitlines()
        if ":" in line
    )
    return dict(
        pid=pid,
        start=fields[19],
        state=fields[0],
        cpu_ticks=int(fields[11]) + int(fields[12]),
        rss_bytes=int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
        peak_rss_bytes=int(status.get("VmHWM", "0 kB").split()[0]) * 1024,
    )


def alive(identity):
    try:
        current = process(identity["pid"])
        return current["start"] == identity["start"] and current["state"] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def verify_contract(root, expected):
    root = Path(root)
    require(
        digest(root / "helper-contract.json") == expected,
        "Study check failed: digest(root / 'helper-contract.json') == expected",
    )
    contract = load(root / "helper-contract.json")
    for name, sha in contract["files"].items():
        require(digest(root / name) == sha, name)
    registration = load(root / "registration.json")
    study_budget(registration)
    return registration


def verify_files(files):
    for path, sha in files.items():
        require(digest(path) == sha, path)
