"""Strict records for the single-core SDOT measurement workflow."""

import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def freeze_check(root):
    frozen = load(root / "freeze.json")
    for name, expected in frozen["files"].items():
        path = root / name
        require(
            not path.is_symlink() and digest(path) == expected,
            "Frozen input changed: " + name,
        )
    for name, mode in frozen["modes"].items():
        require(
            (root / name).stat().st_mode & 0o7777 == mode,
            "Executable mode changed: " + name,
        )
    return frozen
