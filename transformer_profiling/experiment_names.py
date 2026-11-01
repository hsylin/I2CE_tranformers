"""Canonical report IDs for the published experiment sequence."""

import hashlib
import json
from pathlib import Path
import re


SEQUENCE = json.loads(Path(__file__).with_name("sequence.json").read_text())


def profile_names(frame, path):
    """Normalize the registered profiling input before labels are constructed."""
    identities = {
        (stage.get("source_commit"), stage.get("binary_sha256")): stage["id"]
        for stage in SEQUENCE["experiments"]
        if stage.get("source_commit") and stage.get("binary_sha256")
    }
    if {"repo_commit", "binary_sha256"}.issubset(frame.columns):
        names = [
            identities.get((source, binary), old)
            for source, binary, old in zip(
                frame.repo_commit, frame.binary_sha256, frame.exp_id
            )
        ]
        if names != frame.exp_id.tolist():
            result = frame.copy()
            result["exp_id"] = names
            result["implementation"] = [
                re.sub(r"\b" + re.escape(old) + r"\b", new, text)
                for old, new, text in zip(frame.exp_id, names, frame.implementation)
            ]
            return result
    path = Path(path)
    if (
        not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != SEQUENCE["profiling_sha256"]
    ):
        return frame
    ids = sorted(frame.exp_id.unique(), key=lambda value: int(value[1:]))
    stages = SEQUENCE["experiments"]
    if len(ids) != len(stages):
        raise ValueError("Profiling input does not contain every registered stage")
    names = dict(zip(ids, (stage["id"] for stage in stages)))
    result = frame.copy()
    result["exp_id"] = frame.exp_id.map(names)
    result["implementation"] = frame.implementation.map(
        lambda text: re.sub(
            r"\bE\d+\b", lambda match: names.get(match[0], match[0]), text
        )
    )
    return result


def run_names(run):
    """Identify a stage by its measured source and binary, independently of paths."""
    for stage in SEQUENCE["experiments"]:
        if all(
            run.get(key) == stage[key] for key in ("source_commit", "binary_sha256")
        ):
            description = re.sub(r"^E\d+\b\s*", "", run.get("label", ""))
            label = stage["id"] + (" " + description if description else "")
            return stage["id"], label
    return run["id"], run.get("label", run["id"])
