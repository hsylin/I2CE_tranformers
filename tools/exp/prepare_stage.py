#!/usr/bin/env python3
"""Prepare an isolated E01-E13 source tree and optionally reproduce its binary.

Canonical inputs are overlaid only in the new output directory. This command
never executes the resulting binary or launches a simulator.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def relative_path(name):
    path = Path(name)
    require(
        bool(name) and not path.is_absolute() and ".." not in path.parts,
        "Expected a repository-relative path: " + name,
    )
    return path


def git_bytes(revision, name):
    relative_path(name)
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "show", revision + ":" + name],
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    )


def checked_bytes(revision, name, expected):
    data = git_bytes(revision, name)
    require(sha256(data) == expected, "Source hash mismatch: " + name)
    return data


def save(path, record):
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


def prepare(stage, revision, output):
    require(stage in {f"E{i:02}" for i in range(1, 14)}, "Unknown stage")
    commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "--verify", revision + "^{commit}"],
        text=True,
    ).strip()
    prefix = "transformer_profiling/experiments/" + stage + "/"
    manifest = json.loads(git_bytes(commit, prefix + "source.json"))
    require(manifest["stage"] == stage, "Stage manifest mismatch")
    require(not output.exists(), "Output already exists; choose a new directory")
    output.mkdir(parents=True)
    source = output / "source"
    source.mkdir()
    for name, expected in manifest["source_files"].items():
        target = source / relative_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(checked_bytes(commit, name, expected))
    inputs = json.loads(git_bytes(commit, "transformer_profiling/inputs/manifest.json"))
    for name, expected in inputs["files"].items():
        relative_path(name)
        data = checked_bytes(
            commit, "transformer_profiling/inputs/workload/" + name, expected
        )
        if name.startswith("headers/"):
            target = source / "Full_NN/gemm_definitions" / Path(name).name
        elif name.startswith("weights/"):
            target = source / name
        else:
            raise RuntimeError("Unknown scientific input category: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record = {
        "stage": stage,
        "public_commit": commit,
        "source_files": manifest["source_files"],
        "scientific_inputs": inputs["files"],
        "reference_binary_sha256": manifest["binary_sha256"],
        "binary_built": False,
        "binary_executed": False,
        "formal_measurements": 0,
    }
    save(output / "prepared.json", record)
    return source, manifest, record


def build(source, manifest, record, output):
    environment = dict(os.environ, **manifest["build_environment"])
    requirements = {
        "compiler": "A64CXX",
        "libm5": "LIBM5_A",
        "m5_header": "M5_INC",
    }
    for tool, variable in requirements.items():
        require(bool(os.environ.get(variable)), "Set " + variable)
        environment[variable] = os.environ[variable]
        path = Path(environment[variable])
        if tool == "m5_header":
            path /= "gem5/m5ops.h"
        require(
            path.is_file()
            and sha256(path.read_bytes()) == manifest["tool_hashes"][tool],
            "Registered tool identity mismatch: " + variable,
        )
    with (output / "build.log").open("x") as log:
        result = subprocess.run(
            ["bash", "compile_transformer.sh"],
            cwd=source,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=900,
        )
    record = dict(record, binary_built=result.returncode == 0)
    record["returncode"] = result.returncode
    if result.returncode == 0:
        record["binary_sha256"] = sha256((source / "transformer.o").read_bytes())
        record["matches_reference"] = (
            record["binary_sha256"] == manifest["binary_sha256"]
        )
    save(output / "build-result.json", record)
    require(result.returncode == 0, "Build failed; inspect the preserved build.log")
    require(record["matches_reference"], "Binary differs from the accepted reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--revision", help="Defaults to the stage tag")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    source, manifest, record = prepare(
        args.stage, args.revision or args.stage, args.output.resolve()
    )
    if args.build:
        build(source, manifest, record, args.output.resolve())
    print(json.dumps({"stage": args.stage, "output": str(args.output)}))


if __name__ == "__main__":
    main()
