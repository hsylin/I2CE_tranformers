#!/usr/bin/env python3
"""Compare saved output bytes from two separately compiled API probes.

This is a functional check under QEMU. It never runs gem5, changes production
sources, or reuses formal measurement slots. Every output directory is new.
"""
import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
        return value.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.registration.read_text())
    require(plan["formal_measurements"] == 0, "Performance runs are forbidden")
    require(plan["maximum_builds"] == plan["maximum_executions"] == 2, "Unbounded plan")
    require(digest(Path(__file__)) == plan["helper_sha256"], "Helper identity differs")
    probe = Path(plan["probe"])
    require(digest(probe) == plan["probe_sha256"], "Probe identity differs")
    args.output.mkdir(parents=True, exist_ok=False)
    commands = []
    save(
        args.output / "start.json",
        {
            "registration_sha256": digest(args.registration),
            "started_at": time.time(),
            "formal_measurements": 0,
        },
    )

    def run(name, command, cwd):
        record = {"name": name, "command": list(map(str, command)), "cwd": str(cwd)}
        commands.append(record)
        with (args.output / (name + ".log")).open("x") as log:
            result = subprocess.run(
                record["command"],
                cwd=cwd,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        record["returncode"] = result.returncode
        require(result.returncode == 0, name + " failed; inspect preserved log")

    try:
        results = {}
        for name in ("original", "corrected"):
            binding = plan[name]
            source = Path(binding["source"])
            contract_path = Path(binding["contract"])
            require(
                digest(contract_path) == binding["contract_sha256"], "Contract differs"
            )
            contract = json.loads(contract_path.read_text())
            for relative, expected in contract["source_files"].items():
                require(
                    digest(source / relative) == expected, "Source differs: " + relative
                )
            compiler = contract["tools"]["compiler"]
            qemu = contract["tools"]["qemu"]
            for tool in (compiler, qemu):
                require(digest(Path(tool["path"])) == tool["sha256"], "Tool differs")
            executable = args.output / (name + "-probe")
            command = [
                compiler["path"],
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-march=armv8-a+sve",
                "-DSIMD",
                "-static",
                "-ffunction-sections",
                "-fdata-sections",
                "-Wl,--gc-sections",
            ]
            command.extend(
                "-I" + str(source / part)
                for part in (
                    "",
                    "Full_NN/inc",
                    "Full_NN/src",
                    "Full_NN/gemm_definitions",
                    "transformer_layers",
                )
            )
            command.extend(
                [probe, source / "Full_NN/src/gemm_exec.c", "-o", executable]
            )
            run(name + "-build", command, args.output)
            output = args.output / (name + "-outputs.bin")
            run(
                name + "-run",
                [
                    qemu["path"],
                    "-cpu",
                    "max,sve=on,sve-default-vector-length=16",
                    executable,
                    output,
                ],
                args.output,
            )
            results[name] = {
                "probe_binary_sha256": digest(executable),
                "output_sha256": digest(output),
                "output_bytes": output.stat().st_size,
                "source_revision": contract["source_revision"],
            }
            for relative, expected in contract["source_files"].items():
                require(
                    digest(source / relative) == expected, "Source changed: " + relative
                )
        require(
            results["original"]["output_sha256"]
            == results["corrected"]["output_sha256"],
            "Original and corrected API output bytes differ",
        )
        save(
            args.output / "complete.json",
            {
                "passed": True,
                "results": results,
                "formal_measurements": 0,
                "production_binary_executed": False,
            },
        )
    except Exception:
        save(
            args.output / "failure.json",
            {"error": traceback.format_exc(), "automatic_retry": False},
        )
        raise
    finally:
        save(args.output / "commands.json", commands)


if __name__ == "__main__":
    main()
