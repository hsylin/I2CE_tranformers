#!/usr/bin/env python3
"""Build one registered stage and compare its supported behavior with S25.

All outputs go to a new directory. This runner never launches gem5 or reuses
formal slots. QEMU checks use explicit validation copies, not the production
binary, and their instrumentation is recorded separately.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import sys


def digest(path):
    with path.open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hashes(root):
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Commands:
    def __init__(self, output, environment):
        self.directory = output / "logs"
        self.directory.mkdir()
        self.environment = environment
        self.records = []

    def run(
        self,
        name,
        command,
        cwd,
        environment=None,
        stdin=None,
        timeout=900,
        expected_returncode=0,
    ):
        record = {
            "name": name,
            "command": [str(part) for part in command],
            "cwd": str(cwd),
            "started_at": time.time(),
            "expected_returncode": expected_returncode,
        }
        self.records.append(record)
        try:
            with (self.directory / (name + ".log")).open("x") as log:
                result = subprocess.run(
                    record["command"],
                    cwd=cwd,
                    env=self.environment if environment is None else environment,
                    stdin=subprocess.DEVNULL if stdin is None else stdin,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            record["returncode"] = result.returncode
            require(
                result.returncode == expected_returncode,
                name + " failed; inspect its saved log",
            )
        except subprocess.TimeoutExpired:
            record["timed_out_after_seconds"] = timeout
            raise
        except OSError as error:
            record["launch_error"] = str(error)
            raise
        finally:
            record["finished_at"] = time.time()
        return (self.directory / (name + ".log")).read_text(errors="replace")


def replace_once(path, original, replacement):
    text = path.read_text()
    require(text.count(original) == 1, "Ambiguous validation anchor: " + str(path))
    path.write_text(text.replace(original, replacement))


def instrument_copy(source, weights, compressed_indices):
    main_source = source / "transformer.cpp"
    main_source.write_text("#include <cstdio>\n" + main_source.read_text())
    input_anchor = 'std::string dir_name = "/home/thu/TiC-SAT/weights";'
    replace_once(
        source / "transformer.cpp",
        input_anchor,
        "std::string dir_name = " + json.dumps(str(weights)) + ";",
    )
    output_anchor = (
        "        TransformerBlock::computeGroup2(D_SEQ, grouped_blocks, "
        "grouped_inputs, grouped_outputs);"
    )
    capture = """
        for (std::size_t learner = 0; learner < 2; ++learner) {
            const std::string name = "final-learner" + std::to_string(learner) + ".bin";
            FILE *file = std::fopen(name.c_str(), "wb");
            if (!file || std::fwrite(grouped_outputs[learner], 1, D_SEQ * D_MODEL, file)
                         != D_SEQ * D_MODEL) std::abort();
            if (std::fclose(file) != 0) std::abort();
        }
"""
    replace_once(source / "transformer.cpp", output_anchor, output_anchor + capture)
    if compressed_indices:
        replace_once(
            source / "transformer_layers/layerFactory.cc",
            '"/home/thu/TiC-SAT/weights/compressed_indices"',
            json.dumps(str(weights / "compressed_indices")),
        )


def functional_m5(commands, work, tools):
    source = work / "m5-functional.cc"
    source.write_text(
        "#include <cstdint>\n"
        'extern "C" void m5_dump_stats(uint64_t, uint64_t) {}\n'
        'extern "C" void m5_reset_stats(uint64_t, uint64_t) {}\n'
        'extern "C" void m5_dump_reset_stats(uint64_t, uint64_t) {}\n'
        'extern "C" uint64_t m5_rpns() { static uint64_t tick = 0; return ++tick; }\n'
    )
    object_file = work / "m5-functional.o"
    library = work / "libm5-functional.a"
    commands.run(
        "m5-compile", [tools["compiler"]["path"], "-c", source, "-o", object_file], work
    )
    commands.run(
        "m5-archive", [tools["archiver"]["path"], "rcs", library, object_file], work
    )
    return library


def build_production(commands, source, tools):
    commands.run("production-build", ["bash", "compile_transformer.sh"], source)
    binary = source / "transformer.o"
    commands.run(
        "production-disassembly", [tools["objdump"]["path"], "-d", "-C", binary], source
    )
    return binary


def check_contract(contract):
    """Reject an incomplete or nonfunctional registration before any build."""
    require(contract["helper_sha256"] == digest(Path(__file__)), "Unbound helper")
    require(
        contract["experiment_id"] in {f"E{i:02d}" for i in range(1, 14)},
        "Unknown reconstruction stage",
    )
    require(contract["formal_measurements"] == 0, "Formal measurements are forbidden")
    require(contract["automatic_retries"] == 0, "Automatic retries are forbidden")
    require(contract["maximum_production_builds"] == 1, "Register one production build")
    require(
        contract["maximum_validation_model_builds"] == 2,
        "Register both functional copies",
    )
    require(
        contract["maximum_full_model_executions"] == 2,
        "Register both functional executions",
    )
    require(
        contract["rejected_sve_lengths_bytes"] == [32, 64]
        and contract["maximum_startup_rejection_checks"] == 2,
        "Register two incompatible SVE startup checks",
    )
    oracles = contract["oracles"]
    require(
        len({item["name"] for item in oracles}) == len(oracles),
        "Duplicate oracle names",
    )
    require(contract["maximum_oracle_builds"] == len(oracles), "Oracle budget mismatch")
    require(
        contract["maximum_oracle_executions"] == len(oracles),
        "Oracle execution budget mismatch",
    )
    for name in [
        *contract["source_files"],
        *contract["scientific_inputs"],
        *contract["test_files"],
        *contract["additional_inputs"],
    ]:
        require(
            not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Contract paths must stay within their input roots: " + name,
        )
    for item in oracles:
        require(item["source"] in contract["test_files"], "Unbound oracle source")
    for item in contract["native_checks"]:
        require(item["source"] in contract["test_files"], "Unbound native test source")


def validate(args, commands, contract):
    output = args.output
    tools = contract["tools"]
    require(
        shutil.disk_usage(output).free >= 8 * 1024**3,
        "At least 8 GiB of free build space is required",
    )
    memory_info = Path("/proc/meminfo")
    if memory_info.is_file():
        available = next(
            int(line.split()[1]) * 1024
            for line in memory_info.read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        require(
            available >= 4 * 1024**3, "At least 4 GiB of available memory is required"
        )
    for tool in tools.values():
        require(
            digest(Path(tool["path"])) == tool["sha256"],
            "Tool changed: " + tool["path"],
        )
    require(
        digest(args.reference_build / "production-source/transformer.o")
        == contract["original_binary_sha256"],
        "Reference binary mismatch",
    )
    for name, expected in contract["scientific_inputs"].items():
        require(digest(args.inputs / name) == expected, "Input changed: " + name)
    for item in contract["additional_inputs"].values():
        require(digest(Path(item["source"])) == item["sha256"], "CI01 input changed")

    source = output / "production-source"
    source.mkdir()
    for name, expected in contract["source_files"].items():
        require(digest(args.candidate / name) == expected, "Candidate changed: " + name)
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.candidate / name, destination)
    for name, expected in contract["scientific_inputs"].items():
        if name.startswith("headers/"):
            destination = source / "Full_NN/gemm_definitions" / Path(name).name
            shutil.copyfile(args.inputs / name, destination)
            require(digest(destination) == expected, "Header copy mismatch")
    effective = hashes(source)
    save(output / "effective-sources.json", effective)
    differences = {}
    for name, value in effective.items():
        reference = args.reference_build / "production-source" / name
        reference_hash = digest(reference) if reference.is_file() else None
        if reference_hash != value:
            differences[name] = {"reference": reference_hash, "candidate": value}
    save(output / "source-differences.json", differences)

    if args.production_reference is not None:
        previous = args.production_reference
        prior_result = json.loads((previous / "complete.json").read_text())
        require(prior_result["passed"], "The reference validation did not pass")
        previous_start = json.loads((previous / "start.json").read_text())
        require(
            previous_start["contract_sha256"] == digest(args.contract),
            "The tested candidate used a different build/input contract",
        )
        expected = prior_result["production_binary_sha256"]
        require(
            digest(previous / "production-source/transformer.o") == expected,
            "The previously tested candidate binary changed",
        )
        binary = build_production(commands, source, tools)
        require(digest(binary) == expected, "The rebuilt production binary differs")
        return {
            "passed": True,
            "mode": "production-rebuild-comparison",
            "production_binary_sha256": expected,
            "matches_tested_candidate": True,
            "validation_reference": str(previous),
            "validation_receipt_sha256": digest(previous / "complete.json"),
            "formal_runs": 0,
            "new_functional_executions": 0,
            "limitations": [
                "Existing functional checks are reused, not rerun",
                "Binary comparison covers this production configuration only",
            ],
        }

    compiler = tools["compiler"]["path"]
    qemu = [tools["qemu"]["path"], "-cpu", "max,sve=on,sve-default-vector-length=16"]
    for name, expected in contract["test_files"].items():
        require(
            digest(args.candidate / "tests" / name) == expected,
            "Independent oracle source changed: " + name,
        )
    if args.reuse_oracles is not None:
        previous = args.reuse_oracles
        require(
            json.loads((previous / "effective-sources.json").read_text()) == effective,
            "Completed oracle sources do not match this candidate",
        )
        previous_start = json.loads((previous / "start.json").read_text())
        require(
            previous_start["contract_sha256"] == digest(args.contract),
            "Completed oracle contract differs",
        )
        oracle_results = json.loads((previous / "oracles.json").read_text())
        require(
            set(oracle_results) == {item["name"] for item in contract["oracles"]},
            "Completed oracle set is incomplete",
        )
        for name, result in oracle_results.items():
            require(
                result["passed"]
                and digest(previous / (name + "-test")) == result["binary_sha256"],
                "Completed oracle binary changed: " + name,
            )
        save(
            output / "reused-oracle-binding.json",
            {
                "directory": str(previous),
                "oracles_sha256": digest(previous / "oracles.json"),
                "effective_sources_sha256": digest(previous / "effective-sources.json"),
                "note": "Previously completed checks, not new executions",
            },
        )
    else:
        oracle_results = {}
        for oracle in contract["oracles"]:
            name = oracle["name"]
            executable = output / (name + "-test")
            test_source = args.candidate / "tests" / oracle["source"]
            require(
                digest(test_source) == contract["test_files"][oracle["source"]],
                "Independent oracle changed: " + name,
            )
            command = [
                compiler,
                "-std=c++17",
                "-O3",
                "-Wall",
                "-Wextra",
                "-march=armv8-a+sve",
                "-DSIMD",
                "-static",
                "-IFull_NN/inc",
                "-IFull_NN/gemm_definitions",
                "-I.",
                *oracle.get("compiler_flags", []),
                test_source,
                *oracle.get(
                    "translation_units",
                    ["Full_NN/src/gemm_exec.c", "Full_NN/src/gemm_SVE.c"],
                ),
                "-o",
                executable,
            ]
            commands.run(name + "-build", command, source)
            observed = commands.run(
                name + "-run", [*qemu, executable], output, timeout=300
            )
            diagnostics = (
                "[codebook-sdot] ",
                "[codebook-last-output] ",
                "[attention-sdot] ",
            )
            lines = [
                line
                for line in observed.strip().splitlines()
                if not line.startswith(diagnostics)
            ]
            require(lines == [oracle["expected_output"]], "Oracle mismatch: " + name)
            oracle_results[name] = {"passed": True, "binary_sha256": digest(executable)}
    save(output / "oracles.json", oracle_results)

    # Native file/layout tests do not require the ARM SIMD instruction set.
    require(args.host_cxx.is_file(), "Native sanitizer compiler is missing")
    if contract["native_checks"]:
        require(
            digest(args.host_cxx) == contract["native_compiler_sha256"],
            "Native sanitizer compiler changed",
        )
    save(
        output / "host-compiler.json",
        {"path": str(args.host_cxx), "sha256": digest(args.host_cxx)},
    )
    native_environment = commands.environment.copy()
    native_environment["LD_LIBRARY_PATH"] = str(args.host_cxx.parent.parent / "lib")
    for check in contract["native_checks"]:
        name = check["name"]
        executable = output / name
        commands.run(
            name + "-build",
            [
                str(args.host_cxx),
                "-std=c++17",
                "-O1",
                "-g",
                "-Wall",
                "-Wextra",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-I.",
                args.candidate / "tests" / check["source"],
                "-o",
                executable,
            ],
            source,
            native_environment,
        )
        observed = commands.run(
            name + "-run", [executable], output, native_environment, timeout=60
        )
        require(
            observed.strip() == check["expected_output"],
            "Native check mismatch: " + name,
        )

    binary = build_production(commands, source, tools)
    binary_sha = digest(binary)
    results = {}
    for mode in ("capture", "production-flags"):
        work = output / mode
        copy = work / "source"
        shutil.copytree(source, copy, ignore=shutil.ignore_patterns("transformer.o"))
        weights = work / "weights"
        shutil.copytree(args.inputs / "weights", weights)
        for name, item in contract["additional_inputs"].items():
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item["source"], destination)
        compressed_indices = contract["validation_compressed_indices"]
        instrument_copy(copy, weights, compressed_indices)
        changed = {
            name: sha
            for name, sha in hashes(copy).items()
            if effective.get(name) != sha
        }
        expected_changes = {"transformer.cpp"}
        if compressed_indices:
            expected_changes.add("transformer_layers/layerFactory.cc")
        require(
            set(changed) == expected_changes,
            "Unexpected instrumentation changes",
        )
        save(
            work / "instrumentation.json",
            {
                "changed_sources": changed,
                "production_binary_executed": False,
                "purpose": "Input redirection and final-byte capture in a validation copy",
            },
        )
        environment = commands.environment.copy()
        if mode == "capture":
            environment.update(
                PROFILE_GEMM_ONLY_FLAG="0",
                GEM5_PROFILE_REGIONS_FLAG="0",
                I2CE_USE_LIBM5_FLAG="0",
            )
        else:
            environment["LIBM5_A"] = str(functional_m5(commands, work, tools))
        commands.run(
            mode + "-build", ["bash", "compile_transformer.sh"], copy, environment
        )
        executable = copy / "transformer.o"
        with executable.open("rb") as stdin:
            log = commands.run(
                mode + "-run", [*qemu, executable], work, environment, stdin
            )
        for marker in (
            "D_SEQ = 512\n",
            "CODEBOOK_SIZE = 4\n",
            "N_LEARNERS = 2\n",
            "SVE_LENGTH on Gem5/QEMU = 16 bytes = 128 bits",
        ):
            require(marker in log, "Workload marker missing: " + marker)
        require(
            " Not loaded" not in log and "Weight file size mismatch:" not in log,
            "An input failed to load",
        )
        tensors = hashes(weights / "multiple_learner_outputs/c")
        expected = contract["expected_outputs"]["tensors"] if mode == "capture" else {}
        require(tensors == expected, mode + " intermediate tensors differ")
        final = {
            str(learner): digest(work / f"final-learner{learner}.bin")
            for learner in range(2)
        }
        require(
            final == contract["expected_outputs"]["final_output_sha256"],
            mode + " final bytes differ",
        )
        if mode == "production-flags":
            require(log.count("[GEM5_PROFILE] dump ") == 22, "Profiling markers differ")
            for vector_bytes in contract["rejected_sve_lengths_bytes"]:
                rejected = commands.run(
                    f"reject-sve-{vector_bytes}",
                    [
                        tools["qemu"]["path"],
                        "-cpu",
                        f"max,sve=on,sve-default-vector-length={vector_bytes}",
                        executable,
                    ],
                    work,
                    environment,
                    timeout=30,
                    expected_returncode=1,
                )
                require(
                    "ERROR: SVE vector-length mismatch." in rejected
                    and "[GEM5_PROFILE] reset" not in rejected,
                    "Incompatible SVE length was not rejected before ROI",
                )
        results[mode] = {
            "passed": True,
            "binary_sha256": digest(executable),
            "tensors": tensors,
            "final_outputs": final,
        }
        save(work / "complete.json", results[mode])
    require(
        digest(binary) == binary_sha, "Production binary modified during validation"
    )
    return {
        "passed": True,
        "results": results,
        "oracles": oracle_results,
        "production_binary_sha256": binary_sha,
        "binary_matches_reference": binary_sha == contract["original_binary_sha256"],
        "formal_runs": 0,
        "production_binary_executed": False,
        "limitations": [
            "QEMU functional checks are instrumented copies",
            "No new performance claim or exhaustive equivalence proof",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "contract", "reference-build", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--host-cxx", type=Path, required=True)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-oracles", type=Path)
    reuse.add_argument(
        "--production-reference",
        type=Path,
        help="Build only and require full binary identity with a tested candidate",
    )
    args = parser.parse_args()
    for name, value in vars(args).items():
        if value is not None:
            setattr(args, name, value.resolve())
    contract = json.loads(args.contract.read_text())
    check_contract(contract)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(contract["build_environment"])
    environment["PATH"] = (
        str(Path(contract["tools"]["compiler"]["path"]).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    commands = Commands(args.output, environment)
    save(
        args.output / "start.json",
        {
            "started_at": time.time(),
            "pid": os.getpid(),
            "helper_sha256": digest(Path(__file__)),
            "contract_sha256": digest(args.contract),
            "source_revision": contract["source_revision"],
            "python_optimization": sys.flags.optimize,
            "formal_runs": 0,
        },
    )
    try:
        save(args.output / "complete.json", validate(args, commands, contract))
    except Exception:
        save(
            args.output / "failure.json",
            {"error": traceback.format_exc(), "automatic_retry": False},
        )
        raise
    finally:
        save(args.output / "commands.json", commands.records)


if __name__ == "__main__":
    main()
