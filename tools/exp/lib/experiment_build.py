"""Build only the resolved source; keep generated artifacts private to a run."""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from experiment_spec import digest, require
from configuration_snapshot import regular_file_record
from build_inputs import (
    BUILD_SCHEMA,
    GENERATOR_SOURCES,
    capture_executable,
    verify_executable,
    capture_sources,
    verify_sources,
    freeze_guest_inputs,
    verify_guest_inputs,
    clean_build_environment,
)

BUILD_FLAGS = {
    "SIMD_FLAG": "1",
    "RELOAD_WEIGHT_FLAG": "1",
    "USE_NOTEBOOK_GENERATED_WEIGHTS_FLAG": "1",
    "USE_CODEBOOK_GEMM_FLAG": "1",
    "FULL_INTERLEAVED_PIPELINE_FLAG": "1",
    "USE_FP32_TRANSFORMER_FLAG": "0",
    "GEM5_PROFILE_REGIONS_FLAG": "1",
    "ENABLE_CODEBOOK_REFERENCE_FLAG": "0",
    "ENABLE_DEBUG_PRINT_FLAG": "0",
    "DENSE_NO_SIMD_BASELINE_FLAG": "0",
    "PROFILE_GEMM_ONLY_FLAG": "0",
    "CORE_NUM_FLAG": "1",
    "I2CE_USE_LIBM5_FLAG": "1",
    "I2CE_IDX_BITS_BYTE_ALIGNED": "0",
}


def identity(path):
    """Hash one resolved input, including the executable permission contract."""
    path = Path(path).resolve()
    require(path.is_file(), "Missing input file: " + str(path))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "mode": path.stat().st_mode & 0o777,
        "sha256": digest(path),
    }


def file_manifest(root):
    """Describe every regular input; reject symlinks and empty roots."""
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Input symlinks are not accepted: " + str(path))
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    require(records, "Empty input directory: " + str(root))
    return records


def run_checked(command, directory, log, environment=None):
    """Run a build step once, retaining its combined diagnostic log."""
    with Path(log).open("xb") as stream:
        result = subprocess.run(
            list(map(str, command)),
            cwd=directory,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    require(result.returncode == 0, "Command failed; inspect " + str(log))


def export_source(repo, sha, destination):
    """Extract only build inputs from an exact Git commit into a private tree."""
    destination.mkdir()
    paths = [
        "Full_NN",
        "transformer_layers",
        "accelerator",
        "transformer.cpp",
        "transformer.h",
        "compile_transformer.sh",
        "tools/exp",
    ]
    archive = destination.parent / "source.tar"
    with archive.open("xb") as stream:
        subprocess.run(
            ["git", "-C", str(repo), "archive", sha, *paths], stdout=stream, check=True
        )
    with tarfile.open(archive) as stream:
        members = stream.getmembers()
        for member in members:
            require(
                not Path(member.name).is_absolute()
                and ".." not in Path(member.name).parts,
                "Unsafe source member",
            )
            require(
                member.isfile() or member.isdir(),
                "Non-regular source member: " + member.name,
            )
        # Tracked or stale caches must never replace the exported Python source.
        sources = [
            m
            for m in members
            if "__pycache__" not in Path(m.name).parts
            and Path(m.name).suffix not in (".pyc", ".pyo")
        ]
        stream.extractall(destination, members=sources)
    return identity(archive)


def copy_guest_binary(source, destination):
    """Preserve the mode required by the guest's direct ./transformer.o call."""
    require(
        source.is_file() and os.access(source, os.X_OK),
        "Build output is not executable",
    )
    shutil.copy2(source, destination)
    require(os.access(destination, os.X_OK), "Guest-share binary is not executable")


# Keep this small closure explicit. The compiler's dependency file below rejects
# future project includes that were not copied into the private snapshot.
EXPORTER_SOURCE = "tools/exp/export_compressed_indices.cc"
EXPORTER_CODEC = "transformer_layers/compressed_index_file.h"
EXPORTER_FILES = (
    EXPORTER_SOURCE,
    EXPORTER_CODEC,
    "Full_NN/src/compressed_index_layout.h",
)
INCLUDE_ENVIRONMENT = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "DEPENDENCIES_OUTPUT",
    "SUNPRO_DEPENDENCIES",
)


def verify_exporter_inputs(snapshot):
    """Check the complete private input tree; never reread the live checkout."""
    root = Path(snapshot["path"])
    require(root.is_dir() and not root.is_symlink(), "Missing exporter snapshot")
    for path in root.rglob("*"):
        require(path.is_file() or path.is_dir(), "Special exporter input: " + str(path))
    require(
        file_manifest(root) == snapshot["members"],
        "Exporter snapshot changed",
    )
    return root


def freeze_exporter_inputs(headers, destination):
    """Copy the helper closure and generated headers before their first use.

    Originals are checked around copying. Later checkout edits are harmless:
    compilation uses only these private bytes, whose hashes are checked again
    before accepting the compiler and exporter outputs.
    """
    origin = Path(__file__).resolve().parents[3]
    sources = {name: origin / name for name in EXPORTER_FILES}
    sources.update(
        ("generated/" + item["path"], headers / item["path"])
        for item in file_manifest(headers)
    )
    members = [
        dict(path=name, **regular_file_record(path))
        for name, path in sorted(sources.items())
    ]
    destination.mkdir()
    for item in members:
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[item["path"]], target)
    for item in members:
        require(
            regular_file_record(sources[item["path"]])
            == {key: item[key] for key in ("bytes", "sha256")},
            "Exporter source changed during snapshot creation: " + item["path"],
        )
    snapshot = {"path": str(destination.resolve()), "members": members}
    verify_exporter_inputs(snapshot)
    return snapshot


def exporter_dependencies(path, snapshot):
    """Require GCC's non-system dependencies to belong to the captured closure.

    Compile paths are relative to the snapshot, with a fixed Make target. The
    controlled exporter/header names have no Make escapes; reject unsupported
    syntax instead of silently omitting a dependency. System C++ headers remain
    part of the external host toolchain, not this project-source snapshot.
    """
    text = path.read_text().replace("\\\n", " ")
    require(text.startswith("exporter:"), "Unexpected exporter dependency target")
    body = text[len("exporter:") :]
    require(
        not any(char in body for char in "\\\\$#"),
        "Unsupported exporter dependency syntax",
    )
    root = Path(snapshot["path"])
    expected = {item["path"] for item in snapshot["members"]}
    dependencies = set()
    for name in body.split():
        dependency = (root / name).resolve()
        require(
            dependency.is_relative_to(root), "External exporter dependency: " + name
        )
        relative = str(dependency.relative_to(root))
        require(relative in expected, "Unfrozen exporter dependency: " + name)
        dependencies.add(relative)
    require(set(EXPORTER_FILES) <= dependencies, "Incomplete exporter dependency file")
    return sorted(dependencies)


def verify_exporter_artifacts(record):
    """Keep the retained build evidence bound through completion and collection."""
    verify_exporter_inputs(record["snapshot"])
    for name in ("binary", "dependency_file"):
        item = record[name]
        require(identity(item["path"]) == item, "Exporter artifact changed: " + name)


def export_compressed_inputs(headers, share, directory, settings):
    """Compile and execute from one private snapshot, then return its evidence."""
    snapshot = freeze_exporter_inputs(headers, directory / "exporter-inputs")
    root = Path(snapshot["path"])
    compiler = capture_executable(settings["host_compiler"])
    binary = directory / "export-indices"
    dependencies = directory / "exporter-dependencies.d"
    command = [
        compiler["invocation"],
        "-std=c++17",
        "-O2",
        "-I.",
        "-Igenerated",
        "-MMD",
        "-MF",
        str(dependencies),
        "-MT",
        "exporter",
        EXPORTER_SOURCE,
        "-o",
        str(binary),
    ]
    environment = {
        key: value
        for key, value in clean_build_environment().items()
        if key not in INCLUDE_ENVIRONMENT
    }
    run_checked(command, root, directory / "exporter-build.log", environment)
    verify_exporter_inputs(snapshot)
    verify_executable(compiler)
    record = {
        "snapshot": snapshot,
        "compiler": compiler["identity"],
        "compiler_command": compiler,
        "compile_argv": command,
        "compile_cwd": str(root),
        "cleared_include_environment": list(INCLUDE_ENVIRONMENT),
        "dependencies": exporter_dependencies(dependencies, snapshot),
        "dependency_file": identity(dependencies),
        "binary": identity(binary),
    }
    run_checked(
        [binary, share / "weights/compressed_indices"],
        root,
        directory / "exporter.log",
    )
    verify_exporter_artifacts(record)
    verify_executable(compiler)
    return record


def build(repo, plan, directory, settings):
    """Generate and compile a private guest share, then record its provenance."""
    source, share = directory / "source", directory / "share"
    archive = export_source(repo, plan["parameters"]["sha"], source)
    share.mkdir()
    row = plan["parameters"]
    generator = source / "tools/exp/lib/gen_artifacts.py"
    generator_inputs = {
        "sources": capture_sources(source, GENERATOR_SOURCES),
        "python": capture_executable(settings["python"]),
    }
    generator_identity = identity(generator)
    command = [
        generator_inputs["python"]["invocation"],
        "-B",
        generator,
        "--repo",
        source,
        "--stage-root",
        source,
        "--n-learners",
        row["learners"],
        "--codebook-size",
        row["codebook_size"],
        "--seq-len",
        512,
        "--d-model",
        256,
        "--num-heads",
        4,
        "--d-ff",
        1024,
        "--d-q",
        64,
    ]
    generator_environment = dict(
        os.environ, I2CE_IDX_BITS_BYTE_ALIGNED="0", PYTHONDONTWRITEBYTECODE="1"
    )
    run_checked(command, source, directory / "generate.log", generator_environment)
    verify_sources(generator_inputs["sources"])
    verify_executable(generator_inputs["python"])
    headers = source / "Full_NN/gemm_definitions"
    defines = (headers / "codebooks_def.h").read_text()
    for name in ("TILE_L1_SIZE", "TILE_L2_SIZE", "SAME_SEQ"):
        import re

        require(
            re.search(r"#define\s+" + name + r"\s+1(?:\s|$)", defines),
            "Generated artifact has incompatible " + name,
        )
    guest_inputs = freeze_guest_inputs(source, settings, directory)
    private = Path(guest_inputs["snapshot"]["path"])
    environment = clean_build_environment()
    # These explicitly recorded values override ambient build switches.
    environment.update(
        BUILD_FLAGS,
        A64CXX=guest_inputs["compiler"]["invocation"],
        GEM5_ROOT=settings["gem5_root"],
        LIBM5_A=str(private / "libm5.a"),
        M5_INC=str(private / "include"),
    )
    run_checked(
        ["bash", "compile_transformer.sh"], source, directory / "build.log", environment
    )
    verify_guest_inputs(guest_inputs, building=True)
    copy_guest_binary(source / "transformer.o", share / "transformer.o")
    shutil.copytree(
        source / "weights/generated_from_notebook",
        share / "weights/generated_from_notebook",
    )
    exporter = export_compressed_inputs(headers, share, directory, settings)
    verify_guest_inputs(guest_inputs, building=True)
    verify_sources(generator_inputs["sources"])
    exporter_root = Path(exporter["snapshot"]["path"])
    # Every invocation owns a new share; no mutable binary/weights cache is reused.
    plan["build"] = {
        "schema": BUILD_SCHEMA,
        "generator_inputs": generator_inputs,
        "guest_inputs": guest_inputs,
        "environment": {
            key: environment[key]
            for key in (*BUILD_FLAGS, "A64CXX", "LIBM5_A", "M5_INC")
        },
        "generator_argv": list(map(str, command)),
        "source_archive": archive,
        "generator": generator_identity,
        "exporter": exporter,
        "exporter_source": identity(exporter_root / EXPORTER_SOURCE),
        "exporter_codec": identity(exporter_root / EXPORTER_CODEC),
        "compiler": guest_inputs["compiler"]["identity"],
        "libm5": identity(private / "libm5.a"),
        "binary": identity(share / "transformer.o"),
        "inputs": file_manifest(share),
    }
    return share
