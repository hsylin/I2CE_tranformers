"""Build-time identities and private project inputs, shared by both compilers.

Tool executables stay in their installation: their subordinate tools, sysroots,
standard libraries and Python packages must remain stable during a build.
This module does not claim a hermetic operating system or toolchain.
"""

import os
from pathlib import Path
import shutil
import stat

from configuration_snapshot import regular_file_record, tree_manifest


BUILD_SCHEMA = "i2ce.build-inputs.v1"
GUEST_SOURCES = (
    "compile_transformer.sh",
    "transformer.cpp",
    "transformer.h",
    "Full_NN/inc",
    "Full_NN/src",
    "Full_NN/gemm_definitions",
    "weights/generated_from_notebook",
    "transformer_layers",
    "accelerator",
)
GENERATOR_SOURCES = ("tools/exp/lib", "Full_NN/generators")


def capture_executable(command):
    """Resolve PATH once, retaining argv[0] spelling for compiler-driver semantics."""
    found = shutil.which(str(command))
    if found is None:
        raise ValueError("Executable not found or not executable: " + str(command))
    invocation = Path(os.path.abspath(found))
    target = invocation.resolve(strict=True)
    record = dict(path=str(target), **regular_file_record(target))
    record["mode"] = stat.S_IMODE(target.stat().st_mode)
    return {"invocation": str(invocation), "identity": record}


def verify_executable(record):
    """Reject an alias retarget, content/mode change or missing executable."""
    if capture_executable(record["invocation"]) != record:
        raise ValueError("Build executable changed: " + record["invocation"])


def capture_sources(root, groups):
    """Record the declared private source closure, ignoring derived Python bytecode."""
    root = Path(root).resolve(strict=True)
    members = []
    for group in groups:
        path = root / group
        if path.is_dir() and not path.is_symlink():
            members.extend(
                dict(item, path=group + "/" + item["path"])
                for item in tree_manifest(path, ignore_bytecode=True)
            )
        else:
            members.append(dict(path=group, **regular_file_record(path)))
    return {"path": str(root), "groups": list(groups), "members": members}


def verify_sources(record):
    """Recheck the same source inventory, including additions and removals."""
    if capture_sources(record["path"], record["groups"]) != record:
        raise ValueError("Private build sources changed")


def copy_private_tree(source, destination):
    """Copy a stable, link-free project include tree; never reuse a partial copy."""
    before = tree_manifest(source)
    destination.mkdir()
    for item in before:
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / item["path"], target)
    if tree_manifest(source) != before or tree_manifest(destination) != before:
        raise ValueError("Project includes changed during snapshot creation")


def freeze_guest_inputs(source, settings, directory):
    """Bind guest sources and compiler; privately copy libm5 and all its headers."""
    compiler = capture_executable(settings["compiler"])
    sources = capture_sources(source, GUEST_SOURCES)
    root = directory / "guest-build-inputs"
    root.mkdir()
    library = Path(settings["libm5"]).resolve(strict=True)
    original_library = regular_file_record(library)
    shutil.copyfile(library, root / "libm5.a")
    if (
        regular_file_record(library) != original_library
        or regular_file_record(root / "libm5.a") != original_library
    ):
        raise ValueError("libm5 changed during snapshot creation")
    includes = Path(settings["m5_include"]).resolve(strict=True)
    copy_private_tree(includes, root / "include")
    record = {
        "compiler": compiler,
        "sources": sources,
        "snapshot": {"path": str(root), "members": tree_manifest(root)},
        "original_library": dict(path=str(library), **original_library),
        "original_includes": str(includes),
    }
    verify_guest_inputs(record, building=True)
    return record


def verify_guest_inputs(record, *, building=False):
    """Verify retained bytes; require the external compiler only while building."""
    verify_sources(record["sources"])
    snapshot = record["snapshot"]
    if tree_manifest(snapshot["path"]) != snapshot["members"]:
        raise ValueError("Private guest build inputs changed")
    if building:
        verify_executable(record["compiler"])


def clean_build_environment():
    """Remove ambient search-path injections from project compiler invocations."""
    excluded = {
        "CPATH",
        "CPLUS_INCLUDE_PATH",
        "C_INCLUDE_PATH",
        "OBJC_INCLUDE_PATH",
        "DEPENDENCIES_OUTPUT",
        "SUNPRO_DEPENDENCIES",
        "LIBRARY_PATH",
        "LD_LIBRARY_PATH",
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
    }
    return {key: value for key, value in os.environ.items() if key not in excluded}
