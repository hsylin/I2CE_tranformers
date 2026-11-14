#!/usr/bin/env python3
"""Per-commit experiments with immutable configuration and private run directories.

Use exp.sh experiment dryrun/submit --id ... or --manifest batch.tsv.
Alias: exp.sh <action> --id/--manifest ... (selector first). Single-row fields:
--at REF, --s1/--o1/--k1/--s2/--o2/--k2, --learners,
--codebook-size, --l1d-size, --l2-size. All are required. TSV columns are:
id sha s1 o1 k1 s2 o2 k2 learners codebook_size l1d_size l2_size.
Capacities accept bytes or explicit B/KiB/MiB/GiB/kB/MB/GB units.
Use status --id ID and collect --id ID after submit. Failed IDs are never reused.
Submit needs Linux, A64CXX, HOST_CXX, LIBM5_A, M5_INC and an existing
I2CE_CHECKPOINT. Other machine paths come from the existing runner.conf.
DIOD optionally selects the host 9P helper (default /usr/local/sbin/diod).
The helper must be executable; run paths must fit Linux UNIX-socket limits.
Older structured runs without helper identity, tick-frequency receipts or
completion-bound output hashes are not accepted by the strict collector;
their historical files remain unchanged. Output hashes cannot be backfilled
after completion without losing the original execution-to-output guarantee.
Collected results separately record verified collector sources and Python
identity. Checkout HEAD is optional context; source hashes identify dirty or
exported collectors. Collection requires the source-only CLI bootstrap.
dryrun only resolves Git objects and prints the plan; it creates no files.

Reproducibility boundary: private source/generated inputs, project headers,
libm5, imported gem5 configurations and bootloaders are saved before use.
Completion and collection share validation of the captured execution outputs.
The exporter records its own source closure, compiler invocation, dependency
file and binary; its inputs can be rebuilt from the retained closure.

This is not a hermetic toolchain. Compiler subordinate tools, sysroot, standard
libraries, Python packages, shell and OS must remain stable during a build.
Before/after hashes detect persistent drift, not malicious same-UID writes
that restore the original bytes. Keep run directories and receipts unchanged.
Resource admission is advisory, not an OS memory limit. Failed reservations
require inspection; the runner never retries them automatically. Configuration
and numerical validation do not establish performance preservation.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

# Direct script execution reads this entrypoint's .py. Bootstrap the helper
# loader by source as well: -B alone still permits reading existing bytecode.
RUNNER_SOURCES = None
if __name__ == "__main__":
    bootstrap = runpy.run_path(
        str(Path(__file__).with_name("configuration_snapshot.py"))
    )
    RUNNER_SOURCES = bootstrap["activate_runner_sources"](
        Path(__file__).resolve().parent
    )

from experiment_spec import (
    COLUMNS,
    digest,
    git,
    normalize,
    read_manifest,
    reject_policy_overrides,
    require,
    resolve,
    save_new,
)
from experiment_build import (
    BUILD_FLAGS,
    build,
    file_manifest,
    identity,
    verify_exporter_artifacts,
)
from build_inputs import BUILD_SCHEMA, verify_guest_inputs, verify_sources
from configuration_snapshot import (
    runner_manifest,
    freeze_bootloaders,
    freeze_diod,
    verify_diod,
    freeze_configuration,
    verify_bootloaders,
    verify_snapshot,
)
from configuration_identity import (
    HARDWARE_OPTIONS,
    MACHINE_OPTIONS,
    socket_path,
    verify_completed_configuration,
    capture_stage_configuration,
    verify_captured_policy,
)
from experiment_results import METRICS_SCHEMA, CapturedFile, extract_bytes
import simulation_slots

LIBRARY = Path(__file__).resolve().parent


def verify_runner_sources():
    """Keep loaded helper code bound to the source identity recorded for this process."""
    if RUNNER_SOURCES is not None:
        RUNNER_SOURCES.verify()


def collector_checkout_commit():
    """Return optional checkout context; the source manifest is authoritative.

    A dirty checkout can share HEAD with different collector bytes. A copied
    library outside tools/exp/lib must not inherit an unrelated enclosing HEAD.
    """
    if LIBRARY.parts[-3:] != ("tools", "exp", "lib"):
        return None
    repository = LIBRARY.parent.parent.parent
    if not (repository / ".git").exists():
        return None
    try:
        return git(repository, "rev-parse", "HEAD").decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def capture_collection_provenance():
    """Identify the verified collector, independently of the execution checkout.

    Only the source-only CLI bootstrap can attest which helper bytes were
    loaded. An ordinary Python import cannot establish that contract.
    """
    require(
        RUNNER_SOURCES is not None,
        "Collection requires verified source loading; use exp.sh experiment collect",
    )
    require(RUNNER_SOURCES.root == LIBRARY, "Collector source root differs")
    members = RUNNER_SOURCES.verify()
    canonical = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "i2ce.collection.v1",
        "metrics_schema": METRICS_SCHEMA,
        "source_loading": "verified-python-source",
        "source_root": str(LIBRARY),
        "source_manifest": members,
        "source_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "checkout_commit": collector_checkout_commit(),
        "python": {
            "executable": identity(sys.executable),
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
        },
    }


def verify_collection_provenance(provenance):
    """Refuse collector or interpreter drift before publishing derived metrics."""
    require(
        RUNNER_SOURCES.verify() == provenance["source_manifest"],
        "Collector sources changed during collection",
    )
    require(
        identity(sys.executable) == provenance["python"]["executable"],
        "Collector interpreter changed during collection",
    )


def saved_object(record, fields, label):
    """Refuse older/incomplete record layouts with a useful field diagnostic."""
    require(isinstance(record, dict), "Expected a JSON object in " + label)
    missing = sorted(set(fields) - record.keys())
    require(
        not missing,
        "Unsupported or incomplete structured run metadata in "
        + label
        + ": missing "
        + ", ".join(missing),
    )
    return record


class Once(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        seen = getattr(namespace, "_seen", set())
        if self.dest in seen:
            parser.error("Duplicate option: " + option_string)
        seen.add(self.dest)
        namespace._seen = seen
        setattr(namespace, self.dest, values)


def arguments(argv):
    """Parse one command; require complete rows and reject ambiguous options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("action", choices=("submit", "dryrun", "status", "collect"))
    parser.add_argument("--manifest", action=Once, type=Path)
    for name in COLUMNS:
        option = "--at" if name == "sha" else "--" + name.replace("_", "-")
        parser.add_argument(option, dest=name, action=Once)
    args = parser.parse_args(argv)
    fields = {
        key: getattr(args, key) for key in COLUMNS if getattr(args, key) is not None
    }
    if args.action in ("status", "collect"):
        require(
            not args.manifest and set(fields) <= {"id"},
            "status/collect accept only --id",
        )
        require(args.action == "status" or args.id, "collect needs --id")
        if args.id:
            import re

            require(
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.id),
                "Invalid experiment ID",
            )
        return args, []
    require(
        not (args.manifest and fields), "Manifest and row options cannot be combined"
    )
    rows = read_manifest(args.manifest) if args.manifest else [normalize(fields)]
    return args, rows


def settings_from_environment():
    """Validate settings for new execution, separate from saved-run inspection."""
    env = os.environ
    reject_policy_overrides(env)
    require(
        env.get("RUN_CPU", "minor") == "minor" and env.get("NUM_CORES", "1") == "1",
        "Structured experiments currently require one minor CPU",
    )
    hardware = list(HARDWARE_OPTIONS)
    require(
        shlex.split(env.get("GEM5_MACHINE_ARGS", " ".join(hardware))) == hardware,
        "Structured experiments require the recorded 4GHz, 2GiB, single-channel DDR3 hardware settings",
    )

    def value(name, default=""):
        return env.get(name, default)

    root = value("GEM5_CWD")
    settings = {
        "root": value("EXP_ROOT"),
        "repo": value("REPO_ROOT"),
        "python": sys.executable,
        "compiler": value("A64CXX"),
        "host_compiler": value("HOST_CXX", "g++"),
        "gem5_root": root,
        "m5_path": value("M5_PATH", str(Path.home() / "i2ce/gem5_resources")),
        "gem5": value("GEM5_BIN"),
        "starter": str(
            Path(root) / value("GEM5_CFG", "configs/example/arm/starter_fs.py")
        ),
        "kernel": value("KERNEL"),
        "disk": value("DISK"),
        "diod": value("DIOD", "/usr/local/sbin/diod"),
        "checkpoint": value("I2CE_CHECKPOINT"),
        "libm5": value("LIBM5_A", str(Path(root) / "util/m5/build/arm64/out/libm5.a")),
        "m5_include": value("M5_INC", str(Path(root) / "include")),
        "max_parallel": simulation_slots.positive(value("MAX_PARALLEL", "auto")),
        "reserve_cpus": float(value("SLOT_RESERVE_CPUS", "4")),
        "reserve_gib": float(value("SLOT_RESERVE_GIB", "16")),
        "rss_gib": float(value("SLOT_RSS_GIB", "8")),
    }
    simulation_slots.validate_resources(
        settings["max_parallel"],
        settings["reserve_cpus"],
        settings["reserve_gib"],
        settings["rss_gib"],
    )
    return settings


def describe_commands(plan, settings, directory):
    """Describe planned actions without executing or creating files."""
    row = plan["parameters"]
    return {
        "source": ["git", "archive", row["sha"], "<build sources>"],
        "generate": [
            settings["python"],
            "<pinned source>/tools/exp/lib/gen_artifacts.py",
            "--n-learners",
            row["learners"],
            "--codebook-size",
            row["codebook_size"],
        ],
        "build": ["bash", "<isolated source>/compile_transformer.sh"],
        "build_environment": BUILD_FLAGS,
        "guest": ["./transformer.o", *plan["runtime_argv"]],
        "configuration_only": [
            "<gem5>",
            "<false_cache_config.py>",
            "--configuration-only",
            "--l1d-size=" + str(row["l1d_size"]) + "B",
            "--l2-size=" + str(row["l2_size"]) + "B",
        ],
        "run_directory": str(directory),
        "checkpoint": settings["checkpoint"],
        "new_performance_executed_by_dryrun": False,
    }


def guest_script(share, argv):
    """Render a private guest invocation and persist its exit status."""
    template = (LIBRARY / "run.rcS.tmpl").read_text()
    # Keep the established guest boot/mount sequence. Every path is a new share.
    template = template.replace("@SHARE@", shlex.quote(str(share))).replace(
        "@ID@", share.parent.name
    )
    template = (
        template.replace("@CB@", "frozen")
        .replace("@NL@", "frozen")
        .replace("@SVE@", "128")
    )
    template = template.replace(
        "./transformer.o\n", shlex.join(["./transformer.o", *argv]) + "\n"
    )
    template = template.replace(
        "RC=$?\n", 'RC=$?\nprintf "%s\\n" "$RC" > guest-exit-code\n'
    )
    return template


def gem5_command(plan, settings, output, check=False):
    """Build identical machine arguments for configuration and formal stages."""
    row = plan["parameters"]
    command = [
        settings["gem5"],
        "-d",
        str(output),
        "--stats-file=stats.txt",
        "--dump-config=config.ini",
        str(LIBRARY / "false_cache_config.py"),
        "--runner-starter=" + settings["starter"],
        "--runner-spec=" + str(output.parent / "frozen.json"),
        *MACHINE_OPTIONS,
        "--l1d-size=" + str(row["l1d_size"]) + "B",
        "--l2-size=" + str(row["l2_size"]) + "B",
        "--kernel=" + settings["kernel"],
        "--disk-image=" + settings["disk"],
        "--restore=" + settings["checkpoint"],
        "--script=" + plan["paths"]["guest_script"],
        "--vio-9p=" + plan["paths"]["share"],
    ]
    if check:
        command.append("--configuration-only")
    return command


def execute_gem5(command, output, settings, pre_launch=None):
    """Admit one process through the shared resource gate and record its exit."""
    output.mkdir()
    args = SimpleNamespace(
        out=output,
        state=Path.home() / ".cache/i2ce/simulation-slots",
        command=command,
        cwd=Path(settings["gem5_root"]),
        limit=settings["max_parallel"],
        reserve_cpus=settings["reserve_cpus"],
        reserve_gib=settings["reserve_gib"],
        rss_gib=settings["rss_gib"],
        poll=10,
        return_child=True,
        pre_launch=pre_launch,
        environment=dict(
            os.environ, M5_PATH=settings["m5_path"], PYTHONDONTWRITEBYTECODE="1"
        ),
    )
    child = simulation_slots.launch(args)
    code = child.wait()
    save_new(output / "exit.json", {"exit_code": code, "finish": time.time()})
    require(code == 0, "gem5 failed: " + str(output))


def verify_execution_assets(plan):
    """Check saved asset identities without depending on today's runner checkout."""
    saved_object(plan, ("identities", "build", "paths", "parameters"), "frozen.json")
    identities = saved_object(
        plan["identities"],
        (
            "gem5",
            "starter",
            "kernel",
            "disk",
            "guest_script",
            "configuration",
            "bootloaders",
            "diod",
            "checkpoint",
        ),
        "frozen.json identities",
    )
    for name in ("gem5", "starter", "kernel", "disk", "guest_script"):
        item = saved_object(identities[name], ("path",), "frozen.json " + name)
        require(identity(item["path"]) == item, "Frozen input changed: " + name)
    verify_snapshot(
        saved_object(
            identities["configuration"],
            ("path", "members", "starter"),
            "frozen.json configuration",
        )
    )
    verify_bootloaders(identities["bootloaders"])
    verify_diod(identities["diod"])
    checkpoint = saved_object(
        identities["checkpoint"], ("path", "members"), "frozen.json checkpoint"
    )
    require(
        file_manifest(checkpoint["path"]) == checkpoint["members"], "Checkpoint changed"
    )
    verify_build_inputs(plan)


def verify_frozen_inputs(plan):
    """Check execution assets and the owner code before launch and completion."""
    verify_runner_sources()
    verify_execution_assets(plan)
    require(runner_manifest(LIBRARY) == plan["identities"]["runner"], "Runner changed")


def verify_execution_state(directory, plan, frozen_sha):
    """Reject changes to the configuration or assets after their initial freeze."""
    require(digest(directory / "frozen.json") == frozen_sha, "Frozen metadata changed")
    verify_frozen_inputs(plan)


def verify_build_inputs(plan):
    """Verify private binary permissions and every generated input's content."""
    record = saved_object(plan.get("build"), ("schema",), "build provenance")
    require(
        record.get("schema") == BUILD_SCHEMA,
        "Unsupported build provenance; original inputs cannot be backfilled",
    )
    try:
        verify_sources(record["generator_inputs"]["sources"])
        verify_guest_inputs(record["guest_inputs"])
        verify_exporter_artifacts(record["exporter"])
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete build provenance") from error
    if "binary" in plan["build"]:
        item = plan["build"]["binary"]
        require(identity(item["path"]) == item, "Guest binary identity or mode changed")
        require(
            os.access(item["path"], os.X_OK), "Guest binary is no longer executable"
        )
    for item in plan["build"]["inputs"]:
        path = Path(plan["paths"]["share"]) / item["path"]
        require(digest(path) == item["sha256"], "Run input changed: " + item["path"])


def freeze_execution_inputs(plan, settings, directory, share):
    """Freeze private configuration imports and bind every execution asset once."""
    plan["paths"] = {
        "share": str(share),
        "guest_script": str(share / "run.rcS"),
        "checkpoint": str(Path(settings["checkpoint"]).resolve()),
        "check": str(directory / "configuration-check"),
    }
    settings["checkpoint"] = plan["paths"]["checkpoint"]
    configuration = freeze_configuration(
        Path(settings["gem5_root"]) / "configs",
        settings["starter"],
        directory / "gem5-configs",
    )
    settings["starter"] = str(verify_snapshot(configuration))
    plan["identities"] = {
        key: identity(settings[key]) for key in ("gem5", "starter", "kernel", "disk")
    }
    # Commands must use the same resolved paths that were hashed. A mutable
    # symlink in a caller's settings must not redirect a queued execution.
    for key in ("gem5", "starter", "kernel", "disk"):
        settings[key] = plan["identities"][key]["path"]
    plan["identities"]["configuration"] = configuration
    plan["identities"]["diod"] = freeze_diod(settings["diod"])
    bootloaders = freeze_bootloaders(
        settings["m5_path"], settings["gem5_root"], directory / "gem5-resources"
    )
    plan["identities"]["bootloaders"] = bootloaders
    settings["m5_path"] = bootloaders["path"]
    plan["identities"]["checkpoint"] = {
        "path": settings["checkpoint"],
        "members": file_manifest(settings["checkpoint"]),
    }
    verify_runner_sources()
    plan["identities"]["runner"] = runner_manifest(LIBRARY)
    plan["identities"]["guest_script"] = identity(share / "run.rcS")
    plan["runner_commit"] = (
        git(Path(__file__).parents[3], "rev-parse", "HEAD").decode().strip()
    )
    save_new(directory / "frozen.json", plan)
    frozen_sha = digest(directory / "frozen.json")
    save_new(directory / "frozen-hash.json", {"sha256": frozen_sha})

    return frozen_sha


def capture_stage_outputs(directory, prefix):
    """Capture one finished gem5 stage's machine, policy and process status."""
    files = capture_stage_configuration(directory, prefix)
    files[prefix + "_exit"] = CapturedFile.read(directory / "exit.json")
    return files


def capture_execution_outputs(directory, plan):
    """Capture every output used to accept an execution, in one inventory.

    Inputs are frozen before launch. These files are produced by the two stages
    and the guest, so they are captured only after execution. Diagnostic logs,
    PID records and derived result.json are not acceptance inputs.
    """
    files = capture_stage_outputs(directory / "configuration-check", "check")
    files.update(capture_stage_outputs(directory / "formal", "formal"))
    share = Path(plan["paths"]["share"])
    paths = {
        "guest_exit": share / "guest-exit-code",
        "stats": directory / "formal/stats.txt",
        "roi_mapping": share / "gem5_profile_regions.tsv",
    }
    files.update({name: CapturedFile.read(path) for name, path in paths.items()})
    return files


def verify_captured_files(files):
    """Refuse a change to any consumed artifact before publishing a receipt."""
    for captured in files.values():
        captured.verify_unchanged()


def validate_execution_outputs(directory, plan, files):
    """Use the same captured bytes and acceptance rules at completion and collection."""
    actual = verify_completed_configuration(
        directory / "configuration-check", directory / "formal", plan, captured=files
    )
    for name in ("check_exit", "formal_exit"):
        record = saved_object(json.loads(files[name].data), ("exit_code",), name)
        require(
            type(record["exit_code"]) is int and record["exit_code"] == 0,
            "Unsuccessful process exit: " + name,
        )
    require(
        files["guest_exit"].data.strip() == b"0",
        "Guest executable did not finish successfully",
    )
    metrics = extract_bytes(files["stats"].data, files["roi_mapping"].data)
    return actual, metrics


def seal_completed_outputs(directory, plan, frozen_sha, *, checked=None):
    """Bind output identities to this execution before publishing success."""
    files = capture_execution_outputs(directory, plan)
    # A completed preflight is an input to the waiting formal stage. Preserve
    # its original identity instead of accepting a consistently rewritten pair.
    if checked is not None:
        for name, captured in checked.items():
            require(
                files[name].identity == captured.identity,
                "Configuration-only evidence changed: " + name,
            )
    validate_execution_outputs(directory, plan, files)
    verify_captured_files(files)
    record = {
        "schema": "i2ce.execution-outputs.v2",
        "passed": True,
        "frozen_sha256": frozen_sha,
        "checked_at": time.time(),
        "outputs": {name: captured.identity for name, captured in files.items()},
    }
    path = directory / "execution-integrity.json"
    save_new(path, record)
    saved = CapturedFile.read(path)
    require(json.loads(saved.data) == record, "Completion integrity receipt changed")
    verify_captured_files(files)
    return saved.identity["sha256"]


def execute(directory):
    """Own one reserved run from build through policy checks and execution."""
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text())
    plan, settings = request["plan"], request["settings"]
    receipt = {"passed": False, "start": time.time(), "formal_started": False}
    try:
        require(sys.platform.startswith("linux"), "Execution requires Linux")
        require(
            digest(directory / "request.json")
            == json.loads((directory / "request-hash.json").read_text())["sha256"],
            "Submission request changed before execution",
        )
        for stage in ("configuration-check", "formal"):
            socket_path(directory / stage)
        # A global build lock prevents multiple submits exhausting compiler RAM.
        lock_root = Path.home() / ".cache/i2ce/simulation-slots"
        lock_root.mkdir(parents=True, exist_ok=True)
        with (lock_root / "build.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            share = build(settings["repo"], plan, directory, settings)
        (share / "run.rcS").write_text(guest_script(share, plan["runtime_argv"]))
        frozen_sha = freeze_execution_inputs(plan, settings, directory, share)

        def verify_before_launch():
            verify_execution_state(directory, plan, frozen_sha)

        check_command = gem5_command(
            plan, settings, directory / "configuration-check", check=True
        )
        formal_command = gem5_command(plan, settings, directory / "formal")
        save_new(
            directory / "commands.json",
            {"configuration_only": check_command, "formal": formal_command},
        )
        execute_gem5(
            check_command,
            directory / "configuration-check",
            settings,
            verify_before_launch,
        )
        checked = capture_stage_outputs(directory / "configuration-check", "check")
        verify_captured_policy(
            checked["check_config"], checked["check_policy"], False, plan
        )
        verify_captured_files(checked)

        def verify_before_formal():
            verify_before_launch()
            verify_captured_files(checked)
            receipt["formal_started"] = True

        execute_gem5(
            formal_command, directory / "formal", settings, verify_before_formal
        )
        verify_execution_state(directory, plan, frozen_sha)
        receipt["execution_integrity_sha256"] = seal_completed_outputs(
            directory, plan, frozen_sha, checked=checked
        )
        receipt["passed"] = True
    except BaseException as error:
        receipt["error"] = str(error)
    finally:
        receipt["finish"] = time.time()
        save_new(directory / "complete.json", receipt)
    return 0 if receipt["passed"] else 1


def capture_completion_metadata(directory):
    """Bind a successful completion to its original plan and output seal.

    Return the validated records together with the exact files that supplied
    them. The caller rechecks every captured file before publishing a result.
    """
    complete_file = CapturedFile.read(directory / "complete.json")
    complete = saved_object(
        json.loads(complete_file.data), ("passed",), "complete.json"
    )
    require(
        complete["passed"] is True,
        "Failed or incomplete run cannot be collected as valid",
    )
    frozen_hash_file = CapturedFile.read(directory / "frozen-hash.json")
    frozen_hash = saved_object(
        json.loads(frozen_hash_file.data), ("sha256",), "frozen-hash.json"
    )
    frozen_file = CapturedFile.read(directory / "frozen.json")
    require(
        frozen_file.identity["sha256"] == frozen_hash["sha256"],
        "Frozen metadata changed",
    )
    integrity_file = CapturedFile.read(directory / "execution-integrity.json")
    integrity = saved_object(
        json.loads(integrity_file.data),
        ("passed", "frozen_sha256", "outputs"),
        "execution-integrity.json",
    )
    saved_object(complete, ("execution_integrity_sha256",), "complete.json")
    require(
        integrity_file.identity["sha256"] == complete["execution_integrity_sha256"],
        "Completion integrity receipt changed",
    )
    require(
        integrity["passed"] is True
        and integrity["frozen_sha256"] == frozen_file.identity["sha256"],
        "Missing or mismatched execution integrity",
    )
    require(
        integrity.get("schema") == "i2ce.execution-outputs.v2",
        "Unsupported completion output schema; original evidence cannot be backfilled",
    )
    return SimpleNamespace(
        plan=json.loads(frozen_file.data),
        completion=complete,
        integrity=integrity,
        files={
            "complete": complete_file,
            "frozen": frozen_file,
            "frozen_hash": frozen_hash_file,
            "integrity": integrity_file,
        },
    )


def collect(directory):
    """Verify the sealed evidence, derive metrics from it, and publish once."""
    collection = capture_collection_provenance()
    saved = capture_completion_metadata(directory)
    verify_execution_assets(saved.plan)
    files = capture_execution_outputs(directory, saved.plan)
    expected = saved_object(
        saved.integrity["outputs"], files, "completed output identities"
    )
    require(set(expected) == set(files), "Unexpected completed output identities")
    for name, captured in files.items():
        require(
            captured.identity == expected[name], "Completed output changed: " + name
        )
    actual, metrics = validate_execution_outputs(directory, saved.plan, files)
    verify_captured_files(files)
    verify_captured_files(saved.files)
    verify_collection_provenance(collection)
    save_new(
        directory / "result.json",
        {
            "passed": True,
            "collection": collection,
            "frozen": saved.plan,
            "actual_config": actual,
            "execution": saved.completion,
            "execution_integrity_sha256": saved.files["integrity"].identity["sha256"],
            "execution_outputs": {
                name: captured.identity for name, captured in files.items()
            },
            "stats": files["stats"].identity,
            "roi_mapping": files["roi_mapping"].identity,
            "metrics": metrics,
            "comparison_status": "measured counters; no performance comparison claimed",
        },
    )


def experiment_root():
    """Resolve saved-run storage without validating current execution settings."""
    location = os.environ.get("EXP_ROOT", "")
    require(
        location and Path(location).is_absolute(),
        "EXP_ROOT must be an absolute directory",
    )
    # Bind aliases once so planning and gem5 use the same socket pathname.
    return Path(location).resolve() / "experiments"


def print_status(root, identifier=None):
    """Show the latest durable receipt for one run or all reserved runs."""
    paths = (
        [root / identifier]
        if identifier
        else (
            sorted(path for path in root.iterdir() if path.is_dir())
            if root.exists()
            else []
        )
    )
    for path in paths:
        state = next(
            (
                name
                for name in (
                    "result.json",
                    "complete.json",
                    "owner.json",
                    "request.json",
                )
                if (path / name).exists()
            ),
            None,
        )
        print(
            json.dumps(
                {
                    "id": path.name,
                    "receipt": state,
                    "record": json.loads((path / state).read_text()) if state else None,
                }
            )
        )


def prepare_plans(rows, settings, root, action):
    """Resolve a batch before reservation; retain rejected submissions separately."""
    try:
        plans = [resolve(settings["repo"], row) for row in rows]
    except (ValueError, subprocess.CalledProcessError) as error:
        if action == "submit":
            rejected = Path(settings["root"]) / "rejections"
            rejected.mkdir(parents=True, exist_ok=True)
            save_new(
                rejected / (str(time.time_ns()) + ".json"),
                {
                    "passed": False,
                    "formal_started": False,
                    "rows": rows,
                    "error": str(error),
                },
            )
        raise
    for plan in plans:
        destination = root / plan["parameters"]["id"]
        for stage in ("configuration-check", "formal"):
            socket_path(destination / stage)
        require(not destination.exists(), "ID already reserved: " + destination.name)
        plan["commands"] = describe_commands(plan, settings, destination)
    return plans


def validate_submission(settings):
    """Require the configured machine assets before reserving any experiment ID."""
    require(
        settings["checkpoint"] and Path(settings["checkpoint"]).is_dir(),
        "Set I2CE_CHECKPOINT to an existing compatible single-core SVE128 checkpoint",
    )
    freeze_diod(settings["diod"])  # Validate without creating a snapshot or running it.
    for key in ("compiler", "gem5", "starter", "kernel", "disk", "libm5"):
        require(Path(settings[key]).is_file(), "Missing configured " + key)


def start_owner(directory):
    """Launch one reserved owner; its durable request prevents automatic retries."""
    with (directory / "owner.log").open("xb") as log:
        child = subprocess.Popen(
            [
                "nice",
                "-n",
                "10",
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "_execute",
                str(directory),
            ],
            cwd="/tmp",
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save_new(
        directory / "owner.json",
        {
            "pid": child.pid,
            "start": time.time(),
            "identity": simulation_slots.proc_info(child.pid),
        },
    )
    print(str(directory))


def submit_plans(root, plans, settings):
    """Reserve the complete batch under one lock before any owner can execute."""
    validate_submission(settings)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(
            all(not (root / p["parameters"]["id"]).exists() for p in plans),
            "Duplicate submission",
        )
        for plan in plans:
            directory = root / plan["parameters"]["id"]
            directory.mkdir()
            save_new(directory / "request.json", {"plan": plan, "settings": settings})
            save_new(
                directory / "request-hash.json",
                {"sha256": digest(directory / "request.json")},
            )
        for plan in plans:
            start_owner(root / plan["parameters"]["id"])


def main(argv=None):
    """Dispatch saved-run inspection independently of submission settings."""
    args, rows = arguments(sys.argv[1:] if argv is None else argv)
    root = experiment_root()
    if args.action == "status":
        print_status(root, args.id)
    elif args.action == "collect":
        collect(root / args.id)
    else:
        settings = settings_from_environment()
        plans = prepare_plans(rows, settings, root, args.action)
        if args.action == "dryrun":
            print(json.dumps({"plans": plans}, indent=2))
        else:
            submit_plans(root, plans, settings)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_execute"]:
        sys.exit(execute(sys.argv[2]))
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
