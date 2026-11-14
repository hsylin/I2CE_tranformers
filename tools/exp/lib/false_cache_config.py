"""Traceable starter_fs adapter: set L1D policy before instantiate, verify after.

No kernel or shared gem5 source is patched. Configuration-only mode never calls
simulate. Both configuration and formal outputs carry a policy receipt.
"""

import configparser
import hashlib
import importlib.util
import json
import os
import runpy
from pathlib import Path
import sys
import time


# gem5 executes this entrypoint as __m5_main__. Bootstrap before helper imports;
# importing the module for isolated unit tests does not alter their import state.
RUNNER_SOURCES = None
if __name__ in ("__m5_main__", "__main__"):
    bootstrap = runpy.run_path(
        str(Path(__file__).with_name("configuration_snapshot.py"))
    )
    RUNNER_SOURCES = bootstrap["activate_runner_sources"](
        Path(__file__).resolve().parent
    )


def verify_configuration(path, expected=None, allow_uncached=False):
    """Check the actual L1D policy and requested cache sizes after instantiation."""
    config = configparser.ConfigParser(interpolation=None)
    with Path(path).open() as stream:
        config.read_file(stream)
    caches = [section for section in config if section.endswith(".dcache")]
    if not caches and not allow_uncached:
        raise ValueError("No L1D cache in actual gem5 configuration")
    for cache in caches:
        if config[cache].get("writeback_clean", "").lower() != "false":
            raise ValueError("Actual L1D.writeback_clean is not False")
        if expected and int(config[cache]["size"]) != expected["l1d_size"]:
            raise ValueError("Actual L1D capacity differs from frozen parameters")
    outer = [section for section in config if section.endswith(".l2")]
    if expected and (
        len(outer) != 1 or int(config[outer[0]]["size"]) != expected["l2_size"]
    ):
        raise ValueError("Actual L2 capacity differs from frozen parameters")
    return {
        "l1d_sections": caches,
        "config_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "writeback_clean": False,
        "uncached_boot": not caches,
    }


def take_option(name, required=True):
    """Remove one adapter option before forwarding arguments to starter_fs."""
    values = [value for value in sys.argv if value.startswith(name + "=")]
    if len(values) != (1 if required else min(1, len(values))):
        raise ValueError("Missing or duplicate " + name)
    if not values:
        return None
    sys.argv.remove(values[0])
    return values[0].split("=", 1)[1]


def main():
    """Bind the False policy before instantiation; optionally stop without simulate."""
    import m5

    from configuration_snapshot import (
        activate_configuration,
        verify_snapshot,
        verify_configuration_imports,
        verify_bootloaders,
    )
    from configuration_identity import (
        capture_configurations,
        verify_matching_configuration,
        verify_run_configuration,
        bind_diod,
        TICKS_PER_SECOND,
    )
    from experiment_results import CapturedFile

    starter = Path(take_option("--runner-starter")).resolve()
    spec_path = take_option("--runner-spec", required=False)
    check = "--configuration-only" in sys.argv
    if check:
        sys.argv.remove("--configuration-only")
    spec = json.loads(Path(spec_path).read_text()) if spec_path else None
    if (
        spec
        and hashlib.sha256(starter.read_bytes()).hexdigest()
        != spec["identities"]["starter"]["sha256"]
    ):
        raise ValueError("starter_fs changed since run freezing")
    snapshot = spec["identities"]["configuration"] if spec else None
    bootloaders = spec["identities"].get("bootloaders") if spec else None
    if spec:
        m5.ticks.setGlobalFrequency(TICKS_PER_SECOND)
        verify_bootloaders(bootloaders)
        os.environ["M5_PATH"] = bootloaders["path"]
    if snapshot:
        activate_configuration(snapshot, starter)
    else:
        sys.path[:0] = [str(starter.parent), str(starter.parents[2])]
    definition = importlib.util.spec_from_file_location("_pinned_starter", starter)
    original = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(original)
    if snapshot:
        verify_configuration_imports(snapshot)
    create, run = original.create, original.run
    simulate = m5.simulate
    events = []
    receipt = {
        "passed": False,
        "configuration_only": check,
        "start": time.time(),
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "starter_sha256": hashlib.sha256(starter.read_bytes()).hexdigest(),
    }

    boundary_files = None

    def verify_boundary(args=None):
        """Recheck frozen state at each execution boundary, without duplicate checks.

        The run boundary checks before/after starter.run; the simulate boundary
        also catches changes made inside starter.run before each simulation call.
        """
        nonlocal boundary_files
        if RUNNER_SOURCES is not None:
            RUNNER_SOURCES.verify()
        output = Path(m5.options.outdir)
        if not spec:
            return verify_configuration(
                output / "config.ini", allow_uncached=args.cpu == "atomic"
            )
        verify_snapshot(snapshot)
        verify_configuration_imports(snapshot)
        frequency = int(m5.ticks.fromSeconds(1.0))
        if frequency != TICKS_PER_SECOND:
            raise ValueError("Actual machine tick frequency must be 1 THz")
        if boundary_files is None:
            boundary_files = (
                {"formal_config": CapturedFile.read(output / "config.ini")}
                if check
                else capture_configurations(
                    spec["paths"]["check"], output, completed=False
                )
            )
        for captured in boundary_files.values():
            captured.verify_unchanged()
        if check:
            actual = verify_run_configuration(boundary_files["formal_config"], spec)
        else:
            actual = verify_matching_configuration(
                spec["paths"]["check"], output, spec, captured=boundary_files
            )
        return dict(actual, tick_frequency=frequency)

    def guarded_simulate(*args, **kwargs):
        if check:
            raise RuntimeError("Configuration-only check attempted to simulate")
        if spec:
            verify_boundary()
        event = simulate(*args, **kwargs)
        events.append(
            {
                "cause": event.getCause(),
                "code": int(event.getCode()),
                "tick": int(m5.curTick()),
            }
        )
        return event

    def checked_create(args):
        if spec:
            if (
                args.restore != spec["paths"]["checkpoint"]
                or args.vio_9p != spec["paths"]["share"]
            ):
                raise ValueError("Checkpoint or private share differs from frozen run")
            if args.script != spec["paths"]["guest_script"] or args.cpu != "minor":
                raise ValueError("Guest script or CPU differs from frozen run")
            if int(args.num_cores) != 1 or int(args.sve_vl) != 1:
                raise ValueError("Expected one core and SVE128")
        system = create(args)
        if spec:
            actual = [str(path) for path in system.workload.boot_loader]
            if actual != verify_bootloaders(bootloaders):
                raise ValueError(
                    "Constructed ARM bootloaders differ from frozen snapshot"
                )
        if spec:
            bind_diod(
                system,
                spec["identities"].get("diod"),
                spec["paths"]["share"],
                m5.options.outdir,
            )
        count = 0
        for cluster in system.cpu_cluster:
            for cpu in cluster.cpus:
                if hasattr(cpu, "dcache"):
                    cpu.dcache.writeback_clean = False
                    count += 1
        if args.cpu != "atomic" and not count:
            raise ValueError("Unsupported starter: no accessible L1D construction path")
        return system

    def checked_run(args):
        receipt.update(verify_boundary(args))
        if check:
            receipt.update(
                passed=True, simulate_called=False, guest_instructions_executed=0
            )
            return
        run(args)
        receipt.update(verify_boundary(args))
        if not events or events[-1]["code"] != 0:
            raise RuntimeError("gem5 did not finish successfully")
        receipt.update(passed=True, simulate_called=True)

    original.create, original.run, m5.simulate = (
        checked_create,
        checked_run,
        guarded_simulate,
    )
    try:
        original.main()
    except BaseException as error:
        receipt["passed"] = False
        receipt["error"] = str(error)
        raise
    finally:
        receipt.update(finish=time.time(), events=events)
        with (Path(m5.options.outdir) / "cache-policy.json").open("x") as stream:
            json.dump(receipt, stream, indent=2)


if __name__ in ("__m5_main__", "__main__"):
    main()
