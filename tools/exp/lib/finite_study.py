#!/usr/bin/env python3
"""One persistent owner for a finite fixed-workload study; never retry a reservation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import sys

import study_execution as execution
from study_records import (
    require,
    study_budget,
    replace_status,
    digest,
    load,
    save,
    process,
    verify_contract,
    verify_files,
)
from study_resources import sample_admission


def reserve(root, row, registration):
    budget = study_budget(registration)
    rows = registration["rows"]
    require(row in rows, "Cannot reserve an unregistered row")
    used = sum(
        (Path(item["state"]) / "formal-reservation.json").exists() for item in rows
    )
    if used >= budget:
        raise RuntimeError("The registered formal-run budget is exhausted")
    save(
        Path(row["state"]) / "formal-reservation.json",
        dict(
            stage=row["id"],
            epoch=time.time(),
            ordinal=used + 1,
            formal_budget=budget,
            maximum_attempts=1,
            automatic_retries=False,
            replacement_allowed=False,
        ),
    )


def configuration(root, registration, row):
    state = Path(row["state"])
    output = Path(row["output"]) / "configuration"
    output.mkdir(parents=True, exist_ok=False)
    save(state / "configuration-start.json", dict(epoch=time.time(), stage=row["id"]))
    args = execution.command(root, registration, row, configuration=True)
    lockpath = Path(registration["admission_lock"])
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with lockpath.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            decision = sample_admission(Path(registration["scratch"]), root)
            replace_status(state / "configuration-admission-current.json", decision)
            if decision["admitted"]:
                with (state / "configuration-process.log").open("xb") as log:
                    result = subprocess.run(
                        args,
                        cwd=registration["gem5_root"],
                        env=execution.environment(registration),
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=300,
                    )
                break
        time.sleep(20)
    save(
        state / "configuration-exit.json",
        dict(returncode=result.returncode, epoch=time.time(), command=args),
    )
    require(result.returncode == 0, "Study check failed: result.returncode == 0")
    gate, check = load(output / "config-gate.json"), load(
        output / "initialization-check.json"
    )
    require(
        gate["passed"] and check["passed"] and not check["simulate_called"],
        "Study check failed: gate['passed'] and check['passed'] and not check['simulate_called']",
    )
    require(
        check["guest_instructions_executed"] == 0,
        "Study check failed: check['guest_instructions_executed'] == 0",
    )
    require(
        not (Path(row["inputs"]) / "gem5_profile_regions.tsv").exists(),
        "Study check failed: not (Path(row['inputs'])/'gem5_profile_regions.tsv').exists()",
    )
    pins = dict(load(state / "input-binding.json")["files"])
    for name in [
        "config.ini",
        "config.json",
        "config-gate.json",
        "initialization-check.json",
    ]:
        pins[str(output / name)] = digest(output / name)
    save(
        state / "prelaunch-gate.json",
        dict(
            passed=True,
            stage=row["id"],
            pins=pins,
            configuration_only=True,
            formal_runs=0,
            admission=decision,
            epoch=time.time(),
        ),
    )


def launch(root, registration, row, decision):
    state = Path(row["state"])
    verify_files(load(state / "prelaunch-gate.json")["pins"])
    verify_files(registration["runtime_pins"])
    verify_files(registration["checkpoint_pins"])
    require(
        load(state / "prelaunch-gate.json")["passed"],
        "Study check failed: load(state/'prelaunch-gate.json')['passed']",
    )
    reserve(root, row, registration)
    output = Path(row["output"]) / "formal"
    output.mkdir(exist_ok=False)
    args = execution.command(root, registration, row)
    log = (state / "formal-process.log").open("xb")
    try:
        child = subprocess.Popen(
            args,
            cwd=registration["gem5_root"],
            env=execution.environment(registration),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()
    identity = process(child.pid)
    save(
        state / "formal-process.json",
        dict(identity, epoch=time.time(), command=args, admission=decision),
    )
    return dict(child=child, row=row, identity=identity, started=time.time())


def snapshot(job):
    row, identity = job["row"], job["identity"]
    try:
        current = process(identity["pid"])
        require(
            current["start"] == identity["start"],
            "Study check failed: current['start'] == identity['start']",
        )
    except (FileNotFoundError, ProcessLookupError):
        return 0
    current.update(
        epoch=time.time(),
        host_elapsed_seconds=time.time() - job["started"],
        output_bytes=sum(
            p.stat().st_size
            for p in (Path(row["output"]) / "formal").rglob("*")
            if p.is_file()
        ),
    )
    replace_status(Path(row["state"]) / "host-resources-current.json", current)
    return current["peak_rss_bytes"]


def mark_failure(root, stage, error):
    path = root / "evidence" / ("failure-" + stage + ".json")
    if not path.exists():
        save(
            path,
            dict(
                stage=stage,
                epoch=time.time(),
                error=error,
                future_admissions_stopped=True,
                automatic_retry=False,
            ),
        )


def controller(root, expected):
    registration = verify_contract(root, expected)
    budget = study_budget(registration)
    evidence = root / "evidence"
    # start.json is an immutable one-use owner reservation, including after a crash.
    save(
        evidence / "controller-start.json",
        dict(
            process(os.getpid()),
            epoch=time.time(),
            contract_sha256=expected,
            python_optimization=sys.flags.optimize,
            python_debug=__debug__,
        ),
    )
    os.nice(5)
    active, next_index, peak = {}, 0, 0
    failed = False
    try:
        verify_files(registration["runtime_pins"])
        verify_files(registration["checkpoint_pins"])
        verify_files(registration["stage_evidence_pins"])
        for row in registration["rows"]:
            execution.prepare(row, registration)
            configuration(root, registration, row)
        save(
            evidence / "all-configurations-complete.json",
            dict(
                passed=True,
                count=budget,
                simulate_called=False,
                guest_instructions=0,
                formal_runs=0,
                epoch=time.time(),
            ),
        )
        lockpath = Path(registration["admission_lock"])
        lockpath.parent.mkdir(parents=True, exist_ok=True)
        while active or (next_index < budget and not failed):
            for stage, job in list(active.items()):
                rc = job["child"].poll()
                if rc is None:
                    peak = max(peak, snapshot(job))
                    continue
                row = job["row"]
                save(
                    Path(row["state"]) / "formal-exit.json",
                    dict(
                        returncode=rc,
                        process=job["identity"],
                        epoch=time.time(),
                        host_elapsed_seconds=time.time() - job["started"],
                    ),
                )
                del active[stage]
                try:
                    require(rc == 0, "Nonzero simulator exit: " + str(rc))
                    execution.collect(root, registration, row)
                except Exception:
                    failed = True
                    mark_failure(root, stage, traceback.format_exc())
            pilot = registration["rows"][0]
            canary = evidence / "canary-entry.json"
            if next_index and not canary.exists():
                terminal = Path(pilot["output"]) / "formal/system.terminal"
                if terminal.exists():
                    markers = execution.entry_markers(terminal.read_text())
                    if all(markers.values()):
                        save(
                            canary,
                            dict(
                                stage=pilot["id"],
                                epoch=time.time(),
                                direct_markers=markers,
                                terminal_prefix_sha256=digest(terminal),
                                complete_measurement=False,
                            ),
                        )
            if (
                not failed
                and next_index < budget
                and (next_index == 0 or canary.exists())
            ):
                with lockpath.open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    decision = sample_admission(
                        Path(registration["scratch"]), root, peak
                    )
                    replace_status(evidence / "admission-current.json", decision)
                    if decision["admitted"]:
                        row = registration["rows"][next_index]
                        try:
                            active[row["id"]] = launch(
                                root, registration, row, decision
                            )
                            next_index += 1
                        except Exception:
                            failed = True
                            mark_failure(root, row["id"], traceback.format_exc())
            replace_status(
                evidence / "status-current.json",
                dict(
                    epoch=time.time(),
                    launched=next_index,
                    active=list(active),
                    failed=failed,
                    pilot_peak_rss_or_later=peak,
                ),
            )
            if active or (next_index < budget and not failed):
                time.sleep(20)
        verify_files(registration["runtime_pins"])
        verify_files(registration["checkpoint_pins"])
        verify_contract(root, expected)
        save(
            evidence / "controller-complete.json",
            dict(
                passed=not failed and next_index == budget,
                launched=next_index,
                failed=failed,
                epoch=time.time(),
                formal_budget=budget,
                all_children_reaped=True,
                final_runtime_pins_unchanged=True,
            ),
        )
    except Exception:
        mark_failure(root, "controller", traceback.format_exc())
        # Existing children retain their outputs. A crashed owner never restarts
        # or automatically recreates collection; explicit saved-state recovery
        # must inspect receipts and process identities before any missing suffix.
        raise


def status(root):
    result = {}
    for name in [
        "controller-start.json",
        "all-configurations-complete.json",
        "canary-entry.json",
        "controller-complete.json",
    ]:
        path = root / "evidence" / name
        result[name] = load(path) if path.exists() else None
    result["failures"] = [
        load(p) for p in sorted((root / "evidence").glob("failure-*.json"))
    ]
    result["archives"] = [
        load(p) for p in sorted((root / "evidence").glob("E*-archive.json"))
    ]
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status", "collect"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract-sha256")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action != "run":
        # Collection is owned by the controller. This command reads completed
        # receipts and cannot reparse data or launch work, even on repeated use.
        status(root)
        return
    require(args.contract_sha256, "Study check failed: args.contract_sha256")
    with (root / "evidence/controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller(root, args.contract_sha256)


if __name__ == "__main__":
    main()
