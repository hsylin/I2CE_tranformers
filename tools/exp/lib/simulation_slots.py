#!/usr/bin/env python3
"""Resource-aware, cross-runner admission for Linux gem5 processes.

The lock covers inspection and launch, not the simulation lifetime. A live
process (PID + kernel start tick) owns its slot; crashes release it naturally.
The host process budget never changes the simulated CPU/cache configuration.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

GIB = 1024**3


def sync_directory(directory):
    """Persist a rename/unlink before another caller can rely on shared state."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save(path, data):
    """Atomically persist shared admission state while the caller holds its lock."""
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    sync_directory(path.parent)


def require_no_pending_launch(state):
    """Fail closed across all output IDs after an unfinished registration.

    A pending reservation may own a live child whose PID could not be saved.
    Do not infer that it is safe to clear or retry it. Inspection must establish
    what happened before this shared marker can be removed.
    """
    pending = state / "pending-launch.json"
    if os.path.lexists(pending):
        raise RuntimeError("Unresolved launch reservation; inspect " + str(pending))


def positive(value):
    """Parse an explicit cap; zero never means unlimited."""
    if value == "auto":
        return value
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "limit must be auto or a positive integer"
        ) from None
    if number < 1:
        raise argparse.ArgumentTypeError(
            "zero is not unlimited; use auto or a positive integer"
        )
    return number


def proc_info(pid, proc=Path("/proc")):
    """Identify a live process owned by this user using PID and start ticks."""
    try:
        p = proc / str(pid)
        if p.stat().st_uid != os.getuid():
            return None
        # comm may contain spaces or parentheses; fields after its closing ')'
        # start at stat field 3, making starttime (field 22) index 19.
        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        status = dict(
            line.split(":", 1)
            for line in (p / "status").read_text().splitlines()
            if ":" in line
        )
        return dict(
            pid=int(pid),
            start=fields[19],
            comm=(p / "comm").read_text().strip(),
            argv=(p / "cmdline")
            .read_bytes()
            .decode(errors="replace")
            .strip("\0")
            .split("\0"),
            rss=int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
            hwm=int(status.get("VmHWM", "0 kB").split()[0]) * 1024,
        )
    except (OSError, ValueError, IndexError):
        return None


def processes(name):
    """List owned live processes with the executable's Linux comm name."""
    result = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            item = proc_info(p.name)
            if item and item["comm"] == name[:15]:
                result.append(item)
    return result


def cgroup_limits():
    """Read v1/v2 limits at the current cgroup AND every ancestor."""
    cpus, available = [], []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, relative = line.split(":", 2)
        versions = [("v2", Path("/sys/fs/cgroup"))] if not controllers else []
        if "cpu" in controllers.split(","):
            versions.append(("cpu", Path("/sys/fs/cgroup/cpu")))
        if "memory" in controllers.split(","):
            versions.append(("memory", Path("/sys/fs/cgroup/memory")))
        for kind, root in versions:
            current = root / relative.lstrip("/")
            while True:
                try:
                    if kind == "v2":
                        quota, period = (current / "cpu.max").read_text().split()
                        if quota != "max":
                            cpus.append(int(quota) / int(period))
                    elif kind == "cpu":
                        quota = int((current / "cpu.cfs_quota_us").read_text())
                        if quota > 0:
                            cpus.append(
                                quota / int((current / "cpu.cfs_period_us").read_text())
                            )
                except OSError:
                    pass
                try:
                    names = (
                        ("memory.max", "memory.current")
                        if kind == "v2"
                        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
                    )
                    if kind in ("v2", "memory"):
                        limit = (current / names[0]).read_text().strip()
                        if limit != "max":
                            available.append(
                                max(
                                    0,
                                    int(limit) - int((current / names[1]).read_text()),
                                )
                            )
                except OSError:
                    pass
                if current == root:
                    break
                if root not in current.parents:
                    break
                current = current.parent
    return cpus, available


def resources():
    """Read physical-core, affinity, cgroup and available-memory budgets."""
    affinity = os.sched_getaffinity(0)
    physical = set()
    for cpu in affinity:
        top = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        physical.add(
            (
                (top / "physical_package_id").read_text().strip(),
                (top / "core_id").read_text().strip(),
            )
        )
    cpu_limits, mem_limits = cgroup_limits()
    mem = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    available = int(mem["MemAvailable"].split()[0]) * 1024
    return dict(
        physical_cpus=len(physical),
        affinity_cpus=len(affinity),
        cpu_budget=min([float(len(physical)), *cpu_limits]),
        memory_available=min([available, *mem_limits]),
        load1=os.getloadavg()[0],
    )


def validate_resources(limit, reserve_cpus, reserve_memory, worker_memory, poll=None):
    """Validate either CLI GiB or internal byte budgets before admission or writes."""
    if limit != "auto" and (type(limit) is not int or limit <= 0):
        raise ValueError("Process limit must be auto or a positive integer")
    for name, value, positive_only in (
        ("CPU reserve", reserve_cpus, False),
        ("memory reserve", reserve_memory, False),
        ("worker memory budget", worker_memory, True),
        ("poll interval", poll, True),
    ):
        if name == "poll interval" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (positive_only and value == 0)
        ):
            bound = "positive" if positive_only else "nonnegative"
            raise ValueError(f"{name} must be finite and {bound}")


def pending_memory(live, owners, default_budget):
    """Reserve each worker's remaining admitted budget, without pooling RSS.

    Older owner records have no recoverable budget. Wait for those processes
    to exit; a new caller's smaller budget must not replace their reservation.
    Unregistered processes use the current conservative worker budget.
    """
    budgets = {
        (owner["pid"], owner["start"]): owner.get("rss_budget") for owner in owners
    }
    pending = 0
    unbudgeted = []
    for process in live:
        key = (process["pid"], process["start"])
        budget = budgets.get(key, default_budget)
        if budget is None:
            unbudgeted.append(process["pid"])
            continue
        validate_resources("auto", 0, 0, budget)
        pending += max(0, budget - process["rss"])
    return pending, unbudgeted


def admission(
    info,
    live_count,
    limit,
    rss_budget,
    reserve_cpus,
    reserve_bytes,
    live_rss=0,
    *,
    pending_bytes=None,
):
    """Check host headroom after accounting for live worker reservations."""
    validate_resources(limit, reserve_cpus, reserve_bytes, rss_budget)
    # Approximate external CPU pressure conservatively; gem5 jobs already count
    # against cap below. Explicit cap overrides auto CPU sizing, not memory.
    external_load = max(0.0, info["load1"] - live_count)
    available_cpus = max(
        0, math.floor(info["cpu_budget"] - reserve_cpus - external_load)
    )
    cap = (
        available_cpus
        if limit == "auto"
        else min(limit, max(1, math.floor(info["cpu_budget"])))
    )
    # MemAvailable already reflects resident pages. Reserve the remaining
    # budget for admitted jobs whose resident set has not reached that budget.
    # Direct callers may describe equal-budget workers in aggregate. Launch
    # supplies per-owner reservations so different budgets never get pooled.
    if pending_bytes is None:
        pending_bytes = max(0, live_count * rss_budget - live_rss)
    if (
        isinstance(pending_bytes, bool)
        or not isinstance(pending_bytes, (int, float))
        or not math.isfinite(pending_bytes)
        or pending_bytes < 0
    ):
        raise ValueError("Pending memory reservation must be finite and nonnegative")
    usable_bytes = info["memory_available"] - reserve_bytes - pending_bytes
    memory_slots = max(0, math.floor(usable_bytes / rss_budget))
    return dict(
        cap=cap,
        live=live_count,
        memory_slots=memory_slots,
        admitted=live_count < cap and memory_slots > 0,
        rss_budget=rss_budget,
        pending_bytes=pending_bytes,
        external_load=external_load,
        **info,
    )


def inspect_workers(state, executable):
    """Combine discovered workers with live PID/start identities from our state."""
    live = processes(executable)
    owners_path = state / "active-limits.json"
    owners = json.loads(owners_path.read_text()) if owners_path.exists() else []
    active = []
    for owner in owners:
        process = proc_info(owner["pid"])
        if process and process["start"] == owner["start"]:
            active.append(owner)
            if not any(item["pid"] == process["pid"] for item in live):
                live.append(process)
    return live, active


def require_unused_output(output, share, live):
    """Refuse an existing reservation or a directory/share used by a live worker."""
    if (output / "launch.json").exists():
        raise RuntimeError(
            f"{output} already has a launch reservation; inspect/recover, do not relaunch"
        )
    if any(
        str(output) in process["argv"] or (share and share in process["argv"])
        for process in live
    ):
        raise RuntimeError("Output directory or private share is already used by gem5")


def decide_admission(args, state, output, live, owners):
    """Keep live caps and individual budgets while sizing the next worker."""
    caps = [owner["limit"] for owner in owners if owner["limit"] != "auto"]
    if args.limit != "auto":
        caps.append(args.limit)
    limit = min(caps) if caps else "auto"
    history = state / "rss-high-water.json"
    old = json.loads(history.read_text()) if history.exists() else {"bytes": 0}
    peak = max([old["bytes"], *(process["hwm"] for process in live)])
    save(history, {"bytes": peak})
    rss_budget = max(args.rss_gib * GIB, math.ceil(peak * 1.25))
    pending_bytes, unbudgeted = pending_memory(live, owners, rss_budget)
    decision = admission(
        resources(),
        len(live),
        limit,
        rss_budget,
        args.reserve_cpus,
        args.reserve_gib * GIB,
        pending_bytes=pending_bytes,
    )
    if unbudgeted:
        decision.update(admitted=False, memory_slots=0)
    decision.update(
        time=time.time(),
        out=str(output),
        processes=live,
        unbudgeted_owners=unbudgeted,
    )
    with (state / "admission.jsonl").open("a") as log:
        log.write(json.dumps(decision) + "\n")
    return decision


def record_child_launch(child, *, state, output, owners, reservation, limit):
    """Persist either a confirmed exit or a live PID/start identity.

    A short configuration check can finish before its parent is scheduled again.
    Poll even when /proc has no live identity: only Popen's exit status confirms
    that our child finished. Missing identity alone must remain fail-closed.
    """
    identity = proc_info(child.pid)
    exit_code = child.poll()
    if exit_code is not None:
        save(
            output / "launch.json",
            dict(
                reservation,
                status="exited",
                pid=child.pid,
                exit_code=exit_code,
                finish=time.time(),
            ),
        )
        return
    if identity is None:
        raise RuntimeError("Cannot establish live child identity; inspect stdout")
    save(
        state / "pending-launch.json",
        dict(reservation, requested_limit=limit, process=identity),
    )
    (output / "pid").write_text(str(child.pid) + "\n")
    owners.append(
        dict(
            pid=child.pid,
            start=identity["start"],
            limit=limit,
            rss_budget=reservation["rss_budget"],
        )
    )
    save(state / "active-limits.json", owners)
    save(output / "launch.json", dict(reservation, status="launched", process=identity))


def start_reserved_process(args, command, *, state, output, owners, decision):
    """Reserve once, spawn once, and retain diagnostic state on any launch failure.

    The caller holds the admission lock and has just revalidated frozen inputs.
    A failed state write may occur after the child starts: never retry implicitly.
    """
    reservation = dict(
        status="reserved", command=command, cwd=str(args.cwd), **decision
    )
    save(output / "launch.json", reservation)
    # The shared record exists before Popen. A failed PID/owner write can never
    # turn our child into an unregistered worker with a guessed smaller budget.
    pending = state / "pending-launch.json"
    save(pending, dict(reservation, requested_limit=args.limit))
    try:
        with (output / "gem5_stdout.log").open("xb") as stdout:
            child = subprocess.Popen(
                command,
                cwd=args.cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=getattr(args, "environment", None),
                close_fds=True,
            )
        record_child_launch(
            child,
            state=state,
            output=output,
            owners=owners,
            reservation=reservation,
            limit=args.limit,
        )
        # A durable terminal receipt is sufficient for an already reaped child;
        # it never acquires a live slot. wait() still returns its real exit code.
        pending.unlink()
        sync_directory(state)
    except BaseException as error:
        save(
            output / "launch.json",
            dict(reservation, status="needs-inspection", error=str(error)),
        )
        raise
    print(
        f'[slots] pid={child.pid} cap={decision["cap"]} '
        f'RSS-budget={decision["rss_budget"] / GIB:.1f} GiB',
        flush=True,
    )
    return child


def launch(args):
    """Inspect, revalidate, reserve and spawn under one cross-runner lock."""
    validate_resources(
        args.limit, args.reserve_cpus, args.reserve_gib, args.rss_gib, args.poll
    )
    if not Path("/proc/self/stat").exists():
        raise RuntimeError("Simulation admission requires Linux /proc")
    output = args.out.resolve()
    state = args.state.expanduser()
    state.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("Missing simulation command")
    executable = Path(command[0]).name
    share = next((value for value in command if value.startswith("--vio-9p=")), None)
    while True:
        with (state / "admission.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            require_no_pending_launch(state)
            live, owners = inspect_workers(state, executable)
            require_unused_output(output, share, live)
            decision = decide_admission(args, state, output, live, owners)
            if decision["admitted"]:
                # Admission can wait indefinitely. Check again before reserving.
                pre_launch = getattr(args, "pre_launch", None)
                if pre_launch is not None:
                    pre_launch()
                child = start_reserved_process(
                    args,
                    command,
                    state=state,
                    output=output,
                    owners=owners,
                    decision=decision,
                )
                return child if getattr(args, "return_child", False) else None
            print(
                f'[slots] waiting: live={len(live)} cap={decision["cap"]} '
                f'memory-slots={decision["memory_slots"]} '
                f'unbudgeted-owners={decision["unbudgeted_owners"]}',
                flush=True,
            )
        time.sleep(args.poll)


def main():
    """Validate CLI budgets and optionally wait for the admitted child."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state", type=Path, default=Path.home() / ".cache/i2ce/simulation-slots"
    )
    p.add_argument("--limit", type=positive, default="auto")
    p.add_argument("--reserve-cpus", type=float, default=4)
    p.add_argument("--reserve-gib", type=float, default=16)
    p.add_argument(
        "--rss-gib",
        type=float,
        default=8,
        help="Conservative initial HOST RSS budget, not guest RAM",
    )
    p.add_argument("--poll", type=float, default=20)
    p.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the admitted child and propagate its exit code",
    )
    p.add_argument("--cwd", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    try:
        validate_resources(
            args.limit, args.reserve_cpus, args.reserve_gib, args.rss_gib, args.poll
        )
    except ValueError as error:
        p.error(str(error))
    args.return_child = args.wait
    child = launch(args)
    if args.wait:
        raise SystemExit(child.wait())


if __name__ == "__main__":
    main()
