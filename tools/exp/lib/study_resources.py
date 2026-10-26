#!/usr/bin/env python3
"""Fresh admission for frozen studies, extending the shared runner policy."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from study_records import require
import simulation_slots

GIB = 1024**3
POLICY = dict(
    sample_seconds=3,
    cpu_reserve=4,
    memory_reserve=16 * GIB,
    scratch_reserve=16 * GIB,
    package_reserve=4 * GIB,
    minimum_cpu_per_run=2,
    minimum_memory_per_run=8 * GIB,
    disk_per_run=4 * GIB,
    io_some_max=0.20,
    io_full_max=0.05,
    probe_bytes=8 * 1024**2,
    probe_seconds_max=5,
)


def aggregate_sample():
    cpus = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields[0].startswith("cpu") and fields[0][3:].isdigit():
            cpus[int(fields[0][3:])] = [int(x) for x in fields[1:9]]
    vm = dict(
        line.split()[:2] for line in Path("/proc/vmstat").read_text().splitlines()
    )
    pressure = {}
    try:
        for line in Path("/proc/pressure/io").read_text().splitlines():
            fields = line.split()
            pressure[fields[0]] = dict(x.split("=") for x in fields[1:])
    except FileNotFoundError:
        pressure = None
    # simulation_slots checks ownership before reading any process contents.
    processes = simulation_slots.processes("gem5.fast") + simulation_slots.processes(
        "gem5.opt"
    )
    sampled = []
    for item in processes:
        try:
            stat = (
                (Path("/proc") / str(item["pid"]) / "stat")
                .read_text()
                .rsplit(")", 1)[1]
                .split()
            )
        except (FileNotFoundError, ProcessLookupError):
            continue
        if stat[19] != item["start"]:
            continue
        item["cpu_ticks"] = int(stat[11]) + int(stat[12])
        sampled.append(item)
    return dict(
        monotonic=time.monotonic(),
        cpus=cpus,
        io=pressure,
        swap=int(vm["pswpin"]) + int(vm["pswpout"]),
        processes=sampled,
    )


def busy_cpus(first, second, affinity):
    busy = wait = total = 0
    for cpu in affinity:
        delta = [b - a for a, b in zip(first[cpu], second[cpu])]
        if len(delta) != 8 or any(x < 0 for x in delta) or sum(delta) <= 0:
            raise ValueError("Invalid aggregate CPU sample")
        busy += sum(delta[i] for i in (0, 1, 2, 5, 6, 7)) / sum(delta)
        wait += delta[4]
        total += sum(delta)
    return busy, wait / total


def capacity(
    cores,
    busy,
    available,
    memory_pending,
    free_disk,
    package_free,
    active,
    per_cpu,
    per_memory,
    io_passed,
):
    cpu_slots = max(
        0, math.floor((cores - POLICY["cpu_reserve"] - math.ceil(busy)) / per_cpu)
    )
    memory_slots = max(
        0, (available - POLICY["memory_reserve"] - memory_pending) // per_memory
    )
    # Reserve remaining output growth for every live simulation as well.
    disk_slots = max(
        0, (free_disk - POLICY["scratch_reserve"]) // POLICY["disk_per_run"] - active
    )
    passed = io_passed and package_free >= POLICY["package_reserve"]
    slots = min(cpu_slots, memory_slots, disk_slots) if passed else 0
    return dict(
        additional_slots=int(slots),
        cpu_slots=cpu_slots,
        memory_slots=int(memory_slots),
        disk_slots=int(disk_slots),
        admitted=slots >= 1,
        fixed_concurrency_cap=None,
    )


def probe(directory):
    block = os.urandom(1024**2)
    with tempfile.TemporaryFile(dir=directory) as stream:
        started = time.monotonic()
        for _ in range(8):
            stream.write(block)
        stream.flush()
        os.fsync(stream.fileno())
        write_seconds = time.monotonic() - started
        stream.seek(0)
        started = time.monotonic()
        for _ in range(8):
            require(
                stream.read(len(block)) == block,
                "Study check failed: stream.read(len(block)) == block",
            )
        require(not stream.read(1), "Study check failed: not stream.read(1)")
        read_seconds = time.monotonic() - started
    return dict(
        passed=max(write_seconds, read_seconds) < POLICY["probe_seconds_max"],
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        bytes=POLICY["probe_bytes"],
        scope="bounded fsync and cached read; not sustained bandwidth",
    )


def sample_admission(scratch, package, observed_peak=0):
    first = aggregate_sample()
    time.sleep(POLICY["sample_seconds"])
    second = aggregate_sample()
    elapsed = second["monotonic"] - first["monotonic"]
    resources = simulation_slots.resources()
    affinity = sorted(os.sched_getaffinity(0))
    busy, iowait = busy_cpus(first["cpus"], second["cpus"], affinity)
    previous = {(p["pid"], p["start"]): p for p in first["processes"]}
    hz = os.sysconf("SC_CLK_TCK")
    for item in second["processes"]:
        old = previous.get((item["pid"], item["start"]))
        item["cpu_equivalents"] = (
            max(0, (item["cpu_ticks"] - old["cpu_ticks"]) / hz / elapsed) if old else 0
        )
    live = second["processes"]
    per_cpu = max(
        [POLICY["minimum_cpu_per_run"], *[2 * p["cpu_equivalents"] for p in live]]
    )
    per_memory = max(
        [
            POLICY["minimum_memory_per_run"],
            2 * observed_peak + 2 * GIB,
            *[2 * p["hwm"] + 2 * GIB for p in live],
        ]
    )
    busy += sum(max(0, per_cpu - p["cpu_equivalents"]) for p in live)
    pending = sum(max(0, per_memory - p["rss"]) for p in live)
    checks = dict(no_swapping=second["swap"] == first["swap"])
    if first["io"] is not None and second["io"] is not None:
        for key, limit in [
            ("some", POLICY["io_some_max"]),
            ("full", POLICY["io_full_max"]),
        ]:
            old, new = first["io"][key], second["io"][key]
            delta = int(new["total"]) - int(old["total"])
            if delta < 0:
                raise ValueError("I/O pressure counter regressed")
            checks["io_" + key] = (
                max(delta / (elapsed * 1e6), float(new["avg10"]) / 100) < limit
            )
    try:
        result = subprocess.run(
            [sys.executable, "-B", __file__, "--probe", str(scratch)],
            text=True,
            capture_output=True,
            timeout=12,
        )
        io_probe = (
            json.loads(result.stdout)
            if result.returncode == 0
            else dict(passed=False, error=result.stderr[-1000:])
        )
    except subprocess.TimeoutExpired:
        io_probe = dict(passed=False, error="Own filesystem probe exceeded 12 seconds")
    checks["filesystem_probe"] = io_probe["passed"]
    decision = capacity(
        resources["cpu_budget"],
        busy,
        resources["memory_available"],
        pending,
        shutil.disk_usage(scratch).free,
        shutil.disk_usage(package).free,
        len(live),
        per_cpu,
        per_memory,
        all(checks.values()),
    )
    return dict(
        epoch=time.time(),
        policy=POLICY,
        resources=resources,
        sampled_seconds=elapsed,
        busy_core_equivalents=busy,
        iowait_diagnostic_only=iowait,
        own_active_simulations=live,
        memory_pending=pending,
        per_run_cpu=per_cpu,
        per_run_memory=per_memory,
        io_checks=checks,
        filesystem_probe=io_probe,
        **decision,
    )


if __name__ == "__main__":
    require(
        len(sys.argv) == 3 and sys.argv[1] == "--probe",
        "Study check failed: len(sys.argv) == 3 and sys.argv[1] == '--probe'",
    )
    print(json.dumps(probe(Path(sys.argv[2]))))
