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

GIB = 1024 ** 3


def save(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def positive(value):
    if value == 'auto':
        return value
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('limit must be auto or a positive integer') from None
    if number < 1:
        raise argparse.ArgumentTypeError('zero is not unlimited; use auto or a positive integer')
    return number


def proc_info(pid, proc=Path('/proc')):
    try:
        p = proc / str(pid)
        # comm may contain spaces or parentheses; fields after its closing ')'
        # start at stat field 3, making starttime (field 22) index 19.
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        status = dict(line.split(':', 1) for line in (p / 'status').read_text().splitlines() if ':' in line)
        return dict(pid=int(pid), start=fields[19], comm=(p / 'comm').read_text().strip(),
                    argv=(p / 'cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0'),
                    rss=int(status.get('VmRSS', '0 kB').split()[0]) * 1024,
                    hwm=int(status.get('VmHWM', '0 kB').split()[0]) * 1024)
    except (OSError, ValueError, IndexError):
        return None


def processes(name):
    result = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            item = proc_info(p.name)
            if item and item['comm'] == name[:15]:
                result.append(item)
    return result


def cgroup_limits():
    """Read v1/v2 limits at the current cgroup AND every ancestor."""
    cpus, available = [], []
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, relative = line.split(':', 2)
        versions = [('v2', Path('/sys/fs/cgroup'))] if not controllers else []
        if 'cpu' in controllers.split(','):
            versions.append(('cpu', Path('/sys/fs/cgroup/cpu')))
        if 'memory' in controllers.split(','):
            versions.append(('memory', Path('/sys/fs/cgroup/memory')))
        for kind, root in versions:
            current = root / relative.lstrip('/')
            while True:
                try:
                    if kind == 'v2':
                        quota, period = (current / 'cpu.max').read_text().split()
                        if quota != 'max':
                            cpus.append(int(quota) / int(period))
                    elif kind == 'cpu':
                        quota = int((current / 'cpu.cfs_quota_us').read_text())
                        if quota > 0:
                            cpus.append(quota / int((current / 'cpu.cfs_period_us').read_text()))
                except OSError:
                    pass
                try:
                    names = ('memory.max', 'memory.current') if kind == 'v2' else ('memory.limit_in_bytes', 'memory.usage_in_bytes')
                    if kind in ('v2', 'memory'):
                        limit = (current / names[0]).read_text().strip()
                        if limit != 'max':
                            available.append(max(0, int(limit) - int((current / names[1]).read_text())))
                except OSError:
                    pass
                if current == root:
                    break
                if root not in current.parents:
                    break
                current = current.parent
    return cpus, available


def resources():
    affinity = os.sched_getaffinity(0)
    physical = set()
    for cpu in affinity:
        top = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.add(((top / 'physical_package_id').read_text().strip(),
                      (top / 'core_id').read_text().strip()))
    cpu_limits, mem_limits = cgroup_limits()
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(mem['MemAvailable'].split()[0]) * 1024
    return dict(physical_cpus=len(physical), affinity_cpus=len(affinity),
                cpu_budget=min([float(len(physical)), *cpu_limits]),
                memory_available=min([available, *mem_limits]), load1=os.getloadavg()[0])


def admission(info, live_count, limit, rss_budget, reserve_cpus, reserve_bytes):
    # Approximate external CPU pressure conservatively; gem5 jobs already count
    # against cap below. Explicit cap overrides auto CPU sizing, not memory.
    external_load = max(0.0, info['load1'] - live_count)
    available_cpus = max(0, math.floor(info['cpu_budget'] - reserve_cpus - external_load))
    cap = available_cpus if limit == 'auto' else min(limit, max(1, math.floor(info['cpu_budget'])))
    memory_slots = max(0, math.floor((info['memory_available'] - reserve_bytes) / rss_budget))
    return dict(cap=cap, live=live_count, memory_slots=memory_slots,
                admitted=live_count < cap and memory_slots > 0,
                rss_budget=rss_budget, external_load=external_load, **info)


def launch(args):
    if not Path('/proc/self/stat').exists():
        raise RuntimeError('Simulation admission requires Linux /proc')
    out = args.out.resolve()
    state = args.state.expanduser()
    state.mkdir(parents=True, exist_ok=True)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise RuntimeError('Missing simulation command')
    name = Path(command[0]).name
    share = next((v for v in command if v.startswith('--vio-9p=')), None)
    while True:
        with (state / 'admission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            live = processes(name)
            if (out / 'launch.json').exists():
                raise RuntimeError(f'{out} already has a launch reservation; inspect/recover, do not relaunch')
            if any(str(out) in p['argv'] or (share and share in p['argv']) for p in live):
                raise RuntimeError('Output directory or private share is already used by gem5')
            history = state / 'rss-high-water.json'
            old = json.loads(history.read_text()) if history.exists() else {'bytes': 0}
            peak = max([old['bytes'], *(p['hwm'] for p in live)])
            save(history, {'bytes': peak})
            rss_budget = max(args.rss_gib * GIB, math.ceil(peak * 1.25))
            decision = admission(resources(), len(live), args.limit, rss_budget,
                                 args.reserve_cpus, args.reserve_gib * GIB)
            decision.update(time=time.time(), out=str(out), processes=live)
            with (state / 'admission.jsonl').open('a') as log:
                log.write(json.dumps(decision) + '\n')
            if decision['admitted']:
                reservation = dict(status='reserved', command=command, cwd=str(args.cwd), **decision)
                save(out / 'launch.json', reservation)
                try:
                    with (out / 'gem5_stdout.log').open('xb') as stdout:
                        child = subprocess.Popen(command, cwd=args.cwd, stdin=subprocess.DEVNULL,
                                                 stdout=stdout, stderr=subprocess.STDOUT,
                                                 start_new_session=True, close_fds=True)
                    identity = proc_info(child.pid)
                    if identity is None or child.poll() is not None:
                        raise RuntimeError('gem5 exited during launch; inspect stdout')
                    (out / 'pid').write_text(str(child.pid) + '\n')
                    save(out / 'launch.json', dict(reservation, status='launched', process=identity))
                except BaseException as error:
                    # A reservation is retained for recovery even if a process
                    # started just before failure; never silently launch twice.
                    save(out / 'launch.json', dict(reservation, status='needs-inspection', error=str(error)))
                    raise
                print(f'[slots] pid={child.pid} cap={decision["cap"]} RSS-budget={rss_budget / GIB:.1f} GiB', flush=True)
                return
            print(f'[slots] waiting: live={len(live)} cap={decision["cap"]} memory-slots={decision["memory_slots"]}', flush=True)
        time.sleep(args.poll)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=Path.home() / '.cache/i2ce/simulation-slots')
    p.add_argument('--limit', type=positive, default='auto')
    p.add_argument('--reserve-cpus', type=float, default=4)
    p.add_argument('--reserve-gib', type=float, default=16)
    p.add_argument('--rss-gib', type=float, default=8, help='Conservative initial HOST RSS budget, not guest RAM')
    p.add_argument('--poll', type=float, default=20)
    p.add_argument('--cwd', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if min(args.reserve_cpus, args.reserve_gib) < 0 or min(args.rss_gib, args.poll) <= 0:
        p.error('Resource reserves must be nonnegative; RSS budget/poll must be positive')
    launch(args)


if __name__ == '__main__':
    main()
