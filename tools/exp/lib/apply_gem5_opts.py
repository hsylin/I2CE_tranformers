#!/usr/bin/env python3
"""
Add per-run CLI options to gem5's configs/example/arm/starter_fs.py:

    --sve-vl N       SVE vector length in 128-bit units (replaces the
                     hard-coded `root.system.sve_vl = 1`)
    --l1i-size S     override L1 instruction cache size (default 48KiB)
    --l1d-size S     override L1 data cache size        (default 32KiB)
    --l2-size  S     override shared L2 cache size      (default 1MiB)
    --roofline-monitors  observe Core–L1D and L1D–L2 payloads (off by default)

Written against gem5 25.1 (Jerry0209/gem5 @ ed752e78, the tree this project
uses). Idempotent, and each edit is detected independently, so a tree that
already has the --sve-vl option (e.g. from the earlier apply_sve_vl.py patch)
only gains the cache pieces — options are never registered twice. The
unmodified file is saved once as <config>.orig. The result is compile-checked;
on failure the original is restored and the script exits non-zero.

Usage:  apply_gem5_opts.py /path/to/gem5/configs/example/arm/starter_fs.py
"""
import py_compile
import shutil
import sys
from pathlib import Path

ARG_ANCHOR = '    parser.add_argument("--restore", type=str, default=None)'

SVE_OPT = """
    # ---- --sve-vl option (exp.sh patch-gem5) ----
    parser.add_argument(
        "--sve-vl", type=int, default=1, choices=[1, 2, 4, 8, 16],
        help="SVE vector length in 128-bit units: 1=128b, 2=256b, 4=512b",
    )"""

CACHE_OPTS = """
    # ---- cache-size options (exp.sh patch-gem5) ----
    parser.add_argument(
        "--l1i-size", type=str, default=None,
        help="override L1I cache size, e.g. 48KiB (timing CPUs only)",
    )
    parser.add_argument(
        "--l1d-size", type=str, default=None,
        help="override L1D cache size, e.g. 32KiB (timing CPUs only)",
    )
    parser.add_argument(
        "--l2-size", type=str, default=None,
        help="override shared L2 cache size, e.g. 1MiB (timing CPUs only)",
    )"""

SVE_OLD = "    root.system.sve_vl = 1"
SVE_NEW = "    root.system.sve_vl = args.sve_vl  # exp.sh patch-gem5: was hard-coded 1"

CACHE_ANCHOR = "    system.addCaches(want_caches, last_cache_level=2)"
ROOFLINE_OPT = """
    parser.add_argument(
        "--roofline-monitors", action="store_true",
        help="native CommMonitor bytes at Core-L1D and L1D-L2 (single timing CPU)",
    )"""
ROOFLINE_BLOCK = """

    # ---- optional native Roofline monitors ----
    if args.roofline_monitors:
        from m5.objects import CommMonitor
        if (not want_caches or len(system.cpu_cluster) != 1
                or len(system.cpu_cluster[0].cpus) != 1):
            raise ValueError("Roofline monitors require one timing CPU and L1/L2 caches")
        if hasattr(system, "roofline_l1") or hasattr(system, "roofline_l2"):
            raise ValueError("Roofline monitors already attached")
        for monitor_name in ("roofline_l1", "roofline_l2"):
            setattr(system, monitor_name, CommMonitor(
                sample_period="1s", disable_bandwidth_hists=False,
                disable_burst_length_hists=True, disable_latency_hists=True,
                disable_itt_dists=True, disable_outstanding_hists=True,
                disable_transaction_hists=True, disable_addr_dists=True,
            ))
        cpu = system.cpu_cluster[0].cpus[0]
        cpu.dcache_port.splice(
            system.roofline_l1.cpu_side_port, system.roofline_l1.mem_side_port)
        cpu.dcache.mem_side.splice(
            system.roofline_l2.cpu_side_port, system.roofline_l2.mem_side_port)
    # ---- end native Roofline monitors ----
"""
CACHE_BLOCK = """    system.addCaches(want_caches, last_cache_level=2)

    # ---- per-run cache size overrides (exp.sh patch-gem5) ----
    # devices.py fixes L1I=48KiB, L1D=32KiB, L2=1MiB as class attributes;
    # these flags resize the instantiated caches of every cluster instead.
    if want_caches and (args.l1i_size or args.l1d_size or args.l2_size):
        for _cluster in system.cpu_cluster:
            for _cpu in _cluster.cpus:
                if args.l1i_size and getattr(_cpu, "icache", None):
                    _cpu.icache.size = args.l1i_size
                if args.l1d_size and getattr(_cpu, "dcache", None):
                    _cpu.dcache.size = args.l1d_size
            if args.l2_size and getattr(_cluster, "l2", None):
                _cluster.l2.size = args.l2_size
    # ---- end cache size overrides ----"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    cfg = Path(sys.argv[1]).resolve()
    if not cfg.is_file():
        print(f"ERROR: {cfg} not found", file=sys.stderr)
        return 2

    original = cfg.read_text()
    text = original
    done, todo = [], []

    # 1. argparse options — each piece checked on its own, so a file already
    #    carrying --sve-vl (earlier apply_sve_vl.py) is never given a duplicate.
    need_sve_opt = '"--sve-vl"' not in text
    need_cache_opts = '"--l1d-size"' not in text
    need_roofline_opt = '"--roofline-monitors"' not in text
    if not need_sve_opt:
        done.append("--sve-vl option already registered")
    if not need_cache_opts:
        done.append("cache-size options already registered")
    insert = ""
    if need_sve_opt:
        insert += SVE_OPT
    if need_cache_opts:
        insert += CACHE_OPTS
    if need_roofline_opt:
        insert += ROOFLINE_OPT
    if insert:
        if ARG_ANCHOR not in text:
            print(
                "ERROR: could not find the --restore add_argument anchor; "
                "this starter_fs.py differs from gem5 25.1 @ ed752e78 — "
                "patch it manually.",
                file=sys.stderr,
            )
            return 3
        text = text.replace(ARG_ANCHOR, ARG_ANCHOR + insert, 1)
        if need_sve_opt:
            todo.append("added --sve-vl option")
        if need_cache_opts:
            todo.append("added --l1i-size / --l1d-size / --l2-size options")
        if need_roofline_opt:
            todo.append("added optional --roofline-monitors flag")

    # 2. sve_vl assignment
    if "root.system.sve_vl = args.sve_vl" in text:
        done.append("sve_vl already reads args.sve_vl")
    elif SVE_OLD in text:
        text = text.replace(SVE_OLD, SVE_NEW, 1)
        todo.append("root.system.sve_vl now follows --sve-vl")
    else:
        print(
            "ERROR: could not find `root.system.sve_vl = 1` to replace.",
            file=sys.stderr,
        )
        return 3

    # 3. cache size application
    if "per-run cache size overrides" in text:
        done.append("cache override block already present")
    elif CACHE_ANCHOR in text:
        text = text.replace(CACHE_ANCHOR, CACHE_BLOCK, 1)
        todo.append("cache sizes applied after system.addCaches()")
    else:
        print("ERROR: could not find the system.addCaches anchor.", file=sys.stderr)
        return 3

    if "# ---- optional native Roofline monitors ----" not in text:
        text = text.replace(CACHE_ANCHOR, CACHE_ANCHOR + ROOFLINE_BLOCK, 1)
        todo.append("added opt-in native byte monitors; default hardware is unchanged")

    for msg in done:
        print(f"[patch-gem5] skip: {msg}")
    if not todo:
        print("[patch-gem5] nothing to do — config already patched")
        return 0

    backup = cfg.with_suffix(cfg.suffix + ".orig")
    if not backup.exists():
        shutil.copy2(cfg, backup)
        print(f"[patch-gem5] original saved as {backup}")

    cfg.write_text(text)
    try:
        py_compile.compile(str(cfg), doraise=True)
    except py_compile.PyCompileError as e:
        cfg.write_text(original)
        print(f"ERROR: patched file does not compile — reverted. {e}", file=sys.stderr)
        return 4

    # a registered-twice option would crash gem5 at startup; make it impossible
    for opt in (
        '"--sve-vl"',
        '"--l1i-size"',
        '"--l1d-size"',
        '"--l2-size"',
        '"--roofline-monitors"',
    ):
        if cfg.read_text().count(opt) > 1:
            cfg.write_text(original)
            print(
                f"ERROR: option {opt} would be registered twice — reverted.",
                file=sys.stderr,
            )
            return 5

    for msg in todo:
        print(f"[patch-gem5] done: {msg}")
    print(f"[patch-gem5] OK: {cfg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
