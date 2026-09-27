#!/usr/bin/env python3
"""
Report vector-register spilling in an aarch64 binary, per function.

A spill is the register allocator running out of vector registers and parking a
value on the stack. One in a function prologue is ordinary ABI cost; one in the
innermost loop of a GEMM kernel is a store and a reload on every iteration, and
is usually worth restructuring the kernel to remove.

What counts as a spill here is a vector load/store whose address is based on sp
or x29:

    str  z3, [sp, #2, mul vl]     SVE  Z  spill
    ldr  p1, [x29, #6, mul vl]    SVE  P  fill
    stp  q8, q9, [sp, #16]        NEON Q  spill
    addvl sp, sp, #-3             a scalable stack frame, which only exists
                                  because something had to be spilled

Two things are then subtracted, because otherwise every kernel looks guilty:

  * The low 64 bits of v8-v15 and the predicates p4-p15 are callee-saved under
    AAPCS64, so any function that uses them must save and restore them. That is
    the cost of a call, not the allocator running out of registers, and it is
    reported in its own "abi" column.
  * A store nothing ever reads back is a local being written, not a spill. Each
    store is paired with a later load of the same stack slot, and only those
    pairs are counted as spills.

What survives can still be a false positive: writing a vector to the stack and
reading it straight back as a different type is a normal lane-extraction idiom
and has exactly the shape of a spill. So zero is conclusive and a non-zero
count is a place to look, which --show prints.

Loop depth comes from backward branches: every branch whose target is an
earlier address in the same function opens a loop body, and depth is how many
of those bodies contain the instruction.

Static binaries link all of libc, whose memcpy and friends are full of NEON
stack traffic that says nothing about this project. Functions are therefore
matched against --filter (the project's kernels by default); --all disables it.

    check_spills.py transformer.o
    check_spills.py 'executable archive/report_experiments_executables/'*.o
    check_spills.py a.o b.o --filter 'sve_gemm_row'      # A/B two builds
    check_spills.py transformer.o --json

Runs on the python3 that is already on PATH (3.6 and newer); no environment
needs activating.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Tuple

# Functions this project cares about. libc is the rest of a static binary.
DEFAULT_FILTER = r"gemm|transformer|sve_|softmax|addnorm|codebook"

FUNC_RE = re.compile(r"^([0-9a-f]+)\s+<(.+)>:$")
INSN_RE = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+)*\s*(\.?[a-z][a-z0-9.]*)\s*(.*)$")
# What a disassembler prints when it does not know the encoding. An objdump
# too old for SVE emits these instead of `mla z4.s, ...` -- and then a spill
# it cannot decode is invisible, so the tool would report a clean binary.
UNDECODED = {".inst", ".word", ".short", ".byte", ".long"}
# A branch's target address, when objdump resolved it to one.
BRANCH_RE = re.compile(r"^(b|bl|br|b\.[a-z]+|cbz|cbnz|tbz|tbnz)$")
TARGET_RE = re.compile(r"\b([0-9a-f]{4,})\s+<")

# Walking back from a `ret`, these are the only things an epilogue contains.
# A fixed instruction window would swallow the whole body of a short function.
EPILOGUE_OPS = {"ret", "ldp", "ldr", "add", "sub", "mov", "addvl", "addpl",
                "nop", "autiasp", "hint", "ldraa", "ldrab"}

VEC_MOVE = {"str", "ldr", "stp", "ldp"}
VEC_REG_RE = re.compile(r"^\s*([zpqdsvh])(\d+)\b")
STACK_BASE_RE = re.compile(r"\[\s*(sp|x29)\b")
# The exact stack slot, so a store can be matched with the load that reads it
# back. "[sp]", "[sp, #112]", "[sp, #2, mul vl]" all have to normalise.
SLOT_RE = re.compile(r"\[\s*(sp|x29)\s*(?:,\s*#(-?\d+)\s*(?:,\s*mul\s+vl)?)?\s*\]")
STORES = {"str", "stp", "st1b", "st1h", "st1w", "st1d"}
SVE_REG_RE = re.compile(r"\b[zp]\d+\b")


class Function:
    __slots__ = ("name", "start", "insns", "branches", "spills", "sve", "total",
                 "epilogue", "undecoded")

    def __init__(self, name: str, start: int):
        self.name = name
        self.start = start
        self.sve = 0
        self.total = 0
        self.epilogue: set = set()
        self.undecoded = 0
        self.insns: List[int] = []          # addresses, in order
        self.branches: List[Tuple[int, int]] = []   # (from, to) backward only
        self.spills: List[dict] = []

    def count_undecoded(self, pending) -> int:
        return sum(1 for _a, mn, ops in pending
                   if mn in UNDECODED or "(bad)" in ops or "undefined" in ops)

    def loop_ranges(self) -> List[Tuple[int, int]]:
        return [(to, frm) for frm, to in self.branches]

    def depth_at(self, addr: int) -> int:
        # A loop body is approximated by [branch target, branch]. Compilers put
        # cold blocks and epilogues inside such a span, so anything in the run
        # of instructions leading to a `ret` is treated as outside the loop --
        # otherwise every callee-save restore reads as an innermost-loop spill.
        if addr in self.epilogue:
            return 0
        return sum(1 for lo, hi in self.loop_ranges() if lo <= addr <= hi)


def objdump_version(objdump: str) -> str:
    try:
        out = subprocess.run([objdump, "--version"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        return out.stdout.splitlines()[0].strip()
    except Exception:
        return "unknown version"


# GNU binutils only. llvm-objdump prints immediates in hex and lays the line
# out differently, so the parsing below would quietly mis-read it -- and a
# tool that mis-reads is worse here than one that refuses.
CANDIDATES = ("aarch64-linux-gnu-objdump", "aarch64-none-linux-gnu-objdump",
              "aarch64-conda-linux-gnu-objdump", "objdump")


def version_tuple(text: str) -> Tuple[int, int]:
    m = re.search(r"(\d+)\.(\d+)", text)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def supports_aarch64(path: str) -> bool:
    if "llvm" in Path(path).name.lower():
        return False
    if "aarch64" in Path(path).name:
        return True
    try:                                    # a multi-target host objdump may do
        out = subprocess.run([path, "--info"], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
        return "aarch64" in out.stdout
    except Exception:
        return False


def find_objdump(explicit: Optional[str]) -> str:
    """Pick the NEWEST aarch64-capable objdump, not the first one on PATH.

    A machine can easily have an ancient system binutils shadowing a modern one
    in a conda environment -- CentOS 7 ships 2.27, which predates SVE entirely
    and silently disassembles every SVE instruction as `.inst 0x...`.
    """
    if explicit:
        return explicit
    found = []
    conda = os.environ.get("CONDA_PREFIX")
    for name in CANDIDATES:
        if conda:
            candidate = Path(conda) / "bin" / name
            if candidate.is_file():
                found.append(str(candidate))
        which = shutil.which(name)
        if which:
            found.append(which)
    seen, usable = set(), []
    for path in found:
        real = str(Path(path).resolve())
        if real in seen:
            continue
        seen.add(real)
        if supports_aarch64(path):
            usable.append(path)
    if not usable:
        sys.exit("no GNU aarch64 objdump found; pass --objdump /path/to/objdump\n"
                 "(a conda gem5 environment usually has "
                 "aarch64-conda-linux-gnu-objdump)")
    best = max(usable, key=lambda p: version_tuple(objdump_version(p)))
    # binutils only learned SVE in 2.28; CentOS 7 ships 2.27.
    if version_tuple(objdump_version(best)) < (2, 28):
        print(f"warning: {best} is {objdump_version(best)}, which predates SVE\n"
              "         support in binutils. Every SVE instruction will come back\n"
              "         undecoded. Activate an environment with a newer binutils,\n"
              "         or pass --objdump.", file=sys.stderr)
    return best


def disassemble(objdump: str, binary: Path) -> str:
    cmd = [objdump, "-d", "--no-show-raw-insn", str(binary)]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, check=True)
    except FileNotFoundError:
        sys.exit(f"cannot run {objdump}")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"{objdump} failed on {binary}:\n{exc.stderr.strip()[:400]}")
    if "aarch64" not in out.stdout[:4000] and "<" not in out.stdout[:4000]:
        sys.exit(f"{objdump} produced no disassembly for {binary}")
    return out.stdout


def slot_of(operands: str) -> Optional[str]:
    m = SLOT_RE.search(operands)
    if not m:
        return None
    base, off = m.group(1), m.group(2) or "0"
    return f"{base}{'+' if not off.startswith('-') else ''}{off}"


def classify(mnemonic: str, operands: str) -> Optional[str]:
    """Return the spill kind, or None when the instruction is not stack traffic."""
    if mnemonic in ("addvl", "addpl") and operands.lstrip().startswith("sp"):
        return "vl_frame"
    if mnemonic not in VEC_MOVE:
        return None
    reg = VEC_REG_RE.match(operands)
    if not reg or not STACK_BASE_RE.search(operands):
        return None
    letter, number = reg.group(1), int(reg.group(2))
    # AAPCS64: the low 64 bits of v8-v15 and the predicates p4-p15 are
    # callee-saved, so a function that merely uses them must save and restore
    # them. That is ABI cost every call pays, not the allocator running out of
    # registers, and counting it as a spill makes every kernel look guilty.
    if letter == "d" and 8 <= number <= 15:
        return "abi"
    if letter == "p" and 4 <= number <= 15:
        return "abi"
    if letter == "z":
        return "z"
    if letter == "p":
        return "p"
    if letter in ("q", "v"):
        return "q"
    return "fp"          # d/s/h outside the callee-saved range


def parse(text: str, keep: Pattern) -> List[Function]:
    funcs: List[Function] = []
    cur: Optional[Function] = None
    pending: List[Tuple[int, str, str]] = []

    def flush():
        if cur is None:
            return
        # Walk back from each `ret` while the instructions still look like an
        # epilogue, and stop at the first one that does not. A compiler can put
        # the restore sequence inside a backward branch's address span, which
        # would otherwise read as an innermost-loop spill.
        for i, (_addr, mn, _ops) in enumerate(pending):
            if mn != "ret":
                continue
            j = i
            while j >= 0 and pending[j][1] in EPILOGUE_OPS:
                cur.epilogue.add(pending[j][0])
                j -= 1
        for addr, mn, ops in pending:
            cur.insns.append(addr)
            if BRANCH_RE.match(mn):
                tgt = TARGET_RE.search(ops)
                if tgt:
                    dest = int(tgt.group(1), 16)
                    if dest < addr and dest >= cur.start:
                        cur.branches.append((addr, dest))
        for addr, mn, ops in pending:
            kind = classify(mn, ops)
            if kind:
                cur.spills.append({"addr": addr, "kind": kind,
                                   "depth": cur.depth_at(addr),
                                   "store": mn in STORES,
                                   "slot": slot_of(ops),
                                   "insn": f"{mn} {ops}".strip()})
        cur.undecoded = cur.count_undecoded(pending)
        cur.sve = sum(1 for _, _, o in pending if SVE_REG_RE.search(o))
        cur.total = len(pending)
        funcs.append(cur)

    for line in text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            if cur is not None and keep.search(cur.name):
                flush()
            cur = Function(m.group(2), int(m.group(1), 16))
            pending = []
            continue
        if cur is None:
            continue
        m = INSN_RE.match(line)
        if m:
            pending.append((int(m.group(1), 16), m.group(2), m.group(3)))
    if cur is not None and keep.search(cur.name):
        flush()
    return funcs


def summarise(func: Function) -> dict:
    # A store nobody reads back is a local being written, not a spill. Pair
    # each store with a later load of the same stack slot; only those pairs are
    # evidence that a value had nowhere to live but memory.
    loads_at: Dict[str, List[int]] = {}
    for s in func.spills:
        if not s["store"] and s["slot"]:
            loads_at.setdefault(s["slot"], []).append(s["addr"])
    paired = 0
    paired_in_loop = 0
    pair_depth = 0
    for s in func.spills:
        if s["kind"] == "abi" or not s["store"] or not s["slot"]:
            continue
        if any(a > s["addr"] for a in loads_at.get(s["slot"], ())):
            paired += 1
            if s["depth"] > 0:
                paired_in_loop += 1
                pair_depth = max(pair_depth, s["depth"])

    by_kind: Dict[str, int] = {}
    in_loop = 0
    max_depth = 0
    for s in func.spills:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0) + 1
        if s["kind"] == "abi":
            continue
        if s["depth"] > 0:
            in_loop += 1
            max_depth = max(max_depth, s["depth"])
    return {
        "function": func.name,
        "insns": func.total,
        "sve_insns": func.sve,
        "abi_saves": sum(1 for s in func.spills if s["kind"] == "abi"),
        "spill_pairs": paired,
        "spill_pairs_in_loop": paired_in_loop,
        "spills_total": sum(1 for s in func.spills if s["kind"] != "abi"),
        "spills_in_loop": in_loop,
        "max_loop_depth": pair_depth,
        "deepest_stack_access": max_depth,
        "by_kind": by_kind,
        "loops": len(func.branches),
        "undecoded": func.undecoded,
    }


def report(binary: Path, rows: List[dict], show_all: bool, detail: List[Function]) -> int:
    interesting = [r for r in rows if r["spills_total"] or r["sve_insns"]] if not show_all else rows
    interesting.sort(key=lambda r: (-r["spill_pairs_in_loop"], -r["spill_pairs"],
                                    -r["spills_in_loop"], -r["sve_insns"]))

    print(f"\n=== {binary} ===")
    undecoded = sum(r["undecoded"] for r in rows)
    if undecoded:
        print(f"  !! {undecoded} instructions in the matched functions could not be")
        print("     decoded (.inst / .word / (bad)). This disassembler does not know")
        print("     part of the instruction set -- most likely SVE. A spill it cannot")
        print("     decode is a spill it cannot report, so the verdict below is NOT")
        print("     trustworthy. Use a newer binutils, or pass --objdump.")
    if not rows:
        print("  no functions matched --filter; use --all or widen it")
        return 0
    head = (f"  {'function':<48}{'insns':>7}{'sve':>6}{'loops':>7}"
            f"{'abi':>5}{'stack':>7}{'paired':>8}{'inloop':>8}{'depth':>7}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in interesting[:40]:
        print(f"  {r['function'][:46]:<48}{r['insns']:>7}{r['sve_insns']:>6}"
              f"{r['loops']:>7}{r['abi_saves']:>5}{r['spills_total']:>7}"
              f"{r['spill_pairs']:>8}{r['spill_pairs_in_loop']:>8}"
              f"{r['max_loop_depth']:>7}")
    if len(interesting) > 40:
        print(f"  ... {len(interesting) - 40} more")

    loop_spills = sum(r["spill_pairs_in_loop"] for r in rows)
    any_spills = sum(r["spill_pairs"] for r in rows)
    stack_traffic = sum(r["spills_total"] for r in rows)
    sve_funcs = [r for r in rows if r["sve_insns"]]
    print()
    print(f"  {len(rows)} functions matched, {len(sve_funcs)} of them use SVE")
    abi = sum(r["abi_saves"] for r in rows)
    print(f"  {abi} callee-save accesses (ABI cost), "
          f"{stack_traffic} other vector stack accesses, "
          f"{any_spills} of them read back later")
    if any_spills == 0:
        print("  VERDICT: nothing is spilled. Every vector value stayed in a register;")
        print("           the stack traffic above is saved registers and write-only locals.")
    elif loop_spills == 0:
        pl = "" if any_spills == 1 else "s"
        print(f"  VERDICT: {any_spills} store/reload pair{pl}, none inside a loop.")
        print("           Executed once per call; not a hot-path concern.")
    else:
        worst = max(rows, key=lambda r: r["spill_pairs_in_loop"])
        pl = "" if loop_spills == 1 else "s"
        print(f"  VERDICT: {loop_spills} store/reload pair{pl} INSIDE a loop "
              f"(worst: {worst['function'][:40]}, depth {worst['max_loop_depth']}).")
        print("           Look at them with --show: writing a vector to the stack and")
        print("           reading it back as another type is the same shape as a spill.")
    return loop_spills


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binaries", nargs="+", type=Path)
    ap.add_argument("--filter", default=DEFAULT_FILTER,
                    help=f"function-name regex (default: {DEFAULT_FILTER!r})")
    ap.add_argument("--all", action="store_true", help="every function, libc included")
    ap.add_argument("--objdump", help="disassembler to use")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--show", metavar="FUNC",
                    help="print every spill site in functions matching this regex")
    ap.add_argument("--fail-on-loop-spill", action="store_true",
                    help="exit 1 when any spill sits inside a loop")
    args = ap.parse_args(argv)

    objdump = find_objdump(args.objdump)
    if not args.json:
        print(f"disassembler: {objdump}\n              {objdump_version(objdump)}")
    keep = re.compile(".") if args.all else re.compile(args.filter, re.IGNORECASE)

    out: Dict[str, List[dict]] = {}
    total_loop_spills = 0
    for binary in args.binaries:
        if not binary.is_file():
            print(f"skipping {binary}: not a file", file=sys.stderr)
            continue
        funcs = parse(disassemble(objdump, binary), keep)
        rows = [summarise(f) for f in funcs]
        out[str(binary)] = rows
        if not args.json:
            total_loop_spills += report(binary, rows, args.all, funcs)
            if args.show:
                pat = re.compile(args.show, re.IGNORECASE)
                for f in funcs:
                    if not pat.search(f.name) or not f.spills:
                        continue
                    print(f"\n  -- {f.name}")
                    for s in f.spills:
                        print(f"     {s['addr']:x}  depth {s['depth']}  {s['insn']}")
        else:
            total_loop_spills += sum(r["spills_in_loop"] for r in rows)

    if args.json:
        json.dump(out, sys.stdout, indent=2)
        print()
    elif len(args.binaries) > 1:
        print("\n=== across all binaries ===")
        for name, rows in out.items():
            print(f"  {Path(name).name[:66]:<68}{sum(r['spills_in_loop'] for r in rows):>6} in-loop")

    if any(r["undecoded"] for rows in out.values() for r in rows):
        return 2
    return 1 if (args.fail_on_loop_spill and total_loop_spills) else 0


if __name__ == "__main__":
    sys.exit(main())
