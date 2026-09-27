#!/usr/bin/env python3
"""
Self-contained check of check_spills.py's classification, run with no arguments.

Real binaries from this project turn out to contain no spills at all, so they
cannot show that the detector would notice one. These cases are hand-written
disassembly with a known answer, covering what the tool has to tell apart:

  * a genuine spill and reload inside a nested loop           -> counted, depth 2
  * a callee-saved d8/d9 pair saved and restored              -> ABI, not a spill
  * a vector store nothing ever reads back                    -> a local, not a spill
  * a spill and reload outside any loop                       -> counted, depth 0
  * an epilogue sitting inside a backward branch's span       -> not "in a loop"
  * a disassembler that could not decode the instructions     -> flagged, so a
    blind pass never reads as a clean one
"""
import importlib.util
import re
import sys
from pathlib import Path

FIXTURE = """
0000000000001000 <func_real_spill>:
    1000:\tstp\td8, d9, [sp, #16]
    1004:\tptrue\tp0.s
    1008:\tld1w\t{z1.s}, p0/z, [x0]
    100c:\tld1w\t{z2.s}, p0/z, [x1]
    1010:\tstr\tz3, [sp, #1, mul vl]
    1014:\tmla\tz4.s, p0/m, z1.s, z2.s
    1018:\tldr\tz3, [sp, #1, mul vl]
    101c:\tb.ne\t100c <func_real_spill+0xc>
    1020:\tb.ne\t1008 <func_real_spill+0x8>
    1024:\tldp\td8, d9, [sp, #16]
    1028:\tret

0000000000002000 <func_write_only>:
    2000:\tstr\tq5, [sp, #32]
    2004:\tb.ne\t2000 <func_write_only>
    2008:\tret

0000000000003000 <func_spill_outside_loop>:
    3000:\tstr\tz7, [sp, #2, mul vl]
    3004:\tmla\tz4.s, p0/m, z1.s, z2.s
    3008:\tldr\tz7, [sp, #2, mul vl]
    300c:\tmla\tz5.s, p0/m, z1.s, z2.s
    3010:\tret

0000000000005000 <func_undecoded_by_old_objdump>:
    5000:\t.inst\t0x04a10441 ; undefined
    5004:\t.word\t0x25a0c000
    5008:\tmla\tz4.s, p0/m, z1.s, z2.s
    500c:\tret

0000000000004000 <func_epilogue_inside_loop_span>:
    4000:\tstp\td8, d9, [sp, #16]
    4004:\tmla\tz4.s, p0/m, z1.s, z2.s
    4008:\tb.ne\t4004 <func_epilogue_inside_loop_span+0x4>
    400c:\tldp\td8, d9, [sp, #16]
    4010:\tret
"""

EXPECTED = {
    "func_real_spill":                dict(abi=2, stack=2, pairs=1, in_loop=1, depth=2),
    "func_write_only":                dict(abi=0, stack=1, pairs=0, in_loop=0, depth=0),
    "func_spill_outside_loop":        dict(abi=0, stack=2, pairs=1, in_loop=0, depth=0),
    "func_epilogue_inside_loop_span": dict(abi=2, stack=0, pairs=0, in_loop=0, depth=0),
    "func_undecoded_by_old_objdump":   dict(abi=0, stack=0, pairs=0, in_loop=0, depth=0),
}

# An objdump too old for SVE prints .inst/.word instead of the instruction, and
# then a spill hidden in those words cannot be seen. The count must be non-zero
# so the caller is told the verdict is unreliable.
EXPECTED_UNDECODED = {"func_undecoded_by_old_objdump": 2}


def load():
    path = Path(__file__).with_name("check_spills.py")
    spec = importlib.util.spec_from_file_location("check_spills", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_spills"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    cs = load()
    got = {}
    for func in cs.parse(FIXTURE, re.compile("func_")):
        row = cs.summarise(func)
        got[row["function"]] = dict(
            abi=row["abi_saves"], stack=row["spills_total"],
            pairs=row["spill_pairs"], in_loop=row["spill_pairs_in_loop"],
            depth=row["max_loop_depth"], _undecoded=row["undecoded"])

    failures = 0
    for name, want in EXPECTED.items():
        have = dict(got.get(name) or {})
        undecoded = have.pop("_undecoded", 0)
        if undecoded != EXPECTED_UNDECODED.get(name, 0):
            failures += 1
            print(f"  FAIL  {name}: undecoded {undecoded}, "
                  f"expected {EXPECTED_UNDECODED.get(name, 0)}")
        if have == want:
            print(f"  PASS  {name}")
        else:
            failures += 1
            print(f"  FAIL  {name}\n          expected {want}\n          got      {have}")
    missing = set(got) - set(EXPECTED)
    if missing:
        failures += 1
        print(f"  FAIL  unexpected functions parsed: {sorted(missing)}")
    print(("\nall cases pass" if not failures else f"\n{failures} case(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
