"""The complete registered bandwidth, compute and cross-check scan."""

from pathlib import Path
from common import load, require, save

SOURCE = Path(__file__).parent
QUALITY = dict(
    last_two_pass_delta_max=0.05,
    repeat_range_fraction_max=0.05,
    median_empty_overhead_max=0.01,
)


def make_plan():
    source = load(SOURCE / "data/bandwidth-worksets.json")
    sizes = source["tiles"]
    require(
        len(sizes) == len(set(sizes)) == 90 and sizes == sorted(sizes),
        "Expected the 90 distinct original sizes in original order",
    )
    require(
        sizes[0] == 32 and sizes[-1] == 262144 and 256 in sizes,
        "Working-set endpoints or compute anchor changed",
    )
    blocks, cases = [], []

    def add(kind, n, d, u, passes):
        block = dict(
            kind=kind,
            family="stream",
            n=n,
            footprint=48 * n,
            d=d,
            u=u,
            passes=passes,
            block=len(blocks),
            first_id=len(cases),
        )
        blocks.append(block)
        for p in passes:
            cases.append(dict(block, id=len(cases), passes=p))

    for n in sizes:
        cap = min(64, max(8, 262144 // n))
        add("bandwidth", n, 1, 8, [p for p in (1, 2, 4, 8, 16, 32, 64) if p <= cap])
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        for u in (1, 2, 4, 8):
            # This exact condition is already in the bandwidth sweep.
            if (d, u) != (1, 8):
                add("compute", 256, d, u, [1, 2, 4, 8, 16, 32, 64])
    for n in (4096, 65536):
        for d in (4, 16, 64):
            for u in (1, 8):
                add("crosscheck", n, d, u, [1, 2, 4, 8])
    require(
        len(cases) == 812 and len(blocks) == 137,
        "Full scan coverage differs from the registered experiment",
    )
    kernels = [
        dict(d=d, u=u, name=f"ert_sdot_d{d}_u{u}")
        for d in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        for u in (1, 2, 4, 8)
    ]
    return dict(
        schema_version=1,
        profile="full-bandwidth-and-compute",
        source_worksets=source,
        kernels=kernels,
        blocks=blocks,
        cases=cases,
        experiments=3,
        measured_ROIs=2436,
        empty_ROIs=9,
        stats_dumps_per_experiment=816,
        functional_cases=144,
        quality=QUALITY,
        timing="simTicks/simFreq; no overhead subtraction",
        warmup="Initialize once per block; one same-kernel pass before each ROI; reset statistics without flushing caches; cumulative C.",
        counting=dict(
            unique_bytes="48*N",
            nominal_bytes="64*N*passes",
            integer_ops="32*N*D*passes",
            nominal_AI="D/2",
        ),
        scope="One simulated SVE128 core. Full registered staged scan, not a Cartesian product or the complete official ERT suite. Best stable reference within the tested patterns, not a universal maximum.",
    )


def emit(root, plan):
    assembly = [".arch armv8.2-a+sve", ".text"]
    for e in plan["kernels"]:
        d, u, name = e["d"], e["u"], e["name"]
        tag = f"{d}_{u}"
        assembly += [
            ".p2align 4",
            f".global {name}",
            f".type {name}, %function",
            f"{name}:",
            "ptrue p0.b",
            f"cbz x3, .Lend_{tag}",
            f"cbz x4, .Lend_{tag}",
            f".Lpass_{tag}:",
            "mov x5,x0",
            "mov x6,x1",
            "mov x7,x2",
            "mov x8,x3",
            ".p2align 4",
            f".Ltile_{tag}:",
        ]
        for j in range(u):
            assembly += [
                f"ld1b {{z{16+j}.b}},p0/z,[x5,#{j},mul vl]",
                f"ld1b {{z{24+j}.b}},p0/z,[x6,#{j},mul vl]",
                f"ld1w {{z{j}.s}},p0/z,[x7,#{j},mul vl]",
            ]
        for _ in range(d):
            assembly += [f"sdot z{j}.s,z{16+j}.b,z{24+j}.b" for j in range(u)]
        assembly += [f"st1w {{z{j}.s}},p0,[x7,#{j},mul vl]" for j in range(u)]
        assembly += [
            f"add x5,x5,#{u*16}",
            f"add x6,x6,#{u*16}",
            f"add x7,x7,#{u*16}",
            f"subs x8,x8,#{u}",
            f"b.ne .Ltile_{tag}",
            "subs x4,x4,#1",
            f"b.ne .Lpass_{tag}",
            f".Lend_{tag}:",
            "ret",
            f".size {name},.-{name}",
        ]
    assembly += ['.section .note.GNU-stack,"",%progbits']
    header = ["#ifndef HOST_ORACLE"]
    for e in plan["kernels"]:
        header += [
            f'extern "C" void {e["name"]}(const int8_t*,const int8_t*,uint32_t*,uint64_t,uint64_t);'
        ]
    header += [
        "#define FN(d,u) ert_sdot_d##d##_u##u",
        "#else",
        "#define FN(d,u) nullptr",
        "#endif",
        "static Entry entries[]={",
    ]
    header += [f'{{{e["d"]},{e["u"]},FN({e["d"]},{e["u"]})}},' for e in plan["kernels"]]
    header += ["};", "static Block blocks[]={"]
    for b in plan["blocks"]:
        index = next(
            i
            for i, e in enumerate(plan["kernels"])
            if (e["d"], e["u"]) == (b["d"], b["u"])
        )
        header += [
            "{%d,%d,%d,%d,{%s}},"
            % (
                b["n"],
                index,
                b["first_id"],
                len(b["passes"]),
                ",".join(map(str, b["passes"])),
            )
        ]
    header += ["};", f'static const unsigned condition_count={len(plan["cases"])};']
    for name, content in [
        ("kernel_sdot.S", "\n".join(assembly)),
        ("plan.hh", "\n".join(header)),
    ]:
        with (root / "src" / name).open("x") as stream:
            stream.write(content + "\n")
    save(root / "plan.json", plan)
