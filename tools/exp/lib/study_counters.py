"""Parse new run counters using cumulative ticks; keep post-ROI separate.

Derived from hsylin's preserved S25 parser. No old study entrypoint or data
is accessed. Cache latency ratios have finite-window and in-flight limits.
Packet/interface counters do not represent CPU payload bytes.
"""

from pathlib import Path
from study_records import require, digest
from decimal import Decimal
import gzip
import json

CB = ["MHA_QKV", "Projection", "FF1", "FF2"]
ATT = ["MHA_QK", "MHA_softmax", "MHA_SV", "MHA_out"]
NORM = ["non_GEMM_after_projection", "non_GEMM_after_ff2"]
LABELS = [
    "MHA_QKV",
    "MHA_QK",
    "MHA_softmax",
    "MHA_SV",
    "MHA_QKV",
    "MHA_QK",
    "MHA_softmax",
    "MHA_SV",
    "MHA_QKV",
    "MHA_QK",
    "MHA_softmax",
    "MHA_SV",
    "MHA_QKV",
    "MHA_QK",
    "MHA_softmax",
    "MHA_SV",
    "MHA_out",
    "Projection",
    "non_GEMM_after_projection",
    "FF1",
    "FF2",
    "non_GEMM_after_ff2",
]
MAP = {
    "cpu_cycles": "system.cpu_cluster.cpus.numCycles",
    "instructions": "simInsts",
    "ops": "simOps",
    "load_instructions": "system.cpu_cluster.cpus.commitStats0.numLoadInsts",
    "store_instructions": "system.cpu_cluster.cpus.commitStats0.numStoreInsts",
    "committed_branches": "system.cpu_cluster.cpus.branchPred.committed_0::total",
    "branch_mispredictions": "system.cpu_cluster.cpus.branchPred.mispredicted_0::total",
}
PREFIX = {
    c: "system.cpu_cluster." + ("cpus." if c != "l2" else "") + c
    for c in ["icache", "dcache", "l2"]
}
EXTRA = {}
for c, prefix in PREFIX.items():
    for key, stat in [
        ("demand_accesses", "demandAccesses"),
        ("demand_misses", "demandMisses"),
    ]:
        MAP[c + "_" + key] = prefix + "." + stat + "::total"
    for reason in ["no_mshrs", "no_wbuffers", "no_targets"]:
        MAP[c + "_blocked_" + reason] = prefix + ".blockedCycles::" + reason
    for key, stat in [
        ("demand_miss_latency_ticks", "demandMissLatency"),
        ("demand_mshr_misses", "demandMshrMisses"),
        ("demand_mshr_miss_latency_ticks", "demandMshrMissLatency"),
    ]:
        EXTRA[c + "_" + key] = prefix + "." + stat + "::total"
for k in ["readAccesses", "readMisses", "writeAccesses", "writeMisses"]:
    MAP["dtb_" + k] = "system.cpu_cluster.cpus.mmu.dtb." + k
for c in ["icache", "dcache"]:
    MAP["toL2_" + c + "_packet_bytes"] = (
        "system.cpu_cluster.toL2Bus.pktSize_system.cpu_cluster.cpus."
        + c
        + ".mem_side_port::system.cpu_cluster.l2.cpu_side_port"
    )
MAP.update(
    {
        "toL2_walker_packet_bytes": "system.cpu_cluster.toL2Bus.pktSize_system.cpu_cluster.cpus.mmu.walker.port::system.cpu_cluster.l2.cpu_side_port",
        "toL2_total_packet_bytes": "system.cpu_cluster.toL2Bus.pktSize::total",
        "membus_L2_memctrl_packet_bytes": "system.membus.pktSize_system.cpu_cluster.l2.mem_side_port::system.mem_ctrls.port",
        "membus_total_packet_bytes": "system.membus.pktSize::total",
        "controller_read_interface_bytes": "system.mem_ctrls.bytesReadSys",
        "controller_write_interface_bytes": "system.mem_ctrls.bytesWrittenSys",
        "controller_writequeue_read_bytes": "system.mem_ctrls.bytesReadWrQ",
        "controller_read_bursts": "system.mem_ctrls.readBursts",
        "controller_write_bursts": "system.mem_ctrls.writeBursts",
        "dram_read_bursts": "system.mem_ctrls.dram.readBursts",
        "dram_write_bursts": "system.mem_ctrls.dram.writeBursts",
    }
)


def exact_number(value):
    return int(value) if value == int(value) else float(value)


def raw_blocks(path):
    blocks = []
    current_block = None
    for line in path.read_text().splitlines():
        if "Begin Simulation Statistics" in line:
            require(
                current_block is None,
                "A new statistics block started before the preceding block ended",
            )
            current_block = {}
        elif "End Simulation Statistics" in line:
            require(
                current_block is not None,
                "Statistics end marker has no corresponding begin marker",
            )
            blocks.append(current_block)
            current_block = None
        elif current_block is not None:
            fields = line.split("#", 1)[0].split()
            if len(fields) >= 2:
                try:
                    Decimal(fields[1])
                except Exception:
                    continue
                require(
                    fields[0] not in current_block,
                    "A statistics block contains a duplicate counter name",
                )
                current_block[fields[0]] = fields[1]
    require(
        current_block is None and len(blocks) == 23,
        "Expected 22 ROI dumps and one separate post-ROI dump",
    )
    return blocks


def deltas(blocks, mapping):
    previous = {field: Decimal(0) for field in mapping}
    interval_deltas = []
    for block in blocks:
        current = {field: Decimal(block[value]) for field, value in mapping.items()}
        require(
            all(
                (
                    value.is_finite() and value >= previous[field]
                    for field, value in current.items()
                )
            ),
            "Cumulative counters must be finite and must not decrease",
        )
        interval_deltas.append(
            {
                field: exact_number(value - previous[field])
                for field, value in current.items()
            }
        )
        previous = current
    return interval_deltas


def groups(intervals):
    phases = {field: {} for field in dict.fromkeys(LABELS)}
    for interval in intervals:
        for field, value in interval["values"].items():
            phases[interval["label"]][field] = (
                phases[interval["label"]].get(field, 0) + value
            )
    for name, parts in [
        ("BLOCK", list(phases)),
        ("CODEBOOK", CB),
        ("ATTENTION", ATT),
        ("NORMALIZATION_PHASES", NORM),
        ("OTHER", ATT + NORM),
    ]:
        phases[name] = {
            field: sum((phases[phase][field] for phase in parts))
            for field in intervals[0]["values"]
        }
    return phases


def derived(values, frequency):
    for name, miss, access in [
        ("l1d_miss_rate", "dcache_demand_misses", "dcache_demand_accesses"),
        ("l2_miss_rate", "l2_demand_misses", "l2_demand_accesses"),
        ("icache_miss_rate", "icache_demand_misses", "icache_demand_accesses"),
        ("dtb_read_miss_rate", "dtb_readMisses", "dtb_readAccesses"),
        ("dtb_write_miss_rate", "dtb_writeMisses", "dtb_writeAccesses"),
    ]:
        values[name] = values[miss] / values[access] if values[access] else None
    for cache in PREFIX:
        for suffix, ticks, denominator in [
            ("observed_avg_miss_ns", "demand_miss_latency_ticks", "demand_misses"),
            (
                "observed_avg_mshr_miss_ns",
                "demand_mshr_miss_latency_ticks",
                "demand_mshr_misses",
            ),
        ]:
            values[cache + "_" + suffix] = (
                values[cache + "_" + ticks]
                / values[cache + "_" + denominator]
                / frequency
                * 1000000000.0
                if values[cache + "_" + denominator]
                else None
            )
    values["dram_issued_read_bytes"] = values["dram_read_bursts"] * 64
    values["dram_issued_write_bytes"] = values["dram_write_bursts"] * 64
    values["dram_issued_bytes"] = (
        values["dram_issued_read_bytes"] + values["dram_issued_write_bytes"]
    )
    if "sim_ticks" in values:
        values["milliseconds"] = values["sim_ticks"] / frequency * 1000
        values["exact_seconds"] = values["sim_ticks"] / frequency
    require(
        values["toL2_total_packet_bytes"]
        == sum(
            (
                values["toL2_" + cache + "_packet_bytes"]
                for cache in ["icache", "dcache", "walker"]
            )
        ),
        "L2 bus packet total differs from its three registered interfaces",
    )


def parse(folder, evidence):
    folder = Path(folder)
    evidence = Path(evidence)
    path = folder / "stats.txt"
    raw = raw_blocks(path)
    frequency = int(raw[0]["simFreq"])
    require(
        frequency == 10**12
        and all((int(block["simFreq"]) == frequency for block in raw)),
        "Every dump must use the registered one-terahertz tick frequency",
    )
    labels = [
        line.split("\t")[2]
        for line in (folder / "gem5_profile_regions.tsv").read_text().splitlines()[2:]
    ]
    require(
        labels == LABELS, "Profiling labels differ from the fixed S25 interval schema"
    )
    with gzip.open(evidence / "all-raw-stat-dumps.json.gz", "xt") as stream:
        json.dump(raw, stream)
    interval_deltas = deltas(raw, dict(MAP, **EXTRA, sim_ticks="simTicks"))
    intervals = [
        dict(dump_index=index + 1, label=LABELS[index], values=interval_deltas[index])
        for index in range(22)
    ]
    phases = groups(intervals)
    require(
        phases["BLOCK"]["sim_ticks"] == int(raw[21]["simTicks"]),
        "ROI interval ticks do not sum to the final ROI cumulative tick",
    )
    for field in interval_deltas[0]:
        require(
            phases["CODEBOOK"][field]
            + phases["ATTENTION"][field]
            + phases["NORMALIZATION_PHASES"][field]
            == phases["BLOCK"][field],
            "Exclusive phase counters do not sum to the whole ROI",
        )
    for values in (
        list(phases.values())
        + [index["values"] for index in intervals]
        + [interval_deltas[22]]
    ):
        derived(values, frequency)
    for cache, prefix in PREFIX.items():
        for printed, numerator, denominator in [
            ("demandAvgMissLatency", "demand_miss_latency_ticks", "demand_misses"),
            (
                "demandAvgMshrMissLatency",
                "demand_mshr_miss_latency_ticks",
                "demand_mshr_misses",
            ),
        ]:
            values = phases["BLOCK"]
            expected_ratio = (
                values[cache + "_" + numerator] / values[cache + "_" + denominator]
                if values[cache + "_" + denominator]
                else None
            )
            if expected_ratio is not None:
                require(
                    abs(
                        float(raw[21][prefix + "." + printed + "::total"])
                        - expected_ratio
                    )
                    < 1.1e-06,
                    "Reported cache latency ratio differs from its registered counters",
                )
    return dict(
        passed=True,
        stats_sha256=digest(path),
        labels_sha256=digest(folder / "gem5_profile_regions.tsv"),
        raw_dumps=23,
        roi_intervals=22,
        intervals=intervals,
        metrics=phases,
        post_ROI_dump23_delta_separate=interval_deltas[22],
        sim_frequency=frequency,
        all_raw_stat_fields_per_dump=[len(block) for block in raw],
        traffic_scope="Packet/interface/issued burst counters per frozen source meanings, not CPU-L1 payload bytes; selected additive fields only; observed average miss-latency has coverage/inflight/window limits.",
    )
