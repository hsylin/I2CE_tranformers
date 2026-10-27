#!/usr/bin/env python3
"""Plot configuration-derived Roofline references with native gem5 observations.

No simulation or calibration is launched. Unsupported or incomplete experiments
are listed with reasons; a missing counter is never replaced by an estimate.
"""

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
import math
from pathlib import Path
import re
import tempfile

MODEL_REVISION = "ed752e78a2ddd5587c1954e2cda51f94cbbe554e"
MODEL_URL = "https://github.com/Jerry0209/gem5/blob/" + MODEL_REVISION
DEFAULT_INPUT = Path(__file__).with_name("roofline_examples") / "e13" / "roofline.json"
CPU = "system.cpu_cluster.cpus"
DRAM = "system.mem_ctrls.dram"
DOT = CPU + ".commitStats0.committedInstType::SimdDotProd"
MEMBUS_ROUTE = (
    "system.membus.pktSize_system.cpu_cluster.l2.mem_side_port::system.mem_ctrls.port"
)
TRAFFIC = {
    "Core–L1D": (
        "system.roofline_l1.totalReadBytes",
        "system.roofline_l1.totalWrittenBytes",
    ),
    "L1D–L2": (
        "system.roofline_l2.totalReadBytes",
        "system.roofline_l2.totalWrittenBytes",
    ),
    "Physical DRAM": (DRAM + ".dramBytesRead", DRAM + ".dramBytesWritten"),
    "Membus/controller": (DRAM + ".bytesRead::total", DRAM + ".bytesWritten::total"),
}
RATES = {
    "Core–L1D": (
        "system.roofline_l1.averageReadBandwidth",
        "system.roofline_l1.averageWriteBandwidth",
        1,
    ),
    "L1D–L2": (
        "system.roofline_l2.averageReadBandwidth",
        "system.roofline_l2.averageWriteBandwidth",
        1,
    ),
    "Physical DRAM": (DRAM + ".avgRdBW", DRAM + ".avgWrBW", 10**6),
}
COUNTERS = {
    "simTicks",
    "simFreq",
    "finalTick",
    DOT,
    MEMBUS_ROUTE,
    DRAM + ".peakBW",
    DRAM + ".readBursts",
    DRAM + ".writeBursts",
}
COUNTERS.update(k for pair in TRAFFIC.values() for k in pair)
COUNTERS.update(k for read, write, _ in RATES.values() for k in (read, write))
FORMULAS = {DRAM + ".peakBW"} | {k for r, w, _ in RATES.values() for k in (r, w)}
COLORS = {
    "Core–L1D": "#278ac4",
    "L1D–L2": "#529b39",
    "Physical DRAM": "#666f7a",
    "Membus/controller": "#a166bd",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value, label):
    require(
        type(value) in (int, float) and math.isfinite(value) and value > 0,
        label + " must be finite and positive",
    )
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_artifact(root, record):
    relative = Path(record["path"])
    require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Artifact paths must be relative and remain inside the evidence directory",
    )
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, "Artifact escapes evidence directory")
    require(
        digest(path) == record["sha256"], "Artifact SHA256 mismatch: " + relative.name
    )
    return path


def stats_blocks(path):
    """Keep integers exact and retain file line numbers for source inspection."""
    blocks, current, lines = [], None, None
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if "Begin Simulation Statistics" in line:
            require(current is None, "Nested statistics block")
            current, lines = {}, {}
        elif "End Simulation Statistics" in line:
            require(current is not None, "Unmatched statistics end")
            blocks.append((current, lines))
            current = None
        elif current is not None:
            fields = line.split()
            if fields and fields[0] in COUNTERS:
                key = fields[0]
                require(
                    key not in current and len(fields) >= 2,
                    "Duplicate or malformed counter: " + key,
                )
                value = Decimal(fields[1])
                require(value.is_finite() and value >= 0, "Invalid counter: " + key)
                require(
                    key in FORMULAS or value == value.to_integral_value(),
                    "Noninteger counter: " + key,
                )
                current[key] = float(value) if key in FORMULAS else int(value)
                lines[key] = number
    require(current is None and blocks, "Incomplete or empty statistics file")
    return blocks


def theory(config, frequency):
    """Conditional resource bounds for the inspected single-core Minor model."""
    s = config["system"]
    require(
        len(s["cpu_cluster"]) == len(s["mem_ctrls"]) == 1,
        "Only one cluster and one DRAM channel are supported",
    )
    cluster = s["cpu_cluster"][0]
    require(
        len(cluster["cpus"]) == 1 and s["sve_vl"] == 1,
        "Only a single SVE128 core is supported; multicore needs a new model",
    )
    cpu = cluster["cpus"][0]
    require(
        cpu["type"] == "BaseMinorCPU" and cpu["numThreads"] == 1 and cpu["path"] == CPU,
        "Unsupported CPU model, thread count or counter namespace",
    )
    domain = cluster["clk_domain"]
    require(
        cpu["clk_domain"] == domain["path"], "CPU clock domain is not the cluster clock"
    )
    hz = frequency / positive(domain["clock"][0], "CPU clock period")
    system_hz = frequency / positive(s["clk_domain"]["clock"][0], "System clock period")
    require(
        s["cache_line_size"] == 64, "This packet model requires 64-byte cache lines"
    )
    for name, cpu_peer, memory_peer in [
        ("roofline_l1", CPU + ".dcache_port", CPU + ".dcache.cpu_side"),
        (
            "roofline_l2",
            CPU + ".dcache.mem_side",
            "system.cpu_cluster.toL2Bus.cpu_side_ports[1]",
        ),
    ]:
        m = s[name]
        require(
            m["type"] == "CommMonitor" and m["disable_bandwidth_hists"] is False,
            "Missing enabled native CommMonitor: " + name,
        )
        require(
            m["cpu_side_port"]["peer"] == cpu_peer
            and m["mem_side_port"]["peer"] == memory_peer,
            "Unsupported monitor boundary: " + name,
        )
    fus = [
        fu
        for fu in cpu["executeFuncUnits"]["funcUnits"]
        if any(op["opClass"] == "SimdDotProd" for op in fu["opClasses"]["opClasses"])
    ]
    require(
        len(fus) == 1 and fus[0]["issueLat"] == 1,
        "Unsupported dot-product FU model; review issue scheduling first",
    )
    require(
        all(
            t["suppress"] is False
            and t["extraCommitLat"] == 0
            and t["extraCommitLatExpr"] is None
            for t in fus[0]["timings"]
        ),
        "Unsupported dot-product timing override",
    )
    peak = (
        hz
        * 32
        * min(
            positive(cpu["executeIssueLimit"], "Issue limit"),
            positive(cpu["executeCommitLimit"], "Commit limit"),
            1,
        )
    )
    memory = []

    def bound(level, direction, rate, formula, fields, qualification):
        memory.append(
            dict(
                level=level,
                direction=direction,
                bytes_per_second=rate,
                ridge_ops_per_byte=peak / rate,
                formula=formula,
                config_fields=fields,
                qualification=qualification,
            )
        )

    cp = "system.cpu_cluster[0].cpus[0]."
    core_fields = {
        cp + k: cpu[k] for k in ("executeMemoryIssueLimit", "executeMemoryCommitLimit")
    }
    core_fields.update(
        {
            "system.cpu_cluster[0].clk_domain.clock[0]": domain["clock"][0],
            "system.sve_vl": 1,
        }
    )
    bound(
        "Core–L1D",
        "total",
        hz
        * 16
        * min(
            positive(cpu["executeMemoryIssueLimit"], "Memory issue"),
            positive(cpu["executeMemoryCommitLimit"], "Memory commit"),
        ),
        "CPU Hz × 16 B × min(memory issue, memory commit)",
        core_fields,
        "Conditional CPU issue bound for full, aligned, contiguous 16 B accesses; shared by reads and writes, not an intrinsic L1 bank peak.",
    )
    for level, bus, bus_hz, expected_clock, prefix in [
        (
            "L1D–L2",
            cluster["toL2Bus"],
            hz,
            domain["path"],
            "system.cpu_cluster[0].toL2Bus.",
        ),
        (
            "Membus/controller",
            s["membus"],
            system_hz,
            s["clk_domain"]["path"],
            "system.membus.",
        ),
    ]:
        require(
            bus["type"] == "CoherentXBar" and bus["clk_domain"] == expected_clock,
            "Unsupported crossbar or clock domain: " + level,
        )
        width = positive(bus["width"], "Crossbar width")
        header = positive(bus["header_latency"], "Header occupancy")
        rate = 64 * bus_hz / (header + math.ceil(64 / width))
        fields = {prefix + k: bus[k] for k in ("width", "header_latency", "clk_domain")}
        fields["system.cache_line_size"] = 64
        fields["resolved_clock_hz"] = bus_hz
        for direction in ("read", "write"):
            bound(
                level,
                direction,
                rate,
                "64 B × bus Hz / (header cycles + ceil(64 B / width))",
                fields,
                "Per-direction crossbar bound for 64 B payloads. Header-only requests, contention and cache service can reduce throughput; not a cache-bank maximum.",
            )
    dram = s["mem_ctrls"][0]["dram"]
    require(dram["type"] == "DRAMInterface", "Unsupported memory interface")
    burst = (
        math.prod(
            positive(dram[k], k)
            for k in ("burst_length", "devices_per_rank", "device_bus_width")
        )
        / 8
    )
    require(burst == 64, "Only the inspected 64 B DRAM burst model is supported")
    rate = burst * frequency / positive(dram["tBURST"], "DRAM burst ticks")
    fields = {
        "system.mem_ctrls[0].dram." + k: dram[k]
        for k in ("burst_length", "devices_per_rank", "device_bus_width", "tBURST")
    }
    bound(
        "Physical DRAM",
        "total",
        rate,
        "(burst_length × devices_per_rank × device_bus_width / 8) × simFreq / tBURST",
        fields,
        "Single-channel burst bandwidth shared by reads and writes; refresh, bank conflicts and direction changes reduce sustained bandwidth.",
    )
    return dict(
        compute_ops_per_second=peak,
        cpu_hz=hz,
        system_hz=system_hz,
        burst_bytes=burst,
        compute_formula="CPU Hz × 32 ops/SDOT × min(issue limit, commit limit, 1/issueLat)",
        compute_fields={
            cp + "executeIssueLimit": cpu["executeIssueLimit"],
            cp + "executeCommitLimit": cpu["executeCommitLimit"],
            "SDOT FU count": len(fus),
            "SDOT FU issueLat": fus[0]["issueLat"],
            "SDOT FU opLat": fus[0]["opLat"],
            "system.sve_vl": s["sve_vl"],
            "system.cpu_cluster[0].clk_domain.clock[0]": domain["clock"][0],
        },
        memory=memory,
    )


def analyze(root, run):
    require(
        run["gem5_revision"] == MODEL_REVISION,
        "Unreviewed gem5 revision/model semantics",
    )
    require(
        re.fullmatch(r"[a-f0-9]{40}", run["source_commit"]), "Invalid source commit"
    )
    require(
        re.fullmatch(r"[a-f0-9]{64}", run["binary_sha256"]), "Invalid binary SHA256"
    )
    files = {
        name: read_artifact(root, record) for name, record in run["artifacts"].items()
    }
    receipt = json.loads(files["receipt"].read_text())
    require(
        receipt["host_returncode"]
        == receipt["guest_returncode"]
        == receipt["simulation_code"]
        == 0
        and receipt["simulation_cause"] == "m5_exit instruction encountered"
        and receipt["config_gate_passed"] is True
        and receipt["guest_binary_hash_ok"] is True,
        "Run has not passed completion/configuration/guest identity checks",
    )
    require(
        receipt["binary_sha256"] == run["binary_sha256"]
        and receipt["source_commit"] == run["source_commit"],
        "Run identity differs from its receipt",
    )
    snapshots = stats_blocks(files["stats"])
    roi = run["roi"]
    end, begin = roi["end_block"], roi["begin_block"]
    require(
        type(end) is int
        and type(roi["expected_blocks"]) is int
        and len(snapshots) == roi["expected_blocks"]
        and 0 <= end < len(snapshots) - 1,
        "Invalid ROI end/block count; final cleanup dump must be excluded",
    )
    require(
        begin is None or (type(begin) is int and 0 <= begin < end),
        "Invalid ROI beginning",
    )
    require(roi["counter_mode"] == "cumulative_since_reset", "Unsupported reset scheme")
    labels = [
        line.split("\t") for line in files["regions"].read_text().splitlines()[2:]
    ]
    require(
        [int(row[0]) for row in labels] == list(range(1, len(labels) + 1))
        and [row[2] for row in labels] == roi["labels"]
        and end < len(labels),
        "ROI marker sequence differs from the declared measurement",
    )
    selected, locations = snapshots[end]
    required = COUNTERS - FORMULAS
    # Missing zero-suppressed counters remain unsupported: rerun/export an
    # explicit accounting record instead of silently inventing zero traffic.
    for index in range(end + 1):
        values = snapshots[index][0]
        require(
            required <= values.keys(),
            "Missing native counter(s): " + ", ".join(sorted(required - values.keys())),
        )
        require(values["simFreq"] == selected["simFreq"], "Tick frequency changed")
        if index:
            previous = snapshots[index - 1][0]
            require(
                all(values[k] >= previous[k] for k in required),
                "Counter reset inside declared cumulative ROI",
            )
    frequency = positive(selected["simFreq"], "simFreq")
    base = snapshots[begin][0] if begin is not None else dict.fromkeys(required, 0)
    delta = {
        k: selected[k] - base[k] for k in required if k not in ("simFreq", "finalTick")
    }
    seconds = positive(delta["simTicks"], "ROI simTicks") / frequency
    limits = theory(json.loads(files["config"].read_text()), frequency)
    require(
        DRAM + ".peakBW" in selected
        and math.isclose(
            selected[DRAM + ".peakBW"] * 10**6,
            next(
                b["bytes_per_second"]
                for b in limits["memory"]
                if b["level"] == "Physical DRAM"
            ),
            rel_tol=1e-9,
        ),
        "Native DRAM peakBW disagrees with config (decimal MB/s in inspected source)",
    )
    for direction in ("read", "write"):
        byte_key = DRAM + (
            ".dramBytesRead" if direction == "read" else ".dramBytesWritten"
        )
        burst_key = DRAM + (".readBursts" if direction == "read" else ".writeBursts")
        for values, _ in snapshots[: end + 1]:
            require(
                values[byte_key] == values[burst_key] * limits["burst_bytes"],
                "DRAM burst byte mismatch",
            )
    op = run["operation_model"]
    require(
        op["kind"] == "transformer_gemm"
        and op["instruction_model"] == "sve128_int8_sdot",
        "Unsupported operation accounting or precision",
    )
    dims = [op[k] for k in ("learners", "sequence", "dimension", "feedforward")]
    require(all(type(v) is int and v > 0 for v in dims), "Invalid GEMM dimensions")
    learners, sequence, dimension, feedforward = dims
    require(
        begin is None and end == len(labels) - 1,
        "Transformer formula requires the whole-block ROI",
    )
    work = (
        2
        * learners
        * (
            4 * sequence * dimension**2
            + 2 * sequence**2 * dimension
            + 2 * sequence * dimension * feedforward
        )
    )
    traffic, sources = {}, []
    for level, (read, write) in TRAFFIC.items():
        traffic[level] = dict(
            read=delta[read], write=delta[write], total=delta[read] + delta[write]
        )
        if level in RATES:
            r, w, scale = RATES[level]
            for key, count in ((r, read), (w, write)):
                require(
                    key in selected
                    and math.isclose(
                        selected[key] * scale,
                        selected[count] / (selected["simTicks"] / frequency),
                        rel_tol=1e-9,
                        abs_tol=1.1,
                    ),
                    "Native average bandwidth disagrees with cumulative bytes/time: "
                    + key,
                )
        for key in (read, write):
            sources.append(
                dict(
                    quantity=level + (" read bytes" if key == read else " write bytes"),
                    value=delta[key],
                    source=run["artifacts"]["stats"]["path"],
                    field=key,
                    line=locations[key],
                )
            )
    require(
        traffic["Membus/controller"]["total"] == delta[MEMBUS_ROUTE],
        "Controller bytes differ from the selected membus route payload",
    )
    for key in ("simTicks", "simFreq", DOT, DRAM + ".peakBW", MEMBUS_ROUTE):
        sources.append(
            dict(
                quantity=key,
                value=selected[key],
                source=run["artifacts"]["stats"]["path"],
                field=key,
                line=locations[key],
            )
        )
    points = []
    performance = work / seconds
    for bound in limits["memory"]:
        count = traffic[bound["level"]][bound["direction"]]
        if not count:
            continue  # Zero traffic has infinite AI and no finite log-axis point.
        ai = work / count
        ceiling = min(limits["compute_ops_per_second"], bound["bytes_per_second"] * ai)
        points.append(
            dict(
                bound,
                bytes=count,
                ai=ai,
                performance=performance,
                used_bandwidth=count / seconds,
                bandwidth_fraction=count / seconds / bound["bytes_per_second"],
                projection=ceiling,
                model_headroom=ceiling / performance,
            )
        )
    return dict(
        id=run["id"],
        label=run["label"],
        source_commit=run["source_commit"],
        binary_sha256=run["binary_sha256"],
        seconds=seconds,
        operations=work,
        operation_model=op,
        performance=performance,
        traffic=traffic,
        theory=limits,
        points=points,
        quantity_sources=sources,
        roi=roi,
        opclass_count=delta[DOT],
        opclass_equivalent_ops=delta[DOT] * 32,
        opclass_to_useful_ratio=delta[DOT] * 32 / work,
        compute_fraction=performance / limits["compute_ops_per_second"],
        limitations=run["limitations"],
        artifacts=run["artifacts"],
    )


def load(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    require(
        manifest["schema_version"] == 1
        and manifest["method"] == "gem5_theoretical_references",
        "Unsupported Roofline manifest schema",
    )
    supported, skipped, seen = [], [], set()
    for run in manifest["experiments"]:
        name = run["id"]
        require(
            isinstance(name, str) and name not in seen,
            "Duplicate or invalid experiment ID",
        )
        seen.add(name)
        if "unavailable_reason" in run:
            skipped.append(
                dict(id=name, status="unavailable", reason=run["unavailable_reason"])
            )
            continue
        try:
            supported.append(analyze(path.parent, run))
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            OSError,
            InvalidOperation,
        ) as error:
            skipped.append(
                dict(id=name, status="unsupported_or_invalid", reason=str(error))
            )
    return dict(
        method=manifest["method"],
        manifest_sha256=digest(path),
        observations=supported,
        skipped=skipped,
    )


def log_ticks(low, high):
    """Spell out minor tick values instead of ambiguous log-axis digits."""
    return [
        m * 10**e
        for e in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1)
        for m in (1, 2, 5)
        if low <= m * 10**e <= high
    ]


def figure(observation):
    o = observation
    points = [
        p for p in o["points"] if p["level"] in {"Core–L1D", "L1D–L2", "Physical DRAM"}
    ]
    peak = o["theory"]["compute_ops_per_second"] / 1e9
    values = [p["ai"] for p in points] + [p["ridge_ops_per_byte"] for p in points]
    left, right = min(values) / 4, max(values) * 2
    traces = [
        dict(
            x=[left, right],
            y=[peak, peak],
            mode="lines",
            name=f"SDOT compute reference · {peak:g} Gop/s",
            line=dict(color="#b78300", width=3),
        )
    ]
    traces.append(
        dict(
            x=[left, right],
            y=[o["performance"] / 1e9] * 2,
            mode="lines",
            name=f"Observed · {o['performance']/1e9:.2f} Gop/s",
            line=dict(color="#a3abb3", dash="dot", width=1),
        )
    )
    for p in points:
        name = p["level"] + " " + p["direction"]
        color = COLORS[p["level"]]
        rate = p["bytes_per_second"] / 1e9
        ridge = p["ridge_ops_per_byte"]
        line = dict(
            color=color, dash="dash" if p["direction"] == "write" else "solid", width=2
        )
        traces.append(
            dict(
                x=[left, ridge],
                y=[left * rate, peak],
                mode="lines",
                name=f"{name} · {rate:g} GB/s",
                legendgroup=name,
                line=line,
                hovertemplate=html.escape(p["qualification"]) + "<extra></extra>",
            )
        )
        ai, actual, projected = p["ai"], o["performance"] / 1e9, p["projection"] / 1e9
        traces.append(
            dict(
                x=[ai, ai],
                y=[actual, projected],
                mode="lines",
                showlegend=False,
                legendgroup=name,
                line=dict(color=color, dash="dot", width=1),
                hoverinfo="skip",
            )
        )
        traces.append(
            dict(
                x=[ai],
                y=[actual],
                mode="markers",
                name=name + " observed",
                legendgroup=name,
                showlegend=False,
                marker=dict(
                    color=color,
                    size=12,
                    symbol="diamond" if p["direction"] == "write" else "circle",
                ),
                hovertemplate=f"{html.escape(name)}<br>AI: {ai:.6f} ops/B<br>Observed: {actual:.6f} Gop/s<br>Traffic: {p['bytes']:,} B<br>Used: {p['used_bandwidth']/1e9:.6f} GB/s<extra></extra>",
            )
        )
        traces.append(
            dict(
                x=[ai],
                y=[projected],
                mode="markers",
                name=name + " reference projection",
                legendgroup=name,
                showlegend=False,
                marker=dict(
                    color=color, size=11, symbol="circle-open", line=dict(width=2)
                ),
                hovertemplate="Reference projection, not another measurement<extra></extra>",
            )
        )
    bottom = (
        min(
            left * min(p["bytes_per_second"] for p in points) / 1e9,
            o["performance"] / 1e9,
        )
        / 2
    )
    return dict(
        data=traces,
        layout=dict(
            title=dict(text="Hierarchical Roofline Model for " + html.escape(o["id"])),
            template="plotly_white",
            height=620,
            margin=dict(t=95, b=110, l=85, r=30),
            xaxis=dict(
                type="log",
                title=dict(
                    text="Boundary-specific arithmetic intensity (useful integer ops/B)"
                ),
                range=[math.log10(left), math.log10(right)],
                tickmode="array",
                tickvals=log_ticks(left, right),
                ticktext=[f"{v:g}" for v in log_ticks(left, right)],
            ),
            yaxis=dict(
                type="log",
                title=dict(text="Useful integer Gop/s"),
                range=[math.log10(bottom), math.log10(peak * 1.7)],
                tickmode="array",
                tickvals=log_ticks(bottom, peak * 1.7),
                ticktext=[f"{v:g}" for v in log_ticks(bottom, peak * 1.7)],
            ),
            legend=dict(orientation="h", y=-0.23),
            meta=dict(method="theoretical_references", experiment=o["id"]),
        ),
    )


def summary_html(result):
    esc = lambda value: html.escape(str(value))
    parts = [
        "<h2>Hierarchical Roofline</h2><p>Solid points are measured useful integer performance. Open points are reference projections. Each sloped line is capped by the common compute reference. L1 and L2 lines are conditional resource bounds, not measured cache maxima. No empirical calibration is used. Equal-rate lines and nearby points can overlap; use the legend, hover values and table to inspect both directions.</p>"
    ]
    for o in result["observations"]:
        parts.append(
            f"<h3>{esc(o['id'])} — {esc(o['label'])}</h3><p>{o['seconds']:.10f} simulated seconds; {o['operations']:,} useful GEMM operations; <strong>{o['performance']/1e9:.6f} Gop/s</strong> ({o['compute_fraction']:.2%} of the SDOT compute reference).</p>"
        )
        parts.append(
            f"<p>Source commit <code>{esc(o['source_commit'])}</code><br>Binary SHA256 <code>{esc(o['binary_sha256'])}</code>. This is a separately instrumented observation; matching experiment labels do not identify the same run as other report tabs.</p>"
        )
        parts.append(
            "<table><thead><tr><th>Boundary / direction</th><th>Bytes</th><th>AI (ops/B)</th><th>Used GB/s</th><th>Reference GB/s</th><th>Usage/reference</th></tr></thead><tbody>"
        )
        for p in o["points"]:
            parts.append(
                f"<tr><td>{esc(p['level']+' '+p['direction'])}</td><td>{p['bytes']:,}</td><td>{p['ai']:.6f}</td><td>{p['used_bandwidth']/1e9:.6f}</td><td>{p['bytes_per_second']/1e9:g}</td><td>{p['bandwidth_fraction']:.2%}</td></tr>"
            )
        parts.append(
            "</tbody></table><p>Read/write lines use their own byte denominators; shared references use read + write. Bytes at different boundaries must not be added. Membus/controller payload and physical DRAM bursts are distinct quantities.</p>"
        )
        parts.append(
            f"<p>SimdDotProd commits: {o['opclass_count']:,}; ×32 / useful work = {o['opclass_to_useful_ratio']:.6f}. OpClass accounting is a cross-check, not a mnemonic trace. Softmax, packing and other work remain in ROI time but are excluded from useful GEMM operations.</p>"
        )
        parts.append(
            "<details><summary>Quantities, source counters and theoretical derivations</summary><table><tr><th>Quantity</th><th>Value</th><th>Source</th></tr>"
        )
        for row in o["quantity_sources"]:
            parts.append(
                f"<tr><td>{esc(row['quantity'])}</td><td>{esc(row['value'])}</td><td><code>{esc(row['field'])}</code><br>{esc(row['source'])}:{row['line']}</td></tr>"
            )
        parts.append(
            "</table><p>Useful work = 2L(4SD² + 2S²D + 2SDF), derived from declared algorithm dimensions, not a gem5 FLOP counter.</p><pre>"
            + esc(json.dumps(o["operation_model"], indent=2))
            + "</pre>"
        )
        parts.append(
            "<p>"
            + esc(o["theory"]["compute_formula"])
            + "</p><pre>"
            + esc(json.dumps(o["theory"]["compute_fields"], indent=2))
            + "</pre>"
        )
        for b in o["theory"]["memory"]:
            parts.append(
                "<p><b>"
                + esc(b["level"] + " " + b["direction"])
                + "</b>: "
                + esc(b["formula"])
                + "<br>"
                + esc(b["qualification"])
                + "</p><pre>"
                + esc(json.dumps(b["config_fields"], indent=2))
                + "</pre>"
            )
        parts.append(
            "<p>Clock values are ticks/cycle; use simFreq / clock. Native DRAM peakBW and avgRdBW/avgWrBW use decimal MB/s in the inspected implementation.</p></details><ul>"
        )
        parts.extend("<li>" + esc(note) + "</li>" for note in o["limitations"])
        parts.append("</ul>")
    if result["skipped"]:
        parts.append(
            "<h3>Experiments without a supported Roofline observation</h3><ul>"
        )
        parts.extend(
            "<li><b>" + esc(row["id"]) + "</b>: " + esc(row["reason"]) + "</li>"
            for row in result["skipped"]
        )
        parts.append("</ul>")
    parts.append(
        "<p>Inspected model source: "
        + ", ".join(
            '<a href="' + MODEL_URL + "/" + path + '">' + title + "</a>"
            for title, path in [
                ("MinorCPU", "src/cpu/minor/BaseMinorCPU.py"),
                ("Crossbar serialization", "src/mem/xbar.cc"),
                ("Crossbar occupancy", "src/mem/coherent_xbar.cc"),
                ("DRAM bandwidth units", "src/mem/dram_interface.cc"),
                ("CommMonitor payload", "src/mem/comm_monitor.cc"),
            ]
        )
        + "</p>"
    )
    return "\n".join(parts)


def evidence_paths(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    return [path] + [
        path.parent / a["path"]
        for r in manifest["experiments"]
        for a in r.get("artifacts", {}).values()
    ]


def render(result):
    from plotly.offline import get_plotlyjs
    import plotly.io as pio

    charts = "\n".join(
        pio.to_html(
            figure(o),
            include_plotlyjs=False,
            full_html=False,
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(format="svg", filename=o["id"] + "-roofline"),
            ),
        )
        for o in result["observations"]
    )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>Hierarchical Roofline</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#233247}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:9px;border-bottom:1px solid #dae0e7;text-align:left}pre{overflow:auto;background:#f3f6fa;padding:14px}code{overflow-wrap:anywhere}details{margin:18px 0}li{margin:8px 0}</style><script>"
        + get_plotlyjs()
        + "</script>"
        + charts
        + summary_html(result)
        + "</html>"
    )


def write_output(path, content, inputs):
    path = Path(path).resolve()
    require(
        path not in {Path(p).resolve() for p in inputs},
        "Output must not overwrite an input",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path, required=True, help="Offline HTML output"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional recomputed quantities and skip reasons",
    )
    args = parser.parse_args()
    inputs = evidence_paths(args.input)
    require(
        args.json_output is None or args.output.resolve() != args.json_output.resolve(),
        "Output paths must differ",
    )
    require(
        all(
            p.resolve() not in {v.resolve() for v in inputs}
            for p in [args.output, args.json_output]
            if p is not None
        ),
        "Output must not overwrite an input",
    )
    result = load(args.input)
    write_output(args.output, render(result), inputs)
    if args.json_output:
        write_output(
            args.json_output,
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            inputs,
        )
    print(
        f"Roofline: {len(result['observations'])} supported, {len(result['skipped'])} skipped; {args.output}"
    )


if __name__ == "__main__":
    main()
