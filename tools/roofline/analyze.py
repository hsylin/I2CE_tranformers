"""Reconstruct every ROI from exact counters, then apply the registered gates."""

import configparser
import csv
from decimal import Decimal
from pathlib import Path
import re
import statistics

from common import digest, freeze_check, load, require, save

RATES = ["integer_GOPs", "nominal_GBs", "cpu_l1d_GBs", "l1d_l2_GBs", "dram_GBs"]


def stats_blocks(path):
    current = None
    with Path(path).open() as stream:
        for line in stream:
            if "Begin Simulation Statistics" in line:
                require(current is None, "Nested statistics block")
                current = {}
            elif "End Simulation Statistics" in line:
                require(current is not None, "Unmatched statistics end")
                yield current
                current = None
            elif current is not None:
                words = line.split()
                if len(words) > 1 and not words[0].startswith("#"):
                    require(words[0] not in current, "Duplicate counter: " + words[0])
                    current[words[0]] = words[1]
    require(current is None, "Incomplete statistics block")


def count(block, key, omitted=None, context=None):
    if key not in block:
        # Only callers enumerating a known nozero byte/count field may opt in.
        require(omitted is not None, "Missing required counter: " + key)
        prefix = key.rsplit(".", 1)[0] + "."
        require(
            any(k.startswith(prefix) for k in block), "Missing counter scope: " + prefix
        )
        omitted.append(dict(context or {}, counter=key))
        return 0
    value = Decimal(block[key])
    require(
        value.is_finite() and value >= 0 and value == value.to_integral_value(),
        "Counter is not a finite nonnegative integer: " + key,
    )
    return int(value)


def summarize(plan, rows):
    groups, largest, declines = [], [], []
    metrics = RATES + [
        "sim_us",
        "l1d_demand_misses",
        "l1d_demand_accesses",
        "l2_demand_misses",
        "l2_demand_accesses",
        "l1d_miss_ratio",
        "l2_miss_ratio",
        "empty_overhead_fraction",
    ]
    for case in plan["cases"]:
        repeated = [r for r in rows if r["id"] == case["id"]]
        require(
            sorted(r["experiment"] for r in repeated) == [0, 1, 2],
            "Missing or duplicated repeat",
        )
        out = dict(case)
        for metric in metrics:
            values = [r[metric] for r in repeated]
            out.update(
                {
                    metric + "_min": min(values),
                    metric + "_median": statistics.median(values),
                    metric + "_max": max(values),
                }
            )
        groups.append(out)
    q = plan["quality"]
    for block in plan["blocks"]:
        sequence = [g for g in groups if g["block"] == block["block"]]
        require(
            [g["passes"] for g in sequence] == block["passes"], "Pass sequence differs"
        )
        a, z = sequence[-2:]
        out = dict(z)
        for metric in RATES:
            med, prev = z[metric + "_median"], a[metric + "_median"]
            denominator = max(abs(med), abs(prev))
            delta = abs(med - prev) / denominator if denominator else 0
            spread = (z[metric + "_max"] - z[metric + "_min"]) / med if med else 0
            out[metric + "_last_two_pass_delta"] = delta
            out[metric + "_repeat_range_fraction"] = spread
            out[metric + "_stable"] = bool(
                med > 0
                and delta <= q["last_two_pass_delta_max"]
                and spread <= q["repeat_range_fraction_max"]
                and z["empty_overhead_fraction_median"]
                <= q["median_empty_overhead_max"]
            )
        for previous, current in zip(sequence, sequence[1:]):
            for metric in RATES:
                before, after = (
                    previous[metric + "_median"],
                    current[metric + "_median"],
                )
                if after < before:
                    declines.append(
                        dict(
                            block=block["block"],
                            kind=block["kind"],
                            d=block["d"],
                            u=block["u"],
                            footprint=block["footprint"],
                            metric=metric,
                            previous_passes=previous["passes"],
                            passes=current["passes"],
                            relative_change=after / before - 1,
                        )
                    )
        largest.append(out)
    peaks = {}
    for metric in RATES:
        candidates = largest
        stable = [r for r in candidates if r[metric + "_stable"]]
        peaks[metric] = dict(
            selection="all registered blocks at their largest passes",
            stable_blocks=len(stable),
            total_blocks=len(candidates),
            best_observed=max(candidates, key=lambda r: r[metric + "_median"]),
            best_stable=(
                max(stable, key=lambda r: r[metric + "_median"]) if stable else None
            ),
        )
    return groups, largest, peaks, declines


def analyze(root, output, plan=None, marker_prefix="SDOT"):
    freeze_check(root)
    plan = plan or load(root / "plan.json")
    require(
        re.fullmatch(r"[A-Z][A-Z0-9]*", marker_prefix) is not None,
        "Invalid marker prefix",
    )
    require(
        load(root / "execution/simulations-complete.json")["status"] == "passed",
        "All simulations must finish before analysis",
    )
    require(
        f"{marker_prefix}_VERIFY_PASS cases=144"
        in (root / "runs/verify/system.terminal").read_text(),
        "ARM functional coverage missing",
    )
    for tag in ("configuration", "verify", "exp0", "exp1", "exp2"):
        require(
            load(root / f"runs/{tag}/config-gate.json")
            == dict(passed=True, differences=[]),
            "Configuration gate failed: " + tag,
        )
        require(
            isinstance(load(root / f"runs/{tag}/config.json"), dict),
            "Missing full JSON config",
        )
        if tag != "configuration":
            end = load(root / f"runs/{tag}/simulation-exit.json")
            require(
                end["cause"] == "m5_exit instruction encountered" and end["code"] == 0,
                "Abnormal guest exit",
            )
        require(
            load(root / f"execution/{tag}-exit.json")["returncode"] == 0,
            "Abnormal host exit",
        )
    golden = {
        b["block"]: b["hash"]
        for b in load(root / "build/independent-goldens.json")["blocks"]
    }
    rows, empty, omitted, hashes = [], [], [], {}
    for ex in range(3):
        base = root / f"runs/exp{ex}"
        terminal = (base / "system.terminal").read_text()
        require(
            terminal.count(
                f"{marker_prefix}_COMPLETE experiment={ex} blocks=137 cases=812"
            )
            == 1
            and terminal.count(marker_prefix + "_GUEST_COMPLETE") == 1
            and "FAILED" not in terminal
            and "mismatch" not in terminal,
            "Incomplete guest measurement stream",
        )
        markers = re.findall(
            rf"{marker_prefix}_ROI experiment=(\d+) id=(\d+) block=(\d+) tiles=(\d+) d=(\d+) u=(\d+) passes=(\d+)",
            terminal,
        )
        results = re.findall(
            rf"{marker_prefix}_RESULT experiment=(\d+) id=(\d+) bytes=(\d+) ops=(\d+) cumulative=(\d+) valid=1",
            terminal,
        )
        checks = re.findall(
            rf"{marker_prefix}_BLOCK experiment=(\d+) block=(\d+) hash=([0-9a-f]+) valid=1",
            terminal,
        )
        require(
            len(markers) == len(results) == 812 and len(checks) == 137,
            "ROI/block coverage differs",
        )
        require(
            [(int(e), int(b), h) for e, b, h in checks]
            == [(ex, b["block"], golden[b["block"]]) for b in plan["blocks"]],
            "Block hashes differ from independent goldens",
        )
        require(
            re.findall(rf"{marker_prefix}_EMPTY experiment=(\d+) id=(\d+)", terminal)
            == [(str(ex), str(i)) for i in range(3)],
            "Empty-window coverage differs",
        )
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read(base / "config.ini")
        dram = cfg["system.mem_ctrls.dram"]
        burst = (
            int(dram["device_bus_width"])
            * int(dram["devices_per_rank"])
            * int(dram["burst_length"])
            // 8
        )
        require(burst == 64, "Unsupported DRAM burst")
        snapshots = list(stats_blocks(base / "stats.txt"))
        require(len(snapshots) == 816, "Expected 3 empty + 812 measured + 1 exit dumps")
        for i, s in enumerate(snapshots[:3]):
            require(
                count(s, "simFreq") == 10**12 and count(s, "simTicks") > 0,
                "Invalid empty timing",
            )
            empty.append(dict(experiment=ex, id=i, sim_ticks=count(s, "simTicks")))
        overhead = statistics.median(
            r["sim_ticks"] for r in empty if r["experiment"] == ex
        )
        cumulative, previous = 0, -1
        for c, marker, result, s in zip(
            plan["cases"], markers, results, snapshots[3:-1]
        ):
            require(
                list(map(int, marker))
                == [ex, c["id"], c["block"], c["n"], c["d"], c["u"], c["passes"]],
                "Ordered ROI identity differs",
            )
            if c["block"] != previous:
                cumulative, previous = 0, c["block"]
            cumulative += 1 + c["passes"]
            nominal, ops = 64 * c["n"] * c["passes"], 32 * c["n"] * c["passes"] * c["d"]
            require(
                list(map(int, result)) == [ex, c["id"], nominal, ops, cumulative],
                "Numerical ROI receipt differs",
            )
            ticks, frequency = count(s, "simTicks"), count(s, "simFreq")
            require(ticks > 0 and frequency == 10**12, "Invalid measured timing")
            seconds = ticks / frequency
            row = dict(
                c,
                experiment=ex,
                sim_ticks=ticks,
                sim_freq=frequency,
                sim_us=seconds * 1e6,
                integer_ops=ops,
                nominal_bytes=nominal,
                nominal_AI=ops / nominal,
                integer_GOPs=ops / seconds / 1e9,
                nominal_GBs=nominal / seconds / 1e9,
                empty_overhead_fraction=overhead / ticks,
            )
            context = dict(experiment=ex, id=c["id"])
            for label, prefix in [
                ("cpu_l1d", "system.roofline_l1"),
                ("l1d_l2", "system.roofline_l2"),
            ]:
                read = count(s, prefix + ".totalReadBytes", omitted, context)
                write = count(s, prefix + ".totalWrittenBytes", omitted, context)
                row.update(
                    {
                        label + "_read_bytes": read,
                        label + "_write_bytes": write,
                        label + "_GBs": (read + write) / seconds / 1e9,
                        label + "_AI": ops / (read + write) if read + write else None,
                    }
                )
            for label, prefix in [
                ("l1d", "system.cpu_cluster.cpus.dcache"),
                ("l2", "system.cpu_cluster.l2"),
            ]:
                a, m, h = [
                    count(s, prefix + "." + name + "::total", omitted, context)
                    for name in ("demandAccesses", "demandMisses", "demandHits")
                ]
                require(a == h + m, "Cache demand accounting does not close")
                row.update(
                    {
                        label + "_demand_accesses": a,
                        label + "_demand_misses": m,
                        label + "_miss_ratio": m / a if a else 0,
                    }
                )
            read, write = [
                count(s, "system.mem_ctrls.dram." + name, omitted, context) * burst
                for name in ("readBursts", "writeBursts")
            ]
            row.update(
                dram_read_bytes=read,
                dram_write_bytes=write,
                dram_GBs=(read + write) / seconds / 1e9,
                dram_AI=ops / (read + write) if read + write else None,
            )
            row.update(
                mc_read_bytes=count(
                    s, "system.mem_ctrls.bytesReadSys", omitted, context
                ),
                mc_write_bytes=count(
                    s, "system.mem_ctrls.bytesWrittenSys", omitted, context
                ),
            )
            require(
                row["cpu_l1d_read_bytes"] >= 48 * c["n"] * c["passes"]
                and row["cpu_l1d_write_bytes"] >= 16 * c["n"] * c["passes"],
                "Missing expected kernel traffic",
            )
            rows.append(row)
        for name in (
            "stats.txt",
            "system.terminal",
            "config.ini",
            "config.json",
            "config-gate.json",
            "simulation-exit.json",
        ):
            hashes[f"runs/exp{ex}/{name}"] = digest(base / name)
    require(len(rows) == 2436 and len(empty) == 9, "Incomplete aggregate coverage")
    groups, largest, peaks, declines = summarize(plan, rows)
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [("rows", rows), ("groups", groups), ("largest-pass", largest)]:
        save(output / (name + ".json"), data)
        with (output / (name + ".csv")).open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(data[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(data)
    save(
        output / "summary.json",
        dict(
            status="passed",
            rows=2436,
            conditions=812,
            blocks=137,
            experiments=3,
            empty_windows=empty,
            peaks=peaks,
            pass_increase_declines=declines,
            quality=plan["quality"],
            scope=plan["scope"],
            repeat_caveat="Simulated repeatability, not physical hardware variance",
            binary_sha256=digest(root / "build/ert_sdot"),
            plan_sha256=digest(root / "plan.json"),
        ),
    )
    save(
        output / "verification.json",
        dict(
            status="passed",
            raw_hashes=hashes,
            freeze_sha256=digest(root / "freeze.json"),
            functional_cases=144,
            measured_C_checks=2436,
            AW_guard_and_hash_checks=411,
            stats_dumps_per_experiment=816,
        ),
    )
    save(
        output / "omitted-zero-counters.json",
        dict(
            policy="Enumerated nozero native byte/burst/cache-count fields only; scope must exist; cache accesses=hits+misses.",
            events=omitted,
        ),
    )
    return dict(
        status="passed", rows=len(rows), groups=len(groups), largest_pass=len(largest)
    )
