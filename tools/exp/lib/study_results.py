"""Evaluate frozen stages from saved counters, without reentering simulation."""

import argparse
import csv
import gzip
import json
from pathlib import Path
import re
import time
from study_records import digest, load, save, require, study_budget

AGGREGATES = ("BLOCK", "CODEBOOK", "ATTENTION", "NORMALIZATION_PHASES")
TITLES = dict(
    zip(
        ["E%02d" % i for i in range(1, 14)],
        [
            "Widened baseline",
            "SDOT core and adapters",
            "int8 and bounded W packing",
            "L1 blocking",
            "Outer W reuse",
            "All-K A retention",
            "Full-panel decoder",
            "Learner-first consumption",
            "Final-K output fusion",
            "Direct-int8 Attention",
            "Attention blocking",
            "Query16 streaming",
            "Vector integer softmax",
        ],
    )
)


def comparison(current_ticks, reference_ticks):
    if current_ticks <= 0 or reference_ticks <= 0:
        raise ValueError("Whole-workload ticks must be positive")
    return dict(
        delta_ticks=current_ticks - reference_ticks,
        improvement_percent=100 * (reference_ticks - current_ticks) / reference_ticks,
        speedup=reference_ticks / current_ticks,
    )


def summarize(rows, counters):
    result = []
    baseline = counters.get("E01")
    previous = None
    for row in rows:
        stage = row["id"]
        current = counters.get(stage)
        if current is None:
            result.append(dict(stage=stage, status="missing-valid-measurement"))
            previous = None
            continue
        metrics = current["metrics"]
        ticks = metrics["BLOCK"]["sim_ticks"]
        record = dict(
            stage=stage,
            status="valid-single-run",
            whole_ticks=ticks,
            milliseconds={
                key: metrics[key]["sim_ticks"] / 1000000000.0 for key in AGGREGATES
            },
            versus_E01=(
                comparison(ticks, baseline["metrics"]["BLOCK"]["sim_ticks"])
                if baseline
                else None
            ),
            versus_previous=(
                comparison(ticks, previous["metrics"]["BLOCK"]["sim_ticks"])
                if previous
                else None
            ),
        )
        result.append(record)
        previous = current
    return result


def counter_differences(current, reference):
    require(set(current["metrics"]) == set(reference["metrics"]), "Phase sets differ")
    require(
        len(current["intervals"]) == len(reference["intervals"]),
        "Interval counts differ",
    )
    scopes = [
        ("phase", name, value, reference["metrics"][name])
        for name, value in current["metrics"].items()
    ]
    for left, right in zip(current["intervals"], reference["intervals"]):
        require(
            left["dump_index"] == right["dump_index"]
            and left["label"] == right["label"],
            "Saved-result integrity check failed: left['dump_index'] == right['dump_index'] and left['label'] == right['label']",
        )
        scopes.append(
            ("interval", str(left["dump_index"]), left["values"], right["values"])
        )
    scopes.append(
        (
            "post_ROI",
            "23",
            current["post_ROI_dump23_delta_separate"],
            reference["post_ROI_dump23_delta_separate"],
        )
    )
    for scope, name, left, right in scopes:
        require(
            set(left) == set(right),
            "Saved-result integrity check failed: set(left) == set(right)",
        )
        for field in sorted(left):
            a, b = (left[field], right[field])
            yield dict(
                scope=scope,
                name=name,
                field=field,
                current=a,
                reference=b,
                delta=None if a is None or b is None else a - b,
            )


def write_table(path, records, delimiter=","):
    if not records:
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0]), delimiter=delimiter
        )
        writer.writeheader()
        writer.writerows(records)


def verify_freeze(freeze, row):
    """Accept both preserved freeze schemas without weakening source binding."""
    commits = [
        freeze[key] for key in ("merge_commit", "integration_commit") if key in freeze
    ]
    require(
        commits and all(commit == row["source_commit"] for commit in commits),
        "Frozen source binding differs",
    )
    require(
        freeze["tag_object"] == row["tag_object"]
        and freeze["production_binary_sha256"] == row["binary_sha256"],
        "Frozen tag or binary binding differs",
    )


def verify_saved_row(study, row, closure):
    root = study / "rows" / row["id"]
    mirror = load(root / "both-host.json")
    final = closure["stage_final_receipts"][row["id"]]
    require(mirror["passed"] and final["passed"], "Unverified mirrored stage")
    require(
        mirror["sha256"] == final["archive_sha256"], "Final archive binding differs"
    )
    manifest = load(root / "member-manifest.json")
    for name in ["audit/counters.json", "audit/completion-audit.json"]:
        require(digest(root / name) == manifest[name]["sha256"], name)
    audit = load(root / "audit/completion-audit.json")
    require(
        digest(root / "audit/completion-audit.json")
        == final["completion_audit_sha256"],
        "Final audit binding differs",
    )
    require(
        audit["passed"]
        and audit["stage"] == row["id"]
        and audit["formal_attempts"] == 1
        and audit["binary_sha256"] == row["binary_sha256"],
        "Stage or binary binding differs",
    )
    freeze = load(study / "stages" / row["id"] / "freeze.json")
    verify_freeze(freeze, row)
    counters = load(root / "audit/counters.json")
    require(
        counters["passed"]
        and counters["raw_dumps"] == 23
        and counters["roi_intervals"] == 22
        and counters["sim_frequency"] == 10**12,
        "Counter scope differs",
    )
    require(
        len(counters["metrics"]) == 15
        and len(counters["intervals"]) == 22
        and all(len(value) == 65 for value in counters["metrics"].values()),
        "Counter fields are incomplete",
    )
    metrics = counters["metrics"]
    require(
        metrics["BLOCK"]["sim_ticks"]
        == sum(metrics[name]["sim_ticks"] for name in AGGREGATES[1:]),
        "Aggregate phases do not sum to whole workload",
    )
    return counters, dict(
        archive_sha256=mirror["sha256"],
        counters_sha256=digest(root / "audit/counters.json"),
        tag_object=row["tag_object"],
        source_commit=row["source_commit"],
        binary_sha256=row["binary_sha256"],
    )


def report_tables(rows, counters):
    table, phases = ([], [])
    for row in rows:
        stage = row["id"]
        if stage not in counters:
            continue
        stats = counters[stage]
        metrics = stats["metrics"]
        total = metrics["BLOCK"]["sim_ticks"] / 1000000000000.0
        implementation = stage + ": " + TITLES[stage]
        if stage in ("E12", "E13"):
            implementation += " (compare aggregate Attention; internal markers move)"
        metadata = dict(
            exp_id=stage,
            study="E01-E13 reconstructed milestones; single runs",
            model="BERT-mini",
            d_q=64,
            d_seq=512,
            d_model=256,
            num_head=4,
            d_ff=1024,
            implementation=implementation,
            n_learners=2,
            codebook_size=4,
            sve_bits=128,
            repo_commit=row["source_commit"],
            binary_sha256=row["binary_sha256"],
            cores=1,
            l1i="48KiB",
            l1d="32KiB",
            l2="1MiB",
        )
        names = list(dict.fromkeys((x["label"] for x in stats["intervals"])))
        for name in names + ["BLOCK"]:
            value = metrics[name]
            table.append(
                dict(
                    metadata,
                    row_kind="final_total" if name == "BLOCK" else "interval_delta",
                    interval=name,
                    sim_seconds=value["sim_ticks"] / 1000000000000.0,
                    instructions=value["instructions"],
                    cpu_cycles=value["cpu_cycles"],
                    dcache_demand_accesses=value["dcache_demand_accesses"],
                    dcache_demand_misses=value["dcache_demand_misses"],
                    l2_demand_accesses=value["l2_demand_accesses"],
                    l2_demand_misses=value["l2_demand_misses"],
                )
            )
        for name in AGGREGATES[1:]:
            phases.append(
                dict(
                    config_id="SVE128 single-core simulated ROI",
                    variant=implementation,
                    replicate="single",
                    scope="simulated",
                    worker_id="all",
                    phase=name,
                    sim_seconds=metrics[name]["sim_ticks"] / 1000000000000.0,
                    roi_seconds=total,
                )
            )
    return (table, phases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Local verified mirror directory"
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registration = load(args.registration)
    rows = registration["rows"]
    budget = study_budget(registration)
    closure = load(args.root / "complete.json")
    require(
        closure["passed"] and closure["mirrored"] == budget,
        "Saved-result integrity check failed: closure['passed'] and closure['mirrored'] == budget",
    )
    require(
        closure["registration_sha256"] == digest(args.registration),
        "Mirror registration binding differs",
    )
    output = args.output
    output.mkdir(exist_ok=False)
    save(
        output / "start.json",
        dict(
            epoch=time.time(),
            source_sha256=digest(__file__),
            registration_sha256=digest(args.registration),
        ),
    )
    counters = {}
    bindings = {}
    for row in rows:
        counters[row["id"]], bindings[row["id"]] = verify_saved_row(
            args.root, row, closure
        )
    summaries = summarize(rows, counters)
    save(
        output / "summary.json",
        dict(
            rows=summaries,
            measured_stages=budget,
            formal_runs=budget,
            primary="whole-workload integer ticks",
            practical_reference_percent=registration["practical_reference_percent"],
            single_measurement_each=True,
            independent_confirmation=False,
            global_optimum_claim=False,
            phase_boundary_limit="E12/E13 streaming moves work into QK markers; compare aggregate Attention.",
            setup_limit="CI01 setup is partial outside ROI; post-ROI dump23 is not pure cleanup.",
            bindings=bindings,
        ),
    )
    comparison_rows = []
    for entry in summaries:
        comparison_rows.append(
            dict(
                stage=entry["stage"],
                whole_ms=entry["milliseconds"]["BLOCK"],
                codebook_ms=entry["milliseconds"]["CODEBOOK"],
                attention_ms=entry["milliseconds"]["ATTENTION"],
                normalization_ms=entry["milliseconds"]["NORMALIZATION_PHASES"],
                improvement_vs_E01_percent=entry["versus_E01"]["improvement_percent"],
                speedup_vs_E01=entry["versus_E01"]["speedup"],
                improvement_vs_previous_percent=(
                    entry["versus_previous"]["improvement_percent"]
                    if entry["versus_previous"]
                    else None
                ),
            )
        )
    write_table(output / "stage-comparisons.csv", comparison_rows)
    count = 0
    with gzip.open(output / "all-counter-differences.jsonl.gz", "xt") as stream:
        for i, row in enumerate(rows):
            references = [("E01", "baseline")] + (
                [(rows[i - 1]["id"], "adjacent")] if i else []
            )
            for reference, kind in references:
                for difference in counter_differences(
                    counters[row["id"]], counters[reference]
                ):
                    stream.write(
                        json.dumps(
                            dict(
                                stage=row["id"],
                                reference_stage=reference,
                                comparison=kind,
                                **difference,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    count += 1
    require(
        count == (2 * budget - 1) * 38 * 65,
        "Saved-result integrity check failed: count == (2 * budget - 1) * 38 * 65",
    )
    table, phases = report_tables(rows, counters)
    write_table(output / "results_all_experiments.tsv", table, "\t")
    write_table(output / "phase_results.csv", phases)
    setup = []
    for row in rows:
        audit = load(args.root / "rows" / row["id"] / "audit/completion-audit.json")
        lines = audit["setup_outside_ROI"]
        value = int(re.search("setup_ns=(\\d+)", lines[0])[1]) if lines else None
        setup.append(
            dict(
                stage=row["id"],
                CI01_partial_setup_ns=value,
                applicable=bool(lines),
                inside_primary_ROI=False,
            )
        )
    save(output / "partial-setup.json", setup)
    save(
        output / "complete.json",
        dict(
            passed=True,
            epoch=time.time(),
            difference_rows=count,
            outputs={p.name: digest(p) for p in output.iterdir() if p.is_file()},
            simulator_entries=0,
            raw_stats_reparsed=0,
        ),
    )


if __name__ == "__main__":
    main()
