#!/usr/bin/env python3
"""Verify historical False measurements and generate compact profiling data.

Run on the analysis server. Example:
  build_tile_evidence.py --evidence-root /path/to/learner-grid-v1 \
      --output reports/data/false-tiles.json
This reads existing results/configurations; it never runs a simulator or fits a model.
"""
import argparse
import configparser
import hashlib
import itertools
import json
from pathlib import Path

AXES = ("s1", "o1", "k1", "s2", "o2", "k2")
CACHES = {
    "A": (32768, 1048576),
    "B": (65536, 1048576),
    "C": (32768, 2097152),
    "D": (65536, 2097152),
}
BASELINE = (32, 32, 256, 256, 32, 256)
GRID_SHA = "8f44683013bdf0086d17c1505267a20cdee177de9d35b21172b3da1d6b31b64f"
COMPARISON_SHA = "401d6182bb63ff58d33259aeaaba02405a52617d271c282b1660bfb53e8062fa"
FORMULA_SHA = "2cba8531851ad94c7926a2cfd7cbb92e03f049f3d1b251df8602057199084db4"
METRICS = (
    "sim_ticks",
    "dcache_demand_accesses",
    "dcache_demand_misses",
    "l2_demand_accesses",
    "l2_demand_misses",
    "instructions",
)


def require(condition, message):
    """Reject evidence that does not satisfy the frozen study contract."""
    if not condition:
        raise ValueError(message)


def digest(path):
    """Hash source evidence before using it in the report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pinned(path, expected):
    """Read JSON only after checking its expected digest."""
    require(digest(path) == expected, "Source hash mismatch: " + str(path))
    return json.loads(path.read_text())


def geometry(row):
    """Return a stable six-axis tile identity."""
    return tuple(row["tiles"][a] for a in AXES)


def profile(row):
    """Return the cache/cardinality identity of an observation."""
    return row["cache"], row["learners"], row["entries"]


def reduction(selected, reference):
    """Express runtime reduction relative to the reference, preserving regressions."""
    return 100 * (1 - selected / reference)


def verify_config(path, l1, l2):
    """Check cache sizes and writeback_clean=False in the actual configuration."""
    cfg = configparser.ConfigParser()
    with path.open() as stream:
        cfg.read_file(stream)
    data = [s for s in cfg if s.endswith(".dcache")]
    outer = [s for s in cfg if s.endswith(".l2")]
    require(
        len(data) == 1 and len(outer) == 1, "Expected one L1D and one L2: " + str(path)
    )
    require(
        cfg[data[0]].get("writeback_clean") == "false",
        "L1D.writeback_clean is not False",
    )
    require(
        int(cfg[data[0]]["size"]) == l1 and int(cfg[outer[0]]["size"]) == l2,
        "Actual cache capacities differ from profile",
    )


def resolve_result(root, historical):
    """Locate a historical result without changing its stored provenance."""
    marker = "/runtime-tiles-cardinality-v1/"
    require(marker in historical, "Unknown historical path namespace")
    relative = Path(historical.split(marker, 1)[1])
    require(
        not relative.is_absolute() and ".." not in relative.parts, "Unsafe result path"
    )
    path = root.parent / relative
    require(
        root.parent.resolve() in path.resolve().parents, "Result escapes evidence root"
    )
    return path, str(relative)


def verify_measurement(root, ref, cache, learners, entries):
    """Verify one result and retain raw phase counters and binary identity."""
    path, relative = resolve_result(root, ref["result_path"])
    require(digest(path) == ref["result_sha256"], "Result hash mismatch: " + relative)
    raw = json.loads(path.read_text())
    require(raw["passed"] is True and raw["id"] == ref["id"], "Invalid result receipt")
    require(raw["binary_sha256"] == ref["binary_sha256"], "Binary identity mismatch")
    require(geometry(raw) == geometry(ref), "Tile identity mismatch")
    # The original two-learner study predates the learners field. Its identity
    # comes from the pinned grid manifest, not an assumed default for any file.
    recorded_learners = raw.get("learners")
    if recorded_learners is None:
        require(
            relative.startswith("grid/runs/") and learners == 2,
            "Missing learner identity outside the original two-learner study",
        )
        recorded_learners = 2
    require(
        (raw["cache"], recorded_learners, raw["entries"]) == (cache, learners, entries),
        "Profile identity mismatch",
    )
    require(
        (raw["L1D_bytes"], raw["L2_bytes"]) == CACHES[cache],
        "Result capacities mismatch",
    )
    config = path.parent / "formal/config.ini"
    verify_config(config, *CACHES[cache])
    config_sha = digest(config)
    if "config_sha256" in ref:
        require(config_sha == ref["config_sha256"], "Config hash mismatch")
    metrics = raw["metrics"]
    for phase in ("BLOCK", "CODEBOOK"):
        ticks = metrics[phase]["sim_ticks"]
        require(type(ticks) is int and ticks > 0, "Missing integer runtime")
        require(ticks == ref[phase + "_ticks"], "Runtime differs from result receipt")
    return {
        "id": raw["id"],
        "cache": cache,
        "learners": learners,
        "entries": entries,
        "tiles": raw["tiles"],
        "writeback_clean": False,
        "BLOCK_ticks": metrics["BLOCK"]["sim_ticks"],
        "CODEBOOK_ticks": metrics["CODEBOOK"]["sim_ticks"],
        "phases": {p: {k: v.get(k) for k in METRICS} for p, v in metrics.items()},
        "binary_sha256": raw["binary_sha256"],
        "source_commit": raw.get("source_commit", ref.get("source_commit")),
        "learner_identity_source": (
            "result" if "learners" in raw else "pinned_grid_manifest"
        ),
        "result_reference": relative,
        "result_sha256": ref["result_sha256"],
        "config_reference": str(Path(relative).parent / "formal/config.ini"),
        "config_sha256": config_sha,
        "production_tensor_observed": raw.get(
            "actual_production_tensor_values_observed", False
        ),
    }


def load_study_sources(root):
    """Load the pinned study and enforce its measurement accounting."""
    grid_path = root / "false-grid-1728-heatmaps-v1/observations-1728.json"
    comparison_path = (
        root / "false-hierarchical-validation-v9/results-v1/comparison.json"
    )
    grid = load_pinned(grid_path, GRID_SHA)
    comparison = load_pinned(comparison_path, COMPARISON_SHA)
    require(
        len(grid) == 1728 and len(comparison["profiles"]) == 36, "Unexpected coverage"
    )
    require(
        comparison["count_new_selections"] == 15
        and comparison["count_reused_selections"] == 21
        and comparison["matched_controls"] == 12,
        "Unexpected v9 measurement accounting",
    )
    require(comparison["timing_frequency_Hz"] == 10**12, "Unexpected tick frequency")
    return grid, comparison


def expected_geometries():
    """Describe the 48 legal geometries in each historical grid profile."""
    expected = set(
        itertools.product(
            (16, 32), (32, 64), (128, 256), (128, 256), (32, 64), (256, 512)
        )
    )
    expected = {t for t in expected if not (t[1] == 64 and t[4] == 32)}
    return expected


def verify_grid(root, grid, expected):
    """Verify every grid observation and each profile's coverage and binary."""
    points, groups, ids = {}, {}, set()
    for row in grid:
        require(row["policy"] is False, "Non-False grid row")
        key = profile(row)
        require(
            key in set(itertools.product(CACHES, (1, 2, 4), (4, 8, 16))),
            "Unknown profile",
        )
        require(row["id"] not in ids, "Duplicate grid ID")
        ids.add(row["id"])
        groups.setdefault(key, []).append(row)
        points[row["id"]] = verify_measurement(root, row, *key)
    require(len(groups) == 36, "Incomplete profiles")
    for rows in groups.values():
        require(
            len(rows) == 48 and {geometry(r) for r in rows} == expected,
            "Grid coverage mismatch",
        )
        require(
            len({r["binary_sha256"] for r in rows}) == 1,
            "Mixed binaries inside original grid",
        )
    return points, groups


def merge_references(root, r, key, points, new_ids):
    """Resolve comparison roles and reject conflicting reused observations."""
    refs = {}
    for name in (
        "selected",
        "historical_fixed_baseline",
        "historical_grid_best",
        "matched_fixed_baseline",
        "matched_grid_winner_tile",
    ):
        ref = r.get(name)
        refs[name] = ref["id"] if ref else None
        if ref:
            point = verify_measurement(root, ref, *key)
            if point["id"] in points:
                require(points[point["id"]] == point, "Conflicting reused result")
            points[point["id"]] = point
            if point["id"].startswith("HV9-"):
                new_ids.add(point["id"])
    return refs


def comparison_profile(r, key, groups, points, refs, expected):
    """Check the selected geometry and same-binary deltas for one profile."""
    selected = points[refs["selected"]]
    require(
        selected["tiles"] == r["scheduler_tiles"],
        "Selection differs from frozen v9",
    )
    historical_best = points[refs["historical_grid_best"]]
    best_ticks = min(x["BLOCK_ticks"] for x in groups[key])
    require(
        historical_best["BLOCK_ticks"] == best_ticks,
        "Historical winner is not grid minimum",
    )
    require(
        geometry(points[refs["historical_fixed_baseline"]]) == BASELINE,
        "Wrong baseline geometry",
    )
    deltas = {}
    for name in ("matched_fixed_baseline", "matched_grid_winner_tile"):
        reference = points[refs[name]] if refs[name] else None
        matched = reference and reference["binary_sha256"] == selected["binary_sha256"]
        deltas[name] = (
            reduction(selected["BLOCK_ticks"], reference["BLOCK_ticks"])
            if matched
            else None
        )
    require(
        all(value is not None for value in deltas.values()),
        "Primary comparison lacks matching binary",
    )
    require(
        abs(deltas["matched_fixed_baseline"] - r["vs_matched_baseline_reduction_pct"])
        < 1e-9,
        "Stored baseline comparison mismatch",
    )
    require(
        abs(
            deltas["matched_grid_winner_tile"]
            - r["vs_matched_grid_winner_tile_reduction_pct"]
        )
        < 1e-9,
        "Stored winner comparison mismatch",
    )
    return {
        "cache": key[0],
        "learners": key[1],
        "entries": key[2],
        "inputs": r["inputs"],
        "grid_ids": [x["id"] for x in sorted(groups[key], key=geometry)],
        "grid_best_ids": [
            x["id"] for x in groups[key] if x["BLOCK_ticks"] == best_ticks
        ],
        "references": refs,
        "selected_geometry_in_grid": geometry(selected) in expected,
        "selected_binary_is_grid_binary": selected["binary_sha256"]
        == historical_best["binary_sha256"],
        "measurement_status": "MEASURED",
        "reductions_pct": deltas,
        "cross_binary_baseline_reduction_pct": r[
            "vs_historical_baseline_reduction_pct"
        ],
        "selection_origin": r["selected"]["origin"],
    }


def verify_profiles(root, comparison, points, groups, expected):
    """Verify all 36 comparison profiles and the exact 27 new measurements."""
    profiles = []
    seen = set()
    new_ids = set()
    for r in comparison["profiles"]:
        i = r["inputs"]
        key = (r["cache"], i["learners"], i["entries"])
        require(
            key not in seen and key in groups, "Duplicate/missing comparison profile"
        )
        seen.add(key)
        refs = merge_references(root, r, key, points, new_ids)
        profiles.append(comparison_profile(r, key, groups, points, refs, expected))
    require(len(new_ids) == 27, "Expected exactly 27 distinct new v9 measurements")
    return profiles


def summarize_profiles(profiles):
    """Aggregate the four verified cache settings for each learner/CB pair."""
    summaries = []
    for n, cb in itertools.product((1, 2, 4), (4, 8, 16)):
        rows = [p for p in profiles if p["learners"] == n and p["entries"] == cb]
        values = [p["reductions_pct"]["matched_fixed_baseline"] for p in rows]
        summaries.append(
            {
                "learners": n,
                "entries": cb,
                "profiles": 4,
                "mean_baseline_reduction_pct": sum(values) / 4,
                "min_baseline_reduction_pct": min(values),
                "max_baseline_reduction_pct": max(values),
            }
        )
    return summaries


def build(root):
    """Assemble verified grid, comparison profiles and display metadata."""
    grid, comparison = load_study_sources(root)
    expected = expected_geometries()
    points, groups = verify_grid(root, grid, expected)
    profiles = verify_profiles(root, comparison, points, groups, expected)
    summaries = summarize_profiles(profiles)
    return {
        "schema": "i2ce.false-grid-v9.v1",
        "axes": AXES,
        "cache_bytes": CACHES,
        "tick_frequency_hz": 10**12,
        "baseline_tiles": BASELINE,
        "aggregation": "Arithmetic mean of per-profile runtime reductions; each of four caches has weight 1/4.",
        "percent_formula": "100 * (1 - selected_ticks / reference_ticks)",
        "new_code_performance_measured": False,
        "counts": {
            "grid": 1728,
            "profiles": 36,
            "new_selections": 15,
            "reused_selections": 21,
            "new_same_binary_controls": 12,
            "new_measurements": 27,
        },
        "sources": {
            "grid_sha256": GRID_SHA,
            "comparison_sha256": COMPARISON_SHA,
            "frozen_scheduler_sha256": FORMULA_SHA,
        },
        "points": points,
        "profiles": profiles,
        "summaries": summaries,
    }


def main():
    """Write compact evidence and its verification manifest; never simulate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = build(args.evidence_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"
    )
    manifest = {
        "passed": True,
        "data_sha256": digest(args.output),
        "generator_sha256": digest(Path(__file__)),
        "sources": data["sources"],
        "verified_result_config_pairs": len(data["points"]),
        "formal_performance_runs": 0,
        "counts": data["counts"],
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
