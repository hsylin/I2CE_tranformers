"""Check saved-result signs, missing stages and undefined counter differences."""

import tempfile
import json
import subprocess
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/exp/lib"))
import study_results as results


def sample(ticks):
    metrics = {
        name: dict(sim_ticks=ticks if name == "BLOCK" else ticks // 3, rate=None)
        for name in results.AGGREGATES
    }
    return dict(
        metrics=metrics,
        intervals=[dict(dump_index=1, label="phase", values={"x": 2})],
        post_ROI_dump23_delta_separate={"x": 3},
    )


class ResultsTests(unittest.TestCase):
    def test_preserved_freeze_formats_require_same_exact_commit(self):
        row = dict(source_commit="source", tag_object="tag", binary_sha256="binary")
        for key in ("merge_commit", "integration_commit"):
            freeze = dict(
                tag_object="tag", production_binary_sha256="binary", **{key: "source"}
            )
            results.verify_freeze(freeze, row)
            freeze[key] = "wrong revision"
            with self.assertRaisesRegex(ValueError, "source binding"):
                results.verify_freeze(freeze, row)
        with self.assertRaisesRegex(ValueError, "source binding"):
            results.verify_freeze(
                dict(merge_commit="source", integration_commit="conflict"), row
            )

    def test_dropped_interval_is_rejected(self):
        old, new = sample(100), sample(80)
        new["intervals"] = []
        with self.assertRaisesRegex(ValueError, "Interval counts"):
            list(results.counter_differences(new, old))

    def test_optimized_python_still_requires_normal_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registration = dict(
                maximum_formal_runs=1,
                automatic_retries=0,
                rows=[dict(id="E01", state="/unused/state", output="/unused/output")],
            )
            (root / "registration.json").write_text(json.dumps(registration))
            (root / "complete.json").write_text(
                json.dumps(dict(passed=False, mirrored=1))
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-O",
                    results.__file__,
                    "--root",
                    str(root),
                    "--registration",
                    str(root / "registration.json"),
                    "--output",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError", result.stderr)
            self.assertFalse((root / "results").exists())

    def test_complete_saved_evaluation_and_binary_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = dict(
                id="E01",
                state="/unused/state",
                output="/unused/output",
                binary_sha256="binary",
                source_commit="source",
                tag_object="tag",
            )
            registration = dict(
                maximum_formal_runs=1,
                automatic_retries=0,
                rows=[row],
                practical_reference_percent=1,
            )

            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            write(root / "registration.json", registration)
            saved = root / "rows/E01"
            names = [
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
                *results.AGGREGATES,
                "OTHER",
            ]
            required = [
                "sim_ticks",
                "instructions",
                "cpu_cycles",
                "dcache_demand_accesses",
                "dcache_demand_misses",
                "l2_demand_accesses",
                "l2_demand_misses",
            ]
            values = {
                key: 0 for key in required + ["field_%02d" % i for i in range(58)]
            }
            metrics = {name: dict(values, sim_ticks=10) for name in names}
            metrics["BLOCK"]["sim_ticks"] = 30
            counter = dict(
                passed=True,
                raw_dumps=23,
                roi_intervals=22,
                sim_frequency=10**12,
                metrics=metrics,
                intervals=[
                    dict(dump_index=i + 1, label=names[i % 10], values=values)
                    for i in range(22)
                ],
                post_ROI_dump23_delta_separate=values,
            )
            audit = dict(
                passed=True,
                stage="E01",
                formal_attempts=1,
                binary_sha256="binary",
                setup_outside_ROI=[],
            )
            write(saved / "audit/counters.json", counter)
            write(saved / "audit/completion-audit.json", audit)
            write(
                saved / "member-manifest.json",
                {
                    name: dict(sha256=results.digest(saved / name))
                    for name in ["audit/counters.json", "audit/completion-audit.json"]
                },
            )
            write(saved / "both-host.json", dict(passed=True, sha256="archive"))
            write(
                root / "stages/E01/freeze.json",
                dict(
                    tag_object="tag",
                    merge_commit="source",
                    production_binary_sha256="binary",
                ),
            )
            closure = dict(
                passed=True,
                mirrored=1,
                registration_sha256=results.digest(root / "registration.json"),
                stage_final_receipts={
                    "E01": dict(
                        passed=True,
                        archive_sha256="archive",
                        completion_audit_sha256=results.digest(
                            saved / "audit/completion-audit.json"
                        ),
                    )
                },
            )
            write(root / "complete.json", closure)
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    results.__file__,
                    "--root",
                    str(root),
                    "--registration",
                    str(root / "registration.json"),
                    "--output",
                    str(root / "results"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            complete = json.loads((root / "results/complete.json").read_text())
            self.assertEqual(complete["difference_rows"], 38 * 65)
            self.assertEqual(complete["raw_stats_reparsed"], 0)
            self.assertEqual(
                json.loads((root / "results/summary.json").read_text())["rows"][0][
                    "whole_ticks"
                ],
                30,
            )
            row["binary_sha256"] = "another binary"
            with self.assertRaisesRegex(ValueError, "binary binding"):
                results.verify_saved_row(root, row, closure)

    def test_signed_gain_and_regression(self):
        self.assertEqual(
            results.comparison(80, 100),
            dict(delta_ticks=-20, improvement_percent=20, speedup=1.25),
        )
        self.assertEqual(results.comparison(110, 100)["improvement_percent"], -10)
        self.assertEqual(results.comparison(100, 100)["delta_ticks"], 0)
        with self.assertRaises(ValueError):
            results.comparison(0, 100)

    def test_missing_stage_does_not_bridge_adjacent_comparison(self):
        rows = [{"id": "E01"}, {"id": "E02"}, {"id": "E03"}]
        out = results.summarize(rows, {"E01": sample(100), "E03": sample(80)})
        self.assertEqual(out[1]["status"], "missing-valid-measurement")
        self.assertIsNone(out[2]["versus_previous"])
        self.assertEqual(out[2]["versus_E01"]["delta_ticks"], -20)

    def test_export_matches_migrated_report_schemas(self):
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "transformer_profiling")
        )
        import report

        report.load_dependencies()
        names = sorted(report.SCHEMA_V2)
        values = dict(
            sim_ticks=1000,
            instructions=100,
            cpu_cycles=250,
            dcache_demand_accesses=20,
            dcache_demand_misses=2,
            l2_demand_accesses=2,
            l2_demand_misses=1,
        )
        metrics = {n: dict(values) for n in names}
        metrics["BLOCK"] = {k: v * 10 for k, v in values.items()}
        for n, multiplier in [
            ("CODEBOOK", 4),
            ("ATTENTION", 4),
            ("NORMALIZATION_PHASES", 2),
        ]:
            metrics[n] = {k: v * multiplier for k, v in values.items()}
        counter = dict(metrics=metrics, intervals=[{"label": n} for n in names])
        row = dict(id="E13", source_commit="fixture", binary_sha256="fixture")
        table, phases = results.report_tables([row], {"E13": counter})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results.write_table(root / "results.tsv", table, "\t")
            results.write_table(root / "phases.csv", phases)
            frame, total, stage = report.load_data(root / "results.tsv")
            self.assertEqual(len(total), 1)
            self.assertEqual(len(stage), 10)
            charts = report.implementation_phase_charts(root / "phases.csv")
            self.assertEqual(
                charts[0]["layout"]["yaxis"]["title"]["text"],
                "Share of ROI (%)",
            )

    def test_keep_negative_zero_and_undefined_counter_differences(self):
        old, new = sample(100), sample(80)
        new["intervals"][0]["values"]["x"] = 1
        differences = list(results.counter_differences(new, old))
        self.assertIn(-20, [x["delta"] for x in differences])
        self.assertIn(-1, [x["delta"] for x in differences])
        self.assertIn(0, [x["delta"] for x in differences])
        self.assertIn(None, [x["delta"] for x in differences])


if __name__ == "__main__":
    unittest.main()
