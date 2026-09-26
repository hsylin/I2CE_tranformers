#!/usr/bin/env python3
"""Behavior checks for the offline report; requires the visualization requirements.

python tests/profiling_report_test.py
"""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "transformer_profiling/report.py"
SOURCE = ROOT / "transformer_profiling/final/final_all_experiments.tsv"
sys.path.insert(0, str(SCRIPT.parent))
import report
report.load_dependencies()
pd = report.pd


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.raw = pd.read_csv(SOURCE, sep="\t")
        self.raw = self.raw[self.raw.exp_id.str.extract(r"(\d+)")[0].astype(int) <= 36].copy()

    def load(self, data):
        path = self.directory / "input.tsv"
        data.to_csv(path, sep="\t", index=False)
        return report.load_data(path)

    def cli(self, *args, no_site=False):
        return subprocess.run([sys.executable, *(["-S"] if no_site else []), str(SCRIPT), *map(str, args)],
                              cwd=self.directory, capture_output=True, text=True)

    def test_historical_metrics_and_source_anomalies(self):
        frame, total, stage = self.load(self.raw)
        self.assertEqual((len(frame), len(total), len(stage)), (252, 36, 216))
        row = stage[(stage.exp_id == "E16") & (stage.interval == "FF1")].iloc[0]
        self.assertAlmostEqual(row.l2_miss_pct, 95.17305398977166)
        self.assertEqual(total[total.estimated].exp_id.tolist(), ["E05", "E06"])
        self.assertEqual(total[~total.phase_timing_valid].exp_id.tolist(), ["E02"])
        fig = report.phase_chart(total[total.model == "BERT-mini"], stage[stage.model == "BERT-mini"], "BERT-mini")
        for trace in fig["data"]:
            self.assertNotIn("E02", trace["y"])

    def test_new_experiment_keeps_repeated_settings_separate(self):
        new = self.raw[self.raw.exp_id == "E09"].copy()
        new["exp_id"] = "E37"
        new["study"] = "Runner"
        new["sim_seconds"] *= 0.8
        _, total, stage = self.load(pd.concat([self.raw, new], ignore_index=True))
        self.assertEqual(len(total), 37)
        mini = total[total.model == "BERT-mini"]
        curve = report.sve_chart(mini, "BERT-mini")
        four = next(t for t in curve["data"] if t["name"] == "4 learners")
        self.assertEqual(len(four["x"]), 4)
        self.assertEqual(four["mode"], "markers")
        self.assertTrue(all(row[2] is None for row in four["customdata"]))
        grid = report.codebook_chart(mini, "BERT-mini")
        self.assertTrue(math.isnan(grid["data"][0]["z"][1][0]))
        self.assertIn("E09", grid["data"][-1]["customdata"][0])
        self.assertIn("E37", grid["data"][-1]["customdata"][0])
        self.assertEqual(sum(len(t["x"]) for t in report.runtime_chart(mini, "BERT-mini")["data"]), 19)

    def test_zero_denominators_remain_missing(self):
        self.raw.loc[0, ["dcache_demand_accesses", "dcache_demand_misses", "instructions", "cpu_cycles"]] = 0
        frame, _, _ = self.load(self.raw)
        self.assertTrue(pd.isna(frame.iloc[0].l1d_miss_pct))
        self.assertTrue(pd.isna(frame.iloc[0].l1d_mpki))
        self.assertTrue(pd.isna(frame.iloc[0].ipc_calc))

    def test_duplicate_and_incomplete_regions_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load(pd.concat([self.raw, self.raw.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, "six"):
            self.load(self.raw.iloc[1:])

    def test_optional_tile_medians_and_missing_cells(self):
        path = self.directory / "tiles.csv"
        path.write_text("config_id,tile_m,tile_n,tile_k,replicate,sim_seconds,l1d_accesses,l1d_misses,l2_accesses,l2_misses\n"
                        "test_only,32,32,16,1,10,100,20,20,10\n"
                        "test_only,32,32,16,2,8,100,10,20,5\n"
                        "test_only,32,64,16,1,6,100,10,20,5\n"
                        "test_only,64,32,16,1,7,100,15,20,7\n")
        fig = report.clean(report.tile_charts(path)[0])
        self.assertEqual(fig["data"][0]["z"][0][0], 9)
        self.assertIsNone(fig["data"][0]["z"][1][1])
        self.assertEqual(fig["data"][1]["z"][0][0], 15)

    def test_scaling_uses_one_core_not_learner_count(self):
        path = self.directory / "scaling.csv"
        path.write_text("config_id,num_cores,replicate,sim_seconds\ntest_only,1,1,10\ntest_only,2,1,6\ntest_only,4,1,4\n")
        fig = report.scaling_charts(path)[0]
        self.assertEqual(fig["data"][0]["y"], [1, 10 / 6, 2.5])
        path.write_text("config_id,num_cores,replicate,sim_seconds\ntest_only,2,1,6\n")
        with self.assertRaisesRegex(ValueError, "1-core"):
            report.scaling_charts(path)

    def test_tile_footprint_uses_supplied_bytes_and_fixed_capacity(self):
        path = self.directory / "tiles.csv"
        header = "config_id,tile_m,tile_n,tile_k,replicate,sim_seconds,l1d_accesses,l1d_misses,l2_accesses,l2_misses,tile_footprint_bytes,l2_capacity_bytes\n"
        rows = ["test_only,32,32,16,1,10,100,20,20,10,65536,1048576\n",
                "test_only,32,32,16,2,8,100,10,20,5,65536,1048576\n",
                "test_only,64,32,16,1,7,100,15,20,7,131072,1048576\n"]
        path.write_text(header + "".join(rows))
        figures = report.tile_charts(path)
        self.assertEqual(len(figures), 2)
        self.assertEqual(figures[1]["data"][0]["x"], [64, 128])
        self.assertEqual(figures[1]["data"][0]["y"], [9, 7])
        self.assertEqual(figures[1]["layout"]["shapes"][0]["x0"], 1024)
        path.write_text(header + "".join(rows) + "test_only,32,32,32,1,5,100,10,20,5,131072,2097152\n")
        with self.assertRaisesRegex(ValueError, "capacity"):
            report.tile_charts(path)

    def test_efficiency_and_bandwidth_use_per_run_roi_measurements(self):
        path = self.directory / "scaling.csv"
        path.write_text("config_id,num_cores,replicate,sim_seconds,dram_bytes\n"
                        "test_only,1,1,10,1000000000\ntest_only,1,2,2,6000000000\n"
                        "test_only,2,1,4,4000000000\ntest_only,2,2,2,4000000000\n")
        figures = report.scaling_charts(path)
        self.assertEqual(figures[0]["data"][0]["y"], [1, 2])
        self.assertEqual(figures[1]["data"][0]["y"], [100, 100])
        self.assertEqual(figures[2]["data"][0]["y"], [1.55, 1.5])
        data = pd.read_csv(path)
        data.loc[0, "dram_bytes"] = None
        data.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "every row"):
            report.scaling_charts(path)

    def phase_fixture(self):
        rows = []
        for rep, roi, staging, kernel in [(1, 10, 2, 6), (2, 8, 1, 5)]:
            rows.extend([["test_only", "tiled", rep, "wall", "all", phase, duration, roi]
                         for phase, duration in [("staging", staging), ("kernel", kernel), ("barrier", 0)]])
            for worker, compute, wait in [(0, roi - 3, 1), (1, roi - 5, 3)]:
                rows.extend([["test_only", "tiled", rep, "worker", worker, phase, duration, roi]
                             for phase, duration in [("staging", 1), ("kernel", compute), ("barrier", wait)]])
        return pd.DataFrame(rows, columns=["config_id", "variant", "replicate", "scope", "worker_id", "phase", "sim_seconds", "roi_seconds"])

    def test_phases_preserve_wall_and_worker_time_and_unattributed_roi(self):
        path = self.directory / "phases.csv"
        self.phase_fixture().to_csv(path, index=False)
        wall, workers = report.implementation_phase_charts(path)
        self.assertEqual([t["name"] for t in wall["data"]], ["staging", "kernel", "barrier", "Unattributed"])
        self.assertEqual([t["y"] for t in wall["data"]], [[1.5], [5.5], [0.0], [2.0]])
        self.assertAlmostEqual(wall["data"][0]["customdata"][0][1], 100 * 1.5 / 9)
        self.assertEqual(workers["data"][0]["x"], ["tiled / worker 0", "tiled / worker 1"])
        self.assertEqual(workers["data"][1]["y"], [6, 4])
        self.assertEqual([sum(t["y"][i] for t in workers["data"]) for i in [0, 1]], [9, 9])
        for fig in [wall, workers]:
            percentages = fig["layout"]["updatemenus"][0]["buttons"][1]["args"][0]["y"]
            for column in zip(*percentages):
                self.assertAlmostEqual(sum(column), 100)

    def test_invalid_phase_accounting_is_rejected(self):
        path = self.directory / "phases.csv"
        data = self.phase_fixture()
        cases = []
        overlapping = data.copy()
        overlapping.loc[0, "sim_seconds"] = 9
        cases.append((overlapping, "exceed ROI"))
        inconsistent_roi = data.copy()
        inconsistent_roi.loc[(data.scope == "worker") & (data.replicate == 1), "roi_seconds"] = 12
        cases.append((inconsistent_roi, "same ROI"))
        cases.append((data.drop(index=0), "Phase sets"))
        cases.append((data[~((data.scope == "worker") & (data.worker_id == 1) & (data.replicate == 2))], "Worker IDs"))
        reserved_name = data.copy()
        reserved_name.loc[0, "phase"] = "__roi"
        cases.append((reserved_name, "reserved"))
        for frame, message in cases:
            with self.subTest(message=message):
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    report.implementation_phase_charts(path)

    def test_phase_discovery_and_header_only_templates(self):
        output = self.directory / "report.html"
        self.raw.to_csv(self.directory / "final_all_experiments.tsv", sep="\t", index=False)
        for template in (ROOT / "transformer_profiling/templates").glob("*_results.csv"):
            shutil.copy(template, self.directory / template.name)
        result = self.cli("--input", self.directory, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotRegex(output.read_text(), r'data-view="(?:tiles|scaling|phases)"')
        phases = self.directory / "phase_results.csv"
        self.phase_fixture().to_csv(phases, index=False)
        result = self.cli("--input", self.directory, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('data-view="phases"', output.read_text())
        self.assertIn(hashlib.sha256(phases.read_bytes()).hexdigest(), output.read_text())
        previous = output.read_bytes()
        invalid = self.phase_fixture()
        invalid.loc[0, "sim_seconds"] = 100
        invalid.to_csv(phases, index=False)
        result = self.cli("--input", self.directory, "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), previous)

    def test_help_and_missing_dependency_message(self):
        help_result = self.cli("--help", no_site=True)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        result = self.cli("--output", self.directory / "out.html", no_site=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requirements-visualization.txt", result.stderr)

    def test_output_validation_preserves_existing_files(self):
        output = self.directory / "report.html"
        output.write_text("previous report")
        result = self.cli("--experiments", "E999", "--output", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_text(), "previous report")
        result = self.cli("--output", SOURCE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not overwrite", result.stderr)
        result = self.cli("--output", output, "--metrics-output", output)
        self.assertNotEqual(result.returncode, 0)

    def test_one_command_from_other_directory_and_dense_only_subset(self):
        output = self.directory / "all.html"
        env = dict(os.environ, REPORT_PYTHON=sys.executable,
                   GEM5_BIN="/does/not/exist", GEM5_CWD="/does/not/exist")
        original_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        result = subprocess.run(["bash", str(ROOT / "tools/exp/exp.sh"), "report", "--output", str(output)],
                                cwd=self.directory, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        expected = report.load_data(SOURCE.parent)[1]
        self.assertIn("Experiments: " + str(len(expected)), result.stdout)
        text = output.read_text()
        specs = json.loads(re.search(r'id="chart-specs">(.*?)</script>', text, re.S).group(1))
        self.assertTrue(specs)
        self.assertIn("plotly.js v", text)
        self.assertNotRegex(text, r'<script[^>]+src=')
        self.assertIn(original_hash, text)
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), original_hash)
        result = self.cli("--experiments", "E01", "E04", "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Experiments: 2", result.stdout)

    def test_directory_discovers_new_file_and_overrides_stale_combined_rows(self):
        combined = self.directory / "final_all_experiments.tsv"
        self.raw.to_csv(combined, sep="\t", index=False)
        new = self.raw[self.raw.exp_id == "E09"].copy()
        new["exp_id"] = "E37"
        new.to_csv(self.directory / "e37_new.tsv", sep="\t", index=False)
        updated = self.raw[self.raw.exp_id == "E09"].copy()
        updated["sim_seconds"] *= 0.8
        updated.to_csv(self.directory / "e09_updated.tsv", sep="\t", index=False)
        _, total, _ = report.load_data(self.directory)
        self.assertEqual(len(total), 37)
        self.assertEqual(total.loc[total.exp_id == "E09", "sim_seconds"].iloc[0],
                         updated.loc[updated.row_kind == "final_total", "sim_seconds"].iloc[0])
        self.assertEqual(len(pd.read_csv(combined, sep="\t")), 252)
        combined.unlink()
        self.assertEqual(len(report.load_data(self.directory)[1]), 2)

    def test_directory_rejects_duplicate_files_and_mismatched_ids(self):
        data = self.raw[self.raw.exp_id == "E09"]
        data.to_csv(self.directory / "e09_a.tsv", sep="\t", index=False)
        data.to_csv(self.directory / "e09_b.tsv", sep="\t", index=False)
        with self.assertRaisesRegex(ValueError, "Duplicate per-experiment"):
            report.load_data(self.directory)
        (self.directory / "e09_b.tsv").rename(self.directory / "e38_wrong.tsv")
        with self.assertRaisesRegex(ValueError, "filename and experiment ID"):
            report.load_data(self.directory)

    def test_english_charts_only_and_optional_tabs_require_data(self):
        output = self.directory / "report.html"
        result = self.cli("--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = output.read_text()
        main = re.search(r"<main>(.*?)</main>", text, re.S).group(1)
        self.assertIn('<html lang="en">', text)
        self.assertNotRegex(main, r"[\u3400-\u9fff]")
        self.assertNotRegex(main, r"<(?:p|footer|table|details)(?:\s|>)")
        self.assertNotIn('data-view="tiles"', main)
        self.assertNotIn('data-view="scaling"', main)
        self.assertNotIn('data-view="phases"', main)
        self.assertNotIn("Perfetto", main)
        self.raw.to_csv(self.directory / "final_all_experiments.tsv", sep="\t", index=False)
        (self.directory / "scaling_results.csv").write_text(
            "config_id,num_cores,replicate,sim_seconds\ntest_only,1,1,10\ntest_only,2,1,6\n")
        result = self.cli("--input", self.directory, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('data-view="scaling"', output.read_text())
        self.assertNotIn('data-view="tiles"', output.read_text())

    def test_directory_output_cannot_overwrite_or_create_result_inputs(self):
        self.raw.to_csv(self.directory / "final_all_experiments.tsv", sep="\t", index=False)
        for name in ["e37_derived.tsv", "final_all_experiments.tsv", *report.OPTIONAL_RESULTS.values()]:
            result = self.cli("--input", self.directory, "--output", self.directory / name)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite", result.stderr)

    def test_watch_regenerates_after_new_experiment_file(self):
        self.raw.to_csv(self.directory / "final_all_experiments.tsv", sep="\t", index=False)
        output = self.directory / "watched.html"
        with (self.directory / "watch.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, str(SCRIPT), "--input", str(self.directory),
                                     "--output", str(output), "--watch", "--watch-interval", "0.1"],
                                    stdout=log, stderr=log)
            try:
                def wait_for_count(count, has_phases=False):
                    deadline = time.monotonic() + 15
                    while time.monotonic() < deadline:
                        self.assertIsNone(proc.poll(), "Watch process exited")
                        if output.exists():
                            text = output.read_text()
                            meta = json.loads(re.search(r'id="report-provenance">(.*?)</script>', text, re.S).group(1))
                            if len(meta["selected_experiments"]) == count and ('data-view="phases"' in text) == has_phases:
                                return
                        time.sleep(0.05)
                    self.fail("Watch did not refresh to {} experiments".format(count))
                wait_for_count(36)
                new = self.raw[self.raw.exp_id == "E09"].copy()
                new["exp_id"] = "E37"
                staging = self.directory / "staging.tsv"
                new.to_csv(staging, sep="\t", index=False)
                staging.rename(self.directory / "e37_new.tsv")
                wait_for_count(37)
                (self.directory / "e37_new.tsv").unlink()
                wait_for_count(36)
                self.phase_fixture().to_csv(self.directory / "phase_results.csv", index=False)
                wait_for_count(36, has_phases=True)
                (self.directory / "phase_results.csv").unlink()
                wait_for_count(36)
            finally:
                proc.terminate()
                proc.wait(timeout=5)

    def test_collect_refreshes_html_and_dry_run_does_not(self):
        import profiling_add_experiment_test as extractor
        _, rows = extractor.committed_rows("E09")
        root = self.directory / "runner"
        runner = root / "tools/exp"
        runner.mkdir(parents=True)
        shutil.copy(ROOT / "tools/exp/exp.sh", runner / "exp.sh")
        shutil.copy(ROOT / "tools/exp/experiments.tsv", runner / "experiments.tsv")
        conf = ("EXP_ROOT='{}'\nGEM5_BIN=/unused\nGEM5_CWD=/unused\nKERNEL=/unused\nDISK=/unused\nGEN_PYTHON='{}'\n").format(self.directory / "runs", sys.executable)
        (runner / "runner.conf").write_text(conf)
        out = self.directory / "runs/37/out_20260926_120000"
        out.mkdir(parents=True)
        extractor.write_stats(out / "stats.txt", extractor.interval_deltas(rows), ["0.5"] * 6)
        (out / "gem5_profile_regions.tsv").write_text("complete\n")
        results = self.directory / "results"
        output = self.directory / "collected.html"
        env = dict(os.environ, REPO_ROOT=str(ROOT), REPORT_PYTHON=sys.executable, REPORT_OUTPUT=str(output))
        command = ["bash", str(runner / "exp.sh"), "collect", "37", "--output-root", str(results)]
        dry = subprocess.run(command + ["--dry-run"], env=env, capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(output.exists())
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Experiments: 1", result.stdout)
        self.assertTrue(output.is_file())

    def test_python39_syntax(self):
        ast.parse(SCRIPT.read_text(), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
