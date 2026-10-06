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
from unittest import mock

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "transformer_profiling/report.py"
# The historical dataset (previous student's E01-E36). Read only where a test
# explicitly names it; it is no longer the report's default input.
HISTORICAL = ROOT / "transformer_profiling/final"
SOURCE = HISTORICAL / "final_all_experiments.tsv"
# The current run directory, which report.py reads by default. It numbers from
# E01 as well, so a leak between the two is visible as a wrong E01.
CURRENT = ROOT / "transformer_profiling/hsylin"
sys.path.insert(0, str(SCRIPT.parent))
import report
import cache_comparison
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

    def cache_fixture(self):
        frame, _, _ = report.load_data(CURRENT)
        metadata = json.loads((CURRENT / 'cache_comparison.json').read_text())
        path = self.directory / 'cache_comparison.json'
        path.write_text(json.dumps(metadata))
        return frame, metadata, path

    def test_cache_comparison_keeps_single_runs_and_exact_pair_ratios(self):
        frame, _, path = self.cache_fixture()
        figures = cache_comparison.comparison_charts(path, frame, report.layout, report.switch_menu)
        self.assertEqual(len(figures), 4)
        for got, expected in zip(figures[0]['data'][0]['y'], [344.066, 310.451, 214.704, 315.61, 198.159]):
            self.assertAlmostEqual(got, expected)
        for got, expected in zip(figures[0]['data'][1]['y'], [259.626, 224.516, 122.916, 226.315, 122.932]):
            self.assertAlmostEqual(got, expected)
        # E07 -> E09: whole model improves, modified phases do not.
        self.assertAlmostEqual(figures[1]['data'][0]['y'][4], .214704 / .198159)
        self.assertLess(figures[1]['data'][1]['y'][4], 1)
        self.assertIn('MHA_QK', figures[2]['data'][0]['y'])
        self.assertEqual(figures[3]['data'][0]['cells']['values'][3][-1], '128 × 128 × 512')
        self.assertEqual(figures[3]['data'][0]['cells']['values'][6][-1], '409.5')
        # The variant path must not aggregate repeated observations as medians.
        with mock.patch.object(pd.core.groupby.generic.SeriesGroupBy, 'median', side_effect=AssertionError('median')):
            cache_comparison.comparison_charts(path, frame, report.layout, report.switch_menu)

    def test_cache_comparison_rejects_identity_configuration_and_accounting_mismatch(self):
        frame, _, path = self.cache_fixture()
        for field, value in [('repo_commit', '0' * 12), ('binary_sha256', '0' * 64),
                             ('gem5_timestamp', 'another-run'), ('l2', '512KiB'),
                             ('compile_flags', 'different'), ('phase_timing_valid', False)]:
            changed = frame.copy()
            changed.loc[changed.exp_id == 'E09', field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                cache_comparison.load_variants(path, changed)
        # Reusing historical E05 IDs cannot attach current geometry.
        with self.assertRaises(ValueError):
            cache_comparison.load_variants(path, report.load_data(HISTORICAL)[0])

    def test_cache_comparison_subset_and_missing_endpoints(self):
        frame, _, path = self.cache_fixture()
        subset = frame[frame.exp_id.isin(['E07', 'E09'])]
        figures = cache_comparison.comparison_charts(path, subset, report.layout, report.switch_menu)
        self.assertEqual(len(figures[0]['data'][0]['x']), 2)
        self.assertEqual(len(figures[1]['data'][0]['x']), 1)
        self.assertEqual(cache_comparison.comparison_charts(path, frame[frame.exp_id == 'E01'], report.layout, report.switch_menu), [])

    def test_cache_comparison_invalid_metadata_is_rejected(self):
        frame, metadata, path = self.cache_fixture()
        metadata['variants'].append(metadata['variants'][0])
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cache_comparison.load_variants(path, frame)
        metadata['variants'].pop()
        metadata['variants'][0]['l1_tile'] = [0, 32, 128]
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'tile'):
            cache_comparison.load_variants(path, frame)

    def test_cache_tab_offline_generation_and_metadata_fingerprint(self):
        output = self.directory / 'cache.html'
        result = self.cli('--output', output)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = output.read_text()
        self.assertIn('data-view="cache"', text)
        self.assertNotIn('data-view="tiles"', text)
        meta = json.loads(re.search(r'id="report-provenance">(.*?)</script>', text, re.S).group(1))
        self.assertEqual(meta['cache_comparison']['sha256'], hashlib.sha256((CURRENT / 'cache_comparison.json').read_bytes()).hexdigest())
        self.assertNotRegex(text, r'<script[^>]+src=')
        result = self.cli('--output', output, '--experiments', 'E01', 'E03')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('data-view="cache"', output.read_text())

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

    def test_one_command_from_other_directory_uses_current_dataset(self):
        """No --input: the default is hsylin/, never final/."""
        # The current publication aggregates E01-E09 from its study namespaces.
        result = self.cli("--output", self.directory / "all.html")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Experiments: 9", result.stdout)
        self.assertNotIn("Experiments: 36", result.stdout)
        self.assertTrue((self.directory / "all.html").exists())

        # Populated, it reports only what is in that directory.
        results = self.directory / "hsylin"
        results.mkdir()
        self.raw[self.raw.exp_id.isin(["E09", "E16"])].to_csv(
            results / "hsylin_all_experiments.tsv", sep="\t", index=False)
        output = self.directory / "mine.html"
        combined_hash = hashlib.sha256(
            (results / "hsylin_all_experiments.tsv").read_bytes()).hexdigest()
        result = self.cli("--input", results, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Experiments: 2", result.stdout)
        text = output.read_text()
        specs = json.loads(re.search(r'id="chart-specs">(.*?)</script>', text, re.S).group(1))
        self.assertTrue(specs)
        self.assertIn("plotly.js v", text)
        self.assertNotRegex(text, r'<script[^>]+src=')
        self.assertIn(combined_hash, text)
        self.assertNotIn(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), text)

    def test_default_input_is_current_dataset_and_excludes_historical(self):
        """Both datasets number from E01, so the default must read only hsylin/."""
        self.assertEqual(report.DEFAULT_INPUT, CURRENT)
        # The root combined table is a derived view, not a new result namespace.
        combined = CURRENT / (CURRENT.name + "_all_experiments.tsv")
        _, current, _ = report.load_data(CURRENT)
        self.assertEqual(set(current.exp_id), {f"E{i:02}" for i in range(1, 10)})
        self.assertTrue((current.n_learners == 2).all())
        self.assertTrue((current.codebook_size == 4).all())
        self.assertEqual(sorted(p.name for p in CURRENT.glob("e0*.tsv")), [])

        _, historical, _ = report.load_data(HISTORICAL)
        self.assertEqual(len(historical), 36)
        self.assertEqual(historical.exp_id.max(), "E36")
        # The historical E01 is the dense baseline, so a leak would be obvious.
        self.assertEqual(historical[historical.exp_id == "E01"].n_learners.iloc[0], 1)
        # E37-E46 are gone from the historical dataset.
        self.assertEqual([p.name for p in HISTORICAL.glob("e3[7-9]*.tsv")], [])
        self.assertEqual([p.name for p in HISTORICAL.glob("e4*.tsv")], [])

    def test_combined_table_resolved_by_directory_name(self):
        self.assertEqual(report.combined_table(CURRENT).name, "hsylin_all_experiments.tsv")
        self.assertEqual(report.combined_table(HISTORICAL).name, "final_all_experiments.tsv")
        base_mini = ROOT / "transformer_profiling/base_mini"
        if base_mini.is_dir():
            self.assertEqual(report.combined_table(base_mini).name,
                             "base_mini_all_experiments.tsv")
        # A directory whose name matches nothing falls back to its single table.
        self.raw.to_csv(self.directory / "whatever_all_experiments.tsv", sep="\t", index=False)
        self.assertEqual(report.combined_table(self.directory).name,
                         "whatever_all_experiments.tsv")
        # Two candidates are ambiguous between datasets and must be refused.
        self.raw.to_csv(self.directory / "other_all_experiments.tsv", sep="\t", index=False)
        with self.assertRaises(ValueError):
            report.combined_table(self.directory)

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
        self.raw.to_csv(self.directory / "final_all_experiments.tsv", sep="\t", index=False)
        result = self.cli("--input", self.directory, "--output", output)
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

    def test_collect_writes_tables_and_dry_run_does_not(self):
        import profiling_add_experiment_test as extractor
        _, rows = extractor.committed_rows("E09")
        root = self.directory / "runner"
        runner = root / "tools/exp"
        runner.mkdir(parents=True)
        shutil.copy(ROOT / "tools/exp/exp.sh", runner / "exp.sh")
        shutil.copy(ROOT / "tools/exp/experiments.tsv", runner / "experiments.tsv")
        # exp.sh resolves add_experiment.py relative to its own directory, not
        # $REPO_ROOT, so that `--at <sha>` cannot swap the collector out.
        (root / "transformer_profiling").mkdir()
        shutil.copy(ROOT / "transformer_profiling/add_experiment.py",
                    root / "transformer_profiling/add_experiment.py")
        conf = ("EXP_ROOT='{}'\nGEM5_BIN=/unused\nGEM5_CWD=/unused\nKERNEL=/unused\nDISK=/unused\nGEN_PYTHON='{}'\n").format(self.directory / "runs", sys.executable)
        (runner / "runner.conf").write_text(conf)
        out = self.directory / "runs/37/out_20260926_120000"
        out.mkdir(parents=True)
        extractor.write_stats(out / "stats.txt", extractor.interval_deltas(rows), ["0.5"] * 6)
        # exp.sh passes this to the collector, which reads the schema from it.
        (out / "gem5_profile_regions.tsv").write_text(
            "scope\tgroup4_full_interleaved_transformer_block\n"
            "dump_index\tcheckpoint\tinterval_since_previous\n"
            "1\tafter_mha\tMHA\n"
            "2\tafter_projection\tProjection\n"
            "3\tafter_attn_addnorm\tnon_GEMM_after_projection\n"
            "4\tafter_ff1\tFF1\n"
            "5\tafter_ff2\tFF2\n"
            "6\tfinal_total\tnon_GEMM_after_ff2\n")
        results = self.directory / "results"
        env = dict(os.environ, REPO_ROOT=str(ROOT), GEN_PYTHON=sys.executable)
        command = ["bash", str(runner / "exp.sh"), "collect", "37", "--output-root", str(results)]
        dry = subprocess.run(command + ["--dry-run"], env=env, capture_output=True, text=True)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse(results.exists())
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        # An explicit --output-root numbers into that directory, so E01.
        self.assertEqual([p.name for p in sorted(results.glob("e*.tsv"))
                          if not p.name.endswith("_all_experiments.tsv")],
                         ["e01_runner_n4_cb8_sve128_run_20260926_120000.tsv"])
        self.assertTrue((results / "results_all_experiments.tsv").is_file())
        # Reporting is a separate step and reads that directory when told to.
        html = self.directory / "collected.html"
        shown = self.cli("--input", results, "--output", html)
        self.assertEqual(shown.returncode, 0, shown.stderr)
        self.assertIn("Experiments: 1", shown.stdout)
        self.assertTrue(html.is_file())

    def split_mha(self, frame):
        """Rewrite MHA rows into the five sub-regions, conserving every metric."""
        parts = ["MHA_QKV", "MHA_QK", "MHA_softmax", "MHA_SV", "MHA_out"]
        numeric = ["sim_seconds", "instructions", "ops", "cpu_cycles", "memory_references",
                   "load_instructions", "store_instructions", "dcache_demand_accesses",
                   "dcache_demand_misses", "icache_demand_accesses", "icache_demand_misses",
                   "l2_demand_accesses", "l2_demand_misses", "committed_branches",
                   "branch_mispredictions"]
        numeric = [c for c in numeric if c in frame.columns]
        keep = frame[frame.interval != "MHA"].copy()
        rows = []
        for _, mha in frame[frame.interval == "MHA"].iterrows():
            # First four parts take a fifth each; the last takes the remainder so
            # the five add back to exactly the original MHA row.
            running = {c: 0 for c in numeric}
            for index, name in enumerate(parts):
                row = mha.copy()
                row["interval"] = name
                row["checkpoint"] = "after_" + name.lower()
                for c in numeric:
                    if index < len(parts) - 1:
                        share = mha[c] / len(parts)
                        if c != "sim_seconds":
                            share = int(share)
                        row[c] = share
                        running[c] += share
                    else:
                        row[c] = mha[c] - running[c]
                rows.append(row)
        return pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)

    def test_split_mha_schema_is_accepted_and_charted(self):
        """The five sub-regions replace MHA and appear in block execution order."""
        one = self.raw[self.raw.exp_id == "E09"].copy()
        data = self.split_mha(one)
        frame, totals, stages = self.load(data)
        self.assertEqual(sorted(set(stages.interval)), sorted(report.SCHEMA_V2))
        self.assertEqual(len(stages), 10)
        # The schema decides the chart's region list and its order.
        self.assertEqual(report.STAGES,
                         ["MHA_QKV", "MHA_QK", "MHA_softmax", "MHA_SV", "MHA_out",
                          "Projection", "non_GEMM_after_projection", "FF1", "FF2",
                          "non_GEMM_after_ff2"])
        self.assertEqual(len(report.SHORT), len(report.STAGES))
        self.assertEqual(len(report.COLORS), len(report.STAGES))
        self.assertEqual(len(set(report.COLORS)), len(report.COLORS))
        # Splitting conserves the total, so the timings stay valid.
        self.assertTrue(totals.phase_timing_valid.all())
        fig = report.phase_chart(totals, stages, "BERT-mini")
        names = [t["name"] for t in fig["data"]]
        self.assertEqual(names, report.SHORT)
        self.assertIn("MHA Q/K/V", names)
        self.assertNotIn("MHA", names)

    def test_report_renders_both_schemas_together(self):
        """A mixed report carries every region and invents no MHA mapping."""
        v1 = self.raw[self.raw.exp_id == "E09"].copy()
        v2 = self.raw[self.raw.exp_id == "E16"].copy()
        data = pd.concat([v1, self.split_mha(v2)], ignore_index=True)
        frame, totals, stages = self.load(data)
        self.assertEqual(report.STAGES,
                         ["MHA", "MHA_QKV", "MHA_QK", "MHA_softmax", "MHA_SV",
                          "MHA_out", "Projection", "non_GEMM_after_projection",
                          "FF1", "FF2", "non_GEMM_after_ff2"])
        # Each experiment keeps its own schema; nothing is back-filled.
        self.assertEqual(set(stages[stages.exp_id == "E09"].interval), report.SCHEMA_V1)
        self.assertEqual(set(stages[stages.exp_id == "E16"].interval), report.SCHEMA_V2)
        fig = report.phase_chart(totals, stages, "BERT-mini")
        mha = next(t for t in fig["data"] if t["name"] == "MHA")
        qkv = next(t for t in fig["data"] if t["name"] == "MHA Q/K/V")
        # Horizontal bars: the measured value is x, the experiment label is y.
        # The V2 experiment has no MHA cell and the V1 experiment no MHA_QKV cell,
        # and each gap sits opposite the other experiment's label.
        def gaps(trace):
            return {label for label, value in zip(trace["y"], trace["x"])
                    if value is None or (isinstance(value, float) and math.isnan(value))}
        self.assertEqual(len(gaps(mha)), 1)
        self.assertEqual(len(gaps(qkv)), 1)
        self.assertNotEqual(gaps(mha), gaps(qkv))
        # Nothing was invented. The MHA-family regions belong to one schema each,
        # so each carries exactly one value; the shared tail regions carry both.
        mha_labels = {report.REGION_SHORT[n] for n in report.REGION_ORDER
                      if n == "MHA" or n.startswith("MHA_")}
        for trace in fig["data"]:
            filled = [v for v in trace["x"]
                      if v is not None and not (isinstance(v, float) and math.isnan(v))]
            expected = 1 if trace["name"] in mha_labels else 2
            self.assertEqual(len(filled), expected, trace["name"])

    def test_unknown_region_set_is_refused(self):
        data = self.raw[self.raw.exp_id == "E09"].copy()
        data.loc[data.interval == "MHA", "interval"] = "MHA_QKV"
        with self.assertRaises(ValueError) as caught:
            self.load(data)
        self.assertIn("neither known schema", str(caught.exception))

    def test_python39_syntax(self):
        ast.parse(SCRIPT.read_text(), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
