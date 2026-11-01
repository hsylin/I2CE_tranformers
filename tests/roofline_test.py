"""Reference arithmetic, native evidence validation and optional plotting."""

import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transformer_profiling"))
import roofline
import report
import experiment_names
from roofline_fixture import bundle


class RooflineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        bundle(self.root)
        self.path = self.root / "roofline.json"
        self.manifest = json.loads(self.path.read_text())
        self.run = self.manifest["experiments"][-1]

    def save(self):
        self.path.write_text(json.dumps(self.manifest))

    def edit_artifact(self, name, change):
        item = self.run["artifacts"][name]
        path = self.root / item["path"]
        path.write_text(change(path.read_text()))
        item["sha256"] = roofline.digest(path)
        if name != "receipt":
            receipt_path = self.root / "receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["artifact_sha256"][name] = item["sha256"]
            receipt_path.write_text(json.dumps(receipt))
            self.run["artifacts"]["receipt"]["sha256"] = roofline.digest(receipt_path)
        self.save()

    def reject(self, reason):
        self.save()
        result = roofline.load(self.path)
        self.assertEqual(result["observations"], [])
        self.assertIn(reason, result["skipped"][-1]["reason"])

    def test_rejects_duplicate_id_after_public_name_normalization(self):
        identity = experiment_names.SEQUENCE["experiments"][-1]
        self.run.update(
            id="E11",
            source_commit=identity["source_commit"],
            binary_sha256=identity["binary_sha256"],
        )
        second = copy.deepcopy(self.run)
        second["id"] = "E10"
        self.manifest["experiments"] = [self.run, second]
        self.save()
        with self.assertRaisesRegex(
            ValueError, "Duplicate published experiment ID: E10"
        ):
            roofline.load(self.path)

    def test_native_quantities_and_measured_slopes(self):
        result = roofline.load(self.path)
        self.assertEqual(len(result["skipped"]), 1)
        o = result["observations"][0]
        self.assertEqual(o["operations"], 2147483648)
        self.assertEqual(o["seconds"], 0.0720863665)
        self.assertAlmostEqual(o["performance"] / 1e9, 29.79042712605)
        self.assertEqual(o["opclass_to_useful_ratio"], 1)
        self.assertEqual(o["traffic"]["Physical DRAM"]["total"], 11789760)
        self.assertEqual(o["traffic"]["Membus/controller"]["total"], 11804736)
        self.assertEqual(o["references"]["compute_ops_per_second"], 128e9)
        self.assertEqual(len(o["points"]), 3)
        points = {(p["level"], p["direction"]): p for p in o["points"]}
        self.assertEqual(
            points["Core–L1D", "total"]["bytes_per_second"], 50840555117.87165
        )
        self.assertEqual(
            points["L1D–L2", "total"]["bytes_per_second"], 56887924389.25698
        )
        self.assertEqual(
            points["L1D–L2", "total"]["ai"], 2147483648 / (188274659 + 187275620)
        )
        self.assertEqual(points["Physical DRAM", "total"]["bytes_per_second"], 12.8e9)
        self.assertTrue(all(p["projection"] == 128e9 for p in o["points"]))

    def test_figure_preserves_directional_coordinates_and_integer_units(self):
        o = roofline.load(self.path)["observations"][0]
        fig = roofline.figure(o)
        points = [t for t in fig["data"] if t.get("name", "").endswith(" observed")]
        self.assertEqual(len(points), 3)
        self.assertEqual(
            [t["x"][0] for t in points],
            [p["ai"] for p in o["points"] if p["level"] != "Membus/controller"],
        )
        self.assertFalse(
            any("Membus/controller" in t.get("name", "") for t in fig["data"])
        )
        self.assertTrue(all(t["y"] == [o["performance"] / 1e9] for t in points))
        self.assertEqual(
            fig["layout"]["yaxis"]["title"]["text"], "Useful integer Gop/s"
        )

    def test_registered_stage_name_is_shared_by_both_scopes_without_changing_values(
        self,
    ):
        before = {scope: roofline.load(self.path, scope) for scope in roofline.SCOPES}
        sequence = dict(
            experiments=[
                dict(
                    id="E09",
                    source_commit=self.run["source_commit"],
                    binary_sha256=self.run["binary_sha256"],
                )
            ]
        )
        with patch.object(experiment_names, "SEQUENCE", sequence):
            for scope in roofline.SCOPES:
                result = roofline.load(self.path, scope)
                actual = result["observations"][0]
                original = before[scope]["observations"][0]
                self.assertEqual(actual["id"], "E09")
                self.assertTrue(actual["label"].startswith("E09"))
                self.assertEqual(
                    {k: v for k, v in actual.items() if k not in ("id", "label")},
                    {k: v for k, v in original.items() if k not in ("id", "label")},
                )
                self.assertTrue(
                    roofline.figure(actual)["layout"]["title"]["text"].startswith("E09")
                )
            mismatch = dict(self.run, binary_sha256="f" * 64)
            self.assertEqual(experiment_names.run_names(mismatch)[0], self.run["id"])

    def test_missing_calibration_omits_cache_references_without_fallback(self):
        del self.manifest["calibration"]
        self.save()
        o = roofline.load(self.path)["observations"][0]
        self.assertEqual([p["level"] for p in o["points"]], ["Physical DRAM"])
        self.assertIn("No compatible", o["reference_issues"][0])

    def test_nonfinite_calibration_cannot_create_a_trace(self):
        path = self.root / "calibration.json"
        record = json.loads(path.read_text())
        record["references"]["Core–L1D"]["bytes_per_second"] = float("nan")
        path.write_text(json.dumps(record))
        self.manifest["calibration"]["sha256"] = roofline.digest(path)
        self.save()
        o = roofline.load(self.path)["observations"][0]
        self.assertEqual([p["level"] for p in o["points"]], ["Physical DRAM"])
        self.assertIn("finite and positive", o["reference_issues"][0])

    def test_ini_mismatch_cannot_apply_cache_calibration(self):
        self.edit_artifact(
            "config_ini", lambda text: text.replace("size = 32768", "size = 65536")
        )
        o = roofline.load(self.path)["observations"][0]
        self.assertIn("INI hardware", o["reference_issues"][0])
        self.assertEqual([p["level"] for p in o["points"]], ["Physical DRAM"])

    def test_workload_metadata_cannot_be_silently_relabeled(self):
        self.run["operation_model"]["learners"] = 4
        self.reject("Measurement metadata differs")

    def test_export_has_no_theoretical_cache_fallback(self):
        result = roofline.load(self.path)
        chart = roofline.figure(result["observations"][0])
        text = json.dumps(chart) + roofline.summary_html(result)
        for obsolete in (
            "CPU issue limit assuming",
            "crossbar throughput limit",
            "128 GB/s",
            "64 GB/s",
            "No empirical calibration",
        ):
            self.assertNotIn(obsolete, text)
        for p in result["observations"][0]["points"][:2]:
            self.assertEqual(p["source_type"], "measured")
            self.assertEqual(p["direction"], "total")

    def test_mismatched_cache_capacity_omits_cache_references(self):
        def change(text):
            c = json.loads(text)
            c["system"]["cpu_cluster"][0]["cpus"][0]["dcache"]["size"] *= 2
            return json.dumps(c)

        self.edit_artifact("config", change)
        o = roofline.load(self.path)["observations"][0]
        self.assertEqual([p["level"] for p in o["points"]], ["Physical DRAM"])
        self.assertIn("mismatch", o["reference_issues"][0])

    def test_scalar_work_does_not_require_dot_opclass(self):
        self.edit_artifact(
            "stats",
            lambda s: "\n".join(
                line for line in s.splitlines() if roofline.DOT not in line
            ),
        )
        o = roofline.load(self.path)["observations"][0]
        self.assertEqual(o["operations"], 2147483648)
        self.assertIsNone(o["opclass_count"])

    def test_interval_work_and_counter_difference_once(self):
        op = dict(
            kind="gemm_shapes",
            begin_block=19,
            end_block=20,
            labels=["FF2"],
            shapes=[dict(m=512, n=256, k=1024, count=2)],
        )
        self.assertEqual(
            roofline.useful_work(op, 19, 20, self.run["roi"]["labels"]), 536870912
        )
        self.run["roi"].update(begin_block=19, end_block=20)
        self.run["operation_model"] = op
        self.rebind_measurement()
        self.save()
        o = roofline.load(self.path)["observations"][0]
        blocks = roofline.stats_blocks(self.root / "stats.txt")
        self.assertEqual(
            o["seconds"], (blocks[20][0]["simTicks"] - blocks[19][0]["simTicks"]) / 1e12
        )
        key = roofline.TRAFFIC["Core–L1D"][0]
        self.assertEqual(
            o["traffic"]["Core–L1D"]["read"], blocks[20][0][key] - blocks[19][0][key]
        )

    def test_codebook_sums_only_matching_intervals_and_work(self):
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        o = roofline.load(self.path, "codebook")["observations"][0]
        blocks = roofline.stats_blocks(self.root / "stats.txt")
        selected = [0, 4, 8, 12, 17, 19, 20]
        self.assertEqual(o["operations"], 1610612736)
        self.assertEqual([i["end_block"] for i in o["selected_intervals"]], selected)
        expected_ticks = sum(
            blocks[i][0]["simTicks"] - (blocks[i - 1][0]["simTicks"] if i else 0)
            for i in selected
        )
        self.assertEqual(o["seconds"], expected_ticks / 10**12)
        for level, keys in roofline.TRAFFIC.items():
            for direction, key in zip(("read", "write"), keys):
                expected = sum(
                    blocks[i][0][key] - (blocks[i - 1][0][key] if i else 0)
                    for i in selected
                )
                self.assertEqual(o["traffic"][level][direction], expected)
        for point in o["points"]:
            self.assertEqual(point["ai"], 1610612736 / point["bytes"])
            self.assertEqual(point["performance"], 1610612736 / o["seconds"])
        terms = o["quantity_sources"][0]["terms"]
        self.assertIsNone(terms[0]["begin_line"])
        self.assertEqual(
            terms[1]["begin_line"], blocks[3][1][roofline.TRAFFIC["Core–L1D"][0]]
        )
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_codebook_does_not_use_a_fraction_of_block_time_or_bytes(self):
        original = roofline.load(self.path, "codebook")["observations"][0]
        blocks = roofline.stats_blocks(self.root / "stats.txt")
        # Add traffic and time only to the first Attention QK interval.
        for index, (values, _) in enumerate(blocks):
            if index >= 1:
                values["simTicks"] += 10**12
                values["finalTick"] += 10**12
                values[roofline.TRAFFIC["Core–L1D"][0]] += 10**9
            for level, (r, w, scale) in roofline.RATES.items():
                for rate, counter in zip((r, w), roofline.TRAFFIC[level]):
                    values[rate] = (
                        values[counter]
                        / (values["simTicks"] / values["simFreq"])
                        / scale
                    )
        text = "\n".join(
            "---------- Begin Simulation Statistics ----------\n"
            + "\n".join(f"{k} {v}" for k, v in values.items())
            + "\n---------- End Simulation Statistics ----------"
            for values, _ in blocks
        )
        self.edit_artifact("stats", lambda _: text)
        derived = roofline.load(self.path, "codebook")["observations"][0]
        whole = roofline.load(self.path)["observations"][0]
        self.assertEqual(derived["seconds"], original["seconds"])
        self.assertEqual(derived["traffic"], original["traffic"])
        self.assertGreater(whole["seconds"], 1)

    def test_codebook_rejects_unrecognized_partition_without_hiding_block(self):
        self.run["roi"]["labels"][19] = "fused_ff1"
        self.edit_artifact("regions", lambda s: s.replace("\tFF1\n", "\tfused_ff1\n"))
        self.rebind_measurement()
        self.save()
        self.assertEqual(len(roofline.load(self.path)["observations"]), 1)
        result = roofline.load(self.path, "codebook")
        self.assertFalse(result["observations"])
        self.assertIn("separate QKV", result["skipped"][-1]["reason"])

    def test_codebook_keeps_cross_boundary_payload_accounting_distinct(self):
        before = roofline.load(self.path, "codebook")["observations"][0]
        blocks = roofline.stats_blocks(self.root / "stats.txt")
        old = blocks[7][0][roofline.MEMBUS_ROUTE]
        self.edit_artifact(
            "stats",
            lambda s: s.replace(
                f"{roofline.MEMBUS_ROUTE} {old}\n",
                f"{roofline.MEMBUS_ROUTE} {old + 64}\n",
            ),
        )
        after = roofline.load(self.path, "codebook")["observations"][0]
        self.assertEqual(after["traffic"], before["traffic"])
        self.assertEqual(after["membus_route_bytes"], before["membus_route_bytes"] - 64)
        self.assertEqual(len(roofline.load(self.path)["observations"]), 1)

    def test_whole_registered_payload_mismatch_remains_invalid(self):
        self.edit_artifact(
            "stats",
            lambda s: re.sub(
                re.escape(roofline.MEMBUS_ROUTE) + r" (\d+)",
                lambda m: roofline.MEMBUS_ROUTE + " " + str(int(m[1]) + 64),
                s,
            ),
        )
        for scope in ("block", "codebook"):
            result = roofline.load(self.path, scope)
            self.assertFalse(result["observations"])
            self.assertIn("Controller bytes differ", result["skipped"][-1]["reason"])

    def test_codebook_work_uses_registered_dimensions_not_instruction_count(self):
        self.run["operation_model"]["learners"] = 3
        self.run["operation_model"]["dimension"] = 128
        self.rebind_measurement()
        self.save()
        o = roofline.load(self.path, "codebook")["observations"][0]
        self.assertEqual(
            o["operations"], 2 * 3 * (4 * 512 * 128**2 + 2 * 512 * 128 * 1024)
        )

    def rebind_measurement(self):
        path = self.root / "measurement.json"
        data = json.loads(path.read_text())
        data.update(
            roi=self.run["roi"],
            operation_model=self.run["operation_model"],
            gem5_revision=self.run["gem5_revision"],
        )
        path.write_text(json.dumps(data))
        h = roofline.digest(path)
        self.run["measurement_sha256"] = h
        self.run["artifacts"]["measurement"]["sha256"] = h
        receipt = self.root / "receipt.json"
        r = json.loads(receipt.read_text())
        r["measurement_sha256"] = h
        receipt.write_text(json.dumps(r))
        self.run["artifacts"]["receipt"]["sha256"] = roofline.digest(receipt)

    def test_mixed_supported_and_unsupported_experiments(self):
        second = copy.deepcopy(self.run)
        second["id"] = "E14-test-only"
        second["operation_model"]["kind"] = "unknown"
        self.manifest["experiments"].append(second)
        self.save()
        result = roofline.load(self.path)
        self.assertEqual([o["id"] for o in result["observations"]], ["fixture"])
        self.assertIn("Measurement metadata differs", result["skipped"][-1]["reason"])

    def test_no_experiment_id_hardcoding(self):
        self.run["id"] = "trial-42"
        self.save()
        self.assertEqual(roofline.load(self.path)["observations"][0]["id"], "trial-42")

    def test_hash_mismatch_does_not_plot(self):
        with (self.root / "stats.txt").open("a") as out:
            out.write("\nchanged\n")
        self.reject("SHA256 mismatch")

    def test_missing_monitor_counter_is_not_estimated_from_misses(self):
        self.edit_artifact(
            "stats",
            lambda s: "\n".join(
                line
                for line in s.splitlines()
                if "system.roofline_l1.totalReadBytes" not in line
            ),
        )
        self.reject("Missing native counter")

    def test_native_bandwidth_mismatch(self):
        self.edit_artifact(
            "stats", lambda s: s.replace("6098487888.136240", "7098487888.136240")
        )
        self.reject("Native average bandwidth")

    def test_dram_peak_unit_or_config_mismatch(self):
        def change(s):
            c = json.loads(s)
            c["system"]["mem_ctrls"][0]["dram"]["tBURST"] *= 2
            return json.dumps(c)

        self.edit_artifact("config", change)
        self.reject("peakBW disagrees")

    def test_burst_mismatch(self):
        self.edit_artifact("stats", lambda s: s.replace("6716288", "6716289"))
        self.reject("DRAM burst byte mismatch")

    def test_roi_cannot_include_exit_cleanup(self):
        self.run["roi"]["end_block"] = 22
        self.rebind_measurement()
        self.reject("cleanup dump")

    def test_partial_file_is_rejected(self):
        self.edit_artifact(
            "stats", lambda s: s[: s.rfind("---------- End Simulation Statistics")]
        )
        self.reject("Incomplete")

    def test_nonfinite_counter(self):
        self.edit_artifact("stats", lambda s: s.replace("72086366500", "nan"))
        self.reject("Invalid counter")

    def test_unknown_model_revision(self):
        self.run["gem5_revision"] = "f" * 40
        self.reject("Unreviewed gem5 revision")

    def test_multicore_is_skipped(self):
        def change(s):
            c = json.loads(s)
            c["system"]["cpu_cluster"][0]["cpus"] *= 2
            return json.dumps(c)

        self.edit_artifact("config", change)
        self.reject("multicore")

    def test_wrong_monitor_boundary(self):
        self.edit_artifact(
            "config",
            lambda s: s.replace(
                '"peer": "system.cpu_cluster.cpus.dcache_port"',
                '"peer": "system.cpu_cluster.cpus.icache_port"',
            ),
        )
        self.reject("Unsupported monitor boundary")

    def test_unsuccessful_run(self):
        self.edit_artifact(
            "receipt",
            lambda s: s.replace('"guest_returncode": 0', '"guest_returncode": 1'),
        )
        self.reject("completion/configuration/guest identity")

    def test_wrong_binary_identity(self):
        self.run["binary_sha256"] = "b" * 64
        self.reject("identity differs")

    def test_wrong_region_labels(self):
        self.run["roi"]["labels"][-1] = "other"
        self.rebind_measurement()
        self.reject("marker sequence")

    def test_escape_and_artifact_traversal(self):
        self.run["label"] = "<script>alert(1)</script>"
        self.save()
        page = roofline.summary_html(roofline.load(self.path))
        self.assertNotIn("<script>", page)
        self.run["artifacts"]["stats"]["path"] = "../stats.txt"
        self.reject("relative")

    def test_duplicate_id(self):
        self.manifest["experiments"].append(copy.deepcopy(self.run))
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            roofline.load(self.path)

    def test_cli_protects_raw_inputs(self):
        stats = self.root / "stats.txt"
        before = stats.read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "transformer_profiling/roofline.py"),
                "--input",
                str(self.path),
                "--output",
                str(stats),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(stats.read_bytes(), before)

    def test_report_tab_is_optional_and_keeps_run_provenance(self):
        output = self.root.parent / "report.html"
        command = [
            sys.executable,
            "-B",
            str(ROOT / "transformer_profiling/report.py"),
            "--input",
            str(ROOT / "transformer_profiling/final"),
            "--roofline",
            str(self.path),
            "--output",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        page = output.read_text()
        self.assertIn('data-view="roofline"', page)
        self.assertIn(self.run["binary_sha256"], page)
        self.assertIn('data-view="roofline-codebook"', page)
        self.assertNotIn('data-view="curves"', page)
        self.assertIn("fixture \\u00b7 Codebook", page)
        main = re.search(r"<main>(.*?)</main>", page, re.S).group(1)
        self.assertNotIn("Evidence manifest:", main)
        self.assertNotIn("Solid points: measured", main)
        self.assertIn("Unavailable data", main)
        self.assertNotIn("<table", main)
        self.assertIn("No matching native CommMonitor", page)
        before = (self.root / "stats.txt").read_bytes()
        command[-1] = str(self.root / "stats.txt")
        bad = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual((self.root / "stats.txt").read_bytes(), before)

    def test_report_selection_matches_source_and_binary_in_both_scopes(self):
        report.load_dependencies()
        frame, totals, stages = report.load_data(ROOT / "transformer_profiling/final")
        observation = roofline.load(self.path)["observations"][0]
        other = copy.deepcopy(observation)
        other.update(
            id="different-run", label="excluded-other-run", source_commit="f" * 40
        )
        with patch.object(
            roofline,
            "load",
            return_value={"observations": [observation, other], "skipped": []},
        ):
            page = report.build_report(
                frame,
                totals,
                stages,
                "",
                "fixture",
                roofline_path=self.path,
                roofline_identities={
                    (observation["source_commit"], observation["binary_sha256"])
                },
            )
        self.assertNotIn("excluded-other-run", page)
        self.assertNotIn("different-run", page)
        self.assertIn('data-tab="roofline-codebook"', page)
        self.assertIn('data-tab="roofline"', page)
        self.assertEqual(
            page.count('"source_commit": "' + observation["source_commit"] + '"'), 2
        )


class ReportDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.results.mkdir()
        shutil.copyfile(
            ROOT / "transformer_profiling/final/final_all_experiments.tsv",
            self.results / "final_all_experiments.tsv",
        )

    def bundle(self, name):
        folder = self.results / name
        bundle(folder)
        return folder / "roofline.json"

    def cli(self, output):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "transformer_profiling/report.py"),
                "--input",
                str(self.results),
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )

    def test_no_fallback_to_other_datasets_or_raw_stats_guessing(self):
        (self.results / "stats.txt").write_text("not a registered run")
        self.assertEqual(report.roofline_manifests(self.results), [])

    def test_nested_discovery_and_authoritative_selection(self):
        a = self.bundle("runs/a")
        b = self.bundle("runs/b")
        self.assertEqual(report.roofline_manifests(self.results), [a, b])
        self.assertEqual(report.roofline_manifests(self.results, b), [b])
        root_manifest = self.results / "roofline.json"
        root_manifest.write_text("{}")
        self.assertEqual(report.roofline_manifests(self.results), [root_manifest])

    def test_no_automatic_external_symlink_or_hidden_bundle(self):
        self.bundle(".archive/run")
        external = self.root / "external.json"
        external.write_text("{}")
        (self.results / "run").mkdir()
        (self.results / "run/roofline.json").symlink_to(external)
        (self.results / "roofline.json").symlink_to(external)
        self.assertEqual(report.roofline_manifests(self.results), [])

    def test_same_html_automatic_bundles_keep_run_ids_separate(self):
        a = self.bundle("run-a")
        b = self.bundle("run-b")
        # A broken third bundle must not suppress the two valid observations.
        broken = self.results / "incomplete"
        broken.mkdir()
        (broken / "roofline.json").write_text("{incomplete")
        output = self.root / "report.html"
        result = self.cli(output)
        self.assertEqual(result.returncode, 0, result.stderr)
        page = output.read_text()
        self.assertIn('data-view="stages"', page)
        self.assertIn('data-view="roofline"', page)
        self.assertNotIn("Evidence manifest:", page)
        self.assertIn("incomplete", page)
        specs = json.loads(
            re.search(r'id="chart-specs">(.*?)</script>', page, re.S).group(1)
        )
        figures = [
            f
            for f in specs.values()
            if f["layout"].get("meta", {}).get("method") == "measured_cache_references"
        ]
        self.assertEqual(len(figures), 4)  # Two scopes for each distinct run.
        self.assertEqual(
            sorted(f["layout"]["meta"]["scope"] for f in figures),
            ["block", "block", "codebook", "codebook"],
        )
        for figure in figures:
            actual = [
                t for t in figure["data"] if t.get("name", "").endswith(" observed")
            ]
            self.assertEqual(len(actual), 3)
            if figure["layout"]["meta"]["scope"] == "block":
                self.assertAlmostEqual(actual[0]["y"][0], 29.79042712605)
        provenance = json.loads(
            re.search(r'id="report-provenance">(.*?)</script>', page, re.S).group(1)
        )
        self.assertEqual(
            {v["path"] for v in provenance["roofline_manifests"]},
            {str(p.resolve()) for p in (a, b, broken / "roofline.json")},
        )

    def test_discovered_raw_evidence_cannot_be_overwritten(self):
        manifest = self.bundle("run-a")
        stats = manifest.parent / "stats.txt"
        original = stats.read_bytes()
        result = self.cli(stats)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must not overwrite", result.stderr)
        self.assertEqual(stats.read_bytes(), original)

    def test_watch_detects_new_bundle_and_raw_counter_changes(self):
        args = SimpleNamespace(
            input=self.results,
            interpretation=None,
            marker_schemes=None,
            **{key: None for key in report.OPTIONAL_RESULTS},
        )
        before = report.watched_state(args)
        manifest = self.bundle("run-a")
        discovered = report.watched_state(args)
        self.assertNotEqual(before, discovered)
        with (manifest.parent / "stats.txt").open("a") as stream:
            stream.write("\nchanged counter fixture\n")
        self.assertNotEqual(discovered, report.watched_state(args))


class MonitorPatchTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "monitor_options", ROOT / "tools/exp/lib/apply_gem5_opts.py"
        )
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_patch_idempotence_and_disabled_default(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "starter_fs.py"
            shutil.copyfile(ROOT / "gem5/starter_fs.py", path)
            original = path.read_bytes()
            command = [
                sys.executable,
                "-B",
                str(ROOT / "tools/exp/lib/apply_gem5_opts.py"),
                str(path),
            ]
            subprocess.run(command, check=True, capture_output=True)
            first = path.read_bytes()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(path.read_bytes(), first)
            self.assertEqual(path.with_suffix(".py.orig").read_bytes(), original)
            self.assertEqual(first.decode().count('"--roofline-monitors"'), 1)
            import argparse

            parser = argparse.ArgumentParser()
            exec(textwrap.dedent(self.module.ROOFLINE_OPT), {"parser": parser})
            self.assertFalse(parser.parse_args([]).roofline_monitors)

    def test_native_monitor_splices_only_data_ports(self):
        from unittest.mock import Mock

        cpu = SimpleNamespace(
            dcache_port=Mock(), dcache=SimpleNamespace(mem_side=Mock())
        )
        system = SimpleNamespace(cpu_cluster=[SimpleNamespace(cpus=[cpu])])
        factory = Mock(
            side_effect=lambda **kw: SimpleNamespace(
                cpu_side_port=object(), mem_side_port=object()
            )
        )
        env = dict(
            system=system,
            args=SimpleNamespace(roofline_monitors=True),
            want_caches=True,
        )
        with patch.dict(
            sys.modules,
            {
                "m5": SimpleNamespace(),
                "m5.objects": SimpleNamespace(CommMonitor=factory),
            },
        ):
            exec(textwrap.dedent(self.module.ROOFLINE_BLOCK), env)
        cpu.dcache_port.splice.assert_called_once_with(
            system.roofline_l1.cpu_side_port, system.roofline_l1.mem_side_port
        )
        cpu.dcache.mem_side.splice.assert_called_once_with(
            system.roofline_l2.cpu_side_port, system.roofline_l2.mem_side_port
        )
        self.assertEqual(factory.call_count, 2)
        self.assertFalse(factory.call_args.kwargs["disable_bandwidth_hists"])

    def test_unsupported_hardware_is_rejected_before_monitor_creation(self):
        env = dict(
            system=SimpleNamespace(cpu_cluster=[]),
            args=SimpleNamespace(roofline_monitors=True),
            want_caches=False,
        )
        with patch.dict(
            sys.modules,
            {"m5": SimpleNamespace(), "m5.objects": SimpleNamespace(CommMonitor=None)},
        ):
            with self.assertRaisesRegex(ValueError, "one timing CPU"):
                exec(textwrap.dedent(self.module.ROOFLINE_BLOCK), env)


if __name__ == "__main__":
    unittest.main()
