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


class RooflineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "evidence"
        shutil.copytree(roofline.DEFAULT_INPUT.parent, self.root)
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
        self.save()

    def reject(self, reason):
        self.save()
        result = roofline.load(self.path)
        self.assertEqual(result["observations"], [])
        self.assertIn(reason, result["skipped"][-1]["reason"])

    def test_preserved_e13_quantities(self):
        result = roofline.load(self.path)
        self.assertEqual(len(result["skipped"]), 12)
        o = result["observations"][0]
        self.assertEqual(o["operations"], 2147483648)
        self.assertEqual(o["seconds"], 0.0720863665)
        self.assertAlmostEqual(o["performance"] / 1e9, 29.79042712605)
        self.assertEqual(o["opclass_to_useful_ratio"], 1)
        self.assertEqual(o["traffic"]["Physical DRAM"]["total"], 11789760)
        self.assertEqual(o["traffic"]["Membus/controller"]["total"], 11804736)
        self.assertEqual(o["theory"]["compute_ops_per_second"], 128e9)
        self.assertEqual(len(o["points"]), 6)
        points = {(p["level"], p["direction"]): p for p in o["points"]}
        self.assertEqual(points["Core–L1D", "total"]["bytes_per_second"], 64e9)
        self.assertEqual(points["L1D–L2", "read"]["bytes_per_second"], 128e9)
        self.assertAlmostEqual(points["L1D–L2", "read"]["ai"], 11.4061215641)
        self.assertAlmostEqual(points["L1D–L2", "write"]["ai"], 11.4669685675)
        self.assertNotIn(("L1D–L2", "total"), points)
        self.assertEqual(points["Physical DRAM", "total"]["bytes_per_second"], 12.8e9)
        self.assertEqual(
            points["Membus/controller", "read"]["bytes_per_second"], 12.8e9
        )
        self.assertTrue(all(p["projection"] == 128e9 for p in o["points"]))

    def test_figure_preserves_directional_coordinates_and_integer_units(self):
        o = roofline.load(self.path)["observations"][0]
        fig = roofline.figure(o)
        points = [t for t in fig["data"] if t.get("name", "").endswith(" observed")]
        self.assertEqual(len(points), 4)
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

    def test_frequency_comes_from_stats_and_bus_domains_are_separate(self):
        c = json.loads((self.root / "config.json").read_text())
        a, b = roofline.theory(c, 1e12), roofline.theory(c, 2e12)
        self.assertEqual(b["compute_ops_per_second"], 2 * a["compute_ops_per_second"])
        c["system"]["clk_domain"]["clock"][0] *= 2
        changed = roofline.theory(c, 1e12)
        old = {(v["level"], v["direction"]): v["bytes_per_second"] for v in a["memory"]}
        new = {
            (v["level"], v["direction"]): v["bytes_per_second"]
            for v in changed["memory"]
        }
        self.assertEqual(new["L1D–L2", "read"], old["L1D–L2", "read"])
        self.assertEqual(
            new["Membus/controller", "read"], old["Membus/controller", "read"] / 2
        )

    def test_mixed_supported_and_unsupported_experiments(self):
        second = copy.deepcopy(self.run)
        second["id"] = "E14-test-only"
        second["operation_model"]["instruction_model"] = "unknown"
        self.manifest["experiments"].append(second)
        self.save()
        result = roofline.load(self.path)
        self.assertEqual([o["id"] for o in result["observations"]], ["E13"])
        self.assertIn("Unsupported operation", result["skipped"][-1]["reason"])

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
        self.assertIn("Hierarchical Roofline Model for E13", page)
        main = re.search(r"<main>(.*?)</main>", page, re.S).group(1)
        self.assertNotIn("Evidence manifest:", main)
        self.assertNotIn("Solid points are measured", main)
        self.assertNotIn("Experiments without a supported", main)
        self.assertNotIn("<table", main)
        self.assertIn("No matching native CommMonitor", page)
        before = (self.root / "stats.txt").read_bytes()
        command[-1] = str(self.root / "stats.txt")
        bad = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual((self.root / "stats.txt").read_bytes(), before)


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
        shutil.copytree(roofline.DEFAULT_INPUT.parent, folder)
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
            if f["layout"].get("meta", {}).get("method") == "theoretical_references"
        ]
        self.assertEqual(len(figures), 2)  # Never deduplicate distinct runs by E13.
        for figure in figures:
            actual = [
                t for t in figure["data"] if t.get("name", "").endswith(" observed")
            ]
            self.assertEqual(len(actual), 4)
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
