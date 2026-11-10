"""Contract tests for the full scan and its scientific accounting gates."""

import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/roofline"))
from plan import emit, make_plan
from analyze import count, stats_blocks, summarize, RATES
from scan_results import write_results


class ScanContractTest(unittest.TestCase):
    def test_guest_header_executes_every_registered_pass(self):
        plan = make_plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            emit(root, plan)
            header = (
                (root / "src/plan.hh").read_text().split("static Block blocks[]={")[1]
            )
        actual = []
        for n, entry, first, pass_count, pass_list in re.findall(
            r"\{(\d+),(\d+),(\d+),(\d+),\{([\d,]+)\}\}", header
        ):
            kernel = plan["kernels"][int(entry)]
            for offset, passes in enumerate(
                list(map(int, pass_list.split(",")))[: int(pass_count)]
            ):
                actual.append(
                    (int(first) + offset, int(n), kernel["d"], kernel["u"], passes)
                )
        self.assertEqual(
            actual,
            [(c["id"], c["n"], c["d"], c["u"], c["passes"]) for c in plan["cases"]],
        )

    def test_exact_sizes_and_rowwise_coverage(self):
        p = make_plan()
        self.assertEqual(len(p["cases"]), 812)
        self.assertEqual(len(p["blocks"]), 137)
        bandwidth = [b for b in p["blocks"] if b["kind"] == "bandwidth"]
        self.assertEqual([b["n"] for b in bandwidth], p["source_worksets"]["tiles"])
        self.assertEqual(len({c["footprint"] for c in p["cases"]}), 92)
        self.assertEqual([c["id"] for c in p["cases"]], list(range(812)))
        self.assertEqual(
            len({(c["n"], c["d"], c["u"], c["passes"]) for c in p["cases"]}), 812
        )
        self.assertEqual(
            {(c["d"], c["u"], c["passes"]) for c in p["cases"] if c["n"] == 256},
            {
                (d, u, p)
                for d in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                for u in (1, 2, 4, 8)
                for p in (1, 2, 4, 8, 16, 32, 64)
            },
        )
        self.assertEqual(sum(c["kind"] == "bandwidth" for c in p["cases"]), 519)
        self.assertEqual(sum(c["kind"] == "crosscheck" for c in p["cases"]), 48)
        for b in bandwidth:
            expected = [
                p
                for p in (1, 2, 4, 8, 16, 32, 64)
                if p <= min(64, max(8, 262144 // b["n"]))
            ]
            self.assertEqual(b["passes"], expected)
            self.assertEqual((b["d"], b["u"]), (1, 8))

    def test_reprocessing_rejects_changed_order_and_block_identity(self):
        from reprocess import validate_plan

        p = make_plan()
        validate_plan(p, p)
        for field in ("passes", "d", "block", "id"):
            changed = copy.deepcopy(p)
            changed["cases"][0][field] += 1
            with self.assertRaises(ValueError):
                validate_plan(changed, p)
        changed = copy.deepcopy(p)
        changed["blocks"][0]["passes"] = list(reversed(changed["blocks"][0]["passes"]))
        with self.assertRaises(ValueError):
            validate_plan(changed, p)

    def test_exact_integer_and_explicit_zero_policy(self):
        self.assertEqual(
            count({"ticks": "9007199254740993"}, "ticks"), 9007199254740993
        )
        for bad in ("nan", "inf", "-1", "1.5"):
            with self.assertRaises(ValueError):
                count({"ticks": bad}, "ticks")
        with self.assertRaises(ValueError):
            count({}, "simTicks")
        with self.assertRaises(ValueError):
            count({}, "cache.demandHits", [], {})
        events = []
        self.assertEqual(
            count({"cache.present": "1"}, "cache.demandHits", events, dict(id=2)), 0
        )
        self.assertEqual(events, [dict(id=2, counter="cache.demandHits")])

    def test_truncated_duplicate_and_nested_stats_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats.txt"
            for body in (
                "Begin Simulation Statistics\nsimTicks 1\n",
                "Begin Simulation Statistics\nsimTicks 1\nsimTicks 2\nEnd Simulation Statistics\n",
                "Begin Simulation Statistics\nBegin Simulation Statistics\n",
            ):
                path.write_text(body)
                with self.assertRaises(ValueError):
                    list(stats_blocks(path))

    def fixture(self):
        p = make_plan()
        # Two independent blocks cover both eligible reference kinds.
        p["blocks"] = [
            dict(block=0, kind="bandwidth", passes=[32, 64], d=1, u=8, footprint=12288),
            dict(block=1, kind="compute", passes=[32, 64], d=256, u=8, footprint=12288),
        ]
        p["cases"] = [
            dict(b, id=i * 2 + j, passes=passes)
            for i, b in enumerate(p["blocks"])
            for j, passes in enumerate(b["passes"])
        ]
        rows = []
        for c in p["cases"]:
            for ex in range(3):
                rows.append(
                    dict(
                        c,
                        experiment=ex,
                        **{m: 100.0 for m in RATES},
                        sim_us=10,
                        l1d_miss_ratio=0,
                        l2_miss_ratio=0,
                        l1d_demand_misses=0,
                        l1d_demand_accesses=1,
                        l2_demand_misses=0,
                        l2_demand_accesses=1,
                        empty_overhead_fraction=0.001,
                    )
                )
        return p, rows

    def test_nonconvergence_and_decline_retained(self):
        p, rows = self.fixture()
        for r in rows:
            if r["id"] == 3:
                r["integer_GOPs"] = 90
        groups, largest, peaks, declines = summarize(p, rows)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(largest), 2)
        self.assertFalse(largest[-1]["integer_GOPs_stable"])
        self.assertEqual(peaks["integer_GOPs"]["best_stable"]["block"], 0)
        self.assertAlmostEqual(declines[-1]["relative_change"], -0.1)

    def test_repeat_spread_overhead_and_missing_repeat(self):
        p, rows = self.fixture()
        for r in rows:
            if r["id"] == 3 and r["experiment"] == 2:
                r["integer_GOPs"] = 106
        self.assertFalse(summarize(p, rows)[1][-1]["integer_GOPs_stable"])
        p, rows = self.fixture()
        for r in rows:
            if r["id"] == 3:
                r["empty_overhead_fraction"] = 0.01001
        self.assertFalse(summarize(p, rows)[1][-1]["integer_GOPs_stable"])
        with self.assertRaises(ValueError):
            summarize(p, rows[:-1])

    def results_fixture(self, directory):
        root = Path(directory)
        analysis = root / "analysis"
        analysis.mkdir()
        raw = root / "stats.txt"
        raw.write_text("recorded counter fixture\n")
        values = {
            "summary": {"status": "passed", "rows": 2436},
            "verification": {
                "status": "passed",
                "raw_hashes": {
                    "stats.txt": hashlib.sha256(raw.read_bytes()).hexdigest()
                },
            },
            "groups": [{"id": i} for i in range(812)],
            "largest-pass": [{"block": i} for i in range(137)],
        }
        for name, value in values.items():
            (analysis / (name + ".json")).write_text(json.dumps(value))
        for name in ("rows", "groups", "largest-pass"):
            (analysis / (name + ".csv")).write_text("id,value\n0,1\n")
        return root, values

    def test_results_export_preserves_values_and_csv_without_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root, values = self.results_fixture(directory)
            output = root / "results"
            write_results(root, output, plan={"cases": []})
            self.assertEqual(
                {p.name for p in output.iterdir()},
                {
                    "sdot-scan-results.json",
                    "sdot-scan-rows.csv",
                    "sdot-scan-groups.csv",
                    "sdot-scan-largest-pass.csv",
                },
            )
            result = json.loads((output / "sdot-scan-results.json").read_text())
            self.assertEqual(result["summary"], values["summary"])
            self.assertEqual(result["groups"], values["groups"])
            qa = json.loads((root / "automated-qa.json").read_text())
            for name, expected in qa["result_hashes"].items():
                self.assertEqual(
                    hashlib.sha256((output / name).read_bytes()).hexdigest(), expected
                )
            for name in ("rows", "groups", "largest-pass"):
                self.assertEqual(
                    (output / ("sdot-scan-" + name + ".csv")).read_bytes(),
                    (root / "analysis" / (name + ".csv")).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                write_results(root, output, plan={"cases": []})

    def test_results_export_rejects_changed_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.results_fixture(directory)
            (root / "stats.txt").write_text("changed fixture\n")
            with self.assertRaisesRegex(ValueError, "Raw evidence changed"):
                write_results(root, root / "results", plan={"cases": []})
            self.assertFalse((root / "automated-qa.json").exists())

    def test_results_export_rejects_failed_or_incomplete_analysis(self):
        for changed in (
            {"status": "failed", "rows": 2436},
            {"status": "passed", "rows": 2435},
        ):
            with (
                self.subTest(changed=changed),
                tempfile.TemporaryDirectory() as directory,
            ):
                root, _ = self.results_fixture(directory)
                (root / "analysis/summary.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    write_results(root, root / "results", plan={"cases": []})
                self.assertFalse((root / "automated-qa.json").exists())


if __name__ == "__main__":
    unittest.main()
