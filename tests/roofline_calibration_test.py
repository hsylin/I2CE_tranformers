"""Synthetic references: test provenance, gate failures and unchanged workloads."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transformer_profiling"))
import roofline
import roofline_calibration as rc
from roofline_fixture import bundle


def synthetic_record():
    record = json.loads(
        (ROOT / "tests/fixtures/roofline/legacy-cache-calibration.json").read_text()
    )
    hashes = {"runs/exp%d/stats.txt" % ex: str(ex) * 64 for ex in range(3)}

    def sample(rate, memory=False):
        out = dict(
            passes=[32, 64],
            rate_per_second=rate,
            samples=[
                dict(
                    passes=p,
                    experiment=ex,
                    sim_ticks=10**7 * p,
                    count=rate * p // 10**5,
                    empty_ticks=1000,
                    stats_sha256=hashes["runs/exp%d/stats.txt" % ex],
                )
                for p in (32, 64)
                for ex in range(3)
            ],
        )
        if memory:
            out.update(direction="total", source_type="measured")
        return out

    record.update(
        schema_version=2,
        id="synthetic-not-a-measurement",
        pattern="streaming-int8-sve128-sdot",
        quality=rc.QUALITY.copy(),
        source_hashes=hashes,
        compute=sample(125 * 10**9),
        references={
            name: sample(rate, True)
            for name, rate in [
                ("Core–L1D", 43 * 10**9),
                ("L1D–L2", 27 * 10**9),
                ("Physical DRAM", 2 * 10**9),
            ]
        },
    )
    return record


class ScanCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "evidence"
        bundle(self.folder)
        self.manifest = self.folder / "roofline.json"
        self.calibration = self.folder / "scan.json"
        self.record = synthetic_record()

    def write(self):
        self.calibration.write_text(json.dumps(self.record))

    def test_reference_override_preserves_observed_work_and_ai(self):
        old = roofline.load(self.manifest)["observations"][-1]
        self.write()
        new = roofline.load(self.manifest, calibration_override=self.calibration)[
            "observations"
        ][-1]
        for key in (
            "seconds",
            "operations",
            "performance",
            "traffic",
            "source_commit",
            "binary_sha256",
            "roi",
        ):
            self.assertEqual(new[key], old[key])
        self.assertFalse(new["reference_issues"])
        refs = new["references"]
        self.assertEqual(refs["compute_ops_per_second"], 125e9)
        self.assertEqual(refs["theoretical"]["compute_ops_per_second"], 128e9)
        self.assertEqual(refs["theoretical"]["memory"][0]["bytes_per_second"], 12.8e9)
        self.assertEqual(
            [p["bytes_per_second"] for p in new["points"]], [43e9, 27e9, 2e9]
        )
        self.assertEqual(
            {p["level"]: p["ai"] for p in new["points"]},
            {p["level"]: p["ai"] for p in old["points"]},
        )
        self.assertEqual(refs["calibration_sha256"], roofline.digest(self.calibration))
        self.assertNotIn(
            "Reference lines use the stable SDOT scan medians",
            roofline.summary_html(dict(observations=[new], skipped=[])),
        )

    def test_hardware_mismatch_and_bad_later_slope_apply_nothing(self):
        for change in ("hardware", "slope"):
            self.record = synthetic_record()
            if change == "hardware":
                self.record["hardware"]["system"]["cache_line_size"] = 128
            else:
                self.record["references"]["Physical DRAM"]["samples"][-1]["count"] *= 2
            self.write()
            result = roofline.load(
                self.manifest, calibration_override=self.calibration
            )["observations"][-1]
            self.assertTrue(result["reference_issues"])
            self.assertEqual(result["references"]["compute_ops_per_second"], 128e9)
            self.assertEqual(len(result["references"]["memory"]), 1)
            self.assertNotIn("calibration_sha256", result["references"])

    def test_active_limitations_preserve_recorded_wording_and_other_caveats(self):
        legacy = "Cache slopes are fixed independent write-pattern attainable measurements, not universal maxima."
        manifest = json.loads(self.manifest.read_text())
        for run in manifest["experiments"]:
            if "artifacts" in run:
                run["limitations"] = [legacy, "Unchanged workload caveat."]
        self.manifest.write_text(json.dumps(manifest))
        before = self.manifest.read_bytes()
        self.write()
        observed = roofline.load(self.manifest, calibration_override=self.calibration)[
            "observations"
        ][-1]
        self.assertEqual(
            observed["recorded_limitations"], [legacy, "Unchanged workload caveat."]
        )
        self.assertNotIn(legacy, observed["limitations"])
        self.assertIn("Unchanged workload caveat.", observed["limitations"])
        self.assertTrue(
            any(
                "streaming SDOT scan medians" in note
                for note in observed["limitations"]
            )
        )
        self.assertEqual(self.manifest.read_bytes(), before)
        legacy_view = roofline.load(self.manifest)["observations"][-1]
        self.assertEqual(
            legacy_view["limitations"], legacy_view["recorded_limitations"]
        )

    def test_unstable_or_unbound_references_rejected(self):
        changes = [
            lambda s: s["samples"].pop(),
            lambda s: s["samples"].__setitem__(0, copy.deepcopy(s["samples"][1])),
            lambda s: s["samples"][0].update(stats_sha256="f" * 64),
            lambda s: s["samples"][0].update(sim_ticks=0),
            lambda s: s["samples"][0].update(count=1.5),
            lambda s: s.update(rate_per_second=float("nan")),
            lambda s: [
                r.update(empty_ticks=r["sim_ticks"] // 10) for r in s["samples"]
            ],
            lambda s: [
                r.update(count=r["count"] * 2)
                for r in s["samples"]
                if r["passes"] == 32
            ],
            lambda s: s["samples"][-1].update(count=s["samples"][-1]["count"] * 2),
        ]
        for change in changes:
            record = synthetic_record()
            sample = record["compute"]
            change(sample)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                rc.stable_rate(
                    sample, 10**12, record["quality"], record["source_hashes"]
                )
        record = synthetic_record()
        record["quality"]["last_two_pass_delta_max"] = 0.1
        with self.assertRaises(ValueError):
            rc.stable_rate(
                record["compute"], 10**12, record["quality"], record["source_hashes"]
            )


if __name__ == "__main__":
    unittest.main()
