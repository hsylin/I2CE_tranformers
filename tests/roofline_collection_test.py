"""Check measured-run export and acquisition without running gem5."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "transformer_profiling"), str(ROOT / "tools/exp/lib")]
import roofline
import study_execution
import study_roofline
from roofline_fixture import bundle


class CollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "helper-contract.json").write_text("{}")
        manifest = json.loads(bundle(self.root / "fixture").read_text())
        run = manifest["experiments"][-1]
        self.state = self.root / "state"
        self.state.mkdir()
        self.formal = self.root / "output/formal"
        self.formal.mkdir(parents=True)
        (self.root / "evidence/E01").mkdir(parents=True)
        for key, name in dict(
            stats="stats.txt",
            config="config.json",
            config_ini="config.ini",
            regions="gem5_profile_regions.tsv",
        ).items():
            shutil.copyfile(
                self.root / "fixture" / run["artifacts"][key]["path"],
                self.formal / name,
            )
        (self.formal / "config-gate.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    config_json_sha256=roofline.digest(self.formal / "config.json"),
                    config_ini_sha256=roofline.digest(self.formal / "config.ini"),
                )
            )
        )
        (self.formal / "system.terminal").write_text(
            "transformer.o: OK\n[guest] E01 exit code: 0\n[guest] E01 done\n"
        )
        (self.state / "formal-process.log").write_text(
            "m5_exit instruction encountered (0)"
        )
        (self.state / "formal-exit.json").write_text('{"returncode":0}')
        (self.root / "binary").write_bytes(
            b"synthetic executable identity; not runnable"
        )
        binary_hash = roofline.digest(self.root / "binary")
        (self.root / "build.json").write_text(
            json.dumps(
                dict(
                    source_commit=run["source_commit"],
                    binary_sha256=binary_hash,
                    commands=[["synthetic-compile"]],
                    compiler_identity="test fixture",
                )
            )
        )
        self.row = dict(
            id="E01",
            state=str(self.state),
            output=str(self.root / "output"),
            inputs=str(self.root / "input"),
            binary=str(self.root / "binary"),
            binary_sha256=binary_hash,
            build_record=dict(
                path=str(self.root / "build.json"),
                sha256=roofline.digest(self.root / "build.json"),
            ),
            roofline={
                k: run[k]
                for k in (
                    "source_commit",
                    "status",
                    "label",
                    "roi",
                    "operation_model",
                    "limitations",
                )
            },
        )
        self.registration = dict(
            runtime_pins={},
            checkpoint_pins={},
            scientific_inputs={},
            simulator="/fixture/gem5",
            kernel="/fixture/kernel",
            disk="/fixture/disk",
            checkpoint="/fixture/checkpoint",
            roofline=dict(
                study_id="synthetic-study",
                gem5_revision=roofline.MODEL_REVISION,
                calibration=dict(
                    path=str(self.root / "fixture/calibration.json"),
                    sha256=roofline.digest(self.root / "fixture/calibration.json"),
                ),
            ),
        )

    def test_export_binds_evidence_and_refuses_replacement(self):
        study_roofline.collect(self.root, self.registration, self.row)
        out = self.root / "evidence/E01/roofline"
        result = roofline.load(out / "roofline.json")
        self.assertEqual(result["skipped"], [])
        self.assertEqual(result["observations"][0]["reference_issues"], [])
        self.assertEqual(result["observations"][0]["id"], "synthetic-study/E01")
        self.assertEqual(len(result["observations"][0]["points"]), 3)
        with self.assertRaises(FileExistsError):
            study_roofline.collect(self.root, self.registration, self.row)

    def test_monitors_only_when_explicitly_registered(self):
        self.assertIn(
            "--roofline-monitors",
            study_execution.command(self.root, self.registration, self.row),
        )
        del self.registration["roofline"]
        self.assertNotIn(
            "--roofline-monitors",
            study_execution.command(self.root, self.registration, self.row),
        )

    def test_reject_changed_binary_before_copy(self):
        (self.root / "binary").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Binary identity"):
            study_roofline.collect(self.root, self.registration, self.row)
        self.assertFalse((self.root / "evidence/E01/roofline").exists())

    def test_reject_changed_configuration_before_copy(self):
        (self.formal / "config.ini").write_text("changed")
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            study_roofline.collect(self.root, self.registration, self.row)
