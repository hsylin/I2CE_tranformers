"""Exercise the future controller with fake children and synthetic evidence."""

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools/exp/lib"))
import finite_study as controller
import study_execution as execution
from study_records import save, digest, replace_status


class StudyExecutionTests(unittest.TestCase):
    def rows(self, root):
        return [
            dict(
                id="E%02d" % i,
                state=str(root / "state" / ("E%02d" % i)),
                output=str(root / "output" / ("E%02d" % i)),
            )
            for i in range(1, 14)
        ]

    def exercise_controller(self, failure, count=13):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evidence").mkdir()
            rows = self.rows(root)[:count]
            registration = dict(
                maximum_formal_runs=len(rows),
                automatic_retries=0,
                rows=rows,
                runtime_pins={},
                checkpoint_pins={},
                stage_evidence_pins={},
                scratch=str(root),
                admission_lock=str(root / "lock"),
            )
            save(root / "registration.json", registration)
            launches = []

            def prepare(row, _):
                Path(row["state"]).mkdir(parents=True)
                (Path(row["output"]) / "formal").mkdir(parents=True)

            def launch(root, registration, row, decision):
                controller.reserve(root, row, registration)
                launches.append(row["id"])
                (Path(row["output"]) / "formal/system.terminal").write_text(
                    "\n".join(execution.entry_markers("")) + "\n"
                )
                return dict(
                    child=mock.Mock(poll=lambda: 1 if row["id"] == failure else 0),
                    row=row,
                    identity={"pid": 1, "start": "1"},
                    started=0,
                )

            with (
                mock.patch.object(
                    controller, "verify_contract", return_value=registration
                ),
                mock.patch.object(controller, "verify_files"),
                mock.patch.object(
                    controller, "process", return_value={"pid": 1, "start": "1"}
                ),
                mock.patch.object(controller.os, "nice"),
                mock.patch.object(controller.time, "sleep"),
                mock.patch.object(execution, "prepare", side_effect=prepare),
                mock.patch.object(controller, "configuration"),
                mock.patch.object(execution, "collect"),
                mock.patch.object(controller, "launch", side_effect=launch),
                mock.patch.object(
                    controller, "sample_admission", return_value={"admitted": True}
                ),
            ):
                controller.controller(root, "fixture")
                with self.assertRaises(FileExistsError):
                    controller.controller(root, "fixture")
            result = json.loads(
                (root / "evidence/controller-complete.json").read_text()
            )
            self.assertEqual(
                launches, ["E%02d" % i for i in range(1, 4 if failure else count + 1)]
            )
            self.assertEqual(result["passed"], failure is None)
            self.assertTrue((root / "evidence/canary-entry.json").exists())
            before = {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                controller.status(root)
                controller.status(root)
            self.assertEqual(
                before, {str(p): digest(p) for p in root.rglob("*") if p.is_file()}
            )

    def test_complete_registration_with_three_slots(self):
        self.exercise_controller(None, count=3)

    def test_complete_thirteen_once(self):
        self.exercise_controller(None)

    def test_failure_stops_future_admissions(self):
        self.exercise_controller("E03")

    def test_validation_only_diagnostics_are_not_production_requirements(self):
        text = "[reconstruction-ci-validation] logical_indices=16384 passed=1\n[compressed-indices] setup_ns=123 clock=simulator_rpns\n[bounded-runtime-W] calls=15\n"
        self.assertEqual(
            execution.normalized_diagnostics(text),
            [
                "[compressed-indices] setup_ns=POSITIVE clock=simulator_rpns",
                "[bounded-runtime-W] calls=15",
            ],
        )

    def test_reservation_is_not_guest_entry(self):
        markers = execution.entry_markers(
            "[guest] E01 once-only formal\ntransformer.o: OK\n"
        )
        self.assertFalse(all(markers.values()))
        self.assertTrue(markers["transformer.o: OK"])

    def test_budget_duplicate_and_unregistered_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = self.rows(root)[:3]
            registration = dict(rows=rows, maximum_formal_runs=3, automatic_retries=0)
            for row in rows:
                Path(row["state"]).mkdir(parents=True)
            controller.reserve(root, rows[0], registration)
            with self.assertRaises(FileExistsError):
                controller.reserve(root, rows[0], registration)
            with self.assertRaises(ValueError):
                controller.reserve(root, dict(id="E99"), registration)
            for row in rows[1:]:
                controller.reserve(root, row, registration)
            with self.assertRaises(RuntimeError):
                controller.reserve(root, rows[0], registration)
            self.assertEqual(len(list(root.glob("state/*/formal-reservation.json"))), 3)

    def test_failed_launch_consumes_slot_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = self.rows(root)[0]
            state = Path(row["state"])
            state.mkdir(parents=True)
            Path(row["output"]).mkdir(parents=True)
            save(state / "prelaunch-gate.json", dict(passed=True, pins={}))
            registration = dict(
                rows=[row],
                maximum_formal_runs=1,
                automatic_retries=0,
                runtime_pins={},
                checkpoint_pins={},
                gem5_root=str(root),
                gem5_resources=str(root),
            )
            with (
                mock.patch.object(execution, "command", return_value=["fake"]),
                mock.patch.object(
                    controller.subprocess,
                    "Popen",
                    side_effect=OSError("synthetic launch failure"),
                ) as child,
            ):
                with self.assertRaises(OSError):
                    controller.launch(root, registration, row, {})
                with self.assertRaises(RuntimeError):
                    controller.launch(root, registration, row, {})
                self.assertEqual(child.call_count, 1)
            self.assertTrue((state / "formal-reservation.json").exists())

    def test_environment_and_bounded_status(self):
        with mock.patch.dict(os.environ, {"PYTHONOPTIMIZE": "2"}):
            self.assertNotIn(
                "PYTHONOPTIMIZE", execution.environment({"gem5_resources": "/fixture"})
            )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(100):
                replace_status(root / "status-current.json", {"iteration": i})
            self.assertEqual(len(list(root.iterdir())), 1)
            self.assertEqual(
                json.loads((root / "status-current.json").read_text())["iteration"], 99
            )
            with self.assertRaises(ValueError):
                replace_status(root / "completion.json", {})
        for path in (REPO / "tools/exp/lib").glob("study_*.py"):
            self.assertFalse(
                any(
                    isinstance(n, ast.Assert)
                    for n in ast.walk(ast.parse(path.read_text()))
                ),
                str(path),
            )


if __name__ == "__main__":
    unittest.main()
