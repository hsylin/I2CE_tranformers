#!/usr/bin/env python3
"""Exercise validation failures without building or running the model."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "tools/exp/build_stage.py"
SPEC = importlib.util.spec_from_file_location("build_stage", MODULE)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


class ValidationRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.commands = validation.Commands(self.root, os.environ.copy())

    def test_nonzero_exit_preserves_diagnostic_and_status(self):
        command = [sys.executable, "-c", "print('invalid input'); raise SystemExit(7)"]
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.commands.run("reject", command, self.root)
        self.assertEqual(self.commands.records[0]["returncode"], 7)
        self.assertEqual((self.root / "logs/reject.log").read_text(), "invalid input\n")

    def test_timeout_preserves_attempt_identity(self):
        command = [sys.executable, "-c", "import time; time.sleep(10)"]
        with self.assertRaises(subprocess.TimeoutExpired):
            self.commands.run("timeout", command, self.root, timeout=0.1)
        record = self.commands.records[0]
        self.assertEqual(record["command"], command)
        self.assertEqual(record["timed_out_after_seconds"], 0.1)
        self.assertIn("finished_at", record)

    def test_expected_rejection_does_not_accept_success(self):
        self.commands.run(
            "expected-rejection",
            [sys.executable, "-c", "raise SystemExit(1)"],
            self.root,
            expected_returncode=1,
        )
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.commands.run(
                "unexpected-success",
                [sys.executable, "-c", "pass"],
                self.root,
                expected_returncode=1,
            )

    def test_missing_program_preserves_launch_failure(self):
        with self.assertRaises(FileNotFoundError):
            self.commands.run("missing", [self.root / "absent-program"], self.root)
        self.assertIn("launch_error", self.commands.records[0])

    def test_ambiguous_instrumentation_does_not_change_source(self):
        source = self.root / "fixture.cc"
        source.write_text("anchor anchor\n")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            validation.replace_once(source, "anchor", "replacement")
        self.assertEqual(source.read_text(), "anchor anchor\n")

    def test_receipt_cannot_overwrite_existing_evidence(self):
        receipt = self.root / "receipt.json"
        validation.save(receipt, {"first": True})
        original = receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            validation.save(receipt, {"first": False})
        self.assertEqual(receipt.read_bytes(), original)

    def test_gate_remains_active_with_python_optimization(self):
        program = (
            "import runpy; "
            f"module = runpy.run_path({str(MODULE)!r}); "
            "module['require'](False, 'gate remains active')"
        )
        result = subprocess.run(
            [sys.executable, "-O", "-c", program], capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gate remains active", result.stderr)

    def contract(self):
        return {
            "helper_sha256": validation.digest(MODULE),
            "experiment_id": "E01",
            "formal_measurements": 0,
            "automatic_retries": 0,
            "maximum_production_builds": 1,
            "maximum_validation_model_builds": 2,
            "maximum_full_model_executions": 2,
            "rejected_sve_lengths_bytes": [32, 64],
            "maximum_startup_rejection_checks": 2,
            "maximum_oracle_builds": 0,
            "maximum_oracle_executions": 0,
            "oracles": [],
            "native_checks": [],
            "source_files": {"transformer.cpp": "fixture"},
            "scientific_inputs": {},
            "test_files": {},
            "additional_inputs": {},
        }

    def test_contract_rejects_measurements_retries_and_unbound_tests(self):
        validation.check_contract(self.contract())
        for field in ("formal_measurements", "automatic_retries"):
            with self.subTest(field=field):
                contract = self.contract()
                contract[field] = 1
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    validation.check_contract(contract)
        contract = self.contract()
        contract["native_checks"] = [{"source": "missing.cc"}]
        with self.assertRaisesRegex(RuntimeError, "Unbound"):
            validation.check_contract(contract)

    def test_contract_rejects_path_escape_and_oracle_budget_mismatch(self):
        for name in ("../outside.cc", "/absolute.cc"):
            contract = self.contract()
            contract["source_files"] = {name: "fixture"}
            with self.assertRaisesRegex(RuntimeError, "within"):
                validation.check_contract(contract)
        contract = self.contract()
        contract["maximum_oracle_executions"] = 1
        with self.assertRaisesRegex(RuntimeError, "budget mismatch"):
            validation.check_contract(contract)

    def test_baseline_does_not_require_compressed_index_factory(self):
        (self.root / "transformer.cpp").write_text("// fixture\n")
        with patch.object(validation, "replace_once") as replace:
            validation.instrument_copy(self.root, self.root / "weights", False)
        self.assertEqual(replace.call_count, 2)
        self.assertTrue(
            all(
                call.args[0].name == "transformer.cpp"
                for call in replace.call_args_list
            )
        )

    def test_compressed_stage_redirects_factory_input(self):
        (self.root / "transformer.cpp").write_text("// fixture\n")
        with patch.object(validation, "replace_once") as replace:
            validation.instrument_copy(self.root, self.root / "weights", True)
        self.assertEqual(replace.call_count, 3)
        self.assertEqual(replace.call_args.args[0].name, "layerFactory.cc")


if __name__ == "__main__":
    unittest.main()
