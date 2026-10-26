import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/exp/prepare_stage.py"
SPEC = importlib.util.spec_from_file_location("prepare_stage", MODULE_PATH)
stage_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage_tool)


class PrepareStageTests(unittest.TestCase):
    def test_rejects_path_escape(self):
        for name in ["", "/tmp/input", "../input", "a/../../input"]:
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                stage_tool.relative_path(name)

    def test_accepts_nested_source_path(self):
        self.assertEqual(
            stage_tool.relative_path("Full_NN/src/gemm_exec.c"),
            Path("Full_NN/src/gemm_exec.c"),
        )

    def test_rejects_source_hash_mismatch(self):
        with mock.patch.object(stage_tool, "git_bytes", return_value=b"changed"):
            with self.assertRaisesRegex(RuntimeError, "Source hash mismatch"):
                stage_tool.checked_bytes("revision", "source.c", "0" * 64)

    def test_unknown_stage_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with self.assertRaises(RuntimeError):
                stage_tool.prepare("E99", "HEAD", output)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        manifest = {"stage": "E01"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinel = output / "keep"
            sentinel.write_bytes(b"original")
            with (
                mock.patch.object(
                    stage_tool.subprocess, "check_output", return_value="a" * 40
                ),
                mock.patch.object(
                    stage_tool, "git_bytes", return_value=json.dumps(manifest).encode()
                ),
                self.assertRaisesRegex(RuntimeError, "Output already exists"),
            ):
                stage_tool.prepare("E01", "HEAD", output)
            self.assertEqual(sentinel.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
