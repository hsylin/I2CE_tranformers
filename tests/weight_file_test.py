"""Exercise the actual file loader without compiling the transformer model."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WeightFileTest(unittest.TestCase):
    def test_missing_short_malformed_extra_and_valid_inputs(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        self.assertIsNotNone(compiler, "A native C++ compiler is required")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "weight-file-test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-I",
                    str(ROOT),
                    str(ROOT / "tests/weight_file_test.cc"),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run(
                [str(binary), directory], check=True, capture_output=True, timeout=20
            )
