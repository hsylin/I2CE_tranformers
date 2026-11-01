"""Exercise scalar normalization boundaries without running an experiment."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SoftmaxDenominatorTest(unittest.TestCase):
    def test_scalar_and_rearranged_zero_small_and_normal_sums(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        self.assertIsNotNone(compiler, "A native C++ compiler is required")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "softmax-test"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-fsanitize=undefined",
                    "-fno-sanitize-recover=all",
                    "-I",
                    str(ROOT),
                    str(ROOT / "tests/softmax_denominator_test.cc"),
                    str(ROOT / "transformer_layers/softmax.cc"),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, timeout=20)
