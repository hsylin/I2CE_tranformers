"""Check dump ordering and deferred index output with host m5 stubs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProfileBoundaryTest(unittest.TestCase):
    def test_region_and_legacy_modes(self):
        compiler = shutil.which("c++")
        self.assertIsNotNone(compiler, "A host C++ compiler is required")
        for regions in (False, True):
            with self.subTest(regions=regions), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                (root / "gem5").mkdir()
                (root / "gem5/m5ops.h").write_text(
                    "#include <cstdint>\n"
                    'extern "C" {\n'
                    "void m5_reset_stats(uint64_t, uint64_t);\n"
                    "void m5_dump_stats(uint64_t, uint64_t);\n"
                    "void m5_dump_reset_stats(uint64_t, uint64_t);\n}\n"
                )
                harness = root / "main.cc"
                harness.write_text(
                    '#include "profile.h"\n'
                    "#include <cstdint>\n#include <cstdio>\n"
                    'extern "C" void m5_reset_stats(uint64_t, uint64_t) '
                    '{ std::puts("RESET"); }\n'
                    'extern "C" void m5_dump_stats(uint64_t, uint64_t) '
                    '{ std::puts("DUMP"); }\n'
                    'extern "C" void m5_dump_reset_stats(uint64_t, uint64_t) '
                    '{ std::puts("DUMP_RESET"); }\n'
                    'int main() { resetTransformerStatsWindow("test");\n'
                    'dumpTransformerStatsCheckpointIfProfiling("a", "first");\n'
                    'dumpTransformerStatsLegacyBoundary("b", "second");\n'
                    'std::puts("BEFORE_FINALIZE");\n'
                    "finalizeTransformerStatsWindow();\n"
                    "finalizeTransformerStatsWindow(); }\n"
                )
                command = [
                    compiler,
                    "-std=c++17",
                    "-DI2CE_USE_LIBM5",
                    "-I" + str(root),
                    "-I" + str(ROOT / "transformer_layers"),
                    str(ROOT / "transformer_layers/profile.cc"),
                    str(harness),
                    "-o",
                    str(root / "check"),
                ]
                if regions:
                    command.insert(1, "-DGEM5_PROFILE_REGIONS")
                subprocess.run(command, check=True, capture_output=True, timeout=60)
                result = subprocess.run(
                    [str(root / "check")],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                lines = result.stdout.splitlines()
                index = root / "gem5_profile_regions.tsv"
                if regions:
                    self.assertEqual(lines.count("DUMP"), 2)
                    self.assertNotIn("DUMP_RESET", lines)
                    first_echo = next(
                        i for i, line in enumerate(lines) if "dump 1 " in line
                    )
                    self.assertGreater(first_echo, lines.index("BEFORE_FINALIZE"))
                    self.assertEqual(sum("[GEM5_PROFILE] dump " in x for x in lines), 2)
                    self.assertEqual(
                        index.read_text().splitlines(),
                        [
                            "scope\ttest",
                            "dump_index\tcheckpoint\tinterval_since_previous",
                            "1\ta\tfirst",
                            "2\tb\tsecond",
                        ],
                    )
                else:
                    self.assertEqual(lines.count("DUMP_RESET"), 1)
                    self.assertNotIn("DUMP", lines)
                    self.assertFalse(index.exists())


if __name__ == "__main__":
    unittest.main()
