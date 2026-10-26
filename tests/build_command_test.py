"""Validate build argument boundaries without pretending to compile ARM code."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BuildCommandTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "transformer_layers").mkdir()
        (self.root / "transformer_layers/a.cc").touch()
        (self.root / "transformer_layers/b.cc").touch()
        shutil.copyfile(ROOT / "compile_transformer.sh", self.root / "build.sh")
        compiler = self.root / "fake-compiler"
        compiler.write_text(
            "#!" + sys.executable + "\n"
            "import json, os, sys\n"
            "if sys.argv[1:] == ['-print-sysroot']:\n"
            "    print('/test/sysroot')\n"
            "else:\n"
            "    with open(os.environ['ARGUMENT_LOG'], 'w') as stream:\n"
            "        json.dump(sys.argv[1:], stream)\n"
            "    sys.exit(int(os.environ.get('COMPILER_STATUS', '0')))\n"
        )
        compiler.chmod(0o755)
        self.environment = {
            key: value
            for key, value in os.environ.items()
            if not key.endswith("_FLAG")
            and key not in {"GEM5_ROOT", "LIBM5_A", "M5_INC"}
        }
        self.environment.update(
            A64CXX=str(compiler), ARGUMENT_LOG=str(self.root / "arguments.json")
        )

    def run_build(self, **flags):
        return subprocess.run(
            ["bash", "build.sh"],
            cwd=self.root,
            env=dict(self.environment, **flags),
            capture_output=True,
            text=True,
            timeout=20,
        )

    def arguments(self):
        return json.loads((self.root / "arguments.json").read_text())

    def test_default_flags(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        self.assertEqual(args[:3], ["-std=c++17", "-O2", "-Wall"])
        self.assertIn("-DRELOAD_WEIGHT", args)
        self.assertIn("-DUSE_NOTEBOOK_GENERATED_WEIGHTS", args)
        self.assertIn("-DCORE_NUM=1", args)
        self.assertNotIn("-DSIMD", args)
        self.assertNotIn("-DI2CE_USE_LIBM5", args)

    def test_direct_sve_build(self):
        include = self.root / "include"
        (include / "gem5").mkdir(parents=True)
        (include / "gem5/m5ops.h").touch()
        library = self.root / "libm5.a"
        library.touch()
        result = self.run_build(
            SIMD_FLAG="1",
            CORE_NUM_FLAG="3",
            I2CE_USE_LIBM5_FLAG="1",
            LIBM5_A=str(library),
            M5_INC=str(include),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = self.arguments()
        for option in [
            "-march=armv8-a+sve",
            "-DSIMD",
            "-DI2CE_USE_LIBM5",
            "-DCORE_NUM=3",
            "Full_NN/src/gemm_SVE.c",
            "-I" + str(include),
        ]:
            self.assertIn(option, args)
        self.assertGreater(args.index(str(library)), args.index("transformer.o"))

    def test_direct_build_rejects_missing_inputs(self):
        result = self.run_build(I2CE_USE_LIBM5_FLAG="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "arguments.json").exists())

    def test_compiler_failure_is_propagated(self):
        result = self.run_build(COMPILER_STATUS="9")
        self.assertEqual(result.returncode, 9)

    def test_library_paths_with_spaces(self):
        include = self.root / "guest headers"
        (include / "gem5").mkdir(parents=True)
        (include / "gem5/m5ops.h").touch()
        library = self.root / "guest library.a"
        library.touch()
        result = self.run_build(
            I2CE_USE_LIBM5_FLAG="1", LIBM5_A=str(library), M5_INC=str(include)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(library), self.arguments())
        self.assertIn("-I" + str(include), self.arguments())

    def test_direct_build_checks_archive_and_header(self):
        include = self.root / "include"
        library = self.root / "libm5.a"
        for archive_exists in (False, True):
            with self.subTest(archive_exists=archive_exists):
                if archive_exists:
                    library.touch()
                result = self.run_build(
                    I2CE_USE_LIBM5_FLAG="1",
                    LIBM5_A=str(library),
                    M5_INC=str(include),
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((self.root / "arguments.json").exists())


if __name__ == "__main__":
    unittest.main()
