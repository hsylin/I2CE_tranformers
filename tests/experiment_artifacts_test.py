"""Check generator preparation and config patching using temporary inputs."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tools/exp/lib"
spec = importlib.util.spec_from_file_location("artifacts", HELPERS / "gen_artifacts.py")
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class ArtifactTests(unittest.TestCase):
    def test_equal_assignment_is_still_matched(self):
        self.assertEqual(
            artifacts.override_line("N_LEARNERS = 2  # count", "N_LEARNERS", "2"),
            ("N_LEARNERS = 2  # count", True),
        )
        self.assertFalse(
            artifacts.override_line("OTHER_N_LEARNERS = 2", "N_LEARNERS", "4")[1]
        )

    def test_generator_dry_run_never_executes_cells_or_creates_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = root / "repo/Full_NN/generators"
            generator.mkdir(parents=True)
            notebook = dict(
                cells=[
                    dict(
                        cell_type="code",
                        source=[
                            "N_LEARNERS = 2\n",
                            "CODEBOOK_SIZE = 4\n",
                            "SVE_LANES = 4\n",
                            "TICSAT_REPO_ROOT = None\n",
                            "raise RuntimeError('must not execute')\n",
                        ],
                    )
                ]
            )
            (generator / "Transformer_generator.ipynb").write_text(json.dumps(notebook))
            output = root / "output"
            command = [
                sys.executable,
                str(HELPERS / "gen_artifacts.py"),
                "--repo",
                str(root / "repo"),
                "--stage-root",
                str(output),
                "--n-learners",
                "2",
                "--codebook-size",
                "4",
                "--dry-run",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists())
            notebook["cells"][0]["source"].pop(0)
            (generator / "Transformer_generator.ipynb").write_text(json.dumps(notebook))
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 3)
            self.assertFalse(output.exists())

    def test_config_patch_is_idempotent_and_preserves_original(self):
        original = """def configure():
    parser.add_argument("--restore", type=str, default=None)
    root.system.sve_vl = 1
    system.addCaches(want_caches, last_cache_level=2)
"""
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "starter_fs.py"
            config.write_text(original)
            command = [sys.executable, str(HELPERS / "apply_gem5_opts.py"), str(config)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            amended = config.read_bytes()
            self.assertEqual(config.with_suffix(".py.orig").read_text(), original)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(config.read_bytes(), amended)
            self.assertEqual(config.read_text().count('"--sve-vl"'), 1)

    def test_unknown_config_anchor_fails_without_editing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "starter_fs.py"
            config.write_text("pass\n")
            result = subprocess.run(
                [sys.executable, str(HELPERS / "apply_gem5_opts.py"), str(config)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 3)
            self.assertEqual(config.read_text(), "pass\n")
            self.assertFalse(config.with_suffix(".py.orig").exists())


if __name__ == "__main__":
    unittest.main()
