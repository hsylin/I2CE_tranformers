"""Exercise collection using synthetic statistics; never launch gem5."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import profiling_add_experiment_test as extractor

ROOT = Path(__file__).resolve().parents[1]


class CollectionTest(unittest.TestCase):
    def test_dry_collection_has_no_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = root / "tools/exp"
            runner.mkdir(parents=True)
            shutil.copy(ROOT / "tools/exp/exp.sh", runner / "exp.sh")
            (runner / "experiments.tsv").write_text(
                "37\t8\t4\t128\tcodebook_int8\t-\tsynthetic fixture\n"
            )
            (runner / "runner.conf").write_text(
                f"EXP_ROOT='{root}/runs'\nGEM5_BIN=/unused\nGEM5_CWD=/unused\n"
                f"KERNEL=/unused\nDISK=/unused\nGEN_PYTHON='{sys.executable}'\n"
            )
            profiling = root / "transformer_profiling"
            profiling.mkdir()
            shutil.copy(ROOT / "transformer_profiling/add_experiment.py", profiling)
            output = root / "runs/37/out_20260926_120000"
            output.mkdir(parents=True)
            _, rows = extractor.committed_rows("E09")
            extractor.write_stats(
                output / "stats.txt", extractor.interval_deltas(rows), ["0.5"] * 6
            )
            extractor.write_region_index(
                output / "gem5_profile_regions.tsv", extractor.ax.REGIONS
            )
            command = [
                "bash",
                str(runner / "exp.sh"),
                "collect",
                "37",
                "--output-root",
                str(root / "results"),
            ]
            before = {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            dry = subprocess.run(
                command + ["--dry-run"],
                capture_output=True,
                text=True,
                env=dict(os.environ, REPO_ROOT=str(ROOT)),
            )
            self.assertEqual(dry.returncode, 0, dry.stderr)
            after = {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
            self.assertEqual(before, after)
            actual = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=dict(os.environ, REPO_ROOT=str(ROOT)),
            )
            self.assertEqual(actual.returncode, 0, actual.stderr)
            self.assertTrue((output / "collected_as").exists())
            self.assertTrue((root / "results/results_all_experiments.tsv").exists())
            self.assertNotIn("git push", actual.stdout)


if __name__ == "__main__":
    unittest.main()
