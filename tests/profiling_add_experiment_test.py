#!/usr/bin/env python3
"""Check transformer_profiling/add_experiment.py against the committed E01-E36 tables.

Synthetic gem5 stats files are rebuilt from committed per-experiment TSVs, fed
back through the script, and the resulting rows must match the committed rows
byte for byte. Runs anywhere with Python 3.8+:

    python3 tests/profiling_add_experiment_test.py
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The closed historical dataset. The extraction math is checked by reproducing
# its committed E01-E36 rows, so these tests keep reading it deliberately.
FINAL = ROOT / "transformer_profiling" / "final"
# The current run directory, which the script writes to by default.
CURRENT = ROOT / "transformer_profiling" / "hsylin"
SCRIPT = ROOT / "transformer_profiling" / "add_experiment.py"

sys.dont_write_bytecode = True  # keep transformer_profiling/ free of __pycache__
sys.path.insert(0, str(SCRIPT.parent))
import add_experiment as ax  # noqa: E402

TICKS_PER_SECOND = Decimal(10) ** 12


def committed_rows(exp_id: str):
    path = next(FINAL.glob(f"{exp_id.lower()}_*.tsv"))
    with path.open(encoding="utf-8", newline="") as handle:
        return path, list(csv.DictReader(handle, delimiter="\t"))


def unit(column: str) -> Decimal:
    return Decimal("0.000001") if column == "sim_seconds" else Decimal(1)


def write_stats(path: Path, deltas, btb_values, sim_seconds_text=None) -> None:
    """Write cumulative gem5 dump blocks whose adjacent differences are `deltas`."""
    cumulative = {column: Decimal(0) for column, _, _ in ax.ADD_METRICS}
    lines = []
    for index, (delta, btb) in enumerate(zip(deltas, btb_values)):
        for column in cumulative:
            cumulative[column] += delta[column]
        seconds = cumulative["sim_seconds"]
        lines.append("---------- Begin Simulation Statistics ----------")
        text = (sim_seconds_text or {}).get(index, format(seconds, "f"))
        lines.append(f"simSeconds {text} # Number of seconds simulated (Second)")
        lines.append(f"simTicks {int(seconds * TICKS_PER_SECOND)}")
        lines.append(f"simFreq {int(TICKS_PER_SECOND)}")
        for column, source, _ in ax.ADD_METRICS[1:]:
            lines.append(f"{source} {format(cumulative[column], 'f')}")
        lines.append(f"{ax.BTB_HIT_RATIO_STAT} {btb}")
        lines.append("---------- End Simulation Statistics   ----------")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def interval_deltas(rows):
    return [
        {column: Decimal(row[column]) for column, _, _ in ax.ADD_METRICS}
        for row in rows[:6]
    ]



# The schema the interleaved int8 paths emit: four regions per attention head,
# then the block-level tail. Mirrors the markers in selfattention.cc and
# transformerBlock.cc.
PER_HEAD = [("after_qkv_h", "MHA_QKV"), ("after_qk_h", "MHA_QK"),
            ("after_softmax_h", "MHA_softmax"), ("after_sv_h", "MHA_SV")]
BLOCK_TAIL = [("after_mha", "MHA_out"), ("after_projection", "Projection"),
              ("after_attn_addnorm", "non_GEMM_after_projection"),
              ("after_ff1", "FF1"), ("after_ff2", "FF2"),
              ("final_total", "non_GEMM_after_ff2")]


def split_regions(num_heads: int):
    return PER_HEAD * num_heads + BLOCK_TAIL


def write_region_index(path: Path, regions, scope="group2_full_interleaved_transformer_block"):
    lines = [f"scope\t{scope}", "dump_index\tcheckpoint\tinterval_since_previous"]
    for index, (checkpoint, interval) in enumerate(regions, start=1):
        lines.append(f"{index}\t{checkpoint}\t{interval}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cli_args(rows, stats: Path, out: Path):
    first = rows[0]
    args = [
        "--stats", str(stats),
        "--study", first["study"],
        "--model", first["model"],
        "--n-learners", first["n_learners"],
        "--sve-bits", first["sve_bits"],
        "--exp-id", first["exp_id"],
        "--executable", first["executable"],
        "--gem5-timestamp", first["gem5_timestamp"],
        "--recorded-stats-path", first["stats_file"],
        "--output-root", str(out),
    ]
    if first["codebook_size"] == "NA":
        args.append("--dense")
    else:
        args += ["--codebook-size", first["codebook_size"]]
    return args


def run_script(args, check=True):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise AssertionError(f"add_experiment.py failed:\n{result.stderr}")
    return result


class AddExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="i2ce-profiling-test."))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def assert_reproduces(self, exp_id: str, stats: Path, extra=()) -> None:
        path, rows = committed_rows(exp_id)
        out = self.tmp / f"out_{exp_id}"
        run_script([*cli_args(rows, stats, out), *extra])
        self.assertEqual(
            (out / path.name).read_bytes(), path.read_bytes(), f"{exp_id} differs"
        )

    def test_codebook_run_with_exit_block(self) -> None:
        _, rows = committed_rows("E09")
        deltas = interval_deltas(rows)
        # gem5 appends one more block at `m5 exit`; it must be ignored.
        deltas.append({column: unit(column) for column in deltas[0]})
        btb = ["0.5"] * 5 + [rows[-1]["btb_hit_ratio"], "0.1"]
        stats = self.tmp / "stats_e09.txt"
        write_stats(stats, deltas, btb)
        self.assert_reproduces("E09", stats)

    def test_dense_multi_learner_sums_chunks(self) -> None:
        _, rows = committed_rows("E03")
        n_learners = int(rows[0]["n_learners"])
        totals = interval_deltas(rows)
        deltas = []
        for learner in range(n_learners):
            for region in totals:
                deltas.append({
                    column: value - (n_learners - 1) * unit(column) if learner == 0
                    else unit(column)
                    for column, value in region.items()
                })
        btb = ["0.5"] * (len(deltas) - 1) + [rows[-1]["btb_hit_ratio"]]
        stats = self.tmp / "stats_e03.txt"
        write_stats(stats, deltas, btb)
        self.assert_reproduces("E03", stats)

    def test_scaled_first_learner_chunk(self) -> None:
        _, rows = committed_rows("E05")
        n_learners = int(rows[0]["n_learners"])
        chunk = [
            {column: value / n_learners for column, value in region.items()}
            for region in interval_deltas(rows)
        ]
        deltas = chunk + chunk[:4]  # the unfinished second learner
        btb = ["0.5"] * 5 + [rows[-1]["btb_hit_ratio"]] + ["0.1"] * 4
        stats = self.tmp / "stats_e05.txt"
        write_stats(stats, deltas, btb)
        self.assert_reproduces("E05", stats, ["--scale-first-learner"])

    def test_malformed_sim_seconds_uses_ticks(self) -> None:
        _, rows = committed_rows("E07")
        deltas = interval_deltas(rows)
        btb = ["0.5"] * 5 + [rows[-1]["btb_hit_ratio"]]
        stats = self.tmp / "stats_e07.txt"
        # E02 block 1 contains this kind of value (`2953792.`).
        bad = format(deltas[0]["sim_seconds"] * 1000000, "f").split(".")[0] + "."
        write_stats(stats, deltas, btb, {0: bad})
        self.assert_reproduces("E07", stats)

    def test_split_mha_regions_are_summed_across_heads(self) -> None:
        """Four heads x four sub-regions collapse to one row each, totals intact."""
        num_heads = 4
        regions = split_regions(num_heads)
        self.assertEqual(len(regions), num_heads * 4 + 6)

        # A distinct, exactly representable delta per dump so the expected sum of
        # each region is unambiguous.
        deltas = []
        for index in range(len(regions)):
            deltas.append({column: Decimal(index + 1) * unit(column) * 1000
                           for column, _, _ in ax.ADD_METRICS})
        stats = self.tmp / "run_20261001_101500" / "stats_20261001_101500.txt"
        stats.parent.mkdir()
        write_stats(stats, deltas, ["0.5"] * len(regions))
        index_path = self.tmp / "gem5_profile_regions.tsv"
        write_region_index(index_path, regions)

        out = self.tmp / "split"
        run_script([
            "--stats", str(stats), "--study", "Runner", "--model", "BERT-mini",
            "--n-learners", "2", "--codebook-size", "4", "--sve-bits", "128",
            "--output-root", str(out), "--regions", str(index_path),
        ])
        written = sorted(p for p in out.glob("e01_*.tsv"))
        self.assertEqual(len(written), 1, [p.name for p in out.glob("*")])
        with written[0].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))

        # One row per interval name, in order of first appearance, plus the total.
        self.assertEqual([r["interval"] for r in rows],
                         ["MHA_QKV", "MHA_QK", "MHA_softmax", "MHA_SV", "MHA_out",
                          "Projection", "non_GEMM_after_projection", "FF1", "FF2",
                          "non_GEMM_after_ff2", "total_program"])
        self.assertEqual([r["row_kind"] for r in rows],
                         ["interval_delta"] * 10 + ["final_total"])

        # Each per-head region is the sum of its four dumps.
        by_interval = {r["interval"]: r for r in rows}
        for position, (_, interval) in enumerate(PER_HEAD):
            expected = sum(position + 1 + 4 * head for head in range(num_heads)) * 1000
            self.assertEqual(int(by_interval[interval]["instructions"]), expected,
                             f"{interval} should sum its {num_heads} per-head dumps")
        # A block-level region keeps its single dump.
        self.assertEqual(int(by_interval["FF1"]["instructions"]),
                         (num_heads * 4 + 4) * 1000)
        # The total is the last cumulative dump, i.e. every delta added up.
        self.assertEqual(int(by_interval["total_program"]["instructions"]),
                         sum(range(1, len(regions) + 1)) * 1000)
        # The stage rows still account for the whole window.
        self.assertEqual(
            sum(int(r["instructions"]) for r in rows if r["row_kind"] == "interval_delta"),
            int(by_interval["total_program"]["instructions"]))
        self.assertEqual(by_interval["total_program"]["checkpoint"], "final_total")

    def test_region_index_must_be_complete(self) -> None:
        regions = split_regions(2)
        stats = self.tmp / "run_20261001_110000" / "stats_20261001_110000.txt"
        stats.parent.mkdir()
        deltas = [{column: Decimal(1) * unit(column) for column, _, _ in ax.ADD_METRICS}
                  for _ in regions]
        write_stats(stats, deltas, ["0.5"] * len(regions))

        gapped = self.tmp / "gapped.tsv"
        lines = ["scope\tx", "dump_index\tcheckpoint\tinterval_since_previous"]
        for index, (checkpoint, interval) in enumerate(regions, start=1):
            if index == 3:
                continue
            lines.append(f"{index}\t{checkpoint}\t{interval}")
        gapped.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ax.ExtractionError):
            ax.read_region_index(gapped)

        empty = self.tmp / "empty.tsv"
        empty.write_text("complete\n", encoding="utf-8")
        with self.assertRaises(ax.ExtractionError):
            ax.read_region_index(empty)

        # A run whose stats file has fewer blocks than the index is refused.
        short_index = self.tmp / "long.tsv"
        write_region_index(short_index, split_regions(8))
        result = run_script([
            "--stats", str(stats), "--study", "Runner", "--model", "BERT-mini",
            "--n-learners", "2", "--codebook-size", "4", "--sve-bits", "128",
            "--output-root", str(self.tmp / "refused"), "--regions", str(short_index),
        ], check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("region index lists", result.stderr)

    def test_legacy_six_region_runs_need_no_index(self) -> None:
        """The default schema is unchanged, so committed rows still reproduce."""
        self.assertEqual([interval for _, interval in ax.REGIONS],
                         ["MHA", "Projection", "non_GEMM_after_projection",
                          "FF1", "FF2", "non_GEMM_after_ff2"])
        self.assertEqual(ax.output_regions(ax.REGIONS), list(ax.REGIONS))
        # Collapsing is order-preserving and idempotent for a unique schema.
        self.assertEqual(ax.output_regions(split_regions(3)),
                         PER_HEAD + BLOCK_TAIL)

    def test_default_output_root_is_current_dataset(self) -> None:
        """New runs go to hsylin/ and take the next ID in that directory."""
        self.assertEqual(ax.DEFAULT_OUTPUT_ROOT, CURRENT)
        self.assertEqual(ax.combined_table_path(CURRENT).name, "hsylin_all_experiments.tsv")
        self.assertEqual(ax.combined_table_path(FINAL).name, "final_all_experiments.tsv")
        # The two datasets number independently. hsylin/ was emptied when the
        # MHA split changed the interval schema, so it starts again at E01.
        self.assertEqual(ax.next_exp_id(CURRENT / "manifest.tsv"), "E01")
        self.assertEqual(ax.next_exp_id(FINAL / "manifest.tsv"), "E37")
        ids = [row["exp_id"] for row in ax.read_tsv(CURRENT / "manifest.tsv", ax.MANIFEST_COLUMNS)]
        self.assertEqual(ids, [])
        # E37-E46 are gone from the historical manifest, E01-E36 remain.
        historical = [row["exp_id"] for row in ax.read_tsv(FINAL / "manifest.tsv", ax.MANIFEST_COLUMNS)]
        self.assertEqual(historical, [f"E{n:02d}" for n in range(1, 37)])

    def test_next_run_in_current_dataset_becomes_e01(self) -> None:
        """A real collect into a copy of the emptied hsylin/ lands as E01."""
        _, rows = committed_rows("E09")
        stats = self.tmp / "run_20260930_120000" / "stats_20260930_120000.txt"
        stats.parent.mkdir()
        write_stats(stats, interval_deltas(rows), ["0.5"] * 5 + [rows[-1]["btb_hit_ratio"]])
        out = self.tmp / "hsylin"
        shutil.copytree(CURRENT, out)
        combined_name = "hsylin_all_experiments.tsv"
        before = {name: (out / name).read_text(encoding="utf-8")
                  for name in ("manifest.tsv", combined_name)}
        run_script([
            "--stats", str(stats), "--study", "Runner", "--model", "BERT-mini",
            "--n-learners", "2", "--codebook-size", "4", "--sve-bits", "128",
            "--output-root", str(out),
        ])
        new_file = out / "e01_runner_n2_cb4_sve128_run_20260930_120000.tsv"
        self.assertTrue(new_file.exists(), sorted(p.name for p in out.glob("e*.tsv")))
        manifest = (out / "manifest.tsv").read_text(encoding="utf-8")
        combined = (out / combined_name).read_text(encoding="utf-8")
        self.assertTrue(manifest.startswith(before["manifest.tsv"]))
        self.assertTrue(combined.startswith(before[combined_name]))
        self.assertTrue(manifest.splitlines()[-1].startswith("E01\t"))
        self.assertEqual(combined.splitlines()[-7:],
                         new_file.read_text(encoding="utf-8").splitlines()[1:])
        # No stray final_all_experiments.tsv, and the historical tree is untouched.
        self.assertFalse((out / "final_all_experiments.tsv").exists())
        self.assertEqual(ax.next_exp_id(out / "manifest.tsv"), "E02")

    def test_appends_next_id_without_touching_existing_rows(self) -> None:
        _, rows = committed_rows("E09")
        stats = self.tmp / "run_20260921_101500" / "stats_20260921_101500.txt"
        stats.parent.mkdir()
        write_stats(stats, interval_deltas(rows), ["0.5"] * 5 + [rows[-1]["btb_hit_ratio"]])
        out = self.tmp / "final"
        shutil.copytree(FINAL, out)
        before = {
            name: (out / name).read_text(encoding="utf-8")
            for name in ("manifest.tsv", "final_all_experiments.tsv")
        }
        next_id = ax.next_exp_id(out / "manifest.tsv")
        self.assertEqual(next_id, "E37")  # the historical dataset ends at E36
        run_script([
            "--stats", str(stats), "--study", "Learner scaling", "--model", "BERT-mini",
            "--n-learners", "4", "--codebook-size", "8", "--sve-bits", "128",
            "--output-root", str(out),
        ])
        new_file = out / (
            f"{next_id.lower()}_learner_scaling_n4_cb8_sve128_run_20260921_101500.tsv"
        )
        self.assertTrue(new_file.exists())
        manifest = (out / "manifest.tsv").read_text(encoding="utf-8")
        combined = (out / "final_all_experiments.tsv").read_text(encoding="utf-8")
        self.assertTrue(manifest.startswith(before["manifest.tsv"]))
        self.assertTrue(combined.startswith(before["final_all_experiments.tsv"]))
        self.assertTrue(manifest.splitlines()[-1].startswith(f"{next_id}\t"))
        new_rows = new_file.read_text(encoding="utf-8").splitlines()[1:]
        self.assertEqual(combined.splitlines()[-7:], new_rows)

        # The same ID is refused unless --replace is given.
        again = run_script([
            "--stats", str(stats), "--study", "Learner scaling", "--model", "BERT-mini",
            "--n-learners", "4", "--codebook-size", "8", "--sve-bits", "128",
            "--output-root", str(out), "--exp-id", next_id,
        ], check=False)
        self.assertNotEqual(again.returncode, 0)
        self.assertIn("--replace", again.stderr)

    def test_too_few_blocks_is_an_error(self) -> None:
        _, rows = committed_rows("E09")
        stats = self.tmp / "stats_20260921_101500.txt"
        write_stats(stats, interval_deltas(rows)[:5], ["0.5"] * 5)
        result = run_script(cli_args(rows, stats, self.tmp / "out"), check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("need 6", result.stderr)


if __name__ == "__main__":
    outcome = unittest.main(exit=False, verbosity=2).result
    sys.exit(0 if outcome.wasSuccessful() else 1)
