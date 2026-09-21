#!/usr/bin/env python3
"""Add one gem5 profiling run to the transformer_profiling/final tables.

The script converts the stats file of a single `transformer.o` run compiled
with `PROFILE_GEMM_ONLY_FLAG=1 GEM5_PROFILE_REGIONS_FLAG=1` into the same
seven-row TSV format as E01-E36, then adds the experiment to `manifest.tsv`
and `final_all_experiments.tsv`. The interval math, column order and number
formatting follow `transformer_profiling/extract_final_stats.py` in the
Jerry0209/gem5 fork, which produced E01-E36.

The next free experiment ID (E37, E38, ...) is chosen from `manifest.tsv`
unless `--exp-id` is given.

Examples:
  # Codebook SIMD int8, BERT-mini, 4 learners, CB = 8, 256-bit SVE
  python3 transformer_profiling/add_experiment.py \\
      --stats /home/thu/gem5/output/run_20260921_101500/stats_20260921_101500.txt \\
      --study "Learner scaling" --model BERT-mini \\
      --n-learners 4 --codebook-size 8 --sve-bits 256

  # Dense no-SIMD baseline with 2 learners (12 dumps are expected)
  python3 transformer_profiling/add_experiment.py \\
      --stats /home/thu/gem5/output/run_20260921_120000/stats_20260921_120000.txt \\
      --study "Dense baseline" --model BERT-base \\
      --n-learners 2 --dense --sve-bits 128

  # Preview the rows without writing anything
  python3 transformer_profiling/add_experiment.py ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, getcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


getcontext().prec = 40

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "final"

# (checkpoint, interval) for the six `m5 dumpstats` calls of one Transformer
# block, in the order emitted by transformer_layers/profile.cc.
REGIONS = [
    ("after_mha", "MHA"),
    ("after_projection", "Projection"),
    ("after_attn_addnorm", "non_GEMM_after_projection"),
    ("after_ff1", "FF1"),
    ("after_ff2", "FF2"),
    ("final_total", "non_GEMM_after_ff2"),
]

# (output column, gem5 stat, format)
ADD_METRICS = [
    ("sim_seconds", "simSeconds", "seconds"),
    ("instructions", "simInsts", "int"),
    ("ops", "simOps", "int"),
    ("cpu_cycles", "system.cpu_cluster.cpus.numCycles", "int"),
    ("memory_references", "system.cpu_cluster.cpus.commitStats0.numMemRefs", "int"),
    ("load_instructions", "system.cpu_cluster.cpus.commitStats0.numLoadInsts", "int"),
    ("store_instructions", "system.cpu_cluster.cpus.commitStats0.numStoreInsts", "int"),
    ("dcache_demand_accesses", "system.cpu_cluster.cpus.dcache.demandAccesses::total", "int"),
    ("dcache_demand_misses", "system.cpu_cluster.cpus.dcache.demandMisses::total", "int"),
    ("icache_demand_accesses", "system.cpu_cluster.cpus.icache.demandAccesses::total", "int"),
    ("icache_demand_misses", "system.cpu_cluster.cpus.icache.demandMisses::total", "int"),
    ("l2_demand_accesses", "system.cpu_cluster.l2.demandAccesses::total", "int"),
    ("l2_demand_misses", "system.cpu_cluster.l2.demandMisses::total", "int"),
    ("committed_branches", "system.cpu_cluster.cpus.branchPred.committed_0::total", "int"),
    ("branch_mispredictions", "system.cpu_cluster.cpus.branchPred.mispredicted_0::total", "int"),
]

BTB_HIT_RATIO_STAT = "system.cpu_cluster.cpus.branchPred.BTBHitRatio"

OUTPUT_COLUMNS = [
    "exp_id",
    "study",
    "model",
    "d_q",
    "d_seq",
    "d_model",
    "num_head",
    "d_ff",
    "implementation",
    "n_learners",
    "codebook_size",
    "sve_bits",
    "executable",
    "gem5_timestamp",
    "stats_file",
    "row_index",
    "row_kind",
    "dump_index",
    "checkpoint",
    "interval",
    *[column for column, _, _ in ADD_METRICS],
    "ipc",
    "cpi",
    "dcache_demand_miss_rate",
    "icache_demand_miss_rate",
    "l2_demand_miss_rate",
    "branch_misprediction_rate",
    "btb_hit_ratio",
]

MANIFEST_COLUMNS = [
    "exp_id",
    "study",
    "implementation",
    "n_learners",
    "codebook_size",
    "sve_bits",
    "gem5_timestamp",
    "stats_file",
    "output_file",
    "stats_blocks",
    "rows_written",
]

# name -> (d_q, d_seq, d_model, num_head, d_ff), matching transformer.h
MODEL_PRESETS = {
    "BERT-mini": (64, 512, 256, 4, 1024),
    "BERT-base": (64, 512, 768, 12, 3072),
}

EXP_ID_RE = re.compile(r"^E(\d+)$")
TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")
SECONDS_QUANTUM = Decimal("0.000001")

Block = Dict[str, Decimal]
Metrics = Dict[str, Decimal]


class ExtractionError(Exception):
    """Raised for input problems the user has to fix."""


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    study: str
    model: str
    d_q: int
    d_seq: int
    d_model: int
    num_head: int
    d_ff: int
    implementation: str
    n_learners: int
    codebook_size: str
    sve_bits: int
    executable: str
    gem5_timestamp: str
    stats_file: str
    is_dense: bool
    scale_first_learner: bool

    @property
    def number(self) -> int:
        return exp_number(self.exp_id)

    @property
    def output_name(self) -> str:
        return (
            f"e{self.number:02d}_{slugify_study(self.study)}_n{self.n_learners}_"
            f"cb{self.codebook_size}_sve{self.sve_bits}_{self.gem5_timestamp}.tsv"
        )


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stats", type=Path, required=True,
        help="gem5 stats file of the run, e.g. output/run_<ts>/stats_<ts>.txt",
    )
    parser.add_argument(
        "--study", required=True,
        help='study label, e.g. "Dense baseline", "Learner scaling", '
             '"Codebook-size scaling"; also used in the file name',
    )
    parser.add_argument(
        "--model", required=True,
        help="BERT-mini or BERT-base; any other name also needs --dims",
    )
    parser.add_argument(
        "--dims", type=int, nargs=5, metavar=("D_Q", "D_SEQ", "D_MODEL", "NUM_HEAD", "D_FF"),
        help="model dimensions from transformer.h (required for non-preset models)",
    )
    parser.add_argument("--n-learners", type=int, required=True, help="N_LEARNERS of the run")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--codebook-size", type=int, help="CODEBOOK_SIZE of a codebook run")
    kind.add_argument(
        "--dense", action="store_true",
        help="dense baseline run (codebook_size is recorded as NA)",
    )
    parser.add_argument("--sve-bits", type=int, required=True, help="gem5 SVE vector length in bits")
    parser.add_argument(
        "--exp-id",
        help="experiment ID such as E37 (default: next free ID in manifest.tsv)",
    )
    parser.add_argument(
        "--implementation",
        help='implementation label (default: "Dense naive/default" or "Codebook SIMD int8")',
    )
    parser.add_argument(
        "--executable",
        help="executable name to record (default: the E01-E36 naming scheme, e.g. "
             "transformer_BERT_mini_int8_noTiling_4D_CB_8_SVE_256.o)",
    )
    parser.add_argument(
        "--gem5-timestamp",
        help="run_YYYYMMDD_HHMMSS label (default: taken from the stats path)",
    )
    parser.add_argument(
        "--recorded-stats-path",
        help="path to write in the stats_file column (default: absolute --stats path); "
             "useful when the stats file was copied from the simulation server",
    )
    parser.add_argument(
        "--start-block", type=int, default=1,
        help="1-based stats block where this program's dumps start (default: 1); "
             "use it when several programs ran in the same gem5 session",
    )
    parser.add_argument(
        "--scale-first-learner", action="store_true",
        help="dense multi-learner runs only: use the first learner chunk multiplied by "
             "--n-learners (the estimate used for the unfinished BERT-base E05/E06 runs)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="directory holding manifest.tsv and final_all_experiments.tsv "
             "(default: transformer_profiling/final next to this script)",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="overwrite an experiment ID that already exists",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the rows to stdout and write nothing",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Experiment metadata
# --------------------------------------------------------------------------


def exp_number(exp_id: str) -> int:
    match = EXP_ID_RE.match(exp_id)
    if not match:
        raise ExtractionError(f"experiment ID must look like E37, got {exp_id!r}")
    return int(match.group(1))


def normalize_exp_id(exp_id: str) -> str:
    label = exp_id.strip().upper()
    return f"E{exp_number(label if label.startswith('E') else 'E' + label):02d}"


def slugify_study(study: str) -> str:
    return study.lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def model_dims(model: str, dims: Optional[Sequence[int]]) -> Tuple[int, int, int, int, int]:
    preset = MODEL_PRESETS.get(model)
    if dims is not None:
        dims = tuple(dims)
        if preset is not None and dims != preset:
            raise ExtractionError(
                f"--dims {list(dims)} does not match the {model} preset {list(preset)}"
            )
        return dims  # type: ignore[return-value]
    if preset is None:
        known = ", ".join(MODEL_PRESETS)
        raise ExtractionError(f"unknown model {model!r}; use one of {known} or pass --dims")
    return preset


def derive_gem5_timestamp(stats_path: Path, override: Optional[str]) -> str:
    if override:
        label = override.strip()
        return f"run_{label}" if re.fullmatch(r"\d{8}_\d{6}", label) else label
    # Prefer the file name (stats_<ts>.txt), then its directory (run_<ts>).
    for part in (stats_path.name, stats_path.parent.name):
        match = TIMESTAMP_RE.search(part)
        if match:
            return f"run_{match.group(1)}"
    raise ExtractionError(
        f"no YYYYMMDD_HHMMSS timestamp found in {stats_path}; pass --gem5-timestamp"
    )


def next_exp_id(manifest_path: Path) -> str:
    highest = 0
    if manifest_path.exists():
        for row in read_tsv(manifest_path, MANIFEST_COLUMNS):
            highest = max(highest, exp_number(row["exp_id"]))
    return f"E{highest + 1:02d}"


def build_experiment(args: argparse.Namespace) -> Experiment:
    if args.n_learners < 1:
        raise ExtractionError("--n-learners must be at least 1")
    if args.scale_first_learner and not (args.dense and args.n_learners > 1):
        raise ExtractionError("--scale-first-learner only applies to --dense runs with --n-learners > 1")

    d_q, d_seq, d_model, num_head, d_ff = model_dims(args.model, args.dims)
    model_token = args.model.replace("-", "_")

    if args.dense:
        codebook_size = "NA"
        implementation = args.implementation or "Dense naive/default"
        executable_impl = "dense_baseline"
    else:
        if args.codebook_size < 1:
            raise ExtractionError("--codebook-size must be positive")
        codebook_size = str(args.codebook_size)
        implementation = args.implementation or "Codebook SIMD int8"
        executable_impl = f"CB_{codebook_size}"

    executable = args.executable or (
        f"transformer_{model_token}_int8_noTiling_{args.n_learners}D_"
        f"{executable_impl}_SVE_{args.sve_bits}.o"
    )

    stats_path = args.stats.expanduser().resolve()
    manifest_path = args.output_root / "manifest.tsv"
    exp_id = normalize_exp_id(args.exp_id) if args.exp_id else next_exp_id(manifest_path)

    return Experiment(
        exp_id=exp_id,
        study=args.study,
        model=args.model,
        d_q=d_q,
        d_seq=d_seq,
        d_model=d_model,
        num_head=num_head,
        d_ff=d_ff,
        implementation=implementation,
        n_learners=args.n_learners,
        codebook_size=codebook_size,
        sve_bits=args.sve_bits,
        executable=executable,
        gem5_timestamp=derive_gem5_timestamp(stats_path, args.gem5_timestamp),
        stats_file=args.recorded_stats_path or str(stats_path),
        is_dense=args.dense,
        scale_first_learner=args.scale_first_learner,
    )


# --------------------------------------------------------------------------
# gem5 stats parsing
# --------------------------------------------------------------------------


def parse_stats_blocks(path: Path) -> List[Block]:
    """Return one {stat: value} dict per Begin/End Simulation Statistics block."""
    blocks: List[Block] = []
    current: Optional[Block] = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "Begin Simulation Statistics" in line:
                current = {}
                continue
            if "End Simulation Statistics" in line:
                if current is None:
                    raise ExtractionError(f"End marker before Begin marker in {path}")
                blocks.append(current)
                current = None
                continue
            if current is None:
                continue
            parts = line.split("#", 1)[0].split()
            if len(parts) < 2:
                continue
            try:
                current[parts[0]] = Decimal(parts[1])
            except InvalidOperation:
                continue
    if current is not None:
        # gem5 was still running (or was killed) while writing this block.
        print(f"note: ignoring an unfinished stats block at the end of {path}", file=sys.stderr)
    return blocks


def stat(block: Block, name: str, block_number: int) -> Decimal:
    try:
        return block[name]
    except KeyError:
        raise ExtractionError(
            f"stats block {block_number} has no {name!r}; check that the run used the "
            f"same gem5 configuration (MinorCPU, cpu_cluster with l2) as E01-E36"
        ) from None


def repair_sim_seconds(blocks: List[Block], first_block_number: int) -> None:
    """Fall back to simTicks / simFreq when gem5 printed a malformed simSeconds.

    gem5 occasionally prints simSeconds wrongly (E02 block 1 reads
    `2953792.` while simTicks / simFreq is 2.953792). The printed value is
    kept whenever it agrees with the tick count to within its six decimals.
    """
    for offset, block in enumerate(blocks):
        ticks, freq, seconds = block.get("simTicks"), block.get("simFreq"), block.get("simSeconds")
        if ticks is None or not freq:
            continue
        from_ticks = (ticks / freq).quantize(SECONDS_QUANTUM, rounding=ROUND_HALF_UP)
        if seconds is None or not seconds.is_finite() or abs(seconds - ticks / freq) > SECONDS_QUANTUM:
            print(
                f"warning: stats block {first_block_number + offset}: simSeconds={seconds} "
                f"disagrees with simTicks/simFreq={from_ticks}; using simTicks/simFreq",
                file=sys.stderr,
            )
            block["simSeconds"] = from_ticks


def check_single_stats_window(blocks: List[Block], first_block_number: int) -> None:
    """All dumps of one program are cumulative since one `m5 resetstats`."""
    previous: Optional[Decimal] = None
    for offset, block in enumerate(blocks):
        ticks = block.get("simTicks")
        if ticks is not None and previous is not None and ticks < previous:
            raise ExtractionError(
                f"simTicks drops at stats block {first_block_number + offset}: the selected "
                f"blocks span a stats reset (another program run?); set --start-block"
            )
        if ticks is not None:
            previous = ticks


# --------------------------------------------------------------------------
# Interval math (same as extract_final_stats.py)
# --------------------------------------------------------------------------


def zero_metrics() -> Metrics:
    return {column: Decimal(0) for column, _, _ in ADD_METRICS}


def delta_metrics(current: Block, previous: Optional[Block], block_number: int) -> Metrics:
    return {
        column: stat(current, source, block_number)
        - (stat(previous, source, block_number - 1) if previous is not None else Decimal(0))
        for column, source, _ in ADD_METRICS
    }


def add_metrics(left: Metrics, right: Metrics) -> Metrics:
    return {column: left[column] + right[column] for column, _, _ in ADD_METRICS}


def sum_metrics(intervals: Sequence[Metrics]) -> Metrics:
    total = zero_metrics()
    for interval in intervals:
        total = add_metrics(total, interval)
    return total


def dumps_needed(exp: Experiment) -> int:
    if exp.is_dense and exp.n_learners > 1 and not exp.scale_first_learner:
        # DENSE_NO_SIMD_BASELINE runs the learners one after another inside a
        # single stats window, so each learner contributes six dumps.
        return len(REGIONS) * exp.n_learners
    return len(REGIONS)


def aggregate_intervals(
    exp: Experiment, blocks: List[Block], first_block_number: int
) -> Tuple[List[Metrics], Metrics, Block]:
    """Return (six interval metrics, final_total metrics, block for BTBHitRatio)."""
    per_dump: List[Metrics] = []
    previous: Optional[Block] = None
    for offset, current in enumerate(blocks):
        per_dump.append(delta_metrics(current, previous, first_block_number + offset))
        previous = current

    if len(blocks) == len(REGIONS):
        intervals = per_dump
        if exp.scale_first_learner:
            intervals = [
                {column: metrics[column] * exp.n_learners for column in metrics}
                for metrics in intervals
            ]
            return intervals, sum_metrics(intervals), blocks[-1]
        final_block = blocks[-1]
        final = {
            column: stat(final_block, source, first_block_number + len(blocks) - 1)
            for column, source, _ in ADD_METRICS
        }
        return intervals, final, final_block

    # Dense multi-learner: sum the same region across all learner chunks.
    intervals = [zero_metrics() for _ in REGIONS]
    for dump_offset, metrics in enumerate(per_dump):
        region = dump_offset % len(REGIONS)
        intervals[region] = add_metrics(intervals[region], metrics)
    return intervals, sum_metrics(intervals), blocks[-1]


# --------------------------------------------------------------------------
# Row formatting
# --------------------------------------------------------------------------


def fmt_metric(number: Decimal, metric_type: str) -> str:
    if metric_type == "seconds":
        return format(number.quantize(SECONDS_QUANTUM, rounding=ROUND_HALF_UP), "f")
    return str(int(number))


def ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    return Decimal(0) if denominator == 0 else numerator / denominator


def fmt_ratio(number: Decimal, places: int) -> str:
    return format(number.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), "f")


def base_row(exp: Experiment, row_index: int, row_kind: str, dump_index: int) -> Dict[str, str]:
    checkpoint, interval = REGIONS[dump_index - 1]
    row = {column: "" for column in OUTPUT_COLUMNS}
    row.update(
        {
            "exp_id": exp.exp_id,
            "study": exp.study,
            "model": exp.model,
            "d_q": str(exp.d_q),
            "d_seq": str(exp.d_seq),
            "d_model": str(exp.d_model),
            "num_head": str(exp.num_head),
            "d_ff": str(exp.d_ff),
            "implementation": exp.implementation,
            "n_learners": str(exp.n_learners),
            "codebook_size": exp.codebook_size,
            "sve_bits": str(exp.sve_bits),
            "executable": exp.executable,
            "gem5_timestamp": exp.gem5_timestamp,
            "stats_file": exp.stats_file,
            "row_index": str(row_index),
            "row_kind": row_kind,
            "dump_index": str(dump_index),
            "checkpoint": checkpoint,
            "interval": interval,
        }
    )
    return row


def build_rows(exp: Experiment, all_blocks: List[Block], start_block: int) -> List[Dict[str, str]]:
    if start_block < 1 or start_block > max(len(all_blocks), 1):
        raise ExtractionError(
            f"--start-block {start_block} is outside the {len(all_blocks)} stats blocks"
        )
    needed = dumps_needed(exp)
    available = all_blocks[start_block - 1:]
    if len(available) < needed:
        raise ExtractionError(
            f"found {len(available)} stats block(s) from block {start_block}, need {needed} "
            f"({len(REGIONS)} dumps x {needed // len(REGIONS)} learner chunk(s)). Did the run "
            f"finish, and was it compiled with GEM5_PROFILE_REGIONS_FLAG=1?"
            + (" For an unfinished dense multi-learner run see --scale-first-learner."
               if exp.is_dense and exp.n_learners > 1 and not exp.scale_first_learner else "")
        )
    blocks = [dict(block) for block in available[:needed]]
    ignored = len(available) - needed
    if ignored:
        reason = (
            "gem5 adds one when the simulation exits" if ignored == 1
            else "if the file holds several program runs, choose one with --start-block"
        )
        print(
            f"note: using stats blocks {start_block}-{start_block + needed - 1}; ignoring "
            f"{ignored} later block(s) ({reason})",
            file=sys.stderr,
        )

    repair_sim_seconds(blocks, start_block)
    check_single_stats_window(blocks, start_block)
    intervals, final, ratio_block = aggregate_intervals(exp, blocks, start_block)

    rows: List[Dict[str, str]] = []
    for index, metrics in enumerate(intervals, start=1):
        row = base_row(exp, index, "interval_delta", index)
        for column, _, metric_type in ADD_METRICS:
            row[column] = fmt_metric(metrics[column], metric_type)
            if metrics[column] < 0:
                print(
                    f"warning: negative {column} in interval {REGIONS[index - 1][1]}",
                    file=sys.stderr,
                )
        rows.append(row)

    total = base_row(exp, len(REGIONS) + 1, "final_total", len(REGIONS))
    total["interval"] = "total_program"
    for column, _, metric_type in ADD_METRICS:
        total[column] = fmt_metric(final[column], metric_type)
    total["ipc"] = fmt_ratio(ratio(final["instructions"], final["cpu_cycles"]), 6)
    total["cpi"] = fmt_ratio(ratio(final["cpu_cycles"], final["instructions"]), 6)
    total["dcache_demand_miss_rate"] = fmt_ratio(
        ratio(final["dcache_demand_misses"], final["dcache_demand_accesses"]), 12
    )
    total["icache_demand_miss_rate"] = fmt_ratio(
        ratio(final["icache_demand_misses"], final["icache_demand_accesses"]), 12
    )
    total["l2_demand_miss_rate"] = fmt_ratio(
        ratio(final["l2_demand_misses"], final["l2_demand_accesses"]), 12
    )
    total["branch_misprediction_rate"] = fmt_ratio(
        ratio(final["branch_mispredictions"], final["committed_branches"]), 12
    )
    total["btb_hit_ratio"] = fmt_ratio(
        stat(ratio_block, BTB_HIT_RATIO_STAT, start_block + needed - 1), 6
    )
    rows.append(total)
    return rows


# --------------------------------------------------------------------------
# TSV input/output
# --------------------------------------------------------------------------


def read_tsv(path: Path, expected_columns: List[str]) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_columns:
            raise ExtractionError(f"{path} does not have the expected columns")
        return list(reader)


def write_tsv(path: Path, columns: List[str], rows: List[Dict[str, str]]) -> None:
    """Write like extract_final_stats.py, replacing the file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.chmod(tmp_name, mode)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def merge_rows(
    existing: List[Dict[str, str]], exp_id: str, new_rows: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    kept = [row for row in existing if row["exp_id"] != exp_id]
    # A stable sort keeps the row order inside each experiment.
    return sorted(kept + new_rows, key=lambda row: exp_number(row["exp_id"]))


def add_experiment(
    exp: Experiment, rows: List[Dict[str, str]], stats_blocks: int, output_root: Path, replace: bool
) -> Path:
    manifest_path = output_root / "manifest.tsv"
    combined_path = output_root / "final_all_experiments.tsv"
    manifest = read_tsv(manifest_path, MANIFEST_COLUMNS) if manifest_path.exists() else []
    combined = read_tsv(combined_path, OUTPUT_COLUMNS) if combined_path.exists() else []

    previous = [row for row in manifest if row["exp_id"] == exp.exp_id]
    if previous and not replace:
        raise ExtractionError(
            f"{exp.exp_id} already exists in {manifest_path}; pass --replace to overwrite it"
        )

    output_path = output_root.resolve() / exp.output_name
    write_tsv(output_path, OUTPUT_COLUMNS, rows)

    # With --replace, drop the old per-experiment TSV if its name changed.
    for row in previous:
        old_path = output_root.resolve() / Path(row["output_file"]).name
        if old_path != output_path and old_path.exists() and old_path.name.startswith(
            f"e{exp.number:02d}_"
        ):
            old_path.unlink()

    manifest_row = {
        "exp_id": exp.exp_id,
        "study": exp.study,
        "implementation": exp.implementation,
        "n_learners": str(exp.n_learners),
        "codebook_size": exp.codebook_size,
        "sve_bits": str(exp.sve_bits),
        "gem5_timestamp": exp.gem5_timestamp,
        "stats_file": exp.stats_file,
        "output_file": str(output_path),
        "stats_blocks": str(stats_blocks),
        "rows_written": str(len(rows)),
    }
    write_tsv(combined_path, OUTPUT_COLUMNS, merge_rows(combined, exp.exp_id, rows))
    write_tsv(manifest_path, MANIFEST_COLUMNS, merge_rows(manifest, exp.exp_id, [manifest_row]))
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        exp = build_experiment(args)
        if not args.stats.is_file():
            raise ExtractionError(f"stats file not found: {args.stats}")
        blocks = parse_stats_blocks(args.stats)
        rows = build_rows(exp, blocks, args.start_block)

        if args.dry_run:
            writer = csv.DictWriter(
                sys.stdout, fieldnames=OUTPUT_COLUMNS, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            return 0

        output_path = add_experiment(exp, rows, len(blocks), args.output_root, args.replace)
    except ExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = rows[-1]
    print(f"{exp.exp_id}: wrote {output_path}")
    print(f"  total sim_seconds={total['sim_seconds']} ipc={total['ipc']} cpi={total['cpi']}")
    print(f"  updated {args.output_root / 'manifest.tsv'} and final_all_experiments.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
