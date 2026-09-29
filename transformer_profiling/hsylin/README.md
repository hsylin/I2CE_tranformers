# Current profiling runs

gem5 profiling runs for the int8 / SVE performance work on the codebook GEMM
path. This is the directory `tools/exp/exp.sh collect`, `add_experiment.py` and
`report.py` all use by default.

`transformer_profiling/final/` is a **separate, closed** dataset: the previous
student's E01-E36 extraction for BERT-mini and BERT-base. It numbers from E01
as well, so an experiment ID only identifies a run together with its directory.
Cite runs from here as `hsylin/E01`, and historical ones as `final/E09`.

## Contents

| File | What it is |
| --- | --- |
| `eNN_<study>_n<L>_cb<S>_sve<VL>_<commit>_run_<timestamp>.tsv` | One experiment: six `interval_delta` rows and one `final_total` row |
| `hsylin_all_experiments.tsv` | Every experiment's rows concatenated |
| `manifest.tsv` | One row per experiment: settings, source `stats.txt`, commit, binary hash |
| `provenance/ENN.tsv` | The run's `build_config.tsv` verbatim: compile flags, artifact hashes, the full gem5 command line |
| `metric_sources.tsv` | Which gem5 stat each column is read from |

The interval rows use the same six stages as `final/` and `base_mini/`: `MHA`,
`Projection`, `non_GEMM_after_projection`, `FF1`, `FF2`, `non_GEMM_after_ff2`.
Each interval is the current cumulative gem5 dump minus the previous one; the
math and number formatting are the same as the E01-E36 extraction, which
`tests/profiling_add_experiment_test.py` checks by reproducing those rows.

## The runs

E01-E03 are a three-point A/B of the two int8 kernel changes, all at
cb = 4, 2 learners, 128-bit SVE, restored from the same checkpoint, with only
the measured binary differing (`exp.sh submit 38 --at <commit>`):

| ID | Commit | Change measured |
| --- | --- | --- |
| E01 | `3c5501b4` | baseline (PR #12, tooling only) |
| E02 | `fad41a0f` | PR #14 — read interleaved int8 activations in place |
| E03 | `e53973fa` | PR #15 — `svdot_s32` in the dense interleaved kernels |

Note that `provenance/ENN.tsv` has an `exp_id` key holding the **runner** ID
from `tools/exp/experiments.tsv` (38 here), not the `ENN` of this directory;
the `ENN` is in the filename and in `manifest.tsv`'s `runner_id` column.

## Adding a run

```bash
bash tools/exp/exp.sh collect 38            # -> the next free ID here
python3 transformer_profiling/report.py     # refresh the HTML
```

IDs are allocated from this directory's `manifest.tsv`, so the next run is E04.
`add_experiment.py --output-root` can target another directory, which then
numbers into *that* directory's sequence. See `USER_MANUAL.md`, Section 4.8.
