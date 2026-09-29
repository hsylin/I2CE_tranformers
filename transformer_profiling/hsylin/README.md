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
| `eNN_<study>_n<L>_cb<S>_sve<VL>_<commit>_run_<timestamp>.tsv` | One experiment: one `interval_delta` row per region plus one `final_total` row |
| `hsylin_all_experiments.tsv` | Every experiment's rows concatenated |
| `manifest.tsv` | One row per experiment: settings, source `stats.txt`, commit, binary hash |
| `provenance/ENN.tsv` | The run's `build_config.tsv` verbatim: compile flags, artifact hashes, the full gem5 command line |
| `metric_sources.tsv` | Which gem5 stat each column is read from |

## Regions

`final/` and `base_mini/` use six regions: `MHA`, `Projection`,
`non_GEMM_after_projection`, `FF1`, `FF2`, `non_GEMM_after_ff2`. Runs collected
here replace the single `MHA` row with five, because MHA mixed a codebook GEMM
with two dense ones and a change to either kernel could not be attributed:

| Region | What it measures |
| --- | --- |
| `MHA_QKV` | the three codebook Q/K/V projections |
| `MHA_QK` | Q·Kᵀ alone — the dense interleaved kernel, nothing else in the interval |
| `MHA_softmax` | softmax on its own |
| `MHA_SV` | V transpose, the dense softmax(QKᵀ)·V matmul, post-softmax rescale, per-head gather |
| `MHA_out` | the multihead repack after the head loop |

The first four are closed once per attention head, inside
`SingleHeadSelfAttn::computeInterleaved{2,4}Learners` and the block's head loop,
so a run emits `4 x num_heads + 6` stats dumps. Dumps sharing a region name are
summed, which is why the table still has one row per region whatever the head
count. `MHA_SV` has to close after the gather rather than at the end of
`computeInterleaved*()`, or the gather would land in the next head's `MHA_QKV`.

Each interval is the current cumulative gem5 dump minus the previous one; the
math and number formatting are the same as the E01-E36 extraction, which
`tests/profiling_add_experiment_test.py` checks by reproducing those rows.

The sequence is not assumed. Every run writes `gem5_profile_regions.tsv` listing
the region each dump closed, `exp.sh collect` passes it to
`add_experiment.py --regions`, and the extractor reads the schema from it.

## The runs

Empty. The three-point int8 A/B that was here (E01-E03, baseline vs PR #14 vs
PR #15 at cb = 4, 2 learners, 128-bit SVE) was collected under the six-region
profiling schema and was dropped when the MHA region was split: its `MHA` row
has no counterpart in the new schema, so it cannot be compared against anything
measured from here on. Those rows remain in git history at `bc7e3b8b`.

Re-run it under the new schema with:

```bash
./exp.sh submit 38 --at 3c5501b4    # baseline
./exp.sh submit 38 --at fad41a0f    # PR #14, int8 activations in place
./exp.sh submit 38 --at e53973fa    # PR #15, svdot in the dense kernels
```

## Adding a run

```bash
bash tools/exp/exp.sh collect 38            # -> the next free ID here
python3 transformer_profiling/report.py     # refresh the HTML
```

IDs are allocated from this directory's `manifest.tsv`, which is empty, so the
next run is E01.
`add_experiment.py --output-root` can target another directory, which then
numbers into *that* directory's sequence. See `USER_MANUAL.md`, Section 4.8.
