# Final BERT profiling TSVs

This directory contains the E01-E36 profiling extraction for BERT-mini and BERT-base.

Each per-experiment TSV has seven rows: six `interval_delta` rows and one `final_total` row. The interval rows use the same stage separation as `base_mini`: `MHA`, `Projection`, `non_GEMM_after_projection`, `FF1`, `FF2`, and `non_GEMM_after_ff2`.

For normal six-dump files, each interval is the current cumulative gem5 dump minus the previous cumulative dump. Complete dense multi-learner baseline files are aggregated by summing the same stage across each six-dump learner chunk. BERT-base E05 and E06 had only ten dumps available, so their first completed learner chunk was multiplied by `n_learners` to estimate the whole dense baseline run.

The curated inputs were read from `/home/thu/gem5/transformer_profiling/BERT_mini` and `/home/thu/gem5/transformer_profiling/BERT_base`. The `stats_file` column keeps the prior `base_mini` convention of recording the corresponding `/home/thu/gem5/output/run_.../stats_...txt` path.

Use `final_all_experiments.tsv` for a combined table, or the `E*.tsv` files for per-experiment analysis. See `metric_sources.tsv` for source stat names.

## This directory is closed

E01-E36 is the completed historical extraction and nothing is appended to it. New gem5 runs go to `transformer_profiling/hsylin/`, which numbers from E01 independently; see [../hsylin/README.md](../hsylin/README.md). `add_experiment.py` and `tools/exp/exp.sh collect` both default there, and `report.py` reads it by default, so this directory is only loaded when `--input transformer_profiling/final` names it explicitly.

Because both directories start at E01, an experiment ID is only meaningful together with the directory it came from. Cite runs from here as `final/E09`, not `E09`.
