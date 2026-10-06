# E09: L1/L2 decoded weight tiles

Revision: packing-v2. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e09-v2` / `fbc2d5fb6d94d67034c160108b408c5be85e05df`.
- Source tree: `6506596330d22b5eec9865df7fc27705eb8c10ba`; original local SHA: `570248d190c6a2d6736774429e635af604e67d32`.
- Frozen controller: `3346892c009debc2e885646cab26c331b27b75a3`.
- Performance comparison baseline: E07; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.198159 s; host elapsed approximately 3316.73 s (not a speedup metric).
- Original table: [TSV](../../e09_cb4_hierarchical_cache_packing_n2_cb4_sve128_fbc2d5fb_run_20261006_075829.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
