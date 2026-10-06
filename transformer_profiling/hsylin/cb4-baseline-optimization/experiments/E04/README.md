# E04: NT01-NT05, output-first traversal

Revision: original. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e04` / `e62309bca2c25bfca4e1c4e31365672989efcf3f`.
- Source tree: `5dc2f2763cbdd475e6f9d3267900300e6539d1e1`; original local SHA: `d4e76060e38c7ff61b8322f86ab25f54fd9a26d0`.
- Frozen controller: `0fe7aaeee1c44ba6f32909ab967049920403f420`.
- Performance comparison baseline: E03; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.771601 s; host elapsed approximately 3903.02 s (not a speedup metric).
- Original table: [TSV](../../e04_cb4_baseline_optimization_n2_cb4_sve128_e62309bc_run_20261005_211932.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
