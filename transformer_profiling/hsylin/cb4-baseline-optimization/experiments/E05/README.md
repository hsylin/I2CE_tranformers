# E05: E04 plus row-first traversal

Revision: original. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e05` / `e92999ac319bd89b965f59e0a64c4dd93111606f`.
- Source tree: `db1117c7d07101941f5208a2c14164dd8086f51f`; original local SHA: `ce0670b04ca304946d9f75aee440cb0b9dc20852`.
- Frozen controller: `0fe7aaeee1c44ba6f32909ab967049920403f420`.
- Performance comparison baseline: E04; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.344066 s; host elapsed approximately 3727.39 s (not a speedup metric).
- Original table: [TSV](../../e05_cb4_baseline_optimization_n2_cb4_sve128_e92999ac_run_20261005_211934.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
