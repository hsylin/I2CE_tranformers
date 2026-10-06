# E02: Historical in-place int8 attention

Revision: original. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e02` / `f126e2b612666c9cc85d07073e2ac59424bad462`.
- Source tree: `e0bc9202f53462ad54ee77fec7cddf58c74317d8`; original local SHA: `f126e2b612666c9cc85d07073e2ac59424bad462`.
- Frozen controller: `0fe7aaeee1c44ba6f32909ab967049920403f420`.
- Performance comparison baseline: E01; patch application baseline: `None`.
- Whole simulated ROI: 2.523009 s; host elapsed approximately 16487.96 s (not a speedup metric).
- Original table: [TSV](../../e02_cb4_baseline_optimization_n2_cb4_sve128_f126e2b6_run_20261005_211926.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
