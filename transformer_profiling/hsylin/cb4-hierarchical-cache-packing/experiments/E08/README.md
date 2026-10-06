# E08: L1/L2 compressed-index cache tiles

Revision: packing-v2. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e08-v2` / `0f1184b6a44dd979fd4fb2f8075ab6b70822187d`.
- Source tree: `35845207646ac19057d000dfbeeaf91d9aecd2e5`; original local SHA: `7d0d2c29445cadec2ade2f766dbe35099692c8e1`.
- Frozen controller: `3346892c009debc2e885646cab26c331b27b75a3`.
- Performance comparison baseline: E06; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.315610 s; host elapsed approximately 3932.83 s (not a speedup metric).
- Original table: [TSV](../../e08_cb4_hierarchical_cache_packing_n2_cb4_sve128_0f1184b6_run_20261006_075826.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
