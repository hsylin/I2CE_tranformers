# E06: L1 compressed-index cache tile

Revision: packing-v2. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e06-v2` / `6b05de1c6b48305a2a3fd93c8fdac23ee3cdcb73`.
- Source tree: `419316066cab24236c0c0a75f139fe6bc1c084d6`; original local SHA: `8032b1322fe076f9564e65e7e8f6b3e45f8b8070`.
- Frozen controller: `3346892c009debc2e885646cab26c331b27b75a3`.
- Performance comparison baseline: E05; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.310451 s; host elapsed approximately 3830.01 s (not a speedup metric).
- Original table: [TSV](../../e06_cb4_hierarchical_cache_packing_n2_cb4_sve128_6b05de1c_run_20261006_075821.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
