# E07: L1 decoded weight tile

Revision: packing-v2. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e07-v2` / `4a90231e8762b65521b7f8b01f350bc3774fbeca`.
- Source tree: `7ef6f698174634f238348f3659cfc445c14adeb7`; original local SHA: `7249071b19a3baf163efb7ee9f40120dabe3697b`.
- Frozen controller: `3346892c009debc2e885646cab26c331b27b75a3`.
- Performance comparison baseline: E06; patch application baseline: `33d76aa29d8e199caee430f45ae59ae1ae16fd3b`.
- Whole simulated ROI: 0.214704 s; host elapsed approximately 3289.45 s (not a speedup metric).
- Original table: [TSV](../../e07_cb4_hierarchical_cache_packing_n2_cb4_sve128_4a90231e_run_20261006_075824.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
