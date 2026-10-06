# E01: Historical pre-int8 control

Revision: original. Single formal run; no medians.

- Measured source: `exp/cb4-cache/e01` / `eb4a8e4fa59109b64f3bb1aafec3c0cda3c77fff`.
- Source tree: `44d9b303e1bb14a97f48be83bc7d93a3e2861665`; original local SHA: `eb4a8e4fa59109b64f3bb1aafec3c0cda3c77fff`.
- Frozen controller: `0fe7aaeee1c44ba6f32909ab967049920403f420`.
- Performance comparison baseline: None; patch application baseline: `None`.
- Whole simulated ROI: 3.153500 s; host elapsed approximately 16362.30 s (not a speedup metric).
- Original table: [TSV](../../e01_cb4_baseline_optimization_n2_cb4_sve128_eb4a8e4f_run_20261005_211924.tsv).
- Correctness: original receipt records 84 pipeline tensors checked against E03. Kernel-specific limitations and reproduction levels are described in the [study guide](../../../reproduction/README.md).
- [Provenance](provenance.json), original run/launch/configuration receipts and losslessly compressed raw statistics are adjacent.

Historical branch names and absolute server paths in original receipts are evidence, not dependencies on live branches. Fetch the measured tag to recover source. Do not reuse the old formal slot.
