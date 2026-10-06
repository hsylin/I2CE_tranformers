# Reproduction levels

## Recompute and verify saved results (no simulator)

From the repository root, with Python 3.9 or newer:

```bash
python3 transformer_profiling/hsylin/reproduction/verify_results.py
python3 -O transformer_profiling/hsylin/reproduction/verify_results.py
```

The verifier checks archive hashes, source/result/binary identities, all 84 saved tensor hashes, shared input/hardware/checkpoint fingerprints, normal exits and raw additive ROI counters. It prints whole/codebook runtime and the explicit comparison-baseline speedup. Checks use exceptions, not removable Python assertions. The original TSV collector rounds cumulative simulated seconds to six decimal places before taking phase differences; raw ticks are preserved for higher-resolution analyses.

To use the existing interactive report (its pandas/plotly dependencies are separate):

```bash
python3 transformer_profiling/report.py --input transformer_profiling/hsylin/comparisons/canonical_all_experiments.tsv --output /tmp/cb4-cache-report.html
```

## Recover exact source

Keep the publication checkout for these documents and scripts. Use a separate worktree for source:

```bash
git fetch --tags
git worktree add --detach ../cb4-e09-source exp/cb4-cache/e09-v2
git -C ../cb4-e09-source rev-parse HEAD
git -C ../cb4-e09-source rev-parse 'HEAD^{tree}'
```

Expected E09 source: `fbc2d5fb6d94d67034c160108b408c5be85e05df`; tree: `6506596330d22b5eec9865df7fc27705eb8c10ba`.
Every experiment's manifest contains its own tag/tree and ordered patches from the stated patch baseline. Tags preserve original commit identities; applying patches can produce different commit IDs even with identical trees.

With an AArch64 SVE cross compiler and QEMU user emulator installed, E06–E09 offer correctness-only tests:

```bash
A64CXX=/absolute/path/to/aarch64-linux-gnu-g++ I2CE_QEMU=/absolute/path/to/qemu-aarch64 bash ../cb4-e09-source/tests/run_cb4_cache_test.sh
```

SVE128 runs the cache kernel. Wider SVE and incompatible CB/learner variants test rejection; they are not claims of generalized cache support. The guard-page test intentionally faults a child process. No simulation timing is collected.

## Repeat a full simulation (not validated as a public turnkey workflow)

Frozen controller tags are `exp/cb4-cache/controller-baseline` and `exp/cb4-cache/controller-packing-v2`. Original `run.json`, `launch.json` and `build_config.tsv` describe commands, generated inputs, compiler flags, machine and checkpoint identities. Original source/controller commits and completed run slots must remain unchanged.

Full execution still requires the matching gem5 build/config, guest disk/kernel, checkpoint, generated headers/weights and cross-toolchain. Large binaries/checkpoints are intentionally not in this Git publication; their hashes are in the receipts, and the author retains them in the original study packages. Public retrieval of those external assets is not yet packaged. A hash or historical private server path is not a download link. Consequently this publication verifies report reproduction and source recovery, not an independently executed full simulator rerun.

Any future rerun needs its own namespace/output directory and run receipt. It must not overwrite E01–E09 or turn revisions into repeated trials. No new performance runs were launched for this publication.
