# CB4 baseline optimization and hierarchical cache study

This study uses fixed result IDs E01–E09, runner row 38, two learners, four
codebook entries, shared 2-bit indices and SVE128. One formal simulation per
ID; no medians or performance retries. The frozen controller stays detached;
`--at` chooses the measured source. Old CB4/CB16 controllers are not modified.

## Sources and order

E01/E02 are historical source controls; E03 is B0 at
`33d76aa29d8e199caee430f45ae59ae1ae16fd3b`. E04 contains NT01–NT05; E05 adds
row-first traversal. These five belong to `cb4-baseline-optimization`.
Only after reviewing all five results and approving the E05 gate may E06–E09
be implemented and run in `cb4-hierarchical-cache-tiling`. These are L1
compressed, L1 decoded, L1/L2 compressed and L1/L2 decoded variants. There is
no multi-output register blocking or extra H00/Q00/J00 experiment.

`~/I2CE_CB4_cache_study/action_map.json` holds the full local/server SHAs,
source trees and baselines. Empty E06–E09 slots are permitted before their
implementation. `verify` refuses source changes after preparation/submission.

## Server setup

Use a clean detached worktree `~/i2ce/I2CE_cb4_cache_controller` at the controller
commit. Configure its ignored `tools/exp/runner.conf` with REPO_ROOT pointing
at that worktree and EXP_ROOT at `~/i2ce/exp-cb4-cache-study`. Keep the previous
CB4 machine/compiler configuration and set `MAX_PARALLEL=auto`, `USE_LIBM5=1`.
Copy the preserved CB4 hardware and actual-machine fingerprints into
`~/i2ce/cb4-cache-study/expected-hardware.json` and `expected-machine.json`.
Copy the existing boot checkpoint into the new experiment root. Do not boot
another checkpoint: every comparison checks identical checkpoint contents.
Then commit/freeze the controller and run:

```bash
cd ~/i2ce/I2CE_cb4_cache_controller/tools/exp
python3 -B cache_study.py verify --manifest ~/I2CE_CB4_cache_study/action_map.json
python3 -B cache_worker.py baseline
```

The worker first prepares immutable libm5 builds and generated data, starting
with E03. QEMU validates a separate source copy against all 84 E03 pipeline
output tensors. Only the validation copy's weights path and profiling flags
change. E01/E02 are not presumed numerically equivalent: a discrepancy blocks
submission for investigation. QEMU wall time is not a performance result.
All five must pass validation before the worker submits any formal run.

The worker serializes builds/submissions; simulations execute concurrently.
Resource admission considers physical cores, CPU affinity/cgroup quotas,
host load, available host RAM and observed RSS. It has no fixed limit of 8.
Single-core GEMM configuration is unchanged by host scheduling.

The worker has a shared lock and finite 72-hour collector deadline. A launch
receipt and action marker prevent duplicate submission. On interruption,
inspect the worker/processes first; `recover E01` records an existing reserved
run and never launches a second one. Failed evidence must be preserved.

## Monitoring and collection

```bash
cd ~/i2ce/I2CE_cb4_cache_controller/tools/exp
python3 -B cache_study.py status --all
# The worker normally collects. Only run these manually when its collector is idle:
python3 -B cache_study.py collect --finished
python3 -B cache_study.py compare --all
python3 -B cache_study.py export --out ~/i2ce/cb4-cache-study/exports
```

Collection uses a lock and explicit `--run`, `--study`, `--exp-id` and
`--output-root`. Results in separate study namespaces never replace legacy
root E01–E11, now preserved in `transformer_profiling/archives/legacy-hsylin`.
Collector routing options are parsed once, avoiding a second hardcoded study
argument. Wrong source, generated data, compiler, machine, binary or checkpoint
fingerprints reject comparisons. Simulated ROI seconds are compared;
`host_elapsed_seconds_approx` is only a log-timestamp wall-time estimate and
is unsuitable for kernel speedup conclusions.

## Gate and hierarchy

The reviewer writes `baseline-gate.json` with `approved: true`, full
`E05_server_sha`, `E05_result_sha256`, and the documented correctness/performance
rationale after all E01–E05 finish. Register new E06–E09 sources with `verify`
and run `python3 -B cache_worker.py hierarchy`. Do not modify the frozen helper.
The E05 gate is checked before hierarchy builds, validation and submission.

## Recompare an export without running experiments

```bash
scp -r hlin@eslsrv12:~/i2ce/cb4-cache-study/exports ~/Downloads/I2CE_CB4_cache_results
cd ~/Downloads/I2CE_CB4_cache_results
shasum -a 256 -c SHA256SUMS
python3 -B tools/cache_study.py compare --all --exports "$PWD" \
  --out ~/Downloads/I2CE_CB4_cache_comparisons
```

This is read-only with respect to recorded evidence. Source patches, bundles,
exact ordered patch application baselines and correctness logs are delivered
alongside the export. No source branch is automatically merged or deleted by
these helpers.
