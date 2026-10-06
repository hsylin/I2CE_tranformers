# CB4 integration and parameter-interface validation

This record covers integration into `submission`, not a new optimization study.
The measured E09 packing-v2 source remains
`fbc2d5fb6d94d67034c160108b408c5be85e05df` at `exp/cb4-cache/e09-v2`.

## Preservation and integration

- PR #18 integrated the selected E09 lineage without replacing submission's tooling/history.
- PR #19 published all nine results, original namespaces, raw compressed stats, manifests, source/patch mappings and reproduction instructions.
- Nine source tags and two frozen-controller tags were restored from a fresh network clone and checked for exact commit/tree identity before the completed experiment branches were deleted.
- Local full Git bundles and the original study packages remain preserved. Historical controller/worktree/run directories were not changed.
- The original measured results retain their original source SHAs; they are not relabelled as merge-commit measurements.

## New source changes

| Commit | Responsibility |
|---|---|
| `bd4d9133db1df630e5dee761208fb59d9aa6338e` | Defined modulo-2^32 bias/output accumulation in relevant two-learner C32 SVE/scalar fallback paths, with forced-path tests |
| `3c01614b636c550ca8a9194cd0d7f000521bd6db` | Replaced nested scratch arrays with flat storage, preserving bytes, alignment, field order/offsets and staging |
| `1004afabd823ed25d8d19369db0f18d1c34d03a5` | Compile-time tile contract, compile/runner plumbing, boundary/full-C32/consumer and invalid-configuration tests |

All three commits have hsylin as author and committer. The fallback correction is
scoped to the relevant two-learner shared-index paths, not every legacy GEMM API.
The parameter interface uses the E09 default geometry and preserves X1/C1 staging.

## Manual evidence

Toolchain: conda-forge AArch64 GCC 13.4.0-19; QEMU AArch64 10.0.0.

| Check | Outcome |
|---|---|
| Forced C32 SDOT/widening/scalar paths, nonzero products, extreme bias/output, all 32 bits | 56 row cases plus wrapper checks per SVE128/256/512; PASS with signed-overflow sanitizer |
| Same new C32 test against original E09 | Correctly detects signed overflow; preserved negative evidence |
| Default cache oracle and unsupported dispatch controls | PASS; bounds, poisoned tails, aliases, pre-narrowing rejection, allocation failure |
| Parameter configurations | Default 16/32/128 : 128/128/512; small 8/16/64 : 64/64/256; uneven 12/20/48 : 28/36/80 |
| Scalar oracle for those configurations | 184 / 188 / 188 cases; full C2 bits checked before low8 output, plus guard-page cases |
| Consumer integration | All three configurations enter the cache path; allocation/alias/shared-mode/dimension fallback/rejection checks PASS |
| Illegal configurations | Eight illegal geometries and three incompatible generated CB/index/SVE configurations rejected |
| Build/runner interface | Four host tests PASS, including invalid shell values, disabled path, incompatible/ignored overrides and separate cache-size options |
| Full pipeline, final source above, E09 generated inputs | All 84 tensor SHA256 values match the preserved E09 reference |
| Archived result verifier | All nine runs: identities, shared fingerprints, exits, 84 saved tensor hashes and additive raw counters PASS, including Python `-O` |
| Corrupted-publication tests | Six negative tests PASS under Python `-O` |
| Existing HTML report | 26 report tests passed when the nine results were published |

Selected logs and the full-pipeline receipt are in
[`validation/cb4-cache-integration/`](../validation/cb4-cache-integration/).
The guard-page test intentionally faults a child and checks its SIGSEGV; this
message is not a failed kernel test. Pipeline validation changes only the copied
weight path and disables profiling; it is a correctness binary, not the measured
production binary. Generated inputs were hash-checked before reuse.

## Assembly and performance boundary

At `-std=c++17 -O2 -march=armv8-a+sve -ffunction-sections`, with the test CB4
header and no test instrumentation, extracted relocatable
`.text.sve_gemm_cb4_cache_i8` section hashes were:

| Source | SHA256 |
|---|---|
| Measured E09 / C32-only correction | `e9e9f801406f8e3ebf7077b09b169f5c1230198fcff44d4c0fae0899cc4fc280` |
| Flat-buffer correction / default parameter interface | `e3e6683cda22bb6f131e22c4dd098b2e8a3ffb5003b3af67863775a1e4385083` |

Thus the buffer refactor changed code generation; adding default tile parameters
introduced no further change in this checked section. This comparison does not
assert identical linked executables, layouts or whole-model timings.
The reviewed inner K loops retain four operand loads, two SDOTs, an increment,
comparison and branch per 16-K chunk, with no stack access in that loop. Register
allocation and surrounding address/control instructions differ. This is not a
blanket assertion of zero stack traffic in the full function.

No new gem5 performance run was submitted. Before comparing new tile timings,
measure the corrected default once in a fresh namespace; retain original E09
as historical evidence. Do not infer that old timing is the new source's timing.

## Deliberately deferred

CI workflows, branch protection, Releases/DOI, automatic tile selection,
additional optimization variants and arbitrary SVE/multicore support. Existing
hardware size controls need actual-config/provenance verification in future runs.
Large external simulator/data assets are not yet a public turnkey rerun package;
see the archived result [reproduction notes](../transformer_profiling/hsylin/reproduction/README.md).
