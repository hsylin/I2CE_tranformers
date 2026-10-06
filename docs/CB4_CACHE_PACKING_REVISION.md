# CB4 single-output cache packing revision

Reuse preserved E01–E05, including their original controller fingerprints.
Run E06–E09 exactly once each in an independent state/experiment root and the
cb4-hierarchical-cache-packing result namespace. Never submit baseline repeats.
Original P03 and all original run receipts remain unchanged.

The only supported cross-controller comparisons explicitly pair the preserved
baseline controller with this frozen revision controller; machine, compiler,
generated data, actual cache configuration, checkpoint and ROI must match.
The existing-reference controller is exported as reference-controller.json.

Before submission, packing-ready.json must bind each source and production
binary to the manual correctness/assembly gate. Correctness-only tests do not
constitute another performance run.

Kernel scope: SIMD activation/weight/index/accumulator panel transfers;
zero only the final partially used SDOT vector; retain allocated strides and
all tile/traversal/representation choices. Keep the single-output kernel.
Inspect table stack traffic in production assembly without claiming that
source-level hoisting guarantees spill elimination.

Promotion requires correct output, verified provenance, lower whole-model
simulated runtime and lower combined codebook-phase runtime versus that
same action's original run. Preserve both versions and report phase changes;
no medians, repeats, automatic tuning, PRs or merges.
