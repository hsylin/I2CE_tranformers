# CB4 E01–E09: preserved single-run results

**Selected maintenance and parameterization baseline: E09 packing-v2.** Full staging is retained to study cache-tile geometry. Other measured variants remain source tags and comparison controls; their branches need not remain active. Integration or later correctness fixes are not new performance measurements of these tags.

## Quick navigation

- [E01–E05 baseline optimization](cb4-baseline-optimization/README.md).
- [E06–E09 cache tiling, selected packing-v2](cb4-hierarchical-cache-tiling/README.md).
- [Reproduce tables, recover source and understand full-rerun limitations](reproduction/README.md).
- [Machine-readable source/controller/run/patch manifest](manifest.json).
- [Per-phase metrics](comparisons/phase_metrics.tsv) and [explicit comparisons](comparisons/aggregate_comparisons.tsv).
- The unrelated legacy E01–E11 dataset remains preserved in [its archive](../archives/legacy-hsylin/).

## Measurements

CB4, two learners, shared 2-bit indices, SVE128, one MinorCPU at 4 GHz, L1D 32 KiB/2-way, L2 1 MiB/16-way, 64-byte lines. BERT-mini dimensions: S=512, model=256, head=64, four heads, FF=1024. One formal run per source; no medians. Times below are simulated ROI seconds, not host elapsed time.

| ID | Action | Revision | Whole s | Codebook s | Measured source tag |
|---|---|---|---:|---:|---|
| E01 | Historical pre-int8 control | original | 3.153500 | 2.855031 | [exp/cb4-cache/e01](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e01) |
| E02 | Historical in-place int8 attention | original | 2.523009 | 2.121127 | [exp/cb4-cache/e02](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e02) |
| E03 | SDOT B0 reference | original | 1.167828 | 0.957672 | [exp/cb4-cache/e03](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e03) |
| E04 | NT01-NT05, output-first traversal | original | 0.771601 | 0.686740 | [exp/cb4-cache/e04](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e04) |
| E05 | E04 plus row-first traversal | original | 0.344066 | 0.259626 | [exp/cb4-cache/e05](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e05) |
| E06 | L1 compressed-index cache tile | packing-v2 | 0.310451 | 0.224516 | [exp/cb4-cache/e06-v2](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e06-v2) |
| E07 | L1 decoded weight tile | packing-v2 | 0.214704 | 0.122916 | [exp/cb4-cache/e07-v2](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e07-v2) |
| E08 | L1/L2 compressed-index cache tiles | packing-v2 | 0.315610 | 0.226315 | [exp/cb4-cache/e08-v2](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e08-v2) |
| E09 | L1/L2 decoded weight tiles | packing-v2 | 0.198159 | 0.122932 | [exp/cb4-cache/e09-v2](https://github.com/hsylin/I2CE_tranformers/tree/exp/cb4-cache/e09-v2) |

## What each comparison isolates

- E01→E02→E03: historical implementation evolution, not a single-instruction ablation.
- E03→E04: combined non-tiling cleanups; E04→E05: row-first traversal.
- E05→E06 v2: full L1 compressed strategy including packing costs.
- E06 v2→E07 v2: compressed versus decoded L1 strategy.
- E06 v2→E08 v2 and E07 v2→E09 v2: adding L2 within each representation lineage.
- E08 v2→E09 v2: compressed versus decoded two-level strategy.

E09 v2's whole ROI is 0.198159 s versus E07 v2's 0.214704 s, but their codebook sums are 0.122932 s and 0.122916 s respectively. Most of the whole-model difference is in unchanged QK. It is not evidence of a 7.7% improvement in codebook GEMM; cache/layout explanations remain hypotheses. A single fixed configuration does not establish the best tile sizes for other hardware or shapes.

## Preservation and scope

Measured source tags and frozen controller tags remain fixed. Original result TSVs and receipts are byte-preserved; stats are losslessly gzip-compressed. Branch names in receipts describe history. Original v1 cache results remain locally archived and are not combined with v2. Later retired adaptive experiments remain local-only by user decision and are outside this publication.

Each new source/configuration needs a new identity. Source recovery, report reconstruction and full performance reruns are distinct claims. See the reproduction guide before attempting execution.

The root `hsylin_all_experiments.tsv` is a generated nine-experiment summary for the existing report reader. Individual measured TSVs remain in their original study namespaces; no root result IDs are allocated.
