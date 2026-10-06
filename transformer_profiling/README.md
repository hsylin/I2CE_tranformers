# Transformer profiling tools

## Python environment

The repository's `environment.yml` defines `gem5_env` with Python 3.9.23.
For an existing environment, install only the visualization dependencies:

```bash
source "$HOME/i2ce/miniforge3/etc/profile.d/conda.sh"
conda activate gem5_env
python -m pip install -r transformer_profiling/requirements-visualization.txt
python -c "import pandas, plotly; print(pandas.__version__, plotly.__version__)"
python -m pip check
```

The pins are pandas 2.2.3 and Plotly 6.3.0. Jupyter is optional.
`add_experiment.py` itself needs only the standard library; `report.py` needs
the visualization dependencies above.

## Generate or refresh the report

```bash
python3 transformer_profiling/report.py
```

The single script, `report.py`, writes
`transformer_profiling/reports/profiling_report.html`. The English page contains
charts and their controls, plus measured geometry in the cache-comparison tab:
no tutorials, warning banners or empty sections. It embeds Plotly.js and the data, so it opens offline. The
camera button exports individual charts to SVG. Generated reports are ignored
by Git.

Every invocation reads the current run directory, `hsylin/`. The historical
`final/` (E01-E36) and `base_mini/` datasets are **not** read: both number from
E01, so including them silently would collide. To report on one of them, name it
explicitly with `--input transformer_profiling/final`.

Within the selected directory:

- Files named `eNN_*.tsv`, `ENN_*.tsv`, `eNN.tsv`, or `ENN.tsv` contain one complete
  experiment each. The filename ID must match the rows' `exp_id`.
- Individual files override the same experiment's rows in the directory's
  combined table, `<dirname>_all_experiments.tsv` (so
  `hsylin/hsylin_all_experiments.tsv`). The combined table supplies any other
  experiments. This includes newly added files even before it is updated.
- Duplicate individual files for an ID are rejected. Invalid or incomplete
  experiments fail validation, leaving the previous HTML intact.
- Input TSVs are never modified. Manifests, metric mappings and `figures/` are
  not treated as experiments.

Adding further experiments does not require code edits. Use one
`interval_delta` row per profiling region plus one `final_total` row. Two region
schemas are recognised: the six regions every run before the MHA split used, and
the split schema below, where MHA becomes five rows. An experiment must match one
of them exactly; a set that matches neither is refused rather than charted. See [hsylin/README.md](hsylin/README.md) for extraction, and
[final/README.md](final/README.md) for how the historical dataset was produced.

Collection and reporting are separate steps. `exp.sh collect` writes the TSVs;
run the report afterwards:

```bash
bash tools/exp/exp.sh collect 38
python3 transformer_profiling/report.py
```

`collect --dry-run` writes nothing. A run collected with an explicit
`--output-root` needs the matching `report.py --input`; the default input is
`transformer_profiling/hsylin`.

For files copied or edited directly in `hsylin/`, regenerate the report, or
leave this command running to rebuild on changes (Ctrl-C stops it):

```bash
python3 transformer_profiling/report.py --watch
```

Watch mode observes local experiment and optional measurement files. It also
rebuilds on additions/deletions and retries incomplete writes without replacing
the previous valid report. It does not fetch GitHub changes or rerun gem5. Pull
remote results into this checkout before generating, and reload the HTML in the
browser afterward. An HTML previously downloaded to another computer remains a
snapshot and must be downloaded again.

Defaults are relative to the script, not your working directory. Reporting does
not need `runner.conf`, gem5, a guest image or a compiler.

```bash
python3 transformer_profiling/report.py --input /path/to/results_directory
python3 transformer_profiling/report.py --input transformer_profiling/final
python3 transformer_profiling/report.py --output "$HOME/profiling_report.html"
python3 transformer_profiling/report.py --experiments E01 E03
python3 transformer_profiling/report.py --metrics-output /tmp/profiling_metrics.tsv
python3 transformer_profiling/report.py --help
```

## Charts and measurement definitions

Each model has the following charts when the required observations exist:

| Chart | Controls |
| --- | --- |
| Experiment runtimes | Simulated time on a log scale; one point per experiment. |
| Phase breakdown | Every profiling region the data carries; seconds or percentage. |
| Phase metric heatmap | L1D/L2 miss rate, L1D/L2 MPKI, IPC, L2 miss count. |
| SVE sensitivity | CB8, grouped by learner count; time or ratio to 128-bit SVE. |
| Codebook x SVE grid | Four learners; runtime or L1D/L2 miss rate. |
| Runtime vs L2 misses | Demand miss count with cache metrics on hover. |

Time means gem5 simulated seconds, not host elapsed time. Miss rate is demand
misses / demand accesses; MPKI is misses x 1000 / instructions; IPC is
instructions / cycles. Ratios are calculated from interval counts, not by
subtracting cumulative ratios. Zero denominators remain missing. L2 demand
misses are not a measurement of DRAM bytes; `simOps` is not a FLOP count.

Non-GEMM 1 means `non_GEMM_after_projection`; Non-GEMM 2 means
`non_GEMM_after_ff2`.

The interleaved int8 paths split MHA into five regions, closed four times per
attention head and summed across heads:

| Region | What it measures |
| --- | --- |
| `MHA_QKV` | the three codebook Q/K/V projections — the only compressed-weight GEMM in the head |
| `MHA_QK` | Q·Kᵀ alone, the dense interleaved kernel with nothing else in the interval |
| `MHA_softmax` | softmax, measured on its own so it dilutes neither dense region |
| `MHA_SV` | the V transpose, the dense softmax(QKᵀ)·V matmul, the post-softmax rescale and the per-head gather |
| `MHA_out` | the multihead repack left over once the four per-head regions are subtracted |

`MHA_QKV` and `MHA_QK` are the point of the split: one is a codebook GEMM and
the other a dense GEMM, so a change to either kernel can be attributed to it.
Outside those two, the regions still do not isolate staging, decode, MAC,
barriers or reduction. Per-worker timelines require separate buffered guest events.

Repeated settings remain separate observations. SVE groups with duplicate
settings or different model dimensions have no inferred speedup baseline;
ambiguous grid cells show run counts and individual IDs/times on hover. Retain
run manifests to verify matching workload, kernel and architecture before
interpreting timing ratios. Learner count is not CPU core count.

Known historical source issues remain explicit without banners in the HTML:

- E02 has invalid MHA/Projection intervals that cancel in its total. Its phase
  bar is omitted; the original total and cache counters remain available, with
  the issue in runtime hover details and a terminal warning. No replacement
  timing is invented.
- E05/E06 are estimated dense BERT-base baselines, as documented in
  `final/README.md`. They use `(est.)` labels and open diamond runtime markers.
- File hashes, selected IDs, package versions, and the report code revision
  are embedded as non-visible JSON metadata. The report revision is not the
  revision used to build each measured experiment.

## Tile, multicore and implementation-phase measurements

Copy the header-only templates in `templates/` and populate real measurements.
Save them in `hsylin/` with the same filenames for automatic inclusion, or
select explicit files:

```bash
python3 transformer_profiling/report.py \
  --tile-results /path/to/tiles.csv \
  --scaling-results /path/to/scaling.csv \
  --phase-results /path/to/phases.csv
```

These capabilities remain in the script even before tiling or multicore results
exist. Only tabs with measurements are displayed; header-only templates create
no placeholder charts. All three CSVs are monitored by `report.py --watch`.

| Chart | Input file | Measurements |
| --- | --- | --- |
| Tile-size heatmap | `tile_results.csv` | Tile dimensions and simulated ROI time. |
| Cache heatmaps | `tile_results.csv` | L1D/L2 demand accesses and misses; switch the heatmap control. |
| Tile footprint vs runtime | `tile_results.csv` | Optional accounted `tile_footprint_bytes`; optional L2 capacity marker. |
| Multicore speedup | `scaling_results.csv` | Core count and simulated ROI time, including one core. |
| Parallel efficiency | `scaling_results.csv` | The same measured strong-scaling series. |
| DRAM bandwidth vs cores | `scaling_results.csv` | Optional measured `dram_bytes` over the timed ROI. |
| Implementation phase breakdown | `phase_results.csv` | Exclusive wall or per-worker phase durations and ROI duration. |

The CSVs supplement the historical experiment TSVs; they do not require the
TSV region schema and do not change it. The input directory still needs at
least one valid historical-schema experiment. The report does not perform tile
sweeps, instrument kernels or infer new measurements from old aggregate stats.
`--experiments` filters TSV charts and the cache-variant comparison; optional CSVs select
their own independent configurations. Use English configuration and phase names
to keep all chart labels in English.

### Cache tiling comparison: preserved CB4 E05–E09

The current `hsylin/` input includes `cache_comparison.json`, so the default
command adds a **Cache tiling comparison** tab. This compares E05 with the
packing-v2 E06–E09 implementations using their existing single formal runs.
It is separate from **Tile sweeps**: tile dimensions are fixed; representation
and hierarchy change. No repeats, medians, interpolated results or additional
simulations are used.

The tab contains:

- Whole-model and codebook-region simulated milliseconds (not host time).
- Explicit baseline/candidate speedups, separating compressed/decoded and
  L1/L2 comparisons. A ratio above 1 means the candidate is faster.
- All ten phase runtimes and instruction/L1D/L2 demand metrics. Codebook time
  is the sum of `MHA_QKV`, `Projection`, `FF1` and `FF2`. Unchanged QK and other
  phases remain visible; their changes are not attributed to cache GEMM.
- Measured `(S,O,K)` tiles, padded byte strides and allocated arena size.
  Arena size is neither the cache working set nor measured traffic.

For the canonical TSV, supply the metadata explicitly:

```bash
python3 transformer_profiling/report.py \
  --input transformer_profiling/hsylin/comparisons/canonical_all_experiments.tsv \
  --cache-comparison transformer_profiling/hsylin/cache_comparison.json \
  --output /tmp/cb4_profiling.html
```

The metadata is derived from the measured `exp/cb4-cache/e05` and
`exp/cb4-cache/e06-v2` through `e09-v2` source tags, not the later integrated
source. Each entry pins its full source SHA, binary SHA256, study and run ID.
The renderer checks every selected row against that identity and the common
workload/compiler/cache configuration, and records the metadata file hash in
the HTML provenance. Full raw-result/fingerprint verification remains available
through `python3 transformer_profiling/hsylin/reproduction/verify_results.py`.

The schema is intentionally small: `schema_version: 1`, a `configuration`
mapping, `variants` with source/run identity and geometry, and `comparisons`
with baseline/candidate IDs and labels. `--experiments` removes excluded
variants and comparisons needing an excluded endpoint. A metadata file beside
the selected input is discovered automatically and monitored by `--watch`.
Historical datasets receive no cache tab unless given matching metadata.

These results compare complete implementations, not separately timed packing,
decode or SDOT substeps. In particular E07→E09 improves whole-model time while
the summed codebook phases are nearly unchanged; inspect the phase chart before
attributing the improvement to L2 tiling.

### Tile sweeps

Tile CSV columns:

```text
config_id,tile_m,tile_n,tile_k,replicate,sim_seconds,l1d_accesses,l1d_misses,l2_accesses,l2_misses,tile_footprint_bytes,l2_capacity_bytes
```

Hold workload, input, layer/ROI, kernel, datatype, ensemble, SVE, core count,
weight-sharing policy and architecture fixed within `config_id`, and document
them in a run manifest. A heatmap is generated for each config_id/tile_k pair.
Cells show median runtime or median per-run miss rate across repetitions;
unmeasured cells stay blank. Sum participating private L1D counters when
appropriate, and count a shared L2 once.

`tile_footprint_bytes` and `l2_capacity_bytes` are optional. Omit them, or leave
an entire column blank, to retain just the heatmaps. If a column is populated,
supply it for every row. Footprint must be positive and identical across repeats
of a tile; L2 capacity must be positive and constant within `config_id`.

Account for simultaneously live inputs, outputs, packed indices, codebooks and
staging buffers, including the chosen sharing/replication policy. Do not silently
use the footprint of a dense weight matrix. This is an accounted working set,
not a measured reuse distance or proof of cache residency. The L2 marker denotes
configured nominal capacity; other traffic, cache associativity and TLB behavior
can also affect runtime. Record the accounting method in the run manifest.

### Multicore scaling

Scaling CSV columns:

```text
config_id,num_cores,replicate,sim_seconds,dram_bytes
```

Hold everything except core count fixed within `config_id`, including the same
algorithm and tile policy. Each series needs a measured one-core baseline.
Speedup is median T(1)/median T(p). Parallel efficiency has its own plot and is
100 x speedup/p; values above 100% are retained. Hover shows repeats and time
range. Use wall simulated ROI time rather than a sum of per-core cycles.

`dram_bytes` is optional. When supplied, it must be finite and nonnegative for
every row. It means measured read plus write traffic at the memory controller(s)
over the same timed ROI, summed across channels once. Bandwidth is computed for
each run as bytes/seconds/1e9, then the median is plotted in decimal GB/s. Do not
substitute L2 misses x line size, allocated tensor bytes or a host wall-clock
measurement. No DRAM peak bandwidth is assumed. A plateau can motivate further
checks; it alone does not prove bandwidth saturation.

The existing extractor hard-codes single-core CPU stat names; extend and verify
it against the actual multicore configuration before interpreting its counters.

### Implementation phases

Phase CSV columns:

```text
config_id,variant,replicate,scope,worker_id,phase,sim_seconds,roi_seconds
```

Use a fixed workload, architecture and core count within `config_id`; `variant`
identifies the implementation being compared. `phase` is a measured category
such as `staging`, `kernel`, `barrier` or `reduction`. Record one row per phase
and run, summing repeated intervals of that phase within the timed ROI.

- `scope=wall,worker_id=all`: exclusive top-level phases partition elapsed ROI
  time. A parallel region's wall time must not be the sum of worker times.
- `scope=worker,worker_id=0` (or another ID): exclusive phases of that worker,
  measured against the same guest simulated clock and common ROI boundaries.
  Record all workers, including idle workers, consistently across repetitions.
- `sim_seconds` may be zero for a measured absent phase. Phase sets must match
  within each configuration/scope. Missing phase rows are rejected, not assumed
  to be zero. `roi_seconds` is the positive elapsed duration of the common ROI
  and must agree across phases, workers and scopes for each run.

The script plots wall and worker scopes separately. Stacked bars show mean phase
durations across repetitions; the percentage control uses mean phase time /
mean ROI time, so the stack remains additive. Uncovered time is shown in gray
as `Unattributed`, never relabeled as useful compute or waiting. Phase totals
above the ROI duration are rejected. `Unattributed` and `__roi` are reserved
names. Supply mutually exclusive phase intervals: totals alone cannot detect
every overlap. Producer/consumer overlap needs timestamped events rather than
this duration table. Index decode, lookup and MAC within the current SVE kernel
must not be presented as separate measured phases without actual instrumentation.

For multicore diagnosis, compare each worker's kernel and barrier shares.
Different shares can reveal imbalance; they do not identify which physical core
ran a migrating thread. This is an aggregate phase chart, not a Perfetto trace.

## Research basis for the additional plots

- Goto and van de Geijn, [*Anatomy of High-Performance Matrix Multiplication*](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf),
  ACM TOMS (2008), [DOI](https://doi.org/10.1145/1356052.1356053): cache/TLB-aware
  blocking and packing motivate plotting accounted tile footprint against runtime
  and measuring staging cost alongside the kernel.
- Smith et al., [*Anatomy of High-Performance Many-Threaded Matrix Multiplication*](https://www.cs.utexas.edu/~flame/pubs/blis3_ipdps14.pdf),
  IPDPS (2014): parallel loop choice, cache sharing and synchronization motivate
  strong-scaling, efficiency and per-worker phase comparisons.
- McCalpin, [*Sustainable Memory Bandwidth in Current High Performance Computers*](https://www.cs.virginia.edu/~mccalpin/papers/bandwidth/bandwidth.html)
  (1995): measuring sustained bandwidth motivates the measured DRAM-bytes/ROI
  plot. Any reference bandwidth should be measured for the simulated system.

These are diagnostic plots adapted to this project, not reproduced paper results
or assumed performance bounds for Minor/SVE. Prioritize a few promising tile
settings and repeatable baselines before expensive full-system sweeps.

## Validation

```bash
python tests/profiling_report_test.py
python tests/profiling_add_experiment_test.py
```

Tests cover measured ratios, source anomalies, repeated configurations, new
experiment discovery, stale combined tables, English chart-only output, file
protection, tile footprints, strong scaling, measured bandwidth, exclusive phase
accounting, live refresh and collection-triggered regeneration.
Synthetic results are confined to temporary test fixtures.
