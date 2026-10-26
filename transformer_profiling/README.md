# Transformer profiling tools

This directory contains reusable collection and reporting tools alongside
historical datasets. For the reconstructed E01-E13 implementation sequence,
source preparation and build reproduction, use the
[reconstruction workflow](experiment-runner.md) and [stage index](provenance/stages.json).
The reconstruction adds no performance measurements.

## Report existing observations

Run commands from the repository root. Install the pinned visualization
dependencies in a suitable Python environment:

```bash
python -m pip install -r transformer_profiling/requirements-visualization.txt
python -B transformer_profiling/report.py \
  --input transformer_profiling/final \
  --output /tmp/historical-profiling.html
```

`report.py` writes self-contained HTML with Plotly.js and source metadata
embedded for offline viewing. The chart camera exports individual figures.
Reporting does not need a compiler, gem5 or `runner.conf`.

The [final](final/README.md) and [base_mini](base_mini/README.md) datasets are
inherited studies. Their experiment numbering does not identify the new stage
tags. Select one dataset explicitly; combining directories that both start
with E01 would confuse distinct observations. Historical binaries and saved
results retain their original scope.

The default input is `transformer_profiling/hsylin`, a destination for separately
collected results. A checkout does not imply a new result table exists there;
missing inputs are errors, with no fallback to historical data. A report on
another directory needs its matching `--input`.

Within an input directory, individual `eNN_*.tsv`, `ENN_*.tsv`, `eNN.tsv` or
`ENN.tsv` files override that experiment in `<dirname>_all_experiments.tsv`.
The filename ID must match its rows. Duplicate individual IDs and incomplete
experiments are rejected. Each experiment needs one `interval_delta` row per
region and one `final_total` row, matching either the six-region or split
Attention schema. Input TSVs remain unchanged, and failed validation preserves
previous valid HTML. `add_experiment.py` is the standard-library collector;
its region index comes from the run that produced the dumps.

Useful report options include:

```bash
python -B transformer_profiling/report.py --help
python -B transformer_profiling/report.py \
  --input transformer_profiling/final --experiments E01 E03 \
  --output /tmp/selected-profiling.html \
  --metrics-output /tmp/profiling-metrics.tsv
```

`--watch` rebuilds on local input changes and stops with Ctrl-C; it does not
fetch remote data or rerun gem5. Reload the browser after regeneration. A
previously copied HTML file remains a snapshot.

## Metrics and interval scope

Times are simulated seconds, not host elapsed time. Demand miss rate is
misses/accesses, MPKI is misses × 1000/instructions, and IPC is
instructions/cycles. Ratios use interval counts, not differences of cumulative
ratios. Zero denominators remain missing. L2 misses do not measure DRAM bytes,
and `simOps` is not a FLOP count. Learner count is not core count.

The split schema includes `MHA_QKV`, `MHA_QK`, `MHA_softmax`, `MHA_SV` and
`MHA_out`, plus Projection, FF1, FF2 and the two non-GEMM regions. Their scopes
depend on source marker placement. Setup, allocation, initialization and
optional output can occur inside these intervals. The Codebook-labelled sum
(`MHA_QKV` + Projection + FF1 + FF2) is a region sum, not pure kernel time.

E12/E13 move the internal Attention boundaries: the QK-labelled interval
contains the complete streamed QK/softmax/SV computation, V transpose and
enabled captures; the softmax boundary immediately follows it, and the
SV-labelled interval includes gather. Compare aggregate Attention with the
materialized E01-E11 path. Source-compatible marker groups can be supplied
with `--marker-schemes`; this metadata must be bound to the saved inputs.

Repeated settings remain separate observations. Ambiguous SVE baselines or
grid cells do not imply a speedup. Verify workload, binary, compiler,
architecture and marker compatibility before interpreting ratios. Never form
a synthetic result from the best phases of different executions.

Historical source limitations remain visible in report metadata or warnings:
`final/` E02 has invalid Attention/Projection intervals, so its phase bar is
omitted while its recorded total and counters remain. E05/E06 in that dataset
are estimated dense baselines and retain estimated labels. These statements
refer to that historical dataset, not the reconstructed tags. The report's
code revision is distinct from the revision used to build an observed binary.

## Optional measured-data schemas

The header-only [templates](templates/) describe three optional CSV inputs.
Use existing measurements with `--tile-results`, `--scaling-results` and
`--phase-results`, or place the corresponding filenames beside the selected
TSV input. Tabs appear only when measurements exist; templates produce no
placeholder results. At least one valid TSV experiment is still required.
These files do not run sweeps, instrument kernels or infer observations.

### Tile measurements

```text
config_id,tile_m,tile_n,tile_k,replicate,sim_seconds,l1d_accesses,l1d_misses,l2_accesses,l2_misses,tile_footprint_bytes,l2_capacity_bytes
```

Hold workload, datatype, kernel, learners, SVE, core count, architecture and
weight-sharing policy fixed within `config_id`. Heatmap cells show median time
or median per-run miss rate; unmeasured cells stay empty. Sum participating
private L1 counters as appropriate and count a shared L2 once.

Footprint and L2 capacity are optional; a populated column must be complete.
Footprint is positive and fixed across repetitions of a tile; capacity is
positive and fixed within a configuration. Account for simultaneously live
operands, outputs, packed indices, codebooks and staging buffers, including
replication. Allocated arena size is neither measured traffic nor proof of
cache residency; nominal capacity alone omits associativity and TLB effects.

### Strong scaling

```text
config_id,num_cores,replicate,sim_seconds,dram_bytes
```

Hold everything except core count fixed and include a measured one-core
baseline. Speedup is median T(1)/median T(p); parallel efficiency is
100 × speedup/p. Use elapsed simulated ROI time rather than summed core times.
The existing extractor uses single-core stat names; a multicore study needs
separately validated counter collection and configuration.

Optional `dram_bytes` must be measured read-plus-write controller traffic over
the same ROI, summed across channels once, and finite and nonnegative for every
row. Bandwidth is the median of per-run bytes/seconds/1e9 in decimal GB/s.
L2 misses × line size and tensor allocation sizes are not substitutes.
A plateau alone does not prove saturation.

### Implementation phases

```text
config_id,variant,replicate,scope,worker_id,phase,sim_seconds,roi_seconds
```

Record exclusive phase durations for a fixed workload and architecture.
`scope=wall` and `scope=simulated` use `worker_id=all`; `scope=worker` identifies
one worker. Do not mix host and simulated clocks, or substitute summed worker
times for elapsed ROI time. Include all workers consistently.

Phase sets must match within each configuration/scope. Measured zero is
allowed; missing rows are not assumed zero. Positive ROI duration must agree
across phases in a run. Wall and worker scopes share the same physical elapsed
ROI; simulated time is independent. The renderer shows mean phase durations
and mean-phase/mean-ROI percentages; uncovered time is `Unattributed`.
Totals above the ROI are rejected. `Unattributed` and `__roi` are reserved names.
Duration tables cannot detect every overlap or explain thread migration;
producer/consumer overlap needs timestamped events. Decode, lookup and MAC are
not separate measured phases without corresponding instrumentation.

## Optional comparison metadata

`--cache-comparison` accepts schema-version-1 metadata with `configuration`,
`variants` and `comparisons`. Variants bind source, binary and run identities
and tile geometry; comparisons identify explicit baseline/candidate pairs.
The renderer checks selected TSV identities and shared configuration and
records the metadata hash. Matching metadata beside an input can be discovered
automatically. Historical inputs receive no cache comparison without it.

This checkout does not supply a new canonical comparison table or cache
comparison result set. Supply your own matching saved evidence; the tool's
support for a schema is not evidence that its measurements exist.
`--experiments` filters TSV/cache comparisons; optional CSVs retain their own
configuration selection. `--compact-labels` adds short IDs and a key;
`--single-run-study` shows single-observation and boundary limitations.
`--interpretation` accepts evidence-qualified notes bound to saved-input hashes.

## Validate the host tools

```bash
python -m pip install -r transformer_profiling/requirements-tests.txt
python -B tools/exp/validate_host.py --lint
```

Tests use temporary fixtures and cover report formulas, input discovery,
source anomalies, file protection, optional schemas and collection. They do
not launch a simulator or regenerate frozen scientific inputs. Arm numerical
validation and production build identities have the separate scopes recorded
in each stage's `source.json` and the reconstruction workflow.
