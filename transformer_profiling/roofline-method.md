# Native-traffic Roofline evidence

Roofline is an optional view in the existing offline report. A compatible
`roofline.json` bundle beside profiling tables, or in a run subdirectory, is
loaded automatically. `report.py --roofline <manifest>` selects a specific
bundle. No raw-statistics guessing or historical example fallback occurs.
The standalone renderer takes explicit `--input` and `--output` paths.
Both renderers offer Codebook and complete BLOCK views. Standalone
`--scope codebook` or `--scope block` exports one view; the default is both.
The report omits the Parameters page, repeated experiment-key tables,
Roofline summary tables and explanatory notes. Exact run identities,
counter provenance and measurement limits remain in exported data and
this method document.

Core–L1D and L1D–L2 slopes come from the preserved independent Cache A
calibration in `calibration/cache-a.json`: 50.840555 and 56.887924 GB/s.
Calculations retain full precision. Both rates combine reads and writes
from the same window. Their arithmetic intensities use useful work divided
by combined bytes at the matching boundary. Controller payload and physical
DRAM burst traffic remain distinct; bytes from different levels are never
added. The L2 monitor observes the L1D path, excluding instruction-cache
and page-walker paths.

These measurements characterize two write-oriented patterns inside gem5,
not the physical host server or universal cache maxima. The L1 sample uses
a 16 KiB contiguous-write kernel. The L2 sample writes 16 bytes per 64-byte
line across 768 KiB and includes both reads and writebacks. Source hashes,
selected sample identifiers, residency checks, duration/unroll comparisons,
and complete relevant JSON/INI hardware identities accompany the rates.
The raw calibration archive is retained separately. Repetitions establish
reproducibility for those cases, not workload generalization or confidence
intervals. Do not tune the calibration to an experiment stage.

The common compute reference retains the inspected model's conditional
SVE128 SDOT issue limit. Physical DRAM retains its configured burst-rate
reference. These are theoretical values, distinct from the measured cache
slopes. A scalar stage can be plotted against the same machine references;
its useful work does not depend on SDOT instruction counts. No inference
about a causal bottleneck follows just from distance below a reference.

## Registered acquisition

Apply `tools/exp/lib/apply_gem5_opts.py` to a separately pinned supported
gem5 configuration. It adds an opt-in `--roofline-monitors` argument and
splices native CommMonitor objects into the two data boundaries. It does
not change capacities, clocks, policies or production kernels. Pin the
resulting configuration script and simulator. Establish the monitored
reference configuration explicitly; an unmonitored full-configuration
gate must not be weakened to accept different topology.

The finite-study registration enables acquisition through a `roofline`
object containing `study_id`, `gem5_revision`, and optional `calibration`
(`path`, `sha256`). Each registered row supplies a `roofline` object with
`source_commit`, `label`, `status`, `roi`, `operation_model`, `limitations`.
Its `build_record` (`path`, `sha256`) binds the actual `source_commit`,
`binary_sha256`, `commands` and `compiler_identity`. Include these records
in the study's immutable helper and runtime pins before execution. Use a
unique study ID; an E-number alone does not identify a measurement.

The registered command adds native monitors for both initialization and
formal execution. Existing configuration, guest identity and completion
gates remain in force. After ordinary collection succeeds,
`study_roofline.collect` preserves the raw statistics, both configuration
formats, ROI labels, host/guest exits, build record, input hashes,
checkpoint/tool identities and calibration record in the run's evidence
folder. It writes a portable manifest and collection status. Existing
exports are never overwritten. This operation cannot launch a simulation.
The manifest is discovered there or can be selected explicitly by the
report CLI. Collection does not certify full production tensor values.

## Work and ROI accounting

Schema 2 uses method `gem5_measured_cache_references`. Artifact paths are
relative, contained and SHA256-bound. Completion receipts bind the same
source, binary, input, measurement and raw-artifact identities. Metadata
changes therefore require a new explicit record, not relabeling old data.

For `operation_model.kind = transformer_gemm`, positive integer `learners`,
`sequence`, `dimension` and `feedforward` define whole-block useful work:
`2 L (4 S D² + 2 S² D + 2 S D F)`. This counts integer multiply/add operations
in the declared GEMMs, including both learners. It excludes packing,
Softmax and normalization operations while retaining their elapsed time.
SimdDotProd, when present, is an optional diagnostic only.

The Codebook view is a derived selection within that same validated run,
not a new measurement or a rewritten registration. It sums every QKV,
Projection, FF1 and FF2 interval, subtracting each preceding cumulative
dump once. The selection excludes Attention QK/SV, Softmax, normalization
and cleanup. Useful work is `2 L (4 S D² + 2 S D F)`; time and each boundary's
read/write bytes come only from the selected intervals, never a fraction
of BLOCK totals. Packing, conversions and surrounding work inside those
markers remain included. This is not a pure SDOT-microkernel measurement.

Derivation requires the registered whole-transformer operation model and
separate per-head QKV/QK/Softmax/SV markers followed by the six block-tail
markers. A changed or fused marker scheme makes Codebook unsupported,
without suppressing an otherwise valid BLOCK observation. The original
manifest, receipt, ROI and operation-model hashes are preserved. Exported
selection indices and both counter endpoints make each subtraction
independently checkable. Both scopes use the same fixed machine references.
The registered ROI's controller/route consistency gate remains unchanged.
For derived subregions, in-flight transactions can be recorded on opposite
sides of a marker at adjacent boundaries. Keep controller payload and route
payload separately; do not adjust either counter or require their derived
interval totals to agree. DRAM and monitor traffic are time-attributed,
not a causal allocation of every transfer to the operator that initiated it.

A `gemm_shapes` model declares `m`, `n`, `k`, `count` per shape and binds
`begin_block`, `end_block` and the exact selected label slice. Useful work
is the sum of `2 m n k count`. Shape multiplicity must come from the
registered workload; it is not estimated from executed instructions.

The existing protocol has 22 labels and 23 complete statistics dumps.
Use `counter_mode = cumulative_since_reset`, `begin_block = null`, and
zero-based `end_block = 21` for BLOCK. Cleanup dump 23 is excluded.
For an interval, subtract the preceding matching dump once. The parser
rejects inconsistent label counts, counter resets, missing native bytes,
nonfinite values, changed identities and incompatible model semantics.
Zero traffic has undefined finite AI and is omitted from the log plot.
No cache misses are converted into invented byte measurements.

Reports distinguish valid or diagnostic measurements and explicit
not-measured entries. Invalid/unsupported runs retain ordinary profiling
results and an explanation. A missing or incompatible cache calibration
omits its cache references; supported DRAM data can remain visible.
No cache reference is estimated as a fallback. Whole JSON and INI
CPU/cache/monitor/memory identities must match the portable calibration.

## Published experiment names

`sequence.json` defines E01–E10 for the published sequence. Both report
generators use this catalog: Overview recognizes the registered profiling
input by SHA256, and Roofline identifies each stage by its measured source
commit and binary SHA256. IDs, labels, hover text and ordering are normalized
before rendering. Other datasets retain their own experiment namespaces.
The source files, measurements, calibration and receipt checks are unchanged.

The generated reports are in `reports/profiling.html` and
`reports/roofline.html`. Both contain Codebook and Complete BLOCK views.

## Validation and future measurements

Host tests exercise cumulative/interval accounting, scalar useful work,
units, identity mismatches, changed hardware, missing calibration and
native monitor wiring. Synthetic counter fixtures are labeled as tests.
Preserved monitored evidence may demonstrate rendering, clearly labeled
as historical and bound to its original binary. It does not measure any
newly rebuilt stage. Monitoring may perturb execution and must not be
assumed equivalent to an uninstrumented run.

New stage measurements, monitor-impact checks for those binaries, and
full production-output validation remain separate work. Preparing this
reporting support does not run any of them.
