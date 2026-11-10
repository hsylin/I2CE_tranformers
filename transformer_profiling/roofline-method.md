# Native-traffic Roofline evidence

Roofline is an optional view in the existing offline report. A compatible
`roofline.json` bundle beside profiling tables, or in a run subdirectory, is
loaded automatically. `report.py --roofline <manifest>` selects a specific
bundle. No raw-statistics guessing or historical example fallback occurs.
Visualization is provided only by `report.py` in `profiling.html`, with
Codebook, Dense QKᵀ / SV and Complete BLOCK views. `roofline.py` provides
their shared accounting and chart functions, not a standalone report command.
Measurement definitions and downloads are indexed in `reports/README.md`.
Source names and SHA256 identities remain portable; host-specific absolute
paths are not embedded in published HTML.

## Published empirical references

The explicit `calibration/sdot-scan.json` record uses the full registered scan:
812 conditions, three repetitions, 2,436 measured ROIs and nine empty windows.
It selects 125.582664 Gop/s integer throughput, 43.442017 GB/s Core–L1D,
27.684364 GB/s L1D–L2 and 1.987268 GB/s physical DRAM. Full precision is retained.
The exact scan definition and commands are listed below.

References are independently selected largest-pass medians satisfying the
registered 5% pass-change, 5% repeat-range and 1% empty-overhead gates. The
calibration stores the exact last two pass counts and their three repetitions,
raw-counter/tick values, hardware identities and source SHA256 values. The
renderer rechecks these quantities before applying the references together.
The raw recording is reprocessed read-only; none of its inputs, statistics,
previous analyses or execution receipts are overwritten.

Core–L1D and L1D–L2 rates combine native read and write bytes. Each AI uses
useful work divided by combined bytes at that same boundary. Physical DRAM
uses 64-byte bursts; controller payload remains separate. The L2 monitor
observes the L1D path, excluding instruction-cache and page-walker paths.
The kernel's nominal AI, D/2, is not substituted for native L2 or DRAM AI.

The previous write-pattern calibration is removed from the published
calibration directory. Its schema remains readable for archived evidence,
and its isolated test fixture is not a default or a published reference.
The conditional 128 Gop/s SDOT issue limit and theoretical 12.8 GB/s DRAM
interface rate remain separately labeled metadata. The native DRAM peakBW
counter is checked against that interface rate, not the empirical scan rate.

These measured references apply to the declared single-core machine and
streaming SDOT pattern. They are not universal maxima, physical-host noise
estimates or simulated multicore results. Other access patterns or larger
D/U searches may achieve higher rates. A scalar stage can use the same
machine reference, but distance below a line alone does not identify a causal
bottleneck. Do not tune a calibration separately for each optimization stage.

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
dump once. All heads of Q=XW_Q, K=XW_K and V=XW_V are included. Projection covers
W_S applied after concatenating the per-head SV outputs; concatenation/repacking
itself (MHA_out) is excluded. FF1 expands D to F; FF2 returns F to D.
The selection excludes QKᵀ/SV, Softmax, normalization and cleanup. Useful work is `2 L (4 S D² + 2 S D F)`; time and each boundary's
read/write bytes come only from the selected intervals, never a fraction
of BLOCK totals. Packing, conversions and surrounding work inside those
markers remain included. This is not a pure SDOT-microkernel measurement.

The Dense QKᵀ / SV view combines every MHA_QK and MHA_SV interval across
all heads. QKᵀ produces the score matrix; SV multiplies softmax probabilities
by values. The intervals also include score-buffer allocation/initialization,
the V transpose, output rescaling and each head's copy into the multihead
buffer. In the measured workload, the score buffer is 512 KiB and rescaling
uses `>> 6`. Separately marked Softmax and MHA_out, all projections and FF1/FF2
are excluded; preparation or copying inside a selected marker is retained.
With sequence length S, model dimension D and L learners,
useful work is `4 L S² D`: each head contributes two `2 S² d_head` products.
Here S in this formula denotes sequence length, not the probability matrix.
Codebook and Dense useful work sum to BLOCK useful GEMM work, while their
times exclude separately marked non-GEMM phases. This is a time-attributed
sum of marked operators, not isolated kernel timing or the entire attention
pipeline. The same chart layout and boundary-specific reference lines are
used for each view; INT8 SDOT references do not imply every earlier stage
executes that instruction or reaches that pattern's peak.

The exported `limitations` describe the active calibration. When a validated
SDOT calibration replaces the old write-pattern reference, its calibration
caveat is updated. `recorded_limitations` preserves the original manifest
wording; the external manifest and all measured quantities remain unchanged.

Derivation requires the registered whole-transformer operation model and
separate per-head QKV/QK/Softmax/SV markers followed by the six block-tail
markers. A changed or fused marker scheme makes derived views unsupported,
without suppressing an otherwise valid BLOCK observation. The original
manifest, receipt, ROI and operation-model hashes are preserved. Exported
selection indices and both counter endpoints make each subtraction
independently checkable. All scopes use the same fixed machine references.
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

The model report is `reports/profiling.html`, with Overview, Codebook,
Dense QKᵀ / SV and Complete BLOCK views. Updating calibration changes
reference lines, ridges and reference-relative ratios; all 20 historical
observations retain their original time, useful work, bytes, AI and identities.
The ten Dense observations are derived from the same original raw counters;
no model rerun or historical observation replacement is involved.
The calibration scan is published as numerical CSV and JSON data.
The published model observations precede the input and Softmax repairs.
Repaired checkpoints have not been remeasured; the historical E10 time is not
a measurement of a newly rebuilt checkout.

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


## SDOT scan commands

The full scan uses the exact 90 sizes in
`tools/roofline/data/bandwidth-worksets.json`, not evenly spaced substitutes.
Each tile contains distinct 16-byte A, W and C vectors (48 bytes total).
N is the tile count; D is SDOT updates per loaded accumulator, U is the
number of independent interleaved chains, and passes is complete traversals
inside one measured interval.

| Stage | Working sets | D | U | passes | Additional conditions |
|---|---|---|---|---|---:|
| Bandwidth | 90 sizes, 1.5 KiB–12 MiB | 1 | 8 | Powers of two up to min(64, max(8, floor(262144/N))) | 519 |
| Compute | 12 KiB | 1, 2, 4, 8, 16, 32, 64, 128, 256 | 1, 2, 4, 8 | 1, 2, 4, 8, 16, 32, 64 | 245 |
| Cross-check | 192 KiB and 3 MiB | 4, 16, 64 | 1, 8 | 1, 2, 4, 8 | 48 |

Combine values within each row. The shared compute D=1/U=8 block is measured
only by the bandwidth row: 812 conditions, 137 blocks and 92 distinct sizes.
Three independent repetitions give 2,436 measured ROIs plus nine empty
windows, with 816 statistics dumps per repetition including exit.

Initialize each block once. Before every ROI, run one same-kernel warm-up
pass, then reset statistics without flushing caches. Check results outside
the ROI. Use simTicks/simFreq for time; do not subtract empty-window timing.
Each SDOT performs 32 integer operations. Nominal AI is D/2, while native
AI uses measured bytes at each individual boundary. Largest-pass medians
must satisfy 5% last-two-pass change, 5% repeat range/median and 1% empty
overhead; retain unstable results and all adjacent-pass declines.

Builds, tests and simulation run on the Linux simulation server. Preparation
requires a completed reference-machine package with its expected freeze hash,
host C++/ASan/UBSan, an AArch64 compiler/objdump, libm5 and the registered
gem5 ARM checkpoint/kernel/disk assets. It verifies the reference configuration,
scalar oracle and assembly, then freezes a new package. No plotting dependency
is required by the scan.

```sh
python3 tools/roofline/scan.py plan
python3 tools/roofline/scan.py prepare \
  --output "$NEW_RUN" --machine-root "$REFERENCE" \
  --machine-freeze-sha256 "$REFERENCE_FREEZE_SHA256" \
  --host-cxx "$HOST_CXX" --arm-cxx "$ARM_CXX" --objdump "$OBJDUMP" \
  --m5-include "$M5_INCLUDE" --libm5 "$LIBM5" --m5-path "$M5_PATH"
python3 tools/roofline/scan.py run --root "$NEW_RUN" --workers 3
python3 tools/roofline/scan.py calibrate --root "$NEW_RUN" --output "$CALIBRATION"
python3 tools/roofline/scan.py export --root "$NEW_RUN"
```

Each process simulates one core. Execution checks full configuration and
binary modes, validates 144 ARM cases, then runs at most three repetitions
concurrently using SCHED_IDLE/nice19. Admission requires 64 GiB available
memory and 100 GiB free disk; exclusive start claims prevent automatic retries.

To regenerate numerical results from the supplied recording without simulation:

```sh
python3 tools/roofline/scan.py reprocess \
  --root "$RECORDING" --freeze-sha256 "$RECORDING_FREEZE_SHA256" \
  --output "$NEW_RESULTS_DIRECTORY"
```

Reprocessing requires access to the registered gem5 checkout and binary
identified by `registration.json`; these machine assets are separate from
the repository. It verifies frozen evidence and recomputes all observations,
checking 2,436 rows, 812 groups and 137 largest-pass records against the
recording. It writes analysis, calibration and CSV/JSON results only.
The internal `analysis/summary.json` is an input to calibration; the published
summary appears only inside `sdot-scan-results.json`, without a second copy.
