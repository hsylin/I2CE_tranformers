# SDOT Roofline and Transformer reports

Open [Transformer profiling](profiling.html) locally; no web server is required.
It contains Overview, Codebook Roofline, Dense QKᵀ / SV Roofline and
Complete BLOCK Roofline for E01–E10.

`profiling.html` is the single published model report. Its scope tables
highlight the included operations; detailed accounting is documented below.

## Published calibration

| Reference | Stable median |
|---|---:|
| Integer throughput | 125.582664 Gop/s |
| CPU–L1D combined traffic | 43.442017 GB/s |
| L1D–L2 combined traffic | 27.684364 GB/s |
| Physical DRAM burst traffic | 1.987268 GB/s |

Full precision, exact counts/ticks, selected conditions and raw-statistics
hashes are in `../transformer_profiling/calibration/sdot-scan.json`. These are
independently selected best stable observations within the registered
streaming SDOT pattern, not universal maxima or a multicore experiment.

The scan has 90 bandwidth working sets from 1.5 KiB to 12 MiB; compute tests
cover D=1–256 (powers of two), U=1/2/4/8 at 12 KiB; cross-checks use 192 KiB
and 3 MiB. The shared bandwidth/compute condition is counted once. In total,
812 conditions are repeated three times, yielding 2,436 measured ROIs and
nine additional empty windows. All 144 ARM functional cases passed.

[The measurement method](../transformer_profiling/roofline-method.md) defines the exact plan,
warm-up/reset procedure and 5%/5%/1% stability rules. Longer-D and different
operand-sharing kernels are outside this profile and are not pooled into it.

## Data and provenance

The calibration scan was recorded on 8–9 November 2026 (UTC).
`sdot-recording.json` contains the exact start/end timestamps and SHA256
identities of the original execution receipts. Report regeneration does
not change the measurement dates.

* [All 2,436 ROI measurements](sdot-scan-rows.csv).
* [All 812 three-repeat summaries](sdot-scan-groups.csv).
* [All 137 largest-pass results and stability flags](sdot-scan-largest-pass.csv).
* [Complete scan plan, summary, pass-increase declines and verification](sdot-scan-results.json).
* [All three model scopes and calibration provenance](roofline-quantities.json).
* [Recording identity](sdot-recording.json).

Raw statistics, complete INI/JSON, exact measured source and binary, frozen
inputs and execution receipts are supplied in the separate recording archive.
Reprocessing recomputes all observations from raw counters and checks them
against the recorded analysis without overwriting it. The maintained tool
supports new scans; its newly built binary is not silently substituted for
the executable associated with these historical measurements.

## Model interpretation

E01–E10 names come from `transformer_profiling/sequence.json`. E09 introduces
Attention SDOT, and E10 adds Attention QK/SV blocking. Applying this calibration
changes reference slopes, ridges and reference-relative ratios only. All 20
model-scope observations retain their original time, useful work, traffic,
AI and source/binary identity. Ten Dense QKᵀ / SV observations are derived
from the same saved counters, giving 30 scope observations in total.

Codebook covers Q/K/V projections for all heads, the W_S output projection
after concatenation, FF1 expansion and FF2 reduction. Dense QKᵀ / SV covers
the QK and SV marker intervals for all heads. These include score-buffer
allocation/initialization, V transpose, output rescaling and the per-head
copy into the multihead buffer. Separately marked Softmax, MHA_out and the
projections are excluded; preparation inside a selected marker is retained.
Both views sum time and traffic over their own marker intervals;
neither is computed as a fraction of Complete BLOCK.

The model plots describe registered binaries from before the input and
Softmax repairs. Repaired checkpoints are not remeasured here. The E10 BLOCK
time of 68.611685 ms remains attached to its measured binary and is not a
performance validation of the current checkout. See
[the accounting method](../transformer_profiling/roofline-method.md).

## Regenerate

First reprocess the recording using the [scan commands](../transformer_profiling/roofline-method.md#sdot-scan-commands).
For the model reports, use the original model-evidence package and the
published explicit calibration:

```sh
python3 transformer_profiling/report.py \
  --input "$MODEL_EVIDENCE/reports/profiling.tsv" \
  --roofline "$MODEL_EVIDENCE/roofline.json" \
  --calibration transformer_profiling/calibration/sdot-scan.json \
  --output "$NEW_PROFILING_HTML" --single-run-study --compact-labels
```

This command verifies saved evidence and does not launch simulations. The old
write-pattern calibration is no longer published; historical schema support
is retained for evidence compatibility and isolated tests.
