# Native-statistics hierarchical Roofline

This view uses **theoretical/conditional reference lines and measured points**.
The L1/L2 references are resource bounds, not cache-bank peaks or empirical
calibration results. Reporting never launches gem5, rebuilds a model, changes
cache policy, or infers missing bytes from cache misses.

## View the preserved E13 run

From the repository root, with the existing visualization dependencies:

```bash
python -m pip install -r transformer_profiling/requirements-visualization.txt
python -B transformer_profiling/roofline.py \
  --output /tmp/e13-roofline.html \
  --json-output /tmp/e13-roofline-quantities.json
```

The default manifest is `roofline_examples/e13/roofline.json`. E13 is rendered;
E01–E12 are listed as unavailable because matching native boundary traffic is
not retained. The HTML embeds Plotly for offline viewing; the camera exports
SVG. The JSON contains recomputed quantities, source counters/line numbers,
config fields, formulas, hashes and skip reasons. The plot has a separate point
per displayed boundary/direction, dotted projection guides, and open reference
points. It shows Core–L1D, L1D–L2 read/write and physical DRAM. Controller payload
is retained for validation and detailed quantities, but is not plotted.
All actual points share the same useful-work numerator and simulated ROI time.

To add the independent Roofline observations to an existing profiling report:

```bash
python -B transformer_profiling/report.py \
  --input transformer_profiling/final \
  --roofline transformer_profiling/roofline_examples/e13/roofline.json \
  --output /tmp/profiling-with-roofline.html
```

This adds a separate tab. It does not join measurements by `E13` alone or replace
TSV timing. Historical datasets have separate experiment namespaces. The
Roofline run identities remain in the HTML's embedded provenance; measurement
limitations are documented below and in the standalone detailed report. The main
report's model/experiment filters apply to its TSV dataset. Each supplied Roofline
observation is plotted independently.

The normal `report.py --input results --output profiling_report.html` command
discovers evidence automatically in this layout:

```text
results/
  e01_*.tsv                   # Existing profiling observations
  e02_*.tsv
  runs/
    e13-monitored/
      roofline.json           # Manifest, dimensions, ROI, identities and hashes
      stats.txt
      config.json
      regions.tsv
      receipt.json
      provenance.json
    another-run/
      roofline.json
      ...
```

An explicit `--roofline` selects only that manifest. Otherwise, a `roofline.json`
at the input root is authoritative; if absent, manifests in nonhidden run
subdirectories are discovered recursively. Automatic discovery stays inside the
selected input dataset, excludes external symlinks, and never imports examples
from elsewhere. Multiple bundles with the same experiment ID remain distinct,
identified by their manifest paths in the embedded provenance. A malformed
bundle is recorded there without blocking valid bundles. The integrated
Roofline tab displays charts only, titled `Hierarchical Roofline Model for ID`;
run identities and skip reasons remain in the HTML's embedded provenance.
All charts remain in the same self-contained HTML. The standalone report still
provides the detailed quantity tables and derivations.

`--watch` detects added bundles and raw evidence changes; reload the HTML after
regeneration. Without any manifest, the existing TSV report still renders and
the command reports why Roofline was skipped. Raw `stats.txt` and `config.json`
alone are insufficient to infer a successful run, its ROI or useful integer
work. Register that metadata once per run; the report then reads and validates
the native measurements automatically without manually entered bandwidths.

## Quantities and matched denominators

| Quantity | Source / derivation |
|---|---|
| Time | Native `simTicks / simFreq`; never `hostSeconds` |
| Useful work W | `2L(4SD² + 2S²D + 2SDF)` from declared transformer GEMM dimensions; MAC = two integer operations |
| Executed cross-check | `system.cpu_cluster.cpus.commitStats0.committedInstType::SimdDotProd × 32`; OpClass, not a mnemonic trace |
| Core–L1D bytes | `system.roofline_l1.totalReadBytes`, `totalWrittenBytes` |
| L1D–L2 bytes | `system.roofline_l2.totalReadBytes`, `totalWrittenBytes` |
| Physical DRAM bytes | `system.mem_ctrls.dram.dramBytesRead`, `dramBytesWritten`; checked against corresponding `readBursts`, `writeBursts × 64` |
| Controller payload | `system.mem_ctrls.dram.bytesRead::total`, `bytesWritten::total`; checked against the membus L2-to-controller route `pktSize` counter |
| Native average bandwidth | Monitor `averageReadBandwidth`, `averageWriteBandwidth` in B/s; DRAM `avgRdBW`, `avgWrBW` in decimal MB/s in the inspected source |
| Observed point | `AI = W / Q`, `P = W / time`; useful integer Gop/s, not GFLOP/s |
| Reference projection | `min(compute reference, bandwidth reference × AI)`; not another measurement |
| Ridge | `compute reference / bandwidth reference` |
| Bandwidth fraction | `(Q / time) / bandwidth reference`; not a hardware saturation counter |
| Compute fraction | `P / compute reference` |
| Model headroom | `reference projection / P`; not a promised speedup |

Read monitor bytes count successful read-response payload; write bytes count
successful write-request payload. Packet headers/control traffic are not added.
L1D–L2 excludes the separate I-cache and page-walker paths. Physical DRAM includes
system traffic during the ROI. Controller payload and physical burst bytes are
distinct; do not substitute one for the other or add traffic across levels.

The current reference model is the inspected single-core Minor/SVE128/classic
cache model. `config.json` clock periods are ticks/cycle, so clock Hz is
`simFreq / clock[0]`; clock-domain references are checked explicitly.

| Reference | Configuration / formula | E13 | Required AI |
|---|---|---:|---|
| Integer compute | CPU Hz × 32 × min(`executeIssueLimit`, `executeCommitLimit`, one dot FU / `issueLat`) | 128 Gop/s | All points share W/time |
| Core–L1D | CPU Hz × 16 × min(`executeMemoryIssueLimit`, `executeMemoryCommitLimit`) | 64 GB/s | W/(read+write) |
| L1D–L2 | 64 × cluster Hz / (`toL2Bus.header_latency` + ceil(64/`toL2Bus.width`)) | 128 GB/s per direction | W/read, W/write separately |
| Membus/controller | 64 × system Hz / (`membus.header_latency` + ceil(64/`membus.width`)) | 12.8 GB/s per direction | W/read, W/write separately |
| Physical DRAM | (`burst_length` × `devices_per_rank` × `device_bus_width` / 8) × simFreq / `tBURST` | 12.8 GB/s shared | W/(read+write) |

CPU and `toL2Bus` use the cluster's 4 GHz; membus uses the system's 1 GHz.
SVE length `system.sve_vl=1` means 128 bits. One full int8 SDOT performs 16 MACs,
32 integer operations. The dot FU's `opLat=6` requires independent accumulators
to sustain `issueLat=1`; latency is not the reciprocal throughput.
The 16 B L1 assumption requires full aligned contiguous vector accesses, with
reads/writes sharing the memory issue limit. It does not establish L1 bank
bandwidth. Crossbar formulas include header occupancy for a 64 B packet, not
just width × Hz; other requests, competing ports and cache service can lower
throughput. DRAM `peakBW=12800` is decimal MB/s in this source despite its unit
label; it is checked against the configuration-derived 12.8 GB/s. Refresh,
bank conflicts and read/write turnarounds can lower sustained bandwidth.

Softmax, gather and packing remain in measured time but are excluded from W.
Being below all references does not identify a unique bottleneck. No hardware
maximum or calibrated saturation is claimed for L1 or L2.

## Collect and register another experiment

Native `CommMonitor` already supplies the required counters: no gem5 C++ changes
or rebuild are needed. The existing configuration patcher now installs an
**off-by-default** `--roofline-monitors` option:

```bash
python -B tools/exp/lib/apply_gem5_opts.py /path/to/gem5/configs/example/arm/starter_fs.py
```

The patcher retains its backup/idempotence checks. It changes the explicitly
named configuration file, not the gem5 executable. Invoke the patched config
with `--roofline-monitors` for a separately registered timing run. For example,
append that flag to the existing reviewed gem5 restore command, keeping its
CPU, cache, inputs, binary and checkpoint parameters. Do not enable it for an
atomic boot-checkpoint job. `exp.sh patch-gem5` installs flag support too; its
normal run commands do not enable it automatically. Existing external starter
scripts do not gain the option merely by checking out this commit.

The option splices two monitors into the CPU data port and L1D memory-side port,
matching the preserved E13 run. It rejects multicore/atomic configurations and
does not change cache inclusion policy. This commit tests patch installation
and wiring on the host; it does not claim a new gem5 initialization or run.

For each completed run, preserve raw `stats.txt`, `config.json`, region markers,
exit/configuration/binary checks and source identity. Add an entry to a manifest
following the included E13 example, with relative artifact paths and their
SHA256 values. `receipt.json` records verified checks; it is not a substitute
for checking and preserving the original logs. Set the exact expected dump
count, zero-based whole-block end index and labels. Counts must be cumulative
since a single reset; do not select the final cleanup dump. The current useful
work formula applies to a whole transformer block, not an arbitrary subregion.
Subregion operation models must be implemented and reviewed before plotting.

Multiple entries produce multiple plots. Missing evidence, nonfinite values,
hash/identity mismatches, incomplete dumps, resets, unsupported precision/model,
multicore or mismatched monitor wiring produce explicit skip reasons. A skipped
run does not prevent valid runs from rendering. Zero traffic has no finite
log-axis AI and its point is omitted; missing counters are not assumed zero.
To declare an unmeasured experiment, use only `id` and `unavailable_reason`.
Never reuse a calibration or byte denominator from a different configuration.

## Model sources

The implementation pins Jerry0209/gem5 revision
`ed752e78a2ddd5587c1954e2cda51f94cbbe554e`; another revision needs a semantics
review before support is enabled:

- [MinorCPU scheduling parameters](https://github.com/Jerry0209/gem5/blob/ed752e78a2ddd5587c1954e2cda51f94cbbe554e/src/cpu/minor/BaseMinorCPU.py)
- [Packet serialization](https://github.com/Jerry0209/gem5/blob/ed752e78a2ddd5587c1954e2cda51f94cbbe554e/src/mem/xbar.cc)
- [Crossbar layer occupancy](https://github.com/Jerry0209/gem5/blob/ed752e78a2ddd5587c1954e2cda51f94cbbe554e/src/mem/coherent_xbar.cc)
- [DRAM bursts and bandwidth units](https://github.com/Jerry0209/gem5/blob/ed752e78a2ddd5587c1954e2cda51f94cbbe554e/src/mem/dram_interface.cc)
- [CommMonitor payload accounting](https://github.com/Jerry0209/gem5/blob/ed752e78a2ddd5587c1954e2cda51f94cbbe554e/src/mem/comm_monitor.cc)
