# E01-E13 reconstruction workflow

E01-E13 reconstruct the approved Jerry0209 and hsylin contributions as
implementation milestones. Upstream ancestry, existing author notices and
contributor credit remain. Commit and tag dates record this reconstruction,
not the original discovery or measurement dates.

The [stage index](provenance/stages.json) links each configuration, accepted
source map and binary SHA-256. Existing executable archives, model dumps and
[historical profiling datasets](README.md) retain their original meaning.
Their experiment IDs are separate from these tags; the inherited
`transformer.o` is not a stage build.

## Milestone sequence

| Stage | Implementation change |
| --- | --- |
| E01 | Widened Codebook cache and reusable int32 activation workspace; pre-SDOT baseline. |
| E02 | MR4xNR16 SDOT with adapters from widened operands. |
| E03 | Int8 operands, shared CI01 indices and bounded paired weight packing. |
| E04 | L1 16x32 accumulator tiles; K128 weight panels and K256 accumulation. |
| E05 | Outer 128x128x512 blocking and weight-panel reuse. |
| E06 | Retain all packed activation K panels across outer output tiles. |
| E07 | Specialize full weight panels and omit unread padding stores. |
| E08 | Consume full tiles in learner, row4, column16 order. |
| E09 | Write bytes at final-K accumulation and remove full C32 output temporaries. |
| E10 | Workload Attention SDOT; preserve the separate generic MR8xNR8 API. |
| E11 | Reuse bounded QK and SV operand panels. |
| E12 | Stream sixteen queries per panel with explicit private activation ownership. |
| E13 | Vectorize integer softmax while retaining its scalar fallback. |

The registered workload has sequence length 512, model width 256, four heads
of width 64, feed-forward width 1024, two learners, four Codebook entries and
128-bit SVE. Stage builds use one core. E13 retains inner tiles 16x32x256,
physical weight-panel K128 and outer tiles 128x128x512. General runner options
for other shapes, vector lengths or core counts do not extend this support.
The [inherited exceptions](provenance/constraints.json) record compatibility
interfaces, diagnostics and source details kept to preserve the accepted bytes.

## Prepare sources without executing a model

Run commands from the repository root. Preparation needs Python and Git:

```bash
python -B tools/exp/prepare_stage.py --stage E13 --output /tmp/e13-source
```

Choose a new output directory. The helper reads the stage tag, verifies every
`source_files` hash and the [canonical input manifest](inputs/manifest.json),
then writes `source/` and `prepared.json`. It overlays the canonical headers
and weights only in that directory. Use `--stage E01` through `--stage E13`
to select another milestone; `--revision` explicitly selects its commit.
This step builds no executable and launches no simulator.

Canonical bytes can differ from inherited example headers and weights.
Their producer revision remains unresolved; the exact bytes and hashes are
preserved without inferring a generator history.

## Reproduce an accepted binary

Each stage's `source.json` records its build environment, compiler, libm5 and
header hashes. The preserved compiler is AArch64 GCC 13.4; its version string
alone is insufficient to reproduce its dependencies. Set these paths to the
registered tools, replacing the placeholders:

```bash
export A64CXX=/path/to/aarch64-conda-linux-gnu-g++
export LIBM5_A=/path/to/libm5.a
export M5_INC=/path/to/gem5/include
python -B tools/exp/prepare_stage.py --stage E13 --output /tmp/e13-build --build
```

The helper verifies the registered tool identities before compiling with
`compile_transformer.sh`, using the stage's fixed flags. It preserves
`build.log` and `build-result.json` and rejects a binary hash mismatch.
A failed or completed output directory is not reused. The resulting
`source/transformer.o` is compiled only; this command never executes it.
A fresh isolated build for each reconstructed tag reproduced its accepted
full binary hash; build receipts and executables are retained externally.

The accepted E13 SHA-256 is:

```text
5d52441dc570a1a1b2db66bdb80287290c3c2c5318fd82fb3bbff059337b9766
```

It differs from original S25 (`406d00d5…`). Historical S25 timings cannot be
assigned to E13. Always bind a result to the full binary hash, source and input
identities, configuration and marker scheme that produced it.

Guest execution additionally requires compatible simulator, kernel, disk
image and checkpoint artifacts. Those are external dependencies. The frozen
guest input layout uses `/home/thu/TiC-SAT/weights`; an equivalent mount mapping
must preserve those accesses. Build reproduction does not validate a new guest
or simulator configuration.

## Host validation and numerical evidence

Use Python 3.11, matching the host-validation workflow, for the pinned host
test dependencies:

```bash
python -m pip install -r transformer_profiling/requirements-tests.txt
python -B tools/exp/validate_host.py --lint
```

The host checks cover tooling, temporary test fixtures, source contracts,
reporting, shell syntax, formatting and static checks. They neither build Arm
numerical oracles nor launch a simulator, and they do not regenerate frozen
scientific inputs.

The stage records retain reference tensor/final-output hashes in
`expected_outputs` and describe the scope of earlier validation. Full-model
and API comparisons used separately built functional copies. Exact numerical
outputs from the real-libm5 production executables remain unobserved. The
recorded checks establish bounded compatibility, not universal input
equivalence or timing significance. The reconstruction adds no measurements.

## Interpret profiling intervals

Read [the reporting guide](README.md) alongside the source that placed each
marker. Setup, allocation, initialization, console output and first-use work
can occur inside the measured regions. A label alone does not isolate a kernel.

In E12/E13, the QK-labelled interval includes streamed QK, softmax and SV work,
V transpose and enabled captures; the next softmax boundary is immediate and
the SV-labelled interval includes gather. Compare aggregate Attention across
this boundary change. Do not interpret those internal labels as the same
phases as E01-E11. Unchanged arithmetic also does not guarantee unchanged
region time when setup and buffer lifetimes move.

## General experiment tooling

[exp.sh](../tools/exp/exp.sh) is a separate configurable build, checkpoint,
launch and collection interface. Its [experiment table](../tools/exp/experiments.tsv)
is intentionally empty. [runner.conf.example](../tools/exp/runner.conf.example)
describes external paths and requires `I2CE_EXPERIMENT_ROOT` to name a new
absolute study directory.

The runner exposes `list`, `dryrun`, `build`, `checkpoint`, `run`, `submit`,
`status`, `results` and `collect`. Build and launch commands accept `--at` to
select source history; collection uses the current tooling and the run's saved
build configuration and region index. Its general notebook generation path
can change headers and weights and is not the fixed-stage reproduction path.

New simulation work requires its own finite registration, resource budget,
configuration checks and failure policy. The runner alone does not establish
those conditions. Neither this walkthrough nor publication schedules a run.
Simulator patching, input regeneration and garbage collection apply only to
new designated study resources, never the preserved archives or evidence.

Collection and reporting are separate operations. Use explicit report inputs
for existing observations; the [profiling guide](README.md) documents schemas,
formulas and optional comparison metadata.
