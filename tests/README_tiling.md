# Tiled four-learner shared-index compact GEMM

First-stage tiling of the four-learner shared-index compact GEMM: a
loop-reorder plus a small register tile. Correctness only — no gem5
timing was measured for this stage.

## What changed

`sve_gemm_tile_compact_int8_interleaved_4Learners_same_seq` (in
`Full_NN/src/gemm_SVE.c`) and its driver
`gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled[_ex]`
(in `Full_NN/src/gemm_exec.c`) replace the loop structure of the
existing `..._4Learners_same_seq_ex` path:

- sequence-row tiles outer, output features inner, so the activation
  panel is reused across all output features instead of the whole
  activation matrix being re-streamed once per output feature;
- a two-row register tile: both sequence rows share each packed-index
  decode and the four learner codebook lookups;
- only the current `2 x Kc` activation tile is widened to int32 (kept
  within an ~16 KB L1 budget) instead of widening the whole matrix;
- outputs are written row-contiguously.

`CodebookDense::computeInterleaved4LearnersToInt8` (the fully
interleaved production path) calls the tiled driver.
`computeInterleaved4Learners` (the uint32 in/out path) is unchanged and
left for a later stage. The public API and the caller-owned pre-widened
codebook / activation-workspace parameters are preserved.

Vector length is resolved at run time (`svcntw()`), so one build is
correct at SVE VL 128/256/512. This generalizes the compile-time
`N_SVE_REG_CB_*` register-capacity gate of the older wrappers; codebooks
that do not fit in four registers per learner fall back to the scalar
path. The numerical result is identical to
`gemm_exec_compact_int_interleaved_4Learners_same_seq`.

## Running the correctness test

Needs an AArch64 C++ compiler and `qemu-aarch64` (same environment as
`run_gemm_widening_test.sh`):

```
A64CXX=aarch64-linux-gnu-g++ QEMU_AARCH64=qemu-aarch64 \
  bash tests/run_gemm_tiled_test.sh
```

It builds `tests/gemm_tiled_test.cc` with `gemm_exec.c` + `gemm_SVE.c`
and runs it under qemu at `sve-default-vector-length` 16/32/64 bytes
(SVE 128/256/512). Each run compares the tiled kernel (public wrapper
and the `_ex` variant with a caller-owned pre-widened codebook and
workspace) against the scalar reference over codebook sizes 2..32 and
shapes whose sequence/K/N are not divisible by the tile sizes, including
K large enough to span several K blocks.

## Verified

QEMU functional correctness only (exact agreement with the scalar
reference), 50/50 cases at each of VL 128, 256, 512:

```
VL=128: ALL PASSED (50/50 cases passed)
VL=256: ALL PASSED (50/50 cases passed)
VL=512: ALL PASSED (50/50 cases passed)
```

No runtime speedup is claimed here: gem5 was not run for this stage.
Any performance numbers must come from gem5 full-system runs, not from
QEMU (which is functional emulation, not a timing model).
