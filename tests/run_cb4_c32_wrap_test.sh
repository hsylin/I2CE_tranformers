#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${I2CE_TEST_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/i2ce-c32-wrap.XXXXXX")}
mkdir -p "$OUT"
CXX=${A64CXX:-aarch64-linux-gnu-g++}
QEMU=${I2CE_QEMU:-qemu-aarch64}
"$CXX" -std=c++17 -O2 -march=armv8-a+sve -static -DSIMD \
  -fsanitize=signed-integer-overflow -fno-sanitize-recover=all \
  -ffunction-sections -fdata-sections -Wl,--gc-sections \
  -I"$ROOT/Full_NN/inc" -I"$ROOT/tests/gemm_definitions_cb4" \
  "$ROOT/tests/gemm_cb4_c32_wrap_test.cc" "$ROOT/Full_NN/src/gemm_SVE.c" \
  "$ROOT/Full_NN/src/gemm_exec.c" -o "$OUT/wrap"
for BYTES in 16 32 64; do
  "$QEMU" -cpu "max,sve=on,sve-default-vector-length=$BYTES" "$OUT/wrap" < "$OUT/wrap"
done
