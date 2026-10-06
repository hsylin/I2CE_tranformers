#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${I2CE_TEST_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/i2ce-cache-correctness.XXXXXX")}
mkdir -p "$OUT"
CXX=${A64CXX:-$HOME/i2ce/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++}
QEMU=${I2CE_QEMU:-/home/thu/opt/qemu-sve/bin/qemu-aarch64}
"$CXX" -std=c++17 -O2 -Wall -Wextra -march=armv8-a+sve -static \
  -ffunction-sections -fdata-sections -Wl,--gc-sections -DI2CE_TEST_CB4_CACHE \
  -I"$ROOT/Full_NN/inc" -I"$ROOT/tests/gemm_definitions_cb4" \
  "$ROOT/tests/gemm_cb4_cache_test.cc" "$ROOT/Full_NN/src/gemm_SVE.c" -o "$OUT/oracle"
for BYTES in 16 32 64; do
  "$QEMU" -cpu "max,sve=on,sve-default-vector-length=$BYTES" "$OUT/oracle" < "$OUT/oracle"
done
# Compile-time dispatch controls. These binaries only check rejection; they
# are not performance measurements or claims about arbitrary fallback configs.
for MODE in cb16 four_learners; do
  mkdir -p "$OUT/$MODE"
  if [[ "$MODE" == cb16 ]]; then
    sed -e 's/CB_SIZE         4/CB_SIZE         16/' \
        -e 's/BITS_PER_CB     2/BITS_PER_CB     4/' \
        -e 's/IDX_MASK        0b11/IDX_MASK        0b1111/' \
        -e 's/N_SVE_REG_CB_1  1/N_SVE_REG_CB_4  1/' \
        "$ROOT/tests/gemm_definitions_cb4/codebooks_def.h" > "$OUT/$MODE/codebooks_def.h"
  else
    sed 's/N_LEARNERS      2/N_LEARNERS      4/' \
        "$ROOT/tests/gemm_definitions_cb4/codebooks_def.h" > "$OUT/$MODE/codebooks_def.h"
  fi
  "$CXX" -std=c++17 -O2 -Wall -Wextra -march=armv8-a+sve -static \
    -ffunction-sections -fdata-sections -Wl,--gc-sections -DI2CE_TEST_CB4_CACHE \
    -I"$ROOT/Full_NN/inc" -I"$OUT/$MODE" \
    "$ROOT/tests/gemm_cb4_cache_test.cc" "$ROOT/Full_NN/src/gemm_SVE.c" -o "$OUT/$MODE/oracle"
  "$QEMU" -cpu max,sve=on,sve-default-vector-length=16 "$OUT/$MODE/oracle" < "$OUT/$MODE/oracle"
done
echo "Correctness-only artifacts: $OUT"
