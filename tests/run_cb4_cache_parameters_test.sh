#!/usr/bin/env bash
# Correctness only: no timer and no simulator performance run.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${I2CE_TEST_OUT:-$(mktemp -d "${TMPDIR:-/tmp}/i2ce-cache-parameters.XXXXXX")}
mkdir -p "$OUT"
CXX=${A64CXX:-$HOME/i2ce/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++}
QEMU=${I2CE_QEMU:-/home/thu/opt/qemu-sve/bin/qemu-aarch64}
COMMON=(-std=c++17 -O2 -march=armv8-a+sve -static
  -ffunction-sections -fdata-sections -Wl,--gc-sections
  -DI2CE_TEST_CB4_CACHE -DI2CE_CB4_CACHE_CONFIG_REQUESTED=1
  -I"$ROOT/Full_NN/inc" -I"$ROOT/tests/gemm_definitions_cb4")
for GEOMETRY in default small uneven; do
  DEFS=()
  if [[ "$GEOMETRY" == small ]]; then
    DEFS=(-DI2CE_CACHE_S1=8 -DI2CE_CACHE_O1=16 -DI2CE_CACHE_K1=64
      -DI2CE_CACHE_S2=64 -DI2CE_CACHE_O2=64 -DI2CE_CACHE_K2=256)
  elif [[ "$GEOMETRY" == uneven ]]; then
    # No requirement that the inner tile divide the outer tile.
    DEFS=(-DI2CE_CACHE_S1=12 -DI2CE_CACHE_O1=20 -DI2CE_CACHE_K1=48
      -DI2CE_CACHE_S2=28 -DI2CE_CACHE_O2=36 -DI2CE_CACHE_K2=80)
  fi
  "$CXX" "${COMMON[@]}" ${DEFS[@]+"${DEFS[@]}"} -DI2CE_TEST_CB4_CACHE_C32 "$ROOT/tests/gemm_cb4_cache_test.cc" \
    "$ROOT/Full_NN/src/gemm_SVE.c" -o "$OUT/$GEOMETRY"
  "$QEMU" -cpu max,sve=on,sve-default-vector-length=16 \
    "$OUT/$GEOMETRY" < "$OUT/$GEOMETRY"
  "$CXX" "${COMMON[@]}" ${DEFS[@]+"${DEFS[@]}"} -DSIMD \
    -I"$ROOT/transformer_layers" "$ROOT/tests/gemm_cb4_cache_consumer_test.cc" \
    "$ROOT/transformer_layers/codebookDense.cc" "$ROOT/Full_NN/src/gemm_SVE.c" \
    "$ROOT/Full_NN/src/gemm_exec.c" -o "$OUT/$GEOMETRY-consumer"
  "$QEMU" -cpu max,sve=on,sve-default-vector-length=16 \
    "$OUT/$GEOMETRY-consumer" < "$OUT/$GEOMETRY-consumer"
done

printf '#include <gemm_cb4_cache_config.h>\nint main() { return 0; }\n' > "$OUT/config.cc"
for BAD in '-DI2CE_CACHE_S1=0' '-DI2CE_CACHE_O1=-4' '-DI2CE_CACHE_K1=17' \
  '-DI2CE_CACHE_S1=256' '-DI2CE_CACHE_O2=65536' '-DI2CE_CACHE_S1=5' \
  '-DI2CE_CACHE_K2=65536' \
  '-DI2CE_CACHE_S2=65532 -DI2CE_CACHE_O2=65532 -DI2CE_CACHE_K2=65520'; do
  # BAD values are fixed test literals, not shell input.
  read -ra DEFS <<< "$BAD"
  if "$CXX" -std=c++17 -I"$ROOT/Full_NN/inc" ${DEFS[@]+"${DEFS[@]}"} -fsyntax-only \
      "$OUT/config.cc" > "$OUT/rejected.log" 2>&1; then
    echo "FAIL: illegal geometry compiled: $BAD" >&2; exit 1
  fi
  grep -q 'CB4' "$OUT/rejected.log" || { cat "$OUT/rejected.log"; exit 1; }
done
# Explicit configuration must fail instead of silently selecting a fallback.
for MODE in cb16 separate_indices sve256; do
  mkdir -p "$OUT/$MODE"
  case "$MODE" in
    cb16) CHANGE='s/CB_SIZE         4/CB_SIZE         16/' ;;
    separate_indices) CHANGE='s/SAME_SEQ    1/SAME_SEQ    0/' ;;
    sve256) CHANGE='s/N_SVE_BYTE      16/N_SVE_BYTE      32/' ;;
  esac
  sed "$CHANGE" "$ROOT/tests/gemm_definitions_cb4/codebooks_def.h" > "$OUT/$MODE/codebooks_def.h"
  if "$CXX" -std=c++17 -march=armv8-a+sve -DI2CE_CB4_CACHE_CONFIG_REQUESTED=1 \
      -I"$ROOT/Full_NN/inc" -I"$OUT/$MODE" -fsyntax-only "$ROOT/Full_NN/src/gemm_SVE.c" \
      > "$OUT/$MODE/rejected.log" 2>&1; then
    echo "FAIL: incompatible generated configuration compiled: $MODE" >&2; exit 1
  fi
  grep -q 'Explicit cache tiles require' "$OUT/$MODE/rejected.log" || exit 1
done
echo "PASS 3 geometries, 8 illegal geometries and 3 incompatible dispatch configurations; artifacts: $OUT"
