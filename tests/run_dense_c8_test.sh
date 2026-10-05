#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$(mktemp -d "${TMPDIR:-/tmp}/i2ce-dense-c8.XXXXXX")
CXX=${A64CXX:-$HOME/i2ce/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-g++}
GEM5=${I2CE_GEM5_ROOT:-$HOME/i2ce/gem5-tianrui}
"$CXX" -std=c++17 -O2 -Wall -march=armv8-a+sve -DSIMD -static \
    -ffunction-sections -fdata-sections -Wl,--gc-sections \
    -I"$ROOT/Full_NN/inc" -I"$ROOT/tests/gemm_definitions_cb4" \
    -I"$ROOT/transformer_layers" "$ROOT/tests/dense_c8_test.cc" \
    "$ROOT/transformer_layers/interleavedPipeline.cc" "$ROOT/transformer_layers/softmax.cc" \
    "$ROOT/Full_NN/src/gemm_SVE.c" -o "$OUT/oracle"
for VL in 1 2 4; do
    "$GEM5/build/ARM/gem5.fast" -d "$OUT/vl$VL" \
        "$GEM5/configs/example/arm/starter_se.py" --cpu=atomic \
        --param "system.cpu_cluster.cpus[0].isa[0].sve_vl_se=$VL" \
        "$OUT/oracle" > "$OUT/vl$VL.log" 2>&1
    if grep -q '^FAIL' "$OUT/vl$VL.log" || ! grep -q '^PASS' "$OUT/vl$VL.log"; then
        cat "$OUT/vl$VL.log"; exit 1
    fi
    grep '^PASS' "$OUT/vl$VL.log"
done
echo "Logs: $OUT"
