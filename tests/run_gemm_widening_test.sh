#!/usr/bin/env bash
# Run in the Linux build environment (server or ARM64 Docker), not macOS.
set -euo pipefail

I2CE_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2CE_TEST_OUT="$(mktemp -d "${TMPDIR:-/tmp}/i2ce-widening-test.XXXXXX")"
I2CE_TEST_CXX="${A64CXX:-}"
if [[ -z "$I2CE_TEST_CXX" ]]; then
  if [[ -x "${CONDA_PREFIX:-}/bin/aarch64-conda-linux-gnu-g++" ]]; then
    I2CE_TEST_CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
  elif command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    I2CE_TEST_CXX="$(command -v aarch64-linux-gnu-g++)"
  elif [[ "$(uname -m)" == aarch64 ]] && command -v g++ >/dev/null 2>&1; then
    I2CE_TEST_CXX="$(command -v g++)"
  else
    echo 'ERROR: set A64CXX to the AArch64 C++ compiler.' >&2
    exit 1
  fi
fi
I2CE_TEST_QEMU="${QEMU_AARCH64:-qemu-aarch64}"
command -v "$I2CE_TEST_QEMU" >/dev/null 2>&1 || {
  echo 'ERROR: qemu-aarch64 is required; set QEMU_AARCH64 if needed.' >&2
  exit 1
}

I2CE_TEST_FLAGS=(-std=c++17 -O2 -Wall -Wextra -march=armv8-a+sve -DSIMD
  -I"$I2CE_TEST_ROOT/Full_NN/inc"
  -I"$I2CE_TEST_ROOT/Full_NN/gemm_definitions")
"$I2CE_TEST_CXX" --version > "$I2CE_TEST_OUT/compiler.txt"
"$I2CE_TEST_QEMU" --version > "$I2CE_TEST_OUT/qemu.txt"
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -static \
  "$I2CE_TEST_ROOT/tests/gemm_widening_test.cc" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_SVE.c" \
  -o "$I2CE_TEST_OUT/gemm_widening_test"

# Generate assembly from the actual production translation unit, not a copied
# implementation. Inspect the helper and all three callers in this file.
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -S -fverbose-asm \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_exec.c" \
  -o "$I2CE_TEST_OUT/gemm_exec.s"
# Also ensure the new SVE include/helper stays excluded from scalar builds.
"$I2CE_TEST_CXX" -std=c++17 -O2 -Wall \
  -I"$I2CE_TEST_ROOT/Full_NN/inc" \
  -I"$I2CE_TEST_ROOT/Full_NN/gemm_definitions" \
  -c "$I2CE_TEST_ROOT/Full_NN/src/gemm_exec.c" \
  -o "$I2CE_TEST_OUT/gemm_exec_scalar.o"

"$I2CE_TEST_QEMU" -cpu max,sve=on,sve-default-vector-length=16 \
  "$I2CE_TEST_OUT/gemm_widening_test" | tee "$I2CE_TEST_OUT/sve128.log"

# Same sources, but compiled against a CB4 generated header so the kernels take
# the N_SVE_REG_CB_1 codebook path with 2-bit packed indexes. The committed
# header is CB8/N_SVE_REG_CB_2, so this is the only way to cover the CB4 target
# configuration -- changing bits_per_cb alone would test the wrong path.
I2CE_TEST_FLAGS_CB4=(-std=c++17 -O2 -Wall -Wextra -march=armv8-a+sve -DSIMD
  -I"$I2CE_TEST_ROOT/Full_NN/inc"
  -I"$I2CE_TEST_ROOT/tests/gemm_definitions_cb4")
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS_CB4[@]}" -static \
  "$I2CE_TEST_ROOT/tests/gemm_widening_test.cc" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_SVE.c" \
  -o "$I2CE_TEST_OUT/gemm_widening_test_cb4"
"$I2CE_TEST_QEMU" -cpu max,sve=on,sve-default-vector-length=16 \
  "$I2CE_TEST_OUT/gemm_widening_test_cb4" | tee "$I2CE_TEST_OUT/sve128_cb4.log"
# Only the flat helper is tested at VL256; the generated model remains VL128.
"$I2CE_TEST_QEMU" -cpu max,sve=on,sve-default-vector-length=32 \
  "$I2CE_TEST_OUT/gemm_widening_test" --helper-only \
  | tee "$I2CE_TEST_OUT/sve256_helper.log"
sha256sum "$I2CE_TEST_OUT/gemm_widening_test" > "$I2CE_TEST_OUT/binary.sha256"
printf '\nFUNCTIONAL TESTS PASSED. Assembly and logs: %s\n' "$I2CE_TEST_OUT"
printf 'Inspect widening assembly: %s/gemm_exec.s\n' "$I2CE_TEST_OUT"
