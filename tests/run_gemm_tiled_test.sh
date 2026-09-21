#!/usr/bin/env bash
# Correctness test for the tiled four-learner shared-index compact GEMM.
# Runs in the aarch64 build environment (aarch64 g++ + qemu-aarch64), the same
# way tests/run_gemm_widening_test.sh does. Verifies the tiled SVE kernel
# against the scalar reference at SVE vector lengths 128, 256 and 512 bits.
set -euo pipefail

I2CE_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2CE_TEST_OUT="$(mktemp -d "${TMPDIR:-/tmp}/i2ce-tiled-test.XXXXXX")"

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

"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -static \
  "$I2CE_TEST_ROOT/tests/gemm_tiled_test.cc" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_exec.c" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_SVE.c" \
  -o "$I2CE_TEST_OUT/gemm_tiled_test"

status=0
# sve-default-vector-length is in bytes: 16=128-bit, 32=256-bit, 64=512-bit.
for vlbytes in 16 32 64; do
  "$I2CE_TEST_QEMU" -cpu "max,sve=on,sve-default-vector-length=$vlbytes" \
    "$I2CE_TEST_OUT/gemm_tiled_test" | tee -a "$I2CE_TEST_OUT/results.log" || status=1
done

echo
if [[ $status -eq 0 ]]; then
  printf 'TILED GEMM TESTS PASSED. Logs: %s\n' "$I2CE_TEST_OUT"
else
  printf 'TILED GEMM TESTS FAILED. Logs: %s\n' "$I2CE_TEST_OUT" >&2
fi
exit $status
