#!/usr/bin/env bash
# Run in the Linux build environment (server or ARM64 Docker), not macOS.
#
# The AArch64+SVE test binaries need an SVE-capable executor. Two are supported:
#
#   qemu  - qemu-aarch64 user mode, >= 3.0 (SVE landed in 3.0, 2018) and new
#           enough to accept `-cpu max,sve=on`.
#   gem5  - gem5 SE mode, using the same SVE implementation as the performance
#           experiments. Needs no extra software when gem5 is already built.
#
# Selection is automatic; force one with I2CE_TEST_BACKEND=qemu|gem5.
#
# This script previously aborted with no output at all when qemu was unusable
# (CentOS 7 ships qemu 2.0, whose `--version` prints usage and returns 1, which
# `set -e` turned into a silent exit). Every prerequisite is now probed
# explicitly and every failure names what is missing.
set -euo pipefail

die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

I2CE_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
I2CE_TEST_OUT="$(mktemp -d "${TMPDIR:-/tmp}/i2ce-widening-test.XXXXXX")" \
  || die "could not create a temporary directory under ${TMPDIR:-/tmp}. If / is full, set TMPDIR to a writable filesystem."

I2CE_TEST_CXX="${A64CXX:-}"
if [[ -z "$I2CE_TEST_CXX" ]]; then
  if [[ -x "${CONDA_PREFIX:-}/bin/aarch64-conda-linux-gnu-g++" ]]; then
    I2CE_TEST_CXX="$CONDA_PREFIX/bin/aarch64-conda-linux-gnu-g++"
  elif command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
    I2CE_TEST_CXX="$(command -v aarch64-linux-gnu-g++)"
  elif [[ "$(uname -m)" == aarch64 ]] && command -v g++ >/dev/null 2>&1; then
    I2CE_TEST_CXX="$(command -v g++)"
  else
    die 'no AArch64 C++ compiler found; set A64CXX to its path.'
  fi
fi

I2CE_TEST_FLAGS=(-std=c++17 -O2 -Wall -Wextra -march=armv8-a+sve -DSIMD
  -I"$I2CE_TEST_ROOT/Full_NN/inc")

# ---------------------------------------------------------------- backend
I2CE_TEST_QEMU="${QEMU_AARCH64:-qemu-aarch64}"
I2CE_GEM5_BIN="${I2CE_GEM5_BIN:-$HOME/i2ce/gem5-tianrui/build/ARM/gem5.fast}"
I2CE_GEM5_ROOT="${I2CE_GEM5_ROOT:-$(cd "$(dirname "$I2CE_GEM5_BIN")/../.." 2>/dev/null && pwd || true)}"
I2CE_TEST_BACKEND="${I2CE_TEST_BACKEND:-auto}"

# Compile and run a minimal SVE program: the only trustworthy check that an
# executor can actually issue SVE at the width the tests assume.
qemu_supports_sve() {
  command -v "$I2CE_TEST_QEMU" >/dev/null 2>&1 || { echo "  qemu: '$I2CE_TEST_QEMU' not on PATH"; return 1; }
  # qemu < 3.1 spells it -version and returns 1 for --version, so try both.
  local ver
  ver="$("$I2CE_TEST_QEMU" --version 2>/dev/null | head -1)" \
    || ver="$("$I2CE_TEST_QEMU" -version 2>/dev/null | head -1)" || ver="unknown"
  printf '#include <arm_sve.h>\n#include <stdio.h>\nint main(void){printf("%%llu\\n",(unsigned long long)svcntb());return 0;}\n' \
    > "$I2CE_TEST_OUT/svechk.c"
  "$I2CE_TEST_CXX" -std=c++17 -O2 -march=armv8-a+sve -static \
    -x c++ "$I2CE_TEST_OUT/svechk.c" -o "$I2CE_TEST_OUT/svechk" 2>/dev/null \
    || { echo "  qemu: SVE probe failed to compile"; return 1; }
  local got
  got="$("$I2CE_TEST_QEMU" -cpu max,sve=on,sve-default-vector-length=16 \
          "$I2CE_TEST_OUT/svechk" 2>/dev/null || true)"
  if [[ "$got" != "16" ]]; then
    echo "  qemu: cannot run SVE at 128-bit (got '${got:-<nothing>}'); version: ${ver:-unknown}"
    echo "        SVE needs qemu >= 3.0 with '-cpu max,sve=on'."
    return 1
  fi
  return 0
}

gem5_usable() {
  [[ -x "$I2CE_GEM5_BIN" ]] || { echo "  gem5: no executable at $I2CE_GEM5_BIN (set I2CE_GEM5_BIN)"; return 1; }
  [[ -f "$I2CE_GEM5_ROOT/configs/example/arm/starter_se.py" ]] \
    || { echo "  gem5: starter_se.py not found under $I2CE_GEM5_ROOT (set I2CE_GEM5_ROOT)"; return 1; }
  return 0
}

QEMU_WHY=""; GEM5_WHY=""
case "$I2CE_TEST_BACKEND" in
  qemu) qemu_supports_sve || die "I2CE_TEST_BACKEND=qemu but qemu cannot run SVE." ;;
  gem5) gem5_usable       || die "I2CE_TEST_BACKEND=gem5 but gem5 is unusable." ;;
  auto)
    if QEMU_WHY="$(qemu_supports_sve)"; then
      I2CE_TEST_BACKEND=qemu
    elif GEM5_WHY="$(gem5_usable)"; then
      I2CE_TEST_BACKEND=gem5
      printf 'note: qemu unusable, falling back to gem5 SE mode.\n%s\n' "$QEMU_WHY"
    else
      printf '%s\n%s\n' "$QEMU_WHY" "$GEM5_WHY" >&2
      die "no SVE-capable executor. Install qemu >= 3.0, or point I2CE_GEM5_BIN at a built gem5.fast."
    fi ;;
  *) die "I2CE_TEST_BACKEND must be qemu, gem5 or auto (got '$I2CE_TEST_BACKEND')" ;;
esac
printf 'backend: %s\n' "$I2CE_TEST_BACKEND"

# Run one test binary and require a PASS line. gem5 always exits 0 regardless of
# the guest, reporting the guest status as "... thread context (N)" instead, so
# the result is taken from the program's own output in both backends.
# $1=binary  $2=label  $3=vector-length-in-bytes  $4..=program args
run_case() {
  local bin="$1" label="$2" vl_bytes="$3"; shift 3
  local log="$I2CE_TEST_OUT/$label.log"
  if [[ "$I2CE_TEST_BACKEND" == qemu ]]; then
    "$I2CE_TEST_QEMU" -cpu "max,sve=on,sve-default-vector-length=$vl_bytes" \
      "$bin" "$@" > "$log" 2>&1 || true
  else
    local vl_q=$(( vl_bytes / 16 )) extra=()
    # SE mode allocates a plain System, so the FS parameter system.sve_vl does
    # not exist; the SE equivalent lives on the ISA object and defaults to 1.
    if [[ "$vl_q" != 1 ]]; then
      extra=(--param "system.cpu_cluster.cpus[0].isa[0].sve_vl_se=$vl_q")
    fi
    # starter_se.py takes each command as ONE positional string and splits it
    # with shlex; passing guest arguments as separate argv entries makes its own
    # argparse reject them ("unrecognized arguments: --helper-only").
    local cmdstr="$bin"
    [[ $# -gt 0 ]] && cmdstr="$bin $*"
    ( cd "$I2CE_GEM5_ROOT" && "$I2CE_GEM5_BIN" -d "$I2CE_TEST_OUT/gem5_$label" \
        --stats-file=stats.txt configs/example/arm/starter_se.py \
        ${extra[@]+"${extra[@]}"} --cpu=atomic "$cmdstr" ) > "$log" 2>&1 || true
  fi
  if grep -q '^FAIL' "$log" || ! grep -q '^PASS' "$log"; then
    echo "--- $label output ---" >&2; tail -20 "$log" >&2
    die "$label did not report PASS (full log: $log)"
  fi
  grep '^PASS' "$log"
}

"$I2CE_TEST_CXX" --version > "$I2CE_TEST_OUT/compiler.txt"

# Default generated header: CB8 / N_SVE_REG_CB_2.
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -I"$I2CE_TEST_ROOT/Full_NN/gemm_definitions" -static \
  "$I2CE_TEST_ROOT/tests/gemm_widening_test.cc" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_SVE.c" \
  -o "$I2CE_TEST_OUT/gemm_widening_test"

# CB4 / N_SVE_REG_CB_1: CB_SIZE selects the codebook-register path, so covering
# CB4 needs its own header rather than a different bits_per_cb argument.
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -I"$I2CE_TEST_ROOT/tests/gemm_definitions_cb4" -static \
  "$I2CE_TEST_ROOT/tests/gemm_widening_test.cc" \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_SVE.c" \
  -o "$I2CE_TEST_OUT/gemm_widening_test_cb4"

# Generate assembly from the actual production translation unit, not a copied
# implementation. Inspect the helper and all three callers in this file.
"$I2CE_TEST_CXX" "${I2CE_TEST_FLAGS[@]}" -I"$I2CE_TEST_ROOT/Full_NN/gemm_definitions" -S -fverbose-asm \
  "$I2CE_TEST_ROOT/Full_NN/src/gemm_exec.c" \
  -o "$I2CE_TEST_OUT/gemm_exec.s"
# Also ensure the new SVE include/helper stays excluded from scalar builds.
"$I2CE_TEST_CXX" -std=c++17 -O2 -Wall \
  -I"$I2CE_TEST_ROOT/Full_NN/inc" \
  -I"$I2CE_TEST_ROOT/Full_NN/gemm_definitions" \
  -c "$I2CE_TEST_ROOT/Full_NN/src/gemm_exec.c" \
  -o "$I2CE_TEST_OUT/gemm_exec_scalar.o"

run_case "$I2CE_TEST_OUT/gemm_widening_test"     sve128      16
run_case "$I2CE_TEST_OUT/gemm_widening_test_cb4" sve128_cb4  16
# Only the flat helper is tested at VL256; the generated model remains VL128.
run_case "$I2CE_TEST_OUT/gemm_widening_test"     sve256_help 32 --helper-only

sha256sum "$I2CE_TEST_OUT/gemm_widening_test" "$I2CE_TEST_OUT/gemm_widening_test_cb4" \
  > "$I2CE_TEST_OUT/binary.sha256"
printf '\nFUNCTIONAL TESTS PASSED (%s). Assembly and logs: %s\n' \
  "$I2CE_TEST_BACKEND" "$I2CE_TEST_OUT"
printf 'Inspect widening assembly: %s/gemm_exec.s\n' "$I2CE_TEST_OUT"
