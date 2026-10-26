#!/usr/bin/env bash
# Build in an isolated source directory: the output is ./transformer.o.
# Flag variables select the existing run modes; A64CXX selects the compiler.

unset CC CXX CPATH LIBRARY_PATH LD_LIBRARY_PATH PKG_CONFIG_PATH CPPFLAGS LDFLAGS

DEFAULT_A64CXX="${CONDA_PREFIX:-}/bin/aarch64-conda-linux-gnu-g++"
if [ -z "${A64CXX:-}" ]; then
    if [ -x "$DEFAULT_A64CXX" ]; then
        A64CXX="$DEFAULT_A64CXX"
    elif command -v aarch64-linux-gnu-g++ >/dev/null 2>&1; then
        A64CXX="$(command -v aarch64-linux-gnu-g++)"
    elif command -v aarch64-conda-linux-gnu-g++ >/dev/null 2>&1; then
        A64CXX="$(command -v aarch64-conda-linux-gnu-g++)"
    else
        echo "No AArch64 C++ compiler found. Set A64CXX to its path." >&2
        exit 1
    fi
fi
export A64CXX
A64SYSROOT="$("$A64CXX" -print-sysroot 2>/dev/null || true)"
export A64SYSROOT

# Keep arguments separate so include and archive paths may contain spaces.
BUILD_DEFINES=()
BUILD_CXXFLAGS=()
BUILD_SVE_SOURCES=()
BUILD_LIBRARIES=()
for flag in RELOAD_WEIGHT USE_NOTEBOOK_GENERATED_WEIGHTS USE_CODEBOOK_GEMM \
    ENABLE_CODEBOOK_REFERENCE ENABLE_DEBUG_PRINT PROFILE_GEMM_ONLY \
    GEM5_PROFILE_REGIONS FULL_INTERLEAVED_PIPELINE USE_FP32_TRANSFORMER \
    DENSE_NO_SIMD_BASELINE SIMD; do
    default_value=0
    case "$flag" in
        RELOAD_WEIGHT|USE_NOTEBOOK_GENERATED_WEIGHTS) default_value=1 ;;
    esac
    variable_name="${flag}_FLAG"
    flag_value="${!variable_name:-$default_value}"
    printf '  %s=%s\n' "$variable_name" "$flag_value"
    if [ "$flag_value" = 1 ]; then
        BUILD_DEFINES+=("-D$flag")
    fi
done
if [ "${SIMD_FLAG:-0}" = 1 ]; then
    BUILD_CXXFLAGS+=(-march=armv8-a+sve)
    BUILD_SVE_SOURCES+=(Full_NN/src/gemm_SVE.c)
fi
CORE_NUM_VALUE="${CORE_NUM_FLAG:-1}"
printf '  CORE_NUM (build)=%s\n' "$CORE_NUM_VALUE"

if [ "${I2CE_USE_LIBM5_FLAG:-0}" = 1 ]; then
    if [ -z "${LIBM5_A:-}" ] || [ -z "${M5_INC:-}" ]; then
        if [ -z "${GEM5_ROOT:-}" ]; then
            echo "Direct profiling needs LIBM5_A and M5_INC, or GEM5_ROOT." >&2
            exit 1
        fi
        LIBM5_A="${LIBM5_A:-$GEM5_ROOT/util/m5/build/arm64/out/libm5.a}"
        M5_INC="${M5_INC:-$GEM5_ROOT/include}"
    fi
    if [ ! -f "$LIBM5_A" ]; then
        echo "libm5 archive not found: $LIBM5_A" >&2
        echo "Build it with tools/exp/exp.sh build-libm5." >&2
        exit 1
    fi
    if [ ! -f "$M5_INC/gem5/m5ops.h" ]; then
        echo "gem5/m5ops.h not found under: $M5_INC" >&2
        exit 1
    fi
    BUILD_DEFINES+=(-DI2CE_USE_LIBM5)
    BUILD_CXXFLAGS+=("-I$M5_INC")
    # Static libraries follow the objects that reference their symbols.
    BUILD_LIBRARIES+=("$LIBM5_A")
fi
printf '  I2CE_USE_LIBM5_FLAG=%s\n' "${I2CE_USE_LIBM5_FLAG:-0}"

"$A64CXX" -std=c++17 -O2 -Wall \
    "${BUILD_CXXFLAGS[@]}" \
    transformer.cpp transformer_layers/*.cc \
    Full_NN/src/gemm_exec.c "${BUILD_SVE_SOURCES[@]}" \
    accelerator/smm_gem.cpp accelerator/systolic_m2m.cc \
    "${BUILD_DEFINES[@]}" \
    -I. -IFull_NN/inc -IFull_NN/gemm_definitions \
    -DSA_SIZE=4 -DDEVELOP -DCORE_NUM="$CORE_NUM_VALUE" \
    -fopenmp -static -o transformer.o "${BUILD_LIBRARIES[@]}" \
    -Wl,--start-group -lgomp -ldl -Wl,--end-group -pthread
