#pragma once
#include "run_mode_config.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#if CFG_SIMD
#include <arm_sve.h>
#include <gemm_SVE.h>
#endif

namespace attention_sdot {
struct Counters {
    uint64_t calls = 0, core_calls = 0, sdot = 0, allocations = 0, frees = 0,
             requested = 0, max_request = 0;
    uint64_t a_packs = 0, b_packs = 0, a_bytes = 0, b_bytes = 0, c_initialized = 0,
             fallbacks = 0, failures = 0;
};
static Counters counters;
static __attribute__((destructor)) void report() {
    std::fprintf(
        stderr,
        "[attention-sdot] version=1 calls=%llu core_calls=%llu sdot_source_count=%llu "
        "allocations=%llu frees=%llu requested_bytes=%llu max_request_bytes=%llu "
        "A_packs=%llu B_packs=%llu A_assigned_bytes=%llu B_assigned_bytes=%llu "
        "C_initialized_bytes=%llu fallbacks=%llu failures=%llu retained=%llu "
        "cleanup=per_call_ROI\n",
        (unsigned long long)counters.calls, (unsigned long long)counters.core_calls,
        (unsigned long long)counters.sdot, (unsigned long long)counters.allocations,
        (unsigned long long)counters.frees, (unsigned long long)counters.requested,
        (unsigned long long)counters.max_request, (unsigned long long)counters.a_packs,
        (unsigned long long)counters.b_packs, (unsigned long long)counters.a_bytes,
        (unsigned long long)counters.b_bytes,
        (unsigned long long)counters.c_initialized,
        (unsigned long long)counters.fallbacks, (unsigned long long)counters.failures,
        (unsigned long long)(counters.allocations - counters.frees));
}
#ifdef I2CE_TEST_ATTENTION_SDOT
static uint32_t *observed = nullptr;
static void (*partial_observer)(const uint32_t *, size_t, size_t, size_t, size_t,
                                size_t, size_t, size_t) = nullptr;
#endif
static size_t padded(size_t count, size_t quantum) {
    return (count + quantum - 1) / quantum * quantum;
}
static bool multiply(size_t left, size_t right, size_t &product) {
    if (left && right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}
static bool overlap(const void *left, size_t left_bytes, const void *right,
                    size_t right_bytes) {
    const uintptr_t left_address = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_address = reinterpret_cast<uintptr_t>(right);
    return left_bytes && right_bytes &&
           (left_address <= right_address ? right_address - left_address < left_bytes
                                          : left_address - right_address < right_bytes);
}
static bool span(const void *data, size_t bytes) {
    return !bytes || (data && reinterpret_cast<uintptr_t>(data) <=
                                  std::numeric_limits<uintptr_t>::max() - bytes);
}
struct Extents {
    size_t left_bytes, right_bytes, output_bytes;
};
static Extents extents(const int8_t *left, const int8_t *right, size_t rows,
                       size_t columns, size_t features, int8_t *output) {
    Extents result;
    size_t elements;
    if (rows > UINT32_MAX || columns > UINT32_MAX || features > UINT32_MAX ||
        !multiply(rows, features, elements) ||
        !multiply(elements, 2, result.left_bytes) ||
        !multiply(columns, features, elements) ||
        !multiply(elements, 2, result.right_bytes) ||
        !multiply(rows, columns, elements) ||
        !multiply(elements, 2, result.output_bytes) || !span(left, result.left_bytes) ||
        !span(right, result.right_bytes) || !span(output, result.output_bytes)) {
        throw std::invalid_argument("Attention operand extent");
    }
    return result;
}
// One independent aligned allocation. Placement array starts uint32 lifetimes;
// packed bytes access its character representation, C accesses uint32 objects.
struct Arena {
    void *allocation;
    uint32_t *words;
    size_t bytes;
    explicit Arena(size_t allocation_bytes)
        : allocation(nullptr), words(nullptr), bytes(allocation_bytes) {
        if (!allocation_bytes || allocation_bytes % 64) {
            throw std::invalid_argument("Attention scratch alignment");
        }
        try {
            allocation = ::operator new(allocation_bytes, std::align_val_t(64));
        } catch (...) {
            ++counters.failures;
            throw;
        }
        words = ::new (allocation) uint32_t[allocation_bytes / sizeof(uint32_t)];
        ++counters.allocations;
        counters.requested += allocation_bytes;
        counters.max_request =
            std::max(counters.max_request, (uint64_t)allocation_bytes);
    }
    ~Arena() {
        if (allocation) {
            ::operator delete(allocation, std::align_val_t(64));
            ++counters.frees;
        }
    }
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;
    int8_t *data() {
        return reinterpret_cast<int8_t *>(words);
    }
};
// Final SDOT panels: row-major A and output-group4/K4/column4 B. All bytes
// read by the fixed kernel, including missing rows/columns/K, are assigned.
static void pack_sdot(const int8_t *source, size_t logical_rows,
                      size_t logical_features, size_t row_base, size_t feature_base,
                      size_t rows, size_t padded_features, int8_t *destination,
                      bool pack_right) {
    size_t learner_plane_bytes = rows * padded_features;
#if CFG_SIMD
    const svbool_t all = svptrue_b8();
    if (!pack_right) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t feature = 0; feature < padded_features; feature += 16u) {
                size_t valid = (row_base + row < logical_rows &&
                                feature_base + feature < logical_features)
                                   ? std::min(size_t(16),
                                              logical_features - feature_base - feature)
                                   : 0;
                svint8_t x0 = svdup_s8(0), x1 = svdup_s8(0);
                if (valid) {
                    svint8x2_t x =
                        svld2_s8(svwhilelt_b8(size_t(0), valid),
                                 source + ((row_base + row) * logical_features +
                                           feature_base + feature) *
                                              2u);
                    x0 = svget2_s8(x, 0);
                    x1 = svget2_s8(x, 1);
                }
                svst1_s8(all, destination + row * padded_features + feature, x0);
                svst1_s8(all,
                         destination + learner_plane_bytes + row * padded_features +
                             feature,
                         x1);
            }
        }
    } else {
        for (size_t row = 0; row < rows; row += 4u) {
            for (size_t feature = 0; feature < padded_features; feature += 16u) {
                size_t valid = feature_base + feature < logical_features
                                   ? std::min(size_t(16),
                                              logical_features - feature_base - feature)
                                   : 0;
                svint8_t r00 = svdup_s8(0), r01 = svdup_s8(0);
                if (valid && row_base + row + 0u < logical_rows) {
                    svint8x2_t x =
                        svld2_s8(svwhilelt_b8(size_t(0), valid),
                                 source + ((row_base + row + 0u) * logical_features +
                                           feature_base + feature) *
                                              2u);
                    r00 = svget2_s8(x, 0);
                    r01 = svget2_s8(x, 1);
                }
                svint8_t r10 = svdup_s8(0), r11 = svdup_s8(0);
                if (valid && row_base + row + 1u < logical_rows) {
                    svint8x2_t x =
                        svld2_s8(svwhilelt_b8(size_t(0), valid),
                                 source + ((row_base + row + 1u) * logical_features +
                                           feature_base + feature) *
                                              2u);
                    r10 = svget2_s8(x, 0);
                    r11 = svget2_s8(x, 1);
                }
                svint8_t r20 = svdup_s8(0), r21 = svdup_s8(0);
                if (valid && row_base + row + 2u < logical_rows) {
                    svint8x2_t x =
                        svld2_s8(svwhilelt_b8(size_t(0), valid),
                                 source + ((row_base + row + 2u) * logical_features +
                                           feature_base + feature) *
                                              2u);
                    r20 = svget2_s8(x, 0);
                    r21 = svget2_s8(x, 1);
                }
                svint8_t r30 = svdup_s8(0), r31 = svdup_s8(0);
                if (valid && row_base + row + 3u < logical_rows) {
                    svint8x2_t x =
                        svld2_s8(svwhilelt_b8(size_t(0), valid),
                                 source + ((row_base + row + 3u) * logical_features +
                                           feature_base + feature) *
                                              2u);
                    r30 = svget2_s8(x, 0);
                    r31 = svget2_s8(x, 1);
                }
                {
                    svint32_t first_pair_low = svzip1_s32(svreinterpret_s32_s8(r00),
                                                          svreinterpret_s32_s8(r10)),
                              first_pair_high = svzip2_s32(svreinterpret_s32_s8(r00),
                                                           svreinterpret_s32_s8(r10));
                    svint32_t second_pair_low = svzip1_s32(svreinterpret_s32_s8(r20),
                                                           svreinterpret_s32_s8(r30)),
                              second_pair_high = svzip2_s32(svreinterpret_s32_s8(r20),
                                                            svreinterpret_s32_s8(r30));
                    svst1_s8(all,
                             destination + 0u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 0u,
                             svreinterpret_s8_s64(
                                 svzip1_s64(svreinterpret_s64_s32(first_pair_low),
                                            svreinterpret_s64_s32(second_pair_low))));
                    svst1_s8(all,
                             destination + 0u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 16u,
                             svreinterpret_s8_s64(
                                 svzip2_s64(svreinterpret_s64_s32(first_pair_low),
                                            svreinterpret_s64_s32(second_pair_low))));
                    svst1_s8(all,
                             destination + 0u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 32u,
                             svreinterpret_s8_s64(
                                 svzip1_s64(svreinterpret_s64_s32(first_pair_high),
                                            svreinterpret_s64_s32(second_pair_high))));
                    svst1_s8(all,
                             destination + 0u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 48u,
                             svreinterpret_s8_s64(
                                 svzip2_s64(svreinterpret_s64_s32(first_pair_high),
                                            svreinterpret_s64_s32(second_pair_high))));
                }
                {
                    svint32_t first_pair_low = svzip1_s32(svreinterpret_s32_s8(r01),
                                                          svreinterpret_s32_s8(r11)),
                              first_pair_high = svzip2_s32(svreinterpret_s32_s8(r01),
                                                           svreinterpret_s32_s8(r11));
                    svint32_t second_pair_low = svzip1_s32(svreinterpret_s32_s8(r21),
                                                           svreinterpret_s32_s8(r31)),
                              second_pair_high = svzip2_s32(svreinterpret_s32_s8(r21),
                                                            svreinterpret_s32_s8(r31));
                    svst1_s8(all,
                             destination + 1u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 0u,
                             svreinterpret_s8_s64(
                                 svzip1_s64(svreinterpret_s64_s32(first_pair_low),
                                            svreinterpret_s64_s32(second_pair_low))));
                    svst1_s8(all,
                             destination + 1u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 16u,
                             svreinterpret_s8_s64(
                                 svzip2_s64(svreinterpret_s64_s32(first_pair_low),
                                            svreinterpret_s64_s32(second_pair_low))));
                    svst1_s8(all,
                             destination + 1u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 32u,
                             svreinterpret_s8_s64(
                                 svzip1_s64(svreinterpret_s64_s32(first_pair_high),
                                            svreinterpret_s64_s32(second_pair_high))));
                    svst1_s8(all,
                             destination + 1u * learner_plane_bytes +
                                 row * padded_features + 4u * feature + 48u,
                             svreinterpret_s8_s64(
                                 svzip2_s64(svreinterpret_s64_s32(first_pair_high),
                                            svreinterpret_s64_s32(second_pair_high))));
                }
            }
        }
    }
#else
    for (size_t learner = 0; learner < 2u; ++learner) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t feature = 0; feature < padded_features; ++feature) {
                size_t packed_offset = pack_right ? (row / 4u) * 4u * padded_features +
                                                        (feature / 4u) * 16u +
                                                        (row % 4u) * 4u + feature % 4u
                                                  : row * padded_features + feature;
                destination[learner * learner_plane_bytes + packed_offset] =
                    (row_base + row < logical_rows &&
                     feature_base + feature < logical_features)
                        ? source[((row_base + row) * logical_features + feature_base +
                                  feature) *
                                     2u +
                                 learner]
                        : 0;
            }
        }
    }
#endif
    if (pack_right) {
        ++counters.b_packs;
        counters.b_bytes += learner_plane_bytes * 2;
    } else {
        ++counters.a_packs;
        counters.a_bytes += learner_plane_bytes * 2;
    }
}
static size_t c_index(size_t row, size_t column, size_t padded_columns) {
    return row * padded_columns + column;
}
static void scalar(const int8_t *left, const int8_t *right, size_t rows, size_t columns,
                   size_t features, int8_t *output) {
    ++counters.fallbacks;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            for (size_t learner = 0; learner < 2; ++learner) {
                uint32_t sum = 0;
                for (size_t feature = 0; feature < features; ++feature) {
                    sum += uint32_t(
                        int32_t(left[(row * features + feature) * 2 + learner]) *
                        int32_t(right[(column * features + feature) * 2 + learner]));
                }
                size_t output_index = (row * columns + column) * 2 + learner;
                output[output_index] = (int8_t)(uint8_t)sum;
#ifdef I2CE_TEST_ATTENTION_SDOT
                if (observed) {
                    observed[output_index] = sum;
                }
#endif
            }
        }
    }
}
#if CFG_SIMD
static void core_4x16(const int8_t *left, const int8_t *right, uint32_t *accumulators,
                      size_t padded_features, size_t accumulator_stride) {
    attention_sdot_mr4nr16(left, right, accumulators, padded_features, padded_features,
                           4u * padded_features, accumulator_stride);
    ++counters.core_calls;
    counters.sdot += 4u * padded_features;
}
#endif
using Strategy = bool (*)(const int8_t *, const int8_t *, size_t, size_t, size_t,
                          int8_t *);
static void dispatch(const int8_t *left, const int8_t *right, size_t rows,
                     size_t columns, size_t features, int8_t *output,
                     Strategy strategy) {
    if (!rows || !columns) {
        return;
    }
    const Extents checked_extents =
        extents(left, right, rows, columns, features, output);
    if (!features) {
        std::memset(output, 0, checked_extents.output_bytes);
#ifdef I2CE_TEST_ATTENTION_SDOT
        if (observed) {
            std::fill(observed, observed + checked_extents.output_bytes, uint32_t(0));
        }
#endif
        return;
    }
    const bool alias = overlap(left, checked_extents.left_bytes, output,
                               checked_extents.output_bytes) ||
                       overlap(right, checked_extents.right_bytes, output,
                               checked_extents.output_bytes);
    std::vector<int8_t> alias_output;
    if (alias) {
        alias_output.resize(checked_extents.output_bytes);
    }
    int8_t *destination = alias ? alias_output.data() : output;
    if (!strategy(left, right, rows, columns, features, destination)) {
        scalar(left, right, rows, columns, features, destination);
    }
    if (alias) {
        std::memcpy(output, destination, checked_extents.output_bytes);
    }
}
} // namespace attention_sdot
