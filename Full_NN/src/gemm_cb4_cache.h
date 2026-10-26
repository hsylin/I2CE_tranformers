/* One MR4xNR16 consumer with paired K128 weight panels.
 * Activations stay in int8 form; a complete C32 matrix owns accumulated
 * results until the final low-byte conversion pass. No panel is retained
 * across row tiles or output blocks. */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "compressed_sdot_decoder.h"
#include "gemm_sdot_lane.h"

static i2ce_cb4_scratch_stats cb4_scratch_stats;
static uint64_t cb_last_direct_calls = 0, cb_last_source_cores = 0;
static uint64_t cb_sdot_calls = 0, cb_sdot_full_calls = 0, cb_sdot_source = 0,
                cb_sdot_edges = 0;
#ifdef I2CE_TEST_CB4_CACHE
uint32_t *i2ce_cb4_cache_observed = NULL;
#endif
static __attribute__((destructor)) void report_cb_sdot(void) {
    fprintf(stderr,
            "[codebook-last-output] version=1 direct_calls=%llu "
            "final_core_source_calls=%llu "
            "source_scope=registered_full_shape_not_dynamic_traffic\n",
            (unsigned long long)cb_last_direct_calls,
            (unsigned long long)cb_last_source_cores);
    fprintf(stderr,
            "[codebook-sdot] version=3 MR=4 NR=16 Kpanel_max=256 calls=%llu "
            "full_core_source_calls=%llu full_core_sdot_source_count=%llu "
            "edge_outputs=%llu "
            "prediction_scope=registered_full_panel_shape_not_dynamic_opcode_counter\n",
            (unsigned long long)cb_sdot_calls, (unsigned long long)cb_sdot_full_calls,
            (unsigned long long)cb_sdot_source, (unsigned long long)cb_sdot_edges);
}

void sve_cb4_scratch_reset(void) {
    memset(&cb4_scratch_stats, 0, sizeof(cb4_scratch_stats));
    cb_sdot_calls = cb_sdot_full_calls = cb_sdot_source = cb_sdot_edges = 0;
    cb_last_direct_calls = cb_last_source_cores = 0;
}
i2ce_cb4_scratch_stats sve_cb4_scratch_get(void) {
    return cb4_scratch_stats;
}

static size_t cb4_min(size_t a, size_t b) {
    return a < b ? a : b;
}
static int cb4_valid_address_range(const void *pointer, size_t bytes) {
    return !bytes || (pointer && (uintptr_t)pointer <= UINTPTR_MAX - (bytes - 1u));
}
static int cb4_ranges_overlap(const void *a, size_t a_bytes, const void *b,
                              size_t b_bytes) {
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return a_bytes && b_bytes && (x <= y ? y - x < a_bytes : x - y < b_bytes);
}

enum {
    CB4_ROWS = 4,
    CB4_COLUMNS = 16,
    CB4_K = 256,
    CB4_A_STRIDE = 528,
    CB4_W_STRIDE = 144,
    CB4_C_STRIDE = 32,
    CB4_W_PLANE = 32 * CB4_W_STRIDE,
    CB4_W_PANEL = 2 * CB4_W_PLANE
};

static void cb4_pack_activation_rows(int8_t *packed, const int8_t *input,
                                     size_t features, size_t row_start, size_t rows,
                                     size_t k_start, size_t count) {
    const svbool_t all = svptrue_b8();
    for (size_t row = 0; row < CB4_ROWS; ++row) {
        for (size_t k = 0; k < CB4_K; k += 16u) {
            svint8_t first = svdup_s8(0), second = svdup_s8(0);
            if (row < rows && k < count) {
                const svbool_t valid = svwhilelt_b8((uint64_t)k, (uint64_t)count);
                const svint8x2_t pair = svld2_s8(
                    valid, input + ((row_start + row) * features + k_start + k) * 2u);
                first = svget2_s8(pair, 0);
                second = svget2_s8(pair, 1);
            }
            // K256 loads every core row, including dummy edge rows and K tails.
            // The remaining stride padding is never read and stays untouched.
            svst1_s8(all, packed + row * CB4_A_STRIDE + k, first);
            svst1_s8(all, packed + (CB4_ROWS + row) * CB4_A_STRIDE + k, second);
        }
    }
}

static void cb4_initialize_accumulators(uint32_t *c, const int32_t *bias, size_t rows,
                                        size_t columns, size_t output_start) {
    const svbool_t all = svptrue_b32();
    for (size_t row = 0; row < CB4_ROWS; ++row) {
        for (size_t column = 0; column < CB4_COLUMNS; column += 4u) {
            svuint32_t first = svdup_u32(0), second = svdup_u32(0);
            if (bias && row < rows && column < columns) {
                const svbool_t valid =
                    svwhilelt_b32((uint64_t)column, (uint64_t)columns);
                const svint32x2_t pair =
                    svld2_s32(valid, bias + (output_start + column) * 2u);
                first = svreinterpret_u32_s32(svget2_s32(pair, 0));
                second = svreinterpret_u32_s32(svget2_s32(pair, 1));
            }
            svst1_u32(all, c + row * CB4_C_STRIDE + column, first);
            svst1_u32(all, c + (CB4_ROWS + row) * CB4_C_STRIDE + column, second);
        }
    }
}

int sve_gemm_ci_cb4_cache_i8(const uint8_t *indices, size_t index_bytes, size_t nw,
                             size_t rows, size_t columns, size_t features,
                             const int8_t *input, const int32_t *codebook,
                             const int32_t *bias, int8_t *output, uint8_t bits) {
#if CB_SIZE == 4 && N_LEARNERS == 2 && TILE_L1_SIZE == 1 && TILE_L2_SIZE == 1
    if (bits != 2u || svcntb() != 16u || rows > UINT16_MAX || columns > UINT16_MAX ||
        features > UINT16_MAX || nw > UINT16_MAX) {
        return 0;
    }
    if (!rows || !columns) {
        return 1;
    }
    const size_t input_bytes = rows * features * 2u, output_bytes = rows * columns * 2u;
    const size_t payload_bytes = features ? ci_payload_bytes(columns, features) : 0u;
    const size_t bias_bytes = bias ? columns * 8u : 0u;
    if (index_bytes != payload_bytes || (features && !payload_bytes) ||
        input_bytes > UINT32_MAX || output_bytes > UINT32_MAX ||
        nw < (features + 15u) / 16u || !cb4_valid_address_range(input, input_bytes) ||
        !cb4_valid_address_range(output, output_bytes) ||
        !cb4_valid_address_range(indices, payload_bytes) ||
        !cb4_valid_address_range(codebook, features ? 32u : 0u) ||
        !cb4_valid_address_range(bias, bias_bytes)) {
        return 0;
    }
    if (cb4_ranges_overlap(output, output_bytes, input, input_bytes) ||
        cb4_ranges_overlap(output, output_bytes, indices, payload_bytes) ||
        cb4_ranges_overlap(output, output_bytes, codebook, features ? 32u : 0u) ||
        cb4_ranges_overlap(output, output_bytes, bias, bias_bytes) ||
        (features && ((uintptr_t)codebook & 3u)) || (bias && ((uintptr_t)bias & 3u))) {
        return 0;
    }
    if (features && !gemm_sdot_decode_usable(codebook, 4u, 2u, 2u)) {
        return 0;
    }
    if (!features) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns * 2u; ++column) {
                output[row * columns * 2u + column] =
                    bias ? (int8_t)(uint8_t)bias[column] : 0;
#ifdef I2CE_TEST_CB4_CACHE
                if (i2ce_cb4_cache_observed) {
                    i2ce_cb4_cache_observed[row * columns * 2u + column] =
                        bias ? (uint32_t)bias[column] : 0u;
                }
#endif
            }
        }
        return 1;
    }

    // Own the complete modulo-32-bit output until conversion finishes.
    const size_t temporary_bytes = output_bytes * sizeof(uint32_t);
    uint32_t *temporary = (uint32_t *)malloc(temporary_bytes);
    if (!temporary) {
        ++cb4_scratch_stats.allocation_failures;
        return 0;
    }
    ++cb4_scratch_stats.runtime_allocations;
    cb4_scratch_stats.requested_bytes += temporary_bytes;
    cb4_scratch_stats.retained_bytes += temporary_bytes;
    if (temporary_bytes > cb4_scratch_stats.max_request_bytes) {
        cb4_scratch_stats.max_request_bytes = temporary_bytes;
    }

    int8_t activations[2][CB4_ROWS][CB4_A_STRIDE] __attribute__((aligned(64)));
    int8_t weights[2][CB4_W_PANEL] __attribute__((aligned(64)));
    uint32_t accumulators[2][CB4_ROWS][CB4_C_STRIDE] __attribute__((aligned(64)));
    cb4_scratch_stats.peak_W_bytes = sizeof(weights);
    int8_t scratch0[256] = {0}, scratch1[256] = {0};
    const svint8_t table0 = gemm_sdot_cb_table(codebook, 4u, 2u, 0u, scratch0);
    const svint8_t table1 = gemm_sdot_cb_table(codebook, 4u, 2u, 1u, scratch1);
    svuint8_t byte_selection, shifts;
    gemm_sdot_lane_patterns(2u, &byte_selection, &shifts);
    const size_t index_k_panels = (features + 127u) / 128u;
    ++cb_sdot_calls;
    if (rows % 16u == 0u && columns % 32u == 0u && features % 256u == 0u) {
        cb_sdot_full_calls += (rows / 4u) * (columns / 16u) * 2u * (features / 256u);
        cb_sdot_source += (rows / 4u) * (columns / 16u) * 2u * features * 4u;
    }
    cb_sdot_edges += rows * columns - (rows / 4u * 4u) * (columns / 16u * 16u);
    for (size_t column_start = 0; column_start < columns; column_start += CB4_COLUMNS) {
        const size_t valid_columns = cb4_min(CB4_COLUMNS, columns - column_start);
        for (size_t row_start = 0; row_start < rows; row_start += CB4_ROWS) {
            const size_t valid_rows = cb4_min(CB4_ROWS, rows - row_start);
            cb4_initialize_accumulators(&accumulators[0][0][0], bias, valid_rows,
                                        valid_columns, column_start);
            for (size_t k_start = 0; k_start < features; k_start += CB4_K) {
                const size_t valid_k = cb4_min(CB4_K, features - k_start);
                cb4_pack_activation_rows(&activations[0][0][0], input, features,
                                         row_start, valid_rows, k_start, valid_k);
                ++cb4_scratch_stats.A_pack_calls;
                cb4_scratch_stats.A_assigned_bytes += 2u * CB4_ROWS * CB4_K;
                cb4_scratch_stats.A_useful_bytes += 2u * valid_rows * valid_k;
                for (size_t panel = 0; panel < 2u; ++panel) {
                    const size_t consumed = panel * 128u;
                    const size_t remaining =
                        valid_k > consumed ? valid_k - consumed : 0u;
                    ci_pack_sdot_w(weights[panel], valid_columns, 32u, CB4_W_STRIDE,
                                   CB4_W_PLANE, indices, index_k_panels, column_start,
                                   k_start + consumed, cb4_min(128u, remaining), table0,
                                   table1, byte_selection, shifts);
                }
                cb4_scratch_stats.W_pack_calls += 2u;
                cb4_scratch_stats.W_assigned_bytes += sizeof(weights);
                cb4_scratch_stats.W_useful_bytes += 2u * valid_columns * valid_k;
                for (size_t learner = 0; learner < 2u; ++learner) {
                    i2ce_sdot_cb_mr4nr16_k256(&activations[learner][0][0],
                                              &weights[0][learner * CB4_W_PLANE],
                                              &accumulators[learner][0][0]);
                }
            }
            for (size_t row = 0; row < valid_rows; ++row) {
                for (size_t column = 0; column < valid_columns; ++column) {
                    for (size_t learner = 0; learner < 2u; ++learner) {
                        temporary[((row_start + row) * columns + column_start +
                                   column) *
                                      2u +
                                  learner] = accumulators[learner][row][column];
                    }
                }
            }
        }
    }
#ifdef I2CE_TEST_CB4_CACHE
    if (i2ce_cb4_cache_observed) {
        memcpy(i2ce_cb4_cache_observed, temporary, temporary_bytes);
    }
#endif
    for (size_t i = 0; i < output_bytes; ++i) {
        output[i] = (int8_t)(uint8_t)temporary[i];
    }
    free(temporary);
    ++cb4_scratch_stats.frees;
    cb4_scratch_stats.retained_bytes -= temporary_bytes;
    return 1;
#else
    (void)indices;
    (void)index_bytes;
    (void)nw;
    (void)rows;
    (void)columns;
    (void)features;
    (void)input;
    (void)codebook;
    (void)bias;
    (void)output;
    (void)bits;
    return 0;
#endif
}
