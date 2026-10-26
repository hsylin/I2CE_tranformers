#pragma once
#include <stdint.h>
#include <string.h>
#include "gemm_sdot_lane.h"

// Adapt interleaved int32 operands to one learner's MR4xNR16 SDOT call.
// Packed panels are local to each call and are not reused across row tiles.
// Two physical K128 panels form one Kacc256 call. Four-output groups hold
// four adjacent signed K bytes per output, matching SDOT's byte lanes.
enum {
    SDOT_ADAPTER_CORE_ROWS = 4,
    SDOT_ADAPTER_CORE_COLUMNS = 16,
    SDOT_ADAPTER_CORE_K = 256,
    SDOT_ADAPTER_ACTIVATION_STRIDE = 528,
    SDOT_ADAPTER_WEIGHT_GROUP_STRIDE = 576,
    SDOT_ADAPTER_WEIGHT_PANEL_STRIDE = 9216,
    SDOT_ADAPTER_ACCUMULATOR_STRIDE = 32,
    SDOT_ADAPTER_WEIGHT_ADAPTER_BYTES =
        SDOT_ADAPTER_WEIGHT_PANEL_STRIDE + 4 * SDOT_ADAPTER_WEIGHT_GROUP_STRIDE
};

static void pack_sdot_activations_from_i32(int8_t *packed, const int32_t *widened,
                                           uint32_t input_size, uint32_t row_start,
                                           uint32_t row_count, uint32_t k_start,
                                           uint32_t k_count, uint32_t learner) {
    /* The fixed K256 core reads four rows, including dummy edge rows.
     * Initialize exactly those bytes; the 272-byte stride padding is unread.
     * The core always reads all 256 K bytes, even for a short logical tail. */
    for (uint32_t row = 0; row < SDOT_ADAPTER_CORE_ROWS; ++row) {
        for (uint32_t k = 0; k < SDOT_ADAPTER_CORE_K; ++k) {
            int8_t value = 0;
            if (row < row_count && k < k_count) {
                const size_t source =
                    ((size_t)(row_start + row) * input_size + k_start + k) * 2u +
                    learner;
                value = (int8_t)widened[source];
            }
            packed[row * SDOT_ADAPTER_ACTIVATION_STRIDE + k] = value;
        }
    }
}

static void pack_sdot_weights_from_indices(int8_t *packed, const uint32_t *indices,
                                           const int32_t *widened_codebook,
                                           uint32_t words_per_output,
                                           uint32_t output_start, uint32_t output_count,
                                           uint32_t k_start, uint32_t k_count,
                                           uint32_t learner) {
    // Clear the whole physical adapter, including padding the core ignores.
    // Valid outputs overwrite only their decoded signed-byte positions.
    memset(packed, 0, SDOT_ADAPTER_WEIGHT_ADAPTER_BYTES);
    for (uint32_t output = 0; output < output_count; ++output) {
        const uint32_t *index_row =
            indices + (size_t)(output_start + output) * words_per_output;
        for (uint32_t k = 0; k < k_count; ++k) {
            const uint32_t logical_k = k_start + k;
            const uint32_t code =
                (index_row[logical_k / 16u] >> (2u * (logical_k % 16u))) & 3u;
            const uint32_t panel_k = k % 128u;
            const size_t destination =
                (k / 128u) * SDOT_ADAPTER_WEIGHT_PANEL_STRIDE +
                (output / 4u) * SDOT_ADAPTER_WEIGHT_GROUP_STRIDE +
                (panel_k / 4u) * 16u + (output % 4u) * 4u + panel_k % 4u;
            packed[destination] = (int8_t)widened_codebook[code * 2u + learner];
        }
    }
}

/* The arithmetic core loads all 4x16 partial sums. Dummy edge positions
 * therefore require zero, while the unused second half of each C row does not.
 * Bias keeps its uint32 modulo bits; no signed-overflow arithmetic is used. */
static void initialize_sdot_accumulators(uint32_t *accumulators, uint32_t rows,
                                         uint32_t columns, const int32_t *bias,
                                         uint32_t output_start, uint32_t learner) {
    const svbool_t words = svptrue_b32();
    for (uint32_t row = 0; row < SDOT_ADAPTER_CORE_ROWS; ++row) {
        for (uint32_t column = 0; column < SDOT_ADAPTER_CORE_COLUMNS; column += 4u) {
            svuint32_t initial = svdup_u32(0);
            if (bias != NULL && row < rows && column < columns) {
                const svbool_t valid =
                    svwhilelt_b32((uint64_t)column, (uint64_t)columns);
                const svint32x2_t pair =
                    svld2_s32(valid, bias + 2u * (output_start + column));
                initial = svreinterpret_u32_s32(learner == 0u ? svget2_s32(pair, 0)
                                                              : svget2_s32(pair, 1));
            }
            svst1_u32(words,
                      accumulators + row * SDOT_ADAPTER_ACCUMULATOR_STRIDE + column,
                      initial);
        }
    }
}

static void gemm_cb4_sdot_from_widened(gemm_t shape, const int32_t *input,
                                       const uint32_t *indices, const int32_t *codebook,
                                       const int32_t *bias, int32_t *output,
                                       i2ce_cb4_scratch_stats *accounting) {
    int8_t activation_panel[SDOT_ADAPTER_CORE_ROWS * SDOT_ADAPTER_ACTIVATION_STRIDE]
        __attribute__((aligned(64)));
    int8_t weight_panel[SDOT_ADAPTER_WEIGHT_ADAPTER_BYTES] __attribute__((aligned(64)));
    uint32_t accumulators[SDOT_ADAPTER_CORE_ROWS][SDOT_ADAPTER_ACCUMULATOR_STRIDE]
        __attribute__((aligned(64)));

    if (accounting->peak_W_bytes < sizeof(weight_panel)) {
        accounting->peak_W_bytes = sizeof(weight_panel);
    }
    for (uint32_t output_start = 0; output_start < shape.output_size;
         output_start += SDOT_ADAPTER_CORE_COLUMNS) {
        const uint32_t columns =
            shape.output_size - output_start < SDOT_ADAPTER_CORE_COLUMNS
                ? shape.output_size - output_start
                : SDOT_ADAPTER_CORE_COLUMNS;
        for (uint32_t row_start = 0; row_start < shape.seq_len;
             row_start += SDOT_ADAPTER_CORE_ROWS) {
            const uint32_t rows = shape.seq_len - row_start < SDOT_ADAPTER_CORE_ROWS
                                      ? shape.seq_len - row_start
                                      : SDOT_ADAPTER_CORE_ROWS;
            for (uint32_t learner = 0; learner < 2u; ++learner) {
                initialize_sdot_accumulators(&accumulators[0][0], rows, columns, bias,
                                             output_start, learner);
                for (uint32_t k_start = 0; k_start < shape.input_size;
                     k_start += SDOT_ADAPTER_CORE_K) {
                    const uint32_t k_count =
                        shape.input_size - k_start < SDOT_ADAPTER_CORE_K
                            ? shape.input_size - k_start
                            : SDOT_ADAPTER_CORE_K;
                    pack_sdot_activations_from_i32(activation_panel, input,
                                                   shape.input_size, row_start, rows,
                                                   k_start, k_count, learner);
                    pack_sdot_weights_from_indices(weight_panel, indices, codebook,
                                                   shape.n_words_row, output_start,
                                                   columns, k_start, k_count, learner);
                    ++accounting->A_pack_calls;
                    accounting->A_assigned_bytes +=
                        SDOT_ADAPTER_CORE_ROWS * SDOT_ADAPTER_CORE_K;
                    accounting->A_useful_bytes += rows * k_count;
                    accounting->W_pack_calls += 2u; // Two physical K128 panels.
                    accounting->W_assigned_bytes +=
                        sizeof(weight_panel) + columns * k_count;
                    accounting->W_useful_bytes += columns * k_count;
                    i2ce_sdot_cb_mr4nr16_k256(activation_panel, weight_panel,
                                              &accumulators[0][0]);
                }
                for (uint32_t row = 0; row < rows; ++row) {
                    for (uint32_t column = 0; column < columns; ++column) {
                        const size_t destination =
                            ((size_t)(row_start + row) * shape.output_size +
                             output_start + column) *
                                2u +
                            learner;
                        /* Preserve the uint32 modulo bits in the signed API;
                         * avoid signed-overflow arithmetic and narrowing casts. */
                        memcpy(output + destination, &accumulators[row][column],
                               sizeof(int32_t));
                    }
                }
            }
        }
    }
}
