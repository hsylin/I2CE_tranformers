#pragma once
#include "attentionSdot.h"

namespace attention_sdot {
#if CFG_SIMD
static void core_8x8(const int8_t *left, const int8_t *right, uint32_t *accumulators,
                     size_t padded_features, size_t output_row_stride) {
    attention_sdot_mr8nr8(left, right, accumulators, padded_features, padded_features,
                          padded_features * 4, output_row_stride);
    ++counters.core_calls;
    counters.sdot += 4u * padded_features;
}
#endif
// The public generic API retains eight-row/eight-column tiles. The
// blocked workload has a separate MR4xNR16 consumer and dispatch path.
static bool micro_8x8(const int8_t *left, const int8_t *right, size_t rows,
                      size_t columns, size_t features, int8_t *output) {
#if CFG_SIMD
    if (svcntb() != 16 || rows > 65535 || columns > 65535 || features > 512) {
        return false;
    }
    const size_t padded_features = padded(features, 16),
                 left_bytes = 8 * padded_features * 2,
                 right_bytes = 8 * padded_features * 2,
                 accumulator_bytes = 8 * 8 * 2 * sizeof(uint32_t);
    Arena owner(left_bytes + right_bytes + accumulator_bytes);
    int8_t *packed_left = owner.data(), *packed_right = packed_left + left_bytes;
    uint32_t *accumulators = reinterpret_cast<uint32_t *>(packed_right + right_bytes);
    ++counters.calls;
    for (size_t row_base = 0; row_base < rows; row_base += 8) {
        pack_sdot(left, rows, features, row_base, 0, 8, padded_features, packed_left,
                  false);
        for (size_t column_base = 0; column_base < columns; column_base += 8) {
            pack_sdot(right, columns, features, column_base, 0, 8, padded_features,
                      packed_right, true);
            std::fill(accumulators, accumulators + 128, uint32_t(0));
            counters.c_initialized += accumulator_bytes;
            for (size_t learner = 0; learner < 2; ++learner) {
                core_8x8(packed_left + learner * 8 * padded_features,
                         packed_right + learner * 8 * padded_features,
                         accumulators + learner * 64, padded_features, 8);
            }
#ifdef I2CE_TEST_ATTENTION_SDOT
            if (partial_observer) {
                partial_observer(accumulators, 8, 8, row_base, column_base, features,
                                 rows, columns);
            }
#endif
            for (size_t row = 0; row < std::min(size_t(8), rows - row_base); ++row) {
                for (size_t column = 0;
                     column < std::min(size_t(8), columns - column_base); ++column) {
                    for (size_t learner = 0; learner < 2; ++learner) {
                        size_t index =
                            ((row_base + row) * columns + column_base + column) * 2 +
                            learner;
                        uint32_t value =
                            accumulators[learner * 64 + c_index(row, column, 8)];
                        output[index] = (int8_t)(uint8_t)value;
#ifdef I2CE_TEST_ATTENTION_SDOT
                        if (observed) {
                            observed[index] = value;
                        }
#endif
                    }
                }
            }
        }
    }
    return true;
#else
    (void)left;
    (void)right;
    (void)rows;
    (void)columns;
    (void)features;
    (void)output;
    return false;
#endif
}

} // namespace attention_sdot
