#pragma once
#include "attentionSdot.h"

namespace attention_sdot {

// A single MR4xNR16 output tile uses row-major A and grouped-four-column B.
// Each invocation accumulates the complete padded K span before narrowing.
static bool micro_4x16(const int8_t *left, const int8_t *right, size_t rows,
                       size_t columns, size_t features, int8_t *output) {
#if CFG_SIMD
    if (svcntb() != 16u || rows > 65535u || columns > 65535u || features > 512u) {
        return false;
    }
    const size_t padded_features = padded(features, 16u);
    const size_t left_bytes = 2u * 4u * padded_features;
    const size_t right_bytes = 2u * 16u * padded_features;
    const size_t accumulator_bytes = 2u * 4u * 16u * sizeof(uint32_t);
    Arena owner(left_bytes + right_bytes + accumulator_bytes);
    int8_t *packed_left = owner.data();
    int8_t *packed_right = packed_left + left_bytes;
    auto *accumulator = reinterpret_cast<uint32_t *>(packed_right + right_bytes);
    ++counters.calls;
    for (size_t first_row = 0; first_row < rows; first_row += 4u) {
        pack_sdot(left, rows, features, first_row, 0u, 4u, padded_features, packed_left,
                  false);
        for (size_t first_column = 0; first_column < columns; first_column += 16u) {
            pack_sdot(right, columns, features, first_column, 0u, 16u, padded_features,
                      packed_right, true);
            std::fill(accumulator, accumulator + 128u, uint32_t(0));
            counters.c_initialized += accumulator_bytes;
            for (size_t learner = 0; learner < 2u; ++learner) {
                core_4x16(packed_left + learner * 4u * padded_features,
                          packed_right + learner * 16u * padded_features,
                          accumulator + learner * 64u, padded_features, 16u);
            }
            for (size_t row = 0; row < std::min(size_t(4), rows - first_row); ++row) {
                for (size_t column = 0;
                     column < std::min(size_t(16), columns - first_column); ++column) {
                    for (size_t learner = 0; learner < 2u; ++learner) {
                        const size_t index =
                            ((first_row + row) * columns + first_column + column) * 2u +
                            learner;
                        const uint32_t value =
                            accumulator[learner * 64u + row * 16u + column];
                        output[index] = int8_t(uint8_t(value));
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
