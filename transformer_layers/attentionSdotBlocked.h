#pragma once
#include "attentionSdot.h"
namespace attention_sdot {
#ifdef I2CE_TEST_ATTENTION_SDOT
static void (*blocked_inner_observer)(const uint32_t *, size_t, size_t, size_t, size_t,
                                      size_t, size_t, size_t, size_t, size_t, size_t,
                                      size_t, size_t) = nullptr;
#endif
// Panels are directly packed into final M1/K1 row-A or N1/K1 grouped-B subpanels.
// Every consumer of current K2 completes before the same owner is overwritten.
static bool blocked(const int8_t *left, const int8_t *right, size_t rows,
                    size_t columns, size_t features, int8_t *output) {
#if CFG_SIMD
    if (svcntb() != 16 || rows > 65535 || columns > 65535 || features > 65535) {
        return false;
    }
    const size_t row_capacity = padded(std::min(size_t(128), rows), 4),
                 column_capacity = padded(std::min(size_t(128), columns), 16),
                 feature_capacity = padded(std::min(size_t(512), features), 16);
    const size_t left_capacity = 2 * row_capacity * feature_capacity,
                 right_capacity = 2 * column_capacity * feature_capacity,
                 accumulator_capacity =
                     2 * row_capacity * column_capacity * sizeof(uint32_t);
    Arena owner(left_capacity + right_capacity + accumulator_capacity);
    int8_t *packed_left = owner.data(), *packed_right = packed_left + left_capacity;
    uint32_t *accumulators =
        reinterpret_cast<uint32_t *>(packed_right + right_capacity);
    ++counters.calls;
    for (size_t row_base = 0; row_base < rows; row_base += 128) {
        for (size_t column_base = 0; column_base < columns; column_base += 128) {
            const size_t tile_rows = std::min(size_t(128), rows - row_base),
                         tile_columns = std::min(size_t(128), columns - column_base),
                         padded_rows = padded(tile_rows, 4),
                         padded_columns = padded(tile_columns, 16);
            std::fill(accumulators, accumulators + 2 * padded_rows * padded_columns,
                      uint32_t(0));
            counters.c_initialized +=
                2 * padded_rows * padded_columns * sizeof(uint32_t);
            for (size_t feature_base = 0; feature_base < features;
                 feature_base += 512) {
                const size_t tile_features =
                                 std::min(size_t(512), features - feature_base),
                             padded_features = padded(tile_features, 16);
                for (size_t inner_row = 0; inner_row < tile_rows; inner_row += 16) {
                    const size_t inner_padded_rows =
                        padded(std::min(size_t(16), tile_rows - inner_row), 4);
                    for (size_t inner_feature = 0; inner_feature < tile_features;
                         inner_feature += 128) {
                        const size_t inner_padded_features = padded(
                            std::min(size_t(128), tile_features - inner_feature), 16);
                        pack_sdot(left, rows, features, row_base + inner_row,
                                  feature_base + inner_feature, inner_padded_rows,
                                  inner_padded_features,
                                  packed_left + 2 * inner_row * padded_features +
                                      2 * inner_padded_rows * inner_feature,
                                  false);
                    }
                }
                for (size_t inner_column = 0; inner_column < tile_columns;
                     inner_column += 32) {
                    const size_t inner_padded_columns =
                        padded(std::min(size_t(32), tile_columns - inner_column), 16);
                    for (size_t inner_feature = 0; inner_feature < tile_features;
                         inner_feature += 128) {
                        const size_t inner_padded_features = padded(
                            std::min(size_t(128), tile_features - inner_feature), 16);
                        pack_sdot(right, columns, features, column_base + inner_column,
                                  feature_base + inner_feature, inner_padded_columns,
                                  inner_padded_features,
                                  packed_right + 2 * inner_column * padded_features +
                                      2 * inner_padded_columns * inner_feature,
                                  true);
                    }
                }
                for (size_t inner_column = 0; inner_column < tile_columns;
                     inner_column += 32) {
                    const size_t inner_columns =
                                     std::min(size_t(32), tile_columns - inner_column),
                                 inner_padded_columns = padded(inner_columns, 16);
                    for (size_t inner_feature = 0; inner_feature < tile_features;
                         inner_feature += 128) {
                        const size_t inner_features = std::min(
                                         size_t(128), tile_features - inner_feature),
                                     inner_padded_features = padded(inner_features, 16);
                        const int8_t *right_panel =
                            packed_right + 2 * inner_column * padded_features +
                            2 * inner_padded_columns * inner_feature;
                        for (size_t inner_row = 0; inner_row < tile_rows;
                             inner_row += 16) {
                            const size_t inner_padded_rows =
                                padded(std::min(size_t(16), tile_rows - inner_row), 4);
                            const int8_t *left_panel =
                                packed_left + 2 * inner_row * padded_features +
                                2 * inner_padded_rows * inner_feature;
                            for (size_t kernel_row = 0; kernel_row < inner_padded_rows;
                                 kernel_row += 4) {
                                for (size_t kernel_column = 0;
                                     kernel_column < inner_padded_columns;
                                     kernel_column += 16) {
                                    for (size_t learner = 0; learner < 2; ++learner) {
                                        const int8_t *kernel_left =
                                                         left_panel +
                                                         learner * inner_padded_rows *
                                                             inner_padded_features +
                                                         kernel_row *
                                                             inner_padded_features,
                                                     *kernel_right =
                                                         right_panel +
                                                         learner *
                                                             inner_padded_columns *
                                                             inner_padded_features +
                                                         kernel_column *
                                                             inner_padded_features;
                                        uint32_t *kernel_accumulators =
                                            accumulators +
                                            learner * padded_rows * padded_columns +
                                            (inner_row + kernel_row) * padded_columns +
                                            inner_column + kernel_column;
                                        core_4x16(kernel_left, kernel_right,
                                                  kernel_accumulators,
                                                  inner_padded_features,
                                                  padded_columns);
                                    }
                                }
                            }
                        }
#ifdef I2CE_TEST_ATTENTION_SDOT
                        if (blocked_inner_observer) {
                            blocked_inner_observer(
                                accumulators, padded_rows, padded_columns, row_base,
                                column_base, feature_base, tile_features, inner_column,
                                inner_columns, inner_feature, inner_features, rows,
                                columns);
                        }
#endif
                    }
                }
#ifdef I2CE_TEST_ATTENTION_SDOT
                if (partial_observer) {
                    partial_observer(accumulators, padded_rows, padded_columns,
                                     row_base, column_base,
                                     feature_base + tile_features, rows, columns);
                }
#endif
            }
            for (size_t row = 0; row < tile_rows; ++row) {
                for (size_t column = 0; column < tile_columns; ++column) {
                    for (size_t learner = 0; learner < 2; ++learner) {
                        size_t output_index =
                            ((row_base + row) * columns + column_base + column) * 2 +
                            learner;
                        uint32_t value =
                            accumulators[learner * padded_rows * padded_columns +
                                         c_index(row, column, padded_columns)];
                        output[output_index] = (int8_t)(uint8_t)value;
#ifdef I2CE_TEST_ATTENTION_SDOT
                        if (observed) {
                            observed[output_index] = value;
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
