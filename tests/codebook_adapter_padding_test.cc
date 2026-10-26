// Check the core's mandatory read extent and untouched stride padding separately.
#include <arm_sve.h>
#include "gemm_exec.h"
#include "gemm_SVE.h"
#include "Full_NN/src/gemm_cb4_adapter.h"

#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>

static void check(uint32_t rows, uint32_t columns, uint32_t features, uint32_t learner,
                  bool has_bias) {
    constexpr uint32_t input_features = 300;
    constexpr uint32_t row_start = 1, feature_start = 3, output_start = 2;
    constexpr int8_t activation_poison = 0x5a;
    constexpr uint32_t accumulator_poison = 0xa5a55a5a;
    std::array<int32_t, 6 * input_features * 2> input;
    std::array<int32_t, 40> bias;
    std::array<int8_t, SDOT_ADAPTER_CORE_ROWS * SDOT_ADAPTER_ACTIVATION_STRIDE> packed;
    std::array<uint32_t, SDOT_ADAPTER_CORE_ROWS * SDOT_ADAPTER_ACCUMULATOR_STRIDE>
        accumulators;
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = int32_t((index * 19u) % 256u) - 128;
    }
    for (size_t index = 0; index < bias.size(); ++index) {
        bias[index] =
            index % 2u ? INT32_MIN + int32_t(index) : INT32_MAX - int32_t(index);
    }
    packed.fill(activation_poison);
    accumulators.fill(accumulator_poison);
    pack_sdot_activations_from_i32(packed.data(), input.data(), input_features,
                                   row_start, rows, feature_start, features, learner);
    initialize_sdot_accumulators(accumulators.data(), rows, columns,
                                 has_bias ? bias.data() : nullptr, output_start,
                                 learner);
    for (uint32_t row = 0; row < SDOT_ADAPTER_CORE_ROWS; ++row) {
        for (uint32_t k = 0; k < SDOT_ADAPTER_ACTIVATION_STRIDE; ++k) {
            int8_t expected = activation_poison;
            if (k < SDOT_ADAPTER_CORE_K) {
                expected = row < rows && k < features
                               ? int8_t(input[((row_start + row) * input_features +
                                               feature_start + k) *
                                                  2u +
                                              learner])
                               : 0;
            }
            if (packed[row * SDOT_ADAPTER_ACTIVATION_STRIDE + k] != expected) {
                std::abort();
            }
        }
        for (uint32_t column = 0; column < SDOT_ADAPTER_ACCUMULATOR_STRIDE; ++column) {
            uint32_t expected = accumulator_poison;
            if (column < SDOT_ADAPTER_CORE_COLUMNS) {
                expected = has_bias && row < rows && column < columns
                               ? uint32_t(bias[(output_start + column) * 2u + learner])
                               : 0u;
            }
            if (accumulators[row * SDOT_ADAPTER_ACCUMULATOR_STRIDE + column] !=
                expected) {
                std::abort();
            }
        }
    }
}

int main() {
    unsigned cases = 0;
    for (uint32_t rows : {1u, 3u, 4u}) {
        for (uint32_t columns : {1u, 15u, 16u}) {
            for (uint32_t features : {1u, 127u, 128u, 255u, 256u}) {
                for (uint32_t learner : {0u, 1u}) {
                    for (bool bias : {false, true}) {
                        check(rows, columns, features, learner, bias);
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("PASS: %u mandatory-read and untouched-padding cases\n", cases);
}
