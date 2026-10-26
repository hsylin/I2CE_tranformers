// Independent logical CB4 matrix oracle for the SDOT core and widened adapters.
#include "gemm_exec.h"
#include "gemm_exec_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check_case(uint16_t rows, uint16_t features, uint16_t columns,
                       bool use_bias, bool use_cached_codebook) {
    gemm_t shape = {rows, features, columns,
                    static_cast<uint16_t>((features + 15u) / 16u)};
    std::vector<int8_t> input(static_cast<size_t>(rows) * features * 2u);
    const int8_t codebook[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
    std::vector<int32_t> widened(codebook, codebook + 8);
    std::vector<uint32_t> packed(static_cast<size_t>(columns) * shape.n_words_row, 0);
    std::vector<int32_t> bias(columns * 2u);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<int8_t>(
            static_cast<int>((index * 37u + index / 11u) % 256u) - 128);
    }
    for (uint32_t column = 0; column < columns; ++column) {
        bias[column * 2u] = INT32_MAX - static_cast<int32_t>(column);
        bias[column * 2u + 1u] = INT32_MIN + static_cast<int32_t>(column);
        for (uint32_t k = 0; k < features; ++k) {
            const uint32_t code = (column * 3u + k * 5u + k / 7u) % 4u;
            packed[column * shape.n_words_row + k / 16u] |= code << (2u * (k % 16u));
        }
    }
    const int32_t sentinel = 0x12345678;
    const size_t output_elements = static_cast<size_t>(rows) * columns * 2u;
    std::vector<int32_t> actual(output_elements + 2, sentinel);
    std::vector<int32_t> expected(output_elements + 2, sentinel);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < columns; ++column) {
            for (uint32_t learner = 0; learner < 2; ++learner) {
                uint32_t sum =
                    use_bias ? static_cast<uint32_t>(bias[column * 2u + learner]) : 0u;
                for (uint32_t k = 0; k < features; ++k) {
                    const uint32_t code = (column * 3u + k * 5u + k / 7u) % 4u;
                    const int32_t product =
                        static_cast<int32_t>(
                            input[(row * features + k) * 2u + learner]) *
                        static_cast<int32_t>(codebook[code * 2u + learner]);
                    sum += static_cast<uint32_t>(product);
                }
                std::memcpy(&expected[1 + (row * columns + column) * 2u + learner],
                            &sum, sizeof(sum));
            }
        }
    }
    const auto input_before = input;
    const auto packed_before = packed;
    const auto codebook_before = widened;
    gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
        shape, input.data(), packed.data(), codebook, use_bias ? bias.data() : nullptr,
        actual.data() + 1, 2, use_cached_codebook ? widened.data() : nullptr, nullptr,
        0);
    if (actual != expected || input != input_before || packed != packed_before ||
        widened != codebook_before) {
        std::fprintf(stderr, "FAIL rows=%u K=%u N=%u bias=%d cache=%d\n", rows,
                     features, columns, use_bias, use_cached_codebook);
        std::exit(1);
    }
}

int main() {
    unsigned cases = 0;
    for (uint16_t rows : {1, 3, 4, 5, 16}) {
        for (uint16_t columns : {1, 15, 16, 17, 32}) {
            for (uint16_t features :
                 {1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1024}) {
                for (bool bias : {false, true}) {
                    for (bool cache : {false, true}) {
                        check_case(rows, features, columns, bias, cache);
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("PASS: %u independent signed-byte, tail and modulo-bias cases\n",
                cases);
}
