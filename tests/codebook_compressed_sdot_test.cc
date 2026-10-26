// Independent logical CB4 oracle for the cumulative CI01 byte-operand path.
#include "gemm_exec.h"
#include "gemm_exec_internal.h"
#include "gemm_SVE.h"
extern uint32_t *i2ce_cb4_cache_observed;

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int i2ce_cb4_cache_test_allocate(void **pointer, size_t alignment, size_t bytes) {
    const int status = posix_memalign(pointer, alignment, bytes);
    if (status == 0) {
        std::memset(*pointer, 0xa5, bytes);
    }
    return status;
}

static void check_case(uint16_t rows, uint16_t features, uint16_t columns,
                       bool use_bias) {
    std::vector<int8_t> input(static_cast<size_t>(rows) * features * 2u);
    const int32_t codebook[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
    const size_t output_blocks = (columns + 31u) / 32u;
    const size_t k_blocks = (features + 127u) / 128u;
    std::vector<uint8_t> packed(output_blocks * k_blocks * 1024u, 0);
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
            // Build the serialized format from logical coordinates, without
            // calling the production offset, decoder or table helpers.
            const size_t block = (column / 32u) * k_blocks + k / 128u;
            const size_t pair = (column % 32u) / 2u;
            const size_t k_eight = (k % 128u) / 8u;
            const size_t lane = (column % 2u) * 8u + k % 8u;
            const size_t byte = block * 1024u + pair * 64u + k_eight * 4u + lane / 4u;
            packed[byte] |= static_cast<uint8_t>(code << (2u * (lane % 4u)));
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

    std::vector<int8_t> bytes(output_elements + 2u, 0x5a);
    std::vector<int8_t> expected_bytes(output_elements + 2u, 0x5a);
    for (size_t i = 0; i < output_elements; ++i) {
        expected_bytes[i + 1u] = (int8_t)(uint8_t)expected[i + 1u];
    }
    i2ce_cb4_cache_observed = reinterpret_cast<uint32_t *>(actual.data() + 1);
    const int accepted = sve_gemm_ci_cb4_cache_i8(
        packed.data(), packed.size(), (features + 15u) / 16u, rows, columns, features,
        input.data(), codebook, use_bias ? bias.data() : nullptr, bytes.data() + 1, 2);
    i2ce_cb4_cache_observed = nullptr;
    if (!accepted || actual != expected || bytes != expected_bytes ||
        input != input_before || packed != packed_before) {
        std::fprintf(stderr, "FAIL rows=%u K=%u N=%u bias=%d\n", rows, features,
                     columns, use_bias);
        std::exit(1);
    }
    // Exercise the real final-output route without the full-C observer.
    // Both observer and ordinary output routes must agree with the oracle.
    std::fill(bytes.begin(), bytes.end(), int8_t(0x5a));
    const int direct_accepted = sve_gemm_ci_cb4_cache_i8(
        packed.data(), packed.size(), (features + 15u) / 16u, rows, columns, features,
        input.data(), codebook, use_bias ? bias.data() : nullptr, bytes.data() + 1, 2);
    if (!direct_accepted || bytes != expected_bytes || input != input_before ||
        packed != packed_before) {
        std::fprintf(stderr, "FAIL direct output rows=%u K=%u N=%u bias=%d\n", rows,
                     features, columns, use_bias);
        std::exit(1);
    }
}

static void rejected_descriptors_leave_output_unchanged() {
    uint8_t indices[1024] = {};
    int8_t input[2] = {}, output[2] = {12, 34};
    int32_t codebook[8] = {};
    for (unsigned invalid = 0; invalid < 9; ++invalid) {
        const int accepted = sve_gemm_ci_cb4_cache_i8(
            invalid == 0 ? nullptr : indices, invalid == 1 ? 1023 : sizeof(indices), 1,
            invalid == 2 ? 65536 : 1, invalid == 3 ? 65536 : 1,
            invalid == 4 ? 65536 : 1, invalid == 5 ? nullptr : input,
            invalid == 6 ? nullptr : codebook, nullptr, invalid == 7 ? nullptr : output,
            invalid == 8 ? 3 : 2);
        if (accepted || output[0] != 12 || output[1] != 34) {
            std::abort();
        }
    }
}

int main() {
    unsigned cases = 0;
    for (uint16_t rows : {1, 3, 4, 5, 16, 17}) {
        for (uint16_t columns : {1, 15, 16, 17, 31, 32, 33}) {
            for (uint16_t features : {0, 1, 15, 16, 17, 127, 128, 129, 255, 256, 257,
                                      511, 512, 513, 1024}) {
                for (bool bias : {false, true}) {
                    check_case(rows, features, columns, bias);
                    ++cases;
                }
            }
        }
    }
    const uint16_t outer_cases[][3] = {
        {128, 512, 128},  {127, 513, 129},  {129, 511, 127},
        {129, 513, 129},  {256, 1024, 256}, {128, 1024, 256},
        {256, 1024, 128}, {17, 257, 129},   {129, 257, 17}};
    for (const auto &shape : outer_cases) {
        for (bool bias : {false, true}) {
            check_case(shape[0], shape[1], shape[2], bias);
            ++cases;
        }
    }
    rejected_descriptors_leave_output_unchanged();
    std::printf("PASS: %u independent CI01 signed-byte/tail/bias cases, %u "
                "direct-output cases and 9 rejected descriptors\n",
                cases, cases);
}
