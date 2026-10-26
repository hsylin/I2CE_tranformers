// Enumerate physical bytes independently of the production view helpers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
static void require(bool passed) {
    if (!passed) {
        std::abort();
    }
}

#define I2CE_TEST_CB4_CACHE
#include "Full_NN/src/gemm_SVE.c"

int i2ce_cb4_cache_test_allocate(void **pointer, size_t alignment, size_t bytes) {
    return posix_memalign(pointer, alignment, bytes);
}

static unsigned check_activations() {
    unsigned cases = 0;
    for (unsigned poison : {0x5au, 0xa5u}) {
        for (size_t rows : {1u, 3u, 4u, 7u, 15u, 16u, 17u, 127u, 128u}) {
            for (size_t width :
                 {1u, 15u, 16u, 17u, 127u, 128u, 129u, 255u, 256u, 257u, 511u, 512u}) {
                for (size_t first_row : {0u, 1u}) {
                    for (size_t first_k : {0u, 16u, 256u}) {
                        const size_t features = first_k + width;
                        std::vector<int8_t> input((first_row + rows) * features * 2u);
                        for (size_t i = 0; i < input.size(); ++i) {
                            input[i] = int8_t(i * 57u + poison);
                        }
                        const auto input_before = input;
                        const size_t plane = size_t(CACHE_S2) * 528u;
                        std::vector<int8_t> actual(2u * plane + 64u, int8_t(poison));
                        auto expected = actual;
                        cache_pack_x(actual.data() + 32u, rows, 528u, plane,
                                     input.data(), features, first_row, first_k, width);
                        for (size_t learner = 0; learner < 2u; ++learner) {
                            for (size_t row = 0; row < rows; ++row) {
                                for (size_t k = 0; k < (width + 15u) / 16u * 16u; ++k) {
                                    expected[32u + learner * plane + row * 528u + k] =
                                        k < width
                                            ? input[((first_row + row) * features +
                                                     first_k + k) *
                                                        2u +
                                                    learner]
                                            : 0;
                                }
                            }
                        }
                        require(actual == expected && input == input_before);
                        ++cases;
                    }
                }
            }
        }
    }
    return cases;
}

static unsigned check_accumulators() {
    unsigned cases = 0;
    for (unsigned poison : {0x5au, 0xa5u}) {
        for (size_t rows : {1u, 3u, 4u, 7u, 15u, 16u, 17u, 127u, 128u}) {
            for (size_t columns : {1u, 7u, 15u, 16u, 17u, 31u, 32u, 33u, 127u, 128u}) {
                for (bool has_bias : {false, true}) {
                    cb4_cache_arena actual;
                    std::memset(&actual, poison, sizeof(actual));
                    std::vector<unsigned char> expected(sizeof(actual), poison);
                    std::vector<int32_t> bias(columns * 2u);
                    for (size_t i = 0; i < bias.size(); ++i) {
                        uint32_t bits = 0x80000000u + uint32_t(i * 19117u + poison);
                        std::memcpy(&bias[i], &bits, sizeof(bits));
                    }
                    cache_init_c_tiles(&actual, rows, columns,
                                       has_bias ? bias.data() : nullptr, 0);
                    // Enumerate 16x32 tiles inside the 128x128 owner, then learner
                    // planes. All other C slots and the complete W owner stay poisoned.
                    for (size_t row = 0; row < rows; ++row) {
                        for (size_t column = 0; column < columns; ++column) {
                            for (size_t learner = 0; learner < 2u; ++learner) {
                                const size_t tile = (column / 32u) * 8u + row / 16u;
                                const size_t element = tile * 1024u + learner * 512u +
                                                       (row % 16u) * 32u + column % 32u;
                                uint32_t value =
                                    has_bias ? uint32_t(bias[column * 2u + learner])
                                             : 0u;
                                std::memcpy(expected.data() + element * 4u, &value, 4u);
                            }
                        }
                    }
                    require(std::memcmp(&actual, expected.data(), sizeof(actual)) == 0);
                    std::vector<uint32_t> output(rows * columns * 2u + 2u, 0x12345678u);
                    cache_write_y_tiles(&actual, rows, columns, output.data() + 1u,
                                        columns, 0, 0);
                    for (size_t row = 0; row < rows; ++row) {
                        for (size_t column = 0; column < columns * 2u; ++column) {
                            require(output[1u + row * columns * 2u + column] ==
                                    (has_bias ? uint32_t(bias[column]) : 0u));
                        }
                    }
                    require(output.front() == 0x12345678u &&
                            output.back() == 0x12345678u);
                    require(std::memcmp(&actual, expected.data(), sizeof(actual)) == 0);
                    ++cases;
                }
            }
        }
    }
    return cases;
}

int main() {
    require(svcntb() == 16u);
    const unsigned activations = check_activations();
    const unsigned accumulators = check_accumulators();
    // A registered full shape makes source-work accounting independently
    // calculable: 2 outer sequence tiles x 2 output tiles x 2 K chunks.
    std::vector<int8_t> input(256u * 1024u * 2u, 0);
    std::vector<uint8_t> indices(8u * 8u * 1024u, 0);
    std::vector<int8_t> output(256u * 256u * 2u, 42);
    const int32_t codebook[8] = {};
    sve_cb4_scratch_reset();
    require(sve_gemm_ci_cb4_cache_i8(indices.data(), indices.size(), 64u, 256u, 256u,
                                     1024u, input.data(), codebook, nullptr,
                                     output.data(), 2u));
    for (int8_t value : output) {
        require(value == 0);
    }
    const auto stats = sve_cb4_scratch_get();
    require(stats.A_pack_calls == 8u && stats.A_assigned_bytes == 1048576u);
    require(stats.W_pack_calls == 128u && stats.W_assigned_bytes == 1179648u);
    require(stats.runtime_allocations == 1u && stats.frees == 1u);
    require(stats.retained_bytes == 0u);
    require(stats.peak_W_bytes == 147456u);
    require(stats.requested_bytes == 937984u);
    std::printf("PASS: %u A layout cases and %u C initialization/output cases\n",
                activations, accumulators);
}
