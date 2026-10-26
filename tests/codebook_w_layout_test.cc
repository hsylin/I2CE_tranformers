// Check serialized indices and physical W bytes using logical coordinates.
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

static unsigned logical_index(size_t column, size_t k) {
    return unsigned((column * 3u + k * 5u + k / 7u) % 4u);
}

static void check_panel(size_t columns, size_t width, size_t first_column,
                        size_t first_k, unsigned poison) {
    const size_t total_columns = first_column + columns;
    const size_t total_k = first_k + width;
    const size_t k_blocks = (total_k + 127u) / 128u;
    std::vector<uint8_t> indices((total_columns + 31u) / 32u * k_blocks * 1024u, 0);
    for (size_t column = 0; column < total_columns; ++column) {
        for (size_t k = 0; k < total_k; ++k) {
            const size_t block = (column / 32u) * k_blocks + k / 128u;
            const size_t pair = (column % 32u) / 2u;
            const size_t k_eight = (k % 128u) / 8u;
            const size_t lane = (column % 2u) * 8u + k % 8u;
            const size_t byte = block * 1024u + pair * 64u + k_eight * 4u + lane / 4u;
            indices[byte] |= uint8_t(logical_index(column, k) << (2u * (lane % 4u)));
        }
    }
    const auto indices_before = indices;
    const int32_t codebook[] = {-128, 127, -1, 1, 0, -128, 127, 0};
    int8_t scratch0[256] = {}, scratch1[256] = {};
    const svint8_t table0 = gemm_sdot_cb_table(codebook, 4u, 2u, 0u, scratch0);
    const svint8_t table1 = gemm_sdot_cb_table(codebook, 4u, 2u, 1u, scratch1);
    svuint8_t byte_selector, shifts;
    gemm_sdot_lane_patterns(2u, &byte_selector, &shifts);
    std::vector<int8_t> actual(9216u + 64u, int8_t(poison));
    auto expected = actual;
    ci_pack_sdot_w(actual.data() + 32u, columns, 32u, 144u, 4608u, indices.data(),
                   k_blocks, first_column, first_k, width, table0, table1,
                   byte_selector, shifts);
    const bool full_panel = columns == 32u && width == 128u;
    for (size_t learner = 0; learner < 2u; ++learner) {
        for (size_t column = 0; column < 32u; ++column) {
            for (size_t k = 0; k < 144u; ++k) {
                // Four columns share each 16-byte K4 vector. Padding after
                // K128 is untouched only in the complete-panel path.
                if (full_panel && k >= 128u) {
                    continue;
                }
                const size_t offset = learner * 4608u + (column / 4u) * 576u +
                                      (k / 4u) * 16u + (column % 4u) * 4u + k % 4u;
                expected[32u + offset] =
                    column < columns && k < width
                        ? int8_t(codebook[logical_index(first_column + column,
                                                        first_k + k) *
                                              2u +
                                          learner])
                        : 0;
            }
        }
    }
    require(actual == expected && indices == indices_before);
}

int main() {
    require(svcntb() == 16u);
    unsigned cases = 0;
    for (size_t columns : {1u, 2u, 3u, 15u, 16u, 17u, 31u, 32u}) {
        for (size_t width : {1u, 7u, 8u, 15u, 16u, 17u, 63u, 127u, 128u}) {
            for (size_t first_column : {0u, 32u, 128u}) {
                for (size_t first_k : {0u, 128u, 512u}) {
                    for (unsigned poison : {0x5au, 0xa5u}) {
                        check_panel(columns, width, first_column, first_k, poison);
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("PASS: %u independent W physical-layout and padding cases\n", cases);
}
