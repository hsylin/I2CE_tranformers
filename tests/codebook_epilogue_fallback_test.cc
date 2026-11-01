// Independent oracle for column-major packed 2-bit indices and low-byte output.
#include "gemm_SVE.h"
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(uint32_t m, uint32_t n, uint32_t k, bool biased) {
    const uint32_t words = (k + 15u) / 16u;
    std::vector<uint32_t> indices(n * words + 1u, 0u);
    std::vector<int8_t> input(size_t(m) * k * 2u + 1u, 0);
    const int32_t book[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
    std::vector<int32_t> bias(n * 2u + 1u);
    auto code = [](uint32_t col, uint32_t feature) {
        return (col * 3u + feature * 5u + feature / 7u) % 4u;
    };
    for (size_t i = 0; i + 1 < input.size(); ++i)
        input[i] = int8_t(int((i * 37u + i / 11u) % 256u) - 128);
    for (uint32_t col = 0; col < n; ++col) {
        bias[col * 2u] = INT32_MAX - int32_t(col);
        bias[col * 2u + 1u] = INT32_MIN + int32_t(col);
        for (uint32_t feature = 0; feature < k; ++feature)
            indices[col * words + feature / 16u] |= code(col, feature)
                                                    << (2u * (feature % 16u));
    }
    std::vector<int8_t> got(size_t(m) * n * 2u + 2u, 0x5a), want = got;
    for (uint32_t row = 0; row < m; ++row)
        for (uint32_t col = 0; col < n; ++col)
            for (uint32_t learner = 0; learner < 2u; ++learner) {
                uint32_t sum = biased ? uint32_t(bias[col * 2u + learner]) : 0u;
                for (uint32_t feature = 0; feature < k; ++feature)
                    sum += uint32_t(
                        int32_t(input[(size_t(row) * k + feature) * 2u + learner]) *
                        book[code(col, feature) * 2u + learner]);
                want[1u + (size_t(row) * n + col) * 2u + learner] =
                    int8_t(uint8_t(sum));
            }
    const auto input_before = input;
    const auto indices_before = indices;
    const auto bias_before = bias;
    const int accepted =
        sve_gemm_cb4_2l_full_i8(indices.data(), words, m, n, k, input.data(), book,
                                biased ? bias.data() : nullptr, got.data() + 1u);
    if (!accepted || got != want || input != input_before ||
        indices != indices_before || bias != bias_before) {
        std::fprintf(stderr, "FAIL fallback M=%u N=%u K=%u bias=%d\n", m, n, k, biased);
        std::exit(1);
    }
}

int main() {
    if (svcntb() != 16u)
        return 2;
    unsigned cases = 0;
    for (uint32_t m : {0u, 1u, 3u, 4u, 5u, 17u})
        for (uint32_t n : {0u, 1u, 15u, 16u, 17u, 32u, 33u})
            for (uint32_t k : {0u, 1u, 15u, 16u, 17u, 127u, 128u, 129u, 255u, 256u,
                               257u, 511u, 512u, 513u, 1024u})
                for (bool bias : {false, true}) {
                    check(m, n, k, bias);
                    ++cases;
                }
    std::printf("PASS: %u independent fallback low-byte/bias/zero/tail cases\n", cases);
}
