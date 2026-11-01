#include "gemm_SVE.h"
#include "gemm_exec.h"
#include "Full_NN/src/gemm_cb4_adapter.h"
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main() {
    if (svcntb() != 16)
        return 2;
    unsigned cases = 0;
    for (uint16_t m : {1, 3, 4, 5, 16})
        for (uint16_t n : {1, 15, 16, 17, 32, 33})
            for (uint16_t k :
                 {0, 1, 15, 16, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513})
                for (bool biased : {false, true}) {
                    const uint16_t words = (k + 15u) / 16u;
                    gemm_t shape{m, k, n, words};
                    std::vector<int32_t> a(size_t(m) * k * 2), bias(size_t(n) * 2);
                    std::vector<uint32_t> indices(size_t(n) * words, 0);
                    const int32_t cb[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
                    for (size_t i = 0; i < a.size(); ++i)
                        a[i] = int(i * 37u % 256) - 128;
                    for (size_t i = 0; i < bias.size(); ++i) {
                        uint32_t bits = 0x7fffffffU + uint32_t(i) * 19117u;
                        std::memcpy(&bias[i], &bits, 4);
                    }
                    for (size_t j = 0; j < n; ++j)
                        for (size_t f = 0; f < k; ++f)
                            indices[j * words + f / 16] |=
                                uint32_t((j * 3 + f * 5 + f / 7) % 4) << (2 * (f % 16));
                    std::vector<int32_t> got(size_t(m) * n * 2 + 2, 0x12345678),
                        expected = got;
                    for (size_t i = 0; i < m; ++i)
                        for (size_t j = 0; j < n; ++j)
                            for (size_t l = 0; l < 2; ++l) {
                                uint32_t sum = biased ? uint32_t(bias[j * 2 + l]) : 0;
                                for (size_t f = 0; f < k; ++f)
                                    sum += uint32_t(
                                        a[(i * k + f) * 2 + l] *
                                        cb[((j * 3 + f * 5 + f / 7) % 4) * 2 + l]);
                                std::memcpy(&expected[1 + (i * n + j) * 2 + l], &sum,
                                            4);
                            }
                    auto before_a = a;
                    auto before_i = indices;
                    i2ce_cb4_scratch_stats accounting{};
                    gemm_cb4_sdot_from_widened(shape, a.data(), indices.data(), cb,
                                               biased ? bias.data() : nullptr,
                                               got.data() + 1, &accounting);
                    if (got != expected || a != before_a || indices != before_i) {
                        std::fprintf(stderr, "FAIL adapter M=%u N=%u K=%u\n", m, n, k);
                        return 1;
                    }
                    ++cases;
                }
    std::printf("PASS: %u independent scalar int32-adapter cases including guarded "
                "output and tails\n",
                cases);
}
