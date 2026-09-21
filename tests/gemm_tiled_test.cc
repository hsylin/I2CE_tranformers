// Correctness test for the tiled four-learner shared-index compact GEMM.
//
// Compares the tiled SVE kernel (public wrapper and the _ex variant with a
// caller-owned pre-widened codebook and workspace) against the repository's
// scalar reference gemm_exec_compact_int_interleaved_4Learners_same_seq().
//
// Build/run via tests/run_gemm_tiled_test.sh (aarch64 g++ + qemu-aarch64).
// Exercises SVE vector lengths 128/256/512 (the runner sets the qemu VL),
// codebook sizes 2..32, and shapes whose sequence/K/N are not divisible by the
// kernel tile sizes, including K large enough to force multiple K blocks.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <arm_sve.h>

#include "gemm_exec.h"
#include "gemm_exec_internal.h"

namespace {

uint32_t g_rng = 0x1234567u;
uint32_t rnd() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng >> 8;
}

int run_case(uint32_t S, uint32_t K, uint32_t N, uint8_t bits, bool with_bias) {
    const uint32_t ipw = 32u / bits;
    const uint32_t cb = 1u << bits;
    const uint32_t nwr = (K + ipw - 1u) / ipw;
    const uint32_t vl = static_cast<uint32_t>(svcntw());

    gemm_t g;
    g.seq_len = static_cast<uint16_t>(S);
    g.input_size = static_cast<uint16_t>(K);
    g.output_size = static_cast<uint16_t>(N);
    g.n_words_row = static_cast<uint16_t>(nwr);

    std::vector<int8_t> x(static_cast<size_t>(S) * K * 4u);
    std::vector<uint32_t> w(static_cast<size_t>(N) * nwr, 0u);
    std::vector<int8_t> cb8(static_cast<size_t>(cb) * 4u);
    std::vector<int32_t> cb32(static_cast<size_t>(cb) * 4u);
    std::vector<int32_t> bias(static_cast<size_t>(N) * 4u);

    for (size_t i = 0; i < x.size(); i++) x[i] = static_cast<int8_t>(rnd() & 0xffu);
    for (uint32_t n = 0; n < N; n++)
        for (uint32_t k = 0; k < K; k++)
            w[static_cast<size_t>(n) * nwr + k / ipw] |= (rnd() & (cb - 1u)) << ((k % ipw) * bits);
    for (uint32_t i = 0; i < cb * 4u; i++) {
        cb8[i] = static_cast<int8_t>(rnd() & 0xffu);
        cb32[i] = cb8[i];
    }
    for (uint32_t i = 0; i < N * 4u; i++) bias[i] = static_cast<int32_t>(rnd() % 20001u) - 10000;
    const int32_t *b = with_bias ? bias.data() : nullptr;

    std::vector<int32_t> ref(static_cast<size_t>(S) * N * 4u, 0);
    std::vector<int32_t> o_pub(static_cast<size_t>(S) * N * 4u, 0x5a5a5a5a);
    std::vector<int32_t> o_ex(static_cast<size_t>(S) * N * 4u, 0x5a5a5a5a);

    gemm_exec_compact_int_interleaved_4Learners_same_seq(g, x.data(), w.data(), cb8.data(), b, ref.data(), bits);
    gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled(g, x.data(), w.data(), cb8.data(), b, o_pub.data(), bits);

    // _ex path: caller-owned pre-widened int32 codebook + activation workspace
    // (sized like CodebookDense's cache) exercises the workspace-backed panel.
    std::vector<int32_t> ws(static_cast<size_t>(S) * K * 4u);
    gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled_ex(
        g, x.data(), w.data(), cb8.data(), b, o_ex.data(), bits,
        cb32.data(), ws.data(), static_cast<uint32_t>(ws.size()));

    const bool bad_pub = ref != o_pub;
    const bool bad_ex = ref != o_ex;
    if (bad_pub || bad_ex) {
        std::printf("FAIL VL=%u S=%u K=%u N=%u CB=%u bias=%d | public %s | _ex %s\n",
                    vl * 32u, S, K, N, cb, with_bias ? 1 : 0,
                    bad_pub ? "MISMATCH" : "ok", bad_ex ? "MISMATCH" : "ok");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    const uint32_t shapes[][3] = {
        {1, 1, 1},    {5, 3, 3},     {7, 37, 5},    {2, 40, 2},   {9, 161, 7},
        {33, 1024, 19}, {16, 1030, 9}, {3, 3072, 4}, {64, 256, 64}, {512, 256, 64},
    };
    const uint8_t bitss[] = {1, 2, 3, 4, 5};  // codebook sizes 2, 4, 8, 16, 32
    const uint32_t vl = static_cast<uint32_t>(svcntw());

    int fails = 0;
    int cases = 0;
    for (size_t s = 0; s < sizeof(shapes) / sizeof(shapes[0]); s++) {
        for (size_t bi = 0; bi < sizeof(bitss); bi++) {
            fails += run_case(shapes[s][0], shapes[s][1], shapes[s][2], bitss[bi], ((s + bi) & 1u) != 0u);
            cases++;
        }
    }
    std::printf("VL=%u: %s (%d/%d cases passed)\n", vl * 32u,
                fails ? "FAILED" : "ALL PASSED", cases - fails, cases);
    return fails ? 1 : 0;
}
