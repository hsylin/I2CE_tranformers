/*
 * Correctness of the svdot decode path in the codebook GEMM kernel.
 *
 * sve_gemm_row_compact_int8_interleaved_2Learners_same_seq() gained a fast
 * path that decodes a whole packed word into byte lanes and accumulates with
 * svdot_s32 instead of widening to int32 and accumulating with svmla_s32.
 * That path is only legal when a codebook index never straddles a byte
 * boundary, which is bits_per_cb == 2, and the fallback must stay bit-exact
 * for every other width.
 *
 * Both are checked here against an independent scalar reference, over K values
 * that exercise the predicated tail (K not a multiple of 4, of 16, or of the
 * vector length), several row counts, and both the bias and accumulate flags.
 *
 * Layouts, as the kernel defines them:
 *   packed_row[w]     32-bit word, index i at bits [i*bits, i*bits+bits-1],
 *                     so logical k = w * (32 / bits) + i
 *   in_mat[]          byte at (row * ld_in + k * 2 + learner)
 *   codebook_i32[]    index i, learner l  ->  [i * 2 + l]
 */
#include <arm_sve.h>
#include <codebooks_def.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
    const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems,
    const int8_t *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved, uint32_t codebook_size,
    int32_t *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);

namespace {

uint32_t idx_mask_of(uint8_t bits) {
    return (bits >= 32u) ? UINT32_MAX : ((1u << bits) - 1u);
}

void reference(const uint32_t *packed_row, uint32_t k_elems, const int8_t *in,
               uint32_t seq_tile, uint32_t ld_in, const int32_t *cb,
               int32_t *out, uint32_t out_col, uint32_t ld_out,
               const int32_t *bias, int add_bias, int accumulate,
               uint8_t bits_per_cb) {
    const uint32_t ipw = 32u / bits_per_cb;
    const uint32_t mask = idx_mask_of(bits_per_cb);
    for (uint32_t row = 0; row < seq_tile; row++) {
        int32_t a0 = 0, a1 = 0;
        for (uint32_t k = 0; k < k_elems; k++) {
            const uint32_t w = k / ipw, i = k % ipw;
            const uint32_t ix = (packed_row[w] >> (i * bits_per_cb)) & mask;
            a0 += static_cast<int32_t>(in[row * ld_in + k * 2u + 0u]) * cb[ix * 2u + 0u];
            a1 += static_cast<int32_t>(in[row * ld_in + k * 2u + 1u]) * cb[ix * 2u + 1u];
        }
        if (add_bias && bias != nullptr) { a0 += bias[0]; a1 += bias[1]; }
        int32_t *slot = &out[row * ld_out + out_col * 2u];
        if (accumulate) { slot[0] += a0; slot[1] += a1; }
        else            { slot[0] = a0;  slot[1] = a1;  }
    }
}

uint32_t rnd_state = 2166136261u;
uint32_t rnd() { rnd_state = rnd_state * 1664525u + 1013904223u; return rnd_state >> 8; }

int failures = 0;
long checked = 0;

void one_case(uint8_t bits, uint32_t cb_size, uint32_t K, uint32_t R,
              int bias_on, int accum) {
    const uint32_t ipw = 32u / bits;
    const uint32_t NW = (K + ipw - 1u) / ipw ? (K + ipw - 1u) / ipw : 1u;
    const uint32_t ld_in = K * 2u + 5u;       /* deliberately not tight */
    const uint32_t ld_out = 8u, out_col = 1u;

    std::vector<uint32_t> packed(NW);
    std::vector<int8_t>   in(static_cast<size_t>(R) * ld_in);
    std::vector<int32_t>  cb(cb_size * 2u);
    std::vector<int32_t>  bias(2);
    std::vector<int32_t>  seed(static_cast<size_t>(R) * ld_out);

    for (auto &v : packed) v = (rnd() << 16) ^ rnd();
    for (auto &v : in)     v = static_cast<int8_t>(rnd() % 255u) - 127;
    for (auto &v : cb)     v = static_cast<int32_t>(rnd() % 255u) - 127;
    for (auto &v : bias)   v = static_cast<int32_t>(rnd() % 2000u) - 1000;
    for (auto &v : seed)   v = static_cast<int32_t>(rnd() % 2000u) - 1000;

    std::vector<int32_t> got = seed, want = seed;
    sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
        packed.data(), NW, K, in.data(), R, ld_in, cb.data(), cb_size,
        got.data(), out_col, ld_out, bias.data(), bias_on, accum, bits);
    reference(packed.data(), K, in.data(), R, ld_in, cb.data(),
              want.data(), out_col, ld_out, bias.data(), bias_on, accum, bits);

    for (size_t i = 0; i < got.size(); i++) {
        checked++;
        if (got[i] != want[i]) {
            if (failures < 10) {
                std::printf("  mismatch bits=%u cb=%u K=%u R=%u bias=%d acc=%d "
                            "slot=%zu got=%d want=%d\n",
                            bits, cb_size, K, R, bias_on, accum, i, got[i], want[i]);
            }
            failures++;
        }
    }
}

void sweep(uint8_t bits, uint32_t cb_size, const char *label) {
    static const uint32_t Ks[] = {0,1,2,3,4,5,7,8,15,16,17,31,32,33,48,63,64,65,
                                  100,127,128,255,256,257,512,1023,1024};
    static const uint32_t Rs[] = {1,2,3,5,8};
    const long before = failures;
    for (uint32_t K : Ks)
        for (uint32_t R : Rs)
            for (int bias_on = 0; bias_on < 2; bias_on++)
                for (int accum = 0; accum < 2; accum++)
                    one_case(bits, cb_size, K, R, bias_on, accum);
    std::printf("%s %s: bits_per_cb=%u codebook_size=%u\n",
                (failures == before) ? "PASS" : "FAIL", label, bits, cb_size);
}

}  // namespace

int main() {
    std::printf("vector length: %llu bits, CB_SIZE=%d, BITS_PER_CB=%d\n",
                static_cast<unsigned long long>(svcntb()) * 8,
                static_cast<int>(CB_SIZE), static_cast<int>(BITS_PER_CB));

    /* bits_per_cb == 2 is the only width the svdot path accepts. It needs at
     * least 4 codebook entries because a 2-bit index reaches 3. */
    if (CB_SIZE >= 4) sweep(2, 4, "svdot path (2-bit indexes)");

    /* The generated width exercises whichever path the guard selects; for
     * BITS_PER_CB == 3 that is the svmla fallback. */
    if (BITS_PER_CB != 2 && CB_SIZE >= (1 << BITS_PER_CB)) {
        sweep(static_cast<uint8_t>(BITS_PER_CB), CB_SIZE, "generated width");
    }

    std::printf("%ld output values checked, %d mismatches\n", checked, failures);
    if (failures == 0) std::printf("PASS gemm_codebook_sdot_test\n");
    else               std::printf("FAIL gemm_codebook_sdot_test\n");
    return failures != 0;
}
