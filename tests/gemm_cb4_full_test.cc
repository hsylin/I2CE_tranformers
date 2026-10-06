// Full matrix oracle: independent scalar index extraction and int64 products.
#include <gemm_SVE.h>
#include <gemm_exec_internal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    unsigned cases = 0;
    for (unsigned M : {0u, 1u, 2u, 5u})
    for (unsigned N : {0u, 1u, 3u, 7u})
    for (unsigned K : {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 127u, 128u, 129u, 511u, 512u, 513u, 1024u})
    for (unsigned offset : {0u, 1u})
    for (bool biased : {false, true}) {
        const unsigned NW = (K + 15u) / 16u;
        std::vector<uint32_t> packed(N * NW + 1u, 0);
        std::vector<int8_t> storage(M * K * 2u + 2u, -128);
        int8_t *in = storage.data() + offset;
        int8_t cb8[8]; int32_t cb[8];
        std::vector<int32_t> bias(N * 2u);
        // Keep every table entry in range while covering the signed extremes.
        for (unsigned i = 0; i < 8; ++i) cb8[i] = cb[i] = i % 2 ? 127 - int(i % 7) : -128 + int(i % 7);
        for (unsigned i = 0; i < M * K * 2u; ++i) in[i] = int8_t((i * 53u + 19u) & 255u);
        for (unsigned n = 0; n < N; ++n) {
            for (unsigned k = 0; k < NW * 16u; ++k)
                packed[n * NW + k / 16u] |= ((k + 3u * n) & 3u) << (2u * (k % 16u));
            bias[2u*n] = 103; bias[2u*n+1] = -177;
        }
        std::vector<int32_t> got(M * N * 2u + 8u, 0x34561234);
        std::vector<int32_t> want = got;
        for (unsigned m=0; m<M; ++m) for (unsigned n=0; n<N; ++n)
        for (unsigned l=0; l<2; ++l) {
            int64_t sum = biased ? bias[2*n+l] : 0;
            for (unsigned k=0; k<K; ++k) {
                unsigned ix = (packed[n*NW+k/16] >> (2*(k%16))) & 3u;
                sum += int64_t(in[(m*K+k)*2+l]) * cb[ix*2+l];
            }
            uint32_t bits=uint32_t(sum);std::memcpy(&want[(m*N+n)*2+l],&bits,4);
        }
        gemm_t shape{uint16_t(M),uint16_t(K),uint16_t(N),uint16_t(NW)};
        gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
            shape,in,packed.data(),cb8,biased?bias.data():nullptr,got.data(),2,
            cb,nullptr,0);
        if (got != want) { std::printf("FAIL wrapper M=%u N=%u K=%u off=%u\n",M,N,K,offset); return 1; }
        // Exercise the hoisted entry directly, including empty axes.
        std::fill(got.begin(),got.end(),0x34561234);
        if (!sve_gemm_cb4_2l_full_i32(packed.data(),NW,M,N,K,in,cb,
                                     biased?bias.data():nullptr,got.data()) || got!=want) {
            std::printf("FAIL prepared M=%u N=%u K=%u\n",M,N,K); return 1;
        }
        cb[3]=128;
        std::vector<int32_t> saved=got;
        if (sve_gemm_cb4_2l_full_i32(packed.data(),NW,M,N,K,in,cb,nullptr,got.data()) || got!=saved) {
            std::puts("FAIL ineligible codebook modified output");return 1;
        }
        ++cases;
    }
    std::printf("PASS full CB4/I2 oracle: %u cases, both learners and guard values\n",cases);
}
