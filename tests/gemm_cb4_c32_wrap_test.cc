// Force the C32 APIs; checking only the cache/C8 result cannot cover these paths.
#include <gemm_SVE.h>
#include <gemm_exec_internal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <climits>
#include <vector>

static uint32_t bits(int32_t value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

int main() {
    unsigned cases = 0;
    for (unsigned K : {0u, 1u, 15u, 16u, 17u, 129u, 1024u})
    for (bool wide : {false, true})
    for (bool add_bias : {false, true})
    for (bool accumulate : {false, true}) {
        const unsigned nw = (K + 15u)/16u;
        std::vector<uint32_t> indices(nw + 1u, 0xe4e4e4e4u);
        std::vector<int8_t> x(K*2u + 2u, 127);
        int32_t cb[8];
        int8_t cb8[8];
        for (unsigned i=0; i<8; ++i) {
            cb8[i] = i%2 ? -128 : 127;
            cb[i] = wide ? (i%2 ? -129 : 128) : cb8[i];
        }
        int32_t bias[2] = {INT32_MAX, INT32_MIN};
        int32_t out[6] = {73, 73, INT32_MAX, INT32_MIN, 73, 73};
        uint32_t expected[2];
        for (unsigned l=0; l<2; ++l) {
            int64_t sum = accumulate ? out[2+l] : 0;
            if (add_bias) sum += bias[l];
            for (unsigned k=0; k<K; ++k) {
                const unsigned ix = (indices[k/16] >> (2*(k%16))) & 3u;
                sum += int64_t(x[2*k+l])*cb[2*ix+l];
            }
            expected[l] = uint32_t(sum);
        }
        // int8-range tables use SDOT; out-of-range tables force widening SVE.
        sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
            indices.data(), nw, K, x.data(), 1, K*2u, cb, 4,
            out, 1, 6, bias, add_bias, accumulate, 2);
        if (bits(out[2])!=expected[0] || bits(out[3])!=expected[1] ||
            out[0]!=73 || out[1]!=73 || out[4]!=73 || out[5]!=73) {
            std::printf("FAIL C32 row K=%u wide=%d bias=%d accumulate=%d\n", K,wide,add_bias,accumulate);
            return 1;
        }
        if (!wide && !accumulate) {
            gemm_t shape{1, uint16_t(K), 1, uint16_t(nw)};
            int32_t scalar[2] = {}, wrapper[2] = {}, prepared[2] = {};
            // Odd byte address forces the wrapper's scalar fallback.
            gemm_exec_compact_int_interleaved_2Learners_same_seq(
                shape,x.data()+1,indices.data(),cb8,add_bias?bias:nullptr,scalar,2);
            gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
                shape,x.data()+1,indices.data(),cb8,add_bias?bias:nullptr,wrapper,2,cb,nullptr,0);
            if (!sve_gemm_cb4_2l_full_i32(indices.data(),nw,1,1,K,x.data(),cb,add_bias?bias:nullptr,prepared)) return 1;
            for (unsigned l=0; l<2; ++l)
                if (bits(scalar[l])!=expected[l] || bits(wrapper[l])!=expected[l] || bits(prepared[l])!=expected[l]) {
                    std::puts("FAIL scalar/wrapper/prepared C32"); return 1;
                }
        }
        ++cases;
    }
    std::printf("PASS C32 wrap: %u row cases; SDOT/widening/empty K, scalar fallback and prepared sink\n", cases);
}
