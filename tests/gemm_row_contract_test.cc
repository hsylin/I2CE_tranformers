// Independent bounded-value reference for the two-learner SVE entry points.
// The tested functions are linked from gemm_SVE.c under the declared policy.
#include "gemm_SVE.h"
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    if (svcntb() != 16) return 2;
    unsigned row_cases = 0, dense_cases = 0;
    for (uint32_t m : {1u, 3u, 4u, 5u, 16u}) {
        for (uint32_t n : {1u, 15u, 16u, 17u, 32u, 33u}) {
            for (uint32_t k : {0u, 1u, 15u, 16u, 17u, 127u, 128u, 129u,
                               255u, 256u, 257u, 511u, 512u, 513u}) {
                const uint32_t words = (k + 15u) / 16u;
                const uint32_t ld_a = 2u*k + 6u, ld_c = 2u*n + 4u;
                std::vector<int32_t> a(m*ld_a, 0), bias(2u*n);
                std::vector<uint32_t> indices(n*(words ? words : 1u), 0);
                const int32_t cb[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
                for (size_t i=0; i<a.size(); ++i) a[i] = int(i*37u%256u)-128;
                for (size_t i=0; i<bias.size(); ++i) bias[i] = int(i*113u%2000u)-1000;
                for (uint32_t j=0; j<n; ++j)
                    for (uint32_t f=0; f<k; ++f)
                        indices[j*words+f/16u] |= ((j*3u+f*5u+f/7u)%4u) << (2u*(f%16u));
                const auto a_before = a;
                const auto indices_before = indices;
                for (bool biased : {false,true}) {
                    for (bool accumulate : {false,true}) {
                        std::vector<int32_t> got(m*ld_c+2u, 0x12345678), expected=got;
                        for (uint32_t j=0; j<n; ++j) {
                            for (uint32_t i=0; i<m; ++i) {
                                for (uint32_t l=0; l<2; ++l) {
                                    int64_t sum = biased ? bias[2u*j+l] : 0;
                                    if (accumulate) sum += expected[1u+i*ld_c+2u*j+l];
                                    for (uint32_t f=0; f<k; ++f)
                                        sum += int64_t(a[i*ld_a+2u*f+l])*cb[2u*((j*3u+f*5u+f/7u)%4u)+l];
                                    expected[1u+i*ld_c+2u*j+l] = int32_t(sum);
                                }
                            }
                            sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
                                indices.data()+j*words, words, k, a.data(), m, ld_a,
                                cb, 4u, got.data()+1u, j, ld_c,
                                biased ? bias.data()+2u*j : nullptr,
                                biased, accumulate, 2u);
                        }
                        if (got!=expected || a!=a_before || indices!=indices_before) {
                            std::fprintf(stderr,"FAIL row M=%u N=%u K=%u bias=%d accumulate=%d\n",m,n,k,biased,accumulate);
                            return 1;
                        }
                        ++row_cases;
                    }
                }
                std::vector<int32_t> left(m*k*2u+1u), right(n*k*2u+1u);
                for (size_t i=0; i<left.size(); ++i) left[i]=int(i*37u%256u)-128;
                for (size_t i=0; i<right.size(); ++i) right[i]=int(i*91u%256u)-128;
                const auto left_before=left, right_before=right;
                std::vector<int32_t> got(m*n*2u+2u,0x12345678), expected=got;
                for (uint32_t i=0;i<m;++i) for (uint32_t j=0;j<n;++j) for (uint32_t l=0;l<2;++l) {
                    int64_t sum=0;
                    for (uint32_t f=0;f<k;++f) sum+=int64_t(left[(i*k+f)*2u+l])*right[(j*k+f)*2u+l];
                    expected[1u+(i*n+j)*2u+l]=int32_t(sum);
                }
                sve_gemm_dense_int8_interleaved_2Learners(left.data(),right.data(),m,n,k,got.data()+1u);
                if (got!=expected || left!=left_before || right!=right_before) {
                    std::fprintf(stderr,"FAIL dense M=%u N=%u K=%u\n",m,n,k);
                    return 1;
                }
                ++dense_cases;
            }
        }
    }
    std::printf("PASS: %u independent widened-row cases and %u dense cases, bias/accumulate/tails/output sentinels/input immutability\n",row_cases,dense_cases);
}
