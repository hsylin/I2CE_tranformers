#include <gemm_exec_internal.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

int main() {
    unsigned cases=0;
    for (unsigned M : {0u,1u,2u,5u})
    for (unsigned N : {0u,1u,3u,7u})
    for (unsigned K : {0u,1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,511u,512u,513u,1024u})
    for (bool biased : {false,true}) {
        unsigned NW=(K+15)/16;
        gemm_t shape{uint16_t(M),uint16_t(K),uint16_t(N),uint16_t(NW)};
        std::vector<uint32_t> indices(N*NW+1,0);
        std::vector<int8_t> input(M*K*2+2);
        int32_t cb[8];std::vector<int32_t> bias(N*2);
        for(unsigned i=0;i<8;++i) cb[i]=i%2?127-int(i%7):-128+int(i%7);
        for(unsigned i=0;i<M*K*2;++i) input[i]=int8_t(i*53+19);
        for(unsigned n=0;n<N;++n) {
            for(unsigned k=0;k<NW*16;++k) indices[n*NW+k/16]|=((k+n*3)&3)<<(2*(k%16));
            bias[2*n]=std::numeric_limits<int32_t>::max()-int(n);
            bias[2*n+1]=std::numeric_limits<int32_t>::min()+int(n);
        }
        std::vector<int8_t> want(M*N*2+16,91),got=want;
        for(unsigned m=0;m<M;++m) for(unsigned n=0;n<N;++n) for(unsigned l=0;l<2;++l) {
            int64_t sum=biased?bias[2*n+l]:0;
            for(unsigned k=0;k<K;++k) {
                unsigned ix=(indices[n*NW+k/16]>>(2*(k%16)))&3;
                sum+=int64_t(input[(m*K+k)*2+l])*cb[2*ix+l];
            }
            want[(m*N+n)*2+l]=int8_t(uint8_t(uint64_t(sum)));
        }
        if (!gemm_exec_cb4_2l_i8(shape,input.data(),indices.data(),cb,
                biased?bias.data():nullptr,got.data(),2) || got!=want) {
            std::printf("FAIL C8 M=%u N=%u K=%u bias=%d\n",M,N,K,biased);return 1;
        }
        std::fill(got.begin(),got.end(),91);const auto saved=got;
        if (gemm_exec_cb4_2l_i8(shape,input.data()+1,indices.data(),cb,nullptr,got.data(),2) || got!=saved ||
            gemm_exec_cb4_2l_i8(shape,input.data(),indices.data(),cb,nullptr,got.data(),3) || got!=saved) {
            std::puts("FAIL C8 fallback altered output");return 1;
        }
        if(M && N && K) {
            const auto input_saved=input;
            if(gemm_exec_cb4_2l_i8(shape,input.data(),indices.data(),cb,nullptr,input.data(),2) || input!=input_saved ||
               gemm_exec_cb4_2l_i8(shape,input.data(),indices.data(),cb,nullptr,input.data()+1,2) || input!=input_saved) {
                std::puts("FAIL overlapping C8 sink did not preserve fallback");return 1;
            }
            cb[3]=128;
            if(gemm_exec_cb4_2l_i8(shape,input.data(),indices.data(),cb,nullptr,got.data(),2) || got!=saved) {
                std::puts("FAIL C8 invalid codebook altered output");return 1;
            }
        }
        ++cases;
    }
    std::printf("PASS CB4/I2 C8 scalar oracle: %u cases, int8 extremes, wrapping bias, tails and fallback\n",cases);
}
