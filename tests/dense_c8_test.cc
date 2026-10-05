#include "interleavedPipeline.h"
#include "softmax.h"
#ifdef SIMD
#include <gemm_SVE.h>
#endif
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
    unsigned cases=0;
    for(unsigned M : {0u,1u,2u,5u}) for(unsigned N : {0u,1u,3u,7u})
    for(unsigned K : {0u,1u,7u,15u,16u,17u,31u,32u,63u,64u,65u,257u,512u})
    for(unsigned offset : {0u,1u}) {
        std::vector<int8_t> aa(M*K*2+2),bb(N*K*2+2);
        int8_t *a=aa.data()+offset,*b=bb.data()+offset;
        for(unsigned i=0;i<M*K*2;++i)a[i]=int8_t(i*53u+128u);
        for(unsigned i=0;i<N*K*2;++i)b[i]=int8_t(i*71u+127u);
        std::vector<int8_t> got(M*N*2+16,93),want=got;
        for(unsigned m=0;m<M;++m)for(unsigned n=0;n<N;++n)for(unsigned l=0;l<2;++l) {
            int64_t sum=0;
            for(unsigned k=0;k<K;++k)sum+=int64_t(a[(m*K+k)*2+l])*b[(n*K+k)*2+l];
            want[(m*N+n)*2+l]=int8_t(uint8_t(uint64_t(sum)));
        }
        matmulInterleaved2LearnersToInt8(a,b,M,N,K,got.data());
        if(got!=want){std::printf("FAIL dense C8 M=%u N=%u K=%u off=%u\n",M,N,K,offset);return 1;}
        for(unsigned which : {0u,1u}) for(unsigned overlap_offset : {0u,1u}) {
            // Test aliasing either operand, including a partial byte overlap.
            const unsigned input_bytes=(which?N:M)*K*2u;
            std::vector<int8_t> aliased(std::max(input_bytes,M*N*2u+overlap_offset)+16,89);
            std::copy(which?b:a,(which?b:a)+input_bytes,aliased.begin());
            const auto before=aliased;
            matmulInterleaved2LearnersToInt8(which?a:aliased.data(),which?aliased.data():b,
                M,N,K,aliased.data()+overlap_offset);
            for(unsigned i=0;i<aliased.size();++i) {
                int8_t expected=(i>=overlap_offset && i<overlap_offset+M*N*2u)
                    ?want[i-overlap_offset]:before[i];
                if(aliased[i]!=expected){std::puts("FAIL dense C8 alias fallback");return 1;}
            }
        }
#ifdef SIMD
        std::vector<int8_t> direct(M*N*2+16,93);
        sve_gemm_dense_int8_interleaved_2Learners_to_int8(a,b,M,N,K,direct.data());
        if(direct!=want){std::puts("FAIL direct dense sink");return 1;}
#endif
        // Test the real PV consumer: signed int8 cast precedes >> 6.
        Softmax softmax;softmax.post_softmax_interleaved2D(got.data(),M,N);
        for(unsigned i=0;i<M*N*2;++i) {
            // Mathematical floor division is independent of signed-shift code.
            int v=want[i];want[i]=int8_t(v>=0?v/64:-((-v+63)/64));
        }
        if(got!=want){std::puts("FAIL cast before PV shift");return 1;}
        ++cases;
    }
    std::printf("PASS dense C8/PV oracle: %u cases\n",cases);
}
