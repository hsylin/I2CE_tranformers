// Guard pages make both speculative-width reads and tail overreads observable.
#include <gemm_SVE.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstdint>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>

template<class T> struct Guarded {
    size_t page; void *mapping; T *data;
    explicit Guarded(size_t count) : page(size_t(sysconf(_SC_PAGESIZE))) {
        if (count*sizeof(T)>page) std::abort();
        mapping=mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if (mapping==MAP_FAILED || mprotect(static_cast<char*>(mapping)+page,page,PROT_NONE)) std::abort();
        data=reinterpret_cast<T*>(static_cast<char*>(mapping)+page-count*sizeof(T));
    }
    ~Guarded(){munmap(mapping,page*2);}
};
int main() {
    unsigned cases=0;
    for (unsigned K : {0u,1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,511u,512u,513u,1024u})
    for (unsigned NW : {0u,1u,2u,3u,4u,7u,8u,9u,16u,32u,33u,64u}) {
        Guarded<uint32_t> packed(NW);
        Guarded<int8_t> input(K*2u);
        int32_t cb[8],out[2]={0,0},bias[2]={53,-71};
        for (unsigned i=0;i<8;++i) cb[i]=i%2?127:-128;
        for (unsigned w=0;w<NW;++w) packed.data[w]=0xfedcba98u^(w*0x13579u);
        for (unsigned i=0;i<K*2u;++i) input.data[i]=int8_t(i*67u);
        sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
            packed.data,NW,K,input.data,1,K*2,cb,4,out,0,2,bias,1,0,2);
        // I2 row ABI consumes one packed word per complete 16-K group.
        unsigned effective=NW*16;
        if (effective>K) effective=K;
        for(unsigned l=0;l<2;++l) {
            int64_t sum=bias[l];
            for(unsigned k=0;k<effective;++k) {
                unsigned word=k/16<NW?packed.data[k/16]:0;
                unsigned ix=(word>>(2*(k%16)))&3;
                sum+=int64_t(input.data[k*2+l])*cb[ix*2+l];
            }
            if (out[l]!=sum) {std::printf("FAIL guarded K=%u NW=%u l=%u\n",K,NW,l);return 1;}
        }
        ++cases;
    }
    std::printf("PASS guarded CB4/I2 tails: %u cases\n",cases);
}
