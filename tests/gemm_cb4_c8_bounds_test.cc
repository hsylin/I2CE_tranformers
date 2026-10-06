// Place the exact I2, activation and result extents before inaccessible pages.
#include <gemm_exec_internal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

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
    // Confirm that this runtime enforces PROT_NONE (gem5 SE may ignore it).
    Guarded<uint32_t> probe(0);
    const pid_t child=fork();
    if (child<0) return 1;
    if (!child) { volatile uint32_t value=*probe.data; (void)value; _exit(0); }
    int status=0;
    if (waitpid(child,&status,0)!=child || !WIFSIGNALED(status) || WTERMSIG(status)!=SIGSEGV) {
        std::puts("FAIL runtime does not enforce PROT_NONE");return 1;
    }
    unsigned cases=0;
    for (unsigned K : {0u,1u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,511u,512u,513u,1024u})
    for (bool biased : {false,true}) {
        const unsigned nw=(K+15)/16;
        Guarded<uint32_t> packed(nw);
        Guarded<int8_t> input(K*2),output(2);
        Guarded<int32_t> cb(8),bias(2);
        for (unsigned w=0;w<nw;++w) packed.data[w]=0xe4b13927u^(w*0x13579u);
        for (unsigned k=0;k<K*2;++k) input.data[k]=int8_t(k*67u);
        for (unsigned i=0;i<8;++i) cb.data[i]=i%2?127-int(i):-128+int(i);
        bias.data[0]=INT32_MAX;bias.data[1]=INT32_MIN;
        gemm_t shape{1,uint16_t(K),1,uint16_t(nw)};
        if (!gemm_exec_cb4_2l_i8(shape,input.data,packed.data,cb.data,
                biased?bias.data:nullptr,output.data,2)) return 1;
        for(unsigned l=0;l<2;++l) {
            int64_t sum=biased?bias.data[l]:0;
            for(unsigned k=0;k<K;++k) {
                unsigned ix=(packed.data[k/16]>>(2*(k%16)))&3;
                sum+=int64_t(input.data[k*2+l])*cb.data[ix*2+l];
            }
            if (output.data[l]!=int8_t(uint8_t(uint64_t(sum)))) {
                std::printf("FAIL guarded C8 K=%u learner=%u\n",K,l);return 1;
            }
        }
        ++cases;
    }
    std::printf("PASS CB4/I2 C8 guard pages: %u cases; PROT_NONE verified\n",cases);
}
