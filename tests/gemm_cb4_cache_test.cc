// Correctness only: independent scalar oracle, fixed seeds, no timing output.
#include <gemm_SVE.h>
#include <arm_sve.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <limits>
#include <codebooks_def.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>

extern uint64_t i2ce_cb4_cache_calls, i2ce_cb4_cache_outputs;
extern uint64_t i2ce_cb4_cache_chunks, i2ce_cb4_cache_decoded;
extern int i2ce_cb4_cache_fail_alloc;
static unsigned failed_allocations;
int i2ce_cb4_cache_test_allocate(void **p, size_t alignment, size_t bytes) {
    if (i2ce_cb4_cache_fail_alloc) { *p=nullptr; ++failed_allocations; return ENOMEM; }
    return posix_memalign(p,alignment,bytes);
}
static void require(bool b, const char *message) {
    if (!b) { std::printf("FAIL %s\n",message); std::exit(1); }
}
static bool eligible() { return CB_SIZE==4 && N_LEARNERS==2 && svcntb()==16; }

template<class T> struct Guarded {
    size_t bytes; void *mapping; T *data;
    explicit Guarded(size_t count) {
        const size_t page=size_t(sysconf(_SC_PAGESIZE));
        bytes=((count*sizeof(T)+page-1)/page+1)*page;
        mapping=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        require(mapping!=MAP_FAILED,"mmap");
        char *end=static_cast<char*>(mapping)+bytes-page;
        require(!mprotect(end,page,PROT_NONE),"mprotect");
        data=reinterpret_cast<T*>(end-count*sizeof(T));
    }
    ~Guarded(){munmap(mapping,bytes);}
};
static void bounds_tests() {
    Guarded<int8_t> probe(0);
    std::fflush(nullptr);
    const pid_t child=fork(); require(child>=0,"fork guard probe");
    if(!child) {volatile int8_t value=*probe.data; (void)value; _exit(0);}
    int status=0;
    require(waitpid(child,&status,0)==child && WIFSIGNALED(status) && WTERMSIG(status)==SIGSEGV,
            "runtime does not enforce guard pages");
    for(size_t K:{0u,1u,15u,16u,17u,127u,128u,129u,511u,512u,513u}) {
        const size_t M=17,N=33,nw=(K+15)/16;
        Guarded<int8_t> x(M*K*2),y(M*N*2);
        Guarded<uint32_t> indices(N*nw);
        Guarded<int32_t> cb(8),bias(N*2);
        for(size_t i=0;i<M*K*2;++i)x.data[i]=int8_t(i*67u);
        for(size_t i=0;i<N*nw;++i)indices.data[i]=0xe4b13927u;
        for(size_t i=0;i<8;++i)cb.data[i]=i%2?127:-128;
        for(size_t i=0;i<N*2;++i)bias.data[i]=i%2?INT32_MAX:INT32_MIN;
        require(sve_gemm_cb4_cache_i8(indices.data,nw,M,N,K,x.data,cb.data,bias.data,y.data,2),
                "guarded call rejected");
        for(size_t s=0;s<M;++s)for(size_t o=0;o<N;++o)for(size_t l=0;l<2;++l) {
            int64_t sum=bias.data[2*o+l];
            for(size_t k=0;k<K;++k)
                sum+=int64_t(x.data[(s*K+k)*2+l])*cb.data[2*((indices.data[o*nw+k/16]>>(2*(k%16)))&3)+l];
            require(y.data[(s*N+o)*2+l]==int8_t(uint8_t(uint64_t(sum))),"guarded oracle");
        }
    }
}

int main() {
    unsigned cases=0;
    std::vector<std::array<size_t,3>> shapes={
        {0,3,17},{3,0,17},{1,1,0},{17,33,0},{1,1,1},{15,31,15},
        {16,32,16},{17,33,17},{1,1,31},{1,1,32},{1,1,33},
        {15,31,127},{16,32,128},{17,33,129},{127,1,17},{128,1,17},
        {129,1,17},{1,127,17},{1,128,17},{1,129,17},
        {1,1,511},{1,1,512},{1,1,513},{17,33,513},{129,129,129},
        {17,129,513},{129,33,513},{2,4,65535}};
    for (const auto shape:shapes) for (bool biased:{false,true}) for (size_t offset:{0u,1u}) {
        const size_t M=shape[0],N=shape[1],K=shape[2],nw=(K+15)/16;
        std::vector<int8_t> x(M*K*2+2),y(M*N*2+32,91),want=y;
        int8_t *input=x.data()+offset;
        std::vector<uint32_t> idx(N*nw+1,0);
        int32_t cb[8]={-128,127,0,-1,1,0,126,-127};
        std::vector<int32_t> b(N*2);
        for(size_t i=0;i<M*K*2;++i) input[i]=int8_t(i*53u+19u);
        for(size_t o=0;o<N;++o) {
            for(size_t k=0;k<nw*16;++k) idx[o*nw+k/16]|=((k+o*3)&3)<<(2*(k%16));
            b[2*o]=INT32_MAX-int(o); b[2*o+1]=INT32_MIN+int(o);
        }
        for(size_t s=0;s<M;++s) for(size_t o=0;o<N;++o) for(size_t l=0;l<2;++l) {
            int64_t sum=biased?b[2*o+l]:0;
            for(size_t k=0;k<K;++k)
                sum+=int64_t(input[(s*K+k)*2+l])*cb[2*((idx[o*nw+k/16]>>(2*(k%16)))&3)+l];
            want[(s*N+o)*2+l]=int8_t(uint8_t(uint64_t(sum)));
        }
        const int ok=sve_gemm_cb4_cache_i8(idx.data(),nw,M,N,K,input,cb,biased?b.data():nullptr,y.data(),2);
        if(eligible()) require(ok && y==want,"cache scalar oracle or output guard");
        else require(!ok && y==std::vector<int8_t>(y.size(),91),"unsupported dispatch wrote output");
        ++cases;
    }
    // Each rejected invocation must preserve the entire output before fallback.
    uint32_t idx[8]={}; int8_t x[256]={},y[256]; int32_t cb[8]={},b[64]={};
    auto unchanged=[&](size_t M,size_t N,size_t K,size_t nw,uint8_t bits) {
        memset(y,91,sizeof(y));
        require(!sve_gemm_cb4_cache_i8(idx,nw,M,N,K,x,cb,b,y,bits),"invalid call accepted");
        for(auto c:y) require(c==91,"invalid call wrote output");
    };
    unchanged(65536,1,1,1,2); unchanged(1,65536,1,1,2);
    unchanged(1,1,65536,4096,2); unchanged(SIZE_MAX,1,1,1,2);
    unchanged(65535,65535,1,1,2); unchanged(1,1,17,1,2);
    unchanged(1,1,16,1,4);
    require(!sve_gemm_cb4_cache_i8(idx,1,1,1,16,x,cb,b,x,2),"X alias accepted");
    require(!sve_gemm_cb4_cache_i8(idx,1,1,1,16,x,cb,b,reinterpret_cast<int8_t*>(idx),2),"I alias accepted");
    require(!sve_gemm_cb4_cache_i8(idx,1,1,1,16,x,cb,b,reinterpret_cast<int8_t*>(cb),2),"CB alias accepted");
    require(!sve_gemm_cb4_cache_i8(idx,1,1,1,16,x,cb,b,reinterpret_cast<int8_t*>(b),2),"bias alias accepted");
    require(!sve_gemm_cb4_cache_i8(idx,1,1,1,16,x,cb,b,reinterpret_cast<int8_t*>(UINTPTR_MAX),2),"address overflow accepted");
    cb[3]=128; unchanged(1,1,16,1,2); cb[3]=0;
    if(eligible()) {
        bounds_tests();
        i2ce_cb4_cache_fail_alloc=1; unchanged(1,1,16,1,2);
        require(failed_allocations==1,"allocator failure branch not exercised");
        i2ce_cb4_cache_fail_alloc=0;
        // The caller can retain the existing E05 computation after rejection.
        require(sve_gemm_cb4_2l_full_i8(idx,1,1,1,16,x,cb,b,y),"E05 fallback failed");
        require(y[0]==0 && y[1]==0,"fallback result");
        i2ce_cb4_cache_calls=i2ce_cb4_cache_outputs=i2ce_cb4_cache_chunks=i2ce_cb4_cache_decoded=0;
        require(sve_gemm_cb4_cache_i8(idx,2,2,4,32,x,cb,b,y,2),"counter example rejected");
        require(i2ce_cb4_cache_calls==1 && i2ce_cb4_cache_outputs==8 && i2ce_cb4_cache_chunks==16,
                "single-output path not exercised");
        require(i2ce_cb4_cache_decoded==128,"decoded panel accounting");
    } else require(i2ce_cb4_cache_calls==0,"unsupported config entered cache path");
    std::printf("PASS cache oracle %u cases; SVE%zu CB%d L%d; tails/wrap/aliases/range/fault injection\n",
                cases,size_t(svcntb()*8),CB_SIZE,N_LEARNERS);
}
