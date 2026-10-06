// Validation-only integration test, linked against each exact study source.
#include "codebookDense.h"
#include <arm_sve.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
extern uint64_t i2ce_cb4_cache_calls;
extern int i2ce_cb4_cache_fail_alloc;
static unsigned failures;
int i2ce_cb4_cache_test_allocate(void **p,size_t a,size_t b) {
    if(i2ce_cb4_cache_fail_alloc){*p=nullptr;++failures;return ENOMEM;}
    return posix_memalign(p,a,b);
}
static void check(bool ok,const char *why){if(!ok){std::printf("FAIL %s\n",why);std::exit(1);}}
int main() {
    const size_t M=17,N=4,K=32,nw=2;
    uint32_t indices[N*nw]; for(auto &i:indices)i=0xe4b13927;
    int8_t cb[8]={-128,127,0,-1,1,0,126,-127};
    CodebookDenseConfig config{};config.input_size=K;config.output_size=N;
    config.n_words_row=nw;config.bits_per_cb=2;config.n_learners=2;
    config.weight_idx=indices;config.codebook_int8=cb;config.codebook_int8_interleaved=cb;
    CodebookDense layer(config);
    std::vector<int8_t> input(M*K*2),got(M*N*2,91),want=got;
    for(size_t i=0;i<input.size();++i)input[i]=int8_t(i*53+19);
    for(size_t s=0;s<M;++s)for(size_t o=0;o<N;++o)for(size_t l=0;l<2;++l){
        int64_t sum=0;
        for(size_t k=0;k<K;++k)sum+=int64_t(input[(s*K+k)*2+l])*cb[2*((indices[o*nw+k/16]>>(2*(k%16)))&3)+l];
        want[(s*N+o)*2+l]=int8_t(uint8_t(uint64_t(sum)));
    }
    layer.computeInterleaved2LearnersToInt8(M,input.data(),got.data());
    check(got==want,"consumer output");
    check(i2ce_cb4_cache_calls==(svcntb()==16?1u:0u),"consumer dispatch did not reach expected path");
    i2ce_cb4_cache_calls=0;i2ce_cb4_cache_fail_alloc=1;
    got.assign(got.size(),91);layer.computeInterleaved2LearnersToInt8(M,input.data(),got.data());
    check(got==want && i2ce_cb4_cache_calls==0,"allocation-failure fallback output");
    check(failures==(svcntb()==16?1u:0u),"allocator fault injection not exercised");
    i2ce_cb4_cache_fail_alloc=0;
    auto alias=input;layer.computeInterleaved2LearnersToInt8(M,alias.data(),alias.data());
    check(std::vector<int8_t>(alias.begin(),alias.begin()+want.size())==want,"alias-safe fallback output");
    check(i2ce_cb4_cache_calls==0,"overlap entered cache path");
    bool rejected=false;try{layer.computeInterleaved2LearnersToInt8(65536,input.data(),got.data());}
    catch(const std::length_error&){rejected=true;}check(rejected,"sequence narrowed before guard");
    for(int mode=0;mode<3;++mode){
        auto bad=config;
        bad.input_size=mode==0?65536:mode==2?65532:K;
        bad.output_size=mode==1?65536:N;bad.n_words_row=(bad.input_size+15)/16;
        CodebookDense huge(bad);rejected=false;
        try{huge.computeInterleaved2LearnersToInt8(mode==2?65535:1,input.data(),got.data());}
        catch(const std::length_error&){rejected=true;}check(rejected,"dimension/product guard");
    }
    config.same_seq=false;CodebookDense distinct(config);rejected=false;
    try{distinct.computeInterleaved2LearnersToInt8(M,input.data(),got.data());}
    catch(const std::runtime_error&){rejected=true;}check(rejected,"different-index mode accepted");
    std::printf("PASS consumer SVE%zu: actual dispatch, allocation-failure fallback, alias fallback, pre-narrowing guards, shared-mode guard\n",size_t(svcntb()*8));
}
