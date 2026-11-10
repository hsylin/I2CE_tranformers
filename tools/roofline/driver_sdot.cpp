// ERT-derived integer scan. Upstream source/license and exact plan are retained.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <initializer_list>
#ifndef HOST_ORACLE
#include <arm_sve.h>
#include <gem5/m5ops.h>
#endif
using Kernel=void(*)(const int8_t*,const int8_t*,uint32_t*,uint64_t,uint64_t);
struct Entry {unsigned d,u;Kernel fn;};
struct Block {uint64_t n;unsigned entry,first_id,count;uint64_t passes[7];};
#include "plan.hh"
static uint32_t seedc(size_t i){return UINT32_C(0xf0000000)+UINT32_C(2654435761)*uint32_t(i);}
static int8_t av(size_t i){return int8_t(int((i*17+23)%256)-128);}
static int8_t wv(size_t i){return int8_t(int((i*29+71)%256)-128);}
static int32_t dots[64];
static void make_dots(){for(unsigned i=0;i<64;i++){int32_t v=0;for(unsigned q=0;q<4;q++)v+=int32_t(av(4*i+q))*int32_t(wv(4*i+q));dots[i]=v;}}
struct Buffer {unsigned char*raw;size_t bytes;explicit Buffer(size_t n):raw(nullptr),bytes(n){if(posix_memalign((void**)&raw,64,n+128))std::abort();std::memset(raw,0xa5,n+128);}~Buffer(){std::free(raw);}unsigned char*data(){return raw+64;}bool guard()const{for(size_t i=0;i<64;i++)if(raw[i]!=0xa5||raw[bytes+64+i]!=0xa5)return false;return true;}};
static void init(int8_t*a,int8_t*w,uint32_t*c,size_t n){for(size_t i=0;i<16*n;i++){a[i]=av(i);w[i]=wv(i);}for(size_t i=0;i<4*n;i++)c[i]=seedc(i);}
static uint32_t value(size_t i,unsigned d,uint64_t p){return seedc(i)+uint32_t(int64_t(dots[i%64])*int64_t(d)*int64_t(p));}
static bool checkc(const uint32_t*c,size_t n,unsigned d,uint64_t p){for(size_t i=0;i<4*n;i++)if(c[i]!=value(i,d,p)){std::fprintf(stderr,"C mismatch %zu\n",i);return false;}return true;}
static bool checkaw(const int8_t*a,const int8_t*w,size_t n){for(size_t i=0;i<16*n;i++)if(a[i]!=av(i)||w[i]!=wv(i))return false;return true;}
static uint64_t hashc(const uint32_t*c,size_t n){uint64_t h=UINT64_C(14695981039346656037);for(size_t i=0;i<4*n;i++)for(unsigned q=0;q<4;q++){h^=(c[i]>>(q*8))&255;h*=UINT64_C(1099511628211);}return h;}
int main(int argc,char**argv){
 make_dots();
#ifndef HOST_ORACLE
 if(svcntb()!=16)return 2;
#endif
 const bool golden=argc==2&&!std::strcmp(argv[1],"--golden");
 const bool verify=argc==2&&!std::strcmp(argv[1],"--verify");
 const bool scan=argc==3&&!std::strcmp(argv[1],"--experiment")&&std::strlen(argv[2])==1&&argv[2][0]>='0'&&argv[2][0]<='2';
 if(!golden&&!verify&&!scan)return 2;
#ifdef HOST_ORACLE
 if(!golden&&!verify)return 2;
#else
 if(golden)return 2;
#endif
 if(verify){unsigned count=0;for(auto e:entries)for(uint64_t n:{uint64_t(8),uint64_t(24)})for(uint64_t p:{uint64_t(1),uint64_t(3)}){Buffer ba(16*n),bw(16*n),bc(16*n);auto a=(int8_t*)ba.data(),w=(int8_t*)bw.data();auto c=(uint32_t*)bc.data();init(a,w,c,n);
#ifndef HOST_ORACLE
 e.fn(a,w,c,n,p);
#else
 for(uint64_t pass=0;pass<p;pass++)for(size_t i=0;i<4*n;i++){int32_t dot=0;for(unsigned q=0;q<4;q++)dot+=int32_t(a[4*i+q])*int32_t(w[4*i+q]);c[i]+=uint32_t(int64_t(dot)*e.d);}
#endif
 if(!checkc(c,n,e.d,p)||!checkaw(a,w,n)||!ba.guard()||!bw.guard()||!bc.guard())return 3;count++;}std::printf("SDOT_VERIFY_PASS cases=%u\n",count);return 0;}
 const int experiment=scan?argv[2][0]-'0':-1;
#ifndef HOST_ORACLE
 for(unsigned i=0;i<3;i++){std::printf("SDOT_EMPTY experiment=%d id=%u\n",experiment,i);std::fflush(stdout);asm volatile("dsb sy":::"memory");m5_reset_stats(0,0);asm volatile("dsb sy":::"memory");m5_dump_stats(0,0);}
#endif
 unsigned bi=0;
 for(const auto &b:blocks){auto e=entries[b.entry];Buffer ba(16*b.n),bw(16*b.n),bc(16*b.n);auto a=(int8_t*)ba.data(),w=(int8_t*)bw.data();auto c=(uint32_t*)bc.data();init(a,w,c,b.n);uint64_t cumulative=0;
  for(unsigned j=0;j<b.count;j++){uint64_t p=b.passes[j];cumulative+=p+1;
#ifdef HOST_ORACLE
   for(size_t i=0;i<4*b.n;i++)c[i]=value(i,e.d,cumulative);
#else
   std::printf("SDOT_ROI experiment=%d id=%u block=%u tiles=%" PRIu64 " d=%u u=%u passes=%" PRIu64 "\n",experiment,b.first_id+j,bi,b.n,e.d,e.u,p);std::fflush(stdout);
   e.fn(a,w,c,b.n,1);asm volatile("dsb sy":::"memory");m5_reset_stats(0,0);
   e.fn(a,w,c,b.n,p);asm volatile("dsb sy":::"memory");m5_dump_stats(0,0);
   if(!checkc(c,b.n,e.d,cumulative))return 4;
   std::printf("SDOT_RESULT experiment=%d id=%u bytes=%" PRIu64 " ops=%" PRIu64 " cumulative=%" PRIu64 " valid=1\n",experiment,b.first_id+j,64*b.n*p,32*b.n*p*e.d,cumulative);std::fflush(stdout);
#endif
  }
  if(!checkaw(a,w,b.n)||!ba.guard()||!bw.guard()||!bc.guard())return 5;
  std::printf("SDOT_BLOCK experiment=%d block=%u hash=%016" PRIx64 " valid=1\n",experiment,bi,hashc(c,b.n));std::fflush(stdout);bi++;
 }
 std::printf("SDOT_COMPLETE experiment=%d blocks=%u cases=%u\n",experiment,bi,condition_count);return 0;
}
