/* Private implementation, included once by gemm_SVE.c.
 * Fixed CB4/shared-I2/SVE128 cache experiment. External X/Y remain interleaved;
 * only the arena's X uses learner planes. No multi-output register blocking.
 */
#include <stdlib.h>
#include <string.h>

#ifdef I2CE_TEST_CB4_CACHE
uint64_t i2ce_cb4_cache_calls = 0, i2ce_cb4_cache_outputs = 0;
uint64_t i2ce_cb4_cache_chunks = 0, i2ce_cb4_cache_decoded = 0;
int i2ce_cb4_cache_fail_alloc = 0;
extern int i2ce_cb4_cache_test_allocate(void **, size_t, size_t);
#define CACHE_COUNT(name, amount) (i2ce_cb4_cache_##name += (amount))
#else
#define CACHE_COUNT(name, amount) ((void)0)
#endif

#if CB_SIZE == 4 && N_LEARNERS == 2 && TILE_L1_SIZE == 1 && TILE_L2_SIZE == 1
enum { CACHE_S1 = 16, CACHE_O1 = 32, CACHE_K1 = 128,
       CACHE_X1_STRIDE = 144,
       CACHE_S2 = 128, CACHE_O2 = 128, CACHE_K2 = 512, CACHE_X2_STRIDE = 528 };

/* Every field has a size divisible by 64; one aligned allocation, no malloc
 * inside the tile loops. W1 contains two learner-specific decoded planes. */
typedef struct {
    int8_t x1[2][CACHE_S1][CACHE_X1_STRIDE];
    uint32_t c1[2][CACHE_S1][CACHE_O1];
    int8_t w1[2][CACHE_O1][CACHE_X1_STRIDE];
    int8_t x2[2][CACHE_S2][CACHE_X2_STRIDE];
    uint32_t c2[2][CACHE_S2][CACHE_O2];
    int8_t w2[2][CACHE_O2][CACHE_X2_STRIDE];
} cb4_cache_arena;

static size_t cache_min(size_t a, size_t b) { return a < b ? a : b; }

/* Address-range validation precedes both dereference and output writes.
 * The caller owns buffers with the documented extents; this cannot discover
 * an incorrectly sized allocation behind an otherwise valid pointer. */
static int cache_range(const void *p, size_t bytes) {
    return !bytes || (p && (uintptr_t)p <= UINTPTR_MAX - (bytes - 1u));
}
static int cache_overlap(const void *a, size_t an, const void *b, size_t bn) {
    if (!an || !bn) return 0;
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}

static void cache_pack_x(int8_t *dst, size_t rows, size_t stride,
                         size_t plane, const int8_t *input, size_t K,
                         size_t sb, size_t kb, size_t ks) {
    for (size_t s = 0; s < rows; ++s) {
        const int8_t *src = input + ((sb + s) * K + kb) * 2u;
        for (size_t k = 0; k < ks; ++k) {
            dst[s * stride + k] = src[2u * k];
            dst[plane + s * stride + k] = src[2u * k + 1u];
        }
        /* SDOT has no predicate. Zero the remaining activation bytes of the
         * final vector; extra row padding is initialized for bounds tests. */
        memset(dst + s * stride + ks, 0, stride - ks);
        memset(dst + plane + s * stride + ks, 0, stride - ks);
    }
}

/* Decode a shared index once into both learner weight planes. No I1 copy. */
static void cache_pack_w(int8_t *dst, size_t stride, size_t plane,
                         const uint32_t *indices, size_t nw, size_t ob,
                         size_t os, size_t kb, size_t ks,
                         svint8_t cb0, svint8_t cb1,
                         svuint8_t byte_sel, svuint8_t shifts) {
    const svbool_t pg = svptrue_b8();
    for (size_t o = 0; o < os; ++o) {
        for (size_t k = 0; k < ks; k += 16u) {
            const svuint8_t packed = svreinterpret_u8_u32(
                svdup_n_u32(indices[(ob + o) * nw + (kb + k) / 16u]));
            const svuint8_t ix = svand_n_u8_x(pg,
                svlsr_u8_x(pg, svtbl_u8(packed, byte_sel), shifts), 3u);
            const svbool_t tail = svwhilelt_b8((uint64_t)k, (uint64_t)ks);
            svst1_s8(tail, dst + o * stride + k, svtbl_s8(cb0, ix));
            svst1_s8(tail, dst + plane + o * stride + k, svtbl_s8(cb1, ix));
            CACHE_COUNT(decoded, cache_min(16u, ks - k));
        }
        memset(dst + o * stride + ks, 0, stride - ks);
        memset(dst + plane + o * stride + ks, 0, stride - ks);
    }
}

/* Copy a bounded subpanel from two packed learner planes into L1. */
static void cache_copy_panel(int8_t *dst, size_t dp, size_t ds,
                             const int8_t *src, size_t sp, size_t stride,
                             size_t row, size_t k, size_t rows, size_t ks) {
    for (size_t l = 0; l < 2u; ++l)
        for (size_t i = 0; i < rows; ++i) {
            memcpy(dst + l * dp + i * ds, src + l * sp + (row + i) * stride + k, ks);
            memset(dst + l * dp + i * ds + ks, 0, ds - ks);
        }
}

/* C2 holds bias plus all earlier K2 panels. C1 is loaded before the K1
 * sequence and written back after it, never reinitialized with bias. */
static void cache_transfer_c(cb4_cache_arena *a, size_t sb, size_t ob,
                             size_t ss, size_t os, int to_inner) {
    for (size_t l = 0; l < 2u; ++l)
        for (size_t s = 0; s < ss; ++s)
            for (size_t o = 0; o < os; ++o) {
                if (to_inner) a->c1[l][s][o] = a->c2[l][sb + s][ob + o];
                else a->c2[l][sb + s][ob + o] = a->c1[l][s][o];
            }
}

static void cache_init_c(uint32_t *c, size_t plane, size_t stride,
                         size_t rows, size_t cols, const int32_t *bias, size_t ob) {
    for (size_t s = 0; s < rows; ++s)
        for (size_t o = 0; o < cols; ++o) {
            c[s * stride + o] = bias ? (uint32_t)bias[2u * (ob + o)] : 0u;
            c[plane + s * stride + o] = bias ? (uint32_t)bias[2u * (ob + o) + 1u] : 0u;
        }
}

static void cache_write_y(const uint32_t *c, size_t plane, size_t stride,
                          size_t rows, size_t cols, int8_t *output,
                          size_t N, size_t sb, size_t ob) {
    for (size_t s = 0; s < rows; ++s)
        for (size_t o = 0; o < cols; ++o) {
            output[((sb + s) * N + ob + o) * 2u] = (int8_t)(uint8_t)c[s * stride + o];
            output[((sb + s) * N + ob + o) * 2u + 1u] = (int8_t)(uint8_t)c[plane + s * stride + o];
        }
}

/* One (s,o), two independent learners. At most 128 signed-byte products
 * per learner: |partial| <= 128*128*128 = 2,097,152, safely int32.
 * Only the cross-K-tile accumulation uses modulo-2^32 arithmetic. */
static void cache_compute_inner(cb4_cache_arena *a, size_t ss, size_t os, size_t ks) {
    const svbool_t pg = svptrue_b8();
    for (size_t s = 0; s < ss; ++s)
        for (size_t o = 0; o < os; ++o) {
            svint32_t v0 = svdup_s32(0), v1 = svdup_s32(0);
            for (size_t k = 0; k < ks; k += 16u) {
                v0 = svdot_s32(v0, svld1_s8(pg, a->x1[0][s] + k),
                               svld1_s8(pg, a->w1[0][o] + k));
                v1 = svdot_s32(v1, svld1_s8(pg, a->x1[1][s] + k),
                               svld1_s8(pg, a->w1[1][o] + k));
                CACHE_COUNT(chunks, 1);
            }
            a->c1[0][s][o] += (uint32_t)svaddv_s32(svptrue_b32(), v0);
            a->c1[1][s][o] += (uint32_t)svaddv_s32(svptrue_b32(), v1);
            CACHE_COUNT(outputs, 1);
        }
}
#endif

int sve_gemm_cb4_cache_i8(const uint32_t *indices, size_t nw,
                        size_t M, size_t N, size_t K, const int8_t *input,
                        const int32_t *cb, const int32_t *bias, int8_t *output,
                        uint8_t bits) {
#if CB_SIZE == 4 && N_LEARNERS == 2 && TILE_L1_SIZE == 1 && TILE_L2_SIZE == 1
    if (bits != 2u || svcntb() != 16u) return 0;
    /* Retain the supported descriptor domain without narrowing first. These
     * bounds also make all products below fit size_t on AArch64. */
    if (M > UINT16_MAX || N > UINT16_MAX || K > UINT16_MAX || nw > UINT16_MAX) return 0;
    if (!M || !N) return 1;
    const size_t xb = M * K * 2u, yb = M * N * 2u, ib = N * nw * 4u, bb = N * 8u;
    if (xb > UINT32_MAX || yb > UINT32_MAX || N * nw > UINT32_MAX ||
        nw < (K + 15u) / 16u || !cache_range(output, yb) ||
        !cache_range(input, xb) || !cache_range(indices, K ? ib : 0u) ||
        !cache_range(cb, K ? 32u : 0u) || !cache_range(bias, bias ? bb : 0u)) return 0;
    if (cache_overlap(output,yb,input,xb) || cache_overlap(output,yb,indices,K ? ib : 0u) ||
        cache_overlap(output,yb,cb,K ? 32u : 0u) || cache_overlap(output,yb,bias,bias ? bb : 0u)) return 0;
    if (K && (((uintptr_t)indices & 3u) || ((uintptr_t)cb & 3u))) return 0;
    if (bias && ((uintptr_t)bias & 3u)) return 0;
    if (K && !gemm_sdot_decode_usable(cb, 4u, 2u, 2u)) return 0;
    if (!K) {
        for (size_t s = 0; s < M; ++s)
            for (size_t o = 0; o < N * 2u; ++o)
                output[s * N * 2u + o] = bias ? (int8_t)(uint8_t)bias[o] : 0;
        return 1;
    }
    void *storage = NULL;
#ifdef I2CE_TEST_CB4_CACHE
    const int alloc_status = i2ce_cb4_cache_test_allocate(&storage, 64u, sizeof(cb4_cache_arena));
#else
    const int alloc_status = posix_memalign(&storage, 64u, sizeof(cb4_cache_arena));
#endif
    if (alloc_status != 0) return 0;
    cb4_cache_arena *a = (cb4_cache_arena *)storage;
    int8_t scratch0[256] = {0}, scratch1[256] = {0};
    const svint8_t cb0 = gemm_sdot_cb_table(cb,4u,2u,0u,scratch0);
    const svint8_t cb1 = gemm_sdot_cb_table(cb,4u,2u,1u,scratch1);
    svuint8_t byte_sel, shifts; gemm_sdot_lane_patterns(2u,&byte_sel,&shifts);
    CACHE_COUNT(calls, 1);
    for (size_t sb = 0; sb < M; sb += CACHE_S2) {
        const size_t ss = cache_min(CACHE_S2, M - sb);
        for (size_t ob = 0; ob < N; ob += CACHE_O2) {
            const size_t os = cache_min(CACHE_O2, N - ob);
            cache_init_c(&a->c2[0][0][0], CACHE_S2 * CACHE_O2, CACHE_O2, ss, os, bias, ob);
            for (size_t kb = 0; kb < K; kb += CACHE_K2) {
                const size_t ks = cache_min(CACHE_K2, K - kb);
                cache_pack_x(&a->x2[0][0][0],ss,CACHE_X2_STRIDE,CACHE_S2 * CACHE_X2_STRIDE,input,K,sb,kb,ks);
                cache_pack_w(&a->w2[0][0][0],CACHE_X2_STRIDE,CACHE_O2 * CACHE_X2_STRIDE,
                             indices,nw,ob,os,kb,ks,cb0,cb1,byte_sel,shifts);
                for (size_t s1 = 0; s1 < ss; s1 += CACHE_S1) {
                    const size_t ms = cache_min(CACHE_S1, ss - s1);
                    for (size_t o1 = 0; o1 < os; o1 += CACHE_O1) {
                        const size_t ns = cache_min(CACHE_O1, os - o1);
                        cache_transfer_c(a,s1,o1,ms,ns,1);
                        for (size_t k1 = 0; k1 < ks; k1 += CACHE_K1) {
                            const size_t kk = cache_min(CACHE_K1, ks - k1);
                            cache_copy_panel(&a->x1[0][0][0],CACHE_S1 * CACHE_X1_STRIDE,CACHE_X1_STRIDE,
                                &a->x2[0][0][0],CACHE_S2 * CACHE_X2_STRIDE,CACHE_X2_STRIDE,s1,k1,ms,kk);
                            cache_copy_panel(&a->w1[0][0][0],CACHE_O1 * CACHE_X1_STRIDE,CACHE_X1_STRIDE,
                                &a->w2[0][0][0],CACHE_O2 * CACHE_X2_STRIDE,CACHE_X2_STRIDE,o1,k1,ns,kk);
                            cache_compute_inner(a,ms,ns,kk);
                        }
                        cache_transfer_c(a,s1,o1,ms,ns,0);
                    }
                }
            }
            cache_write_y(&a->c2[0][0][0], CACHE_S2 * CACHE_O2, CACHE_O2, ss, os, output, N, sb, ob);
        }
    }
    free(storage);
    return 1;
#else
    (void)indices; (void)nw; (void)M; (void)N; (void)K; (void)input;
    (void)cb; (void)bias; (void)output; (void)bits;
    return 0;
#endif
}
#undef CACHE_COUNT
