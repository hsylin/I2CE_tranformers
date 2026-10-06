#ifndef I2CE_GEMM_CB4_CACHE_CONFIG_H
#define I2CE_GEMM_CB4_CACHE_CONFIG_H

#include <stdint.h>

/* Compile-time cache geometry; independent of generated TILE_L1/L2_SIZE.
 * Defaults reproduce E09 packing-v2. Keep 16 bytes of row padding, both
 * staging levels, and one output's two learner SDOT accumulators. */
#ifndef I2CE_CACHE_S1
#define I2CE_CACHE_S1 16
#endif
#ifndef I2CE_CACHE_O1
#define I2CE_CACHE_O1 32
#endif
#ifndef I2CE_CACHE_K1
#define I2CE_CACHE_K1 128
#endif
#ifndef I2CE_CACHE_S2
#define I2CE_CACHE_S2 128
#endif
#ifndef I2CE_CACHE_O2
#define I2CE_CACHE_O2 128
#endif
#ifndef I2CE_CACHE_K2
#define I2CE_CACHE_K2 512
#endif

#if I2CE_CACHE_S1 <= 0 || I2CE_CACHE_O1 <= 0 || I2CE_CACHE_K1 <= 0 || \
    I2CE_CACHE_S2 <= 0 || I2CE_CACHE_O2 <= 0 || I2CE_CACHE_K2 <= 0
#error "CB4 cache tiles must be positive"
#endif
#if I2CE_CACHE_S2 > 65535 || I2CE_CACHE_O2 > 65535 || I2CE_CACHE_K2 > 65520
#error "CB4 cache tiles exceed the supported descriptor/partial-sum domain"
#endif
#if I2CE_CACHE_S1 > I2CE_CACHE_S2 || I2CE_CACHE_O1 > I2CE_CACHE_O2 || \
    I2CE_CACHE_K1 > I2CE_CACHE_K2
#error "CB4 inner tiles must not exceed outer tiles"
#endif
#if I2CE_CACHE_S1 % 4 || I2CE_CACHE_O1 % 4 || \
    I2CE_CACHE_S2 % 4 || I2CE_CACHE_O2 % 4
#error "CB4 S/O tiles must be multiples of 4 (64-byte arena field alignment)"
#endif
#if I2CE_CACHE_K1 % 16 || I2CE_CACHE_K2 % 16
#error "CB4 K tiles must be multiples of 16 (packed words and SVE128 tails)"
#endif

#define I2CE_CACHE_X1_STRIDE (I2CE_CACHE_K1 + 16)
#define I2CE_CACHE_X2_STRIDE (I2CE_CACHE_K2 + 16)
#define I2CE_CACHE_ARENA_BYTES ( \
    2ULL * (I2CE_CACHE_S1 + I2CE_CACHE_O1) * I2CE_CACHE_X1_STRIDE + \
    8ULL * I2CE_CACHE_S1 * I2CE_CACHE_O1 + \
    2ULL * (I2CE_CACHE_S2 + I2CE_CACHE_O2) * I2CE_CACHE_X2_STRIDE + \
    8ULL * I2CE_CACHE_S2 * I2CE_CACHE_O2)
/* A conservative implementation limit, not a claim that this fits a cache.
 * Together with the dimension bounds it keeps every plane product and arena
 * offset representable in int and size_t on the supported AArch64 target. */
#if I2CE_CACHE_ARENA_BYTES > INT32_MAX
#error "CB4 cache arena exceeds the supported 2-GiB addressing limit"
#endif

#define I2CE_CACHE_DEFAULT_GEOMETRY ( \
    I2CE_CACHE_S1 == 16 && I2CE_CACHE_O1 == 32 && I2CE_CACHE_K1 == 128 && \
    I2CE_CACHE_S2 == 128 && I2CE_CACHE_O2 == 128 && I2CE_CACHE_K2 == 512)

#endif
