#ifndef GEMM_WRAP_H
#define GEMM_WRAP_H

#include <stdint.h>
#include <string.h>

/* C32 outputs retain the low 32 bits, including bias and partial accumulation.
 * Unsigned addition defines wrapping; memcpy preserves the result bits without
 * relying on an out-of-range unsigned-to-signed conversion. */
static inline int32_t gemm_add_i32_wrap(int32_t a, int32_t b) {
    const uint32_t bits = (uint32_t)a + (uint32_t)b;
    int32_t result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

#endif
