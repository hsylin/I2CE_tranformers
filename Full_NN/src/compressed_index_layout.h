#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* CI01 stores a shared two-bit index stream for two learners. Each panel
 * contains 32 output columns and 128 input features (1024 bytes). A word
 * contains eight indices for each of two adjacent output columns. Both
 * learners decode that word through their own four-entry codebook. */
static inline size_t ci_payload_bytes(size_t columns, size_t features) {
    if (!columns || !features || columns > UINT16_MAX || features > UINT16_MAX) {
        return 0;
    }
    const size_t column_panels = (columns + 31u) / 32u;
    const size_t feature_panels = (features + 127u) / 128u;
    const size_t bytes = column_panels * feature_panels * 1024u;
    return bytes <= UINT32_MAX ? bytes : 0;
}

/* The K16 group holds two K8 words. Within each word, the first eight
 * indices belong to the even column and the next eight to its partner. */
static inline size_t ci_pair_k8_offset(size_t feature_panels, size_t column,
                                       size_t feature) {
    const size_t panel = (column / 32u) * feature_panels + feature / 128u;
    const size_t column_pair = (column % 32u) / 2u;
    const size_t feature_group = (feature % 128u) / 16u;
    const size_t word_in_group = (feature % 16u) / 8u;
    return panel * 1024u + column_pair * 64u + feature_group * 8u + word_in_group * 4u;
}

/* memcpy permits an unaligned source without violating aliasing rules.
 * The SIMD decoder uses this representation on little-endian AArch64. */
static inline uint32_t ci_load_word(const uint8_t *source) {
    uint32_t word;
    memcpy(&word, source, sizeof(word));
    return word;
}
