#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* Shared packed indices for one, two or four learners: 32 columns x K128 per panel.
 * Each pair/K8 record has 16 consecutive indices, packed little-endian in
 * exactly 2*bits bytes. CI01 is the original two-bit format; CI02 supports
 * three/four bits without padding a CB8 record to a CB16 record. */
static inline size_t ci_payload_bytes_bits(size_t columns, size_t features,
                                           unsigned bits) {
    if (bits < 2u || bits > 4u || !columns || !features || columns > UINT16_MAX ||
        features > UINT16_MAX) {
        return 0;
    }
    const size_t column_panels = (columns + 31u) / 32u;
    const size_t feature_panels = (features + 127u) / 128u;
    const size_t bytes = column_panels * feature_panels * (512u * bits);
    return bytes <= UINT32_MAX ? bytes : 0;
}
static inline size_t ci_payload_bytes(size_t columns, size_t features) {
    return ci_payload_bytes_bits(columns, features, 2u);
}

/* The K16 group holds two K8 words. Within each word, the first eight
 * indices belong to the even column and the next eight to its partner. */
static inline size_t ci_pair_k8_offset_bits(size_t feature_panels, size_t column,
                                            size_t feature, unsigned bits) {
    const size_t panel = (column / 32u) * feature_panels + feature / 128u;
    const size_t column_pair = (column % 32u) / 2u;
    const size_t feature_group = (feature % 128u) / 16u;
    const size_t word_in_group = (feature % 16u) / 8u;
    return (panel * 256u + column_pair * 16u + feature_group * 2u + word_in_group) *
           (2u * bits);
}
static inline size_t ci_pair_k8_offset(size_t feature_panels, size_t column,
                                       size_t feature) {
    return ci_pair_k8_offset_bits(feature_panels, column, feature, 2u);
}

/* memcpy permits an unaligned source without violating aliasing rules.
 * The SIMD decoder uses this representation on little-endian AArch64. */
static inline uint32_t ci_load_word(const uint8_t *source) {
    uint32_t word;
    memcpy(&word, source, sizeof(word));
    return word;
}
