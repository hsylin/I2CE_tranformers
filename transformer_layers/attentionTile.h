#pragma once
#include "interleavedPipeline.h"
// The query-row stream uses the same final-panel two-level SDOT backend.
// Its effective outer M is 16: one M1 consumer, no retained packed K/V owner.
static inline void attentionMatmulTiled(const int8_t *left, const int8_t *right,
                                        size_t rows, size_t columns, size_t features,
                                        int8_t *output) {
    attentionMatmulSdotBlocked(left, right, rows, columns, features, output);
}
