#pragma once

// Shared build-domain predicate. Expand after the generated codebook definitions
// are visible. Application mode and runtime SVE length are checked separately.
#define I2CE_CODEBOOK_TILE_LAYOUT_SUPPORTED                                            \
    ((CB_SIZE == 4 || CB_SIZE == 8 || CB_SIZE == 16) &&                                \
     (N_LEARNERS == 1 || N_LEARNERS == 2 || N_LEARNERS == 4) && TILE_L1_SIZE == 1 &&   \
     TILE_L2_SIZE == 1)
