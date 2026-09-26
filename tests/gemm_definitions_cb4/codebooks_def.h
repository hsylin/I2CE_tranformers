#ifndef _CODEBOOKS_DEF_H_
#define _CODEBOOKS_DEF_H_

/*
 * Test-only generated-header stand-in for the CB4 / 2-learner / SVE128 target
 * configuration. The field layout mirrors what
 * Full_NN/generators/codebooks_defs_generator.py emits, so the kernels see the
 * same macros they would in a real CB4 build:
 *
 *   idxs_bits = ceil(log2(4))          = 2
 *   n_regs_cb = ceil(CB_SIZE / lanes)  = ceil(4/4) = 1  -> N_SVE_REG_CB_1
 *
 * This exists because the committed Full_NN/gemm_definitions/codebooks_def.h is
 * generated for CB8 and selects N_SVE_REG_CB_2. Simply passing bits_per_cb=2 to
 * a CB8-configured build would exercise the wrong codebook-register path, so
 * CB4 coverage needs its own header rather than a changed constant.
 */

#define N_LEARNERS      2

#define N_SVE_LANES     4
#define N_SVE_HALF      8
#define N_SVE_BYTE      16

#define CB_SIZE         4
#define TILE_L1_SIZE    1
#define TILE_L2_SIZE    1
#define BITS_PER_CB     2
#define IDX_MASK        0b11

// Number of indexes packed per each 32-bits word
#define IDXS_PER_WORD ((32) / (BITS_PER_CB))
#define IDXS_PER_WORD_16 ((16) / (BITS_PER_CB))

// Number of indexes packed per each vector register
#define IDX_PER_VECT    ((N_SVE_LANES) * (IDXS_PER_WORD))

#define N_SVE_REG_CB_1  1
#define N_SVE_REG_CB_F16_1  1

#define USE_F32   1

#define SAME_SEQ    1

#endif
