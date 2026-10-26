#ifndef _CODEBOOKS_DEF_H_
#define _CODEBOOKS_DEF_H_

#define N_LEARNERS      2

#define N_SVE_LANES     4
#define N_SVE_HALF      8
#define N_SVE_BYTE      16

#define CB_SIZE         4
#define TILE_L1_SIZE    1
#define TILE_L2_SIZE    1
#define BITS_PER_CB		2
#define IDX_MASK		a0b11

// Number of indexes packed per each 32-bits word
#define IDXS_PER_WORD ((32) / (BITS_PER_CB))
#define IDXS_PER_WORD_16 ((16) / (BITS_PER_CB))


// Number of indexes packed per each vector register
#define IDX_PER_VECT	((N_SVE_LANES) * (IDXS_PER_WORD))

#define N_SVE_REG_CB_1	1
#define N_SVE_REG_CB_F16_1	1


#define USE_F32   1



#define SAME_SEQ    1





#endif
