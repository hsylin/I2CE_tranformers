/*
 * SVE GEMM kernels for codebook-compressed TiC-SAT weights.
 *
 * This module is used by the generated TiC-SAT neural-network kernels to
 * compute rows of matrix multiplication on Arm SVE targets. Most functions in
 * this file multiply one input row or sequence tile by one compressed weight
 * row, where the weights are not stored directly. Instead, each weight is
 * represented by a small packed integer index. The index selects an entry from
 * an int32 or fp32 codebook, and the selected codebook value is used as the
 * real matrix weight.
 *
 * The compact kernels perform the following high-level steps:
 * 1. Read packed 32-bit words containing several codebook indexes.
 * 2. Use bit shifts and masks to unpack the indexes into SVE lanes.
 * 3. Look up the real weight values from the codebook.
 * 4. Multiply those weights by the corresponding input activations.
 * 5. Horizontally reduce the SVE accumulator into one output scalar.
 * 6. Optionally add bias and either write or accumulate into the output matrix.
 *
 * The interleaved 2D and 4D variants compute multiple related dot products at
 * once. Their inputs, codebooks, and outputs are laid out as repeating groups
 * of two or four values, which lets the generated model evaluate multiple
 * learners or dimensions in the same pass through the packed indexes.
 *
 * The dense helpers at the end of the file are reference-style SVE kernels for
 * already-expanded int32 matrices. They use the same 2- and 4-learner interleaved layout,
 * but do not decode codebook indexes.
 */

#include <stdint.h>
#include <stddef.h>

#include <arm_sve.h>

#include <codebooks_def.h>
#include <gemm_SVE.h>

/*
 * Navigation declarations for the public GEMM SVE entry points.
 *
 * Interleaving factors used by the function names:
 *   2D -> two interleaved learner/output streams per logical element
 *   4D -> four interleaved learner/output streams per logical element
 *
 * Sequence naming:
 *   same_seq -> all interleaved streams share one packed index row
 *   diff_seq -> each interleaved stream has its own packed index row
 */

/* Compact single-output row kernels. */
void sve_gemm_row_compact_int8(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const int8_t *in_mat, uint32_t seq_tile, uint32_t ld_in, const int32_t *codebook_i32, int32_t *out_mat, uint32_t out_col, uint32_t ld_out, int32_t bias_val, int add_bias, int accumulate, uint8_t bits_per_cb);
void sve_gemm_row_compact_fp32(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const float *in_mat, uint32_t seq_tile, uint32_t ld_in, const float *codebook, float *out_mat, uint32_t out_col, uint32_t ld_out, float bias_val, int add_bias, int accumulate, uint8_t bits_per_cb);

/* FP32 compact row kernels for interleaved outputs. */
void sve_gemm_row_compact_fp32_interleaved_4Learners_diff_seq(const uint32_t *packed_rows_interleaved, uint32_t n_words_row, uint32_t k_elems, const float *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const float *codebook_interleaved, uint32_t codebook_size, float *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const float *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);
void sve_gemm_row_compact_fp32_interleaved_2Learners_same_seq(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const float *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const float *codebook_interleaved, uint32_t codebook_size, float *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const float *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);
void sve_gemm_row_compact_fp32_interleaved_4Learners_same_seq(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const float *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const float *codebook_interleaved, uint32_t codebook_size, float *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const float *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);

/* Int8/int32 compact row kernels for interleaved outputs. */
void sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq(const uint32_t *packed_rows_interleaved, uint32_t n_words_row, uint32_t k_elems, const int8_t *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const int32_t *codebook_i32_interleaved, uint32_t codebook_size, int32_t *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const int32_t *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);
void sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const int8_t *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const int32_t *codebook_i32_interleaved, uint32_t codebook_size, int32_t *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const int32_t *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);
void sve_gemm_row_compact_int8_interleaved_4Learners_same_seq(const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems, const int8_t *in_mat_interleaved, uint32_t seq_tile, uint32_t ld_in_interleaved, const int32_t *codebook_i32_interleaved, uint32_t codebook_size, int32_t *out_mat_interleaved, uint32_t out_col, uint32_t ld_out_interleaved, const int32_t *bias_interleaved, int add_bias, int accumulate, uint8_t bits_per_cb);

/* Dense interleaved int8/int32 GEMM helpers. */
void sve_gemm_dense_int8_interleaved_4Learners(const int8_t *lhs_interleaved, const int8_t *rhs_by_col_interleaved, uint32_t lhs_rows, uint32_t rhs_cols, uint32_t k_elems, int32_t *out_interleaved);
void sve_gemm_dense_int8_interleaved_2Learners(const int8_t *lhs_interleaved, const int8_t *rhs_by_col_interleaved, uint32_t lhs_rows, uint32_t rhs_cols, uint32_t k_elems, int32_t *out_interleaved);

/*
 * Build a bit mask for one packed codebook index.
 *
 * Each 32-bit packed word stores floor(32 / bits_per_cb) indexes. After an
 * index is shifted down to bit zero, this mask removes the neighboring packed
 * indexes that belong to the same word.
 */
static uint32_t gemm_sve_idx_mask(uint8_t bits_per_cb) {
    return (bits_per_cb >= 32u) ? UINT32_MAX : ((1u << bits_per_cb) - 1u);
}

/*
 * Load L interleaved int8 activations per K position and sign-extend them to
 * int32, without materialising an int32 copy of the matrix first.
 *
 * The interleaved layout puts the L learner values for one K position in L
 * adjacent bytes, so for L=4 those bytes are exactly one 32-bit word and for
 * L=2 exactly one 16-bit halfword. One contiguous load therefore brings in a
 * whole K position per lane, and each learner is recovered by shifting its byte
 * into the sign position and arithmetic-shifting it back down. This replaces
 * svld4_s32/svld2_s32 over a widened buffer: same lanes, same values, a quarter
 * of the memory traffic.
 *
 * Byte j of memory is bits [8j+7:8j] of the loaded element on little-endian
 * AArch64, which is what makes the shift amounts below the learner order.
 *
 * Alignment: the 32-bit form needs a 4-byte aligned base and the 16-bit form a
 * 2-byte aligned base. Row bases are ld_in_interleaved apart, and that stride
 * is a multiple of L, so checking the matrix base once is enough; the callers
 * in gemm_exec.c do exactly that and fall back to the scalar path otherwise.
 */
static svint32x4_t gemm_load_in4_s8(svbool_t pg, const int8_t *row_in_i8) {
    const svint32_t packed =
        svreinterpret_s32_u32(svld1_u32(pg, (const uint32_t *)(const void *)row_in_i8));
    return svcreate4_s32(
        svasr_n_s32_x(pg, svlsl_n_s32_x(pg, packed, 24), 24),
        svasr_n_s32_x(pg, svlsl_n_s32_x(pg, packed, 16), 24),
        svasr_n_s32_x(pg, svlsl_n_s32_x(pg, packed, 8), 24),
        svasr_n_s32_x(pg, packed, 24));
}

/*
 * Can this call use the svdot decode path?
 *
 * svdot_s32 consumes four int8 products per 32-bit lane, so the decoded
 * weights must be produced as bytes, four consecutive K positions to a lane.
 * That is only possible when a codebook index never straddles a byte
 * boundary, which means 8 % bits_per_cb == 0. Two widths matter here:
 *
 *   bits_per_cb = 2  four indexes per byte, one packed word per 16 K
 *   bits_per_cb = 4  two indexes per byte, two packed words per 16 K
 *
 * bits_per_cb = 3 straddles whenever 3i mod 8 > 5. Reaching this path with an
 * 8-entry codebook therefore needs the index stream padded to 4 bits when it
 * is generated, which costs a third more index memory; the generator does
 * that only when asked, and until then CB8 builds keep the svmla path.
 *
 * svtbl_s8 selects within a single vector, which is svcntb() bytes; the
 * 16-byte bound keeps one table valid at every vector length from 128 bits
 * up. The codebook values are int8_t in the generated header but reach the
 * kernel widened to int32, so the range is re-checked rather than assumed.
 */
static int gemm_sdot_decode_usable(const int32_t *codebook_i32_interleaved,
                                   uint32_t codebook_size,
                                   uint32_t n_learners,
                                   uint8_t bits_per_cb) {
    if ((bits_per_cb != 2u) && (bits_per_cb != 4u)) return 0;
    if ((codebook_size == 0u) || (codebook_size > 16u)) return 0;
    if (codebook_size > (1u << bits_per_cb)) return 0;
    if (svcntb() < 16u) return 0;
    if (codebook_i32_interleaved == NULL) return 0;
    for (uint32_t i = 0; i < codebook_size * n_learners; i++) {
        if ((codebook_i32_interleaved[i] < -128) ||
            (codebook_i32_interleaved[i] > 127)) {
            return 0;
        }
    }
    return 1;
}

/*
 * Narrow one learner's slice of the interleaved int32 codebook into an int8
 * svtbl_s8 table: byte i holds the weight for index i. Callers must have
 * cleared scratch and checked gemm_sdot_decode_usable() first.
 */
static svint8_t gemm_sdot_cb_table(const int32_t *codebook_i32_interleaved,
                                   uint32_t codebook_size,
                                   uint32_t n_learners,
                                   uint32_t learner,
                                   int8_t *scratch) {
    for (uint32_t i = 0; i < codebook_size; i++) {
        scratch[i] = (int8_t)codebook_i32_interleaved[i * n_learners + learner];
    }
    return svld1_s8(svptrue_b8(), scratch);
}

/*
 * Lane patterns for the byte-wise decode, both loop invariant.
 *
 *   byte_sel[i] = i / (8 / bits)   which packed byte feeds output byte i
 *   shifts8[i]  = bits * (i % (8 / bits))   where inside it that index sits
 *
 * For 2-bit indexes that is i/4 and 2*(i%4); for 4-bit, i/2 and 4*(i%2).
 */
static void gemm_sdot_lane_patterns(uint8_t bits_per_cb,
                                    svuint8_t *byte_sel, svuint8_t *shifts8) {
    const svbool_t ptrue8 = svptrue_b8();
    const svuint8_t lanes = svindex_u8(0, 1);
    const uint32_t sel_shift = (bits_per_cb == 2u) ? 2u : 1u;
    const uint8_t mod_mask = (uint8_t)((8u / bits_per_cb) - 1u);
    const uint32_t shift_scale = (bits_per_cb == 2u) ? 1u : 2u;
    *byte_sel = svlsr_n_u8_x(ptrue8, lanes, sel_shift);
    *shifts8 = svlsl_n_u8_x(ptrue8, svand_n_u8_x(ptrue8, lanes, mod_mask), shift_scale);
}

/*
 * Decode the 16 codebook indexes that one svdot accumulator consumes.
 *
 * A 2-bit stream needs 4 packed bytes for 16 indexes and a 4-bit stream needs
 * 8, so the source words are gathered into one 64-bit value and broadcast;
 * byte_sel then reaches at most byte 3 or byte 7 of it.
 *
 * words_per_iter is always passed as a literal so that each width compiles to
 * its own copy: at 2 bits the second word, and the bounds test that guards it,
 * fold away entirely. Leaving it a runtime value cost eight instructions per
 * iteration, a third of the loop.
 */
__attribute__((always_inline))
static inline svuint8_t gemm_sdot_decode_idxs(svbool_t pg8,
                                              const uint32_t *packed_row,
                                              uint32_t cw, uint32_t n_words_row,
                                              const uint32_t words_per_iter,
                                              svuint8_t byte_sel,
                                              svuint8_t shifts8,
                                              uint8_t idx_mask8) {
    svuint8_t src;
    if (words_per_iter > 1u) {
        const uint32_t cw1 = cw + 1u;
        const uint64_t hi = (cw1 < n_words_row) ? (uint64_t)packed_row[cw1] : 0u;
        src = svreinterpret_u8_u64(svdup_n_u64((uint64_t)packed_row[cw] | (hi << 32)));
    } else {
        src = svreinterpret_u8_u32(svdup_n_u32(packed_row[cw]));
    }
    const svuint8_t spread = svtbl_u8(src, byte_sel);
    return svand_n_u8_x(pg8, svlsr_u8_x(pg8, spread, shifts8), idx_mask8);
}

/*
 * One row-block of the svdot path, two interleaved streams.
 *
 * words_per_iter is always passed as a literal so each index width gets its
 * own specialisation; see gemm_sdot_decode_idxs().
 */
__attribute__((always_inline))
static inline void gemm_sdot_rows_2l(
    const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems,
    const int8_t *in_mat_interleaved, uint32_t seq_tile,
    uint32_t ld_in_interleaved, int32_t *out_mat_interleaved,
    uint32_t out_col, uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved, int add_bias, int accumulate,
    svint8_t cb0_b, svint8_t cb1_b, svuint8_t byte_sel, svuint8_t shifts8,
    uint8_t idx_mask8, const uint32_t words_per_iter,
    int8_t *out_i8) {
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        uint32_t input_idx = 0;
        const int8_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        /* Counted loop: one iteration per 16 K positions, also bounded by the
         * packed words available. A single induction variable keeps the test
         * to one compare instead of two. */
        const uint32_t by_k = (k_elems + 15u) / 16u;
        const uint32_t by_w = (n_words_row + words_per_iter - 1u) / words_per_iter;
        const uint32_t n_iters = (by_k < by_w) ? by_k : by_w;

        for (uint32_t it = 0; it < n_iters; it++) {
            const uint32_t cw = it * words_per_iter;
            const uint32_t missing_total = k_elems - input_idx;
            const uint32_t active = (missing_total < 16u) ? missing_total : 16u;
            svbool_t pg8 = svwhilelt_b8((uint64_t)0, (uint64_t)active);

            svuint8_t cb_idxs8 = gemm_sdot_decode_idxs(
                pg8, packed_row, cw, n_words_row, words_per_iter,
                byte_sel, shifts8, idx_mask8);

            svint8_t weights0 = svtbl_s8(cb0_b, cb_idxs8);
            svint8_t weights1 = svtbl_s8(cb1_b, cb_idxs8);

            /*
             * svdot_s32 has no predicated form, so the tail is handled on the
             * activation side: this load is zeroing-predicated, and a zero
             * activation byte contributes nothing whatever weight byte it is
             * paired with.
             */
            svint8x2_t in_vals = svld2_s8(pg8, &row_in[input_idx * 2u]);

            acc_v0 = svdot_s32(acc_v0, svget2_s8(in_vals, 0), weights0);
            acc_v1 = svdot_s32(acc_v1, svget2_s8(in_vals, 1), weights1);

            input_idx += active;
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        if (out_i8 != NULL) {
            /* Full K has completed. Use modulo arithmetic for bias and keep
             * exactly the low byte, matching the consumer's existing cast. */
            const uint32_t b0 = add_bias && bias_interleaved ? (uint32_t)bias_interleaved[0] : 0u;
            const uint32_t b1 = add_bias && bias_interleaved ? (uint32_t)bias_interleaved[1] : 0u;
            int8_t *slot = out_i8 + row * ld_out_interleaved + out_col * 2u;
            slot[0] = (int8_t)(uint8_t)((uint32_t)acc0 + b0);
            slot[1] = (int8_t)(uint8_t)((uint32_t)acc1 + b1);
            continue;
        }
        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
        }
        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
        if (accumulate) { out_slot[0] += acc0; out_slot[1] += acc1; }
        else            { out_slot[0] = acc0;  out_slot[1] = acc1;  }
    }
}

int sve_gemm_cb4_2l_row_i8(
    const uint32_t *indices, uint32_t n_words_row, uint32_t K,
    const int8_t *input, uint32_t M, uint32_t ld_in,
    const int32_t *codebook, const int32_t *bias,
    int8_t *output, uint32_t n, uint32_t ld_out) {
    if (!gemm_sdot_decode_usable(codebook, 4u, 2u, 2u)) return 0;
    int8_t cb_bytes0[256] = {0};
    int8_t cb_bytes1[256] = {0};
    const svint8_t cb0 = gemm_sdot_cb_table(codebook, 4u, 2u, 0u, cb_bytes0);
    const svint8_t cb1 = gemm_sdot_cb_table(codebook, 4u, 2u, 1u, cb_bytes1);
    svuint8_t byte_sel, shifts8;
    gemm_sdot_lane_patterns(2u, &byte_sel, &shifts8);
    gemm_sdot_rows_2l(indices, n_words_row, K, input, M, ld_in, NULL,
        n, ld_out, bias, 1, 0, cb0, cb1, byte_sel, shifts8, 3u, 1u, output);
    return 1;
}

/* Four interleaved streams. All four share one index row in the same_seq
 * layout, so one decoded byte vector drives four table lookups. */
__attribute__((always_inline))
static inline void gemm_sdot_rows_4l(
    const uint32_t *packed_row, uint32_t n_words_row, uint32_t k_elems,
    const int8_t *in_mat_interleaved, uint32_t seq_tile,
    uint32_t ld_in_interleaved, int32_t *out_mat_interleaved,
    uint32_t out_col, uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved, int add_bias, int accumulate,
    svint8_t cb0_b, svint8_t cb1_b, svint8_t cb2_b, svint8_t cb3_b,
    svuint8_t byte_sel, svuint8_t shifts8,
    uint8_t idx_mask8, const uint32_t words_per_iter) {
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        svint32_t acc_v2 = svdup_s32(0);
        svint32_t acc_v3 = svdup_s32(0);
        uint32_t input_idx = 0;
        const int8_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        /* Counted loop: one iteration per 16 K positions, also bounded by the
         * packed words available. A single induction variable keeps the test
         * to one compare instead of two. */
        const uint32_t by_k = (k_elems + 15u) / 16u;
        const uint32_t by_w = (n_words_row + words_per_iter - 1u) / words_per_iter;
        const uint32_t n_iters = (by_k < by_w) ? by_k : by_w;

        for (uint32_t it = 0; it < n_iters; it++) {
            const uint32_t cw = it * words_per_iter;
            const uint32_t missing_total = k_elems - input_idx;
            const uint32_t active = (missing_total < 16u) ? missing_total : 16u;
            svbool_t pg8 = svwhilelt_b8((uint64_t)0, (uint64_t)active);

            svuint8_t cb_idxs8 = gemm_sdot_decode_idxs(
                pg8, packed_row, cw, n_words_row, words_per_iter,
                byte_sel, shifts8, idx_mask8);

            svint8_t weights0 = svtbl_s8(cb0_b, cb_idxs8);
            svint8_t weights1 = svtbl_s8(cb1_b, cb_idxs8);
            svint8_t weights2 = svtbl_s8(cb2_b, cb_idxs8);
            svint8_t weights3 = svtbl_s8(cb3_b, cb_idxs8);

            svint8x4_t in_vals = svld4_s8(pg8, &row_in[input_idx * 4u]);

            acc_v0 = svdot_s32(acc_v0, svget4_s8(in_vals, 0), weights0);
            acc_v1 = svdot_s32(acc_v1, svget4_s8(in_vals, 1), weights1);
            acc_v2 = svdot_s32(acc_v2, svget4_s8(in_vals, 2), weights2);
            acc_v3 = svdot_s32(acc_v3, svget4_s8(in_vals, 3), weights3);

            input_idx += active;
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        int32_t acc2 = svaddv_s32(svptrue_b32(), acc_v2);
        int32_t acc3 = svaddv_s32(svptrue_b32(), acc_v3);
        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0]; acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2]; acc3 += bias_interleaved[3];
        }
        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0; out_slot[1] += acc1;
            out_slot[2] += acc2; out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0; out_slot[1] = acc1;
            out_slot[2] = acc2; out_slot[3] = acc3;
        }
    }
}

static svint32x2_t gemm_load_in2_s8(svbool_t pg, const int8_t *row_in_i8) {
    const svint32_t packed =
        svreinterpret_s32_u32(svld1uh_u32(pg, (const uint16_t *)(const void *)row_in_i8));
    return svcreate2_s32(
        svasr_n_s32_x(pg, svlsl_n_s32_x(pg, packed, 24), 24),
        svasr_n_s32_x(pg, svlsl_n_s32_x(pg, packed, 16), 24));
}

#if defined(N_SVE_REG_CB_2)
/*
 * Look up int32 codebook entries when the logical codebook spans two SVE
 * registers.
 *
 * SVE table lookup operates inside one register at a time, so indexes for the
 * second register are rebased by subtracting the number of lanes. A predicate
 * selects the first-register result for small indexes and the second-register
 * result for indexes that cross the first register boundary.
 */
static svint32_t gemm_extract_weightsx2_s32(svbool_t pg,
                                            svuint32_t idxs,
                                            svint32x2_t cb_regs_x2) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svint32_t weights_0 = svtbl_s32(svget2_s32(cb_regs_x2, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svint32_t weights_1 = svtbl_s32(svget2_s32(cb_regs_x2, 1), idxs_1);

    return svsel_s32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights_0);
}

/*
 * fp32 version of gemm_extract_weightsx2_s32().
 *
 * The index rebasing is identical; only the table element type changes from
 * signed 32-bit integer to single-precision floating point.
 */
static svfloat32_t gemm_extract_weightsx2_f32(svbool_t pg,
                                              svuint32_t idxs,
                                              svfloat32x2_t cb_regs_x2) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svfloat32_t weights_0 = svtbl_f32(svget2_f32(cb_regs_x2, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svfloat32_t weights_1 = svtbl_f32(svget2_f32(cb_regs_x2, 1), idxs_1);

    return svsel_f32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights_0);
}
#endif

#if defined(N_SVE_REG_CB_4)
/*
 * Look up int32 codebook entries when the logical codebook spans four SVE
 * registers.
 *
 * Step by step:
 * 1. Try a table lookup against each physical SVE register.
 * 2. Rebase indexes for registers 1, 2, and 3 by subtracting one, two, or
 *    three register widths.
 * 3. Select the lookup result from the register whose range contains the
 *    original logical codebook index.
 */
static svint32_t gemm_extract_weightsx4_s32(svbool_t pg,
                                            svuint32_t idxs,
                                            svint32x4_t cb_regs_x4) {
    const uint32_t n_lanes = (uint32_t)svcntw(); // Each register stores one contiguous slice of the codebook.
    svint32_t weights_0 = svtbl_s32(svget4_s32(cb_regs_x4, 0), idxs); // Lookup values from codebook slice 0.
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes); // Rebase indexes for codebook slice 1.
    svuint32_t idxs_2 = svsub_n_u32_x(pg, idxs, 2u * n_lanes); // Rebase indexes for codebook slice 2.
    svuint32_t idxs_3 = svsub_n_u32_x(pg, idxs, 3u * n_lanes); // Rebase indexes for codebook slice 3.
    svint32_t weights_1 = svtbl_s32(svget4_s32(cb_regs_x4, 1), idxs_1); // Lookup values from codebook slice 1.
    svint32_t weights_2 = svtbl_s32(svget4_s32(cb_regs_x4, 2), idxs_2); // Lookup values from codebook slice 2.
    svint32_t weights_3 = svtbl_s32(svget4_s32(cb_regs_x4, 3), idxs_3); // Lookup values from codebook slice 3.
    svint32_t weights = weights_0; // Start with the first slice, then overwrite lanes that belong to later slices.

    weights = svsel_s32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights); // Keep slice 1 for indexes >= one register.
    weights = svsel_s32(svcmpge_n_u32(pg, idxs, 2u * n_lanes), weights_2, weights); // Keep slice 2 for indexes >= two registers.
    weights = svsel_s32(svcmpge_n_u32(pg, idxs, 3u * n_lanes), weights_3, weights); // Keep slice 3 for indexes >= three registers.

    return weights; // Return the same logical result as a wider table lookup.
}

/*
 * fp32 version of gemm_extract_weightsx4_s32().
 *
 * This keeps larger floating-point codebooks resident in SVE registers and
 * avoids a gather load for every decoded packed index.
 */
static svfloat32_t gemm_extract_weightsx4_f32(svbool_t pg,
                                              svuint32_t idxs,
                                              svfloat32x4_t cb_regs_x4) {
    const uint32_t n_lanes = (uint32_t)svcntw();
    svfloat32_t weights_0 = svtbl_f32(svget4_f32(cb_regs_x4, 0), idxs);
    svuint32_t idxs_1 = svsub_n_u32_x(pg, idxs, n_lanes);
    svuint32_t idxs_2 = svsub_n_u32_x(pg, idxs, 2u * n_lanes);
    svuint32_t idxs_3 = svsub_n_u32_x(pg, idxs, 3u * n_lanes);
    svfloat32_t weights_1 = svtbl_f32(svget4_f32(cb_regs_x4, 1), idxs_1);
    svfloat32_t weights_2 = svtbl_f32(svget4_f32(cb_regs_x4, 2), idxs_2);
    svfloat32_t weights_3 = svtbl_f32(svget4_f32(cb_regs_x4, 3), idxs_3);
    svfloat32_t weights = weights_0;

    weights = svsel_f32(svcmpge_n_u32(pg, idxs, n_lanes), weights_1, weights);
    weights = svsel_f32(svcmpge_n_u32(pg, idxs, 2u * n_lanes), weights_2, weights);
    weights = svsel_f32(svcmpge_n_u32(pg, idxs, 3u * n_lanes), weights_3, weights);

    return weights;
}
#endif

/*
 * Example masks produced by gemm_sve_idx_mask():
 *   bits_per_cb = 1 -> mask = 0b1
 *   bits_per_cb = 2 -> mask = 0b11
 *   bits_per_cb = 4 -> mask = 0b1111
 *   bits_per_cb = 8 -> mask = 0b11111111
 */

/*
 * Compute one compact int8-input GEMM output column for a tile of sequence rows.
 *
 * packed_row points to one compressed weight row. Each packed word contains
 * several codebook indexes; each index selects an int32 weight from
 * codebook_i32. For every input row in the sequence tile, the kernel computes:
 *
 *   out_mat[row, out_col] = dot(in_mat[row, 0:k_elems], decoded_weights)
 *
 * The input activations are int8 values widened to int32 SVE lanes, while the
 * accumulator and output are int32. When accumulate is nonzero, the result is
 * added to an existing output slot instead of replacing it.
 */
void sve_gemm_row_compact_int8(const uint32_t *packed_row,
                               uint32_t n_words_row,
                               uint32_t k_elems,
                               const int8_t *in_mat,
                               uint32_t seq_tile,
                               uint32_t ld_in,
                               const int32_t *codebook_i32,
                               int32_t *out_mat,
                               uint32_t out_col,
                               uint32_t ld_out,
                               int32_t bias_val,
                               int add_bias,
                               int accumulate,
                               uint8_t bits_per_cb) {
    /*
     * Degenerate case: there are no packed indexes to decode. The output still
     * honors bias and accumulate semantics so callers do not need a separate
     * slow path.
     */
    if ((bits_per_cb == 0u) || (k_elems == 0u)) { // No input elements, so output is just bias (if add_bias) or zero
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t acc = add_bias ? bias_val : 0;
            int32_t *out_slot = &out_mat[row * ld_out + out_col];
            if (accumulate) {
                *out_slot += acc;
            } else {
                *out_slot = acc;
            }
        }
        return;
    }

    /*
     * idxs_per_word controls how many logical weights are encoded in one
     * packed uint32_t. For example, 4-bit indexes store eight weights per word.
     */
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw(); // Auto-detect number of 32-bit lanes in SVE vector

    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

    /*
     * Process one input sequence row at a time. Each row produces one scalar
     * output for the requested output column.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v = svdup_s32(0);
        uint32_t input_idx = 0;

        /*
         * Load a vector of packed words. Tail predicates keep the final load
         * valid when n_words_row is not a multiple of the SVE lane count.
         */
        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg =
                svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row); // n_words_row is the number of uint32_t words in the row
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                /*
                 * Broadcast one packed word, then use an SVE vector of shift
                 * amounts to extract several indexes from that single word at
                 * once. This turns scalar bit packing into vector lanes.
                 */
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane); // duplicate the current lane's packed indices across the vector for processing

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    /*
                     * bits_pg marks only the valid indexes in this chunk. It
                     * handles both the end of a packed word and the end of the
                     * logical K dimension.
                     */
                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb); // shifts for extracting each index from the packed word

                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v); // Get the actual codebook indices for this set of lanes

                    svint32_t in_vals =
                        svld1sb_s32(bits_pg, &in_mat[row * ld_in + input_idx]);
                    svint32_t weights =
                        svld1_gather_u32index_s32(bits_pg, codebook_i32, cb_idxs); // Get the corresponding weights from the codebook for these indices
                    // Example: weights = [codebook_i32[2], codebook_i32[0], codebook_i32[3], codebook_i32[1]]

                    acc_v = svmla_s32_m(bits_pg, acc_v, in_vals, weights); // Add them together into the accumulator vector

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        /*
         * Reduce all partial products in acc_v into the final dot product for
         * this row/column pair, then apply the requested output policy.
         */
        int32_t acc = svaddv_s32(svptrue_b32(), acc_v); // Horizontally add the vector accumulator to get the final dot product for this output element
        if (add_bias) {
            acc += bias_val;
        }

        int32_t *out_slot = &out_mat[row * ld_out + out_col]; 
        if (accumulate) {
            *out_slot += acc;
        } else {
            *out_slot = acc;
        }
    }
}

/*
 * Compute one compact fp32 GEMM output column for a tile of sequence rows.
 *
 * This is the floating-point counterpart of sve_gemm_row_compact_int8(). The
 * packed weights still come from codebook indexes, but both input activations
 * and decoded weights are fp32 and the accumulator is an SVE fp32 vector.
 */
void sve_gemm_row_compact_fp32(const uint32_t *packed_row,
                               uint32_t n_words_row,
                               uint32_t k_elems,
                               const float *in_mat,
                               uint32_t seq_tile,
                               uint32_t ld_in,
                               const float *codebook,
                               float *out_mat,
                               uint32_t out_col,
                               uint32_t ld_out,
                               float bias_val,
                               int add_bias,
                               int accumulate,
                               uint8_t bits_per_cb) {
    /*
     * Degenerate case: no weights are decoded, but the caller still receives
     * either bias or zero according to the normal output policy.
     */
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float acc = add_bias ? bias_val : 0.0f;
            float *out_slot = &out_mat[row * ld_out + out_col];
            if (accumulate) {
                *out_slot += acc;
            } else {
                *out_slot = acc;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

    /*
     * Same unpack -> lookup -> multiply -> reduce structure as the int8
     * compact kernel, but using fp32 loads, codebook gathers, and accumulators.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg = svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts = svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);
                    svuint32_t cb_idxs = svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    svfloat32_t in_vals = svld1_f32(bits_pg, &in_mat[row * ld_in + input_idx]);
                    svfloat32_t weights =
                        svld1_gather_u32index_f32(bits_pg, codebook, cb_idxs);

                    acc_v = svmla_f32_m(bits_pg, acc_v, in_vals, weights);
                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc = svaddv_f32(svptrue_b32(), acc_v);
        if (add_bias) {
            acc += bias_val;
        }

        float *out_slot = &out_mat[row * ld_out + out_col];
        if (accumulate) {
            *out_slot += acc;
        } else {
            *out_slot = acc;
        }
    }
}

/*
 * Compute four fp32 compact GEMM outputs with four independent packed rows.
 *
 * The "4D" layout stores four streams next to each other:
 *
 *   packed_rows_interleaved: [idx0_d0, idx0_d1, idx0_d2, idx0_d3, ...]
 *   in_mat_interleaved:      [x0_d0,   x0_d1,   x0_d2,   x0_d3,   ...]
 *   codebook_interleaved:    [cb0_d0,  cb0_d1,  cb0_d2,  cb0_d3,  ...]
 *   out_mat_interleaved:     [y_d0,    y_d1,    y_d2,    y_d3,    ...]
 *
 * "diff_seq" means each of the four dimensions can use its own compressed
 * packed-row stream. The kernel decodes four independent codebook-index streams
 * and accumulates four dot products for each sequence row.
 */
void sve_gemm_row_compact_fp32_interleaved_4Learners_diff_seq(
    const uint32_t *packed_rows_interleaved,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;
            const float bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0.0f;
            const float bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Load the four interleaved fp32 codebooks into SVE registers before entering
 * the row loop. The generated configuration chooses whether each logical
 * codebook fits in one, two, or four physical SVE registers.
 */
#if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
#endif

    /*
     * Each sequence row reuses the preloaded codebooks and walks the compressed
     * packed rows from the beginning. Four accumulators track four independent
     * dot products for the same logical row.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        svfloat32_t acc_v2 = svdup_f32(0.0f);
        svfloat32_t acc_v3 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            /*
             * svld4_u32 de-interleaves the packed 4D stream. After this load,
             * packed_idxs0..3 contain the compressed index words for the four
             * independent dimensions.
             */
            svuint32x4_t packed_idxs_4d =
                svld4_u32(load_pg, &packed_rows_interleaved[cw * 4u]);
            svuint32_t packed_idxs0 = svget4_u32(packed_idxs_4d, 0);
            svuint32_t packed_idxs1 = svget4_u32(packed_idxs_4d, 1);
            svuint32_t packed_idxs2 = svget4_u32(packed_idxs_4d, 2);
            svuint32_t packed_idxs3 = svget4_u32(packed_idxs_4d, 3);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed0 = svdup_lane_u32(packed_idxs0, lane);
                svuint32_t dup_idxs_packed1 = svdup_lane_u32(packed_idxs1, lane);
                svuint32_t dup_idxs_packed2 = svdup_lane_u32(packed_idxs2, lane);
                svuint32_t dup_idxs_packed3 = svdup_lane_u32(packed_idxs3, lane);

                /*
                 * Decode several indexes from the selected packed word. The
                 * four dimensions use the same shift vector but different
                 * packed words, so their decoded indexes may differ.
                 */
                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    svuint32_t cb_idxs0 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed0, shifts);
                    svuint32_t cb_idxs1 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed1, shifts);
                    svuint32_t cb_idxs2 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed2, shifts);
                    svuint32_t cb_idxs3 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed3, shifts);

                    cb_idxs0 = svand_u32_z(bits_pg, cb_idxs0, idx_mask_v);
                    cb_idxs1 = svand_u32_z(bits_pg, cb_idxs1, idx_mask_v);
                    cb_idxs2 = svand_u32_z(bits_pg, cb_idxs2, idx_mask_v);
                    cb_idxs3 = svand_u32_z(bits_pg, cb_idxs3, idx_mask_v);

                    /*
                     * svld4_f32 de-interleaves input activations into four SVE
                     * vectors. Each vector is multiplied by the matching
                     * codebook lookup result below.
                     */
                    svfloat32x4_t in_vals =
                        svld4_f32(bits_pg, &row_in[input_idx * 4u]);
                    svfloat32_t in0 = svget4_f32(in_vals, 0);
                    svfloat32_t in1 = svget4_f32(in_vals, 1);
                    svfloat32_t in2 = svget4_f32(in_vals, 2);
                    svfloat32_t in3 = svget4_f32(in_vals, 3);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);
                    svfloat32_t weights2 = svdup_n_f32(0);
                    svfloat32_t weights3 = svdup_n_f32(0);

/*
 * Choose the lookup strategy for the codebook size selected at generation
 * time. One-register codebooks can use svtbl directly; larger codebooks use
 * helper functions that combine lookups across multiple registers.
 */
#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs0);
                    weights1 = svtbl_f32(cb1, cb_idxs1);
                    weights2 = svtbl_f32(cb2, cb_idxs2);
                    weights3 = svtbl_f32(cb3, cb_idxs3);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs0, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs1, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs2, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs3, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs0, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs1, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs2, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs3, cb3_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_f32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_f32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        /*
         * Finish the four dot products, add optional per-dimension bias, and
         * write the interleaved output group for this row and output column.
         */
        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);
        float acc2 = svaddv_f32(svptrue_b32(), acc_v2);
        float acc3 = svaddv_f32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

/*
 * Compute two fp32 compact GEMM outputs that share the same packed index row.
 *
 * "2D" means inputs, codebooks, bias, and outputs are stored as pairs. The
 * "same_seq" form reuses one decoded codebook-index stream for both dimensions,
 * then applies that same index to two interleaved codebooks. This is useful
 * when two generated learners share the same compressed structure but have
 * different codebook values.
 */
void sve_gemm_row_compact_fp32_interleaved_2Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Preload the two interleaved fp32 codebooks. The same decoded index stream
 * will select from both codebooks, producing two output dimensions per row.
 */
#if defined(N_SVE_REG_CB_1)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget2_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget2_f32(codebooks_loaded, 1);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 2u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget2_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget2_f32(codebooks_loaded, 1));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x2_t codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 2u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget2_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget2_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget2_f32(codebooks_loaded, 1), svdup_n_f32(0));

    codebooks_loaded = svld2_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget2_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget2_f32(codebooks_loaded, 1));
#endif

    /*
     * Decode one packed row and apply each decoded index to both interleaved
     * codebooks. The input stream is de-interleaved with svld2_f32 so each
     * accumulator receives the matching activation dimension.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    svfloat32x2_t in_vals =
                        svld2_f32(bits_pg, &row_in[input_idx * 2u]);
                    svfloat32_t in0 = svget2_f32(in_vals, 0);
                    svfloat32_t in1 = svget2_f32(in_vals, 1);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs);
                    weights1 = svtbl_f32(cb1, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb1_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb1_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

/*
 * Compute four fp32 compact GEMM outputs that share the same packed index row.
 *
 * This combines the 4-learner interleaved data layout with the "same_seq" packed
 * index convention. One packed index stream is decoded, then used to select
 * weights from four interleaved fp32 codebooks. Four accumulators track the
 * four dot products in parallel.
 */
void sve_gemm_row_compact_fp32_interleaved_4Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const float *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const float *codebook_interleaved,
    uint32_t codebook_size,
    float *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const float *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            float *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const float bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0.0f;
            const float bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0.0f;
            const float bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0.0f;
            const float bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0.0f;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Preload four interleaved fp32 codebooks for the shared-index 4D path. The
 * packed row is decoded once, and the same logical index selects one weight
 * from each of the four codebook streams.
 */
#if defined(N_SVE_REG_CB_1)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_interleaved);
    svfloat32_t cb0 = svget4_f32(codebooks_loaded, 0);
    svfloat32_t cb1 = svget4_f32(codebooks_loaded, 1);
    svfloat32_t cb2 = svget4_f32(codebooks_loaded, 2);
    svfloat32_t cb3 = svget4_f32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x2_t cb0_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    svfloat32x2_t cb1_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    svfloat32x2_t cb2_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    svfloat32x2_t cb3_2regs = svcreate2_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_f32(svget2_f32(cb0_2regs, 0), svget4_f32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_f32(svget2_f32(cb1_2regs, 0), svget4_f32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_f32(svget2_f32(cb2_2regs, 0), svget4_f32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_f32(svget2_f32(cb3_2regs, 0), svget4_f32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svfloat32x4_t codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[0u * (N_SVE_LANES * 4u)]);
    svfloat32x4_t cb0_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb1_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb2_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));
    svfloat32x4_t cb3_4regs = svcreate4_f32(svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(codebooks_loaded, 0), svdup_n_f32(0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(codebooks_loaded, 1), svdup_n_f32(0));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(codebooks_loaded, 2), svdup_n_f32(0));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(codebooks_loaded, 3), svdup_n_f32(0));

    codebooks_loaded = svld4_f32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_f32(svget4_f32(cb0_4regs, 0), svget4_f32(cb0_4regs, 1), svget4_f32(cb0_4regs, 2), svget4_f32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_f32(svget4_f32(cb1_4regs, 0), svget4_f32(cb1_4regs, 1), svget4_f32(cb1_4regs, 2), svget4_f32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_f32(svget4_f32(cb2_4regs, 0), svget4_f32(cb2_4regs, 1), svget4_f32(cb2_4regs, 2), svget4_f32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_f32(svget4_f32(cb3_4regs, 0), svget4_f32(cb3_4regs, 1), svget4_f32(cb3_4regs, 2), svget4_f32(codebooks_loaded, 3));
#endif

    /*
     * The loop structure mirrors the 4-learner diff_seq kernel, except there is only
     * one packed index stream. cb_idxs is therefore reused for all four
     * codebook lookups.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svfloat32_t acc_v0 = svdup_f32(0.0f);
        svfloat32_t acc_v1 = svdup_f32(0.0f);
        svfloat32_t acc_v2 = svdup_f32(0.0f);
        svfloat32_t acc_v3 = svdup_f32(0.0f);
        uint32_t input_idx = 0;

        const float *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    svfloat32x4_t in_vals =
                        svld4_f32(bits_pg, &row_in[input_idx * 4u]);
                    svfloat32_t in0 = svget4_f32(in_vals, 0);
                    svfloat32_t in1 = svget4_f32(in_vals, 1);
                    svfloat32_t in2 = svget4_f32(in_vals, 2);
                    svfloat32_t in3 = svget4_f32(in_vals, 3);

                    svfloat32_t weights0 = svdup_n_f32(0);
                    svfloat32_t weights1 = svdup_n_f32(0);
                    svfloat32_t weights2 = svdup_n_f32(0);
                    svfloat32_t weights3 = svdup_n_f32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_f32(cb0, cb_idxs);
                    weights1 = svtbl_f32(cb1, cb_idxs);
                    weights2 = svtbl_f32(cb2, cb_idxs);
                    weights3 = svtbl_f32(cb3, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_f32(bits_pg, cb_idxs, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_f32(bits_pg, cb_idxs, cb3_4regs);
#endif

                    acc_v0 = svmla_f32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_f32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_f32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_f32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        float acc0 = svaddv_f32(svptrue_b32(), acc_v0);
        float acc1 = svaddv_f32(svptrue_b32(), acc_v1);
        float acc2 = svaddv_f32(svptrue_b32(), acc_v2);
        float acc3 = svaddv_f32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        float *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

/*
 * Compute four int32 compact GEMM outputs with four independent packed rows.
 *
 * This is the int32/int8-style counterpart of
 * sve_gemm_row_compact_fp32_interleaved_4Learners_diff_seq(). Input values and
 * codebook weights are carried as int32 SVE lanes, and the four compressed
 * packed-row streams are decoded independently before multiply-accumulate.
 */
void sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq(
    const uint32_t *packed_rows_interleaved,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int8_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    /*
     * If no packed indexes can be decoded, the dot product is empty. The
     * function still writes the correct two-lane output group: either bias
     * values, when requested, or zeros. accumulate keeps the same meaning as in
     * the normal path.
     */
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;
            const int32_t bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0;
            const int32_t bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    /*
     * One uint32_t packed word stores several codebook indexes. For example:
     * - bits_per_cb = 4 means 8 indexes per word.
     * - bits_per_cb = 8 means 4 indexes per word.
     * idx_mask keeps only the selected index after it has been shifted down.
     */
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Preload four interleaved int32 codebooks. The diff_seq integer path decodes
 * four separate packed index streams, then uses these resident codebooks for
 * table lookup instead of gathering from memory on every multiply.
 */
#if defined(N_SVE_REG_CB_1)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget4_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget4_s32(codebooks_loaded, 1);
    svint32_t cb2 = svget4_s32(codebooks_loaded, 2);
    svint32_t cb3 = svget4_s32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    svint32x2_t cb2_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    svint32x2_t cb3_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget4_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget4_s32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_s32(svget2_s32(cb2_2regs, 0), svget4_s32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_s32(svget2_s32(cb3_2regs, 0), svget4_s32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb2_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb3_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget4_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget4_s32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(cb2_4regs, 2), svget4_s32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(cb3_4regs, 2), svget4_s32(codebooks_loaded, 3));
#endif

    /*
     * Four int32 accumulators track the four interleaved dimensions. Inputs are
     * already int32 here, so the multiply-accumulate uses svmla_s32_m directly.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        svint32_t acc_v2 = svdup_s32(0);
        svint32_t acc_v3 = svdup_s32(0);
        uint32_t input_idx = 0;

        const int8_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32x4_t packed_idxs_4d =
                svld4_u32(load_pg, &packed_rows_interleaved[cw * 4u]);
            svuint32_t packed_idxs0 = svget4_u32(packed_idxs_4d, 0);
            svuint32_t packed_idxs1 = svget4_u32(packed_idxs_4d, 1);
            svuint32_t packed_idxs2 = svget4_u32(packed_idxs_4d, 2);
            svuint32_t packed_idxs3 = svget4_u32(packed_idxs_4d, 3);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                svuint32_t dup_idxs_packed0 = svdup_lane_u32(packed_idxs0, lane);
                svuint32_t dup_idxs_packed1 = svdup_lane_u32(packed_idxs1, lane);
                svuint32_t dup_idxs_packed2 = svdup_lane_u32(packed_idxs2, lane);
                svuint32_t dup_idxs_packed3 = svdup_lane_u32(packed_idxs3, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    svuint32_t cb_idxs0 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed0, shifts);
                    svuint32_t cb_idxs1 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed1, shifts);
                    svuint32_t cb_idxs2 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed2, shifts);
                    svuint32_t cb_idxs3 =
                        svlsr_u32_z(bits_pg, dup_idxs_packed3, shifts);

                    cb_idxs0 = svand_u32_z(bits_pg, cb_idxs0, idx_mask_v);
                    cb_idxs1 = svand_u32_z(bits_pg, cb_idxs1, idx_mask_v);
                    cb_idxs2 = svand_u32_z(bits_pg, cb_idxs2, idx_mask_v);
                    cb_idxs3 = svand_u32_z(bits_pg, cb_idxs3, idx_mask_v);

                    svint32x4_t in_vals =
                        gemm_load_in4_s8(bits_pg, &row_in[input_idx * 4u]);
                    svint32_t in0 = svget4_s32(in_vals, 0);
                    svint32_t in1 = svget4_s32(in_vals, 1);
                    svint32_t in2 = svget4_s32(in_vals, 2);
                    svint32_t in3 = svget4_s32(in_vals, 3);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);
                    svint32_t weights2 = svdup_n_s32(0);
                    svint32_t weights3 = svdup_n_s32(0);

#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs0);
                    weights1 = svtbl_s32(cb1, cb_idxs1);
                    weights2 = svtbl_s32(cb2, cb_idxs2);
                    weights3 = svtbl_s32(cb3, cb_idxs3);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs0, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs1, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs2, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs3, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs0, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs1, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs2, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs3, cb3_4regs);
#endif

                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_s32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_s32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        int32_t acc2 = svaddv_s32(svptrue_b32(), acc_v2);
        int32_t acc3 = svaddv_s32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

/*
 * Compute two int32 compact GEMM outputs that share the same packed index row.
 *
 * One packed index stream is decoded for each input position. The decoded
 * indexes select two interleaved int32 codebook values, which are multiplied
 * by the two interleaved input streams and reduced into two output values.
 */
void sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int8_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    /*
     * Empty-K path for two interleaved streams. No multiply-accumulate work is
     * possible, but the output group must still become either bias or zero, and
     * accumulate must still decide between adding and overwriting.
     */
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
            }
        }
        return;
    }

    /*
     * svdot fast path. Each iteration decodes the 16 indexes one svdot
     * accumulator consumes: one packed word at 2 bits per index, two at
     * 4 bits. The svmla path below stays as the fallback for every width
     * whose indexes straddle a byte boundary.
     */
    if (gemm_sdot_decode_usable(codebook_i32_interleaved, codebook_size, 2u,
                                bits_per_cb)) {
        const uint32_t words_per_iter = (uint32_t)bits_per_cb / 2u;
        const uint8_t idx_mask8 = (uint8_t)gemm_sve_idx_mask(bits_per_cb);
        int8_t cb_bytes0[256] = {0};
        int8_t cb_bytes1[256] = {0};
        const svint8_t cb0_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 2u, 0u, cb_bytes0);
        const svint8_t cb1_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 2u, 1u, cb_bytes1);
        svuint8_t byte_sel, shifts8;
        gemm_sdot_lane_patterns(bits_per_cb, &byte_sel, &shifts8);

        if (words_per_iter == 1u) {
            gemm_sdot_rows_2l(packed_row, n_words_row, k_elems, in_mat_interleaved,
                   seq_tile, ld_in_interleaved, out_mat_interleaved, out_col,
                   ld_out_interleaved, bias_interleaved, add_bias, accumulate,
                   cb0_b, cb1_b, byte_sel, shifts8, idx_mask8, 1u, NULL);
        } else {
            gemm_sdot_rows_2l(packed_row, n_words_row, k_elems, in_mat_interleaved,
                   seq_tile, ld_in_interleaved, out_mat_interleaved, out_col,
                   ld_out_interleaved, bias_interleaved, add_bias, accumulate,
                   cb0_b, cb1_b, byte_sel, shifts8, idx_mask8, 2u, NULL);
        }
        return;
    }

    /*
     * Decode parameters shared by both interleaved streams:
     * - idxs_per_word: number of packed codebook indexes in each uint32_t.
     * - idx_mask: mask used after shifting one index into the low bits.
     * - n_lanes: SVE vector length in 32-bit lanes, detected at runtime.
     */
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Preload the two interleaved int32 codebooks for the shared-index 2D integer
 * path. One decoded index vector selects both weights0 and weights1.
 *
 * codebook_i32_interleaved is arranged as:
 *   [cb0_stream0, cb0_stream1, cb1_stream0, cb1_stream1, ...]
 *
 * svld2_s32 splits this memory into two SVE vectors. If the codebook is wider
 * than one SVE register, the N_SVE_REG_CB_2 or N_SVE_REG_CB_4 branch stores the
 * later slices in tuple registers for table lookup.
 */
#if defined(N_SVE_REG_CB_1)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget2_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget2_s32(codebooks_loaded, 1);

#elif defined(N_SVE_REG_CB_2)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 2u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget2_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget2_s32(codebooks_loaded, 1), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget2_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget2_s32(codebooks_loaded, 1));

#elif defined(N_SVE_REG_CB_4)
    svint32x2_t codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 2u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget2_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget2_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget2_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget2_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget2_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget2_s32(codebooks_loaded, 1), svdup_n_s32(0));

    codebooks_loaded = svld2_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 2u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget2_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget2_s32(codebooks_loaded, 1));
#endif

    /*
     * Decode the shared packed index row and multiply two interleaved int32
     * activation streams by their matching decoded codebook weights.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        uint32_t input_idx = 0;

        /*
         * Select the current input row. The row stride already includes the
         * two interleaved streams, so row_in[input_idx * 2] addresses the next
         * logical K element.
         */
        const int8_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        /*
         * cw walks the packed weight row in groups of SVE lanes. The final
         * iteration is predicated so n_words_row does not need to be a multiple
         * of svcntw().
         */
        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                /*
                 * Work on one packed word at a time. Duplicating it across the
                 * vector lets the next loop extract multiple bit fields with a
                 * vector of shift amounts.
                 */
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    /*
                     * bits_pg activates only the valid extracted indexes. It
                     * protects both the partially-used tail of a packed word
                     * and the final K elements of the row.
                     */
                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    /*
                     * Decode indexes:
                     * 1. Shift the target bit field down to bit zero.
                     * 2. Mask off the neighboring packed fields.
                     * The same cb_idxs vector is reused for both interleaved
                     * codebook streams because this is the same_seq variant.
                     */
                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    /*
                     * Load two interleaved input streams for the same logical
                     * K positions. svld2_s32 returns:
                     *   in0 = stream 0 values
                     *   in1 = stream 1 values
                     */
                    svint32x2_t in_vals =
                        gemm_load_in2_s8(bits_pg, &row_in[input_idx * 2u]);
                    svint32_t in0 = svget2_s32(in_vals, 0);
                    svint32_t in1 = svget2_s32(in_vals, 1);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);

/*
 * Select codebook entries for both streams. The N_SVE_REG_CB_* macro is fixed
 * by generated codebook configuration, so only one of these lookup paths is
 * compiled for a given build.
 */
#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs);
                    weights1 = svtbl_s32(cb1, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb1_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb1_4regs);
#endif

                    /*
                     * Accumulate two dot products in parallel:
                     *   acc_v0 += input_stream0 * weight_stream0
                     *   acc_v1 += input_stream1 * weight_stream1
                     */
                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        /*
         * Reduce each vector accumulator into one scalar output, then add the
         * optional two-lane bias group for this output column.
         */
        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
        }

        /*
         * Store the two interleaved outputs for this logical row/column. The
         * output slot order is [stream0, stream1].
         */
        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 2u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

/*
 * Compute four int32 compact GEMM outputs that share the same packed index row.
 *
 * This version decodes one packed-row stream and applies the same codebook
 * indexes to four interleaved int32 codebooks. It is the 4D integer equivalent
 * of sve_gemm_row_compact_fp32_interleaved_4Learners_same_seq().
 */
void sve_gemm_row_compact_int8_interleaved_4Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int8_t *in_mat_interleaved,
    uint32_t seq_tile,
    uint32_t ld_in_interleaved,
    const int32_t *codebook_i32_interleaved,
    uint32_t codebook_size,
    int32_t *out_mat_interleaved,
    uint32_t out_col,
    uint32_t ld_out_interleaved,
    const int32_t *bias_interleaved,
    int add_bias,
    int accumulate,
    uint8_t bits_per_cb) {
    /*
     * Empty-K path for four interleaved streams. No multiply-accumulate work is
     * possible, but the output group must still become either bias or zero, and
     * accumulate must still decide between adding and overwriting.
     */
    if ((bits_per_cb == 0u) || (k_elems == 0u)) {
        for (uint32_t row = 0; row < seq_tile; row++) {
            int32_t *out_slot =
                &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
            const int32_t bias0 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[0] : 0;
            const int32_t bias1 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[1] : 0;
            const int32_t bias2 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[2] : 0;
            const int32_t bias3 =
                (add_bias && (bias_interleaved != NULL)) ? bias_interleaved[3] : 0;

            if (accumulate) {
                out_slot[0] += bias0;
                out_slot[1] += bias1;
                out_slot[2] += bias2;
                out_slot[3] += bias3;
            } else {
                out_slot[0] = bias0;
                out_slot[1] = bias1;
                out_slot[2] = bias2;
                out_slot[3] = bias3;
            }
        }
        return;
    }

    /*
     * svdot fast path, four streams. Identical decode to the two-stream
     * kernel: all four learners share one index row in the same_seq layout,
     * so one decoded byte vector drives four svtbl_s8 lookups and four
     * accumulators. svld4_s8 de-interleaves the four activation streams.
     */
    if (gemm_sdot_decode_usable(codebook_i32_interleaved, codebook_size, 4u,
                                bits_per_cb)) {
        const uint32_t words_per_iter = (uint32_t)bits_per_cb / 2u;
        const uint8_t idx_mask8 = (uint8_t)gemm_sve_idx_mask(bits_per_cb);
        int8_t cb_bytes0[256] = {0};
        int8_t cb_bytes1[256] = {0};
        int8_t cb_bytes2[256] = {0};
        int8_t cb_bytes3[256] = {0};
        const svint8_t cb0_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 4u, 0u, cb_bytes0);
        const svint8_t cb1_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 4u, 1u, cb_bytes1);
        const svint8_t cb2_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 4u, 2u, cb_bytes2);
        const svint8_t cb3_b =
            gemm_sdot_cb_table(codebook_i32_interleaved, codebook_size, 4u, 3u, cb_bytes3);
        svuint8_t byte_sel, shifts8;
        gemm_sdot_lane_patterns(bits_per_cb, &byte_sel, &shifts8);

        if (words_per_iter == 1u) {
            gemm_sdot_rows_4l(packed_row, n_words_row, k_elems, in_mat_interleaved,
                   seq_tile, ld_in_interleaved, out_mat_interleaved, out_col,
                   ld_out_interleaved, bias_interleaved, add_bias, accumulate,
                   cb0_b, cb1_b, cb2_b, cb3_b, byte_sel, shifts8, idx_mask8, 1u);
        } else {
            gemm_sdot_rows_4l(packed_row, n_words_row, k_elems, in_mat_interleaved,
                   seq_tile, ld_in_interleaved, out_mat_interleaved, out_col,
                   ld_out_interleaved, bias_interleaved, add_bias, accumulate,
                   cb0_b, cb1_b, cb2_b, cb3_b, byte_sel, shifts8, idx_mask8, 2u);
        }
        return;
    }

    /*
     * Decode parameters shared by all four streams:
     * - idxs_per_word: number of packed codebook indexes in each uint32_t.
     * - idx_mask: mask used after shifting one index into the low bits.
     * - n_lanes: SVE vector length in 32-bit lanes, detected at runtime.
     */
    const uint32_t idxs_per_word = 32u / bits_per_cb;
    const uint32_t idx_mask = gemm_sve_idx_mask(bits_per_cb);
    const uint32_t n_lanes = (uint32_t)svcntw();
    const svuint32_t idx_mask_v = svdup_u32(idx_mask);

/*
 * Preload four interleaved int32 codebooks for the shared-index 4D integer
 * path. The same cb_idxs vector is used for all four codebook streams.
 *
 * codebook_i32_interleaved is arranged as repeating groups of four:
 *   [cb0_s0, cb0_s1, cb0_s2, cb0_s3, cb1_s0, cb1_s1, ...]
 *
 * svld4_s32 de-interleaves those groups into stream-specific SVE vectors. When
 * a logical codebook spans multiple SVE registers, tuple registers keep all
 * slices available for the helper lookup functions.
 */
#if defined(N_SVE_REG_CB_1)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), codebook_i32_interleaved);
    svint32_t cb0 = svget4_s32(codebooks_loaded, 0);
    svint32_t cb1 = svget4_s32(codebooks_loaded, 1);
    svint32_t cb2 = svget4_s32(codebooks_loaded, 2);
    svint32_t cb3 = svget4_s32(codebooks_loaded, 3);

#elif defined(N_SVE_REG_CB_2)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x2_t cb0_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    svint32x2_t cb1_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    svint32x2_t cb2_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    svint32x2_t cb3_2regs = svcreate2_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_2regs = svcreate2_s32(svget2_s32(cb0_2regs, 0), svget4_s32(codebooks_loaded, 0));
    cb1_2regs = svcreate2_s32(svget2_s32(cb1_2regs, 0), svget4_s32(codebooks_loaded, 1));
    cb2_2regs = svcreate2_s32(svget2_s32(cb2_2regs, 0), svget4_s32(codebooks_loaded, 2));
    cb3_2regs = svcreate2_s32(svget2_s32(cb3_2regs, 0), svget4_s32(codebooks_loaded, 3));

#elif defined(N_SVE_REG_CB_4)
    svint32x4_t codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[0u * (N_SVE_LANES * 4u)]);
    svint32x4_t cb0_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb1_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb2_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));
    svint32x4_t cb3_4regs = svcreate4_s32(svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[1u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[2u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(codebooks_loaded, 0), svdup_n_s32(0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(codebooks_loaded, 1), svdup_n_s32(0));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(codebooks_loaded, 2), svdup_n_s32(0));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(codebooks_loaded, 3), svdup_n_s32(0));

    codebooks_loaded = svld4_s32(svwhilelt_b32((uint64_t)0, (uint64_t)codebook_size), &codebook_i32_interleaved[3u * (N_SVE_LANES * 4u)]);
    cb0_4regs = svcreate4_s32(svget4_s32(cb0_4regs, 0), svget4_s32(cb0_4regs, 1), svget4_s32(cb0_4regs, 2), svget4_s32(codebooks_loaded, 0));
    cb1_4regs = svcreate4_s32(svget4_s32(cb1_4regs, 0), svget4_s32(cb1_4regs, 1), svget4_s32(cb1_4regs, 2), svget4_s32(codebooks_loaded, 1));
    cb2_4regs = svcreate4_s32(svget4_s32(cb2_4regs, 0), svget4_s32(cb2_4regs, 1), svget4_s32(cb2_4regs, 2), svget4_s32(codebooks_loaded, 2));
    cb3_4regs = svcreate4_s32(svget4_s32(cb3_4regs, 0), svget4_s32(cb3_4regs, 1), svget4_s32(cb3_4regs, 2), svget4_s32(codebooks_loaded, 3));
#endif

    /*
     * Walk one compressed row per output column. The decoded indexes are shared
     * by all four dimensions, while svld4_s32 separates the four interleaved
     * activation streams.
     */
    for (uint32_t row = 0; row < seq_tile; row++) {
        svint32_t acc_v0 = svdup_s32(0);
        svint32_t acc_v1 = svdup_s32(0);
        svint32_t acc_v2 = svdup_s32(0);
        svint32_t acc_v3 = svdup_s32(0);
        uint32_t input_idx = 0;

        /*
         * row_in points to the first value in the current sequence row. Each
         * logical K element occupies four adjacent int32 values, one per
         * interleaved stream.
         */
        const int8_t *row_in = &in_mat_interleaved[row * ld_in_interleaved];

        /*
         * Read packed index words in SVE-width chunks. The same packed row is
         * shared by all four output streams because this is the same_seq path.
         */
        for (uint32_t cw = 0; (cw < n_words_row) && (input_idx < k_elems);
             cw += n_lanes) {
            svbool_t load_pg = svwhilelt_b32((uint64_t)cw, (uint64_t)n_words_row);
            svuint32_t packed_idxs = svld1_u32(load_pg, &packed_row[cw]);
            uint32_t n_loaded_lanes = (uint32_t)svcntp_b32(load_pg, load_pg);

            for (uint32_t lane = 0; lane < n_loaded_lanes; lane++) {
                /*
                 * Each lane of packed_idxs is one uint32_t packed word. By
                 * broadcasting that word, the inner loop can unpack several
                 * codebook indexes with vector shifts.
                 */
                svuint32_t dup_idxs_packed = svdup_lane_u32(packed_idxs, lane);

                for (uint32_t idx_ptr = 0;
                     (idx_ptr < idxs_per_word) && (input_idx < k_elems);
                     idx_ptr += n_lanes) {
                    const uint32_t missing_lane = idxs_per_word - idx_ptr;
                    const uint32_t missing_total = k_elems - input_idx;
                    const uint32_t active_lanes =
                        (missing_lane < missing_total) ? missing_lane : missing_total;

                    /*
                     * active_lanes may be smaller than the SVE width at the end
                     * of a packed word or the end of k_elems. bits_pg keeps
                     * loads, table lookups, and multiply-accumulates masked to
                     * the valid elements only.
                     */
                    svbool_t bits_pg =
                        svwhilelt_b32((uint64_t)0, (uint64_t)active_lanes);
                    svuint32_t shifts =
                        svindex_u32(idx_ptr * bits_per_cb, bits_per_cb);

                    /*
                     * Decode one vector of codebook indexes from the packed
                     * word. Since all four streams share the same index row,
                     * this one cb_idxs vector selects weights from cb0..cb3.
                     */
                    svuint32_t cb_idxs =
                        svlsr_u32_z(bits_pg, dup_idxs_packed, shifts);
                    cb_idxs = svand_u32_z(bits_pg, cb_idxs, idx_mask_v);

                    /*
                     * Load four input streams for the current K positions.
                     * svld4_s32 converts memory groups of
                     * [s0, s1, s2, s3] into four independent SVE vectors.
                     */
                    svint32x4_t in_vals =
                        gemm_load_in4_s8(bits_pg, &row_in[input_idx * 4u]);
                    svint32_t in0 = svget4_s32(in_vals, 0);
                    svint32_t in1 = svget4_s32(in_vals, 1);
                    svint32_t in2 = svget4_s32(in_vals, 2);
                    svint32_t in3 = svget4_s32(in_vals, 3);

                    svint32_t weights0 = svdup_n_s32(0);
                    svint32_t weights1 = svdup_n_s32(0);
                    svint32_t weights2 = svdup_n_s32(0);
                    svint32_t weights3 = svdup_n_s32(0);

/*
 * Translate decoded indexes into four vectors of real int32 weights. Direct
 * svtbl_s32 is enough for one-register codebooks; multi-register codebooks use
 * helper functions that rebase indexes across the tuple register slices.
 */
#ifdef N_SVE_REG_CB_1
                    weights0 = svtbl_s32(cb0, cb_idxs);
                    weights1 = svtbl_s32(cb1, cb_idxs);
                    weights2 = svtbl_s32(cb2, cb_idxs);
                    weights3 = svtbl_s32(cb3, cb_idxs);
#endif

#ifdef N_SVE_REG_CB_2
                    weights0 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb0_2regs);
                    weights1 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb1_2regs);
                    weights2 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb2_2regs);
                    weights3 = gemm_extract_weightsx2_s32(bits_pg, cb_idxs, cb3_2regs);
#endif

#ifdef N_SVE_REG_CB_4
                    weights0 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb0_4regs);
                    weights1 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb1_4regs);
                    weights2 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb2_4regs);
                    weights3 = gemm_extract_weightsx4_s32(bits_pg, cb_idxs, cb3_4regs);
#endif

                    /*
                     * Accumulate four dot products in lockstep. The stream
                     * number is preserved all the way through input load,
                     * codebook lookup, accumulator, and output slot.
                     */
                    acc_v0 = svmla_s32_m(bits_pg, acc_v0, in0, weights0);
                    acc_v1 = svmla_s32_m(bits_pg, acc_v1, in1, weights1);
                    acc_v2 = svmla_s32_m(bits_pg, acc_v2, in2, weights2);
                    acc_v3 = svmla_s32_m(bits_pg, acc_v3, in3, weights3);

                    input_idx += (uint32_t)svcntp_b32(bits_pg, bits_pg);
                }
            }
        }

        /*
         * Horizontally reduce the four vector accumulators into scalar GEMM
         * outputs and add the optional four-lane bias group.
         */
        int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
        int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
        int32_t acc2 = svaddv_s32(svptrue_b32(), acc_v2);
        int32_t acc3 = svaddv_s32(svptrue_b32(), acc_v3);

        if (add_bias && (bias_interleaved != NULL)) {
            acc0 += bias_interleaved[0];
            acc1 += bias_interleaved[1];
            acc2 += bias_interleaved[2];
            acc3 += bias_interleaved[3];
        }

        /*
         * Write the final interleaved output group:
         *   out_slot[0..3] = streams 0..3 for (row, out_col).
         */
        int32_t *out_slot =
            &out_mat_interleaved[row * ld_out_interleaved + out_col * 4u];
        if (accumulate) {
            out_slot[0] += acc0;
            out_slot[1] += acc1;
            out_slot[2] += acc2;
            out_slot[3] += acc3;
        } else {
            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

/*
 * Dense 4-learner interleaved int32 GEMM helper.
 *
 * Unlike the compact kernels above, this function receives already-expanded
 * left-hand-side and right-hand-side matrices. rhs_by_col_interleaved is laid
 * out by output column so each inner loop walks a contiguous RHS column. Four
 * interleaved dot products are accumulated for every (row, column) pair.
 */
void sve_gemm_dense_int8_interleaved_4Learners(
    const int8_t *lhs_interleaved,
    const int8_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved) {
    /* K positions consumed per iteration: one byte per position per stream. */
    const uint32_t k_per_iter = (uint32_t)svcntb();

    /*
     * Dense layout notes:
     * - lhs_interleaved stores four values for each K position of each row.
     * - rhs_by_col_interleaved stores four values for each K position of each
     *   output column.
     * - out_interleaved stores four reduced dot products for each matrix cell.
     */
    for (uint32_t row = 0; row < lhs_rows; row++) {
        const int8_t *lhs_row = &lhs_interleaved[row * k_elems * 4u];

        for (uint32_t col = 0; col < rhs_cols; col++) {
            const int8_t *rhs_col = &rhs_by_col_interleaved[col * k_elems * 4u];
            svint32_t acc_v0 = svdup_s32(0);
            svint32_t acc_v1 = svdup_s32(0);
            svint32_t acc_v2 = svdup_s32(0);
            svint32_t acc_v3 = svdup_s32(0);

            /*
             * One svdot_s32 per stream replaces one svmla_s32 per stream, and
             * covers four K positions per 32-bit lane instead of one.
             *
             * svld4_s8 de-interleaves at byte width, so each stream's vector
             * holds svcntb() consecutive K values. svdot_s32 reads each 32-bit
             * lane as four int8 pairs and accumulates their four products into
             * that lane, which makes lane j cover K = 4j..4j+3 -- exactly the
             * grouping the de-interleaved load just produced. No repacking is
             * needed: the interleaved layout already has the right shape.
             *
             * The loop therefore advances by svcntb() K positions rather than
             * svcntw(), a quarter of the iterations for the same work. The
             * predicate is a byte predicate for the same reason; inactive
             * elements load as zero and contribute zero products, so the tail
             * needs no special case and svdot_s32 needs no predicate of its own.
             */
            for (uint32_t k = 0; k < k_elems; k += k_per_iter) {
                svbool_t pg = svwhilelt_b8((uint64_t)k, (uint64_t)k_elems);
                svint8x4_t lhs_vals = svld4_s8(pg, &lhs_row[k * 4u]);
                svint8x4_t rhs_vals = svld4_s8(pg, &rhs_col[k * 4u]);

                acc_v0 = svdot_s32(acc_v0, svget4_s8(lhs_vals, 0), svget4_s8(rhs_vals, 0));
                acc_v1 = svdot_s32(acc_v1, svget4_s8(lhs_vals, 1), svget4_s8(rhs_vals, 1));
                acc_v2 = svdot_s32(acc_v2, svget4_s8(lhs_vals, 2), svget4_s8(rhs_vals, 2));
                acc_v3 = svdot_s32(acc_v3, svget4_s8(lhs_vals, 3), svget4_s8(rhs_vals, 3));
            }

            int32_t *out_slot = &out_interleaved[(row * rhs_cols + col) * 4u];
            out_slot[0] = svaddv_s32(svptrue_b32(), acc_v0);
            out_slot[1] = svaddv_s32(svptrue_b32(), acc_v1);
            out_slot[2] = svaddv_s32(svptrue_b32(), acc_v2);
            out_slot[3] = svaddv_s32(svptrue_b32(), acc_v3);
        }
    }
}

/*
 * Dense two-learner GEMM with a compile-time-selected C32 or C8 sink.
 * Inputs remain interleaved bytes and both sinks use identical dot products.
 */
__attribute__((always_inline))
static inline void gemm_dense_2l_sink(
    const int8_t *lhs_interleaved,
    const int8_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved, int8_t *out_i8) {
    const uint32_t k_per_iter = (uint32_t)svcntb();

    /*
     * Two-stream dense variant. It follows the same row/column/K traversal as
     * the 4D helper, but svld2_s8 splits each interleaved pair and two
     * accumulators are reduced into the output cell.
     */
    for (uint32_t row = 0; row < lhs_rows; row++) {
        const int8_t *lhs_row = &lhs_interleaved[row * k_elems * 2u];

        for (uint32_t col = 0; col < rhs_cols; col++) {
            const int8_t *rhs_col = &rhs_by_col_interleaved[col * k_elems * 2u];
            svint32_t acc_v0 = svdup_s32(0);
            svint32_t acc_v1 = svdup_s32(0);

            /* Two-stream form of the same svdot_s32 traversal; see the
             * 4Learners kernel above for why the interleaved layout feeds
             * svdot_s32 without repacking. */
            for (uint32_t k = 0; k < k_elems; k += k_per_iter) {
                svbool_t pg = svwhilelt_b8((uint64_t)k, (uint64_t)k_elems);
                svint8x2_t lhs_vals = svld2_s8(pg, &lhs_row[k * 2u]);
                svint8x2_t rhs_vals = svld2_s8(pg, &rhs_col[k * 2u]);

                acc_v0 = svdot_s32(acc_v0, svget2_s8(lhs_vals, 0), svget2_s8(rhs_vals, 0));
                acc_v1 = svdot_s32(acc_v1, svget2_s8(lhs_vals, 1), svget2_s8(rhs_vals, 1));
            }

            const int32_t acc0 = svaddv_s32(svptrue_b32(), acc_v0);
            const int32_t acc1 = svaddv_s32(svptrue_b32(), acc_v1);
            const uint32_t offset = (row * rhs_cols + col) * 2u;
            if (out_i8 != NULL) {
                out_i8[offset] = (int8_t)(uint8_t)acc0;
                out_i8[offset + 1u] = (int8_t)(uint8_t)acc1;
            } else {
                out_interleaved[offset] = acc0;
                out_interleaved[offset + 1u] = acc1;
            }
        }
    }
}

void sve_gemm_dense_int8_interleaved_2Learners(
    const int8_t *lhs, const int8_t *rhs, uint32_t M, uint32_t N,
    uint32_t K, int32_t *output) {
    gemm_dense_2l_sink(lhs, rhs, M, N, K, output, NULL);
}

void sve_gemm_dense_int8_interleaved_2Learners_to_int8(
    const int8_t *lhs, const int8_t *rhs, uint32_t M, uint32_t N,
    uint32_t K, int8_t *output) {
    gemm_dense_2l_sink(lhs, rhs, M, N, K, NULL, output);
}
