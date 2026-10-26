#ifndef _GEMM_SVE_H_
#define _GEMM_SVE_H_

#include <stdint.h>

#include <gemm_exec.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief SVE row kernel for one int8 compact weight row and an input tile.
 *
 * This kernel computes one output column for @p seq_tile input rows. It expands
 * packed codebook indices from @p packed_row, gathers int32 weights from
 * @p codebook_i32, multiplies them by int8 input values, and accumulates int32
 * dot products.
 *
 * Step by step:
 * 1. Read packed uint32_t words from @p packed_row.
 * 2. Extract each bits_per_cb-wide codebook index.
 * 3. Gather the selected weight from @p codebook_i32.
 * 4. Multiply by the matching input element from @p in_mat.
 * 5. Horizontally reduce the SVE accumulator and store or add it to @p out_mat.
 *
 * @param packed_row Packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words available in @p packed_row.
 * @param k_elems Number of input/K elements to process from this row tile.
 * @param in_mat Base pointer to the input tile. Rows are separated by @p ld_in.
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in Leading dimension/row stride of @p in_mat, in int8 elements.
 * @param codebook_i32 Int32-expanded codebook weights indexed by packed values.
 * @param out_mat Base pointer to the output tile. Rows are separated by
 *                @p ld_out.
 * @param out_col Output column inside each output row to update.
 * @param ld_out Leading dimension/row stride of @p out_mat, in int32 elements.
 * @param bias_val Bias value for @p out_col.
 * @param add_bias Nonzero to add @p bias_val to the computed dot product.
 * @param accumulate Nonzero to add into the existing output slot; zero to
 *                   overwrite it.
 * @param bits_per_cb Number of bits used for each packed codebook index.
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
                               uint8_t bits_per_cb);

/**
 * @brief SVE row kernel for one FP32 compact weight row and an input tile.
 *
 * This is the FP32 version of sve_gemm_row_compact_int8(). It expands packed
 * indices, gathers FP32 weights from @p codebook, multiplies by FP32 inputs, and
 * writes one output column for each row in the sequence tile.
 *
 * @param packed_row Packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words available in @p packed_row.
 * @param k_elems Number of input/K elements to process from this row tile.
 * @param in_mat Base pointer to the input tile. Rows are separated by @p ld_in.
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in Leading dimension/row stride of @p in_mat, in float elements.
 * @param codebook FP32 codebook weights indexed by packed values.
 * @param out_mat Base pointer to the output tile. Rows are separated by
 *                @p ld_out.
 * @param out_col Output column inside each output row to update.
 * @param ld_out Leading dimension/row stride of @p out_mat, in float elements.
 * @param bias_val Bias value for @p out_col.
 * @param add_bias Nonzero to add @p bias_val to the computed dot product.
 * @param accumulate Nonzero to add into the existing output slot; zero to
 *                   overwrite it.
 * @param bits_per_cb Number of bits used for each packed codebook index.
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
                               uint8_t bits_per_cb);

/**
 * @brief SVE FP32 kernel for four interleaved learners with separate indices.
 *
 * Each learner has its own packed index stream. Input, codebook, bias, and
 * output values are interleaved in groups of four:
 * [learner0, learner1, learner2, learner3].
 *
 * @param packed_rows_interleaved Packed indices stored as
 *                                [n_words_row][4 learners].
 * @param n_words_row Number of packed uint32_t words per learner.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Input tile stored as [seq_tile][ld][4 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in float
 *                          elements, including interleaved learner lanes.
 * @param codebook_interleaved FP32 codebook stored as
 *                             [codebook_size][4 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Output tile stored with four learner values per
 *                            logical output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in float
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional bias values for this output column, stored
 *                         as [4 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
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
    uint8_t bits_per_cb);

/**
 * @brief SVE FP32 kernel for two interleaved learners sharing one index row.
 *
 * Both learners use the same packed codebook indices, but each learner reads
 * its own input, codebook, bias, and output lane.
 *
 * @param packed_row Shared packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words in @p packed_row.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Input tile stored as [seq_tile][ld][2 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in float
 *                          elements, including interleaved learner lanes.
 * @param codebook_interleaved FP32 codebook stored as
 *                             [codebook_size][2 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Output tile with two learner values per logical
 *                            output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in float
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional bias values for this output column, stored
 *                         as [2 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
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
    uint8_t bits_per_cb);

/**
 * @brief SVE FP32 kernel for four interleaved learners sharing one index row.
 *
 * All four learners use the same packed codebook indices. The four lanes still
 * keep separate input, codebook, bias, and output values.
 *
 * @param packed_row Shared packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words in @p packed_row.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Input tile stored as [seq_tile][ld][4 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in float
 *                          elements, including interleaved learner lanes.
 * @param codebook_interleaved FP32 codebook stored as
 *                             [codebook_size][4 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Output tile with four learner values per logical
 *                            output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in float
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional bias values for this output column, stored
 *                         as [4 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
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
    uint8_t bits_per_cb);

/**
 * @brief SVE int8/int32 kernel for four learners with separate indices.
 *
 * The input and codebook pointers use int32_t elements because this path works
 * on widened/interleaved int8 values that are consumed by 32-bit SVE lanes.
 * Outputs and biases are int32 accumulators.
 *
 * @param packed_rows_interleaved Packed indices stored as
 *                                [n_words_row][4 learners].
 * @param n_words_row Number of packed uint32_t words per learner.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Widened interleaved input tile stored as
 *                           [seq_tile][ld][4 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in int32
 *                          elements, including interleaved learner lanes.
 * @param codebook_i32_interleaved Int32 codebook stored as
 *                                 [codebook_size][4 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Int32 output tile with four learner values per
 *                            logical output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in int32
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional int32 bias values for this output column,
 *                         stored as [4 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq(
    const uint32_t *packed_rows_interleaved,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
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
    uint8_t bits_per_cb);

/**
 * @brief SVE int8/int32 kernel for two learners sharing one index row.
 *
 * Both learners use the same packed index stream. Inputs and codebook values are
 * widened to int32 lanes before multiplication/accumulation.
 *
 * @param packed_row Shared packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words in @p packed_row.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Widened input tile stored as
 *                           [seq_tile][ld][2 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in int32
 *                          elements, including interleaved learner lanes.
 * @param codebook_i32_interleaved Int32 codebook stored as
 *                                 [codebook_size][2 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Int32 output tile with two learner values per
 *                            logical output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in int32
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional int32 bias values for this output column,
 *                         stored as [2 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
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
    uint8_t bits_per_cb);

/**
 * @brief SVE int8/int32 kernel for four learners sharing one index row.
 *
 * All four learners use the same packed codebook-index stream. Each learner has
 * separate widened input, codebook, bias, and output lanes.
 *
 * @param packed_row Shared packed codebook-index words for one output row.
 * @param n_words_row Number of uint32_t words in @p packed_row.
 * @param k_elems Number of input/K elements to process.
 * @param in_mat_interleaved Widened input tile stored as
 *                           [seq_tile][ld][4 learners].
 * @param seq_tile Number of sequence rows processed by this call.
 * @param ld_in_interleaved Row stride of @p in_mat_interleaved, in int32
 *                          elements, including interleaved learner lanes.
 * @param codebook_i32_interleaved Int32 codebook stored as
 *                                 [codebook_size][4 learners].
 * @param codebook_size Number of codebook entries per learner.
 * @param out_mat_interleaved Int32 output tile with four learner values per
 *                            logical output.
 * @param out_col Logical output column to update.
 * @param ld_out_interleaved Row stride of @p out_mat_interleaved, in int32
 *                           elements, including interleaved learner lanes.
 * @param bias_interleaved Optional int32 bias values for this output column,
 *                         stored as [4 learners]. Pass NULL for zero bias.
 * @param add_bias Nonzero to add @p bias_interleaved when it is not NULL.
 * @param accumulate Nonzero to add into the existing output slots; zero to
 *                   overwrite them.
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void sve_gemm_row_compact_int8_interleaved_4Learners_same_seq(
    const uint32_t *packed_row,
    uint32_t n_words_row,
    uint32_t k_elems,
    const int32_t *in_mat_interleaved,
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
    uint8_t bits_per_cb);

/**
 * @brief SVE dense int8/int32 GEMM for four interleaved learners.
 *
 * This kernel multiplies a widened int32 left-hand matrix by a widened int32
 * right-hand matrix that is arranged by output column, and writes four
 * interleaved int32 outputs per logical result.
 *
 * @param lhs_interleaved Left-hand/input matrix stored as
 *                        [lhs_rows][k_elems][4 learners].
 * @param rhs_by_col_interleaved Right-hand/weight matrix stored by output
 *                               column as [rhs_cols][k_elems][4 learners].
 * @param lhs_rows Number of rows in the left-hand/input matrix.
 * @param rhs_cols Number of columns in the right-hand/weight matrix.
 * @param k_elems Shared K dimension to reduce over.
 * @param out_interleaved Output matrix stored as
 *                        [lhs_rows][rhs_cols][4 learners].
 */
void sve_gemm_dense_int8_interleaved_4Learners(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved);

/**
 * @brief SVE dense int8/int32 GEMM for two interleaved learners.
 *
 * This is the two-learner version of sve_gemm_dense_int8_interleaved_4Learners().
 *
 * @param lhs_interleaved Left-hand/input matrix stored as
 *                        [lhs_rows][k_elems][2 learners].
 * @param rhs_by_col_interleaved Right-hand/weight matrix stored by output
 *                               column as [rhs_cols][k_elems][2 learners].
 * @param lhs_rows Number of rows in the left-hand/input matrix.
 * @param rhs_cols Number of columns in the right-hand/weight matrix.
 * @param k_elems Shared K dimension to reduce over.
 * @param out_interleaved Output matrix stored as
 *                        [lhs_rows][rhs_cols][2 learners].
 */
void sve_gemm_dense_int8_interleaved_2Learners(
    const int32_t *lhs_interleaved,
    const int32_t *rhs_by_col_interleaved,
    uint32_t lhs_rows,
    uint32_t rhs_cols,
    uint32_t k_elems,
    int32_t *out_interleaved);

#ifdef __cplusplus
}
#endif

#endif
