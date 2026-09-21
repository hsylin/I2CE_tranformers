#ifndef _GEMM_EXEC_H_
#define _GEMM_EXEC_H_

#include <inttypes.h>
#include <stdint.h>

/**
 * @brief Shape metadata for one GEMM layer.
 *
 * The kernels in this header compute a matrix multiplication over a sequence
 * dimension:
 *
 *   out[seq, out_idx] = bias[out_idx] +
 *                       sum(in[seq, in_idx] * weight[out_idx, in_idx])
 *
 * For compact/codebook GEMM, the dense weight matrix is not stored directly.
 * Each weight is represented by a packed codebook index, and the actual weight
 * value is read from the codebook at execution time.
 */
typedef struct gemm_struct {
    /** Number of input rows/tokens/sequences processed by this GEMM call. */
    uint16_t seq_len;

    /** Number of input features in each sequence row; this is the K dimension. */
    uint16_t input_size;

    /** Number of output features/neurons; this is the number of weight rows. */
    uint16_t output_size;

    /**
     * Number of uint32_t words used to store the packed codebook indices for
     * one output row. Only compact/codebook kernels use this field.
     */
    uint16_t n_words_row;
} gemm_t;

/**
 * @brief Execute a dense FP32 GEMM without codebook compression.
 *
 * @param gemm_layer Layer dimensions: seq_len rows, input_size columns, and
 *                   output_size output columns.
 * @param in Row-major input matrix with shape [seq_len][input_size].
 * @param weights Row-major dense weight matrix with shape
 *                [output_size][input_size].
 * @param bias Optional bias vector with output_size elements. Pass NULL to use
 *             zero bias.
 * @param out Row-major output matrix with shape [seq_len][output_size].
 */
void gemm_exec_noCB(gemm_t gemm_layer,
                    const float *in,
                    const float *weights,
                    const float *bias,
                    float *out);

/**
 * @brief Execute a scalar FP32 GEMM using packed codebook weight indices.
 *
 * Each packed index selects one FP32 value from @p codebook. The selected value
 * is used as the weight for the corresponding input element.
 *
 * @param gemm_layer Layer dimensions and n_words_row for each packed weight row.
 * @param in Row-major input matrix with shape [seq_len][input_size].
 * @param weight_idx Packed uint32_t codebook-index rows with shape
 *                   [output_size][n_words_row].
 * @param codebook FP32 weight codebook. It contains 2^bits_per_cb entries.
 * @param bias Optional FP32 bias vector with output_size elements. Pass NULL for
 *             zero bias.
 * @param out Row-major output matrix with shape [seq_len][output_size].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact(gemm_t gemm_layer,
                       const float *in,
                       const uint32_t *weight_idx,
                       const float *codebook,
                       const float *bias,
                       float *out,
                       uint8_t bits_per_cb);

/**
 * @brief Execute two same-sequence FP32 compact GEMMs from interleaved buffers.
 *
 * "2-learner same_seq" means two learners/models share the same packed index row for
 * each output, but keep separate input values, codebook values, biases, and
 * output accumulators. Values are stored as pairs:
 * [learner0, learner1].
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input matrix with shape
 *                       [seq_len][input_size][2 learners].
 * @param weight_idx Shared packed codebook indices with shape
 *                   [output_size][n_words_row].
 * @param codebook_interleaved Codebook entries interleaved as
 *                             [codebook_index][2 learners].
 * @param bias_interleaved Optional bias values interleaved as
 *                         [output_size][2 learners]. Pass NULL for zero bias.
 * @param out_interleaved Output matrix with shape
 *                        [seq_len][output_size][2 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_fp32_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb);

/**
 * @brief Execute four same-sequence FP32 compact GEMMs from interleaved buffers.
 *
 * "4-learner same_seq" means four learners/models share one packed index row for each
 * output. Their input, codebook, bias, and output values are stored in groups of
 * four: [learner0, learner1, learner2, learner3].
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input matrix with shape
 *                       [seq_len][input_size][4 learners].
 * @param weight_idx Shared packed codebook indices with shape
 *                   [output_size][n_words_row].
 * @param codebook_interleaved Codebook entries interleaved as
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional bias values interleaved as
 *                         [output_size][4 learners]. Pass NULL for zero bias.
 * @param out_interleaved Output matrix with shape
 *                        [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_fp32_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb);

/**
 * @brief Execute four FP32 compact GEMMs with per-learner packed index rows.
 *
 * "4-learner diff_seq" stores four learners/models together, but each learner has its
 * own packed weight-index stream. This is used when the four learners may have
 * different compact weights while still sharing the same traversal.
 *
 * @param gemm_layer Layer dimensions and packed-row length per learner.
 * @param in_interleaved Input matrix with shape
 *                       [seq_len][input_size][4 learners].
 * @param weight_idx_interleaved Packed indices interleaved as
 *                               [output_size][n_words_row][4 learners].
 * @param codebook_interleaved Codebook entries interleaved as
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional bias values interleaved as
 *                         [output_size][4 learners]. Pass NULL for zero bias.
 * @param out_interleaved Output matrix with shape
 *                        [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_fp32_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                   const float *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const float *codebook_interleaved,
                                                   const float *bias_interleaved,
                                                   float *out_interleaved,
                                                   uint8_t bits_per_cb);

/**
 * @brief Execute a dense int8 GEMM with int32 accumulation.
 *
 * @param gemm_layer Layer dimensions: seq_len rows, input_size columns, and
 *                   output_size output columns.
 * @param in Row-major int8 input matrix with shape [seq_len][input_size].
 * @param weights Row-major int8 dense weight matrix with shape
 *                [output_size][input_size].
 * @param bias Optional int32 bias vector with output_size elements. Pass NULL
 *             to use zero bias.
 * @param out Row-major int32 output matrix with shape [seq_len][output_size].
 */
void gemm_exec_noCB_int(gemm_t gemm_layer,
                        const int8_t *in,
                        const int8_t *weights,
                        const int32_t *bias,
                        int32_t *out);

/**
 * @brief Execute a scalar int8 compact GEMM with int32 accumulation.
 *
 * Packed indices select int8 weights from @p codebook. Inputs and selected
 * weights are multiplied as int32 values, and the result is accumulated into an
 * int32 output.
 *
 * @param gemm_layer Layer dimensions and n_words_row for each packed weight row.
 * @param in Row-major int8 input matrix with shape [seq_len][input_size].
 * @param weight_idx Packed uint32_t codebook-index rows with shape
 *                   [output_size][n_words_row].
 * @param codebook Int8 weight codebook. It contains 2^bits_per_cb entries.
 * @param bias Optional int32 bias vector with output_size elements. Pass NULL
 *             for zero bias.
 * @param out Row-major int32 output matrix with shape [seq_len][output_size].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int(gemm_t gemm_layer,
                           const int8_t *in,
                           const uint32_t *weight_idx,
                           const int8_t *codebook,
                           const int32_t *bias,
                           int32_t *out,
                           uint8_t bits_per_cb);

/**
 * @brief Execute four int8 compact GEMMs with per-learner packed index rows.
 *
 * This is the int8/int32 version of the 4-learner diff_seq path. Each output slot
 * stores four int32 accumulators, one per learner.
 *
 * @param gemm_layer Layer dimensions and packed-row length per learner.
 * @param in_interleaved Input values stored as
 *                       [seq_len][input_size][4 learners].
 * @param weight_idx_interleaved Packed indices stored as
 *                               [output_size][n_words_row][4 learners].
 * @param codebook_interleaved Int8 codebook entries stored as
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional int32 bias values stored as
 *                         [output_size][4 learners]. Pass NULL for zero bias.
 * @param out_interleaved Int32 output stored as
 *                        [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb);

/**
 * @brief Execute two same-sequence int8 compact GEMMs from interleaved buffers.
 *
 * The two learners share the packed index row for each output, but use separate
 * interleaved input, codebook, bias, and output values.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input values stored as
 *                       [seq_len][input_size][2 learners].
 * @param weight_idx Shared packed indices stored as
 *                   [output_size][n_words_row].
 * @param codebook_interleaved Int8 codebook entries stored as
 *                             [codebook_index][2 learners].
 * @param bias_interleaved Optional int32 bias values stored as
 *                         [output_size][2 learners]. Pass NULL for zero bias.
 * @param out_interleaved Int32 output stored as
 *                        [seq_len][output_size][2 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb);

/**
 * @brief Execute four same-sequence int8 compact GEMMs from interleaved buffers.
 *
 * The four learners share the packed index row for each output, but use
 * separate interleaved input, codebook, bias, and output values.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input values stored as
 *                       [seq_len][input_size][4 learners].
 * @param weight_idx Shared packed indices stored as
 *                   [output_size][n_words_row].
 * @param codebook_interleaved Int8 codebook entries stored as
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional int32 bias values stored as
 *                         [output_size][4 learners]. Pass NULL for zero bias.
 * @param out_interleaved Int32 output stored as
 *                        [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb);

#ifdef SIMD
/**
 * @brief SVE-accelerated FP32 compact GEMM.
 *
 * Computes the same result as gemm_exec_compact(), but uses SVE row kernels and
 * may tile the sequence and packed-index dimensions for cache locality.
 *
 * @param gemm_layer Layer dimensions and n_words_row for each packed weight row.
 * @param in Row-major input matrix with shape [seq_len][input_size].
 * @param weight_idx Packed uint32_t codebook-index rows with shape
 *                   [output_size][n_words_row].
 * @param codebook FP32 weight codebook with 2^bits_per_cb entries.
 * @param bias Optional FP32 bias vector with output_size elements. Pass NULL for
 *             zero bias.
 * @param out Row-major output matrix with shape [seq_len][output_size].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_sve(gemm_t gemm_layer,
                           const float *in,
                           const uint32_t *weight_idx,
                           const float *codebook,
                           const float *bias,
                           float *out,
                           uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated two-learner FP32 same-sequence compact GEMM.
 *
 * Computes the same layout and result as
 * gemm_exec_compact_fp32_interleaved_2Learners_same_seq(), using SVE when the codebook
 * fits the selected SVE register-cache mode.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input matrix [seq_len][input_size][2 learners].
 * @param weight_idx Shared packed indices [output_size][n_words_row].
 * @param codebook_interleaved Codebook entries [codebook_index][2 learners].
 * @param bias_interleaved Optional bias [output_size][2 learners]. Pass NULL
 *                         for zero bias.
 * @param out_interleaved Output matrix [seq_len][output_size][2 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                        const float *in_interleaved,
                                                        const uint32_t *weight_idx,
                                                        const float *codebook_interleaved,
                                                        const float *bias_interleaved,
                                                        float *out_interleaved,
                                                        uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated four-learner FP32 same-sequence compact GEMM.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input matrix [seq_len][input_size][4 learners].
 * @param weight_idx Shared packed indices [output_size][n_words_row].
 * @param codebook_interleaved Codebook entries [codebook_index][4 learners].
 * @param bias_interleaved Optional bias [output_size][4 learners]. Pass NULL
 *                         for zero bias.
 * @param out_interleaved Output matrix [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                        const float *in_interleaved,
                                                        const uint32_t *weight_idx,
                                                        const float *codebook_interleaved,
                                                        const float *bias_interleaved,
                                                        float *out_interleaved,
                                                        uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated four-learner FP32 compact GEMM with separate indices.
 *
 * @param gemm_layer Layer dimensions and packed-row length per learner.
 * @param in_interleaved Input matrix [seq_len][input_size][4 learners].
 * @param weight_idx_interleaved Packed indices
 *                               [output_size][n_words_row][4 learners].
 * @param codebook_interleaved Codebook entries [codebook_index][4 learners].
 * @param bias_interleaved Optional bias [output_size][4 learners]. Pass NULL
 *                         for zero bias.
 * @param out_interleaved Output matrix [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                       const float *in_interleaved,
                                                       const uint32_t *weight_idx_interleaved,
                                                       const float *codebook_interleaved,
                                                       const float *bias_interleaved,
                                                       float *out_interleaved,
                                                       uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated int8 compact GEMM with int32 accumulation.
 *
 * Computes the same result as gemm_exec_compact_int().
 *
 * @param gemm_layer Layer dimensions and n_words_row for each packed weight row.
 * @param in Row-major int8 input matrix with shape [seq_len][input_size].
 * @param weight_idx Packed uint32_t codebook-index rows with shape
 *                   [output_size][n_words_row].
 * @param codebook Int8 weight codebook with 2^bits_per_cb entries.
 * @param bias Optional int32 bias vector with output_size elements. Pass NULL
 *             for zero bias.
 * @param out Row-major int32 output matrix with shape [seq_len][output_size].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated four-learner int8 compact GEMM with separate indices.
 *
 * @param gemm_layer Layer dimensions and packed-row length per learner.
 * @param in_interleaved Input values [seq_len][input_size][4 learners].
 * @param weight_idx_interleaved Packed indices
 *                               [output_size][n_words_row][4 learners].
 * @param codebook_interleaved Int8 codebook entries
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional int32 bias [output_size][4 learners]. Pass
 *                         NULL for zero bias.
 * @param out_interleaved Int32 output [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                       const int8_t *in_interleaved,
                                                       const uint32_t *weight_idx_interleaved,
                                                       const int8_t *codebook_interleaved,
                                                       const int32_t *bias_interleaved,
                                                       int32_t *out_interleaved,
                                                       uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated two-learner int8 same-sequence compact GEMM.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input values [seq_len][input_size][2 learners].
 * @param weight_idx Shared packed indices [output_size][n_words_row].
 * @param codebook_interleaved Int8 codebook entries
 *                             [codebook_index][2 learners].
 * @param bias_interleaved Optional int32 bias [output_size][2 learners]. Pass
 *                         NULL for zero bias.
 * @param out_interleaved Int32 output [seq_len][output_size][2 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                       const int8_t *in_interleaved,
                                                       const uint32_t *weight_idx,
                                                       const int8_t *codebook_interleaved,
                                                       const int32_t *bias_interleaved,
                                                       int32_t *out_interleaved,
                                                       uint8_t bits_per_cb);

/**
 * @brief SVE-accelerated four-learner int8 same-sequence compact GEMM.
 *
 * @param gemm_layer Layer dimensions and packed-row length.
 * @param in_interleaved Input values [seq_len][input_size][4 learners].
 * @param weight_idx Shared packed indices [output_size][n_words_row].
 * @param codebook_interleaved Int8 codebook entries
 *                             [codebook_index][4 learners].
 * @param bias_interleaved Optional int32 bias [output_size][4 learners]. Pass
 *                         NULL for zero bias.
 * @param out_interleaved Int32 output [seq_len][output_size][4 learners].
 * @param bits_per_cb Number of bits used for each packed codebook index.
 */
void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                       const int8_t *in_interleaved,
                                                       const uint32_t *weight_idx,
                                                       const int8_t *codebook_interleaved,
                                                       const int32_t *bias_interleaved,
                                                       int32_t *out_interleaved,
                                                       uint8_t bits_per_cb);

/* Tiled four-learner shared-index compact GEMM (see gemm_exec.c). */
void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb);

#endif

#endif
