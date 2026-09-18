/**
 * GEMM execution kernels for dense, compact/codebook, and SVE paths.
 *
 * This module is used by the generated TiC-SAT neural network inference code
 * to execute fully connected/projection layers. A GEMM layer is treated as a
 * sequence of input rows multiplied by a matrix of output rows:
 *
 *   out[seq, out_idx] = bias[out_idx] +
 *                       sum(in[seq, in_idx] * weight[out_idx, in_idx])
 *
 * It provides three related families of implementations:
 *
 * 1. Dense reference kernels for FP32 and int8 inputs. These read the complete
 *    weight matrix directly and are useful for uncompressed layers, testing,
 *    and correctness comparisons.
 *
 * 2. Compact/codebook kernels. These do not store every weight value directly.
 *    Instead, each weight position stores a packed integer index. At runtime the
 *    index selects one value from a small codebook, reducing the memory needed
 *    for weights while keeping the GEMM loop structure simple.
 *
 * 3. Interleaved multi-learner and optional SVE kernels. Interleaved variants
 *    compute two or four learners/models in one traversal by storing values as
 *    [item][learner]. When SIMD is enabled, the SVE wrappers use vector row
 *    kernels where possible and fall back to the scalar versions when a layout,
 *    codebook size, or bit width is not suitable for the selected SVE path.
 *
 * The code assumes row-major buffers and flat 1D arrays. The gemm_t metadata
 * supplies the sequence length, input dimension, output dimension, and compact
 * packed-row length.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <gemm_exec.h>
#ifdef SIMD
#include <codebooks_def.h>
#include <gemm_SVE.h>

/*
 * Optional command-line override for TILE_L1_SIZE, plumbed from
 * compile_transformer.sh (TILE_L1_SIZE_FLAG env var -> -DTILE_L1_SIZE_OVERRIDE).
 *
 * codebooks_def.h above defines TILE_L1_SIZE from the notebook-generated
 * configuration. When TILE_L1_SIZE_OVERRIDE is present it wins with an
 * explicit #undef/#define so command-line tile-size experiments do not
 * require regenerating weights. When unset, TILE_L1_SIZE keeps the
 * notebook value unchanged. TILE_L2_SIZE has no such override because
 * no codebook GEMM code path consumes it yet.
 */
#ifdef TILE_L1_SIZE_OVERRIDE
#undef TILE_L1_SIZE
#define TILE_L1_SIZE TILE_L1_SIZE_OVERRIDE
#endif
#endif

/**
 * Decode one codebook index from a packed weight row.
 *
 * Packed compact weights store multiple small indexes in each uint32_t word.
 * For example, with bits_per_cb == 4, each word contains eight indexes. The
 * least significant bits hold the first index in that word, then the next index,
 * and so on.
 *
 * Steps:
 * 1. Compute how many indexes fit in one 32-bit word.
 * 2. Find the word containing elem_idx.
 * 3. Shift the desired index down to bit 0.
 * 4. Mask away any neighboring packed indexes.
 */
static uint32_t get_packed_index(const uint32_t *packed_row,
                                 uint32_t elem_idx,
                                 uint8_t bits_per_cb) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;

    return (packed_row[word_idx] >> offset) & idx_mask;
}

/**
 * Decode one index for a fixed four-learner interleaved packed layout.
 *
 * The packed weight buffer is arranged as:
 *
 *   [word0 learner0][word0 learner1][word0 learner2][word0 learner3]
 *   [word1 learner0][word1 learner1][word1 learner2][word1 learner3] ...
 *
 * This helper selects the packed word for the requested learner, then applies
 * the same shift-and-mask logic used by get_packed_index().
 */
static uint32_t get_packed_index_interleaved_4d(
    const uint32_t *packed_rows_interleaved,
    uint32_t elem_idx,
    uint8_t bits_per_cb,
    uint32_t learner) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    uint32_t packed_word = packed_rows_interleaved[word_idx * 4u + learner];

    return (packed_word >> offset) & idx_mask;
}

/**
 * Decode one index from a generic N-learner interleaved packed layout.
 *
 * This is the same layout as get_packed_index_interleaved_4d(), but the learner
 * stride is supplied by the caller. It is used by code paths that want one
 * helper for two-learner and four-learner arrangements.
 */
static uint32_t get_packed_index_interleaved_nd(
    const uint32_t *packed_rows_interleaved,
    uint32_t elem_idx,
    uint8_t bits_per_cb,
    uint32_t learner,
    uint32_t learner_count) {
    uint32_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    uint32_t word_idx = elem_idx / idxs_per_word;
    uint32_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    uint32_t packed_word = packed_rows_interleaved[word_idx * learner_count + learner];

    return (packed_word >> offset) & idx_mask;
}

#ifdef SIMD
#ifndef TILE_L1_SIZE
#define TILE_L1_SIZE 0
#endif

/**
 * Return the configured L1 tile size, or the full dimension when tiling is off.
 *
 * TILE_L1_SIZE values of 0 or 1 mean "do not tile" for these wrappers. Values
 * larger than the dimension are also treated as the full dimension so the loops
 * below always make forward progress with a valid tile length.
 */
static uint32_t gemm_sve_l1_tile_or_full(uint32_t full_size) {
    const uint32_t tile_size = (uint32_t)TILE_L1_SIZE;
    if ((tile_size <= 1u) || (tile_size >= full_size)) {
        return full_size;
    }
    return tile_size;
}

/**
 * Report how many codebook entries can be kept in SVE registers per learner.
 *
 * The generated configuration selects one of the N_SVE_REG_CB_* modes. The row
 * kernels rely on this capacity when deciding whether a full codebook can be
 * loaded once and reused from registers.
 */
static uint32_t gemm_sve_codebook_capacity(void) {
#if defined(N_SVE_REG_CB_4)
    return N_SVE_LANES * 4u; /* Four SVE registers are available per learner codebook. */
#elif defined(N_SVE_REG_CB_2)
    return N_SVE_LANES * 2u; /* Two SVE registers are available per learner codebook. */
#elif defined(N_SVE_REG_CB_1)
    return N_SVE_LANES; /* One SVE register is available per learner codebook. */
#else
    return 0u; /* No codebook register-cache mode was selected. */
#endif
}

/**
 * Check whether the SVE compact kernels can cache the whole codebook.
 *
 * If the codebook is too large for the selected register-cache mode, the caller
 * intentionally falls back to the scalar implementation. This preserves
 * correctness without requiring a separate vector path for partial codebooks.
 */
static int gemm_sve_codebook_fits_registers(uint32_t codebook_size) {
    const uint32_t capacity = gemm_sve_codebook_capacity();
    return (capacity != 0u) && (codebook_size <= capacity);
}

/*
 * Shared prologue helpers used by every interleaved SVE compact GEMM wrapper.
 *
 * Every interleaved SVE wrapper opens with the same guard chain:
 *   1. Empty output shape       -> return.
 *   2. Degenerate compact input -> fill each output slot with bias/zero.
 *      (bits_per_cb == 0, or input_size == 0, or n_words_row == 0 all
 *       produce the same observable behavior: no K contribution.)
 *   3. Unsupported SVE case     -> dispatch to the corresponding scalar
 *                                  fallback wrapper.
 *
 * Historically each wrapper hand-inlined that chain, and the two "degenerate
 * compact input" branches drifted: the fp32 SVE wrappers used a combined
 * check while four of the int8 SVE wrappers used a split form. The helpers
 * below consolidate the logic; the observable behavior is unchanged in every
 * pre-existing case, and future edits to the guard chain apply once.
 *
 * Only step 3 is per-wrapper - each SVE wrapper still calls its own
 * hand-written scalar counterpart with the exact original argument list.
 */

/*
 * Bias/zero fill for the interleaved int32 codebook output layout.
 *
 * learner_count is the interleaved stride and must be positive. Called for
 * bits_per_cb == 0 or when the K axis is empty; identical observable
 * behavior to the pre-refactor per-wrapper split.
 */
static void gemm_fill_bias_or_zero_int_interleaved(
    gemm_t gemm_layer,
    int32_t *out_interleaved,
    const int32_t *bias_interleaved,
    uint32_t learner_count) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * learner_count];
            for (uint32_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] = (bias_interleaved == NULL)
                                        ? 0
                                        : bias_interleaved[out_idx * learner_count + learner];
            }
        }
    }
}

/*
 * FP32 counterpart to gemm_fill_bias_or_zero_int_interleaved. Kept as a
 * separate function to preserve the scalar 0.0f initializer (no implicit
 * int-to-float conversion) and to keep each wrapper's original type
 * signature.
 */
static void gemm_fill_bias_or_zero_fp32_interleaved(
    gemm_t gemm_layer,
    float *out_interleaved,
    const float *bias_interleaved,
    uint32_t learner_count) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * learner_count];
            for (uint32_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] = (bias_interleaved == NULL)
                                        ? 0.0f
                                        : bias_interleaved[out_idx * learner_count + learner];
            }
        }
    }
}

/*
 * True (non-zero) when the SVE compact path cannot execute this codebook and
 * the caller must delegate to its scalar counterpart. Combines the two
 * pre-existing checks: bits_per_cb > 8 (index width the SVE path never
 * supported) and !gemm_sve_codebook_fits_registers(1u << bits_per_cb) (the
 * cached-codebook precondition).
 */
static int gemm_sve_needs_scalar_fallback(uint8_t bits_per_cb) {
    if (bits_per_cb > 8u) {
        return 1;
    }
    const uint32_t codebook_size = 1u << bits_per_cb;
    if (!gemm_sve_codebook_fits_registers(codebook_size)) {
        return 1;
    }
    return 0;
}
#endif

/**
 * Execute the straightforward FP32 dense GEMM path.
 *
 * This is the uncompressed baseline: every output element is initialized with
 * its bias, then each input feature is multiplied by the corresponding dense
 * weight value and accumulated. The weight matrix is laid out as
 * [output_size][input_size].
 */
void gemm_exec_noCB(gemm_t gemm_layer,
                    const float *in,
                    const float *weights,
                    const float *bias,
                    float *out) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            float acc = bias == NULL ? 0.0f : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                acc += in[(seq * gemm_layer.input_size) + in_idx] *
                       weights[(out_idx * gemm_layer.input_size) + in_idx];
            }

            out[(seq * gemm_layer.output_size) + out_idx] = acc;
        }
    }
}

/**
 * Execute scalar FP32 GEMM with compact/codebook weights.
 *
 * The loop order is the same as gemm_exec_noCB(), but each weight is recovered
 * in two steps:
 * 1. Decode the packed codebook index for the current input position.
 * 2. Use that index to read the actual FP32 weight from codebook.
 */
void gemm_exec_compact(gemm_t gemm_layer,
                       const float *in,
                       const uint32_t *weight_idx,
                       const float *codebook,
                       const float *bias,
                       float *out,
                       uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float acc = bias == NULL ? 0.0f : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                float weight = codebook[cb_idx];
                acc += in[(seq * gemm_layer.input_size) + in_idx] * weight;
            }

            out[(seq * gemm_layer.output_size) + out_idx] = acc;
        }
    }
}


/**
 * Scalar FP32 compact GEMM for two interleaved learners sharing one index row.
 *
 * The input, codebook, bias, and output buffers are stored in pairs:
 * [learner0, learner1]. Both learners use the same packed codebook index for a
 * given (out_idx, in_idx), but each learner reads its own input and codebook
 * value. SVE wrappers use this as the correctness-preserving fallback.
 */
void gemm_exec_compact_fp32_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];
            float acc0 = (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 2u + 0u];
            float acc1 = (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 2u + 1u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 2u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                acc0 += in_vals[0] * codebook_interleaved[cb_idx * 2u + 0u];
                acc1 += in_vals[1] * codebook_interleaved[cb_idx * 2u + 1u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

/**
 * Scalar FP32 compact GEMM for four interleaved learners sharing one index row.
 *
 * This is the four-learner version of the same_seq layout. The packed weights
 * are shared across learners, while input, codebook, bias, and output values are
 * interleaved as [learner0, learner1, learner2, learner3].
 */
void gemm_exec_compact_fp32_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                    const float *in_interleaved,
                                                    const uint32_t *weight_idx,
                                                    const float *codebook_interleaved,
                                                    const float *bias_interleaved,
                                                    float *out_interleaved,
                                                    uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            float acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 4u + learner];
            }

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                for (uint32_t learner = 0; learner < 4u; learner++) {
                    acc[learner] += in_vals[learner] *
                                    codebook_interleaved[cb_idx * 4u + learner];
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = acc[learner];
            }
        }
    }
}

/**
 * Scalar FP32 compact GEMM for four interleaved learners with separate indexes.
 *
 * The diff_seq layout stores one packed index stream per learner:
 * [output][packed word][learner]. This allows each learner to use different
 * compact weights while still sharing the outer GEMM traversal.
 */
void gemm_exec_compact_fp32_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                   const float *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const float *codebook_interleaved,
                                                   const float *bias_interleaved,
                                                   float *out_interleaved,
                                                   uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_rows =
                &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
            float *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];
            float acc[4];
            for (uint32_t learner = 0; learner < 4u; learner++) {
                acc[learner] =
                    (bias_interleaved == NULL) ? 0.0f : bias_interleaved[out_idx * 4u + learner];
            }

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const float *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                for (uint32_t learner = 0; learner < 4u; learner++) {
                    uint32_t cb_idx =
                        get_packed_index_interleaved_nd(packed_rows, in_idx, bits_per_cb, learner, 4u);
                    acc[learner] += in_vals[learner] *
                                    codebook_interleaved[cb_idx * 4u + learner];
                }
            }

            for (uint32_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = acc[learner];
            }
        }
    }
}

/**
 * Execute dense int8 GEMM with int32 accumulation.
 *
 * Inputs and weights are int8_t, but each multiply is promoted to int32_t
 * before accumulation. The output and optional bias are int32_t so this path can
 * preserve a wider accumulation range than the input type.
 */
void gemm_exec_noCB_int(gemm_t gemm_layer,
                        const int8_t *in,
                        const int8_t *weights,
                        const int32_t *bias,
                        int32_t *out) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            int32_t acc = (bias == NULL) ? 0 : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                int32_t in_val = (int32_t)in[seq * gemm_layer.input_size + in_idx];
                int32_t w_val  = (int32_t)weights[out_idx * gemm_layer.input_size + in_idx];
                acc += in_val * w_val;
            }

            out[seq * gemm_layer.output_size + out_idx] = acc;
        }
    }
}

/**
 * Execute scalar int8 compact GEMM with int32 accumulation.
 *
 * This mirrors gemm_exec_compact(), except the selected codebook value is int8_t
 * and both operands are promoted to int32_t before the multiply-accumulate.
 */
void gemm_exec_compact_int(gemm_t gemm_layer,
                           const int8_t *in,
                           const uint32_t *weight_idx,
                           const int8_t *codebook,
                           const int32_t *bias,
                           int32_t *out,
                           uint8_t bits_per_cb) {
    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t acc = (bias == NULL) ? 0 : bias[out_idx];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);
                int32_t in_val = (int32_t)in[seq * gemm_layer.input_size + in_idx];
                int32_t w_val  = (int32_t)codebook[cb_idx];
                acc += in_val * w_val;
            }

            out[seq * gemm_layer.output_size + out_idx] = acc;
        }
    }
}

/**
 * Scalar int8 compact GEMM for four learners with separate packed indexes.
 *
 * Each output element stores four parallel int32 accumulators, one per learner,
 * so the caller can process four different compact weight streams in a single
 * traversal. This function is also the fallback for the matching SVE wrapper.
 */
void gemm_exec_compact_int_interleaved_4Learners_diff_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx_interleaved,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_rows =
                &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
            int32_t acc2 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
            int32_t acc3 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx0 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 0u);
                uint32_t cb_idx1 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 1u);
                uint32_t cb_idx2 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 2u);
                uint32_t cb_idx3 = get_packed_index_interleaved_4d(packed_rows, in_idx, bits_per_cb, 3u);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx0 * 4u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx1 * 4u + 1u];
                acc2 += (int32_t)in_vals[2] * (int32_t)codebook_interleaved[cb_idx2 * 4u + 2u];
                acc3 += (int32_t)in_vals[3] * (int32_t)codebook_interleaved[cb_idx3 * 4u + 3u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

/**
 * Scalar int8 compact GEMM for two interleaved learners sharing one index row.
 *
 * The two learners share each decoded codebook index, then use that index to
 * read learner-specific entries from an interleaved int8 codebook.
 */
void gemm_exec_compact_int_interleaved_2Learners_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 2u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 2u + 1u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 2u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx * 2u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx * 2u + 1u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
        }
    }
}

/**
 * Scalar int8 compact GEMM for four interleaved learners sharing one index row.
 *
 * This same_seq path is useful when the learners share the compact index stream
 * but still have separate interleaved inputs, codebook values, biases, and
 * output accumulators.
 */
void gemm_exec_compact_int_interleaved_4Learners_same_seq(gemm_t gemm_layer,
                                                   const int8_t *in_interleaved,
                                                   const uint32_t *weight_idx,
                                                   const int8_t *codebook_interleaved,
                                                   const int32_t *bias_interleaved,
                                                   int32_t *out_interleaved,
                                                   uint8_t bits_per_cb) {
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
        for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
            const uint32_t *packed_row =
                &weight_idx[out_idx * gemm_layer.n_words_row];
            int32_t *out_slot =
                &out_interleaved[((seq * gemm_layer.output_size) + out_idx) * 4u];

            int32_t acc0 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 0u];
            int32_t acc1 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 1u];
            int32_t acc2 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 2u];
            int32_t acc3 = (bias_interleaved == NULL) ? 0 : bias_interleaved[out_idx * 4u + 3u];

            for (uint32_t in_idx = 0; in_idx < gemm_layer.input_size; in_idx++) {
                const int8_t *in_vals =
                    &in_interleaved[((seq * gemm_layer.input_size) + in_idx) * 4u];
                uint32_t cb_idx = get_packed_index(packed_row, in_idx, bits_per_cb);

                acc0 += (int32_t)in_vals[0] * (int32_t)codebook_interleaved[cb_idx * 4u + 0u];
                acc1 += (int32_t)in_vals[1] * (int32_t)codebook_interleaved[cb_idx * 4u + 1u];
                acc2 += (int32_t)in_vals[2] * (int32_t)codebook_interleaved[cb_idx * 4u + 2u];
                acc3 += (int32_t)in_vals[3] * (int32_t)codebook_interleaved[cb_idx * 4u + 3u];
            }

            out_slot[0] = acc0;
            out_slot[1] = acc1;
            out_slot[2] = acc2;
            out_slot[3] = acc3;
        }
    }
}

#ifdef SIMD
/*
 * Common SVE wrapper pattern used below:
 * 1. Reject empty output shapes immediately.
 * 2. Handle degenerate compact layers by writing bias/zero output.
 * 3. Check whether the selected bit width and codebook size match the SVE row
 *    kernel requirements.
 * 4. Convert int8 buffers to int32 when the SVE kernel accumulates in int32.
 * 5. Iterate over output rows, sequence tiles, and packed-K-word tiles.
 * 6. Call the low-level SVE row kernel for each tile and accumulate partial K
 *    tiles into the same output locations.
 */
/**
 * Execute SVE-accelerated FP32 compact GEMM.
 *
 * This wrapper handles edge cases, breaks the work into optional L1-sized tiles,
 * and delegates the inner row computation to sve_gemm_row_compact_fp32().
 * processed_k tracks how many input features have already been consumed because
 * packed rows are tiled by uint32_t words, not directly by input elements.
 */
void gemm_exec_compact_sve(gemm_t gemm_layer,
                           const float *in,
                           const uint32_t *weight_idx,
                           const float *codebook,
                           const float *bias,
                           float *out,
                           uint8_t bits_per_cb) {
    /* Step 1: no sequence rows or no output rows means there is nothing to fill. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: if there is no valid compact K dimension, the matrix product term
     * disappears and every output element is just its bias or zero.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0.0f : bias[out_idx];
            }
        }
        return;
    }

    /*
     * Step 3: choose tile sizes. A TILE_L1_SIZE of 0/1 means "use the full
     * dimension"; otherwise these values bound the sequence and packed-word
     * tiles used by the loops below.
     */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    /*
     * Outer loop selects one output row. Inner tiles cover sequence rows and
     * packed weight words; the first K tile initializes from bias, later K tiles
     * accumulate into the same output slots.
     */
    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 4: select the packed weight row and scalar bias for this output. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float bias_val = (bias == NULL) ? 0.0f : bias[out_idx];

        /* Step 5: process a tile of sequence rows at a time. */
        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Clamp the final sequence tile so it stops exactly at seq_len. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /*
             * Step 6: walk across the packed K dimension by uint32_t words.
             * processed_k is the matching input-feature offset, because one
             * packed word expands to several codebook indexes.
             */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                /* Clamp the final packed-word tile. */
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /*
                 * Convert packed words to the maximum number of input features
                 * they can describe, then clamp for a partial final K tile.
                 */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 7: run the vector row kernel. The first K tile writes
                 * bias-initialized outputs; later K tiles add to those outputs.
                 */
                sve_gemm_row_compact_fp32(&packed_row[w0],
                                          tile_words,
                                          k_tile,
                                          &in[seq0 * gemm_layer.input_size +
                                              processed_k],
                                          seq_tile,
                                          gemm_layer.input_size,
                                          codebook,
                                          &out[seq0 * gemm_layer.output_size],
                                          out_idx,
                                          gemm_layer.output_size,
                                          bias_val,
                                          (w0 == 0u),
                                          (w0 != 0u),
                                          bits_per_cb);

                /* Step 8: advance the unpacked input-feature offset. */
                processed_k += k_tile;
            }
        }
    }
}

/**
 * SVE FP32 compact GEMM for two same-sequence interleaved learners.
 *
 * The SVE row kernel expects the whole codebook to fit in the configured
 * register-cache mode. If the bit width or codebook size is unsupported, this
 * wrapper calls the scalar 2-learner same_seq implementation with the same arguments.
 */
void gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no output shape, no work. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: no compact K contribution means each two-learner output pair is
     * copied from interleaved bias, or initialized to zero when bias is absent.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_fp32_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 2u);
        return;
    }

    /* Step 3: dispatch to the scalar same_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_fp32_interleaved_2Learners_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /* Step 4: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 6: select the shared packed row and the two bias values. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 2u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 7: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 8: tile the packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 9: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 10: call the SVE kernel. Input/output strides are doubled
                 * because each logical value is stored for two learners.
                 */
                sve_gemm_row_compact_fp32_interleaved_2Learners_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 2u],
                    seq_tile,
                    gemm_layer.input_size * 2u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 2u],
                    out_idx,
                    gemm_layer.output_size * 2u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 11: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }
}

/**
 * SVE FP32 compact GEMM for four same-sequence interleaved learners.
 *
 * This has the same control flow as the 2-learner same_seq wrapper, but all interleaved
 * strides are four learners wide and the row kernel consumes svld4-friendly
 * codebook/input layouts.
 */
void gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no output shape, no work. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: no compact K contribution means each four-learner output group is
     * copied from interleaved bias, or initialized to zero when bias is absent.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_fp32_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 4u);
        return;
    }

    /* Step 3: dispatch to the scalar same_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_fp32_interleaved_4Learners_same_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /* Step 4: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 6: select the shared packed row and the four bias values. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 7: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 8: tile the packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 9: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 10: call the SVE kernel. Input/output strides are four
                 * times wider because four learners are interleaved.
                 */
                sve_gemm_row_compact_fp32_interleaved_4Learners_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 11: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }
}

/**
 * SVE FP32 compact GEMM for four interleaved learners with separate indexes.
 *
 * The packed index rows are interleaved by learner, so each K-word tile starts
 * at &packed_rows[w0 * 4]. Apart from that layout difference, the tiling and
 * fallback policy matches the same_seq SVE wrappers.
 */
void gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq(
    gemm_t gemm_layer,
    const float *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const float *codebook_interleaved,
    const float *bias_interleaved,
    float *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no output shape, no work. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: no compact K contribution means each four-learner output group is
     * copied from interleaved bias, or initialized to zero when bias is absent.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_fp32_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 4u);
        return;
    }

    /* Step 3: dispatch to the scalar diff_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_fp32_interleaved_4Learners_diff_seq(gemm_layer,
                                                       in_interleaved,
                                                       weight_idx_interleaved,
                                                       codebook_interleaved,
                                                       bias_interleaved,
                                                       out_interleaved,
                                                       bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /* Step 4: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 6: select this output row's interleaved per-learner packed rows. */
        const uint32_t *packed_rows =
            &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
        const float *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 7: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 8: tile the packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 9: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 10: call the diff_seq SVE kernel. packed_rows advances by
                 * w0 * 4 because each packed word has four learner-specific words.
                 */
                sve_gemm_row_compact_fp32_interleaved_4Learners_diff_seq(
                    &packed_rows[w0 * 4u],
                    tile_words,
                    k_tile,
                    &in_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 11: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }
}

/**
 * Execute SVE-accelerated int8 compact GEMM with int32 accumulation.
 *
 * The SVE row kernel works with int32 codebook entries, so this wrapper expands
 * the int8 codebook once before running the tiled vector loop. Unsupported bit
 * widths fall back to the scalar int8 compact implementation.
 */
void gemm_exec_compact_int_sve(gemm_t gemm_layer,
                               const int8_t *in,
                               const uint32_t *weight_idx,
                               const int8_t *codebook,
                               const int32_t *bias,
                               int32_t *out,
                               uint8_t bits_per_cb) {
    /* Step 1: no sequence rows or no output rows means there is nothing to fill. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: a zero bit width cannot decode any compact weights, so the GEMM
     * contribution is empty and the output becomes bias/zero.
     */
    if (bits_per_cb == 0u) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0 : bias[out_idx];
            }
        }
        return;
    }

    /* Step 3: empty K dimensions also reduce to bias/zero output. */
    if ((gemm_layer.input_size == 0u) || (gemm_layer.n_words_row == 0u)) {
        for (uint32_t seq = 0; seq < gemm_layer.seq_len; seq++) {
            for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
                out[seq * gemm_layer.output_size + out_idx] =
                    (bias == NULL) ? 0 : bias[out_idx];
            }
        }
        return;
    }

    /* Step 4: wider indexes are valid for scalar decode but not this SVE path. */
    if (bits_per_cb > 8u) {
        gemm_exec_compact_int(gemm_layer,
                              in,
                              weight_idx,
                              codebook,
                              bias,
                              out,
                              bits_per_cb);
        return;
    }

    int32_t codebook_i32[256];
    const uint32_t codebook_size = 1u << bits_per_cb;
    /*
     * Step 5: expand the int8 codebook to int32 once. The SVE kernel can then
     * load codebook values in the same type it uses for accumulation.
     */
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        codebook_i32[cb_idx] = (int32_t)codebook[cb_idx];
    }

    /* Step 6: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 7: select this output row's packed weights and bias. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t bias_val = (bias == NULL) ? 0 : bias[out_idx];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 8: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 9: tile the packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 10: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 11: run the int8 SVE row kernel. The first K tile writes
                 * bias-initialized outputs; later K tiles accumulate into them.
                 */
                sve_gemm_row_compact_int8(&packed_row[w0],
                                          tile_words,
                                          k_tile,
                                          &in[seq0 * gemm_layer.input_size +
                                              processed_k],
                                          seq_tile,
                                          gemm_layer.input_size,
                                          codebook_i32,
                                          &out[seq0 * gemm_layer.output_size],
                                          out_idx,
                                          gemm_layer.output_size,
                                          bias_val,
                                          (w0 == 0u),
                                          (w0 != 0u),
                                          bits_per_cb);

                /* Step 12: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }
}

/**
 * SVE int8 compact GEMM for four learners with separate packed indexes.
 *
 * The input/output layout matches
 * gemm_exec_compact_int_interleaved_4Learners_diff_seq(); only the inner math changes.
 * The wrapper expands both the codebook and input from int8 to int32 because the
 * SVE row kernel operates on int32 lanes for accumulation.
 */
void gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no sequence rows or no output rows means there is nothing to fill. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: a zero bit width or empty K axis leaves no compact contribution,
     * so each four-learner output group becomes bias/zero.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_int_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 4u);
        return;
    }

    /* Step 3: dispatch to the scalar diff_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_int_interleaved_4Learners_diff_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx_interleaved,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /*
     * Step 4: expand the interleaved int8 codebook to int32 while preserving
     * [codebook index][learner], which matches the SVE load pattern.
     */
    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner]; /* Preserve [codebook index][learner] layout for svld4_s32. */
        }
    }

    /*
     * Step 7: allocate a temporary int32 copy of the interleaved input. This
     * avoids doing sign-extension repeatedly inside every output-row tile.
     */
    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 4u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        /* Step 8: allocation failure keeps correctness by using scalar fallback. */
        gemm_exec_compact_int_interleaved_4Learners_diff_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx_interleaved,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    /* Step 9: copy/sign-extend each interleaved input element to int32. */
    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    /* Step 10: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 11: select per-learner packed rows and four interleaved biases. */
        const uint32_t *packed_rows =
            &weight_idx_interleaved[(out_idx * gemm_layer.n_words_row) * 4u];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 12: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 13: tile packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 14: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 15: run the diff_seq SVE row kernel. packed_rows advances
                 * by w0 * 4 because every packed word has four learner words.
                 */
                sve_gemm_row_compact_int8_interleaved_4Learners_diff_seq(
                    &packed_rows[w0 * 4u],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 16: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }

    /* Step 17: release the temporary expanded input buffer. */
    free(input_i32_interleaved);
}

/**
 * SVE int8 compact GEMM for two same-sequence interleaved learners.
 *
 * The packed index row is shared by both learners. The codebook and input are
 * expanded to int32 interleaved buffers before entering the SVE row kernel, and
 * unsupported cases fall back to the scalar 2-learner same_seq implementation.
 */
void gemm_exec_compact_int_sve_interleaved_2Learners_same_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no sequence rows or no output rows means there is nothing to fill. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: a zero bit width or empty K axis leaves no compact contribution,
     * so each two-learner output pair becomes bias/zero.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_int_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 2u);
        return;
    }

    /* Step 3: dispatch to the scalar same_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_int_interleaved_2Learners_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /*
     * Step 4: expand the interleaved int8 codebook to int32 while preserving
     * [codebook index][learner], which matches the SVE load pattern.
     */
    int32_t codebook_i32_interleaved[2u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 2u; learner++) {
            codebook_i32_interleaved[cb_idx * 2u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 2u + learner]; /* Preserve [codebook index][learner] layout for svld2_s32. */
        }
    }

    /*
     * Step 7: allocate a temporary int32 copy of the interleaved input. This
     * avoids doing sign-extension repeatedly inside every output-row tile.
     */
    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 2u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        /* Step 8: allocation failure keeps correctness by using scalar fallback. */
        gemm_exec_compact_int_interleaved_2Learners_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    /* Step 9: copy/sign-extend each interleaved input element to int32. */
    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    /* Step 10: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 11: select the shared packed row and two interleaved biases. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 2u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 12: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 13: tile packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 14: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 15: run the same_seq SVE row kernel. Input/output strides
                 * are doubled because two learners are interleaved.
                 */
                sve_gemm_row_compact_int8_interleaved_2Learners_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 2u],
                    seq_tile,
                    gemm_layer.input_size * 2u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 2u],
                    out_idx,
                    gemm_layer.output_size * 2u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 16: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }

    /* Step 17: release the temporary expanded input buffer. */
    free(input_i32_interleaved);
}

/**
 * SVE int8 compact GEMM for four same-sequence interleaved learners.
 *
 * This is the four-learner counterpart to the 2-learner same_seq SVE wrapper. Learners
 * share packed indexes, while the codebook, input, bias, and output buffers stay
 * interleaved four values at a time.
 */
void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb) {
    /* Step 1: no sequence rows or no output rows means there is nothing to fill. */
    if ((gemm_layer.seq_len == 0u) || (gemm_layer.output_size == 0u)) {
        return;
    }

    /*
     * Step 2: a zero bit width or empty K axis leaves no compact contribution,
     * so each four-learner output group becomes bias/zero.
     */
    if ((bits_per_cb == 0u) || (gemm_layer.input_size == 0u) ||
        (gemm_layer.n_words_row == 0u)) {
        gemm_fill_bias_or_zero_int_interleaved(
            gemm_layer, out_interleaved, bias_interleaved, 4u);
        return;
    }

    /* Step 3: dispatch to the scalar same_seq path when SVE cannot execute. */
    if (gemm_sve_needs_scalar_fallback(bits_per_cb)) {
        gemm_exec_compact_int_interleaved_4Learners_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    const uint32_t codebook_size = 1u << bits_per_cb;
    /*
     * Step 4: expand the interleaved int8 codebook to int32 while preserving
     * [codebook index][learner], which matches the SVE load pattern.
     */
    int32_t codebook_i32_interleaved[4u * 256u] = {0};
    for (uint32_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
        for (uint32_t learner = 0; learner < 4u; learner++) {
            codebook_i32_interleaved[cb_idx * 4u + learner] =
                (int32_t)codebook_interleaved[cb_idx * 4u + learner]; /* Preserve [codebook index][learner] layout for svld4_s32. */
        }
    }

    /*
     * Step 7: allocate a temporary int32 copy of the interleaved input. This
     * avoids doing sign-extension repeatedly inside every output-row tile.
     */
    const uint32_t input_count =
        (uint32_t)gemm_layer.seq_len * (uint32_t)gemm_layer.input_size * 4u;
    int32_t *input_i32_interleaved =
        (int32_t *)malloc((size_t)input_count * sizeof(int32_t));
    if (input_i32_interleaved == NULL) {
        /* Step 8: allocation failure keeps correctness by using scalar fallback. */
        gemm_exec_compact_int_interleaved_4Learners_same_seq(gemm_layer,
                                                      in_interleaved,
                                                      weight_idx,
                                                      codebook_interleaved,
                                                      bias_interleaved,
                                                      out_interleaved,
                                                      bits_per_cb);
        return;
    }

    /* Step 9: copy/sign-extend each interleaved input element to int32. */
    for (uint32_t idx = 0; idx < input_count; idx++) {
        input_i32_interleaved[idx] = (int32_t)in_interleaved[idx];
    }

    /* Step 10: choose sequence and packed-word tile sizes for cache locality. */
    const uint32_t tile_seq = gemm_sve_l1_tile_or_full(gemm_layer.seq_len);
    const uint32_t tile_k_words = gemm_sve_l1_tile_or_full(gemm_layer.n_words_row);

    for (uint32_t out_idx = 0; out_idx < gemm_layer.output_size; out_idx++) {
        /* Step 11: select the shared packed row and four interleaved biases. */
        const uint32_t *packed_row =
            &weight_idx[out_idx * gemm_layer.n_words_row];
        const int32_t *bias_vals =
            (bias_interleaved == NULL) ? NULL : &bias_interleaved[out_idx * 4u];

        for (uint32_t seq0 = 0; seq0 < gemm_layer.seq_len; seq0 += tile_seq) {
            /* Step 12: clamp the final sequence tile. */
            const uint32_t seq_tile =
                ((seq0 + tile_seq) <= gemm_layer.seq_len)
                    ? tile_seq
                    : (gemm_layer.seq_len - seq0);

            /* Step 13: tile packed K words and track the unpacked K offset. */
            uint32_t processed_k = 0;
            for (uint32_t w0 = 0; w0 < gemm_layer.n_words_row;
                 w0 += tile_k_words) {
                const uint32_t tile_words =
                    ((w0 + tile_k_words) <= gemm_layer.n_words_row)
                        ? tile_k_words
                        : (gemm_layer.n_words_row - w0);
                /* Step 14: convert packed words to the number of input features. */
                const uint32_t max_k_in_tile = tile_words * (32u / bits_per_cb);
                const uint32_t k_tile =
                    ((processed_k + max_k_in_tile) <= gemm_layer.input_size)
                        ? max_k_in_tile
                        : (gemm_layer.input_size - processed_k);

                /*
                 * Step 15: run the same_seq SVE row kernel. Input/output strides
                 * are four times wider because four learners are interleaved.
                 */
                sve_gemm_row_compact_int8_interleaved_4Learners_same_seq(
                    &packed_row[w0],
                    tile_words,
                    k_tile,
                    &input_i32_interleaved[((seq0 * gemm_layer.input_size) + processed_k) * 4u],
                    seq_tile,
                    gemm_layer.input_size * 4u,
                    codebook_i32_interleaved,
                    codebook_size,
                    &out_interleaved[(seq0 * gemm_layer.output_size) * 4u],
                    out_idx,
                    gemm_layer.output_size * 4u,
                    bias_vals,
                    (w0 == 0u),
                    (w0 != 0u),
                    bits_per_cb);

                /* Step 16: move to the next unpacked input-feature range. */
                processed_k += k_tile;
            }
        }
    }

    /* Step 17: release the temporary expanded input buffer. */
    free(input_i32_interleaved);
}
#endif
