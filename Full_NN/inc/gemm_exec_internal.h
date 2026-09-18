/*
 * gemm_exec_internal.h — extended internal entry points for the compact
 * int8 SVE interleaved GEMM wrappers.
 *
 * The three public wrappers declared in gemm_exec.h keep their pre-refactor
 * signatures for backward compatibility. Callers that want to reuse a
 * caller-owned pre-widened int32 codebook buffer (avoiding the per-call
 * int8->int32 expansion that the public wrappers perform locally) call the
 * `_ex` variants declared here instead.
 *
 * Contract for the extra parameter:
 *   codebook_i32_interleaved_opt (may be NULL)
 *     - When non-NULL, the wrapper skips its local int8->int32 codebook
 *       expansion and reads this buffer directly.
 *     - When NULL, the wrapper falls back to the public-wrapper behavior
 *       and performs the local expansion into a stack buffer.
 *     - Ownership: the caller owns the buffer; the wrapper never frees it.
 *     - Capacity: must hold at least `learner_count * (1u << bits_per_cb)`
 *       int32 elements, laid out as [codebook_index][learner] to match the
 *       svld2_s32 / svld4_s32 load pattern the row kernels use.
 *     - Alignment: standard 4-byte int32 alignment; no special SVE-vector
 *       alignment is required (the row kernel uses masked loads).
 *     - Values: must equal (int32_t)codebook_interleaved[i] for every i.
 *
 * This header lives under Full_NN/inc/ so C++ callers inside
 * transformer_layers/ can include it, but it is not part of the public C
 * API surface of gemm_exec.h.
 */
#pragma once

#include "gemm_exec.h"

#ifdef SIMD
#ifdef __cplusplus
extern "C" {
#endif

void gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx_interleaved,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt);

void gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt);

void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // SIMD
