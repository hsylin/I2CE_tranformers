/*
 * gemm_exec_internal.h — extended internal entry points for the compact
 * int8 SVE interleaved GEMM wrappers.
 *
 * The three public wrappers declared in gemm_exec.h keep their pre-refactor
 * signatures for backward compatibility. Callers that want to reuse a
 * caller-owned pre-widened int32 codebook buffer, or to supply a
 * caller-owned scratch buffer for the int8->int32 activation widening
 * (both of which the public wrappers otherwise perform locally on every
 * call) call the `_ex` variants declared here instead.
 *
 * Contract for the extra parameters:
 *
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
 *   input_i32_workspace_opt / input_i32_workspace_capacity
 *     - When workspace_opt is non-NULL AND capacity is at least
 *       (gemm_layer.seq_len * gemm_layer.input_size * learner_count),
 *       the wrapper writes widened activations into this buffer and does NOT
 *       malloc. Input order remains [sequence][feature][learner].
 *     - Non-overlapping input/workspace ranges use a predicated SVE copy.
 *       Overlapping ranges retain the original forward scalar-copy behavior;
 *       no memmove-like guarantee for in-place expansion is added.
 *     - When workspace_opt is NULL, or capacity is insufficient, the
 *       wrapper falls back to malloc/free (public-wrapper behavior);
 *       its allocation-failure scalar fallback is preserved.
 *     - Ownership: the caller owns the buffer; the wrapper never frees it.
 *     - Capacity unit: number of int32 elements the buffer can hold.
 *     - Alignment: standard 4-byte int32 alignment.
 *     - Lifetime: only borrowed for the duration of the call; the wrapper
 *       does not retain the pointer.
 *     - Thread-safety: two threads calling the same _ex wrapper must
 *       supply distinct workspace buffers. Supplying one shared mutable
 *       buffer to concurrent calls is undefined behavior.
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
    const int32_t *codebook_i32_interleaved_opt,
    int32_t *input_i32_workspace_opt,
    uint32_t input_i32_workspace_capacity);

void gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt,
    int32_t *input_i32_workspace_opt,
    uint32_t input_i32_workspace_capacity);

void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt,
    int32_t *input_i32_workspace_opt,
    uint32_t input_i32_workspace_capacity);

void gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled_ex(
    gemm_t gemm_layer,
    const int8_t *in_interleaved,
    const uint32_t *weight_idx,
    const int8_t *codebook_interleaved,
    const int32_t *bias_interleaved,
    int32_t *out_interleaved,
    uint8_t bits_per_cb,
    const int32_t *codebook_i32_interleaved_opt,
    int32_t *input_i32_workspace_opt,
    uint32_t input_i32_workspace_capacity);


#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // SIMD
