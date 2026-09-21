#include "selfattention.h"
#include "memory.h"
#include <cmath>
#include <iostream>
#include <string>
#include <algorithm>

//#include <cstdint>
#include "debuggerFunctions.h"

#include "layerFactory.h"
#include "codebookDense.h"
#include "interleavedCodebookDenseValidator.h"
#include "interleavedPipeline.h"

#include <stdexcept>
#include <vector>

// Function signatures in this file:
// (requireInterleavedCodebookDense{2,4} moved to interleavedCodebookDenseValidator.h)
// void computeCodebookDenseInterleaved2Learners(const char* label, LinearLayer* const layers[2], std::size_t seq_len, const int8_t* input_interleaved, int8_t* output_interleaved);
// void computeCodebookDenseInterleaved4Learners(const char* label, LinearLayer* const layers[4], std::size_t seq_len, const int8_t* input_interleaved, int8_t* output_interleaved);
// SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t head_idx, std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, uint32_t** weightVector, std::size_t kernel_dim, std::size_t max_col, std::size_t learner_idx, std::string dump_dir);
// SingleHeadSelfAttn::~SingleHeadSelfAttn();
// void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t* input, uint32_t* output);
// template <std::size_t LearnerCount> void SingleHeadSelfAttn::computeGroupImpl(std::size_t seq_len, SingleHeadSelfAttn** heads, uint32_t* const* inputs, uint32_t* const* outputs);
// void SingleHeadSelfAttn::computeGroup2(std::size_t seq_len, SingleHeadSelfAttn* heads[2], uint32_t* const inputs[2], uint32_t* const outputs[2]);
// void SingleHeadSelfAttn::computeGroup4(std::size_t seq_len, SingleHeadSelfAttn* heads[4], uint32_t* const inputs[4], uint32_t* const outputs[4]);
// void SingleHeadSelfAttn::computeInterleaved2Learners(std::size_t seq_len, SingleHeadSelfAttn* heads[2], const int8_t* input_interleaved, int8_t* output_interleaved);
// void SingleHeadSelfAttn::computeInterleaved4Learners(std::size_t seq_len, SingleHeadSelfAttn* heads[4], const int8_t* input_interleaved, int8_t* output_interleaved);

namespace {

using transformer_internal::requireInterleavedCodebookDense2;
using transformer_internal::requireInterleavedCodebookDense4;

/**
 * @brief Run a two-learner interleaved CodebookDense projection and write int8 interleaved output.
 *
 * @param label Projection label used only for validation/error messages.
 * @param layers Two learner projection layers for the same Q, K, or V projection.
 * @param seq_len Number of tokens/rows to process for each learner.
 * @param input_interleaved Input activations in the 2-learner interleaved learner layout.
 * @param output_interleaved Destination buffer for the projected int8 activations in the same 2-learner interleaved layout.
 *
 * This wrapper keeps validation and execution together: first it checks the layers, then it delegates to the
 * CodebookDense implementation that knows how to consume and produce the interleaved int8 format.
 */
void computeCodebookDenseInterleaved2Learners(const char* label,
                                       LinearLayer* const layers[2],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense2(label, layers);
    primary->computeInterleaved2LearnersToInt8(seq_len, input_interleaved, output_interleaved);
}

/**
 * @brief Run a four-learner interleaved CodebookDense projection and write int8 interleaved output.
 *
 * @param label Projection label used only for validation/error messages.
 * @param layers Four learner projection layers for the same Q, K, or V projection.
 * @param seq_len Number of tokens/rows to process for each learner.
 * @param input_interleaved Input activations in the 4-learner interleaved learner layout.
 * @param output_interleaved Destination buffer for the projected int8 activations in the same 4-learner interleaved layout.
 *
 * The actual projection is implemented by CodebookDense. This function is a small checked entry point used by the
 * interleaved self-attention pipeline.
 */
void computeCodebookDenseInterleaved4Learners(const char* label,
                                       LinearLayer* const layers[4],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense4(label, layers);
    primary->computeInterleaved4LearnersToInt8(seq_len, input_interleaved, output_interleaved);
}

}

/* Flexible configuration of the number of heads and codebooked GEMM */

/**
 * @brief Construct one self-attention head and allocate all temporary buffers used by that head.
 *
 * @param head_idx Index of this attention head. It is used for naming Q/K/V layers and dump files.
 * @param pre_seq_len Maximum or preconfigured sequence length used to size the internal buffers.
 * @param input_dim Number of input features per token before the Q/K/V projections.
 * @param head_hidden_size Number of hidden features produced by this head for Q, K, and V.
 * @param weightVector Array of projection weight pointers. weightVector[0] is Q, [1] is K, and [2] is V.
 * @param kernel_dim Kernel/tile dimension used by the BWMA rearranged attention path.
 * @param max_col Maximum column count used by BWMA transpose/rearrangement helpers.
 * @param learner_idx Learner index for grouped or interleaved execution. It also helps select learner-specific layers.
 * @param dump_dir Optional directory where debug matrix dumps for this head should be written.
 *
 * The constructor creates the Q/K/V LinearLayer objects through LayerFactory, optionally creates dense reference
 * layers for correctness checks, creates the Softmax helper, and allocates packed uint32_t scratch buffers.
 */
SingleHeadSelfAttn::SingleHeadSelfAttn(std::size_t head_idx,
                                       std::size_t pre_seq_len,
                                       std::size_t input_dim,
                                       std::size_t head_hidden_size,
                                       uint32_t** weightVector,
                                       std::size_t kernel_dim,
                                       std::size_t max_col,
                                       std::size_t learner_idx,
                                       std::string dump_dir) {
    head_idx_ = head_idx;
    pre_seq_len_ = pre_seq_len;
    head_hidden_size_ = head_hidden_size;
    kernel_size_ = kernel_dim;
    max_col_ = max_col;
    input_dim_ = input_dim;
    learner_idx_ = learner_idx;
    dump_dir_ = dump_dir;

    const std::string q_name = "q_h" + std::to_string(head_idx_);
    const std::string k_name = "k_h" + std::to_string(head_idx_);
    const std::string v_name = "v_h" + std::to_string(head_idx_);

    auto q_bundle = LayerFactory::create(q_name, input_dim, head_hidden_size, weightVector[0], learner_idx_);
    auto k_bundle = LayerFactory::create(k_name, input_dim, head_hidden_size, weightVector[1], learner_idx_);
    auto v_bundle = LayerFactory::create(v_name, input_dim, head_hidden_size, weightVector[2], learner_idx_);

    query_layer_ = q_bundle.main;
    key_layer_ = k_bundle.main;
    value_layer_ = v_bundle.main;

#if CFG_USE_CODEBOOK_REFERENCE
    query_reference_ = q_bundle.reference;
    key_reference_ = k_bundle.reference;
    value_reference_ = v_bundle.reference;

    query_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    value_reference_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
#endif

    softmax_ = new Softmax();

    query_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    key_transposed_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    value_layer_out_ = new uint32_t[(pre_seq_len * head_hidden_size) >> 2]();
    attention_scores_ = new uint32_t[(pre_seq_len * pre_seq_len) >> 2]();
}

/**
 * @brief Release all heap-owned layers and scratch buffers allocated by the constructor.
 *
 * The object owns its Q/K/V projection layers, optional reference layers, Softmax helper, and temporary output buffers.
 * All of those resources are deleted here.
 */
SingleHeadSelfAttn::~SingleHeadSelfAttn() {
    delete[] query_layer_out_;
    delete[] key_layer_out_;
    delete[] key_transposed_layer_out_;
    delete[] value_layer_out_;
    delete[] attention_scores_;

#if CFG_USE_CODEBOOK_REFERENCE
    delete[] query_reference_out_;
    delete[] key_reference_out_;
    delete[] value_reference_out_;

    delete query_reference_;
    delete key_reference_;
    delete value_reference_;
#endif

    delete query_layer_;
    delete key_layer_;
    delete value_layer_;
    delete softmax_;
}

/**
 * @brief Compute self-attention for a single learner and a single attention head.
 *
 * @param seq_len Actual sequence length to process for this invocation.
 * @param input Packed uint32_t input activation matrix with shape seq_len x input_dim_.
 * @param output Packed uint32_t output activation matrix with shape seq_len x head_hidden_size_.
 *
 * High-level steps:
 * 1. Project input tokens into Q, K, and V using the head-specific linear layers.
 * 2. Compute attention scores Q * K^T using either RWMA or BWMA kernels, depending on compile-time flags.
 * 3. Apply softmax to the attention scores.
 * 4. Multiply softmax scores by V to produce the head output.
 * 5. Apply post-softmax scaling and optionally dump intermediate matrices for debugging.
 */
void SingleHeadSelfAttn::compute(std::size_t seq_len, uint32_t* input, uint32_t* output) {

    // Debug for head 0
    // static bool dumped_input_h0 = false;
    // static bool dumped_q_h0_outputs = false;

    // const std::size_t rows_to_dump = std::min<std::size_t>(seq_len, 2);

    // if (head_idx_ == 0 && !dumped_input_h0) {
    //     dumped_input_h0 = true;

    //     std::size_t rows_to_dump = std::min<std::size_t>(seq_len, 2);

    //     std::cout << "\n===== DEBUG self attention input for head 0 =====\n";
    //     std::cout << "seq_len = " << seq_len << "\n";
    //     std::cout << "input_dim = " << input_dim_ << "\n";
    //     printPackedTensorAsPythonList("input_matrix_test", input, rows_to_dump, input_dim_);
    //     std::cout << "===== END DEBUG =====\n\n";
    // }

    // Step 1: build the head-specific query, key, and value matrices from the packed input activations.
    query_layer_->compute(seq_len, input, query_layer_out_);
    key_layer_->compute(seq_len, input, key_layer_out_);
    value_layer_->compute(seq_len, input, value_layer_out_);

    // Optional debug dumps let Python/reference tooling inspect each projection independently.
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "q_h" + std::to_string(head_idx_) + ".txt",
        query_layer_out_,
        seq_len,
        head_hidden_size_);
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "k_h" + std::to_string(head_idx_) + ".txt",
        key_layer_out_,
        seq_len,
        head_hidden_size_);
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "v_h" + std::to_string(head_idx_) + ".txt",
        value_layer_out_,
        seq_len,
        head_hidden_size_);

#if CFG_USE_CODEBOOK_REFERENCE
    std::cout << "[DEBUG] CFG_USE_CODEBOOK_REFERENCE active in SingleHeadSelfAttn" << std::endl;

    // Recompute Q/K/V with dense reference layers and compare against the active implementation.
    std::fill(query_reference_out_, query_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(key_reference_out_, key_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);
    std::fill(value_reference_out_, value_reference_out_ + ((seq_len * head_hidden_size_) >> 2), 0u);

    query_reference_->compute(seq_len, input, query_reference_out_);
    key_reference_->compute(seq_len, input, key_reference_out_);
    value_reference_->compute(seq_len, input, value_reference_out_);

    // Print q_h0 outputs only once, after both paths have been computed.
    // if (head_idx_ == 0 && !dumped_q_h0_outputs) {
    //     dumped_q_h0_outputs = true;

    //     std::cout << "\n===== DEBUG q_h0 outputs =====\n";
    //     printPackedTensorAsPythonList("q_h0_cpp_codebook", query_layer_out_, rows_to_dump, head_hidden_size_);
    //     printPackedTensorAsPythonList("q_h0_cpp_dense_ref", query_reference_out_, rows_to_dump, head_hidden_size_);
    //     std::cout << "===== END q_h0 OUTPUT DEBUG =====\n\n";
    // }

    comparePackedBuffers(("q_h" + std::to_string(head_idx_)).c_str(),
                         query_reference_out_, query_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);

    comparePackedBuffers(("k_h" + std::to_string(head_idx_)).c_str(),
                         key_reference_out_, key_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);

    comparePackedBuffers(("v_h" + std::to_string(head_idx_)).c_str(),
                         value_reference_out_, value_layer_out_,
                         (seq_len * head_hidden_size_) >> 2);
#endif

#ifndef BWMA
    std::cout << "RWMA method" << std::endl;

    // RWMA path: transpose K into column-major packed form, then compute Q * K^T.
    Transpose::transpose(key_layer_out_, key_transposed_layer_out_,
                         head_hidden_size_, pre_seq_len_);

#ifdef SIMD
    simdComputeRWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                    head_hidden_size_, seq_len);
#else
    smmComputeRWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                   head_hidden_size_, seq_len);
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "qk_scores_h" + std::to_string(head_idx_) + ".txt",
        attention_scores_,
        seq_len,
        seq_len);

    softmax_->compute(attention_scores_, seq_len);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "softmax_qk_h" + std::to_string(head_idx_) + ".txt",
        attention_scores_,
        seq_len,
        seq_len);

#ifdef SIMD
    simdComputeRWMA(seq_len, attention_scores_, output, value_layer_out_,
                    seq_len, head_hidden_size_);
#else
    smmComputeRWMA(seq_len, attention_scores_, output, value_layer_out_,
                   seq_len, head_hidden_size_);
#endif

#else
    std::cout << "BWMA method" << std::endl;

    // BWMA path: rearrange K for tiled/kernel-aware access, then compute Q * K^T.
    Transpose::transpose_rearranged(key_layer_out_, key_transposed_layer_out_,
                                    head_hidden_size_, pre_seq_len_, kernel_size_, max_col_);

#ifdef SIMD
    simdComputeBWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                    head_hidden_size_, seq_len);
#else
    smmComputeBWMA(seq_len, query_layer_out_, attention_scores_, key_transposed_layer_out_,
                   head_hidden_size_, seq_len);
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "qk_scores_h" + std::to_string(head_idx_) + ".txt",
        attention_scores_,
        seq_len,
        seq_len);

    softmax_->computeRearranged(attention_scores_, seq_len, kernel_size_);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "softmax_qk_h" + std::to_string(head_idx_) + ".txt",
        attention_scores_,
        seq_len,
        seq_len);

#ifdef SIMD
    simdComputeBWMA(seq_len, attention_scores_, output, value_layer_out_,
                    seq_len, head_hidden_size_);
#else
    smmComputeBWMA(seq_len, attention_scores_, output, value_layer_out_,
                   seq_len, head_hidden_size_);
#endif
#endif

    // Store the raw attention-weighted V result before final post-softmax scaling.
    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "softmax_v_pre_post_h" + std::to_string(head_idx_) + ".txt",
        output,
        seq_len,
        head_hidden_size_);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "head_out_h" + std::to_string(head_idx_) + ".txt",
        output,
        seq_len,
        head_hidden_size_);

    softmax_->post_softmax(output, seq_len, head_hidden_size_);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "head_out_post_h" + std::to_string(head_idx_) + ".txt",
        output,
        seq_len,
        head_hidden_size_);
}

/**
 * @brief Shared grouped self-attention implementation for two or four learners.
 *
 * @tparam LearnerCount Number of learners processed together. Only 2 and 4 are supported.
 * @param seq_len Actual sequence length to process for every learner in the group.
 * @param heads Array of learner-specific SingleHeadSelfAttn objects. Each element owns its own buffers/layers.
 * @param inputs Array of packed uint32_t input activation matrices, one per learner.
 * @param outputs Array of packed uint32_t output activation matrices, one per learner.
 *
 * This function groups learners at the Q/K/V projection stage. It first tries to compute each projection with a
 * fused CodebookDense grouped kernel. If a grouped kernel is unavailable, it falls back to the normal per-learner
 * LinearLayer::compute() path. After projections are ready, each learner runs the same attention math as compute().
 */
template <std::size_t LearnerCount>
void SingleHeadSelfAttn::computeGroupImpl(std::size_t seq_len,
                                          SingleHeadSelfAttn** heads,
                                          uint32_t* const* inputs,
                                          uint32_t* const* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped self-attention is supported");

    LinearLayer* query_layers[LearnerCount];
    LinearLayer* key_layers[LearnerCount];
    LinearLayer* value_layers[LearnerCount];
    uint32_t* query_outputs[LearnerCount];
    uint32_t* key_outputs[LearnerCount];
    uint32_t* value_outputs[LearnerCount];

    // Collect the per-learner layer pointers and output buffers into parallel arrays for the grouped helpers.
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        query_layers[learner] = heads[learner]->query_layer_;
        key_layers[learner] = heads[learner]->key_layer_;
        value_layers[learner] = heads[learner]->value_layer_;
        query_outputs[learner] = heads[learner]->query_layer_out_;
        key_outputs[learner] = heads[learner]->key_layer_out_;
        value_outputs[learner] = heads[learner]->value_layer_out_;
    }

    // Select the matching grouped CodebookDense helper at compile time based on LearnerCount.
    auto tryGroupedCodebookDense = [&](LinearLayer* const layers[LearnerCount],
                                       uint32_t* const dense_outputs[LearnerCount]) {
        if constexpr (LearnerCount == 2u) {
            return tryComputeGroupedCodebookDense2(layers, seq_len, inputs, dense_outputs);
        } else {
            return tryComputeGroupedCodebookDense4(layers, seq_len, inputs, dense_outputs);
        }
    };

    // Try fused grouped Q/K/V projection first; fall back to normal per-learner projection if needed.
    if (!tryGroupedCodebookDense(query_layers, query_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->query_layer_->compute(seq_len, inputs[learner], query_outputs[learner]);
        }
    }
    if (!tryGroupedCodebookDense(key_layers, key_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->key_layer_->compute(seq_len, inputs[learner], key_outputs[learner]);
        }
    }
    if (!tryGroupedCodebookDense(value_layers, value_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner]->value_layer_->compute(seq_len, inputs[learner], value_outputs[learner]);
        }
    }

    // Once Q/K/V are available, each learner finishes its own attention score, softmax, and output path.
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        SingleHeadSelfAttn* self = heads[learner];

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "q_h" + std::to_string(self->head_idx_) + ".txt",
            self->query_layer_out_,
            seq_len,
            self->head_hidden_size_);
        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "k_h" + std::to_string(self->head_idx_) + ".txt",
            self->key_layer_out_,
            seq_len,
            self->head_hidden_size_);
        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "v_h" + std::to_string(self->head_idx_) + ".txt",
            self->value_layer_out_,
            seq_len,
            self->head_hidden_size_);

#if CFG_USE_CODEBOOK_REFERENCE
        // Reference mode validates each learner projection against the dense reference implementation.
        std::fill(self->query_reference_out_,
                  self->query_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);
        std::fill(self->key_reference_out_,
                  self->key_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);
        std::fill(self->value_reference_out_,
                  self->value_reference_out_ + ((seq_len * self->head_hidden_size_) >> 2),
                  0u);

        self->query_reference_->compute(seq_len, inputs[learner], self->query_reference_out_);
        self->key_reference_->compute(seq_len, inputs[learner], self->key_reference_out_);
        self->value_reference_->compute(seq_len, inputs[learner], self->value_reference_out_);

        const std::string q_name =
            "q_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);
        const std::string k_name =
            "k_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);
        const std::string v_name =
            "v_h" + std::to_string(self->head_idx_) + "_learner" + std::to_string(self->learner_idx_);

        comparePackedBuffers(q_name.c_str(),
                             self->query_reference_out_, self->query_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
        comparePackedBuffers(k_name.c_str(),
                             self->key_reference_out_, self->key_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
        comparePackedBuffers(v_name.c_str(),
                             self->value_reference_out_, self->value_layer_out_,
                             (seq_len * self->head_hidden_size_) >> 2);
#endif

#ifndef BWMA
        std::cout << "RWMA method" << std::endl;

        // RWMA attention: compute softmax(Q * K^T) and then multiply by V for this learner.
        Transpose::transpose(self->key_layer_out_,
                             self->key_transposed_layer_out_,
                             self->head_hidden_size_,
                             self->pre_seq_len_);

#ifdef SIMD
        simdComputeRWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                        self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#else
        smmComputeRWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                       self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#endif

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "qk_scores_h" + std::to_string(self->head_idx_) + ".txt",
            self->attention_scores_,
            seq_len,
            seq_len);

        self->softmax_->compute(self->attention_scores_, seq_len);

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "softmax_qk_h" + std::to_string(self->head_idx_) + ".txt",
            self->attention_scores_,
            seq_len,
            seq_len);

#ifdef SIMD
        simdComputeRWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                        seq_len, self->head_hidden_size_);
#else
        smmComputeRWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                       seq_len, self->head_hidden_size_);
#endif

#else
        std::cout << "BWMA method" << std::endl;

        // BWMA attention uses a rearranged key matrix layout tuned for the BWMA kernels.
        Transpose::transpose_rearranged(self->key_layer_out_, self->key_transposed_layer_out_,
                                        self->head_hidden_size_, self->pre_seq_len_,
                                        self->kernel_size_, self->max_col_);

#ifdef SIMD
        simdComputeBWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                        self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#else
        smmComputeBWMA(seq_len, self->query_layer_out_, self->attention_scores_,
                       self->key_transposed_layer_out_, self->head_hidden_size_, seq_len);
#endif

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "qk_scores_h" + std::to_string(self->head_idx_) + ".txt",
            self->attention_scores_,
            seq_len,
            seq_len);

        self->softmax_->computeRearranged(self->attention_scores_, seq_len, self->kernel_size_);

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "softmax_qk_h" + std::to_string(self->head_idx_) + ".txt",
            self->attention_scores_,
            seq_len,
            seq_len);

#ifdef SIMD
        simdComputeBWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                        seq_len, self->head_hidden_size_);
#else
        smmComputeBWMA(seq_len, self->attention_scores_, outputs[learner], self->value_layer_out_,
                       seq_len, self->head_hidden_size_);
#endif
#endif

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "softmax_v_pre_post_h" + std::to_string(self->head_idx_) + ".txt",
            outputs[learner],
            seq_len,
            self->head_hidden_size_);

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "head_out_h" + std::to_string(self->head_idx_) + ".txt",
            outputs[learner],
            seq_len,
            self->head_hidden_size_);

        self->softmax_->post_softmax(outputs[learner], seq_len, self->head_hidden_size_);

        dumpPackedMatrixIfEnabled(
            self->dump_dir_,
            "head_out_post_h" + std::to_string(self->head_idx_) + ".txt",
            outputs[learner],
            seq_len,
            self->head_hidden_size_);
    }
}

/**
 * @brief Compute grouped self-attention for exactly two learners.
 *
 * @param seq_len Actual sequence length to process for both learners.
 * @param heads Two SingleHeadSelfAttn objects that represent the same head across two learners.
 * @param inputs Two packed uint32_t input activation matrices, one per learner.
 * @param outputs Two packed uint32_t output activation matrices, one per learner.
 *
 * This is the public two-learner wrapper around computeGroupImpl().
 */
void SingleHeadSelfAttn::computeGroup2(std::size_t seq_len,
                                       SingleHeadSelfAttn* heads[2],
                                       uint32_t* const inputs[2],
                                       uint32_t* const outputs[2]) {
    computeGroupImpl<2u>(seq_len, heads, inputs, outputs);
}

/**
 * @brief Compute grouped self-attention for exactly four learners.
 *
 * @param seq_len Actual sequence length to process for all four learners.
 * @param heads Four SingleHeadSelfAttn objects that represent the same head across four learners.
 * @param inputs Four packed uint32_t input activation matrices, one per learner.
 * @param outputs Four packed uint32_t output activation matrices, one per learner.
 *
 * This is the public four-learner wrapper around computeGroupImpl().
 */
void SingleHeadSelfAttn::computeGroup4(std::size_t seq_len,
                                       SingleHeadSelfAttn* heads[4],
                                       uint32_t* const inputs[4],
                                       uint32_t* const outputs[4]) {
    computeGroupImpl<4u>(seq_len, heads, inputs, outputs);
}

/**
 * @brief Compute self-attention for two learners using fully interleaved int8 buffers.
 *
 * @param seq_len Actual sequence length to process for each learner.
 * @param heads Two SingleHeadSelfAttn objects for the same attention head, one per learner.
 * @param input_interleaved Input activations laid out in the 2-learner interleaved int8 learner format.
 * @param output_interleaved Destination buffer for the final head output in the same 2-learner interleaved int8 format.
 *
 * Step-by-step:
 * 1. Gather Q/K/V layers and dump directories from both learner heads.
 * 2. Compute interleaved int8 Q, K, and V projections with CodebookDense.
 * 3. Compute attention scores Q * K^T in interleaved layout.
 * 4. Apply interleaved softmax to the score matrix.
 * 5. Transpose V into column layout, multiply scores by V, and post-scale the output.
 */
void SingleHeadSelfAttn::computeInterleaved2Learners(std::size_t seq_len,
                                              SingleHeadSelfAttn* heads[2],
                                              const int8_t* input_interleaved,
                                              int8_t* output_interleaved) {
    LinearLayer* query_layers[2];
    LinearLayer* key_layers[2];
    LinearLayer* value_layers[2];
    std::string dump_dirs[2];

    // Extract per-learner projection layers and dump directories into arrays used by the interleaved helpers.
    for (std::size_t learner = 0; learner < 2u; learner++) {
        query_layers[learner] = heads[learner]->query_layer_;
        key_layers[learner] = heads[learner]->key_layer_;
        value_layers[learner] = heads[learner]->value_layer_;
        dump_dirs[learner] = heads[learner]->dump_dir_;
    }

    const std::size_t head_hidden_size = heads[0]->head_hidden_size_;
    std::vector<int8_t> query_out(seq_len * head_hidden_size * 2u, 0);
    std::vector<int8_t> key_out(seq_len * head_hidden_size * 2u, 0);
    std::vector<int8_t> value_out(seq_len * head_hidden_size * 2u, 0);

    // Compute Q/K/V projections with interleaved CodebookDense. Each output keeps both learners interleaved.
    computeCodebookDenseInterleaved2Learners("q_h", query_layers, seq_len, input_interleaved, query_out.data());
    computeCodebookDenseInterleaved2Learners("k_h", key_layers, seq_len, input_interleaved, key_out.data());
    computeCodebookDenseInterleaved2Learners("v_h", value_layers, seq_len, input_interleaved, value_out.data());

#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<uint32_t> packed_input((seq_len * heads[0]->input_dim_) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * head_hidden_size) >> 2, 0u);

    // Convert one learner at a time back to packed uint32_t so the dense reference path can compare results.
    auto compareProjection = [&](const char* prefix,
                                 LinearLayer* reference_layer,
                                 uint32_t* reference_output,
                                 const int8_t* candidate_interleaved,
                                 std::size_t learner) {
        std::fill(reference_output,
                  reference_output + ((seq_len * head_hidden_size) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner2(
            seq_len,
            heads[learner]->input_dim_,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner2(
            seq_len,
            head_hidden_size,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        reference_layer->compute(seq_len, packed_input.data(), reference_output);

        const std::string label =
            std::string(prefix) + "_h" + std::to_string(heads[0]->head_idx_) +
            "_learner" + std::to_string(heads[learner]->learner_idx_);
        comparePackedBuffers(
            label.c_str(),
            reference_output,
            packed_candidate.data(),
            (seq_len * head_hidden_size) >> 2);
    };

    for (std::size_t learner = 0; learner < 2u; learner++) {
        compareProjection("q", heads[learner]->query_reference_,
                          heads[learner]->query_reference_out_,
                          query_out.data(), learner);
        compareProjection("k", heads[learner]->key_reference_,
                          heads[learner]->key_reference_out_,
                          key_out.data(), learner);
        compareProjection("v", heads[learner]->value_reference_,
                          heads[learner]->value_reference_out_,
                          value_out.data(), learner);
    }
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "q_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        query_out.data(),
        seq_len,
        head_hidden_size);
    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "k_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        key_out.data(),
        seq_len,
        head_hidden_size);
    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "v_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        value_out.data(),
        seq_len,
        head_hidden_size);

    // Attention score matrix for two learners: each learner gets a seq_len x seq_len score matrix.
    std::vector<int8_t> attention_scores(seq_len * seq_len * 2u, 0);
    matmulInterleaved2LearnersToInt8(
        query_out.data(),
        key_out.data(),
        seq_len,
        seq_len,
        head_hidden_size,
        attention_scores.data());

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "qk_scores_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        attention_scores.data(),
        seq_len,
        seq_len);

    // Softmax is applied independently per learner while preserving the 2-learner interleaved storage layout.
    heads[0]->softmax_->computeInterleaved2Learners(attention_scores.data(), seq_len); // softmax_approx((QK^T) / 8)

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "softmax_qk_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        attention_scores.data(),
        seq_len,
        seq_len);

    // Put V into column-major learner-interleaved layout so the final scores * V matmul can stream columns.
    std::vector<int8_t> value_by_col(head_hidden_size * seq_len * 2u, 0);
    transposeInterleavedRowsToCols2(
        value_out.data(),
        value_by_col.data(),
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "v_by_col_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        value_by_col.data(),
        head_hidden_size,
        seq_len);

    // Final attention output: softmax(QK^T) * V.
    matmulInterleaved2LearnersToInt8(
        attention_scores.data(),
        value_by_col.data(),
        seq_len,
        head_hidden_size,
        seq_len,
        output_interleaved);

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "softmax_v_pre_post_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "head_out_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);

    // Post-softmax scaling is fused into the final output matmul for better int8 accuracy, so we apply it to the output of the matmul here before dumping final head output.
    heads[0]->softmax_->post_softmax_interleaved2D(
        output_interleaved,
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "head_out_post_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);
}

/**
 * @brief Compute self-attention for four learners using fully interleaved int8 buffers.
 *
 * @param seq_len Actual sequence length to process for each learner.
 * @param heads Four SingleHeadSelfAttn objects for the same attention head, one per learner.
 * @param input_interleaved Input activations laid out in the 4-learner interleaved int8 learner format.
 * @param output_interleaved Destination buffer for the final head output in the same 4-learner interleaved int8 format.
 *
 * This follows the same algorithm as computeInterleaved2Learners(), but every intermediate buffer contains four learners:
 * interleaved Q/K/V projections, interleaved QK attention scores, interleaved softmax output, and final interleaved
 * attention-weighted V output.
 */
void SingleHeadSelfAttn::computeInterleaved4Learners(std::size_t seq_len,
                                              SingleHeadSelfAttn* heads[4],
                                              const int8_t* input_interleaved,
                                              int8_t* output_interleaved) {
    LinearLayer* query_layers[4];
    LinearLayer* key_layers[4];
    LinearLayer* value_layers[4];
    std::string dump_dirs[4];

    // Extract per-learner projection layers and dump directories into arrays used by the 4D helpers.
    for (std::size_t learner = 0; learner < 4u; learner++) {
        query_layers[learner] = heads[learner]->query_layer_;
        key_layers[learner] = heads[learner]->key_layer_;
        value_layers[learner] = heads[learner]->value_layer_;
        dump_dirs[learner] = heads[learner]->dump_dir_;
    }

    const std::size_t head_hidden_size = heads[0]->head_hidden_size_;
    std::vector<int8_t> query_out(seq_len * head_hidden_size * 4u, 0);
    std::vector<int8_t> key_out(seq_len * head_hidden_size * 4u, 0);
    std::vector<int8_t> value_out(seq_len * head_hidden_size * 4u, 0);

    // Compute Q/K/V projections for all four learners in a single interleaved CodebookDense format.
    computeCodebookDenseInterleaved4Learners("q_h", query_layers, seq_len, input_interleaved, query_out.data());
    computeCodebookDenseInterleaved4Learners("k_h", key_layers, seq_len, input_interleaved, key_out.data());
    computeCodebookDenseInterleaved4Learners("v_h", value_layers, seq_len, input_interleaved, value_out.data());

#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<uint32_t> packed_input((seq_len * heads[0]->input_dim_) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * head_hidden_size) >> 2, 0u);

    // Convert one learner at a time back to packed uint32_t so the dense reference path can compare results.
    auto compareProjection = [&](const char* prefix,
                                 LinearLayer* reference_layer,
                                 uint32_t* reference_output,
                                 const int8_t* candidate_interleaved,
                                 std::size_t learner) {
        std::fill(reference_output,
                  reference_output + ((seq_len * head_hidden_size) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner4(
            seq_len,
            heads[learner]->input_dim_,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner4(
            seq_len,
            head_hidden_size,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        reference_layer->compute(seq_len, packed_input.data(), reference_output);

        const std::string label =
            std::string(prefix) + "_h" + std::to_string(heads[0]->head_idx_) +
            "_learner" + std::to_string(heads[learner]->learner_idx_);
        comparePackedBuffers(
            label.c_str(),
            reference_output,
            packed_candidate.data(),
            (seq_len * head_hidden_size) >> 2);
    };

    for (std::size_t learner = 0; learner < 4u; learner++) {
        compareProjection("q", heads[learner]->query_reference_,
                          heads[learner]->query_reference_out_,
                          query_out.data(), learner);
        compareProjection("k", heads[learner]->key_reference_,
                          heads[learner]->key_reference_out_,
                          key_out.data(), learner);
        compareProjection("v", heads[learner]->value_reference_,
                          heads[learner]->value_reference_out_,
                          value_out.data(), learner);
    }
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "q_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        query_out.data(),
        seq_len,
        head_hidden_size);
    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "k_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        key_out.data(),
        seq_len,
        head_hidden_size);
    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "v_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        value_out.data(),
        seq_len,
        head_hidden_size);

    // Attention score matrix for four learners: each learner gets a seq_len x seq_len score matrix.
    std::vector<int8_t> attention_scores(seq_len * seq_len * 4u, 0);
    matmulInterleaved4LearnersToInt8(
        query_out.data(),
        key_out.data(),
        seq_len,
        seq_len,
        head_hidden_size,
        attention_scores.data());

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "qk_scores_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        attention_scores.data(),
        seq_len,
        seq_len);

    // Softmax is applied independently per learner while preserving the 4-learner interleaved storage layout.
    heads[0]->softmax_->computeInterleaved4Learners(attention_scores.data(), seq_len);

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "softmax_qk_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        attention_scores.data(),
        seq_len,
        seq_len);

    // Put V into column-major learner-interleaved layout so the final scores * V matmul can stream columns.
    std::vector<int8_t> value_by_col(head_hidden_size * seq_len * 4u, 0);
    transposeInterleavedRowsToCols4(
        value_out.data(),
        value_by_col.data(),
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "v_by_col_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        value_by_col.data(),
        head_hidden_size,
        seq_len);

    // Final attention output: softmax(QK^T) * V.
    matmulInterleaved4LearnersToInt8(
        attention_scores.data(),
        value_by_col.data(),
        seq_len,
        head_hidden_size,
        seq_len,
        output_interleaved);

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "softmax_v_pre_post_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "head_out_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);

    // Apply final post-softmax scaling to the interleaved output buffer.
    heads[0]->softmax_->post_softmax_interleaved4D(
        output_interleaved,
        seq_len,
        head_hidden_size);

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "head_out_post_h" + std::to_string(heads[0]->head_idx_) + ".txt",
        output_interleaved,
        seq_len,
        head_hidden_size);
}
