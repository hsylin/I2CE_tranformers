//
// Created by alireza on 3/2/22.
//

#include "transformerBlock.h"
#include "debuggerFunctions.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "layerFactory.h"
#include "profile.h"
#include "run_mode_config.h"
#include "interleavedPipeline.h"
#include "transformerBlockInterleavedHelpers.h"

// Function signatures in this file:
// TransformerBlock::TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim, std::size_t head_hidden_size, std::size_t num_heads, std::size_t ff_size, uint32_t** weightVector, std::size_t kernelDim, std::size_t maxCol, std::size_t learner_idx, std::string dump_dir);
// TransformerBlock::~TransformerBlock();
// void TransformerBlock::compute(std::size_t seq_len, uint32_t* input, uint32_t* output);
// void DenseNoSimdTransformerBlock::compute(std::size_t seq_len, uint32_t* input, uint32_t* output);
// void TransformerBlock::computeWithoutStatsReset(std::size_t seq_len, uint32_t* input, uint32_t* output);
// void TransformerBlock::computeWithStatsLabel(std::size_t seq_len, uint32_t* input, uint32_t* output, const char* stats_window_label);
// void TransformerBlock::computeBody(std::size_t seq_len, uint32_t* input, uint32_t* output);
// template <std::size_t LearnerCount> void TransformerBlock::computeGroupImpl(std::size_t seq_len, TransformerBlock** blocks, uint32_t* const* inputs, uint32_t* const* outputs);
// void TransformerBlock::computeGroup2FullInterleaved(std::size_t seq_len, TransformerBlock* blocks[2], uint32_t* const inputs[2], uint32_t* const outputs[2]);
// void TransformerBlock::computeGroup4FullInterleaved(std::size_t seq_len, TransformerBlock* blocks[4], uint32_t* const inputs[4], uint32_t* const outputs[4]);
// void TransformerBlock::computeGroup2(std::size_t seq_len, TransformerBlock* blocks[2], uint32_t* const inputs[2], uint32_t* const outputs[2]);
// void TransformerBlock::computeGroup4(std::size_t seq_len, TransformerBlock* blocks[4], uint32_t* const inputs[4], uint32_t* const outputs[4]);


/**
 * @brief Construct a transformer block and allocate its per-layer buffers.
 *
 * Builds one self-attention head per entry in `num_heads`, creates the output
 * projection and feed-forward layers, and allocates the packed intermediate
 * tensors used during block execution.
 *
 * @param pre_seq_len Maximum sequence length used to size internal buffers.
 * @param input_dim Transformer model dimension, also used as the block input and output width.
 * @param head_hidden_size Hidden width produced by each self-attention head.
 * @param num_heads Number of self-attention heads in the multi-head attention stage.
 * @param ff_size Hidden width of the feed-forward expansion layer.
 * @param weightVector Layer weight pointers ordered as per-head Q/K/V weights, then condense, FF0, and FF1.
 * @param kernelDim Hardware/kernel tile dimension passed to attention and AddNorm helpers.
 * @param maxCol Maximum column count passed to attention and AddNorm helpers.
 * @param learner_idx Learner identifier used to select learner-specific codebook/dense layers and labels.
 * @param dump_dir Optional directory where intermediate packed tensors are dumped when dumping is enabled.
 */
TransformerBlock::TransformerBlock(std::size_t pre_seq_len,
                                   std::size_t input_dim,
                                   std::size_t head_hidden_size,
                                   std::size_t num_heads,
                                   std::size_t ff_size,
                                   uint32_t** weightVector,
                                   std::size_t kernelDim,
                                   std::size_t maxCol,
                                   std::size_t learner_idx,
                                   std::string dump_dir) {
    num_heads_ = num_heads;
    head_hidden_size_ = head_hidden_size;
    input_dim_ = input_dim;
    ff_size_ = ff_size;
    learner_idx_ = learner_idx;
    dump_dir_ = dump_dir;

    // Create one self-attention module per head.
    selfatten_.reserve(num_heads_);
    for (std::size_t n = 0; n < num_heads_; ++n) {
        selfatten_.push_back(
            new SingleHeadSelfAttn(
                n,
                pre_seq_len,
                input_dim,
                head_hidden_size,
                weightVector + n * 3,
                kernelDim,
                maxCol,
                learner_idx_,
                dump_dir_));
    }

    // Allocate intermediate buffers.
    multihead_out = new uint32_t[(pre_seq_len * num_heads * head_hidden_size) >> 2]();
    condense_out = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    intermediateFF = new uint32_t[(pre_seq_len * ff_size) >> 2]();

#ifndef BWMA
    multihead_out_reshape = new uint32_t[(pre_seq_len * num_heads * head_hidden_size) >> 2]();
#endif

    addNorm = new AddNormalize(pre_seq_len, input_dim, kernelDim, maxCol);

    // Create post-attention and FFN layers through the factory.
    // The factory automatically uses CodebookDense if the layer exists in registry,
    // otherwise it falls back to Dense.
    auto condense_bundle = LayerFactory::create(
        "condense",
        num_heads * head_hidden_size,
        input_dim,
        weightVector[num_heads * 3],
        learner_idx_);

    condense = condense_bundle.main;

    auto ff0_bundle = LayerFactory::create(
        "ff0",
        input_dim,
        ff_size,
        weightVector[num_heads * 3 + 1],
        learner_idx_);

    feedForward0 = ff0_bundle.main;

    auto ff1_bundle = LayerFactory::create(
        "ff1",
        ff_size,
        input_dim,
        weightVector[num_heads * 3 + 2],
        learner_idx_);

    feedForward1 = ff1_bundle.main;

#if CFG_USE_CODEBOOK_REFERENCE
    // Optional Dense reference path for validation.
    condenseReference = condense_bundle.reference;
    feedForward0Reference = ff0_bundle.reference;
    feedForward1Reference = ff1_bundle.reference;

    referenceCondense = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceCondenseAfterAddNorm = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceFF0 = new uint32_t[(pre_seq_len * ff_size) >> 2]();
    referenceFF1 = new uint32_t[(pre_seq_len * input_dim) >> 2]();
    referenceFinalOutput = new uint32_t[(pre_seq_len * input_dim) >> 2]();
#endif
}
/**
 * @brief Destroy the transformer block and release owned layers and buffers.
 *
 * Frees self-attention heads, projection/FFN layers, AddNorm, intermediate
 * tensors, and optional reference-validation buffers.
 */
TransformerBlock::~TransformerBlock() {
    for (auto* h : selfatten_) {
        delete h;
    }

    delete[] multihead_out;
    delete[] condense_out;
    delete[] intermediateFF;

#ifndef BWMA
    delete[] multihead_out_reshape;
#endif

    delete addNorm;

    delete condense;
    delete feedForward0;
    delete feedForward1;

#if CFG_USE_CODEBOOK_REFERENCE
    delete condenseReference;
    delete feedForward0Reference;
    delete feedForward1Reference;

    delete[] referenceCondense;
    delete[] referenceCondenseAfterAddNorm;
    delete[] referenceFF0;
    delete[] referenceFF1;
    delete[] referenceFinalOutput;
    
#endif
}

/**
 * @brief Execute one transformer block with its normal stats window.
 *
 * This is the public single-learner path. It resets transformer stats using the
 * `single_transformer_block` label and then runs the regular block body.
 *
 * @param seq_len Number of sequence positions to process.
 * @param input Packed int8 input activation buffer, stored as four values per uint32_t.
 * @param output Packed int8 output activation buffer written by the block.
 */
void TransformerBlock::compute(std::size_t seq_len, uint32_t* input, uint32_t* output) {
    computeWithStatsLabel(seq_len, input, output, "single_transformer_block");
}

/**
 * @brief Execute the Dense no-SIMD baseline transformer block.
 *
 * Uses the same single-learner block body as `TransformerBlock::compute`, but
 * records the stats window under the dense baseline label.
 *
 * @param seq_len Number of sequence positions to process.
 * @param input Packed int8 input activation buffer, stored as four values per uint32_t.
 * @param output Packed int8 output activation buffer written by the block.
 */
void DenseNoSimdTransformerBlock::compute(std::size_t seq_len, uint32_t* input, uint32_t* output) {
    computeWithStatsLabel(seq_len, input, output, "dense_no_simd_baseline_transformer_block");
}

/**
 * @brief Execute one transformer block without resetting gem5 stats.
 *
 * Used when an outer caller already opened a stats window, for example when
 * multiple learners are executed sequentially as one measured region.
 *
 * @param seq_len Number of sequence positions to process.
 * @param input Packed int8 input activation buffer, stored as four values per uint32_t.
 * @param output Packed int8 output activation buffer written by the block.
 */
void TransformerBlock::computeWithoutStatsReset(std::size_t seq_len,
                                                uint32_t* input,
                                                uint32_t* output) {
    computeBody(seq_len, input, output);
}

/**
 * @brief Execute one transformer block after opening a named stats window.
 *
 * Resets transformer profiling counters with `stats_window_label`, then runs
 * the common non-grouped block implementation.
 *
 * @param seq_len Number of sequence positions to process.
 * @param input Packed int8 input activation buffer, stored as four values per uint32_t.
 * @param output Packed int8 output activation buffer written by the block.
 * @param stats_window_label Label used for the gem5 transformer stats window.
 */
void TransformerBlock::computeWithStatsLabel(std::size_t seq_len,
                                             uint32_t* input,
                                             uint32_t* output,
                                             const char* stats_window_label) {
    // Start the measured window at the transformer block itself, excluding
    // earlier setup work from the gem5 stats.
    resetTransformerStatsWindow(stats_window_label);
    computeBody(seq_len, input, output);
}


/**
 * @brief Run the standard single-learner transformer block pipeline. (Legacy from TiC-SAT with some adjustments)
 *
 * Executes attention heads, optional multi-head transpose, output projection,
 * first AddNorm, FF0, FF1, and final AddNorm. This path keeps tensors in the
 * normal packed per-learner layout rather than the full interleaved layout.
 *
 * @param seq_len Number of sequence positions to process.
 * @param input Packed int8 input activation buffer, stored as four values per uint32_t.
 * @param output Packed int8 output activation buffer written by the block.
 */
void TransformerBlock::computeBody(std::size_t seq_len, uint32_t* input, uint32_t* output) {
    // Compute each attention head independently and place it in the slice that
    // later forms the concatenated multi-head activation.
    for (std::size_t n = 0; n < num_heads_; ++n) {
        std::cout << "Head : " << n << std::endl;
        selfatten_[n]->compute(
            seq_len,
            input,
            multihead_out + n * ((seq_len * head_hidden_size_) >> 2));
    }

    // Do not overwrite the member pointer multihead_out.
    // Use a local pointer to select the correct tensor fed into condense.
    uint32_t* multihead_for_condense = multihead_out;

#ifndef BWMA
    Transpose::multihead_transpose(
        multihead_out,
        multihead_out_reshape,
        seq_len,
        head_hidden_size_ >> 2,
        num_heads_);

    multihead_for_condense = multihead_out_reshape;
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "multihead_out.txt",
        multihead_for_condense,
        seq_len,
        num_heads_ * head_hidden_size_);
    // In region mode this is a cumulative snapshot; subtract it from the next
    // snapshot to isolate the following Projection interval.
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    std::cout << "Condense" << std::endl;
    // Projection maps concatenated heads back to D_MODEL.
    condense->compute(seq_len, multihead_for_condense, condense_out);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "condense_out.txt",
        condense_out,
        seq_len,
        input_dim_);
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceCondense,
              referenceCondense + ((seq_len * input_dim_) >> 2),
              0u);

    condenseReference->compute(seq_len, multihead_for_condense, referenceCondense);

    comparePackedBuffers(
        "condense_out",
        referenceCondense,
        condense_out,
        (seq_len * input_dim_) >> 2);

    std::copy(referenceCondense,
              referenceCondense + ((seq_len * input_dim_) >> 2),
              referenceCondenseAfterAddNorm); // To keep both the pre-addnorm and post-addnorm reference values for later comparison.
#endif

    std::cout << "Add Norm" << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(input, condense_out);
#else
    // AddNorm mutates condense_out in place after adding the residual input.
    addNorm->compute(input, condense_out); // Directly modify condense_out itself to be the output of addNorm
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "after_attn_addnorm.txt",
        condense_out,
        seq_len,
        input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
    addNorm->computeRearranged(input, referenceCondenseAfterAddNorm);
#else
    addNorm->compute(input, referenceCondenseAfterAddNorm);
#endif

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "condense_out_after_addnorm",
        referenceCondenseAfterAddNorm,
        condense_out,
        (seq_len * input_dim_) >> 2);
#endif

#endif

    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm", "non_GEMM_after_projection");

    std::cout << "Feed Forward 0" << std::endl;
    // FF0 expands the hidden dimension into the feed-forward width.
    feedForward0->compute(seq_len, condense_out, intermediateFF);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "ff0_out.txt",
        intermediateFF,
        seq_len,
        ff_size_);
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

#if CFG_ENABLE_DEBUG_PRINT
    printPackedPreview("ffn0", intermediateFF, (seq_len * ff_size_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceFF0,
              referenceFF0 + ((seq_len * ff_size_) >> 2),
              0u);

    // feedForward0Reference->compute(seq_len, condense_out, referenceFF0);
    feedForward0Reference->compute(seq_len, referenceCondenseAfterAddNorm, referenceFF0);

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "ffn0",
        referenceFF0,
        intermediateFF,
        (seq_len * ff_size_) >> 2);
#endif

#endif

    std::cout << "Feed Forward 1" << std::endl;
    // FF1 contracts the feed-forward activation back into D_MODEL.
    feedForward1->compute(seq_len, intermediateFF, output);

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "ff1_out.txt",
        output,
        seq_len,
        input_dim_);
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

#if CFG_ENABLE_DEBUG_PRINT
    printPackedPreview("ffn1_pre_addnorm", output, (seq_len * input_dim_) >> 2);
#endif


#if CFG_USE_CODEBOOK_REFERENCE
    std::fill(referenceFF1,
              referenceFF1 + ((seq_len * input_dim_) >> 2),
              0u);

    feedForward1Reference->compute(seq_len, referenceFF0, referenceFF1);


#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "ffn1_pre_addnorm",
        referenceFF1,
        output,
        (seq_len * input_dim_) >> 2);
#endif

    std::copy(referenceFF1,
          referenceFF1 + ((seq_len * input_dim_) >> 2),
          referenceFinalOutput);  // std::copy(first, first + count, d_first) Copies the elements in the range [first, last) into the range beginning at d_first.
#endif

    std::cout << "Add Norm" << std::endl;
#ifdef BWMA
    addNorm->computeRearranged(condense_out, output);
#else
    // The final residual uses the post-attention activation as the skip tensor.
    addNorm->compute(condense_out, output);
#endif

    dumpPackedMatrixIfEnabled(
        dump_dir_,
        "final_out.txt",
        output,
        seq_len,
        input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
    addNorm->computeRearranged(referenceCondenseAfterAddNorm, referenceFinalOutput);
#else
    addNorm->compute(referenceCondenseAfterAddNorm, referenceFinalOutput);
#endif

#if CFG_ENABLE_DEBUG_PRINT
    comparePackedBuffers(
        "final_output_after_addnorm",
        referenceFinalOutput,
        output,
        (seq_len * input_dim_) >> 2);
#endif

#endif

    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

/**
 * @brief Execute a grouped transformer block for two or four learners.
 *
 * This templated dispatcher is shared by `computeGroup2` and `computeGroup4`.
 * When `CFG_FULL_INTERLEAVED_PIPELINE` is enabled it immediately forwards to
 * the full interleaved implementation. Otherwise it runs a grouped staged path
 * that keeps each learner's packed buffers separate.
 *
 * @tparam LearnerCount Number of learners in the group. Supported values are 2 and 4.
 * @param seq_len Number of sequence positions to process for every learner.
 * @param blocks Array of learner-specific transformer block objects.
 * @param inputs Array of packed int8 input buffers, one per learner.
 * @param outputs Array of packed int8 output buffers, one per learner.
 */
template <std::size_t LearnerCount>
void TransformerBlock::computeGroupImpl(std::size_t seq_len,
                                        TransformerBlock** blocks,
                                        uint32_t* const* inputs,
                                        uint32_t* const* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped transformer execution is supported");

#if CFG_FULL_INTERLEAVED_PIPELINE
    if constexpr (LearnerCount == 2u) {
        computeGroup2FullInterleaved(seq_len, blocks, inputs, outputs);
        return;
    }
    if constexpr (LearnerCount == 4u) {
        computeGroup4FullInterleaved(seq_len, blocks, inputs, outputs);
        return;
    }
#endif

    // Non-interleaved grouped mode keeps learner buffers separate, but runs the
    // same stage for every learner before moving to the next transformer stage.
    resetTransformerStatsWindow("grouped_transformer_block");

    for (std::size_t n = 0; n < blocks[0]->num_heads_; ++n) {
        std::cout << "Head : " << n << std::endl;

        SingleHeadSelfAttn* heads[LearnerCount];
        uint32_t* head_outputs[LearnerCount];
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner] = blocks[learner]->selfatten_[n];
            head_outputs[learner] =
                blocks[learner]->multihead_out +
                n * ((seq_len * blocks[learner]->head_hidden_size_) >> 2);
        }

        if constexpr (LearnerCount == 2u) {
            SingleHeadSelfAttn::computeGroup2(seq_len, heads, inputs, head_outputs);
        } else {
            SingleHeadSelfAttn::computeGroup4(seq_len, heads, inputs, head_outputs);
        }
    }

    uint32_t* multihead_for_condense[LearnerCount] = {};
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifndef BWMA
        Transpose::multihead_transpose(
            blocks[learner]->multihead_out,
            blocks[learner]->multihead_out_reshape,
            seq_len,
            blocks[learner]->head_hidden_size_ >> 2,
            blocks[learner]->num_heads_);

        multihead_for_condense[learner] = blocks[learner]->multihead_out_reshape;
#else
        multihead_for_condense[learner] = blocks[learner]->multihead_out;
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "multihead_out.txt",
            multihead_for_condense[learner],
            seq_len,
            blocks[learner]->num_heads_ * blocks[learner]->head_hidden_size_);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    std::cout << "Condense" << std::endl;
    LinearLayer* condense_layers[LearnerCount];
    uint32_t* condense_outputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        condense_layers[learner] = blocks[learner]->condense;
        condense_outputs[learner] = blocks[learner]->condense_out;
    }

    auto tryGroupedCodebookDense = [&](LinearLayer* const layers[LearnerCount],
                                       uint32_t* const dense_inputs[LearnerCount],
                                       uint32_t* const dense_outputs[LearnerCount]) {
        // CodebookDense can run grouped learners together; Dense fallback stays
        // per learner so validation and non-codebook configs keep working.
        if constexpr (LearnerCount == 2u) {
            return tryComputeGroupedCodebookDense2(layers, seq_len, dense_inputs, dense_outputs);
        } else {
            return tryComputeGroupedCodebookDense4(layers, seq_len, dense_inputs, dense_outputs);
        }
    };

    if (!tryGroupedCodebookDense(condense_layers, multihead_for_condense, condense_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->condense->compute(
                seq_len, multihead_for_condense[learner], blocks[learner]->condense_out);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "condense_out.txt",
            blocks[learner]->condense_out,
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceCondense,
                  blocks[learner]->referenceCondense + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  0u);

        blocks[learner]->condenseReference->compute(
            seq_len, multihead_for_condense[learner], blocks[learner]->referenceCondense);

        const std::string label =
            "condense_out_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceCondense,
                             blocks[learner]->condense_out,
                             (seq_len * blocks[learner]->input_dim_) >> 2);

        std::copy(blocks[learner]->referenceCondense,
                  blocks[learner]->referenceCondense + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  blocks[learner]->referenceCondenseAfterAddNorm);
#endif
    }
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(inputs[learner], blocks[learner]->condense_out);
#else
        blocks[learner]->addNorm->compute(inputs[learner], blocks[learner]->condense_out);
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "after_attn_addnorm.txt",
            blocks[learner]->condense_out,
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(
            inputs[learner], blocks[learner]->referenceCondenseAfterAddNorm);
#else
        blocks[learner]->addNorm->compute(
            inputs[learner], blocks[learner]->referenceCondenseAfterAddNorm);
#endif

#if CFG_ENABLE_DEBUG_PRINT
        const std::string label =
            "condense_out_after_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceCondenseAfterAddNorm,
                             blocks[learner]->condense_out,
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif
#endif
    }

    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm", "non_GEMM_after_projection");

    std::cout << "Feed Forward 0" << std::endl;
    LinearLayer* ff0_layers[LearnerCount];
    uint32_t* ff0_outputs[LearnerCount];
    uint32_t* ff0_inputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        ff0_layers[learner] = blocks[learner]->feedForward0;
        ff0_outputs[learner] = blocks[learner]->intermediateFF;
        ff0_inputs[learner] = blocks[learner]->condense_out;
    }
    if (!tryGroupedCodebookDense(ff0_layers, ff0_inputs, ff0_outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->feedForward0->compute(
                seq_len, blocks[learner]->condense_out, blocks[learner]->intermediateFF);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        const std::string ff0_label =
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_);

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff0_out.txt",
            blocks[learner]->intermediateFF,
            seq_len,
            blocks[learner]->ff_size_);

#if CFG_ENABLE_DEBUG_PRINT
        printPackedPreview(ff0_label.c_str(), blocks[learner]->intermediateFF, (seq_len * blocks[learner]->ff_size_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceFF0,
                  blocks[learner]->referenceFF0 + ((seq_len * blocks[learner]->ff_size_) >> 2),
                  0u);

        blocks[learner]->feedForward0Reference->compute(
            seq_len, blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFF0);

#if CFG_ENABLE_DEBUG_PRINT
        comparePackedBuffers(ff0_label.c_str(),
                             blocks[learner]->referenceFF0,
                             blocks[learner]->intermediateFF,
                             (seq_len * blocks[learner]->ff_size_) >> 2);
#endif
#endif
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    std::cout << "Feed Forward 1" << std::endl;
    LinearLayer* ff1_layers[LearnerCount];
    uint32_t* ff1_inputs[LearnerCount];
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        ff1_layers[learner] = blocks[learner]->feedForward1;
        ff1_inputs[learner] = blocks[learner]->intermediateFF;
    }
    if (!tryGroupedCodebookDense(ff1_layers, ff1_inputs, outputs)) {
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            blocks[learner]->feedForward1->compute(
                seq_len, blocks[learner]->intermediateFF, outputs[learner]);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        const std::string ff1_label =
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff1_out.txt",
            outputs[learner],
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_ENABLE_DEBUG_PRINT
        printPackedPreview(ff1_label.c_str(), outputs[learner], (seq_len * blocks[learner]->input_dim_) >> 2);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
        std::fill(blocks[learner]->referenceFF1,
                  blocks[learner]->referenceFF1 + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  0u);

        blocks[learner]->feedForward1Reference->compute(
            seq_len, blocks[learner]->referenceFF0, blocks[learner]->referenceFF1);

#if CFG_ENABLE_DEBUG_PRINT
        comparePackedBuffers(ff1_label.c_str(),
                             blocks[learner]->referenceFF1,
                             outputs[learner],
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif

        std::copy(blocks[learner]->referenceFF1,
                  blocks[learner]->referenceFF1 + ((seq_len * blocks[learner]->input_dim_) >> 2),
                  blocks[learner]->referenceFinalOutput);
#endif
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(blocks[learner]->condense_out, outputs[learner]);
#else
        blocks[learner]->addNorm->compute(blocks[learner]->condense_out, outputs[learner]);
#endif

        dumpPackedMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "final_out.txt",
            outputs[learner],
            seq_len,
            blocks[learner]->input_dim_);

#if CFG_USE_CODEBOOK_REFERENCE
#ifdef BWMA
        blocks[learner]->addNorm->computeRearranged(
            blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFinalOutput);
#else
        blocks[learner]->addNorm->compute(
            blocks[learner]->referenceCondenseAfterAddNorm, blocks[learner]->referenceFinalOutput);
#endif

#if CFG_ENABLE_DEBUG_PRINT
        const std::string label =
            "final_output_after_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        comparePackedBuffers(label.c_str(),
                             blocks[learner]->referenceFinalOutput,
                             outputs[learner],
                             (seq_len * blocks[learner]->input_dim_) >> 2);
#endif
#endif
    }

    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

#if CFG_FULL_INTERLEAVED_PIPELINE
/**
 * @brief Execute the full interleaved transformer pipeline for two learners.
 *
 * Converts the two packed learner inputs into a shared `[seq][feature][learner]`
 * int8 layout, keeps that layout through attention, AddNorm, condense, and FFN
 * stages, then packs the final result back into the caller-provided output buffers.
 *
 * @param seq_len Number of sequence positions to process for both learners.
 * @param blocks Two learner-specific transformer block objects whose layers are used.
 * @param inputs Two packed int8 input buffers, one per learner.
 * @param outputs Two packed int8 output buffers that receive the final block outputs.
 */
void TransformerBlock::computeGroup2FullInterleaved(std::size_t seq_len,
                                                    TransformerBlock* blocks[2],
                                                    uint32_t* const inputs[2],
                                                    uint32_t* const outputs[2]) {
    // Full-interleaved mode keeps both learners in one [seq][feature][learner]
    // buffer, so attention, AddNorm, and CodebookDense avoid repeated packing.
    resetTransformerStatsWindow("group2_full_interleaved_transformer_block");

    std::string dump_dirs[2];
#if CFG_USE_CODEBOOK_REFERENCE
    std::size_t learner_ids[2];
#endif
    for (std::size_t learner = 0; learner < 2u; learner++) {
        dump_dirs[learner] = blocks[learner]->dump_dir_;
#if CFG_USE_CODEBOOK_REFERENCE
        learner_ids[learner] = blocks[learner]->learner_idx_;
#endif
    }

    const std::size_t input_dim = blocks[0]->input_dim_;
    const std::size_t head_hidden_size = blocks[0]->head_hidden_size_;
    const std::size_t num_heads = blocks[0]->num_heads_;
    const std::size_t ff_size = blocks[0]->ff_size_;

    std::vector<int8_t> input_interleaved(seq_len * input_dim * 2u, 0);
    // Convert the packed per-learner inputs into lane-adjacent int8 values.
    interleavePackedLearners2(seq_len, input_dim, inputs, input_interleaved.data()); // Convert input matrix to interleaved format

     // Optional: dump the interleaved input for debugging.

    std::vector<int8_t> multihead_interleaved(
        seq_len * num_heads * head_hidden_size * 2u, 0);
    std::vector<int8_t> head_out_interleaved(
        seq_len * head_hidden_size * 2u, 0);

    for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        std::cout << "Head : " << head_idx << std::endl;

        SingleHeadSelfAttn* heads[2];
        for (std::size_t learner = 0; learner < 2u; learner++) {
            heads[learner] = blocks[learner]->selfatten_[head_idx];
        }

        std::fill(head_out_interleaved.begin(), head_out_interleaved.end(), 0);
        SingleHeadSelfAttn::computeInterleaved2Learners(
            seq_len,
            heads,
            input_interleaved.data(),
            head_out_interleaved.data());

        copyHeadToMultiheadInterleaved2Learners(
            head_out_interleaved.data(),
            multihead_interleaved.data(),
            seq_len,
            head_idx,
            head_hidden_size,
            num_heads);
    }

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "multihead_out.txt",
        multihead_interleaved.data(),
        seq_len,
        num_heads * head_hidden_size);
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    std::cout << "Condense" << std::endl;
    LinearLayer* condense_layers[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        condense_layers[learner] = blocks[learner]->condense;
    }

    std::vector<int8_t> condense_interleaved(seq_len * input_dim * 2u, 0);
    // The interleaved CodebookDense call consumes and produces the same
    // [seq][feature][learner] layout.
    computeCodebookDenseInterleaved2Learners(
        "condense",
        condense_layers,
        seq_len,
        multihead_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* condense_references[2];
    uint32_t* condense_reference_outputs[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        condense_references[learner] = blocks[learner]->condenseReference;
        condense_reference_outputs[learner] = blocks[learner]->referenceCondense;
    }
    compareInterleavedDenseReference2D(
        "condense_out",
        condense_references,
        condense_reference_outputs,
        learner_ids,
        seq_len,
        num_heads * head_hidden_size,
        input_dim,
        multihead_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "condense_out.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> condense_before_addnorm_interleaved = condense_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved2Learners(
        input_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference2D(
        "condense_out_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        input_interleaved.data(),
        condense_before_addnorm_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "after_attn_addnorm.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);

    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm", "non_GEMM_after_projection");

    std::cout << "Feed Forward 0" << std::endl;
    LinearLayer* ff0_layers[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        ff0_layers[learner] = blocks[learner]->feedForward0;
    }

    std::vector<int8_t> ff0_interleaved(seq_len * ff_size * 2u, 0);
    // FF0 expands both learners together while preserving the interleaved layout.
    computeCodebookDenseInterleaved2Learners(
        "ff0",
        ff0_layers,
        seq_len,
        condense_interleaved.data(),
        ff0_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 2u; learner++) {
        const std::string ff0_label =
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview2D(
            ff0_label.c_str(),
            ff0_interleaved.data(),
            seq_len,
            ff_size,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff0_references[2];
    uint32_t* ff0_reference_outputs[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        ff0_references[learner] = blocks[learner]->feedForward0Reference;
        ff0_reference_outputs[learner] = blocks[learner]->referenceFF0;
    }
    compareInterleavedDenseReference2D(
        "ffn0",
        ff0_references,
        ff0_reference_outputs,
        learner_ids,
        seq_len,
        input_dim,
        ff_size,
        condense_interleaved.data(),
        ff0_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "ff0_out.txt",
        ff0_interleaved.data(),
        seq_len,
        ff_size);
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    std::cout << "Feed Forward 1" << std::endl;
    LinearLayer* ff1_layers[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        ff1_layers[learner] = blocks[learner]->feedForward1;
    }

    std::vector<int8_t> ff1_interleaved(seq_len * input_dim * 2u, 0);
    // FF1 returns both learners to D_MODEL before the final residual AddNorm.
    computeCodebookDenseInterleaved2Learners(
        "ff1",
        ff1_layers,
        seq_len,
        ff0_interleaved.data(),
        ff1_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 2u; learner++) {
        const std::string ff1_label =
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview2D(
            ff1_label.c_str(),
            ff1_interleaved.data(),
            seq_len,
            input_dim,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff1_references[2];
    uint32_t* ff1_reference_outputs[2];
    for (std::size_t learner = 0; learner < 2u; learner++) {
        ff1_references[learner] = blocks[learner]->feedForward1Reference;
        ff1_reference_outputs[learner] = blocks[learner]->referenceFF1;
    }
    compareInterleavedDenseReference2D(
        "ffn1_pre_addnorm",
        ff1_references,
        ff1_reference_outputs,
        learner_ids,
        seq_len,
        ff_size,
        input_dim,
        ff0_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "ff1_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> ff1_before_addnorm_interleaved = ff1_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved2Learners(
        condense_interleaved.data(),
        ff1_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference2D(
        "final_output_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        condense_interleaved.data(),
        ff1_before_addnorm_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices2(
        dump_dirs,
        "final_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);

    // Pack the final int8 lanes back into the caller's per-learner uint32 buffers.
    packInterleavedLearners2(seq_len, input_dim, ff1_interleaved.data(), outputs);

    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

/**
 * @brief Execute the full interleaved transformer pipeline for four learners.
 *
 * Converts the four packed learner inputs into a shared `[seq][feature][learner]`
 * int8 layout, keeps that layout through attention, AddNorm, condense, and FFN
 * stages, then packs the final result back into the caller-provided output buffers.
 *
 * @param seq_len Number of sequence positions to process for all four learners.
 * @param blocks Four learner-specific transformer block objects whose layers are used.
 * @param inputs Four packed int8 input buffers, one per learner.
 * @param outputs Four packed int8 output buffers that receive the final block outputs.
 */
void TransformerBlock::computeGroup4FullInterleaved(std::size_t seq_len,
                                                    TransformerBlock* blocks[4],
                                                    uint32_t* const inputs[4],
                                                    uint32_t* const outputs[4]) {
    // The 4D path is the same idea as 2D, with four learners packed into the
    // innermost lane of the activation buffers.
    resetTransformerStatsWindow("group4_full_interleaved_transformer_block");

    std::string dump_dirs[4];
#if CFG_USE_CODEBOOK_REFERENCE
    std::size_t learner_ids[4];
#endif
    for (std::size_t learner = 0; learner < 4u; learner++) {
        dump_dirs[learner] = blocks[learner]->dump_dir_;
#if CFG_USE_CODEBOOK_REFERENCE
        learner_ids[learner] = blocks[learner]->learner_idx_;
#endif
    }

    const std::size_t input_dim = blocks[0]->input_dim_;
    const std::size_t head_hidden_size = blocks[0]->head_hidden_size_;
    const std::size_t num_heads = blocks[0]->num_heads_;
    const std::size_t ff_size = blocks[0]->ff_size_;

    std::vector<int8_t> input_interleaved(seq_len * input_dim * 4u, 0);
    // Convert four packed learner tensors into one [seq][feature][learner] tensor.
    interleavePackedLearners4(seq_len, input_dim, inputs, input_interleaved.data());

    std::vector<int8_t> multihead_interleaved(
        seq_len * num_heads * head_hidden_size * 4u, 0);
    std::vector<int8_t> head_out_interleaved(
        seq_len * head_hidden_size * 4u, 0);

    for (std::size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        std::cout << "Head : " << head_idx << std::endl;

        SingleHeadSelfAttn* heads[4];
        for (std::size_t learner = 0; learner < 4u; learner++) {
            heads[learner] = blocks[learner]->selfatten_[head_idx];
        }

        std::fill(head_out_interleaved.begin(), head_out_interleaved.end(), 0);
        SingleHeadSelfAttn::computeInterleaved4Learners(
            seq_len,
            heads,
            input_interleaved.data(),
            head_out_interleaved.data());

        copyHeadToMultiheadInterleaved4Learners(
            head_out_interleaved.data(),
            multihead_interleaved.data(),
            seq_len,
            head_idx,
            head_hidden_size,
            num_heads);
    }

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "multihead_out.txt",
        multihead_interleaved.data(),
        seq_len,
        num_heads * head_hidden_size);
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    std::cout << "Condense" << std::endl;
    LinearLayer* condense_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        condense_layers[learner] = blocks[learner]->condense;
    }

    std::vector<int8_t> condense_interleaved(seq_len * input_dim * 4u, 0);
    // The diff-seq/same-seq choice is hidden inside the CodebookDense helper.
    computeCodebookDenseInterleaved4Learners(
        "condense",
        condense_layers,
        seq_len,
        multihead_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* condense_references[4];
    uint32_t* condense_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        condense_references[learner] = blocks[learner]->condenseReference;
        condense_reference_outputs[learner] = blocks[learner]->referenceCondense;
    }
    compareInterleavedDenseReference4D(
        "condense_out",
        condense_references,
        condense_reference_outputs,
        learner_ids,
        seq_len,
        num_heads * head_hidden_size,
        input_dim,
        multihead_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "condense_out.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> condense_before_addnorm_interleaved = condense_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved4Learners(
        input_interleaved.data(),
        condense_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference4D(
        "condense_out_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        input_interleaved.data(),
        condense_before_addnorm_interleaved.data(),
        condense_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "after_attn_addnorm.txt",
        condense_interleaved.data(),
        seq_len,
        input_dim);

    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm", "non_GEMM_after_projection");

    std::cout << "Feed Forward 0" << std::endl;
    LinearLayer* ff0_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff0_layers[learner] = blocks[learner]->feedForward0;
    }

    std::vector<int8_t> ff0_interleaved(seq_len * ff_size * 4u, 0);
    // FF0 expands all four learners together.
    computeCodebookDenseInterleaved4Learners(
        "ff0",
        ff0_layers,
        seq_len,
        condense_interleaved.data(),
        ff0_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 4u; learner++) {
        const std::string ff0_label =
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview4D(
            ff0_label.c_str(),
            ff0_interleaved.data(),
            seq_len,
            ff_size,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff0_references[4];
    uint32_t* ff0_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff0_references[learner] = blocks[learner]->feedForward0Reference;
        ff0_reference_outputs[learner] = blocks[learner]->referenceFF0;
    }
    compareInterleavedDenseReference4D(
        "ffn0",
        ff0_references,
        ff0_reference_outputs,
        learner_ids,
        seq_len,
        input_dim,
        ff_size,
        condense_interleaved.data(),
        ff0_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "ff0_out.txt",
        ff0_interleaved.data(),
        seq_len,
        ff_size);
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    std::cout << "Feed Forward 1" << std::endl;
    LinearLayer* ff1_layers[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff1_layers[learner] = blocks[learner]->feedForward1;
    }

    std::vector<int8_t> ff1_interleaved(seq_len * input_dim * 4u, 0);
    // FF1 contracts all four learners back to D_MODEL.
    computeCodebookDenseInterleaved4Learners(
        "ff1",
        ff1_layers,
        seq_len,
        ff0_interleaved.data(),
        ff1_interleaved.data());

#if CFG_ENABLE_DEBUG_PRINT
    for (std::size_t learner = 0; learner < 4u; learner++) {
        const std::string ff1_label =
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_);
        printInterleavedPackedPreview4D(
            ff1_label.c_str(),
            ff1_interleaved.data(),
            seq_len,
            input_dim,
            learner);
    }
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* ff1_references[4];
    uint32_t* ff1_reference_outputs[4];
    for (std::size_t learner = 0; learner < 4u; learner++) {
        ff1_references[learner] = blocks[learner]->feedForward1Reference;
        ff1_reference_outputs[learner] = blocks[learner]->referenceFF1;
    }
    compareInterleavedDenseReference4D(
        "ffn1_pre_addnorm",
        ff1_references,
        ff1_reference_outputs,
        learner_ids,
        seq_len,
        ff_size,
        input_dim,
        ff0_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "ff1_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    std::cout << "Add Norm" << std::endl;
#if CFG_USE_CODEBOOK_REFERENCE
    std::vector<int8_t> ff1_before_addnorm_interleaved = ff1_interleaved;
#endif
    blocks[0]->addNorm->computeInterleaved4Learners(
        condense_interleaved.data(),
        ff1_interleaved.data());

#if CFG_USE_CODEBOOK_REFERENCE
    compareInterleavedAddNormReference4D(
        "final_output_after_addnorm",
        blocks[0]->addNorm,
        learner_ids,
        seq_len,
        input_dim,
        condense_interleaved.data(),
        ff1_before_addnorm_interleaved.data(),
        ff1_interleaved.data());
#endif

    dumpInterleavedLearnerMatrices4(
        dump_dirs,
        "final_out.txt",
        ff1_interleaved.data(),
        seq_len,
        input_dim);

    // Restore the public per-learner packed representation expected by callers.
    packInterleavedLearners4(seq_len, input_dim, ff1_interleaved.data(), outputs);

    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}
#endif

/**
 * @brief Public grouped transformer entry point for two learners.
 *
 * Forwards to `computeGroupImpl<2u>`, which selects either the full interleaved
 * path or the staged grouped fallback according to compile-time configuration.
 *
 * @param seq_len Number of sequence positions to process for both learners.
 * @param blocks Two learner-specific transformer block objects.
 * @param inputs Two packed int8 input buffers, one per learner.
 * @param outputs Two packed int8 output buffers that receive the final block outputs.
 */
void TransformerBlock::computeGroup2(std::size_t seq_len,
                                     TransformerBlock* blocks[2],
                                     uint32_t* const inputs[2],
                                     uint32_t* const outputs[2]) {
    computeGroupImpl<2u>(seq_len, blocks, inputs, outputs);
}

/**
 * @brief Public grouped transformer entry point for four learners.
 *
 * Forwards to `computeGroupImpl<4u>`, which selects either the full interleaved
 * path or the staged grouped fallback according to compile-time configuration.
 *
 * @param seq_len Number of sequence positions to process for all four learners.
 * @param blocks Four learner-specific transformer block objects.
 * @param inputs Four packed int8 input buffers, one per learner.
 * @param outputs Four packed int8 output buffers that receive the final block outputs.
 */
void TransformerBlock::computeGroup4(std::size_t seq_len,
                                     TransformerBlock* blocks[4],
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]) {
    computeGroupImpl<4u>(seq_len, blocks, inputs, outputs);
}
