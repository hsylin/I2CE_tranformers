#include "floatTransformerBlock.h"

#include "floatDump.h"
#include "profile.h"

#if CFG_USE_FP32_TRANSFORMER

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace TransformerFloat {

namespace {

// Copy one attention head output into its slice of the concatenated multi-head
// matrix. Each head produces [seq_len x head_hidden_size]; the transformer block
// stores all heads side by side as [seq_len x (num_heads * head_hidden_size)].
void copyHeadToMultihead(const float* head,
                         float* multihead,
                         std::size_t seq_len,
                         std::size_t head_idx,
                         std::size_t head_hidden_size,
                         std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            multihead[seq * multihead_cols + head_col_base + feature] =
                head[seq * head_hidden_size + feature];
        }
    }
}

// Interleaved version of copyHeadToMultihead(). Values for multiple learners
// are stored next to each other in memory:
//   logical_value(row, col, learner)
//     -> buffer[(row * cols + col) * learner_count + learner]
// This layout lets grouped FP32 kernels process the same logical position for
// several learners together.
void copyHeadToMultiheadInterleaved(const float* head_interleaved,
                                    float* multihead_interleaved,
                                    std::size_t learner_count,
                                    std::size_t seq_len,
                                    std::size_t head_idx,
                                    std::size_t head_hidden_size,
                                    std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            const float* in_slot =
                head_interleaved + ((seq * head_hidden_size + feature) * learner_count);
            float* out_slot =
                multihead_interleaved +
                ((seq * multihead_cols + head_col_base + feature) * learner_count);
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

// Profiling label used by the grouped, but not fully interleaved, execution
// path. The static_assert documents the only supported grouped widths.
template <std::size_t LearnerCount>
const char* groupedScope() {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped FP32 blocks are supported");
    if constexpr (LearnerCount == 2u) {
        return "fp32_group2_transformer_block";
    }
    return "fp32_group4_transformer_block";
}

#if CFG_FULL_INTERLEAVED_PIPELINE
// Profiling label used when the whole transformer block keeps learner data in
// interleaved layout from input through output.
template <std::size_t LearnerCount>
const char* fullInterleavedScope() {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner interleaved FP32 blocks are supported");
    if constexpr (LearnerCount == 2u) {
        return "fp32_group2_full_interleaved_transformer_block";
    }
    return "fp32_group4_full_interleaved_transformer_block";
}
#endif

} // namespace

FloatTransformerBlock::FloatTransformerBlock(std::size_t pre_seq_len,
                                             std::size_t input_dim,
                                             std::size_t head_hidden_size,
                                             std::size_t num_heads,
                                             std::size_t ff_size,
                                             std::size_t learner_idx,
                                             std::string dump_dir)
    : num_heads_(num_heads),
      head_hidden_size_(head_hidden_size),
      input_dim_(input_dim),
      ff_size_(ff_size),
      learner_idx_(learner_idx),
      dump_dir_(std::move(dump_dir)),
      add_norm_(pre_seq_len, input_dim),
      condense_("condense", learner_idx),
      feed_forward0_("ff0", learner_idx),
      feed_forward1_("ff1", learner_idx) {
    // Create one single-head attention object per head. The block later calls
    // each head, concatenates their outputs, and projects the combined tensor
    // back to the model dimension with the condense layer.
    selfatten_.reserve(num_heads_);
    for (std::size_t head = 0; head < num_heads_; head++) {
        selfatten_.push_back(new FloatSingleHeadSelfAttn(
            head,
            pre_seq_len,
            input_dim,
            head_hidden_size,
            learner_idx_,
            dump_dir_));
    }
}

FloatTransformerBlock::~FloatTransformerBlock() {
    // The attention heads are allocated explicitly in the constructor, so the
    // destructor releases them here before the block goes away.
    for (auto* head : selfatten_) {
        delete head;
    }
}

// Execute one FP32 transformer block for a single learner.
//
// The block follows the standard transformer encoder structure:
//   1. run all self-attention heads,
//   2. concatenate and project the heads,
//   3. add the residual input and layer-normalize,
//   4. run the two feed-forward layers,
//   5. add the second residual and layer-normalize.
void FloatTransformerBlock::compute(std::size_t seq_len,
                                    const float* input,
                                    float* output) {
    resetTransformerStatsWindow("fp32_single_transformer_block");

    // Step 1: Compute every attention head independently and copy each head into
    // its column range of the multi-head output matrix.
    multihead_out_.assign(seq_len * num_heads_ * head_hidden_size_, 0.0f);
    for (std::size_t head = 0; head < num_heads_; head++) {
        std::cout << "Head : " << head << std::endl;
        Matrix head_out;
        selfatten_[head]->compute(seq_len, input, head_out);
        copyHeadToMultihead(head_out.data(),
                            multihead_out_.data(),
                            seq_len,
                            head,
                            head_hidden_size_,
                            num_heads_);
    }

    dumpFloatMatrixIfEnabled(
        dump_dir_, "multihead_out.txt", multihead_out_.data(),
        seq_len, num_heads_ * head_hidden_size_);
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    // Step 2: Project the concatenated heads back from the multi-head dimension
    // to the transformer model dimension.
    std::cout << "Condense" << std::endl;
    condense_.compute(seq_len, multihead_out_.data(), condense_out_);
    dumpFloatMatrixIfEnabled(
        dump_dir_, "condense_out.txt", condense_out_.data(), seq_len, input_dim_);
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    // Step 3: Apply the attention residual connection. add_norm_ mutates
    // condense_out_ in place by adding input and then layer-normalizing rows.
    std::cout << "Add Norm" << std::endl;
    add_norm_.compute(input, condense_out_.data());
    dumpFloatMatrixIfEnabled(
        dump_dir_, "after_attn_addnorm.txt", condense_out_.data(), seq_len, input_dim_);
    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm",
                                              "non_GEMM_after_projection");

    // Step 4: First feed-forward layer expands from model dimension to the
    // configured feed-forward dimension.
    std::cout << "Feed Forward 0" << std::endl;
    feed_forward0_.compute(seq_len, condense_out_.data(), intermediate_ff_);
    dumpFloatMatrixIfEnabled(
        dump_dir_, "ff0_out.txt", intermediate_ff_.data(), seq_len, ff_size_);
    printFloatPreview("ffn0_learner" + std::to_string(learner_idx_),
                      intermediate_ff_.data(),
                      intermediate_ff_.size());
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    // Step 5: Second feed-forward layer projects back to the model dimension.
    std::cout << "Feed Forward 1" << std::endl;
    feed_forward1_.compute(seq_len, intermediate_ff_.data(), ff1_out_);
    dumpFloatMatrixIfEnabled(
        dump_dir_, "ff1_out.txt", ff1_out_.data(), seq_len, input_dim_);
    printFloatPreview("ffn1_pre_addnorm_learner" + std::to_string(learner_idx_),
                      ff1_out_.data(),
                      ff1_out_.size());
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    // Step 6: Apply the feed-forward residual connection, copy the normalized
    // result to the caller's output buffer, and record the final dump/profile
    // boundary.
    std::cout << "Add Norm" << std::endl;
    add_norm_.compute(condense_out_.data(), ff1_out_.data());
    std::copy(ff1_out_.begin(), ff1_out_.end(), output);

    dumpFloatMatrixIfEnabled(
        dump_dir_, "final_out.txt", output, seq_len, input_dim_);
    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

template <std::size_t LearnerCount>
void FloatTransformerBlock::computeGroupImpl(std::size_t seq_len,
                                             FloatTransformerBlock** blocks,
                                             const float* const* inputs,
                                             float* const* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped FP32 transformer execution is supported");

#if CFG_FULL_INTERLEAVED_PIPELINE
    // When enabled, the grouped path delegates to the fully interleaved pipeline
    // so intermediate tensors remain interleaved across all block stages.
    if constexpr (LearnerCount == 2u) {
        computeGroup2FullInterleaved(seq_len, blocks, inputs, outputs);
        return;
    }
    if constexpr (LearnerCount == 4u) {
        computeGroup4FullInterleaved(seq_len, blocks, inputs, outputs);
        return;
    }
#endif

    resetTransformerStatsWindow(groupedScope<LearnerCount>());

    const std::size_t input_dim = blocks[0]->input_dim_;
    const std::size_t head_hidden_size = blocks[0]->head_hidden_size_;
    const std::size_t num_heads = blocks[0]->num_heads_;
    const std::size_t ff_size = blocks[0]->ff_size_;

    // Step 1: Prepare each learner's multi-head output buffer.
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->multihead_out_.assign(seq_len * num_heads * head_hidden_size, 0.0f);
    }

    // Step 2: For each head index, run the matching head for every learner and
    // copy each learner's result into that learner's concatenated multi-head
    // matrix.
    for (std::size_t head = 0; head < num_heads; head++) {
        std::cout << "Head : " << head << std::endl;

        FloatSingleHeadSelfAttn* heads[LearnerCount];
        Matrix head_outputs[LearnerCount];
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner] = blocks[learner]->selfatten_[head];
        }

        if constexpr (LearnerCount == 2u) {
            FloatSingleHeadSelfAttn::computeGroup2(seq_len, heads, inputs, head_outputs);
        } else {
            FloatSingleHeadSelfAttn::computeGroup4(seq_len, heads, inputs, head_outputs);
        }

        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            copyHeadToMultihead(head_outputs[learner].data(),
                                blocks[learner]->multihead_out_.data(),
                                seq_len,
                                head,
                                head_hidden_size,
                                num_heads);
        }
    }

    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "multihead_out.txt",
            blocks[learner]->multihead_out_.data(),
            seq_len,
            num_heads * head_hidden_size);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    // Step 3: Condense each learner's concatenated heads back to model width.
    std::cout << "Condense" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->condense_.compute(
            seq_len,
            blocks[learner]->multihead_out_.data(),
            blocks[learner]->condense_out_);
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "condense_out.txt",
            blocks[learner]->condense_out_.data(),
            seq_len,
            input_dim);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    // Step 4: Add the original input residual to the attention projection and
    // layer-normalize each learner independently.
    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->add_norm_.compute(inputs[learner],
                                           blocks[learner]->condense_out_.data());
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "after_attn_addnorm.txt",
            blocks[learner]->condense_out_.data(),
            seq_len,
            input_dim);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm",
                                              "non_GEMM_after_projection");

    // Step 5: Run the first feed-forward layer for each learner.
    std::cout << "Feed Forward 0" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->feed_forward0_.compute(
            seq_len,
            blocks[learner]->condense_out_.data(),
            blocks[learner]->intermediate_ff_);
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff0_out.txt",
            blocks[learner]->intermediate_ff_.data(),
            seq_len,
            ff_size);
        printFloatPreview(
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_),
            blocks[learner]->intermediate_ff_.data(),
            blocks[learner]->intermediate_ff_.size());
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    // Step 6: Run the second feed-forward layer to return to model width.
    std::cout << "Feed Forward 1" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->feed_forward1_.compute(
            seq_len,
            blocks[learner]->intermediate_ff_.data(),
            blocks[learner]->ff1_out_);
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "ff1_out.txt",
            blocks[learner]->ff1_out_.data(),
            seq_len,
            input_dim);
        printFloatPreview(
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_),
            blocks[learner]->ff1_out_.data(),
            blocks[learner]->ff1_out_.size());
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    // Step 7: Add the feed-forward residual, normalize, and copy each learner's
    // final matrix into the output buffer supplied by the caller.
    std::cout << "Add Norm" << std::endl;
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        blocks[learner]->add_norm_.compute(blocks[learner]->condense_out_.data(),
                                           blocks[learner]->ff1_out_.data());
        std::copy(blocks[learner]->ff1_out_.begin(),
                  blocks[learner]->ff1_out_.end(),
                  outputs[learner]);
        dumpFloatMatrixIfEnabled(
            blocks[learner]->dump_dir_,
            "final_out.txt",
            outputs[learner],
            seq_len,
            input_dim);
    }
    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

#if CFG_FULL_INTERLEAVED_PIPELINE
template <std::size_t LearnerCount>
void FloatTransformerBlock::computeFullInterleavedBlock(std::size_t seq_len,
                                                        FloatTransformerBlock** blocks,
                                                        const float* const* inputs,
                                                        float* const* outputs) {
    // Fully interleaved execution stores all learners in one physical matrix at
    // each stage. This is used to exercise the interleaved GEMM kernels while
    // preserving the same transformer math as the single-learner path.
    resetTransformerStatsWindow(fullInterleavedScope<LearnerCount>());

    const std::size_t input_dim = blocks[0]->input_dim_;
    const std::size_t head_hidden_size = blocks[0]->head_hidden_size_;
    const std::size_t num_heads = blocks[0]->num_heads_;
    const std::size_t ff_size = blocks[0]->ff_size_;

    std::vector<std::string> dump_dirs(LearnerCount);
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dump_dirs[learner] = blocks[learner]->dump_dir_;
    }

    Matrix input_interleaved;
    // Step 1: Convert separate learner input matrices into the interleaved
    // layout expected by the grouped FP32 kernels.
    interleaveLearnerMatrices(inputs, LearnerCount, seq_len, input_dim, input_interleaved);

    // Step 2: Compute every attention head in interleaved layout, then copy each
    // interleaved head into its multi-head column range.
    Matrix multihead_interleaved(seq_len * num_heads * head_hidden_size * LearnerCount, 0.0f);
    for (std::size_t head = 0; head < num_heads; head++) {
        std::cout << "Head : " << head << std::endl;

        FloatSingleHeadSelfAttn* heads[LearnerCount];
        for (std::size_t learner = 0; learner < LearnerCount; learner++) {
            heads[learner] = blocks[learner]->selfatten_[head];
        }

        Matrix head_out_interleaved;
        if constexpr (LearnerCount == 2u) {
            FloatSingleHeadSelfAttn::computeInterleaved2Learners(
                seq_len, heads, input_interleaved.data(), head_out_interleaved);
        } else {
            FloatSingleHeadSelfAttn::computeInterleaved4Learners(
                seq_len, heads, input_interleaved.data(), head_out_interleaved);
        }

        copyHeadToMultiheadInterleaved(head_out_interleaved.data(),
                                       multihead_interleaved.data(),
                                       LearnerCount,
                                       seq_len,
                                       head,
                                       head_hidden_size,
                                       num_heads);
    }

    dumpInterleavedFloatMatrices(dump_dirs,
                                 "multihead_out.txt",
                                 multihead_interleaved.data(),
                                 seq_len,
                                 num_heads * head_hidden_size,
                                 LearnerCount);
    dumpTransformerStatsCheckpointIfProfiling("after_mha", "MHA");

    // Step 3: Apply the shared interleaved projection layer.
    std::cout << "Condense" << std::endl;
    Matrix condense_interleaved;
    blocks[0]->condense_.computeInterleaved(
        LearnerCount, multihead_interleaved.data(), seq_len, condense_interleaved);
    dumpInterleavedFloatMatrices(dump_dirs,
                                 "condense_out.txt",
                                 condense_interleaved.data(),
                                 seq_len,
                                 input_dim,
                                 LearnerCount);
    dumpTransformerStatsCheckpointIfProfiling("after_projection", "Projection");

    // Step 4: Add the interleaved input residual and layer-normalize each
    // learner stream independently inside the same buffer.
    std::cout << "Add Norm" << std::endl;
    blocks[0]->add_norm_.computeInterleaved(
        input_interleaved.data(), condense_interleaved.data(), LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs,
                                 "after_attn_addnorm.txt",
                                 condense_interleaved.data(),
                                 seq_len,
                                 input_dim,
                                 LearnerCount);
    dumpTransformerStatsCheckpointIfProfiling("after_attn_addnorm",
                                              "non_GEMM_after_projection");

    // Step 5: Run the two feed-forward layers while keeping learner values
    // interleaved.
    std::cout << "Feed Forward 0" << std::endl;
    Matrix ff0_interleaved;
    blocks[0]->feed_forward0_.computeInterleaved(
        LearnerCount, condense_interleaved.data(), seq_len, ff0_interleaved);
    dumpInterleavedFloatMatrices(dump_dirs,
                                 "ff0_out.txt",
                                 ff0_interleaved.data(),
                                 seq_len,
                                 ff_size,
                                 LearnerCount);
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        printInterleavedFloatPreview(
            "ffn0_learner" + std::to_string(blocks[learner]->learner_idx_),
            ff0_interleaved.data(),
            seq_len * ff_size,
            LearnerCount,
            learner);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff1", "FF1");

    std::cout << "Feed Forward 1" << std::endl;
    Matrix ff1_interleaved;
    blocks[0]->feed_forward1_.computeInterleaved(
        LearnerCount, ff0_interleaved.data(), seq_len, ff1_interleaved);
    dumpInterleavedFloatMatrices(dump_dirs,
                                 "ff1_out.txt",
                                 ff1_interleaved.data(),
                                 seq_len,
                                 input_dim,
                                 LearnerCount);
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        printInterleavedFloatPreview(
            "ffn1_pre_addnorm_learner" + std::to_string(blocks[learner]->learner_idx_),
            ff1_interleaved.data(),
            seq_len * input_dim,
            LearnerCount,
            learner);
    }
    dumpTransformerStatsCheckpointIfProfiling("after_ff2", "FF2");

    // Step 6: Add the second residual, dump the interleaved final tensor, and
    // split it back into the per-learner output buffers expected by the caller.
    std::cout << "Add Norm" << std::endl;
    blocks[0]->add_norm_.computeInterleaved(
        condense_interleaved.data(), ff1_interleaved.data(), LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs,
                                 "final_out.txt",
                                 ff1_interleaved.data(),
                                 seq_len,
                                 input_dim,
                                 LearnerCount);
    deinterleaveLearnerMatrices(ff1_interleaved, LearnerCount, seq_len, input_dim, outputs);

    dumpTransformerStatsLegacyBoundary("final_total", "non_GEMM_after_ff2");
}

void FloatTransformerBlock::computeGroup2FullInterleaved(std::size_t seq_len,
                                                         FloatTransformerBlock* blocks[2],
                                                         const float* const inputs[2],
                                                         float* const outputs[2]) {
    computeFullInterleavedBlock<2u>(seq_len, blocks, inputs, outputs);
}

void FloatTransformerBlock::computeGroup4FullInterleaved(std::size_t seq_len,
                                                         FloatTransformerBlock* blocks[4],
                                                         const float* const inputs[4],
                                                         float* const outputs[4]) {
    computeFullInterleavedBlock<4u>(seq_len, blocks, inputs, outputs);
}
#endif

void FloatTransformerBlock::computeGroup2(std::size_t seq_len,
                                          FloatTransformerBlock* blocks[2],
                                          const float* const inputs[2],
                                          float* const outputs[2]) {
    computeGroupImpl<2u>(seq_len, blocks, inputs, outputs);
}

void FloatTransformerBlock::computeGroup4(std::size_t seq_len,
                                          FloatTransformerBlock* blocks[4],
                                          const float* const inputs[4],
                                          float* const outputs[4]) {
    computeGroupImpl<4u>(seq_len, blocks, inputs, outputs);
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
