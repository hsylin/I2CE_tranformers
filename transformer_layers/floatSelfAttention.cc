#include "floatSelfAttention.h"

#include "floatDump.h"
#include "run_mode_config.h"

#if CFG_USE_FP32_TRANSFORMER

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace TransformerFloat {

namespace {

// Matrix multiply for attention scores: output = lhs * transpose(rhs_rows).
// Both inputs are stored row-major. In self-attention this computes Q * K^T,
// producing one score for every query-token/key-token pair.
void matmulTransposedRhs(const float* lhs,
                         const float* rhs_rows,
                         std::size_t lhs_rows,
                         std::size_t rhs_rows_count,
                         std::size_t k_elems,
                         Matrix& output) {
    output.assign(lhs_rows * rhs_rows_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_rows_count; col++) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < k_elems; k++) {
                acc += lhs[row * k_elems + k] * rhs_rows[col * k_elems + k];
            }
            output[row * rhs_rows_count + col] = acc;
        }
    }
}

// Standard row-major matrix multiply: output = lhs * rhs_rows.
// In self-attention this applies the softmax probabilities to V, producing the
// final output for one attention head.
void matmulRows(const float* lhs,
                const float* rhs_rows,
                std::size_t lhs_rows,
                std::size_t rhs_cols,
                std::size_t k_elems,
                Matrix& output) {
    output.assign(lhs_rows * rhs_cols, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            float acc = 0.0f;
            for (std::size_t k = 0; k < k_elems; k++) {
                acc += lhs[row * k_elems + k] * rhs_rows[k * rhs_cols + col];
            }
            output[row * rhs_cols + col] = acc;
        }
    }
}

// Interleaved version of Q * K^T. Each logical matrix element stores all learner
// values next to each other:
//   buffer[(row * cols + col) * learner_count + learner]
// The math is still independent per learner; only the memory layout changes.
void matmulInterleavedTransposedRhs(const float* lhs_interleaved,
                                    const float* rhs_rows_interleaved,
                                    std::size_t lhs_rows,
                                    std::size_t rhs_rows_count,
                                    std::size_t k_elems,
                                    std::size_t learner_count,
                                    Matrix& output_interleaved) {
    output_interleaved.assign(lhs_rows * rhs_rows_count * learner_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_rows_count; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < k_elems; k++) {
                    const float lhs =
                        lhs_interleaved[((row * k_elems) + k) * learner_count + learner];
                    const float rhs =
                        rhs_rows_interleaved[((col * k_elems) + k) * learner_count + learner];
                    acc += lhs * rhs;
                }
                output_interleaved[((row * rhs_rows_count) + col) * learner_count + learner] =
                    acc;
            }
        }
    }
}

// Interleaved version of output = lhs * rhs_rows, used for softmax(QK) * V in
// grouped 2D/4D runs.
void matmulInterleavedRows(const float* lhs_interleaved,
                           const float* rhs_rows_interleaved,
                           std::size_t lhs_rows,
                           std::size_t rhs_cols,
                           std::size_t k_elems,
                           std::size_t learner_count,
                           Matrix& output_interleaved) {
    output_interleaved.assign(lhs_rows * rhs_cols * learner_count, 0.0f);
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                float acc = 0.0f;
                for (std::size_t k = 0; k < k_elems; k++) {
                    const float lhs =
                        lhs_interleaved[((row * k_elems) + k) * learner_count + learner];
                    const float rhs =
                        rhs_rows_interleaved[((k * rhs_cols) + col) * learner_count + learner];
                    acc += lhs * rhs;
                }
                output_interleaved[((row * rhs_cols) + col) * learner_count + learner] = acc;
            }
        }
    }
}

} // namespace

FloatSingleHeadSelfAttn::FloatSingleHeadSelfAttn(std::size_t head_idx,
                                                 std::size_t pre_seq_len,
                                                 std::size_t input_dim,
                                                 std::size_t head_hidden_size,
                                                 std::size_t learner_idx,
                                                 std::string dump_dir)
    : head_idx_(head_idx),
      pre_seq_len_(pre_seq_len),
      input_dim_(input_dim),
      head_hidden_size_(head_hidden_size),
      learner_idx_(learner_idx),
      dump_dir_(std::move(dump_dir)),
      query_layer_("q_h" + std::to_string(head_idx), learner_idx),
      key_layer_("k_h" + std::to_string(head_idx), learner_idx),
      value_layer_("v_h" + std::to_string(head_idx), learner_idx) {}

// Compute one single-head self-attention layer for one learner.
//
// The head projects the input into query, key, and value matrices, computes
// scaled dot-product attention, and returns one [seq_len x head_hidden_size]
// output matrix for the transformer block to concatenate with other heads.
void FloatSingleHeadSelfAttn::compute(std::size_t seq_len,
                                      const float* input,
                                      Matrix& output) const {
    Matrix query;
    Matrix key;
    Matrix value;
    Matrix scores;

    // Step 1: Generate Q, K, and V using the codebook dense projections assigned
    // to this head and learner.
    query_layer_.compute(seq_len, input, query);
    key_layer_.compute(seq_len, input, key);
    value_layer_.compute(seq_len, input, value);

    const std::string head_suffix = std::to_string(head_idx_);
    dumpFloatMatrixIfEnabled(dump_dir_, "q_h" + head_suffix + ".txt",
                             query.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "k_h" + head_suffix + ".txt",
                             key.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "v_h" + head_suffix + ".txt",
                             value.data(), seq_len, head_hidden_size_);

    // Step 2: Compute raw attention scores with Q * K^T. Rows are query tokens
    // and columns are key tokens.
    matmulTransposedRhs(query.data(), key.data(), seq_len, seq_len, head_hidden_size_, scores);
    dumpFloatMatrixIfEnabled(dump_dir_, "qk_scores_h" + head_suffix + ".txt",
                             scores.data(), seq_len, seq_len);

    // Step 3: Scale by 1/sqrt(head_hidden_size) and apply row-wise softmax so
    // each query token receives a probability distribution over key tokens.
    softmax_.compute(scores.data(), seq_len, seq_len,
                     1.0f / std::sqrt(static_cast<float>(head_hidden_size_)));
    dumpFloatMatrixIfEnabled(dump_dir_, "softmax_qk_h" + head_suffix + ".txt",
                             scores.data(), seq_len, seq_len);

    // Step 4: Multiply the attention probabilities by V to produce this head's
    // contextualized output.
    matmulRows(scores.data(), value.data(), seq_len, head_hidden_size_, seq_len, output);
    dumpFloatMatrixIfEnabled(dump_dir_, "softmax_v_pre_post_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "head_out_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
    dumpFloatMatrixIfEnabled(dump_dir_, "head_out_post_h" + head_suffix + ".txt",
                             output.data(), seq_len, head_hidden_size_);
}

template <std::size_t LearnerCount>
void FloatSingleHeadSelfAttn::computeGroupImpl(std::size_t seq_len,
                                               FloatSingleHeadSelfAttn** heads,
                                               const float* const* inputs,
                                               Matrix* outputs) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner grouped FP32 attention is supported");

    // Grouped non-interleaved attention keeps one matrix per learner and simply
    // runs the single-learner attention path for each learner in the group.
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        heads[learner]->compute(seq_len, inputs[learner], outputs[learner]);
    }
}

void FloatSingleHeadSelfAttn::computeGroup2(std::size_t seq_len,
                                            FloatSingleHeadSelfAttn* heads[2],
                                            const float* const inputs[2],
                                            Matrix outputs[2]) {
    computeGroupImpl<2u>(seq_len, heads, inputs, outputs);
}

void FloatSingleHeadSelfAttn::computeGroup4(std::size_t seq_len,
                                            FloatSingleHeadSelfAttn* heads[4],
                                            const float* const inputs[4],
                                            Matrix outputs[4]) {
    computeGroupImpl<4u>(seq_len, heads, inputs, outputs);
}

template <std::size_t LearnerCount>
void FloatSingleHeadSelfAttn::computeInterleavedImpl(std::size_t seq_len,
                                                     FloatSingleHeadSelfAttn** heads,
                                                     const float* input_interleaved,
                                                     Matrix& output_interleaved) {
    static_assert(LearnerCount == 2u || LearnerCount == 4u,
                  "Only 2- and 4-learner interleaved FP32 attention is supported");

    // Interleaved attention performs the same operations as compute(), but every
    // intermediate tensor keeps all learners packed together at each logical
    // matrix element. Only heads[0] is needed to call shared interleaved dense
    // layers because the registry view contains all learners.
    std::vector<std::string> dump_dirs(LearnerCount);
    for (std::size_t learner = 0; learner < LearnerCount; learner++) {
        dump_dirs[learner] = heads[learner]->dump_dir_;
    }

    const std::size_t head_hidden_size = heads[0]->head_hidden_size_;
    const std::string head_suffix = std::to_string(heads[0]->head_idx_);

    Matrix query;
    Matrix key;
    Matrix value;
    // Step 1: Project interleaved input into interleaved Q, K, and V tensors.
    heads[0]->query_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, query);
    heads[0]->key_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, key);
    heads[0]->value_layer_.computeInterleaved(
        LearnerCount, input_interleaved, seq_len, value);

    dumpInterleavedFloatMatrices(dump_dirs, "q_h" + head_suffix + ".txt",
                                 query.data(), seq_len, head_hidden_size, LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "k_h" + head_suffix + ".txt",
                                 key.data(), seq_len, head_hidden_size, LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "v_h" + head_suffix + ".txt",
                                 value.data(), seq_len, head_hidden_size, LearnerCount);

    Matrix scores;
    // Step 2: Compute interleaved Q * K^T scores independently for each learner.
    matmulInterleavedTransposedRhs(query.data(), key.data(), seq_len, seq_len,
                                   head_hidden_size, LearnerCount, scores);
    dumpInterleavedFloatMatrices(dump_dirs, "qk_scores_h" + head_suffix + ".txt",
                                 scores.data(), seq_len, seq_len, LearnerCount);

    // Step 3: Apply scaled softmax per row and per learner.
    heads[0]->softmax_.computeInterleaved(
        scores.data(), seq_len, seq_len, LearnerCount,
        1.0f / std::sqrt(static_cast<float>(head_hidden_size)));
    dumpInterleavedFloatMatrices(dump_dirs, "softmax_qk_h" + head_suffix + ".txt",
                                 scores.data(), seq_len, seq_len, LearnerCount);

    // Step 4: Multiply interleaved attention probabilities by interleaved V to
    // produce this head's interleaved output.
    matmulInterleavedRows(scores.data(), value.data(), seq_len, head_hidden_size,
                          seq_len, LearnerCount, output_interleaved);
    dumpInterleavedFloatMatrices(dump_dirs, "softmax_v_pre_post_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "head_out_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
    dumpInterleavedFloatMatrices(dump_dirs, "head_out_post_h" + head_suffix + ".txt",
                                 output_interleaved.data(), seq_len, head_hidden_size,
                                 LearnerCount);
}

void FloatSingleHeadSelfAttn::computeInterleaved2Learners(std::size_t seq_len,
                                                   FloatSingleHeadSelfAttn* heads[2],
                                                   const float* input_interleaved,
                                                   Matrix& output_interleaved) {
    computeInterleavedImpl<2u>(seq_len, heads, input_interleaved, output_interleaved);
}

void FloatSingleHeadSelfAttn::computeInterleaved4Learners(std::size_t seq_len,
                                                   FloatSingleHeadSelfAttn* heads[4],
                                                   const float* input_interleaved,
                                                   Matrix& output_interleaved) {
    computeInterleavedImpl<4u>(seq_len, heads, input_interleaved, output_interleaved);
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
