#pragma once

#include <cstddef>
#include <string>

#include "floatCodebookDense.h"
#include "floatCommon.h"
#include "floatSoftmax.h"

namespace TransformerFloat {

class FloatSingleHeadSelfAttn {
public:
    FloatSingleHeadSelfAttn(std::size_t head_idx,
                            std::size_t pre_seq_len,
                            std::size_t input_dim,
                            std::size_t head_hidden_size,
                            std::size_t learner_idx = 0,
                            std::string dump_dir = "");

    void compute(std::size_t seq_len, const float* input, Matrix& output) const;

    static void computeGroup2(std::size_t seq_len,
                              FloatSingleHeadSelfAttn* heads[2],
                              const float* const inputs[2],
                              Matrix outputs[2]);
    static void computeGroup4(std::size_t seq_len,
                              FloatSingleHeadSelfAttn* heads[4],
                              const float* const inputs[4],
                              Matrix outputs[4]);

    static void computeInterleaved2Learners(std::size_t seq_len,
                                     FloatSingleHeadSelfAttn* heads[2],
                                     const float* input_interleaved,
                                     Matrix& output_interleaved);
    static void computeInterleaved4Learners(std::size_t seq_len,
                                     FloatSingleHeadSelfAttn* heads[4],
                                     const float* input_interleaved,
                                     Matrix& output_interleaved);

    std::size_t headIndex() const { return head_idx_; }
    std::size_t headHiddenSize() const { return head_hidden_size_; }
    const std::string& dumpDir() const { return dump_dir_; }

private:
    template <std::size_t LearnerCount>
    static void computeGroupImpl(std::size_t seq_len,
                                 FloatSingleHeadSelfAttn** heads,
                                 const float* const* inputs,
                                 Matrix* outputs);

    template <std::size_t LearnerCount>
    static void computeInterleavedImpl(std::size_t seq_len,
                                       FloatSingleHeadSelfAttn** heads,
                                       const float* input_interleaved,
                                       Matrix& output_interleaved);

    std::size_t head_idx_;
    std::size_t pre_seq_len_;
    std::size_t input_dim_;
    std::size_t head_hidden_size_;
    std::size_t learner_idx_;
    std::string dump_dir_;

    FloatCodebookDense query_layer_;
    FloatCodebookDense key_layer_;
    FloatCodebookDense value_layer_;
    FloatSoftmax softmax_;
};

} // namespace TransformerFloat
