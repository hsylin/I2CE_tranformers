#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

#include "util.h"

#include "run_mode_config.h"
// #include "dense.h"
#include "linearLayer.h"
#include "softmax.h"
#include "transpose.h"
#include "../accelerator/smm_gem.h"


class SingleHeadSelfAttn {
public:
    SingleHeadSelfAttn(std::size_t head_idx,
                       std::size_t pre_seq_len,
                       std::size_t input_dim,
                       std::size_t head_hidden_size,
                       uint32_t** weightVector,
                       std::size_t kernel_dim,
                       std::size_t max_col,
                       std::size_t learner_idx = 0,
                       std::string dump_dir = "");

    ~SingleHeadSelfAttn();

    void compute(std::size_t seq_len, uint32_t* input, uint32_t* output);
    static void computeGroup2(std::size_t seq_len,
                              SingleHeadSelfAttn* heads[2],
                              uint32_t* const inputs[2],
                              uint32_t* const outputs[2]);
    static void computeGroup4(std::size_t seq_len,
                              SingleHeadSelfAttn* heads[4],
                              uint32_t* const inputs[4],
                              uint32_t* const outputs[4]);
    static void computeInterleaved2Learners(std::size_t seq_len,
                                     SingleHeadSelfAttn* heads[2],
                                     const int8_t* input_interleaved,
                                     int8_t* output_interleaved);
    static void computeInterleaved4Learners(std::size_t seq_len,
                                     SingleHeadSelfAttn* heads[4],
                                     const int8_t* input_interleaved,
                                     int8_t* output_interleaved);

private:
    template <std::size_t LearnerCount>
    static void computeGroupImpl(std::size_t seq_len,
                                 SingleHeadSelfAttn** heads,
                                 uint32_t* const* inputs,
                                 uint32_t* const* outputs);

    std::size_t head_idx_;
    std::size_t pre_seq_len_;
    std::size_t head_hidden_size_;
    std::size_t kernel_size_;
    std::size_t max_col_;
    std::size_t input_dim_;
    std::size_t learner_idx_;
    std::string dump_dir_;

    LinearLayer* query_layer_;
    LinearLayer* key_layer_;
    LinearLayer* value_layer_;

#if CFG_USE_CODEBOOK_REFERENCE
    LinearLayer* query_reference_;
    LinearLayer* key_reference_;
    LinearLayer* value_reference_;

    uint32_t* query_reference_out_;
    uint32_t* key_reference_out_;
    uint32_t* value_reference_out_;
#endif

    Softmax* softmax_;

    uint32_t* query_layer_out_;
    uint32_t* key_layer_out_;
    uint32_t* key_transposed_layer_out_;
    uint32_t* value_layer_out_;
    uint32_t* attention_scores_;
};
