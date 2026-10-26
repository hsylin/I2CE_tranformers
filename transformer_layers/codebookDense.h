#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "linearLayer.h"

struct CodebookDenseConfig {
    std::size_t input_size;
    std::size_t output_size;
    std::size_t n_words_row;
    uint8_t bits_per_cb;
    std::size_t n_learners = 1;
    bool same_seq = true;
    std::size_t selected_learner = 0;
    const uint32_t *weight_idx;
    const uint32_t *weight_idx_by_learner = nullptr;
    const uint32_t *weight_idx_interleaved = nullptr;
    const float *codebook = nullptr;
    const int8_t *codebook_int8 = nullptr;
    const int8_t *codebooks_int8 = nullptr;
    const int8_t *codebook_int8_interleaved = nullptr;
    const float *bias = nullptr;
    const float *biases = nullptr;
    const float *bias_interleaved = nullptr;
    float input_dequant_scale = 1.0f;
    float output_quant_scale = 1.0f;
};

class CodebookDense : public LinearLayer {
public:
    explicit CodebookDense(const CodebookDenseConfig &config);
    ~CodebookDense() override = default;

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) override;
    bool supportsInterleaved2LearnersSameSeq() const;
    void computeInterleaved2LearnersSameSeq(std::size_t seq_len,
                                     uint32_t* const inputs[2],
                                     uint32_t* const outputs[2]) const;
    void computeInterleaved2LearnersToInt8(std::size_t seq_len,
                                    const int8_t* input_interleaved,
                                    int8_t* output_interleaved) const;
    bool supportsInterleaved4Learners() const;
    void computeInterleaved4Learners(std::size_t seq_len,
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]) const;
    void computeInterleaved4LearnersToInt8(std::size_t seq_len,
                                    const int8_t* input_interleaved,
                                    int8_t* output_interleaved) const;

private:
    static uint32_t getPackedIndex(const uint32_t *packed_row,
                                   std::size_t elem_idx,
                                   uint8_t bits_per_cb);
    static int8_t unpackInt8(const uint32_t *packed, std::size_t elem_idx);
    static void packInt8(const std::vector<int8_t> &src, uint32_t *dst);
    static int8_t clampInt32ToInt8(int32_t value);

    void runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const;
    void buildInterleavedCachesIfNeeded();

    std::size_t input_size_;
    std::size_t output_size_;
    std::size_t n_words_row_;
    std::size_t n_learners_;
    bool same_seq_;
    std::size_t selected_learner_;
    uint8_t bits_per_cb_;
    const uint32_t *weight_idx_;
    const uint32_t *weight_idx_by_learner_;
    const uint32_t *weight_idx_interleaved_;
    const float *codebook_; // Redundant
    const int8_t *codebooks_int8_;
    const int8_t *codebook_int8_interleaved_;
    const float *bias_;
    const float *biases_;
    const float *bias_interleaved_;

    // Reserved for future mixed-scale path; currently compact int GEMM consumes int8 directly.
    float input_dequant_scale_;
    float output_quant_scale_;

    // For integer weights
    std::vector<int8_t> codebook_q_;
    std::vector<int8_t> codebooks_q_;
    std::vector<int8_t> codebook_interleaved_q_;
    std::vector<int32_t> bias_q_;
    std::vector<int32_t> biases_q_;
    std::vector<int32_t> bias_interleaved_q_;
    std::vector<uint32_t> weight_idx_interleaved_cache_;
};
