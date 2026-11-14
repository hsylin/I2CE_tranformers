//
// Created by alireza on 3/2/22.
//
#pragma once // New
#include <cstddef>
#include <cstdint>
#include <string>
#include "../Full_NN/inc/codebook_tiles.h"
#include <vector> // New

#include "run_mode_config.h"
#include "selfattention.h"
#include "addNorm.h"
// #include "dense.h"
#include "linearLayer.h"

class TransformerBlock {
  public:
    TransformerBlock(std::size_t pre_seq_len, std::size_t input_dim,
                     std::size_t head_hidden_size, std::size_t num_heads,
                     std::size_t ff_size, uint32_t **weightVector,
                     std::size_t kernelDim, std::size_t maxCol,
                     std::size_t learner_idx = 0, std::string dump_dir = "",
                     const i2ce::RuntimeTiles &runtime_tiles = std::nullopt);

    virtual ~TransformerBlock();

    virtual void compute(std::size_t seq_len, uint32_t *input, uint32_t *output);
    void computeWithoutStatsReset(std::size_t seq_len, uint32_t *input,
                                  uint32_t *output);
    static void computeGroup2(std::size_t seq_len, TransformerBlock *blocks[2],
                              uint32_t *const inputs[2], uint32_t *const outputs[2]);
    static void computeGroup4(std::size_t seq_len, TransformerBlock *blocks[4],
                              uint32_t *const inputs[4], uint32_t *const outputs[4]);

  private:
    template <std::size_t LearnerCount>
    static void computeGroupImpl(std::size_t seq_len, TransformerBlock **blocks,
                                 uint32_t *const *inputs, uint32_t *const *outputs);

#if CFG_FULL_INTERLEAVED_PIPELINE
    static void computeGroup2FullInterleaved(std::size_t seq_len,
                                             TransformerBlock *blocks[2],
                                             uint32_t *const inputs[2],
                                             uint32_t *const outputs[2]);
    static void computeGroup4FullInterleaved(std::size_t seq_len,
                                             TransformerBlock *blocks[4],
                                             uint32_t *const inputs[4],
                                             uint32_t *const outputs[4]);
#endif

  protected:
    void computeWithStatsLabel(std::size_t seq_len, uint32_t *input, uint32_t *output,
                               const char *stats_window_label);
    void computeBody(std::size_t seq_len, uint32_t *input, uint32_t *output);

  private:
    std::size_t num_heads_;
    std::size_t head_hidden_size_;
    std::size_t input_dim_;
    std::size_t ff_size_;
    std::size_t learner_idx_;
    std::string dump_dir_;

    std::vector<SingleHeadSelfAttn *> selfatten_;

    uint32_t *multihead_out;
    uint32_t *condense_out;
    uint32_t *intermediateFF;

#ifndef BWMA
    uint32_t *multihead_out_reshape;
#endif

    AddNormalize *addNorm;

    LinearLayer *condense;
    LinearLayer *feedForward0;
    LinearLayer *feedForward1;

#if CFG_USE_CODEBOOK_REFERENCE
    // LinearLayer* condenseReference;
    // LinearLayer* feedForward0Reference;
    // LinearLayer* feedForward1Reference;

    // uint32_t* referenceCondense;
    // uint32_t* referenceFF0;
    // uint32_t* referenceFF1;

    LinearLayer *condenseReference = nullptr;
    LinearLayer *feedForward0Reference = nullptr;
    LinearLayer *feedForward1Reference = nullptr;

    uint32_t *referenceCondense = nullptr;
    uint32_t *referenceCondenseAfterAddNorm = nullptr;
    uint32_t *referenceFF0 = nullptr;
    uint32_t *referenceFF1 = nullptr;
    uint32_t *referenceFinalOutput = nullptr;
#endif
};

class DenseNoSimdTransformerBlock : public TransformerBlock {
  public:
    using TransformerBlock::TransformerBlock;

    void compute(std::size_t seq_len, uint32_t *input, uint32_t *output) override;
};
