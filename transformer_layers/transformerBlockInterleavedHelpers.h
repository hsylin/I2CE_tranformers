#pragma once

#include "run_mode_config.h"

#if CFG_FULL_INTERLEAVED_PIPELINE

#include <cstddef>
#include <cstdint>

#include "addNorm.h"
#include "linearLayer.h"

void computeCodebookDenseInterleaved2Learners(const char* label,
                                       LinearLayer* const layers[2],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved);

void computeCodebookDenseInterleaved4Learners(const char* label,
                                       LinearLayer* const layers[4],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved);

#if CFG_ENABLE_DEBUG_PRINT
void printInterleavedPackedPreview2D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner);

void printInterleavedPackedPreview4D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
void compareInterleavedDenseReference2D(const char* label,
                                        LinearLayer* const references[2],
                                        uint32_t* const reference_outputs[2],
                                        const std::size_t learner_ids[2],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved);

void compareInterleavedDenseReference4D(const char* label,
                                        LinearLayer* const references[4],
                                        uint32_t* const reference_outputs[4],
                                        const std::size_t learner_ids[4],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved);

void compareInterleavedAddNormReference2D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[2],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved);

void compareInterleavedAddNormReference4D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[4],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved);
#endif

#endif
