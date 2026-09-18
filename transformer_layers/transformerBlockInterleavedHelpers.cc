#include "transformerBlockInterleavedHelpers.h"

#if CFG_FULL_INTERLEAVED_PIPELINE

#include "codebookDense.h"
#include "debuggerFunctions.h"
#include "interleavedCodebookDenseValidator.h"
#include "interleavedPipeline.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transformer_internal::requireInterleavedCodebookDense2;
using transformer_internal::requireInterleavedCodebookDense4;

} // namespace

/**
 * Run a two-learner CodebookDense layer on interleaved int8 activations.
 *
 * @param label Layer name used in validation error messages.
 * @param layers Two learner-specific layer pointers from the layer factory.
 *        They are checked for CodebookDense 2D interleaved support.
 * @param seq_len Number of sequence rows to process.
 * @param input_interleaved Input activation buffer laid out as
 *        [seq][input_feature][learner] with two learner lanes.
 * @param output_interleaved Destination buffer laid out as
 *        [seq][output_feature][learner] with two learner lanes.
 */
void computeCodebookDenseInterleaved2D(const char* label,
                                       LinearLayer* const layers[2],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense2(label, layers);
    primary->computeInterleaved2DToInt8(seq_len, input_interleaved, output_interleaved);
}

/**
 * Run a four-learner CodebookDense layer on interleaved int8 activations.
 *
 * @param label Layer name used in validation error messages.
 * @param layers Four learner-specific layer pointers from the layer factory.
 *        They are checked for CodebookDense 4D interleaved support.
 * @param seq_len Number of sequence rows to process.
 * @param input_interleaved Input activation buffer laid out as
 *        [seq][input_feature][learner] with four learner lanes.
 * @param output_interleaved Destination buffer laid out as
 *        [seq][output_feature][learner] with four learner lanes.
 */
void computeCodebookDenseInterleaved4D(const char* label,
                                       LinearLayer* const layers[4],
                                       std::size_t seq_len,
                                       const int8_t* input_interleaved,
                                       int8_t* output_interleaved) {
    CodebookDense* primary = requireInterleavedCodebookDense4(label, layers);
    primary->computeInterleaved4DToInt8(seq_len, input_interleaved, output_interleaved);
}

#if CFG_ENABLE_DEBUG_PRINT
/**
 * Print one learner's packed view of a 2D interleaved int8 buffer.
 *
 * @param label Prefix passed to printPackedPreview.
 * @param input_interleaved Source buffer laid out as [row][col][learner].
 * @param rows Number of logical matrix rows.
 * @param cols Number of logical matrix columns.
 * @param learner Learner lane to extract before packing and printing.
 */
void printInterleavedPackedPreview2D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner) {
    std::vector<uint32_t> packed((rows * cols) >> 2, 0u);
    packInterleavedLearner2(rows, cols, input_interleaved, learner, packed.data());
    printPackedPreview(label, packed.data(), packed.size());
}

/**
 * Print one learner's packed view of a 4D interleaved int8 buffer.
 *
 * @param label Prefix passed to printPackedPreview.
 * @param input_interleaved Source buffer laid out as [row][col][learner].
 * @param rows Number of logical matrix rows.
 * @param cols Number of logical matrix columns.
 * @param learner Learner lane to extract before packing and printing.
 */
void printInterleavedPackedPreview4D(const char* label,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols,
                                     std::size_t learner) {
    std::vector<uint32_t> packed((rows * cols) >> 2, 0u);
    packInterleavedLearner4(rows, cols, input_interleaved, learner, packed.data());
    printPackedPreview(label, packed.data(), packed.size());
}
#endif

#if CFG_USE_CODEBOOK_REFERENCE
/**
 * Compare a two-learner interleaved CodebookDense result against Dense output.
 *
 * @param label Base label used in comparison messages.
 * @param references Dense reference layer for each of the two learner lanes.
 * @param reference_outputs Scratch output buffers, one per learner. Each buffer
 *        must hold (seq_len * output_cols) / 4 packed uint32_t words.
 * @param learner_ids Model learner identifiers used only to make comparison
 *        labels match the original learner numbering.
 * @param seq_len Number of sequence rows in the input/candidate matrices.
 * @param input_cols Number of input features per row before packing.
 * @param output_cols Number of output features per row before packing.
 * @param input_interleaved CodebookDense input laid out as
 *        [seq][input_feature][learner].
 * @param candidate_interleaved CodebookDense output laid out as
 *        [seq][output_feature][learner].
 */
void compareInterleavedDenseReference2D(const char* label,
                                        LinearLayer* const references[2],
                                        uint32_t* const reference_outputs[2],
                                        const std::size_t learner_ids[2],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_input((seq_len * input_cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * output_cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 2u; learner++) {
        std::fill(reference_outputs[learner],
                  reference_outputs[learner] + ((seq_len * output_cols) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner2(
            seq_len,
            input_cols,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner2(
            seq_len,
            output_cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        references[learner]->compute(
            seq_len,
            packed_input.data(),
            reference_outputs[learner]);

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            reference_outputs[learner],
            packed_candidate.data(),
            (seq_len * output_cols) >> 2);
    }
}

/**
 * Compare a four-learner interleaved CodebookDense result against Dense output.
 *
 * @param label Base label used in comparison messages.
 * @param references Dense reference layer for each of the four learner lanes.
 * @param reference_outputs Scratch output buffers, one per learner. Each buffer
 *        must hold (seq_len * output_cols) / 4 packed uint32_t words.
 * @param learner_ids Model learner identifiers used only to make comparison
 *        labels match the original learner numbering.
 * @param seq_len Number of sequence rows in the input/candidate matrices.
 * @param input_cols Number of input features per row before packing.
 * @param output_cols Number of output features per row before packing.
 * @param input_interleaved CodebookDense input laid out as
 *        [seq][input_feature][learner].
 * @param candidate_interleaved CodebookDense output laid out as
 *        [seq][output_feature][learner].
 */
void compareInterleavedDenseReference4D(const char* label,
                                        LinearLayer* const references[4],
                                        uint32_t* const reference_outputs[4],
                                        const std::size_t learner_ids[4],
                                        std::size_t seq_len,
                                        std::size_t input_cols,
                                        std::size_t output_cols,
                                        const int8_t* input_interleaved,
                                        const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_input((seq_len * input_cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * output_cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 4u; learner++) {
        std::fill(reference_outputs[learner],
                  reference_outputs[learner] + ((seq_len * output_cols) >> 2),
                  0u);
        std::fill(packed_input.begin(), packed_input.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner4(
            seq_len,
            input_cols,
            input_interleaved,
            learner,
            packed_input.data());
        packInterleavedLearner4(
            seq_len,
            output_cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        references[learner]->compute(
            seq_len,
            packed_input.data(),
            reference_outputs[learner]);

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            reference_outputs[learner],
            packed_candidate.data(),
            (seq_len * output_cols) >> 2);
    }
}

/**
 * Compare a two-learner interleaved AddNorm result against scalar AddNorm.
 *
 * @param label Base label used in comparison messages.
 * @param add_norm Reference AddNormalize implementation. It updates its second
 *        packed buffer argument in place.
 * @param learner_ids Model learner identifiers used only for comparison labels.
 * @param seq_len Number of sequence rows in the matrices.
 * @param cols Number of features per row before packing.
 * @param residual_interleaved Residual input laid out as [seq][col][learner].
 * @param pre_addnorm_interleaved Candidate's pre-AddNorm value laid out as
 *        [seq][col][learner]; each learner is packed and used as the mutable
 *        reference buffer.
 * @param candidate_interleaved Interleaved AddNorm result to compare against
 *        the reference output.
 */
void compareInterleavedAddNormReference2D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[2],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_residual((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_reference((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 2u; learner++) {
        std::fill(packed_residual.begin(), packed_residual.end(), 0u);
        std::fill(packed_reference.begin(), packed_reference.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner2(
            seq_len,
            cols,
            residual_interleaved,
            learner,
            packed_residual.data());
        packInterleavedLearner2(
            seq_len,
            cols,
            pre_addnorm_interleaved,
            learner,
            packed_reference.data());
        packInterleavedLearner2(
            seq_len,
            cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        add_norm->compute(packed_residual.data(), packed_reference.data());

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            packed_reference.data(),
            packed_candidate.data(),
            (seq_len * cols) >> 2);
    }
}

/**
 * Compare a four-learner interleaved AddNorm result against scalar AddNorm.
 *
 * @param label Base label used in comparison messages.
 * @param add_norm Reference AddNormalize implementation. It updates its second
 *        packed buffer argument in place.
 * @param learner_ids Model learner identifiers used only for comparison labels.
 * @param seq_len Number of sequence rows in the matrices.
 * @param cols Number of features per row before packing.
 * @param residual_interleaved Residual input laid out as [seq][col][learner].
 * @param pre_addnorm_interleaved Candidate's pre-AddNorm value laid out as
 *        [seq][col][learner]; each learner is packed and used as the mutable
 *        reference buffer.
 * @param candidate_interleaved Interleaved AddNorm result to compare against
 *        the reference output.
 */
void compareInterleavedAddNormReference4D(const char* label,
                                          AddNormalize* add_norm,
                                          const std::size_t learner_ids[4],
                                          std::size_t seq_len,
                                          std::size_t cols,
                                          const int8_t* residual_interleaved,
                                          const int8_t* pre_addnorm_interleaved,
                                          const int8_t* candidate_interleaved) {
    std::vector<uint32_t> packed_residual((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_reference((seq_len * cols) >> 2, 0u);
    std::vector<uint32_t> packed_candidate((seq_len * cols) >> 2, 0u);

    for (std::size_t learner = 0; learner < 4u; learner++) {
        std::fill(packed_residual.begin(), packed_residual.end(), 0u);
        std::fill(packed_reference.begin(), packed_reference.end(), 0u);
        std::fill(packed_candidate.begin(), packed_candidate.end(), 0u);

        packInterleavedLearner4(
            seq_len,
            cols,
            residual_interleaved,
            learner,
            packed_residual.data());
        packInterleavedLearner4(
            seq_len,
            cols,
            pre_addnorm_interleaved,
            learner,
            packed_reference.data());
        packInterleavedLearner4(
            seq_len,
            cols,
            candidate_interleaved,
            learner,
            packed_candidate.data());

        add_norm->compute(packed_residual.data(), packed_reference.data());

        const std::string learner_label =
            std::string(label) + "_learner" + std::to_string(learner_ids[learner]);
        comparePackedBuffers(
            learner_label.c_str(),
            packed_reference.data(),
            packed_candidate.data(),
            (seq_len * cols) >> 2);
    }
}
#endif

#endif
