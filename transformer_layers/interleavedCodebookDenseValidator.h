// Shared validators for the fused interleaved CodebookDense pipeline.
//
// selfattention.cc and transformerBlockInterleavedHelpers.cc historically
// each carried an anonymous-namespace copy of these two functions with
// identical bodies. The validators are inline in this header so both
// translation units observe exactly one definition and so any future change
// to the validation contract propagates without silent drift.
//
// The helpers are kept internal to the Transformer implementation by living
// in the namespace transformer_internal and being included only from
// transformer_layers/*.cc.
#pragma once

#include "codebookDense.h"
#include "linearLayer.h"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace transformer_internal {

// Validate that two learner projection layers can run the 2D SAME_SEQ
// interleaved CodebookDense path. All learner entries are checked so a
// partially constructed layer bundle fails early rather than crashing
// inside the kernel. Returns layers[0] as a CodebookDense*, which owns
// the interleaved compute entry point.
inline CodebookDense* requireInterleavedCodebookDense2(
    const char* label, LinearLayer* const layers[2]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved2DSameSeq()) {
        throw std::runtime_error(std::string(label) +
                                 " does not support the 2D interleaved pipeline");
    }

    for (std::size_t learner = 1; learner < 2u; learner++) {
        auto* layer = dynamic_cast<CodebookDense*>(layers[learner]);
        if (layer == nullptr || !layer->supportsInterleaved2DSameSeq()) {
            throw std::runtime_error(std::string(label) +
                                     " learner layer does not support the 2D interleaved pipeline");
        }
    }

    return primary;
}

// Validate that four learner projection layers can run the 4D DIFF_SEQ
// interleaved CodebookDense path (which also serves SAME_SEQ at runtime;
// see CodebookDense::computeInterleaved4DDiffSeq). Every learner entry is
// checked. Returns layers[0] as a CodebookDense*, which owns the
// interleaved compute entry point.
inline CodebookDense* requireInterleavedCodebookDense4(
    const char* label, LinearLayer* const layers[4]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved4DDiffSeq()) {
        throw std::runtime_error(std::string(label) +
                                 " does not support the 4D interleaved pipeline");
    }

    for (std::size_t learner = 1; learner < 4u; learner++) {
        auto* layer = dynamic_cast<CodebookDense*>(layers[learner]);
        if (layer == nullptr || !layer->supportsInterleaved4DDiffSeq()) {
            throw std::runtime_error(std::string(label) +
                                     " learner layer does not support the 4D interleaved pipeline");
        }
    }

    return primary;
}

}  // namespace transformer_internal
