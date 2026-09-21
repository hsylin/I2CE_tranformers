#include "floatCodebookDense.h"

#include "run_mode_config.h"

#if CFG_USE_FP32_TRANSFORMER

#include "../Full_NN/gemm_definitions/generated_codebook_registry.h"
#include "../Full_NN/inc/gemm_exec.h"

#include <stdexcept>
#include <utility>

namespace TransformerFloat {

namespace {

// The generated codebook stores quantized weights as indexes into a codebook.
// bits_per_cb gives the number of bits per index, so the table has 2^bits
// entries.
std::size_t codebookSize(const GeneratedCodebookLayerView& view) {
    return static_cast<std::size_t>(1u) << view.bits_per_cb;
}

// Look up one generated layer by name and make sure the registry contains
// enough learner-specific data for the requested execution mode.
const GeneratedCodebookLayerView& layerView(const std::string& name,
                                            std::size_t required_learners) {
    const GeneratedCodebookLayerView* view = findGeneratedCodebookLayer(name.c_str());
    if (view == nullptr) {
        throw std::runtime_error("FP32 transformer layer not found in registry: " + name);
    }
    if (view->n_learners < required_learners) {
        throw std::runtime_error(
            "FP32 transformer registry has fewer learners than requested for layer: " + name);
    }
    return *view;
}

// Select the compact weight-index stream for one learner. SAME_SEQ layers share
// one index stream across learners; diff-seq layers keep per-learner streams
// packed one after another.
const uint32_t* weightIdxForLearner(const GeneratedCodebookLayerView& view,
                                    std::size_t learner) {
    if (view.same_seq || view.weight_idx_by_learner == nullptr) {
        return view.weight_idx;
    }
    return view.weight_idx_by_learner + learner * getGeneratedWeightIdxCount(view);
}

// Select the int8 codebook for one learner. Some generated layers expose one
// shared codebook, while others expose a codebook array indexed by learner.
const int8_t* codebookForLearner(const GeneratedCodebookLayerView& view,
                                 std::size_t learner) {
    if (view.codebooks_int8 == nullptr) {
        return view.codebook_int8;
    }
    return view.codebooks_int8 + learner * codebookSize(view);
}

// Select the bias vector for one learner, following the same shared/per-learner
// convention used by the generated registry.
const float* biasForLearner(const GeneratedCodebookLayerView& view,
                            std::size_t learner) {
    if (view.biases == nullptr) {
        return view.bias;
    }
    return view.biases + learner * view.output_size;
}

// Return an FP32 codebook for the single-learner path.
//
// Preferred path: use an FP32 codebook emitted by the generator.
// Fallback path: widen the int8 codebook into scratch storage so the FP32
// reference kernels can consume the same registry entry.
const float* codebookFp32ForLearner(const GeneratedCodebookLayerView& view,
                                    std::size_t learner,
                                    Matrix& scratch) {
#ifdef GENERATED_CODEBOOK_REGISTRY_HAS_FP32
    const float* codebook = getGeneratedCodebookFp32(&view, learner);
    if (codebook != nullptr) {
        return codebook;
    }
#endif

    const int8_t* codebook_int8 = codebookForLearner(view, learner);
    if (codebook_int8 == nullptr) {
        return nullptr;
    }

    const std::size_t cb_size = codebookSize(view);
    scratch.resize(cb_size);
    for (std::size_t idx = 0; idx < cb_size; idx++) {
        scratch[idx] = static_cast<float>(codebook_int8[idx]);
    }
    return scratch.data();
}

// Build or retrieve the interleaved FP32 codebook used by grouped kernels.
// Layout is [codebook_entry][learner], so all learners' values for the same
// codebook entry are adjacent in memory.
const float* codebookFp32Interleaved(const GeneratedCodebookLayerView& view,
                                     std::size_t learner_count,
                                     Matrix& scratch) {
#ifdef GENERATED_CODEBOOK_REGISTRY_HAS_FP32
    const float* codebook = getGeneratedCodebookFp32Interleaved(&view);
    if (codebook != nullptr) {
        return codebook;
    }
#endif

    const std::size_t cb_size = codebookSize(view);
    scratch.assign(cb_size * learner_count, 0.0f);
    for (std::size_t cb_idx = 0; cb_idx < cb_size; cb_idx++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            const int8_t* learner_codebook = codebookForLearner(view, learner);
            if (learner_codebook == nullptr) {
                return nullptr;
            }
            scratch[cb_idx * learner_count + learner] =
                static_cast<float>(learner_codebook[cb_idx]);
        }
    }
    return scratch.data();
}

// Return bias values in the same [output_channel][learner] interleaved layout
// used by the grouped FP32 kernels. Missing bias data is represented as nullptr;
// missing per-learner bias values are treated as zero in the scratch fallback.
const float* biasFp32Interleaved(const GeneratedCodebookLayerView& view,
                                 std::size_t learner_count,
                                 Matrix& scratch) {
    if (view.bias_interleaved != nullptr) {
        return view.bias_interleaved;
    }
    if (view.biases == nullptr && view.bias == nullptr) {
        return nullptr;
    }

    scratch.assign(view.output_size * learner_count, 0.0f);
    for (std::size_t out_idx = 0; out_idx < view.output_size; out_idx++) {
        for (std::size_t learner = 0; learner < learner_count; learner++) {
            const float* learner_bias = biasForLearner(view, learner);
            scratch[out_idx * learner_count + learner] =
                (learner_bias == nullptr) ? 0.0f : learner_bias[out_idx];
        }
    }
    return scratch.data();
}

// Convert registry metadata into the compact GEMM descriptor consumed by the
// generated dense-layer kernels. The row count comes from the runtime sequence
// length, while input/output dimensions and packing metadata come from the
// generated registry view.
gemm_t makeGemmLayer(const GeneratedCodebookLayerView& view, std::size_t rows) {
    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(rows);
    layer.input_size = static_cast<uint16_t>(view.input_size);
    layer.output_size = static_cast<uint16_t>(view.output_size);
    layer.n_words_row = static_cast<uint16_t>(view.n_words_row);
    return layer;
}

} // namespace

FloatCodebookDense::FloatCodebookDense(std::string layer_name, std::size_t learner)
    : layer_name_(std::move(layer_name)), learner_(learner) {}

// Execute one generated codebook-dense layer for a single learner.
//
// The layer is used for Q/K/V projections, the attention output projection
// ("condense"), and the two feed-forward projections. The compact GEMM kernel
// reconstructs weights from weight indexes plus codebook values instead of
// reading a full dense weight matrix.
void FloatCodebookDense::compute(std::size_t rows,
                                 const float* input,
                                 Matrix& output) const {
    // Step 1: Fetch the generated layer view and choose the learner-specific
    // weight indexes, codebook, and bias.
    const auto& view = layerView(layer_name_, learner_ + 1u);
    const uint32_t* weight_idx = weightIdxForLearner(view, learner_);
    Matrix codebook_scratch;
    const float* codebook = codebookFp32ForLearner(view, learner_, codebook_scratch);
    const float* bias = biasForLearner(view, learner_);

    if (weight_idx == nullptr || codebook == nullptr) {
        throw std::runtime_error("FP32 transformer registry entry is incomplete: " + layer_name_);
    }

    // Step 2: Allocate the output matrix and call the scalar or SVE compact GEMM
    // implementation selected at build time.
    output.assign(rows * view.output_size, 0.0f);
    const gemm_t layer = makeGemmLayer(view, rows);

#ifdef SIMD
    gemm_exec_compact_sve(layer, input, weight_idx, codebook, bias, output.data(),
                          view.bits_per_cb);
#else
    gemm_exec_compact(layer, input, weight_idx, codebook, bias, output.data(),
                      view.bits_per_cb);
#endif
}

// Execute the same dense layer for a grouped 2D or 4D learner run using
// interleaved buffers. The input and output layout is [matrix_element][learner],
// which lets the GEMM implementation process multiple learners together.
void FloatCodebookDense::computeInterleaved(std::size_t learner_count,
                                            const float* input_interleaved,
                                            std::size_t rows,
                                            Matrix& output_interleaved) const {
    // Step 1: Fetch registry data for all learners and materialize interleaved
    // codebook/bias arrays when the generator did not provide them directly.
    const auto& view = layerView(layer_name_, learner_count);
    output_interleaved.assign(rows * view.output_size * learner_count, 0.0f);

    Matrix codebook_scratch;
    Matrix bias_scratch;
    const float* codebook_interleaved =
        codebookFp32Interleaved(view, learner_count, codebook_scratch);
    const float* bias_interleaved =
        biasFp32Interleaved(view, learner_count, bias_scratch);
    if (codebook_interleaved == nullptr) {
        throw std::runtime_error("FP32 transformer registry entry is incomplete: " + layer_name_);
    }

    const gemm_t layer = makeGemmLayer(view, rows);

    // Step 2: Two-learner grouped FP32 execution currently supports only
    // SAME_SEQ, where all learners share the same weight-index stream.
    if (learner_count == 2u) {
        if (!view.same_seq) {
            throw std::runtime_error(
                "FP32 2-learner interleaved path currently requires SAME_SEQ: " + layer_name_);
        }
#ifdef SIMD
        gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq(
            layer, input_interleaved, view.weight_idx, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
        gemm_exec_compact_fp32_interleaved_2Learners_same_seq(
            layer, input_interleaved, view.weight_idx, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
        return;
    }

    // Step 3: Four-learner execution has both SAME_SEQ and diff-seq kernels. The
    // diff-seq path needs a generated interleaved weight-index stream.
    if (learner_count == 4u) {
        if (view.same_seq) {
#ifdef SIMD
            gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq(
                layer, input_interleaved, view.weight_idx, codebook_interleaved,
                bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
            gemm_exec_compact_fp32_interleaved_4Learners_same_seq(
                layer, input_interleaved, view.weight_idx, codebook_interleaved,
                bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
            return;
        }

        const uint32_t* weight_idx_interleaved =
            getGeneratedCodebookWeightIdxInterleaved(&view);
        if (weight_idx_interleaved == nullptr) {
            throw std::runtime_error(
                "FP32 4D diff-seq path is missing interleaved indexes: " + layer_name_);
        }
#ifdef SIMD
        gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq(
            layer, input_interleaved, weight_idx_interleaved, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#else
        gemm_exec_compact_fp32_interleaved_4Learners_diff_seq(
            layer, input_interleaved, weight_idx_interleaved, codebook_interleaved,
            bias_interleaved, output_interleaved.data(), view.bits_per_cb);
#endif
        return;
    }

    // Step 4: Any other grouped width is rejected because there are no matching
    // compact GEMM kernels in this implementation.
    throw std::runtime_error(
        "FP32 interleaved path supports only 2D or 4D learners: " + layer_name_);
}

std::size_t FloatCodebookDense::inputSize() const {
    return layerView(layer_name_, learner_ + 1u).input_size;
}

std::size_t FloatCodebookDense::outputSize() const {
    return layerView(layer_name_, learner_ + 1u).output_size;
}

} // namespace TransformerFloat

#endif // CFG_USE_FP32_TRANSFORMER
