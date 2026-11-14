#include "layerFactory.h"

#include <iostream>
#include <stdexcept>
#include <string>

#include "dense.h"
#include "run_mode_config.h"
#include "../Full_NN/inc/codebook_tiles.h"

#if CFG_USE_CODEBOOK_GEMM
#include "registry_adapter.h"
#include "codebookDense.h"
#endif

namespace {

#if CFG_USE_CODEBOOK_GEMM
/**
 * Build and validate the CodebookDense configuration for one named layer.
 *
 * @param layer_name Registry key for the generated layer weights, usually the
 *        model layer name used when exporting the generated codebook registry.
 * @param expected_input_size Input feature count expected by the caller. This
 *        is checked against the registry so a wrong layer entry cannot be used
 *        with a differently shaped activation matrix.
 * @param expected_output_size Output feature count expected by the caller.
 *        This must match the registry entry because the GEMM output buffer is
 *        sized from this value.
 * @param learner Learner index to select from a multi-learner codebook layer.
 *        It chooses the learner-specific weight indices, codebook, and bias
 *        when the registry stores separate data per learner.
 *
 * @return A CodebookDenseConfig populated from the generated registry.
 *
 * @throws std::invalid_argument if the registry shape does not match the
 *         caller's expected input/output sizes.
 * @throws std::runtime_error if the layer or selected learner is missing from
 *         the generated registry.
 */
CodebookDenseConfig makeCheckedCodebookConfig(const char *layer_name,
                                              std::size_t expected_input_size,
                                              std::size_t expected_output_size,
                                              std::size_t learner,
                                              const i2ce::RuntimeTiles &runtime_tiles) {

    CodebookDenseConfig cfg = makeCodebookDenseConfigFromRegistry(layer_name, learner);
    cfg.runtime_tiles = runtime_tiles;

    if (cfg.input_size != expected_input_size ||
        cfg.output_size != expected_output_size) {
        throw std::invalid_argument("Codebook registry shape mismatch for layer: " +
                                    std::string(layer_name));
    }

    const bool legacy_compressed_path =
        cfg.bits_per_cb == 2u && cfg.n_learners == 2u && cfg.same_seq;
    if (legacy_compressed_path || runtime_tiles.has_value()) {
        cfg.compressed_index_directory = "weights/compressed_indices";
    }
    return cfg;
}
#endif

} // namespace

/**
 * Create the implementation objects used for a transformer linear layer.
 *
 * @param layer_name Human-readable layer name and, when codebook GEMM is
 *        enabled, the lookup key into the generated CodebookDense registry.
 * @param input_size Number of input features per sequence position. Dense
 *        fallback and CodebookDense validation both use this to size/check the
 *        matrix multiply.
 * @param output_size Number of output features per sequence position. The
 *        output activation buffer must be sized for seq_len * output_size
 *        logical int8 values by the caller.
 * @param fallback_weight Pointer to the packed Dense weight matrix used when
 *        CodebookDense is disabled, when a registry entry is unavailable and
 *        fallback is allowed, or when a Dense reference path is requested.
 * @param learner Learner index for multi-learner runs. This selects the
 *        learner-specific CodebookDense data and identifies which Dense
 *        fallback/reference weights correspond to this layer instance.
 *
 * @return A bundle whose main pointer is the execution layer. The reference
 *         pointer is populated only when CFG_USE_CODEBOOK_REFERENCE is enabled.
 */
LinearLayerBundle LayerFactory::create(const std::string &layer_name,
                                       std::size_t input_size, std::size_t output_size,
                                       uint32_t *fallback_weight, std::size_t learner,
                                       const i2ce::RuntimeTiles &runtime_tiles) {

    LinearLayerBundle bundle;

#if CFG_USE_CODEBOOK_GEMM
    try {
        CodebookDenseConfig cfg = makeCheckedCodebookConfig(
            layer_name.c_str(), input_size, output_size, learner,
            runtime_tiles); // Get config from registry and validate shape

        bundle.main = new CodebookDense(cfg);
#if CFG_ENABLE_DEBUG_PRINT
        std::cout << "[LayerFactory] Using CodebookDense for " << layer_name
                  << std::endl;
#endif
    } catch (const std::exception &ex) {
        if (runtime_tiles.has_value()) {
            throw std::runtime_error(
                "Explicit runtime tiles require the Codebook backend for " +
                layer_name + ": " + ex.what());
        }
#if CFG_PROFILE_GEMM_ONLY
        throw std::runtime_error(
            "PROFILE_GEMM_ONLY requires a matching CodebookDense registry entry for " +
            layer_name + ": " + ex.what());
#elif CFG_CODEBOOK_ONLY_MODE
        throw std::runtime_error(
            "Codebook-only mode requires a matching CodebookDense registry entry for " +
            layer_name +
            ". Dense fallback weights are intentionally skipped when "
            "ENABLE_CODEBOOK_REFERENCE is disabled: " +
            ex.what());
#else
        bundle.main = new Dense(input_size, output_size, fallback_weight);
#if CFG_ENABLE_DEBUG_PRINT
        std::cout << "[LayerFactory] Fallback to Dense for " << layer_name << std::endl;
#endif
#endif
    }
#else
    if (runtime_tiles.has_value()) {
        throw std::invalid_argument("Runtime tiles require the Codebook backend");
    }
    bundle.main = new Dense(input_size, output_size, fallback_weight);
#endif

#if CFG_USE_CODEBOOK_REFERENCE
    bundle.reference = new Dense(input_size, output_size, fallback_weight);
#endif

    return bundle;
}
