#include "codebookDense.h"

#include "../Full_NN/inc/gemm_exec.h"
#ifdef SIMD
#include "../Full_NN/inc/gemm_exec_internal.h"
#endif

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// Function signatures in this file:
// int8_t clampToInt8(float value);
// int32_t clampToInt32(double value);
// CodebookDense::CodebookDense(const CodebookDenseConfig &config);
// uint32_t CodebookDense::getPackedIndex(const uint32_t *packed_row, std::size_t elem_idx, uint8_t bits_per_cb);
// int8_t CodebookDense::unpackInt8(const uint32_t *packed, std::size_t elem_idx);
// void CodebookDense::packInt8(const std::vector<int8_t> &src, uint32_t *dst);
// int8_t CodebookDense::clampInt32ToInt8(int32_t value);


// void CodebookDense::runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const;
// void CodebookDense::buildInterleavedCachesIfNeeded();
// bool CodebookDense::supportsInterleaved2LearnersSameSeq() const;
// bool CodebookDense::supportsInterleaved4Learners() const;
// void CodebookDense::computeInterleaved2LearnersSameSeq(std::size_t seq_len, uint32_t* const inputs[2], uint32_t* const outputs[2]) const;
// void CodebookDense::computeInterleaved2LearnersToInt8(std::size_t seq_len, const int8_t* input_interleaved, int8_t* output_interleaved) const;
// void CodebookDense::computeInterleaved4Learners(std::size_t seq_len, uint32_t* const inputs[4], uint32_t* const outputs[4]) const;
// void CodebookDense::computeInterleaved4LearnersToInt8(std::size_t seq_len, const int8_t* input_interleaved, int8_t* output_interleaved) const;
// void CodebookDense::compute(std::size_t seq_len, uint32_t *input, uint32_t *output);

namespace {

/**
 * @brief Round a floating-point value to the nearest int8_t and saturate if out of range.
 *
 * This helper is used when the registry provides a floating-point codebook
 * instead of a pre-quantized int8 codebook. The caller applies the output
 * quantization scale before calling this function, and this helper finishes the
 * conversion by rounding to the nearest integer and clamping to the valid int8
 * range [-128, 127].
 *
 * @param value Floating-point value after any layer/output scaling has already
 *              been applied.
 * @return Rounded and saturated int8_t value that can be passed to the compact
 *         GEMM codebook lookup.
 */
int8_t clampToInt8(float value) {
    float rounded = std::round(value);
    if (rounded > static_cast<float>(std::numeric_limits<int8_t>::max())) {
        return std::numeric_limits<int8_t>::max();
    }
    if (rounded < static_cast<float>(std::numeric_limits<int8_t>::min())) {
        return std::numeric_limits<int8_t>::min();
    }
    return static_cast<int8_t>(rounded);
}


/**
 * @brief Round a floating-point-like value to the nearest int32_t and saturate if out of range.
 *
 * Bias values are added in the GEMM accumulator domain, so they are stored as
 * int32 values rather than int8 values. This helper converts a floating-point
 * bias value into that accumulator domain while avoiding overflow if the scaled
 * value falls outside the int32 range.
 *
 * @param value Floating-point bias value after output quantization scaling has
 *              already been applied.
 * @return Rounded and saturated int32_t value suitable for GEMM accumulation.
 */
int32_t clampToInt32(double value) {
    double rounded = std::round(value);
    if (rounded > static_cast<double>(std::numeric_limits<int32_t>::max())) {
        return std::numeric_limits<int32_t>::max();
    }
    if (rounded < static_cast<double>(std::numeric_limits<int32_t>::min())) {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(rounded);
}
}

/**
 * @brief Construct a CodebookDense layer from a configuration object.
 *
 * This constructor:
 * - stores the layer dimensions and configuration,
 * - validates input arguments,
 * - quantizes the codebook into int8 form if only float codebook is provided,
 * - quantizes the bias into int32 form if bias exists,
 * - prepares interleaved cache layouts when the layer is configured for 2-way
 *   or 4-way learner execution.
 *
 * Expected assumptions:
 * - input and output tensors are packed int8 values stored in uint32_t arrays
 *   (4 int8 values per uint32_t),
 * - therefore input_size and output_size must both be divisible by 4.
 *
 * The important config fields are:
 * - input_size/output_size: logical int8 feature dimensions for one token.
 * - n_words_row: number of uint32_t words used to store one packed weight-index row.
 * - n_learners/same_seq/selected_learner: describe whether this instance is a
 *   single learner or a grouped interleaved learner bundle.
 * - bits_per_cb: number of bits used for each packed codebook index.
 * - weight_idx/weight_idx_by_learner/weight_idx_interleaved: packed codebook-index
 *   streams for normal, per-learner, or already-interleaved layouts.
 * - codebook/codebook_int8/codebooks_int8/codebook_int8_interleaved: floating-point
 *   or int8 codebook values used by the compact GEMM backend.
 * - bias/biases/bias_interleaved: optional bias values for single-learner,
 *   per-learner, or interleaved layouts.
 * - input_dequant_scale/output_quant_scale: scaling factors from the quantized
 *   registry metadata; this implementation uses output_quant_scale when it must
 *   quantize floating-point codebooks or biases locally.
 *
 * @param config Structure containing the layer dimensions, packed weight indexes,
 *               codebook values, optional bias values, learner grouping metadata,
 *               and quantization scales.
 * @throws std::invalid_argument if configuration is invalid.
 */
CodebookDense::CodebookDense(const CodebookDenseConfig &config)
    : input_size_(config.input_size),
      output_size_(config.output_size),
      n_words_row_(config.n_words_row),
      n_learners_(config.n_learners == 0 ? 1 : config.n_learners),
      same_seq_(config.same_seq),
      selected_learner_(config.selected_learner),
      bits_per_cb_(config.bits_per_cb),
      weight_idx_(config.weight_idx),
      weight_idx_by_learner_(config.weight_idx_by_learner),
      weight_idx_interleaved_(config.weight_idx_interleaved),
      codebook_(config.codebook),
      codebooks_int8_(config.codebooks_int8),
      codebook_int8_interleaved_(config.codebook_int8_interleaved),
      bias_(config.bias),
      biases_(config.biases),
      bias_interleaved_(config.bias_interleaved),
      input_dequant_scale_(config.input_dequant_scale == 0.0f ? 1.0f : config.input_dequant_scale),
      output_quant_scale_(config.output_quant_scale == 0.0f ? 1.0f : config.output_quant_scale) {
    if (input_size_ == 0 || output_size_ == 0) {
        throw std::invalid_argument("CodebookDense requires non-zero input and output sizes");
    }
    if (bits_per_cb_ == 0 || bits_per_cb_ > 16) {
        throw std::invalid_argument("CodebookDense bits_per_cb must be between 1 and 16");
    }
    if ((input_size_ % 4) != 0 || (output_size_ % 4) != 0) {
        throw std::invalid_argument("CodebookDense expects packed int8 tensors with dimensions divisible by 4");
    }

    std::size_t idxs_per_word = 32u / bits_per_cb_;
    std::size_t required_n_words_row = (input_size_ + idxs_per_word - 1) / idxs_per_word;
    if (n_words_row_ < required_n_words_row) {
        throw std::invalid_argument("CodebookDense n_words_row is too small for the provided input size and bits_per_cb");
    }
    if (weight_idx_ == nullptr) {
        throw std::invalid_argument("CodebookDense requires a valid weight_idx pointer");
    }

    std::size_t codebook_size = static_cast<std::size_t>(1u) << bits_per_cb_;
    if (config.codebook_int8 != nullptr) {
        codebook_q_.assign(config.codebook_int8, config.codebook_int8 + codebook_size);
    } else if (codebook_ != nullptr) {
        codebook_q_.reserve(codebook_size);
        for (std::size_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
            codebook_q_.push_back(clampToInt8(codebook_[cb_idx] * output_quant_scale_));
        }
    } else {
        throw std::invalid_argument("CodebookDense requires either float or int8 codebook data");
    }

    if (codebooks_int8_ != nullptr && n_learners_ > 0) {
        codebooks_q_.assign(codebooks_int8_, codebooks_int8_ + (n_learners_ * codebook_size));
    } else if (n_learners_ == 1u) {
        codebooks_q_ = codebook_q_;
    }

    if (codebook_int8_interleaved_ != nullptr && n_learners_ > 0) {
        codebook_interleaved_q_.assign(
            codebook_int8_interleaved_,
            codebook_int8_interleaved_ + (n_learners_ * codebook_size));
    }

    if (bias_ != nullptr) {
        bias_q_.reserve(output_size_);
        for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
            bias_q_.push_back(clampToInt32(static_cast<double>(bias_[out_idx]) * output_quant_scale_));
        }
    }

    if (biases_ != nullptr && n_learners_ > 0) {
        biases_q_.reserve(n_learners_ * output_size_);
        for (std::size_t idx = 0; idx < n_learners_ * output_size_; idx++) {
            biases_q_.push_back(clampToInt32(static_cast<double>(biases_[idx]) * output_quant_scale_));
        }
    }

    if (bias_interleaved_ != nullptr && n_learners_ > 0) {
        bias_interleaved_q_.reserve(n_learners_ * output_size_);
        for (std::size_t idx = 0; idx < n_learners_ * output_size_; idx++) {
            bias_interleaved_q_.push_back(clampToInt32(static_cast<double>(bias_interleaved_[idx]) * output_quant_scale_));
        }
    }

    buildInterleavedCachesIfNeeded();
}

/**
 * @brief Extract one packed codebook index from a row of packed index words.
 *
 * The weight matrix is not stored directly. Instead, each weight entry is represented
 * by a small integer index into the codebook. Multiple indexes are packed into a uint32_t.
 *
 * This function:
 * - determines which 32-bit word contains the requested index,
 * - computes the bit offset inside that word,
 * - extracts the index using bit shift and masking.
 *
 * @param packed_row Pointer to the first uint32_t word of one packed weight-index
 *                   row. The row contains indexes for consecutive input features.
 * @param elem_idx Logical input-feature index whose codebook index should be read.
 * @param bits_per_cb Number of bits used per codebook index. This controls how
 *                    many indexes fit into each uint32_t word.
 * @return Extracted integer codebook index, ready to index into the quantized
 *         codebook table.
 */
uint32_t CodebookDense::getPackedIndex(const uint32_t *packed_row,
                                       std::size_t elem_idx,
                                       uint8_t bits_per_cb) {
    std::size_t idxs_per_word = 32u / bits_per_cb;
    uint32_t idx_mask = (1u << bits_per_cb) - 1u;
    std::size_t word_idx = elem_idx / idxs_per_word;
    std::size_t offset = (elem_idx % idxs_per_word) * bits_per_cb;
    return (packed_row[word_idx] >> offset) & idx_mask;
}


/**
 * @brief Unpack one int8 value from a uint32_t-packed tensor.
 *
 * In this storage format, each uint32_t contains 4 signed int8 values:
 *   byte 0 -> element 0
 *   byte 1 -> element 1
 *   byte 2 -> element 2
 *   byte 3 -> element 3
 *
 * This function reads the correct 32-bit word, selects the desired byte,
 * and returns it as signed int8.
 *
 * @param packed Pointer to the first uint32_t word of a packed int8 tensor row.
 *               The tensor uses little-endian byte order inside each word.
 * @param elem_idx Logical int8 element index within that packed row.
 * @return Sign-preserving int8 value extracted from the selected byte.
 */
int8_t CodebookDense::unpackInt8(const uint32_t *packed, std::size_t elem_idx) {
    std::size_t word_idx = elem_idx / 4;
    std::size_t byte_idx = elem_idx % 4;
    uint32_t word = packed[word_idx];
    return static_cast<int8_t>((word >> (byte_idx * 8)) & 0xFF);
}


/**
 * @brief Pack a vector of int8 values into uint32_t words.
 *
 * Every 4 int8 values are packed into one uint32_t:
 *   src[0] -> bits [7:0]
 *   src[1] -> bits [15:8]
 *   src[2] -> bits [23:16]
 *   src[3] -> bits [31:24]
 *
 * This matches the storage format expected by the rest of the framework.
 *
 * @param src Source int8 vector in logical row-major order. Its size must be
 *            divisible by 4 because each output word stores exactly 4 bytes.
 * @param dst Destination buffer that receives src.size() / 4 packed uint32_t
 *            words. The caller must allocate enough storage before calling.
 * @throws std::invalid_argument if src.size() is not divisible by 4.
 */
void CodebookDense::packInt8(const std::vector<int8_t> &src, uint32_t *dst) {
    if ((src.size() % 4) != 0) {
        throw std::invalid_argument("CodebookDense output size must be divisible by 4 for packed int8 storage");
    }
    std::size_t packed_size = src.size() / 4;
    for (std::size_t word_idx = 0; word_idx < packed_size; word_idx++) {
        uint32_t packed_word = 0;
        for (std::size_t byte_idx = 0; byte_idx < 4; byte_idx++) {
            uint8_t value = static_cast<uint8_t>(src[word_idx * 4 + byte_idx]);
            packed_word |= static_cast<uint32_t>(value) << (byte_idx * 8);
        }
        dst[word_idx] = packed_word;
    }
}

/**
 * @brief Saturate an int32 accumulator value into int8 range.
 *
 * After GEMM, results are accumulated in int32 to avoid overflow during summation.
 * This function compresses the final value back into int8 with saturation.
 *
 * @param value Accumulator value produced by GEMM or by another int32 arithmetic path.
 * @return Saturated int8 result in the range [-128, 127].
 *
 * @note The current compute paths intentionally cast accumulators to int8 instead
 *       of calling this helper, because the reference Dense/RWMA path wraps on
 *       int8 cast. This helper remains available for saturation-based behavior.
 */
int8_t CodebookDense::clampInt32ToInt8(int32_t value) {
    if (value > static_cast<int32_t>(std::numeric_limits<int8_t>::max())) {
        return std::numeric_limits<int8_t>::max();
    }
    if (value < static_cast<int32_t>(std::numeric_limits<int8_t>::min())) {
        return std::numeric_limits<int8_t>::min();
    }
    return static_cast<int8_t>(value);
}

/**
 * @brief Execute the dense layer using the compact GEMM backend.
 *
 * Main workflow:
 * 1. Unpack the packed uint32_t input into a plain int8 vector.
 * 2. Fill a gemm_t descriptor with seq_len, input_size, output_size, and n_words_row.
 * 3. Call the scalar or SVE compact GEMM backend, depending on the SIMD build flag.
 * 4. Convert int32 accumulators to int8 using the same wrapping cast as the Dense
 *    reference path.
 * 5. Pack the int8 output back into uint32_t format.
 *
 * The GEMM backend interprets:
 * - input_unpacked as regular int8 activations,
 * - weight_idx_ as packed codebook indexes,
 * - codebook_q_ as the quantized value lookup table,
 * - bias_q_ as optional int32 bias.
 *
 * @param seq_len Number of input rows / sequence tokens to process.
 * @param input Pointer to the packed input tensor. Layout is [seq][input_word],
 *              where each uint32_t word stores 4 int8 input features.
 * @param output Pointer to the packed output tensor. Layout is [seq][output_word],
 *               where each uint32_t word stores 4 int8 output features.
 */
void CodebookDense::runCompactGemm(std::size_t seq_len, const uint32_t *input, uint32_t *output) const {
    std::vector<int8_t> input_unpacked(seq_len * input_size_, 0);
    for (std::size_t seq = 0; seq < seq_len; seq++) {
        const uint32_t *input_row = input + seq * (input_size_ / 4); // Input matrix from previous layer
        for (std::size_t in_idx = 0; in_idx < input_size_; in_idx++) {
            input_unpacked[seq * input_size_ + in_idx] = unpackInt8(input_row, in_idx);
        }
    }

    std::vector<int32_t> output_acc(seq_len * output_size_, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

#ifdef SIMD
    gemm_exec_compact_int_sve(layer,
                              input_unpacked.data(), // vector, take the head address
                              weight_idx_,
                              codebook_q_.data(), // quantized codebook
                              bias_q_.empty() ? nullptr : bias_q_.data(),
                              output_acc.data(),
                              bits_per_cb_);
#else
    gemm_exec_compact_int(layer,
                          input_unpacked.data(), // vector, take the head address
                          weight_idx_,
                          codebook_q_.data(), // quantized codebook
                          bias_q_.empty() ? nullptr : bias_q_.data(),
                          output_acc.data(),
                          bits_per_cb_);
#endif

    std::vector<int8_t> output_int8(seq_len * output_size_, 0);
    for (std::size_t i = 0; i < output_acc.size(); i++) {
        // output_int8[i] = clampInt32ToInt8(output_acc[i]);
        // Match the current Dense / RWMA path behavior exactly.
        // Do NOT saturate here during reference validation, because the Dense path
        // effectively wraps on int8 cast for out-of-range accumulators.
        output_int8[i] = static_cast<int8_t>(output_acc[i]);
    }

    packInt8(output_int8, output); // Pack to 32-bit words and output
}

/**
 * @brief Build local interleaved lookup tables when the registry did not provide them.
 *
 * The 2D and 4D grouped GEMM kernels consume learner-major data in an interleaved
 * layout so they can process multiple learners together. Some registry entries
 * already provide interleaved weight indexes, codebooks, and biases. When they do
 * not, this function creates equivalent local cache vectors from the normal or
 * per-learner data.
 *
 * Cache layouts produced here:
 * - weight_idx_interleaved_: [packed_weight_index][learner].
 * - codebook_interleaved_q_: [codebook_entry][learner].
 * - bias_interleaved_q_: [output_channel][learner].
 *
 * Step-by-step:
 * 1. Return immediately for non-grouped layers; only 2 and 4 learners need these caches.
 * 2. Build an interleaved packed-index stream from either the shared weight_idx_
 *    stream or weight_idx_by_learner_ streams.
 * 3. Build an interleaved codebook from per-learner codebooks when available, or
 *    duplicate the single codebook across learners.
 * 4. Build an interleaved bias table from per-learner biases when available.
 *
 * @note This method has no parameters. It uses the layer configuration captured in
 *       member variables during construction.
 */
void CodebookDense::buildInterleavedCachesIfNeeded() {
    if (n_learners_ != 2u && n_learners_ != 4u) {
        return;
    }

    const std::size_t learner_count = n_learners_;
    const std::size_t packed_idx_count = output_size_ * n_words_row_;
    if (weight_idx_interleaved_ == nullptr) {
        weight_idx_interleaved_cache_.resize(packed_idx_count * learner_count);
        for (std::size_t idx = 0; idx < packed_idx_count; idx++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                if (same_seq_ || weight_idx_by_learner_ == nullptr) {
                    weight_idx_interleaved_cache_[idx * learner_count + learner] = weight_idx_[idx];
                } else {
                    weight_idx_interleaved_cache_[idx * learner_count + learner] =
                        weight_idx_by_learner_[learner * packed_idx_count + idx];
                }
            }
        }
        weight_idx_interleaved_ = weight_idx_interleaved_cache_.data();
    }

    const std::size_t codebook_size = static_cast<std::size_t>(1u) << bits_per_cb_;
    if (codebook_interleaved_q_.empty()) {
        codebook_interleaved_q_.resize(codebook_size * learner_count);
        for (std::size_t cb_idx = 0; cb_idx < codebook_size; cb_idx++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                const std::size_t src_base =
                    (!codebooks_q_.empty() && codebooks_q_.size() >= (learner_count * codebook_size))
                        ? (learner * codebook_size)
                        : 0u;
                codebook_interleaved_q_[cb_idx * learner_count + learner] =
                    (!codebooks_q_.empty() && codebooks_q_.size() >= (learner_count * codebook_size))
                        ? codebooks_q_[src_base + cb_idx]
                        : codebook_q_[cb_idx];
            }
        }
    }

    if (bias_interleaved_q_.empty() && !biases_q_.empty()) {
        bias_interleaved_q_.resize(output_size_ * learner_count);
        for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
            for (std::size_t learner = 0; learner < learner_count; learner++) {
                bias_interleaved_q_[out_idx * learner_count + learner] =
                    biases_q_[learner * output_size_ + out_idx];
            }
        }
    }

    // Pre-widen the interleaved int8 codebook to int32 once, so every SVE
    // wrapper call can skip its per-call widening loop. This mirror is
    // consumed via the int8 SVE wrappers' codebook_i32_interleaved_opt
    // parameter; the scalar backend still reads codebook_interleaved_q_
    // directly and is unaffected.
    if (codebook_widened_i32_interleaved_cache_.empty() &&
        !codebook_interleaved_q_.empty()) {
        codebook_widened_i32_interleaved_cache_.resize(codebook_interleaved_q_.size());
        for (std::size_t i = 0; i < codebook_interleaved_q_.size(); i++) {
            codebook_widened_i32_interleaved_cache_[i] =
                static_cast<int32_t>(codebook_interleaved_q_[i]);
        }
    }
}

/**
 * @brief Check whether this layer can run the 2-learner same-sequence grouped GEMM path.
 *
 * The 2D path is only valid when exactly two learners share the same sequence and
 * the constructor has prepared an interleaved codebook. The actual packed weight
 * indexes are still read from the shared weight_idx_ stream.
 *
 * @return true when the layer has exactly 2 learners, same_seq_ is enabled, the
 *         interleaved codebook is available, and weight_idx_ is non-null.
 */
bool CodebookDense::supportsInterleaved2LearnersSameSeq() const {
    if ((n_learners_ != 2u) || !same_seq_ || codebook_interleaved_q_.empty()) {
        return false;
    }

    return weight_idx_ != nullptr;
}

/**
 * @brief Check whether this layer can run the 4-learner grouped GEMM path.
 *
 * The name mentions diff-seq because this path supports the per-learner
 * different-sequence layout, but the implementation also handles same_seq_ by
 * selecting the 4D same-sequence GEMM backend at runtime.
 *
 * @return true when the layer has exactly 4 learners, the interleaved codebook is
 *         available, and the required packed-index stream exists. For same_seq_
 *         this requires weight_idx_; for diff-seq this requires weight_idx_interleaved_.
 */
bool CodebookDense::supportsInterleaved4Learners() const {
    if ((n_learners_ != 4u) || codebook_interleaved_q_.empty()) {
        return false;
    }

    return same_seq_ ? (weight_idx_ != nullptr) : (weight_idx_interleaved_ != nullptr);
}

/**
 * @brief Execute two learners together and return packed uint32_t outputs.
 *
 * This is the older grouped CodebookDense path used when callers still pass each
 * learner as a separate packed uint32_t buffer. The function temporarily unpacks
 * both input buffers into an interleaved int8 layout, calls the 2D same-sequence
 * GEMM backend once, then splits and repacks the result into one output buffer per
 * learner.
 *
 * Input and output layouts:
 * - inputs[learner]: packed [seq][input_word] tensor for one learner.
 * - internal input_interleaved: [seq][input_channel][learner].
 * - internal output_acc_interleaved/output_int8: [seq][output_channel][learner].
 * - outputs[learner]: packed [seq][output_word] tensor for one learner.
 *
 * Step-by-step:
 * 1. Validate that the 2D same-sequence backend is available.
 * 2. Unpack both learner inputs from uint32_t words into one interleaved int8 buffer.
 * 3. Fill the GEMM descriptor with this layer's dimensions.
 * 4. Run the scalar or SVE 2D same-sequence compact GEMM backend.
 * 5. Cast int32 accumulators to int8.
 * 6. De-interleave each learner's output and pack it back into uint32_t words.
 *
 * @param seq_len Number of sequence rows/tokens to process for both learners.
 * @param inputs Array of two packed input buffers, one per learner.
 * @param outputs Array of two packed output buffers, one per learner.
 * @throws std::runtime_error if the layer is not configured for the 2D same-seq path.
 */
void CodebookDense::computeInterleaved2LearnersSameSeq(std::size_t seq_len,
                                                uint32_t* const inputs[2],
                                                uint32_t* const outputs[2]) const {
    if (!supportsInterleaved2LearnersSameSeq()) {
        throw std::runtime_error("CodebookDense interleaved 2D same-seq path is not available");
    }

    std::vector<int8_t> input_interleaved(seq_len * input_size_ * 2u, 0);
    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t in_idx = 0; in_idx < input_size_; in_idx++) {
            for (std::size_t learner = 0; learner < 2u; learner++) {
                input_interleaved[((seq * input_size_) + in_idx) * 2u + learner] =
                    unpackInt8(inputs[learner] + seq * (input_size_ / 4u), in_idx);
            }
        }
    }

    std::vector<int32_t> output_acc_interleaved(seq_len * output_size_ * 2u, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

#ifdef SIMD
    static thread_local std::vector<int32_t> activation_workspace;
    const std::size_t activation_workspace_needed = seq_len * input_size_ * 2u;
    if (activation_workspace.size() < activation_workspace_needed) {
        activation_workspace.resize(activation_workspace_needed);
    }
    gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
        layer,
        input_interleaved.data(),
        weight_idx_,
        codebook_interleaved_q_.data(),
        bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
        output_acc_interleaved.data(),
        bits_per_cb_,
        codebook_widened_i32_interleaved_cache_.empty()
            ? nullptr
            : codebook_widened_i32_interleaved_cache_.data(),
        activation_workspace.data(),
        static_cast<uint32_t>(activation_workspace.size()));
#else
    gemm_exec_compact_int_interleaved_2Learners_same_seq(
        layer,
        input_interleaved.data(),
        weight_idx_,
        codebook_interleaved_q_.data(),
        bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
        output_acc_interleaved.data(),
        bits_per_cb_);
#endif

    std::vector<int8_t> output_int8(seq_len * output_size_ * 2u, 0);
    for (std::size_t idx = 0; idx < output_acc_interleaved.size(); idx++) {
        output_int8[idx] = static_cast<int8_t>(output_acc_interleaved[idx]);
    }

    std::vector<int8_t> learner_output(seq_len * output_size_, 0);
    for (std::size_t learner = 0; learner < 2u; learner++) {
        for (std::size_t seq = 0; seq < seq_len; seq++) {
            for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
                learner_output[seq * output_size_ + out_idx] =
                    output_int8[((seq * output_size_) + out_idx) * 2u + learner];
            }
        }
        packInt8(learner_output, outputs[learner]);
    }
}

/**
 * @brief Execute two learners together using already-interleaved int8 buffers.
 *
 * This is the fully interleaved 2D pipeline entry point. Unlike
 * computeInterleaved2LearnersSameSeq(), it does not unpack per-learner uint32_t inputs
 * and does not repack outputs. The caller keeps the whole transformer block in
 * the shared [seq][feature][learner] int8 layout.
 *
 * Step-by-step:
 * 1. Validate that the 2D same-sequence backend is available.
 * 2. Allocate an int32 accumulator buffer in interleaved output layout.
 * 3. Fill the GEMM descriptor with this layer's dimensions.
 * 4. Run the scalar or SVE 2D same-sequence compact GEMM backend.
 * 5. Cast each int32 accumulator into the caller-provided int8 output buffer.
 *
 * @param seq_len Number of sequence rows/tokens to process for both learners.
 * @param input_interleaved Pointer to int8 input data in [seq][input_channel][learner]
 *                          layout with exactly 2 learners.
 * @param output_interleaved Pointer to int8 output storage in
 *                           [seq][output_channel][learner] layout with exactly
 *                           2 learners.
 * @throws std::runtime_error if the layer is not configured for the 2D same-seq path.
 */
void CodebookDense::computeInterleaved2LearnersToInt8(std::size_t seq_len,
                                               const int8_t* input_interleaved,
                                               int8_t* output_interleaved) const {
    if (!supportsInterleaved2LearnersSameSeq()) {
        throw std::runtime_error("CodebookDense interleaved 2D pipeline path is not available");
    }

    std::vector<int32_t> output_acc_interleaved(seq_len * output_size_ * 2u, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

#ifdef SIMD
    static thread_local std::vector<int32_t> activation_workspace;
    const std::size_t activation_workspace_needed = seq_len * input_size_ * 2u;
    if (activation_workspace.size() < activation_workspace_needed) {
        activation_workspace.resize(activation_workspace_needed);
    }
    gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
        layer,
        input_interleaved,
        weight_idx_,
        codebook_interleaved_q_.data(),
        bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
        output_acc_interleaved.data(),
        bits_per_cb_,
        codebook_widened_i32_interleaved_cache_.empty()
            ? nullptr
            : codebook_widened_i32_interleaved_cache_.data(),
        activation_workspace.data(),
        static_cast<uint32_t>(activation_workspace.size()));
#else
    gemm_exec_compact_int_interleaved_2Learners_same_seq(
        layer,
        input_interleaved,
        weight_idx_,
        codebook_interleaved_q_.data(),
        bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
        output_acc_interleaved.data(),
        bits_per_cb_);
#endif

    for (std::size_t idx = 0; idx < output_acc_interleaved.size(); idx++) {
        output_interleaved[idx] = static_cast<int8_t>(output_acc_interleaved[idx]);
    }
}

/**
 * @brief Execute four learners together and return packed uint32_t outputs.
 *
 * This is the older 4-learner grouped path for callers that still use separate
 * packed uint32_t buffers per learner. It supports both same_seq_ and diff-seq
 * configurations. For same_seq_, all learners share weight_idx_. For diff-seq,
 * each learner uses its own packed-index stream through weight_idx_interleaved_.
 *
 * Input and output layouts:
 * - inputs[learner]: packed [seq][input_word] tensor for one learner.
 * - internal input_interleaved: [seq][input_channel][learner].
 * - internal output_acc_interleaved/output_int8: [seq][output_channel][learner].
 * - outputs[learner]: packed [seq][output_word] tensor for one learner.
 *
 * Step-by-step:
 * 1. Validate that the 4D grouped backend is available.
 * 2. Unpack all four learner inputs into one interleaved int8 buffer.
 * 3. Fill the GEMM descriptor with this layer's dimensions.
 * 4. Choose same-sequence or different-sequence compact GEMM based on same_seq_.
 * 5. Use the SVE backend when SIMD is defined; otherwise use the scalar backend.
 * 6. Cast int32 accumulators to int8.
 * 7. De-interleave each learner's output and pack it back into uint32_t words.
 *
 * @param seq_len Number of sequence rows/tokens to process for all four learners.
 * @param inputs Array of four packed input buffers, one per learner.
 * @param outputs Array of four packed output buffers, one per learner.
 * @throws std::runtime_error if the layer is not configured for the 4D grouped path.
 */
void CodebookDense::computeInterleaved4Learners(std::size_t seq_len,
                                                uint32_t* const inputs[4],
                                                uint32_t* const outputs[4]) const {
    if (!supportsInterleaved4Learners()) {
        throw std::runtime_error("CodebookDense interleaved 4D diff-seq path is not available");
    }

    std::vector<int8_t> input_interleaved(seq_len * input_size_ * 4u, 0);
    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t in_idx = 0; in_idx < input_size_; in_idx++) {
            for (std::size_t learner = 0; learner < 4u; learner++) {
                input_interleaved[((seq * input_size_) + in_idx) * 4u + learner] =
                    unpackInt8(inputs[learner] + seq * (input_size_ / 4u), in_idx);
            }
        }
    }

    std::vector<int32_t> output_acc_interleaved(seq_len * output_size_ * 4u, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

#ifdef SIMD
    static thread_local std::vector<int32_t> activation_workspace;
    const std::size_t activation_workspace_needed = seq_len * input_size_ * 4u;
    if (activation_workspace.size() < activation_workspace_needed) {
        activation_workspace.resize(activation_workspace_needed);
    }
    if (same_seq_) {
        // Shared-index path: one packed index stream drives all 4 learners.
        gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_ex(
            layer,
            input_interleaved.data(),
            weight_idx_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_,
            codebook_widened_i32_interleaved_cache_.empty()
                ? nullptr
                : codebook_widened_i32_interleaved_cache_.data(),
            activation_workspace.data(),
            static_cast<uint32_t>(activation_workspace.size()));
    } else {
        // Per-learner index path: each learner has its own packed index stream.
        gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq_ex(
            layer,
            input_interleaved.data(),
            weight_idx_interleaved_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_,
            codebook_widened_i32_interleaved_cache_.empty()
                ? nullptr
                : codebook_widened_i32_interleaved_cache_.data(),
            activation_workspace.data(),
            static_cast<uint32_t>(activation_workspace.size()));
    }
#else
    if (same_seq_) {
        gemm_exec_compact_int_interleaved_4Learners_same_seq(
            layer,
            input_interleaved.data(),
            weight_idx_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_);
    } else {
        gemm_exec_compact_int_interleaved_4Learners_diff_seq(
            layer,
            input_interleaved.data(),
            weight_idx_interleaved_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_);
    }
#endif

    std::vector<int8_t> output_int8(seq_len * output_size_ * 4u, 0);
    for (std::size_t idx = 0; idx < output_acc_interleaved.size(); idx++) {
        output_int8[idx] = static_cast<int8_t>(output_acc_interleaved[idx]);
    }

    std::vector<int8_t> learner_output(seq_len * output_size_, 0);
    for (std::size_t learner = 0; learner < 4u; learner++) {
        for (std::size_t seq = 0; seq < seq_len; seq++) {
            for (std::size_t out_idx = 0; out_idx < output_size_; out_idx++) {
                learner_output[seq * output_size_ + out_idx] =
                    output_int8[((seq * output_size_) + out_idx) * 4u + learner];
            }
        }
        packInt8(learner_output, outputs[learner]);
    }
}

/**
 * @brief Execute four learners together using already-interleaved int8 buffers.
 *
 * This is the fully interleaved 4D pipeline entry point. The input and output
 * buffers stay in [seq][feature][learner] format, so this function avoids the
 * unpack/de-interleave/repack work done by computeInterleaved4Learners().
 *
 * The function supports both 4D same-sequence and 4D different-sequence registry
 * layouts:
 * - same_seq_ == true: use weight_idx_ and the 4D same-sequence GEMM backend.
 * - same_seq_ == false: use weight_idx_interleaved_ and the 4D diff-sequence GEMM backend.
 *
 * Step-by-step:
 * 1. Validate that the 4D grouped backend is available.
 * 2. Allocate an int32 accumulator buffer in interleaved output layout.
 * 3. Fill the GEMM descriptor with this layer's dimensions.
 * 4. Select same-seq or diff-seq GEMM, then select SVE or scalar backend.
 * 5. Cast each int32 accumulator into the caller-provided int8 output buffer.
 *
 * @param seq_len Number of sequence rows/tokens to process for all four learners.
 * @param input_interleaved Pointer to int8 input data in
 *                          [seq][input_channel][learner] layout with exactly
 *                          4 learners.
 * @param output_interleaved Pointer to int8 output storage in
 *                           [seq][output_channel][learner] layout with exactly
 *                           4 learners.
 * @throws std::runtime_error if the layer is not configured for the 4D grouped path.
 */
void CodebookDense::computeInterleaved4LearnersToInt8(std::size_t seq_len,
                                               const int8_t* input_interleaved,
                                               int8_t* output_interleaved) const {
    if (!supportsInterleaved4Learners()) {
        throw std::runtime_error("CodebookDense interleaved 4D pipeline path is not available");
    }

    std::vector<int32_t> output_acc_interleaved(seq_len * output_size_ * 4u, 0);

    gemm_t layer;
    layer.seq_len = static_cast<uint16_t>(seq_len);
    layer.input_size = static_cast<uint16_t>(input_size_);
    layer.output_size = static_cast<uint16_t>(output_size_);
    layer.n_words_row = static_cast<uint16_t>(n_words_row_);

#ifdef SIMD
    static thread_local std::vector<int32_t> activation_workspace;
    const std::size_t activation_workspace_needed = seq_len * input_size_ * 4u;
    if (activation_workspace.size() < activation_workspace_needed) {
        activation_workspace.resize(activation_workspace_needed);
    }
    if (same_seq_) {
        gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_tiled_ex(
            layer,
            input_interleaved,
            weight_idx_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_,
            codebook_widened_i32_interleaved_cache_.empty()
                ? nullptr
                : codebook_widened_i32_interleaved_cache_.data(),
            activation_workspace.data(),
            static_cast<uint32_t>(activation_workspace.size()));
    } else {
        gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq_ex(
            layer,
            input_interleaved,
            weight_idx_interleaved_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_,
            codebook_widened_i32_interleaved_cache_.empty()
                ? nullptr
                : codebook_widened_i32_interleaved_cache_.data(),
            activation_workspace.data(),
            static_cast<uint32_t>(activation_workspace.size()));
    }
#else
    if (same_seq_) {
        gemm_exec_compact_int_interleaved_4Learners_same_seq(
            layer,
            input_interleaved,
            weight_idx_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_);
    } else {
        gemm_exec_compact_int_interleaved_4Learners_diff_seq(
            layer,
            input_interleaved,
            weight_idx_interleaved_,
            codebook_interleaved_q_.data(),
            bias_interleaved_q_.empty() ? nullptr : bias_interleaved_q_.data(),
            output_acc_interleaved.data(),
            bits_per_cb_);
    }
#endif

    for (std::size_t idx = 0; idx < output_acc_interleaved.size(); idx++) {
        output_interleaved[idx] = static_cast<int8_t>(output_acc_interleaved[idx]);
    }
}


/**
 * @brief Public inference entry point for the dense layer.
 *
 * This overrides LinearLayer::compute() for callers that use the normal packed
 * uint32_t tensor interface. It currently delegates all work to runCompactGemm().
 * Fully interleaved pipeline callers bypass this entry point and call
 * computeInterleaved2LearnersToInt8() or computeInterleaved4LearnersToInt8() instead.
 *
 * @param seq_len Number of sequence rows/tokens in the input tensor.
 * @param input Pointer to packed input storage in [seq][input_word] layout, where
 *              each uint32_t contains 4 int8 features.
 * @param output Pointer to packed output storage in [seq][output_word] layout,
 *               where each uint32_t contains 4 int8 features.
 */
void CodebookDense::compute(std::size_t seq_len, uint32_t *input, uint32_t *output) {
    runCompactGemm(seq_len, input, output);
}
