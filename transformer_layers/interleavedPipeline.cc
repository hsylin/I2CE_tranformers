#include "interleavedPipeline.h"

#include "debuggerFunctions.h"
#include "run_mode_config.h"

#if CFG_SIMD
#include <gemm_SVE.h>
#endif

#include <algorithm>
#include <vector>

namespace {

int8_t unpackPackedValueLocal(const uint32_t* row, std::size_t elem_idx) {
    const std::size_t word_idx = elem_idx / 4u;
    const std::size_t byte_idx = elem_idx % 4u;
    return static_cast<int8_t>((row[word_idx] >> (byte_idx * 8u)) & 0xFFu);
}

void packOneLearner(const int8_t* input_interleaved,
                    std::size_t rows,
                    std::size_t cols,
                    std::size_t learner_count,
                    std::size_t learner,
                    std::vector<uint32_t>& packed) {
    const std::size_t packed_cols = (cols + 3u) / 4u;
    packed.assign(rows * packed_cols, 0u);

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const std::size_t packed_idx = row * packed_cols + (col / 4u);
            const std::size_t byte_idx = col % 4u;
            const uint8_t value = static_cast<uint8_t>(
                input_interleaved[((row * cols) + col) * learner_count + learner]);
            packed[packed_idx] |= static_cast<uint32_t>(value) << (byte_idx * 8u);
        }
    }
}

} // namespace

void interleavePackedLearners4(std::size_t rows,
                               std::size_t cols,
                               uint32_t* const inputs[4],
                               int8_t* output_interleaved) {
    const std::size_t packed_cols = (cols + 3u) / 4u;

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < 4u; learner++) {
                output_interleaved[((row * cols) + col) * 4u + learner] =
                    unpackPackedValueLocal(inputs[learner] + row * packed_cols, col);
            }
        }
    }
}

void interleavePackedLearners2(std::size_t rows,
                               std::size_t cols,
                               uint32_t* const inputs[2],
                               int8_t* output_interleaved) {
    const std::size_t packed_cols = (cols + 3u) / 4u;

    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            for (std::size_t learner = 0; learner < 2u; learner++) {
                output_interleaved[((row * cols) + col) * 2u + learner] =
                    unpackPackedValueLocal(inputs[learner] + row * packed_cols, col);
            }
        }
    }
}

void packInterleavedLearners4(std::size_t rows,
                              std::size_t cols,
                              const int8_t* input_interleaved,
                              uint32_t* const outputs[4]) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 4u; learner++) {
        packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
        std::copy(packed.begin(), packed.end(), outputs[learner]);
    }
}

void packInterleavedLearners2(std::size_t rows,
                              std::size_t cols,
                              const int8_t* input_interleaved,
                              uint32_t* const outputs[2]) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 2u; learner++) {
        packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
        std::copy(packed.begin(), packed.end(), outputs[learner]);
    }
}

void packInterleavedLearner4(std::size_t rows,
                             std::size_t cols,
                             const int8_t* input_interleaved,
                             std::size_t learner,
                             uint32_t* output) {
    std::vector<uint32_t> packed;
    packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
    std::copy(packed.begin(), packed.end(), output);
}

void packInterleavedLearner2(std::size_t rows,
                             std::size_t cols,
                             const int8_t* input_interleaved,
                             std::size_t learner,
                             uint32_t* output) {
    std::vector<uint32_t> packed;
    packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
    std::copy(packed.begin(), packed.end(), output);
}

void dumpInterleavedLearnerMatrices4(const std::string dump_dirs[4],
                                     const std::string& filename,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 4u; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        packOneLearner(input_interleaved, rows, cols, 4u, learner, packed);
        dumpPackedMatrixIfEnabled(dump_dirs[learner], filename, packed.data(), rows, cols);
    }
}

void dumpInterleavedLearnerMatrices2(const std::string dump_dirs[2],
                                     const std::string& filename,
                                     const int8_t* input_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    std::vector<uint32_t> packed;

    for (std::size_t learner = 0; learner < 2u; learner++) {
        if (dump_dirs[learner].empty()) {
            continue;
        }
        packOneLearner(input_interleaved, rows, cols, 2u, learner, packed);
        dumpPackedMatrixIfEnabled(dump_dirs[learner], filename, packed.data(), rows, cols);
    }
}

void transposeInterleavedRowsToCols4(const int8_t* input_interleaved,
                                     int8_t* output_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const int8_t* in_slot = input_interleaved + ((row * cols + col) * 4u);
            int8_t* out_slot = output_interleaved + ((col * rows + row) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void transposeInterleavedRowsToCols2(const int8_t* input_interleaved,
                                     int8_t* output_interleaved,
                                     std::size_t rows,
                                     std::size_t cols) {
    for (std::size_t row = 0; row < rows; row++) {
        for (std::size_t col = 0; col < cols; col++) {
            const int8_t* in_slot = input_interleaved + ((row * cols + col) * 2u);
            int8_t* out_slot = output_interleaved + ((col * rows + row) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

/**
 * @brief Multiply two four-learner interleaved matrices and return int8 output.
 *
 * Matrix layout:
 * - lhs_interleaved is [lhs_row][k][learner] with four learner lanes.
 * - rhs_by_col_interleaved is [rhs_col][k][learner] with four learner lanes.
 *   The RHS is already stored by output column, which lets the inner product
 *   read both LHS and RHS with the same k index.
 * - output_interleaved is [lhs_row][rhs_col][learner].
 *
 * Step-by-step:
 * 1. Allocate int32 accumulators for every output element and learner lane.
 * 2. If CFG_SIMD is enabled, widen int8 inputs to int32 staging buffers and
 *    call the SVE interleaved 4D GEMM backend.
 * 3. Otherwise, run the scalar triple loop over row, column, and k.
 * 4. For each row/column pair, accumulate four independent dot products, one
 *    per learner lane.
 * 5. Store the int32 accumulators in interleaved output order.
 * 6. Cast the final accumulators down to int8_t in the caller's output buffer.
 */
void matmulInterleaved4LearnersToInt8(const int8_t* lhs_interleaved,
                               const int8_t* rhs_by_col_interleaved,
                               std::size_t lhs_rows,
                               std::size_t rhs_cols,
                               std::size_t k_elems,
                               int8_t* output_interleaved) {
    const std::size_t total_out = lhs_rows * rhs_cols * 4u;
    std::vector<int32_t> output_acc(total_out, 0);

#if CFG_SIMD
    std::vector<int32_t> lhs_i32(lhs_rows * k_elems * 4u);
    std::vector<int32_t> rhs_i32(rhs_cols * k_elems * 4u);

    // The SVE backend consumes int32_t arrays even though the logical matrix
    // values are int8_t, so widen the interleaved inputs without changing order.
    for (std::size_t idx = 0; idx < lhs_i32.size(); idx++) {
        lhs_i32[idx] = static_cast<int32_t>(lhs_interleaved[idx]);
    }
    for (std::size_t idx = 0; idx < rhs_i32.size(); idx++) {
        rhs_i32[idx] = static_cast<int32_t>(rhs_by_col_interleaved[idx]);
    }

    sve_gemm_dense_int8_interleaved_4Learners(
        lhs_i32.data(),
        rhs_i32.data(),
        static_cast<uint32_t>(lhs_rows),
        static_cast<uint32_t>(rhs_cols),
        static_cast<uint32_t>(k_elems),
        output_acc.data());
#else
    // Scalar fallback: compute C[row][col][learner] =
    // sum_k lhs[row][k][learner] * rhs_by_col[col][k][learner].
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            int32_t acc[4] = {0, 0, 0, 0};
            for (std::size_t k = 0; k < k_elems; k++) {
                const int8_t* lhs_slot = lhs_interleaved + ((row * k_elems + k) * 4u);
                const int8_t* rhs_slot = rhs_by_col_interleaved + ((col * k_elems + k) * 4u);
                for (std::size_t learner = 0; learner < 4u; learner++) {
                    acc[learner] += static_cast<int32_t>(lhs_slot[learner]) *
                                    static_cast<int32_t>(rhs_slot[learner]);
                }
            }
            // Write the four learner accumulators back into the same adjacent
            // lane order used everywhere else in the interleaved pipeline.
            int32_t* out_slot = output_acc.data() + ((row * rhs_cols + col) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = acc[learner];
            }
        }
    }
#endif

    // The rest of the integer pipeline expects int8 activations; this mirrors
    // the truncating cast used by the compact dense layers.
    for (std::size_t idx = 0; idx < total_out; idx++) {
        output_interleaved[idx] = static_cast<int8_t>(output_acc[idx]);
    }
}

/**
 * @brief Multiply two two-learner interleaved matrices and return int8 output.
 *
 * Matrix layout:
 * - lhs_interleaved is [lhs_row][k][learner] with two learner lanes.
 * - rhs_by_col_interleaved is [rhs_col][k][learner] with two learner lanes.
 * - output_interleaved is [lhs_row][rhs_col][learner].
 *
 * Step-by-step:
 * 1. Allocate int32 accumulators for the complete interleaved output.
 * 2. Use the SVE interleaved 2D GEMM backend when CFG_SIMD is enabled, after
 *    widening the int8 inputs into int32 staging buffers.
 * 3. Otherwise, run the scalar row/column/k dot-product loops.
 * 4. Keep learner 0 and learner 1 independent: each lane multiplies only the
 *    matching lane from lhs and rhs.
 * 5. Store the two accumulators beside each other in output order.
 * 6. Cast the int32 accumulators back to int8_t for downstream int8 layers.
 */
void matmulInterleaved2LearnersToInt8(const int8_t* lhs_interleaved,
                               const int8_t* rhs_by_col_interleaved,
                               std::size_t lhs_rows,
                               std::size_t rhs_cols,
                               std::size_t k_elems,
                               int8_t* output_interleaved) {
    const std::size_t total_out = lhs_rows * rhs_cols * 2u;
    std::vector<int32_t> output_acc(total_out, 0);

#if CFG_SIMD
    std::vector<int32_t> lhs_i32(lhs_rows * k_elems * 2u);
    std::vector<int32_t> rhs_i32(rhs_cols * k_elems * 2u);

    // Preserve [row/col][k][learner] order while widening for the SVE backend.
    for (std::size_t idx = 0; idx < lhs_i32.size(); idx++) {
        lhs_i32[idx] = static_cast<int32_t>(lhs_interleaved[idx]);
    }
    for (std::size_t idx = 0; idx < rhs_i32.size(); idx++) {
        rhs_i32[idx] = static_cast<int32_t>(rhs_by_col_interleaved[idx]);
    }

    sve_gemm_dense_int8_interleaved_2Learners(
        lhs_i32.data(),
        rhs_i32.data(),
        static_cast<uint32_t>(lhs_rows),
        static_cast<uint32_t>(rhs_cols),
        static_cast<uint32_t>(k_elems),
        output_acc.data());
#else
    // Scalar fallback: each output slot contains two independent dot products.
    for (std::size_t row = 0; row < lhs_rows; row++) {
        for (std::size_t col = 0; col < rhs_cols; col++) {
            int32_t acc[2] = {0, 0};
            for (std::size_t k = 0; k < k_elems; k++) {
                const int8_t* lhs_slot = lhs_interleaved + ((row * k_elems + k) * 2u);
                const int8_t* rhs_slot = rhs_by_col_interleaved + ((col * k_elems + k) * 2u);
                for (std::size_t learner = 0; learner < 2u; learner++) {
                    acc[learner] += static_cast<int32_t>(lhs_slot[learner]) *
                                    static_cast<int32_t>(rhs_slot[learner]);
                }
            }
            // Store learner accumulators adjacent to each other:
            // [row][col][0], [row][col][1].
            int32_t* out_slot = output_acc.data() + ((row * rhs_cols + col) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = acc[learner];
            }
        }
    }
#endif

    // Down-convert to the int8 activation format consumed by the next layer.
    for (std::size_t idx = 0; idx < total_out; idx++) {
        output_interleaved[idx] = static_cast<int8_t>(output_acc[idx]);
    }
}

void copyHeadToMultiheadInterleaved4Learners(const int8_t* head_interleaved,
                                      int8_t* multihead_interleaved,
                                      std::size_t seq_len,
                                      std::size_t head_idx,
                                      std::size_t head_hidden_size,
                                      std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            const int8_t* in_slot =
                head_interleaved + ((seq * head_hidden_size + feature) * 4u);
            int8_t* out_slot =
                multihead_interleaved + ((seq * multihead_cols + head_col_base + feature) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}

void copyHeadToMultiheadInterleaved2Learners(const int8_t* head_interleaved,
                                      int8_t* multihead_interleaved,
                                      std::size_t seq_len,
                                      std::size_t head_idx,
                                      std::size_t head_hidden_size,
                                      std::size_t num_heads) {
    const std::size_t multihead_cols = num_heads * head_hidden_size;
    const std::size_t head_col_base = head_idx * head_hidden_size;

    for (std::size_t seq = 0; seq < seq_len; seq++) {
        for (std::size_t feature = 0; feature < head_hidden_size; feature++) {
            const int8_t* in_slot =
                head_interleaved + ((seq * head_hidden_size + feature) * 2u);
            int8_t* out_slot =
                multihead_interleaved + ((seq * multihead_cols + head_col_base + feature) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                out_slot[learner] = in_slot[learner];
            }
        }
    }
}
