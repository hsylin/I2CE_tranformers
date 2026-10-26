//
// Created by alireza on 4/24/23.
//
#define MAX_COL (SA_SIZE/4)

#include "debuggerFunctions.h"

#include <algorithm>
#include <cstdlib>

#include "codebookDense.h"
#include "run_mode_config.h"

#ifdef SIMD
uint64_t getSveLengthBytes()
{
    uint64_t vl;
    asm volatile("cntb %0" : "=r"(vl));
    return vl;
}

uint64_t getSveInt32Lanes()
{
    uint64_t lanes;
    asm volatile("cntw %0" : "=r"(lanes));
    return lanes;
}
#endif

void print_weight(uint32_t* kernel, int n_row, int n_col){
    for (int i=0; i< n_row; i++){
        for (int j=0; j<n_col; j++){
            printf("0x%08x,\t", kernel[i*n_col + j]);
        }
        printf("\n");
    }
}
void write_weight_to_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.write(reinterpret_cast<const char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}

void read_weight_from_file(const std::string& filename, uint32_t* kernel, int n_row, int n_col) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    for (int i = 0; i < n_row; ++i) {
        for (int j = 0; j < n_col; ++j) {
            file.read(reinterpret_cast<char*>(&kernel[i * n_col + j]), sizeof(uint32_t));
        }
    }

    file.close();
}



void blockWise2RowWise(const uint32_t * blockWise, uint32_t* rowWise, int n_row, int n_col){
    uint32_t* initialRowWise = rowWise;
    for (int col=0; col<n_col/MAX_COL; col++){
        rowWise = initialRowWise + col * MAX_COL;
        for (int row=0; row < n_row; row++){
            for (int i=0; i<MAX_COL; i++){
                *(rowWise + i) = *(blockWise+ i);
                // printf("blockWise2RowWise: current i, %d, rowWise, %u\n", i, *(rowWise + i));
            }
            blockWise += MAX_COL;
            rowWise += n_col;
        }
    }
}

void rowWise2BlockWise(const uint32_t* rowWise, uint32_t* blockWise, int n_row, int n_col) {
    const uint32_t* initialRowWise = rowWise;
    for (int col = 0; col < n_col / MAX_COL; col++) {
        rowWise = initialRowWise + col * MAX_COL;
        for (int row = 0; row < n_row; row++) {
            for (int i = 0; i < MAX_COL; i++) {
                *(blockWise + i) = *(rowWise + i);
            }
            rowWise += n_col;
            blockWise += MAX_COL;
        }
    }
}


void interleave_hidden_flag(uint32_t* kernel, int n_row, int n_col, uint32_t hidden_flag) {
    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * SA_SIZE * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (all_zeros) {
                kernel[tile_index] = hidden_flag;
            }
        }
    }
}

void interleave_hidden_flag_zero_free(uint32_t*& kernel, int n_row, int n_col, uint32_t hidden_flag) {
    uint32_t * new_kernel;
    new_kernel = new uint32_t [n_row * n_col]();
    uint32_t * new_kernel_ptr = new_kernel;
    int counter = 0;

    for (int i = 0; i < n_row / SA_SIZE; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            int tile_index = (i * (n_col / MAX_COL) + j) * SA_SIZE * MAX_COL;
            bool all_zeros = true;

            for (int ii = 0; ii < SA_SIZE; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t value = kernel[tile_index + ii * MAX_COL + jj];
                    if (value != 0) {
                        all_zeros = false;
                        break;
                    }
                }
                if (!all_zeros) {
                    break;
                }
            }

            if (!all_zeros) {
                for (int ii = 0; ii < SA_SIZE; ii++) {
                    for (int jj = 0; jj < MAX_COL; jj++) {
                        *new_kernel ++ = kernel[tile_index + ii * MAX_COL + jj];
                    }
                }
            }
            else{
                *new_kernel ++ = hidden_flag;
                counter ++;
            }
        }
    }
    kernel = new_kernel_ptr;
}


// Jerry: print the output for debugging
void printOutputPreview(const uint32_t *out, int size) {
    int preview = std::min(size, 8);
    std::cout << "Output preview (first " << preview << " packed words):" << std::endl;
    for (int i = 0; i < preview; i++) {
        std::cout << "out[" << i << "] = " << out[i] << " -> [";
        for (int j = 0; j < 4; j++) {
            int8_t value = static_cast<int8_t>((out[i] >> (8 * j)) & 0xFF);
            std::cout << static_cast<int>(value);
            if (j != 3) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}

// Jerry: save the output for debugging
void saveOutput(int size, const uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/output.bin";
    std::ofstream fout(filename);
    if (fout.is_open()) {
        for (int i = 0; i < size; i++) {
            fout << array[i] << " ";
        }
        fout.close();
    } else {
        std::cout << filename + " Not saved" << std::endl;
    }
}


// New debug helpers moved from transformerBlock.cc
/* Print the entire matrix for debugging */
void printPackedMatrix(const char *label,
                       const uint32_t *buffer,
                       std::size_t rows,
                       std::size_t cols) {
    std::size_t packed_cols = (cols + 3) / 4;

    std::cout << label << " full matrix (" << rows << " x " << cols << "):" << std::endl;
    for (std::size_t r = 0; r < rows; r++) {
        std::cout << "row " << r << ": ";
        for (std::size_t c = 0; c < cols; c++) {
            uint32_t word = buffer[r * packed_cols + (c / 4)];
            int8_t value = static_cast<int8_t>((word >> (8 * (c % 4))) & 0xFF);
            std::cout << static_cast<int>(value);
            if (c + 1 != cols) {
                std::cout << " ";
            }
        }
        std::cout << std::endl;
    }
}

void savePackedMatrixText(const char *filename,
                          const uint32_t *buffer,
                          std::size_t rows,
                          std::size_t cols) {
    std::size_t packed_cols = (cols + 3) / 4;

    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cout << filename << " Not saved" << std::endl;
        return;
    }

    for (std::size_t r = 0; r < rows; r++) {
        for (std::size_t c = 0; c < cols; c++) {
            uint32_t word = buffer[r * packed_cols + (c / 4)];
            int8_t value = static_cast<int8_t>((word >> (8 * (c % 4))) & 0xFF);
            fout << static_cast<int>(value);
            if (c + 1 != cols) {
                fout << " ";
            }
        }
        fout << "\n";
    }

    fout.close();
}

void dumpPackedMatrixIfEnabled(const std::string& dump_dir,
                               const std::string& filename,
                               const uint32_t* buffer,
                               std::size_t rows,
                               std::size_t cols) {
#if CFG_PROFILE_GEMM_ONLY || CFG_GEM5_PROFILE_REGIONS
    (void)dump_dir;
    (void)filename;
    (void)buffer;
    (void)rows;
    (void)cols;
    return;
#else
    if (dump_dir.empty()) {
        return;
    }

    const std::string path = dump_dir + "/" + filename;
    savePackedMatrixText(path.c_str(), buffer, rows, cols);
    std::cout << "[DUMP] packed matrix -> " << path << std::endl;
#endif
}

void printPackedTensorAsPythonList(const std::string& var_name,
                                   const uint32_t* packed,
                                   std::size_t rows,
                                   std::size_t cols) {
    const std::size_t packed_cols = (cols + 3) / 4;

    std::cout << var_name << " = [\n";
    for (std::size_t r = 0; r < rows; ++r) {
        std::cout << "    [";
        for (std::size_t c = 0; c < cols; ++c) {
            const int v = static_cast<int>(unpackPackedValue(packed + r * packed_cols, c));
            std::cout << v;
            if (c + 1 != cols) {
                std::cout << ", ";
            }
        }
        std::cout << "]";
        if (r + 1 != rows) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "]\n";
}

// Return true only when both layers are CodebookDense objects that can use the
// 2D same-sequence grouped path. Otherwise callers must run each layer normally.
bool tryComputeGroupedCodebookDense2(LinearLayer* const layers[2],
                                     std::size_t seq_len,
                                     uint32_t* const inputs[2],
                                     uint32_t* const outputs[2]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved2LearnersSameSeq()) {
        return false;
    }

    auto* learner_layer = dynamic_cast<CodebookDense*>(layers[1]);
    if (learner_layer == nullptr || !learner_layer->supportsInterleaved2LearnersSameSeq()) {
        return false;
    }

    primary->computeInterleaved2LearnersSameSeq(seq_len, inputs, outputs);
    return true;
}

// Return true only when all 4 layers are CodebookDense objects that already
// prepared the interleaved 4D caches. Only in that case do we collapse 4 learner
// executions into one GEMM call; otherwise callers must run each layer separately.
bool tryComputeGroupedCodebookDense4(LinearLayer* const layers[4],
                                     std::size_t seq_len,
                                     uint32_t* const inputs[4],
                                     uint32_t* const outputs[4]) {
    auto* primary = dynamic_cast<CodebookDense*>(layers[0]);
    if (primary == nullptr || !primary->supportsInterleaved4Learners()) {
        return false;
    }

    for (std::size_t learner = 1; learner < 4; learner++) {
        auto* learner_layer = dynamic_cast<CodebookDense*>(layers[learner]);
        if (learner_layer == nullptr || !learner_layer->supportsInterleaved4Learners()) {
            return false;
        }
    }

    primary->computeInterleaved4Learners(seq_len, inputs, outputs);
    return true;
}

/* Unpack the results and print 
Feed Forward 1
ffn1_pre_addnorm preview (first 4 packed words):
ffn1_pre_addnorm[0] = 33489149 -> [-3, 0, -1, 1]
ffn1_pre_addnorm[1] = 4278124289 -> [1, -1, -2, -2]
ffn1_pre_addnorm[2] = 65928689 -> [-15, -3, -19, 3]
ffn1_pre_addnorm[3] = 4125492229 -> [5, 4, -26, -11] */

void printPackedPreview(const char *label, const uint32_t *buffer, std::size_t packed_size) { // Redundant, consider removing it later
    std::size_t preview = std::min<std::size_t>(packed_size, 8);
    std::cout << label << " preview (first " << preview << " packed words):" << std::endl;
    for (std::size_t i = 0; i < preview; i++) {
        std::cout << label << "[" << i << "] = " << buffer[i] << " -> [";
        for (int j = 0; j < 4; j++) {
            int8_t value = static_cast<int8_t>((buffer[i] >> (8 * j)) & 0xFF);
            std::cout << static_cast<int>(value);
            if (j != 3) {
                std::cout << ", ";
            }
        }
        std::cout << "]" << std::endl;
    }
}


// Save intermediate results for debugging
void savePackedBuffer(const char *filename, const uint32_t *buffer, std::size_t packed_size) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cout << filename << " Not saved" << std::endl;
        return;
    }

    for (std::size_t i = 0; i < packed_size; i++) {
        fout << buffer[i] << " ";
    }
    fout.close();
}

// Auxiliary function: convert 32-bit packed values to 8-bit unpacked values
int8_t unpackPackedValue(const uint32_t *buffer, std::size_t elem_idx) {
    std::size_t word_idx = elem_idx / 4;
    std::size_t byte_idx = elem_idx % 4;
    return static_cast<int8_t>((buffer[word_idx] >> (byte_idx * 8)) & 0xFF);
}

namespace {
// Process-wide tally of reference comparisons and of those that differed. See
// the header for why this exists rather than a changed return type.
std::size_t g_packed_buffer_comparisons = 0;
std::size_t g_packed_buffer_mismatches = 0;
}  // namespace

std::size_t packedBufferMismatchCount() { return g_packed_buffer_mismatches; }
std::size_t packedBufferComparisonCount() { return g_packed_buffer_comparisons; }
void resetPackedBufferMismatchCount() {
    g_packed_buffer_comparisons = 0;
    g_packed_buffer_mismatches = 0;
}

// Auxiliary function: compare the results of gemm_exec and orginal tranformer code
void comparePackedBuffers(const char *label,
                          const uint32_t *dense_reference,
                          const uint32_t *candidate,
                          std::size_t packed_size) { 
    std::size_t total_values = packed_size * 4;
    int max_abs_diff = 0;
    std::size_t mismatch_count = 0;
    std::size_t first_mismatch = total_values;
    int first_dense_value = 0;
    int first_candidate_value = 0;

    for (std::size_t idx = 0; idx < total_values; idx++) {
        int dense_value = static_cast<int>(unpackPackedValue(dense_reference, idx));
        int candidate_value = static_cast<int>(unpackPackedValue(candidate, idx));
        int abs_diff = dense_value >= candidate_value ? (dense_value - candidate_value)
                                                     : (candidate_value - dense_value);
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        if (abs_diff != 0) {
            if (first_mismatch == total_values) {
                first_mismatch = idx;
                first_dense_value = dense_value;
                first_candidate_value = candidate_value;
            }
            mismatch_count++;
        }
    }

    std::cout << label << " diff vs Dense reference: max_abs_diff=" << max_abs_diff
              << ", mismatches=" << mismatch_count << "/" << total_values << std::endl;
    g_packed_buffer_comparisons++;
    if (mismatch_count != 0) {
        g_packed_buffer_mismatches++;
        std::cout << label << " first mismatch at value[" << first_mismatch << "]: dense="
                  << first_dense_value << ", candidate=" << first_candidate_value << std::endl;
    }
}
