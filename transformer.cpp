#include"transformer_layers/transformerBlock.h"
//#include"gtest/gtest.h"
#include "transformer.h"
#include "accelerator/smm_gem.h"
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <vector>

#include "transformer_layers/run_mode_config.h"
#include <codebooks_def.h>

// Optional command-line override for TILE_L1_SIZE, matching the same override
// applied at the top of Full_NN/src/gemm_exec.c so both translation units see
// the same TILE_L1_SIZE value. See compile_transformer.sh for the
// TILE_L1_SIZE_FLAG -> -DTILE_L1_SIZE_OVERRIDE plumbing. When the override is
// unset, TILE_L1_SIZE keeps the notebook value from codebooks_def.h and the
// banner below prints it unchanged.
#ifdef TILE_L1_SIZE_OVERRIDE
#undef TILE_L1_SIZE
#define TILE_L1_SIZE TILE_L1_SIZE_OVERRIDE
#endif

#if CFG_USE_CODEBOOK_GEMM
#include "transformer_layers/registry_adapter.h"
#endif

// #ifndef RELOAD_WEIGHT
#include <filesystem>

#include "transformer_layers/debuggerFunctions.h"
#include "transformer_layers/profile.h"
#if CFG_USE_FP32_TRANSFORMER
#include "transformer_layers/transformerFloat.h"
#endif

#define KERNEL_DIM SA_SIZE
#define MAX_COL (SA_SIZE/4)


void fill_kernel(uint32_t *kernel, int kernel_size) {
    for (int i = 0; i < kernel_size; i++) {
        uint32_t result = 0;
        for (int j = 0; j < 4; j++) {
            result |= ((uint8_t) (rand() % 5 - 2)) << (8 * j);
        }
        kernel[i] = result;
    }
}

void fill_weight(uint32_t *kernel, int n_row, int n_col) { // ？
    uint32_t *kernel_ptr = kernel;
    for (int i = 0; i < n_row / KERNEL_DIM; i++) {
        for (int j = 0; j < n_col / MAX_COL; j++) {
            for (int ii = 0; ii < KERNEL_DIM; ii++) {
                for (int jj = 0; jj < MAX_COL; jj++) {
                    uint32_t result = 0;
                    for (int k = 0; k < 4; k++) {
                        result |= ((uint8_t) (rand() % 5 - 2)) << (8 * k);
                    }
                    *kernel_ptr = result;
                    kernel_ptr++;
                }
            }
        }
    }
}

void saveWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    // Write the kernel array to file
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
                           std::to_string(qkv) + ".bin";
    std::ofstream fout(filename);
    if (fout.is_open()) {
        for (int i = 0; i < size; i++) {
            fout << array[i] << " ";
        }
        fout.close();
    }
}


bool tryLoadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        return false;
    }

    for (int i = 0; i < size; i++) {
        if (!(fin >> array[i])) {
            std::cout << "Weight file size mismatch: " << filename
                      << " (expected " << size << " words)" << std::endl;
            fin.close();
            return false;
        }
    }

    uint32_t extra;
    if (fin >> extra) {
        std::cout << "Weight file has extra data: " << filename
                  << " (expected exactly " << size << " words)" << std::endl;
        fin.close();
        return false;
    }

    fin.close();
    return true;
}

void loadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    if (!tryLoadWeight(n_head, qkv, size, array, dir_name)) {
        std::cout << filename + " Not loaded" << std::endl;
    }
}

std::size_t getTransformerLearnerCount() {
#if CFG_DENSE_NO_SIMD_BASELINE
    return N_LEARNERS;
#elif CFG_USE_CODEBOOK_GEMM
    try {
        return getCodebookDenseLearnerCount("q_h0");
    } catch (const std::exception&) {
        return 1;
    }
#else
    return 1;
#endif
}

std::string getNotebookWeightsDirForLearner(const std::string& notebook_weights_dir,
                                            std::size_t learner_idx) {
    return notebook_weights_dir + "/learner" + std::to_string(learner_idx);
}

#if CFG_DENSE_NO_SIMD_BASELINE
using ActiveTransformerBlock = DenseNoSimdTransformerBlock;
#else
using ActiveTransformerBlock = TransformerBlock;
#endif


namespace {

// Startup banner: build flags, compile-time dimensions, SVE vector length,
// and normalized CFG_* configuration. Behavior-preserving: same lines in the
// same order, still gated by the same compile-time #ifs as before extraction.
void printRunBanner() {
    std::cout << "Welcome to TiC-SAT" << std::endl;

#ifdef BWMA
    std::cout << "BWMA method" << std::endl;
#else
    std::cout<<"RWMA method" << std::endl;
#endif
    std::cout << "SA_SIZE = " << SA_SIZE << std::endl;
    std::cout << "KERNEL_DIM = " << KERNEL_DIM << std::endl;
    std::cout << "MAX_COL = " << MAX_COL << std::endl;
    std::cout << "SVE_LENGTH = " << (N_SVE_BYTE * 8) << " bits" << std::endl;
#ifdef SIMD
    const uint64_t runtime_sve_bytes = getSveLengthBytes();
    const uint64_t runtime_sve_int32_lanes = getSveInt32Lanes();
    std::cout << "SVE_LENGTH on Gem5/QEMU = " << runtime_sve_bytes
              << " bytes = " << (runtime_sve_bytes * 8) << " bits" << std::endl;
    std::cout << "float lanes = " << runtime_sve_int32_lanes
              << ", int32 lanes = " << runtime_sve_int32_lanes << std::endl;

    // Codebook layouts, packed-index strides, and codebook-register capacity
    // are baked in at generation time using N_SVE_LANES / N_SVE_BYTE from
    // codebooks_def.h. A runtime SVE vector length that disagrees would make
    // the SVE row kernels read the wrong lanes and produce silently incorrect
    // output, so reject the mismatch here instead of masking it downstream.
    if (runtime_sve_bytes != static_cast<uint64_t>(N_SVE_BYTE) ||
        runtime_sve_int32_lanes != static_cast<uint64_t>(N_SVE_LANES)) {
        std::cerr << "ERROR: SVE vector-length mismatch." << std::endl
                  << "  expected: " << N_SVE_BYTE << " bytes ("
                  << (N_SVE_BYTE * 8) << " bit), " << N_SVE_LANES
                  << " int32 lanes"
                  << " (from codebooks_def.h N_SVE_BYTE / N_SVE_LANES)"
                  << std::endl
                  << "  runtime : " << runtime_sve_bytes << " bytes ("
                  << (runtime_sve_bytes * 8) << " bit), "
                  << runtime_sve_int32_lanes << " int32 lanes"
                  << std::endl
                  << "Regenerate the notebook artifacts for this SVE length"
                  << " or run under a matching QEMU/gem5 configuration."
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif

    std::cout << "D_Q = " << D_Q << std::endl;
    std::cout << "D_SEQ = " << D_SEQ << std::endl;
    std::cout << "D_MODEL = " << D_MODEL << std::endl;
    std::cout << "NUM_HEAD = " << NUM_HEAD << std::endl;
    std::cout << "D_FF = " << D_FF << std::endl;

    std::cout << "N_LEARNERS = " << N_LEARNERS << std::endl;
    std::cout << "CODEBOOK_SIZE = " << CB_SIZE << std::endl;
    std::cout << "TILE_L1_SIZE = " << TILE_L1_SIZE << std::endl;
    std::cout << "TILE_L2_SIZE = " << TILE_L2_SIZE << std::endl;
    std::cout << "USE_F32 = " << USE_F32 << std::endl;

    std::cout << "CFG_RELOAD_WEIGHT = " << CFG_RELOAD_WEIGHT << std::endl;
    std::cout << "CFG_USE_NOTEBOOK_GENERATED_WEIGHTS = " << CFG_USE_NOTEBOOK_GENERATED_WEIGHTS << std::endl;
    std::cout << "CFG_USE_CODEBOOK_GEMM = " << CFG_USE_CODEBOOK_GEMM << std::endl;
    std::cout << "CFG_USE_CODEBOOK_REFERENCE = " << CFG_USE_CODEBOOK_REFERENCE << std::endl;
    std::cout << "CFG_ENABLE_DEBUG_PRINT = " << CFG_ENABLE_DEBUG_PRINT << std::endl;
    std::cout << "CFG_PROFILE_GEMM_ONLY = " << CFG_PROFILE_GEMM_ONLY << std::endl;
    std::cout << "CFG_GEM5_PROFILE_REGIONS = " << CFG_GEM5_PROFILE_REGIONS << std::endl;
    std::cout << "CFG_FULL_INTERLEAVED_PIPELINE = " << CFG_FULL_INTERLEAVED_PIPELINE << std::endl;
    std::cout << "CFG_USE_FP32_TRANSFORMER = " << CFG_USE_FP32_TRANSFORMER << std::endl;
    std::cout << "CFG_SIMD = " << CFG_SIMD << std::endl;
    std::cout << "CFG_DENSE_NO_SIMD_BASELINE = " << CFG_DENSE_NO_SIMD_BASELINE << std::endl;
}

// Return the runtime weights directory. Prefer the host-side project path;
// fall back to the 9P mount used inside the gem5 guest.
std::string resolveWeightsDir() {
    // Prefer the host-side project path and fall back to the 9p mount in gem5.
    std::string dir_name = "/home/thu/TiC-SAT/weights";
    if (!std::filesystem::exists(dir_name)) {
        dir_name = "/mnt/weights";
    }
    return dir_name;
}

// Allocate and populate the input tensor on the heap. When RELOAD_WEIGHT is
// enabled, try the notebook-generated .bin first (if that source is enabled)
// and fall back to the legacy loadWeight path; otherwise fill in place and
// save. Ownership of the returned buffer is transferred to the caller.
uint32_t* loadInputTensorHeap(const std::string& dir_name,
                              const std::string& notebook_weights_dir) {
    uint32_t *tensor_in = new uint32_t[D_SEQ * D_MODEL >> 2];
#if CFG_RELOAD_WEIGHT
    // Load the tensor input from file
    // We assign -1 and -1 to n_head and qkv to indicate that we are not loading the weight
    bool input_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
    printf("Input matrix: trying notebook-generated input\n");
#endif
    input_loaded_from_notebook = tryLoadWeight(
        -1, -1, D_SEQ * D_MODEL >> 2, tensor_in, notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
    if (input_loaded_from_notebook) {
        std::cout << "Loaded notebook-generated input matrix from "
                  << notebook_weights_dir << std::endl;
    }
#endif
#endif

#if CFG_ENABLE_DEBUG_PRINT
    if (!input_loaded_from_notebook) {
        printf("Input matrix: loading weights of input matrix\n");
    }
#endif
    if (!input_loaded_from_notebook) {
        loadWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, dir_name);
    }
#else
    fill_kernel(tensor_in, D_SEQ * D_MODEL >> 2);
    // Save the tensor input to file
    // We assign -1 and -1 to n_head and qkv to indicate that we are not saving the weight
    saveWeight(-1, -1, D_SEQ * D_MODEL >> 2, tensor_in, dir_name);
#endif
    return tensor_in;
}

// Standard "=== LEARNER N ===" banner used before per-learner weight loading
// in the grouped and sequential paths. Callers that only want it in the
// multi-learner case gate on learner_count > 1 themselves.
void printLearnerHeader(std::size_t learner_idx) {
    std::cout << "\n=============== LEARNER " << learner_idx
              << " ===============\n" << std::endl;
}

// Per-learner path bundle plus its side effect: when make_dump_dir is true,
// build the per-learner dump directory string AND create it on disk. The
// grouped 2/4-learner paths always pass true; the default single-learner
// loop only passes true when learner_count > 1 (single-learner runs write to
// the top-level dump location instead).
struct LearnerPaths {
    std::string notebook_weights_dir;
    std::string dump_dir;
};

LearnerPaths preparePathsForLearner(std::size_t learner_idx,
                                    const std::string& notebook_weights_dir,
                                    const std::string& multiple_learner_output_root,
                                    bool make_dump_dir) {
    LearnerPaths paths;
    paths.notebook_weights_dir =
        getNotebookWeightsDirForLearner(notebook_weights_dir, learner_idx);
    if (make_dump_dir) {
        paths.dump_dir = multiple_learner_output_root + "/learner" +
                         std::to_string(learner_idx);
        std::filesystem::create_directories(paths.dump_dir);
    }
    return paths;
}

}  // namespace


void test() {
    printRunBanner();

    // Legacy commented-out draft kept in a prior commit; retained here as a
    // reminder that USE_F32 is now printed unconditionally by printRunBanner().
// #ifdef USE_F32
//     std::cout << "USE_F32 = " << USE_F32 << std::endl;
// #else
//     std::cout << "USE_F32 = 0" << std::endl;
// #endif

    std::string dir_name = resolveWeightsDir();
    std::string notebook_weights_dir = dir_name + "/generated_from_notebook";
#if !CFG_RELOAD_WEIGHT
    std::filesystem::create_directories(dir_name);
#endif

    const std::size_t learner_count = getTransformerLearnerCount();
    constexpr bool dump_outputs_enabled =
        !(CFG_PROFILE_GEMM_ONLY || CFG_GEM5_PROFILE_REGIONS);
#if CFG_DENSE_NO_SIMD_BASELINE
    std::cout << "DENSE_NO_SIMD_BASELINE_N_LEARNERS = " << learner_count << std::endl;
#else
    std::cout << "CODEBOOK_REGISTRY_N_LEARNERS = " << learner_count << std::endl;
#endif

#if CFG_USE_FP32_TRANSFORMER
    TransformerFloat::run(
        learner_count,
        dump_outputs_enabled ? dir_name + "/multiple_learner_outputs/c" : std::string());
    return;
#endif

    uint32_t *tensor_in = loadInputTensorHeap(dir_name, notebook_weights_dir);

#ifndef BWMA
    uint32_t tensorInRowWise[D_SEQ * D_MODEL >> 2];
    // By default, the saved tensor is in block-wise format
    // We need to convert it to row-wise format
    blockWise2RowWise(tensor_in, tensorInRowWise, D_SEQ, D_MODEL >> 2);
    tensor_in = tensorInRowWise;
#endif

    const std::string multiple_learner_output_root = dir_name + "/multiple_learner_outputs/c";
    if (dump_outputs_enabled && learner_count > 1) {
        std::filesystem::create_directories(multiple_learner_output_root);
    }

    auto buildTransformerBlockForLearner = // a lambda function to build a transformer block for a given learner index and its corresponding notebook weights directory and dump directory
        [&](std::size_t learner_idx,
            const std::string& learner_notebook_weights_dir,
            const std::string& learner_dump_dir) -> ActiveTransformerBlock* {

            // Q weight
            // K weight
            // V weight
            // output projection
            // FFN layer 1
            // FFN layer 2
        std::vector<uint32_t*> weightVec(3 * NUM_HEAD + 3, nullptr);
#if !CFG_CODEBOOK_ONLY_MODE
        const int head_qkv_size = D_Q * D_MODEL >> 2;
#endif

        for (int n = 0; n < NUM_HEAD; n++) {
            uint32_t* query_kernel = nullptr;
            uint32_t* key_kernel = nullptr;
            uint32_t* value_kernel = nullptr;

#if !CFG_CODEBOOK_ONLY_MODE
            query_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
            key_kernel = new uint32_t[D_Q * D_MODEL >> 2]();
            value_kernel = new uint32_t[D_Q * D_MODEL >> 2]();

            bool q_loaded_from_notebook = false;
            bool k_loaded_from_notebook = false;
            bool v_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
            printf("Head %d : trying notebook-generated Q/K/V weights\n", n);
#endif

            q_loaded_from_notebook = tryLoadWeight(
                n, 0, head_qkv_size, query_kernel, learner_notebook_weights_dir);

            k_loaded_from_notebook = tryLoadWeight(
                n, 1, head_qkv_size, key_kernel, learner_notebook_weights_dir);

            v_loaded_from_notebook = tryLoadWeight(
                n, 2, head_qkv_size, value_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
            if (q_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated Q weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
            }
            if (k_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated K weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
            }
            if (v_loaded_from_notebook) {
                std::cout << "Loaded notebook-generated V weights for head "
                          << n << " from " << learner_notebook_weights_dir << std::endl;
            }
#endif
#endif

#if CFG_RELOAD_WEIGHT
#if CFG_ENABLE_DEBUG_PRINT
            printf("Head %d : Q, K, V weight matrix: loading weights of Q, K, V kernels\n", n);
#endif
            if (!q_loaded_from_notebook) {
                loadWeight(n, 0, head_qkv_size, query_kernel, dir_name);
            }
            if (!k_loaded_from_notebook) {
                loadWeight(n, 1, head_qkv_size, key_kernel, dir_name);
            }
            if (!v_loaded_from_notebook) {
                loadWeight(n, 2, head_qkv_size, value_kernel, dir_name);
            }
#else
            fill_weight(query_kernel, D_MODEL, D_Q >> 2);
            fill_weight(key_kernel, D_MODEL, D_Q >> 2);
            fill_weight(value_kernel, D_MODEL, D_Q >> 2);

            saveWeight(n, 0, head_qkv_size, query_kernel, dir_name);
            saveWeight(n, 1, head_qkv_size, key_kernel, dir_name);
            saveWeight(n, 2, head_qkv_size, value_kernel, dir_name);
#endif

#ifndef BWMA
            uint32_t* queryRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(query_kernel, queryRowWise, D_MODEL, D_Q >> 2);
            query_kernel = queryRowWise;

            uint32_t* keyRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(key_kernel, keyRowWise, D_MODEL, D_Q >> 2);
            key_kernel = keyRowWise;

            uint32_t* valueRowWise = new uint32_t[D_MODEL * D_Q >> 2];
            blockWise2RowWise(value_kernel, valueRowWise, D_MODEL, D_Q >> 2);
            value_kernel = valueRowWise;
#endif
#endif

            weightVec[n * 3] = query_kernel;
            weightVec[n * 3 + 1] = key_kernel;
            weightVec[n * 3 + 2] = value_kernel; 
            // End for loading/generating Q/K/V weights for each head
        }

        uint32_t* condense_kernel = nullptr;
        uint32_t* ff0_kernel = nullptr;
        uint32_t* ff1_kernel = nullptr;

#if !CFG_CODEBOOK_ONLY_MODE
        condense_kernel = new uint32_t[NUM_HEAD * D_Q * D_MODEL >> 2]();
        ff0_kernel = new uint32_t[D_MODEL * D_FF >> 2]();
        ff1_kernel = new uint32_t[D_FF * D_MODEL >> 2]();

        int n = -1;
        bool condense_loaded_from_notebook = false;
        bool ff0_loaded_from_notebook = false;
        bool ff1_loaded_from_notebook = false;

#if CFG_USE_NOTEBOOK_GENERATED_WEIGHTS
#if CFG_ENABLE_DEBUG_PRINT
        printf("Condense/projection layer: trying notebook-generated condense weights\n");
#endif
        condense_loaded_from_notebook = tryLoadWeight(
            n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 0: trying notebook-generated FF0 weights\n");
#endif
        ff0_loaded_from_notebook = tryLoadWeight(
            n, 1, D_MODEL * D_FF >> 2, ff0_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        printf("Feed forward layer 1: trying notebook-generated FF1 weights\n");
#endif
        ff1_loaded_from_notebook = tryLoadWeight(
            n, 2, D_FF * D_MODEL >> 2, ff1_kernel, learner_notebook_weights_dir);

#if CFG_ENABLE_DEBUG_PRINT
        if (condense_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated condense weights from "
                      << learner_notebook_weights_dir << std::endl;
        }
        if (ff0_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF0 weights from "
                      << learner_notebook_weights_dir << std::endl;
        }
        if (ff1_loaded_from_notebook) {
            std::cout << "Loaded notebook-generated FF1 weights from "
                      << learner_notebook_weights_dir << std::endl;
        }
#endif
#endif

#if CFG_RELOAD_WEIGHT
        if (!condense_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
            printf("Condense/projection layer: loading original condense weights\n");
#endif
            loadWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, dir_name);
        }

        if (!ff0_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
            printf("Feed forward layer 0: loading original FF0 weights\n");
#endif
            loadWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, dir_name);
        }

        if (!ff1_loaded_from_notebook) {
#if CFG_ENABLE_DEBUG_PRINT
            printf("Feed forward layer 1: loading original FF1 weights\n");
#endif
            loadWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, dir_name);
        }
#else
        fill_weight(condense_kernel, NUM_HEAD * D_Q, D_MODEL >> 2);
        fill_weight(ff0_kernel, D_MODEL, D_FF >> 2);
        fill_weight(ff1_kernel, D_FF, D_MODEL >> 2);

        saveWeight(n, 0, NUM_HEAD * D_Q * D_MODEL >> 2, condense_kernel, dir_name);
        saveWeight(n, 1, D_MODEL * D_FF >> 2, ff0_kernel, dir_name);
        saveWeight(n, 2, D_FF * D_MODEL >> 2, ff1_kernel, dir_name);
#endif

#ifndef BWMA
        uint32_t* condenseRowWise = new uint32_t[NUM_HEAD * D_Q * D_MODEL >> 2];
        blockWise2RowWise(condense_kernel, condenseRowWise, NUM_HEAD * D_Q, D_MODEL >> 2);
        condense_kernel = condenseRowWise;

        uint32_t* ff0RowWise = new uint32_t[D_MODEL * D_FF >> 2];
        blockWise2RowWise(ff0_kernel, ff0RowWise, D_MODEL, D_FF >> 2);
        ff0_kernel = ff0RowWise;

        uint32_t* ff1RowWise = new uint32_t[D_FF * D_MODEL >> 2];
        blockWise2RowWise(ff1_kernel, ff1RowWise, D_FF, D_MODEL >> 2);
        ff1_kernel = ff1RowWise;
#endif
#endif

        weightVec[NUM_HEAD * 3] = condense_kernel;
        weightVec[NUM_HEAD * 3 + 1] = ff0_kernel;
        weightVec[NUM_HEAD * 3 + 2] = ff1_kernel;

        return new ActiveTransformerBlock(
            D_SEQ,
            D_MODEL,
            D_Q,
            NUM_HEAD,
            D_FF,
            weightVec.data(),
            KERNEL_DIM,
            MAX_COL,
            learner_idx,
            learner_dump_dir);
        // End of lambda function to build a transformer block for a given learner index and its corresponding notebook weights directory and dump directory
    }; 

#if !CFG_DENSE_NO_SIMD_BASELINE
    // The grouped execution path is enabled when the registry exposes a learner
    // count that has an interleaved GEMM backend. With 2 learners, CodebookDense
    // only groups same-sequence layers; with 4 learners, it chooses same-seq or
    // diff-seq internally from the registry metadata.
    // 2 Learners
    if (learner_count == 2) {
        TransformerBlock* grouped_blocks[2] = {nullptr, nullptr};
        uint32_t* grouped_inputs[2] = {tensor_in, tensor_in};
        uint32_t* grouped_outputs[2] = {
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
        };

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            printLearnerHeader(learner_idx);

            const auto paths = preparePathsForLearner(
                learner_idx, notebook_weights_dir,
                multiple_learner_output_root, dump_outputs_enabled);
            grouped_blocks[learner_idx] = buildTransformerBlockForLearner(
                learner_idx,
                paths.notebook_weights_dir,
                paths.dump_dir);
        }

        TransformerBlock::computeGroup2(D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            delete[] grouped_outputs[learner_idx];
            delete grouped_blocks[learner_idx];
        }

        return;
    }

    //  4 Leaners
    if (learner_count == 4) {
        TransformerBlock* grouped_blocks[4] = {nullptr, nullptr, nullptr, nullptr};
        uint32_t* grouped_inputs[4] = {tensor_in, tensor_in, tensor_in, tensor_in};
        uint32_t* grouped_outputs[4] = {
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
            new uint32_t[D_SEQ * D_MODEL >> 2](),
        };

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            printLearnerHeader(learner_idx);

            const auto paths = preparePathsForLearner(
                learner_idx, notebook_weights_dir,
                multiple_learner_output_root, dump_outputs_enabled);
            grouped_blocks[learner_idx] = buildTransformerBlockForLearner(
                learner_idx,
                paths.notebook_weights_dir,
                paths.dump_dir);
        }

        // This call does not jump to GEMM directly. It enters the grouped
        // transformer path, where each attention/FFN dense layer first tries the
        // 4-learner CodebookDense fast path and only then dispatches to either the
        // SVE or scalar interleaved GEMM backend.
        TransformerBlock::computeGroup4(D_SEQ, grouped_blocks, grouped_inputs, grouped_outputs);

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            delete[] grouped_outputs[learner_idx];
            delete grouped_blocks[learner_idx];
        }

        return;
    }
#endif

#if CFG_DENSE_NO_SIMD_BASELINE
    // For the DenseNoSimd baseline, we always execute learners sequentially in a loop, even if the registry exposes multiple learners. 
    // This is because the baseline does not have an interleaved GEMM backend and thus cannot benefit from grouped execution.
    if (learner_count > 1) {
        std::vector<TransformerBlock*> learner_blocks(learner_count, nullptr);
        std::vector<uint32_t*> learner_outputs(learner_count, nullptr);

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            printLearnerHeader(learner_idx);

            const auto paths = preparePathsForLearner(
                learner_idx, notebook_weights_dir,
                multiple_learner_output_root, dump_outputs_enabled);

            learner_outputs[learner_idx] = new uint32_t[D_SEQ * D_MODEL >> 2]();
            learner_blocks[learner_idx] = buildTransformerBlockForLearner(
                learner_idx,
                paths.notebook_weights_dir,
                paths.dump_dir);
        }

        // Keep one gem5 stats window around the whole sequential learner loop,
        // after all .bin weights have been loaded into the learner blocks.
        resetTransformerStatsWindow("dense_no_simd_baseline_multi_learner_for_loop");

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            learner_blocks[learner_idx]->computeWithoutStatsReset(
                D_SEQ,
                tensor_in,
                learner_outputs[learner_idx]);
        }

        for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
            delete learner_blocks[learner_idx];
            delete[] learner_outputs[learner_idx];
        }

        return;
    }
#endif

    // Default single-learner execution path (also used for multi-learner when the registry does not expose a learner count).
    for (std::size_t learner_idx = 0; learner_idx < learner_count; learner_idx++) {
        if (learner_count > 1) {
            printLearnerHeader(learner_idx);
        }

        const auto paths = preparePathsForLearner(
            learner_idx, notebook_weights_dir,
            multiple_learner_output_root,
            dump_outputs_enabled && learner_count > 1);

        uint32_t *out = new uint32_t[D_SEQ * D_MODEL >> 2]();
        TransformerBlock* selfatten = buildTransformerBlockForLearner(
            learner_idx,
            paths.notebook_weights_dir,
            paths.dump_dir);
        selfatten->compute(D_SEQ, tensor_in, out);
        delete selfatten;

        delete[] out;
    }

}

int main() {
    test();

#if CFG_USE_CODEBOOK_REFERENCE
    /*
     * Reference-validation builds exist to catch numerically wrong kernels, but
     * comparePackedBuffers() only printed its findings, so such a build still
     * exited 0 and the experiment runner reported success. Fail the process
     * instead, so a mismatch cannot reach the results tables unnoticed.
     */
    const std::size_t comparisons = packedBufferComparisonCount();
    const std::size_t mismatches = packedBufferMismatchCount();
    std::cout << "[reference] " << comparisons << " buffer comparison(s), "
              << mismatches << " with mismatches" << std::endl;
    if (mismatches != 0) {
        std::cerr << "ERROR: " << mismatches << " of " << comparisons
                  << " reference comparisons differed; failing the run."
                  << std::endl;
        return 1;
    }
    if (comparisons == 0) {
        std::cerr << "ERROR: reference validation was enabled but no comparison"
                  << " ran; the check is not covering anything." << std::endl;
        return 1;
    }
#endif
    return 0;
}
