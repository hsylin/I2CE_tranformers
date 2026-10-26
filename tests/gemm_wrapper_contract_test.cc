// Check the public wrapper contracts against a small independently decoded GEMM.
// Values stay small enough that both integer and float accumulations are exact.
#include "gemm_exec.h"
#include "gemm_exec_internal.h"
#include "codebooks_def.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

unsigned checks = 0;

template <typename Input, typename Output>
using Wrapper = void (*)(gemm_t, const Input *, const uint32_t *, const Input *,
                         const Output *, Output *, uint8_t);

template <typename Input, typename Output>
void checkCase(const char *name, Wrapper<Input, Output> wrapper, uint32_t learners,
               bool separate_indices, gemm_t shape, uint8_t bits, bool use_bias,
               bool empty_index_rows) {
    const uint32_t entries = 1u << bits;
    const uint32_t indices_per_word = bits == 0 ? 32 : 32 / bits;
    shape.n_words_row =
        empty_index_rows ? 0
                         : (shape.input_size + indices_per_word - 1) / indices_per_word;
    const uint32_t streams = separate_indices ? learners : 1;
    std::vector<Input> input(shape.seq_len * shape.input_size * learners);
    // The configured register kernel may load its full table even for smaller
    // logical codebooks. Provide padding; indices still address logical entries.
    std::vector<Input> codebook(std::max(entries, 256u) * learners, 0);
    std::vector<Output> bias(shape.output_size * learners);
    std::vector<uint32_t> indices(shape.output_size * shape.n_words_row * streams, 0);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<Input>(static_cast<int>(i % 17) - 8);
    }
    for (size_t i = 0; i < entries * learners; ++i) {
        codebook[i] = static_cast<Input>(static_cast<int>((i * 3) % 13) - 6);
    }
    for (size_t i = 0; i < bias.size(); ++i) {
        bias[i] = static_cast<Output>(static_cast<int>(i) - 5);
    }
    for (uint32_t out = 0; out < shape.output_size; ++out) {
        for (uint32_t k = 0; k < shape.input_size && shape.n_words_row; ++k) {
            for (uint32_t stream = 0; stream < streams; ++stream) {
                const uint32_t index = (out * 3 + k * 5 + stream) % entries;
                const size_t word =
                    (out * shape.n_words_row + k / indices_per_word) * streams + stream;
                indices[word] |= index << ((k % indices_per_word) * bits);
            }
        }
    }
    const size_t count = shape.seq_len * shape.output_size * learners;
    const Output sentinel = static_cast<Output>(12345);
    std::vector<Output> actual(count + 2, sentinel);
    std::vector<Output> expected(count + 2, sentinel);
    for (uint32_t row = 0; row < shape.seq_len; ++row) {
        for (uint32_t out = 0; out < shape.output_size; ++out) {
            for (uint32_t learner = 0; learner < learners; ++learner) {
                Output sum = use_bias ? bias[out * learners + learner] : 0;
                if (bits && shape.n_words_row) {
                    for (uint32_t k = 0; k < shape.input_size; ++k) {
                        const uint32_t stream = separate_indices ? learner : 0;
                        const uint32_t index = (out * 3 + k * 5 + stream) % entries;
                        sum +=
                            static_cast<Output>(
                                input[(row * shape.input_size + k) * learners +
                                      learner]) *
                            static_cast<Output>(codebook[index * learners + learner]);
                    }
                }
                expected[1 + (row * shape.output_size + out) * learners + learner] =
                    sum;
            }
        }
    }
    const auto original_input = input;
    const auto original_indices = indices;
    const auto original_codebook = codebook;
    wrapper(shape, input.data(), indices.data(), codebook.data(),
            use_bias ? bias.data() : nullptr, actual.data() + 1, bits);
    if (actual != expected || input != original_input || indices != original_indices ||
        codebook != original_codebook) {
        std::fprintf(
            stderr,
            "FAIL: %s rows=%u k=%u outputs=%u bits=%u bias=%d empty_indices=%d\n", name,
            shape.seq_len, shape.input_size, shape.output_size, bits, use_bias,
            empty_index_rows);
        std::exit(1);
    }
    ++checks;
}

template <typename Input, typename Output>
void checkWrapper(const char *name, Wrapper<Input, Output> wrapper, uint32_t learners,
                  bool separate_indices) {
    const gemm_t shapes[] = {{0, 5, 3, 0}, {3, 5, 0, 0}, {3, 0, 3, 0},
                             {1, 1, 1, 0}, {3, 5, 3, 0}, {5, 17, 7, 0}};
    // 0 exercises bias-only; 4 exceeds this fixture's SVE table capacity;
    // 9 also exercises the explicit unsupported-index-width fallback.
    for (uint8_t bits : {0, 2, 3, 4, 9}) {
        for (bool bias : {false, true}) {
            for (const gemm_t shape : shapes) {
                checkCase(name, wrapper, learners, separate_indices, shape, bits, bias,
                          false);
            }
            checkCase(name, wrapper, learners, separate_indices, {3, 5, 3, 0}, bits,
                      bias, true);
        }
    }
}

// Adapt the internal cached-codebook API to the same independent oracle.
// The caller retains ownership and the wrapper must not modify this mirror.
template <uint32_t Learners, bool SeparateIndices, bool UseCache, int WorkspaceMode = 0>
void cachedWrapper(gemm_t shape, const int8_t *input, const uint32_t *indices,
                   const int8_t *codebook, const int32_t *bias, int32_t *output,
                   uint8_t bits) {
    const size_t entries = std::max(1u << bits, 256u) * Learners;
    std::vector<int32_t> widened(codebook, codebook + entries);
    const auto original = widened;
    const int32_t *cached = UseCache ? widened.data() : nullptr;
    const uint32_t input_count = shape.seq_len * shape.input_size * Learners;
    constexpr int32_t sentinel = 0x31415926;
    std::vector<int32_t> workspace(input_count + 9, sentinel);
    uint32_t capacity = input_count;
    if constexpr (WorkspaceMode == 1) {
        capacity = input_count ? input_count - 1 : 0;
    }
    if constexpr (WorkspaceMode == 3) {
        capacity = input_count + 7;
    }
    int32_t *borrowed = WorkspaceMode == 0 ? nullptr : workspace.data() + 1;

    if constexpr (Learners == 2) {
        gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex(
            shape, input, indices, codebook, bias, output, bits, cached, borrowed,
            capacity);
    } else if constexpr (SeparateIndices) {
        gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq_ex(
            shape, input, indices, codebook, bias, output, bits, cached, borrowed,
            capacity);
    } else {
        gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_ex(
            shape, input, indices, codebook, bias, output, bits, cached, borrowed,
            capacity);
    }
    const bool uses_workspace = WorkspaceMode >= 2 && shape.seq_len &&
                                shape.output_size && shape.input_size &&
                                shape.n_words_row && bits && (1u << bits) <= CB_SIZE;
    for (size_t i = 0; i < workspace.size(); ++i) {
        const int32_t expected = uses_workspace && i > 0 && i <= input_count
                                     ? static_cast<int32_t>(input[i - 1])
                                     : sentinel;
        if (workspace[i] != expected) {
            std::fprintf(stderr,
                         "FAIL: borrowed activation workspace changed incorrectly\n");
            std::exit(1);
        }
    }
    if (widened != original) {
        std::fprintf(stderr, "FAIL: borrowed codebook changed\n");
        std::exit(1);
    }
}

template <int WorkspaceMode, bool UseCache> void checkWorkspaceMode() {
    checkWrapper("workspace2", cachedWrapper<2, false, UseCache, WorkspaceMode>, 2,
                 false);
    checkWrapper("workspace4/shared", cachedWrapper<4, false, UseCache, WorkspaceMode>,
                 4, false);
    checkWrapper("workspace4/separate", cachedWrapper<4, true, UseCache, WorkspaceMode>,
                 4, true);
}

} // namespace

int main() {
    checkWrapper("int2/shared",
                 gemm_exec_compact_int_sve_interleaved_2Learners_same_seq, 2, false);
    checkWrapper("int4/shared",
                 gemm_exec_compact_int_sve_interleaved_4Learners_same_seq, 4, false);
    checkWrapper("int4/separate",
                 gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq, 4, true);
    checkWrapper("float2/shared",
                 gemm_exec_compact_sve_fp32_interleaved_2Learners_same_seq, 2, false);
    checkWrapper("float4/shared",
                 gemm_exec_compact_sve_fp32_interleaved_4Learners_same_seq, 4, false);
    checkWrapper("float4/separate",
                 gemm_exec_compact_sve_fp32_interleaved_4Learners_diff_seq, 4, true);
    checkWrapper("cache2", cachedWrapper<2, false, true>, 2, false);
    checkWrapper("cache4/shared", cachedWrapper<4, false, true>, 4, false);
    checkWrapper("cache4/separate", cachedWrapper<4, true, true>, 4, true);
    checkWrapper("null-cache2", cachedWrapper<2, false, false>, 2, false);
    checkWrapper("null-cache4/shared", cachedWrapper<4, false, false>, 4, false);
    checkWrapper("null-cache4/separate", cachedWrapper<4, true, false>, 4, true);
    checkWorkspaceMode<1, false>();
    checkWorkspaceMode<1, true>();
    checkWorkspaceMode<2, false>();
    checkWorkspaceMode<2, true>();
    checkWorkspaceMode<3, false>();
    checkWorkspaceMode<3, true>();
    std::printf("PASS: %u independent wrapper cases\n", checks);
}
