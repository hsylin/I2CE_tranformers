// A common public-API probe compiled against each implementation separately.
// Saved bytes compare numerical outputs directly; allocation failures and
// arithmetic are also checked against explicit invariants and a scalar oracle.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <new>
#include <vector>
#include "gemm_SVE.c"
#include "interleavedPipeline.cc"
#include "softmax.cc"
#include "attentionSmmlaBlocked.h"
#include "attentionStream.h"

static bool fail_aligned_allocation = false;
__attribute__((noinline)) void *operator new(size_t bytes, std::align_val_t alignment) {
    if (fail_aligned_allocation) {
        throw std::bad_alloc();
    }
    void *pointer = nullptr;
    if (posix_memalign(&pointer, size_t(alignment), bytes)) {
        throw std::bad_alloc();
    }
    std::memset(pointer, 0xa5, bytes);
    return pointer;
}
__attribute__((noinline)) void operator delete(void *pointer,
                                               std::align_val_t) noexcept {
    std::free(pointer);
}
__attribute__((noinline)) void operator delete(void *pointer, size_t,
                                               std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

using Matmul = void (*)(const int8_t *, const int8_t *, size_t, size_t, size_t,
                        int8_t *);
static void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("S25 compatibility probe mismatch");
    }
}
static void save(FILE *output, const std::vector<int8_t> &bytes) {
    require(std::fwrite(bytes.data(), 1, bytes.size(), output) == bytes.size());
}
static uint32_t state = 0x56ab1298;
static int8_t sample() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return int8_t(uint8_t(state));
}
static std::vector<int8_t> reference(const std::vector<int8_t> &left,
                                     const std::vector<int8_t> &right, size_t rows,
                                     size_t columns, size_t features) {
    std::vector<int8_t> output(rows * columns * 2);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            for (size_t learner = 0; learner < 2; ++learner) {
                uint32_t sum = 0;
                for (size_t k = 0; k < features; ++k) {
                    sum +=
                        uint32_t(int32_t(left[(row * features + k) * 2 + learner]) *
                                 int32_t(right[(column * features + k) * 2 + learner]));
                }
                output[(row * columns + column) * 2 + learner] = int8_t(uint8_t(sum));
            }
        }
    }
    return output;
}
static void smmla_micro(const int8_t *left, const int8_t *right, size_t rows,
                        size_t columns, size_t features, int8_t *output) {
    attention_smmla::dispatch(left, right, rows, columns, features, output,
                              attention_smmla::micro);
}
static void smmla_blocked(const int8_t *left, const int8_t *right, size_t rows,
                          size_t columns, size_t features, int8_t *output) {
    attention_smmla::dispatch(left, right, rows, columns, features, output,
                              attention_smmla::blocked);
}
static void matrix_cases(FILE *file) {
    const Matmul paths[] = {matmulInterleaved2LearnersToInt8,
                            attentionMatmulSmmlaBlocked, smmla_micro, smmla_blocked};
    for (Matmul path : paths) {
        for (size_t rows : {1u, 7u, 8u, 9u, 17u, 129u}) {
            for (size_t features : {0u, 1u, 15u, 16u, 17u, 128u, 512u, 513u}) {
                const size_t columns = rows == 129 ? 33 : 17;
                std::vector<int8_t> left(rows * features * 2),
                    right(columns * features * 2);
                for (auto &value : left) {
                    value = sample();
                }
                for (auto &value : right) {
                    value = sample();
                }
                const auto before_left = left, before_right = right;
                std::vector<int8_t> output(rows * columns * 2, 0x5a);
                path(left.data(), right.data(), rows, columns, features, output.data());
                require(output == reference(left, right, rows, columns, features));
                require(left == before_left && right == before_right);
                save(file, output);
            }
        }
        std::vector<int8_t> failure_input(1024, 1), failure_output(9 * 17 * 2, 0x5a);
        bool allocation_rejected = false;
        fail_aligned_allocation = true;
        try {
            path(failure_input.data(), failure_input.data(), 9, 17, 16,
                 failure_output.data());
        } catch (const std::bad_alloc &) {
            allocation_rejected = true;
        }
        fail_aligned_allocation = false;
        require(allocation_rejected);
        require(std::all_of(failure_output.begin(), failure_output.end(),
                            [](int8_t value) { return value == 0x5a; }));
        // Both exact input/output aliases preserve the original operands.
        for (bool alias_left : {false, true}) {
            std::vector<int8_t> left(9 * 32 * 2), right(17 * 32 * 2);
            for (auto &value : left) {
                value = sample();
            }
            for (auto &value : right) {
                value = sample();
            }
            auto expected = reference(left, right, 9, 17, 32);
            int8_t *output = alias_left ? left.data() : right.data();
            path(left.data(), right.data(), 9, 17, 32, output);
            require(std::equal(expected.begin(), expected.end(), output));
            save(file, expected);
        }
    }
}
static void softmax_and_stream(FILE *file) {
    Softmax softmax;
    for (size_t keys : {1u, 15u, 16u, 17u, 255u, 256u, 257u, 512u, 513u, 65535u}) {
        std::vector<int8_t> values(3 * keys * 2);
        for (size_t index = 0; index < values.size(); ++index) {
            values[index] = int8_t(index);
        }
        softmax.computeInterleaved2LearnersRows(values.data(), 3, keys);
        save(file, values);
    }
    for (size_t rows : {1u, 15u, 16u, 17u, 33u}) {
        const size_t keys = 129, depth = 17;
        std::vector<int8_t> query(rows * depth * 2), key(keys * depth * 2),
            values(keys * depth * 2);
        for (auto &value : query) {
            value = sample();
        }
        for (auto &value : key) {
            value = sample();
        }
        for (auto &value : values) {
            value = sample();
        }
        std::vector<int8_t> output(rows * depth * 2), borrowed(output.size());
        std::vector<int8_t> scores(rows * keys * 2), probabilities(scores.size()),
            before_scale(output.size());
        std::vector<int8_t> scratch(std::min(rows, size_t(16)) * keys * 2, 0x5a);
        attentionStreamRows(softmax, query.data(), key.data(), values.data(), rows,
                            keys, depth, output.data(), scores.data(),
                            probabilities.data(), before_scale.data());
        attentionStreamRowsWithScratch(softmax, query.data(), key.data(), values.data(),
                                       rows, keys, depth, borrowed.data(),
                                       scratch.data(), scratch.size());
        require(output == borrowed &&
                scores == reference(query, key, rows, keys, depth));
        save(file, scores);
        save(file, probabilities);
        save(file, before_scale);
        save(file, output);
    }
}
int main(int argc, char **argv) {
    require(argc == 2);
    FILE *file = std::fopen(argv[1], "wbx");
    require(file != nullptr);
    matrix_cases(file);
    softmax_and_stream(file);
    require(std::fclose(file) == 0);
    std::puts("PASS: 192 matrix cases, 8 aliases, 4 allocation failures, 10 softmax "
              "and 5 stream comparisons");
}
