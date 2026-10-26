#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

static void require_at(bool passed, int line) {
    if (!passed) {
        std::fprintf(stderr, "Attention contract failed at line %d\n", line);
        std::abort();
    }
}

#define require(condition) require_at((condition), __LINE__)

static bool fail_aligned_allocation = false;
static unsigned char allocation_poison = 0xa5;
static size_t aligned_allocations = 0;
static size_t aligned_frees = 0;

__attribute__((noinline)) void *operator new(size_t bytes, std::align_val_t alignment) {
    if (fail_aligned_allocation) {
        throw std::bad_alloc();
    }
    void *pointer = nullptr;
    if (posix_memalign(&pointer, size_t(alignment), bytes)) {
        throw std::bad_alloc();
    }
    std::memset(pointer, allocation_poison, bytes);
    ++aligned_allocations;
    return pointer;
}
__attribute__((noinline)) void operator delete(void *pointer,
                                               std::align_val_t) noexcept {
    if (pointer) {
        ++aligned_frees;
    }
    std::free(pointer);
}
__attribute__((noinline)) void operator delete(void *pointer, size_t,
                                               std::align_val_t alignment) noexcept {
    ::operator delete(pointer, alignment);
}

#define I2CE_TEST_ATTENTION_SDOT
#include "transformer_layers/attentionSdotGeneric.h"
#ifdef I2CE_TEST_ATTENTION_BLOCKED
#include "transformer_layers/attentionSdotBlocked.h"
static constexpr auto strategy = attention_sdot::blocked;
#elif defined(I2CE_TEST_ATTENTION_MICRO4)
#include "transformer_layers/attentionSdotMicro.h"
static constexpr auto strategy = attention_sdot::micro_4x16;
#else
#include "transformer_layers/attentionSdotGeneric.h"
static constexpr auto strategy = attention_sdot::micro_8x8;
#endif

static uint32_t state = 0x91827364u;
static int8_t sample() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return int8_t(uint8_t(state));
}
static uint32_t expected(const int8_t *left, const int8_t *right, size_t row,
                         size_t column, size_t features, size_t learner) {
    uint32_t sum = 0;
    for (size_t k = 0; k < features; ++k) {
        sum += uint32_t(int32_t(left[(row * features + k) * 2u + learner]) *
                        int32_t(right[(column * features + k) * 2u + learner]));
    }
    return sum;
}
static void execute(const int8_t *left, const int8_t *right, size_t rows,
                    size_t columns, size_t features, int8_t *output) {
    attention_sdot::dispatch(left, right, rows, columns, features, output, strategy);
}
static void arithmetic(size_t rows, size_t columns, size_t features) {
    std::vector<int8_t> left(rows * features * 2u), right(columns * features * 2u);
    for (auto &value : left) {
        value = sample();
    }
    for (auto &value : right) {
        value = sample();
    }
    const auto left_before = left, right_before = right;
    const size_t elements = rows * columns * 2u;
    std::vector<int8_t> output(elements + 2u, 0x5a);
    std::vector<uint32_t> full(elements + 2u, 0xa5a5a5a5u);
    attention_sdot::observed = full.data() + 1u;
    execute(left.data(), right.data(), rows, columns, features, output.data() + 1u);
    attention_sdot::observed = nullptr;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            for (size_t learner = 0; learner < 2u; ++learner) {
                const size_t index = (row * columns + column) * 2u + learner;
                const uint32_t want =
                    expected(left.data(), right.data(), row, column, features, learner);
                require(full[index + 1u] == want);
                require(output[index + 1u] == int8_t(uint8_t(want)));
            }
        }
    }
    require(output.front() == 0x5a && output.back() == 0x5a);
    require(full.front() == 0xa5a5a5a5u && full.back() == 0xa5a5a5a5u);
    require(left == left_before && right == right_before);
}
static void layouts() {
    for (unsigned poison : {0x5au, 0xa5u}) {
        for (size_t rows : {1u, 3u, 4u, 5u, 16u}) {
            for (size_t features :
                 {1u, 15u, 16u, 17u, 127u, 128u, 129u, 511u, 512u, 513u}) {
                for (size_t first_row : {0u, 1u}) {
                    for (size_t first_k : {0u, 16u}) {
                        const size_t total_rows = rows + first_row;
                        const size_t total_k = features + first_k;
                        const size_t padded_rows = (rows + 15u) / 16u * 16u;
                        const size_t padded_k = (features + 15u) / 16u * 16u;
                        const size_t plane = padded_rows * padded_k;
                        std::vector<int8_t> input(total_rows * total_k * 2u);
                        for (auto &value : input) {
                            value = sample();
                        }
                        const auto before = input;
                        for (bool right : {false, true}) {
                            std::vector<int8_t> output(2u * plane + 64u,
                                                       int8_t(poison));
                            auto want = output;
                            attention_sdot::pack_sdot(
                                input.data(), total_rows, total_k, first_row, first_k,
                                padded_rows, padded_k, output.data() + 32u, right);
                            for (size_t learner = 0; learner < 2u; ++learner) {
                                for (size_t row = 0; row < padded_rows; ++row) {
                                    for (size_t k = 0; k < padded_k; ++k) {
                                        const size_t at =
                                            right ? (row / 4u) * 4u * padded_k +
                                                        (k / 4u) * 16u +
                                                        (row % 4u) * 4u + k % 4u
                                                  : row * padded_k + k;
                                        want[32u + learner * plane + at] =
                                            row < rows && k < features
                                                ? input[((first_row + row) * total_k +
                                                         first_k + k) *
                                                            2u +
                                                        learner]
                                                : 0;
                                    }
                                }
                            }
                            require(output == want && input == before);
                        }
                    }
                }
            }
        }
    }
}
static void wrapping_core() {
    const size_t features = 131088u;
    std::vector<int8_t> left(4u * features * 2u, -128),
        right(16u * features * 2u, -128);
    std::vector<int8_t> packed_left(left.size()), packed_right(right.size());
    attention_sdot::pack_sdot(left.data(), 4u, features, 0u, 0u, 4u, features,
                              packed_left.data(), false);
    attention_sdot::pack_sdot(right.data(), 16u, features, 0u, 0u, 16u, features,
                              packed_right.data(), true);
    uint32_t accumulator[4u * 20u];
    std::fill(std::begin(accumulator), std::end(accumulator), 0xfedcba98u);
    attention_sdot::core_4x16(packed_left.data(), packed_right.data(), accumulator,
                              features, 20u);
    const uint32_t product = uint32_t(uint64_t(features) * 16384u);
    for (size_t row = 0; row < 4u; ++row) {
        for (size_t column = 0; column < 20u; ++column) {
            require(accumulator[row * 20u + column] ==
                    0xfedcba98u + (column < 16u ? product : 0u));
        }
    }
}
static void wrapping_generic_core() {
    const size_t features = 131088u;
    std::vector<int8_t> left(8u * features * 2u, -128), right(8u * features * 2u, -128);
    std::vector<int8_t> packed_left(left.size()), packed_right(right.size());
    attention_sdot::pack_sdot(left.data(), 8u, features, 0u, 0u, 8u, features,
                              packed_left.data(), false);
    attention_sdot::pack_sdot(right.data(), 8u, features, 0u, 0u, 8u, features,
                              packed_right.data(), true);
    uint32_t accumulator[8u * 12u];
    std::fill(std::begin(accumulator), std::end(accumulator), 0xfedcba98u);
    attention_sdot::core_8x8(packed_left.data(), packed_right.data(), accumulator,
                             features, 12u);
    const uint32_t product = uint32_t(uint64_t(features) * 16384u);
    for (size_t row = 0; row < 8u; ++row) {
        for (size_t column = 0; column < 12u; ++column) {
            require(accumulator[row * 12u + column] ==
                    0xfedcba98u + (column < 8u ? product : 0u));
        }
    }
}
static void generic_strategy_contract() {
    std::vector<int8_t> left(9u * 16u * 2u, -128);
    std::vector<int8_t> right(17u * 16u * 2u, 127);
    std::vector<int8_t> output(9u * 17u * 2u);
    const auto before = attention_sdot::counters;
    attention_sdot::dispatch(left.data(), right.data(), 9u, 17u, 16u, output.data(),
                             attention_sdot::micro_8x8);
    const auto after = attention_sdot::counters;
    require(after.calls - before.calls == 1u);
    require(after.core_calls - before.core_calls == 12u);
    require(after.a_packs - before.a_packs == 2u);
    require(after.b_packs - before.b_packs == 6u);
    require(after.requested - before.requested == 1024u);
    require(after.c_initialized - before.c_initialized == 3072u);
    require(after.allocations - before.allocations == 1u);
    require(after.frees - before.frees == 1u);
}
static void aliases_and_failures() {
    constexpr size_t rows = 5u, columns = 17u, features = 129u;
    const size_t elements = rows * columns * 2u;
    for (bool alias_left : {false, true}) {
        std::vector<int8_t> left(std::max(rows * features * 2u, elements));
        std::vector<int8_t> right(std::max(columns * features * 2u, elements));
        for (auto &value : left) {
            value = sample();
        }
        for (auto &value : right) {
            value = sample();
        }
        const auto before_left = left, before_right = right;
        int8_t *output = alias_left ? left.data() : right.data();
        execute(left.data(), right.data(), rows, columns, features, output);
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns; ++column) {
                for (size_t learner = 0; learner < 2u; ++learner) {
                    require(
                        output[(row * columns + column) * 2u + learner] ==
                        int8_t(uint8_t(expected(before_left.data(), before_right.data(),
                                                row, column, features, learner))));
                }
            }
        }
        left = before_left;
        right = before_right;
        fail_aligned_allocation = true;
        bool threw = false;
        try {
            execute(left.data(), right.data(), rows, columns, features,
                    alias_left ? left.data() : right.data());
        } catch (const std::bad_alloc &) {
            threw = true;
        }
        fail_aligned_allocation = false;
        require(threw && left == before_left && right == before_right);
    }
}
static void invalid_extents() {
    int8_t input[64] = {}, output[64];
    std::memset(output, 0x5a, sizeof(output));
    for (unsigned choice = 0; choice < 7u; ++choice) {
        bool threw = false;
        try {
            execute(choice == 0u ? nullptr : input, choice == 1u ? nullptr : input,
                    choice == 3u ? size_t(UINT32_MAX) + 1u : 1u,
                    choice == 4u ? size_t(UINT32_MAX) + 1u : 1u,
                    choice == 5u ? size_t(UINT32_MAX) + 1u : 1u,
                    choice == 2u   ? nullptr
                    : choice == 6u ? reinterpret_cast<int8_t *>(UINTPTR_MAX)
                                   : output);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw);
        for (auto value : output) {
            require(value == 0x5a);
        }
    }
}
static void guarded_tail() {
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    std::vector<void *> mappings;
    auto guard = [&](size_t bytes) {
        void *allocation = mmap(nullptr, 2u * page, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(allocation != MAP_FAILED && bytes <= page);
        require(mprotect(static_cast<char *>(allocation) + page, page, PROT_NONE) == 0);
        mappings.push_back(allocation);
        return reinterpret_cast<int8_t *>(allocation) + page - bytes;
    };
    int8_t *left = guard(3u * 17u * 2u);
    int8_t *right = guard(17u * 17u * 2u);
    int8_t *output = guard(3u * 17u * 2u);
    std::fill(left, left + 3u * 17u * 2u, -128);
    std::fill(right, right + 17u * 17u * 2u, 127);
    execute(left, right, 3u, 17u, 17u, output);
    const int8_t want = int8_t(uint8_t(uint32_t(-128 * 127 * 17)));
    for (size_t i = 0; i < 3u * 17u * 2u; ++i) {
        require(output[i] == want);
    }
    for (void *allocation : mappings) {
        require(munmap(allocation, 2u * page) == 0);
    }
}
static void source_work() {
    std::vector<int8_t> left(32u * 512u * 2u, 0), right(64u * 512u * 2u, 0);
    std::vector<int8_t> output(32u * 64u * 2u, 0x5a);
    const auto before = attention_sdot::counters;
    execute(left.data(), right.data(), 32u, 64u, 512u, output.data());
    const auto after = attention_sdot::counters;
    for (auto value : output) {
        require(value == 0);
    }
    require(after.a_bytes - before.a_bytes == 32768u);
    require(after.c_initialized - before.c_initialized == 16384u);
    require(after.sdot - before.sdot == 131072u);
    require(after.allocations - before.allocations == 1u);
    require(after.frees - before.frees == 1u);
#ifdef I2CE_TEST_ATTENTION_BLOCKED
    require(after.a_packs - before.a_packs == 8u);
    require(after.b_packs - before.b_packs == 8u);
    require(after.b_bytes - before.b_bytes == 65536u);
    require(after.core_calls - before.core_calls == 256u);
    require(after.requested - before.requested == 114688u);
#elif defined(I2CE_TEST_ATTENTION_MICRO4)
    require(after.a_packs - before.a_packs == 8u);
    require(after.b_packs - before.b_packs == 32u);
    require(after.b_bytes - before.b_bytes == 524288u);
    require(after.core_calls - before.core_calls == 64u);
    require(after.requested - before.requested == 20992u);
#else
    require(after.a_packs - before.a_packs == 4u);
    require(after.b_packs - before.b_packs == 32u);
    require(after.b_bytes - before.b_bytes == 262144u);
    require(after.core_calls - before.core_calls == 64u);
    require(after.requested - before.requested == 16896u);
#endif
}
int main() {
    require(svcntb() == 16u);
    source_work();
    layouts();
    wrapping_core();
    wrapping_generic_core();
    generic_strategy_contract();
    unsigned cases = 0;
    const size_t widths[] = {0u,  1u,   15u,  16u,  17u,  63u,
                             64u, 127u, 128u, 129u, 512u, 513u};
    for (unsigned poison : {0x5au, 0xa5u}) {
        allocation_poison = poison;
        for (size_t rows : {0u, 1u, 3u, 4u, 5u, 15u, 16u, 17u, 127u, 128u, 129u}) {
            for (size_t columns :
                 {0u, 1u, 15u, 16u, 17u, 31u, 32u, 33u, 127u, 128u, 129u}) {
                arithmetic(rows, columns, widths[(rows + 3u * columns) % 12u]);
                ++cases;
            }
        }
    }
    aliases_and_failures();
    invalid_extents();
    guarded_tail();
    require(aligned_allocations == aligned_frees);
    std::printf("PASS: %u Attention arithmetic cases, layout, alias, rejection and "
                "ownership checks\n",
                cases);
}
