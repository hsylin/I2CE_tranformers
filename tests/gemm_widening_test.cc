// Focused regression test for activation widening and its three GEMM callers.
// Include the implementation to exercise the private helper without adding a
// production API. Do not also link gemm_exec.c when building this test.
#include "../Full_NN/src/gemm_exec.c"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int32_t kCanary = 0x31415926;
unsigned int checks = 0;

void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

class GuardedTail {
 public:
  explicit GuardedTail(size_t bytes) {
    const long page_size = sysconf(_SC_PAGESIZE);
    Check(page_size > 0, "page size");
    page_size_ = static_cast<size_t>(page_size);
    Check(bytes + sizeof(kCanary) <= page_size_, "guarded buffer size");
    mapping_ = mmap(nullptr, 3 * page_size_, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Check(mapping_ != MAP_FAILED, "mmap");
    auto* middle = static_cast<unsigned char*>(mapping_) + page_size_;
    Check(mprotect(middle, page_size_, PROT_READ | PROT_WRITE) == 0,
          "mprotect");
    data_ = middle + page_size_ - bytes;
  }

  ~GuardedTail() { munmap(mapping_, 3 * page_size_); }
  GuardedTail(const GuardedTail&) = delete;
  GuardedTail& operator=(const GuardedTail&) = delete;
  void* data() { return data_; }

 private:
  void* mapping_ = nullptr;
  void* data_ = nullptr;
  size_t page_size_ = 0;
};

void TestWideningTails() {
  gemm_widen_input_sve(nullptr, nullptr, 0);
  const uint32_t lengths[] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16,
                              17, 31, 32, 33, 127, 255, 256, 257, 511};
  for (uint32_t count : lengths) {
    GuardedTail input(count);
    GuardedTail output(static_cast<size_t>(count) * sizeof(int32_t));
    auto* src = static_cast<int8_t*>(input.data());
    auto* dst = static_cast<int32_t*>(output.data());
    src[-1] = 37;
    dst[-1] = kCanary;
    for (uint32_t i = 0; i < count; ++i) {
      src[i] = static_cast<int8_t>(static_cast<int>(i % 256) - 128);
      dst[i] = kCanary;
    }
    gemm_widen_input_sve(src, dst, count);
    for (uint32_t i = 0; i < count; ++i) {
      const int32_t expected = static_cast<int32_t>(i % 256) - 128;
      Check(dst[i] == expected, "signed widening value");
      Check(src[i] == expected, "widening changed input");
    }
    Check(src[-1] == 37 && dst[-1] == kCanary, "prefix canary");
    // Both buffers end at an inaccessible page: overreads/overwrites fault.
    ++checks;
  }
}

void TestOverlappingBuffers() {
  const uint32_t lengths[] = {0, 1, 3, 4, 5, 9, 17};
  const size_t source_offsets[] = {0, 1, 3, 4, 8, 20, 63};
  const size_t destination_offsets[] = {0, 1, 7, 16};
  for (uint32_t count : lengths) {
    for (size_t source_offset : source_offsets) {
      for (size_t destination_offset : destination_offsets) {
        std::vector<int32_t> actual(128, 0);
        auto* bytes = reinterpret_cast<int8_t*>(actual.data());
        for (size_t i = 0; i < actual.size() * sizeof(int32_t); ++i) {
          bytes[i] = static_cast<int8_t>(static_cast<int>(i % 256) - 128);
        }
        std::vector<int32_t> expected = actual;
        const auto* reference_src =
            reinterpret_cast<const int8_t*>(expected.data()) + source_offset;
        int32_t* reference_dst = expected.data() + destination_offset;
        // The old forward-copy semantics, including source bytes overwritten
        // by earlier iterations when the ranges overlap.
        for (uint32_t i = 0; i < count; ++i) {
          reference_dst[i] = static_cast<int32_t>(reference_src[i]);
        }
        gemm_widen_input_sve(bytes + source_offset,
                            actual.data() + destination_offset, count);
        Check(actual == expected, "overlap changed forward-copy behavior");
        ++checks;
      }
    }
  }
}

using ExtendedGemm = void (*)(gemm_t, const int8_t*, const uint32_t*,
                             const int8_t*, const int32_t*, int32_t*, uint8_t,
                             const int32_t*, int32_t*, uint32_t);
using PublicGemm = void (*)(gemm_t, const int8_t*, const uint32_t*,
                           const int8_t*, const int32_t*, int32_t*, uint8_t);

struct Variant {
  uint32_t learners;
  bool different_indices;
  ExtendedGemm extended;
  PublicGemm legacy;
};

// Derive the packed-index width from the generated header instead of hard-coding
// it. CB_SIZE also fixes which N_SVE_REG_CB_* path the kernels compile, so the
// two must agree: a CB8 header with bits_per_cb=2 would exercise the wrong
// codebook-register path. Building this file against a different generated
// header is therefore the only correct way to cover another codebook size.
constexpr uint8_t BitsForCodebook(uint32_t entries) {
  uint8_t bits = 0;
  while ((1u << bits) < entries) ++bits;
  return bits;
}

void TestGemmCase(const Variant& variant, gemm_t shape, bool use_bias,
                  bool use_cached_codebook, int workspace_mode) {
  constexpr uint8_t kBits = BitsForCodebook(CB_SIZE);
  constexpr uint32_t kEntries = CB_SIZE;
  constexpr uint32_t kIndicesPerWord = 32u / kBits;
  static_assert(kEntries == (1u << kBits),
                "CB_SIZE must be a power of two for packed-index coverage");
  const uint32_t learners = variant.learners;
  const uint32_t input_count = shape.seq_len * shape.input_size * learners;
  const uint32_t output_count = shape.seq_len * shape.output_size * learners;
  const uint32_t streams = variant.different_indices ? learners : 1u;
  shape.n_words_row =
      (shape.input_size + kIndicesPerWord - 1) / kIndicesPerWord;

  std::vector<int8_t> input(input_count);
  for (uint32_t i = 0; i < input_count; ++i) {
    input[i] = static_cast<int8_t>(static_cast<int>((i * 37u) % 256u) - 128);
  }
  const std::vector<int8_t> original_input = input;
  std::vector<int8_t> codebook(kEntries * learners);
  std::vector<int32_t> cached_codebook(codebook.size());
  for (size_t i = 0; i < codebook.size(); ++i) {
    codebook[i] = static_cast<int8_t>(static_cast<int>((i * 5) % 17) - 8);
    cached_codebook[i] = codebook[i];
  }
  std::vector<int32_t> bias(shape.output_size * learners);
  for (size_t i = 0; i < bias.size(); ++i) {
    bias[i] = static_cast<int32_t>(i) - 7;
  }
  std::vector<uint32_t> packed(
      shape.output_size * shape.n_words_row * streams, 0);
  std::vector<int32_t> expected(output_count, 0);
  for (uint32_t out = 0; out < shape.output_size; ++out) {
    for (uint32_t k = 0; k < shape.input_size; ++k) {
      for (uint32_t stream = 0; stream < streams; ++stream) {
        const uint32_t index = (out * 3 + k * 5 + stream * 2) % kEntries;
        const uint32_t word = (out * shape.n_words_row + k / kIndicesPerWord) *
                                  streams + stream;
        packed[word] |= index << ((k % kIndicesPerWord) * kBits);
      }
    }
    for (uint32_t row = 0; row < shape.seq_len; ++row) {
      for (uint32_t learner = 0; learner < learners; ++learner) {
        int32_t value = use_bias ? bias[out * learners + learner] : 0;
        for (uint32_t k = 0; k < shape.input_size; ++k) {
          const uint32_t stream = variant.different_indices ? learner : 0;
          const uint32_t index = (out * 3 + k * 5 + stream * 2) % kEntries;
          value += input[(row * shape.input_size + k) * learners + learner] *
                   codebook[index * learners + learner];
        }
        expected[(row * shape.output_size + out) * learners + learner] = value;
      }
    }
  }

  std::vector<int32_t> output(output_count + 2, kCanary);
  std::vector<int32_t> workspace(input_count + 9, kCanary);
  uint32_t capacity = 0;
  if (workspace_mode == 1) capacity = input_count == 0 ? 0 : input_count - 1;
  if (workspace_mode == 2) capacity = input_count;
  if (workspace_mode == 3) capacity = input_count + 7;
  if (workspace_mode == 4) {
    variant.legacy(shape, input.data(), packed.data(), codebook.data(),
                   use_bias ? bias.data() : nullptr, output.data() + 1, kBits);
  } else {
    variant.extended(shape, input.data(), packed.data(), codebook.data(),
                     use_bias ? bias.data() : nullptr, output.data() + 1, kBits,
                     use_cached_codebook ? cached_codebook.data() : nullptr,
                     workspace_mode == 0 ? nullptr : workspace.data() + 1,
                     capacity);
  }
  for (uint32_t i = 0; i < output_count; ++i) {
    Check(output[i + 1] == expected[i],
          "GEMM differs from independent reference");
  }
  Check(output.front() == kCanary && output.back() == kCanary, "output canary");
  Check(input == original_input, "GEMM changed input");
  const bool uses_workspace = workspace_mode >= 2 && workspace_mode <= 3 &&
                              shape.seq_len != 0 && shape.output_size != 0 &&
                              shape.input_size != 0;
  for (size_t i = 0; i < workspace.size(); ++i) {
    const int32_t value = uses_workspace && i > 0 && i <= input_count
                              ? static_cast<int32_t>(input[i - 1]) : kCanary;
    Check(workspace[i] == value, "workspace content or canary");
  }
  ++checks;
}

void TestGemmWrappers() {
  Check(svcntw() == N_SVE_LANES, "wrapper test needs generated SVE lane count");
  const Variant variants[] = {
      {2, false, gemm_exec_compact_int_sve_interleaved_2Learners_same_seq_ex,
       gemm_exec_compact_int_sve_interleaved_2Learners_same_seq},
      {4, false, gemm_exec_compact_int_sve_interleaved_4Learners_same_seq_ex,
       gemm_exec_compact_int_sve_interleaved_4Learners_same_seq},
      {4, true, gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq_ex,
       gemm_exec_compact_int_sve_interleaved_4Learners_diff_seq}};
  const gemm_t shapes[] = {{3, 0, 3, 0}, {1, 1, 1, 0}, {3, 3, 3, 0},
                           {3, 5, 3, 0}, {3, 9, 3, 0}, {3, 10, 3, 0},
                           {3, 11, 3, 0}, {3, 17, 3, 0}, {3, 31, 3, 0},
                           {0, 11, 3, 0}, {3, 11, 0, 0}};
  for (const Variant& variant : variants) {
    for (gemm_t shape : shapes) {
      for (bool use_bias : {false, true}) {
        for (bool cached_codebook : {false, true}) {
          for (int mode = 0; mode < 5; ++mode) {
            TestGemmCase(variant, shape, use_bias, cached_codebook, mode);
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool helper_only =
      argc == 2 && std::strcmp(argv[1], "--helper-only") == 0;
  Check(argc == 1 || helper_only, "usage: gemm_widening_test [--helper-only]");
  TestWideningTails();
  TestOverlappingBuffers();
  if (!helper_only) TestGemmWrappers();
  std::printf("PASS: %u cases; SVE=%llu bits; CB_SIZE=%d (%d-bit indexes); %s\n",
              checks, static_cast<unsigned long long>(svcntb() * 8),
              static_cast<int>(CB_SIZE),
              static_cast<int>(BitsForCodebook(CB_SIZE)),
              helper_only ? "helper only" : "helper and three GEMM wrappers");
}
