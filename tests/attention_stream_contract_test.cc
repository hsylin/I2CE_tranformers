// Derived from the preserved S25 streaming correctness oracle by hsylin.
// Independent arithmetic and LUT references check every quantization boundary.
// Allocation hooks poison fresh allocations and inject bounded failures; they are
// compiled only into this test translation unit, never into production.
#include <algorithm>
#include <arm_sve.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <new>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
static void check(bool passed, const char *condition, int line) {
    if (!passed) {
        std::fprintf(stderr, "check failed at line %d: %s\n", line, condition);
        std::abort();
    }
}
#define require(condition) check((condition), #condition, __LINE__)

static bool track_allocations = false, fail_general_allocation = false;
static size_t allocations = 0, frees = 0, aligned_attempts = 0, fail_aligned_at = 0;
__attribute__((noinline)) void *operator new(size_t n) {
    if (track_allocations) {
        ++allocations;
        if (fail_general_allocation) {
            throw std::bad_alloc();
        }
    }
    void *p = std::malloc(n ? n : 1);
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}
__attribute__((noinline)) void *operator new[](size_t n) {
    return ::operator new(n);
}
__attribute__((noinline)) void operator delete(void *p) noexcept {
    if (p && track_allocations) {
        ++frees;
    }
    std::free(p);
}
__attribute__((noinline)) void operator delete[](void *p) noexcept {
    ::operator delete(p);
}
__attribute__((noinline)) void operator delete(void *p, size_t) noexcept {
    ::operator delete(p);
}
__attribute__((noinline)) void operator delete[](void *p, size_t) noexcept {
    ::operator delete(p);
}
#define I2CE_TEST_ATTENTION_SDOT
__attribute__((noinline)) void *operator new(size_t n, std::align_val_t a) {
    if (track_allocations) {
        ++aligned_attempts;
        if (fail_aligned_at && aligned_attempts == fail_aligned_at) {
            throw std::bad_alloc();
        }
        ++allocations;
    }
    void *p = nullptr;
    if (posix_memalign(&p, (size_t)a, n ? n : 64)) {
        throw std::bad_alloc();
    }
    if (track_allocations) {
        std::memset(p, 0xa5, n);
    }
    return p;
}
__attribute__((noinline)) void operator delete(void *p, std::align_val_t) noexcept {
    if (p && track_allocations) {
        ++frees;
    }
    std::free(p);
}
__attribute__((noinline)) void operator delete(void *p, size_t,
                                               std::align_val_t a) noexcept {
    ::operator delete(p, a);
}
#include "gemm_SVE.c"
#include "interleavedPipeline.cc"
#include "softmax.cc"
#define I2CE_TEST_OVERWRITE
#include "staticActivation.h"
#define I2CE_TEST_ATTENTION_TILE
#include "attentionTile.h"
static size_t wide_words = 0, low_bytes = 0, stream_cases = 0, soft_cases = 0,
              soft_bytes = 0, rejects = 0, page_cases = 0, owner_cases = 0;
static uint32_t observed[16 * 513 * 2];
static void check_matmul_intermediate(const int8_t *a, const int8_t *b, size_t m,
                                      size_t n, size_t k, int8_t *out) {
    require(m <= 16 && n <= 513);
    std::fill(std::begin(observed), std::end(observed), 0xdeadbeef);
    attention_sdot::observed = observed;
    attentionMatmulTiled(a, b, m, n, k, out);
    attention_sdot::observed = nullptr;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < 2; ++l) {
                uint32_t ref = 0;
                for (size_t z = 0; z < k; ++z) {
                    ref += uint32_t(int32_t(a[2 * (i * k + z) + l]) *
                                    int32_t(b[2 * (j * k + z) + l]));
                }
                size_t at = 2 * (i * n + j) + l;
                require(uint8_t(out[at]) == uint8_t(ref));
                ++low_bytes;
                if (CFG_SIMD && svcntb() == 16) {
                    require(observed[at] == ref);
                    ++wide_words;
                }
            }
        }
    }
}
#define attentionMatmulTiled check_matmul_intermediate
#include "attentionStream.h"
#undef attentionMatmulTiled
static uint32_t random_state = 0x476211bd;
static uint32_t random_value() {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static const uint8_t lut_ref[32] = {4,  5,  7,   8,   11,  14, 18, 23, 30, 38, 49,
                                    63, 80, 103, 132, 170, 0,  0,  0,  0,  0,  0,
                                    0,  0,  1,   1,   1,   1,  1,  2,  2,  3};
static void reference_softmax(int8_t *p, size_t rows, size_t keys) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t l = 0; l < 2; ++l) {
            uint64_t sum = 0;
            for (size_t k = 0; k < keys; ++k) {
                sum += lut_ref[uint8_t(p[(r * keys + k) * 2 + l]) / 8];
            }
            uint64_t d = std::max(uint64_t(1), sum / 256);
            for (size_t k = 0; k < keys; ++k) {
                p[(r * keys + k) * 2 + l] =
                    int8_t(lut_ref[uint8_t(p[(r * keys + k) * 2 + l]) / 8] / d);
            }
        }
    }
}
static std::vector<int8_t> reference_matmul(const int8_t *a, const int8_t *b, size_t m,
                                            size_t n, size_t k) {
    std::vector<int8_t> out(2 * m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < 2; ++l) {
                int64_t v = 0;
                for (size_t z = 0; z < k; ++z) {
                    v += int64_t(a[(i * k + z) * 2 + l]) * b[(j * k + z) * 2 + l];
                }
                out[(i * n + j) * 2 + l] = int8_t(uint8_t(uint64_t(v)));
            }
        }
    }
    return out;
}
struct GuardedBuffer {
    std::vector<int8_t> v;
    size_t n;
    uint8_t poison;
    GuardedBuffer(size_t count, uint8_t p)
        : v(count + 74, int8_t(p)), n(count), poison(p) {}
    int8_t *data() {
        return v.data() + 37;
    }
    void guards() {
        for (size_t z = 0; z < 37; ++z) {
            require(uint8_t(v[z]) == poison && uint8_t(v[37 + n + z]) == poison);
        }
    }
};
static void check_stream(size_t rows, size_t keys, size_t depth, uint8_t poison,
                         bool capture) {
    GuardedBuffer q(rows * depth * 2, poison), k(keys * depth * 2, poison),
        v(keys * depth * 2, poison), out(rows * depth * 2, poison),
        scratch(std::min(rows, size_t(16)) * keys * 2 + 19, poison),
        qc(rows * keys * 2, poison), sc(rows * keys * 2, poison),
        pc(rows * depth * 2, poison);
    for (auto *b : {&q, &k, &v}) {
        for (size_t z = 0; z < b->n; ++z) {
            b->data()[z] = int8_t(random_value());
        }
    }
    auto qsave = q.v, ksave = k.v, vsave = v.v;
    auto scores = reference_matmul(q.data(), k.data(), rows, keys, depth);
    auto probs = scores;
    reference_softmax(probs.data(), rows, keys);
    auto pre = reference_matmul(probs.data(), v.data(), rows, depth, keys);
    auto expected = pre;
    for (auto &x : expected) {
        x = int8_t(int(x) >> 6);
    }
    Softmax soft;
    allocations = frees = aligned_attempts = 0;
    track_allocations = true;
    attentionStreamRowsWithScratch(
        soft, q.data(), k.data(), v.data(), rows, keys, depth, out.data(),
        scratch.data(), scratch.n, capture ? qc.data() : nullptr,
        capture ? sc.data() : nullptr, capture ? pc.data() : nullptr);
    track_allocations = false;
    require(allocations == (CFG_SIMD && svcntb() == 16 ? 2 * ((rows + 15) / 16) : 0) &&
            frees == allocations);
    require(std::equal(expected.begin(), expected.end(), out.data()));
    if (capture) {
        require(std::equal(scores.begin(), scores.end(), qc.data()));
        require(std::equal(probs.begin(), probs.end(), sc.data()));
        require(std::equal(pre.begin(), pre.end(), pc.data()));
    }
    require(q.v == qsave && k.v == ksave && v.v == vsave);
    for (size_t z = scratch.n - 19; z < scratch.n; ++z) {
        require(uint8_t(scratch.data()[z]) == poison);
    }
    for (auto *b : {&q, &k, &v, &out, &scratch, &qc, &sc, &pc}) {
        b->guards();
    }
    ++stream_cases;
}
static void check_softmax_patterns() {
    Softmax soft;
    for (size_t keys : {size_t(0), size_t(1), size_t(3), size_t(15), size_t(16),
                        size_t(17), size_t(255), size_t(256), size_t(257), size_t(512),
                        size_t(513), size_t(65535)}) {
        for (size_t pattern = 0; pattern < 4; ++pattern) {
            GuardedBuffer b(keys * 6, 0xa5);
            for (size_t z = 0; z < b.n; ++z) {
                b.data()[z] = int8_t(pattern == 0   ? z
                                     : pattern == 1 ? 0x80
                                     : pattern == 2 ? 0x78
                                                    : random_value());
            }
            auto ref = b.v;
            reference_softmax(ref.data() + 37, 3, keys);
            soft.computeInterleaved2LearnersRows(b.data(), 3, keys);
            require(b.v == ref);
            soft_bytes += b.n;
            ++soft_cases;
        }
    }
    soft.computeInterleaved2LearnersRows(nullptr, 0, 1);
    soft.computeInterleaved2LearnersRows(nullptr, 1, 0);
}
template <class F> static void expect_rejection(F f) {
    bool caught = false;
    try {
        f();
    } catch (const std::exception &) {
        caught = true;
    }
    require(caught);
    ++rejects;
}
static void check_contract_rejections() {
    Softmax soft;
    std::vector<int8_t> mem(16384, 0x37);
    auto copy = mem;
    int8_t *q = mem.data(), *k = q + 2048, *v = q + 4096, *o = q + 6144, *s = q + 8192;
    auto run = [&](const int8_t *qq, const int8_t *kk, const int8_t *vv, size_t r,
                   size_t n, size_t d, int8_t *oo, int8_t *ss, size_t len,
                   int8_t *aa = nullptr, int8_t *bb = nullptr, int8_t *cc = nullptr) {
        attentionStreamRowsWithScratch(soft, qq, kk, vv, r, n, d, oo, ss, len, aa, bb,
                                       cc);
    };
    for (size_t r : {size_t(65536), SIZE_MAX}) {
        expect_rejection([&] { run(q, k, v, r, 3, 5, o, s, 1024); });
    }
    for (size_t n : {size_t(65536), SIZE_MAX}) {
        expect_rejection([&] { run(q, k, v, 3, n, 5, o, s, 1024); });
    }
    for (size_t d : {size_t(65536), SIZE_MAX}) {
        expect_rejection([&] { run(q, k, v, 3, 5, d, o, s, 1024); });
    }
    expect_rejection([&] { run(q, k, v, 3, 5, 7, o, nullptr, 1024); });
    expect_rejection([&] { run(q, k, v, 3, 5, 7, o, s, 29); });
    for (auto *x : {q, k, v}) {
        expect_rejection([&] { run(q, k, v, 3, 5, 7, x + 1, s, 1024); });
        expect_rejection([&] { run(q, k, v, 3, 5, 7, o, x + 1, 1024); });
    }
    expect_rejection([&] { run(q, k, v, 3, 5, 7, o, o + 1, 1024); });
    expect_rejection([&] { run(q, k, v, 3, 5, 7, o, s, 1024, s + 1); });
    expect_rejection([&] { run(q, k, v, 3, 5, 7, o, s, 1024, nullptr, s + 1); });
    expect_rejection(
        [&] { run(q, k, v, 3, 5, 7, o, s, 1024, nullptr, nullptr, s + 1); });
    require(mem == copy);
}
struct GuardPage {
    char *base;
    size_t length;
    int8_t *p;
    explicit GuardPage(size_t bytes) {
        size_t page = sysconf(_SC_PAGESIZE);
        length = ((bytes + page - 1) / page + 1) * page;
        base = (char *)mmap(nullptr, length, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(base != MAP_FAILED);
        require(mprotect(base + length - page, page, PROT_NONE) == 0);
        p = (int8_t *)(base + length - page - bytes);
    }
    ~GuardPage() {
        require(munmap(base, length) == 0);
    }
};
static void check_guard_pages() {
    for (size_t keys : {size_t(1), size_t(17), size_t(129), size_t(513)}) {
        size_t rows = 17, depth = 33;
        GuardPage q(rows * depth * 2), k(keys * depth * 2), v(keys * depth * 2),
            o(rows * depth * 2), s(16 * keys * 2);
        for (size_t z = 0; z < rows * depth * 2; ++z) {
            q.p[z] = int8_t(random_value());
        }
        for (size_t z = 0; z < keys * depth * 2; ++z) {
            k.p[z] = int8_t(random_value());
            v.p[z] = int8_t(random_value());
        }
        auto prob = reference_matmul(q.p, k.p, rows, keys, depth);
        reference_softmax(prob.data(), rows, keys);
        auto expected = reference_matmul(prob.data(), v.p, rows, depth, keys);
        Softmax soft;
        attentionStreamRowsWithScratch(soft, q.p, k.p, v.p, rows, keys, depth, o.p, s.p,
                                       16 * keys * 2);
        for (size_t z = 0; z < expected.size(); ++z) {
            require(o.p[z] == int8_t(int(expected[z]) >> 6));
        }
        ++page_cases;
    }
}
static void check_owner_lifetime() {
    using namespace i2ce;
    requireStaticActivationShape(512, 256, 4, 64, 1024, 2, 16);
    for (size_t which = 0; which < 7; ++which) {
        size_t dims[] = {512, 256, 4, 64, 1024, 2, 16};
        ++dims[which];
        expect_rejection([&] {
            requireStaticActivationShape(dims[0], dims[1], dims[2], dims[3], dims[4],
                                         dims[5], dims[6]);
        });
    }
    for (uint8_t p : {uint8_t(0x5a), uint8_t(0xa5)}) {
        overwrite_test_state = {};
        overwrite_test_state.poison = p;
        StaticActivationReceipt receipt;
        {
            StaticActivationFrame a(receipt);
            for (size_t i = 0; i < static_activation::capacity; ++i) {
                require(uint8_t(a.data()[i]) == p);
            }
            std::memset(a.data(), 0x33, static_activation::capacity);
            {
                StaticActivationFrame b(receipt);
                require(a.data() != b.data());
                for (size_t i = 0; i < static_activation::capacity; ++i) {
                    require(uint8_t(b.data()[i]) == p && a.data()[i] == 0x33);
                }
            }
            require(receipt.live_bytes == static_activation::capacity);
        }
        require(receipt.allocations == 2 && receipt.frees == 2 &&
                receipt.live_bytes == 0);
        require(overwrite_test_state.allocations == 2 &&
                overwrite_test_state.frees == 2 &&
                overwrite_test_state.live_bytes == 0);
        ++owner_cases;
    }
    overwrite_test_state = {};
    overwrite_test_state.fail_attempt = 1;
    StaticActivationReceipt failed;
    bool caught = false;
    try {
        StaticActivationFrame a(failed);
    } catch (const std::bad_alloc &) {
        caught = true;
    }
    require(caught && failed.allocations == 0 && failed.live_bytes == 0 &&
            overwrite_test_state.allocations == 0);
    ++owner_cases;
    overwrite_test_state = {};
    StaticActivationReceipt unwind;
    try {
        StaticActivationFrame a(unwind);
        throw std::runtime_error("owner unwind");
    } catch (const std::runtime_error &) {
    }
    require(unwind.allocations == 1 && unwind.frees == 1 && unwind.live_bytes == 0);
    ++owner_cases;
    expect_rejection([&] { overwrite_count(SIZE_MAX, 2); });
    expect_rejection([&] { overwrite_count(2, SIZE_MAX); });
}

static size_t private_checks = 0, head_stream_cases = 0;
static std::vector<i2ce::StaticActivationBytes>
head_and_global_views(i2ce::StaticActivationFrame &f) {
    using namespace i2ce;
    namespace l = static_activation;
    std::vector<StaticActivationBytes> v = {
        {f.data() + l::input_offset, l::input_bytes},
        {f.data() + l::multihead_offset, l::multihead_bytes},
        {f.data() + l::condense_offset, l::condense_bytes},
        {f.data() + l::ff0_offset, l::ff0_bytes},
        {f.data() + l::ff1_offset, l::ff1_bytes}};
    for (size_t h = 0; h < 4; ++h) {
        auto w = f.head(h);
        require(w.headIndex() == h);
        for (auto x : {w.head_out(), w.query(), w.key(), w.value(), w.value_columns(),
                       w.scores()}) {
            v.push_back(x);
        }
    }
    return v;
}
static void check_private_heads() {
    using namespace i2ce;
    const size_t global_offsets[] = {0, 262144, 524288, 786432, 1835008};
    const size_t global_lengths[] = {262144, 262144, 262144, 1048576, 262144};
    const size_t local_offsets[] = {0, 65536, 131072, 196608, 262144, 327680};
    for (uint8_t poison : {uint8_t(0x5a), uint8_t(0xa5)}) {
        overwrite_test_state = {};
        overwrite_test_state.poison = poison;
        StaticActivationReceipt receipt;
        {
            StaticActivationFrame frame(receipt);
            auto all = head_and_global_views(frame);
            require(all.size() == 29);
            require(static_activation::capacity == 3473408);
            size_t sum = 0;
            for (size_t i = 0; i < all.size(); ++i) {
                auto v = all[i];
                size_t off = i < 5 ? global_offsets[i]
                                   : 2097152 + ((i - 5) / 6) * 344064 +
                                         local_offsets[(i - 5) % 6];
                size_t len = i < 5              ? global_lengths[i]
                             : (i - 5) % 6 == 5 ? 16384
                                                : 65536;
                require(v.data() == frame.data() + off && v.length == len);
                sum += len;
                for (size_t j = i + 1; j < all.size(); ++j) {
                    auto w = all[j];
                    auto x = (uintptr_t)v.data(), y = (uintptr_t)w.data();
                    require(x + v.length <= y || y + w.length <= x);
                    ++private_checks;
                }
            }
            require(sum == 3473408);
            expect_rejection([&] { frame.head(4); });
            expect_rejection([&] { frame.head(SIZE_MAX); });
            // Keep all other head and global bytes unchanged while each head streams.
            for (size_t h = 0; h < 4; ++h) {
                for (size_t repetition = 0; repetition < 2; ++repetition) {
                    auto w = frame.head(h);
                    size_t rows = 17, keys = 33, depth = 15;
                    for (auto v : {w.query(), w.key(), w.value_columns()}) {
                        for (size_t z = 0; z < v.length; ++z) {
                            v.data()[z] = int8_t(random_value());
                        }
                    }
                    auto prob = reference_matmul(w.query().data(), w.key().data(), rows,
                                                 keys, depth);
                    reference_softmax(prob.data(), rows, keys);
                    auto expected = reference_matmul(
                        prob.data(), w.value_columns().data(), rows, depth, keys);
                    std::vector<int8_t> before(frame.data(), frame.data() + 3473408);
                    Softmax soft;
                    attentionStreamRowsWithScratch(
                        soft, w.query().data(), w.key().data(),
                        w.value_columns().data(), rows, keys, depth,
                        w.head_out().data(), w.scores().data(), w.scores().length);
                    for (size_t z = 0; z < expected.size(); ++z) {
                        require(w.head_out().data()[z] ==
                                int8_t(int(expected[z]) >> 6));
                    }
                    for (size_t z = 0; z < 3473408; ++z) {
                        auto q = frame.data() + z;
                        bool output = q >= w.head_out().data() &&
                                      q < w.head_out().data() + rows * depth * 2;
                        bool score = q >= w.scores().data() &&
                                     q < w.scores().data() + 16 * keys * 2;
                        if (!output && !score) {
                            require(*q == before[z]);
                        }
                    }
                    ++head_stream_cases;
                }
            }
            {
                StaticActivationReceipt nested;
                StaticActivationFrame inner(nested);
                auto iv = head_and_global_views(inner);
                for (auto x : all) {
                    for (auto y : iv) {
                        uintptr_t a = (uintptr_t)x.data(), b = (uintptr_t)y.data();
                        require(a + x.length <= b || b + y.length <= a);
                        ++private_checks;
                    }
                }
            }
        }
        require(receipt.allocations == 1 && receipt.frees == 1 &&
                receipt.live_bytes == 0);
        require(overwrite_test_state.live_bytes == 0);
    }
}

static size_t scratch_faults = 0;
static void check_scratch_failures() {
    if (!CFG_SIMD || svcntb() != 16) {
        return;
    }
    const size_t rows = 33, keys = 17, depth = 15;
    GuardedBuffer q(rows * depth * 2, 0x5a), k(keys * depth * 2, 0x5a),
        v(keys * depth * 2, 0x5a), o(rows * depth * 2, 0x5a), sc(16 * keys * 2, 0x5a);
    for (auto *b : {&q, &k, &v}) {
        for (size_t z = 0; z < b->n; ++z) {
            b->data()[z] = int8_t(random_value());
        }
    }
    for (size_t failure : {size_t(1), size_t(2), size_t(3), size_t(6)}) {
        std::fill(o.v.begin(), o.v.end(), int8_t(0x5a));
        auto before = o.v;
        Softmax soft;
        allocations = frees = aligned_attempts = 0;
        fail_aligned_at = failure;
        track_allocations = true;
        bool caught = false;
        try {
            attentionStreamRowsWithScratch(soft, q.data(), k.data(), v.data(), rows,
                                           keys, depth, o.data(), sc.data(), sc.n);
        } catch (const std::bad_alloc &) {
            caught = true;
        }
        track_allocations = false;
        fail_aligned_at = 0;
        require(caught && allocations == frees && aligned_attempts == failure);
        if (failure <= 2) {
            require(o.v == before);
        } else {
            o.guards();
        }
        ++scratch_faults;
    }
}

int main() {
    require(CFG_SIMD && svcntb() == 16u);
    check_contract_rejections();
    check_private_heads();
    check_guard_pages();
    check_scratch_failures();
    check_owner_lifetime();
    check_softmax_patterns();
    for (uint8_t poison : {uint8_t(0x5a), uint8_t(0xa5)}) {
        for (size_t rows : {1u, 15u, 16u, 17u, 33u}) {
            for (size_t keys : {1u, 15u, 17u, 127u, 129u, 513u}) {
                for (size_t depth : {1u, 15u, 17u, 64u, 129u}) {
                    check_stream(rows, keys, depth, poison, (rows + keys + depth) % 2u);
                }
            }
        }
    }
    require(stream_cases == 300u && soft_cases == 48u);
    require(page_cases == 4u && owner_cases == 4u);
    require(head_stream_cases == 16u && scratch_faults == 4u);
    require(private_checks == 2494u && rejects == 31u);
    require(wide_words == low_bytes && wide_words > 0);
#ifdef I2CE_EXPECT_VECTOR_SOFTMAX
    require(i2ce_vector_phase_calls > 0);
#else
    require(i2ce_vector_phase_calls == 0);
#endif
    std::puts(
        "PASS: 300 streams, 48 softmax patterns, 4 guard pages, 4 owner cases and 16 "
        "isolated head streams");
}
