#pragma once
#include "run_mode_config.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>
#if CFG_SIMD
#include <arm_sve.h>
#include <gemm_SVE.h>
#endif

namespace attention_smmla {
struct Counters {
    uint64_t calls = 0, core_calls = 0, smmla = 0, allocations = 0, frees = 0,
             requested = 0, max_request = 0;
    uint64_t a_packs = 0, b_packs = 0, a_bytes = 0, b_bytes = 0, c_initialized = 0,
             fallbacks = 0, failures = 0;
};
static Counters counters;
static __attribute__((destructor)) void report() {
    std::fprintf(
        stderr,
        "[attention-smmla] version=1 calls=%llu core_calls=%llu "
        "smmla_source_count=%llu allocations=%llu frees=%llu requested_bytes=%llu "
        "max_request_bytes=%llu A_packs=%llu B_packs=%llu A_assigned_bytes=%llu "
        "B_assigned_bytes=%llu C_initialized_bytes=%llu fallbacks=%llu failures=%llu "
        "retained=%llu cleanup=per_call_ROI\n",
        (unsigned long long)counters.calls, (unsigned long long)counters.core_calls,
        (unsigned long long)counters.smmla, (unsigned long long)counters.allocations,
        (unsigned long long)counters.frees, (unsigned long long)counters.requested,
        (unsigned long long)counters.max_request, (unsigned long long)counters.a_packs,
        (unsigned long long)counters.b_packs, (unsigned long long)counters.a_bytes,
        (unsigned long long)counters.b_bytes,
        (unsigned long long)counters.c_initialized,
        (unsigned long long)counters.fallbacks, (unsigned long long)counters.failures,
        (unsigned long long)(counters.allocations - counters.frees));
}
#ifdef I2CE_TEST_ATTENTION_SMMLA
static uint32_t *observed = nullptr;
static void (*partial_observer)(const uint32_t *, size_t, size_t, size_t, size_t,
                                size_t, size_t, size_t) = nullptr;
#endif
static size_t padded(size_t n, size_t quantum) {
    return (n + quantum - 1) / quantum * quantum;
}
static bool multiply(size_t a, size_t b, size_t &out) {
    if (a && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    out = a * b;
    return true;
}
static bool overlap(const void *a, size_t na, const void *b, size_t nb) {
    uintptr_t u = (uintptr_t)a, v = (uintptr_t)b;
    return na && nb && (u <= v ? v - u < na : u - v < nb);
}
static bool span(const void *p, size_t n) {
    return !n || (p && (uintptr_t)p <= std::numeric_limits<uintptr_t>::max() - n);
}
struct Extents {
    size_t a, b, c;
};
static Extents extents(const int8_t *a, const int8_t *b, size_t M, size_t N, size_t K,
                       int8_t *c) {
    Extents v;
    size_t x;
    if (M > UINT32_MAX || N > UINT32_MAX || K > UINT32_MAX || !multiply(M, K, x) ||
        !multiply(x, 2, v.a) || !multiply(N, K, x) || !multiply(x, 2, v.b) ||
        !multiply(M, N, x) || !multiply(x, 2, v.c) || !span(a, v.a) || !span(b, v.b) ||
        !span(c, v.c)) {
        throw std::invalid_argument("Attention operand extent");
    }
    return v;
}
// One independent aligned allocation. Placement array starts uint32 lifetimes;
// packed bytes access its character representation, C accesses uint32 objects.
struct Arena {
    void *allocation;
    uint32_t *words;
    size_t bytes;
    explicit Arena(size_t n) : allocation(nullptr), words(nullptr), bytes(n) {
        if (!n || n % 64) {
            throw std::invalid_argument("Attention scratch alignment");
        }
        try {
            allocation = ::operator new(n, std::align_val_t(64));
        } catch (...) {
            ++counters.failures;
            throw;
        }
        words = ::new (allocation) uint32_t[n / sizeof(uint32_t)];
        ++counters.allocations;
        counters.requested += n;
        counters.max_request = std::max(counters.max_request, (uint64_t)n);
    }
    ~Arena() {
        if (allocation) {
            ::operator delete(allocation, std::align_val_t(64));
            ++counters.frees;
        }
    }
    Arena(const Arena &) = delete;
    Arena &operator=(const Arena &) = delete;
    int8_t *data() {
        return reinterpret_cast<int8_t *>(words);
    }
};
// Final paired panel: learner / pair / K8 / partner / 8 bytes. All bytes
// read by the fixed kernel, including missing rows/columns/K, are assigned.
static void pack_pairs(const int8_t *src, size_t logical_rows, size_t logical_K,
                       size_t row_base, size_t K_base, size_t rows, size_t kp,
                       int8_t *dst, bool right) {
    const size_t plane = rows * kp;
#if CFG_SIMD
    const svbool_t eight = svwhilelt_b8(size_t(0), size_t(8));
#endif
    for (size_t pair = 0; pair < rows / 2; ++pair) {
        for (size_t k = 0; k < kp; k += 8) {
            for (size_t partner = 0; partner < 2; ++partner) {
                size_t row = row_base + pair * 2 + partner;
                int8_t *d0 = dst + pair * kp * 2 + k * 2 + partner * 8,
                       *d1 = d0 + plane;
                const size_t valid = (row < logical_rows && K_base + k < logical_K)
                                         ? std::min(size_t(8), logical_K - K_base - k)
                                         : 0;
                if (!valid) {
                    std::memset(d0, 0, 8);
                    std::memset(d1, 0, 8);
                    continue;
                }
#if CFG_SIMD
                svbool_t pg = svwhilelt_b8(size_t(0), valid);
                svint8x2_t v = svld2_s8(pg, src + (row * logical_K + K_base + k) * 2);
                svst1_s8(eight, d0, svget2_s8(v, 0));
                svst1_s8(eight, d1, svget2_s8(v, 1));
#else
                for (size_t j = 0; j < 8; ++j) {
                    d0[j] = j < valid ? src[(row * logical_K + K_base + k + j) * 2] : 0;
                    d1[j] =
                        j < valid ? src[(row * logical_K + K_base + k + j) * 2 + 1] : 0;
                }
#endif
            }
        }
    }
    if (right) {
        ++counters.b_packs;
        counters.b_bytes += plane * 2;
    } else {
        ++counters.a_packs;
        counters.a_bytes += plane * 2;
    }
}
static size_t c_index(size_t m, size_t n, size_t np) {
    return (m / 2) * np * 2 + (n / 2) * 4 + (m % 2) * 2 + n % 2;
}
static void scalar(const int8_t *a, const int8_t *b, size_t M, size_t N, size_t K,
                   int8_t *out) {
    ++counters.fallbacks;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            for (size_t l = 0; l < 2; ++l) {
                uint32_t v = 0;
                for (size_t k = 0; k < K; ++k) {
                    v += uint32_t(int32_t(a[(m * K + k) * 2 + l]) *
                                  int32_t(b[(n * K + k) * 2 + l]));
                }
                size_t i = (m * N + n) * 2 + l;
                out[i] = (int8_t)(uint8_t)v;
#ifdef I2CE_TEST_ATTENTION_SMMLA
                if (observed) {
                    observed[i] = v;
                }
#endif
            }
        }
    }
}
#if CFG_SIMD
static void core(const int8_t *a, const int8_t *b, uint32_t *c, size_t kp,
                 size_t cp_stride) {
    attention_smmla_mr4nr8(a, b, c, kp, kp * 2, kp * 2, cp_stride);
    ++counters.core_calls;
    counters.smmla += kp;
}
#endif
static bool micro(const int8_t *a, const int8_t *b, size_t M, size_t N, size_t K,
                  int8_t *out) {
#if CFG_SIMD
    if (svcntb() != 16 || M > 65535 || N > 65535 || K > 512) {
        return false;
    }
    const size_t kp = padded(K, 16), as = 4 * kp * 2, bs = 8 * kp * 2,
                 cs = 4 * 8 * 2 * sizeof(uint32_t);
    Arena arena(as + bs + cs);
    int8_t *ap = arena.data(), *bp = ap + as;
    uint32_t *c = reinterpret_cast<uint32_t *>(bp + bs);
    ++counters.calls;
    for (size_t mb = 0; mb < M; mb += 4) {
        pack_pairs(a, M, K, mb, 0, 4, kp, ap, false);
        for (size_t nb = 0; nb < N; nb += 8) {
            pack_pairs(b, N, K, nb, 0, 8, kp, bp, true);
            std::fill(c, c + 64, uint32_t(0));
            counters.c_initialized += cs;
            for (size_t l = 0; l < 2; ++l) {
                core(ap + l * 4 * kp, bp + l * 8 * kp, c + l * 32, kp, 16);
            }
#ifdef I2CE_TEST_ATTENTION_SMMLA
            if (partial_observer) {
                partial_observer(c, 4, 8, mb, nb, K, M, N);
            }
#endif
            for (size_t m = 0; m < std::min(size_t(4), M - mb); ++m) {
                for (size_t n = 0; n < std::min(size_t(8), N - nb); ++n) {
                    for (size_t l = 0; l < 2; ++l) {
                        size_t i = ((mb + m) * N + nb + n) * 2 + l;
                        uint32_t v = c[l * 32 + c_index(m, n, 8)];
                        out[i] = (int8_t)(uint8_t)v;
#ifdef I2CE_TEST_ATTENTION_SMMLA
                        if (observed) {
                            observed[i] = v;
                        }
#endif
                    }
                }
            }
        }
    }
    return true;
#else
    (void)a;
    (void)b;
    (void)M;
    (void)N;
    (void)K;
    (void)out;
    return false;
#endif
}
using Strategy = bool (*)(const int8_t *, const int8_t *, size_t, size_t, size_t,
                          int8_t *);
static void dispatch(const int8_t *a, const int8_t *b, size_t M, size_t N, size_t K,
                     int8_t *out, Strategy strategy) {
    if (!M || !N) {
        return;
    }
    const Extents v = extents(a, b, M, N, K, out);
    if (!K) {
        std::memset(out, 0, v.c);
#ifdef I2CE_TEST_ATTENTION_SMMLA
        if (observed) {
            std::fill(observed, observed + v.c, uint32_t(0));
        }
#endif
        return;
    }
    const bool alias = overlap(a, v.a, out, v.c) || overlap(b, v.b, out, v.c);
    std::vector<int8_t> tmp;
    if (alias) {
        tmp.resize(v.c);
    }
    int8_t *dest = alias ? tmp.data() : out;
    if (!strategy(a, b, M, N, K, dest)) {
        scalar(a, b, M, N, K, dest);
    }
    if (alias) {
        std::memcpy(out, dest, v.c);
    }
}
} // namespace attention_smmla
