#pragma once
#include "attentionSmmla.h"
namespace attention_smmla {
#ifdef I2CE_TEST_ATTENTION_SMMLA
static void (*blocked_inner_observer)(const uint32_t *, size_t, size_t, size_t, size_t,
                                      size_t, size_t, size_t, size_t, size_t, size_t,
                                      size_t, size_t) = nullptr;
#endif
// Panels are directly packed into final M1/K1 or N1/K1 paired subpanels.
// Every consumer of current K2 completes before the same owner is overwritten.
static bool blocked(const int8_t *a, const int8_t *b, size_t M, size_t N, size_t K,
                    int8_t *out) {
#if CFG_SIMD
    if (svcntb() != 16 || M > 65535 || N > 65535 || K > 65535) {
        return false;
    }
    const size_t mcap = padded(std::min(size_t(128), M), 4),
                 ncap = padded(std::min(size_t(128), N), 8),
                 kcap = padded(std::min(size_t(512), K), 16);
    const size_t acap = 2 * mcap * kcap, bcap = 2 * ncap * kcap,
                 ccap = 2 * mcap * ncap * sizeof(uint32_t);
    Arena owner(acap + bcap + ccap);
    int8_t *ap = owner.data(), *bp = ap + acap;
    uint32_t *c = reinterpret_cast<uint32_t *>(bp + bcap);
    ++counters.calls;
    for (size_t mb = 0; mb < M; mb += 128) {
        for (size_t nb = 0; nb < N; nb += 128) {
            const size_t ms = std::min(size_t(128), M - mb),
                         ns = std::min(size_t(128), N - nb), mp = padded(ms, 4),
                         np = padded(ns, 8);
            std::fill(c, c + 2 * mp * np, uint32_t(0));
            counters.c_initialized += 2 * mp * np * sizeof(uint32_t);
            for (size_t kb = 0; kb < K; kb += 512) {
                const size_t ks = std::min(size_t(512), K - kb), kp = padded(ks, 16);
                for (size_t mi = 0; mi < ms; mi += 16) {
                    const size_t mip = padded(std::min(size_t(16), ms - mi), 4);
                    for (size_t ki = 0; ki < ks; ki += 128) {
                        const size_t kip = padded(std::min(size_t(128), ks - ki), 16);
                        pack_pairs(a, M, K, mb + mi, kb + ki, mip, kip,
                                   ap + 2 * mi * kp + 2 * mip * ki, false);
                    }
                }
                for (size_t ni = 0; ni < ns; ni += 32) {
                    const size_t nip = padded(std::min(size_t(32), ns - ni), 8);
                    for (size_t ki = 0; ki < ks; ki += 128) {
                        const size_t kip = padded(std::min(size_t(128), ks - ki), 16);
                        pack_pairs(b, N, K, nb + ni, kb + ki, nip, kip,
                                   bp + 2 * ni * kp + 2 * nip * ki, true);
                    }
                }
                for (size_t ni = 0; ni < ns; ni += 32) {
                    const size_t nis = std::min(size_t(32), ns - ni),
                                 nip = padded(nis, 8);
                    for (size_t ki = 0; ki < ks; ki += 128) {
                        const size_t kis = std::min(size_t(128), ks - ki),
                                     kip = padded(kis, 16);
                        const int8_t *bv = bp + 2 * ni * kp + 2 * nip * ki;
                        for (size_t mi = 0; mi < ms; mi += 16) {
                            const size_t mip = padded(std::min(size_t(16), ms - mi), 4);
                            const int8_t *av = ap + 2 * mi * kp + 2 * mip * ki;
                            for (size_t mm = 0; mm < mip; mm += 4) {
                                for (size_t nn = 0; nn < nip; nn += 8) {
                                    for (size_t l = 0; l < 2; ++l) {
                                        const int8_t *ax =
                                                         av + l * mip * kip + mm * kip,
                                                     *bx =
                                                         bv + l * nip * kip + nn * kip;
                                        uint32_t *cx = c + l * mp * np +
                                                       ((mi + mm) / 2) * np * 2 +
                                                       ((ni + nn) / 2) * 4;
                                        core(ax, bx, cx, kip, np * 2);
                                    }
                                }
                            }
                        }
#ifdef I2CE_TEST_ATTENTION_SMMLA
                        if (blocked_inner_observer) {
                            blocked_inner_observer(c, mp, np, mb, nb, kb, ks, ni, nis,
                                                   ki, kis, M, N);
                        }
#endif
                    }
                }
#ifdef I2CE_TEST_ATTENTION_SMMLA
                if (partial_observer) {
                    partial_observer(c, mp, np, mb, nb, kb + ks, M, N);
                }
#endif
            }
            for (size_t m = 0; m < ms; ++m) {
                for (size_t n = 0; n < ns; ++n) {
                    for (size_t l = 0; l < 2; ++l) {
                        size_t i = ((mb + m) * N + nb + n) * 2 + l;
                        uint32_t v = c[l * mp * np + c_index(m, n, np)];
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
} // namespace attention_smmla
