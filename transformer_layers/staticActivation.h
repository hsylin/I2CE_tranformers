#pragma once
#include "staticActivationLayout.h"
#include "overwriteVector.h"
#include "run_mode_config.h"
#include <cstdio>
#include <stdexcept>
#if CFG_SIMD
#include <arm_sve.h>
#endif
namespace i2ce {
// Non-owning view of already-live int8 elements; no lease, allocation or planner.
struct StaticActivationBytes {
    int8_t *pointer;
    std::size_t length;
    int8_t *data() const noexcept {
        return pointer;
    }
    int8_t *begin() const noexcept {
        return pointer;
    }
    int8_t *end() const noexcept {
        return pointer + length;
    }
};
inline std::size_t staticActivationVectorBytes() noexcept {
#if CFG_SIMD
    return svcntb();
#else
    return 0u;
#endif
}
inline void requireStaticActivationShape(std::size_t rows, std::size_t input,
                                         std::size_t heads, std::size_t depth,
                                         std::size_t ff, std::size_t learners,
                                         std::size_t vector_bytes) {
    if (rows != 512u || input != 256u || heads != 4u || depth != 64u || ff != 1024u ||
        learners != 2u || vector_bytes != 16u) {
        throw std::invalid_argument("SM03 unsupported static activation shape/VL");
    }
}
struct StaticActivationReceipt {
    std::size_t allocations = 0, frees = 0, requested_bytes = 0, live_bytes = 0;
    void print() const {
        std::printf("[static-activation-layout] version=%u identity=%s "
                    "capacity_bytes=%zu allocations=%zu frees=%zu requested_bytes=%zu "
                    "retained_bytes=%zu cleanup=after_final_dump dynamic_fallback=0\n",
                    static_activation::version, static_activation::identity,
                    static_activation::capacity, allocations, frees, requested_bytes,
                    live_bytes);
    }
};
class StaticActivationFrame;
// A head-specific non-owning view. Only the backing owner constructs it.
class StaticHeadWorkspace {
    int8_t *base_;
    std::size_t index_;
    friend class StaticActivationFrame;
    StaticHeadWorkspace(int8_t *base, std::size_t index) noexcept
        : base_(base), index_(index) {}

  public:
    std::size_t headIndex() const noexcept {
        return index_;
    }
    StaticActivationBytes head_out() const noexcept {
        return {base_ + static_activation::head_out_offset,
                static_activation::head_out_bytes};
    }
    StaticActivationBytes query() const noexcept {
        return {base_ + static_activation::query_offset,
                static_activation::query_bytes};
    }
    StaticActivationBytes key() const noexcept {
        return {base_ + static_activation::key_offset, static_activation::key_bytes};
    }
    StaticActivationBytes value() const noexcept {
        return {base_ + static_activation::value_offset,
                static_activation::value_bytes};
    }
    StaticActivationBytes value_columns() const noexcept {
        return {base_ + static_activation::value_columns_offset,
                static_activation::value_columns_bytes};
    }
    StaticActivationBytes scores() const noexcept {
        return {base_ + static_activation::scores_offset,
                static_activation::scores_bytes};
    }
};
class StaticActivationFrame {
    OverwriteBytes bytes_;
    StaticActivationReceipt &receipt_;

  public:
    explicit StaticActivationFrame(StaticActivationReceipt &receipt)
        : bytes_(static_activation::capacity), receipt_(receipt) {
        ++receipt_.allocations;
        receipt_.requested_bytes += bytes_.size();
        receipt_.live_bytes += bytes_.size();
    }
    StaticActivationFrame(const StaticActivationFrame &) = delete;
    StaticActivationFrame &operator=(const StaticActivationFrame &) = delete;
    ~StaticActivationFrame() {
        // Release first, then record the actual completed release. The caller's
        // scope ends after the same final dump as the parent's outer vectors.
        {
            OverwriteBytes releasing;
            releasing.swap(bytes_);
        }
        ++receipt_.frees;
        receipt_.live_bytes -= static_activation::capacity;
    }
    int8_t *data() noexcept {
        return bytes_.data();
    }
    StaticHeadWorkspace head(std::size_t index) {
        if (index >= static_activation::head_count) {
            throw std::out_of_range("head workspace index");
        }
        return StaticHeadWorkspace(bytes_.data() + static_activation::heads_offset +
                                       index * static_activation::head_stride,
                                   index);
    }
};
} // namespace i2ce
