#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#ifdef I2CE_TEST_OVERWRITE
#include <cstdlib>
#include <cstring>
#endif

namespace i2ce {
// Only buffers whose complete consumed extent is written by their producer
// may use this allocator. Vector size and real C++ element lifetimes remain
// intact; allocate/deallocate use the same std::allocator as the parent.
inline std::size_t overwrite_count(std::size_t a, std::size_t b,
                                   std::size_t lanes = 2u) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if ((b && a > max / b) || (lanes && a * b > max / lanes)) {
        throw std::length_error("overwrite vector extent overflow");
    }
    return a * b * lanes;
}

#ifdef I2CE_TEST_OVERWRITE
struct OverwriteTestState {
    std::size_t attempts = 0, allocations = 0, frees = 0, requested_bytes = 0;
    std::size_t live_bytes = 0, default_constructed_bytes = 0;
    std::size_t fail_attempt = 0;
    unsigned char poison = 0x55;
};
inline OverwriteTestState overwrite_test_state;
inline unsigned char overwrite_poison() {
    const char *p = std::getenv("I2CE_OVERWRITE_POISON");
    return p ? static_cast<unsigned char>(std::strtoul(p, nullptr, 0))
             : overwrite_test_state.poison;
}
#endif

template <class T> struct OverwriteAllocator : std::allocator<T> {
    using value_type = T;
    template <class U> struct rebind {
        using other = OverwriteAllocator<U>;
    };
    OverwriteAllocator() noexcept = default;
    template <class U> OverwriteAllocator(const OverwriteAllocator<U> &) noexcept {}
    template <class U> void construct(U *p) {
        static_assert(std::is_trivial<U>::value, "only trivial overwrite elements");
        ::new (static_cast<void *>(p))
            U; // default-init: lifetime starts, no value-init
#ifdef I2CE_TEST_OVERWRITE
        std::memset(p, overwrite_poison(), sizeof(U));
        overwrite_test_state.default_constructed_bytes += sizeof(U);
#endif
    }
    template <class U, class A, class... Args>
    void construct(U *p, A &&a, Args &&...args) {
        ::new (static_cast<void *>(p))
            U(std::forward<A>(a), std::forward<Args>(args)...);
    }
#ifdef I2CE_TEST_OVERWRITE
    T *allocate(std::size_t n) {
        auto &s = overwrite_test_state;
        ++s.attempts;
        if (s.fail_attempt && s.attempts == s.fail_attempt) {
            throw std::bad_alloc();
        }
        T *p = std::allocator<T>::allocate(n);
        ++s.allocations;
        s.requested_bytes += n * sizeof(T);
        s.live_bytes += n * sizeof(T);
        return p;
    }
    void deallocate(T *p, std::size_t n) noexcept {
        ++overwrite_test_state.frees;
        overwrite_test_state.live_bytes -= n * sizeof(T);
        std::allocator<T>::deallocate(p, n);
    }
#endif
};
template <class T, class U>
inline bool operator==(const OverwriteAllocator<T> &, const OverwriteAllocator<U> &) {
    return true;
}
template <class T, class U>
inline bool operator!=(const OverwriteAllocator<T> &, const OverwriteAllocator<U> &) {
    return false;
}
using OverwriteBytes = std::vector<int8_t, OverwriteAllocator<int8_t>>;
} // namespace i2ce
