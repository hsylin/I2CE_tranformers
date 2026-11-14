#pragma once
#include "codebook_tile_support.h"
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

namespace i2ce {
// K1 is the accumulation step (historical KACC), not the physical panel depth.
// Compressed indices retain K128 panels; decoded W uses min(128, K2).
struct CodebookTiles {
    size_t s1 = 16;
    size_t o1 = 32;
    size_t k1 = 256;
    size_t s2 = 128;
    size_t o2 = 128;
    size_t k2 = 512;

    size_t physical_k() const {
        return k2 < 128 ? k2 : 128;
    }
};

inline const char *validate_tiles(const CodebookTiles &t) {
    if (t.s1 < 4 || t.s1 > 32768 || (t.s1 & (t.s1 - 1))) {
        return "S1 must be a power of two in [4,32768]";
    }
    if (t.s2 < t.s1 || t.s2 > 65535 || t.s2 % t.s1) {
        return "S2 must be a multiple of S1, between S1 and 65535";
    }
    if ((t.o1 != 32 && t.o1 != 64) || (t.o2 != 32 && t.o2 != 64 && t.o2 != 128) ||
        t.o2 % t.o1) {
        return "O1 must be 32/64; O2 must be 32/64/128 and divisible by O1";
    }
    if (t.k1 < 16 || t.k1 > 256 || (t.k1 & (t.k1 - 1)) || t.k2 < 16 || t.k2 > 512 ||
        (t.k2 & (t.k2 - 1)) || t.k1 > t.k2) {
        return "K1 must be 16..256 power-two; K2 16..512 power-two; K1<=K2";
    }
    if ((t.k1 < 128 || t.k2 < 128) && t.k1 != t.k2) {
        return "Compact depths below 128 require K1=K2";
    }
    return nullptr;
}

// An absent option preserves the existing backend choice. An explicit value
// requires the tiled Codebook backend and never falls back to another geometry.
using RuntimeTiles = std::optional<CodebookTiles>;

inline void tiles_usage(FILE *out) {
    std::fputs(
        "Usage: transformer.o [--s1 N --o1 N --k1 N --s2 N --o2 N --k2 N]\n"
        "All six options are required together; Codebook only, Attention fixed.\n"
        "No options preserve the existing backend choice; explicit tiles require "
        "SIMD.\n"
        "The default tiled backend applies to two learners with CB4.\n"
        "Default tile geometry: S1=16 O1=32 K1=256 S2=128 O2=128 K2=512.\n"
        "K1 is accumulation depth; decoded W depth=min(128,K2).\n",
        out);
}

// No partial update, signs, trailing text, duplicate options or silent fallback.
inline const char *parse_tiles(int argc, const char *const *argv, CodebookTiles &out) {
    if (argc == 1) {
        out = CodebookTiles{};
        return nullptr;
    }
    CodebookTiles t;
    const char *names[] = {"--s1", "--o1", "--k1", "--s2", "--o2", "--k2"};
    size_t *fields[] = {&t.s1, &t.o1, &t.k1, &t.s2, &t.o2, &t.k2};
    unsigned seen = 0;
    for (int a = 1; a < argc; a += 2) {
        size_t i = 0;
        while (i < 6 && std::strcmp(argv[a], names[i])) {
            ++i;
        }
        if (i == 6) {
            return "Unknown tile option";
        }
        if (seen & (1u << i)) {
            return "Duplicate tile option";
        }
        if (a + 1 == argc || !*argv[a + 1]) {
            return "Missing tile value";
        }
        size_t value = 0;
        for (const char *p = argv[a + 1]; *p; ++p) {
            if (*p < '0' || *p > '9') {
                return "Tile values must be decimal integers";
            }
            value = value * 10 + size_t(*p - '0');
            if (value > 65535) {
                return "Tile value exceeds 65535";
            }
        }
        *fields[i] = value;
        seen |= 1u << i;
    }
    if (seen != 63) {
        return "Supply all six tile options together";
    }
    if (const char *error = validate_tiles(t)) {
        return error;
    }
    out = t;
    return nullptr;
}

inline size_t checked_product(size_t left, size_t right) {
    if (right && left > std::numeric_limits<size_t>::max() / right) {
        throw std::overflow_error("Codebook scratch product overflow");
    }
    return left * right;
}

inline size_t checked_sum(size_t left, size_t right) {
    if (left > std::numeric_limits<size_t>::max() - right) {
        throw std::overflow_error("Codebook scratch sum overflow");
    }
    return left + right;
}

// Byte strides for packed operands. X is row-major within each learner plane;
// W groups four outputs into SDOT lanes, with learners inside each K panel.
struct PackedOperandLayout {
    size_t x_row_stride;
    size_t x_learner_stride;
    size_t w_depth;
    size_t w_learner_stride;
    size_t w_panel_stride;
    size_t w_group_stride;

    PackedOperandLayout(const CodebookTiles &tiles, size_t learners)
        : x_row_stride(tiles.k2), x_learner_stride(tiles.s2 * tiles.k2),
          w_depth(tiles.physical_k()), w_learner_stride(tiles.o1 * w_depth),
          w_panel_stride(learners * w_learner_stride), w_group_stride(4 * w_depth) {}

    size_t x_offset(size_t learner, size_t row, size_t feature) const {
        return learner * x_learner_stride + row * x_row_stride + feature;
    }
    size_t w_group_offset(size_t learner, size_t column) const {
        return learner * w_learner_stride + (column / 4) * w_group_stride;
    }
    size_t w_lane_offset(size_t column, size_t feature) const {
        return (feature / w_depth) * w_panel_stride + ((feature % w_depth) / 4) * 16 +
               (column % 4) * 4 + feature % 4;
    }
};

// C contains S1 x O1 tiles, each with one plane per learner. W contains
// O1 x physical_k panels. X keeps all K panels for the current S2 rows.
// The validated domain makes each region and learner plane 64-byte aligned.
struct CodebookLayout {
    CodebookTiles t;
    size_t c_bytes;
    size_t w_bytes;
    size_t a_panel_bytes;
    size_t a_panels;
    size_t total_bytes;
    size_t learners;
    CodebookLayout(CodebookTiles tiles, size_t features, size_t count = 2)
        : t(tiles), learners(count) {
        if (const char *error = validate_tiles(t)) {
            throw std::invalid_argument(error);
        }
        if (features > UINT16_MAX || (count != 1 && count != 2 && count != 4)) {
            throw std::invalid_argument("Unsupported Codebook layout dimensions");
        }
        c_bytes = checked_product(checked_product(sizeof(uint32_t), learners),
                                  checked_product(t.s2, t.o2));
        w_bytes = checked_product(learners, checked_product(t.o2, t.k2));
        a_panel_bytes = checked_product(learners, checked_product(t.s2, t.k2));
        a_panels = (features + t.k2 - 1) / t.k2;
        total_bytes = checked_sum(checked_sum(c_bytes, w_bytes),
                                  checked_product(a_panels, a_panel_bytes));
    }
};
} // namespace i2ce
