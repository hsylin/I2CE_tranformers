#pragma once
#include "../Full_NN/src/compressed_index_layout.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
namespace i2ce_ci {
constexpr size_t header_bytes = 96;
constexpr uint64_t seed = UINT64_C(14695981039346656037);
inline uint64_t hash_bytes(uint64_t hash, const void *data, size_t bytes) {
    const auto *encoded = static_cast<const uint8_t *>(data);
    for (size_t offset = 0; offset < bytes; ++offset) {
        hash ^= encoded[offset];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}
// File identities use little-endian integers, independent of host byte order.
inline void put(uint8_t *data, uint64_t value, size_t bytes) {
    for (size_t offset = 0; offset < bytes; ++offset) {
        data[offset] = uint8_t(value >> (8 * offset));
    }
}
inline uint64_t get(const uint8_t *data, size_t bytes) {
    uint64_t value = 0;
    for (size_t offset = 0; offset < bytes; ++offset) {
        value |= uint64_t(data[offset]) << (8 * offset);
    }
    return value;
}
inline uint64_t hu32(uint64_t hash, uint32_t value) {
    uint8_t encoded[4];
    put(encoded, value, 4);
    return hash_bytes(hash, encoded, 4);
}
// Checks address wraparound; allocation ownership remains the caller's responsibility.
inline bool range(const void *data, size_t bytes) {
    return !bytes ||
           (data && reinterpret_cast<uintptr_t>(data) <= UINTPTR_MAX - (bytes - 1u));
}
inline uint64_t identity(const uint32_t *indices, size_t words_per_column,
                         size_t columns, size_t features, const int32_t *codebook,
                         const int32_t *bias) {
    if (!ci_payload_bytes(columns, features) ||
        words_per_column < (features + 15u) / 16u || words_per_column > UINT16_MAX ||
        !range(indices, columns * words_per_column * 4u) || !range(codebook, 32u) ||
        !range(bias, bias ? columns * 8u : 0u) ||
        (reinterpret_cast<uintptr_t>(indices) & 3u) ||
        (reinterpret_cast<uintptr_t>(codebook) & 3u) ||
        (bias && (reinterpret_cast<uintptr_t>(bias) & 3u))) {
        throw std::invalid_argument("compressed source descriptor");
    }
    uint64_t hash = hu32(hu32(hu32(seed, uint32_t(columns)), uint32_t(features)),
                         uint32_t(words_per_column));
    // Source padding participates in the identity, even though it is not computed.
    for (size_t i = 0; i < columns * words_per_column; ++i) {
        hash = hu32(hash, indices[i]);
    }
    for (size_t i = 0; i < 8u; ++i) {
        if (codebook[i] < -128 || codebook[i] > 127) {
            throw std::invalid_argument("compressed codebook domain");
        }
        hash = hu32(hash, uint32_t(codebook[i]));
    }
    hash = hu32(hash, bias ? 1u : 0u);
    if (bias) {
        for (size_t i = 0; i < columns * 2u; ++i) {
            hash = hu32(hash, uint32_t(bias[i]));
        }
    }
    return hash;
}
inline std::string filename(uint64_t source_identity) {
    char name[40];
    std::snprintf(name, sizeof(name), "%016llx.ci2",
                  static_cast<unsigned long long>(source_identity));
    return name;
}
inline uint8_t index(const uint8_t *data, size_t columns, size_t features,
                     size_t column, size_t feature) {
    (void)columns;
    const size_t lane = (column % 2u) * 8u + feature % 8u;
    return (data[ci_pair_k8_offset((features + 127u) / 128u, column, feature) +
                 lane / 4u] >>
            (2u * (lane % 4u))) &
           3u;
}
inline bool padding_zero(const uint8_t *data, size_t columns, size_t features) {
    const size_t padded_columns = (columns + 31u) / 32u * 32u,
                 padded_features = (features + 127u) / 128u * 128u;
    if (padded_columns == columns && padded_features == features) {
        return true;
    }
    for (size_t column = 0; column < padded_columns; ++column) {
        for (size_t feature = 0; feature < padded_features; ++feature) {
            if ((column >= columns || feature >= features) &&
                index(data, columns, features, column, feature)) {
                return false;
            }
        }
    }
    return true;
}
inline void load(const std::string &path, uint64_t source_identity,
                 size_t words_per_column, size_t columns, size_t features,
                 uint8_t *destination, size_t size) {
    const size_t payload_bytes = ci_payload_bytes(columns, features);
    if (!payload_bytes || words_per_column < (features + 15u) / 16u ||
        words_per_column > UINT16_MAX || size != payload_bytes ||
        !range(destination, size)) {
        throw std::invalid_argument("compressed destination extent");
    }
    std::unique_ptr<FILE, int (*)(FILE *)> file(std::fopen(path.c_str(), "rb"),
                                                std::fclose);
    if (!file) {
        throw std::runtime_error("compressed index file missing: " + path);
    }
    uint8_t header[header_bytes];
    if (std::fread(header, 1, sizeof(header), file.get()) != sizeof(header)) {
        throw std::runtime_error("compressed short header");
    }
    // Header size, version, N, K, source words, learners, layout, panel N/K, bits.
    const uint32_t expected[] = {96,
                                 1,
                                 uint32_t(columns),
                                 uint32_t(features),
                                 uint32_t(words_per_column),
                                 2,
                                 1,
                                 32,
                                 128,
                                 2};
    if (std::memcmp(header, "I2CECI01", 8)) {
        throw std::runtime_error("compressed magic");
    }
    for (size_t i = 0; i < 10; ++i) {
        if (get(header + 8 + 4 * i, 4) != expected[i]) {
            throw std::runtime_error("compressed version/shape/layout");
        }
    }
    if (get(header + 48, 8) != payload_bytes ||
        get(header + 56, 8) != source_identity ||
        get(header + 72, 8) != UINT64_C(0x314b385249415032)) {
        throw std::runtime_error("compressed extent/source/layout");
    }
    for (size_t i = 80; i < 96; ++i) {
        if (header[i]) {
            throw std::runtime_error("compressed reserved header");
        }
    }
    if (std::fread(destination, 1, payload_bytes, file.get()) != payload_bytes ||
        std::fgetc(file.get()) != EOF || std::ferror(file.get())) {
        throw std::runtime_error("compressed payload length/EOF");
    }
    if (hash_bytes(seed, destination, payload_bytes) != get(header + 64, 8) ||
        !padding_zero(destination, columns, features)) {
        throw std::runtime_error("compressed checksum/padding");
    }
    // On failure, the destination may be modified and must remain unpublished.
}
} // namespace i2ce_ci
