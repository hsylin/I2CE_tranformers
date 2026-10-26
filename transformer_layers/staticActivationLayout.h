#pragma once
#include <cstddef>
// Fixed disjoint partitions generated at development time.
namespace i2ce {
namespace static_activation {
inline constexpr unsigned version = 3;
inline constexpr const char identity[] =
    "4edc5e7a1745fc1f172395f24ebb01bcf1cb8d58761d8d57a1698ce89cd125d8";
inline constexpr std::size_t capacity = 3473408u;
inline constexpr std::size_t input_offset = 0u, input_bytes = 262144u;
inline constexpr std::size_t multihead_offset = 262144u, multihead_bytes = 262144u;
inline constexpr std::size_t condense_offset = 524288u, condense_bytes = 262144u;
inline constexpr std::size_t ff0_offset = 786432u, ff0_bytes = 1048576u;
inline constexpr std::size_t ff1_offset = 1835008u, ff1_bytes = 262144u;
inline constexpr std::size_t heads_offset = 2097152u, head_stride = 344064u,
                             head_count = 4u;
inline constexpr std::size_t head_out_offset = 0u, head_out_bytes = 65536u;
inline constexpr std::size_t query_offset = 65536u, query_bytes = 65536u;
inline constexpr std::size_t key_offset = 131072u, key_bytes = 65536u;
inline constexpr std::size_t value_offset = 196608u, value_bytes = 65536u;
inline constexpr std::size_t value_columns_offset = 262144u,
                             value_columns_bytes = 65536u;
inline constexpr std::size_t scores_offset = 327680u, scores_bytes = 16384u;
static_assert(heads_offset + head_count * head_stride == capacity,
              "head partition capacity");
} // namespace static_activation
} // namespace i2ce
