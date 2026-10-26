#pragma once

#include "attentionTile.h"
#include "overwriteVector.h"
#include "softmax.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace attention_stream {

constexpr size_t query_tile_rows = 16;

inline bool overlaps(const void *left, size_t left_bytes, const void *right,
                     size_t right_bytes) {
    const uintptr_t left_address = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_address = reinterpret_cast<uintptr_t>(right);
    return left_bytes && right_bytes &&
           (left_address <= right_address ? right_address - left_address < left_bytes
                                          : left_address - right_address < right_bytes);
}

inline void validate_output(const int8_t *query, const int8_t *key,
                            const int8_t *value_columns, size_t rows, size_t keys,
                            size_t depth, int8_t *output) {
    if (rows > 65535u || keys > 65535u || depth > 65535u) {
        throw std::length_error("stream shape domain");
    }
    const size_t output_bytes = rows * depth * 2u;
    const size_t key_value_bytes = keys * depth * 2u;
    if (overlaps(output, output_bytes, query, output_bytes) ||
        overlaps(output, output_bytes, key, key_value_bytes) ||
        overlaps(output, output_bytes, value_columns, key_value_bytes)) {
        throw std::invalid_argument("stream output aliases input");
    }
}

// Scores hold complete key rows for at most sixteen queries. Each operation
// finishes before its buffer is overwritten: int8 QK -> integer softmax ->
// int8 SV -> signed post-scaling. Moving these quantization boundaries would
// change the algorithm, even if the surrounding matrix products were exact.
inline void compute_rows(Softmax &softmax, const int8_t *query, const int8_t *key,
                         const int8_t *value_columns, size_t rows, size_t keys,
                         size_t depth, int8_t *output, int8_t *scores,
                         int8_t *qk_capture, int8_t *softmax_capture,
                         int8_t *sv_capture) {
    for (size_t row_base = 0; row_base < rows; row_base += query_tile_rows) {
        const size_t tile_rows = std::min(query_tile_rows, rows - row_base);
        const size_t score_offset = row_base * keys * 2u;
        const size_t score_bytes = tile_rows * keys * 2u;
        attentionMatmulTiled(query + row_base * depth * 2u, key, tile_rows, keys, depth,
                             scores);
        if (qk_capture) {
            std::memcpy(qk_capture + score_offset, scores, score_bytes);
        }
        softmax.computeInterleaved2LearnersRows(scores, tile_rows, keys);
        if (softmax_capture) {
            std::memcpy(softmax_capture + score_offset, scores, score_bytes);
        }
        const size_t output_offset = row_base * depth * 2u;
        int8_t *output_rows = output + output_offset;
        attentionMatmulTiled(scores, value_columns, tile_rows, depth, keys,
                             output_rows);
        if (sv_capture) {
            std::memcpy(sv_capture + output_offset, output_rows,
                        tile_rows * depth * 2u);
        }
        softmax.post_softmax_interleaved2D(output_rows, tile_rows, depth);
    }
}

} // namespace attention_stream

// The owned overload allocates only the score rows needed by this call.
// Optional captures are caller-owned, correctly sized diagnostic buffers.
static void attentionStreamRows(Softmax &softmax, const int8_t *query,
                                const int8_t *key, const int8_t *value_columns,
                                size_t rows, size_t keys, size_t depth, int8_t *output,
                                int8_t *qk_capture = nullptr,
                                int8_t *softmax_capture = nullptr,
                                int8_t *sv_capture = nullptr) {
    attention_stream::validate_output(query, key, value_columns, rows, keys, depth,
                                      output);
    i2ce::OverwriteBytes scores(
        i2ce::overwrite_count(std::min(rows, attention_stream::query_tile_rows), keys));
    attention_stream::compute_rows(softmax, query, key, value_columns, rows, keys,
                                   depth, output, scores.data(), qk_capture,
                                   softmax_capture, sv_capture);
}

// The borrowed overload never resizes or releases the caller's score storage.
// Its used prefix must be disjoint from operands, output, and optional captures.
static void attentionStreamRowsWithScratch(
    Softmax &softmax, const int8_t *query, const int8_t *key,
    const int8_t *value_columns, size_t rows, size_t keys, size_t depth, int8_t *output,
    int8_t *scores, size_t score_capacity_bytes, int8_t *qk_capture = nullptr,
    int8_t *softmax_capture = nullptr, int8_t *sv_capture = nullptr) {
    attention_stream::validate_output(query, key, value_columns, rows, keys, depth,
                                      output);
    const size_t required_bytes =
        i2ce::overwrite_count(std::min(rows, attention_stream::query_tile_rows), keys);
    const size_t output_bytes = rows * depth * 2u;
    const size_t key_value_bytes = keys * depth * 2u;
    const size_t capture_bytes = rows * keys * 2u;
    using attention_stream::overlaps;
    if (score_capacity_bytes < required_bytes || (required_bytes && !scores) ||
        overlaps(scores, required_bytes, query, output_bytes) ||
        overlaps(scores, required_bytes, key, key_value_bytes) ||
        overlaps(scores, required_bytes, value_columns, key_value_bytes) ||
        overlaps(scores, required_bytes, output, output_bytes) ||
        (qk_capture && overlaps(scores, required_bytes, qk_capture, capture_bytes)) ||
        (softmax_capture &&
         overlaps(scores, required_bytes, softmax_capture, capture_bytes)) ||
        (sv_capture && overlaps(scores, required_bytes, sv_capture, output_bytes))) {
        throw std::invalid_argument("SM03 invalid external score extent/alias");
    }
    attention_stream::compute_rows(softmax, query, key, value_columns, rows, keys,
                                   depth, output, scores, qk_capture, softmax_capture,
                                   sv_capture);
}
