/* Private implementation, included once by gemm_SVE.c.
 * CB4/8/16 shared-index/SVE128 blocked matrix multiplication. External X/Y remain
 * interleaved; only the arena's X uses learner planes. MR4xNR16 indexed SDOT inside
 * each inner cache tile; bounded runtime final W.
 */
#include <stdlib.h>
#include <string.h>
#include "compressed_sdot_decoder.h"
#include "gemm_sdot_lane.h"
#include "codebook_tiles.h"
#include <stdio.h>
#ifdef __cplusplus
#include <new>
#endif

/* Only the compatibility wrapper uses this thread-local summary. The kernel
 * receives an invocation-local result; its hot loops never update shared state. */
static thread_local i2ce_codebook_execution_stats cache_thread_stats;

static void cache_merge_statistics(i2ce_codebook_execution_stats &total,
                                   const i2ce_codebook_execution_stats &call) {
    auto &dst = total.scratch;
    const auto &src = call.scratch;
    dst.offline_allocations += src.offline_allocations;
    dst.runtime_allocations += src.runtime_allocations;
    dst.frees += src.frees;
    dst.requested_bytes += src.requested_bytes;
    dst.allocation_failures += src.allocation_failures;
    dst.retained_bytes += src.retained_bytes;
    dst.W_pack_calls += src.W_pack_calls;
    dst.W_outer_pack_calls += src.W_outer_pack_calls;
    dst.W_assigned_bytes += src.W_assigned_bytes;
    dst.W_useful_bytes += src.W_useful_bytes;
    dst.A_pack_calls += src.A_pack_calls;
    dst.A_assigned_bytes += src.A_assigned_bytes;
    dst.A_useful_bytes += src.A_useful_bytes;
    if (src.max_request_bytes > dst.max_request_bytes) {
        dst.max_request_bytes = src.max_request_bytes;
    }
    if (src.peak_W_bytes > dst.peak_W_bytes) {
        dst.peak_W_bytes = src.peak_W_bytes;
    }
    total.sdot_calls += call.sdot_calls;
    total.sdot_full_calls += call.sdot_full_calls;
    total.sdot_source += call.sdot_source;
    total.sdot_edges += call.sdot_edges;
}

static __attribute__((destructor)) void report_cb_sdot(void) {
    // Process-exit reporting covers the exiting thread's compatibility calls.
    // A parallel driver must combine the explicit results from its workers.
    const auto &stats = cache_thread_stats;
    fprintf(
        stderr,
        "[codebook-sdot] backend=runtime-tiles MR=4 NR=16 physical_K=min(128,K2) "
        "calls=%llu "
        "full_core_source_calls=%llu full_core_sdot_source_count=%llu "
        "edge_outputs=%llu "
        "prediction_scope=registered_full_panel_shape_not_dynamic_opcode_counter\n",
        (unsigned long long)stats.sdot_calls, (unsigned long long)stats.sdot_full_calls,
        (unsigned long long)stats.sdot_source, (unsigned long long)stats.sdot_edges);
}

void sve_cb4_scratch_reset(void) {
    cache_thread_stats = {};
}
i2ce_cb4_scratch_stats sve_cb4_scratch_get(void) {
    return cache_thread_stats.scratch;
}

#ifdef I2CE_TEST_CB4_CACHE
uint32_t *i2ce_cb4_cache_observed = NULL;
extern int i2ce_cb4_cache_test_allocate(void **, size_t, size_t);
#endif

#if I2CE_CODEBOOK_TILE_LAYOUT_SUPPORTED
// One owner allocation; these pointers are views, never separate buffers.
struct PackedCodebookArena {
    uint32_t *c;
    int8_t *w;
    i2ce::CodebookLayout layout;
};
static size_t cache_min(size_t a, size_t b) {
    return a < b ? a : b;
}

/* Views partition the accumulator region into S1 x O1 tiles.
 * Each tile contains N_LEARNERS planes; no packing or copy pass.
 * Caller supplies row/column tile indices bounded by the active outer tile. */
typedef struct {
    uint32_t *base;
    size_t plane, stride;
} AccumulatorTileView;
static inline __attribute__((always_inline)) AccumulatorTileView
cache_c_view_data(const i2ce::CodebookTiles &tiles, uint32_t *base, size_t row_tile,
                  size_t column_tile) {
    const size_t plane = tiles.s1 * tiles.o1;
    const size_t row_stride = N_LEARNERS * plane;
    const size_t column_stride = N_LEARNERS * tiles.s2 * tiles.o1;
    return (AccumulatorTileView){
        base + column_tile * column_stride + row_tile * row_stride, plane, tiles.o1};
}
/* Address-range validation precedes both dereference and output writes.
 * The caller owns buffers with the documented extents; this cannot discover
 * an incorrectly sized allocation behind an otherwise valid pointer. */
static int cache_range(const void *p, size_t bytes) {
    return !bytes || (p && (uintptr_t)p <= UINTPTR_MAX - (bytes - 1u));
}
static int cache_overlap(const void *a, size_t an, const void *b, size_t bn) {
    if (!an || !bn) {
        return 0;
    }
    const uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}

static inline __attribute__((always_inline)) void
cache_pack_x(int8_t *dst, size_t rows, size_t stride, [[maybe_unused]] size_t plane,
             const int8_t *input, size_t features, size_t row_base, size_t feature_base,
             size_t tile_features) {

    const svbool_t all = svptrue_b8();
    for (size_t row = 0; row < rows; ++row) {
        const int8_t *src =
            input + ((row_base + row) * features + feature_base) * N_LEARNERS;
        for (size_t feature = 0; feature < tile_features; feature += 16u) {
            const svbool_t valid =
                svwhilelt_b8((uint64_t)feature, (uint64_t)tile_features);
            int8_t *row_dst = dst + row * stride + feature;
#if N_LEARNERS == 1
            svst1_s8(all, row_dst, svld1_s8(valid, src + feature));
#elif N_LEARNERS == 2
            const svint8x2_t x = svld2_s8(valid, src + 2 * feature);
            svst1_s8(all, row_dst, svget2_s8(x, 0));
            svst1_s8(all, row_dst + plane, svget2_s8(x, 1));
#else
            const svint8x4_t x = svld4_s8(valid, src + 4 * feature);
            svst1_s8(all, row_dst, svget4_s8(x, 0));
            svst1_s8(all, row_dst + plane, svget4_s8(x, 1));
            svst1_s8(all, row_dst + 2 * plane, svget4_s8(x, 2));
            svst1_s8(all, row_dst + 3 * plane, svget4_s8(x, 3));
#endif
        }
    }
}

/* Decode directly into each final inner-panel layout. Index decoding and
 * all learner lookups still occur once per outer tile, not per sequence row.
 * Decoded rows use tiles.physical_k() bytes; compressed indices retain their
 * K128 layout. Zero inactive final-vector lanes in decoded panels. */
static inline __attribute__((always_inline)) void
cache_pack_w_blocks(const i2ce::CodebookTiles &tiles, PackedCodebookArena *arena,
                    const i2ce::PackedOperandLayout &operands, const uint8_t *indices,
                    size_t feature_panel_count, size_t column_base, size_t tile_columns,
                    size_t feature_base, size_t tile_features, svint8x4_t tables,
                    svuint8_t byte_selector, svuint8_t index_shifts) {
    const size_t depth = operands.w_depth;
    const size_t column_stride = N_LEARNERS * tiles.o1 * tiles.k2;
    const size_t column_tiles = (tile_columns + tiles.o1 - 1u) / tiles.o1;
    const size_t panels = (tile_features + depth - 1u) / depth;
    for (size_t column_tile = 0; column_tile < column_tiles; ++column_tile) {
        const size_t inner_column = column_tile * tiles.o1;
        for (size_t panel_index = 0; panel_index < panels; ++panel_index) {
            const size_t inner_feature = panel_index * depth;
            const WeightPanel panel{cache_min(tiles.o1, tile_columns - inner_column),
                                    tiles.o1,
                                    depth,
                                    operands.w_learner_stride,
                                    column_base + inner_column,
                                    feature_base + inner_feature,
                                    cache_min(depth, tile_features - inner_feature)};
            int8_t *weights = arena->w + column_tile * column_stride +
                              panel_index * operands.w_panel_stride;
            ci_pack_sdot_w(weights, panel, {indices, feature_panel_count}, tables,
                           byte_selector, index_shifts);
        }
    }
}

/* Each learner uses row-major C within an S1 x O1 tile.
 * Initialize only the active rows and columns. */
static void cache_init_c(uint32_t *c, size_t plane, size_t stride, size_t rows,
                         size_t cols, const int32_t *bias, size_t column_base) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < cols; column += 4u) {
            svbool_t pg = svwhilelt_b32((uint64_t)column, (uint64_t)cols);
            for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
                const svuint32_t v =
                    bias ? svreinterpret_u32_s32(svld1_gather_u32index_s32(
                               pg, bias + (column_base + column) * N_LEARNERS + learner,
                               svindex_u32(0, N_LEARNERS)))
                         : svdup_u32(0);
                svst1_u32(pg, c + learner * plane + row * stride + column, v);
            }
        }
    }
}
/* Separate epilogue: read completed C tiles only after every K panel.
 * This never changes the SDOT core or consumes its live registers. */
static __attribute__((noinline)) void
cache_store_i8_tile(const uint32_t *c, size_t plane, size_t stride, size_t rows,
                    size_t cols, int8_t *output, size_t columns, size_t row_base,
                    size_t column_base) {
#ifdef I2CE_TEST_CB4_CACHE
    if (i2ce_cb4_cache_observed) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < cols; ++column) {
                for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
                    i2ce_cb4_cache_observed[((row_base + row) * columns + column_base +
                                             column) *
                                                N_LEARNERS +
                                            learner] =
                        c[learner * plane + row * stride + column];
                }
            }
        }
    }
#endif
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < cols; column += 4u) {
            svbool_t pg = svwhilelt_b32((uint64_t)column, (uint64_t)cols);
            uint8_t *dst =
                reinterpret_cast<uint8_t *>(output) +
                ((row_base + row) * columns + column_base + column) * N_LEARNERS;
            for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
                const auto values =
                    svld1_u32(pg, c + learner * plane + row * stride + column);
                svst1b_scatter_u32offset_u32(pg, dst + learner,
                                             svindex_u32(0, N_LEARNERS), values);
            }
        }
    }
}

/* Initialize and finalize each S1 x O1 accumulator tile in consumption order. */
static void cache_init_c_tiles_data(const i2ce::CodebookTiles &tiles,
                                    uint32_t *accumulators, size_t rows, size_t cols,
                                    const int32_t *bias, size_t column_base) {
    const size_t column_tiles = (cols + tiles.o1 - 1u) / tiles.o1;
    const size_t row_tiles = (rows + tiles.s1 - 1u) / tiles.s1;
    for (size_t column_tile = 0; column_tile < column_tiles; ++column_tile) {
        const size_t column = column_tile * tiles.o1;
        for (size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
            const size_t row = row_tile * tiles.s1;
            const auto c =
                cache_c_view_data(tiles, accumulators, row_tile, column_tile);
            cache_init_c(c.base, c.plane, c.stride, cache_min(tiles.s1, rows - row),
                         cache_min(tiles.o1, cols - column), bias,
                         column_base + column);
        }
    }
}
static void cache_store_i8_tiles(const i2ce::CodebookTiles &tiles,
                                 uint32_t *accumulators, size_t rows, size_t cols,
                                 int8_t *output, size_t columns, size_t row_base,
                                 size_t column_base) {
    const size_t column_tiles = (cols + tiles.o1 - 1u) / tiles.o1;
    const size_t row_tiles = (rows + tiles.s1 - 1u) / tiles.s1;
    for (size_t column_tile = 0; column_tile < column_tiles; ++column_tile) {
        const size_t column = column_tile * tiles.o1;
        for (size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
            const size_t row = row_tile * tiles.s1;
            const auto c =
                cache_c_view_data(tiles, accumulators, row_tile, column_tile);
            cache_store_i8_tile(c.base, c.plane, c.stride,
                                cache_min(tiles.s1, rows - row),
                                cache_min(tiles.o1, cols - column), output, columns,
                                row_base + row, column_base + column);
        }
    }
}

/* One (s,o), independent learner planes. At most 256 signed-byte products
 * per learner: |partial| <= 256*128*128 = 4,194,304, safely int32.
 * Only the cross-K-tile accumulation uses modulo-2^32 arithmetic. */
/* Logical M/N edges read the final SDOT packed group directly.
 * Full MR4xNR16 tiles use indexed SDOT; complete K16 operand tails are zero. */
static void cache_compute_pair_edge(const i2ce::PackedOperandLayout &layout,
                                    AccumulatorTileView c, size_t row_base,
                                    size_t feature_base, size_t row, size_t column,
                                    size_t tile_features, const int8_t *weights,
                                    const int8_t *input_panel) {
    for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
        uint32_t v = c.base[learner * c.plane + row * c.stride + column];
        const int8_t *x =
            input_panel + layout.x_offset(learner, row_base + row, feature_base);
        const int8_t *w = weights + layout.w_group_offset(learner, column);
        for (size_t feature = 0; feature < tile_features; ++feature) {
            v += (uint32_t)(int32_t(x[feature]) *
                            int32_t(w[layout.w_lane_offset(column, feature)]));
        }
        c.base[learner * c.plane + row * c.stride + column] = v;
    }
}
static void cache_compute_mr4nr16(const i2ce::PackedOperandLayout &layout,
                                  AccumulatorTileView c, size_t row_base,
                                  size_t feature_base, size_t row, size_t column,
                                  size_t tile_features, const int8_t *weights,
                                  const int8_t *input_panel) {
    for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
        if (layout.w_depth < 128u) {
            i2ce_sdot_intrinsic_mr4nr16(
                input_panel + layout.x_offset(learner, row_base + row, feature_base),
                weights + layout.w_group_offset(learner, column),
                c.base + learner * c.plane + row * c.stride + column,
                (tile_features + 15u) / 16u * 16u, layout.x_row_stride,
                layout.w_group_stride, c.stride);
        } else {
            i2ce_sdot_cb_mr4nr16(
                input_panel + layout.x_offset(learner, row_base + row, feature_base),
                weights + layout.w_group_offset(learner, column),
                c.base + learner * c.plane + row * c.stride + column,
                (tile_features + 15u) / 16u * 16u, layout.x_row_stride, c.stride);
        }
    }
}
static void cache_compute_inner(const i2ce::PackedOperandLayout &layout,
                                AccumulatorTileView c, size_t row_base,
                                size_t feature_base, size_t tile_rows,
                                size_t tile_columns, size_t tile_features,
                                const int8_t *weights, const int8_t *input_panel) {
    size_t full_rows = tile_rows / 4u * 4u, full_columns = tile_columns / 16u * 16u;
    for (size_t row = 0; row < full_rows; row += 4u) {
        for (size_t column = 0; column < full_columns; column += 16u) {
            cache_compute_mr4nr16(layout, c, row_base, feature_base, row, column,
                                  tile_features, weights, input_panel);
        }
        for (size_t column = full_columns; column < tile_columns; ++column) {
            for (size_t r = 0; r < 4u; ++r) {
                cache_compute_pair_edge(layout, c, row_base, feature_base, row + r,
                                        column, tile_features, weights, input_panel);
            }
        }
    }
    for (size_t row = full_rows; row < tile_rows; ++row) {
        for (size_t column = 0; column < tile_columns; ++column) {
            cache_compute_pair_edge(layout, c, row_base, feature_base, row, column,
                                    tile_features, weights, input_panel);
        }
    }
}

static void cache_compute_step(const i2ce::PackedOperandLayout &layout,
                               AccumulatorTileView c, size_t row_base,
                               size_t feature_base, size_t tile_rows,
                               size_t tile_columns, size_t tile_features,
                               const int8_t *weights, const int8_t *input_panel) {
    if (tile_features != 128u && tile_features != 256u) {
        // Short and compact reductions use decoded panels of min(128, K2).
        // Compressed indices retain their separate K128 layout.
        for (size_t feature = 0; feature < tile_features; feature += layout.w_depth) {
            cache_compute_inner(
                layout, c, row_base, feature_base + feature, tile_rows, tile_columns,
                cache_min(layout.w_depth, tile_features - feature),
                weights + (feature / layout.w_depth) * layout.w_panel_stride,
                input_panel);
        }
        return;
    }
    const size_t full_rows = tile_rows / 4u * 4u;
    const size_t full_columns = tile_columns / 16u * 16u;
    // Finish one learner plane; reuse each four-row A group across W groups.
    for (size_t learner = 0; learner < N_LEARNERS; ++learner) {
        const int8_t *learner_weights = weights + learner * layout.w_learner_stride;
        uint32_t *learner_result = c.base + learner * c.plane;
        for (size_t row = 0; row < full_rows; row += 4u) {
            const int8_t *row_input =
                input_panel + layout.x_offset(learner, row_base + row, feature_base);
            for (size_t column = 0; column < full_columns; column += 16u) {
                const int8_t *group_weights =
                    learner_weights + (column / 4u) * layout.w_group_stride;
                i2ce_sdot_cb_mr4nr16_accum(row_input, group_weights,
                                           learner_result + row * c.stride + column,
                                           layout.x_row_stride, c.stride,
                                           layout.w_panel_stride, tile_features);
            }
        }
    }
    // Only incomplete M/N ranges use scalar outputs; Kacc is the requested 128/256
    // depth.
    for (size_t row = 0; row < full_rows; ++row) {
        for (size_t column = full_columns; column < tile_columns; ++column) {
            cache_compute_pair_edge(layout, c, row_base, feature_base, row, column,
                                    tile_features, weights, input_panel);
        }
    }
    for (size_t row = full_rows; row < tile_rows; ++row) {
        for (size_t column = 0; column < tile_columns; ++column) {
            cache_compute_pair_edge(layout, c, row_base, feature_base, row, column,
                                    tile_features, weights, input_panel);
        }
    }
}

struct CacheProblem {
    const uint8_t *indices;
    size_t index_bytes;
    size_t words_per_column;
    size_t rows;
    size_t columns;
    size_t features;
    const int8_t *input;
    const int32_t *codebook;
    const int32_t *bias;
    int8_t *output;
    uint8_t bits;
};
// Validate dimensions before forming products or narrowing into descriptors.
static bool cache_valid_dimensions(const CacheProblem &p) {
    return p.rows <= UINT16_MAX && p.columns <= UINT16_MAX &&
           p.features <= UINT16_MAX && p.words_per_column <= UINT16_MAX;
}

struct CacheExtents {
    size_t input;
    size_t output;
    size_t indices;
    size_t codebook;
    size_t bias;

    explicit CacheExtents(const CacheProblem &p)
        : input(p.rows * p.features * N_LEARNERS),
          output(p.rows * p.columns * N_LEARNERS),
          indices(p.features ? ci_payload_bytes_bits(p.columns, p.features, p.bits)
                             : 0),
          codebook(p.features ? CB_SIZE * N_LEARNERS * sizeof(int32_t) : 0),
          bias(p.bias ? p.columns * N_LEARNERS * sizeof(int32_t) : 0) {}
};

// The validated uint16 dimensions keep these products within size_t on AArch64.
static bool cache_valid_storage(const CacheProblem &p, const CacheExtents &bytes) {
    const size_t indices_per_word = 32u / p.bits;
    if (p.index_bytes != bytes.indices || (p.features && !bytes.indices) ||
        bytes.input > UINT32_MAX || bytes.output > UINT32_MAX ||
        p.words_per_column < (p.features + indices_per_word - 1) / indices_per_word ||
        !cache_range(p.output, bytes.output)) {
        return false;
    }
    const struct {
        const void *data;
        size_t bytes;
    } inputs[] = {{p.input, bytes.input},
                  {p.indices, bytes.indices},
                  {p.codebook, bytes.codebook},
                  {p.bias, bytes.bias}};
    for (const auto &input : inputs) {
        if (!cache_range(input.data, input.bytes) ||
            cache_overlap(p.output, bytes.output, input.data, input.bytes)) {
            return false;
        }
    }
    return true;
}

static bool cache_valid_values(const CacheProblem &p) {
    if ((p.features && (reinterpret_cast<uintptr_t>(p.codebook) & 3u)) ||
        (p.bias && (reinterpret_cast<uintptr_t>(p.bias) & 3u))) {
        return false;
    }
    for (size_t i = 0; p.features && i < CB_SIZE * N_LEARNERS; ++i) {
        if (p.codebook[i] < -128 || p.codebook[i] > 127) {
            return false;
        }
    }
    return true;
}

static bool cache_validate_problem(const CacheProblem &p,
                                   const i2ce::CodebookTiles &tiles) {
    if (i2ce::validate_tiles(tiles) || p.bits != CI_KERNEL_BITS || svcntb() != 16u ||
        !cache_valid_dimensions(p)) {
        return false;
    }
    if (!p.rows || !p.columns) {
        return true;
    }
    return cache_valid_storage(p, CacheExtents(p)) && cache_valid_values(p);
}
static void cache_store_bias_only(const CacheProblem &p) {
#ifdef I2CE_TEST_CB4_CACHE
    if (i2ce_cb4_cache_observed) {
        for (size_t row = 0; row < p.rows; ++row) {
            for (size_t column = 0; column < p.columns * N_LEARNERS; ++column) {
                i2ce_cb4_cache_observed[row * p.columns * N_LEARNERS + column] =
                    p.bias ? (uint32_t)p.bias[column] : 0u;
            }
        }
    }
#endif
    for (size_t row = 0; row < p.rows; ++row) {
        for (size_t column = 0; column < p.columns * N_LEARNERS; ++column) {
            p.output[row * p.columns * N_LEARNERS + column] =
                p.bias ? (int8_t)(uint8_t)p.bias[column] : 0;
        }
    }
}
static void cache_compute_outer_step(const i2ce::CodebookTiles &tiles,
                                     const i2ce::PackedOperandLayout &operands,
                                     PackedCodebookArena *arena,
                                     const int8_t *input_panel, size_t tile_rows,
                                     size_t tile_columns, size_t tile_features) {
    // All K updates for an output tile use its own accumulator view. Direct tile
    // indices make each view independent of the previous loop iteration.
    const size_t column_tiles = (tile_columns + tiles.o1 - 1u) / tiles.o1;
    const size_t row_tiles = (tile_rows + tiles.s1 - 1u) / tiles.s1;
    const size_t feature_steps = (tile_features + tiles.k1 - 1u) / tiles.k1;
    const size_t column_stride = N_LEARNERS * tiles.o1 * tiles.k2;
    const size_t step_stride = (tiles.k1 / operands.w_depth) * operands.w_panel_stride;
    for (size_t column_tile = 0; column_tile < column_tiles; ++column_tile) {
        const size_t column = column_tile * tiles.o1;
        const size_t columns = cache_min(tiles.o1, tile_columns - column);
        for (size_t step = 0; step < feature_steps; ++step) {
            const size_t feature = step * tiles.k1;
            const size_t features = cache_min(tiles.k1, tile_features - feature);
            const int8_t *weights =
                arena->w + column_tile * column_stride + step * step_stride;
            for (size_t row_tile = 0; row_tile < row_tiles; ++row_tile) {
                const size_t row = row_tile * tiles.s1;
                const auto c =
                    cache_c_view_data(tiles, arena->c, row_tile, column_tile);
                cache_compute_step(operands, c, row, feature,
                                   cache_min(tiles.s1, tile_rows - row), columns,
                                   features, weights, input_panel);
            }
        }
    }
}
static void cache_execute_tiles(const CacheProblem &p, const i2ce::CodebookTiles &tiles,
                                PackedCodebookArena *arena, int8_t *input_panels,
                                i2ce_codebook_execution_stats &stats) {
    const i2ce::PackedOperandLayout operands(tiles, N_LEARNERS);
    const size_t feature_panels = (p.features + 127u) / 128u;
    int8_t scratch[16] = {};
    const auto t0 = gemm_sdot_cb_table(p.codebook, CB_SIZE, N_LEARNERS, 0, scratch);
#if N_LEARNERS >= 2
    const auto t1 = gemm_sdot_cb_table(p.codebook, CB_SIZE, N_LEARNERS, 1, scratch);
#else
    const auto t1 = t0;
#endif
#if N_LEARNERS == 4
    const auto t2 = gemm_sdot_cb_table(p.codebook, CB_SIZE, 4, 2, scratch);
    const auto t3 = gemm_sdot_cb_table(p.codebook, CB_SIZE, 4, 3, scratch);
#else
    const auto t2 = t0, t3 = t0;
#endif
    const auto tables = svcreate4_s8(t0, t1, t2, t3);
    svuint8_t byte_selector, index_shifts;
    const svuint8_t bit_positions =
        svmul_n_u8_x(svptrue_b8(), svindex_u8(0, 1), CI_KERNEL_BITS);
    byte_selector = svlsr_n_u8_x(svptrue_b8(), bit_positions, 3);
    index_shifts = svand_n_u8_x(svptrue_b8(), bit_positions, 7u);

    ++stats.sdot_calls;
    if (p.rows % tiles.s1 == 0u && p.columns % tiles.o1 == 0u &&
        p.features % tiles.k1 == 0u) {
        stats.sdot_full_calls +=
            (p.rows / 4u) * (p.columns / 16u) * N_LEARNERS * (p.features / tiles.k1);
        stats.sdot_source +=
            (p.rows / 4u) * (p.columns / 16u) * N_LEARNERS * p.features * 4u;
    }
    stats.sdot_edges +=
        p.rows * p.columns - (p.rows / 4u * 4u) * (p.columns / 16u * 16u);
    for (size_t row_base = 0; row_base < p.rows; row_base += tiles.s2) {
        const size_t tile_rows = cache_min(tiles.s2, p.rows - row_base);
        for (size_t panel = 0; panel < arena->layout.a_panels; ++panel) {
            const size_t feature_base = panel * tiles.k2;
            const size_t tile_features = cache_min(tiles.k2, p.features - feature_base);
            int8_t *input_panel = input_panels + panel * arena->layout.a_panel_bytes;
            cache_pack_x(input_panel, tile_rows, operands.x_row_stride,
                         operands.x_learner_stride, p.input, p.features, row_base,
                         feature_base, tile_features);
            ++stats.scratch.A_pack_calls;
            stats.scratch.A_assigned_bytes +=
                N_LEARNERS * tile_rows * ((tile_features + 15u) / 16u * 16u);
            stats.scratch.A_useful_bytes += N_LEARNERS * tile_rows * tile_features;
        }
        for (size_t column_base = 0; column_base < p.columns; column_base += tiles.o2) {
            const size_t tile_columns = cache_min(tiles.o2, p.columns - column_base);
            const size_t column_panels = (tile_columns + tiles.o1 - 1u) / tiles.o1;
            cache_init_c_tiles_data(tiles, arena->c, tile_rows, tile_columns, p.bias,
                                    column_base);
            for (size_t panel = 0; panel < arena->layout.a_panels; ++panel) {
                const size_t feature_base = panel * tiles.k2;
                const size_t tile_features =
                    cache_min(tiles.k2, p.features - feature_base);
                const int8_t *input_panel =
                    input_panels + panel * arena->layout.a_panel_bytes;
                cache_pack_w_blocks(tiles, arena, operands, p.indices, feature_panels,
                                    column_base, tile_columns, feature_base,
                                    tile_features, tables, byte_selector, index_shifts);
                ++stats.scratch.W_outer_pack_calls;
                const size_t weight_panels =
                    column_panels *
                    ((tile_features + operands.w_depth - 1u) / operands.w_depth);
                stats.scratch.W_pack_calls += weight_panels;
                stats.scratch.W_assigned_bytes +=
                    weight_panels * operands.w_panel_stride;
                stats.scratch.W_useful_bytes +=
                    N_LEARNERS * tile_columns * tile_features;
                cache_compute_outer_step(tiles, operands, arena, input_panel, tile_rows,
                                         tile_columns, tile_features);
            }
            cache_store_i8_tiles(tiles, arena->c, tile_rows, tile_columns, p.output,
                                 p.columns, row_base, column_base);
        }
    }
}
static int cache_run_problem(const CacheProblem &p, const i2ce::CodebookTiles &tiles,
                             i2ce_codebook_execution_stats &stats) {
    // One aligned owner contains bounded C/W tiles and all K panels of A.
    // Completed C tiles are narrowed separately; no full output matrix is owned.
    const i2ce::CodebookLayout layout(tiles, p.features, N_LEARNERS);
    const size_t prefix_bytes = layout.c_bytes + layout.w_bytes;
    const size_t storage_bytes = layout.total_bytes;
    void *storage = NULL;
#ifdef I2CE_TEST_CB4_CACHE
    const int alloc_status = i2ce_cb4_cache_test_allocate(&storage, 64u, storage_bytes);
#else
    const int alloc_status = posix_memalign(&storage, 64u, storage_bytes);
#endif
    if (alloc_status != 0) {
        ++stats.scratch.allocation_failures;
        return 0;
    }
    ++stats.scratch.runtime_allocations;
    stats.scratch.retained_bytes += storage_bytes;
    stats.scratch.peak_W_bytes = layout.w_bytes;
    stats.scratch.requested_bytes += storage_bytes;
    if (storage_bytes > stats.scratch.max_request_bytes) {
        stats.scratch.max_request_bytes = storage_bytes;
    }
    // Placement new starts the accumulator lifetime in the aligned owner.
    uint32_t *accumulators =
        ::new (storage) uint32_t[layout.c_bytes / sizeof(uint32_t)];
    PackedCodebookArena views{accumulators, (int8_t *)storage + layout.c_bytes, layout};
    PackedCodebookArena *arena = &views;
    int8_t *input_panels = (int8_t *)storage + prefix_bytes;
    cache_execute_tiles(p, tiles, arena, input_panels, stats);
    free(storage);
    ++stats.scratch.frees;
    stats.scratch.retained_bytes -= storage_bytes;
    return 1;
}
#endif

int sve_gemm_ci_cache_i8_with_stats(const uint8_t *indices, size_t index_bytes,
                                    size_t words_per_column, size_t rows,
                                    size_t columns, size_t features,
                                    const int8_t *input, const int32_t *codebook,
                                    const int32_t *bias, int8_t *output, uint8_t bits,
                                    size_t learners, const i2ce::CodebookTiles &tiles,
                                    i2ce_codebook_execution_stats &stats) {
    stats = {};
#if I2CE_CODEBOOK_TILE_LAYOUT_SUPPORTED
    if (learners != N_LEARNERS) {
        return 0;
    }
    const CacheProblem problem{indices,  index_bytes, words_per_column, rows, columns,
                               features, input,       codebook,         bias, output,
                               bits};
    if (!cache_validate_problem(problem, tiles)) {
        return 0;
    }
    if (!rows || !columns) {
        return 1;
    }
    if (!features) {
        cache_store_bias_only(problem);
        return 1;
    }
    return cache_run_problem(problem, tiles, stats);
#else
    (void)indices;
    (void)index_bytes;
    (void)words_per_column;
    (void)rows;
    (void)columns;
    (void)features;
    (void)input;
    (void)codebook;
    (void)bias;
    (void)output;
    (void)bits;
    (void)learners;
    (void)tiles;
    return 0;
#endif
}

int sve_gemm_ci_cache_i8(const uint8_t *indices, size_t index_bytes,
                         size_t words_per_column, size_t rows, size_t columns,
                         size_t features, const int8_t *input, const int32_t *codebook,
                         const int32_t *bias, int8_t *output, uint8_t bits,
                         size_t learners, const i2ce::CodebookTiles &tiles) {
    i2ce_codebook_execution_stats stats{};
    const int result = sve_gemm_ci_cache_i8_with_stats(
        indices, index_bytes, words_per_column, rows, columns, features, input,
        codebook, bias, output, bits, learners, tiles, stats);
    cache_merge_statistics(cache_thread_stats, stats);
    return result;
}

int sve_gemm_ci_cb4_cache_i8(const uint8_t *indices, size_t bytes, size_t nw, size_t m,
                             size_t n, size_t k, const int8_t *input, const int32_t *cb,
                             const int32_t *bias, int8_t *output, uint8_t bits) {
    return sve_gemm_ci_cache_i8(indices, bytes, nw, m, n, k, input, cb, bias, output,
                                bits, 2, i2ce::CodebookTiles{});
}
