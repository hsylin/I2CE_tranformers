/* Private implementation, included once by gemm_SVE.c.
 * CB4/shared-index/SVE128 blocked matrix multiplication. External X/Y remain
 * interleaved; only the arena's X uses learner planes. MR4xNR16 indexed SDOT inside
 * each inner cache tile; bounded runtime final W.
 */
#include <stdlib.h>
#include <string.h>
#include "compressed_sdot_decoder.h"
#include "gemm_sdot_lane.h"
#include <stdio.h>
#ifdef __cplusplus
#include <new>
#endif

/* Allocation-accounting metadata only, reset before the single inference.
 * No backing storage is retained. These scalar updates are charged to ROI. */
static i2ce_cb4_scratch_stats cb4_scratch_stats;
static uint64_t cb_last_direct_calls = 0, cb_last_source_cores = 0;
static uint64_t cb_sdot_calls = 0, cb_sdot_full_calls = 0, cb_sdot_source = 0,
                cb_sdot_edges = 0;
static __attribute__((destructor)) void report_cb_sdot(void) {
    fprintf(stderr,
            "[codebook-last-output] version=1 direct_calls=%llu "
            "final_core_source_calls=%llu "
            "source_scope=registered_full_shape_not_dynamic_traffic\n",
            (unsigned long long)cb_last_direct_calls,
            (unsigned long long)cb_last_source_cores);
    fprintf(stderr,
            "[codebook-sdot] version=3 MR=4 NR=16 Kpanel_max=256 calls=%llu "
            "full_core_source_calls=%llu full_core_sdot_source_count=%llu "
            "edge_outputs=%llu "
            "prediction_scope=registered_full_panel_shape_not_dynamic_opcode_counter\n",
            (unsigned long long)cb_sdot_calls, (unsigned long long)cb_sdot_full_calls,
            (unsigned long long)cb_sdot_source, (unsigned long long)cb_sdot_edges);
}

void sve_cb4_scratch_reset(void) {
    memset(&cb4_scratch_stats, 0, sizeof(cb4_scratch_stats));
    cb_sdot_calls = cb_sdot_full_calls = cb_sdot_source = cb_sdot_edges = 0;
    cb_last_direct_calls = cb_last_source_cores = 0;
}
i2ce_cb4_scratch_stats sve_cb4_scratch_get(void) {
    return cb4_scratch_stats;
}

/* Fixed layout identifiers exposed only to the independent test oracle. */
#define I2CE_WEIGHT_BLOCK_POLICY 2
#define I2CE_MMLA_PAIRED_A 0
/* One A panel belongs to the active sequence/output tile. */
#define I2CE_X_REUSE_OUTPUTS 0

#ifdef I2CE_TEST_CB4_CACHE
uint64_t i2ce_cb4_cache_x_pack_bytes = 0, i2ce_cb4_cache_x_pack_calls = 0;
int i2ce_cb4_cache_x_reuse = I2CE_X_REUSE_OUTPUTS;
int i2ce_cb4_cache_paired_a = I2CE_MMLA_PAIRED_A;
uint64_t i2ce_cb4_cache_mmla_instructions = 0;
uint64_t i2ce_cb4_cache_sdot_instructions = 0;
uint64_t i2ce_cb4_cache_mr_tiles = 0, i2ce_cb4_cache_a_vectors = 0,
         i2ce_cb4_cache_w_vectors = 0;
uint64_t i2ce_cb4_cache_calls = 0, i2ce_cb4_cache_outputs = 0;
uint64_t i2ce_cb4_cache_chunks = 0, i2ce_cb4_cache_decoded = 0;
uint64_t i2ce_cb4_cache_w_copy_bytes = 0, i2ce_cb4_cache_c_transfer_bytes = 0;
uint32_t *i2ce_cb4_cache_observed = NULL;
int i2ce_cb4_cache_weight_policy = I2CE_WEIGHT_BLOCK_POLICY;
int i2ce_cb4_cache_fail_alloc = 0;
extern int i2ce_cb4_cache_test_allocate(void **, size_t, size_t);
#define CACHE_COUNT(name, amount) (i2ce_cb4_cache_##name += (amount))
#else
#define CACHE_COUNT(name, amount) ((void)0)
#endif

#if CB_SIZE == 4 && N_LEARNERS == 2 && TILE_L1_SIZE == 1 && TILE_L2_SIZE == 1
enum {
    CACHE_S1 = 16,
    CACHE_O1 = 32,
    CACHE_K1 = 128,
    CACHE_X1_STRIDE = 144,
    CACHE_S2 = 16,
    CACHE_O2 = 32,
    CACHE_K2 = 256,
    CACHE_X2_STRIDE = 528
};

/* Complete necessary runtime owner. Every field and appended A panel is
 * a multiple of 64 bytes. The arena contains C and final W panels only. */
typedef struct {
    uint32_t c[2u * CACHE_S2 * CACHE_O2];
    int8_t w2_blocks[CACHE_O2 / CACHE_O1][CACHE_K2 / CACHE_K1][2][CACHE_O1]
                    [CACHE_X1_STRIDE];
} cb4_cache_arena;
typedef char cb4_runtime_size_check[(sizeof(cb4_cache_arena) == 22528u) ? 1 : -1];
/* A is repacked for each sequence/output tile and K chunk. */
typedef int8_t cb4_x_panel[2][CACHE_S2][CACHE_X2_STRIDE];

static size_t cache_min(size_t a, size_t b) {
    return a < b ? a : b;
}

/* Views partition the accumulator region into 16x32 tiles.
 * Each tile contains two learner planes; no packing or copy pass.
 * Caller supplies S1/O1-aligned offsets bounded by the active outer tile. */
typedef struct {
    uint32_t *base;
    size_t plane, stride;
} cb4_c_view;
static inline __attribute__((always_inline)) cb4_c_view
cache_c_view_data(uint32_t *base, size_t inner_row, size_t inner_column) {
    return (cb4_c_view){base + ((inner_column / CACHE_O1) * (CACHE_S2 / CACHE_S1) +
                                inner_row / CACHE_S1) *
                                   2u * CACHE_S1 * CACHE_O1,
                        CACHE_S1 * CACHE_O1, CACHE_O1};
}
/* Arena view used by the independent accumulator-layout oracle. */
static inline cb4_c_view cache_c_view(cb4_cache_arena *arena, size_t inner_row,
                                      size_t inner_column) {
    return cache_c_view_data(arena->c, inner_row, inner_column);
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
cache_pack_x(int8_t *dst, size_t rows, size_t stride, size_t plane, const int8_t *input,
             size_t features, size_t row_base, size_t feature_base,
             size_t tile_features) {
    CACHE_COUNT(x_pack_bytes, 2u * rows * tile_features);
    CACHE_COUNT(x_pack_calls, 1);
    const svbool_t all = svptrue_b8();
    for (size_t row = 0; row < rows; ++row) {
        const int8_t *src = input + ((row_base + row) * features + feature_base) * 2u;
        for (size_t feature = 0; feature < tile_features; feature += 16u) {
            const svbool_t valid =
                svwhilelt_b8((uint64_t)feature, (uint64_t)tile_features);
            const svint8x2_t x = svld2_s8(valid, src + 2u * feature);
            /* Predicated LD2 zeros inactive lanes. Store the complete final
             * vector because SDOT is unpredicated; later stride padding is
             * neither read nor cleared. The entry guard requires SVE128. */
            svst1_s8(all, dst + row * stride + feature, svget2_s8(x, 0));
            svst1_s8(all, dst + plane + row * stride + feature, svget2_s8(x, 1));
        }
    }
}

/* Decode directly into each final inner-panel layout. Index decoding and
 * both learner lookups still occur once per outer tile, not per sequence row.
 * Keep 144-byte rows and zero inactive final-vector lanes, for the SDOT consumer. */
static inline __attribute__((always_inline)) void
cache_pack_w_blocks(cb4_cache_arena *arena, const uint8_t *indices,
                    size_t feature_panel_count, size_t column_base, size_t tile_columns,
                    size_t feature_base, size_t tile_features, svint8_t learner0_table,
                    svint8_t learner1_table, svuint8_t byte_selector,
                    svuint8_t index_shifts) {
    for (size_t inner_column = 0; inner_column < tile_columns;
         inner_column += CACHE_O1) {
        for (size_t inner_feature = 0; inner_feature < tile_features;
             inner_feature += CACHE_K1) {
            const size_t cols = cache_min(CACHE_O1, tile_columns - inner_column),
                         width = cache_min(CACHE_K1, tile_features - inner_feature);
            ci_pack_sdot_w(&arena->w2_blocks[inner_column / CACHE_O1]
                                            [inner_feature / CACHE_K1][0][0][0],
                           cols, CACHE_O1, CACHE_X1_STRIDE, CACHE_O1 * CACHE_X1_STRIDE,
                           indices, feature_panel_count, column_base + inner_column,
                           feature_base + inner_feature, width, learner0_table,
                           learner1_table, byte_selector, index_shifts);
            CACHE_COUNT(decoded, cols * width);
        }
    }
}

/* Each learner uses row-major C within the same 16x32 block.
 * Same owner and capacity;
 * inactive row/column slots stay untouched. No C conversion buffer. */
static inline size_t cache_native_c_offset(size_t stride, size_t row, size_t column) {
    return row * stride + column;
}
static void cache_init_c(uint32_t *c, size_t plane, size_t stride, size_t rows,
                         size_t cols, const int32_t *bias, size_t column_base) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < cols; column += 4u) {
            svbool_t pg = svwhilelt_b32((uint64_t)column, (uint64_t)cols);
            svuint32_t b0 = svdup_u32(0), b1 = svdup_u32(0);
            if (bias) {
                svint32x2_t v = svld2_s32(pg, bias + 2u * (column_base + column));
                b0 = svreinterpret_u32_s32(svget2_s32(v, 0));
                b1 = svreinterpret_u32_s32(svget2_s32(v, 1));
            }
            svst1_u32(pg, c + row * stride + column, b0);
            svst1_u32(pg, c + plane + row * stride + column, b1);
        }
    }
}
/* Preserve the complete modulo-32-bit matrix before low-byte conversion. */
static void cache_write_y(const uint32_t *c, size_t plane, size_t stride, size_t rows,
                          size_t cols, uint32_t *output, size_t columns,
                          size_t row_base, size_t column_base) {
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < cols; ++column) {
            for (size_t learner = 0; learner < 2u; ++learner) {
                output[((row_base + row) * columns + column_base + column) * 2u +
                       learner] = c[learner * plane + row * stride + column];
            }
        }
    }
}

/* Initialize and finalize each 16x32 accumulator tile in consumption order. */
static void cache_init_c_tiles_data(uint32_t *accumulators, size_t rows, size_t cols,
                                    const int32_t *bias, size_t column_base) {
    for (size_t inner_column = 0; inner_column < cols; inner_column += CACHE_O1) {
        for (size_t inner_row = 0; inner_row < rows; inner_row += CACHE_S1) {
            const cb4_c_view c =
                cache_c_view_data(accumulators, inner_row, inner_column);
            cache_init_c(c.base, c.plane, c.stride,
                         cache_min(CACHE_S1, rows - inner_row),
                         cache_min(CACHE_O1, cols - inner_column), bias,
                         column_base + inner_column);
        }
    }
}
static void cache_write_y_tiles_data(uint32_t *accumulators, size_t rows, size_t cols,
                                     uint32_t *output, size_t columns, size_t row_base,
                                     size_t column_base) {
    for (size_t inner_column = 0; inner_column < cols; inner_column += CACHE_O1) {
        for (size_t inner_row = 0; inner_row < rows; inner_row += CACHE_S1) {
            const cb4_c_view c =
                cache_c_view_data(accumulators, inner_row, inner_column);
            cache_write_y(c.base, c.plane, c.stride,
                          cache_min(CACHE_S1, rows - inner_row),
                          cache_min(CACHE_O1, cols - inner_column), output, columns,
                          row_base + inner_row, column_base + inner_column);
        }
    }
}

/* Arena wrappers expose initialization and output layout to the oracle. */
static void cache_init_c_tiles(cb4_cache_arena *arena, size_t rows, size_t cols,
                               const int32_t *bias, size_t column_base) {
    cache_init_c_tiles_data(arena->c, rows, cols, bias, column_base);
}
static void cache_write_y_tiles(cb4_cache_arena *arena, size_t rows, size_t cols,
                                uint32_t *output, size_t columns, size_t row_base,
                                size_t column_base) {
    cache_write_y_tiles_data(arena->c, rows, cols, output, columns, row_base,
                             column_base);
}

/* One (s,o), two independent learners. At most 128 signed-byte products
 * per learner: |partial| <= 128*128*128 = 2,097,152, safely int32.
 * Only the cross-K-tile accumulation uses modulo-2^32 arithmetic. */
/* Logical M/N edges read the final SDOT packed group directly.
 * Full MR4xNR16 tiles use indexed SDOT; complete K16 operand tails are zero. */
static void cache_compute_pair_edge(cb4_c_view c, size_t row_base, size_t feature_base,
                                    size_t row, size_t column, size_t tile_features,
                                    const int8_t *weights, const int8_t *input_panel) {
    for (size_t learner = 0; learner < 2u; ++learner) {
        uint32_t v = c.base[learner * c.plane + row * c.stride + column];
        const int8_t *x = input_panel + learner * CACHE_S2 * CACHE_X2_STRIDE +
                          (row_base + row) * CACHE_X2_STRIDE + feature_base;
        const int8_t *w = weights + learner * CACHE_O1 * CACHE_X1_STRIDE +
                          (column / 4u) * 4u * CACHE_X1_STRIDE;
        for (size_t feature = 0; feature < tile_features; ++feature) {
            v += (uint32_t)(int32_t(x[feature]) *
                            int32_t(w[(feature / 4u) * 16u + (column % 4u) * 4u +
                                      feature % 4u]));
        }
        c.base[learner * c.plane + row * c.stride + column] = v;
    }
    CACHE_COUNT(outputs, 1);
}
static void cache_compute_mr4nr16(cb4_c_view c, size_t row_base, size_t feature_base,
                                  size_t row, size_t column, size_t tile_features,
                                  const int8_t *weights, const int8_t *input_panel) {
    for (size_t learner = 0; learner < 2u; ++learner) {
        i2ce_sdot_cb_mr4nr16(input_panel + learner * CACHE_S2 * CACHE_X2_STRIDE +
                                 (row_base + row) * CACHE_X2_STRIDE + feature_base,
                             weights + learner * CACHE_O1 * CACHE_X1_STRIDE +
                                 column * CACHE_X1_STRIDE,
                             c.base + learner * c.plane + row * c.stride + column,
                             (tile_features + 15u) / 16u * 16u);
    }
    CACHE_COUNT(sdot_instructions, ((tile_features + 15u) / 16u) * 128u);
    CACHE_COUNT(a_vectors, ((tile_features + 15u) / 16u) * 8u);
    CACHE_COUNT(w_vectors, ((tile_features + 15u) / 16u) * 32u);
    CACHE_COUNT(outputs, 64);
    CACHE_COUNT(mr_tiles, 2);
}
static void cache_compute_inner(cb4_c_view c, size_t row_base, size_t feature_base,
                                size_t tile_rows, size_t tile_columns,
                                size_t tile_features, const int8_t *weights,
                                const int8_t *input_panel) {
    size_t full_rows = tile_rows / 4u * 4u, full_columns = tile_columns / 16u * 16u;
    for (size_t row = 0; row < full_rows; row += 4u) {
        for (size_t column = 0; column < full_columns; column += 16u) {
            cache_compute_mr4nr16(c, row_base, feature_base, row, column, tile_features,
                                  weights, input_panel);
        }
        for (size_t column = full_columns; column < tile_columns; ++column) {
            for (size_t r = 0; r < 4u; ++r) {
                cache_compute_pair_edge(c, row_base, feature_base, row + r, column,
                                        tile_features, weights, input_panel);
            }
        }
    }
    for (size_t row = full_rows; row < tile_rows; ++row) {
        for (size_t column = 0; column < tile_columns; ++column) {
            cache_compute_pair_edge(c, row_base, feature_base, row, column,
                                    tile_features, weights, input_panel);
        }
    }
}

static void cache_compute_step(cb4_c_view c, size_t row_base, size_t feature_base,
                               size_t tile_rows, size_t tile_columns,
                               size_t tile_features, const int8_t *weights,
                               const int8_t *input_panel) {
    if (tile_features == 256u && tile_rows == 16u && tile_columns == 32u) {
        for (size_t column = 0; column < 32u; column += 16u) {
            for (size_t row = 0; row < 16u; row += 4u) {
                for (size_t learner = 0; learner < 2u; ++learner) {
                    i2ce_sdot_cb_mr4nr16_k256(
                        input_panel + learner * CACHE_S2 * CACHE_X2_STRIDE +
                            (row_base + row) * CACHE_X2_STRIDE + feature_base,
                        weights + learner * CACHE_O1 * CACHE_X1_STRIDE +
                            column * CACHE_X1_STRIDE,
                        c.base + learner * c.plane + row * c.stride + column);
                    CACHE_COUNT(sdot_instructions, 1024u);
                    CACHE_COUNT(a_vectors, 64u);
                    CACHE_COUNT(w_vectors, 256u);
                    CACHE_COUNT(outputs, 32u);
                    CACHE_COUNT(mr_tiles, 1u);
                }
            }
        }
    } else {
        for (size_t feature = 0; feature < tile_features; feature += 128u) {
            cache_compute_inner(c, row_base, feature_base + feature, tile_rows,
                                tile_columns, cache_min(128u, tile_features - feature),
                                weights + (feature / 128u) * 9216u, input_panel);
        }
    }
}

#endif

static int cache_gemm_runtime_i8(const uint8_t *indices, size_t index_bytes,
                                 size_t words_per_column, size_t rows, size_t columns,
                                 size_t features, const int8_t *input,
                                 const int32_t *codebook, const int32_t *bias,
                                 int8_t *output, uint8_t bits) {
#if CB_SIZE == 4 && N_LEARNERS == 2 && TILE_L1_SIZE == 1 && TILE_L2_SIZE == 1
    if (bits != 2u || svcntb() != 16u) {
        return 0;
    }
    /* Retain the supported descriptor domain without narrowing first. These
     * bounds also make all products below fit size_t on AArch64. */
    if (rows > UINT16_MAX || columns > UINT16_MAX || features > UINT16_MAX ||
        words_per_column > UINT16_MAX) {
        return 0;
    }
    if (!rows || !columns) {
        return 1;
    }
    const size_t input_bytes = rows * features * 2u, output_bytes = rows * columns * 2u,
                 index_payload_bytes =
                     features ? ci_payload_bytes(columns, features) : 0u,
                 bias_bytes = columns * 8u;
    if (index_bytes != index_payload_bytes || (features && !index_payload_bytes) ||
        input_bytes > UINT32_MAX || output_bytes > UINT32_MAX ||
        words_per_column < (features + 15u) / 16u ||
        !cache_range(output, output_bytes) || !cache_range(input, input_bytes) ||
        !cache_range(indices, features ? index_payload_bytes : 0u) ||
        !cache_range(codebook, features ? 32u : 0u) ||
        !cache_range(bias, bias ? bias_bytes : 0u)) {
        return 0;
    }
    if (cache_overlap(output, output_bytes, input, input_bytes) ||
        cache_overlap(output, output_bytes, indices,
                      features ? index_payload_bytes : 0u) ||
        cache_overlap(output, output_bytes, codebook, features ? 32u : 0u) ||
        cache_overlap(output, output_bytes, bias, bias ? bias_bytes : 0u)) {
        return 0;
    }
    if (features && ((uintptr_t)codebook & 3u)) {
        return 0;
    }
    if (bias && ((uintptr_t)bias & 3u)) {
        return 0;
    }
    if (features && !gemm_sdot_decode_usable(codebook, 4u, 2u, 2u)) {
        return 0;
    }
    if (!features) {
#ifdef I2CE_TEST_CB4_CACHE
        if (i2ce_cb4_cache_observed) {
            for (size_t row = 0; row < rows; ++row) {
                for (size_t column = 0; column < columns * 2u; ++column) {
                    i2ce_cb4_cache_observed[row * columns * 2u + column] =
                        bias ? (uint32_t)bias[column] : 0u;
                }
            }
        }
#endif
        for (size_t row = 0; row < rows; ++row) {
            for (size_t column = 0; column < columns * 2u; ++column) {
                output[row * columns * 2u + column] =
                    bias ? (int8_t)(uint8_t)bias[column] : 0;
            }
        }
        return 1;
    }
    // One aligned owner contains C, two physical W panels, one A panel,
    // and the full C32 output temporary. The owner lasts through the complete output
    // conversion.
    const size_t x_panels = 1u;
    const size_t prefix_bytes = sizeof(cb4_cache_arena);
    const size_t temporary_offset = prefix_bytes + sizeof(cb4_x_panel);
    const size_t storage_bytes = temporary_offset + output_bytes * sizeof(uint32_t);
    void *storage = NULL;
#ifdef I2CE_TEST_CB4_CACHE
    const int alloc_status = i2ce_cb4_cache_test_allocate(&storage, 64u, storage_bytes);
#else
    const int alloc_status = posix_memalign(&storage, 64u, storage_bytes);
#endif
    if (alloc_status != 0) {
        ++cb4_scratch_stats.allocation_failures;
        return 0;
    }
    ++cb4_scratch_stats.runtime_allocations;
    cb4_scratch_stats.retained_bytes += storage_bytes;
    cb4_scratch_stats.peak_W_bytes = sizeof(((cb4_cache_arena *)0)->w2_blocks);
    cb4_scratch_stats.requested_bytes += storage_bytes;
    if (storage_bytes > cb4_scratch_stats.max_request_bytes) {
        cb4_scratch_stats.max_request_bytes = storage_bytes;
    }
#ifdef __cplusplus
    cb4_cache_arena *arena = ::new (storage) cb4_cache_arena;
    cb4_x_panel *input_panels =
        ::new ((unsigned char *)storage + prefix_bytes) cb4_x_panel[x_panels];
#else
    cb4_cache_arena *arena = (cb4_cache_arena *)storage;
    cb4_x_panel *input_panels =
        (cb4_x_panel *)((unsigned char *)storage + prefix_bytes);
#endif
#ifdef __cplusplus
    uint32_t *temporary =
        ::new ((unsigned char *)storage + temporary_offset) uint32_t[output_bytes];
#else
    uint32_t *temporary = (uint32_t *)((unsigned char *)storage + temporary_offset);
#endif
    uint32_t *accumulators = arena->c;
    int8_t scratch0[256] = {0}, scratch1[256] = {0};
    const svint8_t learner0_table = gemm_sdot_cb_table(codebook, 4u, 2u, 0u, scratch0);
    const svint8_t learner1_table = gemm_sdot_cb_table(codebook, 4u, 2u, 1u, scratch1);
    svuint8_t byte_selector, index_shifts;
    gemm_sdot_lane_patterns(2u, &byte_selector, &index_shifts);
    CACHE_COUNT(calls, 1);
    ++cb_sdot_calls;
    if (rows % 16u == 0u && columns % 32u == 0u && features % 256u == 0u) {
        cb_sdot_full_calls += (rows / 4u) * (columns / 16u) * 2u * (features / 256u);
        cb_sdot_source += (rows / 4u) * (columns / 16u) * 2u * features * 4u;
    }
    cb_sdot_edges += rows * columns - (rows / 4u * 4u) * (columns / 16u * 16u);
    for (size_t row_base = 0; row_base < rows; row_base += CACHE_S2) {
        const size_t tile_rows = cache_min(CACHE_S2, rows - row_base);
        for (size_t column_base = 0; column_base < columns; column_base += CACHE_O2) {
            const size_t tile_columns = cache_min(CACHE_O2, columns - column_base);
            cache_init_c_tiles_data(accumulators, tile_rows, tile_columns, bias,
                                    column_base);
            for (size_t feature_base = 0; feature_base < features;
                 feature_base += CACHE_K2) {
                const size_t tile_features =
                    cache_min(CACHE_K2, features - feature_base);
                int8_t *input_panel = &input_panels[0][0][0][0];
                cache_pack_x(input_panel, tile_rows, CACHE_X2_STRIDE,
                             CACHE_S2 * CACHE_X2_STRIDE, input, features, row_base,
                             feature_base, tile_features);
                ++cb4_scratch_stats.A_pack_calls;
                cb4_scratch_stats.A_assigned_bytes +=
                    2u * tile_rows * ((tile_features + 15u) / 16u * 16u);
                cb4_scratch_stats.A_useful_bytes += 2u * tile_rows * tile_features;
                cache_pack_w_blocks(arena, indices, (features + 127u) / 128u,
                                    column_base, tile_columns, feature_base,
                                    tile_features, learner0_table, learner1_table,
                                    byte_selector, index_shifts);
                ++cb4_scratch_stats.W_outer_pack_calls;
                cb4_scratch_stats.W_pack_calls +=
                    ((tile_columns + 31u) / 32u) * ((tile_features + 127u) / 128u);
                cb4_scratch_stats.W_assigned_bytes += ((tile_columns + 31u) / 32u) *
                                                      ((tile_features + 127u) / 128u) *
                                                      9216u;
                cb4_scratch_stats.W_useful_bytes += 2u * tile_columns * tile_features;
                /* Reuse each final SDOT W block across sequence tiles. C carries the
                 * partial sums directly between K1 panels; no C1 staging. */
                for (size_t inner_column = 0; inner_column < tile_columns;
                     inner_column += CACHE_O1) {
                    const size_t columns_in_tile =
                        cache_min(CACHE_O1, tile_columns - inner_column);
                    for (size_t inner_feature = 0; inner_feature < tile_features;) {
                        const size_t features_in_step =
                            (columns_in_tile == 32u &&
                             tile_features - inner_feature >= 256u)
                                ? 256u
                                : cache_min(CACHE_K1, tile_features - inner_feature);
                        /* Every inner sequence consumer finishes before this
                         * outer W allocation can be overwritten by the next
                         * feature_base. */
                        const int8_t *weights =
                            &arena->w2_blocks[inner_column / CACHE_O1]
                                             [inner_feature / CACHE_K1][0][0][0];
                        for (size_t inner_row = 0; inner_row < tile_rows;
                             inner_row += CACHE_S1) {
                            const size_t rows_in_tile =
                                cache_min(CACHE_S1, tile_rows - inner_row);
                            cache_compute_step(
                                cache_c_view_data(accumulators, inner_row,
                                                  inner_column),
                                inner_row, inner_feature, rows_in_tile, columns_in_tile,
                                features_in_step, weights, input_panel);
                        }
                        inner_feature += features_in_step;
                    }
                }
            }
            cache_write_y_tiles_data(accumulators, tile_rows, tile_columns, temporary,
                                     columns, row_base, column_base);
        }
    }
#ifdef I2CE_TEST_CB4_CACHE
    if (i2ce_cb4_cache_observed) {
        memcpy(i2ce_cb4_cache_observed, temporary, output_bytes * sizeof(uint32_t));
    }
#endif
    for (size_t index = 0; index < output_bytes; ++index) {
        output[index] = (int8_t)(uint8_t)temporary[index];
    }
    free(storage);
    ++cb4_scratch_stats.frees;
    cb4_scratch_stats.retained_bytes -= storage_bytes;
    return 1;
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
    return 0;
#endif
}
int sve_gemm_ci_cb4_cache_i8(const uint8_t *indices, size_t index_bytes,
                             size_t words_per_column, size_t rows, size_t columns,
                             size_t features, const int8_t *input,
                             const int32_t *codebook, const int32_t *bias,
                             int8_t *output, uint8_t bits) {
    return cache_gemm_runtime_i8(indices, index_bytes, words_per_column, rows, columns,
                                 features, input, codebook, bias, output, bits);
}
#undef CACHE_COUNT
