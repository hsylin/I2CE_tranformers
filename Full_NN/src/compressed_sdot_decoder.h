#pragma once
#include "compressed_index_layout.h"
// Build-time cardinality; raw registry words and CI records have different
// packing at three bits. Only CI records reach this decoder.
enum { CI_KERNEL_BITS = CB_SIZE == 8 ? 3 : CB_SIZE == 16 ? 4 : 2 };
static inline __attribute__((always_inline)) svuint8_t
ci_decode_pair(const uint8_t *indices, size_t feature_panels, size_t column,
               size_t feature, svuint8_t byte_selector, svuint8_t shifts) {
    const svbool_t all = svptrue_b8();
    const uint8_t *record = indices + ci_pair_k8_offset_bits(feature_panels, column,
                                                             feature, CI_KERNEL_BITS);
#if CB_SIZE == 4
    const svuint8_t packed = svreinterpret_u8_u32(svdup_u32(ci_load_word(record)));
#else
    // Read exactly six/eight bytes, including at a guarded allocation tail.
    uint64_t word = 0;
    memcpy(&word, record, 2u * CI_KERNEL_BITS);
    const svuint8_t packed = svreinterpret_u8_u64(svdup_u64(word));
#endif
    svuint8_t decoded = svlsr_u8_x(all, svtbl_u8(packed, byte_selector), shifts);
#if CB_SIZE == 8
    const svuint8_t next = svtbl_u8(packed, svadd_n_u8_x(all, byte_selector, 1u));
    decoded = svorr_u8_x(all, decoded,
                         svlsl_u8_x(all, next, svsub_u8_x(all, svdup_u8(8u), shifts)));
#endif
    return svand_n_u8_x(all, decoded, (1u << CI_KERNEL_BITS) - 1u);
}
// Both column pairs are decoded once. Each active learner looks up its own
// table and writes a distinct learner plane. SVE tuple lanes are compile-time
// constants.
template <int Lane, bool Full>
static inline __attribute__((always_inline)) void
ci_store_learner_groups(int8_t *destination, size_t learner_plane_bytes,
                        svint8x4_t tables, svuint8_t first, svuint8_t second,
                        svbool_t valid0, svbool_t valid1) {
    const svint8_t table = svget4_s8(tables, Lane);
    svint8_t a = svtbl_s8(table, first), b = svtbl_s8(table, second);
    if constexpr (!Full) {
        a = svsel_s8(valid0, a, svdup_s8(0));
        b = svsel_s8(valid1, b, svdup_s8(0));
    }
    const svint32_t x = svreinterpret_s32_s8(a), y = svreinterpret_s32_s8(b);
    svst1_s8(svptrue_b8(), destination + Lane * learner_plane_bytes,
             svreinterpret_s8_s32(svuzp1_s32(x, y)));
    svst1_s8(svptrue_b8(), destination + Lane * learner_plane_bytes + 16,
             svreinterpret_s8_s32(svuzp2_s32(x, y)));
    if constexpr (Lane + 1 < N_LEARNERS && Lane < 3) {
        ci_store_learner_groups<Lane + 1, Full>(destination, learner_plane_bytes,
                                                tables, first, second, valid0, valid1);
    }
}

// CI uses shared compressed K128 panels; decoded W may have a smaller depth.
struct CompressedIndexView {
    const uint8_t *data;
    size_t feature_panels;
};
struct WeightPanel {
    size_t valid_columns;
    size_t physical_columns;
    size_t stride;
    size_t learner_plane_bytes;
    size_t first_column;
    size_t first_feature;
    size_t valid_features;
};

template <bool Full>
static inline void ci_pack_sdot_panel(int8_t *destination, const WeightPanel &panel,
                                      CompressedIndexView source, svint8x4_t tables,
                                      svuint8_t byte_selector, svuint8_t shifts) {
    const svbool_t all = svptrue_b8();
    const svuint8_t lane = svindex_u8(0, 1);
    for (size_t column = 0; column < panel.physical_columns; column += 4) {
        for (size_t feature = 0; feature < panel.stride; feature += 8) {
            svuint8_t a = svdup_u8(0), b = svdup_u8(0);
            svbool_t first_valid = all, second_valid = all;
            if (Full ||
                (column < panel.valid_columns && feature < panel.valid_features)) {
                a = ci_decode_pair(
                    source.data, source.feature_panels, panel.first_column + column,
                    panel.first_feature + feature, byte_selector, shifts);
            }
            if (Full ||
                (column + 2 < panel.valid_columns && feature < panel.valid_features)) {
                b = ci_decode_pair(
                    source.data, source.feature_panels, panel.first_column + column + 2,
                    panel.first_feature + feature, byte_selector, shifts);
            }
            if constexpr (!Full) {
                const svbool_t feature_valid = svcmplt_n_u8(
                    all, svand_n_u8_x(all, lane, 7),
                    feature < panel.valid_features ? panel.valid_features - feature
                                                   : 0);
                const svuint8_t column_lanes = svlsr_n_u8_x(all, lane, 3);
                first_valid = svand_b_z(all, feature_valid,
                                        svcmplt_n_u8(all, column_lanes,
                                                     column < panel.valid_columns
                                                         ? panel.valid_columns - column
                                                         : 0));
                second_valid =
                    svand_b_z(all, feature_valid,
                              svcmplt_n_u8(all, column_lanes,
                                           column + 2 < panel.valid_columns
                                               ? panel.valid_columns - column - 2
                                               : 0));
            }
            ci_store_learner_groups<0, Full>(
                destination + column * panel.stride + 4 * feature,
                panel.learner_plane_bytes, tables, a, b, first_valid, second_valid);
        }
    }
}

static inline void ci_pack_sdot_w(int8_t *destination, const WeightPanel &panel,
                                  CompressedIndexView source, svint8x4_t tables,
                                  svuint8_t byte_selector, svuint8_t shifts) {
    if (panel.valid_columns == panel.physical_columns && panel.valid_features == 128 &&
        panel.stride == 128) {
        ci_pack_sdot_panel<true>(destination, panel, source, tables, byte_selector,
                                 shifts);
    } else {
        ci_pack_sdot_panel<false>(destination, panel, source, tables, byte_selector,
                                  shifts);
    }
}
