#pragma once
#include "compressed_index_layout.h"
/* Read two original CI01 pair/K8 words directly into final four-output
 * K4 vectors. Shared packed_indices are decoded once for both learner lookups.
 * The general path assigns padding; the full-panel path writes useful bytes. */
/* Full 32-column, 128-feature panel only. Useful vectors complete; unused stride
 * padding remains untouched and is never consumed by the K128 fast path. */
static inline __attribute__((always_inline)) void
ci_pack_sdot_w_full32k128(int8_t *destination, size_t learner_plane_bytes,
                          const uint8_t *packed_indices, size_t feature_panels,
                          size_t first_column, size_t first_feature,
                          svint8_t learner0_table, svint8_t learner1_table,
                          svuint8_t byte_selector, svuint8_t index_shifts) {
    const svbool_t all = svptrue_b8();
    for (size_t column = 0; column < 32u; column += 4u) {
        for (size_t feature = 0; feature < 128u; feature += 8u) {
            const uint32_t first_pair_word =
                ci_load_word(packed_indices +
                             ci_pair_k8_offset(feature_panels, first_column + column,
                                               first_feature + feature));
            const uint32_t second_pair_word = ci_load_word(
                packed_indices + ci_pair_k8_offset(feature_panels,
                                                   first_column + column + 2u,
                                                   first_feature + feature));
            const svuint8_t first_pair_indices = svand_n_u8_x(
                all,
                svlsr_u8_x(all,
                           svtbl_u8(svreinterpret_u8_u32(svdup_u32(first_pair_word)),
                                    byte_selector),
                           index_shifts),
                3u);
            const svuint8_t second_pair_indices = svand_n_u8_x(
                all,
                svlsr_u8_x(all,
                           svtbl_u8(svreinterpret_u8_u32(svdup_u32(second_pair_word)),
                                    byte_selector),
                           index_shifts),
                3u);
            const svint8_t learner0_first_pair =
                               svtbl_s8(learner0_table, first_pair_indices),
                           learner1_first_pair =
                               svtbl_s8(learner1_table, first_pair_indices),
                           learner0_second_pair =
                               svtbl_s8(learner0_table, second_pair_indices),
                           learner1_second_pair =
                               svtbl_s8(learner1_table, second_pair_indices);
            int8_t *output_group = destination + column * 144u + 4u * feature;
            svst1_s8(all, output_group,
                     svreinterpret_s8_s32(
                         svuzp1_s32(svreinterpret_s32_s8(learner0_first_pair),
                                    svreinterpret_s32_s8(learner0_second_pair))));
            svst1_s8(all, output_group + 16u,
                     svreinterpret_s8_s32(
                         svuzp2_s32(svreinterpret_s32_s8(learner0_first_pair),
                                    svreinterpret_s32_s8(learner0_second_pair))));
            svst1_s8(all, output_group + learner_plane_bytes,
                     svreinterpret_s8_s32(
                         svuzp1_s32(svreinterpret_s32_s8(learner1_first_pair),
                                    svreinterpret_s32_s8(learner1_second_pair))));
            svst1_s8(all, output_group + learner_plane_bytes + 16u,
                     svreinterpret_s8_s32(
                         svuzp2_s32(svreinterpret_s32_s8(learner1_first_pair),
                                    svreinterpret_s32_s8(learner1_second_pair))));
        }
    }
}
/* The general path initializes physical padding and dummy output columns.
 * Only the full-panel path may leave the unread stride padding untouched. */
static inline void
ci_pack_sdot_w(int8_t *destination, size_t columns, size_t physical_columns,
               size_t stride, size_t learner_plane_bytes, const uint8_t *packed_indices,
               size_t feature_panels, size_t first_column, size_t first_feature,
               size_t valid_features, svint8_t learner0_table, svint8_t learner1_table,
               svuint8_t byte_selector, svuint8_t index_shifts) {
    if (columns == 32u && physical_columns == 32u && stride == 144u &&
        learner_plane_bytes == 4608u && valid_features == 128u) {
        ci_pack_sdot_w_full32k128(destination, learner_plane_bytes, packed_indices,
                                  feature_panels, first_column, first_feature,
                                  learner0_table, learner1_table, byte_selector,
                                  index_shifts);
        return;
    }
    const svbool_t all = svptrue_b8();
    const svuint8_t lane = svindex_u8(0, 1);
    const svuint8_t feature_lane = svand_n_u8_x(all, lane, 7u),
                    column_lane = svlsr_n_u8_x(all, lane, 3);
    for (size_t column = 0; column < physical_columns; column += 4u) {
        for (size_t feature = 0; feature < stride; feature += 8u) {
            svint8_t learner0_first_pair = svdup_s8(0),
                     learner1_first_pair = svdup_s8(0),
                     learner0_second_pair = svdup_s8(0),
                     learner1_second_pair = svdup_s8(0);
            if (column < columns && feature < valid_features) {
                uint32_t word = ci_load_word(
                    packed_indices + ci_pair_k8_offset(feature_panels,
                                                       first_column + column,
                                                       first_feature + feature));
                svuint8_t packed = svreinterpret_u8_u32(svdup_u32(word));
                svuint8_t decoded_indices = svand_n_u8_x(
                    all, svlsr_u8_x(all, svtbl_u8(packed, byte_selector), index_shifts),
                    3u);
                svbool_t valid = svand_b_z(
                    all,
                    svcmplt_n_u8(all, feature_lane, uint8_t(valid_features - feature)),
                    svcmplt_n_u8(all, column_lane, uint8_t(columns - column)));
                learner0_first_pair = svsel_s8(
                    valid, svtbl_s8(learner0_table, decoded_indices), svdup_s8(0));
                learner1_first_pair = svsel_s8(
                    valid, svtbl_s8(learner1_table, decoded_indices), svdup_s8(0));
            }
            if (column + 2u < columns && feature < valid_features) {
                uint32_t word = ci_load_word(
                    packed_indices + ci_pair_k8_offset(feature_panels,
                                                       first_column + column + 2u,
                                                       first_feature + feature));
                svuint8_t packed = svreinterpret_u8_u32(svdup_u32(word));
                svuint8_t decoded_indices = svand_n_u8_x(
                    all, svlsr_u8_x(all, svtbl_u8(packed, byte_selector), index_shifts),
                    3u);
                svbool_t valid = svand_b_z(
                    all,
                    svcmplt_n_u8(all, feature_lane, uint8_t(valid_features - feature)),
                    svcmplt_n_u8(all, column_lane, uint8_t(columns - column - 2u)));
                learner0_second_pair = svsel_s8(
                    valid, svtbl_s8(learner0_table, decoded_indices), svdup_s8(0));
                learner1_second_pair = svsel_s8(
                    valid, svtbl_s8(learner1_table, decoded_indices), svdup_s8(0));
            }
            int8_t *output_group = destination + column * stride + 4u * feature;
            svst1_s8(all, output_group,
                     svreinterpret_s8_s32(
                         svuzp1_s32(svreinterpret_s32_s8(learner0_first_pair),
                                    svreinterpret_s32_s8(learner0_second_pair))));
            svst1_s8(all, output_group + 16u,
                     svreinterpret_s8_s32(
                         svuzp2_s32(svreinterpret_s32_s8(learner0_first_pair),
                                    svreinterpret_s32_s8(learner0_second_pair))));
            svst1_s8(all, output_group + learner_plane_bytes,
                     svreinterpret_s8_s32(
                         svuzp1_s32(svreinterpret_s32_s8(learner1_first_pair),
                                    svreinterpret_s32_s8(learner1_second_pair))));
            svst1_s8(all, output_group + learner_plane_bytes + 16u,
                     svreinterpret_s8_s32(
                         svuzp2_s32(svreinterpret_s32_s8(learner1_first_pair),
                                    svreinterpret_s32_s8(learner1_second_pair))));
        }
    }
}
