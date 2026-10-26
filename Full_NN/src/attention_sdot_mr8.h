#pragma once
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>
enum { ATTENTION_GENERIC_ROWS = 8, ATTENTION_GENERIC_COLUMNS = 8 };
/* SVE128 caller contract; full uint32 modulo partial C. C/C++ intrinsics only.
 * Compiler allocates registers; K16 packed tails complete. No horizontal sum. */
static inline __attribute__((always_inline)) void
attention_sdot_intrinsic_8x8(const int8_t *packed_left, const int8_t *packed_right,
                             uint32_t *accumulators, size_t padded_features,
                             size_t left_row_stride, size_t right_group_stride,
                             size_t output_row_stride) {
    const svbool_t bytes = svptrue_b8(), words = svptrue_b32();
    svint32_t sum_row0_group0 = svdup_s32(0);
    svint32_t sum_row0_group1 = svdup_s32(0);
    svint32_t sum_row1_group0 = svdup_s32(0);
    svint32_t sum_row1_group1 = svdup_s32(0);
    svint32_t sum_row2_group0 = svdup_s32(0);
    svint32_t sum_row2_group1 = svdup_s32(0);
    svint32_t sum_row3_group0 = svdup_s32(0);
    svint32_t sum_row3_group1 = svdup_s32(0);
    svint32_t sum_row4_group0 = svdup_s32(0);
    svint32_t sum_row4_group1 = svdup_s32(0);
    svint32_t sum_row5_group0 = svdup_s32(0);
    svint32_t sum_row5_group1 = svdup_s32(0);
    svint32_t sum_row6_group0 = svdup_s32(0);
    svint32_t sum_row6_group1 = svdup_s32(0);
    svint32_t sum_row7_group0 = svdup_s32(0);
    svint32_t sum_row7_group1 = svdup_s32(0);
    for (size_t feature = 0; feature < padded_features; feature += 16u) {
        const svint8_t left_row0 =
            svld1_s8(bytes, packed_left + 0u * left_row_stride + feature);
        const svint8_t left_row1 =
            svld1_s8(bytes, packed_left + 1u * left_row_stride + feature);
        const svint8_t left_row2 =
            svld1_s8(bytes, packed_left + 2u * left_row_stride + feature);
        const svint8_t left_row3 =
            svld1_s8(bytes, packed_left + 3u * left_row_stride + feature);
        const svint8_t left_row4 =
            svld1_s8(bytes, packed_left + 4u * left_row_stride + feature);
        const svint8_t left_row5 =
            svld1_s8(bytes, packed_left + 5u * left_row_stride + feature);
        const svint8_t left_row6 =
            svld1_s8(bytes, packed_left + 6u * left_row_stride + feature);
        const svint8_t left_row7 =
            svld1_s8(bytes, packed_left + 7u * left_row_stride + feature);
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 0u * right_group_stride + 4u * feature + 0u);
            sum_row0_group0 = svdot_lane_s32(sum_row0_group0, weights, left_row0, 0);
            sum_row1_group0 = svdot_lane_s32(sum_row1_group0, weights, left_row1, 0);
            sum_row2_group0 = svdot_lane_s32(sum_row2_group0, weights, left_row2, 0);
            sum_row3_group0 = svdot_lane_s32(sum_row3_group0, weights, left_row3, 0);
            sum_row4_group0 = svdot_lane_s32(sum_row4_group0, weights, left_row4, 0);
            sum_row5_group0 = svdot_lane_s32(sum_row5_group0, weights, left_row5, 0);
            sum_row6_group0 = svdot_lane_s32(sum_row6_group0, weights, left_row6, 0);
            sum_row7_group0 = svdot_lane_s32(sum_row7_group0, weights, left_row7, 0);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 1u * right_group_stride + 4u * feature + 0u);
            sum_row0_group1 = svdot_lane_s32(sum_row0_group1, weights, left_row0, 0);
            sum_row1_group1 = svdot_lane_s32(sum_row1_group1, weights, left_row1, 0);
            sum_row2_group1 = svdot_lane_s32(sum_row2_group1, weights, left_row2, 0);
            sum_row3_group1 = svdot_lane_s32(sum_row3_group1, weights, left_row3, 0);
            sum_row4_group1 = svdot_lane_s32(sum_row4_group1, weights, left_row4, 0);
            sum_row5_group1 = svdot_lane_s32(sum_row5_group1, weights, left_row5, 0);
            sum_row6_group1 = svdot_lane_s32(sum_row6_group1, weights, left_row6, 0);
            sum_row7_group1 = svdot_lane_s32(sum_row7_group1, weights, left_row7, 0);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 0u * right_group_stride + 4u * feature + 16u);
            sum_row0_group0 = svdot_lane_s32(sum_row0_group0, weights, left_row0, 1);
            sum_row1_group0 = svdot_lane_s32(sum_row1_group0, weights, left_row1, 1);
            sum_row2_group0 = svdot_lane_s32(sum_row2_group0, weights, left_row2, 1);
            sum_row3_group0 = svdot_lane_s32(sum_row3_group0, weights, left_row3, 1);
            sum_row4_group0 = svdot_lane_s32(sum_row4_group0, weights, left_row4, 1);
            sum_row5_group0 = svdot_lane_s32(sum_row5_group0, weights, left_row5, 1);
            sum_row6_group0 = svdot_lane_s32(sum_row6_group0, weights, left_row6, 1);
            sum_row7_group0 = svdot_lane_s32(sum_row7_group0, weights, left_row7, 1);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 1u * right_group_stride + 4u * feature + 16u);
            sum_row0_group1 = svdot_lane_s32(sum_row0_group1, weights, left_row0, 1);
            sum_row1_group1 = svdot_lane_s32(sum_row1_group1, weights, left_row1, 1);
            sum_row2_group1 = svdot_lane_s32(sum_row2_group1, weights, left_row2, 1);
            sum_row3_group1 = svdot_lane_s32(sum_row3_group1, weights, left_row3, 1);
            sum_row4_group1 = svdot_lane_s32(sum_row4_group1, weights, left_row4, 1);
            sum_row5_group1 = svdot_lane_s32(sum_row5_group1, weights, left_row5, 1);
            sum_row6_group1 = svdot_lane_s32(sum_row6_group1, weights, left_row6, 1);
            sum_row7_group1 = svdot_lane_s32(sum_row7_group1, weights, left_row7, 1);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 0u * right_group_stride + 4u * feature + 32u);
            sum_row0_group0 = svdot_lane_s32(sum_row0_group0, weights, left_row0, 2);
            sum_row1_group0 = svdot_lane_s32(sum_row1_group0, weights, left_row1, 2);
            sum_row2_group0 = svdot_lane_s32(sum_row2_group0, weights, left_row2, 2);
            sum_row3_group0 = svdot_lane_s32(sum_row3_group0, weights, left_row3, 2);
            sum_row4_group0 = svdot_lane_s32(sum_row4_group0, weights, left_row4, 2);
            sum_row5_group0 = svdot_lane_s32(sum_row5_group0, weights, left_row5, 2);
            sum_row6_group0 = svdot_lane_s32(sum_row6_group0, weights, left_row6, 2);
            sum_row7_group0 = svdot_lane_s32(sum_row7_group0, weights, left_row7, 2);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 1u * right_group_stride + 4u * feature + 32u);
            sum_row0_group1 = svdot_lane_s32(sum_row0_group1, weights, left_row0, 2);
            sum_row1_group1 = svdot_lane_s32(sum_row1_group1, weights, left_row1, 2);
            sum_row2_group1 = svdot_lane_s32(sum_row2_group1, weights, left_row2, 2);
            sum_row3_group1 = svdot_lane_s32(sum_row3_group1, weights, left_row3, 2);
            sum_row4_group1 = svdot_lane_s32(sum_row4_group1, weights, left_row4, 2);
            sum_row5_group1 = svdot_lane_s32(sum_row5_group1, weights, left_row5, 2);
            sum_row6_group1 = svdot_lane_s32(sum_row6_group1, weights, left_row6, 2);
            sum_row7_group1 = svdot_lane_s32(sum_row7_group1, weights, left_row7, 2);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 0u * right_group_stride + 4u * feature + 48u);
            sum_row0_group0 = svdot_lane_s32(sum_row0_group0, weights, left_row0, 3);
            sum_row1_group0 = svdot_lane_s32(sum_row1_group0, weights, left_row1, 3);
            sum_row2_group0 = svdot_lane_s32(sum_row2_group0, weights, left_row2, 3);
            sum_row3_group0 = svdot_lane_s32(sum_row3_group0, weights, left_row3, 3);
            sum_row4_group0 = svdot_lane_s32(sum_row4_group0, weights, left_row4, 3);
            sum_row5_group0 = svdot_lane_s32(sum_row5_group0, weights, left_row5, 3);
            sum_row6_group0 = svdot_lane_s32(sum_row6_group0, weights, left_row6, 3);
            sum_row7_group0 = svdot_lane_s32(sum_row7_group0, weights, left_row7, 3);
        }
        {
            const svint8_t weights = svld1_s8(
                bytes, packed_right + 1u * right_group_stride + 4u * feature + 48u);
            sum_row0_group1 = svdot_lane_s32(sum_row0_group1, weights, left_row0, 3);
            sum_row1_group1 = svdot_lane_s32(sum_row1_group1, weights, left_row1, 3);
            sum_row2_group1 = svdot_lane_s32(sum_row2_group1, weights, left_row2, 3);
            sum_row3_group1 = svdot_lane_s32(sum_row3_group1, weights, left_row3, 3);
            sum_row4_group1 = svdot_lane_s32(sum_row4_group1, weights, left_row4, 3);
            sum_row5_group1 = svdot_lane_s32(sum_row5_group1, weights, left_row5, 3);
            sum_row6_group1 = svdot_lane_s32(sum_row6_group1, weights, left_row6, 3);
            sum_row7_group1 = svdot_lane_s32(sum_row7_group1, weights, left_row7, 3);
        }
    }
    svst1_u32(words, accumulators + 0u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 0u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row0_group0)));
    svst1_u32(words, accumulators + 0u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 0u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row0_group1)));
    svst1_u32(words, accumulators + 1u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 1u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row1_group0)));
    svst1_u32(words, accumulators + 1u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 1u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row1_group1)));
    svst1_u32(words, accumulators + 2u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 2u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row2_group0)));
    svst1_u32(words, accumulators + 2u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 2u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row2_group1)));
    svst1_u32(words, accumulators + 3u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 3u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row3_group0)));
    svst1_u32(words, accumulators + 3u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 3u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row3_group1)));
    svst1_u32(words, accumulators + 4u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 4u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row4_group0)));
    svst1_u32(words, accumulators + 4u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 4u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row4_group1)));
    svst1_u32(words, accumulators + 5u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 5u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row5_group0)));
    svst1_u32(words, accumulators + 5u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 5u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row5_group1)));
    svst1_u32(words, accumulators + 6u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 6u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row6_group0)));
    svst1_u32(words, accumulators + 6u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 6u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row6_group1)));
    svst1_u32(words, accumulators + 7u * output_row_stride + 0u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 7u * output_row_stride + 0u),
                          svreinterpret_u32_s32(sum_row7_group0)));
    svst1_u32(words, accumulators + 7u * output_row_stride + 4u,
              svadd_u32_x(words,
                          svld1_u32(words, accumulators + 7u * output_row_stride + 4u),
                          svreinterpret_u32_s32(sum_row7_group1)));
}
/* Eight-by-eight interface retains existing packing and ownership contracts. */
static inline __attribute__((always_inline)) void
attention_sdot_kernel_8x8(const int8_t *packed_left, const int8_t *packed_right,
                          uint32_t *accumulators, size_t padded_features,
                          size_t left_row_stride, size_t right_group_stride,
                          size_t output_row_stride) {
    attention_sdot_intrinsic_8x8(
        packed_left + 0u * left_row_stride, packed_right + 0u * right_group_stride,
        accumulators + 0u * output_row_stride + 0u, padded_features, left_row_stride,
        right_group_stride, output_row_stride);
}
