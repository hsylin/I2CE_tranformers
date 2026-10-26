#pragma once
#include <arm_sve.h>
#include <stddef.h>
#include <stdint.h>

// SVE128: K16-multiple panel extent, sixteen vector sums, modulo accumulation.
static inline __attribute__((always_inline)) void
i2ce_sdot_intrinsic_mr4nr16(const int8_t *packed_a, const int8_t *packed_b,
                            uint32_t *partial_c, size_t padded_features,
                            size_t a_stride, size_t b_group_stride, size_t c_stride) {
    const svbool_t bytes = svptrue_b8(), words = svptrue_b32();
    svint32_t sum_r0_g0 = svdup_s32(0);
    svint32_t sum_r0_g1 = svdup_s32(0);
    svint32_t sum_r0_g2 = svdup_s32(0);
    svint32_t sum_r0_g3 = svdup_s32(0);
    svint32_t sum_r1_g0 = svdup_s32(0);
    svint32_t sum_r1_g1 = svdup_s32(0);
    svint32_t sum_r1_g2 = svdup_s32(0);
    svint32_t sum_r1_g3 = svdup_s32(0);
    svint32_t sum_r2_g0 = svdup_s32(0);
    svint32_t sum_r2_g1 = svdup_s32(0);
    svint32_t sum_r2_g2 = svdup_s32(0);
    svint32_t sum_r2_g3 = svdup_s32(0);
    svint32_t sum_r3_g0 = svdup_s32(0);
    svint32_t sum_r3_g1 = svdup_s32(0);
    svint32_t sum_r3_g2 = svdup_s32(0);
    svint32_t sum_r3_g3 = svdup_s32(0);
    for (size_t feature = 0; feature < padded_features; feature += 16u) {
        const svint8_t input_row0 = svld1_s8(bytes, packed_a + 0u * a_stride + feature);
        const svint8_t input_row1 = svld1_s8(bytes, packed_a + 1u * a_stride + feature);
        const svint8_t input_row2 = svld1_s8(bytes, packed_a + 2u * a_stride + feature);
        const svint8_t input_row3 = svld1_s8(bytes, packed_a + 3u * a_stride + feature);
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 0u * b_group_stride + 4u * feature + 0u);
            sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 0);
            sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 0);
            sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 0);
            sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 0);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 1u * b_group_stride + 4u * feature + 0u);
            sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 0);
            sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 0);
            sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 0);
            sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 0);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 2u * b_group_stride + 4u * feature + 0u);
            sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 0);
            sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 0);
            sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 0);
            sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 0);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 3u * b_group_stride + 4u * feature + 0u);
            sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 0);
            sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 0);
            sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 0);
            sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 0);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 0u * b_group_stride + 4u * feature + 16u);
            sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 1);
            sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 1);
            sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 1);
            sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 1);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 1u * b_group_stride + 4u * feature + 16u);
            sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 1);
            sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 1);
            sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 1);
            sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 1);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 2u * b_group_stride + 4u * feature + 16u);
            sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 1);
            sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 1);
            sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 1);
            sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 1);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 3u * b_group_stride + 4u * feature + 16u);
            sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 1);
            sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 1);
            sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 1);
            sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 1);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 0u * b_group_stride + 4u * feature + 32u);
            sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 2);
            sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 2);
            sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 2);
            sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 2);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 1u * b_group_stride + 4u * feature + 32u);
            sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 2);
            sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 2);
            sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 2);
            sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 2);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 2u * b_group_stride + 4u * feature + 32u);
            sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 2);
            sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 2);
            sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 2);
            sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 2);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 3u * b_group_stride + 4u * feature + 32u);
            sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 2);
            sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 2);
            sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 2);
            sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 2);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 0u * b_group_stride + 4u * feature + 48u);
            sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 3);
            sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 3);
            sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 3);
            sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 3);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 1u * b_group_stride + 4u * feature + 48u);
            sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 3);
            sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 3);
            sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 3);
            sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 3);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 2u * b_group_stride + 4u * feature + 48u);
            sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 3);
            sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 3);
            sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 3);
            sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 3);
        }
        {
            const svint8_t w =
                svld1_s8(bytes, packed_b + 3u * b_group_stride + 4u * feature + 48u);
            sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 3);
            sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 3);
            sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 3);
            sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 3);
        }
    }
    svst1_u32(words, partial_c + 0u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r0_g0)));
    svst1_u32(words, partial_c + 0u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r0_g1)));
    svst1_u32(words, partial_c + 0u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r0_g2)));
    svst1_u32(words, partial_c + 0u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r0_g3)));
    svst1_u32(words, partial_c + 1u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r1_g0)));
    svst1_u32(words, partial_c + 1u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r1_g1)));
    svst1_u32(words, partial_c + 1u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r1_g2)));
    svst1_u32(words, partial_c + 1u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r1_g3)));
    svst1_u32(words, partial_c + 2u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r2_g0)));
    svst1_u32(words, partial_c + 2u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r2_g1)));
    svst1_u32(words, partial_c + 2u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r2_g2)));
    svst1_u32(words, partial_c + 2u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r2_g3)));
    svst1_u32(words, partial_c + 3u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r3_g0)));
    svst1_u32(words, partial_c + 3u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r3_g1)));
    svst1_u32(words, partial_c + 3u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r3_g2)));
    svst1_u32(words, partial_c + 3u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r3_g3)));
}

// Full K256 consumer: two adjacent physical K128 panels.
static __attribute__((noinline)) void i2ce_sdot_cb_mr4nr16_k256(const int8_t *packed_a,
                                                                const int8_t *packed_b,
                                                                uint32_t *partial_c) {
    const size_t a_stride = 528u, b_group_stride = 576u, c_stride = 32u;
    const svbool_t bytes = svptrue_b8(), words = svptrue_b32();
    svint32_t sum_r0_g0 = svdup_s32(0);
    svint32_t sum_r0_g1 = svdup_s32(0);
    svint32_t sum_r0_g2 = svdup_s32(0);
    svint32_t sum_r0_g3 = svdup_s32(0);
    svint32_t sum_r1_g0 = svdup_s32(0);
    svint32_t sum_r1_g1 = svdup_s32(0);
    svint32_t sum_r1_g2 = svdup_s32(0);
    svint32_t sum_r1_g3 = svdup_s32(0);
    svint32_t sum_r2_g0 = svdup_s32(0);
    svint32_t sum_r2_g1 = svdup_s32(0);
    svint32_t sum_r2_g2 = svdup_s32(0);
    svint32_t sum_r2_g3 = svdup_s32(0);
    svint32_t sum_r3_g0 = svdup_s32(0);
    svint32_t sum_r3_g1 = svdup_s32(0);
    svint32_t sum_r3_g2 = svdup_s32(0);
    svint32_t sum_r3_g3 = svdup_s32(0);
    for (size_t panel = 0; panel < 2u; ++panel) {
        for (size_t feature = 0; feature < 128u; feature += 16u) {
            const svint8_t input_row0 =
                svld1_s8(bytes, packed_a + 0u * a_stride + panel * 128u + feature);
            const svint8_t input_row1 =
                svld1_s8(bytes, packed_a + 1u * a_stride + panel * 128u + feature);
            const svint8_t input_row2 =
                svld1_s8(bytes, packed_a + 2u * a_stride + panel * 128u + feature);
            const svint8_t input_row3 =
                svld1_s8(bytes, packed_a + 3u * a_stride + panel * 128u + feature);
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 0u * b_group_stride +
                                        4u * feature + 0u);
                sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 0);
                sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 0);
                sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 0);
                sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 0);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 1u * b_group_stride +
                                        4u * feature + 0u);
                sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 0);
                sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 0);
                sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 0);
                sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 0);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 2u * b_group_stride +
                                        4u * feature + 0u);
                sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 0);
                sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 0);
                sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 0);
                sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 0);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 3u * b_group_stride +
                                        4u * feature + 0u);
                sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 0);
                sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 0);
                sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 0);
                sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 0);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 0u * b_group_stride +
                                        4u * feature + 16u);
                sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 1);
                sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 1);
                sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 1);
                sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 1);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 1u * b_group_stride +
                                        4u * feature + 16u);
                sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 1);
                sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 1);
                sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 1);
                sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 1);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 2u * b_group_stride +
                                        4u * feature + 16u);
                sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 1);
                sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 1);
                sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 1);
                sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 1);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 3u * b_group_stride +
                                        4u * feature + 16u);
                sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 1);
                sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 1);
                sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 1);
                sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 1);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 0u * b_group_stride +
                                        4u * feature + 32u);
                sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 2);
                sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 2);
                sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 2);
                sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 2);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 1u * b_group_stride +
                                        4u * feature + 32u);
                sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 2);
                sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 2);
                sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 2);
                sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 2);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 2u * b_group_stride +
                                        4u * feature + 32u);
                sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 2);
                sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 2);
                sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 2);
                sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 2);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 3u * b_group_stride +
                                        4u * feature + 32u);
                sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 2);
                sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 2);
                sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 2);
                sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 2);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 0u * b_group_stride +
                                        4u * feature + 48u);
                sum_r0_g0 = svdot_lane_s32(sum_r0_g0, w, input_row0, 3);
                sum_r1_g0 = svdot_lane_s32(sum_r1_g0, w, input_row1, 3);
                sum_r2_g0 = svdot_lane_s32(sum_r2_g0, w, input_row2, 3);
                sum_r3_g0 = svdot_lane_s32(sum_r3_g0, w, input_row3, 3);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 1u * b_group_stride +
                                        4u * feature + 48u);
                sum_r0_g1 = svdot_lane_s32(sum_r0_g1, w, input_row0, 3);
                sum_r1_g1 = svdot_lane_s32(sum_r1_g1, w, input_row1, 3);
                sum_r2_g1 = svdot_lane_s32(sum_r2_g1, w, input_row2, 3);
                sum_r3_g1 = svdot_lane_s32(sum_r3_g1, w, input_row3, 3);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 2u * b_group_stride +
                                        4u * feature + 48u);
                sum_r0_g2 = svdot_lane_s32(sum_r0_g2, w, input_row0, 3);
                sum_r1_g2 = svdot_lane_s32(sum_r1_g2, w, input_row1, 3);
                sum_r2_g2 = svdot_lane_s32(sum_r2_g2, w, input_row2, 3);
                sum_r3_g2 = svdot_lane_s32(sum_r3_g2, w, input_row3, 3);
            }
            {
                const svint8_t w =
                    svld1_s8(bytes, packed_b + panel * 9216u + 3u * b_group_stride +
                                        4u * feature + 48u);
                sum_r0_g3 = svdot_lane_s32(sum_r0_g3, w, input_row0, 3);
                sum_r1_g3 = svdot_lane_s32(sum_r1_g3, w, input_row1, 3);
                sum_r2_g3 = svdot_lane_s32(sum_r2_g3, w, input_row2, 3);
                sum_r3_g3 = svdot_lane_s32(sum_r3_g3, w, input_row3, 3);
            }
        }
    }
    svst1_u32(words, partial_c + 0u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r0_g0)));
    svst1_u32(words, partial_c + 0u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r0_g1)));
    svst1_u32(words, partial_c + 0u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r0_g2)));
    svst1_u32(words, partial_c + 0u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 0u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r0_g3)));
    svst1_u32(words, partial_c + 1u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r1_g0)));
    svst1_u32(words, partial_c + 1u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r1_g1)));
    svst1_u32(words, partial_c + 1u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r1_g2)));
    svst1_u32(words, partial_c + 1u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 1u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r1_g3)));
    svst1_u32(words, partial_c + 2u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r2_g0)));
    svst1_u32(words, partial_c + 2u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r2_g1)));
    svst1_u32(words, partial_c + 2u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r2_g2)));
    svst1_u32(words, partial_c + 2u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 2u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r2_g3)));
    svst1_u32(words, partial_c + 3u * c_stride + 0u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 0u),
                          svreinterpret_u32_s32(sum_r3_g0)));
    svst1_u32(words, partial_c + 3u * c_stride + 4u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 4u),
                          svreinterpret_u32_s32(sum_r3_g1)));
    svst1_u32(words, partial_c + 3u * c_stride + 8u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 8u),
                          svreinterpret_u32_s32(sum_r3_g2)));
    svst1_u32(words, partial_c + 3u * c_stride + 12u,
              svadd_u32_x(words, svld1_u32(words, partial_c + 3u * c_stride + 12u),
                          svreinterpret_u32_s32(sum_r3_g3)));
}

/* Variable K16 extent for partial L1 panels; the packed layout is unchanged. */
static __attribute__((noinline)) void i2ce_sdot_cb_mr4nr16(const int8_t *packed_a,
                                                           const int8_t *packed_b,
                                                           uint32_t *partial_c,
                                                           size_t padded_features) {
    i2ce_sdot_intrinsic_mr4nr16(packed_a, packed_b, partial_c, padded_features, 528u,
                                576u, 32u);
}
