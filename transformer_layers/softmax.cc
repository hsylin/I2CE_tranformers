#include "softmax.h"
#include <cmath>
#include <iostream>

static const  uint8_t  lookup[32] = {
        4, 5, 7, 8, 11, 14, 18, 23, 30, 38, 49, 63, 80, 103, 132, 170, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3,
};

Softmax::Softmax()= default;

Softmax::~Softmax()= default;

void Softmax::compute(uint32_t *input, std::size_t seq_len){
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = (uint8_t*) (input + i * (seq_len >> 2));

        for (int j=0; j< seq_len; j++){
            *(input_uptr) = lookup[(* (uint8_t *) input_uptr) >> 3]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
            sum += *(input_uptr);
            input_uptr ++;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
        for (int j=0; j< seq_len; j++){
//            std::cout << "Ptr " << i << "\t: " << (int) *(input_ptr)  << std::endl;
            *(input_uptr) = (uint8_t) ((*(input_uptr)) /(sum >> 8)); // divide the sum by 256 otherwise all the outputs will be 0!
//            std::cout << "LUT " << i << "\t: " << (int) *(input_ptr) << std::endl;
            input_uptr ++;
        }
//        input_uptr = (uint8_t*) (input + i * (seq_len >> 2));
//        int sum_softmax = 0;
//        for (int j=0; j< seq_len; j++)
//            sum_softmax += *(input_uptr++);
    }
}

/**
 * @brief Approximate softmax for two learners stored in one interleaved score matrix.
 *
 * The attention score buffer is laid out as [query][key][learner] with two
 * learners per score.  The function updates the buffer in place.  Values are
 * reinterpreted as uint8_t because the lookup-table approximation operates on
 * byte values rather than signed arithmetic.
 *
 * Step-by-step for each query row:
 * 1. Walk across all key positions.
 * 2. For each learner lane, divide the score by sqrt(d_q) through a right
 *    shift by 3, then use that value as an index into the exponent LUT.
 * 3. Accumulate the approximated exponent sum separately for each learner.
 * 4. Clamp zero sums to one so normalization never divides by zero.
 * 5. Walk the same row again and divide each LUT output by sum / 256.
 * 6. If sum / 256 rounds down to zero, clamp the denominator to one.
 */
void Softmax::computeInterleaved2Learners(int8_t *input, std::size_t seq_len) {
    // The signed int8_t buffer is treated as raw bytes for the LUT-based
    // softmax approximation, matching the scalar compute() path.
    auto* input_u8 = reinterpret_cast<uint8_t*>(input);
    for (std::size_t query = 0; query < seq_len; query++) {
        uint32_t sum[2] = {0, 0};

        // First pass: replace each score with its exponent approximation and
        // accumulate a separate normalization sum for each learner.
        for (std::size_t key = 0; key < seq_len; key++) {
            uint8_t* slot = input_u8 + ((query * seq_len + key) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                slot[learner] = lookup[slot[learner] >> 3];
                sum[learner] += slot[learner];
            }
        }

        // A zero sum would make the normalization undefined; keep the row
        // stable by using the smallest non-zero denominator source.
        for (std::size_t learner = 0; learner < 2u; learner++) {
            if (sum[learner] == 0u) {
                sum[learner] = 1u;
            }
        }

        // Second pass: normalize the approximated exponentials in place.
        // Dividing the sum by 256 preserves useful int8-scale outputs instead
        // of truncating every small probability to zero.
        for (std::size_t key = 0; key < seq_len; key++) {
            uint8_t* slot = input_u8 + ((query * seq_len + key) * 2u);
            for (std::size_t learner = 0; learner < 2u; learner++) {
                uint32_t denom = sum[learner] >> 8;
                if (denom == 0u) {
                    denom = 1u;
                }
                slot[learner] = static_cast<uint8_t>(slot[learner] / denom);
            }
        }
    }
}

/**
 * @brief Approximate softmax for four learners stored in one interleaved score matrix.
 *
 * The attention score buffer is laid out as [query][key][learner] with four
 * adjacent learner lanes per score.  Each learner row is normalized
 * independently, but all four learners share the same query/key traversal.
 *
 * Step-by-step for each query row:
 * 1. Visit every key position and locate its four-lane interleaved slot.
 * 2. Convert each learner's score into an exponent approximation using
 *    lookup[score >> 3], where the shift approximates division by sqrt(64).
 * 3. Accumulate one exponent sum per learner.
 * 4. Clamp zero sums to one before normalization.
 * 5. Revisit the row and normalize every learner lane by sum / 256.
 * 6. Clamp sum / 256 to one if the fixed-point shift would produce zero.
 */
void Softmax::computeInterleaved4Learners(int8_t *input, std::size_t seq_len) {
    // The LUT path works on bytes; this mirrors computeInterleaved2Learners() and the
    // non-interleaved compute() implementation.
    auto* input_u8 = reinterpret_cast<uint8_t*>(input);
    for (std::size_t query = 0; query < seq_len; query++) {
        uint32_t sum[4] = {0, 0, 0, 0};

        // First pass: LUT exponent approximation and per-learner row sums.
        for (std::size_t key = 0; key < seq_len; key++) {
            uint8_t* slot = input_u8 + ((query * seq_len + key) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                slot[learner] = lookup[slot[learner] >> 3];
                sum[learner] += slot[learner];
            }
        }

        // Avoid a zero normalization source for rows whose LUT outputs summed
        // to zero.
        for (std::size_t learner = 0; learner < 4u; learner++) {
            if (sum[learner] == 0u) {
                sum[learner] = 1u;
            }
        }

        // Second pass: write the normalized softmax approximation back into
        // the same [query][key][learner] layout.
        for (std::size_t key = 0; key < seq_len; key++) {
            uint8_t* slot = input_u8 + ((query * seq_len + key) * 4u);
            for (std::size_t learner = 0; learner < 4u; learner++) {
                uint32_t denom = sum[learner] >> 8;
                if (denom == 0u) {
                    denom = 1u;
                }
                slot[learner] = static_cast<uint8_t>(slot[learner] / denom);
            }
        }
    }
}

void Softmax::computeRearranged(uint32_t *input, std::size_t seq_len, std::size_t kernelDim) {
    // We assume that the input value are fixed-point with 2 bits of fraction.
    for (int i =0; i< seq_len; i++){
        int32_t sum = 0;
        auto* input_uptr = ((uint8_t*) input) + i * kernelDim;
        for (int j =0; j< seq_len / kernelDim; j++){
            for (int k=0; k< kernelDim; k++) {
                *(input_uptr+k) = lookup[(* (uint8_t *) (input_uptr+ k)) >> 3]; // divide by the sqrt od the d_q which is sqrt(64) -> 8
                sum += *(input_uptr+k);
            }
            input_uptr += seq_len* kernelDim;
        }
        sum = (sum==0) ? sum + 1 : sum;
        input_uptr = ((uint8_t*) input) + i * kernelDim;
        for (int j =0; j< seq_len / kernelDim; j++){
            for (int k=0; k< kernelDim; k++) {
                *(input_uptr+k) = (uint8_t) ((*(input_uptr+k)) /(sum >> 8));
            }
            input_uptr += seq_len* kernelDim;
        }
    }
}

void Softmax::post_softmax(uint32_t *input, std::size_t seq_len, std::size_t headSize){
    auto* input_ptr = (int8_t*) input;
    for (int i =0; i< seq_len * headSize; i++){
        *input_ptr = (int8_t) (*(input_ptr) >> 6);
        input_ptr++;
    }
}

void Softmax::post_softmax_interleaved2D(int8_t *input, std::size_t seq_len, std::size_t headSize) {
    for (std::size_t idx = 0; idx < seq_len * headSize * 2u; idx++) {
        input[idx] = static_cast<int8_t>(input[idx] >> 6);
    }
}

void Softmax::post_softmax_interleaved4D(int8_t *input, std::size_t seq_len, std::size_t headSize) {
    for (std::size_t idx = 0; idx < seq_len * headSize * 4u; idx++) {
        input[idx] = static_cast<int8_t>(input[idx] >> 6);
    }
}


// void Softmax::computeFloat(uint32_t *in_matrix, std::size_t seq_len) {
//     int32_t fractional_bits = 16;

//     // Split the input uint32_t array into int8_t values
//     std::vector<int8_t> input_int8(seq_len);
//     for (std::size_t i = 0; i < seq_len / 4; ++i) {
//         uint32_t value = in_matrix[i];
//         for (int j = 0; j < 4; ++j) {
//             input_int8[i * 4 + j] = static_cast<int8_t>((value >> (8 * j)) & 0xFF);
//         }
//     }

//     // Convert int8_t values into fixed-point representation
//     std::vector<int32_t> input_fixed(seq_len);
//     for (std::size_t i = 0; i < seq_len; ++i) {
//         input_fixed[i] = float_to_fixed(static_cast<float>(input_int8[i]), fractional_bits);
//     }

//     // Calculate the softmax for each chunk using fixed-point arithmetic
//     softmax_fixed(input_fixed, fractional_bits);

//     // (Optional) Convert the fixed-point result back to floating-point representation
//     std::vector<float> result(seq_len);
//     for (std::size_t i = 0; i < seq_len; ++i) {
//         result[i] = fixed_to_float(input_fixed[i], fractional_bits);
//         std::cout << "result[" << i << "]: " << result[i] << std::endl;
//     }
// }

// int32_t Softmax::float_to_fixed(float value, int32_t fractional_bits) {
//     return static_cast<int32_t>(round(value * (1 << fractional_bits)));
// }

// float Softmax::fixed_to_float(int32_t fixed_value, int32_t fractional_bits) {
//     return static_cast<float>(fixed_value) / (1 << fractional_bits);
// }

// void Softmax::softmax_fixed(std::vector<int32_t> &input, int32_t fractional_bits) {
//     std::size_t chunk_size = input.size() / 4;

//     for (std::size_t i = 0; i < 4; ++i) {
//         int32_t max_val = input[i * chunk_size];

//         // Find the maximum value in the chunk
//         for (std::size_t j = i * chunk_size + 1; j < (i + 1) * chunk_size; ++j) {
//             if (input[j] > max_val) {
//                 max_val = input[j];
//             }
//         }

//         int32_t sum_exp = 0;
//         for (std::size_t j = i * chunk_size; j < (i + 1) * chunk_size; ++j) {
//             // Subtract max_val and exponentiate using fixed-point arithmetic
//             int32_t exp_val = float_to_fixed(exp(fixed_to_float(input[j] - max_val, fractional_bits)), fractional_bits);
//             input[j] = exp_val;
//             sum_exp += exp_val;
//         }

//         // Normalize the values in the chunk by dividing by the sum of exponentiated values
//         for (std::size_t j = i * chunk_size; j < (i + 1) * chunk_size; ++j) {
//             input[j] = (input[j] << fractional_bits) / sum_exp;
//         }
//     }
// }
