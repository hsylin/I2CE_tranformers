// Construct the file format in storage order and read it in logical order.
// This oracle deliberately does not call the production offset calculator.
#include "transformer_layers/compressed_index_file.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

static uint8_t logical_index(size_t column, size_t feature) {
    return uint8_t((column * 13 + feature * 7 + column / 3 + feature / 5) % 4);
}

int main() {
    size_t cases = 0;
    for (size_t columns : {1, 2, 3, 4, 17, 31, 32, 33, 64}) {
        for (size_t features : {1, 7, 8, 15, 16, 17, 127, 128, 129, 256}) {
            std::vector<uint8_t> payload;
            for (size_t column_base = 0; column_base < columns; column_base += 32) {
                for (size_t feature_base = 0; feature_base < features;
                     feature_base += 128) {
                    for (size_t pair = 0; pair < 16; ++pair) {
                        for (size_t group = 0; group < 8; ++group) {
                            for (size_t half = 0; half < 2; ++half) {
                                uint32_t word = 0;
                                for (size_t lane = 0; lane < 16; ++lane) {
                                    const size_t column =
                                        column_base + pair * 2 + lane / 8;
                                    const size_t feature =
                                        feature_base + group * 16 + half * 8 + lane % 8;
                                    const uint8_t value =
                                        column < columns && feature < features
                                            ? logical_index(column, feature)
                                            : 0;
                                    word |= uint32_t(value) << (2 * lane);
                                }
                                for (size_t byte = 0; byte < 4; ++byte) {
                                    payload.push_back(uint8_t(word >> (8 * byte)));
                                }
                            }
                        }
                    }
                }
            }
            if (payload.size() != ci_payload_bytes(columns, features) ||
                !i2ce_ci::padding_zero(payload.data(), columns, features)) {
                std::abort();
            }
            for (size_t column = 0; column < columns; ++column) {
                for (size_t feature = 0; feature < features; ++feature) {
                    if (i2ce_ci::index(payload.data(), columns, features, column,
                                       feature) != logical_index(column, feature)) {
                        std::abort();
                    }
                }
            }
            ++cases;
        }
    }
    if (ci_payload_bytes(0, 1) || ci_payload_bytes(1, 0) ||
        ci_payload_bytes(65536, 1) || ci_payload_bytes(1, 65536)) {
        std::abort();
    }
    std::printf("PASS: %zu independent CI01 storage-order layouts\n", cases);
}
