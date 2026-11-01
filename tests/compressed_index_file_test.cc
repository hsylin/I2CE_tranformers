// File-format validation is separate from the arithmetic and packing oracle.
#include "transformer_layers/compressed_index_file.h"
#include <array>
#include <cstdlib>
#include <fstream>
#include <vector>

static void write_file(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    if (!file) {
        std::abort();
    }
}

int main() {
    const size_t columns = 17, features = 129, words = 9;
    std::vector<uint32_t> original(columns * words, 0);
    const int32_t codebook[8] = {-128, 127, -1, 1, 0, -128, 127, 0};
    const uint64_t identity =
        i2ce_ci::identity(original.data(), words, columns, features, codebook, nullptr);
    std::vector<uint8_t> valid(96 + 2048, 0);
    std::memcpy(valid.data(), "I2CECI01", 8);
    const uint32_t fields[] = {96, 1, columns, features, words, 2, 1, 32, 128, 2};
    for (size_t field = 0; field < 10; ++field) {
        i2ce_ci::put(valid.data() + 8 + field * 4, fields[field], 4);
    }
    i2ce_ci::put(valid.data() + 48, 2048, 8);
    i2ce_ci::put(valid.data() + 56, identity, 8);
    i2ce_ci::put(valid.data() + 64,
                 i2ce_ci::hash_bytes(i2ce_ci::seed, valid.data() + 96, 2048), 8);
    i2ce_ci::put(valid.data() + 72, UINT64_C(0x314b385249415032), 8);
    const std::string path = "compressed-index-format-fixture.ci2";
    std::vector<uint8_t> destination(2048, 0x5a);
    write_file(path, valid);
    i2ce_ci::load(path, identity, words, columns, features, destination.data(),
                  destination.size());
    for (uint8_t value : destination) {
        if (value != 0) {
            std::abort();
        }
    }

    // Bad magic, each layout field, extents/identity/checksum/layout ID and
    // reserved bytes must fail, including a valid checksum on nonzero padding.
    unsigned rejected = 0;
    for (size_t position :
         {0, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80}) {
        auto bytes = valid;
        bytes[position] ^= 1;
        write_file(path, bytes);
        try {
            i2ce_ci::load(path, identity, words, columns, features, destination.data(),
                          destination.size());
        } catch (const std::exception &) {
            ++rejected;
            continue;
        }
        std::abort();
    }
    for (unsigned mutation = 0; mutation < 4; ++mutation) {
        auto bytes = valid;
        if (mutation == 0) {
            bytes.resize(95);
        }
        if (mutation == 1) {
            bytes.pop_back();
        }
        if (mutation == 2) {
            bytes.push_back(0);
        }
        if (mutation == 3) {
            bytes.back() = 1; // N/K padding, with deliberately updated checksum.
            i2ce_ci::put(bytes.data() + 64,
                         i2ce_ci::hash_bytes(i2ce_ci::seed, bytes.data() + 96, 2048),
                         8);
        }
        write_file(path, bytes);
        try {
            i2ce_ci::load(path, identity, words, columns, features, destination.data(),
                          destination.size());
        } catch (const std::exception &) {
            ++rejected;
            continue;
        }
        std::abort();
    }
    std::printf("PASS: valid CI01 file and %u rejected header/payload/padding cases\n",
                rejected);
}
