// Host-side export from the exact generated registry used by a build.
// Usage: c++ -std=c++17 -I. -I<headers> tools/exp/export_compressed_indices.cc -o
// exporter
//        exporter <new-output-directory>
#include "generated_codebook_registry.h"
#include "transformer_layers/compressed_index_file.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("expected a new output directory");
        }
        const std::filesystem::path out(argv[1]);
        if (!std::filesystem::create_directory(out)) {
            throw std::runtime_error("output directory already exists");
        }
        for (size_t layer = 0; layer < kGeneratedCodebookLayerCount; ++layer) {
            const auto &v = kGeneratedCodebookLayers[layer];
            const unsigned bits = v.bits_per_cb;
            if (bits < 2 || bits > 4 ||
                (v.n_learners != 1 && v.n_learners != 2 && v.n_learners != 4) ||
                (v.n_learners != 1 && !v.same_seq) ||
                !(v.n_learners == 1 ? v.codebook_int8 : v.codebook_int8_interleaved) ||
                v.bias_interleaved || v.bias || v.biases) {
                throw std::runtime_error(
                    "expected bias-free shared 1/2/4-learner CB4/8/16 registry");
            }
            const unsigned entries = 1u << bits;
            std::vector<int32_t> cb(entries * v.n_learners);
            for (size_t i = 0; i < cb.size(); ++i) {
                cb[i] = v.n_learners == 1 ? v.codebook_int8[i]
                                          : v.codebook_int8_interleaved[i];
            }
            const auto id =
                i2ce_ci::identity(v.weight_idx, v.n_words_row, v.output_size,
                                  v.input_size, cb.data(), nullptr, bits, v.n_learners);
            const size_t bytes =
                ci_payload_bytes_bits(v.output_size, v.input_size, bits);
            std::vector<uint8_t> encoded(i2ce_ci::header_bytes + bytes, 0);
            auto *payload = encoded.data() + i2ce_ci::header_bytes;
            const unsigned per_word = 32u / bits;
            for (size_t col = 0; col < v.output_size; ++col) {
                for (size_t k = 0; k < v.input_size; ++k) {
                    const unsigned index =
                        (v.weight_idx[col * v.n_words_row + k / per_word] >>
                         ((k % per_word) * bits)) &
                        (entries - 1u);
                    const size_t record = ci_pair_k8_offset_bits(
                        (v.input_size + 127u) / 128u, col, k, bits);
                    const size_t bit_position = ((col % 2u) * 8u + k % 8u) * bits;
                    for (unsigned bit = 0; bit < bits; ++bit) {
                        payload[record + (bit_position + bit) / 8u] |= uint8_t(
                            ((index >> bit) & 1u) << ((bit_position + bit) % 8u));
                    }
                }
            }
            std::memcpy(encoded.data(), bits == 2u ? "I2CECI01" : "I2CECI02", 8);
            const uint32_t fields[] = {96,
                                       bits == 2u ? 1u : 2u,
                                       uint32_t(v.output_size),
                                       uint32_t(v.input_size),
                                       uint32_t(v.n_words_row),
                                       uint32_t(v.n_learners),
                                       1,
                                       32,
                                       128,
                                       bits};
            for (size_t i = 0; i < 10; ++i) {
                i2ce_ci::put(encoded.data() + 8 + i * 4, fields[i], 4);
            }
            i2ce_ci::put(encoded.data() + 48, bytes, 8);
            i2ce_ci::put(encoded.data() + 56, id, 8);
            i2ce_ci::put(encoded.data() + 64,
                         i2ce_ci::hash_bytes(i2ce_ci::seed, payload, bytes), 8);
            i2ce_ci::put(encoded.data() + 72, UINT64_C(0x314b385249415032), 8);
            const auto path = out / i2ce_ci::filename(id, bits);
            if (std::filesystem::exists(path)) {
                throw std::runtime_error("duplicate source identity");
            }
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char *>(encoded.data()), encoded.size());
            file.close();
            if (!file) {
                throw std::runtime_error("write failed");
            }
            std::vector<uint8_t> checked(bytes);
            i2ce_ci::load(path.string(), id, v.n_words_row, v.output_size, v.input_size,
                          checked.data(), checked.size(), bits, v.n_learners);
            std::cout << v.name << '\t' << entries << '\t' << bytes << '\t'
                      << path.filename().string() << '\n';
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
