#include "transformer_layers/softmax.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    struct Case { std::size_t length; uint8_t input; uint8_t expected; };
    for (const auto item : {Case{4, 0, 4}, Case{512, 128, 0}, Case{256, 0, 1}}) {
        for (const bool rearranged : {false, true}) {
            std::vector<uint32_t> words(item.length * item.length / 4);
            auto* bytes = reinterpret_cast<uint8_t*>(words.data());
            std::fill(bytes, bytes + item.length * item.length, item.input);
            Softmax softmax;
            if (rearranged) softmax.computeRearranged(words.data(), item.length, 4);
            else softmax.compute(words.data(), item.length);
            assert(std::all_of(bytes, bytes + item.length * item.length,
                              [item](uint8_t value) { return value == item.expected; }));
        }
    }
}
