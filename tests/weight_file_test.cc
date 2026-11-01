#include "transformer_layers/weight_file.h"
#include <array>
#include <cassert>

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string directory = argv[1];
    const auto filename = directory + "/H-1_L-1.bin";
    std::array<uint32_t, 3> output{};
    assert(!tryLoadWeight(-1, -1, 3, output.data(), directory));
    for (const std::string contents : {"", "1 2", "1 2 bad", "1 2 3 4", "1 2 3 bad"}) {
        std::ofstream(filename) << contents;
        assert(!tryLoadWeight(-1, -1, 3, output.data(), directory));
        bool rejected = false;
        try { loadWeight(-1, -1, 3, output.data(), directory); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    std::ofstream(filename) << "1 2 4294967295\n\t ";
    loadWeight(-1, -1, 3, output.data(), directory);
    assert((output == std::array<uint32_t, 3>{1, 2, UINT32_MAX}));
    std::remove(filename.c_str());
    bool rejected = false;
    try { loadWeight(-1, -1, 3, output.data(), directory); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
