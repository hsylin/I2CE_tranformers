#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

// Required files fail before inference; optional notebook lookup may fall back.
inline bool tryLoadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        return false;
    }

    for (int i = 0; i < size; i++) {
        if (!(fin >> array[i])) {
            std::cout << "Weight file size mismatch: " << filename
                      << " (expected " << size << " words)" << std::endl;
            fin.close();
            return false;
        }
    }

    fin >> std::ws;
    if (!fin.eof()) {
        std::cout << "Weight file has extra data: " << filename
                  << " (expected exactly " << size << " words)" << std::endl;
        fin.close();
        return false;
    }

    fin.close();
    return true;
}

inline void loadWeight(int n_head, int qkv, int size, uint32_t *array, const std::string &dir_name) {
    std::string filename = dir_name + "/H" + std::to_string(n_head) + "_L" +
            std::to_string(qkv) + ".bin";
    if (!tryLoadWeight(n_head, qkv, size, array, dir_name)) {
        throw std::runtime_error("Cannot load complete weight/input file: " + filename);
    }
}

