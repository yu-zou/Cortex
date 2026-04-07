#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>

namespace cortex {
namespace test {

inline std::vector<float> load_float_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg(); f.seekg(0);
    std::vector<float> data(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

inline std::vector<uint8_t> load_uint8_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg(); f.seekg(0);
    std::vector<uint8_t> data(bytes);
    f.read(reinterpret_cast<char*>(data.data()), bytes);
    return data;
}

}
}
