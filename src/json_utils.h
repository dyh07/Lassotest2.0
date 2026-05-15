#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <fstream>

using json = nlohmann::json;

namespace JsonUtils {

inline json readJsonFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    json j;
    f >> j;
    return j;
}

inline void writeJsonFile(const std::string& path, const json& j) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot write JSON file: " + path);
    }
    f << j.dump(2);
}

} // namespace JsonUtils
