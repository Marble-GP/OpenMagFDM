#pragma once
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <array>
#include <yaml-cpp/yaml.h>
#include "json.hpp"
#include "third_party/picosha2/picosha2.h"

namespace omfdm {
inline std::string inputDigest(const std::filesystem::path& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot read snapshot input: " + filename.string());
    picosha2::hash256_one_by_one hash;
    std::array<char, 65536> buffer{};
    while (input) { input.read(buffer.data(), buffer.size()); hash.process(buffer.begin(), buffer.begin() + input.gcount()); }
    if (!input.eof()) throw std::runtime_error("Snapshot input read failed");
    hash.finish(); return picosha2::get_hash_hex_string(hash);
}
inline void saveInputSnapshot(const std::string& output, std::string& configPath,
                              std::string& imagePath, const std::string& version) {
    namespace fs = std::filesystem;
    using nlohmann::json;
    const auto root = fs::absolute(output);
    const auto snapshot = root / ".inputs";
    const auto manifestPath = root / "run-inputs.json";
    // WebUI/MCP already froze these inputs at queue acceptance. Verify, do not overwrite.
    if (fs::absolute(configPath).lexically_normal() == (snapshot / "config.yaml").lexically_normal()) {
        std::ifstream stream(manifestPath); json manifest; stream >> manifest;
        if (manifest.at("effectiveConfigHash").get<std::string>() != inputDigest(configPath))
            throw std::runtime_error("Saved configuration hash mismatch");
        for (const auto& item : manifest.at("files")) {
            const auto relative = item.at("path").get<std::string>();
            const fs::path rel(relative);
            if (rel.is_absolute() || relative.find("..") != std::string::npos)
                throw std::runtime_error("Invalid snapshot file path");
            if (inputDigest(root / rel) != item.at("sha256").get<std::string>())
                throw std::runtime_error("Saved input hash mismatch");
        }
        std::ofstream stamp(root / "solver-version.txt"); stamp << version << '\n';
        if (!stamp) throw std::runtime_error("Cannot write solver version");
        return;
    }
    if (fs::exists(snapshot) || fs::exists(manifestPath))
        throw std::runtime_error("Input snapshot already exists; choose a new result directory");
    fs::create_directory(snapshot);
    json manifest = {{"version", 1}, {"solverVersion", version}, {"files", json::array()}};
    auto copy = [&](const fs::path& source, const fs::path& destination) {
        if (!fs::is_regular_file(source)) throw std::runtime_error("Snapshot source is not a regular file");
        fs::copy_file(source, destination, fs::copy_options::none);
        manifest["files"].push_back({{"id", source.filename().string()},
            {"path", fs::relative(destination, root).generic_string()},
            {"size", fs::file_size(destination)}, {"sha256", inputDigest(destination)}});
    };
    copy(configPath, snapshot / "source-config.yaml");
    YAML::Node config = YAML::LoadFile((snapshot / "source-config.yaml").string());
    std::size_t visited = 0;
    std::function<void(YAML::Node, unsigned)> visit = [&](YAML::Node node, unsigned depth) {
        if (depth > 64 || ++visited > 50000) throw std::runtime_error("Configuration nesting limit exceeded");
        if (node.IsMap()) for (auto entry : node) {
            const auto key = entry.first.as<std::string>();
            if ((key == "initial_az_path" || key == "value_profile" || key == "gamma_profile") && entry.second.IsScalar()) {
                const auto source = entry.second.as<std::string>();
                if (!source.empty()) {
                    const auto destination = snapshot / ("reference-" + std::to_string(manifest["files"].size()) + ".bin");
                    copy(source, destination); node[key] = fs::relative(destination, root).generic_string();
                }
            } else visit(entry.second, depth + 1);
        }
        else if (node.IsSequence()) for (auto entry : node) visit(entry, depth + 1);
    };
    visit(config, 0);
    const auto savedImage = snapshot / ("model" + fs::path(imagePath).extension().string());
    copy(imagePath, savedImage);
    const auto savedConfig = snapshot / "config.yaml";
    { std::ofstream stream(savedConfig); stream << config; if (!stream) throw std::runtime_error("Cannot save effective configuration"); }
    manifest["effectiveConfigHash"] = inputDigest(savedConfig);
    { std::ofstream stream(manifestPath); stream << manifest.dump(2); if (!stream) throw std::runtime_error("Cannot save input manifest"); }
    configPath = savedConfig.string(); imagePath = savedImage.string();
}
}
