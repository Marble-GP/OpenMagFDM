#pragma once
#include <array>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include "json.hpp"

namespace omfdm::d3 {
namespace fs = std::filesystem;
using Json = nlohmann::json;
using Vec = std::array<double, 3>;
using Index = std::array<int, 3>;
void require(bool condition, const std::string& message);
std::string digest(const fs::path& path);
Json readJson(const fs::path& path);
void writeJson(const fs::path& path, const Json& value);
struct Grid {
    Index size{};
    Vec origin{}, spacing{};
    std::vector<uint32_t> material, solid;
    std::string hash;
    size_t index(int x, int y, int z) const;
    Index position(size_t i) const;
    size_t count() const;
};
Grid readGrid(const fs::path& path, size_t maxCells = 2000000);
void writeGrid(const fs::path& path, const Grid& grid,
               const std::map<std::string, std::vector<double>>& scalars,
               const std::vector<Vec>& vectors = {});
struct FaceKey {
    int axis = 0;
    Index index{};
    bool operator<(const FaceKey& other) const;
};
struct SelectedFace { FaceKey key; int sign = 1; };
std::vector<SelectedFace> readFaces(const fs::path& path, const Grid& grid);
void writeFaces(const fs::path& path, const Grid& grid, const std::vector<SelectedFace>& faces);
void writeSurface(const fs::path& path, const Grid& grid, const std::vector<SelectedFace>& faces);
struct Region { std::set<uint32_t> solids; std::string material; };
struct Coil {
    std::string name, model;
    std::vector<std::string> regions;
    double current = 0, turns = 1, uniformityTolerance = .05;
    fs::path inlet, outlet, cut;
};
struct Config {
    fs::path source, voxels, manifest;
    YAML::Node yaml;
    std::map<std::string, YAML::Node> materials;
    std::map<std::string, Region> regions;
    std::vector<Coil> coils;
    std::map<std::string, double> variables;
    int step = 0, threads = 0;
    size_t maxCells = 2000000;
    double number(const YAML::Node& node) const;
};
Config readConfig(const fs::path& path, int step = 0);
struct Face {
    FaceKey key;
    int a = -1, b = -1; // lower and upper active-cell indices; -1 = exterior
    double area = 0, distance = 0, conductance = 0;
    int terminal = 0, cutSign = 0; // terminal: 1 inlet, -1 outlet
};
struct Domain {
    std::vector<size_t> cells;
    std::vector<double> sigma;
    std::vector<Face> faces;
    std::vector<SelectedFace> selected;
    bool closed = false;
};
Domain makeDomain(const Config& config, const Grid& grid, const Coil& coil);
std::vector<SelectedFace> selectFaces(const Config& config, const Grid& grid, const Coil& coil,
    const Vec& point, const Vec& normal, const Vec& lower, const Vec& upper, double tolerance, bool cut);
struct Result {
    std::vector<double> potential, flux;
    std::vector<Vec> density;
    Json diagnostics;
    bool accepted = false;
};
Result prepareCurrent(const Grid& grid, const Domain& domain, const Coil& coil, bool direct = false);
void saveResult(const fs::path& folder, const Grid& grid, const Domain& domain, const Result& result);
Config snapshot(const Config& config, const fs::path& folder);
} // namespace omfdm::d3
