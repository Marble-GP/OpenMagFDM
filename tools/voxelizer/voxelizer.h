#pragma once

#include <TopoDS_Shape.hxx>
#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>
#include "json.hpp"

namespace omfdm::voxel {
using Vec3 = std::array<double, 3>;
struct Bounds {
    Vec3 min{}, max{};
};
struct Solid {
    TopoDS_Shape shape;
    Bounds bounds;
    double volumeMm3 = 0;
    std::uint32_t materialId = 0;
};
struct Model {
    std::vector<Solid> solids;
    Bounds bounds;
};
struct Options {
    Vec3 spacingMm{1, 1, 1};
    std::uint64_t padding = 1;
    std::uint64_t maxCells = 20000000;
    std::uint64_t maxSurfaceFaces = 2000000;
    double toleranceMm = 1e-7;
    bool surface = true;
};
struct Grid {
    Vec3 originMm{}, spacingMm{};
    std::array<std::uint64_t, 3> size{};
    std::vector<std::uint32_t> material, solid;
    std::vector<std::uint64_t> solidCellCounts;
    std::uint64_t boundarySamples = 0;
    std::uint64_t index(std::uint64_t x, std::uint64_t y, std::uint64_t z) const {
        return x + size[0] * (y + size[1] * z);
    }
};

Model readStep(const std::filesystem::path& path);
Model modelFromShape(const TopoDS_Shape& shape);
void applyMaterialMap(Model& model, const std::filesystem::path& path);
Grid voxelize(const Model& model, const Options& options,
              const std::function<void(std::size_t, std::size_t)>& progress = {});
nlohmann::json describeModel(const Model& model);
nlohmann::json describeGrid(const Model& model, const Grid& grid, const Options& options);
std::uint64_t surfaceFaceCount(const Grid& grid);
void writeVti(const std::filesystem::path& path, const Grid& grid);
void writeVtp(const std::filesystem::path& path, const Grid& grid, std::uint64_t faces);
} // namespace omfdm::voxel
