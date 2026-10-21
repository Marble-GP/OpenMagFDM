#include "voxelizer.h"

#include <BRepBndLib.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepClass3d_SolidClassifier.hxx>
#include <BRepGProp.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <GProp_GProps.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Version.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Iterator.hxx>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace omfdm::voxel {
namespace {
Bounds boundsOf(const TopoDS_Shape& shape) {
    Bnd_Box box;
    // Geometry bounds, independent of display triangulation and mesh deflection.
    BRepBndLib::AddOptimal(shape, box, false, false);
    if (box.IsVoid() || box.IsOpen())
        throw std::runtime_error("CAD shape has empty or unbounded geometry");
    Bounds b;
    box.Get(b.min[0], b.min[1], b.min[2], b.max[0], b.max[1], b.max[2]);
    for (int a = 0; a < 3; ++a)
        if (!std::isfinite(b.min[a]) || !std::isfinite(b.max[a]) || b.max[a] <= b.min[a])
            throw std::runtime_error("CAD shape has invalid or zero-volume bounds");
    return b;
}

// Do not silently discard loose shells/faces/wires next to otherwise valid solids.
void requireOnlySolids(const TopoDS_Shape& shape) {
    if (shape.IsNull()) throw std::runtime_error("Empty CAD shape");
    if (shape.ShapeType() == TopAbs_SOLID) return;
    if (shape.ShapeType() != TopAbs_COMPOUND && shape.ShapeType() != TopAbs_COMPSOLID)
        throw std::runtime_error("Only closed solids are supported; loose faces/shells/wires found");
    bool any = false;
    for (TopoDS_Iterator it(shape); it.More(); it.Next()) {
        any = true;
        requireOnlySolids(it.Value());
    }
    if (!any) throw std::runtime_error("Empty CAD compound");
}

std::uint64_t dimension(double length, double spacing, std::uint64_t padding,
                        std::uint64_t limit, double tolerance) {
    const double ratio = length / spacing;
    // STEP decimal serialization / unit conversion can perturb exact dimensions.
    // Snap only within classification tolerance (and only to a positive count).
    const double rounded = std::round(ratio);
    const double cells = rounded >= 1 && std::abs(ratio - rounded) <=
        std::max(tolerance / spacing,
                 32 * std::numeric_limits<double>::epsilon() * std::max(1.0, ratio))
        ? rounded : std::ceil(ratio);
    if (!std::isfinite(cells) || cells < 1 || cells > static_cast<double>(limit) ||
        padding > limit / 2)
        throw std::runtime_error("Grid dimension exceeds --max-cells (or is not representable)");
    const auto n = static_cast<std::uint64_t>(cells);
    if (n > limit - 2 * padding)
        throw std::runtime_error("Padded grid exceeds --max-cells");
    return n + 2 * padding;
}
Vec3 meters(const Vec3& v) { return {v[0] * .001, v[1] * .001, v[2] * .001}; }
} // namespace

Model modelFromShape(const TopoDS_Shape& shape) {
    requireOnlySolids(shape);
    TopTools_IndexedMapOfShape shapes;
    // Location participates in identity, so translated/rotated instances are kept.
    TopExp::MapShapes(shape, TopAbs_SOLID, shapes);
    if (shapes.IsEmpty()) throw std::runtime_error("No solid bodies in CAD input");
    Model model;
    model.bounds = boundsOf(shape);
    for (int i = 1; i <= shapes.Extent(); ++i) {
        const auto& s = shapes(i);
        if (!BRepCheck_Analyzer(s).IsValid())
            throw std::runtime_error("Invalid BRep solid " + std::to_string(i));
        for (TopExp_Explorer ex(s, TopAbs_SHELL); ex.More(); ex.Next())
            if (!BRep_Tool::IsClosed(ex.Current()))
                throw std::runtime_error("Open shell in solid " + std::to_string(i));
        BRepClass3d_SolidClassifier classifier(s);
        classifier.PerformInfinitePoint(1e-7);
        if (classifier.State() != TopAbs_OUT)
            throw std::runtime_error("Inverted or unbounded solid " + std::to_string(i));
        GProp_GProps props;
        BRepGProp::VolumeProperties(s, props);
        if (!(props.Mass() > 0) || !std::isfinite(props.Mass()))
            throw std::runtime_error("Non-positive volume in solid " + std::to_string(i));
        model.solids.push_back({s, boundsOf(s), props.Mass(), static_cast<std::uint32_t>(i)});
    }
    return model;
}

Model readStep(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext != ".step" && ext != ".stp")
        throw std::runtime_error("Input must be .step or .stp (closed CAD solids)");
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("STEP input is not a regular file");
    STEPControl_Reader reader;
    // Use a C++ stream so Windows paths are not passed through OCCT's narrow file API.
    std::ifstream input(path, std::ios::binary);
    if (!input || reader.ReadStream("input.step", input) != IFSelect_RetDone)
        throw std::runtime_error("Failed to read STEP input");
    // OCCT's unit factor is relative to millimeters. Set AFTER ReadFile/ReadStream.
    reader.SetSystemLengthUnit(1.0);
    const int roots = reader.NbRootsForTransfer();
    if (roots < 1) throw std::runtime_error("STEP input has no transferable roots");
    for (int i = 1; i <= roots; ++i)
        if (!reader.TransferRoot(i))
            throw std::runtime_error("Failed to transfer STEP root " + std::to_string(i));
    return modelFromShape(reader.OneShape());
}

void applyMaterialMap(Model& model, const std::filesystem::path& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open material map");
    nlohmann::json map;
    stream >> map;
    if (!map.is_object() || map.size() != 1 || !map.contains("solid_material_ids") ||
        !map["solid_material_ids"].is_array() ||
        map["solid_material_ids"].size() != model.solids.size())
        throw std::runtime_error("Material map must contain only solid_material_ids, one ID per solid");
    for (std::size_t i = 0; i < model.solids.size(); ++i) {
        const auto& v = map["solid_material_ids"][i];
        if (!v.is_number_integer() || (v.is_number_integer() && !v.is_number_unsigned() &&
            v.get<std::int64_t>() <= 0) || v.get<std::uint64_t>() == 0 ||
            v.get<std::uint64_t>() > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Material IDs must be integers in [1, 4294967295]; 0 is air");
        model.solids[i].materialId = v.get<std::uint32_t>();
    }
}

Grid voxelize(const Model& model, const Options& o,
              const std::function<void(std::size_t, std::size_t)>& progress) {
    if (model.solids.empty()) throw std::runtime_error("Cannot voxelize an empty model");
    if (!std::isfinite(o.toleranceMm) || o.toleranceMm < 1e-7)
        throw std::runtime_error("Classification tolerance must be finite and >= 1e-7 mm");
    // Keep all integer-to-double dimension conversions exact and allocations bounded.
    if (!o.maxCells || o.maxCells > (std::uint64_t{1} << 40))
        throw std::runtime_error("--max-cells must be in [1, 1099511627776]");
    Grid g;
    g.spacingMm = o.spacingMm;
    std::uint64_t count = 1;
    for (int a = 0; a < 3; ++a) {
        if (!std::isfinite(o.spacingMm[a]) || o.spacingMm[a] <= 2 * o.toleranceMm)
            throw std::runtime_error("Spacing must be finite and greater than twice the tolerance");
        g.size[a] = dimension(model.bounds.max[a] - model.bounds.min[a],
                              o.spacingMm[a], o.padding, o.maxCells, o.toleranceMm);
        if (g.size[a] > std::numeric_limits<std::int32_t>::max())
            throw std::runtime_error("Grid dimension exceeds VTK ImageData extent limit");
        if (g.size[a] > o.maxCells / count)
            throw std::runtime_error("Grid exceeds --max-cells; increase spacing or the explicit limit");
        count *= g.size[a];
        g.originMm[a] = model.bounds.min[a] - static_cast<double>(o.padding) * o.spacingMm[a];
        const double end = g.originMm[a] + static_cast<double>(g.size[a]) * o.spacingMm[a];
        if (!std::isfinite(g.originMm[a]) || !std::isfinite(end) ||
            g.originMm[a] + o.spacingMm[a] == g.originMm[a] ||
            end - o.spacingMm[a] == end)
            throw std::runtime_error("Grid coordinates cannot be represented at this spacing");
    }
    if (count > std::numeric_limits<std::size_t>::max() / 9)
        throw std::runtime_error("Grid exceeds addressable memory");
    g.material.assign(static_cast<std::size_t>(count), 0);
    g.solid.assign(static_cast<std::size_t>(count), 0);
    std::vector<std::uint8_t> interior(static_cast<std::size_t>(count), 0);
    g.solidCellCounts.resize(model.solids.size(), 0);
    for (std::size_t s = 0; s < model.solids.size(); ++s) {
        const auto& body = model.solids[s];
        BRepClass3d_SolidClassifier classifier(body.shape);
        std::array<std::uint64_t, 3> lo{}, hi{};
        for (int a = 0; a < 3; ++a) {
            const double first = std::ceil((body.bounds.min[a] - o.toleranceMm -
                                           g.originMm[a]) / g.spacingMm[a] - .5);
            const double last = std::floor((body.bounds.max[a] + o.toleranceMm -
                                           g.originMm[a]) / g.spacingMm[a] - .5) + 1;
            lo[a] = static_cast<std::uint64_t>(std::clamp(first, 0.0, double(g.size[a])));
            hi[a] = static_cast<std::uint64_t>(std::clamp(last, 0.0, double(g.size[a])));
        }
        for (auto z = lo[2]; z < hi[2]; ++z)
            for (auto y = lo[1]; y < hi[1]; ++y)
                for (auto x = lo[0]; x < hi[0]; ++x) {
                    const gp_Pnt p(g.originMm[0] + (double(x) + .5) * g.spacingMm[0],
                                   g.originMm[1] + (double(y) + .5) * g.spacingMm[1],
                                   g.originMm[2] + (double(z) + .5) * g.spacingMm[2]);
                    classifier.Perform(p, o.toleranceMm);
                    const auto state = classifier.State();
                    if (state == TopAbs_OUT) continue;
                    if (state != TopAbs_IN && state != TopAbs_ON)
                        throw std::runtime_error("Indeterminate CAD point classification");
                    const auto n = g.index(x, y, z);
                    if (state == TopAbs_ON) ++g.boundarySamples;
                    if (state == TopAbs_IN && interior[n])
                        throw std::runtime_error("Overlapping solid interiors at cell (" +
                            std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) +
                            "): solids " + std::to_string(g.solid[n]) + " and " + std::to_string(s + 1));
                    // Interior wins over ON; shared ON boundary chooses the first solid.
                    if (!g.solid[n] || (state == TopAbs_IN && !interior[n])) {
                        g.solid[n] = static_cast<std::uint32_t>(s + 1);
                        g.material[n] = body.materialId;
                        interior[n] = state == TopAbs_IN;
                    }
                }
        if (progress) progress(s + 1, model.solids.size());
    }
    for (auto id : g.solid)
        if (id) ++g.solidCellCounts[id - 1];
    if (std::all_of(g.solidCellCounts.begin(), g.solidCellCounts.end(),
                    [](auto n) { return n == 0; }))
        throw std::runtime_error("No cell centers inside the CAD model; use smaller spacing");
    return g;
}

nlohmann::json describeModel(const Model& model) {
    nlohmann::json j = {
        {"schema", "openmagfdm.cad-model/1"},
        {"length_unit", "m"},
        {"occt_version", OCC_VERSION_COMPLETE},
        {"bounds_m", {{"min", meters(model.bounds.min)}, {"max", meters(model.bounds.max)}}},
        {"solids", nlohmann::json::array()}
    };
    for (std::size_t i = 0; i < model.solids.size(); ++i) {
        const auto& s = model.solids[i];
        j["solids"].push_back({{"solid_id", i + 1}, {"material_id", s.materialId},
            {"cad_volume_m3", s.volumeMm3 * 1e-9},
            {"bounds_m", {{"min", meters(s.bounds.min)}, {"max", meters(s.bounds.max)}}}});
    }
    return j;
}

nlohmann::json describeGrid(const Model& model, const Grid& g, const Options& o) {
    auto j = describeModel(model);
    j["schema"] = "openmagfdm.voxel-grid/1";
    j["tool_version"] = "0.1.0";
    j["sampling"] = "cell-center";
    j["index_order"] = "x + nx * (y + ny * z)";
    j["origin_m"] = meters(g.originMm);
    j["spacing_m"] = meters(g.spacingMm);
    j["dimensions_cells"] = g.size;
    j["cell_count"] = g.material.size();
    j["grid_working_array_bytes"] = std::uint64_t(g.material.size()) * 9;
    j["padding_cells"] = o.padding;
    j["classification_tolerance_m"] = o.toleranceMm * .001;
    j["boundary_sample_hits"] = g.boundarySamples;
    j["air_material_id"] = 0;
    j["overlap_policy"] = "reject sampled IN/IN; IN wins over ON; first solid wins ON/ON";
    j["warnings"] = nlohmann::json::array({
        "Cell-center sampling can miss sub-voxel features and overlaps between sample points.",
        "Material IDs are labels; CAD material properties and names are not imported.",
        "Solid IDs depend on this STEP transfer order; preserve this manifest with the source."
    });
    std::uint64_t occupied = 0;
    const double cellVolume = g.spacingMm[0] * g.spacingMm[1] * g.spacingMm[2] * 1e-9;
    for (std::size_t i = 0; i < model.solids.size(); ++i) {
        const auto count = g.solidCellCounts[i];
        occupied += count;
        const auto volume = double(count) * cellVolume;
        j["solids"][i]["voxel_cells"] = count;
        j["solids"][i]["voxel_volume_m3"] = volume;
        j["solids"][i]["relative_volume_error"] =
            (volume - model.solids[i].volumeMm3 * 1e-9) / (model.solids[i].volumeMm3 * 1e-9);
        if (!count) j["warnings"].push_back("Solid " + std::to_string(i + 1) +
                                          " has no sampled cells; decrease spacing.");
    }
    j["occupied_cells"] = occupied;
    j["air_cells"] = g.material.size() - occupied;
    j["files"] = {{"volume", "model.vti"}};
    if (o.surface) {
        j["files"]["surface"] = "surface.vtp";
        j["surface_description"] = "exposed solid-to-air faces only; outward quads; no internal interfaces";
    }
    return j;
}
} // namespace omfdm::voxel
