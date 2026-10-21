#include "voxelizer.h"
#include <Standard_Failure.hxx>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omfdm::voxel;
namespace fs = std::filesystem;
namespace {
void help() {
    std::cout <<
R"(MagVoxelizer 0.1.0 - STEP solid to Cartesian voxel grid

  MagVoxelizer --input model.step --output NEW_DIRECTORY --pitch-mm 0.5
  MagVoxelizer --input model.stp --output NEW_DIRECTORY --spacing-mm 0.5 0.5 1
  MagVoxelizer --input model.step --inspect

Options:
  --input PATH              STEP (.step/.stp), with closed solid bodies
  --output PATH             New directory; existing paths are never overwritten
  --pitch-mm N              Isotropic cell size in millimeters (required for conversion)
  --spacing-mm DX DY DZ     Alternative anisotropic sizes in millimeters
  --padding N               Air padding cells on each side (default 1)
  --material-map PATH       JSON: {"solid_material_ids":[1,2,...]}; 0 reserved for air
  --max-cells N             Allocation guard (default 20000000, about 180 MB grid arrays)
  --max-surface-faces N     Surface output guard (default 2000000)
  --no-surface              Write volume only, omit exterior quad surface
  --inspect                 Print solid IDs, bounds, and CAD volumes as JSON; no output files
  --help                    Show this help
  --version                 Show tool version

Output: model.vti (material_id + solid_id), surface.vtp (exterior quads),
        manifest.json (units, grid, per-solid volume errors, warnings).
All output coordinates are METERS. Input STEP length units are converted automatically.
Cell-center sampling can miss thin features. Sampled solid interior overlaps are errors.
)";
}
double real(const std::string& text) {
    std::size_t end = 0;
    const double value = std::stod(text, &end);
    if (end != text.size() || !std::isfinite(value) || value <= 0)
        throw std::runtime_error("Expected a positive finite number: " + text);
    return value;
}
std::uint64_t integer(const std::string& text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected a non-negative integer: " + text);
    std::size_t end = 0;
    const auto n = std::stoull(text, &end);
    if (end != text.size()) throw std::runtime_error("Invalid integer: " + text);
    return n;
}

int run(const std::vector<std::string>& args) {
    Options options;
    fs::path input, output, map;
    bool inspect = false, spacing = false;
    std::set<std::string> seen;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto key = args[i];
        if (!seen.insert(key).second) throw std::runtime_error("Duplicate option: " + key);
        auto next = [&]() -> const std::string& {
            if (++i >= args.size()) throw std::runtime_error("Missing value for " + key);
            return args[i];
        };
        if (key == "--help") { help(); return 0; }
        if (key == "--version") { std::cout << "MagVoxelizer 0.1.0\n"; return 0; }
        if (key == "--input") input = fs::u8path(next());
        else if (key == "--output") output = fs::u8path(next());
        else if (key == "--material-map") map = fs::u8path(next());
        else if (key == "--pitch-mm" || key == "--spacing-mm") {
            if (spacing) throw std::runtime_error("Use either --pitch-mm or --spacing-mm, not both");
            spacing = true;
            options.spacingMm[0] = real(next());
            options.spacingMm[1] = key == "--pitch-mm" ? options.spacingMm[0] : real(next());
            options.spacingMm[2] = key == "--pitch-mm" ? options.spacingMm[0] : real(next());
        } else if (key == "--padding") options.padding = integer(next());
        else if (key == "--max-cells") options.maxCells = integer(next());
        else if (key == "--max-surface-faces") options.maxSurfaceFaces = integer(next());
        else if (key == "--no-surface") options.surface = false;
        else if (key == "--inspect") inspect = true;
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (input.empty()) throw std::runtime_error("--input is required; see --help");
    if (!inspect && (output.empty() || !spacing))
        throw std::runtime_error("Conversion requires --output and --pitch-mm or --spacing-mm");
    if (inspect && (spacing || !output.empty() || seen.count("--padding") ||
        seen.count("--max-cells") || seen.count("--no-surface") || seen.count("--max-surface-faces")))
        throw std::runtime_error("--inspect accepts only --input and optional --material-map");
    if (!inspect && fs::exists(output))
        throw std::runtime_error("Output path already exists; choose a new directory");
    const auto start = std::chrono::steady_clock::now();
    std::cerr << "Reading STEP solids...\n";
    auto model = readStep(input);
    if (!map.empty()) applyMaterialMap(model, map);
    if (inspect) {
        std::cout << describeModel(model).dump(2) << "\n";
        return 0;
    }
    const auto grid = voxelize(model, options, [](auto done, auto total) {
        std::cerr << "Classified solid " << done << "/" << total << "\n";
    });
    const auto faces = options.surface ? surfaceFaceCount(grid) : 0;
    if (faces > options.maxSurfaceFaces)
        throw std::runtime_error("Surface exceeds --max-surface-faces; use --no-surface or increase limit");
    auto manifest = describeGrid(model, grid, options);
    manifest["source_file"] = fs::absolute(input).u8string();
    manifest["surface_faces"] = faces;
    manifest["surface_payload_bytes"] = faces * 144;
    // Create only after validation/classification. Directory creation is exclusive.
    // manifest.json is the completion marker, written last; failed I/O leaves no valid marker.
    if (!fs::create_directory(output))
        throw std::runtime_error("Could not create a new output directory");
    writeVti(output / "model.vti", grid);
    if (options.surface) writeVtp(output / "surface.vtp", grid, faces);
    manifest["elapsed_seconds"] =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto temporaryManifest = output / "manifest.json.tmp";
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(temporaryManifest, std::ios::binary);
    out << manifest.dump(2) << "\n";
    out.close();
    fs::rename(temporaryManifest, output / "manifest.json");
    std::cout << "Created " << output.u8string() << "\n"
              << "Grid: " << grid.size[0] << " x " << grid.size[1] << " x " << grid.size[2]
              << "; occupied cells: " << manifest["occupied_cells"]
              << "; exterior quads: " << faces << "\n";
    for (const auto& warning : manifest["warnings"])
        std::cerr << "Warning: " << warning.get<std::string>() << "\n";
    return 0;
}
int guarded(const std::vector<std::string>& args) {
    try { return run(args); }
    catch (const Standard_Failure& e) {
        std::cerr << "CAD error: " << e.GetMessageString() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 1;
}
} // namespace

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(fs::path(argv[i]).u8string());
    return guarded(args);
}
#else
int main(int argc, char** argv) {
    return guarded(std::vector<std::string>(argv + 1, argv + argc));
}
#endif
