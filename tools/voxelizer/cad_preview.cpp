#include "voxelizer.h"
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRep_Tool.hxx>
#include <Poly_Triangulation.hxx>
#include <Standard_Failure.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using namespace omfdm::voxel;

int runPreview(const std::vector<std::string>& args) {
    if (args.size() < 2 || args.size() > 3)
        throw std::runtime_error("Usage: MagCadPreview input.step NEW_OUTPUT.json [deflection_mm=0.05]");
    const fs::path input = fs::u8path(args[0]), output = fs::u8path(args[1]);
    double deflection = .05;
    if (args.size() == 3) {
        std::size_t end = 0;
        deflection = std::stod(args[2], &end);
        if (end != args[2].size()) throw std::runtime_error("Invalid deflection");
    }
    if (!std::isfinite(deflection) || deflection < 1e-5)
        throw std::runtime_error("Deflection must be finite and >= 1e-5 mm");
    if (fs::exists(output)) throw std::runtime_error("Output already exists");
    auto model = readStep(input);
    auto document = describeModel(model);
    document["preview_deflection_mm"] = deflection;
    document["preview_angular_deflection_rad"] = .35;
    std::uint64_t triangles = 0;
    for (std::size_t i = 0; i < model.solids.size(); ++i) {
        BRepMesh_IncrementalMesh mesher(model.solids[i].shape, deflection, false, .35, false);
        if (!mesher.IsDone()) throw std::runtime_error("CAD display tessellation failed");
        std::vector<double> positions;
        for (TopExp_Explorer ex(model.solids[i].shape, TopAbs_FACE); ex.More(); ex.Next()) {
            const auto face = TopoDS::Face(ex.Current());
            TopLoc_Location location;
            const auto mesh = BRep_Tool::Triangulation(face, location);
            if (mesh.IsNull()) throw std::runtime_error("A CAD face has no triangulation");
            triangles += mesh->NbTriangles();
            if (triangles > 2000000) throw std::runtime_error("CAD preview exceeds 2 million triangles");
            for (int t = 1; t <= mesh->NbTriangles(); ++t) {
                int a, b, c;
                mesh->Triangle(t).Get(a, b, c);
                if (face.Orientation() == TopAbs_REVERSED) std::swap(b, c);
                for (const int v : {a, b, c}) {
                    const auto p = mesh->Node(v).Transformed(location.Transformation());
                    positions.insert(positions.end(), {p.X() * .001, p.Y() * .001, p.Z() * .001});
                }
            }
        }
        document["solids"][i]["positions"] = std::move(positions);
    }
    document["preview_triangles"] = triangles;
    std::ofstream stream;
    stream.exceptions(std::ios::badbit | std::ios::failbit);
    stream.open(output, std::ios::binary);
    stream << document.dump();
    stream.close();
    std::cout << triangles << " CAD preview triangles\n";
    return 0;
}
int guardedPreview(const std::vector<std::string>& args) {
    try { return runPreview(args); }
    catch (const Standard_Failure& e) { std::cerr << e.GetMessageString() << "\n"; }
    catch (const std::exception& e) { std::cerr << e.what() << "\n"; }
    return 1;
}
#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) args.push_back(fs::path(argv[i]).u8string());
    return guardedPreview(args);
}
#else
int main(int argc, char** argv) {
    return guardedPreview(std::vector<std::string>(argv + 1, argv + argc));
}
#endif
