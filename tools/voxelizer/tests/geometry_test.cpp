#include "voxelizer.h"
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRep_Builder.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Writer.hxx>
#include <StepData_StepModel.hxx>
#include <TopLoc_Location.hxx>
#include <TopoDS_Compound.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Pln.hxx>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace omfdm::voxel;
namespace fs = std::filesystem;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F operation, const char* message) {
    try { operation(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}
TopoDS_Shape group(const std::vector<TopoDS_Shape>& shapes) {
    BRep_Builder builder;
    TopoDS_Compound c;
    builder.MakeCompound(c);
    for (const auto& shape : shapes) builder.Add(c, shape);
    return c;
}
void step(const fs::path& path, const TopoDS_Shape& shape, double unit = 1.0) {
    STEPControl_Writer writer;
    require(Interface_Static::SetCVal("write.step.unit",
        unit == 25.4 ? "INCH" : unit == 1000 ? "M" : "MM"), "STEP output unit parameter");
    writer.Model()->SetWriteLengthUnit(unit);
    require(writer.Transfer(shape, STEPControl_AsIs) == IFSelect_RetDone, "STEP transfer failed");
    require(writer.Write(path.string().c_str()) == IFSelect_RetDone, "STEP write failed");
}
std::uint64_t occupied(const Grid& g) {
    std::uint64_t result = 0;
    for (auto n : g.solidCellCounts) result += n;
    return result;
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "fixture directory argument required");
        const fs::path dir(argv[1]);
        fs::create_directories(dir);
        const TopoDS_Shape box = BRepPrimAPI_MakeBox(2, 3, 4).Shape();
        step(dir / "box.step", box);
        step(dir / "box_inch.step", box, 25.4);
        step(dir / "box_meter.step", box, 1000);
        const auto m = readStep(dir / "box.step");
        require(m.solids.size() == 1, "box solid count");
        Options o;
        auto g = voxelize(m, o);
        require(g.size == std::array<std::uint64_t,3>{4,5,6}, "box grid dimensions");
        require(occupied(g) == 24, "box occupancy");
        require(surfaceFaceCount(g) == 52, "box exposed faces");
        for (const char* name : {"box_inch.step", "box_meter.step"}) {
            const auto units = readStep(dir / name);
            require(std::abs(units.bounds.max[0] - 2) < 1e-8, "STEP unit conversion");
            require(voxelize(units, o).material == g.material, "unit-independent occupancy");
        }
        o.spacingMm = {1, .5, 2};
        o.padding = 0;
        auto anisotropic = voxelize(m, o);
        require(anisotropic.size == std::array<std::uint64_t,3>{2,6,2}, "anisotropic dimensions");
        require(occupied(anisotropic) == 24, "anisotropic occupancy");

        const TopoDS_Shape hollow = BRepAlgoAPI_Cut(
            BRepPrimAPI_MakeBox(4,4,4).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(1,1,1),2,2,2).Shape()).Shape();
        step(dir / "hollow.step", hollow);
        o.spacingMm = {1,1,1};
        const auto hg = voxelize(readStep(dir / "hollow.step"), o);
        require(occupied(hg) == 56, "hollow interior must remain air");
        require(hg.material[hg.index(1,1,1)] == 0, "cavity cell");
        require(surfaceFaceCount(hg) == 120, "cavity surface faces");

        const TopoDS_Shape cylinder = BRepPrimAPI_MakeCylinder(5,4).Shape();
        step(dir / "cylinder.step", cylinder);
        const auto cm = readStep(dir / "cylinder.step");
        const double exact = 100 * std::acos(-1.0);
        o.spacingMm = {2,2,2};
        const double coarse = double(occupied(voxelize(cm,o))) * 8;
        o.spacingMm = {.5,.5,.5};
        const double fine = double(occupied(voxelize(cm,o))) * .125;
        require(std::abs(fine-exact) < std::abs(coarse-exact), "cylinder volume refinement");
        require(std::abs(fine-exact) / exact < .02, "fine cylinder volume error");
        std::cout << "Cylinder relative volume errors: "
                  << (coarse-exact)/exact << ", " << (fine-exact)/exact << "\n";

        const auto part = BRepPrimAPI_MakeBox(1,2,1).Shape();
        gp_Trsf move;
        move.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)), std::acos(-1.0)/2);
        move.SetTranslationPart(gp_Vec(4,0,0));
        const auto assembly = group({part, part.Moved(TopLoc_Location(move))});
        step(dir / "assembly.step", assembly);
        const auto am = readStep(dir / "assembly.step");
        require(am.solids.size() == 2, "placed instances must stay separate");
        require(std::abs(am.bounds.max[0] - 4) < 1e-8, "instance transform");
        o.spacingMm = {1,1,1};
        auto ag = voxelize(am, o);
        require(ag.solidCellCounts == std::vector<std::uint64_t>{2,2}, "instance cell counts");
        require(ag.solid[ag.index(0,1,0)] == 1 && ag.solid[ag.index(3,0,0)] == 2,
                "instance placement and x-fast order");

        const auto overlap = group({BRepPrimAPI_MakeBox(2,2,2).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(1,0,0),2,2,2).Shape()});
        step(dir / "overlap.step", overlap);
        rejects([&] { voxelize(readStep(dir / "overlap.step"),o); }, "overlap must reject");
        const auto face = BRepBuilderAPI_MakeFace(gp_Pln(gp_Pnt(0,0,0),gp_Dir(0,0,1)),0,2,0,2).Shape();
        step(dir / "sheet.step", face);
        rejects([&] { readStep(dir / "sheet.step"); }, "sheet must reject");
        step(dir / "mixed.step", group({box, face}));
        rejects([&] { readStep(dir / "mixed.step"); }, "loose geometry must not disappear");

        // Adjacent solids share a center on their common face: deterministic first-ID tie.
        const auto touching = modelFromShape(group({
            BRepPrimAPI_MakeBox(1,2,2).Shape(),
            BRepPrimAPI_MakeBox(gp_Pnt(1,0,0),1,2,2).Shape()}));
        o.spacingMm = {2,2,2};
        const auto tg = voxelize(touching,o);
        require(tg.material.size() == 1 && tg.solid[0] == 1, "shared ON boundary policy");
        require(tg.boundarySamples == 2, "ON diagnostics");

        // Small, intentionally simplified multi-body model for the first 3D workflow.
        const auto stator = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(12,6).Shape(),
                                          BRepPrimAPI_MakeCylinder(10,6).Shape()).Shape();
        const auto rotor = BRepAlgoAPI_Cut(BRepPrimAPI_MakeCylinder(6,6).Shape(),
                                         BRepPrimAPI_MakeCylinder(3,6).Shape()).Shape();
        const auto shaft = BRepPrimAPI_MakeCylinder(
            gp_Ax2(gp_Pnt(0,0,-2),gp_Dir(0,0,1)),2.5,10).Shape();
        std::vector<TopoDS_Shape> demo{stator, rotor, shaft};
        const auto magnet = BRepPrimAPI_MakeBox(gp_Pnt(6.5,-1,0),2,2,6).Shape();
        for (int i = 0; i < 4; ++i) {
            gp_Trsf rotation;
            rotation.SetRotation(gp_Ax1(gp_Pnt(0,0,0),gp_Dir(0,0,1)),
                                 i * std::acos(-1.0) / 2);
            demo.push_back(magnet.Moved(TopLoc_Location(rotation)));
        }
        step(dir / "rotor_demo.step", group(demo));
        const auto dm = readStep(dir / "rotor_demo.step");
        o = Options{};
        o.spacingMm = {.5,.5,.5};
        require(dm.solids.size() == 7, "demo solids");
        const auto dg = voxelize(dm, o);
        for (auto count : dg.solidCellCounts) require(count > 0, "demo body must be represented");

        o = Options{};
        o.maxCells = 1;
        rejects([&] { voxelize(m,o); }, "allocation guard");
        o = Options{};
        o.spacingMm[0] = std::numeric_limits<double>::infinity();
        rejects([&] { voxelize(m,o); }, "nonfinite spacing");
        o = Options{};
        o.padding = std::numeric_limits<std::uint64_t>::max();
        rejects([&] { voxelize(m,o); }, "padding overflow");
        std::cout << "Geometry, units, cavity, transforms, refinement, rejection tests passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
