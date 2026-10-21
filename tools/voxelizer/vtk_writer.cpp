#include "voxelizer.h"
#include <cstring>
#include <fstream>
#include <iomanip>
#include <locale>
#include <stdexcept>

namespace omfdm::voxel {
namespace {
// Raw-appended VTK XML; UInt64 block lengths, explicitly little-endian data.
void u32(std::ostream& out, std::uint32_t value) {
    char b[4];
    for (int i = 0; i < 4; ++i) b[i] = static_cast<char>(value >> (8 * i));
    out.write(b, 4);
}
void u64(std::ostream& out, std::uint64_t value) {
    char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<char>(value >> (8 * i));
    out.write(b, 8);
}
void f64(std::ostream& out, double value) {
    static_assert(sizeof(double) == 8, "VTK Float64 requires an 8-byte double");
    std::uint64_t bits;
    std::memcpy(&bits, &value, 8);
    u64(out, bits);
}
std::ofstream output(const std::filesystem::path& path) {
    std::ofstream out;
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out.open(path, std::ios::binary);
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    return out;
}
void array(std::ostream& out, const char* type, const char* name,
           std::uint64_t offset, int components = 1) {
    out << "<DataArray type=\"" << type << "\" Name=\"" << name
        << "\" NumberOfComponents=\"" << components
        << "\" format=\"appended\" offset=\"" << offset << "\"/>\n";
}
void start(std::ostream& out, const char* type) {
    out << "<?xml version=\"1.0\"?>\n<VTKFile type=\"" << type
        << "\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
}
void extent(std::ostream& out, const Grid& g) {
    out << "0 " << g.size[0] << " 0 " << g.size[1] << " 0 " << g.size[2];
}
// Vertex order: right-handed outward normals for -X,+X,-Y,+Y,-Z,+Z.
constexpr int corners[6][4][3] = {
    {{0,0,0},{0,0,1},{0,1,1},{0,1,0}},
    {{1,0,0},{1,1,0},{1,1,1},{1,0,1}},
    {{0,0,0},{1,0,0},{1,0,1},{0,0,1}},
    {{0,1,0},{0,1,1},{1,1,1},{1,1,0}},
    {{0,0,0},{0,1,0},{1,1,0},{1,0,0}},
    {{0,0,1},{1,0,1},{1,1,1},{0,1,1}}
};
template<class F> void eachFace(const Grid& g, F visit) {
    for (std::uint64_t z = 0; z < g.size[2]; ++z)
        for (std::uint64_t y = 0; y < g.size[1]; ++y)
            for (std::uint64_t x = 0; x < g.size[0]; ++x) {
                const auto n = g.index(x, y, z);
                if (!g.material[n]) continue;
                const bool exposed[6] = {
                    x == 0 || !g.material[n - 1],
                    x + 1 == g.size[0] || !g.material[n + 1],
                    y == 0 || !g.material[n - g.size[0]],
                    y + 1 == g.size[1] || !g.material[n + g.size[0]],
                    z == 0 || !g.material[n - g.size[0] * g.size[1]],
                    z + 1 == g.size[2] || !g.material[n + g.size[0] * g.size[1]]
                };
                for (int f = 0; f < 6; ++f)
                    if (exposed[f]) visit(x, y, z, n, f);
            }
}
} // namespace

void writeVti(const std::filesystem::path& path, const Grid& g) {
    auto out = output(path);
    const auto bytes = std::uint64_t(g.material.size()) * 4;
    start(out, "ImageData");
    out << "<ImageData WholeExtent=\""; extent(out, g);
    out << "\" Origin=\"" << g.originMm[0] * .001 << " " << g.originMm[1] * .001
        << " " << g.originMm[2] * .001 << "\" Spacing=\""
        << g.spacingMm[0] * .001 << " " << g.spacingMm[1] * .001 << " "
        << g.spacingMm[2] * .001 << "\">\n<Piece Extent=\"";
    extent(out, g);
    out << "\">\n<PointData/>\n<CellData Scalars=\"material_id\">\n";
    array(out, "UInt32", "material_id", 0);
    array(out, "UInt32", "solid_id", bytes + 8);
    out << "</CellData>\n</Piece>\n</ImageData>\n<AppendedData encoding=\"raw\">\n_";
    u64(out, bytes);
    for (auto value : g.material) u32(out, value);
    u64(out, bytes);
    for (auto value : g.solid) u32(out, value);
    out << "\n</AppendedData>\n</VTKFile>\n";
    out.close();
}

std::uint64_t surfaceFaceCount(const Grid& g) {
    std::uint64_t count = 0;
    eachFace(g, [&](auto, auto, auto, auto, int) { ++count; });
    return count;
}

void writeVtp(const std::filesystem::path& path, const Grid& g, std::uint64_t faces) {
    auto out = output(path);
    const std::uint64_t pointsBytes = faces * 4 * 3 * 8;
    const std::uint64_t connectivityBytes = faces * 4 * 8;
    const std::uint64_t offsetsBytes = faces * 8;
    const std::uint64_t idsBytes = faces * 4;
    start(out, "PolyData");
    out << "<PolyData><Piece NumberOfPoints=\"" << faces * 4
        << "\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\""
        << faces << "\">\n<PointData/>\n<CellData Scalars=\"material_id\">\n";
    const auto idOffset = pointsBytes + connectivityBytes + offsetsBytes + 24;
    array(out, "UInt32", "material_id", idOffset);
    array(out, "UInt32", "solid_id", idOffset + idsBytes + 8);
    out << "</CellData>\n<Points>\n";
    array(out, "Float64", "Points", 0, 3);
    out << "</Points>\n<Polys>\n";
    array(out, "Int64", "connectivity", pointsBytes + 8);
    array(out, "Int64", "offsets", pointsBytes + connectivityBytes + 16);
    out << "</Polys>\n</Piece></PolyData>\n<AppendedData encoding=\"raw\">\n_";
    u64(out, pointsBytes);
    eachFace(g, [&](auto x, auto y, auto z, auto, int face) {
        const std::uint64_t cell[3] = {x, y, z};
        for (int v = 0; v < 4; ++v)
            for (int a = 0; a < 3; ++a)
                f64(out, (g.originMm[a] + double(cell[a] + corners[face][v][a]) *
                          g.spacingMm[a]) * .001);
    });
    u64(out, connectivityBytes);
    for (std::uint64_t i = 0; i < faces * 4; ++i) u64(out, i);
    u64(out, offsetsBytes);
    for (std::uint64_t i = 1; i <= faces; ++i) u64(out, i * 4);
    u64(out, idsBytes);
    eachFace(g, [&](auto, auto, auto, auto n, int) { u32(out, g.material[n]); });
    u64(out, idsBytes);
    eachFace(g, [&](auto, auto, auto, auto n, int) { u32(out, g.solid[n]); });
    out << "\n</AppendedData>\n</VTKFile>\n";
    out.close();
}
} // namespace omfdm::voxel
