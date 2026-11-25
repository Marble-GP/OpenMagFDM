#include "FemEngine.h"

#include "MagneticFieldAnalyzer.h"

#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace omfdm {
namespace fem {

namespace fs = std::filesystem;

namespace {

static double mu0() { return 4.0 * M_PI * 1e-7; }

static std::string engineName(const YAML::Node& c) {
    if (c["solver"] && c["solver"]["engine"]) return c["solver"]["engine"].as<std::string>();
    return "fdm";
}

static bool pointInTri(double px, double py, const FemMesh& m, int ti, double& l0, double& l1, double& l2) {
    const auto& t = m.triangles[ti];
    const auto& a = m.vertices[t.v[0]].p;
    const auto& b = m.vertices[t.v[1]].p;
    const auto& c = m.vertices[t.v[2]].p;
    double denom = (b.y - c.y)*(a.x - c.x) + (c.x - b.x)*(a.y - c.y);
    if (std::fabs(denom) < 1e-30) return false;
    l0 = ((b.y - c.y)*(px - c.x) + (c.x - b.x)*(py - c.y)) / denom;
    l1 = ((c.y - a.y)*(px - c.x) + (a.x - c.x)*(py - c.y)) / denom;
    l2 = 1.0 - l0 - l1;
    return l0 >= -1e-12 && l1 >= -1e-12 && l2 >= -1e-12;
}

static int findTri(const FemMesh& m, double px, double py) {
    for (int i = 0; i < (int)m.triangles.size(); ++i) {
        double l0,l1,l2;
        if (pointInTri(px,py,m,i,l0,l1,l2)) return i;
    }
    return -1;
}

static YAML::Node mergeYaml(const YAML::Node& base, const YAML::Node& over) {
    if (!base.IsDefined()) return YAML::Clone(over);
    if (!over.IsDefined()) return YAML::Clone(base);
    if (base.IsMap() && over.IsMap()) {
        YAML::Node out = YAML::Clone(base);
        for (auto it = over.begin(); it != over.end(); ++it) {
            const std::string key = it->first.as<std::string>();
            out[key] = mergeYaml(base[key], it->second);
        }
        return out;
    }
    return YAML::Clone(over);
}

static bool readFiniteDouble(const YAML::Node& n, double& value) {
    if (!n.IsDefined() || !n.IsScalar()) return false;
    try { value = n.as<double>(); }
    catch (...) { return false; }
    return std::isfinite(value);
}

static double polygonSignedArea(const std::vector<Point2D>& points) {
    double area = 0.0;
    if (points.size() < 3) return 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& a = points[i];
        const auto& b = points[(i + 1) % points.size()];
        area += a.x * b.y - b.x * a.y;
    }
    return 0.5 * area;
}

static void triangleGradients(const Point2D& p0, const Point2D& p1, const Point2D& p2,
                              double g[3][2], double& area, double& signedArea) {
    signedArea = 0.5 * ((p1.x - p0.x) * (p2.y - p0.y) -
                        (p2.x - p0.x) * (p1.y - p0.y));
    area = std::fabs(signedArea);
    if (area < 1e-30) {
        for (int i = 0; i < 3; ++i) { g[i][0] = 0.0; g[i][1] = 0.0; }
        return;
    }
    const double inv2A = 1.0 / (2.0 * signedArea);
    g[0][0] = (p1.y - p2.y) * inv2A; g[0][1] = (p2.x - p1.x) * inv2A;
    g[1][0] = (p2.y - p0.y) * inv2A; g[1][1] = (p0.x - p2.x) * inv2A;
    g[2][0] = (p0.y - p1.y) * inv2A; g[2][1] = (p1.x - p0.x) * inv2A;
}

} // anon

FemEngine::FemEngine(const std::string& configPath, const std::string& imagePath)
    : configPath_(configPath), imagePath_(imagePath) {}

bool FemEngine::parseConfig(std::string& err) {
    try { config_ = YAML::LoadFile(configPath_); }
    catch (const std::exception& e) { err = std::string("config load failed: ") + e.what(); return false; }

    // mesh
    nx_ = 0; ny_ = 0; dx_ = 1.0; dy_ = 1.0;
    if (config_["mesh"]) {
        if (config_["mesh"]["dx"]) dx_ = config_["mesh"]["dx"].as<double>();
        if (config_["mesh"]["dy"]) dy_ = config_["mesh"]["dy"].as<double>();
    }
    if (!std::isfinite(dx_) || !std::isfinite(dy_) || dx_ <= 0.0 || dy_ <= 0.0) {
        err = "FEM_UNSUPPORTED: mesh dx/dy must be finite and positive";
        return false;
    }
    // polar rejection
    std::string coord = config_["coordinate_system"] ? config_["coordinate_system"].as<std::string>() : "cartesian";
    if (coord != "cartesian") { err = "FEM_UNSUPPORTED: polar coordinates not supported in FEM mode"; return false; }

    // materials (presets and numeric properties follow the FDM configuration contract)
    matData_.clear();
    std::map<std::string, YAML::Node> presets;
    if (config_["material_presets"]) {
        for (auto it = config_["material_presets"].begin(); it != config_["material_presets"].end(); ++it)
            presets[it->first.as<std::string>()] = it->second;
    }
    if (config_["materials"]) {
        for (auto it = config_["materials"].begin(); it != config_["materials"].end(); ++it) {
            MatData md;
            md.name = it->first.as<std::string>();
            YAML::Node props = it->second;
            if (props["preset"]) {
                const std::string presetName = props["preset"].as<std::string>();
                auto pit = presets.find(presetName);
                if (pit == presets.end()) {
                    err = "FEM_UNSUPPORTED: unknown material preset '" + presetName + "'";
                    return false;
                }
                props = mergeYaml(pit->second, props);
            }
            if (props["rgb"]) {
                const YAML::Node& r = props["rgb"];
                if (!r.IsSequence() || r.size() < 3) {
                    err = "FEM_UNSUPPORTED: material '" + md.name + "' rgb must contain three values";
                    return false;
                }
                double vr = 0, vg = 0, vb = 0;
                if (!readFiniteDouble(r[0], vr) || !readFiniteDouble(r[1], vg) || !readFiniteDouble(r[2], vb) ||
                    vr < 0 || vr > 255 || vg < 0 || vg > 255 || vb < 0 || vb > 255) {
                    err = "FEM_UNSUPPORTED: material '" + md.name + "' has invalid rgb";
                    return false;
                }
                md.rgb = {(uint8_t)std::lround(vr), (uint8_t)std::lround(vg), (uint8_t)std::lround(vb)};
            }
            if (props["mu_r"] && !readFiniteDouble(props["mu_r"], md.mu_r)) {
                err = "FEM_UNSUPPORTED: formula mu_r is not supported for material '" + md.name + "'";
                return false;
            }
            if (props["jz"] && !readFiniteDouble(props["jz"], md.jz)) {
                err = "FEM_UNSUPPORTED: formula jz is not supported for material '" + md.name + "'";
                return false;
            }
            if (md.mu_r <= 0.0) md.mu_r = 1.0;
            if (props["B-H"]) {
                const YAML::Node& bh = props["B-H"];
                if (!bh.IsSequence() || bh.size() != 2 || !bh[0].IsSequence() || !bh[1].IsSequence()) {
                    err = "FEM_INVALID_BH: material '" + md.name + "' must use [[H,...],[B,...]]";
                    return false;
                }
                for (auto v : bh[0]) {
                    double x = 0.0;
                    if (!readFiniteDouble(v, x)) { err = "FEM_INVALID_BH: non-finite H value"; return false; }
                    md.bhH.push_back(x);
                }
                for (auto v : bh[1]) {
                    double x = 0.0;
                    if (!readFiniteDouble(v, x)) { err = "FEM_INVALID_BH: non-finite B value"; return false; }
                    md.bhB.push_back(x);
                }
                if (md.bhH.size() < 2 || md.bhH.size() != md.bhB.size()) {
                    err = "FEM_INVALID_BH: material '" + md.name + "' needs at least two matching H/B points";
                    return false;
                }
                for (size_t k = 0; k < md.bhB.size(); ++k) {
                    if (md.bhH[k] < -1e-15 || (k > 0 && (md.bhB[k] <= md.bhB[k-1] || md.bhH[k] < md.bhH[k-1]))) {
                        err = "FEM_INVALID_BH: material '" + md.name + "' requires increasing B and nondecreasing nonnegative H";
                        return false;
                    }
                }
            }
            if (props["magnetization"]) {
                const YAML::Node& mg = props["magnetization"];
                const std::string pattern = mg["pattern"] ? mg["pattern"].as<std::string>() : "parallel";
                if (pattern != "parallel") {
                    err = "FEM_UNSUPPORTED: magnetization pattern '" + pattern + "' is not supported";
                    return false;
                }
                md.hasMagnetization = true;
                if (mg["Br"]) {
                    if (!readFiniteDouble(mg["Br"], md.Br)) { err = "FEM_UNSUPPORTED: invalid magnetization Br"; return false; }
                } else if (mg["Hc"]) {
                    double hc = 0.0;
                    if (!readFiniteDouble(mg["Hc"], hc)) { err = "FEM_UNSUPPORTED: invalid magnetization Hc"; return false; }
                    md.Br = hc * mu0();
                }
                double angleDeg = 0.0;
                if (mg["angle"] && !readFiniteDouble(mg["angle"], angleDeg)) {
                    err = "FEM_UNSUPPORTED: invalid magnetization angle";
                    return false;
                }
                md.magAngle = angleDeg * M_PI / 180.0;
            }
            matData_.push_back(md);
        }
    }
    // ensure at least air
    if (matData_.empty()) { MatData md; md.name = "air"; matData_.push_back(md); }

    // mesh opts
    meshOpts_ = MeshBuildOptions();
    if (config_["solver"] && config_["solver"]["fem"] && config_["solver"]["fem"]["mesh"]) {
        const auto& mm = config_["solver"]["fem"]["mesh"];
        if (mm["boundary_tolerance"]) meshOpts_.boundaryTolerance = mm["boundary_tolerance"].as<double>();
        if (mm["min_spacing"]) meshOpts_.minSpacing = mm["min_spacing"].as<double>();
        if (mm["max_spacing"]) meshOpts_.maxSpacing = mm["max_spacing"].as<double>();
        if (mm["boundary_attraction"]) meshOpts_.boundaryAttraction = mm["boundary_attraction"].as<double>();
        if (mm["min_angle_deg"]) meshOpts_.minAngleDeg = mm["min_angle_deg"].as<double>();
        if (mm["max_nodes"]) meshOpts_.maxNodes = mm["max_nodes"].as<int>();
        if (mm["max_growth_ratio"]) meshOpts_.maxGrowthRatio = mm["max_growth_ratio"].as<double>();
    }
    if (meshOpts_.minSpacing <= 0) meshOpts_.minSpacing = dx_;
    if (meshOpts_.maxSpacing <= 0) meshOpts_.maxSpacing = 10.0 * dx_;
    if (meshOpts_.boundaryTolerance <= 0) meshOpts_.boundaryTolerance = 0.5 * dx_;

    // transient rejection
    if (config_["transient"] && config_["transient"]["enabled"]) {
        bool en = false;
        try { en = config_["transient"]["enabled"].as<bool>(); } catch (...) {}
        if (en) { err = "FEM_UNSUPPORTED: transient analysis not supported in FEM mode"; return false; }
    }
    if (config_["domain_decomposition"] && config_["domain_decomposition"]["enabled"]) {
        bool en = false;
        try { en = config_["domain_decomposition"]["enabled"].as<bool>(); } catch (...) {}
        if (en) { err = "FEM_UNSUPPORTED: domain decomposition not supported in FEM mode"; return false; }
    }
    if (config_["boundary_conditions"]) {
        const auto& bc = config_["boundary_conditions"];
        for (const char* side : {"left", "right", "bottom", "top"}) {
            if (!bc[side]) continue;
            const std::string type = bc[side]["type"] ? bc[side]["type"].as<std::string>() : "dirichlet";
            if (type != "dirichlet" && type != "neumann" && type != "robin" && type != "periodic") {
                err = std::string("FEM_UNSUPPORTED: unknown boundary condition '") + type + "'";
                return false;
            }
            if (type == "dirichlet" && bc[side]["value"]) {
                double value = 0.0;
                if (!readFiniteDouble(bc[side]["value"], value)) { err = "FEM_UNSUPPORTED: invalid Dirichlet value"; return false; }
            }
            if (type == "robin") {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                if (!readFiniteDouble(bc[side]["alpha"], alpha) || !readFiniteDouble(bc[side]["beta"], beta) ||
                    !readFiniteDouble(bc[side]["gamma"], gamma) || std::fabs(beta) < 1e-30) {
                    err = "FEM_UNSUPPORTED: robin requires finite alpha,beta,gamma with beta != 0";
                    return false;
                }
            }
        }
    }
    parseNonlinearConfig();
    // refinement config
    refinementEnabled_ = false; refinementMaxRounds_ = 8; refinementTheta_ = 0.5; refinementTarget_ = 1e-3;
    if (config_["solver"] && config_["solver"]["fem"] && config_["solver"]["fem"]["refinement"]) {
        const auto& r = config_["solver"]["fem"]["refinement"];
        if (r["enabled"]) refinementEnabled_ = r["enabled"].as<bool>();
        if (r["max_rounds"]) refinementMaxRounds_ = r["max_rounds"].as<int>();
        if (r["theta"]) refinementTheta_ = r["theta"].as<double>();
        if (r["target_relative_error"]) refinementTarget_ = r["target_relative_error"].as<double>();
    }
    if (nonlinearMaxIter_ < 1 || nonlinearTol_ <= 0.0 || lineSearchMax_ < 1 ||
        lineSearchRho_ <= 0.0 || lineSearchRho_ >= 1.0 || relaxation_ <= 0.0 || relaxation_ > 1.0) {
        err = "FEM_UNSUPPORTED: invalid nonlinear solver parameters";
        return false;
    }
    if (refinementMaxRounds_ < 0 || refinementTheta_ <= 0.0 || refinementTheta_ > 1.0 || refinementTarget_ < 0.0 ||
        meshOpts_.maxNodes < 3 || meshOpts_.minAngleDeg < 0.0 || meshOpts_.minAngleDeg > 180.0 ||
        meshOpts_.maxGrowthRatio < 1.0 || meshOpts_.maxSpacing < meshOpts_.minSpacing) {
        err = "FEM_UNSUPPORTED: invalid FEM mesh/refinement parameters";
        return false;
    }
    return true;
}

bool FemEngine::loadImage(std::string& err) {
    imageRaw_ = cv::imread(imagePath_, cv::IMREAD_COLOR);
    if (imageRaw_.empty()) { err = "FEM_UNSUPPORTED: cannot load image"; return false; }
    cv::cvtColor(imageRaw_, imageFlippedRgb_, cv::COLOR_BGR2RGB);
    cv::flip(imageFlippedRgb_, imageFlippedRgb_, 0);
    nx_ = imageFlippedRgb_.cols;
    ny_ = imageFlippedRgb_.rows;
    // materials_ for MeshBuilder (MaterialDesc)
    materials_.clear();
    for (const auto& md : matData_) {
        MaterialDesc m; m.name = md.name; m.rgb = md.rgb; m.antiAliasing = false;
        materials_.push_back(m);
    }
    if (materials_.empty()) { MaterialDesc m; m.name = "air"; m.rgb = {255,255,255}; materials_.push_back(m); }
    return true;
}

bool FemEngine::buildMesh(std::string& err) {
    try {
        mesh_ = MeshBuilder::build(imageRaw_, materials_, dx_, dy_, meshOpts_, extraInteriorPoints_);
    } catch (const std::exception& e) { err = std::string("mesh build failed: ") + e.what(); return false; }
    if (!mesh_.quality.passed) { err = "FEM_MESH_FAILED: " + mesh_.quality.failureReason; return false; }
    if (mesh_.vertices.size() > (size_t)meshOpts_.maxNodes) { err = "FEM_MESH_TOO_LARGE"; return false; }
    // mask and vertex incidence (shared by export, forces and refinement)
    mask_ = MeshBuilder::buildMask(imageFlippedRgb_, materials_);
    vertexTriangles_.assign(mesh_.vertices.size(), {});
    for (int ti = 0; ti < (int)mesh_.triangles.size(); ++ti) {
        for (int k = 0; k < 3; ++k)
            vertexTriangles_[mesh_.triangles[ti].v[k]].push_back(ti);
    }
    buildTriangleIndex();
    return true;
}

void FemEngine::buildTriangleIndex() {
    const double Lx = (nx_-1)*dx_, Ly = (ny_-1)*dy_;
    triangleCell_ = std::max({meshOpts_.maxSpacing, dx_, dy_, 1e-30});
    gridNx_ = std::max(1, (int)std::ceil(Lx / triangleCell_) + 1);
    gridNy_ = std::max(1, (int)std::ceil(Ly / triangleCell_) + 1);
    triangleGrid_.assign((size_t)gridNx_ * gridNy_, {});
    for (int ti = 0; ti < (int)mesh_.triangles.size(); ++ti) {
        const auto& t = mesh_.triangles[ti];
        double minx = 1e300, maxx = -1e300, miny = 1e300, maxy = -1e300;
        for (int k = 0; k < 3; ++k) {
            const auto& p = mesh_.vertices[t.v[k]].p;
            minx = std::min(minx, p.x); maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y); maxy = std::max(maxy, p.y);
        }
        const int ix0 = std::max(0, (int)std::floor(minx / triangleCell_));
        const int ix1 = std::min(gridNx_-1, (int)std::floor(maxx / triangleCell_));
        const int iy0 = std::max(0, (int)std::floor(miny / triangleCell_));
        const int iy1 = std::min(gridNy_-1, (int)std::floor(maxy / triangleCell_));
        for (int iy = iy0; iy <= iy1; ++iy)
            for (int ix = ix0; ix <= ix1; ++ix)
                triangleGrid_[(size_t)iy * gridNx_ + ix].push_back(ti);
    }
}

int FemEngine::findTriangle(double x, double y) const {
    if (!triangleGrid_.empty()) {
        const int ix = std::max(0, std::min(gridNx_-1, (int)std::floor(x / triangleCell_)));
        const int iy = std::max(0, std::min(gridNy_-1, (int)std::floor(y / triangleCell_)));
        for (int ti : triangleGrid_[(size_t)iy * gridNx_ + ix]) {
            double l0,l1,l2;
            if (pointInTri(x,y,mesh_,ti,l0,l1,l2)) return ti;
        }
    }
    return findTri(mesh_, x, y);
}

std::map<int,double> FemEngine::dirichletNodes() const {
    std::map<int,double> fixed;
    if (mesh_.vertices.empty()) return fixed;
    const double Lx = (nx_ - 1) * dx_, Ly = (ny_ - 1) * dy_;
    const double tol = 1e-9 * std::max({std::fabs(dx_), std::fabs(dy_), 1e-12});
    auto onSide = [&](int vi, const std::string& side) {
        const auto& p = mesh_.vertices[vi].p;
        if (side == "left")   return std::fabs(p.x) < tol;
        if (side == "right")  return std::fabs(p.x - Lx) < tol;
        if (side == "bottom") return std::fabs(p.y) < tol;
        if (side == "top")    return std::fabs(p.y - Ly) < tol;
        return false;
    };
    const char* sides[] = {"left", "right", "bottom", "top"};
    for (const char* side : sides) {
        std::string type = "dirichlet";
        double value = 0.0;
        if (config_["boundary_conditions"] && config_["boundary_conditions"][side]) {
            const auto& bc = config_["boundary_conditions"][side];
            type = bc["type"] ? bc["type"].as<std::string>() : "dirichlet";
            if (type == "dirichlet" && bc["value"]) value = bc["value"].as<double>();
        }
        if (type != "dirichlet") continue;
        for (int vi = 0; vi < (int)mesh_.vertices.size(); ++vi)
            if (onSide(vi, side)) fixed[vi] = value;
    }
    if (fixed.empty()) fixed[0] = 0.0; // gauge for pure Neumann / periodic domains
    return fixed;
}

std::vector<int> FemEngine::periodicDofRoots(const std::map<int,double>& fixed,
                                             std::vector<int>& freeNodes,
                                             std::vector<int>& signs) const {
    const int N = (int)mesh_.vertices.size();
    std::vector<int> roots(N, -1);
    signs.assign(N, 0);
    freeNodes.clear();
    for (int vi = 0; vi < N; ++vi) {
        if (fixed.count(vi)) continue;
        roots[vi] = (int)freeNodes.size();
        signs[vi] = 1;
        freeNodes.push_back(vi);
    }
    return roots;
}

void FemEngine::assemble() {
    const int N = (int)mesh_.vertices.size();
    K_.resize(N, N);
    std::vector<Eigen::Triplet<double>> trips;
    rhs_.setZero(N);
    elemMu_.clear();
    elemMu_.reserve(mesh_.triangles.size());

    // Element stiffness evaluated at the current A_z. For B-H materials this
    // makes K_ * A_ - rhs_ the actual nonlinear residual, not a linear surrogate.
    for (const auto& tri : mesh_.triangles) {
        const int vi[3] = {tri.v[0], tri.v[1], tri.v[2]};
        const auto& p0 = mesh_.vertices[vi[0]].p;
        const auto& p1 = mesh_.vertices[vi[1]].p;
        const auto& p2 = mesh_.vertices[vi[2]].p;
        double g[3][2], area = 0.0, signedArea = 0.0;
        triangleGradients(p0, p1, p2, g, area, signedArea);
        const int mid = tri.material;
        if (area < 1e-30) { elemMu_.push_back(mu0()); continue; }

        double Bx = 0.0, By = 0.0;
        for (int a = 0; a < 3; ++a) {
            Bx += A_[vi[a]] * g[a][1];
            By += -A_[vi[a]] * g[a][0];
        }
        const double Bmag = std::hypot(Bx, By);
        const double nu = evalNu(Bmag, mid);
        elemMu_.push_back(1.0 / nu);
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                trips.push_back(Eigen::Triplet<double>(vi[a], vi[b],
                    nu * area * (g[a][0]*g[b][0] + g[a][1]*g[b][1])));

        const double matJz = (mid >= 0 && mid < (int)matData_.size()) ? matData_[mid].jz : 0.0;
        const double jz = sourceFn_ ? sourceFn_((p0.x+p1.x+p2.x)/3.0, (p0.y+p1.y+p2.y)/3.0) : matJz;
        if (jz != 0.0)
            for (int a = 0; a < 3; ++a) rhs_[vi[a]] -= jz * area / 3.0;
    }

    // Prescribed permanent-magnet surface currents. Induced magnetization is
    // already represented by nu(|B|) in the constitutive matrix above.
    std::map<std::pair<long long,long long>, int> vertexIndex;
    for (int vi = 0; vi < N; ++vi) {
        const auto& p = mesh_.vertices[vi].p;
        vertexIndex[{std::llround(p.x * 1e9), std::llround(p.y * 1e9)}] = vi;
    }
    for (const auto& c : mesh_.contours) {
        const int mid = c.materialInside;
        if (mid < 0 || mid >= (int)matData_.size() || !matData_[mid].hasMagnetization) continue;
        const double Mx = (matData_[mid].Br / mu0()) * std::cos(matData_[mid].magAngle);
        const double My = (matData_[mid].Br / mu0()) * std::sin(matData_[mid].magAngle);
        const size_t n = c.points.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& p0 = c.points[i];
            const auto& p1 = c.points[(i + 1) % n];
            const double dxs = p1.x - p0.x, dys = p1.y - p0.y;
            const double len = std::hypot(dxs, dys);
            if (len < 1e-30) continue;
            // Contours are normalized to outer CCW / hole CW. The right normal
            // therefore points from material to air for both cases.
            const double nx = dys / len, ny = -dxs / len;
            const double Kz = Mx * ny - My * nx;
            auto findVertex = [&](const Point2D& p) {
                auto it = vertexIndex.find({std::llround(p.x * 1e9), std::llround(p.y * 1e9)});
                return it == vertexIndex.end() ? -1 : it->second;
            };
            const int vi0 = findVertex(p0), vi1 = findVertex(p1);
            if (vi0 < 0 || vi1 < 0) continue;
            const double contrib = Kz * len * 0.5;
            rhs_[vi0] -= contrib;
            rhs_[vi1] -= contrib;
        }
    }

    // Robin natural BC: P1 mass matrix len/6 * [[2,1],[1,2]].
    for (const auto& re : robinEdges_) {
        const auto& pi = mesh_.vertices[re.i].p;
        const auto& pj = mesh_.vertices[re.j].p;
        const double len = std::hypot(pj.x-pi.x, pj.y-pi.y);
        if (len < 1e-30 || std::fabs(re.beta) < 1e-30) continue;
        const int mx = std::max(0, std::min(nx_-1, (int)std::llround(((pi.x+pj.x)*0.5)/dx_)));
        const int my = std::max(0, std::min(ny_-1, (int)std::llround(((pi.y+pj.y)*0.5)/dy_)));
        const int mid = (mx + my*nx_ < (int)mask_.size()) ? mask_[my*nx_+mx] : -1;
        const double nu = evalNu(0.0, mid);
        const double c = nu * (re.alpha / re.beta) * len / 6.0;
        trips.push_back(Eigen::Triplet<double>(re.i, re.i, 2.0*c));
        trips.push_back(Eigen::Triplet<double>(re.j, re.j, 2.0*c));
        trips.push_back(Eigen::Triplet<double>(re.i, re.j, c));
        trips.push_back(Eigen::Triplet<double>(re.j, re.i, c));
        const double f = nu * (re.gamma / re.beta) * len * 0.5;
        rhs_[re.i] += f;
        rhs_[re.j] += f;
    }

    // Periodic/antiperiodic constraints via a high-weight symmetric penalty.
    for (size_t k = 0; k < periodicPairs_.size(); ++k) {
        const auto& pr = periodicPairs_[k];
        const int sign = (k < periodicSigns_.size()) ? periodicSigns_[k] : 1;
        const double P = 1e10;
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.first, P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.second, P));
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.second, -P*sign));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.first, -P*sign));
    }
    K_.setFromTriplets(trips.begin(), trips.end());

    nodeMu_.assign(N, mu0());
    for (int vi = 0; vi < N; ++vi) {
        double sum = 0.0;
        int count = 0;
        for (int ti : vertexTriangles_[vi]) {
            if (ti >= 0 && ti < (int)elemMu_.size()) { sum += elemMu_[ti]; ++count; }
        }
        if (count > 0) nodeMu_[vi] = sum / count;
    }
}

bool FemEngine::solveLinear() {
    auto fixed = dirichletNodes();
    for (size_t k = 0; k < periodicPairs_.size(); ++k) {
        const auto& pr = periodicPairs_[k];
        if (fixed.count(pr.first) || fixed.count(pr.second)) {
            const int sign = (k < periodicSigns_.size()) ? periodicSigns_[k] : 1;
            const double value = fixed.count(pr.first) ? fixed.at(pr.first) : fixed.at(pr.second);
            fixed[pr.first] = value;
            fixed[pr.second] = sign * value;
        }
    }
    std::vector<int> freeNodes, signs;
    const std::vector<int> roots = periodicDofRoots(fixed, freeNodes, signs);
    if (freeNodes.empty()) {
        A_.setZero(K_.rows());
        for (const auto& kv : fixed) A_[kv.first] = kv.second;
        lastResidual_ = 0.0;
        return true;
    }
    Eigen::SparseMatrix<double> Kff(freeNodes.size(), freeNodes.size());
    std::vector<Eigen::Triplet<double>> trips;
    for (int k = 0; k < K_.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(K_, k); it; ++it) {
            const int r = roots[it.row()], c = roots[it.col()];
            if (r >= 0 && c >= 0)
                trips.push_back(Eigen::Triplet<double>(r, c, it.value() * signs[it.row()] * signs[it.col()]));
        }
    }
    Kff.setFromTriplets(trips.begin(), trips.end());
    A_.setZero(K_.rows());
    for (const auto& kv : fixed) A_[kv.first] = kv.second;
    const Eigen::VectorXd r0 = K_*A_ - rhs_;
    Eigen::VectorXd rf = Eigen::VectorXd::Zero(freeNodes.size());
    for (int vi = 0; vi < (int)roots.size(); ++vi)
        if (roots[vi] >= 0) rf[roots[vi]] -= signs[vi] * r0[vi];

    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(Kff);
    if (solver.info() != Eigen::Success || freeNodes.empty()) {
        std::cerr << "FEM: factorization failed" << std::endl;
        A_.setZero(K_.rows());
        return false;
    }
    const Eigen::VectorXd af = solver.solve(rf);
    if (solver.info() != Eigen::Success) {
        std::cerr << "FEM: solve failed" << std::endl;
        A_.setZero(K_.rows());
        return false;
    }
    A_.setZero(K_.rows());
    for (const auto& kv : fixed) A_[kv.first] = kv.second;
    for (int vi = 0; vi < (int)roots.size(); ++vi)
        if (roots[vi] >= 0) A_[vi] = signs[vi] * af[roots[vi]];
    const Eigen::VectorXd residual = K_*A_ - rhs_;
    Eigen::VectorXd reducedResidual = Eigen::VectorXd::Zero(freeNodes.size());
    Eigen::VectorXd reducedR0 = Eigen::VectorXd::Zero(freeNodes.size());
    for (int vi = 0; vi < (int)roots.size(); ++vi) {
        if (roots[vi] < 0) continue;
        reducedResidual[roots[vi]] += signs[vi] * residual[vi];
        reducedR0[roots[vi]] += signs[vi] * r0[vi];
    }
    lastResidual_ = reducedResidual.norm() / std::max(reducedR0.norm(), 1e-30);
    return lastResidual_ < 1e-8;
}

void FemEngine::exportResults(const std::string& outDir) {
    fs::create_directories(outDir);
    // create subdirs
    for (const char* d : {"Az","Mu","H","Jz","EnergyDensity","Forces","FluxLinkage","BoundaryImg","InputImg","Mesh"}) {
        fs::create_directories(fs::path(outDir) / d);
    }
    int nx = nx_, ny = ny_;
    Eigen::MatrixXd Az(nx, ny), Mu(nx, ny), Hmag(nx, ny), Jz(nx, ny), Ed(nx, ny);
    Az.setZero(); Mu.setZero(); Hmag.setZero(); Jz.setZero(); Ed.setZero();
    int missingSamples = 0;
    const double missing = std::numeric_limits<double>::quiet_NaN();

    // FDM/WebUI point lattice: pixel (i,j) is the physical point (i*dx,j*dy).
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            const double x = ix * dx_, y = iy * dy_;
            const int ti = findTriangle(x, y);
            if (ti < 0) {
                Az(ix, iy) = Mu(ix, iy) = Hmag(ix, iy) = Jz(ix, iy) = Ed(ix, iy) = missing;
                ++missingSamples;
                continue;
            }
            double l0,l1,l2;
            pointInTri(x,y,mesh_,ti,l0,l1,l2);
            const auto& t = mesh_.triangles[ti];
            Az(ix, iy) = l0*A_[t.v[0]] + l1*A_[t.v[1]] + l2*A_[t.v[2]];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            double g[3][2], area = 0.0, signedArea = 0.0;
            triangleGradients(p0, p1, p2, g, area, signedArea);
            double Bx = 0.0, By = 0.0;
            for (int a = 0; a < 3; ++a) {
                Bx += A_[t.v[a]] * g[a][1];
                By += -A_[t.v[a]] * g[a][0];
            }
            const double Bmag = std::hypot(Bx, By);
            const double mu = elemMu_[ti];
            Hmag(ix, iy) = Bmag / mu;
            Mu(ix, iy) = mu;
            const int mx = std::min(nx-1, std::max(0, (int)std::llround(x/dx_)));
            const int my = std::min(ny-1, std::max(0, (int)std::llround(y/dy_)));
            const int mid = (mx + my*nx_ < (int)mask_.size()) ? mask_[my*nx_+mx] : -1;
            Jz(ix, iy) = (mid >= 0 && mid < (int)matData_.size()) ? matData_[mid].jz : 0.0;
            Ed(ix, iy) = coEnergyDensity(Bmag, t.material);
        }
    }

    // FDM writes row-major with rows=y, cols=x? Eigen matrix (rows,cols). writeMatrixCSV expects (rows,cols).
    // FDM Az is ny rows x nx cols. Our Az is (nx,ny) currently (cols,rows). Transpose.
    Eigen::MatrixXd AzT = Az.transpose();   // ny x nx
    Eigen::MatrixXd MuT = Mu.transpose();
    Eigen::MatrixXd Ht = Hmag.transpose();
    Eigen::MatrixXd Jt = Jz.transpose();
    Eigen::MatrixXd Et = Ed.transpose();

    MagneticFieldAnalyzer::ExportConfig opts;
    bool writeCsv = false, writeTiff = true;
    if (config_["export"]) {
        const std::string fmt = config_["export"]["format"].as<std::string>("tiff");
        writeCsv = (fmt == "csv" || fmt == "both");
        writeTiff = (fmt == "tiff" || fmt == "both");
        const std::string prec = config_["export"]["precision"].as<std::string>("double");
        opts.precision = (prec == "float") ? MagneticFieldAnalyzer::ExportConfig::Precision::F32
                                           : MagneticFieldAnalyzer::ExportConfig::Precision::F64;
    }
    if (writeCsv) {
        MagneticFieldAnalyzer::writeMatrixCSV(AzT, outDir + "/Az/step_0001.csv");
        MagneticFieldAnalyzer::writeMatrixCSV(MuT, outDir + "/Mu/step_0001.csv");
        MagneticFieldAnalyzer::writeMatrixCSV(Ht, outDir + "/H/step_0001.csv");
        MagneticFieldAnalyzer::writeMatrixCSV(Jt, outDir + "/Jz/step_0001.csv");
        MagneticFieldAnalyzer::writeMatrixCSV(Et, outDir + "/EnergyDensity/step_0001.csv");
    }
    if (writeTiff) {
        MagneticFieldAnalyzer::writeMatrixTIFF(AzT, outDir + "/Az/step_0001.tiff", opts);
        MagneticFieldAnalyzer::writeMatrixTIFF(MuT, outDir + "/Mu/step_0001.tiff", opts);
        MagneticFieldAnalyzer::writeMatrixTIFF(Ht, outDir + "/H/step_0001.tiff", opts);
        MagneticFieldAnalyzer::writeMatrixTIFF(Jt, outDir + "/Jz/step_0001.tiff", opts);
        MagneticFieldAnalyzer::writeMatrixTIFF(Et, outDir + "/EnergyDensity/step_0001.tiff", opts);
    }

    // copy input image
    cv::Mat inp = cv::imread(imagePath_, cv::IMREAD_COLOR);
    if (!inp.empty()) cv::imwrite(outDir + "/InputImg/step_0001.png", inp);

    // Mesh artifact includes bounds, coordinate convention and missing samples.
    std::ofstream mf(outDir + "/Mesh/mesh.json");
    mf << "{\"nodes\":" << mesh_.vertices.size()
       << ",\"triangles\":" << mesh_.triangles.size()
       << ",\"hash\":\"" << mesh_.hash() << "\""
       << ",\"bounds\":{\"xmin\":0,\"xmax\":" << (nx_-1)*dx_
       << ",\"ymin\":0,\"ymax\":" << (ny_-1)*dy_ << "}"
       << ",\"coordinates\":\"pixel-point-y-up\""
       << ",\"missing_samples\":" << missingSamples << "}";
    mf.close();

    // Direct FemEngine users do not go through MagFDMsolver's conditions
    // exporter. Create a minimal file only when it is absent; never overwrite
    // the full FDM/WebUI metadata written by the CLI.
    const fs::path conditionsPath = fs::path(outDir) / "conditions.json";
    if (!fs::exists(conditionsPath)) {
        std::ofstream cf(conditionsPath);
        cf << "{\"solver\":{\"engine\":\"fem\"},\"coordinate_system\":\"cartesian\""
           << ",\"image_width\":" << nx_ << ",\"image_height\":" << ny_
           << ",\"dx\":" << dx_ << ",\"dy\":" << dy_
           << ",\"mesh\":{\"nodes\":" << mesh_.vertices.size()
           << ",\"triangles\":" << mesh_.triangles.size() << "}}";
        cf.close();
    }

    // Amperian forces per material: free current plus bound surface current
    // on every material interface (permanent + induced magnetization).
    struct ForceAccum { double fx = 0.0, fy = 0.0, tq = 0.0; int count = 0; };
    std::map<std::string, ForceAccum> frc;
    auto addForce = [&](const std::string& name, double fx, double fy, double x, double y) {
        auto& r = frc[name];
        r.fx += fx; r.fy += fy; r.tq += x*fy - y*fx; ++r.count;
    };
    auto fieldAt = [&](double x, double y, double& bx, double& by, int* material = nullptr) {
        const int ti = findTriangle(x, y);
        if (ti < 0) return false;
        const auto& t = mesh_.triangles[ti];
        const auto& p0 = mesh_.vertices[t.v[0]].p;
        const auto& p1 = mesh_.vertices[t.v[1]].p;
        const auto& p2 = mesh_.vertices[t.v[2]].p;
        double g[3][2], area = 0.0, signedArea = 0.0;
        triangleGradients(p0, p1, p2, g, area, signedArea);
        bx = by = 0.0;
        for (int a = 0; a < 3; ++a) { bx += A_[t.v[a]]*g[a][1]; by += -A_[t.v[a]]*g[a][0]; }
        if (material) *material = t.material;
        return true;
    };
    auto magnetizationAt = [&](double x, double y, double& mx, double& my) {
        double bx = 0.0, by = 0.0;
        int mid = -1;
        if (!fieldAt(x, y, bx, by, &mid)) { mx = my = 0.0; return; }
        const double b = std::hypot(bx, by);
        const double nu = evalNu(b, mid);
        mx = (1.0/mu0() - nu) * bx;
        my = (1.0/mu0() - nu) * by;
        if (mid >= 0 && mid < (int)matData_.size() && matData_[mid].hasMagnetization) {
            mx += (matData_[mid].Br/mu0()) * std::cos(matData_[mid].magAngle);
            my += (matData_[mid].Br/mu0()) * std::sin(matData_[mid].magAngle);
        }
    };

    for (const auto& tri : mesh_.triangles) {
        const auto& p0 = mesh_.vertices[tri.v[0]].p;
        const auto& p1 = mesh_.vertices[tri.v[1]].p;
        const auto& p2 = mesh_.vertices[tri.v[2]].p;
        double g[3][2], area = 0.0, signedArea = 0.0;
        triangleGradients(p0, p1, p2, g, area, signedArea);
        if (area < 1e-30) continue;
        double Bx = 0.0, By = 0.0;
        for (int a = 0; a < 3; ++a) { Bx += A_[tri.v[a]]*g[a][1]; By += -A_[tri.v[a]]*g[a][0]; }
        const int mid = tri.material;
        const std::string name = (mid >= 0 && mid < (int)matData_.size()) ? matData_[mid].name : "air";
        const double jz = (mid >= 0 && mid < (int)matData_.size()) ? matData_[mid].jz : 0.0;
        if (jz != 0.0) {
            // J_z z-hat x B = (-J_z By, +J_z Bx).
            const double fx = -jz * By * area;
            const double fy =  jz * Bx * area;
            addForce(name, fx, fy, (p0.x+p1.x+p2.x)/3.0, (p0.y+p1.y+p2.y)/3.0);
        }
    }

    const double eps = 1e-6 * std::min(dx_, dy_);
    for (const auto& c : mesh_.contours) {
        const int mid = c.materialInside;
        if (mid < 0 || mid >= (int)matData_.size()) continue;
        const std::string name = matData_[mid].name;
        const size_t n = c.points.size();
        for (size_t i = 0; i < n; ++i) {
            const auto& p0 = c.points[i];
            const auto& p1 = c.points[(i + 1) % n];
            const double dxs = p1.x - p0.x, dys = p1.y - p0.y;
            const double len = std::hypot(dxs, dys);
            if (len < 1e-30) continue;
            // Outer CCW / hole CW: right normal points from material to outside.
            const double nx = dys / len, ny = -dxs / len;
            const double mx = (p0.x + p1.x) * 0.5, my = (p0.y + p1.y) * 0.5;
            double mix = 0.0, miy = 0.0, mox = 0.0, moy = 0.0;
            magnetizationAt(mx - eps*nx, my - eps*ny, mix, miy);
            magnetizationAt(mx + eps*nx, my + eps*ny, mox, moy);
            const double Kz = (mix - mox)*ny - (miy - moy)*nx;
            double bix = 0.0, biy = 0.0, box = 0.0, boy = 0.0;
            fieldAt(mx - eps*nx, my - eps*ny, bix, biy);
            fieldAt(mx + eps*nx, my + eps*ny, box, boy);
            const double bx = 0.5*(bix + box), by = 0.5*(biy + boy);
            addForce(name, -Kz*by*len, Kz*bx*len, mx, my);
        }
    }
    // Total co-energy, consistent with the nonlinear B-H constitutive law.
    double E = 0.0;
    for (const auto& t : mesh_.triangles) {
        const auto& p0 = mesh_.vertices[t.v[0]].p;
        const auto& p1 = mesh_.vertices[t.v[1]].p;
        const auto& p2 = mesh_.vertices[t.v[2]].p;
        double g[3][2], area = 0.0, signedArea = 0.0;
        triangleGradients(p0, p1, p2, g, area, signedArea);
        if (area < 1e-30) continue;
        double Bx = 0.0, By = 0.0;
        for (int a = 0; a < 3; ++a) { Bx += A_[t.v[a]]*g[a][1]; By += -A_[t.v[a]]*g[a][0]; }
        E += coEnergyDensity(std::hypot(Bx, By), t.material) * area;
    }
    std::ofstream ef(outDir + "/Energy.txt");
    ef << std::setprecision(15) << E << std::endl;
    ef.close();

    // Keep the long-standing FDM/WebUI Forces CSV contract.
    std::ofstream ff(outDir + "/Forces/step_0001.csv");
    ff << "Material,RGB_R,RGB_G,RGB_B,Force_X[N/m],Force_Y[N/m],Force_Magnitude[N/m],Force_Radial[N/m],Torque_Origin[N*m/m],Torque_Center[N*m/m],Magnetic_Energy[J/m],Boundary_Pixels\n";
    ff << "# Note: Forces and energies are per unit depth (2D analysis)\n";
    ff << "# Torque_Origin: torque about coordinate origin; multiply by axial depth [m] for N*m\n";
    ff << "# Torque_Center: Cartesian nodal-grid midpoint, or polar origin\n";
    ff << "# _SYSTEM_TOTAL: Existing co-energy diagnostic; derivatives require fixed excitation and consistent PM energy\n";
    ff << "# Boundary_Pixels: number of selected material nodes (legacy column name)\n";
    ff << "# Force calculation method: Amperian, J = J_free + curl(M_source) + div((nu_body-nu0_body) grad(Az)), F = integral(J x B)\n";
    ff << "# Per-material Magnetic_Energy is not evaluated (legacy zero placeholder); use _SYSTEM_TOTAL only\n";
    ff << "# Touching-material rows are an additive source decomposition, not contact tractions\n";
    ff << std::scientific << std::setprecision(15);
    std::map<std::string, const MatData*> materialByName;
    for (const auto& md : matData_) materialByName[md.name] = &md;
    const double centerX = 0.5*(nx_-1)*dx_, centerY = 0.5*(ny_-1)*dy_;
    double tFx = 0.0, tFy = 0.0, tTq = 0.0;
    for (const auto& kv : frc) {
        const auto& r = kv.second;
        int cr = 255, cg = 255, cb = 255;
        auto it = materialByName.find(kv.first);
        if (it != materialByName.end()) { cr = it->second->rgb[0]; cg = it->second->rgb[1]; cb = it->second->rgb[2]; }
        const double fmag = std::hypot(r.fx, r.fy);
        const double tqCenter = r.tq - (centerX*r.fy - centerY*r.fx);
        ff << kv.first << "," << cr << "," << cg << "," << cb << ","
           << r.fx << "," << r.fy << "," << fmag << ",0,"
           << r.tq << "," << tqCenter << ",0," << r.count << "\n";
        tFx += r.fx; tFy += r.fy; tTq += r.tq;
    }
    ff << "_SYSTEM_TOTAL,0,0,0,0,0,0,0,0,0," << E << ",0\n";
    ff.close();

    // flux linkage (material-pair variant)
    if (config_["flux_linkage"]) {
        auto sampleAz = [&](double x, double y, double& value) {
            const int ti = findTriangle(x, y);
            if (ti < 0) return false;
            double l0,l1,l2;
            if (!pointInTri(x,y,mesh_,ti,l0,l1,l2)) return false;
            const auto& t = mesh_.triangles[ti];
            value = l0*A_[t.v[0]] + l1*A_[t.v[1]] + l2*A_[t.v[2]];
            return true;
        };
        std::vector<std::string> names;
        std::vector<double> values;
        for (const auto& p : config_["flux_linkage"]) {
            names.push_back(p["name"] ? p["name"].as<std::string>() : "path");
            const bool hasA = (bool)p["material_a"], hasB = (bool)p["material_b"];
            const bool useMat = hasA || hasB || (p["use_material"] && p["use_material"].as<bool>());
            double phi = std::numeric_limits<double>::quiet_NaN();
            if (useMat) {
                const std::string ma = hasA ? p["material_a"].as<std::string>() : "";
                const std::string mb = hasB ? p["material_b"].as<std::string>() : "";
                double sa = 0.0, sb = 0.0;
                int ca = 0, cb = 0;
                for (int iy = 0; iy < ny_; ++iy) {
                    for (int ix = 0; ix < nx_; ++ix) {
                        const int midx = mask_[iy*nx_+ix];
                        const std::string mname = (midx >= 0 && midx < (int)matData_.size()) ? matData_[midx].name : "air";
                        if ((!hasA || mname != ma) && (!hasB || mname != mb)) continue;
                        const double a = Az(ix, iy);
                        if (!std::isfinite(a)) continue;
                        if (hasA && mname == ma) { sa += a; ++ca; }
                        if (hasB && mname == mb) { sb += a; ++cb; }
                    }
                }
                if ((!hasA || ca > 0) && (!hasB || cb > 0))
                    phi = (ca > 0 ? sa/ca : 0.0) - (cb > 0 ? sb/cb : 0.0);
            } else if (p["start"] && p["end"]) {
                const auto s = p["start"].as<std::vector<double>>();
                const auto e = p["end"].as<std::vector<double>>();
                double a0 = 0.0, a1 = 0.0;
                if (s.size() >= 2 && e.size() >= 2 && sampleAz(s[0], s[1], a0) && sampleAz(e[0], e[1], a1))
                    phi = a1 - a0;
            }
            values.push_back(phi);
        }
        std::ofstream fl(outDir + "/FluxLinkage/flux_linkage.csv");
        fl << "step";
        for (const auto& name : names) fl << "," << name;
        fl << "\n0";
        for (double value : values) fl << "," << value;
        fl << "\n";
        fl.close();

        std::ofstream statusFile(outDir + "/FluxLinkage/flux_linkage_status.csv");
        statusFile << "step,nonlinear,strict_converged,state_reusable,iterations,residual\n";
        statusFile << "0," << ((anyNonlinear() && nonlinearEnabled_) ? 1 : 0) << ","
           << (lastConverged_ ? 1 : 0) << "," << (lastConverged_ ? 1 : 0) << ","
           << lastIterations_ << "," << std::scientific << std::setprecision(10) << lastResidual_ << "\n";
        statusFile.close();
    }
}

// ---- Phase 2 helpers ----
void FemEngine::parseNonlinearConfig() {
    nonlinearEnabled_ = true; nonlinearMaxIter_ = 50; nonlinearTol_ = 5e-4; relaxation_ = 0.7;
    jacobianMode_ = "tangent"; lineSearchC_ = 1e-4; lineSearchRho_ = 0.65; lineSearchMax_ = 50;
    if (config_["nonlinear_solver"]) {
        const auto& n = config_["nonlinear_solver"];
        if (n["enabled"]) nonlinearEnabled_ = n["enabled"].as<bool>();
        if (n["max_iterations"]) nonlinearMaxIter_ = n["max_iterations"].as<int>();
        if (n["tolerance"]) nonlinearTol_ = n["tolerance"].as<double>();
        if (n["relaxation"]) relaxation_ = n["relaxation"].as<double>();
        if (n["line_search_c"]) lineSearchC_ = n["line_search_c"].as<double>();
        if (n["line_search_rho"]) lineSearchRho_ = n["line_search_rho"].as<double>();
        if (n["line_search_max_trials"]) lineSearchMax_ = n["line_search_max_trials"].as<int>();
        if (n["jacobian"]) jacobianMode_ = n["jacobian"].as<std::string>();
    }
}

void FemEngine::parseRobinPeriodic() {
    robinEdges_.clear(); periodicPairs_.clear(); periodicSigns_.clear();
    if (!config_["boundary_conditions"]) return;
    const auto& bc = config_["boundary_conditions"];
    double Lx = (nx_ - 1) * dx_, Ly = (ny_ - 1) * dy_;
    std::map<std::pair<int,int>,int> edgeCount;
    for (const auto& t : mesh_.triangles) {
        int v[3]={t.v[0],t.v[1],t.v[2]};
        for (int k=0;k<3;++k){int a=v[k],b=v[(k+1)%3]; if(a>b) std::swap(a,b); edgeCount[{a,b}]++;}
    }
    for (const auto& kv : edgeCount) {
        if (kv.second != 1) continue;
        int i=kv.first.first, j=kv.first.second;
        const auto& pi = mesh_.vertices[i].p; const auto& pj = mesh_.vertices[j].p;
        double mx=(pi.x+pj.x)/2, my=(pi.y+pj.y)/2;
        std::string side;
        if (std::fabs(my) < 1e-9) side="bottom";
        else if (std::fabs(my-Ly) < 1e-9) side="top";
        else if (std::fabs(mx) < 1e-9) side="left";
        else if (std::fabs(mx-Lx) < 1e-9) side="right";
        if (side.empty() || !bc[side]) continue;
        std::string type = bc[side]["type"].as<std::string>("dirichlet");
        if (type == "robin") {
            double alpha = bc[side]["alpha"].as<double>(1.0);
            double beta = bc[side]["beta"].as<double>(0.0);
            double gamma = bc[side]["gamma"].as<double>(0.0);
            if (std::fabs(beta) > 1e-30) robinEdges_.push_back({i,j,alpha,beta,gamma});
        }
    }
    // periodic pairs
    auto isPeriodic = [&](const std::string& side) {
        if (!bc[side]) return false;
        return bc[side]["type"].as<std::string>("") == "periodic";
    };
    auto pairSides = [&](std::vector<int> a, std::vector<int> b, bool alongY, int sign) {
        auto coord = [&](int vi) { return alongY ? mesh_.vertices[vi].p.y : mesh_.vertices[vi].p.x; };
        std::sort(a.begin(), a.end(), [&](int x,int y){return coord(x)<coord(y);});
        std::sort(b.begin(), b.end(), [&](int x,int y){return coord(x)<coord(y);});
        std::vector<char> used(b.size(), 0);
        for (int ai : a) {
            int best=-1; double bestD=1e300;
            for (int j=0;j<(int)b.size();++j) if(!used[j]) {
                const double d=std::fabs(coord(ai)-coord(b[j]));
                if(d<bestD){bestD=d;best=j;}
            }
            if(best>=0){used[best]=1;periodicPairs_.push_back({ai,b[best]});periodicSigns_.push_back(sign);}
        }
    };
    if (isPeriodic("left") && isPeriodic("right")) {
        std::vector<int> ls, rs;
        for (int vi=0; vi<(int)mesh_.vertices.size(); ++vi) {
            if (std::fabs(mesh_.vertices[vi].p.x) < 1e-9) ls.push_back(vi);
            if (std::fabs(mesh_.vertices[vi].p.x - ((nx_-1)*dx_)) < 1e-9) rs.push_back(vi);
        }
        const int sign = (bc["left"]["value"] && bc["left"]["value"].as<int>() == -1) ||
                         (bc["right"]["value"] && bc["right"]["value"].as<int>() == -1) ? -1 : 1;
        pairSides(ls, rs, true, sign);
    }
    if (isPeriodic("bottom") && isPeriodic("top")) {
        std::vector<int> bs, ts;
        for (int vi=0; vi<(int)mesh_.vertices.size(); ++vi) {
            if (std::fabs(mesh_.vertices[vi].p.y) < 1e-9) bs.push_back(vi);
            if (std::fabs(mesh_.vertices[vi].p.y - ((ny_-1)*dy_)) < 1e-9) ts.push_back(vi);
        }
        const int sign = (bc["bottom"]["value"] && bc["bottom"]["value"].as<int>() == -1) ||
                         (bc["top"]["value"] && bc["top"]["value"].as<int>() == -1) ? -1 : 1;
        pairSides(bs, ts, false, sign);
    }
}

std::array<double,2> FemEngine::remanentMagnetization(int materialIndex) const {
    if (materialIndex < 0 || materialIndex >= (int)matData_.size()) return {0.0, 0.0};
    const auto& md = matData_[materialIndex];
    if (!md.hasMagnetization) return {0.0, 0.0};
    const double m = md.Br / mu0();
    return {m * std::cos(md.magAngle), m * std::sin(md.magAngle)};
}

bool FemEngine::anyNonlinear() const {
    for (const auto& md : matData_) if (md.hasBH()) return true;
    return false;
}

double FemEngine::evalNu(double Bmag, int mid) const {
    if (mid < 0 || mid >= (int)matData_.size()) return 1.0 / mu0();
    const auto& md = matData_[mid];
    if (!md.hasBH()) return 1.0 / (mu0() * (md.mu_r > 0 ? md.mu_r : 1.0));
    const auto& H = md.bhH;
    const auto& B = md.bhB;
    const double b = std::max(0.0, Bmag);
    const double eps = 1e-30;
    if (b <= B.front()) {
        const double slope = (H[1] - H[0]) / (B[1] - B[0]);
        const double h = H[0] + slope * (b - B[0]);
        return b > eps ? h / b : slope;
    }
    if (b >= B.back()) {
        const double slope = 1.0 / mu0(); // saturating extension: M -> 0
        return (H.back() + slope * (b - B.back())) / b;
    }
    const size_t i = (size_t)(std::upper_bound(B.begin(), B.end(), b) - B.begin() - 1);
    const double t = (b - B[i]) / (B[i+1] - B[i]);
    return (H[i] + t * (H[i+1] - H[i])) / b;
}

double FemEngine::evalDnu(double Bmag, int mid) const {
    if (mid < 0 || mid >= (int)matData_.size() || !matData_[mid].hasBH()) return 0.0;
    const auto& md = matData_[mid];
    const auto& H = md.bhH;
    const auto& B = md.bhB;
    const double b = std::max(0.0, Bmag);
    if (b <= 1e-30) return 0.0;
    size_t i = 0;
    double slope = 0.0, h = 0.0;
    if (b <= B.front()) {
        i = 0;
        slope = (H[1] - H[0]) / (B[1] - B[0]);
        h = H[0] + slope * (b - B[0]);
    } else if (b >= B.back()) {
        i = B.size() - 2;
        slope = 1.0 / mu0();
        h = H.back() + slope * (b - B.back());
    } else {
        i = (size_t)(std::upper_bound(B.begin(), B.end(), b) - B.begin() - 1);
        slope = (H[i+1] - H[i]) / (B[i+1] - B[i]);
        h = H[i] + slope * (b - B[i]);
    }
    return (slope * b - h) / (b * b);
}

double FemEngine::coEnergyDensity(double Bmag, int mid) const {
    const double b = std::max(0.0, Bmag);
    if (mid < 0 || mid >= (int)matData_.size()) return b*b / (2.0*mu0());
    const auto& md = matData_[mid];
    if (!md.hasBH()) {
        const double mu = mu0() * (md.mu_r > 0 ? md.mu_r : 1.0);
        return b*b / (2.0*mu);
    }
    if (b <= 1e-30) return 0.0;
    const auto& H = md.bhH;
    const auto& B = md.bhB;
    const double h = evalNu(b, mid) * b;
    const double slope0 = (H[1] - H[0]) / (B[1] - B[0]);
    const double slopeN = 1.0 / mu0();
    double integral = 0.0; // integral_0^b H(db) dB
    if (b <= B.front()) {
        const double h0 = H[0] - slope0 * B[0];
        integral = 0.5 * (h0 + h) * b;
    } else {
        double x = 0.0, hx = H[0] - slope0 * B[0];
        if (B.front() > 0.0) {
            integral += 0.5 * (hx + H[0]) * B.front();
            x = B.front(); hx = H[0];
        }
        for (size_t i = 0; i + 1 < B.size() && x < b; ++i) {
            const double xe = std::min(b, B[i+1]);
            const double slope = (H[i+1] - H[i]) / (B[i+1] - B[i]);
            const double he = H[i] + slope * (xe - B[i]);
            integral += 0.5 * (hx + he) * (xe - x);
            x = xe; hx = he;
        }
        if (x < b) {
            const double he = hx + slopeN * (b - x);
            integral += 0.5 * (hx + he) * (b - x);
        }
    }
    return std::max(0.0, b*h - integral); // co-energy density: integral B dH
}

void FemEngine::assembleJacobian(Eigen::SparseMatrix<double,Eigen::RowMajor>& J) const {
    const int N = (int)mesh_.vertices.size();
    J.resize(N, N);
    std::vector<Eigen::Triplet<double>> trips;
    for (const auto& tri : mesh_.triangles) {
        const int vi[3] = {tri.v[0], tri.v[1], tri.v[2]};
        const auto& p0 = mesh_.vertices[vi[0]].p;
        const auto& p1 = mesh_.vertices[vi[1]].p;
        const auto& p2 = mesh_.vertices[vi[2]].p;
        double g[3][2], area = 0.0, signedArea = 0.0;
        triangleGradients(p0, p1, p2, g, area, signedArea);
        if (area < 1e-30) continue;
        double Bx = 0.0, By = 0.0;
        for (int a = 0; a < 3; ++a) {
            Bx += A_[vi[a]] * g[a][1];
            By += -A_[vi[a]] * g[a][0];
        }
        const double Bmag = std::hypot(Bx, By);
        const double nu = evalNu(Bmag, tri.material);
        const double dnu = evalDnu(Bmag, tri.material);
        for (int a = 0; a < 3; ++a) {
            for (int c = 0; c < 3; ++c) {
                double value = nu * area * (g[a][0]*g[c][0] + g[a][1]*g[c][1]);
                if (Bmag > 1e-12 && dnu != 0.0) {
                    // B . B_c = g_c . grad(A) = g_c.y*Bx - g_c.x*By.
                    const double gcGradA = g[c][1]*Bx - g[c][0]*By;
                    const double gaGradA = g[a][1]*Bx - g[a][0]*By;
                    value += (dnu / Bmag) * area * gcGradA * gaGradA;
                }
                trips.push_back(Eigen::Triplet<double>(vi[a], vi[c], value));
            }
        }
    }
    for (const auto& re : robinEdges_) {
        const auto& pi = mesh_.vertices[re.i].p;
        const auto& pj = mesh_.vertices[re.j].p;
        const double len = std::hypot(pj.x-pi.x, pj.y-pi.y);
        if (len < 1e-30 || std::fabs(re.beta) < 1e-30) continue;
        const int mx = std::max(0, std::min(nx_-1, (int)std::llround(((pi.x+pj.x)*0.5)/dx_)));
        const int my = std::max(0, std::min(ny_-1, (int)std::llround(((pi.y+pj.y)*0.5)/dy_)));
        const int mid = (mx + my*nx_ < (int)mask_.size()) ? mask_[my*nx_+mx] : -1;
        const double nu = evalNu(0.0, mid);
        const double c = nu * (re.alpha / re.beta) * len / 6.0;
        trips.push_back(Eigen::Triplet<double>(re.i, re.i, 2.0*c));
        trips.push_back(Eigen::Triplet<double>(re.j, re.j, 2.0*c));
        trips.push_back(Eigen::Triplet<double>(re.i, re.j, c));
        trips.push_back(Eigen::Triplet<double>(re.j, re.i, c));
    }
    for (size_t k = 0; k < periodicPairs_.size(); ++k) {
        const auto& pr = periodicPairs_[k];
        const int sign = (k < periodicSigns_.size()) ? periodicSigns_[k] : 1;
        const double P = 1e10;
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.first, P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.second, P));
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.second, -P*sign));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.first, -P*sign));
    }
    J.setFromTriplets(trips.begin(), trips.end());
}

int FemEngine::solve() {
    const int N = (int)mesh_.vertices.size();
    A_.setZero(N);
    auto fixed = dirichletNodes();
    for (size_t k = 0; k < periodicPairs_.size(); ++k) {
        const auto& pr = periodicPairs_[k];
        if (fixed.count(pr.first) || fixed.count(pr.second)) {
            const int sign = (k < periodicSigns_.size()) ? periodicSigns_[k] : 1;
            const double value = fixed.count(pr.first) ? fixed.at(pr.first) : fixed.at(pr.second);
            fixed[pr.first] = value;
            fixed[pr.second] = sign * value;
        }
    }
    for (const auto& kv : fixed) A_[kv.first] = kv.second;
    std::vector<int> freeNodes, signs;
    const std::vector<int> roots = periodicDofRoots(fixed, freeNodes, signs);

    if (!anyNonlinear() || !nonlinearEnabled_) {
        assemble();
        const bool ok = solveLinear();
        lastIterations_ = 1;
        lastConverged_ = ok;
        return ok ? 0 : 1;
    }

    lastConverged_ = false;
    lastIterations_ = 0;
    lastResidual_ = 0.0;
    double rhsNorm = 0.0;
    for (int iter = 0; iter < nonlinearMaxIter_; ++iter) {
        assemble();
        Eigen::VectorXd R = K_ * A_ - rhs_;
        for (const auto& kv : fixed) R[kv.first] = 0.0;
        Eigen::VectorXd reducedR = Eigen::VectorXd::Zero(freeNodes.size());
        for (int vi = 0; vi < N; ++vi)
            if (roots[vi] >= 0) reducedR[roots[vi]] += signs[vi] * R[vi];
        if (iter == 0) {
            rhsNorm = 0.0;
            for (int vi = 0; vi < N; ++vi)
                if (roots[vi] >= 0) rhsNorm += rhs_[vi]*rhs_[vi];
            rhsNorm = std::sqrt(rhsNorm);
        }
        const double scale = rhsNorm > 1e-30 ? rhsNorm : 1.0;
        const double rrel = reducedR.norm() / scale;
        if (rrel < nonlinearTol_) {
            lastIterations_ = iter + 1;
            lastResidual_ = rrel;
            lastConverged_ = true;
            return 0;
        }

        Eigen::SparseMatrix<double,Eigen::RowMajor> J;
        assembleJacobian(J);
        auto solveReduced = [&](const Eigen::SparseMatrix<double,Eigen::RowMajor>& matrix,
                                const Eigen::VectorXd& fullRhs, Eigen::VectorXd& result) {
            result.setZero(N);
            if (freeNodes.empty()) return true;
            Eigen::SparseMatrix<double> reduced(freeNodes.size(), freeNodes.size());
            std::vector<Eigen::Triplet<double>> trips;
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(matrix, k); it; ++it) {
                    const int r = roots[it.row()], c = roots[it.col()];
                    if (r >= 0 && c >= 0)
                trips.push_back(Eigen::Triplet<double>(r, c, it.value() * signs[it.row()] * signs[it.col()]));
                }
            }
            reduced.setFromTriplets(trips.begin(), trips.end());
            Eigen::VectorXd rf = Eigen::VectorXd::Zero(freeNodes.size());
            for (int vi = 0; vi < N; ++vi)
                if (roots[vi] >= 0) rf[roots[vi]] += signs[vi] * fullRhs[vi];
            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(reduced);
            if (solver.info() != Eigen::Success) return false;
            const Eigen::VectorXd x = solver.solve(rf);
            if (solver.info() != Eigen::Success) return false;
            for (int vi = 0; vi < N; ++vi)
                if (roots[vi] >= 0) result[vi] = signs[vi] * x[roots[vi]];
            return true;
        };
        Eigen::VectorXd dA;
        if (!solveReduced(J, -R, dA) && !solveReduced(K_, -R, dA)) return 1;

        const Eigen::VectorXd A0 = A_;
        const double R0 = reducedR.norm();
        double alpha = 1.0;
        bool accepted = false;
        for (int ls = 0; ls < lineSearchMax_; ++ls) {
            A_ = A0 + alpha * dA;
            for (const auto& kv : fixed) A_[kv.first] = kv.second;
            assemble();
            Eigen::VectorXd Rn = K_ * A_ - rhs_;
            for (const auto& kv : fixed) Rn[kv.first] = 0.0;
            Eigen::VectorXd reducedRn = Eigen::VectorXd::Zero(freeNodes.size());
            for (int vi = 0; vi < N; ++vi)
                if (roots[vi] >= 0) reducedRn[roots[vi]] += signs[vi] * Rn[vi];
            if (reducedRn.norm() <= (1.0 - lineSearchC_ * alpha) * R0 ||
                reducedRn.norm() / scale < nonlinearTol_) {
                accepted = true;
                break;
            }
            alpha *= lineSearchRho_;
        }
        if (!accepted) {
            A_ = A0;
            assemble();
            return 2;
        }
    }

    assemble();
    Eigen::VectorXd R = K_ * A_ - rhs_;
    for (const auto& kv : fixed) R[kv.first] = 0.0;
    Eigen::VectorXd reducedR = Eigen::VectorXd::Zero(freeNodes.size());
    for (int vi = 0; vi < N; ++vi)
        if (roots[vi] >= 0) reducedR[roots[vi]] += signs[vi] * R[vi];
    const double scale = rhsNorm > 1e-30 ? rhsNorm : 1.0;
    lastIterations_ = nonlinearMaxIter_;
    lastResidual_ = reducedR.norm() / scale;
    lastConverged_ = lastResidual_ < nonlinearTol_;
    return lastConverged_ ? 0 : 2;
}

void FemEngine::refine(const std::string& outDir) {
    if (!refinementEnabled_) return;
    std::vector<Point2D> extra = extraInteriorPoints_;
    std::ostringstream records;
    records << "{\"rounds\":[";
    bool firstRecord = true;
    for (int round = 0; round < refinementMaxRounds_; ++round) {
        if (solve() != 0) break;
        const int ne = (int)mesh_.triangles.size();
        const int N = (int)mesh_.vertices.size();
        std::vector<double> eta(ne, 0.0), gx(N,0.0), gy(N,0.0), w(N,0.0);
        std::vector<double> gradX(ne,0.0), gradY(ne,0.0), elemArea(ne,0.0);

        // ZZ recovery: area-weighted average of the actual element gradient |A|.
        for (int ti = 0; ti < ne; ++ti) {
            const auto& t = mesh_.triangles[ti];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            double g[3][2], area = 0.0, signedArea = 0.0;
            triangleGradients(p0, p1, p2, g, area, signedArea);
            if (area < 1e-30) continue;
            double ax = 0.0, ay = 0.0;
            for (int a = 0; a < 3; ++a) { ax += A_[t.v[a]]*g[a][0]; ay += A_[t.v[a]]*g[a][1]; }
            gradX[ti] = ax; gradY[ti] = ay; elemArea[ti] = area;
            for (int a = 0; a < 3; ++a) {
                gx[t.v[a]] += ax*area; gy[t.v[a]] += ay*area; w[t.v[a]] += area;
            }
        }
        for (int vi = 0; vi < N; ++vi) if (w[vi] > 0.0) { gx[vi] /= w[vi]; gy[vi] /= w[vi]; }

        double total = 0.0, gradNorm2 = 0.0;
        for (int ti = 0; ti < ne; ++ti) {
            const auto& t = mesh_.triangles[ti];
            double e = 0.0;
            for (int a = 0; a < 3; ++a) {
                const double dxg = gradX[ti] - gx[t.v[a]];
                const double dyg = gradY[ti] - gy[t.v[a]];
                e += dxg*dxg + dyg*dyg;
            }
            eta[ti] = elemArea[ti] * e / 3.0;
            total += eta[ti];
            gradNorm2 += elemArea[ti] * (gradX[ti]*gradX[ti] + gradY[ti]*gradY[ti]);
        }
        const double etaRel = std::sqrt(total / std::max(gradNorm2, 1e-30));
        if (!firstRecord) records << ",";
        firstRecord = false;
        records << "{\"round\":" << round << ",\"nodes\":" << N << ",\"elements\":" << ne
                << ",\"eta\":" << std::sqrt(total) << ",\"eta_rel\":" << etaRel << "}";
        if (etaRel <= refinementTarget_ || total < 1e-30) break;

        std::vector<int> order(ne);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return eta[a] > eta[b]; });
        double cumulative = 0.0;
        const double target = refinementTheta_ * total;
        std::vector<char> marked(ne, 0);
        for (int ti : order) {
            if (cumulative >= target) break;
            marked[ti] = 1;
            cumulative += eta[ti];
        }

        const double Lx = (nx_-1)*dx_, Ly = (ny_-1)*dy_;
        const double minDist = 1e-6 * std::min(dx_, dy_);
        std::vector<Point2D> newPts;
        for (int ti = 0; ti < ne; ++ti) {
            if (!marked[ti]) continue;
            const auto& t = mesh_.triangles[ti];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            const double D = 2.0*(p0.x*(p1.y-p2.y)+p1.x*(p2.y-p0.y)+p2.x*(p0.y-p1.y));
            if (std::fabs(D) < 1e-30) continue;
            const double ux = ((p0.x*p0.x+p0.y*p0.y)*(p1.y-p2.y)+(p1.x*p1.x+p1.y*p1.y)*(p2.y-p0.y)+(p2.x*p2.x+p2.y*p2.y)*(p0.y-p1.y))/D;
            const double uy = ((p0.x*p0.x+p0.y*p0.y)*(p2.x-p1.x)+(p1.x*p1.x+p1.y*p1.y)*(p0.x-p2.x)+(p2.x*p2.x+p2.y*p2.y)*(p1.x-p0.x))/D;
            if (ux < 0.0 || ux > Lx || uy < 0.0 || uy > Ly) continue;
            bool duplicate = false;
            for (const auto& p : extra) if (std::hypot(p.x-ux, p.y-uy) < minDist) { duplicate = true; break; }
            if (!duplicate) for (const auto& p : newPts) if (std::hypot(p.x-ux, p.y-uy) < minDist) { duplicate = true; break; }
            if (!duplicate) newPts.push_back({ux,uy});
        }
        if (newPts.empty()) break;
        extra.insert(extra.end(), newPts.begin(), newPts.end());
        extraInteriorPoints_ = extra;

        const FemMesh oldMesh = mesh_;
        const Eigen::VectorXd oldA = A_;
        std::string err;
        if (!buildMesh(err)) break;
        parseRobinPeriodic();
        Eigen::VectorXd warm = Eigen::VectorXd::Zero(mesh_.vertices.size());
        for (int vi = 0; vi < (int)mesh_.vertices.size(); ++vi) {
            const auto& p = mesh_.vertices[vi].p;
            const int ti = findTri(oldMesh, p.x, p.y);
            if (ti < 0) continue;
            double l0,l1,l2;
            if (!pointInTri(p.x,p.y,oldMesh,ti,l0,l1,l2)) continue;
            const auto& t = oldMesh.triangles[ti];
            warm[vi] = l0*oldA[t.v[0]] + l1*oldA[t.v[1]] + l2*oldA[t.v[2]];
        }
        A_ = warm;
    }
    records << "]}";
    fs::create_directories(fs::path(outDir) / "Mesh");
    std::ofstream rf(fs::path(outDir) / "Mesh" / "refinement.json");
    rf << records.str();
    rf.close();
    extraInteriorPoints_ = extra;
}

int FemEngine::run(const std::string& outDir, FemEngineResult* info) {
    FemEngineResult res;
    std::string err;
    if (!parseConfig(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!loadImage(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!buildMesh(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    parseRobinPeriodic();
    int rc = 0;
    if (refinementEnabled_) { refine(outDir); rc = solve(); }
    else { rc = solve(); }
    res.nodes = (int)mesh_.vertices.size();
    res.triangles = (int)mesh_.triangles.size();
    exportResults(outDir);
    res.exitCode = rc;
    if (rc == 1) res.error = "FEM_SOLVE_FAILED: factorization or solve failure";
    else if (rc == 2) res.error = "FEM_NONCONVERGED: nonlinear solver did not reach tolerance";
    if (info) *info = res;
    if (rc == 0)
        std::cout << "FEM analysis completed. nodes=" << res.nodes << " triangles=" << res.triangles << std::endl;
    else
        std::cerr << res.error << std::endl;
    return rc;
}

} // namespace fem
} // namespace omfdm
