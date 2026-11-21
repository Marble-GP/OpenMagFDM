#include "FemEngine.h"

#include "MagneticFieldAnalyzer.h"

#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
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
    // polar rejection
    std::string coord = config_["coordinate_system"] ? config_["coordinate_system"].as<std::string>() : "cartesian";
    if (coord != "cartesian") { err = "FEM_UNSUPPORTED: polar coordinates not supported in FEM mode"; return false; }

    // materials
    matData_.clear();
    if (config_["materials"]) {
        for (auto it = config_["materials"].begin(); it != config_["materials"].end(); ++it) {
            MatData md;
            md.name = it->first.as<std::string>();
            const YAML::Node& m = it->second;
            if (m["rgb"]) { auto r = m["rgb"]; md.rgb = {(uint8_t)r[0].as<int>(),(uint8_t)r[1].as<int>(),(uint8_t)r[2].as<int>()}; }
            if (m["mu_r"]) md.mu_r = m["mu_r"].as<double>();
            if (m["jz"]) md.jz = m["jz"].as<double>();
            if (m["magnetization"]) {
                const auto& mg = m["magnetization"];
                md.hasMagnetization = true;
                if (mg["Br"]) md.Br = mg["Br"].as<double>();
                if (mg["angle"]) md.magAngle = mg["angle"].as<double>();
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
        mesh_ = MeshBuilder::build(imageRaw_, materials_, dx_, dy_, meshOpts_);
    } catch (const std::exception& e) { err = std::string("mesh build failed: ") + e.what(); return false; }
    if (!mesh_.quality.passed) { err = "FEM_MESH_FAILED: " + mesh_.quality.failureReason; return false; }
    if (mesh_.vertices.size() > (size_t)meshOpts_.maxNodes) { err = "FEM_MESH_TOO_LARGE"; return false; }
    // mask
    mask_ = MeshBuilder::buildMask(imageFlippedRgb_, materials_);
    return true;
}

void FemEngine::assemble(std::string& err) {
    int N = (int)mesh_.vertices.size();
    K_.resize(N, N);
    std::vector<Eigen::Triplet<double>> trips;
    rhs_.setZero(N);

    // element stiffness + free current
    for (const auto& tri : mesh_.triangles) {
        int i0 = tri.v[0], i1 = tri.v[1], i2 = tri.v[2];
        const auto& p0 = mesh_.vertices[i0].p;
        const auto& p1 = mesh_.vertices[i1].p;
        const auto& p2 = mesh_.vertices[i2].p;
        double area = 0.5 * std::fabs((p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y));
        if (area < 1e-30) continue;
        // signed area for gradient orientation
        double signedArea = 0.5 * ((p1.x-p0.x)*(p2.y-p0.y) - (p2.x-p0.x)*(p1.y-p0.y));
        double inv2A = 1.0 / (2.0 * signedArea); // may be negative; gradients use signed area
        // gradients of basis (constant per element)
        // dphi0/dx = (p1.y - p2.y)/(2A), etc.
        double g0x = (p1.y - p2.y) * inv2A;
        double g0y = (p2.x - p1.x) * inv2A;
        double g1x = (p2.y - p0.y) * inv2A;
        double g1y = (p0.x - p2.x) * inv2A;
        double g2x = (p0.y - p1.y) * inv2A;
        double g2y = (p1.x - p0.x) * inv2A;
        int mid = tri.material;
        double mu_r = 1.0;
        // material jz (used only if no source override)
        double matJz = (mid >= 0 && mid < (int)matData_.size()) ? matData_[mid].jz : 0.0;
        if (mid >= 0 && mid < (int)matData_.size()) mu_r = matData_[mid].mu_r;
        double nu = 1.0 / (mu0() * (mu_r > 0 ? mu_r : 1.0));
        // stiffness
        double gx[3] = {g0x,g1x,g2x};
        double gy[3] = {g0y,g1y,g2y};
        int vi[3] = {i0,i1,i2};
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                trips.push_back(Eigen::Triplet<double>(vi[a], vi[b], nu * area * (gx[a]*gx[b] + gy[a]*gy[b])));
        // free current source
    // free current source
    double jz = sourceFn_ ? sourceFn_((p0.x+p1.x+p2.x)/3.0, (p0.y+p1.y+p2.y)/3.0) : matJz;
        if (jz != 0.0) {
            for (int a = 0; a < 3; ++a) rhs_[vi[a]] += jz * area / 3.0;
        }
        // store elem mu
        elemMu_.push_back(mu0() * (mu_r > 0 ? mu_r : 1.0));
    }

    std::cerr << "FEM dbg: rhs_norm=" << rhs_.norm() << " K_norm=" << K_.norm() << std::endl;

    // PM surface currents: line integrals on contours of magnetized materials.
    for (const auto& c : mesh_.contours) {
        int mid = c.materialInside;
        if (mid < 0 || mid >= (int)matData_.size()) continue;
        if (!matData_[mid].hasMagnetization) continue;
        double Br = matData_[mid].Br;
        double magAngle = matData_[mid].magAngle;
        double Mx = (Br / mu0()) * std::cos(magAngle);
        double My = (Br / mu0()) * std::sin(magAngle);
        size_t n = c.points.size();
        for (size_t i = 0; i < n; ++i) {
            const Point2D& p0 = c.points[i];
            const Point2D& p1 = c.points[(i+1)%n];
            double dxs = p1.x - p0.x, dys = p1.y - p0.y;
            double len = std::hypot(dxs, dys);
            if (len < 1e-30) continue;
            // outward normal of the material
            double nx, ny;
            if (!c.isHole) { nx = dys / len; ny = -dxs / len; } // right normal (CCW)
            else { nx = -dys / len; ny = dxs / len; }             // left normal (CW hole)
            double Kz = Mx * ny - My * nx;
            // midpoint
            // find vertex indices for p0,p1 in mesh (search)
            int vi0 = -1, vi1 = -1;
            for (int vi = 0; vi < (int)mesh_.vertices.size(); ++vi) {
                if (std::fabs(mesh_.vertices[vi].p.x - p0.x) < 1e-12 && std::fabs(mesh_.vertices[vi].p.y - p0.y) < 1e-12) vi0 = vi;
                if (std::fabs(mesh_.vertices[vi].p.x - p1.x) < 1e-12 && std::fabs(mesh_.vertices[vi].p.y - p1.y) < 1e-12) vi1 = vi;
            }
            if (vi0 < 0 || vi1 < 0) continue;
            double contrib = Kz * len * 0.5;
            rhs_[vi0] += contrib;
            rhs_[vi1] += contrib;
        }
    }

    // Dirichlet: constrain vertices on domain boundary (x=0,Lx or y=0,Ly).
    std::set<int> fixed;
    double Lx = nx_ * dx_, Ly = ny_ * dy_;
    for (int vi = 0; vi < N; ++vi) {
        const auto& p = mesh_.vertices[vi].p;
        if (std::fabs(p.x) < 1e-6 || std::fabs(p.x - Lx) < 1e-6 ||
            std::fabs(p.y) < 1e-6 || std::fabs(p.y - Ly) < 1e-6) {
            fixed.insert(vi);
        }
    }
    if (fixed.empty() && N > 0) fixed.insert(0); // ground

    // Eliminate Dirichlet rows.
    // Build fixed set; for each fixed vertex, zero its column contributions and set diagonal=1, rhs=0.
    // We'll do it on the triplet list: remove triplets in fixed columns, add identity.
    std::vector<Eigen::Triplet<double>> trips2;
    for (const auto& tr : trips) {
        if (fixed.count(tr.col()) || fixed.count(tr.row())) {
            // keep diagonal identity for fixed rows; skip others
            if (tr.row() == tr.col() && fixed.count(tr.row())) {
                trips2.push_back(Eigen::Triplet<double>(tr.row(), tr.col(), 1.0));
            }
            // skip contributions involving fixed cols/rows otherwise
            continue;
        }
        trips2.push_back(tr);
    }
    // For non-fixed rows that had a fixed column contribution, we already dropped them (zero column approx).
    // Proper elimination: subtract K_i,fixed * 0 = 0, so rhs unchanged. Fine for homogeneous Dirichlet.
    std::cerr << "FEM dbg: N=" << N << " fixed=" << fixed.size() << " trips2=" << trips2.size() << " xmin=" << mesh_.vertices[0].p.x << std::endl;
    // print min/max coords
    double xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9;
    for (const auto& v : mesh_.vertices) { xmin=std::min(xmin,v.p.x); xmax=std::max(xmax,v.p.x); ymin=std::min(ymin,v.p.y); ymax=std::max(ymax,v.p.y); }
    std::cerr << "FEM dbg: x range [" << xmin << "," << xmax << "] y range [" << ymin << "," << ymax << "]" << std::endl;
    K_.setFromTriplets(trips2.begin(), trips2.end());
    for (int vi : fixed) rhs_[vi] = 0.0;

    // nodeMu for export: per-vertex mu (nearest element average). Use material mu_r.
    nodeMu_.assign(N, mu0());
    for (int vi = 0; vi < N; ++vi) {
        // find an element containing this vertex
        for (const auto& t : mesh_.triangles) {
            if (t.v[0]==vi || t.v[1]==vi || t.v[2]==vi) {
                int mid = t.material;
                double mu_r = 1.0;
                if (mid >= 0 && mid < (int)matData_.size()) mu_r = matData_[mid].mu_r;
                nodeMu_[vi] = mu0() * (mu_r > 0 ? mu_r : 1.0);
                break;
            }
        }
    }
    elemMu_.clear();
}

void FemEngine::solveLinear() {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(K_);
    if (solver.info() != Eigen::Success) {
        std::cerr << "FEM: factorization failed" << std::endl;
        A_.setZero(K_.rows());
        return;
    }
    A_ = solver.solve(rhs_);
    Eigen::VectorXd res = K_ * A_ - rhs_;
    if (solver.info() != Eigen::Success) {
        std::cerr << "FEM: solve failed" << std::endl;
        A_.setZero(K_.rows());
    }
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

    // raster sampling
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            double x = (ix + 0.5) * dx_;
            double y = (iy + 0.5) * dy_;
            int ti = findTri(mesh_, x, y);
            if (ti < 0) {
                Az(ix, iy) = 0; // missing -> 0 for now (status handling later)
                continue;
            }
            double l0,l1,l2;
            pointInTri(x,y,mesh_,ti,l0,l1,l2);
            const auto& t = mesh_.triangles[ti];
            double A = l0*A_[t.v[0]] + l1*A_[t.v[1]] + l2*A_[t.v[2]];
            Az(ix, iy) = A;
            // B at element centroid gradients -> evaluate B at (x,y) using element constant B
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            double signedArea = 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
            double inv2A = 1.0/(2.0*signedArea);
            double g0x=(p1.y-p2.y)*inv2A, g0y=(p2.x-p1.x)*inv2A;
            double g1x=(p2.y-p0.y)*inv2A, g1y=(p0.x-p2.x)*inv2A;
            double g2x=(p0.y-p1.y)*inv2A, g2y=(p1.x-p0.x)*inv2A;
            double Bx = A_[t.v[0]]*g0y + A_[t.v[1]]*g1y + A_[t.v[2]]*g2y; // Bx = dA/dy
            double By = -(A_[t.v[0]]*g0x + A_[t.v[1]]*g1x + A_[t.v[2]]*g2x); // By = -dA/dx
            double Bmag = std::hypot(Bx, By);
            double mu = elemMu_.empty() ? mu0() : elemMu_[ti];
            Hmag(ix, iy) = Bmag / mu;
            Mu(ix, iy) = mu;
            // Jz from mask material
            int mx = std::min(nx-1, std::max(0, (int)(x/dx_)));
            int my = std::min(ny-1, std::max(0, (int)(y/dy_)));
            int mid = (mx + my*nx_ < (int)mask_.size()) ? mask_[my*nx_+mx] : -1;
            double jz = 0;
            if (mid >= 0 && mid < (int)matData_.size()) jz = matData_[mid].jz;
            Jz(ix, iy) = jz;
            Ed(ix, iy) = (Bmag*Bmag)/(2.0*mu);
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
    MagneticFieldAnalyzer::writeMatrixCSV(AzT, outDir + "/Az/step_0001.csv");
    MagneticFieldAnalyzer::writeMatrixCSV(MuT, outDir + "/Mu/step_0001.csv");
    MagneticFieldAnalyzer::writeMatrixCSV(Ht, outDir + "/H/step_0001.csv");
    MagneticFieldAnalyzer::writeMatrixCSV(Jt, outDir + "/Jz/step_0001.csv");
    MagneticFieldAnalyzer::writeMatrixCSV(Et, outDir + "/EnergyDensity/step_0001.csv");

    // TIFF
    MagneticFieldAnalyzer::writeMatrixTIFF(AzT, outDir + "/Az/step_0001.tiff", opts);
    MagneticFieldAnalyzer::writeMatrixTIFF(MuT, outDir + "/Mu/step_0001.tiff", opts);
    MagneticFieldAnalyzer::writeMatrixTIFF(Ht, outDir + "/H/step_0001.tiff", opts);
    MagneticFieldAnalyzer::writeMatrixTIFF(Jt, outDir + "/Jz/step_0001.tiff", opts);
    MagneticFieldAnalyzer::writeMatrixTIFF(Et, outDir + "/EnergyDensity/step_0001.tiff", opts);

    // copy input image
    cv::Mat inp = cv::imread(imagePath_, cv::IMREAD_COLOR);
    if (!inp.empty()) cv::imwrite(outDir + "/InputImg/step_0001.png", inp);

    // mesh artifact
    std::ofstream mf(outDir + "/Mesh/mesh.json");
    mf << "{\"nodes\":" << mesh_.vertices.size() << ",\"triangles\":" << mesh_.triangles.size() << ",\"hash\":\"" << mesh_.hash() << "\"}";
    mf.close();

    // conditions.json minimal
    std::ofstream cf(outDir + "/conditions.json");
    cf << "{\"solver\":{\"engine\":\"fem\"},\"mesh\":{\"nodes\":" << mesh_.vertices.size()
       << ",\"triangles\":" << mesh_.triangles.size() << "}}";
    cf.close();

    // forces unavailable marker
    std::ofstream ff(outDir + "/Forces/step_0001.csv");
    ff << "material,force_x,force_y,torque,status\n";
    ff << "all,0,0,0,unavailable\n";
    ff.close();
}

int FemEngine::run(const std::string& outDir, FemEngineResult* info) {
    FemEngineResult res;
    std::string err;
    if (!parseConfig(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!loadImage(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!buildMesh(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    assemble(err);
    solveLinear();
    res.nodes = (int)mesh_.vertices.size();
    res.triangles = (int)mesh_.triangles.size();
    exportResults(outDir);
    res.exitCode = 0;
    if (info) *info = res;
    std::cout << "FEM analysis completed. nodes=" << res.nodes << " triangles=" << res.triangles << std::endl;
    return 0;
}

} // namespace fem
} // namespace omfdm
