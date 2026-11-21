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
            if (m["B-H"]) {
                const auto& bh = m["B-H"];
                if (bh.IsSequence() && bh.size() == 2) {
                    for (auto v : bh[0]) md.bhH.push_back(v.as<double>());
                    for (auto v : bh[1]) md.bhB.push_back(v.as<double>());
                }
            }
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
    parseNonlinearConfig();
    parseRobinPeriodic();
    // refinement config
    refinementEnabled_ = false; refinementMaxRounds_ = 8; refinementTheta_ = 0.5; refinementTarget_ = 1e-3;
    if (config_["solver"] && config_["solver"]["fem"] && config_["solver"]["fem"]["refinement"]) {
        const auto& r = config_["solver"]["fem"]["refinement"];
        if (r["enabled"]) refinementEnabled_ = r["enabled"].as<bool>();
        if (r["max_rounds"]) refinementMaxRounds_ = r["max_rounds"].as<int>();
        if (r["theta"]) refinementTheta_ = r["theta"].as<double>();
        if (r["target_relative_error"]) refinementTarget_ = r["target_relative_error"].as<double>();
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

    // Robin BC contributions (natural boundary).
    for (const auto& re : robinEdges_) {
        const auto& pi = mesh_.vertices[re.i].p;
        const auto& pj = mesh_.vertices[re.j].p;
        double len = std::hypot(pj.x-pi.x, pj.y-pi.y);
        if (len < 1e-30) continue;
        if (std::fabs(re.beta) < 1e-30) continue;
        double coef = (re.alpha / re.beta) * len / 2.0; // per endpoint (trapezoid)
        // stiffness: (alpha/beta)*len/2 on diagonal of each endpoint? Use full: add to (i,i),(j,j) and (i,j),(j,i)
        trips.push_back(Eigen::Triplet<double>(re.i, re.i, coef));
        trips.push_back(Eigen::Triplet<double>(re.j, re.j, coef));
        trips.push_back(Eigen::Triplet<double>(re.i, re.j, coef));
        trips.push_back(Eigen::Triplet<double>(re.j, re.i, coef));
        double rhsContrib = (re.gamma / re.beta) * len / 2.0;
        rhs_[re.i] += rhsContrib;
        rhs_[re.j] += rhsContrib;
    }
    // Periodic BC via penalty.
    for (const auto& pr : periodicPairs_) {
        const double P = 1e6;
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.first, P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.second, P));
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.second, -P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.first, -P));
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

    // forces: Amperian per-material
    std::map<std::string, std::tuple<double,double,double,int>> frc; // name -> fx,fy,torque,count
    // free current + induced magnetization per element
    for (const auto& tri : mesh_.triangles) {
        const auto& p0 = mesh_.vertices[tri.v[0]].p;
        const auto& p1 = mesh_.vertices[tri.v[1]].p;
        const auto& p2 = mesh_.vertices[tri.v[2]].p;
        double signedArea = 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
        double area = std::fabs(signedArea);
        if (area < 1e-30) continue;
        double inv2A = 1.0/(2.0*signedArea);
        double g[3][2] = { {(p1.y-p2.y)*inv2A, (p2.x-p1.x)*inv2A},
                           {(p2.y-p0.y)*inv2A, (p0.x-p2.x)*inv2A},
                           {(p0.y-p1.y)*inv2A, (p1.x-p0.x)*inv2A} };
        int vi[3]={tri.v[0],tri.v[1],tri.v[2]};
        double Bx=0,By=0;
        for (int a=0;a<3;++a){ Bx += A_[vi[a]]*g[a][1]; By += -A_[vi[a]]*g[a][0]; }
        int mid = tri.material;
        std::string matName = (mid>=0 && mid<(int)matData_.size()) ? matData_[mid].name : "air";
        double jz = (mid>=0 && mid<(int)matData_.size()) ? matData_[mid].jz : 0.0;
        double cx = (p0.x+p1.x+p2.x)/3.0, cy=(p0.y+p1.y+p2.y)/3.0;
        if (jz != 0.0) {
            double fx = jz * By * area;
            double fy = -jz * Bx * area;
            double tq = cx*fy - cy*fx;
            auto& r = frc[matName];
            std::get<0>(r)+=fx; std::get<1>(r)+=fy; std::get<2>(r)+=tq; std::get<3>(r)++;
        }
    }
    // PM surface currents
    for (const auto& c : mesh_.contours) {
        int mid = c.materialInside;
        if (mid < 0 || mid >= (int)matData_.size()) continue;
        if (!matData_[mid].hasMagnetization) continue;
        std::string matName = matData_[mid].name;
        double Br = matData_[mid].Br;
        double magAngle = matData_[mid].magAngle;
        double Mx = (Br/mu0())*std::cos(magAngle);
        double My = (Br/mu0())*std::sin(magAngle);
        size_t n = c.points.size();
        for (size_t i=0;i<n;++i) {
            const Point2D& p0 = c.points[i];
            const Point2D& p1 = c.points[(i+1)%n];
            double dxs=p1.x-p0.x, dys=p1.y-p0.y;
            double len=std::hypot(dxs,dys);
            if (len<1e-30) continue;
            double nx,ny;
            if (!c.isHole) { nx=dys/len; ny=-dxs/len; }
            else { nx=-dys/len; ny=dxs/len; }
            double Kz = Mx*ny - My*nx;
            // B at midpoint
            double mx=(p0.x+p1.x)/2, my=(p0.y+p1.y)/2;
            int ti = findTri(mesh_, mx, my);
            double Bx=0,By=0;
            if (ti>=0) {
                const auto& t = mesh_.triangles[ti];
                const auto& q0 = mesh_.vertices[t.v[0]].p;
                const auto& q1 = mesh_.vertices[t.v[1]].p;
                const auto& q2 = mesh_.vertices[t.v[2]].p;
                double sA = 0.5*((q1.x-q0.x)*(q2.y-q0.y)-(q2.x-q0.x)*(q1.y-q0.y));
                double i2A = 1.0/(2.0*sA);
                double g[3][2] = { {(q1.y-q2.y)*i2A,(q2.x-q1.x)*i2A},
                                   {(q2.y-q0.y)*i2A,(q0.x-q2.x)*i2A},
                                   {(q0.y-q1.y)*i2A,(q1.x-q0.x)*i2A} };
                for (int a=0;a<3;++a){ Bx += A_[t.v[a]]*g[a][1]; By += -A_[t.v[a]]*g[a][0]; }
            }
            double fx = -Kz * By * len;
            double fy = Kz * Bx * len;
            double tq = mx*fy - my*fx;
            auto& r = frc[matName];
            std::get<0>(r)+=fx; std::get<1>(r)+=fy; std::get<2>(r)+=tq; std::get<3>(r)++;
        }
    }
    std::ofstream ff(outDir + "/Forces/step_0001.csv");
    ff << "material,force_x,force_y,torque,status\n";
    double tFx=0,tFy=0,tTq=0;
    for (const auto& kv : frc) {
        ff << kv.first << "," << std::get<0>(kv.second) << "," << std::get<1>(kv.second) << "," << std::get<2>(kv.second) << ",ok\n";
        tFx += std::get<0>(kv.second); tFy += std::get<1>(kv.second); tTq += std::get<2>(kv.second);
    }
    ff << "all," << tFx << "," << tFy << "," << tTq << ",ok\n";
    ff.close();

    // total energy
    double E = 0;
    for (size_t ti=0; ti<mesh_.triangles.size(); ++ti) {
        const auto& t = mesh_.triangles[ti];
        const auto& p0 = mesh_.vertices[t.v[0]].p;
        const auto& p1 = mesh_.vertices[t.v[1]].p;
        const auto& p2 = mesh_.vertices[t.v[2]].p;
        double area = std::fabs(0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y)));
        int mid = t.material;
        double mu = (mid>=0 && mid<(int)matData_.size()) ? mu0()*(matData_[mid].mu_r>0?matData_[mid].mu_r:1.0) : mu0();
        // B at centroid
        double cx=(p0.x+p1.x+p2.x)/3.0, cy=(p0.y+p1.y+p2.y)/3.0;
        int ti2 = findTri(mesh_, cx, cy);
        double Bx=0,By=0;
        if (ti2>=0) {
            const auto& q = mesh_.triangles[ti2];
            const auto& a=mesh_.vertices[q.v[0]].p, &b=mesh_.vertices[q.v[1]].p, &c=mesh_.vertices[q.v[2]].p;
            double sA=0.5*((b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y));
            double i2A=1.0/(2.0*sA);
            double g[3][2]={{(b.y-c.y)*i2A,(c.x-b.x)*i2A},{(c.y-a.y)*i2A,(a.x-c.x)*i2A},{(a.y-b.y)*i2A,(b.x-a.x)*i2A}};
            for(int k=0;k<3;++k){ Bx += A_[q.v[k]]*g[k][1]; By += -A_[q.v[k]]*g[k][0]; }
        }
        E += (Bx*Bx+By*By)/(2.0*mu) * area;
    }
    // write energy to a small file
    std::ofstream ef(outDir + "/Energy.txt"); ef << E << std::endl; ef.close();

    // flux linkage (material-pair variant)
    if (config_["flux_linkage"]) {
        std::ofstream fl(outDir + "/FluxLinkage/flux_linkage.csv");
        fl << "name,flux\n";
        for (const auto& p : config_["flux_linkage"]) {
            std::string name = p["name"] ? p["name"].as<std::string>() : "path";
            bool useMat = p["use_material"] && p["use_material"].as<bool>();
            if (useMat) {
                std::string ma = p["material_a"].as<std::string>("");
                std::string mb = p["material_b"].as<std::string>("");
                double sa=0,sb=0; int ca=0,cb=0;
                for (int iy=0;iy<ny_;++iy) for (int ix=0;ix<nx_;++ix) {
                    int midx = mask_[iy*nx_+ix];
                    std::string mname = (midx>=0 && midx<(int)matData_.size()) ? matData_[midx].name : "air";
                    double x=(ix+0.5)*dx_, y=(iy+0.5)*dy_;
                    int ti=findTri(mesh_,x,y);
                    double A=0;
                    if (ti>=0){ double l0,l1,l2; pointInTri(x,y,mesh_,ti,l0,l1,l2); const auto&t=mesh_.triangles[ti]; A=l0*A_[t.v[0]]+l1*A_[t.v[1]]+l2*A_[t.v[2]]; }
                    if (mname==ma){sa+=A;ca++;}
                    if (mname==mb){sb+=A;cb++;}
                }
                double phi = (ca>0?sa/ca:0) - (cb>0?sb/cb:0);
                fl << name << "," << phi << "\n";
            } else {
                fl << name << ",0\n";
            }
        }
        fl.close();
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
    robinEdges_.clear(); periodicPairs_.clear();
    if (!config_["boundary_conditions"]) return;
    const auto& bc = config_["boundary_conditions"];
    double Lx = nx_ * dx_, Ly = ny_ * dy_;
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
    if (isPeriodic("left") && isPeriodic("right")) {
        std::vector<int> ls, rs;
        for (int vi=0; vi<(int)mesh_.vertices.size(); ++vi) {
            if (std::fabs(mesh_.vertices[vi].p.x) < 1e-9) ls.push_back(vi);
            if (std::fabs(mesh_.vertices[vi].p.x - (nx_*dx_)) < 1e-9) rs.push_back(vi);
        }
        for (int li : ls) for (int ri : rs) {
            if (std::fabs(mesh_.vertices[li].p.y - mesh_.vertices[ri].p.y) < 1e-9) { periodicPairs_.push_back({li,ri}); break; }
        }
    }
    if (isPeriodic("bottom") && isPeriodic("top")) {
        std::vector<int> bs, ts;
        for (int vi=0; vi<(int)mesh_.vertices.size(); ++vi) {
            if (std::fabs(mesh_.vertices[vi].p.y) < 1e-9) bs.push_back(vi);
            if (std::fabs(mesh_.vertices[vi].p.y - (ny_*dy_)) < 1e-9) ts.push_back(vi);
        }
        for (int bi : bs) for (int ti : ts) {
            if (std::fabs(mesh_.vertices[bi].p.x - mesh_.vertices[ti].p.x) < 1e-9) { periodicPairs_.push_back({bi,ti}); break; }
        }
    }
}

bool FemEngine::anyNonlinear() const {
    for (const auto& md : matData_) if (md.hasBH()) return true;
    return false;
}

double FemEngine::evalNu(double Bmag, int mid) const {
    if (mid < 0 || mid >= (int)matData_.size()) return 1.0 / mu0();
    const auto& md = matData_[mid];
    if (!md.hasBH()) return 1.0 / (mu0() * (md.mu_r > 0 ? md.mu_r : 1.0));
    const auto& H = md.bhH; const auto& Bt = md.bhB;
    if (Bt.empty()) return 1.0 / (mu0() * (md.mu_r > 0 ? md.mu_r : 1.0));
    double Bc = Bmag;
    if (Bc < Bt.front()) Bc = Bt.front();
    if (Bc > Bt.back()) Bc = Bt.back();
    size_t i = 0;
    while (i + 1 < Bt.size() && Bt[i+1] < Bc) ++i;
    double t = (Bt[i+1]-Bt[i]) > 1e-30 ? (Bc - Bt[i])/(Bt[i+1]-Bt[i]) : 0.0;
    double Hc = H[i] + t*(H[i+1]-H[i]);
    if (Bc < 1e-30) return 1.0 / (mu0() * (md.mu_r > 0 ? md.mu_r : 1.0));
    return Hc / Bc; // nu = H/B
}

double FemEngine::evalDnu(double Bmag, int mid) const {
    if (mid < 0 || mid >= (int)matData_.size()) return 0.0;
    const auto& md = matData_[mid];
    if (!md.hasBH()) return 0.0;
    const auto& H = md.bhH; const auto& Bt = md.bhB;
    if (Bt.size() < 2) return 0.0;
    double Bc = Bmag;
    if (Bc < Bt.front()) Bc = Bt.front();
    if (Bc > Bt.back()) Bc = Bt.back();
    size_t i = 0;
    while (i + 1 < Bt.size() && Bt[i+1] < Bc) ++i;
    double dB = Bt[i+1]-Bt[i];
    if (dB < 1e-30) return 0.0;
    double dH = (H[i+1]-H[i])/dB;
    double Hc = H[i] + (Bc-Bt[i])/dB*(H[i+1]-H[i]);
    if (Bc < 1e-30) return 0.0;
    return (dH*Bc - Hc)/(Bc*Bc);
}

void FemEngine::assembleJacobian(Eigen::SparseMatrix<double,Eigen::RowMajor>& J) const {
    int N = (int)mesh_.vertices.size();
    J.resize(N,N);
    std::vector<Eigen::Triplet<double>> trips;
    // element tangent
    for (const auto& tri : mesh_.triangles) {
        int i0=tri.v[0],i1=tri.v[1],i2=tri.v[2];
        const auto& p0=mesh_.vertices[i0].p; const auto& p1=mesh_.vertices[i1].p; const auto& p2=mesh_.vertices[i2].p;
        double signedArea = 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
        if (std::fabs(signedArea) < 1e-30) continue;
        double inv2A = 1.0/(2.0*signedArea);
        double g[3][2] = { {(p1.y-p2.y)*inv2A, (p2.x-p1.x)*inv2A},
                           {(p2.y-p0.y)*inv2A, (p0.x-p2.x)*inv2A},
                           {(p0.y-p1.y)*inv2A, (p1.x-p0.x)*inv2A} };
        int vi[3]={i0,i1,i2};
        int mid = tri.material;
        // B at this A
        double Bx=0,By=0;
        for (int a=0;a<3;++a){ Bx += A_[vi[a]]*g[a][1]; By += -A_[vi[a]]*g[a][0]; } // Bx=dA/dy, By=-dA/dx
        double Bmag = std::hypot(Bx,By);
        double nu = evalNu(Bmag, mid);
        double dnu = evalDnu(Bmag, mid);
        double area = std::fabs(signedArea);
        for (int a=0;a<3;++a)
            for (int c=0;c<3;++c) {
                double base = nu * area * (g[a][0]*g[c][0] + g[a][1]*g[c][1]);
                double tangent = 0.0;
                if (Bmag > 1e-12) {
                    double gcB = g[c][0]*Bx + g[c][1]*By; // g_c . B (note g is grad phi, B = sum A gradphi rotated; check)
                    // Actually B = (sum A dphi/dy, -sum A dphi/dx). g·B = g_x*Bx + g_y*By.
                    double gaB = g[a][0]*Bx + g[a][1]*By;
                    tangent = (dnu / Bmag) * area * gcB * gaB;
                }
                // account for periodic/robin? base assembly handles BC separately
                trips.push_back(Eigen::Triplet<double>(vi[a], vi[c], base + tangent));
            }
    }
    // Robin BC contributions to Jacobian
    for (const auto& re : robinEdges_) {
        const auto& pi = mesh_.vertices[re.i].p;
        const auto& pj = mesh_.vertices[re.j].p;
        double len = std::hypot(pj.x-pi.x, pj.y-pi.y);
        if (len < 1e-30) continue;
        if (std::fabs(re.beta) < 1e-30) continue;
        double coef = (re.alpha / re.beta) * len * 0.5;
        trips.push_back(Eigen::Triplet<double>(re.i, re.i, coef));
        trips.push_back(Eigen::Triplet<double>(re.j, re.j, coef));
    }
    // periodic: handled by penalty (add large diagonal to slave? use elimination instead)
    // For Phase 2, apply periodic via penalty: A_i - A_j = 0 -> add penalty to both.
    for (const auto& pr : periodicPairs_) {
        // penalty: (A_i - A_j)^2 * P added to residual; Jacobian gets 2P on diag and -P off-diag
        const double P = 1e6;
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.first, P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.second, P));
        trips.push_back(Eigen::Triplet<double>(pr.first, pr.second, -P));
        trips.push_back(Eigen::Triplet<double>(pr.second, pr.first, -P));
    }
    J.setFromTriplets(trips.begin(), trips.end());
}

int FemEngine::solve() {
    // apply periodic elimination? We use penalty, so no elimination.
    if (!anyNonlinear() || !nonlinearEnabled_) {
        assemble(std::string());
        solveLinear();
        return 0;
    }
    // nonlinear Newton
    A_.setZero(mesh_.vertices.size());
    double rhsNorm = 0;
    // initial residual
    for (int iter = 0; iter < nonlinearMaxIter_; ++iter) {
        assemble(std::string());
        Eigen::VectorXd R = K_ * A_ - rhs_;
        if (iter == 0) rhsNorm = rhs_.norm();
        double rrel = (rhsNorm > 1e-30) ? R.norm() / rhsNorm : R.norm();
        if (rrel < nonlinearTol_) {
            return 0;
        }
        Eigen::SparseMatrix<double,Eigen::RowMajor> J;
        assembleJacobian(J);
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(J);
        if (solver.info() != Eigen::Success) {
            // fallback to Picard (diagonal)
            // Picard: solve K(A) dA = -R with K frozen
            Eigen::SparseLU<Eigen::SparseMatrix<double>> s2; s2.compute(K_);
            if (s2.info() != Eigen::Success) return 1;
            Eigen::VectorXd dA = s2.solve(-R);
            // line search
            double alpha = 1.0;
            Eigen::VectorXd A0 = A_;
            double R0 = R.norm();
            for (int ls = 0; ls < lineSearchMax_; ++ls) {
                A_ = A0 + alpha * dA;
                assemble(std::string());
                Eigen::VectorXd Rn = K_ * A_ - rhs_;
                if (Rn.norm() < R0 * (1 - lineSearchC_ * alpha) || Rn.norm() < R0) break;
                alpha *= lineSearchRho_;
                if (alpha < 1e-12) break;
            }
            continue;
        }
        Eigen::VectorXd dA = solver.solve(-R);
        if (solver.info() != Eigen::Success) return 1;
        // line search
        Eigen::VectorXd A0 = A_;
        double R0 = R.norm();
        double alpha = 1.0;
        for (int ls = 0; ls < lineSearchMax_; ++ls) {
            A_ = A0 + alpha * dA;
            assemble(std::string());
            Eigen::VectorXd Rn = K_ * A_ - rhs_;
            if (Rn.norm() <= R0) break;
            alpha *= lineSearchRho_;
            if (alpha < 1e-12) { A_ = A0; assemble(std::string()); break; }
        }
    }
    // final convergence check
    assemble(std::string());
    Eigen::VectorXd R = K_ * A_ - rhs_;
    double rrel = (rhsNorm > 1e-30) ? R.norm() / rhsNorm : R.norm();
    return (rrel < nonlinearTol_) ? 0 : 1;
}

void FemEngine::refine() {
    if (!refinementEnabled_) return;
    std::vector<Point2D> extra = extraInteriorPoints_;
    for (int round = 0; round < refinementMaxRounds_; ++round) {
        solve();
        // Compute ZZ indicator per element.
        int ne = (int)mesh_.triangles.size();
        std::vector<double> eta(ne, 0.0);
        // nodal recovered gradient (area-weighted)
        int N = (int)mesh_.vertices.size();
        std::vector<double> gx(N,0), gy(N,0), w(N,0);
        for (int ti = 0; ti < ne; ++ti) {
            const auto& t = mesh_.triangles[ti];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            double signedArea = 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
            double area = std::fabs(signedArea);
            if (area < 1e-30) continue;
            double inv2A = 1.0/(2.0*signedArea);
            double g[3][2] = { {(p1.y-p2.y)*inv2A, (p2.x-p1.x)*inv2A},
                               {(p2.y-p0.y)*inv2A, (p0.x-p2.x)*inv2A},
                               {(p0.y-p1.y)*inv2A, (p1.x-p0.x)*inv2A} };
            for (int a=0;a<3;++a) {
                double ga = g[a][0]*A_[t.v[a]]? 0 : 0; // placeholder
                (void)ga;
                gx[t.v[a]] += g[a][0]*area;
                gy[t.v[a]] += g[a][1]*area;
                w[t.v[a]] += area;
            }
        }
        for (int vi=0; vi<N; ++vi) { if (w[vi]>0){ gx[vi]/=w[vi]; gy[vi]/=w[vi]; } }

        double total = 0;
        for (int ti = 0; ti < ne; ++ti) {
            const auto& t = mesh_.triangles[ti];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            double signedArea = 0.5*((p1.x-p0.x)*(p2.y-p0.y)-(p2.x-p0.x)*(p1.y-p0.y));
            double area = std::fabs(signedArea);
            if (area < 1e-30) continue;
            double inv2A = 1.0/(2.0*signedArea);
            double g[3][2] = { {(p1.y-p2.y)*inv2A, (p2.x-p1.x)*inv2A},
                               {(p2.y-p0.y)*inv2A, (p0.x-p2.x)*inv2A},
                               {(p0.y-p1.y)*inv2A, (p1.x-p0.x)*inv2A} };
            // element gradient (area-weighted avg of nodal? use vertex gradient from nodal recovery)
            // eta = area * sum_a (grad_elem[a] - grad_recovered[a])^2 approximated
            double e = 0;
            for (int a=0;a<3;++a) {
                double dgx = g[a][0] - gx[t.v[a]];
                double dgy = g[a][1] - gy[t.v[a]];
                e += (dgx*dgx + dgy*dgy);
            }
            eta[ti] = area * e / 3.0;
            total += eta[ti];
        }
        if (total < 1e-30) break;
        // Dörfler marking
        std::vector<int> order(ne);
        for (int i=0;i<ne;++i) order[i]=i;
        std::sort(order.begin(), order.end(), [&](int a, int b){ return eta[a] > eta[b]; });
        double cum = 0, target = refinementTheta_ * total;
        std::vector<char> marked(ne, 0);
        for (int ti : order) {
            if (cum >= target) break;
            marked[ti] = 1;
            cum += eta[ti];
        }
        // collect circumcenters
        std::vector<Point2D> newPts;
        for (int ti = 0; ti < ne; ++ti) {
            if (!marked[ti]) continue;
            const auto& t = mesh_.triangles[ti];
            const auto& p0 = mesh_.vertices[t.v[0]].p;
            const auto& p1 = mesh_.vertices[t.v[1]].p;
            const auto& p2 = mesh_.vertices[t.v[2]].p;
            // circumcenter
            double D = 2*(p0.x*(p1.y-p2.y)+p1.x*(p2.y-p0.y)+p2.x*(p0.y-p1.y));
            if (std::fabs(D) < 1e-30) continue;
            double ux = ((p0.x*p0.x+p0.y*p0.y)*(p1.y-p2.y)+(p1.x*p1.x+p1.y*p1.y)*(p2.y-p0.y)+(p2.x*p2.x+p2.y*p2.y)*(p0.y-p1.y))/D;
            double uy = ((p0.x*p0.x+p0.y*p0.y)*(p2.x-p1.x)+(p1.x*p1.x+p1.y*p1.y)*(p0.x-p2.x)+(p2.x*p2.x+p2.y*p2.y)*(p1.x-p0.x))/D;
            // check inside domain
            double Lx = nx_*dx_, Ly = ny_*dy_;
            if (ux < 0 || ux > Lx || uy < 0 || uy > Ly) continue;
            newPts.push_back({ux,uy});
        }
        if (newPts.empty()) break;
        // dedup against existing extra
        for (const auto& p : newPts) extra.push_back(p);
        // rebuild mesh
        extraInteriorPoints_ = extra;
        std::string err;
        if (!buildMesh(err)) break;
        // interpolate solution (simple: nearest vertex)
        // keep A_ sized; solve() will be called next iteration
        A_.setZero(mesh_.vertices.size());
    }
    extraInteriorPoints_ = extra;
}

int FemEngine::run(const std::string& outDir, FemEngineResult* info) {
    FemEngineResult res;
    std::string err;
    if (!parseConfig(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!loadImage(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (!buildMesh(err)) { res.exitCode = 1; res.error = err; if(info)*info=res; std::cerr << err << std::endl; return 1; }
    if (refinementEnabled_) { refine(); solve(); }
    else { solve(); }
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
