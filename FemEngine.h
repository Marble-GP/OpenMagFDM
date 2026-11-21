#pragma once
// Phase 1: linear (and basic nonlinear-source) FEM engine for 2D magnetostatics.
// Invoked from MagFDMsolver when solver.engine == "fem".
#ifndef FEMENGINE_H
#define FEMENGINE_H

#include "FemMesh.h"

#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>

namespace omfdm {
namespace fem {

struct FemEngineResult {
    int exitCode = 0;
    std::string error;
    int nodes = 0;
    int triangles = 0;
    double solveTimeMs = 0;
};

class FemEngine {
public:
    FemEngine(const std::string& configPath, const std::string& imagePath);
    // Run the full pipeline and write results under outDir. Returns exit code.
    int run(const std::string& outDir, FemEngineResult* info = nullptr);

    // Accessors for tests.
    const FemMesh& mesh() const { return mesh_; }
    const Eigen::VectorXd& solution() const { return A_; }
    const std::vector<double>& nodeMu() const { return nodeMu_; }
    // Test hook: override the source J_z(x,y) (used by MMS).
    void setSourceFunction(std::function<double(double,double)> f) { sourceFn_ = f; }
    double dx() const { return dx_; }
    double dy() const { return dy_; }
    int nx() const { return nx_; }
    int ny() const { return ny_; }

private:
    std::string configPath_;
    std::string imagePath_;
    YAML::Node config_;

    // mesh / grid
    cv::Mat imageRaw_;
    cv::Mat imageFlippedRgb_;
    std::vector<MaterialDesc> materials_;
    std::vector<int> mask_;
    FemMesh mesh_;
    int nx_ = 0, ny_ = 0;
    double dx_ = 0, dy_ = 0;

    // per-material data parsed from config
    struct MatData {
        std::string name;
        std::array<uint8_t,3> rgb = {255,255,255};
        double mu_r = 1.0;
        double jz = 0.0;
        bool hasMagnetization = false;
        double Br = 0.0;
        double magAngle = 0.0; // radians
    };
    std::vector<MatData> matData_;
    std::vector<double> nodeMu_;   // per-vertex mu (for export)
    std::vector<double> elemMu_;   // per-element mu

    Eigen::VectorXd A_;            // solution (nodal A_z)
    Eigen::VectorXd rhs_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> K_;
    std::function<double(double,double)> sourceFn_; // optional MMS source

    // parsed mesh params
    std::string engine_ = "fem";
    MeshBuildOptions meshOpts_;

    bool parseConfig(std::string& err);
    bool loadImage(std::string& err);
    bool buildMesh(std::string& err);
    void assemble(std::string& err);
    void solveLinear();
    void exportResults(const std::string& outDir);
};

} // namespace fem
} // namespace omfdm

#endif // FEMENGINE_H
