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
#include <map>

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

    // Solve (linear or nonlinear depending on materials). Returns 0 on success.
    int solve();

    // Accessors for tests.
    const FemMesh& mesh() const { return mesh_; }
    const Eigen::VectorXd& solution() const { return A_; }
    const std::vector<double>& nodeMu() const { return nodeMu_; }
    const std::vector<double>& elementMu() const { return elemMu_; }
    std::array<double,2> remanentMagnetization(int materialIndex) const;
    double residual() const { return lastResidual_; }
    bool converged() const { return lastConverged_; }
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
        double magAngle = 0.0;
        // B-H table (optional): pairs (H, B), monotone.
        std::vector<double> bhH, bhB;
        bool hasBH() const { return bhH.size() >= 2 && bhH.size() == bhB.size(); }
    };
    std::vector<MatData> matData_;
    std::vector<double> nodeMu_;   // per-vertex mu (for export)
    std::vector<double> jzScale_;  // per-material current-conservation scale for thin regions
    std::vector<double> elemMu_;   // per-element secant mu
    std::vector<std::vector<int>> vertexTriangles_; // vertex -> incident elements
    std::vector<std::vector<int>> triangleGrid_;    // uniform point-location grid
    double triangleCell_ = 0.0;
    int gridNx_ = 0, gridNy_ = 0;

    Eigen::VectorXd A_;            // solution (nodal A_z)
    Eigen::VectorXd rhs_;
    Eigen::SparseMatrix<double, Eigen::RowMajor> K_;
    std::function<double(double,double)> sourceFn_; // optional MMS source

    // parsed mesh params
    std::string engine_ = "fem";
    MeshBuildOptions meshOpts_;
    // nonlinear config
    bool nonlinearEnabled_ = true;
    int nonlinearMaxIter_ = 50;
    double nonlinearTol_ = 5e-4;
    double relaxation_ = 0.7;
    std::string jacobianMode_ = "tangent"; // tangent | secant (secant->tangent fallback)
    double lineSearchC_ = 1e-4;
    double lineSearchRho_ = 0.65;
    int lineSearchMax_ = 50;
    // Robin BCs: list of (edge endpoints index pair, alpha,beta,gamma)
    struct RobinEdge { int i, j; double alpha,beta,gamma; };
    std::vector<RobinEdge> robinEdges_;
    std::vector<std::pair<int,int>> periodicPairs_;
    std::vector<int> periodicSigns_; // +1 periodic, -1 antiperiodic
    // refinement config
    bool refinementEnabled_ = false;
    int refinementMaxRounds_ = 8;
    double refinementTheta_ = 0.5;
    double refinementTarget_ = 1e-3;
    std::vector<Point2D> extraInteriorPoints_;
    int lastIterations_ = 0;
    double lastResidual_ = 0.0;
    bool lastConverged_ = false;

    // helpers
    bool anyNonlinear() const;
    void parseNonlinearConfig();
    void parseRobinPeriodic();
    double evalNu(double Bmag, int mid) const;
    double evalDnu(double Bmag, int mid) const;
    double coEnergyDensity(double Bmag, int mid) const;
    std::map<int,double> dirichletNodes() const;
    std::vector<int> periodicDofRoots(const std::map<int,double>& fixed,
                                      std::vector<int>& freeNodes,
                                      std::vector<int>& signs) const;
    void assembleJacobian(Eigen::SparseMatrix<double,Eigen::RowMajor>& J) const;

    bool parseConfig(std::string& err);
    bool loadImage(std::string& err);
    bool buildMesh(std::string& err);
    void buildTriangleIndex();
    int findTriangle(double x, double y) const;
    void assemble();
    bool solveLinear();
    void refine(const std::string& outDir);
    void exportResults(const std::string& outDir);
};

} // namespace fem
} // namespace omfdm

#endif // FEMENGINE_H
