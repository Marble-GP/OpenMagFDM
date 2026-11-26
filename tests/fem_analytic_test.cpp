#include "FemEngine.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

using namespace omfdm::fem;
static int fail(const std::string& msg) { std::cerr << "FAIL: " << msg << std::endl; return 1; }
namespace fs = std::filesystem;

static void makeFiles(const fs::path& dir, int n) {
    fs::create_directories(dir);
    cv::Mat img(n, n, CV_8UC3, cv::Scalar(255,255,255));
    cv::imwrite((dir / "model.png").string(), img);
    double dx = 0.01 / (n - 1);
    std::ofstream c(dir / "config.yaml");
    c << "coordinate_system: cartesian\n";
    c << "mesh: { dx: " << dx << ", dy: " << dx << " }\n";
    c << "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n";
    c << "boundary_conditions:\n";
    c << "  left: { type: dirichlet, value: 0.0 }\n  right: { type: dirichlet, value: 0.0 }\n";
    c << "  bottom: { type: dirichlet, value: 0.0 }\n  top: { type: dirichlet, value: 0.0 }\n";
    c << "solver:\n  engine: fem\n";
    c << "  fem:\n    mesh: { min_spacing: " << dx << ", max_spacing: " << (dx*3) << ", min_angle_deg: 10 }\n";
    c << "export: { format: csv, async: false }\n";
}

int main() {
    try {
        // Analytic case: Laplace equation with zero source -> A == 0 everywhere.
        fs::path dir = fs::temp_directory_path() / "omfdm_fem_analytic";
        makeFiles(dir, 24);
        FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
        FemEngineResult info;
        int rc = eng.run((dir / "out").string(), &info);
        if (rc != 0) return fail("analytic: engine failed");
        double maxA = 0;
        for (int i = 0; i < eng.solution().size(); ++i) maxA = std::max(maxA, std::fabs(eng.solution()[i]));
        if (maxA > 1e-10) return fail("analytic: nonzero field " + std::to_string(maxA));
        // Exported Az CSV must exist.
        if (!fs::exists(dir / "out" / "Az" / "step_0001.csv")) return fail("analytic: missing Az csv");
        std::cout << "All fem_analytic checks passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}