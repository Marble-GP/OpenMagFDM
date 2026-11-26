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

int main() {
    try {
        int n = 24;
        fs::path dir = fs::temp_directory_path() / "omfdm_fem_adaptive";
        fs::create_directories(dir);
        cv::Mat img(n, n, CV_8UC3, cv::Scalar(255,255,255));
        for (int y = n/3; y < 2*n/3; ++y)
            for (int x = n/3; x < 2*n/3; ++x) img.at<cv::Vec3b>(y,x) = cv::Vec3b(255,0,0);
        cv::imwrite((dir / "model.png").string(), img);
        double dx = 0.01 / (n - 1);
        std::ofstream c(dir / "config.yaml");
        c << "coordinate_system: cartesian\n";
        c << "mesh: { dx: " << dx << ", dy: " << dx << " }\n";
        c << "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n  iron: { rgb: [255,0,0], mu_r: 100.0, jz: 0.0 }\n";
        c << "boundary_conditions:\n  left: { type: dirichlet, value: 0.0 }\n  right: { type: dirichlet, value: 1.0 }\n";
        c << "  bottom: { type: dirichlet, value: 0.0 }\n  top: { type: dirichlet, value: 0.0 }\n";
        c << "solver:\n  engine: fem\n";
        c << "  fem:\n    mesh: { min_spacing: " << dx << ", max_spacing: " << (dx*4) << ", min_angle_deg: 10 }\n";
        c << "    refinement: { enabled: true, max_rounds: 3, theta: 0.5, target_relative_error: 1e-3 }\n";
        c << "export: { format: csv, async: false }\n";
        FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
        FemEngineResult info;
        int rc = eng.run((dir / "out").string(), &info);
        if (rc != 0) return fail("adaptive: engine failed rc=" + std::to_string(rc));
        std::cout << "All fem_adaptive checks passed. nodes=" << info.nodes << " triangles=" << info.triangles << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}