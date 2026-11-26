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
        fs::path dir = fs::temp_directory_path() / "omfdm_fem_nonlinear";
        fs::create_directories(dir);
        // small iron block in air
        int n = 32;
        cv::Mat img(n, n, CV_8UC3, cv::Scalar(255,255,255));
        for (int y = n/3; y < 2*n/3; ++y)
            for (int x = n/3; x < 2*n/3; ++x) img.at<cv::Vec3b>(y,x) = cv::Vec3b(0,0,255);
        cv::imwrite((dir / "model.png").string(), img);
        double dx = 0.01 / (n - 1);
        std::ofstream c(dir / "config.yaml");
        c << "coordinate_system: cartesian\n";
        c << "mesh: { dx: " << dx << ", dy: " << dx << " }\n";
        c << "materials:\n";
        c << "  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n";
        c << "  iron: { rgb: [0,0,255], mu_r: 1.0, jz: 1.0e6, B-H: [[0,100,1000],[0,0.5,1.2]] }\n";
        c << "boundary_conditions:\n";
        c << "  left: { type: dirichlet, value: 0.0 }\n  right: { type: dirichlet, value: 0.0 }\n";
        c << "  bottom: { type: dirichlet, value: 0.0 }\n  top: { type: dirichlet, value: 0.0 }\n";
        c << "solver:\n  engine: fem\n";
        c << "nonlinear_solver:\n  enabled: true\n";
        c << "  fem:\n    mesh: { min_spacing: " << dx << ", max_spacing: " << (dx*3) << ", min_angle_deg: 10 }\n";
        c << "export: { format: csv, async: false }\n";
        FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
        FemEngineResult info;
        int rc = eng.run((dir / "out").string(), &info);
        if (rc != 0) return fail("nonlinear: engine failed rc=" + std::to_string(rc));
        if (!eng.converged()) return fail("nonlinear: did not converge");
        std::cout << "All fem_nonlinear checks passed. residual=" << eng.residual() << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}