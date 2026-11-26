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
        int n = 16;
        fs::path dir = fs::temp_directory_path() / "omfdm_fem_export_compat";
        fs::create_directories(dir);
        cv::Mat img(n, n, CV_8UC3, cv::Scalar(255,255,255));
        cv::imwrite((dir / "model.png").string(), img);
        double dx = 0.01 / (n - 1);
        std::ofstream c(dir / "config.yaml");
        c << "coordinate_system: cartesian\n";
        c << "mesh: { dx: " << dx << ", dy: " << dx << " }\n";
        c << "materials:\n  air: { rgb: [255,255,255], mu_r: 1.0, jz: 0.0 }\n";
        c << "boundary_conditions:\n  left: { type: dirichlet, value: 0.0 }\n  right: { type: dirichlet, value: 0.0 }\n";
        c << "  bottom: { type: dirichlet, value: 0.0 }\n  top: { type: dirichlet, value: 0.0 }\n";
        c << "solver:\n  engine: fem\n";
        c << "  fem:\n    mesh: { min_spacing: " << dx << ", max_spacing: " << (dx*3) << ", min_angle_deg: 10 }\n";
        c << "export: { format: csv, async: false }\n";
        FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
        FemEngineResult info;
        int rc = eng.run((dir / "out").string(), &info);
        if (rc != 0) return fail("export_compat: engine failed");
        // Expected export files
        for (const char* sub : {"Az", "Mu", "H", "Jz", "EnergyDensity"}) {
            if (!fs::exists(dir / "out" / sub / "step_0001.tiff")) return fail(std::string("export_compat: missing ") + sub + " tiff");
        }
        if (!fs::exists(dir / "out" / "Forces" / "step_0001.csv")) return fail("export_compat: missing forces csv");
        if (!fs::exists(dir / "out" / "conditions.json")) return fail("export_compat: missing conditions.json");
        std::cout << "All fem_export_compat checks passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}