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
    // uniform air image (white), n x n
    cv::Mat img(n, n, CV_8UC3, cv::Scalar(255,255,255));
    cv::imwrite((dir / "model.png").string(), img);
    double dx = 0.01 / n; // domain Lx = 0.01
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
        // --- Test 1: zero source -> A ~ 0 ---
        {
            fs::path dir = fs::temp_directory_path() / "omfdm_fem_linear_zero";
            makeFiles(dir, 16);
            FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
            FemEngineResult info;
            int rc = eng.run((dir / "out").string(), &info);
            if (rc != 0) return fail("zero: engine failed");
            double maxA = 0;
            for (int i = 0; i < eng.solution().size(); ++i) maxA = std::max(maxA, std::fabs(eng.solution()[i]));
            if (maxA > 1e-10) return fail("zero: nonzero solution " + std::to_string(maxA));
            std::cout << "zero: OK maxA=" << maxA << std::endl;
        }

        // --- Test 2: MMS convergence ---
        const double L = 0.01;
        const double nu = 1.0 / (4.0 * M_PI * 1e-7); // mu_r = 1
        std::vector<double> errs;
        for (int n : {16, 32, 64}) {
            fs::path dir = fs::temp_directory_path() / ("omfdm_fem_mms_" + std::to_string(n));
            makeFiles(dir, n);
            FemEngine eng((dir / "config.yaml").string(), (dir / "model.png").string());
            eng.setSourceFunction([&](double x, double y) {
                return nu * (M_PI*M_PI/(L*L) + M_PI*M_PI/(L*L)) * std::sin(M_PI*x/L) * std::sin(M_PI*y/L);
            });
            FemEngineResult info;
            int rc = eng.run((dir / "out").string(), &info);
            if (rc != 0) return fail("mms: engine failed n=" + std::to_string(n));
            double sum = 0; int cnt = 0;
            for (int i = 0; i < eng.solution().size(); ++i) {
                double x = eng.mesh().vertices[i].p.x;
                double y = eng.mesh().vertices[i].p.y;
                double exact = std::sin(M_PI*x/L) * std::sin(M_PI*y/L);
                double e = eng.solution()[i] - exact;
                sum += e*e; cnt++;
            }
            double l2 = std::sqrt(sum / cnt);
            errs.push_back(l2);
            // print center value
            int ci = -1; double best = 1e9;
            for (int i=0;i<eng.solution().size();++i){ double d=std::hypot(eng.mesh().vertices[i].p.x-L/2, eng.mesh().vertices[i].p.y-L/2); if(d<best){best=d;ci=i;} }
            std::cerr << "mms contours=" << eng.mesh().contours.size() << " pts0=" << (eng.mesh().contours.empty()?0:eng.mesh().contours[0].points.size());
            if (!eng.mesh().contours.empty()) {
                double xmin=1e9,xmax=-1e9,ymin=1e9,ymax=-1e9;
                for (const auto& p : eng.mesh().contours[0].points) { xmin=std::min(xmin,p.x); xmax=std::max(xmax,p.x); ymin=std::min(ymin,p.y); ymax=std::max(ymax,p.y); }
                std::cerr << " bbox=[" << xmin << "," << xmax << "]x[" << ymin << "," << ymax << "]";
            }
            std::cerr << std::endl;
            std::cout << "mms n=" << n << " centerA=" << eng.solution()[ci] << " exact=1" << std::endl;
        }
        if (errs.size() != 3) return fail("mms: missing errors");
        // convergence order (L2 should be ~2 for P1)
        double order1 = std::log(errs[0]/errs[1]) / std::log(2.0);
        double order2 = std::log(errs[1]/errs[2]) / std::log(2.0);
        std::cout << "mms orders: " << order1 << ", " << order2 << std::endl;
        if (order1 < 1.0 || order2 < 1.0) return fail("mms: convergence order too low");

        std::cout << "All fem_linear checks passed." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }
}
