#include "MagneticFieldAnalyzer.h"
#include "PolarGridAccess.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "legacy_stencil_reference.inc"

struct SolverNumericsTest {
    static void require(bool yes, const char* message) {
        if (!yes) throw std::runtime_error(message);
    }
    static MagneticFieldAnalyzer::BoundaryCondition bc(const std::string& type, int size = 0) {
        MagneticFieldAnalyzer::BoundaryCondition b;
        b.type = type; b.value = 0.017; b.alpha = 1.3; b.beta = 0.004; b.gamma = -0.008;
        for (int k = 0; k < size; ++k) b.profile.push_back(0.013 * std::sin(0.7 * k));
        return b;
    }
    static void grid(MagneticFieldAnalyzer& a, bool polar, bool horizontal, int radial = 8, int theta = 12) {
        a.coordinate_system = polar ? "polar" : "cartesian";
        a.r_orientation = horizontal ? "horizontal" : "vertical";
        a.nr = radial; a.ntheta = theta;
        a.nx = horizontal ? radial : theta; a.ny = horizontal ? theta : radial;
        a.dx = 0.0013; a.dy = 0.0021; a.dr = 0.0027; a.dtheta = 0.23;
        a.r_start = 0.009;
        a.r_coords.resize(radial);
        for (int i = 0; i < radial; ++i) a.r_coords[i] = a.r_start + i * a.dr;
        a.mu_map.resize(a.ny, a.nx);
        a.jz_map.resize(a.ny, a.nx);
        a.Jz_mag_map.resize(a.ny, a.nx);
    }
    static double check(MagneticFieldAnalyzer& a, bool polar, MagneticFieldAnalyzer::RowMatrix& A) {
        const int n = a.nx * a.ny;
        Eigen::VectorXd x(n), b, R, direct_b, old_b;
        double worst = 0;
        for (int update = 0; update < 3; ++update) {
            // Changing every material coefficient/source simulates slide and
            // rollback invalidation; a cached numeric value fails this test.
            for (int j = 0; j < a.ny; ++j) for (int i = 0; i < a.nx; ++i) {
                a.mu_map(j, i) = 4e-7 * M_PI * (1.1 + ((3*i + 7*j + 11*update) % 29) * 47.0);
                a.jz_map(j, i) = 9e4 * std::sin(0.8*i + 0.3*j + update);
                a.Jz_mag_map(j, i) = 3e4 * std::cos(0.2*i - 0.9*j + update);
            }
            for (int k = 0; k < n; ++k) x[k] = 0.027 * std::sin(0.31*k + update);
            Eigen::SparseMatrix<double> reference;
            if (polar) a.buildMatrixPolarReference(reference, old_b);
            else a.buildMatrixReference(reference, old_b);
            a.buildRowSystem(A, b, polar);
            a.evaluateDiscreteResidual(x, R, direct_b, polar);
            const double scale = reference.norm() * x.norm() + old_b.norm();
            Eigen::SparseMatrix<double> converted = A;
            const double matrix_error = (converted - reference).norm() / std::max(1.0, reference.norm());
            const double residual_error = (R - (reference * x - old_b)).norm() / std::max(1.0, scale);
            const double rhs_error = (b - old_b).norm() / std::max(1.0, scale);
            require(std::isfinite(residual_error) && residual_error <= 1e-11, "direct residual differs from v1.6.2");
            require(matrix_error <= 1e-13 && rhs_error <= 1e-13, "cached assembly differs from v1.6.2");
            require((b - direct_b).norm() <= 1e-13 * std::max(1.0, scale), "residual RHS mismatch");
            worst = std::max(worst, residual_error);
        }
        return worst;
    }
    static void run(MagneticFieldAnalyzer& a) {
        MagneticFieldAnalyzer::RowMatrix A;
        int cases = 0;
        double worst = 0;
        for (bool horizontal : {false, true}) {
            grid(a, false, horizontal);
            for (const std::string left : {"dirichlet", "neumann", "robin", "periodic"})
            for (const std::string bottom : {"dirichlet", "neumann", "robin", "periodic"}) {
                a.bc_left = bc(left); a.bc_right = bc(left);
                a.bc_bottom = bc(bottom); a.bc_top = bc(bottom);
                worst = std::max(worst, check(a, false, A)); ++cases;
            }
            grid(a, true, horizontal);
            for (const std::string radial : {"dirichlet", "neumann", "robin"})
            for (const std::string angular : {"dirichlet", "robin", "periodic"})
            for (bool profile : {false, true})
            for (double sign : {1.0, -1.0}) {
                a.bc_inner = bc(radial, profile ? a.ntheta : 0);
                a.bc_outer = bc(radial, profile ? a.ntheta : 0);
                a.bc_theta_min = bc(angular, profile ? a.nr : 0);
                a.bc_theta_max = bc(angular, profile ? a.nr : 0);
                if (angular == "periodic") a.bc_theta_min.value = a.bc_theta_max.value = sign;
                worst = std::max(worst, check(a, true, A)); ++cases;
            }
            // Axis row is Dirichlet in the supported polar model.
            a.bc_inner = bc("dirichlet", a.ntheta); a.bc_outer = bc("robin", a.ntheta);
            a.bc_theta_min = a.bc_theta_max = bc("periodic");
            for (int i = 0; i < a.nr; ++i) a.r_coords[i] = i * a.dr;
            worst = std::max(worst, check(a, true, A)); ++cases;
            // Same DOF count, different connections: must invalidate cache.
            grid(a, true, horizontal, 12, 8);
            a.bc_inner = a.bc_outer = bc("dirichlet");
            worst = std::max(worst, check(a, true, A)); ++cases;
        }
        std::cout << "stencil cases=" << cases << " updates=" << cases*3
                  << " max_normalized_residual_error=" << worst << '\n';
    }
};

int main(int argc, char** argv) {
    try {
        const auto dir = std::filesystem::path(argc > 1 ? argv[1] : "solver_numerics_fixture");
        std::filesystem::create_directories(dir);
        const auto yaml = dir / "fixture.yaml", png = dir / "fixture.png";
        std::ofstream f(yaml);
        f << "omp: {threads: 2}\ncoordinate_system: cartesian\nmesh: {dx: 0.001, dy: 0.001}\n"
             "boundary_conditions:\n  left: {type: dirichlet, value: 0}\n"
             "  right: {type: dirichlet, value: 0}\n  bottom: {type: dirichlet, value: 0}\n"
             "  top: {type: dirichlet, value: 0}\n"
             "materials:\n  air: {rgb: [255,255,255], mu_r: 1, jz: 0}\nexport: {async: false}\n";
        f.close();
        if (!cv::imwrite(png.string(), cv::Mat(12, 8, CV_8UC3, cv::Scalar(255,255,255))))
            throw std::runtime_error("fixture image write failed");
        MagneticFieldAnalyzer a(yaml.string(), png.string());
        SolverNumericsTest::run(a);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
