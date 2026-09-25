#include "MagneticFieldAnalyzer.h"
#include "PolarGridAccess.h"
#include "TangentKrylovWorkspace.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <sstream>
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/adapter/zero_copy.hpp>
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
        a.r_end = a.r_start + (radial - 1) * a.dr;
        a.config["polar_domain"]["r_start"] = a.r_start;
        a.config["polar_domain"]["r_end"] = a.r_end;
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
    static void nonlinearFixture(MagneticFieldAnalyzer& a, bool polar, bool horizontal) {
        grid(a, polar, horizontal);
        a.image = cv::Mat(a.ny, a.nx, CV_8UC3);
        a.material_mu.clear(); a.material_bh_tables.clear(); a.rgb_to_material.clear();
        MagneticFieldAnalyzer::MuValue m;
        m.type = MagneticFieldAnalyzer::MuType::TABLE;
        m.H_table = {0, 10, 100, 400, 1600, 6400, 25600};
        const double B[] = {0, 0.1, 0.55, 1.1, 1.55, 1.8, 2.0};
        m.mu_table.push_back(8000);
        for (size_t k = 1; k < m.H_table.size(); ++k)
            m.mu_table.push_back(B[k] / (4e-7*M_PI*m.H_table[k]));
        a.precomputeMuTableCache(m);
        a.material_mu["iron"] = m;
        a.material_mu["air"] = MagneticFieldAnalyzer::MuValue();
        a.rgb_to_material[0xffffff].name = "iron";
        a.rgb_to_material[0x808080].name = "air";
        a.generateBHTable("iron", m);
        for (int row = 0; row < a.ny; ++row) for (int col = 0; col < a.nx; ++col) {
            const bool iron = (row + 3*col) % 5 != 0;
            a.image.at<cv::Vec3b>(a.ny-1-row, col) = iron ? cv::Vec3b(255,255,255) : cv::Vec3b(128,128,128);
            a.mu_map(row,col) = 4e-7*M_PI * (iron ? 8000 : 1);
            a.jz_map(row,col) = 1e4*std::sin(row + col);
            a.Jz_mag_map(row,col) = 3e4*std::cos(0.31*row + col);
        }
    }
    static Eigen::VectorXd residual(MagneticFieldAnalyzer& a, const Eigen::VectorXd& x, bool polar) {
        a.Az.resize(a.ny, a.nx);
        if (polar) {
            for (int i = 0; i < a.nr; ++i) for (int j = 0; j < a.ntheta; ++j) {
                if (a.r_orientation == "horizontal") a.Az(j,i) = x[i*a.ntheta+j];
                else a.Az(i,j) = x[i*a.ntheta+j];
            }
            a.calculateMagneticFieldPolar();
        } else {
            for (int j = 0; j < a.ny; ++j) for (int i = 0; i < a.nx; ++i) a.Az(j,i) = x[j*a.nx+i];
            a.calculateMagneticField();
        }
        a.calculateHField(); a.updateMuDistribution();
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        if (polar) a.buildMatrixPolarReference(A,b); else a.buildMatrixReference(A,b);
        return A*x-b;
    }
    static void derivativeTests(MagneticFieldAnalyzer& a) {
        double worst = 0;
        int directions = 0;
        for (bool polar : {false, true}) for (bool horizontal : {false, true}) {
            nonlinearFixture(a, polar, horizontal);
            for (int boundary = 0; boundary < 4; ++boundary) {
                a.bc_left = bc(boundary == 2 ? "robin" : "dirichlet");
                a.bc_right = bc("dirichlet");
                a.bc_bottom = a.bc_top = bc(boundary == 0 ? "periodic" : "dirichlet");
                a.bc_inner = bc(boundary == 2 ? "robin" : boundary == 3 ? "neumann" : "dirichlet", a.ntheta);
                a.bc_outer = bc("dirichlet", a.ntheta);
                a.bc_theta_min = a.bc_theta_max = bc(boundary < 2 ? "periodic" : "robin", a.nr);
                if (boundary == 1) a.bc_theta_min.value = a.bc_theta_max.value = -1;
                for (double amplitude : {0.0002, 0.003, 0.015}) {
                    const int n = a.nx*a.ny;
                    Eigen::VectorXd x(n);
                    for (int k = 0; k < n; ++k) x[k] = amplitude * (std::sin(0.3*k) + 0.41*std::cos(0.77*k));
                    const auto R = residual(a,x,polar);
                    MagneticFieldAnalyzer::RowMatrix A,J;
                    Eigen::VectorXd b;
                    a.buildRowSystem(A,b,polar);
                    a.buildConsistentTangent(J,x);
                    MagneticFieldAnalyzer::TangentMaterialWorkspace workspace;
                    MagneticFieldAnalyzer::RowMatrix cached;
                    for (int repeat = 0; repeat < 2; ++repeat) {
                        a.buildConsistentTangent(cached,x,&workspace);
                        require((cached-J).norm() <= 1e-13*std::max(1.0,J.norm()),
                                "material workspace changed tangent");
                    }
                    Eigen::SparseMatrix<double> jc = J;
                    Eigen::SparseLU<Eigen::SparseMatrix<double>> lu;
                    lu.compute(jc);
                    require(lu.info() == Eigen::Success, "small tangent factorization failed");
                    const Eigen::VectorXd update = lu.solve(-R);
                    for (int which = 0; which < 4; ++which) {
                        Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
                        if (which == 0) for (int k = 0; k < n; ++k) v[k] = std::cos(0.43*k);
                        if (which == 1) v[polar ? a.ntheta*3 : a.nx*3] = 1; // seam/boundary
                        if (which == 2) v[n/2+3] = 1; // material interface / saturation
                        if (which == 3) v = update; // actual Newton update direction
                        const Eigen::VectorXd jv = J*v;
                        double best = 1e100;
                        for (double epsilon : {1e-4,1e-5,1e-6,1e-7,1e-8}) {
                            const double h = epsilon*std::max(x.norm(),1e-4)/std::max(v.norm(),1e-30);
                            const Eigen::VectorXd rp = residual(a,x+h*v,polar);
                            const Eigen::VectorXd rm = residual(a,x-h*v,polar);
                            const double error = ((rp-rm)/(2*h)-jv).norm()/std::max(jv.norm(),1e-30);
                            best = std::min(best,error);
                        }
                        if (!(std::isfinite(best) && best < 1e-5))
                            std::cerr << "Jv failure polar=" << polar << " horizontal=" << horizontal
                                      << " bc=" << boundary << " amplitude=" << amplitude
                                      << " direction=" << which << " error=" << best << '\n';
                        require(std::isfinite(best) && best < 1e-5, "tangent Jv mismatch");
                        worst = std::max(worst,best); ++directions;
                    }
                    residual(a,x,polar);
                    // Structural reuse after a new iterate changes every mu.
                    a.buildConsistentTangent(J,x);
                }
            }
        }
        std::cout << "tangent directions=" << directions << " max_best_relative_Jv_error=" << worst << '\n';
    }
    static void slidingTangentTests(MagneticFieldAnalyzer& a) {
        double worst = 0;
        int checks = 0;
        for (bool horizontal : {false, true}) for (bool axis : {false, true})
        for (bool anti : {false, true}) {
            nonlinearFixture(a, true, horizontal);
            if (axis) {
                a.r_start = 0;
                a.r_end = (a.nr-1)*a.dr;
                for (int i = 0; i < a.nr; ++i) a.r_coords[i] = i*a.dr;
            }
            a.bc_inner = a.bc_outer = bc("dirichlet");
            a.bc_inner.value = a.bc_outer.value = 0;
            a.bc_theta_min = a.bc_theta_max = bc("periodic");
            a.bc_theta_min.value = a.bc_theta_max.value = anti ? -1 : 1;
            MagneticFieldAnalyzer::SlideRegion slide;
            slide.direction = horizontal ? "vertical" : "horizontal";
            slide.region_start = 0; slide.region_end = a.nr/2;
            slide.pixels_per_step = 3;
            slide.wrap_mode = anti ? "antiperiodic" : "periodic";
            a.transient_config.slides = {slide};
            a.slide_sign_map.release();
            a.rect_slide_states.clear(); a.slide_step_counter = 0;
            const int n = a.nr*a.ntheta;
            Eigen::VectorXd x(n), v(n), b;
            for (int k = 0; k < n; ++k) {
                x[k] = 0.002*(std::sin(0.3*k)+0.41*std::cos(0.77*k));
                v[k] = std::cos(0.43*k);
            }
            MagneticFieldAnalyzer::RowMatrix A, cached;
            const double* storage = nullptr;
            for (int step = 0; step < 4; ++step) {
                if (step) {
                    const auto before = a.image.clone();
                    a.slideImageRegion();
                    require(cv::norm(before, a.image, cv::NORM_INF) > 0,
                            "sliding fixture did not move the material interface");
                }
                // Seed static/nonlinear cells according to the moved image;
                // residual() then refreshes B/H/mu from the candidate state.
                for (int row = 0; row < a.ny; ++row) for (int col = 0; col < a.nx; ++col) {
                    const auto pixel = a.image.at<cv::Vec3b>(a.ny-1-row,col);
                    a.mu_map(row,col) = 4e-7*M_PI*(pixel[0] == 255 ? 8000 : 1);
                }
                residual(a,x,true);
                a.buildRowSystem(A,b,true);
                a.buildConsistentTangent(cached,x);
                if (!step) storage = cached.valuePtr();
                else require(storage == cached.valuePtr(), "slide rebuilt the fixed tangent graph");
                MagneticFieldAnalyzer::RowMatrix fresh;
                a.buildConsistentTangent(fresh,x);
                require((fresh-cached).norm() <= 1e-13*std::max(1.0,fresh.norm()),
                        "sliding tangent retained stale coefficients");
                const Eigen::VectorXd jv = cached*v;
                double best = 1e100;
                for (double epsilon : {1e-4,1e-5,1e-6,1e-7,1e-8}) {
                    const double h = epsilon*x.norm()/v.norm();
                    const auto rp = residual(a,x+h*v,true);
                    const auto rm = residual(a,x-h*v,true);
                    best = std::min(best, ((rp-rm)/(2*h)-jv).norm()/std::max(jv.norm(),1e-30));
                }
                require(std::isfinite(best) && best < 1e-5, "axis/sliding tangent Jv mismatch");
                worst = std::max(worst,best); ++checks;
            }
        }
        a.transient_config.slides.clear(); a.slide_sign_map.release();
        std::cout << "axis/sliding tangent checks=" << checks << " max_relative_Jv_error=" << worst << '\n';
    }
    static void materialDerivativeTests(MagneticFieldAnalyzer& a) {
        nonlinearFixture(a, false, true);
        auto m = a.material_mu.at("iron");
        double worst = 0;
        for (double H : {1e-7, 0.17, 4.1, 37.0, 210.0, 917.0, 3091.0, 13013.0, 27111.0, 3e6}) {
            const double h = 1e-5 * std::max(H, 1e-5);
            const double fd = (a.evaluateMu(m,H+h)-a.evaluateMu(m,H-h))/(2*h);
            const double d = a.evaluateMuValueDerivative(m,H);
            const double err = std::abs(fd-d)/std::max({std::abs(d),std::abs(fd),1e-6});
            // Very low H involves cancellation in B(H)/H; evaluate its limiting
            // derivative away from the special H<=1e-12 branch separately.
            if (H > 1e-5) { require(err < 1e-5, "PCHIP value derivative mismatch"); worst = std::max(worst,err); }
        }
        // Branch-consistent one-sided derivatives at original-table endpoints.
        for (double H : {m.H_table[1], m.H_table[3], m.H_table.back()}) {
            const double h = 1e-7 * H;
            const double fd = (a.evaluateMu(m,H+h)-a.evaluateMu(m,H))/h;
            const double d = a.evaluateMuValueDerivative(m,H);
            require(std::abs(fd-d) < 1e-4*std::max(std::abs(d),1e-6), "one-sided mu derivative mismatch");
        }
        m.has_dmu_extrapolation = true; m.dmu_r_extrap_const = 123;
        require(a.evaluateMuValueDerivative(m,1e6) == 0, "constant extrapolation must have zero value derivative");
        auto clamp = a.parseMuValue(YAML::Load("[[1,2,3], [0.2,0.2,0.2]]"));
        a.precomputeMuTableCache(clamp);
        require(a.evaluateMuValueDerivative(clamp,1.7) == 0, "clamped table derivative must be zero");
        auto formula = a.parseMuValue(YAML::Load("'1 + 4000/(1 + $H/100)'"));
        for (double H : {0.0,3.7,271.0,13903.0}) {
            const double exact = -40/std::pow(1+H/100,2);
            require(std::abs(a.evaluateMuValueDerivative(formula,H)/exact-1) < 1e-6,
                    "numerical formula derivative mismatch");
        }
        const auto& table = a.material_bh_tables.at("iron");
        for (size_t k : {size_t(3),size_t(111),table.B_values.size()-1}) {
            const double B = table.B_values[k];
            const double h = 1e-8*std::max(B,1e-4);
            const double fd = (a.interpolateH_from_B(table,B+h)-a.interpolateH_from_B(table,B))/h;
            const double d = a.inverseBHDerivative(table,B);
            require(std::abs(fd/d-1) < 1e-5, "one-sided inverse-table derivative mismatch");
        }
        std::cout << "material derivatives max_relative_error=" << worst << '\n';
    }
    static void energyDiagnostic(MagneticFieldAnalyzer& a) {
        nonlinearFixture(a,false,true);
        a.material_bh_tables.clear();
        a.material_mu["iron"] = MagneticFieldAnalyzer::MuValue();
        a.mu_map.setConstant(4e-7*M_PI);
        a.jz_map.setZero(); a.Jz_mag_map.setZero();
        a.bc_left = a.bc_right = a.bc_top = a.bc_bottom = bc("dirichlet");
        a.bc_left.value = a.bc_right.value = a.bc_top.value = a.bc_bottom.value = 0;
        Eigen::VectorXd x(a.nx*a.ny), v = Eigen::VectorXd::Zero(a.nx*a.ny);
        for (int k = 0; k < x.size(); ++k) x[k] = 0.003*std::sin(0.31*k);
        v[a.nx*4+3] = 1;
        const auto R = residual(a,x,false);
        const double h = 1e-7;
        residual(a,x+h*v,false); const double wp = a.computeEnergyObjective();
        residual(a,x-h*v,false); const double wm = a.computeEnergyObjective();
        const double derivative = (wp-wm)/(2*h), residual_derivative = -a.dx*a.dy*R.dot(v);
        std::cout << "energy diagnostic FD=" << derivative << " scaled_negative_R_dot_v="
                  << residual_derivative << " (not assumed equal)\n";
        require(std::isfinite(derivative) && std::isfinite(residual_derivative), "energy diagnostic nonfinite");
    }
    static void environment(const char* name, const char* value) {
#ifdef _WIN32
        _putenv_s(name,value);
#else
        if (*value) setenv(name,value,1); else unsetenv(name);
#endif
    }
    struct Capture {
        std::ostringstream stream;
        std::streambuf* original = std::cout.rdbuf(stream.rdbuf());
        ~Capture() { std::cout.rdbuf(original); }
    };
    static void accelerationTests(MagneticFieldAnalyzer& a) {
        int accepted = 0, rejected = 0;
        for (double amplitude : {0.001,0.01,0.04}) {
            Eigen::MatrixXd reference_Az, reference_mu;
            double reference_residual = 0;
            for (int assembled = 0; assembled < 2; ++assembled) {
                nonlinearFixture(a,true,true);
                a.bc_inner = a.bc_outer = bc("dirichlet");
                a.bc_inner.value = a.bc_outer.value = 0;
                a.bc_theta_min = a.bc_theta_max = bc("periodic");
                a.has_nonlinear_materials = true;
                a.nonlinear_config.max_iterations = 12;
                a.nonlinear_config.tolerance = 1e-8;
                a.nonlinear_config.verbose = true;
                a.nonlinear_config.anderson.enabled = true;
                a.nonlinear_config.anderson.depth = 3;
                a.nonlinear_config.anderson.beta = 1;
                a.nonlinear_config.jacobian_tangent = false;
                a.dd_warm_start_ = true;
                Eigen::VectorXd x(a.nr*a.ntheta);
                for (int k = 0; k < x.size(); ++k) x[k] = amplitude*std::sin(0.4*k);
                residual(a,x,true);
                environment("OMFDM_ASSEMBLED_RESIDUAL", assembled ? "1" : "");
                std::string log;
                { Capture capture; a.solveNonlinearNewtonKrylov(); log = capture.stream.str(); }
                environment("OMFDM_ASSEMBLED_RESIDUAL", "");
                require(a.Az.allFinite() && a.mu_map.allFinite(), "AA exported nonfinite state");
                for (int i = 0; i < a.nr; ++i) for (int j = 0; j < a.ntheta; ++j) x[i*a.ntheta+j] = a.Az(j,i);
                const auto R = residual(a,x,true);
                Eigen::SparseMatrix<double> matrix;
                Eigen::VectorXd b;
                a.buildMatrixPolarReference(matrix,b);
                const double checked = R.norm()/(b.norm()+1e-12);
                require(std::abs(checked-a.last_nonlinear_residual_) < 1e-10*std::max(1.0,checked),
                        "AA/max-iteration rollback residual does not describe exported state");
                if (!assembled) {
                    reference_Az = a.Az; reference_mu = a.mu_map; reference_residual = checked;
                    accepted += log.find("[AA accepted:") != std::string::npos;
                    rejected += log.find("[AA rejected ") != std::string::npos;
                } else {
                    require((reference_Az-a.Az).norm() < 1e-9*std::max(1.0,a.Az.norm()), "AA direct/assembled state differs");
                    require((reference_mu-a.mu_map).norm() < 1e-9*std::max(1.0,a.mu_map.norm()), "AA direct/assembled mu differs");
                    require(std::abs(reference_residual-checked) < 1e-9*std::max(1.0,checked), "AA direct/assembled residual differs");
                }
            }
        }
        require(accepted > 0 && rejected > 0, "AA fixtures must exercise both acceptance and rejection");
        std::cout << "AA acceptance cases=" << accepted << " rejection/rollback cases=" << rejected << '\n';
        a.nonlinear_config.verbose = false;
    }
    static void innerSolverTests(MagneticFieldAnalyzer& a) {
        MagneticFieldAnalyzer::RowMatrix A(40,40), J(40,40);
        std::vector<Eigen::Triplet<double>> entries;
        for (int k = 0; k < 40; ++k) {
            entries.emplace_back(k,k,3.0);
            if (k > 0) entries.emplace_back(k,k-1,-1.0);
            if (k < 39) entries.emplace_back(k,k+1,-1.0);
        }
        A.setFromTriplets(entries.begin(),entries.end()); J = A;
        for (int k = 0; k < 39; ++k) J.coeffRef(k,k+1) += 0.31;
        J.makeCompressed();
        // Borrowed operator must neither copy nor own the analyzer's CRS.
        // Compare copied and borrowed products, including a numeric refresh.
        {
            const auto view = amgcl::adapter::zero_copy_direct(J.rows(), J.cols(),
                J.outerIndexPtr(), J.innerIndexPtr(), J.valuePtr());
            require(!view->own_data && view->ptr == J.outerIndexPtr() &&
                    view->col == J.innerIndexPtr() && view->val == J.valuePtr(),
                    "Krylov operator is not a borrowed CRS view");
            for (int refresh = 0; refresh < 2; ++refresh) {
                if (refresh) J.valuePtr()[0] += 0.125;
                amgcl::backend::crs<double> copied(J);
                std::vector<double> x(40, 1.0), borrowed(40), reference(40);
                amgcl::backend::spmv(1.0, *view, x, 0.0, borrowed);
                amgcl::backend::spmv(1.0, copied, x, 0.0, reference);
                require(borrowed == reference, "borrowed/copied Krylov product differs");
            }
        }
        const MagneticFieldAnalyzer::RowMatrix saved_J = J;
        Eigen::VectorXd b = Eigen::VectorXd::Ones(40), direction;
        for (const char* candidate : {"secant-amg", "tangent-amg", "tangent-ilut"}) {
            environment("OMFDM_TANGENT_PRECONDITIONER",candidate);
            require(a.solveTangentSystem(J,A,b,1e-6,direction), "FGMRES candidate did not solve small system");
            require((J*direction-b).norm()/b.norm() <= 1e-6, "FGMRES returned invalid direction");
            require((J-saved_J).norm() == 0, "FGMRES mutated borrowed tangent matrix");
        }
        b[3] = std::numeric_limits<double>::quiet_NaN();
        require(!a.solveTangentSystem(J,A,b,1e-6,direction), "nonfinite inner RHS accepted");
        environment("OMFDM_TANGENT_PRECONDITIONER","tangent-amg");
        omfdm::TangentKrylovWorkspace work;
        for (int repeat = 0; repeat < 6; ++repeat) {
            b = repeat == 1 ? Eigen::VectorXd::Zero(40).eval() : Eigen::VectorXd::Constant(40,1.0+repeat).eval();
            J.coeffRef(0,0) += 0.1;
            const double tol = repeat % 2 ? 1e-8 : 1e-4;
            Eigen::VectorXd fresh;
            const auto allocations = a.linear_profile_.workspace_allocations;
            require(a.solveTangentSystem(J,A,b,tol,direction,&work), "reused FGMRES failed");
            require(a.linear_profile_.workspace_allocations == allocations + (repeat == 0), "workspace not reused");
            require(a.solveTangentSystem(J,A,b,tol,fresh), "fresh FGMRES failed");
            require((direction-fresh).norm() < 1e-13*std::max(1.0,fresh.norm()), "FGMRES storage leaked history");
        }
        b[3] = std::numeric_limits<double>::quiet_NaN();
        require(!a.solveTangentSystem(J,A,b,1e-6,direction,&work) && !work.solver && work.x.capacity() == 0,
                "failed FGMRES retained storage");
        b.setOnes();
        require(a.solveTangentSystem(J,A,b,1e-6,direction,&work), "FGMRES did not recover after invalid RHS");
        require(work.prepare(41,1e-4) && work.x.size() == 41, "FGMRES size change not rebuilt");
        work.clear();
        require(!work.solver && work.b.capacity() == 0 && work.x.capacity() == 0, "workspace not released");
        environment("OMFDM_TANGENT_PRECONDITIONER","");
        std::cout << "FGMRES A/B/C and nonfinite rejection passed\n";
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
        SolverNumericsTest::derivativeTests(a);
        SolverNumericsTest::slidingTangentTests(a);
        SolverNumericsTest::materialDerivativeTests(a);
        SolverNumericsTest::energyDiagnostic(a);
        SolverNumericsTest::accelerationTests(a);
        SolverNumericsTest::innerSolverTests(a);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
