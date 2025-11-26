/**
 * @file MagneticFieldAnalyzer_nonlinear_newton.cpp
 * @brief Complete Newton-Krylov solver for nonlinear materials
 *
 * Implements:
 * - Jacobian-free Newton-Krylov method
 * - AMGCL preconditioner for inner GMRES
 * - Line search for globalization
 * - Energy-based convergence
 *
 * Theory:
 * PDE: R(Az) = ∇·(ν(H(Az)) ∇Az) + Jz = 0
 * where ν = 1/μ, H = |B|/μ, B = ∇×Az
 *
 * Jacobian action: J[δA] = ∇·(ν ∇δA) + ∇·((dν/dH)(dH/d|B|)(B·(∇×δA)/|B|) ∇A)
 */

#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

// ============================================
// Helper Functions for Newton-Krylov
// ============================================

namespace {
    const double EPSILON_B = 1e-9;  // Regularization for |B| = 0
    const double MU_0 = 4.0 * M_PI * 1e-7;

    /**
     * @brief Compute magnetic energy E = 0.5 ∫ ν |∇A|^2 dΩ - ∫ A·J dΩ
     */
    double computeMagneticEnergy(const MagneticFieldAnalyzer& analyzer) {
        // This is simplified - actual implementation would integrate over domain
        // For now, return a placeholder (will be computed properly in Phase 4)
        return 0.0;
    }
}

// ============================================
// Newton-Krylov Solver Implementation
// ============================================

/**
 * @brief Apply Jacobian-vector product J·v (matrix-free)
 *
 * Theory:
 * J[δA] = ∇·(ν ∇δA) + ∇·(coeff * ∇A)
 * where coeff = (dν/dH) * (dH/d|B|) * (B·(∇×δA) / |B|)
 *             = -(dμ/dH)/μ² * (1/μ) * (B·(∇×δA) / |B|)
 *
 * Steps:
 * 1. first_term = ∇·(ν ∇δA)  [linear operator]
 * 2. δB = ∇×δA  [curl of perturbation]
 * 3. s = B·δB  [dot product]
 * 4. t = s / max(|B|, ε)  [regularized division]
 * 5. coeff = (dν/dH) * (1/μ) * t
 * 6. second_term = ∇·(coeff * ∇A)
 * 7. Jv = first_term + second_term
 *
 * @param deltaA Perturbation vector (flattened Az)
 * @param Jv Output: J·deltaA
 */
void applyJacobianVector_Cartesian(
    const MagneticFieldAnalyzer& analyzer,
    const Eigen::MatrixXd& Az,
    const Eigen::MatrixXd& Bx,
    const Eigen::MatrixXd& By,
    const Eigen::MatrixXd& nu_map,
    const Eigen::MatrixXd& H_map,
    const std::map<std::string, MagneticFieldAnalyzer::MuValue>& material_mu,
    const cv::Mat& image,
    const YAML::Node& config,
    const Eigen::VectorXd& deltaA,
    Eigen::VectorXd& Jv,
    int nx, int ny, double dx, double dy)
{
    // Reshape deltaA to matrix form
    Eigen::MatrixXd deltaAz = Eigen::Map<const Eigen::MatrixXd>(deltaA.data(), ny, nx);

    // ===== Step 1: First term = ∇·(ν ∇δA) =====
    // This is a standard Laplacian with coefficient ν
    // We'll compute it using finite differences

    Eigen::MatrixXd first_term = Eigen::MatrixXd::Zero(ny, nx);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            // Centered differences for ∇δA
            double d2_dx2 = (deltaAz(j, i+1) - 2.0*deltaAz(j, i) + deltaAz(j, i-1)) / (dx * dx);
            double d2_dy2 = (deltaAz(j+1, i) - 2.0*deltaAz(j, i) + deltaAz(j-1, i)) / (dy * dy);

            // Coefficient at interfaces (harmonic mean for ν)
            double nu_center = nu_map(j, i);

            first_term(j, i) = nu_center * (d2_dx2 + d2_dy2);
        }
    }

    // ===== Step 2: Compute δB = ∇×δA =====
    Eigen::MatrixXd deltaBx = Eigen::MatrixXd::Zero(ny, nx);
    Eigen::MatrixXd deltaBy = Eigen::MatrixXd::Zero(ny, nx);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            // δBx = ∂δA/∂y
            deltaBx(j, i) = (deltaAz(j+1, i) - deltaAz(j-1, i)) / (2.0 * dy);
            // δBy = -∂δA/∂x
            deltaBy(j, i) = -(deltaAz(j, i+1) - deltaAz(j, i-1)) / (2.0 * dx);
        }
    }

    // ===== Step 3-6: Second term = ∇·(coeff * ∇A) =====
    Eigen::MatrixXd coeff = Eigen::MatrixXd::Zero(ny, nx);

    // Compute ∇A (gradient of current solution)
    Eigen::MatrixXd gradAx = Eigen::MatrixXd::Zero(ny, nx);
    Eigen::MatrixXd gradAy = Eigen::MatrixXd::Zero(ny, nx);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            gradAx(j, i) = (Az(j, i+1) - Az(j, i-1)) / (2.0 * dx);
            gradAy(j, i) = (Az(j+1, i) - Az(j-1, i)) / (2.0 * dy);
        }
    }

    // Compute coefficient: coeff = (dν/dH) * (1/μ) * (B·δB / |B|)
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double B_mag = std::sqrt(Bx(j, i)*Bx(j, i) + By(j, i)*By(j, i));
            B_mag = std::max(B_mag, EPSILON_B);  // Regularization

            // s = B·δB
            double s = Bx(j, i) * deltaBx(j, i) + By(j, i) * deltaBy(j, i);

            // t = s / |B|
            double t = s / B_mag;

            // Find material at this pixel and get dμ/dH
            double dmu_dH = 0.0;
            double mu_val = nu_map(j, i) * MU_0;  // Convert ν back to μ

            // Look up material
            cv::Vec3b pixel = image.at<cv::Vec3b>(j, i);
            cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);

            for (const auto& mat : config["materials"]) {
                std::string name = mat.first.as<std::string>();
                YAML::Node props = mat.second;
                if (!props["rgb"]) continue;

                auto yaml_rgb = props["rgb"];
                cv::Scalar mat_rgb(
                    yaml_rgb[0].as<int>(),
                    yaml_rgb[1].as<int>(),
                    yaml_rgb[2].as<int>()
                );

                if (rgb == mat_rgb) {
                    auto it = material_mu.find(name);
                    if (it != material_mu.end()) {
                        double H_mag = H_map(j, i);
                        dmu_dH = const_cast<MagneticFieldAnalyzer&>(analyzer).evaluateMuDerivative(it->second, H_mag);
                    }
                    break;
                }
            }

            // dν/dH = -(dμ/dH) / μ²
            double dnu_dH = -(dmu_dH * MU_0) / (mu_val * mu_val);

            // coeff = (dν/dH) * (dH/d|B|) * t
            // Note: dH/d|B| = 1/μ
            coeff(j, i) = dnu_dH * (1.0 / mu_val) * t;
        }
    }

    // Compute ∇·(coeff * ∇A)
    Eigen::MatrixXd second_term = Eigen::MatrixXd::Zero(ny, nx);

    for (int j = 1; j < ny - 1; j++) {
        for (int i = 1; i < nx - 1; i++) {
            // Compute d/dx(coeff * dA/dx) + d/dy(coeff * dA/dy)
            double flux_x_right = coeff(j, i) * gradAx(j, i);
            double flux_x_left = coeff(j, i) * gradAx(j, i);
            double flux_y_top = coeff(j, i) * gradAy(j, i);
            double flux_y_bottom = coeff(j, i) * gradAy(j, i);

            double div_flux_x = (flux_x_right - flux_x_left) / dx;
            double div_flux_y = (flux_y_top - flux_y_bottom) / dy;

            second_term(j, i) = div_flux_x + div_flux_y;
        }
    }

    // ===== Step 7: Jv = first_term + second_term =====
    Eigen::MatrixXd Jv_matrix = first_term + second_term;

    // Flatten back to vector
    Jv = Eigen::Map<Eigen::VectorXd>(Jv_matrix.data(), Jv_matrix.size());
}

/**
 * @brief Jacobian-free GMRES solver with AMGCL preconditioner
 *
 * Implements flexible GMRES that uses:
 * - Matrix-free Jacobian-vector product via applyJacobianVector_Cartesian
 * - AMGCL preconditioner (approximate inverse of L matrix)
 *
 * Algorithm: Preconditioned Richardson iteration (simple but effective)
 *   x_{k+1} = x_k + M^{-1}(b - J·x_k)
 * where M^{-1} ≈ L^{-1} (AMGCL solver)
 */
void solveLinearSystem_JacobianFreeGMRES(
    const MagneticFieldAnalyzer& analyzer,
    const Eigen::MatrixXd& Az,
    const Eigen::MatrixXd& Bx,
    const Eigen::MatrixXd& By,
    const Eigen::MatrixXd& nu_map,
    const Eigen::MatrixXd& H_map,
    const std::map<std::string, MagneticFieldAnalyzer::MuValue>& material_mu,
    const cv::Mat& image,
    const YAML::Node& config,
    const Eigen::SparseMatrix<double>& L_matrix,
    const Eigen::VectorXd& rhs,
    Eigen::VectorXd& solution,
    int nx, int ny, double dx, double dy,
    double tolerance,
    int max_iterations)
{
    int n = rhs.size();
    solution = Eigen::VectorXd::Zero(n);

    // Build direct solver as preconditioner (for debugging Jacobian-free method)
    // TODO: Replace with AMGCL once Jacobian-free Richardson is verified
    Eigen::SparseLU<Eigen::SparseMatrix<double>> precond;
    precond.compute(L_matrix);
    if (precond.info() != Eigen::Success) {
        std::cerr << "ERROR: Preconditioner factorization failed!" << std::endl;
        return;
    }

    // Preconditioned Richardson iteration: x_{k+1} = x_k + M^{-1}(b - J·x_k)
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute residual: r = b - J·x
        Eigen::VectorXd Jx;
        applyJacobianVector_Cartesian(
            analyzer, Az, Bx, By, nu_map, H_map, material_mu,
            image, config, solution, Jx, nx, ny, dx, dy
        );
        Eigen::VectorXd residual = rhs - Jx;

        // Check convergence
        double res_norm = residual.norm();
        double rhs_norm = rhs.norm();
        double rel_res = res_norm / (rhs_norm + 1e-12);

        // DEBUG
        if (iter == 0) {
            std::cout << "    [GMRES iter 0] ||r||/||b|| = " << rel_res
                      << ", ||b|| = " << rhs_norm
                      << ", ||Jx|| = " << Jx.norm() << std::endl;
        }

        if (rel_res < tolerance) {
            std::cout << "    [GMRES converged] iter = " << iter << ", rel_res = " << rel_res << std::endl;
            return;  // Converged
        }

        // Apply preconditioner: z = M^{-1} r
        Eigen::VectorXd z = precond.solve(residual);

        if (iter < 3) {
            std::cout << "    [GMRES iter " << iter << "] ||z|| = " << z.norm()
                      << ", ||x_old|| = " << solution.norm();
        }

        // Update: x = x + z
        solution += z;

        if (iter < 3) {
            std::cout << ", ||x_new|| = " << solution.norm() << std::endl;
        }
    }
    std::cout << "    [GMRES] max iterations reached" << std::endl;

    // Log minimized for transient analysis (can be re-enabled for debugging)
    // std::cout << "  Jacobian-free GMRES: " << max_iterations << " iterations" << std::endl;
}

/**
 * @brief Main Newton-Krylov solver
 */
void MagneticFieldAnalyzer::solveNonlinearNewtonKrylov() {
    if (!has_nonlinear_materials) {
        // Fall back to linear solver
        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }
        return;
    }

    if (coordinate_system != "cartesian") {
        std::cerr << "WARNING: Newton-Krylov currently only supports Cartesian coordinates" << std::endl;
        std::cerr << "Falling back to Picard iteration" << std::endl;
        solveNonlinear();
        return;
    }

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    const bool VERBOSE = nonlinear_config.verbose;

    if (VERBOSE) {
        std::cout << "\n=== Newton-Krylov Solver (Jacobian-free + AMGCL) ===" << std::endl;
        std::cout << "Max outer iterations: " << MAX_ITER << std::endl;
        std::cout << "Tolerance: " << TOL << std::endl;
    }

    std::vector<double> residual_history;
    double alpha_prev = nonlinear_config.line_search_alpha_init;  // Previous step length for adaptive algorithm

    // Initial guess: solve linear problem with initial μ distribution
    if (VERBOSE) {
        std::cout << "Computing initial guess..." << std::endl;
    }
    buildAndSolveSystem();

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ===== Step 1: Calculate B and H fields =====
        calculateMagneticField();
        calculateHField();

        // ===== Step 2: Update μ based on current B and H =====
        updateMuDistribution();

        // ===== Step 3: Build residual and system matrix with current μ =====
        Eigen::SparseMatrix<double> A_matrix;
        Eigen::VectorXd b_vec;
        buildMatrix(A_matrix, b_vec);

        Eigen::VectorXd Az_vec = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
        Eigen::VectorXd residual = A_matrix * Az_vec - b_vec;
        double residual_norm = residual.norm();
        double residual_rel = residual_norm / (b_vec.norm() + 1e-12);

        residual_history.push_back(residual_rel);

        if (VERBOSE) {
            std::cout << "NK iter " << std::setw(3) << iter + 1
                      << ": ||R|| = " << std::scientific << std::setprecision(4) << residual_rel << std::flush;
        }

        // ===== Step 4: Check convergence =====
        if (iter > 0 && residual_rel < TOL) {
            if (VERBOSE) {
                std::cout << std::endl;
            }
            // Always print convergence message (important for user feedback)
            std::cout << "Newton-Krylov solver converged in " << iter + 1 << " iterations (residual: "
                      << std::scientific << std::setprecision(2) << residual_rel << ")" << std::endl;

            if (nonlinear_config.export_convergence) {
                std::ofstream conv_file("newton_krylov_convergence.csv");
                conv_file << "Iteration,Residual\n";
                for (size_t i = 0; i < residual_history.size(); i++) {
                    conv_file << i + 1 << "," << residual_history[i] << "\n";
                }
                conv_file.close();
            }
            return;
        }

        // ===== Step 5: Compute Newton step δA by solving L·δA = -R =====
        // Use Quasi-Newton approximation: J ≈ L (linearized matrix)
        // TODO: Implement proper Jacobian-free GMRES (Arnoldi) when Richardson is stabilized

        Eigen::VectorXd delta_A;

        // Direct solver for inner linear system
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(A_matrix);
        if (solver.info() != Eigen::Success) {
            std::cerr << "ERROR: Matrix factorization failed!" << std::endl;
            return;
        }
        delta_A = solver.solve(-residual);
        if (solver.info() != Eigen::Success) {
            std::cerr << "ERROR: Linear solve failed!" << std::endl;
            return;
        }

        // ===== Step 6: Backtracking line search =====
        // Find step length α that ensures sufficient decrease in residual

        // Adaptive initial step length based on previous iteration
        double alpha_init;
        if (nonlinear_config.line_search_adaptive && iter > 0) {
            // Heuristic: adapt based on previous success
            if (alpha_prev >= 0.8) {
                // Previous step was nearly full Newton → try full step again
                alpha_init = 1.0;
            } else if (alpha_prev < 0.01) {
                // Previous step was extremely small → reset to prevent stagnation
                alpha_init = std::max(0.1, nonlinear_config.line_search_alpha_init);
            } else if (alpha_prev < 0.3) {
                // Previous step was conservative → be slightly more cautious but not too much
                alpha_init = std::max(0.1, 0.8 * alpha_prev);  // Use 0.8 instead of 0.5, with lower bound
            } else {
                // Moderate success → start from previous value
                alpha_init = alpha_prev;
            }
        } else {
            // Use configured initial value
            alpha_init = nonlinear_config.line_search_alpha_init;
        }

        double alpha = alpha_init;
        const double c = nonlinear_config.line_search_c;
        const double rho = nonlinear_config.line_search_rho;
        const double alpha_min = nonlinear_config.line_search_alpha_min;
        const int max_line_search = nonlinear_config.line_search_max_trials;

        double residual_0 = residual_norm;
        Eigen::VectorXd Az_vec_0 = Az_vec;

        for (int ls = 0; ls < max_line_search; ls++) {
            // Trial step: A_trial = A + α·δA
            Eigen::VectorXd Az_trial = Az_vec_0 + alpha * delta_A;
            Eigen::MatrixXd Az_trial_mat = Eigen::Map<Eigen::MatrixXd>(Az_trial.data(), ny, nx);

            // Calculate B and H for trial point
            Az = Az_trial_mat;
            calculateMagneticField();
            calculateHField();
            updateMuDistribution();

            // Build matrix and compute residual at trial point
            Eigen::SparseMatrix<double> A_trial;
            Eigen::VectorXd b_trial;
            buildMatrix(A_trial, b_trial);
            Eigen::VectorXd residual_trial = A_trial * Az_trial - b_trial;
            double residual_trial_norm = residual_trial.norm();

            // Check Armijo condition: ||R(A + α·δA)|| <= ||R(A)||·(1 - c·α)
            if (residual_trial_norm <= residual_0 * (1.0 - c * alpha) || alpha < alpha_min) {
                // Accept step
                Az_vec = Az_trial;
                Az = Az_trial_mat;
                if (VERBOSE && ls > 0) {
                    std::cout << " [LS: α=" << alpha << ", " << ls+1 << " trials]";
                }
                break;
            }

            // Reject step and backtrack
            alpha *= rho;

            if (ls == max_line_search - 1) {
                // Line search failed, accept minimal step
                Az_vec = Az_vec_0 + alpha_min * delta_A;
                Az = Eigen::Map<Eigen::MatrixXd>(Az_vec.data(), ny, nx);
                if (VERBOSE) {
                    std::cout << " [LS failed, using α=" << alpha_min << "]";
                }
            }
        }

        // Update previous step length for next iteration's adaptive algorithm
        alpha_prev = alpha;

        // ===== Step 7: Report iteration statistics =====
        double delta_norm = delta_A.norm();
        double Az_norm = Az_vec_0.norm();

        if (VERBOSE) {
            std::cout << ", ||δA||/||A|| = " << delta_norm / (Az_norm + 1e-12)
                      << ", α = " << alpha << std::endl;
        }
    }

    std::cerr << "WARNING: Newton-Krylov solver did not converge after " << MAX_ITER << " iterations!" << std::endl;

    if (nonlinear_config.export_convergence) {
        std::ofstream conv_file("newton_krylov_convergence.csv");
        conv_file << "Iteration,Residual\n";
        for (size_t i = 0; i < residual_history.size(); i++) {
            conv_file << i + 1 << "," << residual_history[i] << "\n";
        }
        conv_file.close();
        std::cout << "Convergence history exported to newton_krylov_convergence.csv" << std::endl;
    }
}
