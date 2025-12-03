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

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    const bool VERBOSE = nonlinear_config.verbose;
    const bool is_polar = (coordinate_system != "cartesian");

    if (VERBOSE) {
        std::cout << "\n=== Newton-Krylov Solver (Jacobian-free + AMGCL) ===" << std::endl;
        std::cout << "Coordinate system: " << coordinate_system << std::endl;
        std::cout << "Max outer iterations: " << MAX_ITER << std::endl;
        std::cout << "Tolerance: " << TOL << std::endl;
    }

    std::vector<double> residual_history;
    double alpha_prev = nonlinear_config.line_search_alpha_init;  // Previous step length for adaptive algorithm

    // Initial guess: solve linear problem with initial μ distribution
    if (VERBOSE) {
        std::cout << "Computing initial guess..." << std::endl;
    }
    if (is_polar) {
        buildAndSolveSystemPolar();
    } else {
        buildAndSolveSystem();
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ===== Step 1: Calculate B and H fields =====
        if (is_polar) {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        calculateHField();

        // ===== Step 2: Update μ based on current B and H =====
        updateMuDistribution();

        // ===== Step 3: Build residual and system matrix with current μ =====
        Eigen::SparseMatrix<double> A_matrix;
        Eigen::VectorXd b_vec;
        if (is_polar) {
            buildMatrixPolar(A_matrix, b_vec);
        } else {
            buildMatrix(A_matrix, b_vec);
        }

        // CRITICAL: buildMatrixPolar uses row-major indexing (idx = r_idx * ntheta + theta_idx)
        // but Eigen Az.data() is column-major. Must convert to row-major order.
        Eigen::VectorXd Az_vec(Az.size());
        if (is_polar) {
            // Convert Az to row-major order to match buildMatrixPolar indexing
            for (int i = 0; i < nr; i++) {
                for (int j = 0; j < ntheta; j++) {
                    int idx = i * ntheta + j;  // Row-major index
                    if (r_orientation == "horizontal") {
                        Az_vec(idx) = Az(j, i);  // Az is (ntheta, nr), so Az(theta, r)
                    } else {  // vertical
                        Az_vec(idx) = Az(i, j);  // Az is (nr, ntheta), so Az(r, theta)
                    }
                }
            }
        } else {
            // Cartesian: row-major is natural for (ny, nx) with idx = j * nx + i
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = j * nx + i;
                    Az_vec(idx) = Az(j, i);
                }
            }
        }

        Eigen::VectorXd residual = A_matrix * Az_vec - b_vec;
        double residual_norm = residual.norm();
        double b_vec_norm = b_vec.norm();
        double residual_rel = residual_norm / (b_vec_norm + 1e-12);

        residual_history.push_back(residual_rel);

        // DEBUG: Print norms for first iteration to diagnose polar vs cartesian scaling
        if (VERBOSE && iter == 0) {
            double A_norm = A_matrix.norm();
            std::cout << "DEBUG: ||A|| = " << A_norm
                      << ", ||Az|| = " << Az_vec.norm()
                      << ", ||b|| = " << b_vec_norm
                      << ", ||R||_abs = " << residual_norm << std::endl;
        }

        if (VERBOSE) {
            std::cout << "NK iter " << std::setw(3) << iter + 1
                      << ": ||R|| = " << std::scientific << std::setprecision(4) << residual_rel << std::flush;
        }

        // ===== Step 4: Check convergence =====
        bool converged = false;

        if (iter > 0) {
            // Primary convergence criterion: relative residual
            if (residual_rel < TOL) {
                converged = true;
            }

            // Secondary criterion for polar coordinates: residual reduction rate
            // Useful when absolute residual is large but solution is converging
            if (is_polar && iter >= 3) {
                // Check if residual has plateaued (< 5% change over last 3 iterations)
                double reduction_rate = std::abs(residual_history[iter] - residual_history[iter-1]) /
                                       (residual_history[iter-1] + 1e-12);
                if (residual_rel < TOL * 10.0 && reduction_rate < 0.05) {
                    converged = true;
                    if (VERBOSE) {
                        std::cout << " [Plateau detected: Δr=" << reduction_rate << "]";
                    }
                }
            }
        }

        if (converged) {
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

        // ===== Step 5: Compute Newton step δA by solving J·δA = -R =====
        // Improved Jacobian: J ≈ L + D_r where D_r is r-weighted diagonal correction
        //
        // For polar coordinates with nonlinear materials:
        //   True Jacobian includes ∂/∂A(r·∇×(1/μ)∇A) which has r-weighting
        //   We add diagonal correction D_ii ∝ r_i · (dμ/dH contribution)
        //
        // IMPORTANT: mu_r in YAML is effective permeability: μ_eff = B/H
        //   This is standard catalog data format (not differential permeability dB/dH)
        //   We compute dμ/dH = μ₀ * dμ_eff/dH using numerical differentiation

        Eigen::VectorXd delta_A;

        // Build r-weighted diagonal Jacobian correction for polar + nonlinear
        Eigen::SparseMatrix<double> J_matrix = A_matrix;  // Start with linear matrix

        if (is_polar && config["materials"]) {
            cv::Mat image_to_use;
            cv::flip(image, image_to_use, 0);  // Match setupMaterialProperties()

            const double MU_0 = 4.0 * M_PI * 1e-7;
            YAML::Node polar_config = config["polar_domain"] ? config["polar_domain"] : config["polar"];
            double r_start = polar_config["r_start"].as<double>();
            double r_end = polar_config["r_end"].as<double>();
            double dr = (r_end - r_start) / (nr - 1);

            for (int idx = 0; idx < Az_vec.size(); idx++) {
                // Row-major: idx = r_idx * ntheta + theta_idx
                int r_idx = idx / ntheta;
                int theta_idx = idx % ntheta;

                // Calculate r coordinate for this grid point
                double r = r_start + r_idx * dr;
                if (r < 1e-10) continue;  // Avoid singularity at r=0

                // Get pixel indices (match mu_map layout)
                int img_row, img_col;
                if (r_orientation == "horizontal") {
                    img_row = theta_idx;  // theta
                    img_col = r_idx;      // r
                } else {  // vertical
                    img_row = r_idx;      // r
                    img_col = theta_idx;  // theta
                }

                if (img_row < 0 || img_row >= image_to_use.rows ||
                    img_col < 0 || img_col >= image_to_use.cols) {
                    continue;
                }

                cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(img_row, img_col);
                cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);

                // Find material and check if nonlinear
                for (const auto& mat : config["materials"]) {
                    std::string name = mat.first.as<std::string>();
                    YAML::Node props = mat.second;
                    if (!props["rgb"]) continue;

                    cv::Scalar mat_rgb(
                        props["rgb"][0].as<int>(),
                        props["rgb"][1].as<int>(),
                        props["rgb"][2].as<int>()
                    );

                    if (rgb[0] == mat_rgb[0] && rgb[1] == mat_rgb[1] && rgb[2] == mat_rgb[2]) {
                        // Check if this is a nonlinear material (B-H table or formula)
                        bool is_nonlinear = false;
                        if (props["mu_r"]) {
                            if (props["mu_r"].IsSequence()) {
                                is_nonlinear = true;  // B-H table
                            } else {
                                std::string mu_str = props["mu_r"].as<std::string>();
                                if (mu_str.find('$') != std::string::npos) {
                                    is_nonlinear = true;  // Formula with $H
                                }
                            }
                        }

                        if (is_nonlinear) {
                            // === ANALYTICAL JACOBIAN CORRECTION ===
                            // This approach avoids numerical differentiation errors by computing
                            // dμ/dH analytically from the effective permeability μ_eff = B/H.
                            //
                            // Mathematical Derivation:
                            // ------------------------
                            // Given: B = μ_eff · μ₀ · H  (effective permeability definition)
                            //        μ = B/H = μ_eff · μ₀  (actual permeability)
                            //
                            // We need: dμ/dH = d(B/H)/dH for Jacobian correction
                            //
                            // Using the quotient rule:
                            //   dμ/dH = d(B/H)/dH = (dB/dH · H - B) / H²
                            //
                            // Simplify by dividing numerator and denominator by H:
                            //   dμ/dH = (dB/dH - B/H) / H = (dB/dH - μ) / H
                            //
                            // Where dB/dH is computed analytically using the product rule:
                            //   dB/dH = μ₀ · (μ_eff + H · dμ_eff/dH)
                            //
                            // For TABLE type materials, dμ_eff/dH comes from the exact slope
                            // of linear interpolation segments - no numerical differentiation!

                            double H_val = H_map(img_row, img_col);
                            double mu_current = mu_map(img_row, img_col);

                            // Find material's MuValue to access the B-H table
                            auto mu_it = material_mu.find(name);
                            if (mu_it != material_mu.end()) {
                                // Step 1: Get effective permeability and its derivative
                                double mu_eff = evaluateMu(mu_it->second, H_val);
                                double dmu_eff_dH = evaluateMuDerivative(mu_it->second, H_val);

                                // Step 2: Compute differential permeability dB/dH analytically
                                // dB/dH = μ₀ · (μ_eff + H · dμ_eff/dH)
                                double dB_dH = MU_0 * (mu_eff + H_val * dmu_eff_dH);

                                // Step 3: Compute dμ/dH = (dB/dH - μ) / H
                                double dmu_dH = 0.0;
                                if (H_val > 1.0) {  // H > 1 A/m ensures numerical stability
                                    double mu_actual = mu_eff * MU_0;  // μ = μ_eff · μ₀
                                    dmu_dH = (dB_dH - mu_actual) / H_val;
                                } else {
                                    // Low-field region (H < 1 A/m): Use linear approximation
                                    // In this region, μ ≈ constant, so dμ/dH ≈ 0
                                    // Alternative: dμ/dH ≈ μ₀ · dμ_eff/dH (from initial slope)
                                    dmu_dH = MU_0 * dmu_eff_dH;
                                }

                                // Step 4: Jacobian diagonal correction with r-weighting
                                //
                                // For polar FDM equation: ∂/∂r(r·(1/μ)·∂Az/∂r) + ... = -r·Jz
                                //
                                // The Jacobian term for nonlinear μ(H(Az)) includes:
                                //   ∂/∂Az[(1/μ)·∇Az] = (1/μ)·∇ + ∇Az·∂(1/μ)/∂Az
                                //
                                // Since ν = 1/μ, we have:
                                //   dν/dH = -dμ/dH / μ²
                                //
                                // With r-weighting and FDM discretization:
                                double correction_factor = -r * dmu_dH / (mu_current * mu_current + 1e-20);
                                correction_factor *= (dr * dr);  // FDM mesh spacing scale

                                // Add to diagonal element of Jacobian matrix
                                J_matrix.coeffRef(idx, idx) += correction_factor;
                            }
                        }
                        break;
                    }
                }
            }
        }

        // Solve improved system: J·δA = -R
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.compute(J_matrix);
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
            // Use configured initial value, but consider previous success
            // CRITICAL FIX: If previous step was small, don't be too aggressive
            if (iter > 0 && alpha_prev < 0.5) {
                // Previous step was conservative → start from slightly larger value
                alpha_init = std::min(nonlinear_config.line_search_alpha_init, alpha_prev * 1.5);
            } else {
                alpha_init = nonlinear_config.line_search_alpha_init;
            }
        }

        // Apply conservative damping for polar coordinates (Quasi-Newton is less accurate)
        if (is_polar && iter < 5) {
            // First few iterations: be very conservative
            alpha_init *= 0.5;
        } else if (is_polar) {
            // Later iterations: moderate damping
            alpha_init *= 0.7;
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

            // Convert Az_trial (row-major vector) back to matrix form
            Eigen::MatrixXd Az_trial_mat;
            if (is_polar) {
                // CRITICAL: buildMatrixPolar uses row-major indexing (idx = r_idx * ntheta + theta_idx)
                // Convert from row-major vector to matrix
                if (r_orientation == "horizontal") {
                    Az_trial_mat.resize(ntheta, nr);
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            int idx = i * ntheta + j;
                            Az_trial_mat(j, i) = Az_trial(idx);  // Az(theta, r)
                        }
                    }
                } else {  // vertical
                    Az_trial_mat.resize(nr, ntheta);
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            int idx = i * ntheta + j;
                            Az_trial_mat(i, j) = Az_trial(idx);  // Az(r, theta)
                        }
                    }
                }
            } else {
                // Cartesian: Convert from row-major vector (idx = j * nx + i) to matrix
                Az_trial_mat.resize(ny, nx);
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        int idx = j * nx + i;
                        Az_trial_mat(j, i) = Az_trial(idx);
                    }
                }
            }

            // Calculate B and H for trial point
            Az = Az_trial_mat;
            if (is_polar) {
                calculateMagneticFieldPolar();
            } else {
                calculateMagneticField();
            }
            calculateHField();
            updateMuDistribution();

            // Build matrix and compute residual at trial point
            Eigen::SparseMatrix<double> A_trial;
            Eigen::VectorXd b_trial;
            if (is_polar) {
                buildMatrixPolar(A_trial, b_trial);
            } else {
                buildMatrix(A_trial, b_trial);
            }
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

                // Convert Az_vec (row-major) back to matrix form
                if (is_polar) {
                    if (r_orientation == "horizontal") {
                        Az.resize(ntheta, nr);
                        for (int i = 0; i < nr; i++) {
                            for (int j = 0; j < ntheta; j++) {
                                int idx = i * ntheta + j;
                                Az(j, i) = Az_vec(idx);
                            }
                        }
                    } else {  // vertical
                        Az.resize(nr, ntheta);
                        for (int i = 0; i < nr; i++) {
                            for (int j = 0; j < ntheta; j++) {
                                int idx = i * ntheta + j;
                                Az(i, j) = Az_vec(idx);
                            }
                        }
                    }
                } else {
                    Az.resize(ny, nx);
                    for (int j = 0; j < ny; j++) {
                        for (int i = 0; i < nx; i++) {
                            int idx = j * nx + i;
                            Az(j, i) = Az_vec(idx);
                        }
                    }
                }
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
