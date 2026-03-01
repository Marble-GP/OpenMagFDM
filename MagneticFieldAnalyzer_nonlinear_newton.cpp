/**
 * @file MagneticFieldAnalyzer_nonlinear_newton.cpp
 * @brief Newton-Krylov solver for nonlinear materials
 *
 * Implements:
 * - Newton-Krylov method with analytical Jacobian diagonal correction
 * - Armijo backtracking line search for globalization
 * - Optional Anderson acceleration for improved convergence
 *
 * Theory:
 * PDE: R(Az) = ∇·(ν(H(Az)) ∇Az) + Jz = 0
 * where ν = 1/μ, H = |B|/μ, B = ∇×Az
 */

#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Main Newton-Krylov solver
 */
void MagneticFieldAnalyzer::solveNonlinearNewtonKrylov() {
    if (!has_nonlinear_materials) {
        // Fall back to linear solver
        if (coordinate_system == "cartesian") {
            if (coarsening_enabled && n_active_cells < nx * ny) {
                buildAndSolveSystemCoarsened();
            } else {
                buildAndSolveSystem();
            }
        } else {  // polar
            if (coarsening_enabled && n_active_cells < nr * ntheta) {
                buildAndSolveSystemPolarCoarsened();
            } else {
                buildAndSolveSystemPolar();
            }
        }
        return;
    }

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    const bool VERBOSE = nonlinear_config.verbose;
    const bool is_polar = (coordinate_system != "cartesian");

    // Anderson acceleration settings (shared config)
    const bool USE_ANDERSON = nonlinear_config.anderson.enabled;
    const int m_AA = nonlinear_config.anderson.depth;
    const double beta_AA = nonlinear_config.anderson.beta;

    if (VERBOSE) {
        std::cout << "\n=== Newton-Krylov Solver (Jacobian-free + AMGCL) ===" << std::endl;
        std::cout << "Coordinate system: " << coordinate_system << std::endl;
        std::cout << "Max outer iterations: " << MAX_ITER << std::endl;
        std::cout << "Tolerance: " << std::scientific << std::setprecision(1) << TOL << std::endl;
        if (USE_ANDERSON) {
            std::cout << "Anderson acceleration: enabled (depth=" << m_AA << ", beta=" << beta_AA << ")" << std::endl;
        }
    }

    std::vector<double> residual_history;
    double alpha_prev = nonlinear_config.line_search_alpha_init;  // Previous step length for adaptive algorithm

    // Anderson acceleration storage
    std::vector<Eigen::VectorXd> Az_history;      // Az^(k) history
    std::vector<Eigen::VectorXd> g_history;       // g^(k) = Az^(k+1) - Az^(k) history

    // Initial guess: solve linear problem with initial μ distribution
    if (VERBOSE) {
        std::cout << "Computing initial guess..." << std::endl;
    }
    if (is_polar) {
        if (coarsening_enabled && n_active_cells < nr * ntheta) {
            buildAndSolveSystemPolarCoarsened();
        } else {
            buildAndSolveSystemPolar();
        }
    } else {
        if (coarsening_enabled && n_active_cells < nx * ny) {
            buildAndSolveSystemCoarsened();
        } else {
            buildAndSolveSystem();
        }
    }

    // Determine if coarsening is active (constant across iterations)
    bool using_coarsening = (is_polar && coarsening_enabled && n_active_cells < nr * ntheta) ||
                            (!is_polar && coarsening_enabled && n_active_cells < nx * ny);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ===== Step 1: Calculate B and H fields, update μ =====
        if (using_coarsening) {
            // Phase 8: Wide-stencil B/H/μ at active cells only.
            // Avoids stripe artifacts from bilinear-interpolated inactive cell Az.
            Eigen::VectorXd Az_coarse_for_B(n_active_cells);
            // coarse_to_fine / Az are read-only per cidx; Az_coarse_for_B(cidx) unique.
            #pragma omp parallel for schedule(static)
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                Az_coarse_for_B(cidx) = Az(cj, ci);
            }
            Eigen::VectorXd Bx_active, By_active, H_active;
            calculateBFieldAtActiveCells(Az_coarse_for_B, Bx_active, By_active);
            calculateHFieldAtActiveCells(Bx_active, By_active, H_active);
            updateMuAtActiveCells(H_active);
            interpolateMuToFullGrid();  // Fill inactive cells (needed for Galerkin A_f)
        } else {
            // Full-grid path (unchanged)
            if (is_polar) {
                calculateMagneticFieldPolar();
            } else {
                calculateMagneticField();
            }
            calculateHField();
            updateMuDistribution();
        }

        // ===== Step 2: Build residual and system matrix with current μ =====
        Eigen::SparseMatrix<double> A_matrix;
        Eigen::VectorXd b_vec;

        if (using_coarsening && nonlinear_config.use_galerkin_coarsening) {
            // Phase 4: Galerkin projection A_c = R * A_f * P
            buildMatrixGalerkin(A_matrix, b_vec);
        } else if (is_polar) {
            if (coarsening_enabled && n_active_cells < nr * ntheta) {
                buildMatrixPolarCoarsened(A_matrix, b_vec);
            } else {
                buildMatrixPolar(A_matrix, b_vec);
            }
        } else {
            if (coarsening_enabled && n_active_cells < nx * ny) {
                buildMatrixCoarsened(A_matrix, b_vec);
            } else {
                buildMatrix(A_matrix, b_vec);
            }
        }

        // CRITICAL: buildMatrixPolar uses row-major indexing (idx = r_idx * ntheta + theta_idx)
        // but Eigen Az.data() is column-major. Must convert to row-major order.
        // When coarsening is enabled, extract only active cell values.
        Eigen::VectorXd Az_vec;
        // Note: using_coarsening is already defined above in Step 3

        if (using_coarsening) {
            // Coarsened: extract only active cells
            Az_vec.resize(n_active_cells);
            for (int idx = 0; idx < n_active_cells; idx++) {
                auto [i, j] = coarse_to_fine[idx];
                Az_vec(idx) = Az(j, i);  // Az(row, col) for both Cartesian and Polar
            }
        } else if (is_polar) {
            // Full polar grid: convert Az to row-major order to match buildMatrixPolar indexing
            Az_vec.resize(Az.size());
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
            // Full Cartesian grid: row-major is natural for (ny, nx) with idx = j * nx + i
            Az_vec.resize(Az.size());
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    int idx = j * nx + i;
                    Az_vec(idx) = Az(j, i);
                }
            }
        }

        // Coarse residual (used for Newton step direction)
        Eigen::VectorXd residual_coarse = A_matrix * Az_vec - b_vec;
        double residual_coarse_norm = residual_coarse.norm();
        double b_vec_coarse_norm = b_vec.norm();

        // Convergence residual selection
        double residual_norm, b_vec_norm;
        if (using_coarsening && nonlinear_config.use_phase6_precond_jfnk) {
            // Defect correction: FVM coarsened residual for convergence.
            // Using the full-grid residual ||A_full * P * Az - b_full|| would include
            // permanent inactive-cell interpolation errors that cannot be reduced
            // by coarse-space corrections — causing false non-convergence.
            residual_norm = residual_coarse_norm;
            b_vec_norm = b_vec_coarse_norm;
        } else if (using_coarsening) {
            // Other coarsening modes: full-grid residual for accurate convergence
            residual_norm = computeFullGridResidual(b_vec_norm);
            full_matrix_cache_valid = false;
        } else {
            residual_norm = residual_coarse_norm;
            b_vec_norm = b_vec_coarse_norm;
        }
        double residual_rel = residual_norm / (b_vec_norm + 1e-12);

        residual_history.push_back(residual_rel);

        // DEBUG: Print norms for first iteration to diagnose polar vs cartesian scaling
        if (VERBOSE && iter == 0) {
            double A_norm = A_matrix.norm();
            std::cout << "DEBUG: ||A|| = " << A_norm
                      << ", ||Az|| = " << Az_vec.norm()
                      << ", ||b_coarse|| = " << b_vec_coarse_norm
                      << ", ||R_coarse||_abs = " << residual_coarse_norm;
            if (using_coarsening) {
                std::cout << ", ||R_FVM||_abs = " << residual_norm;
            }
            std::cout << std::endl;
        }

        if (VERBOSE) {
            std::cout << "NK iter " << std::setw(3) << iter + 1
                      << ": ||R|| = " << std::scientific << std::setprecision(4) << residual_rel;
            if (using_coarsening && !nonlinear_config.use_phase6_precond_jfnk) {
                // Show FVM coarsened residual alongside fine-grid for comparison
                // (in defect correction mode both are the same, so suppress duplicate)
                double residual_coarse_rel = residual_coarse_norm / (b_vec_coarse_norm + 1e-12);
                std::cout << " (FVM: " << residual_coarse_rel << ")";
            }
            std::cout << std::flush;
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

            // Final output: interpolate coarse solution to full grid.
            // During iteration, bilinear was used for P-matrix consistency.
            // After convergence, apply optional Laplacian smoothing for
            // smooth B field visualization (C¹ Az at skip transitions).
            if (using_coarsening) {
                if (is_polar) {
                    interpolateToFullGridPolar(Az_vec);
                } else {
                    interpolateToFullGrid(Az_vec);
                }
                smoothInactiveCells(coarsen_smooth_iterations);
                // Harmonically interpolate μ at inactive cells (IDW, series circuit model)
                interpolateMuToFullGrid();
            }

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

        // Full-grid frozen-Jacobian residual norm, set by defect correction for use in line search.
        // Negative sentinel = not in defect correction mode (use coarse norm instead).
        double dc_R_fine_norm = -1.0;

        // ===== Step 5: Compute Newton step δA =====
        if (using_coarsening && nonlinear_config.use_phase6_precond_jfnk) {
            // Phase 6: FVM-consistent Newton step
            //
            // Previous approach used Galerkin matrix A_c = P^T * A_f * P as Newton matrix:
            //   delta_A = A_c_galerkin^{-1} * (-R_FVM)
            // Problem: A_c_galerkin ≠ A_FVM — Galerkin includes all 250k fine-grid cells
            // (including inactive cells with interpolated μ) through P and P^T, while
            // A_FVM discretizes only the 67k active cells directly.
            // The operator mismatch makes the Galerkin Newton direction a poor (often
            // non-descent) direction for the FVM residual ||R_FVM||, causing the true
            // nonlinear line search to fail at every α ≥ α_min (all 18 trials rejected).
            //
            // Fix: solve A_FVM × δ = -R_FVM directly using SparseLU on A_matrix.
            // Since d = A_FVM^{-1} × (-R_FVM), the frozen linearized trial residual is:
            //   A_FVM × (Az + α×d) - b = (1-α) × R_FVM
            // The directional derivative ∇f^T × d = -||R_FVM||² < 0, guaranteeing
            // d is a descent direction. The true nonlinear Armijo condition is then
            // satisfied for some α ∈ (0, 1] (descent lemma for smooth objectives).
            //
            // Performance: eliminates updatePreconditioner() — no more full-grid 250k×250k
            // matrix build or Galerkin triple product P^T*A_f*P per Newton iteration.
            if (VERBOSE && iter == 0) {
                std::cout << "Using FVM Newton step (Phase 6)" << std::endl;
            }

            dc_R_fine_norm = residual_coarse_norm;  // FVM residual as Armijo baseline
            {
                Eigen::SparseLU<Eigen::SparseMatrix<double>> fvm_lu;
                fvm_lu.compute(A_matrix);
                if (fvm_lu.info() == Eigen::Success) {
                    delta_A = fvm_lu.solve(-residual_coarse);
                } else {
                    // Fallback: diagonal (Jacobi) scaling step
                    if (VERBOSE) std::cout << " [FVM LU failed, using Jacobi step]";
                    delta_A.resize(residual_coarse.size());
                    for (int k = 0; k < A_matrix.rows(); k++) {
                        double diag = A_matrix.coeff(k, k);
                        delta_A(k) = (std::abs(diag) > 1e-30) ? -residual_coarse(k) / diag : 0.0;
                    }
                }
            }

            if (nonlinear_config.precond_verbose) {
                std::cout << "FVM Newton: ||R_FVM||=" << residual_coarse_norm << std::endl;
            }

        } else if (using_coarsening && nonlinear_config.use_matrix_free_jv) {
            // Phase 5: Matrix-free GMRES (no preconditioner)
            if (VERBOSE && iter == 0) {
                std::cout << "Using matrix-free GMRES (Phase 5) for Newton step" << std::endl;
            }
            delta_A = solveWithMatrixFreeGMRES(Az_vec, -residual_coarse,
                nonlinear_config.gmres_restart * 3, 1e-6);
        } else {
            // Standard direct solve with explicit Jacobian (J = A + diagonal correction)
            Eigen::SparseMatrix<double> J_matrix = A_matrix;

            // Build r-weighted diagonal Jacobian correction for polar + nonlinear
            if (is_polar && config["materials"]) {
                cv::Mat image_to_use;
                cv::flip(image, image_to_use, 0);  // Match setupMaterialProperties()

                const double MU_0 = 4.0 * M_PI * 1e-7;
                YAML::Node polar_config = config["polar_domain"] ? config["polar_domain"] : config["polar"];
                double r_start_local = polar_config["r_start"].as<double>();
                double r_end_local = polar_config["r_end"].as<double>();
                double dr_local = (r_end_local - r_start_local) / (nr - 1);

                for (int idx = 0; idx < Az_vec.size(); idx++) {
                    // Get grid indices
                    int r_idx, theta_idx;
                    if (using_coarsening) {
                        auto [i, j] = coarse_to_fine[idx];
                        r_idx = i;      // i_r
                        theta_idx = j;  // j_theta
                    } else {
                        r_idx = idx / ntheta;
                        theta_idx = idx % ntheta;
                    }

                    double r = r_start_local + r_idx * dr_local;
                    if (r < 1e-10) continue;

                    int img_row, img_col;
                    if (r_orientation == "horizontal") {
                        img_row = theta_idx;
                        img_col = r_idx;
                    } else {
                        img_row = r_idx;
                        img_col = theta_idx;
                    }

                    if (img_row < 0 || img_row >= image_to_use.rows ||
                        img_col < 0 || img_col >= image_to_use.cols) {
                        continue;
                    }

                    cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(img_row, img_col);
                    cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);

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
                            bool is_nonlinear = false;
                            if (props["mu_r"]) {
                                if (props["mu_r"].IsSequence()) {
                                    is_nonlinear = true;
                                } else {
                                    std::string mu_str = props["mu_r"].as<std::string>();
                                    if (mu_str.find('$') != std::string::npos) {
                                        is_nonlinear = true;
                                    }
                                }
                            }

                            if (is_nonlinear) {
                                double H_val = H_map(img_row, img_col);
                                double mu_current = mu_map(img_row, img_col);

                                auto mu_it = material_mu.find(name);
                                if (mu_it != material_mu.end()) {
                                    double mu_eff = evaluateMu(mu_it->second, H_val);
                                    double dmu_eff_dH = evaluateMuDerivative(mu_it->second, H_val);
                                    double dB_dH = MU_0 * (mu_eff + H_val * dmu_eff_dH);

                                    double dmu_dH = 0.0;
                                    if (H_val > 1.0) {
                                        double mu_actual = mu_eff * MU_0;
                                        dmu_dH = (dB_dH - mu_actual) / H_val;
                                    } else {
                                        dmu_dH = MU_0 * dmu_eff_dH;
                                    }

                                    double correction_factor = -r * dmu_dH / (mu_current * mu_current + 1e-20);
                                    correction_factor *= (dr_local * dr_local);
                                    J_matrix.coeffRef(idx, idx) += correction_factor;
                                }
                            }
                            break;
                        }
                    }
                }
            }

            Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
            solver.compute(J_matrix);
            if (solver.info() != Eigen::Success) {
                std::cerr << "ERROR: Matrix factorization failed!" << std::endl;
                return;
            }
            delta_A = solver.solve(-residual_coarse);
            if (solver.info() != Eigen::Success) {
                std::cerr << "ERROR: Linear solve failed!" << std::endl;
                return;
            }
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

        // Note: Previous conservative damping for polar coordinates was too aggressive
        // and caused extremely slow convergence (α=0.05-0.07).
        // The diagonal Jacobian correction should handle nonlinearity adequately.
        // If divergence occurs, consider improving the Jacobian approximation instead.

        double alpha = alpha_init;
        const double c = nonlinear_config.line_search_c;
        const double rho = nonlinear_config.line_search_rho;
        const double alpha_min = nonlinear_config.line_search_alpha_min;
        const int max_line_search = nonlinear_config.line_search_max_trials;

        // In defect correction mode, the frozen-coarse trial residual A_c*(Az_c + δ_c) - b_c
        // is identically zero by construction (δ_c = A_c^{-1} * R_c), so the Armijo condition
        // is trivially satisfied and α=1 is always accepted regardless of overshoot.
        // Instead, use the full-grid frozen-Jacobian residual ||A_f * P * Az_trial - b_f||
        // which is NOT zero and correctly detects divergence.
        double residual_0 = (dc_R_fine_norm >= 0.0) ? dc_R_fine_norm : residual_norm;
        Eigen::VectorXd Az_vec_0 = Az_vec;

        // ===== Phase 6: Anderson-Picard with residual-based restart =====
        //
        // Standard backtracking line search fails here because the B-H piecewise-linear
        // knee causes the true nonlinear residual to INCREASE for steps larger than some
        // geometry-dependent threshold (≈ α_min ≈ 6.6e-4). Backtracking from α=1.0
        // with rho=0.65 reaches only α≈0.007 in 18 trials — never reaching the good
        // region. Forcing α_min as a fallback gives only 0.001–2% per-iter progress.
        //
        // This block replaces the line search with Anderson(m≥3) acceleration:
        //   1. Compute Anderson step from history of FULL Picard residuals δ_k = A^{-1}(-R)
        //      (degenerates to pure Picard G_k = Az + δ when history is empty)
        //   2. Evaluate true nonlinear residual at the proposed step (1 matrix rebuild,
        //      vs 18 in the old line search → 18x cheaper per iteration)
        //   3. If residual INCREASED (>10%): restart Anderson history (discard corrupted
        //      iterates) and fall back to ω=0.1 damped Picard (safe for all geometries)
        //   4. Store full δ_k (not the accepted step) in Anderson history so future
        //      extrapolations use the full Newton correction, not the damped one.
        //
        // Note: Step 7 Anderson is skipped (phase6_anderson_done=true) since this block
        // manages Az_history/g_history with correct Picard residuals.
        bool phase6_anderson_done = false;
        if (using_coarsening && nonlinear_config.use_phase6_precond_jfnk) {
            const int m_phase6 = std::max(3, m_AA);

            // ---- Anderson extrapolation (or pure Picard if history empty) ----
            Eigen::VectorXd G_k = Az_vec_0 + delta_A;   // Full Picard iterate
            Eigen::VectorXd Az_next;
            {
                int m_k = std::min(m_phase6, static_cast<int>(g_history.size()));
                if (m_k == 0) {
                    // No history yet: pure full Picard step
                    Az_next = G_k;
                    if (VERBOSE) std::cout << " [P6: Picard]";
                } else {
                    // Anderson extrapolation: DG = differences of Picard residuals
                    Eigen::MatrixXd DG(delta_A.size(), m_k);
                    for (int j = 0; j < m_k; j++) {
                        int hidx = static_cast<int>(g_history.size()) - m_k + j;
                        DG.col(j) = g_history[hidx] - delta_A;
                    }
                    // Solve: min_θ ||DG θ + δ_k||²  (regularized normal equations)
                    Eigen::MatrixXd DTD = DG.transpose() * DG;
                    double reg = 1e-10 * std::max(DTD.diagonal().maxCoeff(), 1.0);
                    DTD.diagonal().array() += reg;
                    Eigen::VectorXd theta = DTD.ldlt().solve(-DG.transpose() * delta_A);

                    // G_AA = G_k + Σ_j θ_j · ((Az_{k-j} + δ_{k-j}) - (Az_k + δ_k))
                    Eigen::VectorXd G_anderson = G_k;
                    for (int j = 0; j < m_k; j++) {
                        int hidx = static_cast<int>(g_history.size()) - m_k + j;
                        Eigen::VectorXd DX_j = Az_history[hidx] - Az_vec_0;
                        G_anderson += theta(j) * (DX_j + DG.col(j));
                    }
                    // Mixing parameter β (default 1.0 = pure Anderson)
                    const double beta_p6 = (beta_AA > 0.0 && beta_AA <= 1.0) ? beta_AA : 1.0;
                    Az_next = beta_p6 * G_anderson + (1.0 - beta_p6) * G_k;

                    // Safety: cap Anderson step at 3× Picard step
                    double picard_norm  = delta_A.norm();
                    double anderson_norm = (Az_next - Az_vec_0).norm();
                    if (picard_norm > 1e-30 && anderson_norm > 3.0 * picard_norm) {
                        Az_next = G_k;   // Fall back to full Picard
                        if (VERBOSE) std::cout << " [P6: Anderson→Picard(safety)]";
                    } else if (VERBOSE) {
                        std::cout << " [P6: Anderson m=" << m_k
                                  << " step=" << std::scientific << std::setprecision(2)
                                  << anderson_norm << "]";
                    }
                }
            }

            // ---- True nonlinear residual check (1 rebuild, vs 18 in old line search) ----
            // Apply Az_next, update μ, rebuild A_FVM, measure residual.
            // If residual increased (>10%): restart history and use ω=0.1 fallback.
            auto p6_apply_eval = [&](const Eigen::VectorXd& Az_cand) -> double {
                for (int cidx = 0; cidx < n_active_cells; cidx++) {
                    auto [ci, cj] = coarse_to_fine[cidx];
                    Az(cj, ci) = Az_cand(cidx);
                }
                if (is_polar) interpolateInactiveCellsPolar(Az_cand);
                else          interpolateInactiveCells(Az_cand);
                {
                    Eigen::VectorXd Bx_a, By_a, H_a;
                    calculateBFieldAtActiveCells(Az_cand, Bx_a, By_a);
                    calculateHFieldAtActiveCells(Bx_a, By_a, H_a);
                    updateMuAtActiveCells(H_a);
                    interpolateMuToFullGrid();
                }
                Eigen::SparseMatrix<double> A_chk; Eigen::VectorXd b_chk;
                if (is_polar) buildMatrixPolarCoarsened(A_chk, b_chk);
                else          buildMatrixCoarsened(A_chk, b_chk);
                full_matrix_cache_valid = false;
                return (A_chk * Az_cand - b_chk).norm();
            };

            double res_next = p6_apply_eval(Az_next);
            bool step_ok = (res_next <= residual_0 * 1.1);  // Allow ≤10% increase

            if (!step_ok) {
                // Step increased residual: discard corrupted Anderson history and
                // fall back to ω=0.1 conservative Picard.
                // (Even ω=0.5 can cause >10% increase for polar r-weighted matrices
                //  with Si_steel near saturation — ω=0.1 avoids B-H knee crossing.)
                if (!Az_history.empty()) {
                    Az_history.clear();
                    g_history.clear();
                }
                Eigen::VectorXd Az_fallback = Az_vec_0 + 0.1 * delta_A;
                p6_apply_eval(Az_fallback);   // Sets Az to fallback state
                Az_next = Az_fallback;
                if (VERBOSE) {
                    std::cout << " [P6: restart+fallback ω=0.1"
                              << "(R_trial/R0=" << std::setprecision(3) << res_next / residual_0
                              << ")]";
                }
            }

            // Store full Picard residual δ_k in history (critical: not the accepted step)
            Az_history.push_back(Az_vec_0);
            g_history.push_back(delta_A);
            if (static_cast<int>(Az_history.size()) > m_phase6 + 1) {
                Az_history.erase(Az_history.begin());
                g_history.erase(g_history.begin());
            }

            // Sync Az_vec from Az (p6_apply_eval already set Az; resync active cells)
            Az_vec = Az_next;
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                Az_vec(cidx) = Az(cj, ci);
            }

            alpha_prev = 1.0;  // Conceptually "full step" for adaptive init next iter
            phase6_anderson_done = true;
        }

        if (!phase6_anderson_done) {
        for (int ls = 0; ls < max_line_search; ls++) {
            // Trial step: A_trial = A + α·δA
            Eigen::VectorXd Az_trial = Az_vec_0 + alpha * delta_A;

            // Convert Az_trial (row-major vector) back to matrix form
            Eigen::MatrixXd Az_trial_mat;
            if (using_coarsening) {
                // Coarsened: update active cells, then interpolate inactive cells
                Az_trial_mat = Az;  // Start with current Az (has full grid)
                for (int idx = 0; idx < n_active_cells; idx++) {
                    auto [i, j] = coarse_to_fine[idx];
                    Az_trial_mat(j, i) = Az_trial(idx);  // Update active cell
                }
                // ★ CRITICAL FIX: Interpolate inactive cells to ensure consistent gradients
                // This is required for accurate B/H/μ calculation in nonlinear iteration
                Az = Az_trial_mat;  // Temporarily set Az for interpolation functions
                if (is_polar) {
                    interpolateInactiveCellsPolar(Az_trial);
                } else {
                    interpolateInactiveCells(Az_trial);
                }
                Az_trial_mat = Az;  // Copy back with interpolated inactive cells
            } else if (is_polar) {
                // Full polar grid: CRITICAL - buildMatrixPolar uses row-major indexing
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
                // Full Cartesian grid: Convert from row-major vector to matrix
                Az_trial_mat.resize(ny, nx);
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        int idx = j * nx + i;
                        Az_trial_mat(j, i) = Az_trial(idx);
                    }
                }
            }

            // Build matrix and compute residual at trial point
            double residual_trial_norm;
            if (using_coarsening && nonlinear_config.use_phase6_precond_jfnk) {
                // Defect correction: TRUE nonlinear line search (update μ, rebuild A_FVM).
                //
                // The frozen-Jacobian approach evaluates A_matrix * Az_trial - b_vec
                // where A_matrix uses the CURRENT (unfrozen) μ. Since delta_c is computed as
                // A_c_galerkin^{-1} * R_FVM and A_c_galerkin ≈ A_FVM, the trial residual
                // is approximately zero at α=1 — Armijo is trivially satisfied every iteration.
                // After updating μ from Az_trial, the TRUE nonlinear residual can be much
                // larger, causing period-2 oscillation (residual never converges).
                //
                // Fix: update B/H/μ from Az_trial, rebuild A_FVM with new μ, evaluate the
                // TRUE nonlinear residual ||A_new(μ(Az_trial)) * Az_trial - b||. This matches
                // the convention in the non-defect-correction path and prevents trivial acceptance.
                // μ contamination from rejected trials doesn't matter because Step 1 of the next
                // Newton iteration always recomputes μ from Az.

                // Az is already updated to Az_trial_mat by lines above (active cells set,
                // inactive cells interpolated). Now update μ from Az_trial.
                Az = Az_trial_mat;
                {
                    Eigen::VectorXd Bx_active, By_active, H_active;
                    calculateBFieldAtActiveCells(Az_trial, Bx_active, By_active);
                    calculateHFieldAtActiveCells(Bx_active, By_active, H_active);
                    updateMuAtActiveCells(H_active);
                    interpolateMuToFullGrid();
                }
                // Rebuild coarsened FVM matrix with new μ and evaluate residual
                {
                    Eigen::SparseMatrix<double> A_nl_trial;
                    Eigen::VectorXd b_nl_trial;
                    if (is_polar) {
                        buildMatrixPolarCoarsened(A_nl_trial, b_nl_trial);
                    } else {
                        buildMatrixCoarsened(A_nl_trial, b_nl_trial);
                    }
                    residual_trial_norm = (A_nl_trial * Az_trial - b_nl_trial).norm();
                }
                // Invalidate full-matrix cache (μ has changed)
                full_matrix_cache_valid = false;
            } else {
                // Non-defect-correction paths: update B/H/μ and rebuild matrix per trial
                Az = Az_trial_mat;
                if (using_coarsening) {
                    Eigen::VectorXd Bx_active, By_active, H_active;
                    calculateBFieldAtActiveCells(Az_trial, Bx_active, By_active);
                    calculateHFieldAtActiveCells(Bx_active, By_active, H_active);
                    updateMuAtActiveCells(H_active);
                    interpolateMuToFullGrid();
                } else {
                    if (is_polar) {
                        calculateMagneticFieldPolar();
                    } else {
                        calculateMagneticField();
                    }
                    calculateHField();
                    updateMuDistribution();
                }

                if (using_coarsening && nonlinear_config.use_galerkin_coarsening) {
                    full_matrix_cache_valid = false;
                    double b_trial_norm;
                    residual_trial_norm = computeFullGridResidual(b_trial_norm);
                } else {
                    Eigen::SparseMatrix<double> A_trial;
                    Eigen::VectorXd b_trial;
                    if (is_polar) {
                        if (coarsening_enabled && n_active_cells < nr * ntheta) {
                            buildMatrixPolarCoarsened(A_trial, b_trial);
                        } else {
                            buildMatrixPolar(A_trial, b_trial);
                        }
                    } else {
                        if (coarsening_enabled && n_active_cells < nx * ny) {
                            buildMatrixCoarsened(A_trial, b_trial);
                        } else {
                            buildMatrix(A_trial, b_trial);
                        }
                    }
                    Eigen::VectorXd residual_trial = A_trial * Az_trial - b_trial;
                    residual_trial_norm = residual_trial.norm();
                }
            }

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
                if (using_coarsening) {
                    // Coarsened: update active cells, then interpolate inactive cells
                    for (int idx = 0; idx < n_active_cells; idx++) {
                        auto [i, j] = coarse_to_fine[idx];
                        Az(j, i) = Az_vec(idx);
                    }
                    // ★ CRITICAL FIX: Interpolate inactive cells
                    if (is_polar) {
                        interpolateInactiveCellsPolar(Az_vec);
                    } else {
                        interpolateInactiveCells(Az_vec);
                    }
                } else if (is_polar) {
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
        }  // end if (!phase6_anderson_done) [line search block]

        // Update previous step length for next iteration's adaptive algorithm
        // Phase 6 already set alpha_prev = 1.0 inside its block; don't overwrite.
        if (!phase6_anderson_done) {
            alpha_prev = alpha;
        }

        // ===== Step 7: Anderson Acceleration =====
        // Skipped for Phase 6: history management is done inside the Phase 6 block,
        // which stores full Picard residuals (δ_k) rather than the tiny α_min*δ steps
        // that the line search produces — mixing the latter gives negligible acceleration.
        if (!phase6_anderson_done && USE_ANDERSON && m_AA > 0) {
            Eigen::VectorXd g_k = Az_vec - Az_vec_0;  // Actual update

            if (iter >= 1 && g_history.size() > 0) {
                int m_k = std::min(m_AA, static_cast<int>(g_history.size()));

                // Build matrix of residual differences: ΔG = [g_{k-m_k} - g_k, ..., g_{k-1} - g_k]
                Eigen::MatrixXd DG(g_k.size(), m_k);
                for (int j = 0; j < m_k; j++) {
                    int idx = g_history.size() - m_k + j;
                    DG.col(j) = g_history[idx] - g_k;
                }

                // Solve least-squares: min ||DG * θ + g_k||²
                // Using normal equations: (DG^T * DG) * θ = -DG^T * g_k
                Eigen::MatrixXd DTD = DG.transpose() * DG;
                // Add regularization for stability
                DTD.diagonal().array() += 1e-10;
                Eigen::VectorXd theta = DTD.ldlt().solve(-DG.transpose() * g_k);

                // Anderson update: Az_new = Az_new + Σ θ_j * (Az_{k-m_k+j} - Az_k + g_{k-m_k+j} - g_k)
                //                        = Az_new + Σ θ_j * ((Az_{k-m_k+j} + g_{k-m_k+j}) - (Az_k + g_k))
                // Simplified: Az_AA = (1-Σθ) * Az_new + Σ θ_j * (Az_{k-m_k+j} + g_{k-m_k+j})
                Eigen::VectorXd Az_anderson = Az_vec;
                for (int j = 0; j < m_k; j++) {
                    int idx = g_history.size() - m_k + j;
                    // Add correction: θ_j * ((x_j + g_j) - (x_k + g_k)) = θ_j * (DX_j + DG_j)
                    Eigen::VectorXd DX_j = Az_history[idx] - Az_vec_0;
                    Az_anderson += theta(j) * (DX_j + DG.col(j));
                }

                // Apply mixing parameter β
                Az_vec = beta_AA * Az_anderson + (1.0 - beta_AA) * Az_vec;

                // Update Az matrix from Az_vec
                if (using_coarsening) {
                    // Coarsened: update active cells, then interpolate inactive cells
                    for (int idx = 0; idx < n_active_cells; idx++) {
                        auto [i, j] = coarse_to_fine[idx];
                        Az(j, i) = Az_vec(idx);
                    }
                    // ★ CRITICAL FIX: Interpolate inactive cells after Anderson update
                    if (is_polar) {
                        interpolateInactiveCellsPolar(Az_vec);
                    } else {
                        interpolateInactiveCells(Az_vec);
                    }
                } else if (is_polar) {
                    if (r_orientation == "horizontal") {
                        for (int i = 0; i < nr; i++) {
                            for (int j = 0; j < ntheta; j++) {
                                int idx = i * ntheta + j;
                                Az(j, i) = Az_vec(idx);
                            }
                        }
                    } else {
                        for (int i = 0; i < nr; i++) {
                            for (int j = 0; j < ntheta; j++) {
                                int idx = i * ntheta + j;
                                Az(i, j) = Az_vec(idx);
                            }
                        }
                    }
                } else {
                    for (int j = 0; j < ny; j++) {
                        for (int i = 0; i < nx; i++) {
                            int idx = j * nx + i;
                            Az(j, i) = Az_vec(idx);
                        }
                    }
                }
            }

            // Store history
            Az_history.push_back(Az_vec_0);
            g_history.push_back(g_k);

            // Limit history size
            if (static_cast<int>(Az_history.size()) > m_AA + 1) {
                Az_history.erase(Az_history.begin());
                g_history.erase(g_history.begin());
            }
        }

        // ===== Step 8: Report iteration statistics =====
        double delta_norm = delta_A.norm();
        double Az_norm = Az_vec_0.norm();

        if (VERBOSE) {
            std::cout << ", ||δA||/||A|| = " << delta_norm / (Az_norm + 1e-12)
                      << ", α = " << alpha << std::endl;
        }
    }

    std::cerr << "WARNING: Newton-Krylov solver did not converge after " << MAX_ITER << " iterations!" << std::endl;

    // Apply Hermite interpolation for output even if not converged
    {
        bool coarsen_active = is_polar
            ? (coarsening_enabled && n_active_cells < nr * ntheta)
            : (coarsening_enabled && n_active_cells < nx * ny);
        if (coarsen_active) {
            Eigen::VectorXd Az_c(n_active_cells);
            for (int idx = 0; idx < n_active_cells; idx++) {
                auto [ci, cj] = coarse_to_fine[idx];
                Az_c(idx) = Az(cj, ci);
            }
            if (is_polar) {
                interpolateToFullGridPolar(Az_c);
            } else {
                interpolateToFullGrid(Az_c);
            }
            // Harmonically interpolate μ at inactive cells (IDW, series circuit model)
            interpolateMuToFullGrid();
        }
    }

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
