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
#include <sstream>
#include <cmath>
#include <Eigen/Dense>
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Main Newton-Krylov solver
 */
void MagneticFieldAnalyzer::solveNonlinearNewtonKrylov() {
    // [Phase BJ-8 Path D / v1.5.1] All NK iterations and the initial-guess
    // dispatch run on the full grid via AMGCL+EW. The custom Galerkin
    // coarsening machinery (Phase 6 + Galerkin, Phase 5 matrix-free GMRES,
    // fine-finishing Newton-Picard, coarse mask + P/R operators) has been
    // removed; the previously-named "Standard direct solve" path is the
    // only path. AMGCL's internal smoothed_aggregation multigrid handles
    // multi-resolution natively, replacing the OpenMagFDM-side P/R/A_c
    // machinery that produced wrong-answer flux (~1/10 of true) on
    // saturated polar IPMSM problems (Phase BJ-1/4/5 diagnostics).
    if (!has_nonlinear_materials) {
        // Fall back to linear solver (always full grid post-BJ-8)
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

    // [v1.6 Stage 2] Field-adaptive coarsening: build the coarse mask from an
    // initial full-grid solution's |B| field (keeps saturated/high-gradient iron +
    // air gap + boundaries fine, coarsens only smooth bulk). Must run BEFORE the
    // use_coarse flag below so the coarse-native path picks up the new mask.
    bool coarse_skip_init_guess = false;
    if (adaptive_mesh_enabled && is_polar) {
        if (std::getenv("OMFDM_COARSE_TRUEINIT")) {
            // [DIAGNOSTIC] Start the coarse solve from the DECIMATED true (full
            // nonlinear) solution. If the coarse residual stays small and the flux
            // stays correct, the coarse fixed point == the true solution (so the
            // earlier 1/6 was an initial-guess/convergence artefact). If it drifts
            // back to ~1/6, the coarse operator's fixed point is genuinely different
            // (a coarse mu/discretisation accuracy limit, not a bug).
            std::cout << "[COARSE_TRUEINIT] full nonlinear solve for the true initial guess..." << std::endl;
            adaptive_mesh_enabled = false;     // inner call runs the FULL nonlinear NK
            solveNonlinearNewtonKrylov();      // -> member Az = true full solution
            adaptive_mesh_enabled = true;
            calculateMagneticFieldPolar();     // -> Br/Btheta from the true Az
            generateAdaptiveCoarseningMask();  // mask from the true field
            coarse_skip_init_guess = true;     // keep member Az = true solution (don't re-solve)
        } else {
            if (VERBOSE) std::cout << "Adaptive mesh: initial full solve for the |B| indicator..." << std::endl;
            buildAndSolveSystemPolar();        // full linear solve -> member Az
            calculateMagneticFieldPolar();     // -> Br, Btheta
            generateAdaptiveCoarseningMask();  // -> active_cells / coarse_to_fine / n_active_cells / coarsening_enabled
        }
    }

    // [v1.6 Stage 1b] Coarse-native Newton-Krylov. When an adaptive coarsening
    // mask is active, run the NK on the coarse (active-cell) DOFs with the
    // geometric FVM stencil buildMatrixPolarCoarsened. mu/B stay full-resolution:
    // the working vector is prolonged to the full member Az before every field/mu
    // update (Step 1) and at convergence (for force/flux), so this avoids the
    // Phase BK "coarse mu from coarse B" 327% trap -- only the linear algebra
    // (matrix assembly + solve + Newton/line-search vectors) is coarse.
    const bool use_coarse = is_polar && coarsening_enabled
                            && n_active_cells > 0 && n_active_cells < nr * ntheta;
    const bool horiz = (r_orientation == "horizontal");

    // working solve-vector <-> member Az (full). Coarse: restrict/prolong via the
    // adaptive index maps; otherwise the original row-major (polar) / natural
    // (cartesian) layout. These two helpers centralise every Az<->vector
    // conversion in the loop (equivalent to the previous inline code for the
    // full-grid path).
    auto syncMemberAz = [&](const Eigen::VectorXd& v) {
        if (use_coarse) {
            // [Stage 1e] Active-only scatter: write the coarse values into member Az
            // at the active cells. The coarse curl (updateCoarseFieldAndMu) and
            // buildMatrixPolarCoarsened only read active cells, so the O(N) full
            // interpolation of inactive cells is NOT needed per iteration (done once
            // at convergence for force/flux). Member Az is already full-sized from
            // the initial guess.
            for (int idx = 0; idx < n_active_cells; idx++) {
                int i_r = coarse_to_fine[idx].first, j_th = coarse_to_fine[idx].second;
                if (horiz) Az(j_th, i_r) = v(idx); else Az(i_r, j_th) = v(idx);
            }
            return;
        }
        if (is_polar) {
            if (horiz) { Az.resize(ntheta, nr); for (int i=0;i<nr;i++) for(int j=0;j<ntheta;j++) Az(j,i)=v(i*ntheta+j); }
            else       { Az.resize(nr, ntheta); for (int i=0;i<nr;i++) for(int j=0;j<ntheta;j++) Az(i,j)=v(i*ntheta+j); }
        } else {
            Az.resize(ny, nx); for (int j=0;j<ny;j++) for(int i=0;i<nx;i++) Az(j,i)=v(j*nx+i);
        }
    };
    auto buildSolveVec = [&](Eigen::VectorXd& v) {
        if (use_coarse) {
            v.resize(n_active_cells);
            for (int idx=0; idx<n_active_cells; idx++) {
                int i_r = coarse_to_fine[idx].first, j_th = coarse_to_fine[idx].second;
                v(idx) = horiz ? Az(j_th, i_r) : Az(i_r, j_th);
            }
        } else if (is_polar) {
            v.resize(Az.size());
            for (int i=0;i<nr;i++) for(int j=0;j<ntheta;j++) v(i*ntheta+j) = horiz ? Az(j,i) : Az(i,j);
        } else {
            v.resize(Az.size());
            for (int j=0;j<ny;j++) for(int i=0;i<nx;i++) v(j*nx+i)=Az(j,i);
        }
    };
    auto buildMatrixForSolve = [&](Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) {
        if (use_coarse) buildMatrixPolarCoarsened(A, b);
        else if (is_polar) buildMatrixPolar(A, b);
        else buildMatrix(A, b);
    };
    // [Stage 1e] Field + mu update. Coarse: B/H/mu at active cells from the COARSE
    // curl (consistent with the coarse operator + O(n_active)); full: original path.
    auto updateFieldAndMu = [&]() {
        if (use_coarse) { updateCoarseFieldAndMu(); return; }
        if (is_polar) calculateMagneticFieldPolar(); else calculateMagneticField();
        calculateHField();
        updateMuDistribution();
    };

    // [v1.6 DD warm-start] If nonlinear_solver.initial_az_path is set, load that
    // file as the initial NK iterate and SKIP the linear init guess. This lets an
    // outer Schwarz loop warm-start each subdomain from its previous-iterate Az so
    // later outer iterations converge in a few NK steps. File = raw float64,
    // row-major Az(j,i)=buf[j*ncols+i] (the Az TIFF layout; ncols=nr horizontal).
    bool loaded_warm = false;
    if (dd_warm_start_ && is_polar && !use_coarse) {
        // v1.6 DD: the orchestrator already set member Az to the warm iterate -> use it, skip init.
        loaded_warm = true;
        if (VERBOSE) std::cout << "DD warm-start: NK from member Az (skip init guess)" << std::endl;
    }
    if (!loaded_warm && is_polar && !use_coarse && config["nonlinear_solver"] &&
        config["nonlinear_solver"]["initial_az_path"]) {
        std::string init_az_path =
            config["nonlinear_solver"]["initial_az_path"].as<std::string>("");
        if (!init_az_path.empty()) {
            const int rows = (r_orientation == "horizontal") ? ntheta : nr;
            const int cols = (r_orientation == "horizontal") ? nr : ntheta;
            std::ifstream f(init_az_path, std::ios::binary);
            std::vector<double> buf((size_t)rows * cols);
            if (f && f.read(reinterpret_cast<char*>(buf.data()),
                            (std::streamsize)buf.size() * sizeof(double))) {
                if (Az.rows() != rows || Az.cols() != cols) Az.resize(rows, cols);
                for (int j = 0; j < rows; ++j)
                    for (int i = 0; i < cols; ++i)
                        Az(j, i) = buf[(size_t)j * cols + i];
                loaded_warm = true;
                if (VERBOSE)
                    std::cout << "Warm-start: loaded initial Az from " << init_az_path
                              << " (skipping linear init guess)" << std::endl;
            } else {
                std::cerr << "Warning: failed to read initial_az_path '" << init_az_path
                          << "', falling back to linear init guess" << std::endl;
            }
        }
    }

    // Initial guess: solve linear problem with initial μ distribution. Coarse path
    // uses the FVM coarsened linear solve (sets the full interpolated member Az).
    if (VERBOSE && !loaded_warm) {
        std::cout << "Computing initial guess..."
                  << (use_coarse ? " [coarse-native NK active]" : "") << std::endl;
    }
    if (loaded_warm) {
        // member Az already holds the warm-start iterate; do not overwrite it.
    } else if (use_coarse && coarse_skip_init_guess) {
        // [DIAGNOSTIC] member Az already holds the decimated true solution; do not
        // overwrite it with the linear coarse solve.
        std::cout << "[COARSE_TRUEINIT] starting coarse NK from the decimated true solution." << std::endl;
    } else if (use_coarse) {
        buildAndSolveSystemPolarCoarsened();
    } else if (is_polar) {
        buildAndSolveSystemPolar();
    } else {
        buildAndSolveSystem();
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // ===== Step 1: Calculate B and H fields, update μ =====
        updateFieldAndMu();

        // ===== Step 2: Build residual and system matrix with current μ =====
        // Coarse-native: buildMatrixForSolve -> FVM coarse operator (n_active),
        // buildSolveVec -> restrict the (interpolated) full member Az to the
        // active cells. Full path: original buildMatrixPolar/buildMatrix + the
        // row-major Az_vec (buildSolveVec is equivalent there).
        Eigen::SparseMatrix<double> A_matrix;
        Eigen::VectorXd b_vec;
        buildMatrixForSolve(A_matrix, b_vec);

        Eigen::VectorXd Az_vec;
        buildSolveVec(Az_vec);

        // Coarse residual (used for Newton step direction in non-DC paths)
        Eigen::VectorXd residual_coarse = A_matrix * Az_vec - b_vec;
        double residual_coarse_norm = residual_coarse.norm();
        double b_vec_coarse_norm = b_vec.norm();

        // [Phase BJ-8 Path D / v1.5.1] Convergence residual is the full-grid
        // residual (the only one we have now). Phase 6's coarse-grid R + FVM
        // computeFullGridResidual branches are gone; the renamed
        // `residual_coarse_*` carry over by name only (computed at line ~144).
        double residual_norm = residual_coarse_norm;
        double b_vec_norm    = b_vec_coarse_norm;
        double residual_rel  = residual_norm / (b_vec_norm + 1e-12);

        residual_history.push_back(residual_rel);

        // DEBUG: Print norms for first iteration to diagnose polar vs cartesian scaling
        if (VERBOSE && iter == 0) {
            double A_norm = A_matrix.norm();
            std::cout << "DEBUG: ||A|| = " << A_norm
                      << ", ||Az|| = " << Az_vec.norm()
                      << ", ||b|| = " << b_vec_norm
                      << ", ||R||_abs = " << residual_norm
                      << std::endl;
        }

        if (VERBOSE) {
            std::cout << "NK iter " << std::setw(3) << iter + 1
                      << ": ||R|| = " << std::scientific << std::setprecision(4) << residual_rel
                      << std::flush;
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
                double reduction_rate = std::abs(residual_history[iter] - residual_history[iter-1]) /
                                       (residual_history[iter-1] + 1e-12);
                if (residual_rel < TOL * 10.0 && reduction_rate < 0.05) {
                    converged = true;
                    if (VERBOSE) {
                        std::cout << " [Plateau detected: Δr=" << reduction_rate << "]";
                    }
                }
            }

            // [Phase BJ-8 Path D / v1.5.1] Az-stagnation + Coarse-plateau
            // criteria were Phase 6 / Galerkin-specific (handled the
            // coarsening error floor for the now-removed coarse-grid NK).
            // Deleted.
        }

        if (converged) {
            if (VERBOSE) {
                std::cout << std::endl;
            }
            // Always print convergence message (important for user feedback)
            std::cout << "Newton-Krylov solver converged in " << iter + 1 << " iterations (residual: "
                      << std::scientific << std::setprecision(2) << residual_rel << ")" << std::endl;

            // [Stage 1e] Coarse path keeps only active cells current (active-only
            // scatter + coarse curl). Prolong to the full grid ONCE and recompute
            // full B/H/mu so downstream force/flux/export see a consistent full field.
            if (use_coarse) {
                interpolateToFullGridPolar(Az_vec);
                calculateMagneticFieldPolar();
                calculateHField();
                updateMuDistribution();
            }

            // [Phase BJ-8 Path D / v1.5.1] Final-output interpolation +
            // optional Laplacian smoothing + fine-finishing Newton-Picard
            // block deleted. With custom Galerkin coarsening removed the NK
            // loop already converges on the full grid, so the previous
            // "interpolate coarse → full, then polish on fine" step is
            // unnecessary.

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

        // [Phase BJ-8 Path D] Az_vec_at_prev_check / stagnation save was for
        // Phase 6 coarse-grid oscillation detection. Deleted.

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
        // [Phase BJ-8 Path D / v1.5.1] Phase 6 Newton-Picard (Galerkin
        // A(μ_diff) tangent) and Phase 5 matrix-free GMRES branches were
        // removed. Only the Standard direct solve with explicit Jacobian
        // (J = A_matrix + r-weighted diagonal correction) + AMGCL+EW
        // remains. AMGCL's internal smoothed_aggregation multigrid replaces
        // the custom Galerkin coarsening that the deleted branches used.
        {
            // Standard direct solve with explicit Jacobian (J = A + diagonal correction)
            Eigen::SparseMatrix<double> J_matrix = A_matrix;

            // Build r-weighted diagonal Jacobian correction for polar + nonlinear.
            //
            // Previously this loop did cv::flip + a full config["materials"]
            // iteration per cell, costing ~1 s per Newton outer iteration on a
            // 250k-DOF problem (saturated nonlinear case). Same anti-pattern as
            // the calculateCoEnergyDensity fix: hoist the flip, replace YAML
            // iteration with the rgb_to_material LUT, and parallelise.
            //
            // We also pre-resolve material_mu pointers so the inner loop only
            // touches thread-safe data (no YAML node access).
            // Skip for the coarse path: the correction indexes the FULL grid
            // (idx = i_r*ntheta+j_theta) but J_matrix is n_active x n_active. The
            // correction is numerically ~negligible anyway (Phase BL: ~3e-19% of
            // ||J v||), so J_coarse = A_coarse.
            if (!use_coarse && is_polar && !rgb_to_material.empty() && !material_mu.empty()) {
                cv::Mat image_to_use;
                cv::flip(image, image_to_use, 0);  // Match setupMaterialProperties()

                const double MU_0 = 4.0 * M_PI * 1e-7;
                YAML::Node polar_config = config["polar_domain"] ? config["polar_domain"] : config["polar"];
                double r_start_local = polar_config["r_start"].as<double>();
                double r_end_local = polar_config["r_end"].as<double>();
                double dr_local = (r_end_local - r_start_local) / (nr - 1);

                const int n_dof = static_cast<int>(Az_vec.size());

                #pragma omp parallel for schedule(static)
                for (int idx = 0; idx < n_dof; idx++) {
                    int r_idx     = idx / ntheta;
                    int theta_idx = idx % ntheta;

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
                    int rgb_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];  // Phase W: image is RGB; LUT keys are R<<16|G<<8|B

                    auto lut_it = rgb_to_material.find(rgb_key);
                    if (lut_it == rgb_to_material.end()) continue;

                    auto mu_it = material_mu.find(lut_it->second.name);
                    if (mu_it == material_mu.end()) continue;
                    if (mu_it->second.type == MuType::STATIC) continue;  // linear material

                    double H_val = H_map(img_row, img_col);
                    double mu_current = mu_map(img_row, img_col);

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
                    // J_matrix.coeffRef writes to a unique diagonal entry per
                    // idx — no race even though we are inside a parallel for.
                    J_matrix.coeffRef(idx, idx) += correction_factor;
                }
            }

            // Adaptive: SparseLU below the AMGCL threshold, AMGCL above.
            // For full-grid problems (n ~ 250k) this routes through the
            // OpenMP-parallel builtin backend rather than serial SparseLU.
            //
            // Phase BC: Eisenstat-Walker forcing. The inner AMGCL CG
            // doesn't need to converge to 1e-6 when the outer Newton
            // residual is still at 1e+1 -- iterate-quality-wise, the
            // Newton step δ has uncertainty proportional to the outer
            // residual anyway. Tighten the inner tolerance only as the
            // outer residual decreases.
            //   eta_k = γ * (||R_k|| / ||R_{k-1}||)^α
            // clipped to [eta_min, eta_max]. At iter 0 we have no
            // previous residual to ratio against, so use eta_max.
            //
            // Choice 2 (α=2) is the more aggressive form; matches the
            // Eisenstat-Walker 1996 paper's recommended setting for
            // problems where the Newton step is well-aligned with the
            // descent direction.
            double inner_tol = -1.0;  // sentinel: solveLinearSystem uses default
            if (nonlinear_config.eisenstat_walker_enabled) {
                const double g  = nonlinear_config.eisenstat_walker_gamma;
                const double a  = nonlinear_config.eisenstat_walker_alpha;
                const double lo = nonlinear_config.eisenstat_walker_eta_min;
                const double hi = nonlinear_config.eisenstat_walker_eta_max;
                if (iter == 0 || residual_history.size() < 2 ||
                    residual_history[residual_history.size() - 2] <= 0.0) {
                    inner_tol = hi;
                } else {
                    const double r_curr = residual_history.back();
                    const double r_prev = residual_history[residual_history.size() - 2];
                    const double ratio  = (r_prev > 0.0) ? (r_curr / r_prev) : 1.0;
                    double eta = g * std::pow(ratio, a);
                    if (eta < lo) eta = lo;
                    if (eta > hi) eta = hi;
                    inner_tol = eta;
                }
                if (VERBOSE) {
                    std::cout << " [EW: inner_tol=" << std::scientific
                              << std::setprecision(2) << inner_tol << "]";
                }
            }
            delta_A = solveLinearSystem(J_matrix, -residual_coarse,
                                        Eigen::VectorXd(), inner_tol);
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

        // [Phase BJ-8 Path D / v1.5.1] Phase 6 damped Picard + Anderson
        // acceleration block was the alternate update path used when the
        // (now-removed) Galerkin coarse system fed an A(μ_diff) tangent
        // Newton step that needed globalisation against the B-H knee. With
        // custom Galerkin coarsening eliminated, the only update path is the
        // classic Armijo backtracking line search below.
        for (int ls = 0; ls < max_line_search; ls++) {
            // Trial step: A_trial = A + α·δA
            Eigen::VectorXd Az_trial = Az_vec_0 + alpha * delta_A;

            // Set member Az from the trial vector (coarse: active-only scatter;
            // full: row-major write), then recompute field/mu (coarse curl at active
            // cells, or full grid) and the trial residual on the matching operator.
            syncMemberAz(Az_trial);
            updateFieldAndMu();

            double residual_trial_norm;
            {
                Eigen::SparseMatrix<double> A_trial;
                Eigen::VectorXd b_trial;
                buildMatrixForSolve(A_trial, b_trial);
                residual_trial_norm = (A_trial * Az_trial - b_trial).norm();
            }

            // Check Armijo condition: ||R(A + α·δA)|| <= ||R(A)||·(1 - c·α)
            if (residual_trial_norm <= residual_0 * (1.0 - c * alpha) || alpha < alpha_min) {
                // Accept step. Member Az already holds this trial's (full) field
                // from syncMemberAz above, so no extra write is needed.
                Az_vec = Az_trial;
                if (VERBOSE && ls > 0) {
                    std::cout << " [LS: α=" << alpha << ", " << ls+1 << " trials]";
                }
                break;
            }

            // Reject step and backtrack
            alpha *= rho;

            if (ls == max_line_search - 1) {
                // Line search failed, accept minimal step.
                Az_vec = Az_vec_0 + alpha_min * delta_A;
                syncMemberAz(Az_vec);
                if (VERBOSE) {
                    std::cout << " [LS failed, using α=" << alpha_min << "]";
                }
            }
        }

        // Update previous step length for next iteration's adaptive algorithm
        alpha_prev = alpha;

        // ===== Step 7: Anderson Acceleration =====
        if (USE_ANDERSON && m_AA > 0) {
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

                // Update member Az from the accelerated vector (coarse: prolong;
                // full: original row-major write) so the next iteration's field/mu
                // update sees it.
                syncMemberAz(Az_vec);
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

    // [Stage 1e] As in the converged branch: promote the coarse solution to the
    // full grid for downstream force/flux/export. (Az_vec is loop-scoped, so
    // reconstruct the coarse vector from the active cells of member Az.)
    if (use_coarse) {
        Eigen::VectorXd Az_c; buildSolveVec(Az_c);
        interpolateToFullGridPolar(Az_c);
        calculateMagneticFieldPolar();
        calculateHField();
        updateMuDistribution();
    }

    // [Phase BJ-8 Path D / v1.5.1] Hermite-interpolation fallback (coarse →
    // full + μ interpolation) was needed only when the coarsened NK could
    // exit at a coarse-grid solution that had to be promoted to full grid
    // before export. The NK loop now runs on the full grid throughout, so
    // Az / mu_map are already full-grid when this maxiter-exit path runs.

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
