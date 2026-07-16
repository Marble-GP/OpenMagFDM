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
#include <chrono>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <Eigen/Dense>
#include <amgcl/solver/bicgstab.hpp>   // tangent-Jacobian inner solver (nonsymmetric)
#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @brief Main Newton-Krylov solver
 */
void MagneticFieldAnalyzer::solveNonlinearNewtonKrylov() {
    last_nonlinear_iterations_ = 0;
    last_nonlinear_residual_ = std::numeric_limits<double>::infinity();
    last_nonlinear_converged_ = false;
    last_nonlinear_warm_start_safe_ = false;
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
        last_nonlinear_residual_ = 0.0;
        last_nonlinear_converged_ = true;
        last_nonlinear_warm_start_safe_ = true;
        return;
    }

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    // Warm starts need a quality gate distinct from strict convergence. Some
    // stable production cases sit at a discretisation residual floor around
    // 2e-4 with TOL=1e-5; forcing those to cold-start every step is harmful.
    // Conversely, residual-O(1) states must never propagate. This bounded gate
    // accepts the former and rejects the latter.
    const double WARM_START_RESIDUAL_LIMIT =
        std::min(1e-2, std::max(10.0 * TOL, 1e-3));
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
        if (USE_ANDERSON && m_AA > 0 && beta_AA > 0.0) {
            std::cout << "Anderson acceleration: enabled (depth=" << m_AA << ", beta=" << beta_AA << ")" << std::endl;
        }
    }

    std::vector<double> residual_history;
    double alpha_prev = nonlinear_config.line_search_alpha_init;  // Previous step length for adaptive algorithm

    // Keep the best *evaluated* nonlinear iterate.  A line-search/Anderson
    // update is applied near the end of each iteration, while its nonlinear
    // residual is not assembled until the beginning of the next one.  In
    // particular, the update made by the final allowed iteration has no
    // corresponding residual.  Exporting that unchecked state used to make
    // the reported residual, Az and mu_map describe different iterates (and a
    // bad Anderson extrapolation could therefore escape at max-iteration).
    // We only checkpoint states after field/mu and the matching residual have
    // all been evaluated, and restore this checkpoint on a non-converged exit.
    Eigen::VectorXd best_Az_vec;
    double best_residual_rel = std::numeric_limits<double>::infinity();
    int best_iteration = 0;

    // Anderson acceleration storage
    std::vector<Eigen::VectorXd> Az_history;      // Az^(k) history
    std::vector<Eigen::VectorXd> g_history;       // g^(k) = Az^(k+1) - Az^(k) history
    bool anderson_active = USE_ANDERSON && m_AA > 0 && beta_AA > 0.0;
    int anderson_consecutive_rejections = 0;
    constexpr int ANDERSON_MAX_CONSECUTIVE_REJECTIONS = 3;

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
    // Evaluate the nonlinear residual for the member Az and the current,
    // matching mu_map.  Callers must update field/mu first.  Keeping this in
    // one helper makes the max-iteration rollback and the normal convergence
    // exit report exactly the state that downstream export will consume.
    auto evaluateCurrentResidual = [&]() {
        Eigen::SparseMatrix<double> A_final;
        Eigen::VectorXd b_final;
        Eigen::VectorXd Az_final;
        buildMatrixForSolve(A_final, b_final);
        buildSolveVec(Az_final);
        return (A_final * Az_final - b_final).norm() / (b_final.norm() + 1e-12);
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

    // [NKPROF] temporary per-iteration phase timing (diagnostic; enabled with
    // env NK_PROF=1). Measures where the 1.3-1.5 s/iter actually goes.
    const bool NK_PROF = (std::getenv("NK_PROF") != nullptr);
    auto nkprof_now = []() { return std::chrono::high_resolution_clock::now(); };
    auto nkprof_ms  = [](std::chrono::high_resolution_clock::time_point a,
                         std::chrono::high_resolution_clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count() / 1000.0;
    };

    for (int iter = 0; iter < MAX_ITER; iter++) {
        auto nkp_t0 = nkprof_now();
        // ===== Step 1: Calculate B and H fields, update μ =====
        updateFieldAndMu();
        auto nkp_t_mu = nkprof_now();

        // ===== Step 2: Build residual and system matrix with current μ =====
        // Coarse-native: buildMatrixForSolve -> FVM coarse operator (n_active),
        // buildSolveVec -> restrict the (interpolated) full member Az to the
        // active cells. Full path: original buildMatrixPolar/buildMatrix + the
        // row-major Az_vec (buildSolveVec is equivalent there).
        Eigen::SparseMatrix<double> A_matrix;
        Eigen::VectorXd b_vec;
        buildMatrixForSolve(A_matrix, b_vec);
        auto nkp_t_build = nkprof_now();

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
        last_nonlinear_residual_ = residual_rel;

        if (std::isfinite(residual_rel) && residual_rel < best_residual_rel) {
            best_residual_rel = residual_rel;
            best_Az_vec = Az_vec;
            best_iteration = iter + 1;
        }

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

            // [Phase BJ-8 Path D / v1.5.1] Az-stagnation + Coarse-plateau
            // criteria were Phase 6 / Galerkin-specific (handled the
            // coarsening error floor for the now-removed coarse-grid NK).
            // Deleted.
        }

        if (converged) {
            if (VERBOSE) {
                std::cout << std::endl;
            }
            // [Stage 1e] Coarse path keeps only active cells current (active-only
            // scatter + coarse curl). Prolong to the full grid ONCE and recompute
            // full B/H/mu so downstream force/flux/export see a consistent full field.
            if (use_coarse) {
                interpolateToFullGridPolar(Az_vec);
                calculateMagneticFieldPolar();
                calculateHField();
                updateMuDistribution();
            }

            // Re-evaluate after every final-state transformation.  This is a
            // no-op-equivalent rebuild for the normal full-grid path, and is
            // essential for the coarse path where prolongation changes the Az
            // representation consumed by export.
            const double final_residual_rel = evaluateCurrentResidual();
            last_nonlinear_iterations_ = iter + 1;
            last_nonlinear_residual_ = final_residual_rel;
            last_nonlinear_converged_ = std::isfinite(final_residual_rel)
                                      && final_residual_rel < TOL;
            last_nonlinear_warm_start_safe_ = std::isfinite(final_residual_rel)
                                            && final_residual_rel
                                               <= WARM_START_RESIDUAL_LIMIT;
            if (!last_nonlinear_converged_) analysis_convergence_ok_ = false;

            // Always print convergence diagnostics when verbose.  Only the
            // requested relative-residual tolerance is accepted; the value
            // printed here is the re-evaluated residual of the exported state.
            if (!quiet_solver_ && VERBOSE) {
                if (last_nonlinear_converged_) {
                    std::cout << "Newton-Krylov solver converged in " << iter + 1
                              << " iterations (residual: " << std::scientific
                              << std::setprecision(2) << final_residual_rel << ")"
                              << std::endl;
                } else {
                    std::cerr << "WARNING: final-state residual changed after output "
                                 "reconstruction (residual="
                              << std::scientific << std::setprecision(3)
                              << final_residual_rel << ", target=" << TOL << ")."
                              << std::defaultfloat << std::endl;
                }
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
        auto nkp_t_conv = nkprof_now();
        auto nkp_t_jc = nkp_t_conv;   // re-captured after the J diagonal correction (NK_PROF granularity)

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
            // Consistent tangent (jacobian: tangent): add the per-cell
            // rank-one (nu_d - nu)(g g^T) curvature of the discrete energy
            // Hessian — the Stage G gate measured this to be what unlocks
            // alpha = 1 under the energy merit (25 true-tolerance iterations).
            // Supersedes the r-weighted diagonal correction below.
            if (nonlinear_config.jacobian_tangent && !use_coarse && is_polar) {
                addTangentCorrection(J_matrix);
            } else
            // Energy-LS mode keeps J symmetric POSITIVE DEFINITE so CG from
            // x0=0 yields a guaranteed energy-descent direction: the raw
            // diagonal correction goes NEGATIVE where dμ/dH > 0 (the rising-
            // μ_r branch that early iterates sweep through), making J
            // indefinite — measured to produce energy-ASCENT directions. In
            // energy mode the correction is therefore CLAMPED to >= 0 (keeps
            // the saturated-region stiffening, drops the indefinite part).
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
                    if (nonlinear_config.line_search_energy && correction_factor < 0.0)
                        correction_factor = 0.0;  // keep J SPD in energy-LS mode
                    // J_matrix.coeffRef writes to a unique diagonal entry per
                    // idx — no race even though we are inside a parallel for.
                    J_matrix.coeffRef(idx, idx) += correction_factor;
                }
            }
            nkp_t_jc = nkprof_now();

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
                // Residual-proportional cap: the ratio formula never tightens
                // while the outer iteration crawls at a ~constant linear rate
                // (γ·ratio^α ≈ 0.78 → clamped to eta_max forever), which left
                // the Newton direction 10% inexact even at ||R||~1e-2 and
                // stalled the plateau. Tie the cap to the outer residual so
                // inner accuracy follows outer progress. ≤0 disables.
                {
                    const double rc = nonlinear_config.eisenstat_walker_residual_cap;
                    if (rc > 0.0) {
                        double cap = rc * residual_rel;
                        if (cap < lo) cap = lo;
                        if (inner_tol > cap) inner_tol = cap;
                    }
                }
                if (VERBOSE) {
                    std::cout << " [EW: inner_tol=" << std::scientific
                              << std::setprecision(2) << inner_tol << "]";
                }
            }
            if (nonlinear_config.jacobian_tangent && !use_coarse && is_polar) {
                // Tangent path: J is NONSYMMETRIC (raw mu chain rule), so solve
                // with BiCGStab preconditioned by AMG built on the SPD secant A
                // — the assembled twin of the Stage G matrix-free gate
                // (FD-Jv GMRES preconditioned by A), which converged in 25
                // true-tolerance iterations.
                typedef amgcl::backend::builtin<double> TBackend;
                typedef amgcl::amg<TBackend, amgcl::coarsening::smoothed_aggregation,
                                   amgcl::relaxation::spai0> TPrecond;
                auto toCrs = [](const Eigen::SparseMatrix<double>& M,
                                std::vector<ptrdiff_t>& ptr, std::vector<ptrdiff_t>& col,
                                std::vector<double>& val) {
                    Eigen::SparseMatrix<double, Eigen::RowMajor> Mr = M;
                    Mr.makeCompressed();
                    const ptrdiff_t rows = Mr.rows();
                    ptr.assign(Mr.outerIndexPtr(), Mr.outerIndexPtr() + rows + 1);
                    col.assign(Mr.innerIndexPtr(), Mr.innerIndexPtr() + Mr.nonZeros());
                    val.assign(Mr.valuePtr(), Mr.valuePtr() + Mr.nonZeros());
                    return rows;
                };
                std::vector<ptrdiff_t> ap, ac, jp, jc;
                std::vector<double> av, jv;
                ptrdiff_t n_rows = toCrs(A_matrix, ap, ac, av);
                toCrs(J_matrix, jp, jc, jv);
                auto A_crs = std::tie(n_rows, ap, ac, av);
                TPrecond P(A_crs);
                TBackend::matrix J_b(std::tie(n_rows, jp, jc, jv));
                amgcl::solver::bicgstab<TBackend>::params sprm;
                sprm.tol = (inner_tol > 0.0) ? std::min(inner_tol, 1e-2) : 1e-2;
                sprm.maxiter = 100;
                amgcl::solver::bicgstab<TBackend> S(n_rows, sprm);
                std::vector<double> rhs_v(residual_coarse.size());
                for (int q = 0; q < residual_coarse.size(); ++q) rhs_v[q] = -residual_coarse[q];
                std::vector<double> x_v(residual_coarse.size(), 0.0);
                auto [t_it, t_err] = S(J_b, P, rhs_v, x_v);
                if (VERBOSE) std::cout << " [tangent BiCGStab " << t_it << " its, res="
                                       << std::scientific << std::setprecision(1) << t_err << "]";
                delta_A = Eigen::Map<Eigen::VectorXd>(x_v.data(), (Eigen::Index)x_v.size());
            } else {
                delta_A = solveLinearSystem(J_matrix, -residual_coarse,
                                            Eigen::VectorXd(), inner_tol);
            }
        }
        auto nkp_t_solve = nkprof_now();

        // ===== [Stage G gate] OMFDM_PROBE_WNEWTON: true-Newton x energy merit =====
        // Decides GO/NO-GO for building the consistent tangent (CTSM) WITHOUT
        // building it: the true Newton direction δ_N = J_true^{-1}(-R) is
        // computed MATRIX-FREE (central-FD Jv oracle + right-preconditioned
        // GMRES, precond = the assembled secant A via AMGCL), then evaluated
        // under the energy merit W. Phase BL measured the true direction under
        // the ||R|| merit only (Finding 3: full step diverges warm); the CONVEX
        // energy view says W-Armijo is the correct acceptance — this is the one
        // untested combination.
        //   OMFDM_PROBE_WNEWTON=1: at iters in OMFDM_PROBE_WNEWTON_ITERS
        //     (default "5,10,20") print an alpha/W/||R|| scan along δ_N.
        //   OMFDM_PROBE_WNEWTON=2: DRIVE mode — replace the step direction with
        //     δ_N every iteration (matrix-free CTSM); the configured line search
        //     (use line_search_objective: energy) handles acceptance. The
        //     resulting iteration count IS the gate: <=~20 iters => building the
        //     assembled CTSM is worth it.
        {
            static const char* wn_env = std::getenv("OMFDM_PROBE_WNEWTON");
            const int wn_mode = wn_env ? std::atoi(wn_env) : 0;
            bool wn_probe_this_iter = false;
            if (wn_mode == 1) {
                static std::set<int> wn_iters = [] {
                    std::set<int> s;
                    const char* e = std::getenv("OMFDM_PROBE_WNEWTON_ITERS");
                    std::string str = e ? e : "5,10,20";
                    size_t i = 0;
                    while (i < str.size()) {
                        s.insert(std::atoi(str.c_str() + i));
                        size_t n = str.find(',', i);
                        if (n == std::string::npos) break;
                        i = n + 1;
                    }
                    return s;
                }();
                wn_probe_this_iter = wn_iters.count(iter + 1) > 0;
            }
            if (!use_coarse && (wn_mode == 2 || wn_probe_this_iter)) {
                // ~21 preconditioner solves per NK iteration — silence the
                // per-call AMGCL banners for the duration of the probe.
                const bool wn_prev_quiet = quiet_solver_;
                quiet_solver_ = true;
                // Residual oracle at arbitrary x (mutates member fields; the
                // caller below restores them to the current iterate).
                auto evalResidualVec = [&](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                    syncMemberAz(x);
                    updateFieldAndMu();
                    Eigen::SparseMatrix<double> A_t;
                    Eigen::VectorXd b_t;
                    buildMatrixForSolve(A_t, b_t);
                    return A_t * x - b_t;
                };
                const double eps_rel = [] {
                    const char* e = std::getenv("OMFDM_PROBE_EPS");
                    return e ? std::atof(e) : 1e-6;
                }();
                const double xnorm = Az_vec.norm();
                auto applyJ = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
                    const double vn = v.norm();
                    if (vn < 1e-30) return Eigen::VectorXd::Zero(v.size());
                    const double h = eps_rel * std::max(xnorm, 1.0) / vn;
                    Eigen::VectorXd Rp = evalResidualVec(Az_vec + h * v);
                    Eigen::VectorXd Rm = evalResidualVec(Az_vec - h * v);
                    return (Rp - Rm) / (2.0 * h);
                };
                // Right-preconditioned GMRES(m): solve J M^{-1} y = -R, δ_N = M^{-1} y,
                // M = assembled secant A (AMGCL, loose tol).
                const int m_kry = 20;
                auto applyM = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
                    return solveLinearSystem(A_matrix, v, Eigen::VectorXd(), 1e-2);
                };
                const Eigen::VectorXd rhs = -residual_coarse;
                const double bnorm = rhs.norm() + 1e-30;
                std::vector<Eigen::VectorXd> V;
                V.reserve(m_kry + 1);
                Eigen::MatrixXd Hh = Eigen::MatrixXd::Zero(m_kry + 1, m_kry);
                V.push_back(rhs / bnorm);
                int kdim = 0;
                for (int k = 0; k < m_kry; ++k) {
                    Eigen::VectorXd w = applyJ(applyM(V[k]));
                    for (int i2 = 0; i2 <= k; ++i2) {
                        Hh(i2, k) = w.dot(V[i2]);
                        w -= Hh(i2, k) * V[i2];
                    }
                    Hh(k + 1, k) = w.norm();
                    kdim = k + 1;
                    if (Hh(k + 1, k) < 1e-12 * bnorm) break;
                    V.push_back(w / Hh(k + 1, k));
                }
                Eigen::VectorXd e1 = Eigen::VectorXd::Zero(kdim + 1);
                e1(0) = bnorm;
                const Eigen::VectorXd y =
                    Hh.topLeftCorner(kdim + 1, kdim).householderQr().solve(e1);
                Eigen::VectorXd yv = Eigen::VectorXd::Zero(V[0].size());
                for (int k = 0; k < kdim; ++k) yv += y(k) * V[k];
                Eigen::VectorXd delta_N = applyM(yv);
                const double gmres_rel =
                    (Hh.topLeftCorner(kdim + 1, kdim) * y - e1).norm() / bnorm;

                if (wn_probe_this_iter) {
                    // alpha scan along δ_N under BOTH merits.
                    const double W0p = [&] {
                        syncMemberAz(Az_vec); updateFieldAndMu();
                        return computeEnergyObjective();
                    }();
                    std::cout << "\n[WNEWTON probe iter " << iter + 1
                              << "] ||R||=" << residual_norm
                              << " gmres_rel=" << gmres_rel
                              << " ||dN||/||dP||=" << delta_N.norm() / (delta_A.norm() + 1e-30)
                              << " cos(dN,dP)=" << delta_N.dot(delta_A) /
                                     (delta_N.norm() * delta_A.norm() + 1e-30) << std::endl;
                    for (double a : {1.0, 0.65, 0.42, 0.27, 0.18, 0.12, 0.08}) {
                        Eigen::VectorXd xt = Az_vec + a * delta_N;
                        const double Rn = evalResidualVec(xt).norm();
                        const double Wt = computeEnergyObjective();  // fields already at xt
                        std::cout << "  alpha=" << a
                                  << "  dW=" << std::scientific << Wt - W0p
                                  << "  ||R||/||R0||=" << Rn / (residual_norm + 1e-30)
                                  << std::endl;
                    }
                }
                if (wn_mode == 2) {
                    delta_A = delta_N;   // drive the solver along the true direction
                    if (VERBOSE) std::cout << " [WN drive: gmres_rel=" << std::scientific
                                           << std::setprecision(1) << gmres_rel << "]";
                }
                // Restore member fields to the CURRENT iterate for the line search.
                syncMemberAz(Az_vec);
                updateFieldAndMu();
                quiet_solver_ = wn_prev_quiet;
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

        // Energy-objective line search: Armijo on the CONVEX functional W(Az)
        // (∇W = R) instead of on ||R||. ||R|| is not monotone along descent
        // paths of a non-quadratic convex function, so the residual Armijo
        // rejects legitimate long steps and pins α at ~0.1 (the linear-rate
        // crawl). Requires the member fields to currently match Az_vec_0
        // (guaranteed: Step 1 ran updateFieldAndMu for this iterate).
        const bool LS_ENERGY = nonlinear_config.line_search_energy && !use_coarse;
        double W0 = 0.0, g0 = 0.0;
        bool energy_ok = false;
        if (LS_ENERGY) {
            W0 = computeEnergyObjective();
            // Directional derivative dW/dα at α=0 by finite difference. The
            // assembled residual R is NOT usable here: the FV rows carry
            // O(1e9) scale factors, so R·δ has neither the scale nor
            // (numerically) the sign of dW/dα.
            const double eps = 1e-6 * std::sqrt((Az_vec_0.squaredNorm() + 1e-30) /
                                                (delta_A.squaredNorm() + 1e-30));
            Eigen::VectorXd Az_eps = Az_vec_0 + eps * delta_A;
            syncMemberAz(Az_eps);
            updateFieldAndMu();
            const double W_eps = computeEnergyObjective();
            g0 = (W_eps - W0) / eps;
            energy_ok = std::isfinite(W0) && std::isfinite(g0) && g0 < 0.0;
            if (energy_ok) alpha = 1.0;  // always probe the full step first
            else if (VERBOSE) std::cout << " [energy-LS: g0=" << g0
                                        << " not a descent direction -> residual LS]";
        }

        // [Phase BJ-8 Path D / v1.5.1] Phase 6 damped Picard + Anderson
        // acceleration block was the alternate update path used when the
        // (now-removed) Galerkin coarse system fed an A(μ_diff) tangent
        // Newton step that needed globalisation against the B-H knee. With
        // custom Galerkin coarsening eliminated, the only update path is the
        // classic Armijo backtracking line search below.
        int nkp_ls_trials = 0;
        for (int ls = 0; ls < max_line_search; ls++) {
            nkp_ls_trials++;
            // Trial step: A_trial = A + α·δA
            Eigen::VectorXd Az_trial = Az_vec_0 + alpha * delta_A;

            // Set member Az from the trial vector (coarse: active-only scatter;
            // full: row-major write), then recompute field/mu (coarse curl at active
            // cells, or full grid) and the trial residual on the matching operator.
            syncMemberAz(Az_trial);
            updateFieldAndMu();

            bool accept;
            if (energy_ok) {
                // Armijo on the convex energy: W(x+αδ) ≤ W(x) + c·α·(∇W·δ).
                // Skips the per-trial matrix rebuild entirely (the residual is
                // recomputed at the top of the next NK iteration anyway).
                const double W_trial = computeEnergyObjective();
                accept = (std::isfinite(W_trial) &&
                          W_trial <= W0 + c * alpha * g0) || alpha < alpha_min;
                if (accept && VERBOSE && ls == 0) std::cout << " [W-LS α=" << alpha << "]";
            } else {
                double residual_trial_norm;
                {
                    Eigen::SparseMatrix<double> A_trial;
                    Eigen::VectorXd b_trial;
                    buildMatrixForSolve(A_trial, b_trial);
                    residual_trial_norm = (A_trial * Az_trial - b_trial).norm();
                }
                // Check Armijo condition: ||R(A + α·δA)|| <= ||R(A)||·(1 - c·α)
                accept = residual_trial_norm <= residual_0 * (1.0 - c * alpha) ||
                         alpha < alpha_min;
            }

            if (accept) {
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

        if (NK_PROF) {
            auto nkp_t_ls = nkprof_now();
            std::cout << "\n[NKPROF] iter " << iter + 1
                      << ": mu=" << nkprof_ms(nkp_t0, nkp_t_mu) << "ms"
                      << " build=" << nkprof_ms(nkp_t_mu, nkp_t_build) << "ms"
                      << " resid+conv=" << nkprof_ms(nkp_t_build, nkp_t_conv) << "ms"
                      << " jcorr=" << nkprof_ms(nkp_t_conv, nkp_t_jc) << "ms"
                      << " amgcl=" << nkprof_ms(nkp_t_jc, nkp_t_solve) << "ms"
                      << " ls=" << nkprof_ms(nkp_t_solve, nkp_t_ls) << "ms"
                      << " (trials=" << nkp_ls_trials << ")"
                      << " total=" << nkprof_ms(nkp_t0, nkp_t_ls) << "ms" << std::endl;
        }

        // ===== Step 7: Safeguarded Anderson Acceleration =====
        if (anderson_active) {
            const Eigen::VectorXd Az_base = Az_vec;    // Armijo-accepted update
            const Eigen::VectorXd g_k = Az_base - Az_vec_0;
            bool anderson_attempted = false;
            bool anderson_accepted = false;
            bool anderson_restart_history = false;

            if (iter >= 1 && g_history.size() > 0) {
                anderson_attempted = true;
                int m_k = std::min(m_AA, static_cast<int>(g_history.size()));

                // Build matrix of residual differences: ΔG = [g_{k-m_k} - g_k, ..., g_{k-1} - g_k]
                Eigen::MatrixXd DG(g_k.size(), m_k);
                for (int j = 0; j < m_k; j++) {
                    int idx = g_history.size() - m_k + j;
                    DG.col(j) = g_history[idx] - g_k;
                }

                // Solve least-squares: min ||DG * θ + g_k||²
                // A column-pivoted QR solve avoids squaring the condition number
                // as the former normal-equations/LDLT implementation did.
                Eigen::VectorXd theta = Eigen::VectorXd::Zero(m_k);
                const bool decomposition_input_finite = DG.allFinite() && g_k.allFinite();
                if (decomposition_input_finite) {
                    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(DG);
                    qr.setThreshold(1e-10);
                    theta = qr.solve(-g_k);
                }

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

                // Form the full Anderson correction; safeguarded beta mixing is below.
                Eigen::VectorXd correction = Az_anderson - Az_base;
                bool proposal_finite = decomposition_input_finite && theta.allFinite() &&
                                       correction.allFinite();

                // Bound the extrapolation by the norm of the already-globalized
                // Newton update. Near a fixed point, retain a tiny Az-relative
                // floor so a roundoff-sized update cannot create an unbounded ratio.
                const double correction_norm = correction.norm();
                const double update_norm = g_k.norm();
                const double current_norm = Az_vec_0.norm();
                proposal_finite = proposal_finite &&
                    std::isfinite(correction_norm) && std::isfinite(update_norm) &&
                    std::isfinite(current_norm);
                if (proposal_finite) {
                    const double correction_limit = std::max(
                        update_norm, 1e-12 * std::max(current_norm, 1.0));
                    if (correction_norm > correction_limit) {
                        correction *= correction_limit / correction_norm;
                    }
                }

                // Evaluate each state with its own B/H/mu and rebuilt nonlinear
                // operator. A frozen-Jacobian residual is not a valid safeguard.
                auto evaluateAndersonState = [&](const Eigen::VectorXd& state,
                                                 double& residual_rel_out,
                                                 double& energy_out) -> bool {
                    if (!state.allFinite()) return false;
                    syncMemberAz(state);
                    updateFieldAndMu();
                    Eigen::SparseMatrix<double> A_eval;
                    Eigen::VectorXd b_eval;
                    buildMatrixForSolve(A_eval, b_eval);
                    const Eigen::VectorXd R_eval = A_eval * state - b_eval;
                    residual_rel_out = R_eval.norm() / (b_eval.norm() + 1e-12);
                    energy_out = use_coarse ? 0.0 : computeEnergyObjective();
                    return R_eval.allFinite() && std::isfinite(residual_rel_out) &&
                           (use_coarse || std::isfinite(energy_out));
                };

                double base_residual_rel = 0.0;
                double base_energy = 0.0;
                const bool base_finite = evaluateAndersonState(
                    Az_base, base_residual_rel, base_energy);
                proposal_finite = proposal_finite && base_finite;
                if (base_finite && base_residual_rel < best_residual_rel) {
                    // This post-line-search state has a rebuilt nonlinear
                    // residual, so it is a valid max-iteration checkpoint.
                    best_Az_vec = Az_base;
                    best_residual_rel = base_residual_rel;
                    best_iteration = iter + 1;
                }

                // Backtrack the Anderson mixing only; the Armijo base remains
                // untouched. Accept a candidate only if the true residual strictly
                // improves and the convex energy does not increase.
                double beta_trial = std::min(1.0, std::max(0.0, beta_AA));
                constexpr int ANDERSON_MAX_BETA_TRIALS = 3;
                constexpr double ANDERSON_BETA_RHO = 0.5;
                for (int trial = 0;
                     proposal_finite && trial < ANDERSON_MAX_BETA_TRIALS;
                     ++trial, beta_trial *= ANDERSON_BETA_RHO) {
                    const Eigen::VectorXd Az_candidate =
                        Az_base + beta_trial * correction;
                    double candidate_residual_rel = 0.0;
                    double candidate_energy = 0.0;
                    const bool candidate_finite = evaluateAndersonState(
                        Az_candidate, candidate_residual_rel, candidate_energy);
                    const double required_ratio =
                        1.0 - 1e-4 * std::max(beta_trial, 1e-3);
                    const double energy_slack =
                        1e-10 * std::max(std::abs(base_energy), 1.0);
                    const bool residual_improved = candidate_finite &&
                        candidate_residual_rel <= base_residual_rel * required_ratio;
                    const bool energy_not_worse = use_coarse ||
                        candidate_energy <= base_energy + energy_slack;
                    if (residual_improved && energy_not_worse) {
                        Az_vec = Az_candidate;
                        anderson_accepted = true;
                        anderson_consecutive_rejections = 0;
                        if (candidate_residual_rel < best_residual_rel) {
                            // Unlike the historical unchecked extrapolation,
                            // this candidate has matching mu and a true residual.
                            best_Az_vec = Az_candidate;
                            best_residual_rel = candidate_residual_rel;
                            best_iteration = iter + 1;
                        }
                        if (VERBOSE) {
                            std::cout << " [AA accepted: beta=" << beta_trial << "]";
                        }
                        break;
                    }
                }

                if (!anderson_accepted) {
                    // Roll back both Az and derived material fields to the trusted
                    // Armijo result, then restart the secant history from here.
                    Az_vec = Az_base;
                    syncMemberAz(Az_base);
                    updateFieldAndMu();
                    anderson_restart_history = true;
                    ++anderson_consecutive_rejections;
                    if (VERBOSE) {
                        std::cout << " [AA rejected "
                                  << anderson_consecutive_rejections << "/"
                                  << ANDERSON_MAX_CONSECUTIVE_REJECTIONS << "]";
                    }
                }

                // On acceptance, the last candidate evaluation left member Az and
                // B/H/mu at Az_vec. On rejection, the rollback above did the same
                // for Az_base. Do not perform a vector-only sync here: that would
                // make it easier for future changes to leave derived fields stale.
            }

            if (anderson_restart_history) {
                Az_history.clear();
                g_history.clear();
            }

            // Store only the underlying Armijo-globalized map pair. After a
            // rejection, this safe pair seeds a fresh Anderson history.
            Az_history.push_back(Az_vec_0);
            g_history.push_back(g_k);

            // Limit history size
            if (static_cast<int>(Az_history.size()) > m_AA + 1) {
                Az_history.erase(Az_history.begin());
                g_history.erase(g_history.begin());
            }

            if (anderson_attempted && !anderson_accepted &&
                anderson_consecutive_rejections >=
                    ANDERSON_MAX_CONSECUTIVE_REJECTIONS) {
                anderson_active = false;
                Az_history.clear();
                g_history.clear();
                if (VERBOSE) {
                    std::cout << " [AA disabled for this solve]";
                }
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

    // The final update has not been residual-evaluated.  Restore the best
    // checkpoint instead of exporting that unchecked state, then rebuild every
    // derived quantity.  This also makes a rejected/unstable accelerator unable
    // to contaminate force, energy, flux linkage or the next transient seed.
    if (best_Az_vec.size() > 0) {
        syncMemberAz(best_Az_vec);
    }
    updateFieldAndMu();

    // [Stage 1e] Promote the restored coarse state to the full grid for
    // downstream force/flux/export, and derive full-grid B/H/mu from that exact
    // exported Az.
    if (use_coarse) {
        Eigen::VectorXd Az_c;
        if (best_Az_vec.size() > 0) Az_c = best_Az_vec;
        else buildSolveVec(Az_c);
        interpolateToFullGridPolar(Az_c);
        calculateMagneticFieldPolar();
        calculateHField();
        updateMuDistribution();
    }

    last_nonlinear_iterations_ = MAX_ITER;
    last_nonlinear_residual_ = evaluateCurrentResidual();
    last_nonlinear_converged_ = std::isfinite(last_nonlinear_residual_)
                              && last_nonlinear_residual_ < TOL;
    last_nonlinear_warm_start_safe_ = std::isfinite(last_nonlinear_residual_)
                                    && last_nonlinear_residual_
                                       <= WARM_START_RESIDUAL_LIMIT;
    if (!last_nonlinear_converged_) analysis_convergence_ok_ = false;
    if (!quiet_solver_ && !last_nonlinear_converged_) {
        std::cerr << "WARNING: Newton-Krylov did not converge after " << MAX_ITER
                  << " iterations; restored best evaluated iterate "
                  << best_iteration << " (residual=" << std::scientific
                  << std::setprecision(3) << last_nonlinear_residual_
                  << ", target=" << TOL << ")."
                  << std::defaultfloat << std::endl;
    } else if (!quiet_solver_ && VERBOSE) {
        std::cout << "Newton-Krylov solver converged on the final evaluated "
                     "update (residual="
                  << std::scientific << std::setprecision(3)
                  << last_nonlinear_residual_ << ")."
                  << std::defaultfloat << std::endl;
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
