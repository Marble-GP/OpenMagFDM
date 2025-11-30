/**
 * @file MagneticFieldAnalyzer_nonlinear.cpp
 * @brief Nonlinear material support for MagneticFieldAnalyzer
 *
 * This file contains all nonlinear permeability-related functions:
 * - μ_r(H) parsing and evaluation
 * - B-H table generation and interpolation
 * - Nonlinear Picard iteration solver
 * - Anderson acceleration
 */

#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// ============================================
// Nonlinear Material Support Functions
// ============================================

/**
 * @brief Parse mu_r value from YAML (static, formula, or table)
 */
MagneticFieldAnalyzer::MuValue MagneticFieldAnalyzer::parseMuValue(const YAML::Node& mu_node) {
    MuValue result;

    if (!mu_node) {
        result.type = MuType::STATIC;
        result.static_value = 1.0;
        return result;
    }

    // Check if it's a scalar (number or formula string)
    if (mu_node.IsScalar()) {
        std::string mu_str = mu_node.as<std::string>();

        // Check if it contains formula characters (including $H)
        if (mu_str.find('$') != std::string::npos ||
            mu_str.find('*') != std::string::npos ||
            mu_str.find('/') != std::string::npos ||
            mu_str.find('+') != std::string::npos ||
            mu_str.find('(') != std::string::npos ||
            mu_str.find("exp") != std::string::npos ||
            mu_str.find("tanh") != std::string::npos) {
            // It's a formula
            result.type = MuType::FORMULA;
            result.formula = mu_str;
            // Replace $H with H for tinyexpr
            size_t pos = 0;
            while ((pos = result.formula.find("$H", pos)) != std::string::npos) {
                result.formula.replace(pos, 2, "H");
                pos += 1;
            }
        } else {
            // It's a plain number
            result.type = MuType::STATIC;
            result.static_value = mu_node.as<double>();
        }
    }
    // Check if it's a 2D array [[H_values], [mu_r_values]]
    else if (mu_node.IsSequence() && mu_node.size() == 2) {
        result.type = MuType::TABLE;

        // First row: H values
        if (mu_node[0].IsSequence()) {
            for (const auto& val : mu_node[0]) {
                result.H_table.push_back(val.as<double>());
            }
        } else {
            throw std::runtime_error("mu_r table: first row must be H values array");
        }

        // Second row: mu_r values
        if (mu_node[1].IsSequence()) {
            for (const auto& val : mu_node[1]) {
                result.mu_table.push_back(val.as<double>());
            }
        } else {
            throw std::runtime_error("mu_r table: second row must be mu_r values array");
        }

        // Validate array sizes
        if (result.H_table.size() != result.mu_table.size()) {
            throw std::runtime_error("mu_r table: H and mu_r arrays must have same size");
        }

        if (result.H_table.size() < 2) {
            throw std::runtime_error("mu_r table: must have at least 2 points");
        }
    }
    else {
        throw std::runtime_error("Invalid mu_r value format");
    }

    return result;
}

/**
 * @brief Evaluate mu_r at given |H| magnitude
 */
double MagneticFieldAnalyzer::evaluateMu(const MuValue& mu_val, double H_magnitude) {
    switch (mu_val.type) {
        case MuType::STATIC:
            return mu_val.static_value;

        case MuType::FORMULA: {
            te_parser parser;
            te_variable H_var;
            H_var.m_name = "H";
            H_var.m_value = H_magnitude;

            std::set<te_variable> vars = {H_var};
            parser.set_variables_and_functions(vars);

            double mu_r = parser.evaluate(mu_val.formula);
            if (!parser.success()) {
                throw std::runtime_error("Failed to evaluate mu_r formula: " + mu_val.formula);
            }

            // Ensure mu_r >= 1 (physical constraint)
            if (mu_r < 1.0) {
                std::cerr << "WARNING: mu_r < 1 at H=" << H_magnitude
                          << ", clamping to 1.0" << std::endl;
                mu_r = 1.0;
            }

            return mu_r;
        }

        case MuType::TABLE: {
            // Linear interpolation in mu_r(H) table
            const auto& H_tab = mu_val.H_table;
            const auto& mu_tab = mu_val.mu_table;

            // Handle out-of-range (extrapolate with constant)
            if (H_magnitude <= H_tab.front()) {
                return mu_tab.front();
            }
            if (H_magnitude >= H_tab.back()) {
                return mu_tab.back();
            }

            // Find interpolation interval
            auto it = std::upper_bound(H_tab.begin(), H_tab.end(), H_magnitude);
            size_t idx = std::distance(H_tab.begin(), it) - 1;

            // Linear interpolation
            double H0 = H_tab[idx];
            double H1 = H_tab[idx + 1];
            double mu0 = mu_tab[idx];
            double mu1 = mu_tab[idx + 1];

            double alpha = (H_magnitude - H0) / (H1 - H0);
            double mu_r = mu0 + alpha * (mu1 - mu0);

            return mu_r;
        }

        default:
            return 1.0;
    }
}

/**
 * @brief Evaluate derivative dμ_r/dH at given |H| magnitude
 *
 * This is needed for Newton-Krylov Jacobian calculation.
 * Returns dμ_r/dH (not dμ/dH - caller must multiply by μ_0 if needed)
 */
double MagneticFieldAnalyzer::evaluateMuDerivative(const MuValue& mu_val, double H_magnitude) {
    const double epsilon = 1e-6;  // Finite difference step for numerical derivative

    switch (mu_val.type) {
        case MuType::STATIC:
            // Constant mu_r -> derivative is zero
            return 0.0;

        case MuType::FORMULA: {
            // Numerical differentiation using central difference
            // dμ/dH ≈ [μ(H+ε) - μ(H-ε)] / (2ε)

            double H_plus = H_magnitude + epsilon;
            double H_minus = std::max(0.0, H_magnitude - epsilon);

            double mu_plus = evaluateMu(mu_val, H_plus);
            double mu_minus = evaluateMu(mu_val, H_minus);

            double derivative = (mu_plus - mu_minus) / (H_plus - H_minus);

            return derivative;
        }

        case MuType::TABLE: {
            // Analytic derivative from linear interpolation
            const auto& H_tab = mu_val.H_table;
            const auto& mu_tab = mu_val.mu_table;

            // Handle out-of-range (derivative = 0 at boundaries)
            if (H_magnitude <= H_tab.front() || H_magnitude >= H_tab.back()) {
                return 0.0;
            }

            // Find interpolation interval
            auto it = std::upper_bound(H_tab.begin(), H_tab.end(), H_magnitude);
            size_t idx = std::distance(H_tab.begin(), it) - 1;

            // Linear segment slope: dμ/dH = (mu1 - mu0) / (H1 - H0)
            double H0 = H_tab[idx];
            double H1 = H_tab[idx + 1];
            double mu0 = mu_tab[idx];
            double mu1 = mu_tab[idx + 1];

            double derivative = (mu1 - mu0) / (H1 - H0);

            return derivative;
        }

        default:
            return 0.0;
    }
}

/**
 * @brief Validate mu_r table data
 */
void MagneticFieldAnalyzer::validateMuTable(const std::vector<double>& H_vals,
                                             const std::vector<double>& mu_vals,
                                             const std::string& material_name) {
    // Check H values are strictly monotonically increasing (REQUIRED)
    for (size_t i = 1; i < H_vals.size(); i++) {
        if (H_vals[i] <= H_vals[i-1]) {
            throw std::runtime_error(
                "Material '" + material_name + "': H values in mu_r table must be strictly increasing");
        }
    }

    // Check mu_r values are non-negative
    for (size_t i = 0; i < mu_vals.size(); i++) {
        if (mu_vals[i] < 1.0) {
            std::cerr << "WARNING: Material '" << material_name
                      << "': mu_r[" << i << "] = " << mu_vals[i]
                      << " < 1.0 (unphysical)" << std::endl;
        }
    }

    // Check mu_r values are monotonically decreasing (RECOMMENDED, warning only)
    bool is_monotonic_decreasing = true;
    for (size_t i = 1; i < mu_vals.size(); i++) {
        if (mu_vals[i] > mu_vals[i-1]) {
            is_monotonic_decreasing = false;
            break;
        }
    }

    if (!is_monotonic_decreasing) {
        std::cerr << "WARNING: Material '" << material_name
                  << "': mu_r is not monotonically decreasing. "
                  << "This may cause convergence issues in nonlinear solver." << std::endl;
    }
}

/**
 * @brief Generate B(H) table from mu_r(H) via numerical integration
 *
 * B(H) = ∫[0 to H] μ(H') dH' = μ_0 ∫[0 to H] μ_r(H') dH'
 *
 * Uses trapezoidal rule for integration.
 */
void MagneticFieldAnalyzer::generateBHTable(const std::string& material_name, const MuValue& mu_val) {
    const double MU_0 = 4.0 * M_PI * 1e-7;  // H/m

    BHTable& table = material_bh_tables[material_name];
    table.H_values.clear();
    table.B_values.clear();
    table.mu_values.clear();

    // Determine H range for sampling
    double H_min = 0.0;
    double H_max = 1e6;  // 1 MA/m (very high field)
    int num_points = 1000;

    if (mu_val.type == MuType::TABLE) {
        // Use table's H range + some extension
        H_max = std::max(mu_val.H_table.back() * 1.2, 1e5);
        num_points = std::max(500, static_cast<int>(mu_val.H_table.size() * 10));
    }

    // Generate H samples (logarithmic spacing for better resolution at low H)
    std::vector<double> H_samples;
    H_samples.push_back(0.0);

    // Logarithmic spacing from H_min_log to H_max
    double H_min_log = 1e-3;  // Start from 1 mA/m for log spacing
    for (int i = 0; i < num_points; i++) {
        double log_H = std::log10(H_min_log) +
                       i * (std::log10(H_max) - std::log10(H_min_log)) / (num_points - 1);
        H_samples.push_back(std::pow(10.0, log_H));
    }

    // Numerical integration using trapezoidal rule: B(H_i) = B(H_{i-1}) + μ(H_i) * ΔH
    double B_current = 0.0;
    for (size_t i = 0; i < H_samples.size(); i++) {
        double H = H_samples[i];
        double mu_r = evaluateMu(mu_val, H);
        double mu = mu_r * MU_0;

        if (i > 0) {
            double H_prev = H_samples[i-1];
            double mu_r_prev = evaluateMu(mu_val, H_prev);
            double mu_prev = mu_r_prev * MU_0;

            // Trapezoidal integration
            double dH = H - H_prev;
            B_current += 0.5 * (mu + mu_prev) * dH;
        }

        table.H_values.push_back(H);
        table.B_values.push_back(B_current);
        table.mu_values.push_back(mu);
    }

    table.is_valid = true;

    std::cout << "Generated B-H table for '" << material_name << "': "
              << table.H_values.size() << " points, "
              << "H = [" << table.H_values.front() << ", " << table.H_values.back() << "] A/m, "
              << "B = [" << table.B_values.front() << ", " << table.B_values.back() << "] T"
              << std::endl;
}

/**
 * @brief Interpolate |H| from |B| using inverse B-H table
 */
double MagneticFieldAnalyzer::interpolateH_from_B(const BHTable& table, double B_magnitude) {
    if (!table.is_valid || table.B_values.empty()) {
        std::cerr << "ERROR: B-H table is not valid" << std::endl;
        return 0.0;
    }

    const auto& B_tab = table.B_values;
    const auto& H_tab = table.H_values;

    // Handle out-of-range (extrapolate linearly)
    if (B_magnitude <= B_tab.front()) {
        // Linear extrapolation: H = B / μ(0)
        if (table.mu_values.front() > 1e-20) {
            return B_magnitude / table.mu_values.front();
        }
        return 0.0;
    }

    if (B_magnitude >= B_tab.back()) {
        // Linear extrapolation using last segment slope
        size_t n = B_tab.size();
        double dH = H_tab[n-1] - H_tab[n-2];
        double dB = B_tab[n-1] - B_tab[n-2];
        if (std::abs(dB) > 1e-20) {
            double slope = dH / dB;  // dH/dB
            return H_tab[n-1] + slope * (B_magnitude - B_tab[n-1]);
        }
        return H_tab[n-1];
    }

    // Find interpolation interval (binary search)
    auto it = std::upper_bound(B_tab.begin(), B_tab.end(), B_magnitude);
    size_t idx = std::distance(B_tab.begin(), it) - 1;

    // Linear interpolation
    double B0 = B_tab[idx];
    double B1 = B_tab[idx + 1];
    double H0 = H_tab[idx];
    double H1 = H_tab[idx + 1];

    double alpha = (B_magnitude - B0) / (B1 - B0);
    double H = H0 + alpha * (H1 - H0);

    return H;
}

/**
 * @brief Calculate |H| field from Bx, By (or Br, Btheta in polar)
 */
void MagneticFieldAnalyzer::calculateHField() {
    const double MU_0 = 4.0 * M_PI * 1e-7;

    // Ensure H_map has correct size (must match mu_map shape)
    if (coordinate_system == "cartesian") {
        H_map.resize(ny, nx);
    } else {  // polar
        // CRITICAL: H_map shape must match mu_map shape, which depends on r_orientation
        if (r_orientation == "horizontal") {
            H_map.resize(ntheta, nr);  // (theta, r) - theta is rows, r is cols
        } else {  // vertical
            H_map.resize(nr, ntheta);  // (r, theta) - r is rows, theta is cols
        }
    }

    // IMPORTANT: Match the image orientation used in setupMaterialProperties()
    // setupMaterialProperties() always flips the image vertically for BOTH cartesian and polar
    cv::Mat image_to_use;
    cv::flip(image, image_to_use, 0);  // Flip vertically: y down -> y up

    // Calculate H from B using proper inverse B-H relationship
    for (int j = 0; j < H_map.rows(); j++) {
        for (int i = 0; i < H_map.cols(); i++) {
            double Bx_val, By_val;

            if (coordinate_system == "cartesian") {
                Bx_val = Bx(j, i);
                By_val = By(j, i);
            } else {  // polar
                Bx_val = Br(j, i);     // Use Br, Btheta as proxy
                By_val = Btheta(j, i);
            }

            double B_mag = std::sqrt(Bx_val * Bx_val + By_val * By_val);

            // For nonlinear materials, use inverse B-H interpolation
            // For linear materials, use H = B/μ
            // Use image_to_use (flipped) to match setupMaterialProperties()
            cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(j, i);
            cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);  // BGR to RGB

            // Find material for this pixel
            std::string material_name = "";
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
                    material_name = name;
                    break;
                }
            }

            // Check if this material has a B-H table (nonlinear)
            auto bh_it = material_bh_tables.find(material_name);
            if (bh_it != material_bh_tables.end() && bh_it->second.is_valid) {
                // Nonlinear material: use inverse B-H interpolation
                H_map(j, i) = interpolateH_from_B(bh_it->second, B_mag);
            } else {
                // Linear material: use H = B/μ
                double mu = mu_map(j, i);
                if (mu > 1e-20) {
                    H_map(j, i) = B_mag / mu;
                } else {
                    H_map(j, i) = 0.0;
                }
            }
        }
    }
}

/**
 * @brief Update mu_map distribution based on current H_map
 */
void MagneticFieldAnalyzer::updateMuDistribution() {
    const double MU_0 = 4.0 * M_PI * 1e-7;

    if (!config["materials"]) {
        return;
    }

    // IMPORTANT: Match the image orientation used in setupMaterialProperties()
    // setupMaterialProperties() always flips the image vertically for BOTH cartesian and polar
    cv::Mat image_to_use;
    cv::flip(image, image_to_use, 0);  // Flip vertically: y down -> y up

    // For each pixel, look up material and update mu
    for (int j = 0; j < image_to_use.rows; j++) {
        for (int i = 0; i < image_to_use.cols; i++) {
            // Use image_to_use (flipped) to match setupMaterialProperties()
            cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(j, i);
            cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);  // BGR to RGB

            // Find matching material
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
                    // Found matching material
                    double H_mag = H_map(j, i);
                    double mu_r = 1.0;

                    auto it = material_mu.find(name);
                    if (it != material_mu.end()) {
                        mu_r = evaluateMu(it->second, H_mag);
                    } else if (props["mu_r"]) {
                        // Fallback: parse inline (for static mu_r)
                        try {
                            mu_r = props["mu_r"].as<double>();
                        } catch (...) {
                            mu_r = 1.0;
                        }
                    }

                    mu_map(j, i) = mu_r * MU_0;
                    break;
                }
            }
        }
    }
}

// ============================================
// Nonlinear Solver Methods
// ============================================

/**
 * @brief Nonlinear Picard iteration solver with relaxation
 *
 * Algorithm:
 * 1. Initialize μ = μ(H=0)
 * 2. Solve linear system with current μ distribution
 * 3. Calculate B from Az
 * 4. Calculate H = B/μ
 * 5. Update μ = μ(H) with relaxation
 * 6. Check convergence
 * 7. Repeat from step 2
 */
void MagneticFieldAnalyzer::solveNonlinear() {
    if (!has_nonlinear_materials) {
        // No nonlinear materials, use standard linear solver
        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }
        return;
    }

    std::cout << "\n=== Nonlinear Solver (Picard Iteration) ===" << std::endl;
    std::cout << "Max iterations: " << nonlinear_config.max_iterations << std::endl;
    std::cout << "Tolerance: " << nonlinear_config.tolerance << std::endl;
    std::cout << "Relaxation: " << nonlinear_config.relaxation << std::endl;

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    const double OMEGA = nonlinear_config.relaxation;

    // Save previous solution for convergence check
    Eigen::VectorXd Az_old;
    if (coordinate_system == "cartesian") {
        Az_old = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
    } else {
        Az_old = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
    }

    // Convergence history (for optional export)
    std::vector<double> residual_history;
    std::vector<double> mu_change_history;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Step 1: Solve linear system with current μ distribution
        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }

        // Step 2: Calculate magnetic field B
        if (coordinate_system == "cartesian") {
            calculateMagneticField();
        } else {
            calculateMagneticFieldPolar();
        }

        // Step 3: Calculate |H| from B and current μ
        calculateHField();

        // Step 4: Update μ distribution with relaxation
        Eigen::MatrixXd mu_old = mu_map;
        updateMuDistribution();

        // Apply relaxation: μ_new = ω * μ_new + (1-ω) * μ_old
        mu_map = OMEGA * mu_map + (1.0 - OMEGA) * mu_old;

        // Step 5: Check convergence
        Eigen::VectorXd Az_new = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
        double Az_diff_norm = (Az_new - Az_old).norm();
        double Az_norm = Az_new.norm();

        // Avoid division by zero: use relative residual if ||Az|| is large, absolute otherwise
        double Az_residual = (Az_norm > 1e-12) ? (Az_diff_norm / Az_norm) : Az_diff_norm;

        // Calculate relative mu change
        Eigen::VectorXd mu_old_vec = Eigen::Map<Eigen::VectorXd>(mu_old.data(), mu_old.size());
        Eigen::VectorXd mu_new_vec = Eigen::Map<Eigen::VectorXd>(mu_map.data(), mu_map.size());
        double mu_change_norm = (mu_new_vec - mu_old_vec).norm();
        double mu_old_norm = mu_old_vec.norm();
        double mu_change_rel = (mu_old_norm > 1e-20) ? (mu_change_norm / mu_old_norm) : mu_change_norm;

        residual_history.push_back(Az_residual);
        mu_change_history.push_back(mu_change_rel);

        if (nonlinear_config.verbose) {
            std::cout << "NL iter " << std::setw(3) << iter + 1
                      << ": ||ΔAz|| = " << std::scientific << std::setprecision(4) << Az_residual
                      << ",  ||Δμ||/||μ|| = " << mu_change_rel
                      << std::endl;
        }

        // Convergence check (both Az and mu must converge)
        if (Az_residual < TOL && mu_change_rel < TOL) {
            std::cout << "Nonlinear solver converged in " << iter + 1 << " iterations" << std::endl;

            // Export convergence history if requested
            if (nonlinear_config.export_convergence) {
                std::ofstream conv_file("nonlinear_convergence.csv");
                conv_file << "iteration,Az_residual,mu_change\n";
                for (size_t i = 0; i < residual_history.size(); i++) {
                    conv_file << i + 1 << "," << residual_history[i] << "," << mu_change_history[i] << "\n";
                }
                conv_file.close();
                std::cout << "Convergence history saved to: nonlinear_convergence.csv" << std::endl;
            }

            return;
        }

        Az_old = Az_new;
    }

    std::cerr << "WARNING: Nonlinear solver did not converge after "
              << MAX_ITER << " iterations!" << std::endl;
    std::cerr << "Final residual: " << residual_history.back() << std::endl;
}

/**
 * @brief Nonlinear solver with Anderson acceleration
 *
 * Anderson acceleration (AA) is a method to accelerate fixed-point iteration:
 * x_{k+1} = g(x_k) → accelerated update using history of {x_k, g(x_k)}
 *
 * Reference: Walker & Ni, "Anderson Acceleration for Fixed-Point Iterations", SIAM J. Numer. Anal., 2011
 */
void MagneticFieldAnalyzer::solveNonlinearWithAnderson() {
    if (!has_nonlinear_materials) {
        // No nonlinear materials, use standard linear solver
        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }
        return;
    }

    const int m_AA = nonlinear_config.anderson_depth;  // Anderson depth
    if (m_AA <= 0) {
        // Anderson disabled, fall back to standard Picard
        solveNonlinear();
        return;
    }

    std::cout << "\n=== Nonlinear Solver (Picard + Anderson Acceleration) ===" << std::endl;
    std::cout << "Anderson depth: " << m_AA << std::endl;

    const int MAX_ITER = nonlinear_config.max_iterations;
    const double TOL = nonlinear_config.tolerance;
    const double OMEGA = nonlinear_config.relaxation;

    // Anderson acceleration storage
    std::vector<Eigen::VectorXd> mu_history;      // μ^(k)
    std::vector<Eigen::VectorXd> residual_history_AA; // r^(k) = μ^(k+1) - μ^(k)

    // Flatten mu_map to vector for Anderson
    auto flatten_mu = [&]() -> Eigen::VectorXd {
        return Eigen::Map<Eigen::VectorXd>(mu_map.data(), mu_map.size());
    };

    auto unflatten_mu = [&](const Eigen::VectorXd& mu_vec) {
        mu_map = Eigen::Map<const Eigen::MatrixXd>(mu_vec.data(), mu_map.rows(), mu_map.cols());
    };

    Eigen::VectorXd Az_old = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
    std::vector<double> Az_residual_history;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Standard Picard step
        Eigen::VectorXd mu_old = flatten_mu();

        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }

        if (coordinate_system == "cartesian") {
            calculateMagneticField();
        } else {
            calculateMagneticFieldPolar();
        }

        calculateHField();

        Eigen::MatrixXd mu_map_before_relax = mu_map;
        updateMuDistribution();

        // Relaxation before Anderson
        mu_map = OMEGA * mu_map + (1.0 - OMEGA) * mu_map_before_relax;

        Eigen::VectorXd mu_new = flatten_mu();
        Eigen::VectorXd residual = mu_new - mu_old;

        // Anderson acceleration
        if (iter >= 1 && mu_history.size() > 0) {
            int m_k = std::min(m_AA, static_cast<int>(mu_history.size()));

            // Build least-squares problem: min || F * alpha - residual ||
            // where F = [r^(k-m_k), ..., r^(k-1)]
            Eigen::MatrixXd F(residual.size(), m_k);
            for (int j = 0; j < m_k; j++) {
                int idx = residual_history_AA.size() - m_k + j;
                F.col(j) = residual_history_AA[idx] - residual;
            }

            // Solve least-squares: F^T F alpha = F^T residual
            Eigen::VectorXd alpha = (F.transpose() * F).ldlt().solve(F.transpose() * residual);

            // Anderson update: μ^(k+1) = μ^(k) + residual - Σ alpha_j (r^(k-m_k+j) - residual)
            Eigen::VectorXd mu_anderson = mu_new;
            for (int j = 0; j < m_k; j++) {
                int idx = mu_history.size() - m_k + j;
                mu_anderson -= alpha(j) * (mu_history[idx] - mu_old);
            }

            unflatten_mu(mu_anderson);
        }

        // Store history
        mu_history.push_back(mu_old);
        residual_history_AA.push_back(residual);

        // Limit history size
        if (static_cast<int>(mu_history.size()) > m_AA + 1) {
            mu_history.erase(mu_history.begin());
            residual_history_AA.erase(residual_history_AA.begin());
        }

        // Convergence check
        Eigen::VectorXd Az_new = Eigen::Map<Eigen::VectorXd>(Az.data(), Az.size());
        double Az_diff = (Az_new - Az_old).norm();
        double Az_norm = Az_new.norm();
        double mu_diff = residual.norm();
        double mu_old_norm = mu_old.norm();

        // Avoid division by zero: use relative residual if norm is large, absolute otherwise
        double Az_res = (Az_norm > 1e-12) ? (Az_diff / Az_norm) : Az_diff;
        double mu_res = (mu_old_norm > 1e-12) ? (mu_diff / mu_old_norm) : mu_diff;

        Az_residual_history.push_back(Az_res);

        if (nonlinear_config.verbose) {
            std::cout << "AA iter " << std::setw(3) << iter + 1
                      << ": ||ΔAz|| = " << std::scientific << std::setprecision(4) << Az_res
                      << ",  ||Δμ||/||μ|| = " << mu_res
                      << std::endl;
        }

        if (Az_res < TOL && mu_res < TOL) {
            std::cout << "Anderson-accelerated solver converged in " << iter + 1 << " iterations" << std::endl;
            return;
        }

        Az_old = Az_new;
    }

    std::cerr << "WARNING: Anderson solver did not converge after " << MAX_ITER << " iterations!" << std::endl;
}
