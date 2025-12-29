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
#include "tinyexpr/tinyexpr.h"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

// ============================================
// PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) Helper Functions
// Guarantees monotonicity preservation for B-H curve interpolation
// ============================================

namespace {

/**
 * @brief Compute PCHIP slopes that preserve monotonicity
 *
 * Uses Fritsch-Carlson method to ensure interpolated curve is monotonic
 * in regions where input data is monotonic.
 */
std::vector<double> computePCHIPSlopes(const std::vector<double>& x, const std::vector<double>& y) {
    int n = static_cast<int>(x.size());
    if (n < 2) return std::vector<double>(n, 0.0);

    std::vector<double> h(n-1), delta(n-1);

    // Compute intervals and secants
    for (int i = 0; i < n-1; i++) {
        h[i] = x[i+1] - x[i];
        if (h[i] > 0) {
            delta[i] = (y[i+1] - y[i]) / h[i];
        } else {
            delta[i] = 0.0;
        }
    }

    // Compute slopes with monotonicity preservation
    std::vector<double> d(n);

    // Endpoint slopes
    d[0] = delta[0];
    d[n-1] = delta[n-2];

    // Interior slopes
    for (int i = 1; i < n-1; i++) {
        if (delta[i-1] * delta[i] <= 0) {
            // Sign change or zero: set slope to zero for monotonicity
            d[i] = 0.0;
        } else {
            // Weighted harmonic mean (Fritsch-Carlson)
            double w1 = 2.0 * h[i] + h[i-1];
            double w2 = h[i] + 2.0 * h[i-1];
            d[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i]);
        }
    }

    return d;
}

/**
 * @brief PCHIP interpolation at query point
 */
double pchipInterpolate(const std::vector<double>& x, const std::vector<double>& y,
                        const std::vector<double>& d, double xq) {
    int n = static_cast<int>(x.size());

    // Handle out of range
    if (xq <= x[0]) return y[0];
    if (xq >= x[n-1]) return y[n-1];

    // Find interval using binary search
    auto it = std::upper_bound(x.begin(), x.end(), xq);
    int idx = static_cast<int>(std::distance(x.begin(), it)) - 1;
    if (idx < 0) idx = 0;
    if (idx >= n-1) idx = n-2;

    double h = x[idx+1] - x[idx];
    if (h <= 0) return y[idx];

    double t = (xq - x[idx]) / h;

    // Hermite basis functions
    double h00 = (1.0 + 2.0*t) * (1.0-t) * (1.0-t);
    double h10 = t * (1.0-t) * (1.0-t);
    double h01 = t * t * (3.0 - 2.0*t);
    double h11 = t * t * (t - 1.0);

    return h00*y[idx] + h10*h*d[idx] + h01*y[idx+1] + h11*h*d[idx+1];
}

} // anonymous namespace

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

            // Validate coordinate system-specific variables
            bool has_dx_dy = (mu_str.find("$dx") != std::string::npos ||
                              mu_str.find("$dy") != std::string::npos);
            bool has_dr_dtheta = (mu_str.find("$dr") != std::string::npos ||
                                  mu_str.find("$dtheta") != std::string::npos);

            if (has_dx_dy && coordinate_system == "polar") {
                throw std::runtime_error("mu_r formula error: $dx, $dy can only be used in Cartesian coordinates. "
                                         "Use $dr, $dtheta for polar coordinates.");
            }
            if (has_dr_dtheta && coordinate_system == "cartesian") {
                throw std::runtime_error("mu_r formula error: $dr, $dtheta can only be used in polar coordinates. "
                                         "Use $dx, $dy for Cartesian coordinates.");
            }

            // Replace formula variables with tinyexpr-compatible names
            // Order matters: longer names first to avoid partial replacements
            size_t pos = 0;
            while ((pos = result.formula.find("$dtheta", pos)) != std::string::npos) {
                result.formula.replace(pos, 7, "dtheta");
                pos += 6;
            }
            pos = 0;
            while ((pos = result.formula.find("$H", pos)) != std::string::npos) {
                result.formula.replace(pos, 2, "H");
                pos += 1;
            }
            pos = 0;
            while ((pos = result.formula.find("$dx", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dx");
                pos += 2;
            }
            pos = 0;
            while ((pos = result.formula.find("$dy", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dy");
                pos += 2;
            }
            pos = 0;
            while ((pos = result.formula.find("$dr", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dr");
                pos += 2;
            }

            // Replace user-defined variables (sorted by name length descending to avoid partial replacements)
            std::vector<std::pair<std::string, double>> sorted_vars(user_variables.begin(), user_variables.end());
            std::sort(sorted_vars.begin(), sorted_vars.end(),
                      [](const auto& a, const auto& b) { return a.first.length() > b.first.length(); });

            for (const auto& [var_name, var_value] : sorted_vars) {
                std::string search_str = "$" + var_name;
                pos = 0;
                while ((pos = result.formula.find(search_str, pos)) != std::string::npos) {
                    result.formula.replace(pos, search_str.length(), var_name);
                    pos += var_name.length();
                }
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

            // Prepare variables
            std::set<te_variable> vars;

            // H variable (magnetic field intensity)
            te_variable H_var;
            H_var.m_name = "H";
            H_var.m_value = H_magnitude;
            vars.insert(H_var);

            // Coordinate system-specific variables
            if (coordinate_system == "cartesian") {
                te_variable dx_var, dy_var;
                dx_var.m_name = "dx";
                dx_var.m_value = dx;
                dy_var.m_name = "dy";
                dy_var.m_value = dy;
                vars.insert(dx_var);
                vars.insert(dy_var);
            } else {  // polar
                te_variable dr_var, dtheta_var;
                dr_var.m_name = "dr";
                dr_var.m_value = dr;
                dtheta_var.m_name = "dtheta";
                dtheta_var.m_value = dtheta;
                vars.insert(dr_var);
                vars.insert(dtheta_var);
            }

            // User-defined variables
            for (const auto& [var_name, var_value] : user_variables) {
                te_variable user_var;
                user_var.m_name = var_name.c_str();
                user_var.m_value = var_value;
                vars.insert(user_var);
            }

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
            // PCHIP interpolation of B = μ₀ × μ_r × H to ensure monotonicity
            // Then compute μ_r = B / (μ₀ × H)
            const double MU_0 = 4.0 * M_PI * 1e-7;
            const auto& H_tab = mu_val.H_table;
            const auto& mu_tab = mu_val.mu_table;

            // Handle H = 0 case
            if (H_magnitude <= 1e-12) {
                return mu_tab.front();
            }

            // Handle out-of-range (extrapolate with constant μ_r)
            if (H_magnitude <= H_tab.front()) {
                return mu_tab.front();
            }
            if (H_magnitude >= H_tab.back()) {
                return mu_tab.back();
            }

            // Compute B values at table points: B_i = μ₀ × μ_r_i × H_i
            std::vector<double> B_tab(H_tab.size());
            for (size_t i = 0; i < H_tab.size(); i++) {
                B_tab[i] = MU_0 * mu_tab[i] * H_tab[i];
            }

            // Compute PCHIP slopes for B(H) curve
            std::vector<double> slopes = computePCHIPSlopes(H_tab, B_tab);

            // PCHIP interpolation to get B at H_magnitude
            double B_interp = pchipInterpolate(H_tab, B_tab, slopes, H_magnitude);

            // Compute μ_r = B / (μ₀ × H)
            double mu_r = B_interp / (MU_0 * H_magnitude);

            // Ensure μ_r >= 1 (physical constraint)
            if (mu_r < 1.0) {
                mu_r = 1.0;
            }

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

            // Handle out-of-range with user-defined extrapolation
            if (H_magnitude <= H_tab.front() || H_magnitude >= H_tab.back()) {
                // Use extrapolation function if specified
                if (mu_val.has_dmu_extrapolation) {
                    if (!mu_val.dmu_r_extrap_formula.empty()) {
                        // Evaluate formula-based extrapolation
                        te_parser parser;
                        te_variable H_var;
                        H_var.m_name = "H";
                        H_var.m_value = H_magnitude;

                        std::set<te_variable> vars = {H_var};
                        parser.set_variables_and_functions(vars);

                        double result = parser.evaluate(mu_val.dmu_r_extrap_formula);
                        if (!parser.success()) {
                            std::cerr << "WARNING: Failed to evaluate dmu_r extrapolation formula at H="
                                      << H_magnitude << ", using constant fallback" << std::endl;
                            return mu_val.dmu_r_extrap_const;
                        }
                        return result;
                    } else {
                        // Use constant extrapolation
                        return mu_val.dmu_r_extrap_const;
                    }
                } else {
                    // Default: dμ_r/dH = 1.0 (equivalent to vacuum permeability)
                    return 1.0;
                }
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

    // Direct calculation using effective permeability: B(H) = μ_eff(H) * μ₀ * H
    // IMPORTANT: mu_r in YAML is the effective permeability μ_eff = B/H
    // This is standard catalog data format (not differential permeability dB/dH)
    for (size_t i = 0; i < H_samples.size(); i++) {
        double H = H_samples[i];
        double mu_eff = evaluateMu(mu_val, H);  // μ_eff = B/H from catalog
        double mu = mu_eff * MU_0;  // Absolute permeability [H/m]
        double B = mu * H;  // Direct calculation: B = μ * H (no integration!)

        table.H_values.push_back(H);
        table.B_values.push_back(B);
        table.mu_values.push_back(mu);  // Store μ = μ_eff * μ₀
    }

    // ========================================
    // B-H Curve Validation (ALWAYS SHOW WARNINGS)
    // ========================================

    // Check 1: B-H curve monotonicity (dB/dH > 0)
    bool is_BH_monotonic = true;
    for (size_t i = 1; i < table.B_values.size(); i++) {
        double dB = table.B_values[i] - table.B_values[i-1];
        double dH = table.H_values[i] - table.H_values[i-1];
        if (dB < 0.0) {
            std::cerr << "\n**************************************************\n"
                      << "WARNING: Material '" << material_name << "'\n"
                      << "  B-H curve is NOT monotonically increasing!\n"
                      << "  At H=" << table.H_values[i] << " A/m: dB/dH = " << (dB/dH) << " < 0\n"
                      << "  This will cause severe convergence issues!\n"
                      << "  Please check your mu_r table definition in YAML.\n"
                      << "**************************************************\n" << std::endl;
            is_BH_monotonic = false;
            break;
        }
    }

    // Check 2: Extrapolation function validation (if specified)
    if (mu_val.has_dmu_extrapolation) {
        bool extrapolation_valid = true;

        // Test extrapolation function at several high-H values
        std::vector<double> test_H_values = {
            mu_val.H_table.back() * 1.5,
            mu_val.H_table.back() * 2.0,
            mu_val.H_table.back() * 5.0,
            mu_val.H_table.back() * 10.0
        };

        for (double H_test : test_H_values) {
            double dmu_r_extrap = 0.0;

            if (!mu_val.dmu_r_extrap_formula.empty()) {
                // Evaluate formula
                te_parser parser;
                te_variable H_var;
                H_var.m_name = "H";
                H_var.m_value = H_test;

                std::set<te_variable> vars = {H_var};
                parser.set_variables_and_functions(vars);

                dmu_r_extrap = parser.evaluate(mu_val.dmu_r_extrap_formula);
                if (!parser.success()) {
                    dmu_r_extrap = mu_val.dmu_r_extrap_const;
                }
            } else {
                dmu_r_extrap = mu_val.dmu_r_extrap_const;
            }

            if (dmu_r_extrap < 0.0) {
                std::cerr << "\n**************************************************\n"
                          << "WARNING: Material '" << material_name << "'\n"
                          << "  Extrapolation function dmu_r/dH is NEGATIVE!\n"
                          << "  At H=" << H_test << " A/m: dmu_r/dH = " << dmu_r_extrap << "\n"
                          << "  This will cause Newton-Krylov divergence!\n"
                          << "  Please fix your dmu_r_extrapolation definition.\n"
                          << "**************************************************\n" << std::endl;
                extrapolation_valid = false;
                break;
            }
        }
    }

    table.is_valid = true;

    std::cout << "Generated B-H table for '" << material_name << "': "
              << table.H_values.size() << " points, "
              << "H = [" << table.H_values.front() << ", " << table.H_values.back() << "] A/m, "
              << "B = [" << table.B_values.front() << ", " << table.B_values.back() << "] T";
    if (is_BH_monotonic) {
        std::cout << " [Monotonic: OK]";
    } else {
        std::cout << " [Monotonic: FAILED]";
    }
    std::cout << std::endl;
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
 * @brief Interpolate |B| from |H| using B-H table
 */
double MagneticFieldAnalyzer::interpolateB_from_H(const BHTable& table, double H_magnitude) {
    if (!table.is_valid || table.H_values.empty()) {
        std::cerr << "ERROR: B-H table is not valid" << std::endl;
        return 0.0;
    }

    const auto& H_tab = table.H_values;
    const auto& B_tab = table.B_values;

    // Handle out-of-range
    if (H_magnitude <= H_tab.front()) {
        // Linear extrapolation: B = μ(0) * H
        return table.mu_values.front() * H_magnitude;
    }

    if (H_magnitude >= H_tab.back()) {
        // Linear extrapolation using last segment slope (dB/dH at saturation ≈ μ₀)
        size_t n = H_tab.size();
        double dB = B_tab[n-1] - B_tab[n-2];
        double dH = H_tab[n-1] - H_tab[n-2];
        if (std::abs(dH) > 1e-20) {
            double slope = dB / dH;  // dB/dH (differential permeability)
            return B_tab[n-1] + slope * (H_magnitude - H_tab[n-1]);
        }
        return B_tab[n-1];
    }

    // Find interpolation interval (binary search)
    auto it = std::upper_bound(H_tab.begin(), H_tab.end(), H_magnitude);
    size_t idx = std::distance(H_tab.begin(), it) - 1;

    // Linear interpolation
    double H0 = H_tab[idx];
    double H1 = H_tab[idx + 1];
    double B0 = B_tab[idx];
    double B1 = B_tab[idx + 1];

    double alpha = (H_magnitude - H0) / (H1 - H0);
    double B = B0 + alpha * (B1 - B0);

    return B;
}

/**
 * @brief Integrate magnetic co-energy density W' = ∫₀^H B(H') dH' using Simpson's rule
 *
 * For current-source systems (Jz specified), the force is given by F = +∂W'/∂x|_I
 * where W' is the magnetic co-energy (Legendre transform of energy W).
 *
 * Co-energy is computed by integrating B as a function of H from 0 to H_magnitude.
 * This uses composite Simpson's rule with adaptive subdivision.
 *
 * For linear materials with μ = const, W' = W = B²/(2μ) = μH²/2.
 * For nonlinear materials, W' ≠ W in general.
 *
 * @param table B-H table for the nonlinear material
 * @param H_magnitude Target magnetic field intensity |H| [A/m]
 * @return Magnetic co-energy density W' [J/m³]
 */
double MagneticFieldAnalyzer::integrateMagneticCoEnergy(const BHTable& table, double H_magnitude) {
    if (!table.is_valid || table.H_values.empty()) {
        // Fallback: linear approximation using secant permeability
        double mu = (table.mu_values.empty()) ? (4.0 * M_PI * 1e-7) : table.mu_values.front();
        return 0.5 * mu * H_magnitude * H_magnitude;  // W' = μH²/2
    }

    if (H_magnitude <= 0.0) {
        return 0.0;
    }

    // Number of subintervals for Simpson's rule (must be even)
    // Use more points for larger H to maintain accuracy
    int n = 100;  // Default: 100 subintervals (101 points)
    if (H_magnitude > table.H_values.back()) {
        n = 200;  // More points for extrapolation region
    }

    double h = H_magnitude / n;  // Step size

    // Simpson's rule: W' = (h/3) * [B(0) + 4*B(h) + 2*B(2h) + 4*B(3h) + ... + B(H)]
    double sum = 0.0;

    // B(0) term
    double B0 = interpolateB_from_H(table, 0.0);
    sum += B0;

    // Intermediate terms
    for (int i = 1; i < n; i++) {
        double H_i = i * h;
        double B_i = interpolateB_from_H(table, H_i);
        if (i % 2 == 1) {
            sum += 4.0 * B_i;  // Odd indices: coefficient 4
        } else {
            sum += 2.0 * B_i;  // Even indices: coefficient 2
        }
    }

    // B(H_magnitude) term
    double B_n = interpolateB_from_H(table, H_magnitude);
    sum += B_n;

    double coenergy = (h / 3.0) * sum;

    return coenergy;
}

/**
 * @brief Calculate magnetic co-energy density at grid point (j, i)
 *
 * For current-source systems (Jz specified), force is F = +∂W'/∂x|_I
 * where W' is the magnetic co-energy.
 *
 * For nonlinear materials: W' = ∫₀^H B(H') dH' using Simpson integration
 * For linear materials: W' = W = B²/(2μ) = μH²/2
 *
 * NOTE: This function uses a static cached flipped image for efficiency.
 * The cache is invalidated when the image dimensions change.
 *
 * @param j Row index in grid
 * @param i Column index in grid
 * @param B_magnitude Magnetic flux density |B| at this point [T]
 * @return Co-energy density W' [J/m³]
 */
double MagneticFieldAnalyzer::calculateCoEnergyDensity(int j, int i, double B_magnitude) {
    const double MU_0 = 4.0 * M_PI * 1e-7;

    // Default: linear material with mu from mu_map
    double mu = mu_map(j, i);
    if (mu < 1e-20) mu = MU_0;  // Safety check

    // Check if this pixel belongs to a nonlinear material
    if (!has_nonlinear_materials || !config["materials"]) {
        // No nonlinear materials: use linear formula (W' = W for linear)
        return B_magnitude * B_magnitude / (2.0 * mu);
    }

    // Flip image for each call to ensure consistency with current image content
    // Note: For sliding simulations, image content changes while dimensions stay same,
    // so we cannot use static caching based on dimensions alone.
    cv::Mat cached_image_flipped;
    cv::flip(image, cached_image_flipped, 0);

    // Bounds check
    if (j < 0 || j >= cached_image_flipped.rows || i < 0 || i >= cached_image_flipped.cols) {
        return B_magnitude * B_magnitude / (2.0 * mu);
    }

    cv::Vec3b pixel = cached_image_flipped.at<cv::Vec3b>(j, i);
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
            // Found matching material - check if it has a B-H table
            auto bh_it = material_bh_tables.find(name);
            if (bh_it != material_bh_tables.end() && bh_it->second.is_valid) {
                // Nonlinear material: compute H from B, then integrate W' = ∫B dH
                double H_magnitude = interpolateH_from_B(bh_it->second, B_magnitude);
                return integrateMagneticCoEnergy(bh_it->second, H_magnitude);
            } else {
                // Linear material: W' = W = B²/(2μ)
                return B_magnitude * B_magnitude / (2.0 * mu);
            }
        }
    }

    // Material not found: use linear formula
    return B_magnitude * B_magnitude / (2.0 * mu);
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
                    double mu_r = 1.0;  // Default to 1 (air)

                    auto it = material_mu.find(name);
                    if (it != material_mu.end()) {
                        // Nonlinear material: interpolate μ_eff(H) from YAML table
                        // IMPORTANT: mu_r in YAML is effective permeability μ_eff = B/H
                        mu_r = evaluateMu(it->second, H_mag);
                    } else if (props["mu_r"]) {
                        // Linear material: use constant mu_r from YAML
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

    // Anderson acceleration settings (shared config)
    const int m_AA = nonlinear_config.anderson.depth;
    const double beta_AA = nonlinear_config.anderson.beta;

    if (!nonlinear_config.anderson.enabled || m_AA <= 0) {
        // Anderson disabled, fall back to standard Picard
        solveNonlinear();
        return;
    }

    std::cout << "\n=== Nonlinear Solver (Picard + Anderson Acceleration) ===" << std::endl;
    std::cout << "Anderson depth: " << m_AA << ", beta: " << beta_AA << std::endl;

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
