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
#include <cmath>
#include <limits>
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
double MagneticFieldAnalyzer::dHdB_FromTable(const BHTable& table, double B_magnitude) const {
    // Differential reluctivity dH/dB from the (piecewise-linear) B-H table.
    // Beyond the table end B(H) = B_end + mu0*(H - H_end), so dH/dB -> 1/mu0.
    const double MU0 = 4.0 * M_PI * 1e-7;
    const auto& Bt = table.B_values;
    const auto& Ht = table.H_values;
    if (!table.is_valid || Bt.size() < 2) return 1.0 / MU0;
    if (B_magnitude >= Bt.back()) return 1.0 / MU0;
    size_t k;
    if (B_magnitude <= Bt.front()) {
        k = 0;
    } else {
        auto it = std::upper_bound(Bt.begin(), Bt.end(), B_magnitude);
        k = static_cast<size_t>(std::distance(Bt.begin(), it)) - 1;
        if (k + 1 >= Bt.size()) k = Bt.size() - 2;
    }
    const double dB = Bt[k + 1] - Bt[k];
    const double dH = Ht[k + 1] - Ht[k];
    return (dB > 1e-30) ? (dH / dB) : 1.0 / MU0;
}

void MagneticFieldAnalyzer::addTangentCorrection(Eigen::SparseMatrix<double>& J) {
    // Consistent-tangent correction (Stage C of the CTSM x energy program).
    //
    // STATUS (2026-07 benchmark): the DIRECTION quality of the assembled
    // tangent reproduces the matrix-free Stage G gate (alpha=1 accepted for a
    // third of the iterations under the energy merit, vs the 0.1 crawl), but
    // the INNER solve cost explodes: BiCGStab on the tangent J preconditioned
    // by AMG(secant A) needs 100+ iterations where the secant path needs 3-8
    // CG its, so per-NK-iteration cost is ~6 s vs ~0.5 s and the iteration
    // savings are eaten (measured 248 s / 40 iters vs 37 s / 48 secant).
    // NEGATIVE for speed; kept opt-in (jacobian: tangent) because the energy
    // x tangent pairing is the only configuration that pierces the 5e-3
    // plateau (gate: 3.6e-4 true convergence) — the identified follow-ups are
    // a preconditioner FOR the tangent operator, or a hybrid that switches
    // secant -> tangent only after plateau detection (a true-convergence
    // QUALITY option, not a speed one).
    //
    // The discrete magnetostatic energy is W = sum_c w(|B_c|) vol_c - <j, Az>
    // with B_c from the central-difference curl (calculateMagneticFieldPolar).
    // Its Hessian is sum_c vol_c G_c^T [ nu I + (nu_d - nu) bhat bhat^T ] G_c:
    // the isotropic nu part is (spectrally) the assembled secant operator A,
    // and the missing curvature is the per-cell RANK-ONE term
    //     vol_c (nu_d - nu) (G_c^T bhat)(G_c^T bhat)^T,
    // where g = G_c^T bhat has four entries on the cell's theta/r neighbours:
    //     +-bhat_r/(2 dtheta r) on Az(i, j+-1),  -+bhat_theta/(2 dr) on Az(i+-1, j).
    // nu = H/B (secant), nu_d = dH/dB (differential). Linear cells have
    // nu_d = nu and contribute nothing. The Stage G gate measured this
    // curvature to be what unlocks alpha = 1 under the energy merit
    // (25 true-tolerance iterations vs 48 plateau-stalled).
    //
    // Safeguards: entries touching the Dirichlet radial boundary rows are
    // dropped (those rows stay pure BC rows), and the coefficient is floored
    // at -0.9*nu*vol so a rising-mu region (nu_d < nu) cannot push the
    // assembled J indefinite (the exact Hessian is SPD since nu, nu_d > 0,
    // but A only approximates the isotropic part).
    if (coordinate_system == "cartesian") return;
    const double MU0 = 4.0 * M_PI * 1e-7;
    const bool horiz = (r_orientation == "horizontal");
    const bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");

    cv::Mat image_to_use;
    cv::flip(image, image_to_use, 0);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve((size_t)16 * 1024 * 1024);   // ~860k NL cells x up to 16 entries

    for (int i = 1; i < nr - 1; ++i) {          // skip radial boundary cells (Dirichlet rows)
        const double r = r_start + i * dr;
        if (r < 1e-12) continue;
        for (int j = 0; j < ntheta; ++j) {
            const int mr = horiz ? j : i;       // map/grid row of cell (theta,r) layout
            const int mc = horiz ? i : j;
            const cv::Vec3b px = image_to_use.at<cv::Vec3b>(mr, mc);
            const int rgb_key = (px[0] << 16) | (px[1] << 8) | px[2];
            auto lut_it = rgb_to_material.find(rgb_key);
            if (lut_it == rgb_to_material.end()) continue;
            auto bh_it = material_bh_tables.find(lut_it->second.name);
            if (bh_it == material_bh_tables.end() || !bh_it->second.is_valid) continue;

            const double br = Br(mr, mc), bt = Btheta(mr, mc);
            const double Bmag = std::sqrt(br * br + bt * bt);
            if (Bmag < 1e-12) continue;
            const double H_mag = H_map(mr, mc);
            const double nu   = H_mag / Bmag;
            const double nu_d = dHdB_FromTable(bh_it->second, Bmag);
            // dnu_c/dB = (nu_d - nu)/B; no volume factor — the h vector below
            // already carries the r-weighted FV row scaling of the residual.
            double coef = (nu_d - nu) / Bmag;
            if (coef < -0.9 * nu / Bmag) coef = -0.9 * nu / Bmag;   // definiteness floor
            if (std::abs(coef) < 1e-30) continue;

            const double bhr = br / Bmag, bht = bt / Bmag;
            // g = d|B_c|/dAz (the central-difference curl stencil, 4 entries).
            int    gidx[4];
            double gw[4];
            int    ng = 0;
            const int jp = is_periodic ? (j + 1) % ntheta : std::min(j + 1, ntheta - 1);
            const int jm = is_periodic ? (j - 1 + ntheta) % ntheta : std::max(j - 1, 0);
            if (jp != jm) {
                const double wth = bhr / (2.0 * dtheta * r);
                gidx[ng] = i * ntheta + jp; gw[ng++] = +wth;
                gidx[ng] = i * ntheta + jm; gw[ng++] = -wth;
            }
            const double wr = -bht / (2.0 * dr);
            gidx[ng] = (i + 1) * ntheta + j; gw[ng++] = +wr;
            gidx[ng] = (i - 1) * ntheta + j; gw[ng++] = -wr;

            // h = dR/dnu_c: each face f=(c,n) has coefficient s_f*nu_f with
            // nu_f = (nu_c + nu_n)/2 (harmonic-mu face), so dc_f/dnu_c = s_f/2
            // and face f contributes (s_f/2)*(Az_n - Az_c) to row c and the
            // negative to row n. Geometric factors from buildMatrixPolar:
            // radial s = r_{i +/- 1/2}/dr^2, theta s = 1/(r*dtheta^2).
            auto AzAt = [&](int ri, int tj) -> double {
                return horiz ? Az(tj, ri) : Az(ri, tj);
            };
            const double Az_c = AzAt(i, j);
            int    hidx[5];
            double hw[5];
            int    nh = 0;
            double h_center = 0.0;
            auto addFace = [&](int ni, int nj, double s_f) {
                const double d = 0.5 * s_f * (AzAt(ni, nj) - Az_c);
                hidx[nh] = ni * ntheta + nj; hw[nh++] = -d;   // row n gets -(s/2)(Az_n - Az_c)... sign per row-n's face term
                h_center += d;                                 // row c gets +(s/2)(Az_n - Az_c)
            };
            addFace(i + 1, j, (r + 0.5 * dr) / (dr * dr));
            addFace(i - 1, j, (r - 0.5 * dr) / (dr * dr));
            if (jp != jm) {
                addFace(i, jp, 1.0 / (r * dtheta * dtheta));
                addFace(i, jm, 1.0 / (r * dtheta * dtheta));
            }
            hidx[nh] = i * ntheta + j; hw[nh++] = h_center;

            // Scatter the RAW (nonsymmetric) rank-two update coef * h g^T —
            // this IS dR/dAz restricted to the mu chain rule. (Symmetrizing it
            // was measured to wreck the inner-solve conditioning: AMGCL-CG went
            // 3 -> 1800+ iterations. The tangent path therefore solves with
            // BiCGStab preconditioned by AMG on the SPD secant A, mirroring the
            // matrix-free Stage G gate.)
            for (int a2 = 0; a2 < nh; ++a2) {
                const int ra = hidx[a2] / ntheta;
                if (ra == 0 || ra == nr - 1) continue;       // keep Dirichlet rows clean
                for (int b2 = 0; b2 < ng; ++b2) {
                    const int rb = gidx[b2] / ntheta;
                    if (rb == 0 || rb == nr - 1) continue;
                    trips.emplace_back(hidx[a2], gidx[b2], coef * hw[a2] * gw[b2]);
                }
            }
        }
    }
    if (trips.empty()) return;
    Eigen::SparseMatrix<double> dJ(J.rows(), J.cols());
    dJ.setFromTriplets(trips.begin(), trips.end());
    J += dJ;
}

void MagneticFieldAnalyzer::precomputeMuTableCache(MuValue& mu_val) {
    mu_val.B_table_cache.clear();
    mu_val.pchip_slopes_cache.clear();
    if (mu_val.type != MuType::TABLE || mu_val.H_table.size() < 2) return;
    const double MU_0 = 4.0 * M_PI * 1e-7;
    mu_val.B_table_cache.resize(mu_val.H_table.size());
    for (size_t i = 0; i < mu_val.H_table.size(); i++) {
        mu_val.B_table_cache[i] = MU_0 * mu_val.mu_table[i] * mu_val.H_table[i];
    }
    mu_val.pchip_slopes_cache = computePCHIPSlopes(mu_val.H_table, mu_val.B_table_cache);
}

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

            // Predefined physical constant: mu0 (vacuum permeability = 4π×10⁻⁷ H/m)
            const double MU_0 = 4.0 * M_PI * 1e-7;
            te_variable mu0_var;
            mu0_var.m_name = "mu0";
            mu0_var.m_value = MU_0;
            vars.insert(mu0_var);

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
                // Deep-saturation extrapolation: B(H) = B_end + μ₀·(H − H_end),
                // i.e. dB/dH → μ₀ ⇒ μ_r(H) = 1 + (B_end/μ₀ − H_end)/H → 1.
                // The previous constant-μ_r extrapolation implied dB/dH =
                // μ_r_end·μ₀ (wrong beyond saturation) and disagreed with
                // interpolateH_from_B, which already extrapolates H(B) with
                // the last-segment slope — the (H(B), μ(H)) pair was
                // inconsistent whenever an iterate drove H past the table end.
                // Users who supplied dmu_r_extrapolation keep the old base
                // value so their derivative spec stays coherent.
                if (mu_val.has_dmu_extrapolation) {
                    return mu_tab.back();
                }
                const double MU_0_loc = 4.0 * M_PI * 1e-7;
                const double H_end = H_tab.back();
                const double B_end = MU_0_loc * mu_tab.back() * H_end;
                return std::max(1.0, 1.0 + (B_end / MU_0_loc - H_end) / H_magnitude);
            }

            // B(H) samples + PCHIP slopes are pure functions of the table, so
            // they are precomputed once at load (precomputeMuTableCache). The
            // in-place recompute below only remains as a fallback for MuValues
            // that never went through the load path; it was measured at ~40%
            // of updateFieldAndMu wall when executed per cell (1.34M DOF).
            double B_interp;
            if (mu_val.B_table_cache.size() == H_tab.size() &&
                mu_val.pchip_slopes_cache.size() == H_tab.size()) {
                B_interp = pchipInterpolate(H_tab, mu_val.B_table_cache,
                                            mu_val.pchip_slopes_cache, H_magnitude);
            } else {
                std::vector<double> B_tab(H_tab.size());
                for (size_t i = 0; i < H_tab.size(); i++) {
                    B_tab[i] = MU_0 * mu_tab[i] * H_tab[i];
                }
                std::vector<double> slopes = computePCHIPSlopes(H_tab, B_tab);
                B_interp = pchipInterpolate(H_tab, B_tab, slopes, H_magnitude);
            }

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
                } else if (H_magnitude >= H_tab.back()) {
                    // Consistent with evaluateMu's deep-saturation extrapolation
                    // μ_r(H) = 1 + (B_end/μ₀ − H_end)/H:
                    //   dμ_r/dH = −(B_end/μ₀ − H_end)/H² (small, negative).
                    // The old default returned +1.0, a dimensionally meaningless
                    // large positive slope that corrupted the diagonal Jacobian
                    // correction whenever an iterate drove H past the table end.
                    const double MU_0_loc = 4.0 * M_PI * 1e-7;
                    const double H_end = H_tab.back();
                    const double B_end = MU_0_loc * mu_tab.back() * H_end;
                    return -(B_end / MU_0_loc - H_end) / (H_magnitude * H_magnitude);
                } else {
                    // Below table start: μ_r is extrapolated as a constant.
                    return 0.0;
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

    // Cumulative co-energy Wc(H_i) = ∫₀^{H_i} B dH (trapezoid; exact for the
    // piecewise-linear B(H) the fast interpolators use). Consumed by
    // coenergyFromTable (energy-objective line search).
    table.Wc_values.resize(table.H_values.size());
    table.Wc_values[0] = 0.0;
    for (size_t i = 1; i < table.H_values.size(); i++) {
        const double dH = table.H_values[i] - table.H_values[i-1];
        table.Wc_values[i] = table.Wc_values[i-1] +
            0.5 * (table.B_values[i] + table.B_values[i-1]) * dH;
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
double MagneticFieldAnalyzer::coenergyFromTable(const BHTable& table, double H) const {
    // Wc(H) = ∫₀^H B dH from the cumulative table (exact for the piecewise-linear
    // B(H) the fast interpolators use). O(log n) per call — the per-cell workhorse
    // of the energy-objective line search (integrateMagneticCoEnergy's 100-point
    // Simpson is reserved for the one-off co-energy export).
    const double MU0 = 4.0 * M_PI * 1e-7;
    if (H <= 0.0) return 0.0;
    if (!table.is_valid || table.H_values.size() < 2 ||
        table.Wc_values.size() != table.H_values.size()) {
        const double mu = table.mu_values.empty() ? MU0 : table.mu_values.front();
        return 0.5 * mu * H * H;
    }
    const auto& Ht = table.H_values;
    const auto& Bt = table.B_values;
    const auto& Wt = table.Wc_values;
    if (H >= Ht.back()) {
        // Deep saturation: B = B_end + μ0·(H − H_end), matching evaluateMu /
        // interpolateH_from_B extrapolation.
        const double dH = H - Ht.back();
        return Wt.back() + Bt.back() * dH + 0.5 * MU0 * dH * dH;
    }
    auto it = std::upper_bound(Ht.begin(), Ht.end(), H);
    const size_t k = static_cast<size_t>(std::distance(Ht.begin(), it)) - 1;
    const double t  = (H - Ht[k]) / (Ht[k+1] - Ht[k]);
    const double Bh = Bt[k] + t * (Bt[k+1] - Bt[k]);
    return Wt[k] + 0.5 * (Bt[k] + Bh) * (H - Ht[k]);
}

double MagneticFieldAnalyzer::computeEnergyObjective() {
    // Discrete magnetostatic energy functional
    //   W(Az) = Σ_cells w(|B|)·vol − Σ_cells (Jz + Jz_mag)·Az·vol,
    // where w(B) = ∫₀^B H db = B·H − Wc(H) for B-H (table) materials and
    // B²/(2μ) for linear ones. Since B(H) is monotone, w is CONVEX, W is convex
    // in Az, and the Picard residual A(μ(Az))·Az − b is (up to discretization
    // consistency) its gradient — so W is the correct line-search merit
    // function: any SPD-preconditioned residual direction is a descent
    // direction for W, and steps that transiently RAISE ||R|| while lowering W
    // are legitimate. Assumes calculateMagneticFieldPolar/calculateHField/
    // updateMuDistribution have run for the current member Az.
    const bool is_polar = (coordinate_system != "cartesian");
    cv::Mat image_to_use;
    cv::flip(image, image_to_use, 0);  // match setupMaterialProperties orientation

    const int n_rows = image_to_use.rows;
    const int n_cols = image_to_use.cols;
    const bool r_horizontal = is_polar && (r_orientation == "horizontal");
    const bool has_jmag = (Jz_mag_map.rows() == n_rows && Jz_mag_map.cols() == n_cols);
    const bool has_jz   = (jz_map.rows()    == n_rows && jz_map.cols()    == n_cols);

    double W = 0.0;
    #pragma omp parallel
    {
        double W_local = 0.0;
        #pragma omp for schedule(static)
        for (int k = 0; k < n_rows * n_cols; k++) {
            const int j = k / n_cols, i = k % n_cols;

            double vol;
            if (is_polar) {
                const double r = r_start + (r_horizontal ? i : j) * dr;
                vol = r * dr * dtheta;
            } else {
                vol = dx * dy;
            }

            double Bx_val, By_val;
            if (is_polar) { Bx_val = Br(j, i); By_val = Btheta(j, i); }
            else          { Bx_val = Bx(j, i); By_val = By(j, i);     }
            const double B_mag = std::sqrt(Bx_val * Bx_val + By_val * By_val);

            // Energy density: table material -> B·H − Wc(H); linear -> B²/(2μ)
            double w;
            const cv::Vec3b px = image_to_use.at<cv::Vec3b>(j, i);
            const int rgb_key = (px[0] << 16) | (px[1] << 8) | px[2];
            auto lut_it = rgb_to_material.find(rgb_key);
            const BHTable* bh = nullptr;
            if (lut_it != rgb_to_material.end()) {
                auto bh_it = material_bh_tables.find(lut_it->second.name);
                if (bh_it != material_bh_tables.end() && bh_it->second.is_valid)
                    bh = &bh_it->second;
            }
            if (bh) {
                const double H_mag = H_map(j, i);
                w = B_mag * H_mag - coenergyFromTable(*bh, H_mag);
            } else {
                const double mu = std::max(mu_map(j, i), 1e-20);
                w = 0.5 * B_mag * B_mag / mu;
            }

            double src = 0.0;
            if (has_jz)   src += jz_map(j, i);
            if (has_jmag) src += Jz_mag_map(j, i);

            W_local += (w - src * Az(j, i)) * vol;
        }
        #pragma omp critical
        W += W_local;
    }
    return W;
}

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

    int h_rows = H_map.rows();
    int h_cols = H_map.cols();
    bool is_cartesian = (coordinate_system == "cartesian");

    // rgb_to_material LUT replaces config["materials"] iteration for thread-safety
    // flat k = j*h_cols+i avoids collapse(2) for MSVC OpenMP 2.0 compatibility
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < h_rows * h_cols; k++) {
        int j = k / h_cols, i = k % h_cols;
        double Bx_val, By_val;

        if (is_cartesian) {
            Bx_val = Bx(j, i);
            By_val = By(j, i);
        } else {
            Bx_val = Br(j, i);
            By_val = Btheta(j, i);
        }

        double B_mag = std::sqrt(Bx_val * Bx_val + By_val * By_val);

        cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(j, i);
        int rgb_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];  // Phase W: image is RGB; LUT keys are R<<16|G<<8|B

        auto lut_it = rgb_to_material.find(rgb_key);
        if (lut_it != rgb_to_material.end()) {
            const std::string& material_name = lut_it->second.name;
            auto bh_it = material_bh_tables.find(material_name);
            if (bh_it != material_bh_tables.end() && bh_it->second.is_valid) {
                H_map(j, i) = interpolateH_from_B(bh_it->second, B_mag);
            } else {
                double mu = mu_map(j, i);
                H_map(j, i) = (mu > 1e-20) ? B_mag / mu : 0.0;
            }
        } else {
            double mu = mu_map(j, i);
            H_map(j, i) = (mu > 1e-20) ? B_mag / mu : 0.0;
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

    int n_rows = image_to_use.rows;
    int n_cols = image_to_use.cols;

    // Phase V: diagnostic — accumulate per-call stats over nonlinear
    // material cells so the verbose log can confirm whether updateMu
    // actually changed the μ distribution between iterations. The cost
    // is one double-precision min/max/mean reduction per call (~5 ms
    // for a 2976×450 grid), negligible vs. AMGCL.
    double H_min = std::numeric_limits<double>::infinity();
    double H_max = -std::numeric_limits<double>::infinity();
    double H_sum = 0.0;
    double mu_r_min = std::numeric_limits<double>::infinity();
    double mu_r_max = -std::numeric_limits<double>::infinity();
    double mu_r_sum = 0.0;
    long long n_nl = 0;
    long long n_changed = 0;

    // rgb_to_material LUT replaces config["materials"] iteration for thread-safety
    // flat k = j*n_cols+i avoids collapse(2) for MSVC OpenMP 2.0 compatibility.
    // MSVC OpenMP 2.0 doesn't support reduction(min:) / reduction(max:); use
    // per-thread accumulators merged via #pragma omp critical instead.
    #pragma omp parallel
    {
        double H_min_local = std::numeric_limits<double>::infinity();
        double H_max_local = -std::numeric_limits<double>::infinity();
        double H_sum_local = 0.0;
        double mu_r_min_local = std::numeric_limits<double>::infinity();
        double mu_r_max_local = -std::numeric_limits<double>::infinity();
        double mu_r_sum_local = 0.0;
        long long n_nl_local = 0;
        long long n_changed_local = 0;

        #pragma omp for schedule(static)
        for (int k = 0; k < n_rows * n_cols; k++) {
            int j = k / n_cols, i = k % n_cols;
            cv::Vec3b pixel = image_to_use.at<cv::Vec3b>(j, i);
            int rgb_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];  // Phase W: image is RGB; LUT keys are R<<16|G<<8|B

            auto lut_it = rgb_to_material.find(rgb_key);
            if (lut_it != rgb_to_material.end()) {
                const std::string& name = lut_it->second.name;
                auto it = material_mu.find(name);

                // Skip linear (STATIC) materials — μ is constant, already set
                if (it != material_mu.end() && it->second.type == MuType::STATIC) {
                    continue;
                }

                double H_mag = H_map(j, i);
                double mu_r = 1.0;

                if (it != material_mu.end()) {
                    mu_r = evaluateMu(it->second, H_mag);
                }

                const double mu_new = mu_r * MU_0;
                const double mu_old = mu_map(j, i);
                mu_map(j, i) = mu_new;

                n_nl_local++;
                if (std::abs(mu_new - mu_old) > 1e-15 * std::abs(mu_old)) n_changed_local++;
                if (H_mag < H_min_local) H_min_local = H_mag;
                if (H_mag > H_max_local) H_max_local = H_mag;
                H_sum_local += H_mag;
                if (mu_r < mu_r_min_local) mu_r_min_local = mu_r;
                if (mu_r > mu_r_max_local) mu_r_max_local = mu_r;
                mu_r_sum_local += mu_r;
            }
        }

        #pragma omp critical
        {
            if (H_min_local < H_min) H_min = H_min_local;
            if (H_max_local > H_max) H_max = H_max_local;
            H_sum += H_sum_local;
            if (mu_r_min_local < mu_r_min) mu_r_min = mu_r_min_local;
            if (mu_r_max_local > mu_r_max) mu_r_max = mu_r_max_local;
            mu_r_sum += mu_r_sum_local;
            n_nl     += n_nl_local;
            n_changed += n_changed_local;
        }
    }

    if (nonlinear_config.verbose && n_nl > 0) {
        std::cout << " [updateMu: NL_cells=" << n_nl
                  << " (changed=" << n_changed << ")"
                  << " H=[" << std::scientific << std::setprecision(2)
                  << H_min << ", " << (H_sum / static_cast<double>(n_nl))
                  << ", " << H_max << "]"
                  << " mu_r=[" << mu_r_min
                  << ", " << (mu_r_sum / static_cast<double>(n_nl))
                  << ", " << mu_r_max << "]]"
                  << std::flush;
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
    last_nonlinear_iterations_ = 0;
    if (!has_nonlinear_materials) {
        // No nonlinear materials, use standard linear solver
        if (coordinate_system == "cartesian") {
            buildAndSolveSystem();
        } else {
            buildAndSolveSystemPolar();
        }
        return;
    }

    if (nonlinear_config.verbose) {
        std::cout << "\n=== Nonlinear Solver (Picard Iteration) ===" << std::endl;
        std::cout << "Max iterations: " << nonlinear_config.max_iterations << std::endl;
        std::cout << "Tolerance: " << nonlinear_config.tolerance << std::endl;
        std::cout << "Relaxation: " << nonlinear_config.relaxation << std::endl;
    }

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

        // Step 2-4: Calculate B, H, update μ with relaxation
        bool using_coarsening = coarsening_enabled &&
            ((coordinate_system == "polar" && n_active_cells < nr * ntheta) ||
             (coordinate_system != "polar" && n_active_cells < nx * ny));

        Eigen::MatrixXd mu_old = mu_map;

        if (using_coarsening) {
            // Phase 8: Wide-stencil B/H/μ at active cells only
            Eigen::VectorXd Az_coarse_for_B(n_active_cells);
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                Az_coarse_for_B(cidx) = Az(cj, ci);
            }
            Eigen::VectorXd Bx_active, By_active, H_active;
            calculateBFieldAtActiveCells(Az_coarse_for_B, Bx_active, By_active);
            calculateHFieldAtActiveCells(Bx_active, By_active, H_active);
            updateMuAtActiveCells(H_active);
            // Relaxation at active cells only
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                mu_map(cj, ci) = OMEGA * mu_map(cj, ci) + (1.0 - OMEGA) * mu_old(cj, ci);
            }
            interpolateMuToFullGrid();
        } else {
            // Full-grid path (unchanged)
            if (coordinate_system == "cartesian") {
                calculateMagneticField();
            } else {
                calculateMagneticFieldPolar();
            }
            calculateHField();
            updateMuDistribution();
            mu_map = OMEGA * mu_map + (1.0 - OMEGA) * mu_old;
        }

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
            last_nonlinear_iterations_ = iter + 1;
            if (nonlinear_config.verbose)
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

    last_nonlinear_iterations_ = MAX_ITER;
    if (nonlinear_config.verbose) {
        std::cerr << "WARNING: Nonlinear solver did not converge after "
                  << MAX_ITER << " iterations!" << std::endl;
        std::cerr << "Final residual: " << residual_history.back() << std::endl;
    }
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
    last_nonlinear_iterations_ = 0;
    if (!has_nonlinear_materials) {
        // No nonlinear materials, use standard linear solver
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

    // Anderson acceleration settings (shared config)
    const int m_AA = nonlinear_config.anderson.depth;
    const double beta_AA = nonlinear_config.anderson.beta;

    if (!nonlinear_config.anderson.enabled || m_AA <= 0) {
        // Anderson disabled, fall back to standard Picard
        solveNonlinear();
        return;
    }

    if (nonlinear_config.verbose) {
        std::cout << "\n=== Nonlinear Solver (Picard + Anderson Acceleration) ===" << std::endl;
        std::cout << "Anderson depth: " << m_AA << ", beta: " << beta_AA << std::endl;
    }

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

        bool using_coarsening = coarsening_enabled &&
            ((coordinate_system == "polar" && n_active_cells < nr * ntheta) ||
             (coordinate_system != "polar" && n_active_cells < nx * ny));

        Eigen::MatrixXd mu_map_before_relax = mu_map;

        if (using_coarsening) {
            // Phase 8: Wide-stencil B/H/μ at active cells only
            Eigen::VectorXd Az_coarse_for_B(n_active_cells);
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                Az_coarse_for_B(cidx) = Az(cj, ci);
            }
            Eigen::VectorXd Bx_active, By_active, H_active;
            calculateBFieldAtActiveCells(Az_coarse_for_B, Bx_active, By_active);
            calculateHFieldAtActiveCells(Bx_active, By_active, H_active);
            updateMuAtActiveCells(H_active);
            // Relaxation at active cells only
            for (int cidx = 0; cidx < n_active_cells; cidx++) {
                auto [ci, cj] = coarse_to_fine[cidx];
                mu_map(cj, ci) = OMEGA * mu_map(cj, ci) + (1.0 - OMEGA) * mu_map_before_relax(cj, ci);
            }
            interpolateMuToFullGrid();
        } else {
            // Full-grid path (unchanged)
            if (coordinate_system == "cartesian") {
                calculateMagneticField();
            } else {
                calculateMagneticFieldPolar();
            }
            calculateHField();
            updateMuDistribution();
            mu_map = OMEGA * mu_map + (1.0 - OMEGA) * mu_map_before_relax;
        }

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
            last_nonlinear_iterations_ = iter + 1;
            if (nonlinear_config.verbose)
                std::cout << "Anderson-accelerated solver converged in " << iter + 1 << " iterations" << std::endl;
            return;
        }

        Az_old = Az_new;
    }

    last_nonlinear_iterations_ = MAX_ITER;
    if (nonlinear_config.verbose)
        std::cerr << "WARNING: Anderson solver did not converge after " << MAX_ITER << " iterations!" << std::endl;
}
