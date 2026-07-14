#ifndef OPENMAGFDM_POLAR_MAGNETIZATION_CURL_H
#define OPENMAGFDM_POLAR_MAGNETIZATION_CURL_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace openmagfdm::detail {

struct PolarMagnetizationCurlOptions {
    int nr = 0;
    int ntheta = 0;
    double r_start = 0.0;
    double dr = 0.0;
    double dtheta = 0.0;
    double theta_offset = 0.0;
    bool radial_axis_is_columns = true;
    bool theta_periodic = false;
    bool theta_antiperiodic = false;
};

// Return curl(M)_z on a polar grid when M is stored in global Cartesian
// components. Matrix only needs Eigen-like rows(), cols(), setZero(), and
// operator()(row, col), which keeps this numerical kernel independently
// testable without constructing a full MagneticFieldAnalyzer.
template <typename Matrix>
Matrix computePolarMagnetizationCurl(
    const Matrix& mx,
    const Matrix& my,
    const PolarMagnetizationCurlOptions& options) {

    if (options.nr < 2 || options.ntheta < 2) {
        throw std::invalid_argument("Polar magnetization curl requires nr >= 2 and ntheta >= 2");
    }
    if (!(options.dr > 0.0) || !(options.dtheta > 0.0)) {
        throw std::invalid_argument("Polar magnetization curl requires positive dr and dtheta");
    }
    if (options.theta_antiperiodic && !options.theta_periodic) {
        throw std::invalid_argument("Anti-periodic theta curl also requires periodic theta indexing");
    }

    const int expected_rows = options.radial_axis_is_columns
        ? options.ntheta : options.nr;
    const int expected_cols = options.radial_axis_is_columns
        ? options.nr : options.ntheta;
    if (mx.rows() != expected_rows || mx.cols() != expected_cols
        || my.rows() != expected_rows || my.cols() != expected_cols) {
        throw std::invalid_argument("Magnetization matrix shape does not match the polar grid");
    }

    Matrix curl(expected_rows, expected_cols);
    curl.setZero();

    std::vector<double> cos_theta(options.ntheta);
    std::vector<double> sin_theta(options.ntheta);
    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double theta = options.theta_offset + jt * options.dtheta;
        cos_theta[jt] = std::cos(theta);
        sin_theta[jt] = std::sin(theta);
    }

    auto read = [&](const Matrix& matrix, int ir, int jt) -> double {
        const int row = options.radial_axis_is_columns ? jt : ir;
        const int col = options.radial_axis_is_columns ? ir : jt;
        return matrix(row, col);
    };
    auto write = [&](Matrix& matrix, int ir, int jt, double value) {
        const int row = options.radial_axis_is_columns ? jt : ir;
        const int col = options.radial_axis_is_columns ? ir : jt;
        matrix(row, col) = value;
    };

    struct ThetaSample {
        int index;
        double sign;
    };
    auto normalizeTheta = [&](int raw_jt) -> ThetaSample {
        int jt = raw_jt;
        double sign = 1.0;
        if (options.theta_periodic) {
            while (jt < 0) {
                jt += options.ntheta;
                if (options.theta_antiperiodic) sign = -sign;
            }
            while (jt >= options.ntheta) {
                jt -= options.ntheta;
                if (options.theta_antiperiodic) sign = -sign;
            }
        } else {
            jt = std::max(0, std::min(options.ntheta - 1, jt));
        }
        return {jt, sign};
    };

    // Crucially, each sample is projected using that sample's own theta
    // basis. Projecting theta-neighbours with the centre cell's basis creates
    // an O(M/r) fictitious volume current even for uniform Cartesian M.
    auto getMr = [&](int ir, int raw_jt) -> double {
        const ThetaSample sample = normalizeTheta(raw_jt);
        const double value = read(mx, ir, sample.index) * cos_theta[sample.index]
                           + read(my, ir, sample.index) * sin_theta[sample.index];
        return sample.sign * value;
    };
    auto getMtheta = [&](int ir, int raw_jt) -> double {
        const ThetaSample sample = normalizeTheta(raw_jt);
        const double value = -read(mx, ir, sample.index) * sin_theta[sample.index]
                            + read(my, ir, sample.index) * cos_theta[sample.index];
        return sample.sign * value;
    };

    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            if (r <= 1e-10) {
                // The axis is an identity row under the required inner
                // Dirichlet condition, so no singular 1/r source is used.
                continue;
            }

            const int ir_m = std::max(ir - 1, 0);
            const int ir_p = std::min(ir + 1, options.nr - 1);
            const double r_m = options.r_start + ir_m * options.dr;
            const double r_p = options.r_start + ir_p * options.dr;
            const double denom_r = (ir_p - ir_m) * options.dr;

            double d_rMtheta_dr = 0.0;
            if (denom_r > 1e-15) {
                d_rMtheta_dr = (r_p * getMtheta(ir_p, jt)
                               - r_m * getMtheta(ir_m, jt)) / denom_r;
            }

            double dMr_dtheta = 0.0;
            if (options.theta_periodic) {
                dMr_dtheta = (getMr(ir, jt + 1) - getMr(ir, jt - 1))
                            / (2.0 * options.dtheta);
            } else {
                const int jt_m = std::max(jt - 1, 0);
                const int jt_p = std::min(jt + 1, options.ntheta - 1);
                const double denom_theta = (jt_p - jt_m) * options.dtheta;
                if (denom_theta > 1e-15) {
                    dMr_dtheta = (getMr(ir, jt_p) - getMr(ir, jt_m))
                                / denom_theta;
                }
            }

            write(curl, ir, jt, d_rMtheta_dr / r - dMr_dtheta / r);
        }
    }

    return curl;
}

}  // namespace openmagfdm::detail

#endif  // OPENMAGFDM_POLAR_MAGNETIZATION_CURL_H
