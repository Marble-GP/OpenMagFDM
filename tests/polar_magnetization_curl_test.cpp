#include "PolarMagnetizationCurl.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

class DenseMatrix {
public:
    DenseMatrix(int rows, int cols)
        : rows_(rows), cols_(cols), values_(static_cast<std::size_t>(rows * cols), 0.0) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    void setZero() { std::fill(values_.begin(), values_.end(), 0.0); }

    double& operator()(int row, int col) {
        return values_[static_cast<std::size_t>(row * cols_ + col)];
    }
    double operator()(int row, int col) const {
        return values_[static_cast<std::size_t>(row * cols_ + col)];
    }

private:
    int rows_;
    int cols_;
    std::vector<double> values_;
};

using Options = openmagfdm::detail::PolarMagnetizationCurlOptions;

struct TestRunner {
    int failures = 0;

    void check(bool condition, const std::string& name, double metric, double limit) {
        std::cout << (condition ? "[PASS] " : "[FAIL] ") << name
                  << ": " << std::setprecision(12) << metric
                  << " (limit " << limit << ")\n";
        if (!condition) ++failures;
    }
};

DenseMatrix makeMatrix(const Options& options) {
    return DenseMatrix(
        options.radial_axis_is_columns ? options.ntheta : options.nr,
        options.radial_axis_is_columns ? options.nr : options.ntheta);
}

void setValue(DenseMatrix& matrix, const Options& options, int ir, int jt, double value) {
    const int row = options.radial_axis_is_columns ? jt : ir;
    const int col = options.radial_axis_is_columns ? ir : jt;
    matrix(row, col) = value;
}

double getValue(const DenseMatrix& matrix, const Options& options, int ir, int jt) {
    const int row = options.radial_axis_is_columns ? jt : ir;
    const int col = options.radial_axis_is_columns ? ir : jt;
    return matrix(row, col);
}

Options fullCircleOptions(int ntheta, bool radial_axis_is_columns) {
    Options options;
    options.nr = 9;
    options.ntheta = ntheta;
    options.r_start = 0.04;
    options.dr = 0.0025;
    options.dtheta = 2.0 * PI / ntheta;
    options.theta_offset = 0.37;
    options.radial_axis_is_columns = radial_axis_is_columns;
    options.theta_periodic = true;
    return options;
}

double tangentialFieldError(bool radial_axis_is_columns) {
    const Options options = fullCircleOptions(128, radial_axis_is_columns);
    DenseMatrix mx = makeMatrix(options);
    DenseMatrix my = makeMatrix(options);
    constexpr double magnitude = 1.7;

    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double theta = options.theta_offset + jt * options.dtheta;
        for (int ir = 0; ir < options.nr; ++ir) {
            setValue(mx, options, ir, jt, -magnitude * std::sin(theta));
            setValue(my, options, ir, jt,  magnitude * std::cos(theta));
        }
    }

    const DenseMatrix curl = openmagfdm::detail::computePolarMagnetizationCurl(
        mx, my, options);
    double max_relative_error = 0.0;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            const double expected = magnitude / r;
            max_relative_error = std::max(
                max_relative_error,
                std::abs(getValue(curl, options, ir, jt) - expected) / expected);
        }
    }
    return max_relative_error;
}

double uniformCartesianError(int ntheta) {
    const Options options = fullCircleOptions(ntheta, true);
    DenseMatrix mx = makeMatrix(options);
    DenseMatrix my = makeMatrix(options);
    constexpr double mx_value = 0.8;
    constexpr double my_value = -0.3;
    const double magnitude = std::hypot(mx_value, my_value);

    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 0; ir < options.nr; ++ir) {
            setValue(mx, options, ir, jt, mx_value);
            setValue(my, options, ir, jt, my_value);
        }
    }

    const DenseMatrix curl = openmagfdm::detail::computePolarMagnetizationCurl(
        mx, my, options);
    double max_normalized_error = 0.0;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            max_normalized_error = std::max(
                max_normalized_error,
                std::abs(getValue(curl, options, ir, jt)) * r / magnitude);
        }
    }
    return max_normalized_error;
}

double antiPeriodicSectorError(bool radial_axis_is_columns) {
    Options options;
    options.nr = 7;
    options.ntheta = 128;
    options.r_start = 0.03;
    options.dr = 0.003;
    options.dtheta = PI / options.ntheta;
    options.theta_offset = 0.41;
    options.radial_axis_is_columns = radial_axis_is_columns;
    options.theta_periodic = true;
    options.theta_antiperiodic = true;

    DenseMatrix mx = makeMatrix(options);
    DenseMatrix my = makeMatrix(options);
    constexpr double magnitude = 2.3;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double local_theta = jt * options.dtheta;
        const double global_theta = options.theta_offset + local_theta;
        const double mr = magnitude * std::cos(local_theta);
        for (int ir = 0; ir < options.nr; ++ir) {
            setValue(mx, options, ir, jt, mr * std::cos(global_theta));
            setValue(my, options, ir, jt, mr * std::sin(global_theta));
        }
    }

    const DenseMatrix curl = openmagfdm::detail::computePolarMagnetizationCurl(
        mx, my, options);
    double max_normalized_error = 0.0;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double local_theta = jt * options.dtheta;
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            const double expected = magnitude * std::sin(local_theta) / r;
            max_normalized_error = std::max(
                max_normalized_error,
                std::abs(getValue(curl, options, ir, jt) - expected) * r / magnitude);
        }
    }
    return max_normalized_error;
}

double radialDerivativeError() {
    const Options options = fullCircleOptions(96, true);
    DenseMatrix mx = makeMatrix(options);
    DenseMatrix my = makeMatrix(options);
    constexpr double slope = 2.7;

    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double theta = options.theta_offset + jt * options.dtheta;
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            const double mtheta = slope * r;
            setValue(mx, options, ir, jt, -mtheta * std::sin(theta));
            setValue(my, options, ir, jt,  mtheta * std::cos(theta));
        }
    }

    const DenseMatrix curl = openmagfdm::detail::computePolarMagnetizationCurl(
        mx, my, options);
    double max_error = 0.0;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 1; ir < options.nr - 1; ++ir) {
            max_error = std::max(
                max_error,
                std::abs(getValue(curl, options, ir, jt) - 2.0 * slope));
        }
    }
    return max_error;
}

double nonPeriodicOneSidedError() {
    Options options;
    options.nr = 6;
    options.ntheta = 41;
    options.r_start = 0.05;
    options.dr = 0.004;
    options.dtheta = (PI / 3.0) / (options.ntheta - 1);
    options.theta_offset = 0.29;
    options.radial_axis_is_columns = false;

    DenseMatrix mx = makeMatrix(options);
    DenseMatrix my = makeMatrix(options);
    constexpr double slope = 1.4;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        const double local_theta = jt * options.dtheta;
        const double global_theta = options.theta_offset + local_theta;
        const double mr = slope * local_theta;
        for (int ir = 0; ir < options.nr; ++ir) {
            setValue(mx, options, ir, jt, mr * std::cos(global_theta));
            setValue(my, options, ir, jt, mr * std::sin(global_theta));
        }
    }

    const DenseMatrix curl = openmagfdm::detail::computePolarMagnetizationCurl(
        mx, my, options);
    double max_error = 0.0;
    for (int jt = 0; jt < options.ntheta; ++jt) {
        for (int ir = 0; ir < options.nr; ++ir) {
            const double r = options.r_start + ir * options.dr;
            max_error = std::max(
                max_error,
                std::abs(getValue(curl, options, ir, jt) + slope / r));
        }
    }
    return max_error;
}

}  // namespace

int main() {
    TestRunner runner;

    const double tangential_horizontal = tangentialFieldError(true);
    const double tangential_vertical = tangentialFieldError(false);
    runner.check(
        tangential_horizontal < 2e-12,
        "tangential field, horizontal storage",
        tangential_horizontal, 2e-12);
    runner.check(
        tangential_vertical < 2e-12,
        "tangential field, vertical storage",
        tangential_vertical, 2e-12);

    const double coarse_uniform_error = uniformCartesianError(96);
    const double fine_uniform_error = uniformCartesianError(192);
    const bool uniform_errors_at_roundoff =
        coarse_uniform_error < 1e-12 && fine_uniform_error < 1e-12;
    const double convergence_ratio = fine_uniform_error > 1e-15
        ? coarse_uniform_error / fine_uniform_error
        : (uniform_errors_at_roundoff ? 4.0 : 0.0);
    runner.check(
        fine_uniform_error < 2.0e-4,
        "uniform Cartesian M has no O(M/r) fictitious curl",
        fine_uniform_error, 2.0e-4);
    runner.check(
        uniform_errors_at_roundoff
            || (convergence_ratio > 3.8 && convergence_ratio < 4.2),
        "uniform Cartesian M angular error converges at second order",
        convergence_ratio, 4.2);

    const double anti_horizontal = antiPeriodicSectorError(true);
    const double anti_vertical = antiPeriodicSectorError(false);
    runner.check(
        anti_horizontal < 1.1e-4,
        "anti-periodic theta seam, horizontal storage",
        anti_horizontal, 1.1e-4);
    runner.check(
        anti_vertical < 1.1e-4,
        "anti-periodic theta seam, vertical storage",
        anti_vertical, 1.1e-4);

    const double radial_error = radialDerivativeError();
    runner.check(
        radial_error < 2e-12,
        "radial derivative for Mtheta=a*r",
        radial_error, 2e-12);

    const double one_sided_error = nonPeriodicOneSidedError();
    runner.check(
        one_sided_error < 2e-12,
        "non-periodic theta uses one-sided endpoint differences",
        one_sided_error, 2e-12);

    if (runner.failures != 0) {
        std::cerr << runner.failures << " polar magnetization curl test(s) failed\n";
        return 1;
    }
    std::cout << "All polar magnetization curl tests passed\n";
    return 0;
}
