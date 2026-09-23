#include "MagneticFieldAnalyzer.h"
#include "PolarGridAccess.h"
#include "MagneticFieldAnalyzer_stencil.inl"
#include <stdexcept>

void MagneticFieldAnalyzer::buildRowSystem(RowMatrix& A, Eigen::VectorXd& rhs, bool polar) {
    const int n = polar ? nr * ntheta : nx * ny;
    auto rowAt = [&](int k) {
        if (polar) return polarRow<double>(k, [&](int i, int j) {
            return getMuPolar(mu_map, i, j, r_orientation);
        });
        return cartesianRow<double>(k, [&](int i, int j) { return muAtGrid(i, j); });
    };
    // Numeric BC/profile values, sources and materials are deliberately NOT
    // cached. They are read on every update, including after slide/rollback.
    const std::string key = polar
        ? "p:" + std::to_string(nr) + ":" + std::to_string(ntheta) + ":" + r_orientation
          + ":" + bc_inner.type + ":" + bc_outer.type
          + ":" + bc_theta_min.type + ":" + bc_theta_max.type
        : "c:" + std::to_string(nx) + ":" + std::to_string(ny)
          + ":" + bc_left.type + ":" + bc_right.type
          + ":" + bc_bottom.type + ":" + bc_top.type;
    const bool rebuild = key != stencil_pattern_key_;
    if (rebuild) {
        std::vector<int> offsets(n + 1, 0);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) offsets[k + 1] = rowAt(k).count;
        for (int k = 0; k < n; ++k) offsets[k + 1] += offsets[k];
        stencil_pattern_.resize(n, n);
        stencil_pattern_.makeCompressed();
        stencil_pattern_.resizeNonZeros(offsets[n]);
        std::copy(offsets.begin(), offsets.end(), stencil_pattern_.outerIndexPtr());
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) {
            const auto row = rowAt(k);
            for (int q = 0; q < row.count; ++q) {
                const int p = offsets[k] + q;
                stencil_pattern_.innerIndexPtr()[p] = row.column[q];
                stencil_pattern_.valuePtr()[p] = 0.0;
            }
        }
        stencil_pattern_key_ = key;
        // A same-size BC change may alter SparseLU's pattern as well.
        nl_solver_pattern_valid_ = false;
    }
    if (rebuild || A.rows() != n || A.cols() != n || A.nonZeros() != stencil_pattern_.nonZeros())
        A = stencil_pattern_;
    rhs.resize(n);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const auto row = rowAt(k);
        rhs[k] = row.rhs;
        const int begin = A.outerIndexPtr()[k];
        for (int q = 0; q < row.count; ++q) A.valuePtr()[begin + q] = row.value[q];
    }
}

void MagneticFieldAnalyzer::evaluateDiscreteResidual(
    const Eigen::VectorXd& state, Eigen::VectorXd& residual, Eigen::VectorXd& rhs,
    bool polar) const {
    const int n = polar ? nr * ntheta : nx * ny;
    if (state.size() != n) throw std::invalid_argument("residual state/grid size mismatch");
    residual.resize(n);
    rhs.resize(n);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const auto row = polar
            ? polarRow<double>(k, [&](int i, int j) { return getMuPolar(mu_map, i, j, r_orientation); })
            : cartesianRow<double>(k, [&](int i, int j) { return muAtGrid(i, j); });
        double ax = 0.0;
        for (int q = 0; q < row.count; ++q) ax += row.value[q] * state[row.column[q]];
        rhs[k] = row.rhs;
        residual[k] = ax - row.rhs;
    }
}
