#include "MagneticFieldAnalyzer.h"
#include "PolarGridAccess.h"
#include "SparseDifferential.h"
#include "MagneticFieldAnalyzer_stencil.inl"
#include <amgcl/adapter/eigen.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <sstream>

void MagneticFieldAnalyzer::buildConsistentTangent(RowMatrix& J, const Eigen::VectorXd& state) {
    using D = omfdm::SparseDifferential;
    const bool polar = coordinate_system == "polar";
    const bool horizontal = r_orientation == "horizontal";
    const int n = polar ? nr * ntheta : nx * ny;
    const double mu0 = 4.0 * M_PI * 1e-7;
    if (state.size() != n) throw std::invalid_argument("tangent state/grid size mismatch");
    // Matching B/H/mu must already have been evaluated by the NK caller.
    // Only the scalar chain derivative is stored; no dense AD tape or fixed
    // multi-million-triplet reservation survives an iteration.
    Eigen::VectorXd dmu_dB = Eigen::VectorXd::Zero(n);
    cv::Mat material_image;
    cv::flip(image, material_image, 0);
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const int i = polar ? k / ntheta : k % nx;
        const int j = polar ? k % ntheta : k / nx;
        const int row = polar && !horizontal ? i : j;
        const int col = polar && !horizontal ? j : i;
        const auto pixel = material_image.at<cv::Vec3b>(row, col);
        const int rgb = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];
        const auto found = rgb_to_material.find(rgb);
        if (found == rgb_to_material.end()) continue;
        const auto material = material_mu.find(found->second.name);
        if (material == material_mu.end() || material->second.type == MuType::STATIC) continue;
        const auto table = material_bh_tables.find(found->second.name);
        const double b1 = polar ? Br(row, col) : Bx(row, col);
        const double b2 = polar ? Btheta(row, col) : By(row, col);
        const double B = std::sqrt(b1*b1 + b2*b2);
        const double mu = mu_map(row, col);
        const double dH = table != material_bh_tables.end() && table->second.is_valid
            ? inverseBHDerivative(table->second, B) : (mu > 1e-20 ? 1.0 / mu : 0.0);
        dmu_dB[k] = mu0 * evaluateMuValueDerivative(material->second, H_map(row, col)) * dH;
    }
    if (!dmu_dB.allFinite()) throw std::runtime_error("nonfinite material derivative in tangent");
    const bool theta_periodic = bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic";
    const bool anti = theta_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    const bool xp = bc_left.type == "periodic" && bc_right.type == "periodic";
    const bool yp = bc_bottom.type == "periodic" && bc_top.type == "periodic";
    auto materialAt = [&](int i, int j) {
        const int k = polar ? i * ntheta + j : j * nx + i;
        const int row = polar && !horizontal ? i : j;
        const int col = polar && !horizontal ? j : i;
        D m(mu_map(row, col));
        const double b1 = polar ? Br(row, col) : Bx(row, col);
        const double b2 = polar ? Btheta(row, col) : By(row, col);
        const double B = std::sqrt(b1*b1 + b2*b2);
        const double scale = B > 0 ? dmu_dB[k] / B : 0.0;
        // Keep zero derivatives structurally: saturation, B=0, a sliding
        // material interface or a clamp must not change the cached graph.
        if (polar) {
            int prev = j-1, next = j+1;
            double dt = 2*dtheta, sp = 1, sn = 1;
            if (theta_periodic) {
                prev = (j-1+ntheta) % ntheta; next = (j+1) % ntheta;
                if (anti && j == 0) sp = -1;
                if (anti && j == ntheta-1) sn = -1;
            } else if (j == 0) { prev = 0; next = 1; dt = dtheta; }
            else if (j == ntheta-1) { prev = j-1; next = j; dt = dtheta; }
            const double safe_r = std::max(r_coords[i], 0.5*dr);
            m.add(i*ntheta+next, scale*b1*sn/(safe_r*dt));
            m.add(i*ntheta+prev, -scale*b1*sp/(safe_r*dt));
            const int lo = std::max(0, i-1), hi = std::min(nr-1, i+1);
            const double rd = (i == 0 || i == nr-1) ? dr : 2*dr;
            m.add(hi*ntheta+j, -scale*b2/rd);
            m.add(lo*ntheta+j, scale*b2/rd);
        } else {
            int lo = std::max(0, j-1), hi = std::min(ny-1, j+1);
            double yd = (j == 0 || j == ny-1) && !yp ? dy : 2*dy;
            if (yp) { lo = (j-1+ny)%ny; hi = (j+1)%ny; }
            m.add(hi*nx+i, scale*b1/yd); m.add(lo*nx+i, -scale*b1/yd);
            lo = std::max(0, i-1); hi = std::min(nx-1, i+1);
            double xd = (i == 0 || i == nx-1) && !xp ? dx : 2*dx;
            if (xp) { lo = (i-1+nx)%nx; hi = (i+1)%nx; }
            m.add(j*nx+hi, -scale*b2/xd); m.add(j*nx+lo, scale*b2/xd);
        }
        return m;
    };
    auto derivativeRow = [&](int k) {
        const auto row = polar ? polarRow<D>(k, materialAt) : cartesianRow<D>(k, materialAt);
        D R = -row.rhs; // BC elimination/profile terms also depend on material.
        for (int q = 0; q < row.count; ++q) {
            R += row.value[q] * state[row.column[q]];
            R.add(row.column[q], row.value[q].value);
        }
        return R;
    };
    // The secant assembler refreshes this key before the tangent is requested.
    const bool rebuild = J.rows() != n || tangent_pattern_key_ != stencil_pattern_key_;
    if (rebuild) {
        std::vector<int> offsets(n + 1, 0);
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; ++k) offsets[k+1] = derivativeRow(k).count;
        for (int k = 0; k < n; ++k) offsets[k+1] += offsets[k];
        J.resize(n, n); J.makeCompressed(); J.resizeNonZeros(offsets[n]);
        std::copy(offsets.begin(), offsets.end(), J.outerIndexPtr());
        tangent_pattern_key_ = stencil_pattern_key_;
    }
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < n; ++k) {
        const auto r = derivativeRow(k);
        const int offset = J.outerIndexPtr()[k];
        assert(J.outerIndexPtr()[k+1] - offset == r.count);
        for (int q = 0; q < r.count; ++q) {
            J.innerIndexPtr()[offset+q] = r.column[q];
            J.valuePtr()[offset+q] = r.derivative[q];
        }
    }
}

bool MagneticFieldAnalyzer::solveTangentSystem(const RowMatrix& J, const RowMatrix& secant,
    const Eigen::VectorXd& rhs, double tolerance, Eigen::VectorXd& direction) {
    using Backend = amgcl::backend::builtin<double>;
    using AMG = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;
    using ILUT = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilut>;
    using Solver = amgcl::solver::fgmres<Backend>;
    const char* env = std::getenv("OMFDM_TANGENT_PRECONDITIONER");
    const std::string candidate = env ? env : "secant-amg";
    if (candidate != "secant-amg" && candidate != "tangent-amg" && candidate != "tangent-ilut")
        throw std::invalid_argument("OMFDM_TANGENT_PRECONDITIONER must be secant-amg, tangent-amg or tangent-ilut");
    const double tol = tolerance > 0 ? tolerance : SOLVER_TOLERANCE;
    if (!rhs.allFinite() || !Eigen::Map<const Eigen::VectorXd>(J.valuePtr(), J.nonZeros()).allFinite())
        return false;
    const auto conversion_start = omfdm::ProfileClock::now();
    Backend::matrix operator_matrix(J);
    std::vector<double> b(rhs.data(), rhs.data() + rhs.size()), x(rhs.size(), 0.0);
    const auto setup_start = omfdm::ProfileClock::now();
    linear_profile_.conversion_ms += omfdm::milliseconds(conversion_start, setup_start);
    Solver::params params;
    params.M = 20; params.maxiter = 100; params.tol = tol;
    Solver solver(rhs.size(), params);
    size_t iterations = 0;
    double reported = std::numeric_limits<double>::infinity();
    auto apply = [&](const auto& P) {
        const auto solve_start = omfdm::ProfileClock::now();
        linear_profile_.setup_ms += omfdm::milliseconds(setup_start, solve_start);
        // P.apply is one AMG cycle / ILUT application, not an inner CG solve.
        // The same numerical P is used throughout this outer iteration.
        std::tie(iterations, reported) = solver(operator_matrix, P, b, x);
        linear_profile_.solve_ms += omfdm::milliseconds(solve_start, omfdm::ProfileClock::now());
    };
    try {
        if (candidate == "tangent-ilut") {
            ILUT::params p; p.p = 2; p.tau = 1e-2;
            ILUT P(J, p); apply(P);
        } else {
            AMG P(candidate == "tangent-amg" ? J : secant); apply(P);
        }
    } catch (const std::runtime_error& e) {
        if (nonlinear_config.verbose) std::cout << " [tangent setup/solve rejected: " << e.what() << "]";
        return false;
    }
    linear_profile_.iterations += iterations; ++linear_profile_.solves;
    total_linear_iters_ += static_cast<long>(iterations); ++num_linear_solves_;
    direction = Eigen::Map<Eigen::VectorXd>(x.data(), static_cast<Eigen::Index>(x.size()));
    const double true_error = (J * direction - rhs).norm() / std::max(rhs.norm(), 1e-30);
    const bool accepted = direction.allFinite() && std::isfinite(reported) && std::isfinite(true_error)
                       && iterations < params.maxiter && reported <= tol && true_error <= tol;
    if (nonlinear_config.verbose || omfdm::profilingEnabled()) {
        std::ostringstream s;
        s << "\n[TANGENT] candidate=" << candidate << " iterations=" << iterations
          << " reported=" << reported << " true_residual=" << true_error << " tolerance=" << tol
          << " accepted=" << accepted << '\n';
        std::cout << s.str();
    }
    return accepted;
}
