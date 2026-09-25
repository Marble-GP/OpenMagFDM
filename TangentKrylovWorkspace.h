#pragma once
#include <amgcl/backend/builtin.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <memory>
#include <vector>

namespace omfdm {
// Only storage is reused. AMGCL overwrites the active Arnoldi basis/Hessenberg
// on every call; x is reset explicitly, including after zero-RHS calls.
// Never contains an operator view or numerical preconditioner.
struct TangentKrylovWorkspace {
    using Solver = amgcl::solver::fgmres<amgcl::backend::builtin<double>>;
    std::unique_ptr<Solver> solver;
    std::vector<double> b, x;
    bool prepare(size_t n, double tolerance) {
        const bool allocate = !solver || x.size() != n;
        if (allocate) {
            clear();
            Solver::params params;
            params.M = 20; params.maxiter = 100; params.tol = tolerance;
            solver = std::make_unique<Solver>(n, params);
            b.resize(n); x.resize(n);
        }
        solver->prm.tol = tolerance;
        std::fill(x.begin(), x.end(), 0.0);
        return allocate;
    }
    void clear() {
        solver.reset();
        std::vector<double>().swap(b);
        std::vector<double>().swap(x);
    }
};
} // namespace omfdm
