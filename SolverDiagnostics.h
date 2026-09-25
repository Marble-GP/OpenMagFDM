#pragma once

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>

namespace omfdm {
using ProfileClock = std::chrono::steady_clock;
inline double milliseconds(ProfileClock::time_point a, ProfileClock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}
inline bool profilingEnabled() {
    return std::getenv("NK_PROF") || std::getenv("OMFDM_PROFILE");
}
struct LinearSolveCounters {
    double conversion_ms = 0, setup_ms = 0, solve_ms = 0;
    long long iterations = 0, solves = 0;
    double material_prepare_ms = 0, tangent_rows_ms = 0, fallback_build_ms = 0;
    double workspace_ms = 0, preconditioner_ms = 0;
    long long workspace_allocations = 0, workspace_reuses = 0, fallbacks = 0;
    long long field_reuses = 0, field_invalidations = 0, field_updates = 0;
};

// Declared before iteration workspaces so even early convergence and cleanup
// are timed. Counters belong to each analyzer (never shared between chunks).
struct IterationProfile {
    bool enabled = profilingEnabled();
    int iteration;
    const LinearSolveCounters& counters;
    LinearSolveCounters before;
    ProfileClock::time_point start = ProfileClock::now();
    double mu_ms = 0, build_ms = 0, residual_ms = 0, jacobian_ms = 0;
    double linear_ms = 0, line_search_ms = 0, anderson_ms = 0;
    double residual = 0;
    int ls_trials = 0, aa_evaluations = 0;
    const char* aa = "inactive";
    const char* outcome = "update";
    IterationProfile(int i, const LinearSolveCounters& c)
        : iteration(i), counters(c), before(c) {}
    ~IterationProfile() {
        if (!enabled) return;
        // One insertion avoids interleaved field-level writes between chunks.
        std::ostringstream s;
        s.precision(12);
        s << "\n[NKPROF] iter=" << iteration
          << " mu_ms=" << mu_ms << " build_ms=" << build_ms
          << " residual_ms=" << residual_ms << " jacobian_ms=" << jacobian_ms
          << " linear_ms=" << linear_ms
          << " conversion_ms=" << counters.conversion_ms - before.conversion_ms
          << " setup_ms=" << counters.setup_ms - before.setup_ms
          << " krylov_ms=" << counters.solve_ms - before.solve_ms
          << " inner_iters=" << counters.iterations - before.iterations
          << " linear_solves=" << counters.solves - before.solves
          << " material_prepare_ms=" << counters.material_prepare_ms - before.material_prepare_ms
          << " tangent_rows_ms=" << counters.tangent_rows_ms - before.tangent_rows_ms
          << " fallback_build_ms=" << counters.fallback_build_ms - before.fallback_build_ms
          << " workspace_ms=" << counters.workspace_ms - before.workspace_ms
          << " preconditioner_ms=" << counters.preconditioner_ms - before.preconditioner_ms
          << " workspace_allocations=" << counters.workspace_allocations - before.workspace_allocations
          << " workspace_reuses=" << counters.workspace_reuses - before.workspace_reuses
          << " fallbacks=" << counters.fallbacks - before.fallbacks
          << " field_reuses=" << counters.field_reuses - before.field_reuses
          << " field_invalidations=" << counters.field_invalidations - before.field_invalidations
          << " field_updates=" << counters.field_updates - before.field_updates
          << " ls_ms=" << line_search_ms << " ls_trials=" << ls_trials
          << " aa_ms=" << anderson_ms << " aa_evaluations=" << aa_evaluations
          << " aa=" << aa << " residual=" << residual << " outcome=" << outcome
          << " total_ms=" << milliseconds(start, ProfileClock::now()) << '\n';
        std::cout << s.str();
    }
};
} // namespace omfdm
