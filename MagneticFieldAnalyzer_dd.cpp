// MagneticFieldAnalyzer_dd.cpp
// =============================================================================
// v1.6 domain decomposition (optimized Schwarz) -- PRODUCTION solve path.
//
// Opt-in, polar only. Two complementary modes:
//
// 1) VARIABLE-RESOLUTION accuracy mode (the original banded path): radial bands,
//    each solved on its OWN uniform grid (coarsened by cf_r x cf_theta where the
//    field is smooth, kept fine across the air gap / saturated zone), coupled by
//    symmetric Robin transmission iterated to global consistency.
//
// 2) PARALLELIZATION mode (band spec 5th element = theta sectors + parallel:
//    true): FINE (cf=1) bands are split into theta-sector patches at material-
//    safe cuts (theta_cuts; must pass through iron, never coils/magnets — and
//    the air gap must stay inside a full-theta ring band, never theta-cut).
//    Sector patches are exact-resolution (write-back is a plain copy, no
//    mortar). With parallel: true the sweep runs ADDITIVE Schwarz: every patch
//    reads the sweep-start snapshot and the patch loop is OpenMP-parallel with
//    single-threaded inner solves. Small patches are cache-resident, so this
//    scales far better than AMGCL's memory-bandwidth-bound intra-solve
//    threading (dd_bench study: 7.7x vs 3.8x on 24 cores).
//
// Both modes use FLATTENED sub-solves (max_inner NK iterations per patch per
// sweep + a final uncapped polish sweep) — nesting full NK solves inside every
// sweep multiplies the two iteration counts (measured 315 s vs 37 s monolithic).
//
// The converged composite is written into the member Az; downstream stress /
// energy / export are unchanged. Enable via the `domain_decomposition` YAML
// block.
// =============================================================================
#include "MagneticFieldAnalyzer.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

void MagneticFieldAnalyzer::solveDomainDecomposition() {
    const int    NTH   = ntheta;
    const int    NR    = nr;
    const double DR    = dr;
    const double DTH   = dtheta;
    const double RS    = r_start;
    const double alpha = dd_config.robin_p;   // radial Robin coefficient (alpha*Az + beta*dAz/dn = gamma)
    // Theta-interface Robin coefficient: the theta face coupling c=1/(r*mu*dth)
    // is ~40x the radial c=r/(mu*dr) on this geometry class, so the optimal
    // theta alpha is ~alpha/40 (dd_bench S3 tuning). robin_p_theta overrides.
    const double alpha_th = (dd_config.robin_p_theta > 0.0)
                              ? dd_config.robin_p_theta : alpha / 40.0;
    const int    ov    = dd_config.overlap;
    const int    maxo  = dd_config.max_outer;
    const double tol   = dd_config.tol;
    const double omega = dd_config.relax;
    const bool   PAR   = dd_config.parallel;

    std::cout << "\n=== Domain decomposition (optimized Schwarz, polar) ===" << std::endl;
    std::cout << "  " << dd_config.bands.size() << " band spec(s) | robin_p=" << alpha
              << " robin_p_theta=" << alpha_th
              << " overlap=" << ov << " max_outer=" << maxo
              << " tol=" << tol << " relax=" << omega
              << (PAR ? " | PARALLEL (additive Schwarz + OMP patches)" : "") << std::endl;

    // Reload the RAW base config so each sub-domain re-parses variables / materials
    // exactly as a standalone run would (mirrors the validated harness path).
    YAML::Node base = YAML::LoadFile(config_path);

    // Per-patch sub-analyzer. Full-theta rings (ft=true) may be coarsened
    // (cf_r x cf_theta on their own uniform grid); theta-sector patches are
    // always FINE (cf=1) and exact-resolution.
    struct Band {
        int c0, c1;          // radial core column range [c0, c1)
        int cfr, cft;        // radial / theta coarsen factors (1/1 for sectors)
        int er0, er1;        // extended (overlapped) radial column range solved
        int et0, et1;        // extended global-theta row range (sectors; may wrap)
        int ct0, ct1;        // theta core range [ct0, ct1) (sectors; ct1 may exceed NTH)
        bool ft;             // full-theta ring (periodic theta)
        bool gap_out = false;  // outer edge sits on the analytic gap circle cR
        bool gap_in  = false;  // inner edge sits on the analytic gap circle cS
        int nth, nrb;        // band grid size (theta, radial)
        std::unique_ptr<MagneticFieldAnalyzer> an;
    };
    std::vector<Band> B;
    { std::error_code ec; std::filesystem::create_directories("dd_tmp", ec); }  // portable (Windows + POSIX)
    long agg = 0;
    int bid = 0;
    const auto wrapTh = [NTH](int gt) { return ((gt % NTH) + NTH) % NTH; };
    // Grid theta g <-> image row: the solver flips the image vertically, so
    // image row = NTH-1-g (theta-sector crops must apply this; full-theta ring
    // crops keep raw rows because the sub-analyzer's internal flip already
    // aligns ring grid g with composite grid g).
    const auto flipTh = [NTH](int gt) { return ((NTH - 1 - (gt % NTH)) % NTH + NTH) % NTH; };

    // Shared per-patch config scaffolding (crop image written by caller).
    auto makeAnalyzer = [&](Band& b, const cv::Mat& crop_rgb,
                            bool th_robin) -> void {
        cv::Mat cb;
        cv::cvtColor(crop_rgb, cb, cv::COLOR_RGB2BGR);  // image member is RGB; imwrite expects BGR
        const std::string png = "dd_tmp/dd_band" + std::to_string(bid) + ".png";
        cv::imwrite(png, cb);

        YAML::Node cfg = YAML::Clone(base);
        cfg["polar_domain"]["r_start"]      = RS + b.er0 * DR;
        cfg["polar_domain"]["r_end"]        = RS + (b.er1 - 1) * DR;
        if (b.ft) {
            cfg["polar_domain"]["theta_range"]  = "2*pi";
            cfg["polar_domain"]["theta_offset"] = 0.0;
        } else {
            cfg["polar_domain"]["theta_range"]  = (b.et1 - b.et0) * DTH;
            cfg["polar_domain"]["theta_offset"] = b.et0 * DTH;
        }
        if (cfg["transient"])           cfg["transient"]["enabled"] = false;
        if (cfg["nonlinear_solver"])    cfg["nonlinear_solver"]["verbose"] = false;
        if (cfg["domain_decomposition"]) cfg["domain_decomposition"]["enabled"] = false;  // no recursion

        YAML::Node bc(YAML::NodeType::Map);
        auto edge = [&](const char* e, const char* t, double a) {
            YAML::Node n(YAML::NodeType::Map);
            n["type"] = std::string(t);
            if (std::string(t) == "robin") { n["alpha"] = a; n["beta"] = 1.0; n["gamma"] = 0.0; }
            else if (std::string(t) == "dirichlet") n["value"] = 0;
            else n["value"] = 1;  // periodic
            bc[e] = n;
        };
        edge("inner", b.er0       == 0      ? "dirichlet" : "robin", alpha);
        edge("outer", b.er1 - 1   == NR - 1 ? "dirichlet" : "robin", alpha);
        if (th_robin) {
            edge("theta_min", "robin", alpha_th);
            edge("theta_max", "robin", alpha_th);
        } else {
            edge("theta_min", "periodic", 0);
            edge("theta_max", "periodic", 0);
        }
        cfg["polar_boundary_conditions"] = bc;

        const std::string yp = "dd_tmp/dd_band" + std::to_string(bid) + ".yaml";
        std::ofstream(yp) << cfg;
        b.an = std::make_unique<MagneticFieldAnalyzer>(yp, png);
        b.an->setDDWarmStart(true);  // each sweep NK-warm-starts from the orchestrator-set Az
        if (PAR) b.an->setQuietSolver(true);  // concurrent prints would garble the log
        B.push_back(std::move(b));
        ++bid;
    };

    for (const auto& bd : dd_config.bands) {
        const int c0  = std::max(0, std::min(NR, bd.c0));  // clamp the core range to [0, NR]
        const int c1  = std::max(0, std::min(NR, bd.c1));
        const int cfr = std::max(1, bd.cf_r);
        const int cft = std::max(1, bd.cf_theta);
        if (c1 <= c0) {
            std::cerr << "WARNING: DD band [" << bd.c0 << ", " << bd.c1
                      << ") is empty/degenerate (radial range is [0, " << NR << ")); skipped." << std::endl;
            continue;
        }

        // Analytic gap link: a band ending exactly at the rotor circle (c1 ==
        // gap_r0+1) / starting at the stator circle (c0 == gap_r1) must NOT
        // extend its overlap into the analytic annulus, and its gap-side Robin
        // gamma uses the harmonic-map derivative instead of a finite difference.
        const int gR = dd_config.gap_r0, gS = dd_config.gap_r1;
        const bool gap_on = (gR >= 0 && gS > gR && gS < NR);
        const bool b_gap_out = gap_on && (c1 == gR + 1);
        const bool b_gap_in  = gap_on && (c0 == gS);
        if (gap_on && c0 < gS && c1 > gR + 1) {
            throw std::runtime_error(
                "domain_decomposition: band [" + std::to_string(c0) + ", " + std::to_string(c1) +
                ") crosses the analytic gap annulus (" + std::to_string(gR) + ", " +
                std::to_string(gS) + ") — with gap_link, rotor-side bands must end at c1=" +
                std::to_string(gR + 1) + " and stator-side bands start at c0=" + std::to_string(gS) + ".");
        }

        if (bd.sectors <= 1) {
            // ---- full-theta ring (the validated banded path) ----
            Band b;
            b.c0 = c0; b.c1 = c1; b.cfr = cfr; b.cft = cft;
            b.er0 = std::max(0,  c0 - ov * cfr);
            b.er1 = std::min(NR, c1 + ov * cfr);
            if (b_gap_out) { b.er1 = c1; b.gap_out = true; }   // do not overlap into the annulus
            if (b_gap_in)  { b.er0 = c0; b.gap_in  = true; }
            b.et0 = 0; b.et1 = NTH; b.ct0 = 0; b.ct1 = NTH; b.ft = true;
            if (b.er1 - b.er0 < 2) {
                std::cerr << "WARNING: DD band [" << bd.c0 << ", " << bd.c1
                          << ") degenerate after clamping; skipped." << std::endl;
                continue;
            }
            b.nth = NTH / b.cft;
            b.nrb = std::max(2, (b.er1 - b.er0) / b.cfr);
            agg  += (long)b.nth * b.nrb;

            cv::Mat crop(NTH, b.er1 - b.er0, CV_8UC3);
            for (int j = 0; j < NTH; ++j)
                for (int c = 0; c < b.er1 - b.er0; ++c)
                    crop.at<cv::Vec3b>(j, c) = image.at<cv::Vec3b>(j, b.er0 + c);
            cv::Mat cs;
            cv::resize(crop, cs, cv::Size(b.nrb, b.nth), 0, 0, cv::INTER_NEAREST);
            makeAnalyzer(b, cs, /*th_robin=*/false);
        } else {
            // ---- theta-sector patches (fine only; parse already forces cf=1) ----
            std::vector<std::pair<int,int>> secs;
            if (!dd_config.theta_cuts.empty()) {
                const auto& tc = dd_config.theta_cuts;
                for (size_t k = 0; k < tc.size(); ++k) {
                    const int a = tc[k];
                    const int bnd = (k + 1 < tc.size()) ? tc[k + 1] : tc[0] + NTH;
                    secs.push_back({a, bnd});
                }
            } else {
                for (int s = 0; s < bd.sectors; ++s)
                    secs.push_back({(int)((long)NTH * s / bd.sectors),
                                    (int)((long)NTH * (s + 1) / bd.sectors)});
            }
            for (auto [tb0, tb1] : secs) {
                Band b;
                b.c0 = c0; b.c1 = c1; b.cfr = 1; b.cft = 1;
                b.er0 = std::max(0,  c0 - ov);
                b.er1 = std::min(NR, c1 + ov);
                if (b_gap_out) { b.er1 = c1; b.gap_out = true; }
                if (b_gap_in)  { b.er0 = c0; b.gap_in  = true; }
                b.ct0 = tb0; b.ct1 = tb1;
                b.et0 = tb0 - ov; b.et1 = tb1 + ov;   // theta overlap (may wrap)
                b.ft  = false;
                b.nth = b.et1 - b.et0;
                b.nrb = b.er1 - b.er0;
                if (b.nrb < 2 || b.nth < 2) continue;
                agg  += (long)b.nth * b.nrb;

                cv::Mat crop(b.nth, b.nrb, CV_8UC3);
                for (int r = 0; r < b.nth; ++r) {
                    const int gt = b.et0 + (b.nth - 1 - r);  // grid row r <-> global theta gt
                    const int ir = flipTh(gt);               // -> image row (solver flips vertically)
                    for (int cc = 0; cc < b.nrb; ++cc)
                        crop.at<cv::Vec3b>(r, cc) = image.at<cv::Vec3b>(ir, b.er0 + cc);
                }
                makeAnalyzer(b, crop, /*th_robin=*/true);
            }
        }
    }
    if (B.empty()) {
        throw std::runtime_error("domain_decomposition: no valid bands after clamping -- check the "
                                 "'bands' ranges (each must lie within [0, nr) and be non-empty).");
    }
    // (B) Coverage check: the band CORES must tile the full radial range [0, NR). Any uncovered
    // column stays ZERO in the composite (-> wrong flux/field there). Warn loudly if so.
    const int  gapR   = dd_config.gap_r0;
    const int  gapS   = dd_config.gap_r1;
    const bool gap_on = (gapR >= 0 && gapS > gapR && gapS < NR);
    if (gap_on)
        std::cout << "  analytic gap link: annulus columns (" << gapR << ", " << gapS
                  << ") solved by the harmonic (Laplace) transfer map, not the FD mesh" << std::endl;
    {
        std::vector<bool> covered(NR, false);
        for (const auto& b : B)
            for (int c = std::max(0, b.c0); c < std::min(NR, b.c1); ++c) covered[c] = true;
        if (gap_on)  // the annulus interior is analytic by design, not uncovered
            for (int c = gapR + 1; c < gapS; ++c) covered[c] = true;
        int nuncov = 0, first = -1, last = -1;
        for (int c = 0; c < NR; ++c) if (!covered[c]) { ++nuncov; if (first < 0) first = c; last = c; }
        if (nuncov > 0)
            std::cerr << "WARNING: domain_decomposition bands leave " << nuncov << " radial column(s) "
                      << "UNCOVERED (e.g. [" << first << ".." << last << "] of [0," << NR << ")); those stay "
                      << "ZERO in the composite -> flux/field there will be wrong. Make the bands tile [0,"
                      << NR << ")." << std::endl;
    }
    std::cout << "  " << B.size() << " patch(es), aggregate DOF = " << agg << " ("
              << (100.0 * agg / ((double)NTH * NR)) << "% of monolithic "
              << ((long)NTH * NR) << ")" << std::endl;

    // In-loop write-back / restriction stability depends on whether a FINE band is present.
    // With a fine band (the validated mixed case, e.g. fine-active + coarse-rings), nearest-neighbour
    // write-back is stable -- the fine band feeds smooth interface traces. With ALL bands coarse
    // (e.g. a uniformly-coarsened domain split into bands), NN traces are staircased and the
    // coarse<->coarse Robin coupling DIVERGES; BILINEAR write-back smooths the traces and converges to
    // ~the uniform-downsample solution. So pick the in-loop write-back accordingly. (NOTE: for a true
    // uniform downsample, a SINGLE band [[0,nr,cf,cf]] is exact + has no coupling -- prefer that.)
    bool any_fine = false;
    for (const auto& b : B) if (b.cfr == 1 && b.cft == 1) { any_fine = true; break; }
    const bool loop_bilinear = !any_fine;
    if (loop_bilinear)
        std::cout << "  (all bands coarse -> bilinear in-loop write-back for stability)" << std::endl;

    // ---- Schwarz outer loop (multiplicative serial / additive parallel) ---------
    // Composite solution G is indexed (theta, r) to match the analyzer's Az.
    //
    // FLATTENED mode (max_inner > 0, default): each band solve is capped at
    // max_inner NK iterations, so the nonlinear relaxation happens ACROSS the
    // Schwarz sweeps (nonlinear block Gauss-Seidel) instead of being re-paid in
    // full inside every sweep. A final UNCAPPED polish sweep runs after the loop.
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(NTH, NR);
    Eigen::MatrixXd Gprev;
    const int cap = dd_config.max_inner;
    std::vector<int> full_iters(B.size());
    for (size_t i = 0; i < B.size(); ++i) {
        full_iters[i] = B[i].an->getNKMaxIterations();
        if (cap > 0) B[i].an->setNKMaxIterations(cap);
    }
    if (cap > 0)
        std::cout << "  flattened Schwarz: sub-solves capped at " << cap
                  << " NK iteration(s)/sweep + final uncapped polish sweep" << std::endl;
#ifdef _OPENMP
#if _OPENMP >= 200805
    if (PAR) omp_set_max_active_levels(1);  // inner per-patch AMGCL runs single-threaded
#endif
    if (PAR) std::cout << "  OMP patch-parallel: max_threads=" << omp_get_max_threads() << std::endl;
#else
    if (PAR) std::cout << "  (parallel requested but no OpenMP in this build -> serial sweeps)" << std::endl;
#endif

    // One band spanning the full radial range has Dirichlet at both edges and
    // no transmission to iterate -- a single uncapped solve IS the answer
    // (this is the documented uniform-downsample use). Skip the Schwarz loop.
    const bool no_coupling = (B.size() == 1 && B[0].ft && B[0].er0 == 0 && B[0].er1 == NR);

    // ---- Two-level Galerkin coarse space (dd_bench DD_COARSE port) -------------
    // Bilinear r x theta prolongation P (theta periodic, r clamped); each sweep:
    // refresh the FULL operator at the current composite (buildPolarOperator uses
    // the member Az) and apply G += damp * P * (P^T A P)^-1 * P^T (b - A*G).
    // Intended to bound the outer sweep count for many-sector configs (the
    // 1-level theta Schwarz only contracts at ~0.9/sweep).
    //
    // STATUS (2026-07 benchmark): NEGATIVE on IEEJ-D even with FINE sectors —
    // the combination the linear coarse_space_poc predicted would work. The
    // early composite's patch-seam discontinuities poison the NONLINEAR coarse
    // operator (mu evaluated at the seamy field): its LU solve returns spiky
    // corrections that blow the field up ~100x/sweep, and warm-up delay, an L2
    // trust clamp, coarse_damp 0.3 and relax 0.5 all failed to stabilize it.
    // The base flattened-sector iteration itself also fails to contract
    // (wanders at res 1.2-2.6 indefinitely), so there is no convergent
    // iteration for the coarse space to accelerate. Kept as an OFF-by-default
    // experimental knob with the guards below; do not enable in production.
    const bool use_cs = (dd_config.coarse_r > 0 && dd_config.coarse_th > 0) && !no_coupling;
    const double cs_damp = dd_config.coarse_damp;
    Eigen::SparseMatrix<double> Pcs;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> cs_lu;
    bool cs_pattern_done = false;
    int  cs_nrc = 0, cs_nthc = 0;
    if (use_cs) {
        const int CR = dd_config.coarse_r, CTH = dd_config.coarse_th;
        cs_nrc  = std::max(2, NR  / CR);
        cs_nthc = std::max(2, NTH / CTH);
        const long Nc = (long)cs_nrc * cs_nthc;
        std::vector<Eigen::Triplet<double>> tp;
        tp.reserve((size_t)NR * NTH * 4);
        for (int i = 0; i < NR; ++i) {
            const double fic = (double)i / CR;
            const int ic0 = std::min((int)fic, cs_nrc - 1), ic1 = std::min(ic0 + 1, cs_nrc - 1);
            const double wr = fic - (int)fic;
            for (int j = 0; j < NTH; ++j) {
                const double fjc = (double)j / CTH;
                const int jc0 = ((int)fjc) % cs_nthc, jc1 = (jc0 + 1) % cs_nthc;  // theta periodic
                const double wt = fjc - (int)fjc;
                const int f = i * NTH + j;   // matches buildMatrixPolar's row ordering
                tp.push_back({f, ic0 * cs_nthc + jc0, (1 - wr) * (1 - wt)});
                tp.push_back({f, ic0 * cs_nthc + jc1, (1 - wr) * wt});
                tp.push_back({f, ic1 * cs_nthc + jc0, wr * (1 - wt)});
                tp.push_back({f, ic1 * cs_nthc + jc1, wr * wt});
            }
        }
        Pcs.resize((long)NTH * NR, Nc);
        Pcs.setFromTriplets(tp.begin(), tp.end());
        std::cout << "  coarse space: CR=" << CR << " CTH=" << CTH
                  << " coarseDOF=" << Nc << " damp=" << cs_damp
                  << " (Galerkin, rebuilt each sweep at current mu)" << std::endl;
    }
    auto coarse_correct = [&](Eigen::MatrixXd& Gc) {
        // Refresh the full nonlinear operator at the current composite.
        Az = Gc;
        Eigen::SparseMatrix<double> Afull;
        Eigen::VectorXd bvec;
        buildPolarOperator(Afull, bvec);
        Eigen::SparseMatrix<double> Ac =
            (Eigen::SparseMatrix<double>(Pcs.transpose()) * Afull * Pcs).pruned();
        if (!cs_pattern_done) { cs_lu.analyzePattern(Ac); cs_pattern_done = true; }
        cs_lu.factorize(Ac);
        if (cs_lu.info() != Eigen::Success) {
            std::cerr << "WARNING: DD coarse-space factorization failed; skipping correction this sweep."
                      << std::endl;
            return;
        }
        Eigen::VectorXd Gv((long)NTH * NR);
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j) Gv[(long)i * NTH + j] = Gc(j, i);
        const Eigen::VectorXd R = bvec - Afull * Gv;
        Eigen::VectorXd d = Pcs * cs_lu.solve(Eigen::VectorXd(Pcs.transpose() * R));
        // Trust clamp: early composites carry seam discontinuities, so the
        // nonlinear operator (mu evaluated at the seamy field) can make the
        // correction an AMPLIFIER (measured: unclamped corrections grew G
        // ~100x per sweep). Never let one correction move G by more than half
        // its own norm.
        const double dn = d.norm(), gn = Gv.norm() + 1e-30;
        if (dn > 0.5 * gn) d *= 0.5 * gn / dn;
        for (int i = 0; i < NR; ++i)
            for (int j = 0; j < NTH; ++j) Gc(j, i) += cs_damp * d[(long)i * NTH + j];
    };

    // ---- Analytic gap transfer map ---------------------------------------------
    // Source-free air annulus [Ra, Rb]: Az is harmonic, so per theta-harmonic k
    // the field is a_k*(r/Rb)^k + b_k*(Ra/r)^k (k=0: a + b*ln r). Given the two
    // circle traces (rows of the composite at columns gapR/gapS), the radial
    // derivative on each circle follows EXACTLY — this replaces the finite
    // difference across the gap in the Robin gammas, which is what couples the
    // two sides. O(K*N) DFT per sweep (~2x 4.4M mul-adds, a few ms).
    //
    // STATUS (2026-07 benchmark): NEGATIVE on IEEJ-D under FLATTENED sub-solves.
    // The map itself is spectrally exact and FD-consistent (difference-quotient
    // form below), and the fine-fine split contracts early (res 0.38 at sweep 2
    // with the harmonically-optimal robin_p~300 = sqrt(dtn_min*dtn_max)), but a
    // gap interface carries strong slot harmonics whose transmission makes large
    // per-sweep field moves that iteration-capped nonlinear sub-solves cannot
    // track: every tested combination (robin_p 12/300, relax 0.5-0.7, max_inner
    // 3/8, cf 1/2) oscillates or explodes by sweep ~6-8. Nested (max_inner: 0)
    // 2-domain gap splits DO converge (dd_bench, 0.05-0.4% flux) but cost more
    // than the monolithic solve. Kept OFF-by-default for traceability and for a
    // future NK that converges fast enough to make nested sub-solves cheap.
    const double gapRa = RS + gapR * DR, gapRb = RS + gapS * DR;
    std::vector<double> gap_dRa(gap_on ? NTH : 0, 0.0);   // difference-quotient dAz/dr at the rotor edge
    std::vector<double> gap_dRb(gap_on ? NTH : 0, 0.0);   // difference-quotient dAz/dr at the stator edge
    // The FD Robin gammas use one-sided DIFFERENCE QUOTIENTS (the slope over the
    // first cell outside the boundary node), not point derivatives — a point
    // derivative on the circle disagrees with the discrete closure by up to
    // e^{k·dr/R} per harmonic (measured to destabilize the coupling). So the
    // analytic map supplies the field VALUE at each side's neighbour-node
    // radius (harmonically exact) and the gamma uses the same quotient the FD
    // path would; a coarsened side (cfr>1) gets the quotient over ITS spacing.
    int gap_cfr_out = 1, gap_cfr_in = 1;
    for (const auto& b : B) {
        if (b.gap_out) gap_cfr_out = b.cfr;
        if (b.gap_in)  gap_cfr_in  = b.cfr;
    }
    auto computeGapDeriv = [&](const Eigen::MatrixXd& Gs) {
        if (!gap_on) return;
        const int N = NTH, K = N / 2;
        const double lnratio = std::log(gapRa / gapRb);   // < 0
        const double r_out = std::min(gapRa + gap_cfr_out * DR, gapRb);  // rotor-side face node
        const double r_in  = std::max(gapRb - gap_cfr_in  * DR, gapRa);  // stator-side face node
        const double h_out = r_out - gapRa, h_in = gapRb - r_in;
        // v_out/v_in = harmonic field at the two evaluation radii.
        std::vector<double> v_out(N, 0.0), v_in(N, 0.0);
        double ua0 = 0, ub0 = 0;
        for (int j = 0; j < N; ++j) { ua0 += Gs(j, gapR); ub0 += Gs(j, gapS); }
        ua0 /= N; ub0 /= N;
        const double b0 = (ub0 - ua0) / (-lnratio);
        for (int j = 0; j < N; ++j) {
            v_out[j] = ua0 + b0 * std::log(r_out / gapRa);
            v_in[j]  = ua0 + b0 * std::log(r_in  / gapRa);
        }
#ifdef _OPENMP
        #pragma omp parallel
#endif
        {
            std::vector<double> vo_loc(N, 0.0), vi_loc(N, 0.0);
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int k = 1; k <= K; ++k) {
                // forward DFT of both traces at harmonic k (cos & sin parts)
                double uac = 0, uas = 0, ubc = 0, ubs = 0;
                const double dth = 2.0 * M_PI * k / N;
                const double cstep = std::cos(dth), sstep = std::sin(dth);
                double cj = 1.0, sj = 0.0;
                for (int j = 0; j < N; ++j) {
                    const double ga = Gs(j, gapR), gb = Gs(j, gapS);
                    uac += ga * cj; uas += ga * sj;
                    ubc += gb * cj; ubs += gb * sj;
                    const double cn = cj * cstep - sj * sstep;
                    sj = cj * sstep + sj * cstep; cj = cn;
                }
                const double norm = (2 * k == N) ? 1.0 / N : 2.0 / N;  // Nyquist has no sin partner
                uac *= norm; uas *= norm; ubc *= norm; ubs *= norm;
                // annulus transfer: t=(Ra/Rb)^k; basis (r/Rb)^k and (Ra/r)^k
                const double t = std::exp(k * lnratio);
                const double det = t * t - 1.0;
                const double aC = (t * uac - ubc) / det, bC = (t * ubc - uac) / det;
                const double aS = (t * uas - ubs) / det, bS = (t * ubs - uas) / det;
                const double phO = std::exp(k * std::log(r_out / gapRb));
                const double psO = std::exp(k * std::log(gapRa / r_out));
                const double phI = std::exp(k * std::log(r_in  / gapRb));
                const double psI = std::exp(k * std::log(gapRa / r_in));
                const double voc = aC * phO + bC * psO, vos = aS * phO + bS * psO;
                const double vic = aC * phI + bC * psI, vis = aS * phI + bS * psI;
                // accumulate the inverse transform
                cj = 1.0; sj = 0.0;
                for (int j = 0; j < N; ++j) {
                    vo_loc[j] += voc * cj + vos * sj;
                    vi_loc[j] += vic * cj + vis * sj;
                    const double cn = cj * cstep - sj * sstep;
                    sj = cj * sstep + sj * cstep; cj = cn;
                }
            }
#ifdef _OPENMP
            #pragma omp critical
#endif
            for (int j = 0; j < N; ++j) { v_out[j] += vo_loc[j]; v_in[j] += vi_loc[j]; }
        }
        for (int j = 0; j < N; ++j) {
            gap_dRa[j] = (v_out[j] - Gs(j, gapR)) / h_out;
            gap_dRb[j] = (Gs(j, gapS) - v_in[j])  / h_in;
        }
    };

    auto do_band = [&](Band& b, const Eigen::MatrixXd& Gs) {
        if (b.ft) {
            // ---- full-theta ring (validated banded path; reads Gs, writes G) ----
            // Symmetric-Robin transmission gamma from the composite's edge traces
            // (Az and dAz/dn averaged over the band's theta-coarsening window).
            std::vector<double> gin(b.nth, 0.0), gout(b.nth, 0.0);
            for (int mb = 0; mb < b.nth; ++mb) {
                double ui = 0, di = 0, uo = 0, doo = 0;
                const int n = b.cft;
                for (int t = 0; t < b.cft; ++t) {
                    const int j = mb * b.cft + t;
                    if (b.er0 != 0) {
                        ui += Gs(j, b.er0);
                        di += b.gap_in ? gap_dRb[j]
                                       : (Gs(j, b.er0) - Gs(j, b.er0 - 1)) / DR;
                    }
                    if (b.er1 - 1 != NR - 1) {
                        uo += Gs(j, b.er1 - 1);
                        doo += b.gap_out ? gap_dRa[j]
                                         : (Gs(j, b.er1) - Gs(j, b.er1 - 1)) / DR;
                    }
                }
                if (b.er0     != 0)      gin[mb]  = alpha * (ui / n) - (di / n);   // inner edge: alpha*Az - dAz/dn
                if (b.er1 - 1 != NR - 1) gout[mb] = alpha * (uo / n) + (doo / n);  // outer edge: alpha*Az + dAz/dn
            }
            if (b.er0     != 0)      b.an->setBoundaryProfile("inner", gin);
            if (b.er1 - 1 != NR - 1) b.an->setBoundaryProfile("outer", gout);

            // Warm start: sample the composite into the band's (coarse) grid.
            Eigen::MatrixXd pAz(b.nth, b.nrb);
            for (int mb = 0; mb < b.nth; ++mb)
                for (int kb = 0; kb < b.nrb; ++kb) {
                    const int j = mb * b.cft + b.cft / 2;
                    const int c = b.er0 + (int)std::llround((double)kb * (b.er1 - 1 - b.er0) / (b.nrb - 1));
                    pAz(mb, kb) = Gs(std::min(j, NTH - 1), std::min(c, NR - 1));
                }
            b.an->setAz(pAz);
            b.an->solve();
            const Eigen::MatrixXd& sol = b.an->getAz();

            // Write the band CORE [c0, c1) back into the composite G. Mixed (fine present) -> NN
            // (validated, stable mortar; a bilinear OUTPUT pass smooths it at the end). All-coarse ->
            // BILINEAR in-loop (smooth traces so coarse<->coarse coupling converges).
            for (int j = 0; j < NTH; ++j) {
                if (loop_bilinear) {
                    const double fmb = (double)j / b.cft;
                    int mb0 = (int)std::floor(fmb);
                    const double wt = fmb - mb0;
                    mb0 = ((mb0 % b.nth) + b.nth) % b.nth;
                    const int mb1 = (mb0 + 1) % b.nth;            // periodic wrap in theta
                    for (int c = b.c0; c < b.c1; ++c) {
                        double fkb = (double)(c - b.er0) * (b.nrb - 1) / (b.er1 - 1 - b.er0);
                        fkb = std::max(0.0, std::min(fkb, (double)(b.nrb - 1)));
                        const int kb0 = (int)std::floor(fkb);
                        const int kb1 = std::min(kb0 + 1, b.nrb - 1);
                        const double wr = fkb - kb0;
                        G(j, c) = (1.0 - wt) * ((1.0 - wr) * sol(mb0, kb0) + wr * sol(mb0, kb1))
                                +        wt  * ((1.0 - wr) * sol(mb1, kb0) + wr * sol(mb1, kb1));
                    }
                } else {
                    int mb = j / b.cft;
                    if (mb >= b.nth) mb = b.nth - 1;
                    for (int c = b.c0; c < b.c1; ++c) {
                        int kb = (int)std::llround((double)(c - b.er0) * (b.nrb - 1) / (b.er1 - 1 - b.er0));
                        kb = std::max(0, std::min(kb, b.nrb - 1));
                        G(j, c) = sol(mb, kb);
                    }
                }
            }
        } else {
            // ---- theta-sector patch (fine, exact resolution) ----
            // Radial Robin gammas per grid theta row j <-> global theta et0+j.
            std::vector<double> gin(b.nth, 0.0), gout(b.nth, 0.0);
            auto GS = [&](int gt, int c) -> double { return Gs(wrapTh(gt), c); };
            for (int j = 0; j < b.nth; ++j) {
                const int gt = b.et0 + j;
                const int gw = wrapTh(gt);
                if (b.er0     != 0)
                    gin[j]  = alpha * GS(gt, b.er0)
                            - (b.gap_in ? gap_dRb[gw]
                                        : (GS(gt, b.er0) - GS(gt, b.er0 - 1)) / DR);
                if (b.er1 - 1 != NR - 1)
                    gout[j] = alpha * GS(gt, b.er1 - 1)
                            + (b.gap_out ? gap_dRa[gw]
                                         : (GS(gt, b.er1) - GS(gt, b.er1 - 1)) / DR);
            }
            if (b.er0     != 0)      b.an->setBoundaryProfile("inner", gin);
            if (b.er1 - 1 != NR - 1) b.an->setBoundaryProfile("outer", gout);
            // Theta Robin gammas per radial grid index i <-> column er0+i (per-r profile).
            {
                const int t0 = b.et0, t1 = b.et0 + b.nth - 1;  // boundary global thetas
                std::vector<double> gtmin(b.nrb, 0.0), gtmax(b.nrb, 0.0);
                for (int i = 0; i < b.nrb; ++i) {
                    const int c = b.er0 + i;
                    gtmin[i] = alpha_th * GS(t0, c) - (GS(t0, c)     - GS(t0 - 1, c)) / DTH;
                    gtmax[i] = alpha_th * GS(t1, c) + (GS(t1 + 1, c) - GS(t1, c))     / DTH;
                }
                b.an->setBoundaryProfile("theta_min", gtmin);
                b.an->setBoundaryProfile("theta_max", gtmax);
            }
            // Warm start (exact resolution: direct sample).
            Eigen::MatrixXd pAz(b.nth, b.nrb);
            for (int j = 0; j < b.nth; ++j)
                for (int i = 0; i < b.nrb; ++i)
                    pAz(j, i) = GS(b.et0 + j, std::min(b.er0 + i, NR - 1));
            b.an->setAz(pAz);
            b.an->solve();
            const Eigen::MatrixXd& sol = b.an->getAz();
            // Write the 2D CORE back (plain copy — no mortar at fine resolution).
            for (int j = 0; j < b.nth; ++j) {
                const int gug = b.et0 + j;
                if (gug < b.ct0 || gug >= b.ct1) continue;
                const int gt = wrapTh(gug);
                for (int c = b.c0; c < b.c1; ++c)
                    G(gt, c) = sol(j, c - b.er0);
            }
        }
    };

    auto sweep_all = [&](const Eigen::MatrixXd& Gs) {
        computeGapDeriv(Gs);   // analytic gap derivatives for this sweep's gammas
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1) if (PAR)
#endif
        for (int pi = 0; pi < (int)B.size(); ++pi) do_band(B[pi], Gs);
    };

    int it = 0;
    double res = 1.0;
    double Gnorm_ref = 0.0;
    if (no_coupling) {
        B[0].an->setNKMaxIterations(full_iters[0]);
        do_band(B[0], G);
        it = 1;
        res = 0.0;
        std::cout << "  single full-range band (no transmission): solved once, skipping Schwarz loop" << std::endl;
    } else for (it = 1; it <= maxo; ++it) {
        Gprev = G;
        // Additive (parallel): every patch reads the sweep-start snapshot ->
        // independent solves, disjoint-core write-back (race-free).
        // Multiplicative (serial): patches read the live composite.
        sweep_all(PAR ? Gprev : G);
        if (omega != 1.0) G = Gprev + omega * (G - Gprev);   // outer under-relaxation (damps a 2-cycle)
        // 2-level global correction, after a warm-up: the first sweeps' composite
        // still carries patch-seam discontinuities that poison the nonlinear
        // coarse operator (measured blow-up when corrected from sweep 1).
        if (use_cs && it >= 4) coarse_correct(G);
        res = (G - Gprev).norm() / (G.norm() + 1e-30);       // relative Schwarz residual (no reference needed)
        std::cout << "  DD sweep " << it << ": residual = " << res << std::endl;
        // Magnitude divergence guard: a ratio-constant geometric explosion keeps
        // the relative Schwarz residual ~1 while the field grows without bound
        // (seen with an unclamped coarse correction: x100/sweep to |Az|~1e36).
        // Sweep 1's patch solves already produce physical-scale Az even when the
        // composite is seamy, so 100x that magnitude is unambiguously divergent
        // (a relax<1 ramp-up only approaches ~2x of it).
        if (it == 1) Gnorm_ref = G.cwiseAbs().maxCoeff();
        if (it > 1 && Gnorm_ref > 0.0 && G.cwiseAbs().maxCoeff() > 100.0 * Gnorm_ref) {
            throw std::runtime_error(
                "domain_decomposition: the composite field magnitude grew " +
                std::to_string(G.cwiseAbs().maxCoeff() / Gnorm_ref) + "x beyond the sweep-1 scale "
                "-- the iteration is EXPLODING (coarse-space/transmission instability). Reduce "
                "coarse_damp, disable 'coarse', or check the sector layout (cuts must pass through "
                "iron; the air gap must stay in a full-theta fine ring).");
        }
        // (A) Divergence guard: bail out with a clear message instead of exporting blown-up garbage.
        // A healthy radial-band DD drops the residual to O(1e-2) within ~3 sweeps; a residual that
        // blows up or stays order-1 means the ACTIVE region is being coarsened or an interface sits
        // in air/coils (the Schwarz iteration is then unstable -- see README band-design rules).
        // Flattened mode legitimately makes larger early moves (the nonlinear relaxation itself
        // happens across sweeps), so give it more sweeps before the stagnation check bites.
        // Sectored configs propagate information ~one patch per sweep around the ring, so scale
        // the stagnation horizon with the patch count — but cap it so a non-convergent wander
        // (measured: flattened theta-sectors oscillate at res 1.2-2.6 indefinitely) still aborts
        // instead of exporting a garbage composite.
        const int stall_check_from = (cap > 0)
            ? std::min<int>(16, std::max<int>(8, 2 * (int)B.size())) : 4;
        if (!std::isfinite(res) || res > 5.0 || (it >= stall_check_from && res > 0.8)) {
            throw std::runtime_error(
                "domain_decomposition: the Schwarz iteration is DIVERGING (residual=" +
                std::to_string(res) + " at sweep " + std::to_string(it) + "). Likely a band interface "
                "sits in AIR or COILS, or the coarsening is too aggressive for the geometry. Place band "
                "interfaces in IRON, keep the air gap inside a fine band, and lower the coarsening factor. "
                "For a plain uniform downsample use a SINGLE band [[0, nr, cf, cf]] (exact, no coupling).");
        }
        if (res < tol) break;
    }
    const int sweeps_done = std::min(it, maxo);  // 'it' is maxo+1 if the loop ran to the cap without converging
    std::cout << "=== DD done: " << sweeps_done << " sweep(s), final residual = " << res
              << (res < tol ? " (converged)" : " (reached max_outer)") << " ===" << std::endl;

    // Final UNCAPPED polish sweep (flattened mode only): each band converges on
    // its own grid against the final transmission data, so the composite gets
    // fully-converged band solutions (the capped sweeps only relaxed them).
    if (cap > 0 && !no_coupling) {
        for (size_t i = 0; i < B.size(); ++i) B[i].an->setNKMaxIterations(full_iters[i]);
        Gprev = G;
        sweep_all(PAR ? Gprev : G);
        const double pres = (G - Gprev).norm() / (G.norm() + 1e-30);
        std::cout << "  DD polish sweep (uncapped): residual = " << pres << std::endl;
    }

    // (C) Final BILINEAR + PARTITION-OF-UNITY blend pass (OUTPUT ONLY -- does NOT touch the validated
    // NN Schwarz iteration above). Each band's converged sub-solution is rendered bilinearly over its
    // FULL extended range with a taper weight (1 in the core, linearly ramping to 0 across the
    // overlaps), and overlapping bands are averaged. This removes BOTH the nearest-neighbour staircase
    // INSIDE coarse bands AND the band-interface B spikes (the fine<->coarse seam becomes a smooth
    // blend). Interpolation only (theta periodic, radial clamped) -> never extrapolates outside the
    // domain. The coarse-region B is still approximate; the trustworthy DD output is the flux.
    auto taper_r = [](const Band& b, int c) -> double {
        if (c >= b.c0 && c < b.c1) return 1.0;
        if (c <  b.c0) return (b.c0 > b.er0)     ? std::max(0.0, (double)(c - b.er0) / (b.c0 - b.er0))         : 0.0;
        /* c >= c1 */  return (b.er1 - 1 > b.c1) ? std::max(0.0, (double)(b.er1 - 1 - c) / (b.er1 - 1 - b.c1)) : 0.0;
    };
    auto taper_th = [](const Band& b, int gug) -> double {
        if (b.ft) return 1.0;
        if (gug >= b.ct0 && gug < b.ct1) return 1.0;
        if (gug <  b.ct0) return (b.ct0 > b.et0)     ? std::max(0.0, (double)(gug - b.et0) / (b.ct0 - b.et0))         : 0.0;
        /* gug >= ct1 */   return (b.et1 - 1 > b.ct1) ? std::max(0.0, (double)(b.et1 - 1 - gug) / (b.et1 - 1 - b.ct1)) : 0.0;
    };
    {
        Eigen::MatrixXd Gacc = Eigen::MatrixXd::Zero(NTH, NR);
        Eigen::MatrixXd Wacc = Eigen::MatrixXd::Zero(NTH, NR);
        for (const auto& b : B) {
            const Eigen::MatrixXd& sol = b.an->getAz();
            if (b.ft) {
                for (int j = 0; j < NTH; ++j) {
                    const double fmb = (double)j / b.cft;
                    int mb0 = (int)std::floor(fmb);
                    const double wt = fmb - mb0;
                    mb0 = ((mb0 % b.nth) + b.nth) % b.nth;
                    const int mb1 = (mb0 + 1) % b.nth;             // periodic wrap in theta
                    for (int c = b.er0; c < b.er1; ++c) {
                        const double w = taper_r(b, c);
                        if (w <= 0.0) continue;
                        double fkb = (double)(c - b.er0) * (b.nrb - 1) / (b.er1 - 1 - b.er0);
                        fkb = std::max(0.0, std::min(fkb, (double)(b.nrb - 1)));   // clamp (no radial extrapolation)
                        const int kb0 = (int)std::floor(fkb);
                        const int kb1 = std::min(kb0 + 1, b.nrb - 1);
                        const double wr = fkb - kb0;
                        const double val = (1.0 - wt) * ((1.0 - wr) * sol(mb0, kb0) + wr * sol(mb0, kb1))
                                         +        wt  * ((1.0 - wr) * sol(mb1, kb0) + wr * sol(mb1, kb1));
                        Gacc(j, c) += w * val;
                        Wacc(j, c) += w;
                    }
                }
            } else {
                // Sector patch: exact-resolution values, 2D taper (theta x r).
                for (int j = 0; j < b.nth; ++j) {
                    const int gug = b.et0 + j;
                    const double wth = taper_th(b, gug);
                    if (wth <= 0.0) continue;
                    const int gt = wrapTh(gug);
                    for (int c = b.er0; c < b.er1; ++c) {
                        const double w = wth * taper_r(b, c);
                        if (w <= 0.0) continue;
                        Gacc(gt, c) += w * sol(j, c - b.er0);
                        Wacc(gt, c) += w;
                    }
                }
            }
        }
        for (int j = 0; j < NTH; ++j)
            for (int c = 0; c < NR; ++c)
                if (Wacc(j, c) > 1e-12) G(j, c) = Gacc(j, c) / Wacc(j, c);
    }

    // Fill the analytic annulus interior from the harmonic solution of the FINAL
    // circle traces so exports/stress see a complete field.
    if (gap_on && gapS - gapR > 1) {
        const int N = NTH, K = N / 2;
        const double lnratio = std::log(gapRa / gapRb);
        double ua0 = 0, ub0 = 0;
        for (int j = 0; j < N; ++j) { ua0 += G(j, gapR); ub0 += G(j, gapS); }
        ua0 /= N; ub0 /= N;
        const double b0 = (ub0 - ua0) / (-lnratio);
        const double a0 = ua0;
        for (int c = gapR + 1; c < gapS; ++c) {
            const double r = RS + c * DR;
            const double v0 = a0 + b0 * std::log(r / gapRa);
            for (int j = 0; j < N; ++j) G(j, c) = v0;
        }
        for (int k = 1; k <= K; ++k) {
            double uac = 0, uas = 0, ubc = 0, ubs = 0;
            const double dth = 2.0 * M_PI * k / N;
            const double cstep = std::cos(dth), sstep = std::sin(dth);
            double cj = 1.0, sj = 0.0;
            for (int j = 0; j < N; ++j) {
                const double ga = G(j, gapR), gb = G(j, gapS);
                uac += ga * cj; uas += ga * sj;
                ubc += gb * cj; ubs += gb * sj;
                const double cn = cj * cstep - sj * sstep;
                sj = cj * sstep + sj * cstep; cj = cn;
            }
            const double norm = (2 * k == N) ? 1.0 / N : 2.0 / N;
            uac *= norm; uas *= norm; ubc *= norm; ubs *= norm;
            const double t = std::exp(k * lnratio), det = t * t - 1.0;
            const double aC = (t * uac - ubc) / det, bC = (t * ubc - uac) / det;
            const double aS = (t * uas - ubs) / det, bS = (t * ubs - uas) / det;
            for (int c = gapR + 1; c < gapS; ++c) {
                const double r   = RS + c * DR;
                const double phi = std::exp(k * std::log(r / gapRb));   // (r/Rb)^k
                const double psi = std::exp(k * std::log(gapRa / r));   // (Ra/r)^k
                const double vc = aC * phi + bC * psi, vs = aS * phi + bS * psi;
                cj = 1.0; sj = 0.0;
                for (int j = 0; j < N; ++j) {
                    G(j, c) += vc * cj + vs * sj;
                    const double cn = cj * cstep - sj * sstep;
                    sj = cj * sstep + sj * cstep; cj = cn;
                }
            }
        }
    }

    // Composite solution -> member Az, then refresh B/H/mu on the full grid so the
    // Mu/H exports, energy, and Maxwell-stress are consistent with the composite
    // (same sequence buildPolarOperator uses to refresh from a given Az).
    Az = G;
    calculateMagneticFieldPolar();
    calculateHField();
    updateMuDistribution();
}
