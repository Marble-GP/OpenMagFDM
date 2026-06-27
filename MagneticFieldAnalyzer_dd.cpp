// MagneticFieldAnalyzer_dd.cpp
// =============================================================================
// v1.6 domain decomposition (optimized Schwarz) -- PRODUCTION solve path.
//
// Opt-in, polar only. A VARIABLE-RESOLUTION accuracy mode: the domain is split
// into radial bands; each band is solved on its OWN uniform grid (coarsened by
// cf_r x cf_theta where the field is smooth, kept fine across the air gap /
// saturated zone), and adjacent bands are coupled by a SYMMETRIC, conservative
// Robin transmission iterated (multiplicative Schwarz) to global consistency.
// The converged composite (fine-where-fine, coarse-where-coarse) is written into
// the member Az; downstream stress / energy / export are unchanged.
//
// NOTE (positioning): on a dense machine this is NOT faster than the monolithic
// solve -- uniform downsampling wins raw speed. DD's niche is keeping the gap /
// saturation FULLY RESOLVED while coarsening the smooth bulk, i.e. an accuracy /
// verification mode. Enable via the `domain_decomposition` YAML block.
//
// The algorithm is the validated banded path from the dd_bench research harness
// (symmetric Robin + warm-start + optional under-relaxation), here driven by the
// parsed DDConfig and sourcing geometry/image/config from the owning analyzer.
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

void MagneticFieldAnalyzer::solveDomainDecomposition() {
    const int    NTH   = ntheta;
    const int    NR    = nr;
    const double DR    = dr;
    const double RS    = r_start;
    const double alpha = dd_config.robin_p;   // effective Robin coefficient (alpha*Az + beta*dAz/dn = gamma)
    const int    ov    = dd_config.overlap;
    const int    maxo  = dd_config.max_outer;
    const double tol   = dd_config.tol;
    const double omega = dd_config.relax;

    std::cout << "\n=== Domain decomposition (optimized Schwarz, polar) ===" << std::endl;
    std::cout << "  " << dd_config.bands.size() << " bands | robin_p=" << alpha
              << " overlap=" << ov << " max_outer=" << maxo
              << " tol=" << tol << " relax=" << omega << std::endl;

    // Reload the RAW base config so each sub-domain re-parses variables / materials
    // exactly as a standalone run would (mirrors the validated harness path).
    YAML::Node base = YAML::LoadFile(config_path);

    // Per-band sub-analyzer: cropped + (optionally) coarsened annulus solved on its
    // own uniform grid, with Robin (interior edges) / Dirichlet (domain ends) /
    // periodic (theta) boundary conditions.
    struct Band {
        int c0, c1;          // core column range [c0, c1) written back into the composite
        int cfr, cft;        // radial / theta coarsen factors
        int er0, er1;        // extended (overlapped) column range actually solved
        int nth, nrb;        // band grid size (theta, radial)
        std::unique_ptr<MagneticFieldAnalyzer> an;
    };
    std::vector<Band> B;
    { std::error_code ec; std::filesystem::create_directories("dd_tmp", ec); }  // portable (Windows + POSIX)
    long agg = 0;
    int bid = 0;
    for (const auto& bd : dd_config.bands) {
        Band b;
        b.c0  = std::max(0, std::min(NR, bd.c0));  // clamp the core range to [0, NR]
        b.c1  = std::max(0, std::min(NR, bd.c1));
        b.cfr = std::max(1, bd.cf_r);
        b.cft = std::max(1, bd.cf_theta);
        b.er0 = std::max(0,  b.c0 - ov * b.cfr);
        b.er1 = std::min(NR, b.c1 + ov * b.cfr);
        if (b.c1 <= b.c0 || b.er1 - b.er0 < 2) {
            std::cerr << "WARNING: DD band [" << bd.c0 << ", " << bd.c1
                      << ") is empty/degenerate (radial range is [0, " << NR << ")); skipped." << std::endl;
            continue;
        }
        b.nth = NTH / b.cft;
        b.nrb = std::max(2, (b.er1 - b.er0) / b.cfr);
        agg  += (long)b.nth * b.nrb;

        // Crop full-theta x extended columns from the RGB source image, then
        // nearest-neighbour downsample to the band's own (coarse) grid. No theta
        // pre-flip: the band keeps full theta and the sub-analyzer's internal
        // grid<->image flip already aligns band grid g with the composite grid g.
        cv::Mat crop(NTH, b.er1 - b.er0, CV_8UC3);
        for (int j = 0; j < NTH; ++j)
            for (int c = 0; c < b.er1 - b.er0; ++c)
                crop.at<cv::Vec3b>(j, c) = image.at<cv::Vec3b>(j, b.er0 + c);
        cv::Mat cs;
        cv::resize(crop, cs, cv::Size(b.nrb, b.nth), 0, 0, cv::INTER_NEAREST);
        cv::Mat cb;
        cv::cvtColor(cs, cb, cv::COLOR_RGB2BGR);  // image member is RGB; imwrite expects BGR
        const std::string png = "dd_tmp/dd_band" + std::to_string(bid) + ".png";
        cv::imwrite(png, cb);

        // Clone the base config; restrict it to this band's annulus + DD BCs.
        YAML::Node cfg = YAML::Clone(base);
        cfg["polar_domain"]["r_start"]      = RS + b.er0 * DR;
        cfg["polar_domain"]["r_end"]        = RS + (b.er1 - 1) * DR;
        cfg["polar_domain"]["theta_range"]  = "2*pi";
        cfg["polar_domain"]["theta_offset"] = 0.0;
        if (cfg["transient"])           cfg["transient"]["enabled"] = false;
        if (cfg["nonlinear_solver"])    cfg["nonlinear_solver"]["verbose"] = false;
        if (cfg["domain_decomposition"]) cfg["domain_decomposition"]["enabled"] = false;  // sub-domains solve monolithically (no recursion)

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
        edge("theta_min", "periodic", 0);
        edge("theta_max", "periodic", 0);
        cfg["polar_boundary_conditions"] = bc;

        const std::string yp = "dd_tmp/dd_band" + std::to_string(bid) + ".yaml";
        std::ofstream(yp) << cfg;
        b.an = std::make_unique<MagneticFieldAnalyzer>(yp, png);
        b.an->setDDWarmStart(true);  // each sweep NK-warm-starts from the orchestrator-set Az
        B.push_back(std::move(b));
        ++bid;
    }
    if (B.empty()) {
        throw std::runtime_error("domain_decomposition: no valid bands after clamping -- check the "
                                 "'bands' ranges (each must lie within [0, nr) and be non-empty).");
    }
    std::cout << "  aggregate DOF = " << agg << " ("
              << (100.0 * agg / ((double)NTH * NR)) << "% of monolithic "
              << ((long)NTH * NR) << ")" << std::endl;

    // ---- multiplicative Schwarz outer loop -------------------------------------
    // Composite solution G is indexed (theta, r) to match the analyzer's Az.
    Eigen::MatrixXd G = Eigen::MatrixXd::Zero(NTH, NR);
    Eigen::MatrixXd Gprev;
    int it = 0;
    double res = 1.0;
    for (it = 1; it <= maxo; ++it) {
        Gprev = G;
        for (auto& b : B) {
            // Symmetric-Robin transmission gamma from the composite G's edge traces
            // (Az and dAz/dn averaged over the band's theta-coarsening window).
            std::vector<double> gin(b.nth, 0.0), gout(b.nth, 0.0);
            for (int mb = 0; mb < b.nth; ++mb) {
                double ui = 0, di = 0, uo = 0, doo = 0;
                const int n = b.cft;
                for (int t = 0; t < b.cft; ++t) {
                    const int j = mb * b.cft + t;
                    if (b.er0     != 0)      { ui += G(j, b.er0);     di  += (G(j, b.er0)   - G(j, b.er0 - 1)) / DR; }
                    if (b.er1 - 1 != NR - 1) { uo += G(j, b.er1 - 1); doo += (G(j, b.er1)   - G(j, b.er1 - 1)) / DR; }
                }
                if (b.er0     != 0)      gin[mb]  = alpha * (ui / n) - (di / n);   // inner edge: alpha*Az - dAz/dn
                if (b.er1 - 1 != NR - 1) gout[mb] = alpha * (uo / n) + (doo / n);  // outer edge: alpha*Az + dAz/dn
            }
            if (b.er0     != 0)      b.an->setBoundaryProfile("inner", gin);
            if (b.er1 - 1 != NR - 1) b.an->setBoundaryProfile("outer", gout);

            // Warm start: sample the composite G into the band's (coarse) grid.
            Eigen::MatrixXd pAz(b.nth, b.nrb);
            for (int mb = 0; mb < b.nth; ++mb)
                for (int kb = 0; kb < b.nrb; ++kb) {
                    const int j = mb * b.cft + b.cft / 2;
                    const int c = b.er0 + (int)std::llround((double)kb * (b.er1 - 1 - b.er0) / (b.nrb - 1));
                    pAz(mb, kb) = G(std::min(j, NTH - 1), std::min(c, NR - 1));
                }
            b.an->setAz(pAz);
            b.an->solve();
            const Eigen::MatrixXd& sol = b.an->getAz();

            // Write the band CORE [c0, c1) back into the composite G (NN upsample).
            for (int j = 0; j < NTH; ++j) {
                int mb = j / b.cft;
                if (mb >= b.nth) mb = b.nth - 1;
                for (int c = b.c0; c < b.c1; ++c) {
                    int kb = (int)std::llround((double)(c - b.er0) * (b.nrb - 1) / (b.er1 - 1 - b.er0));
                    kb = std::max(0, std::min(kb, b.nrb - 1));
                    G(j, c) = sol(mb, kb);
                }
            }
        }
        if (omega != 1.0) G = Gprev + omega * (G - Gprev);   // outer under-relaxation (damps a 2-cycle)
        res = (G - Gprev).norm() / (G.norm() + 1e-30);       // relative Schwarz residual (no reference needed)
        std::cout << "  DD sweep " << it << ": residual = " << res << std::endl;
        if (res < tol) break;
    }
    const int sweeps_done = std::min(it, maxo);  // 'it' is maxo+1 if the loop ran to the cap without converging
    std::cout << "=== DD done: " << sweeps_done << " sweep(s), final residual = " << res
              << (res < tol ? " (converged)" : " (reached max_outer)") << " ===" << std::endl;

    // Composite solution -> member Az, then refresh B/H/mu on the full grid so the
    // Mu/H exports, energy, and Maxwell-stress are consistent with the composite
    // (same sequence buildPolarOperator uses to refresh from a given Az).
    Az = G;
    calculateMagneticFieldPolar();
    calculateHField();
    updateMuDistribution();
}
