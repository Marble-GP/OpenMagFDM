#ifndef OPENMAGFDM_AMPERIAN_FORCE_H
#define OPENMAGFDM_AMPERIAN_FORCE_H

#include "PolarMagnetizationCurl.h"
#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

// The surrounding Windows Release build uses /fp:fast. Preserve summation
// compensation and finite-value checks in this small postprocessing kernel.
#ifdef _MSC_VER
#pragma float_control(precise, on, push)
#endif

namespace openmagfdm::detail {

// Normalized storage: rows = y/theta, columns = x/r. All inputs belong to
// ONE solved state. deltaNu, mx, my and freeJ contain ONLY the selected body;
// Az and B contain the field of the complete system, including that body.
struct AmperianGrid {
    bool polar = false;
    double h1 = 1.0, h2 = 1.0, rStart = 0.0, thetaOffset = 0.0;
    bool periodic1 = false, periodic2 = false;
    bool anti1 = false, anti2 = false;
};

inline Eigen::MatrixXd amperianCurrent(
    const Eigen::MatrixXd& az, const Eigen::MatrixXd& deltaNu,
    const Eigen::MatrixXd& mx, const Eigen::MatrixXd& my,
    const Eigen::MatrixXd& freeJ, const AmperianGrid& g) {
    const int nx = static_cast<int>(az.cols()), ny = static_cast<int>(az.rows());
    if (nx < 3 || ny < 3 || !(g.h1 > 0) || !(g.h2 > 0)
        || (g.polar && (g.rStart < 0 || g.periodic1))
        || (g.anti1 && !g.periodic1) || (g.anti2 && !g.periodic2))
        throw std::invalid_argument("Invalid Amperian force grid");
    for (const auto* m : {&az, &deltaNu, &mx, &my, &freeJ})
        if (m->rows() != ny || m->cols() != nx || !m->allFinite())
            throw std::invalid_argument("Invalid Amperian force arrays");

    // No extrapolation of a magnetized body beyond a physical boundary:
    // an enclosing force integral would need the missing exterior field.
    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        const bool edge = (!g.periodic1 && (x == 0 || x == nx - 1))
                       || (!g.periodic2 && (y == 0 || y == ny - 1));
        if (edge && (deltaNu(y,x) != 0 || mx(y,x) != 0 || my(y,x) != 0))
            throw std::invalid_argument("Amperian force requires the magnetized body to be inside the physical domain; enlarge the domain or select an interior body");
    }

    auto sample = [&](const Eigen::MatrixXd& a, int y, int x, bool signedField) {
        double sign = 1;
        if (g.periodic1) {
            if (x < 0 || x >= nx) { x = (x + nx) % nx; if (g.anti1 && signedField) sign = -sign; }
        } else x = std::clamp(x, 0, nx - 1);
        if (g.periodic2) {
            if (y < 0 || y >= ny) { y = (y + ny) % ny; if (g.anti2 && signedField) sign = -sign; }
        } else y = std::clamp(y, 0, ny - 1);
        return sign * a(y,x);
    };

    Eigen::MatrixXd current = freeJ;
    if (g.polar) {
        PolarMagnetizationCurlOptions p;
        p.nr = nx; p.ntheta = ny; p.r_start = g.rStart;
        p.dr = g.h1; p.dtheta = g.h2; p.theta_offset = g.thetaOffset;
        p.theta_periodic = g.periodic2; p.theta_antiperiodic = g.anti2;
        current += computePolarMagnetizationCurl(mx, my, p);
    }

    for (int y = 0; y < ny; ++y) for (int x = 0; x < nx; ++x) {
        const double r = g.polar ? g.rStart + x * g.h1 : 1.0;
        if (r <= 1e-10) continue; // Same axis identity row as the field solver.
        const double a = az(y,x), d = deltaNu(y,x);
        // mu_face = harmonic mean(mu), hence nu_face = arithmetic mean(nu).
        // curl(M_induced) = div((nu - nu0) grad(Az)). Using the solver's
        // face gradient avoids differentiating a discontinuous cell-centre B
        // twice. Signed seam continuation applies to A/M, NEVER permeability.
        const double qp = .5 * (d + sample(deltaNu,y,x+1,false)) * (sample(az,y,x+1,true) - a);
        const double qm = .5 * (d + sample(deltaNu,y,x-1,false)) * (a - sample(az,y,x-1,true));
        const double tp = .5 * (d + sample(deltaNu,y+1,x,false)) * (sample(az,y+1,x,true) - a);
        const double tm = .5 * (d + sample(deltaNu,y-1,x,false)) * (a - sample(az,y-1,x,true));
        const double rp = g.polar ? r + .5*g.h1 : 1;
        const double rm = g.polar ? r - .5*g.h1 : 1;
        current(y,x) += (rp*qp - rm*qm)/(r*g.h1*g.h1)
                     + (tp-tm)/(r*r*g.h2*g.h2);
        if (!g.polar) {
            const double dx = (!g.periodic1 && (x == 0 || x == nx-1)) ? g.h1 : 2*g.h1;
            const double dy = (!g.periodic2 && (y == 0 || y == ny-1)) ? g.h2 : 2*g.h2;
            current(y,x) += (sample(my,y,x+1,true)-sample(my,y,x-1,true))/dx
                          - (sample(mx,y+1,x,true)-sample(mx,y-1,x,true))/dy;
        }
    }
    return current;
}

struct AmperianLoad { double fx = 0, fy = 0, torque = 0, torqueCenter = 0; };

inline AmperianLoad integrateAmperianCurrent(
    const Eigen::MatrixXd& current, const Eigen::MatrixXd& b1,
    const Eigen::MatrixXd& b2, const AmperianGrid& g) {
    if (b1.rows()!=current.rows() || b1.cols()!=current.cols()
        || b2.rows()!=current.rows() || b2.cols()!=current.cols()
        || !current.allFinite() || !b1.allFinite() || !b2.allFinite())
        throw std::invalid_argument("Invalid Amperian integration arrays");
    // Compensated summation matters when large opposing surface currents cancel.
    struct Sum { double value=0, correction=0;
        void add(double v) { const double t=value+v;
            correction += std::abs(value)>=std::abs(v) ? (value-t)+v : (v-t)+value; value=t; }
        double get() const { return value+correction; }
    } fx, fy, torque, tc;
    const double cx = g.polar ? 0 : (current.cols()-1)*g.h1/2;
    const double cy = g.polar ? 0 : (current.rows()-1)*g.h2/2;
    for (int y=0; y<current.rows(); ++y) for (int x=0; x<current.cols(); ++x) {
        const double r = g.polar ? g.rStart+x*g.h1 : 1;
        const double volume = r*g.h1*g.h2;
        double px=x*g.h1, py=y*g.h2;
        double dfx=-current(y,x)*b2(y,x)*volume, dfy=current(y,x)*b1(y,x)*volume;
        if (g.polar) {
            const double theta=g.thetaOffset+y*g.h2, c=std::cos(theta), s=std::sin(theta);
            px=r*c; py=r*s;
            const double fr=dfx, ft=dfy; dfx=fr*c-ft*s; dfy=fr*s+ft*c;
        }
        fx.add(dfx); fy.add(dfy);
        torque.add(px*dfy-py*dfx); tc.add((px-cx)*dfy-(py-cy)*dfx);
    }
    return {fx.get(),fy.get(),torque.get(),tc.get()};
}
} // namespace openmagfdm::detail
#ifdef _MSC_VER
#pragma float_control(pop)
#endif
#endif
