// Shared full-grid equations. Extracted without changing boundary/source laws
// from the v1.6.2 Cartesian and polar assemblers. Tests retain the original
// triplet implementations as independent regression oracles.
// MuAccessor permits both ordinary coefficients and differentiated material
// coefficients; it must use the same grid/orientation as the production fields.

template<class Scalar, class MuAccessor>
omfdm::StencilRow<Scalar> MagneticFieldAnalyzer::cartesianRow(int k, const MuAccessor& mu) const {
    omfdm::StencilRow<Scalar> row;
    const bool x_periodic = bc_left.type == "periodic" && bc_right.type == "periodic";
    const bool y_periodic = bc_bottom.type == "periodic" && bc_top.type == "periodic";
        int j = k / nx, i = k % nx;
        int idx = k;

            bool is_left = (i == 0);
            bool is_right = (i == nx - 1);
            bool is_bottom = (j == 0);
            bool is_top = (j == ny - 1);

            // Dirichlet boundary conditions (skip if periodic)
            if (is_left && bc_left.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_left.value;
                return row;
            } else if (is_right && bc_right.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_right.value;
                return row;
            } else if (is_bottom && bc_bottom.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_bottom.value;
                return row;
            } else if (is_top && bc_top.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_top.value;
                return row;
            }

            // Robin boundary conditions: alpha*Az + beta*(dAz/dn) = gamma
            // Left boundary (i=0): outward normal is -x direction
            // dAz/dn = -dAz/dx ≈ (Az(0,j) - Az(1,j))/dx
            // => (alpha + beta/dx)*Az(0,j) - (beta/dx)*Az(1,j) = gamma
            if (is_left && bc_left.type == "robin") {
                double a = bc_left.alpha;
                double b = bc_left.beta;
                double g = bc_left.gamma;
                row.add(idx, a + b/dx);
                row.add(idx + 1, -b/dx);
                row.rhs = g;
                return row;
            }
            // Right boundary (i=nx-1): outward normal is +x direction
            // dAz/dn = dAz/dx ≈ (Az(nx-1,j) - Az(nx-2,j))/dx
            // => (alpha + beta/dx)*Az(nx-1,j) - (beta/dx)*Az(nx-2,j) = gamma
            else if (is_right && bc_right.type == "robin") {
                double a = bc_right.alpha;
                double b = bc_right.beta;
                double g = bc_right.gamma;
                row.add(idx, a + b/dx);
                row.add(idx - 1, -b/dx);
                row.rhs = g;
                return row;
            }
            // Bottom boundary (j=0): outward normal is -y direction
            // dAz/dn = -dAz/dy ≈ (Az(i,0) - Az(i,1))/dy
            // => (alpha + beta/dy)*Az(i,0) - (beta/dy)*Az(i,1) = gamma
            else if (is_bottom && bc_bottom.type == "robin") {
                double a = bc_bottom.alpha;
                double b = bc_bottom.beta;
                double g = bc_bottom.gamma;
                row.add(idx, a + b/dy);
                row.add(idx + nx, -b/dy);
                row.rhs = g;
                return row;
            }
            // Top boundary (j=ny-1): outward normal is +y direction
            // dAz/dn = dAz/dy ≈ (Az(i,ny-1) - Az(i,ny-2))/dy
            // => (alpha + beta/dy)*Az(i,ny-1) - (beta/dy)*Az(i,ny-2) = gamma
            else if (is_top && bc_top.type == "robin") {
                double a = bc_top.alpha;
                double b = bc_top.beta;
                double g = bc_top.gamma;
                row.add(idx, a + b/dy);
                row.add(idx - nx, -b/dy);
                row.rhs = g;
                return row;
            }

            // Interior points (including periodic boundary points) - finite difference stencil
            Scalar coeff_center = 0.0;

            // X-direction terms
            // Determine left and right neighbor indices
            int i_west = i - 1;
            int i_east = i + 1;

            // Handle periodic boundaries in X direction
            if (x_periodic) {
                if (i == 0) i_west = nx - 1;  // Wrap to right edge
                if (i == nx - 1) i_east = 0;   // Wrap to left edge
            }

            // West neighbor (i-1 or wrapped)
            if (i > 0 || x_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                Scalar mu_center = mu(i, j);
                Scalar mu_neighbor = mu(i_west, j);
                Scalar mu_west = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                Scalar coeff_west = 1.0 / (mu_west * dx * dx);

                int idx_west = j * nx + i_west;

                // Check if left neighbor is Dirichlet boundary
                bool left_neighbor_is_dirichlet = (i == 1) && (bc_left.type == "dirichlet");

                if (!left_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    row.add(idx_west, coeff_west);
                } else {
                    // Left neighbor is Dirichlet: move bc value to RHS
                    row.rhs -= coeff_west * bc_left.value;
                }
                coeff_center -= coeff_west;
            }

            // East neighbor (i+1 or wrapped)
            if (i < nx - 1 || x_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                Scalar mu_center = mu(i, j);
                Scalar mu_neighbor = mu(i_east, j);
                Scalar mu_east = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                Scalar coeff_east = 1.0 / (mu_east * dx * dx);

                int idx_east = j * nx + i_east;

                // Check if right neighbor is Dirichlet boundary
                bool right_neighbor_is_dirichlet = (i == nx - 2) && (bc_right.type == "dirichlet");

                if (!right_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    row.add(idx_east, coeff_east);
                } else {
                    // Right neighbor is Dirichlet: move bc value to RHS
                    row.rhs -= coeff_east * bc_right.value;
                }
                coeff_center -= coeff_east;
            }

            // Y-direction terms
            // Determine bottom and top neighbor indices
            int j_south = j - 1;
            int j_north = j + 1;

            // Handle periodic boundaries in Y direction
            if (y_periodic) {
                if (j == 0) j_south = ny - 1;  // Wrap to top edge
                if (j == ny - 1) j_north = 0;   // Wrap to bottom edge
            }

            // South neighbor (j-1 or wrapped)
            if (j > 0 || y_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                Scalar mu_center = mu(i, j);
                Scalar mu_neighbor = mu(i, j_south);
                Scalar mu_south = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                Scalar coeff_south = 1.0 / (mu_south * dy * dy);

                int idx_south = j_south * nx + i;

                // Check if bottom neighbor is Dirichlet boundary
                bool bottom_neighbor_is_dirichlet = (j == 1) && (bc_bottom.type == "dirichlet");

                if (!bottom_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    row.add(idx_south, coeff_south);
                } else {
                    // Bottom neighbor is Dirichlet: move bc value to RHS
                    row.rhs -= coeff_south * bc_bottom.value;
                }
                coeff_center -= coeff_south;
            }

            // North neighbor (j+1 or wrapped)
            if (j < ny - 1 || y_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                Scalar mu_center = mu(i, j);
                Scalar mu_neighbor = mu(i, j_north);
                Scalar mu_north = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                Scalar coeff_north = 1.0 / (mu_north * dy * dy);

                int idx_north = j_north * nx + i;

                // Check if top neighbor is Dirichlet boundary
                bool top_neighbor_is_dirichlet = (j == ny - 2) && (bc_top.type == "dirichlet");

                if (!top_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    row.add(idx_north, coeff_north);
                } else {
                    // Top neighbor is Dirichlet: move bc value to RHS
                    row.rhs -= coeff_north * bc_top.value;
                }
                coeff_center -= coeff_north;
            }

            // Center coefficient
            row.add(idx, coeff_center);

            // Right-hand side (current density + magnetization equivalent current)
            // Jz_mag_map = dMy/dx - dMx/dy (equivalent current from magnetization)
            // Use += to preserve any accumulated Dirichlet boundary contributions
            row.rhs += -(jz_map(j, i) + Jz_mag_map(j, i));
    return row;
}

template<class Scalar, class MuAccessor>
omfdm::StencilRow<Scalar> MagneticFieldAnalyzer::polarRow(int k, const MuAccessor& mu) const {
    omfdm::StencilRow<Scalar> row;
    const bool is_periodic = bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic";
    const bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    const double periodic_sign = is_antiperiodic ? -1.0 : 1.0;
    auto faceMu = [&](int low, int j) -> Scalar {
        if (low >= 0 && low < nr - 1)
            return 2.0 / (1.0 / mu(low, j) + 1.0 / mu(low + 1, j));
        return mu(std::min(std::max(low, 0), nr - 1), j);
    };
        int i = k / ntheta, j = k % ntheta;
        int idx = k;
            double r = r_coords[i];

            // Angular boundary conditions (only for non-periodic boundaries)
            if (!is_periodic) {
                // Handle theta_min boundary (j == 0); profile[i] (per-r) overrides scalar
                if (j == 0 && bc_theta_min.type == "dirichlet") {
                    row.add(idx, 1.0);
                    row.rhs = bc_theta_min.profile.empty() ? bc_theta_min.value
                                                            : bc_theta_min.profile[i];
                    return row;
                }
                // Handle theta_max boundary (j == ntheta - 1); profile[i] overrides scalar
                if (j == ntheta - 1 && bc_theta_max.type == "dirichlet") {
                    row.add(idx, 1.0);
                    row.rhs = bc_theta_max.profile.empty() ? bc_theta_max.value
                                                            : bc_theta_max.profile[i];
                    return row;
                }
            }

            // Radial boundary conditions (Dirichlet only handled here)
            // Neumann boundaries use ghost-elimination in interior stencil below
            // v1.6 DD: profile[j] overrides the scalar along the boundary if present.
            if (i == 0 && bc_inner.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_inner.profile.empty() ? bc_inner.value : bc_inner.profile[j];
                return row;
            }

            if (i == nr - 1 && bc_outer.type == "dirichlet") {
                row.add(idx, 1.0);
                row.rhs = bc_outer.profile.empty() ? bc_outer.value : bc_outer.profile[j];
                return row;
            }

            // Radial Robin boundary conditions are assembled SYMMETRICALLY inside
            // the radial stencil below (v1.6 DD / optimized Schwarz). The boundary
            // node keeps its full FV balance (kept radial face + theta coupling +
            // source); only the boundary face flux is replaced by the Robin flux
            //   r*nu/dr * (gamma - alpha*Az_B)/beta,
            // which adds a symmetric -c*alpha/beta diagonal term (c=r*nu/dr) and an
            // -c*gamma/beta RHS term. Off-diagonals then match the interior coupling
            // -> SPD M-matrix (AMGCL-CG works). The OLD node-replacing form
            //   (alpha+beta/dr)Az_B-(beta/dr)Az_{B-1}=gamma was non-symmetric +
            //   badly scaled (CG diverged) and dropped the node's source/theta terms.

            // Interior points and Neumann boundary points (using ghost-elimination)
            // Polar Poisson equation with variable permeability (divergence form):
            // (1/r) ∂/∂r(r · (1/μ) · ∂Az/∂r) + (1/r²) ∂/∂θ((1/μ) · ∂Az/∂θ) = -Jz
            //
            // Discretization using cell-centered finite differences with interface fluxes:
            // F_{i+1/2} = (1/μ_{i+1/2}) · (Az_{i+1} - Az_i) / dr
            // F_{i-1/2} = (1/μ_{i-1/2}) · (Az_i - Az_{i-1}) / dr
            //
            // Radial term: (1/r_i) · [r_{i+1/2} · F_{i+1/2} - r_{i-1/2} · F_{i-1/2}] / dr

            Scalar coeff_center = 0.0;

            // Get permeability at interfaces
            Scalar mu_inner = faceMu(i - 1, j);  // μ_{i-1/2}
            Scalar mu_outer = faceMu(i, j);  // μ_{i+1/2}
            Scalar mu_current = mu(i, j);

            // Radial coefficients with r-weighting (divergence form)
            //  Use r-weighted formulation for symmetry
            // Multiply equation by r to get: ∂/∂r(r·(1/μ)·∂Az/∂r) + (1/r)·∂/∂θ((1/μ)·∂Az/∂θ) = -r·Jz
            // This makes the radial term symmetric
            double r_iph = r + 0.5 * dr;  // r_{i+1/2}
            double r_imh = r - 0.5 * dr;  // r_{i-1/2}

            Scalar a_ip = r_iph / (mu_outer * dr * dr);  // neighbor i+1 (removed /r for symmetry)
            Scalar a_im = r_imh / (mu_inner * dr * dr);  // neighbor i-1 (removed /r for symmetry)

            // Apply ghost-elimination for Neumann boundaries
            // Neumann BC: dAz/dr = 0 → ghost point equals opposite neighbor
            //   Inner (i=0): Az_{-1} = Az_1 → eliminate Az_{-1}, add a_im to a_ip
            //   Outer (i=nr-1): Az_{nr} = Az_{nr-2} → eliminate Az_{nr}, add a_ip to a_im

            if (i == 0 && bc_inner.type == "neumann") {
                // Ghost elimination: Az_{-1} = Az_1
                // Check if r_imh is positive (should be caught by setupPolarSystem validation)
                if (r_imh <= 0.0) {
                    // Fallback: use symmetry assumption (mirror boundary)
                    // Treat as if r_{-1/2} = r_{+1/2} for stability
                    std::cerr << "Warning: r_imh <= 0 at inner Neumann BC (r=" << r << "), using mirror approximation" << std::endl;
                    double r_imh_eff = r_iph;  // Mirror approximation
                    Scalar a_im_eff = r_imh_eff / (mu_inner * dr * dr);
                    row.add((i + 1) * ntheta + j, a_im_eff + a_ip);
                    coeff_center -= (a_im_eff + a_ip);
                } else {
                    // Standard ghost elimination
                    row.add((i + 1) * ntheta + j, a_im + a_ip);
                    coeff_center -= (a_im + a_ip);
                }
            } else if (i == 0 && bc_inner.type == "robin") {
                // Symmetric Robin (inner, outward normal -r): keep the OUTER radial
                // face (-> node i+1, same a_ip as interior) and replace the inner
                // face flux by the Robin flux r*nu/dr*(gamma-alpha*Az)/beta. Adds a
                // symmetric -c*alpha/beta diagonal and -c*gamma/beta RHS (c=r*nu/dr).
                // theta coupling + source are added by the common code below.
                double a = bc_inner.alpha, b = bc_inner.beta;
                double g = bc_inner.profile.empty() ? bc_inner.gamma : bc_inner.profile[j];
                Scalar c = r / (mu_current * dr);   // r_0 * nu_0 / dr
                row.add((i + 1) * ntheta + j, a_ip);
                coeff_center -= a_ip;
                coeff_center -= c * a / b;
                row.rhs -= c * g / b;
            } else if (i == nr - 1 && bc_outer.type == "robin") {
                // Symmetric Robin (outer, outward normal +r): keep the INNER radial
                // face (-> node i-1, same a_im as interior) and replace the outer
                // face flux by the Robin flux. Symmetric -> SPD (AMGCL-CG works).
                double a = bc_outer.alpha, b = bc_outer.beta;
                double g = bc_outer.profile.empty() ? bc_outer.gamma : bc_outer.profile[j];
                Scalar c = r / (mu_current * dr);   // r_B * nu_B / dr
                row.add((i - 1) * ntheta + j, a_im);
                coeff_center -= a_im;
                coeff_center -= c * a / b;
                row.rhs -= c * g / b;
            } else if (i == nr - 1 && bc_outer.type == "neumann") {
                // Ghost elimination: Az_{nr} = Az_{nr-2}
                // Stencil: a_im·Az_{nr-2} + a_ip·Az_{nr} → (a_im + a_ip)·Az_{nr-2}
                row.add((i - 1) * ntheta + j, a_im + a_ip);
                coeff_center -= (a_im + a_ip);
            } else {
                // Standard interior stencil
                // Check if neighbors are Dirichlet boundaries and move bc values to RHS
                bool inner_neighbor_is_dirichlet = (i == 1) && (bc_inner.type == "dirichlet");
                bool outer_neighbor_is_dirichlet = (i == nr - 2) && (bc_outer.type == "dirichlet");

                if (!inner_neighbor_is_dirichlet) {
                    row.add((i - 1) * ntheta + j, a_im);
                } else {
                    // Inner neighbor is Dirichlet: move bc value to RHS (profile[j] if present)
                    double bv = bc_inner.profile.empty() ? bc_inner.value : bc_inner.profile[j];
                    row.rhs -= a_im * bv;
                }

                if (!outer_neighbor_is_dirichlet) {
                    row.add((i + 1) * ntheta + j, a_ip);
                } else {
                    // Outer neighbor is Dirichlet: move bc value to RHS (profile[j] if present)
                    double bv = bc_outer.profile.empty() ? bc_outer.value : bc_outer.profile[j];
                    row.rhs -= a_ip * bv;
                }

                coeff_center -= (a_im + a_ip);
            }

            // Angular second derivative term: (1/r²) · (1/μ) · ∂²Az/∂θ²
            // Simplified using cell-centered μ (for θ-varying μ, use interface values)
            // Scalar coeff_theta = 1.0 / (r * r * mu_current * dtheta * dtheta);

            // v1.6 DD: symmetric theta-Robin at a theta-sector boundary node. Keep the interior
            // theta face; replace the boundary theta face by the Robin flux (c_th = 1/(r*mu*dtheta)),
            // adding a symmetric -c_th*alpha/beta diagonal + -c_th*gamma/beta RHS. profile[i] is per-r.
            if (!is_periodic && j == 0 && bc_theta_min.type == "robin") {
                Scalar mu_ij = mu(i, j);
                Scalar mu_next = mu(i, j + 1);
                Scalar mu_tn = 2.0 / (1.0 / mu_ij + 1.0 / mu_next);
                Scalar ct_next = 1.0 / (r * mu_tn * dtheta * dtheta);
                row.add(i * ntheta + (j + 1), ct_next);
                coeff_center -= ct_next;
                double a = bc_theta_min.alpha, b = bc_theta_min.beta;
                double g = bc_theta_min.profile.empty() ? bc_theta_min.gamma : bc_theta_min.profile[i];
                Scalar c_th = 1.0 / (r * mu_current * dtheta);
                coeff_center -= c_th * a / b;
                row.rhs -= c_th * g / b;
            } else if (!is_periodic && j == ntheta - 1 && bc_theta_max.type == "robin") {
                Scalar mu_ij = mu(i, j);
                Scalar mu_prev = mu(i, j - 1);
                Scalar mu_tp = 2.0 / (1.0 / mu_ij + 1.0 / mu_prev);
                Scalar ct_prev = 1.0 / (r * mu_tp * dtheta * dtheta);
                row.add(i * ntheta + (j - 1), ct_prev);
                coeff_center -= ct_prev;
                double a = bc_theta_max.alpha, b = bc_theta_max.beta;
                double g = bc_theta_max.profile.empty() ? bc_theta_max.gamma : bc_theta_max.profile[i];
                Scalar c_th = 1.0 / (r * mu_current * dtheta);
                coeff_center -= c_th * a / b;
                row.rhs -= c_th * g / b;
            } else {
            // Determine neighbor indices for angular direction
            // For periodic boundary: use modulo wrapping (% ntheta)
            // For sector domain: boundaries already handled, interior points use simple ±1
            int j_prev_idx = (j - 1 + ntheta) % ntheta;
            int j_next_idx = (j + 1) % ntheta;

            // For non-periodic (sector) do not wrap — ensure j_prev/j_next are interior
            if (!is_periodic) {
                j_prev_idx = j - 1;
                j_next_idx = j + 1;
            }

            // harmonic mean of mu between (i,j) and (i,j_prev)
            Scalar mu_ij = mu(i, j);
            Scalar mu_prev = mu(i, j_prev_idx);
            Scalar mu_next = mu(i, j_next_idx);
            Scalar mu_theta_prev = 2.0 / (1.0 / mu_ij + 1.0 / mu_prev);
            // harmonic mean of mu between (i,j) and (i,j_next)
            Scalar mu_theta_next = 2.0 / (1.0 / mu_ij + 1.0 / mu_next);

            // coeffs for interface flux (symmetric)
            // After r-weighting: (1/r²) term becomes (1/r)
            Scalar coeff_theta_prev = 1.0 / (r * mu_theta_prev * dtheta * dtheta);
            Scalar coeff_theta_next = 1.0 / (r * mu_theta_next * dtheta * dtheta);

            // Check if neighbors are theta Dirichlet boundaries
            // For non-periodic domain: j=0 (theta_min) and j=ntheta-1 (theta_max) may have Dirichlet BC
            bool theta_prev_is_dirichlet = (!is_periodic) && (j == 1) && (bc_theta_min.type == "dirichlet");
            bool theta_next_is_dirichlet = (!is_periodic) && (j == ntheta - 2) && (bc_theta_max.type == "dirichlet");

            // For (anti-)periodic BC, check if coupling crosses the theta boundary
            // j=0 couples to j_prev=ntheta-1 (crosses boundary)
            // j=ntheta-1 couples to j_next=0 (crosses boundary)
            bool prev_crosses_boundary = is_periodic && (j == 0);
            bool next_crosses_boundary = is_periodic && (j == ntheta - 1);

            if (!theta_prev_is_dirichlet) {
                double sign = prev_crosses_boundary ? periodic_sign : 1.0;
                row.add(i * ntheta + j_prev_idx, sign * coeff_theta_prev);
            } else {
                // Theta_min neighbor is Dirichlet: move bc value to RHS (profile[i] per-r if present)
                double bv = bc_theta_min.profile.empty() ? bc_theta_min.value : bc_theta_min.profile[i];
                row.rhs -= coeff_theta_prev * bv;
            }

            if (!theta_next_is_dirichlet) {
                double sign = next_crosses_boundary ? periodic_sign : 1.0;
                row.add(i * ntheta + j_next_idx, sign * coeff_theta_next);
            } else {
                // Theta_max neighbor is Dirichlet: move bc value to RHS (profile[i] per-r if present)
                double bv = bc_theta_max.profile.empty() ? bc_theta_max.value : bc_theta_max.profile[i];
                row.rhs -= coeff_theta_next * bv;
            }

            coeff_center -= (coeff_theta_prev + coeff_theta_next);
            }  // end else (interior / theta-Dirichlet theta term; theta-Robin handled above)

            // Center coefficient
            row.add(idx, coeff_center);

            // Right-hand side (current density + magnetization equivalent current)
            // After r-weighting: RHS becomes -r·(Jz + Jz_mag)
            // Jz_mag_map = polar curl of M (1/r·d(r·Mθ)/dr - 1/r·dMr/dθ)
            // Use += to preserve any accumulated Dirichlet boundary contributions
            row.rhs += -(getJzPolar(jz_map, i, j, r_orientation) + getJzPolar(Jz_mag_map, i, j, r_orientation)) * r;
    // Preserve the historical additive nullspace pin, including its RHS.
    if (k == 0 && bc_inner.type == "neumann" && bc_outer.type == "neumann" && is_periodic) {
        row.add(0, 1.0);
        row.rhs = 0.0;
    }
    return row;
}
