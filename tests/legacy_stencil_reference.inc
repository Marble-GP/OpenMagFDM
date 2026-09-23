// Frozen v1.6.2 oracle (25859a7); compiled only into the numerical tests.
void MagneticFieldAnalyzer::buildMatrixReference(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
    // Build FDM system matrix and right-hand side (Cartesian coordinates)
    int n = nx * ny;
    A.resize(n, n);
    rhs.resize(n);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(5 * n);  // Estimate: 5 non-zeros per row

    // Check for periodic boundary conditions
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    // Build equation for each grid point.
    // Thread-local triplets are merged after the loop; rhs(idx) writes are unique per idx.
    // flat k = j*nx+i for MSVC OpenMP 2.0 compatibility.
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> local_triplets;
        int nth = 1;
        #ifdef _OPENMP
        nth = omp_get_num_threads();
        #endif
        local_triplets.reserve(5 * n / (1 + nth) + 16);

    #pragma omp for schedule(static)
    for (int k = 0; k < ny * nx; k++) {
        int j = k / nx, i = k % nx;
        int idx = k;

            bool is_left = (i == 0);
            bool is_right = (i == nx - 1);
            bool is_bottom = (j == 0);
            bool is_top = (j == ny - 1);

            // Dirichlet boundary conditions (skip if periodic)
            if (is_left && bc_left.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_left.value;
                continue;
            } else if (is_right && bc_right.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_right.value;
                continue;
            } else if (is_bottom && bc_bottom.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_bottom.value;
                continue;
            } else if (is_top && bc_top.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_top.value;
                continue;
            }

            // Robin boundary conditions: alpha*Az + beta*(dAz/dn) = gamma
            // Left boundary (i=0): outward normal is -x direction
            // dAz/dn = -dAz/dx ≈ (Az(0,j) - Az(1,j))/dx
            // => (alpha + beta/dx)*Az(0,j) - (beta/dx)*Az(1,j) = gamma
            if (is_left && bc_left.type == "robin") {
                double a = bc_left.alpha;
                double b = bc_left.beta;
                double g = bc_left.gamma;
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dx));
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx + 1, -b/dx));
                rhs(idx) = g;
                continue;
            }
            // Right boundary (i=nx-1): outward normal is +x direction
            // dAz/dn = dAz/dx ≈ (Az(nx-1,j) - Az(nx-2,j))/dx
            // => (alpha + beta/dx)*Az(nx-1,j) - (beta/dx)*Az(nx-2,j) = gamma
            else if (is_right && bc_right.type == "robin") {
                double a = bc_right.alpha;
                double b = bc_right.beta;
                double g = bc_right.gamma;
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dx));
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx - 1, -b/dx));
                rhs(idx) = g;
                continue;
            }
            // Bottom boundary (j=0): outward normal is -y direction
            // dAz/dn = -dAz/dy ≈ (Az(i,0) - Az(i,1))/dy
            // => (alpha + beta/dy)*Az(i,0) - (beta/dy)*Az(i,1) = gamma
            else if (is_bottom && bc_bottom.type == "robin") {
                double a = bc_bottom.alpha;
                double b = bc_bottom.beta;
                double g = bc_bottom.gamma;
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dy));
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx + nx, -b/dy));
                rhs(idx) = g;
                continue;
            }
            // Top boundary (j=ny-1): outward normal is +y direction
            // dAz/dn = dAz/dy ≈ (Az(i,ny-1) - Az(i,ny-2))/dy
            // => (alpha + beta/dy)*Az(i,ny-1) - (beta/dy)*Az(i,ny-2) = gamma
            else if (is_top && bc_top.type == "robin") {
                double a = bc_top.alpha;
                double b = bc_top.beta;
                double g = bc_top.gamma;
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dy));
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx - nx, -b/dy));
                rhs(idx) = g;
                continue;
            }

            // Interior points (including periodic boundary points) - finite difference stencil
            double coeff_center = 0.0;

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
                double mu_center = muAtGrid(i, j);
                double mu_neighbor = muAtGrid(i_west, j);
                double mu_west = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                double coeff_west = 1.0 / (mu_west * dx * dx);

                int idx_west = j * nx + i_west;

                // Check if left neighbor is Dirichlet boundary
                bool left_neighbor_is_dirichlet = (i == 1) && (bc_left.type == "dirichlet");

                if (!left_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx_west, coeff_west));
                } else {
                    // Left neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= coeff_west * bc_left.value;
                }
                coeff_center -= coeff_west;
            }

            // East neighbor (i+1 or wrapped)
            if (i < nx - 1 || x_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                double mu_center = muAtGrid(i, j);
                double mu_neighbor = muAtGrid(i_east, j);
                double mu_east = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                double coeff_east = 1.0 / (mu_east * dx * dx);

                int idx_east = j * nx + i_east;

                // Check if right neighbor is Dirichlet boundary
                bool right_neighbor_is_dirichlet = (i == nx - 2) && (bc_right.type == "dirichlet");

                if (!right_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx_east, coeff_east));
                } else {
                    // Right neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= coeff_east * bc_right.value;
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
                double mu_center = muAtGrid(i, j);
                double mu_neighbor = muAtGrid(i, j_south);
                double mu_south = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                double coeff_south = 1.0 / (mu_south * dy * dy);

                int idx_south = j_south * nx + i;

                // Check if bottom neighbor is Dirichlet boundary
                bool bottom_neighbor_is_dirichlet = (j == 1) && (bc_bottom.type == "dirichlet");

                if (!bottom_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx_south, coeff_south));
                } else {
                    // Bottom neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= coeff_south * bc_bottom.value;
                }
                coeff_center -= coeff_south;
            }

            // North neighbor (j+1 or wrapped)
            if (j < ny - 1 || y_periodic) {
                // Compute harmonic mean with wrapped neighbor for periodic BC
                double mu_center = muAtGrid(i, j);
                double mu_neighbor = muAtGrid(i, j_north);
                double mu_north = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
                double coeff_north = 1.0 / (mu_north * dy * dy);

                int idx_north = j_north * nx + i;

                // Check if top neighbor is Dirichlet boundary
                bool top_neighbor_is_dirichlet = (j == ny - 2) && (bc_top.type == "dirichlet");

                if (!top_neighbor_is_dirichlet) {
                    // Normal interior-interior or periodic coupling
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx_north, coeff_north));
                } else {
                    // Top neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= coeff_north * bc_top.value;
                }
                coeff_center -= coeff_north;
            }

            // Center coefficient
            local_triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

            // Right-hand side (current density + magnetization equivalent current)
            // Jz_mag_map = dMy/dx - dMx/dy (equivalent current from magnetization)
            // Use += to preserve any accumulated Dirichlet boundary contributions
            rhs(idx) += -(jz_map(j, i) + Jz_mag_map(j, i));
        }  // end omp for

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
    }  // end omp parallel

    // Construct sparse matrix
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

void MagneticFieldAnalyzer::buildMatrixPolarReference(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
    // Build FDM system matrix and right-hand side (Polar coordinates)
    int n = nr * ntheta;
    A.resize(n, n);
    rhs.resize(n);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 7);  // Estimate: 7 non-zeros per row

    // Determine boundary condition type from YAML settings
    // theta is periodic only if both theta_min and theta_max are set to "periodic"
    // Anti-periodic: type="periodic" with value < 0
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    double periodic_sign = is_antiperiodic ? -1.0 : 1.0;  // Sign for coupling across theta boundary

    // Build equation for each grid point.
    // Thread-local triplets merged via critical section; rhs(idx) writes unique per idx.
    // flat k = i*ntheta+j for MSVC OpenMP 2.0 compatibility.
    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<double>> local_triplets;
        int nth = 1;
        #ifdef _OPENMP
        nth = omp_get_num_threads();
        #endif
        local_triplets.reserve(7 * n / (1 + nth) + 16);

    #pragma omp for schedule(static)
    for (int k = 0; k < nr * ntheta; k++) {  // Flat loop: radial × angular
        int i = k / ntheta, j = k % ntheta;
        int idx = k;
            double r = r_coords[i];

            // Angular boundary conditions (only for non-periodic boundaries)
            if (!is_periodic) {
                // Handle theta_min boundary (j == 0); profile[i] (per-r) overrides scalar
                if (j == 0 && bc_theta_min.type == "dirichlet") {
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_min.profile.empty() ? bc_theta_min.value
                                                            : bc_theta_min.profile[i];
                    continue;
                }
                // Handle theta_max boundary (j == ntheta - 1); profile[i] overrides scalar
                if (j == ntheta - 1 && bc_theta_max.type == "dirichlet") {
                    local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_max.profile.empty() ? bc_theta_max.value
                                                            : bc_theta_max.profile[i];
                    continue;
                }
            }

            // Radial boundary conditions (Dirichlet only handled here)
            // Neumann boundaries use ghost-elimination in interior stencil below
            // v1.6 DD: profile[j] overrides the scalar along the boundary if present.
            if (i == 0 && bc_inner.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_inner.profile.empty() ? bc_inner.value : bc_inner.profile[j];
                continue;
            }

            if (i == nr - 1 && bc_outer.type == "dirichlet") {
                local_triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_outer.profile.empty() ? bc_outer.value : bc_outer.profile[j];
                continue;
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

            double coeff_center = 0.0;

            // Get permeability at interfaces
            double mu_inner = getMuAtInterfacePolar(i - 0.5, j, "r");  // μ_{i-1/2}
            double mu_outer = getMuAtInterfacePolar(i + 0.5, j, "r");  // μ_{i+1/2}
            double mu_current = getMuPolar(mu_map, i, j, r_orientation);

            // Radial coefficients with r-weighting (divergence form)
            //  Use r-weighted formulation for symmetry
            // Multiply equation by r to get: ∂/∂r(r·(1/μ)·∂Az/∂r) + (1/r)·∂/∂θ((1/μ)·∂Az/∂θ) = -r·Jz
            // This makes the radial term symmetric
            double r_iph = r + 0.5 * dr;  // r_{i+1/2}
            double r_imh = r - 0.5 * dr;  // r_{i-1/2}

            double a_ip = r_iph / (mu_outer * dr * dr);  // neighbor i+1 (removed /r for symmetry)
            double a_im = r_imh / (mu_inner * dr * dr);  // neighbor i-1 (removed /r for symmetry)

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
                    double a_im_eff = r_imh_eff / (mu_inner * dr * dr);
                    local_triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_im_eff + a_ip));
                    coeff_center -= (a_im_eff + a_ip);
                } else {
                    // Standard ghost elimination
                    local_triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_im + a_ip));
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
                double c = r / (mu_current * dr);   // r_0 * nu_0 / dr
                local_triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_ip));
                coeff_center -= a_ip;
                coeff_center -= c * a / b;
                rhs(idx) -= c * g / b;
            } else if (i == nr - 1 && bc_outer.type == "robin") {
                // Symmetric Robin (outer, outward normal +r): keep the INNER radial
                // face (-> node i-1, same a_im as interior) and replace the outer
                // face flux by the Robin flux. Symmetric -> SPD (AMGCL-CG works).
                double a = bc_outer.alpha, b = bc_outer.beta;
                double g = bc_outer.profile.empty() ? bc_outer.gamma : bc_outer.profile[j];
                double c = r / (mu_current * dr);   // r_B * nu_B / dr
                local_triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, a_im));
                coeff_center -= a_im;
                coeff_center -= c * a / b;
                rhs(idx) -= c * g / b;
            } else if (i == nr - 1 && bc_outer.type == "neumann") {
                // Ghost elimination: Az_{nr} = Az_{nr-2}
                // Stencil: a_im·Az_{nr-2} + a_ip·Az_{nr} → (a_im + a_ip)·Az_{nr-2}
                local_triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, a_im + a_ip));
                coeff_center -= (a_im + a_ip);
            } else {
                // Standard interior stencil
                // Check if neighbors are Dirichlet boundaries and move bc values to RHS
                bool inner_neighbor_is_dirichlet = (i == 1) && (bc_inner.type == "dirichlet");
                bool outer_neighbor_is_dirichlet = (i == nr - 2) && (bc_outer.type == "dirichlet");

                if (!inner_neighbor_is_dirichlet) {
                    local_triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, a_im));
                } else {
                    // Inner neighbor is Dirichlet: move bc value to RHS (profile[j] if present)
                    double bv = bc_inner.profile.empty() ? bc_inner.value : bc_inner.profile[j];
                    rhs(idx) -= a_im * bv;
                }

                if (!outer_neighbor_is_dirichlet) {
                    local_triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_ip));
                } else {
                    // Outer neighbor is Dirichlet: move bc value to RHS (profile[j] if present)
                    double bv = bc_outer.profile.empty() ? bc_outer.value : bc_outer.profile[j];
                    rhs(idx) -= a_ip * bv;
                }

                coeff_center -= (a_im + a_ip);
            }

            // Angular second derivative term: (1/r²) · (1/μ) · ∂²Az/∂θ²
            // Simplified using cell-centered μ (for θ-varying μ, use interface values)
            // double coeff_theta = 1.0 / (r * r * mu_current * dtheta * dtheta);

            // v1.6 DD: symmetric theta-Robin at a theta-sector boundary node. Keep the interior
            // theta face; replace the boundary theta face by the Robin flux (c_th = 1/(r*mu*dtheta)),
            // adding a symmetric -c_th*alpha/beta diagonal + -c_th*gamma/beta RHS. profile[i] is per-r.
            if (!is_periodic && j == 0 && bc_theta_min.type == "robin") {
                double mu_ij = getMuPolar(mu_map, i, j, r_orientation);
                double mu_next = getMuPolar(mu_map, i, j + 1, r_orientation);
                double mu_tn = 2.0 / (1.0 / mu_ij + 1.0 / mu_next);
                double ct_next = 1.0 / (r * mu_tn * dtheta * dtheta);
                local_triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + (j + 1), ct_next));
                coeff_center -= ct_next;
                double a = bc_theta_min.alpha, b = bc_theta_min.beta;
                double g = bc_theta_min.profile.empty() ? bc_theta_min.gamma : bc_theta_min.profile[i];
                double c_th = 1.0 / (r * mu_current * dtheta);
                coeff_center -= c_th * a / b;
                rhs(idx) -= c_th * g / b;
            } else if (!is_periodic && j == ntheta - 1 && bc_theta_max.type == "robin") {
                double mu_ij = getMuPolar(mu_map, i, j, r_orientation);
                double mu_prev = getMuPolar(mu_map, i, j - 1, r_orientation);
                double mu_tp = 2.0 / (1.0 / mu_ij + 1.0 / mu_prev);
                double ct_prev = 1.0 / (r * mu_tp * dtheta * dtheta);
                local_triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + (j - 1), ct_prev));
                coeff_center -= ct_prev;
                double a = bc_theta_max.alpha, b = bc_theta_max.beta;
                double g = bc_theta_max.profile.empty() ? bc_theta_max.gamma : bc_theta_max.profile[i];
                double c_th = 1.0 / (r * mu_current * dtheta);
                coeff_center -= c_th * a / b;
                rhs(idx) -= c_th * g / b;
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
            double mu_ij = getMuPolar(mu_map, i, j, r_orientation);
            double mu_prev = getMuPolar(mu_map, i, j_prev_idx, r_orientation);
            double mu_next = getMuPolar(mu_map, i, j_next_idx, r_orientation);
            double mu_theta_prev = 2.0 / (1.0 / mu_ij + 1.0 / mu_prev);
            // harmonic mean of mu between (i,j) and (i,j_next)
            double mu_theta_next = 2.0 / (1.0 / mu_ij + 1.0 / mu_next);

            // coeffs for interface flux (symmetric)
            // After r-weighting: (1/r²) term becomes (1/r)
            double coeff_theta_prev = 1.0 / (r * mu_theta_prev * dtheta * dtheta);
            double coeff_theta_next = 1.0 / (r * mu_theta_next * dtheta * dtheta);

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
                local_triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_prev_idx, sign * coeff_theta_prev));
            } else {
                // Theta_min neighbor is Dirichlet: move bc value to RHS (profile[i] per-r if present)
                double bv = bc_theta_min.profile.empty() ? bc_theta_min.value : bc_theta_min.profile[i];
                rhs(idx) -= coeff_theta_prev * bv;
            }

            if (!theta_next_is_dirichlet) {
                double sign = next_crosses_boundary ? periodic_sign : 1.0;
                local_triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_next_idx, sign * coeff_theta_next));
            } else {
                // Theta_max neighbor is Dirichlet: move bc value to RHS (profile[i] per-r if present)
                double bv = bc_theta_max.profile.empty() ? bc_theta_max.value : bc_theta_max.profile[i];
                rhs(idx) -= coeff_theta_next * bv;
            }

            coeff_center -= (coeff_theta_prev + coeff_theta_next);
            }  // end else (interior / theta-Dirichlet theta term; theta-Robin handled above)

            // Center coefficient
            local_triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

            // Right-hand side (current density + magnetization equivalent current)
            // After r-weighting: RHS becomes -r·(Jz + Jz_mag)
            // Jz_mag_map = polar curl of M (1/r·d(r·Mθ)/dr - 1/r·dMr/dθ)
            // Use += to preserve any accumulated Dirichlet boundary contributions
            rhs(idx) += -(getJzPolar(jz_map, i, j, r_orientation) + getJzPolar(Jz_mag_map, i, j, r_orientation)) * r;
        }  // end omp for

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
    }  // end omp parallel

    // Check for potential singularity (all-Neumann + periodic case)
    // In this case, the solution is only defined up to a constant
    // Add pin constraint BEFORE assembling matrix
    bool need_pin = (bc_inner.type == "neumann" && bc_outer.type == "neumann" && is_periodic);

    if (need_pin) {
        std::cout << "Detected all-Neumann + periodic boundary: pinning Az(0,0)=0 to remove nullspace" << std::endl;
        int pin_idx = 0;
        // Add strong diagonal term to enforce Az(pin_idx) = 0
        // Use 1.0 with separate row handling (triplets will sum with existing entries)
        triplets.push_back(Eigen::Triplet<double>(pin_idx, pin_idx, 1.0));
        rhs(pin_idx) = 0.0;
    }

    // Assemble sparse matrix
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}
