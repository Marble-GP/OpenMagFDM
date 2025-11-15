#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <chrono>

constexpr double MU_0 = 4.0 * M_PI * 1e-7;  // Vacuum permeability [H/m]

MagneticFieldAnalyzer::MagneticFieldAnalyzer(const std::string& config_path,
                                             const std::string& image_path) {
    loadConfig(config_path);
    loadImage(image_path);

    // Determine coordinate system
    coordinate_system = config["coordinate_system"].as<std::string>("cartesian");
    std::cout << "Coordinate system: " << coordinate_system << std::endl;

    if (coordinate_system == "polar") {
        setupPolarSystem();
    } else {
        setupCartesianSystem();
    }

    setupMaterialProperties();
    validateBoundaryConditions();

    // Initialize transient analysis optimization flags
    transient_solver_initialized = false;
    transient_matrix_nnz = 0;
    boundary_cache_valid = false;
    use_iterative_solver = false;

    // Initialize magnetic field step tracking
    current_field_step = -1;  // Not calculated yet
}

void MagneticFieldAnalyzer::loadConfig(const std::string& config_path) {
    try {
        config = YAML::LoadFile(config_path);
        std::cout << "Configuration loaded from: " << config_path << std::endl;

        // Load transient analysis configuration
        if (config["transient"]) {
            auto trans = config["transient"];
            transient_config.enabled = trans["enabled"].as<bool>(false);
            transient_config.enable_sliding = trans["enable_sliding"].as<bool>(true);
            transient_config.total_steps = trans["total_steps"].as<int>(0);
            transient_config.slide_direction = trans["slide_direction"].as<std::string>("vertical");
            transient_config.slide_region_start = trans["slide_region_start"].as<int>(0);
            transient_config.slide_region_end = trans["slide_region_end"].as<int>(0);
            transient_config.slide_pixels_per_step = trans["slide_pixels_per_step"].as<int>(0);

            if (transient_config.enabled) {
                std::cout << "Transient analysis enabled: " << transient_config.total_steps << " steps" << std::endl;
                std::cout << "  Sliding: " << (transient_config.enable_sliding ? "enabled" : "disabled") << std::endl;
                if (transient_config.enable_sliding) {
                    std::cout << "  Slide direction: " << transient_config.slide_direction << std::endl;
                    std::string axis = (transient_config.slide_direction == "vertical") ? "x" : "y";
                    std::cout << "  Slide region: " << axis << " in [" << transient_config.slide_region_start
                              << ", " << transient_config.slide_region_end << "]" << std::endl;
                    std::cout << "  Pixels per step: " << transient_config.slide_pixels_per_step << std::endl;
                }
            }
        }
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load YAML config: " + std::string(e.what()));
    }
}

void MagneticFieldAnalyzer::loadImage(const std::string& image_path) {
    image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    std::cout << "Image loaded: " << image_path
              << " (size: " << image.cols << "x" << image.rows << ")" << std::endl;
}

void MagneticFieldAnalyzer::setupCartesianSystem() {
    ny = image.rows;
    nx = image.cols;

    // Get mesh spacing from config
    if (config["mesh"]) {
        dx = config["mesh"]["dx"].as<double>(1.0);
        dy = config["mesh"]["dy"].as<double>(1.0);
    } else {
        dx = 1.0;
        dy = 1.0;
    }

    std::cout << "Cartesian mesh setup: nx=" << nx << ", ny=" << ny
              << ", dx=" << dx << ", dy=" << dy << std::endl;

    // Initialize material property matrices
    mu_map = Eigen::MatrixXd::Constant(ny, nx, MU_0);
    jz_map = Eigen::MatrixXd::Zero(ny, nx);
}

void MagneticFieldAnalyzer::setupPolarSystem() {
    // Get polar domain parameters
    if (config["polar_domain"]) {
        r_start = config["polar_domain"]["r_start"].as<double>(0.01);
        r_end = config["polar_domain"]["r_end"].as<double>(1.0);
        r_orientation = config["polar_domain"]["r_orientation"].as<std::string>("horizontal");

        // Parse theta_range (supports tinyexpr formula like "2*pi" or "pi/2", or numeric values)
        if (config["polar_domain"]["theta_range"]) {
            std::string theta_str;
            try {
                // Try to read as string first (for formulas like "2*pi/12")
                theta_str = config["polar_domain"]["theta_range"].as<std::string>();
            } catch (...) {
                // If that fails, try to read as double and convert to string
                try {
                    double val = config["polar_domain"]["theta_range"].as<double>();
                    theta_str = std::to_string(val);
                } catch (...) {
                    std::cerr << "Warning: Failed to read theta_range, using default 2*pi" << std::endl;
                    theta_str = "2*pi";
                }
            }
            te_parser parser;
            theta_range = parser.evaluate(theta_str);
            if (std::isnan(theta_range)) {
                std::cerr << "Warning: Failed to parse theta_range '" << theta_str
                          << "', using default 2*pi" << std::endl;
                theta_range = 2.0 * M_PI;
            }
        } else {
            theta_range = 2.0 * M_PI;
        }
    } else {
        r_start = 0.01;
        r_end = 1.0;
        r_orientation = "horizontal";
        theta_range = 2.0 * M_PI;
    }

    // Determine nr and ntheta based on r_orientation
    if (r_orientation == "horizontal") {
        // r direction: horizontal (columns), theta direction: vertical (rows)
        nr = image.cols;
        ntheta = image.rows;
    } else {  // vertical
        // r direction: vertical (rows), theta direction: horizontal (columns)
        nr = image.rows;
        ntheta = image.cols;
    }

    std::cout << "Polar mesh (r_orientation=" << r_orientation << "): nr=" << nr << ", ntheta=" << ntheta << std::endl;

    // Validate polar domain
    if (r_start <= 0.0 || r_start >= r_end) {
        throw std::runtime_error("Invalid polar domain: must satisfy 0 < r_start < r_end");
    }
    if (theta_range <= 0.0 || theta_range > 2.0 * M_PI) {
        throw std::runtime_error("Invalid theta_range: must satisfy 0 < theta_range <= 2*pi");
    }

    // Calculate mesh spacing
    dr = (r_end - r_start) / (nr - 1);
    // Angular spacing: In polar coordinates, θ is always periodic
    // θ=0 and θ=theta_range represent the same radial line
    // Therefore, we divide theta_range by ntheta (NOT ntheta-1)
    dtheta = theta_range / static_cast<double>(ntheta);  // Periodic sampling: [0, theta_range)

    std::cout << "Polar domain: r = [" << r_start << ", " << r_end << "] m, theta = [0, "
              << theta_range << "] rad (" << (theta_range * 180.0 / M_PI) << " deg)" << std::endl;
    std::cout << "Mesh spacing: dr=" << dr << ", dtheta=" << dtheta << " rad" << std::endl;

    // Validate mesh spacing for Neumann boundary stability
    // For ghost-elimination at inner boundary (i=0), we need r_{i-1/2} = r_start - 0.5*dr > 0
    if (r_start < 0.5 * dr) {
        std::cerr << "Warning: r_start (" << r_start << ") < 0.5*dr (" << 0.5*dr
                  << "). Inner Neumann BC may be unstable." << std::endl;
        std::cerr << "Recommendation: Increase r_start or decrease nr to satisfy r_start >= 0.5*dr" << std::endl;
        // Not throwing error - allow user to proceed with caution
    }

    // Generate radial coordinates
    r_coords.resize(nr);
    for (int i = 0; i < nr; i++) {
        r_coords[i] = r_start + i * dr;
    }

    // Get polar boundary conditions
    if (config["polar_boundary_conditions"]) {
        auto bc_cfg = config["polar_boundary_conditions"];

        if (bc_cfg["inner"]) {
            bc_inner.type = bc_cfg["inner"]["type"].as<std::string>("dirichlet");
            bc_inner.value = bc_cfg["inner"]["value"].as<double>(0.0);
        }

        if (bc_cfg["outer"]) {
            bc_outer.type = bc_cfg["outer"]["type"].as<std::string>("dirichlet");
            bc_outer.value = bc_cfg["outer"]["value"].as<double>(0.0);
        }
    }

    std::cout << "Boundary conditions: inner=" << bc_inner.type
              << ", outer=" << bc_outer.type << std::endl;

    // Initialize material property matrices with r_orientation-dependent shape
    // This must match the shape of Br, Btheta, and Az matrices
    if (r_orientation == "horizontal") {
        // Image is (ntheta, nr) = (rows, cols)
        // Matrices: (ntheta, nr) with indexing (theta_idx, r_idx)
        mu_map = Eigen::MatrixXd::Constant(ntheta, nr, MU_0);
        jz_map = Eigen::MatrixXd::Zero(ntheta, nr);
        std::cout << "Material property matrices: (" << ntheta << " x " << nr << ") = (theta x r)" << std::endl;
    } else {  // vertical
        // Image is (nr, ntheta) = (rows, cols)
        // Matrices: (nr, ntheta) with indexing (r_idx, theta_idx)
        mu_map = Eigen::MatrixXd::Constant(nr, ntheta, MU_0);
        jz_map = Eigen::MatrixXd::Zero(nr, ntheta);
        std::cout << "Material property matrices: (" << nr << " x " << ntheta << ") = (r x theta)" << std::endl;
    }
}

void MagneticFieldAnalyzer::setupMaterialProperties() {
    if (!config["materials"]) {
        std::cout << "No materials defined in config" << std::endl;
        return;
    }

    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;

        // Get RGB values
        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});
        double mu_r = props["mu_r"].as<double>(1.0);

        // Parse Jz value (static, formula, or array)
        JzValue jz_value = parseJzValue(props["jz"]);
        material_jz[name] = jz_value;

        // Evaluate Jz for step 0 (initial state)
        double jz = evaluateJz(jz_value, 0);

        // Find matching pixels
        int count = 0;

        if (coordinate_system == "polar") {
            // For polar coordinates, flip image to match analysis coordinates
            // Convention: image bottom (row=ntheta-1) -> theta=0, image top (row=0) -> theta=theta_range
            // This ensures theta increases upward (counterclockwise from x-axis, like standard y-axis)
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);  // Flip vertically: y down -> y up

            if (r_orientation == "horizontal") {
                // r: cols (i), theta: rows (j, after flip)
                // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx) = (j, i)
                for (int j = 0; j < ntheta; j++) {
                    for (int i = 0; i < nr; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            mu_map(j, i) = mu_r * MU_0;  // (theta_idx, r_idx)
                            jz_map(j, i) = jz;
                            count++;
                        }
                    }
                }
            } else {  // vertical
                // r: rows (j), theta: cols (i)
                // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx) = (j, i)
                for (int j = 0; j < nr; j++) {
                    for (int i = 0; i < ntheta; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            mu_map(j, i) = mu_r * MU_0;  // (r_idx, theta_idx)
                            jz_map(j, i) = jz;
                            count++;
                        }
                    }
                }
            }
        } else {
            // Cartesian coordinates
            // Create flipped image to match analysis coordinates (y up)
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);  // Flip vertically: y down -> y up

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        mu_map(j, i) = mu_r * MU_0;
                        jz_map(j, i) = jz;
                        count++;
                    }
                }
            }
        }

        // Display Jz type
        std::string jz_type_str;
        switch (jz_value.type) {
            case JzType::STATIC: jz_type_str = "static"; break;
            case JzType::FORMULA: jz_type_str = "formula"; break;
            case JzType::ARRAY: jz_type_str = "array"; break;
        }

        std::cout << "Material '" << name << "': RGB(" << rgb[0] << "," << rgb[1] << "," << rgb[2]
                  << ") -> mu_r=" << mu_r << ", Jz=" << jz << " A/m^2 (" << jz_type_str << ")" << std::endl;
        std::cout << "  Matching pixels: " << count << std::endl;
    }
}

void MagneticFieldAnalyzer::validateBoundaryConditions() {
    if (!config["boundary_conditions"]) {
        std::cout << "Using default Dirichlet boundary conditions (value=0)" << std::endl;
        return;
    }

    auto bc = config["boundary_conditions"];

    // Left boundary
    if (bc["left"]) {
        bc_left.type = bc["left"]["type"].as<std::string>("dirichlet");
        bc_left.value = bc["left"]["value"].as<double>(0.0);
    }

    // Right boundary
    if (bc["right"]) {
        bc_right.type = bc["right"]["type"].as<std::string>("dirichlet");
        bc_right.value = bc["right"]["value"].as<double>(0.0);
    }

    // Bottom boundary
    if (bc["bottom"]) {
        bc_bottom.type = bc["bottom"]["type"].as<std::string>("dirichlet");
        bc_bottom.value = bc["bottom"]["value"].as<double>(0.0);
    }

    // Top boundary
    if (bc["top"]) {
        bc_top.type = bc["top"]["type"].as<std::string>("dirichlet");
        bc_top.value = bc["top"]["value"].as<double>(0.0);
    }

    std::cout << "Boundary conditions:" << std::endl;
    std::cout << "  Left: " << bc_left.type << std::endl;
    std::cout << "  Right: " << bc_right.type << std::endl;
    std::cout << "  Bottom: " << bc_bottom.type << std::endl;
    std::cout << "  Top: " << bc_top.type << std::endl;
}

double MagneticFieldAnalyzer::getMuAtInterface(int i, int j, const std::string& direction) const {
    // Harmonic mean of permeability at cell interface
    if (direction == "x+") {
        if (i < nx - 1) {
            return 2.0 / (1.0 / mu_map(j, i) + 1.0 / mu_map(j, i + 1));
        }
        return mu_map(j, i);
    } else if (direction == "x-") {
        if (i > 0) {
            return 2.0 / (1.0 / mu_map(j, i) + 1.0 / mu_map(j, i - 1));
        }
        return mu_map(j, i);
    } else if (direction == "y+") {
        if (j < ny - 1) {
            return 2.0 / (1.0 / mu_map(j, i) + 1.0 / mu_map(j + 1, i));
        }
        return mu_map(j, i);
    } else if (direction == "y-") {
        if (j > 0) {
            return 2.0 / (1.0 / mu_map(j, i) + 1.0 / mu_map(j - 1, i));
        }
        return mu_map(j, i);
    }
    return mu_map(j, i);
}

// ============================================================================
// Unified mu accessors (coordinate-system aware)
// ============================================================================

double MagneticFieldAnalyzer::muAtGrid(int i, int j) const {
    // i: column index (x-direction or r-direction depending on orientation)
    // j: row index (y-direction or theta-direction depending on orientation)

    if (coordinate_system == "polar") {
        // Map (i,j) -> (ir, jt) depending on r_orientation
        int ir, jt;
        if (r_orientation == "horizontal") {
            // r: horizontal (cols), theta: vertical (rows)
            ir = i;
            jt = j;
        } else {
            // r: vertical (rows), theta: horizontal (cols)
            ir = j;
            jt = i;
        }

        // Clamp to valid indices
        ir = std::min(std::max(ir, 0), nr - 1);
        jt = std::min(std::max(jt, 0), ntheta - 1);

        // Access mu_map with orientation-dependent indexing
        if (r_orientation == "horizontal") {
            // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx)
            return mu_map(jt, ir);
        } else {
            // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx)
            return mu_map(ir, jt);
        }
    } else {
        // Cartesian: mu_map stored as (ny, nx)
        // mu_map(rows=ny, cols=nx) -> mu_map(j, i)
        int jj = std::min(std::max(j, 0), ny - 1);
        int ii = std::min(std::max(i, 0), nx - 1);
        return mu_map(jj, ii);
    }
}

double MagneticFieldAnalyzer::getMuAtInterfaceSym(int i, int j, const std::string& direction) const {
    // Symmetric interface mu calculation using harmonic mean
    // This ensures A(i,j) == A(j,i) in coefficient matrix

    if (direction == "x+") {
        // Interface between (i,j) and (i+1,j)
        if (i < nx - 1) {
            double mu_left = muAtGrid(i, j);
            double mu_right = muAtGrid(i + 1, j);
            return 2.0 / (1.0 / mu_left + 1.0 / mu_right);
        }
        return muAtGrid(i, j);
    } else if (direction == "x-") {
        // Interface between (i-1,j) and (i,j)
        if (i > 0) {
            double mu_left = muAtGrid(i - 1, j);
            double mu_right = muAtGrid(i, j);
            return 2.0 / (1.0 / mu_left + 1.0 / mu_right);
        }
        return muAtGrid(i, j);
    } else if (direction == "y+") {
        // Interface between (i,j) and (i,j+1)
        if (j < ny - 1) {
            double mu_bottom = muAtGrid(i, j);
            double mu_top = muAtGrid(i, j + 1);
            return 2.0 / (1.0 / mu_bottom + 1.0 / mu_top);
        }
        return muAtGrid(i, j);
    } else if (direction == "y-") {
        // Interface between (i,j-1) and (i,j)
        if (j > 0) {
            double mu_bottom = muAtGrid(i, j - 1);
            double mu_top = muAtGrid(i, j);
            return 2.0 / (1.0 / mu_bottom + 1.0 / mu_top);
        }
        return muAtGrid(i, j);
    }
    return muAtGrid(i, j);
}

void MagneticFieldAnalyzer::solve() {
    if (coordinate_system == "polar") {
        buildAndSolveSystemPolar();
    } else {
        buildAndSolveSystem();
    }
}

void MagneticFieldAnalyzer::buildMatrix(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
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

    if (x_periodic) {
        std::cout << "Periodic boundary detected in X direction (left-right)" << std::endl;
    }
    if (y_periodic) {
        std::cout << "Periodic boundary detected in Y direction (bottom-top)" << std::endl;
    }

    // Build equation for each grid point
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = j * nx + i;

            bool is_left = (i == 0);
            bool is_right = (i == nx - 1);
            bool is_bottom = (j == 0);
            bool is_top = (j == ny - 1);

            // Dirichlet boundary conditions (skip if periodic)
            if (is_left && bc_left.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_left.value;
                continue;
            } else if (is_right && bc_right.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_right.value;
                continue;
            } else if (is_bottom && bc_bottom.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_bottom.value;
                continue;
            } else if (is_top && bc_top.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_top.value;
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
                    triplets.push_back(Eigen::Triplet<double>(idx, idx_west, coeff_west));
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
                    triplets.push_back(Eigen::Triplet<double>(idx, idx_east, coeff_east));
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
                    triplets.push_back(Eigen::Triplet<double>(idx, idx_south, coeff_south));
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
                    triplets.push_back(Eigen::Triplet<double>(idx, idx_north, coeff_north));
                } else {
                    // Top neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= coeff_north * bc_top.value;
                }
                coeff_center -= coeff_north;
            }

            // Center coefficient
            triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

            // Right-hand side (current density)
            rhs(idx) = -jz_map(j, i);
        }
    }

    // Construct sparse matrix
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

void MagneticFieldAnalyzer::buildAndSolveSystem() {
    std::cout << "\n=== Building FDM system of equations ===" << std::endl;

    int n = nx * ny;
    Eigen::SparseMatrix<double> A(n, n);
    Eigen::VectorXd rhs(n);

    // Build matrix using the separated method
    buildMatrix(A, rhs);

    std::cout << "Matrix size: " << A.rows() << "x" << A.cols() << std::endl;
    std::cout << "Non-zero elements: " << A.nonZeros() << std::endl;

    // Check matrix symmetry (critical for numerical accuracy)
    Eigen::SparseMatrix<double> A_T = A.transpose();
    double symmetry_error = (A - A_T).norm();
    double A_norm = A.norm();
    double relative_symmetry_error = (A_norm > 1e-12) ? (symmetry_error / A_norm) : symmetry_error;
    std::cout << "Matrix symmetry check:" << std::endl;
    std::cout << "  ||A - A^T|| = " << symmetry_error << std::endl;
    std::cout << "  ||A|| = " << A_norm << std::endl;
    std::cout << "  Relative error = " << relative_symmetry_error << std::endl;
    if (relative_symmetry_error > 1e-8) {
        std::cerr << "WARNING: Matrix is not symmetric! Relative error = "
                  << relative_symmetry_error << std::endl;
        std::cerr << "This may indicate incorrect discretization or mu interface calculation." << std::endl;
    } else {
        std::cout << "  Matrix is symmetric (relative error < 1e-8)" << std::endl;
    }

    // Solve using SparseLU
    std::cout << "\n=== Solving linear system ===" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Matrix decomposition failed");
    }

    Eigen::VectorXd Az_flat = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Linear system solving failed");
    }

    // Reshape to 2D matrix
    Az.resize(ny, nx);
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            Az(j, i) = Az_flat(j * nx + i);
        }
    }

    std::cout << "Solution complete!" << std::endl;
}

void MagneticFieldAnalyzer::exportAzToCSV(const std::string& output_path) const {
    if (Az.size() == 0) {
        throw std::runtime_error("No solution to export. Run solve() first.");
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    file << std::scientific << std::setprecision(16);

    int rows = Az.rows();
    int cols = Az.cols();

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            file << Az(j, i);
            if (i < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Az array exported to: " << output_path << std::endl;
    std::cout << "Array size: " << rows << " rows x " << cols << " columns" << std::endl;
}

void MagneticFieldAnalyzer::exportMuToCSV(const std::string& output_path) const {
    if (mu_map.size() == 0) {
        throw std::runtime_error("No permeability data to export.");
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    file << std::scientific << std::setprecision(16);

    int rows = mu_map.rows();
    int cols = mu_map.cols();

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            file << mu_map(j, i);
            if (i < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Permeability distribution exported to: " << output_path << std::endl;
    std::cout << "Array size: " << rows << " rows x " << cols << " columns" << std::endl;
}

void MagneticFieldAnalyzer::exportJzToCSV(const std::string& output_path) const {
    if (jz_map.size() == 0) {
        throw std::runtime_error("No current density data to export.");
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    file << std::scientific << std::setprecision(16);

    int rows = jz_map.rows();
    int cols = jz_map.cols();

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            file << jz_map(j, i);
            if (i < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Current density distribution exported to: " << output_path << std::endl;
    std::cout << "Array size: " << rows << " rows x " << cols << " columns" << std::endl;
}

// ===== Maxwell Stress Calculation =====

void MagneticFieldAnalyzer::calculateMagneticField() {
    // Calculate magnetic flux density: B = rot(A)
    // Bx = ∂Az/∂y
    // By = -∂Az/∂x

    Bx.resize(ny, nx);
    By.resize(ny, nx);

    // Check for periodic boundary conditions
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Bx = ∂Az/∂y
            if (j == 0) {
                if (y_periodic) {
                    // Periodic boundary: use central difference with wrap
                    Bx(j, i) = (Az(1, i) - Az(ny-1, i)) / (2.0 * dy);
                } else {
                    // Forward difference
                    Bx(j, i) = (Az(1, i) - Az(0, i)) / dy;
                }
            } else if (j == ny - 1) {
                if (y_periodic) {
                    // Periodic boundary: use central difference with wrap
                    Bx(j, i) = (Az(0, i) - Az(ny-2, i)) / (2.0 * dy);
                } else {
                    // Backward difference
                    Bx(j, i) = (Az(ny-1, i) - Az(ny-2, i)) / dy;
                }
            } else {
                // Central difference
                Bx(j, i) = (Az(j+1, i) - Az(j-1, i)) / (2.0 * dy);
            }

            // By = -∂Az/∂x
            if (i == 0) {
                if (x_periodic) {
                    // Periodic boundary: use central difference with wrap
                    By(j, i) = -(Az(j, 1) - Az(j, nx-1)) / (2.0 * dx);
                } else {
                    // Forward difference
                    By(j, i) = -(Az(j, 1) - Az(j, 0)) / dx;
                }
            } else if (i == nx - 1) {
                if (x_periodic) {
                    // Periodic boundary: use central difference with wrap
                    By(j, i) = -(Az(j, 0) - Az(j, nx-2)) / (2.0 * dx);
                } else {
                    // Backward difference
                    By(j, i) = -(Az(j, nx-1) - Az(j, nx-2)) / dx;
                }
            } else {
                // Central difference
                By(j, i) = -(Az(j, i+1) - Az(j, i-1)) / (2.0 * dx);
            }
        }
    }

    std::cout << "Magnetic field calculated" << std::endl;
}

// ===== Helper methods for periodic boundary-aware filtering =====

/**
 * @brief Apply Laplacian filter with periodic boundary conditions
 * @param src Source image
 * @param dst Destination image (CV_16S)
 * @param ksize Kernel size (default: 3)
 */
void MagneticFieldAnalyzer::applyLaplacianWithPeriodicBC(const cv::Mat& src, cv::Mat& dst, int ksize) {
    // Check which boundaries are periodic
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    if (!x_periodic && !y_periodic) {
        // No periodic boundaries - use standard Laplacian
        cv::Laplacian(src, dst, CV_16S, ksize);
        return;
    }

    // Expand image with appropriate boundary conditions
    int border = ksize / 2;
    cv::Mat expanded;

    if (x_periodic && y_periodic) {
        // Both periodic - use BORDER_WRAP
        cv::copyMakeBorder(src, expanded, border, border, border, border, cv::BORDER_WRAP);
    } else if (x_periodic && !y_periodic) {
        // X periodic, Y replicate - do in two steps
        cv::Mat temp;
        cv::copyMakeBorder(src, temp, 0, 0, border, border, cv::BORDER_WRAP);
        cv::copyMakeBorder(temp, expanded, border, border, 0, 0, cv::BORDER_REPLICATE);
    } else if (!x_periodic && y_periodic) {
        // X replicate, Y periodic - do in two steps
        cv::Mat temp;
        cv::copyMakeBorder(src, temp, border, border, 0, 0, cv::BORDER_WRAP);
        cv::copyMakeBorder(temp, expanded, 0, 0, border, border, cv::BORDER_REPLICATE);
    }

    // Apply Laplacian on expanded image
    cv::Mat expanded_result;
    cv::Laplacian(expanded, expanded_result, CV_16S, ksize);

    // Extract the center region (original size)
    cv::Rect roi(border, border, src.cols, src.rows);
    dst = expanded_result(roi).clone();
}

/**
 * @brief Apply Sobel filter with periodic boundary conditions
 * @param src Source image
 * @param dst Destination image (CV_64F)
 * @param dx Derivative order in x
 * @param dy Derivative order in y
 * @param ksize Kernel size (default: 3)
 */
void MagneticFieldAnalyzer::applySobelWithPeriodicBC(const cv::Mat& src, cv::Mat& dst, int dx, int dy, int ksize) {
    // Check which boundaries are periodic
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    if (!x_periodic && !y_periodic) {
        // No periodic boundaries - use standard Sobel
        cv::Sobel(src, dst, CV_64F, dx, dy, ksize);
        return;
    }

    // Expand image with appropriate boundary conditions
    int border = ksize / 2;
    cv::Mat expanded;

    if (x_periodic && y_periodic) {
        // Both periodic - use BORDER_WRAP
        cv::copyMakeBorder(src, expanded, border, border, border, border, cv::BORDER_WRAP);
    } else if (x_periodic && !y_periodic) {
        // X periodic, Y replicate - do in two steps
        cv::Mat temp;
        cv::copyMakeBorder(src, temp, 0, 0, border, border, cv::BORDER_WRAP);
        cv::copyMakeBorder(temp, expanded, border, border, 0, 0, cv::BORDER_REPLICATE);
    } else if (!x_periodic && y_periodic) {
        // X replicate, Y periodic - do in two steps
        cv::Mat temp;
        cv::copyMakeBorder(src, temp, border, border, 0, 0, cv::BORDER_WRAP);
        cv::copyMakeBorder(temp, expanded, 0, 0, border, border, cv::BORDER_REPLICATE);
    }

    // Apply Sobel on expanded image
    cv::Mat expanded_result;
    cv::Sobel(expanded, expanded_result, CV_64F, dx, dy, ksize);

    // Extract the center region (original size)
    cv::Rect roi(border, border, src.cols, src.rows);
    dst = expanded_result(roi).clone();
}

cv::Mat MagneticFieldAnalyzer::detectBoundaries() {
    // Laplacian kernel size = 3, so influence range is ±1 pixel (use ±2 for safety)
    const int KERNEL_MARGIN = 2;

    // Check which boundaries are periodic
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    // Check if we can use incremental update (transient analysis with sliding)
    // Disable incremental update if periodic BC is active (filter results are position-dependent)
    bool use_incremental = boundary_cache_valid && transient_config.enabled && transient_config.enable_sliding
                          && !x_periodic && !y_periodic;

    cv::Mat boundaries;

    if (!use_incremental) {
        // Full boundary detection (first time, cache invalid, or periodic BC active)
        if (x_periodic || y_periodic) {
            std::cout << "Boundaries: Full detection (periodic BC requires full recomputation)" << std::endl;
        } else {
            std::cout << "Boundaries: Full detection (Laplacian filter)" << std::endl;
        }

        // Convert material map to grayscale for boundary detection
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

        // Apply Laplacian filter for edge detection (with periodic BC if applicable)
        cv::Mat laplacian_s16;
        applyLaplacianWithPeriodicBC(gray, laplacian_s16, 3);

        // Convert to absolute value
        cv::Mat laplacian_abs;
        cv::convertScaleAbs(laplacian_s16, laplacian_abs);

        // Threshold to get binary boundary map
        cv::threshold(laplacian_abs, boundaries, 10, 255, cv::THRESH_BINARY);

        // Cache the result
        cached_boundaries = boundaries.clone();
        boundary_cache_valid = true;
    } else {
        // Incremental update: slide cached boundaries + recompute border regions
        std::cout << "Boundaries: Incremental update (slide + border recompute)" << std::endl;
        std::cout << "[DEBUG] Starting incremental boundary detection..." << std::endl;

        boundaries = cached_boundaries.clone();
        int shift = transient_config.slide_pixels_per_step;
        std::cout << "[DEBUG] shift = " << shift << std::endl;

        if (transient_config.slide_direction == "vertical") {
            // Vertical slide: shift rows (y direction)
            int x_start = transient_config.slide_region_start;
            int x_end = transient_config.slide_region_end;

            // Slide the boundary detection results within the region
            cv::Mat slide_region = boundaries(cv::Rect(x_start, 0, x_end - x_start, boundaries.rows)).clone();
            cv::Mat shifted_region = cv::Mat::zeros(slide_region.size(), slide_region.type());

            for (int row = 0; row < slide_region.rows; row++) {
                int src_row = (row + slide_region.rows + shift) % slide_region.rows;
                slide_region.row(src_row).copyTo(shifted_region.row(row));
            }

            shifted_region.copyTo(boundaries(cv::Rect(x_start, 0, x_end - x_start, boundaries.rows)));

            // Recompute border region (±KERNEL_MARGIN around the circular shift seam)
            // Circular shift creates ONE seam at row=0 where data wraps around
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

            std::cout << "[DEBUG] Recomputing seam at circular shift boundary (row=0)" << std::endl;
            const int KERNEL_MARGIN = 2;  // Laplacian kernel margin
            int seam_row = 0;  // Circular shift seam is always at row=0

            // Recompute region around the seam with margin for kernel
            int y_min = std::max(0, seam_row + shift - KERNEL_MARGIN);
            int y_max = std::min(boundaries.rows, seam_row + shift + KERNEL_MARGIN + 1);

            if (y_max > y_min) {
                cv::Rect roi(x_start, y_min, x_end - x_start, y_max - y_min);
                cv::Mat gray_roi = gray(roi);
                cv::Mat laplacian_s16, laplacian_abs, boundaries_roi;

                // Note: ROI processing doesn't fully respect periodic BC at ROI boundaries
                // For full accuracy, consider recomputing the entire image
                applyLaplacianWithPeriodicBC(gray_roi, laplacian_s16, 3);
                cv::convertScaleAbs(laplacian_s16, laplacian_abs);
                cv::threshold(laplacian_abs, boundaries_roi, 10, 255, cv::THRESH_BINARY);

                boundaries_roi.copyTo(boundaries(roi));
                std::cout << "[DEBUG] Seam recomputed: y_range=[" << y_min << ", " << y_max << ")" << std::endl;
            }
        } else {  // horizontal
            // Horizontal slide: shift columns (x direction)
            int y_start = transient_config.slide_region_start;
            int y_end = transient_config.slide_region_end;

            // Slide the boundary detection results within the region
            cv::Mat slide_region = boundaries(cv::Rect(0, y_start, boundaries.cols, y_end - y_start)).clone();
            cv::Mat shifted_region = cv::Mat::zeros(slide_region.size(), slide_region.type());

            for (int col = 0; col < slide_region.cols; col++) {
                int src_col = (col + slide_region.cols + shift) % slide_region.cols;
                slide_region.col(src_col).copyTo(shifted_region.col(col));
            }

            shifted_region.copyTo(boundaries(cv::Rect(0, y_start, boundaries.cols, y_end - y_start)));

            // Recompute border region (±KERNEL_MARGIN around the circular shift seam)
            // Circular shift creates ONE seam at col=0 where data wraps around
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

            std::cout << "[DEBUG] Recomputing seam at circular shift boundary (col=0)" << std::endl;
            const int KERNEL_MARGIN = 2;  // Laplacian kernel margin
            int seam_col = 0;  // Circular shift seam is always at col=0

            // Recompute region around the seam with margin for kernel
            int x_min = std::max(0, seam_col + shift - KERNEL_MARGIN);
            int x_max = std::min(boundaries.cols, seam_col + shift + KERNEL_MARGIN + 1);

            if (x_max > x_min) {
                cv::Rect roi(x_min, y_start, x_max - x_min, y_end - y_start);
                cv::Mat gray_roi = gray(roi);
                cv::Mat laplacian_s16, laplacian_abs, boundaries_roi;

                // Note: ROI processing doesn't fully respect periodic BC at ROI boundaries
                applyLaplacianWithPeriodicBC(gray_roi, laplacian_s16, 3);
                cv::convertScaleAbs(laplacian_s16, laplacian_abs);
                cv::threshold(laplacian_abs, boundaries_roi, 10, 255, cv::THRESH_BINARY);

                boundaries_roi.copyTo(boundaries(roi));
                // std::cout << "[DEBUG] Seam recomputed: x_range=[" << x_min << ", " << x_max << ")" << std::endl;
            }
        }

        // std::cout << "[DEBUG] Incremental boundary detection complete" << std::endl;
        // Update cache
        cached_boundaries = boundaries.clone();
    }

    // std::cout << "[DEBUG] Creating boundary visualization..." << std::endl;
    // Create boundary visualization
    boundary_image = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
    for (int j = 0; j < boundaries.rows; j++) {
        for (int i = 0; i < boundaries.cols; i++) {
            if (boundaries.at<uchar>(j, i) > 0) {
                cv::Vec3b rgb_pixel = image.at<cv::Vec3b>(j, i);
                boundary_image.at<cv::Vec3b>(j, i) = cv::Vec3b(rgb_pixel[2], rgb_pixel[1], rgb_pixel[0]);
            }
        }
    }

    // Count boundary pixels
    int boundary_count = cv::countNonZero(boundaries);
    std::cout << "  Total boundary pixels: " << boundary_count << std::endl;

    return boundaries;
}

// ============================================================================
// Helper functions for polar coordinate indexing
// ============================================================================

// Helper functions to access mu_map and jz_map in polar coordinates with correct indexing
inline double getMuPolar(const Eigen::MatrixXd& mu_map, int r_idx, int theta_idx, const std::string& r_orientation) {
    if (r_orientation == "horizontal") {
        // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx)
        return mu_map(theta_idx, r_idx);
    } else {
        // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx)
        return mu_map(r_idx, theta_idx);
    }
}

inline double getJzPolar(const Eigen::MatrixXd& jz_map, int r_idx, int theta_idx, const std::string& r_orientation) {
    if (r_orientation == "horizontal") {
        // jz_map shape: (ntheta, nr), indexing: (theta_idx, r_idx)
        return jz_map(theta_idx, r_idx);
    } else {
        // jz_map shape: (nr, ntheta), indexing: (r_idx, theta_idx)
        return jz_map(r_idx, theta_idx);
    }
}

// Bilinear interpolation for polar coordinates with periodic theta boundary
// Input: physical coordinates (r_phys, theta_phys)
// Output: interpolated field value
inline double bilinearInterpolatePolar(
    const Eigen::MatrixXd& field,
    double r_phys, double theta_phys,
    double r_start, double dr, double dtheta, double theta_range,
    int nr, int ntheta,
    const std::string& r_orientation) {

    // Convert physical coordinates to continuous grid indices
    double r_idx_cont = (r_phys - r_start) / dr;

    // FIX5: Normalize theta to [0, theta_range) in radians BEFORE converting to index
    // CRITICAL: This handles sector models (theta_range < 2π) correctly
    // Periodic boundary: theta = theta_range wraps to theta = 0
    double theta_norm = std::fmod(theta_phys, theta_range);
    if (theta_norm < 0.0) theta_norm += theta_range;
    double theta_idx_cont = theta_norm / dtheta;

    // Clamp r index to valid range [0, nr-2] (need r0+1 to be valid)
    r_idx_cont = std::max(0.0, std::min(r_idx_cont, static_cast<double>(nr - 2) + 0.999));

    // Get integer indices (floor)
    int r0 = static_cast<int>(std::floor(r_idx_cont));
    int t0 = static_cast<int>(std::floor(theta_idx_cont));

    // Clamp to ensure valid indices
    r0 = std::max(0, std::min(r0, nr - 2));
    int r1 = r0 + 1;

    // Theta wraps periodically
    int t1 = (t0 + 1) % ntheta;

    // Interpolation weights
    double wr = r_idx_cont - r0;
    double wt = theta_idx_cont - t0;

    // Clamp weights to [0, 1]
    wr = std::max(0.0, std::min(1.0, wr));
    wt = std::max(0.0, std::min(1.0, wt));

    // Sample field at 4 corners (handle r_orientation)
    double v00, v10, v01, v11;
    if (r_orientation == "horizontal") {
        // field shape: (ntheta, nr), indexing: (theta_idx, r_idx)
        v00 = field(t0, r0);
        v10 = field(t0, r1);
        v01 = field(t1, r0);
        v11 = field(t1, r1);
    } else {
        // field shape: (nr, ntheta), indexing: (r_idx, theta_idx)
        v00 = field(r0, t0);
        v10 = field(r1, t0);
        v01 = field(r0, t1);
        v11 = field(r1, t1);
    }

    // Bilinear interpolation: first interpolate in r, then in theta
    double v0 = (1.0 - wr) * v00 + wr * v10;  // at t0
    double v1 = (1.0 - wr) * v01 + wr * v11;  // at t1
    return (1.0 - wt) * v0 + wt * v1;
}

// Sample B, μ, and coordinates at a physical point with consistent interpolation
// CRITICAL: This ensures B and μ are evaluated at the SAME physical location
MagneticFieldAnalyzer::PolarSample MagneticFieldAnalyzer::sampleFieldsAtPhysicalPoint(double x_phys, double y_phys) {
    MagneticFieldAnalyzer::PolarSample sample;
    sample.x_phys = x_phys;
    sample.y_phys = y_phys;

    if (coordinate_system == "cartesian") {
        // Cartesian coordinates: straightforward interpolation
        // Convert physical coords to grid indices
        double i_cont = x_phys / dx;
        double j_cont = y_phys / dy;

        // Clamp to valid range
        i_cont = std::max(0.0, std::min(i_cont, static_cast<double>(nx - 1)));
        j_cont = std::max(0.0, std::min(j_cont, static_cast<double>(ny - 1)));

        // Get integer indices
        int i0 = static_cast<int>(std::floor(i_cont));
        int j0 = static_cast<int>(std::floor(j_cont));
        int i1 = std::min(i0 + 1, nx - 1);
        int j1 = std::min(j0 + 1, ny - 1);

        // Interpolation weights
        double wi = i_cont - i0;
        double wj = j_cont - j0;

        // Bilinear interpolation for Bx, By, mu
        double Bx00 = Bx(j0, i0), Bx10 = Bx(j0, i1), Bx01 = Bx(j1, i0), Bx11 = Bx(j1, i1);
        double By00 = By(j0, i0), By10 = By(j0, i1), By01 = By(j1, i0), By11 = By(j1, i1);
        double mu00 = mu_map(j0, i0), mu10 = mu_map(j0, i1), mu01 = mu_map(j1, i0), mu11 = mu_map(j1, i1);

        sample.Bx = (1-wi)*(1-wj)*Bx00 + wi*(1-wj)*Bx10 + (1-wi)*wj*Bx01 + wi*wj*Bx11;
        sample.By = (1-wi)*(1-wj)*By00 + wi*(1-wj)*By10 + (1-wi)*wj*By01 + wi*wj*By11;
        sample.mu = (1-wi)*(1-wj)*mu00 + wi*(1-wj)*mu10 + (1-wi)*wj*mu01 + wi*wj*mu11;

        // Polar coords for reference
        sample.r_phys = std::sqrt(x_phys*x_phys + y_phys*y_phys);
        sample.theta_phys = std::atan2(y_phys, x_phys);
        if (sample.theta_phys < 0.0) sample.theta_phys += 2.0 * M_PI;
        sample.Br = 0.0;      // Not computed for Cartesian
        sample.Btheta = 0.0;

    } else {
        // Polar coordinates: use bilinear interpolation with theta wrapping
        sample.r_phys = std::sqrt(x_phys*x_phys + y_phys*y_phys);
        sample.theta_phys = std::atan2(y_phys, x_phys);
        if (sample.theta_phys < 0.0) sample.theta_phys += 2.0 * M_PI;

        // Interpolate Br, Btheta, mu at (r_phys, theta_phys)
        sample.Br = bilinearInterpolatePolar(Br, sample.r_phys, sample.theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);
        sample.Btheta = bilinearInterpolatePolar(Btheta, sample.r_phys, sample.theta_phys,
                                                r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);
        sample.mu = bilinearInterpolatePolar(mu_map, sample.r_phys, sample.theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);

        // Convert to Cartesian using the SAME theta from atan2
        sample.Bx = sample.Br * std::cos(sample.theta_phys) - sample.Btheta * std::sin(sample.theta_phys);
        sample.By = sample.Br * std::sin(sample.theta_phys) + sample.Btheta * std::cos(sample.theta_phys);
    }

    return sample;
}

// Polar coordinate version: directly use r_phys and theta_phys to avoid atan2 inconsistency
// CRITICAL: This ensures exact consistency between boundary point and sample point calculations
MagneticFieldAnalyzer::PolarSample MagneticFieldAnalyzer::sampleFieldsAtPolarPoint(double r_phys, double theta_phys) {
    MagneticFieldAnalyzer::PolarSample sample;

    // Convert to Cartesian for the record
    sample.x_phys = r_phys * std::cos(theta_phys);
    sample.y_phys = r_phys * std::sin(theta_phys);
    sample.r_phys = r_phys;
    sample.theta_phys = theta_phys;

    if (coordinate_system == "polar") {
        // Interpolate Br, Btheta, mu at (r_phys, theta_phys)
        sample.Br = bilinearInterpolatePolar(Br, r_phys, theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);
        sample.Btheta = bilinearInterpolatePolar(Btheta, r_phys, theta_phys,
                                                r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);
        sample.mu = bilinearInterpolatePolar(mu_map, r_phys, theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation);

        // Convert to Cartesian using the SAME theta (no atan2!)
        sample.Bx = sample.Br * std::cos(theta_phys) - sample.Btheta * std::sin(theta_phys);
        sample.By = sample.Br * std::sin(theta_phys) + sample.Btheta * std::cos(theta_phys);
    } else {
        // Cartesian system: fall back to x-y version
        return sampleFieldsAtPhysicalPoint(sample.x_phys, sample.y_phys);
    }

    return sample;
}

void MagneticFieldAnalyzer::calculateMaxwellStress(int step) {
    std::cout << "\n=== Calculating Maxwell Stress (polar-aware, Sobel normals) ===" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    // current_field_step is initialized to -1, so first call always recalculates
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;  // Mark field as calculated for this step
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    // --- get boundaries (image coords y-down) but DO NOT modify class image ---
    cv::Mat boundaries_img = detectBoundaries(); // image-coords (y down)
    if (boundaries_img.empty()) {
        std::cerr << "[ERROR] detectBoundaries() returned empty map." << std::endl;
        return;
    }

    // --- build mat_mask from class image (image coords) without modifying image ---
    cv::Mat mat_mask_img(image.rows, image.cols, CV_8U, cv::Scalar(0));
    // We will build full-material mask from config rgb
    // NOTE: we will create mat_mask_img for each material inside loop below, but if many materials
    // reuse logic can be adjusted. For clarity, build per-material later. Here just ensure boundaries_img valid.

    // --- Flip boundaries to analysis coordinates (both Cartesian and Polar) ---
    cv::Mat boundaries_flipped;
    cv::flip(boundaries_img, boundaries_flipped, 0); // 0 = flip around x-axis (vertical flip)

    // Precompute physical center (using same convention as previous outputs)
    double cx_physical, cy_physical;
    if (coordinate_system == "cartesian") {
        cx_physical = (static_cast<double>(image.cols) * dx) / 2.0;
        cy_physical = (static_cast<double>(image.rows) * dy) / 2.0;
    } else {
        // For polar coordinates, center is at origin (r=0, theta=any)
        // Torque calculation is about origin, so center offset is zero
        cx_physical = 0.0;
        cy_physical = 0.0;
    }

    // Calculate cumulative rotation angle offset for transient analysis
    // CRITICAL: This offset is used for physical coordinate calculation (x_phys, y_phys)
    // but must be REMOVED when sampling magnetic fields from image-based arrays
    double theta_offset = 0.0;

    // DEBUG: Print all conditions
    std::cout << "DEBUG calculateMaxwellStress: step=" << step
              << ", coord_sys=" << coordinate_system
              << ", transient.enabled=" << (transient_config.enabled ? "true" : "false")
              << ", transient.enable_sliding=" << (transient_config.enable_sliding ? "true" : "false") << std::endl;

    if (coordinate_system == "polar" && step >= 0 && transient_config.enabled && transient_config.enable_sliding) {
        theta_offset = step * transient_config.slide_pixels_per_step * dtheta;
        std::cout << "Transient analysis: cumulative rotation offset = " << theta_offset << " rad ("
                  << (theta_offset * 180.0 / M_PI) << " deg) at step " << step << std::endl;
    }

    // Prepare results containers
    force_results.clear();
    boundary_stress_vectors.clear();  // Clear boundary stress vectors for new calculation
    if (!config["materials"]) {
        std::cout << "No materials defined for force calculation" << std::endl;
        return;
    }

    const double mu0 = MU_0;
    const double EPS_NORMAL = 1e-12;
    const double DS_MIN = 1e-12;
    const int MAX_OUTER_SEARCH = 3;

    // For each material
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;
        bool calc_force = props["calc_force"].as<bool>(false);
        if (!calc_force) continue;

        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255,255,255});

        // Build mat_mask from image
        cv::Mat mat_mask_img_local(image.rows, image.cols, CV_8U, cv::Scalar(0));
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                cv::Vec3b px = image.at<cv::Vec3b>(r, c);
                if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                    mat_mask_img_local.at<uchar>(r, c) = 255;
                }
            }
        }

        // Flip material mask to analysis coords (both Cartesian and Polar)
        cv::Mat mat_mask_flipped;
        cv::flip(mat_mask_img_local, mat_mask_flipped, 0); // flip to analysis coords (y up)

        // compute Sobel on flipped mat_mask (we want gradients consistent with analysis indices)
        // Use periodic BC-aware Sobel for accurate normal vectors at boundaries
        cv::Mat gx, gy;
        applySobelWithPeriodicBC(mat_mask_flipped, gx, 1, 0, 3);
        applySobelWithPeriodicBC(mat_mask_flipped, gy, 0, 1, 3);

        ForceResult result;
        result.material_name = name;
        result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
        result.force_x = result.force_y = result.force_radial = 0.0;
        result.torque = result.torque_origin = result.torque_center = 0.0;
        result.pixel_count = 0;
        result.magnetic_energy = 0.0;  // Initialize magnetic potential energy

        std::cout << "\nCalculating force for material: " << name << " (using flipped masks)" << std::endl;

        int rows = mat_mask_flipped.rows;
        int cols = mat_mask_flipped.cols;

        // First pass: calculate stress at boundary pixels and store in map
        std::map<std::pair<int,int>, BoundaryStressPoint> stress_map;

        // Determine loop range based on periodic boundary conditions
        // For periodic BC, include image boundaries; otherwise skip them
        bool x_periodic_loop = (bc_left.type == "periodic" && bc_right.type == "periodic");
        bool y_periodic_loop = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
        int j_start = y_periodic_loop ? 0 : 1;
        int j_end = y_periodic_loop ? rows : (rows - 1);
        int i_start = x_periodic_loop ? 0 : 1;
        int i_end = x_periodic_loop ? cols : (cols - 1);

        // iterate over pixels in analysis coords (y up)
        for (int j = j_start; j < j_end; ++j) {
            for (int i = i_start; i < i_end; ++i) {
                // must be material pixel (mat_mask_flipped) and a boundary pixel (boundaries_flipped)
                if (mat_mask_flipped.at<uchar>(j, i) == 0) continue;

                // Check if this is a boundary pixel
                bool is_boundary = (boundaries_flipped.at<uchar>(j, i) != 0);

                // Skip non-boundary pixels in this pass
                if (!is_boundary) continue;

                // image-space (here analysis-space) gradient: gx = d/dx, gy = d/dy (but note y now up)
                double g_x = gx.at<double>(j, i);
                double g_y = gy.at<double>(j, i);

                // outward normal in image-space (material->background) = -grad
                double n_img_x = -g_x;
                double n_img_y = -g_y;
                double n_norm = std::sqrt(n_img_x*n_img_x + n_img_y*n_img_y);

                // fallback: 8-neighbor search to find adjacent background pixel
                if (n_norm < EPS_NORMAL) {
                    bool found = false;
                    // Check which boundaries are periodic
                    bool x_periodic_fallback = (bc_left.type == "periodic" && bc_right.type == "periodic");
                    bool y_periodic_fallback = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

                    const int offsets[8][2] = { {1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{1,-1} };
                    for (int k=0;k<8 && !found;++k) {
                        int ii = i + offsets[k][0];
                        int jj = j + offsets[k][1];

                        // Apply periodic wrapping or skip if out of bounds
                        bool valid = true;
                        if (ii < 0 || ii >= cols) {
                            if (x_periodic_fallback) {
                                ii = (ii + cols) % cols;
                            } else {
                                valid = false;
                            }
                        }
                        if (jj < 0 || jj >= rows) {
                            if (y_periodic_fallback) {
                                jj = (jj + rows) % rows;
                            } else {
                                valid = false;
                            }
                        }
                        if (!valid) continue;

                        if (mat_mask_flipped.at<uchar>(jj, ii) == 0) {
                            n_img_x = static_cast<double>(offsets[k][0]);
                            n_img_y = static_cast<double>(offsets[k][1]);
                            n_norm = std::sqrt(n_img_x*n_img_x + n_img_y*n_img_y);
                            found = true;
                        }
                    }
                    if (!found) continue;
                }

                // normalize image-space normal
                n_img_x /= n_norm;
                n_img_y /= n_norm;

                // ensure normal points to background (probe along normal)
                // For periodic boundaries, use wrap instead of clamp
                bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
                bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

                auto probe_bg = [&](int step)->bool {
                    int ii = i + static_cast<int>(std::round(n_img_x * step));
                    int jj = j + static_cast<int>(std::round(n_img_y * step));

                    // Apply periodic wrapping or clamping
                    if (x_periodic) {
                        ii = (ii + cols) % cols;
                    } else {
                        ii = std::clamp(ii, 0, cols-1);
                    }
                    if (y_periodic) {
                        jj = (jj + rows) % rows;
                    } else {
                        jj = std::clamp(jj, 0, rows-1);
                    }
                    return mat_mask_flipped.at<uchar>(jj, ii) == 0;
                };
                if (!probe_bg(1) && probe_bg(-1)) {
                    // flip normal
                    n_img_x = -n_img_x;
                    n_img_y = -n_img_y;
                }

                // find an outside sample pixel by stepping along normal
                // x_periodic and y_periodic already defined above for probe_bg()
                int i_out = i, j_out = j;
                bool got_out = false;
                for (int s=1; s<=MAX_OUTER_SEARCH; ++s) {
                    int ii = i + static_cast<int>(std::round(n_img_x * s));
                    int jj = j + static_cast<int>(std::round(n_img_y * s));

                    // Apply periodic wrapping or clamping
                    if (x_periodic) {
                        ii = (ii + cols) % cols;  // Wrap for periodic BC
                    } else {
                        ii = std::clamp(ii, 0, cols-1);  // Clamp for non-periodic BC
                    }
                    if (y_periodic) {
                        jj = (jj + rows) % rows;  // Wrap for periodic BC
                    } else {
                        jj = std::clamp(jj, 0, rows-1);  // Clamp for non-periodic BC
                    }

                    if (mat_mask_flipped.at<uchar>(jj, ii) == 0) {
                        i_out = ii; j_out = jj; got_out = true; break;
                    }
                }
                if (!got_out) {
                    // fallback choose nearest neighbor in direction where probe_bg true or opposite
                    if (probe_bg(1)) {
                        i_out = i + static_cast<int>(std::round(n_img_x));
                        j_out = j + static_cast<int>(std::round(n_img_y));
                    } else {
                        i_out = i - static_cast<int>(std::round(n_img_x));
                        j_out = j - static_cast<int>(std::round(n_img_y));
                    }

                    // Apply periodic wrapping or clamping
                    if (x_periodic) {
                        i_out = (i_out + cols) % cols;
                    } else {
                        i_out = std::clamp(i_out, 0, cols-1);
                    }
                    if (y_periodic) {
                        j_out = (j_out + rows) % rows;
                    } else {
                        j_out = std::clamp(j_out, 0, rows-1);
                    }
                }

                // Map pixel(i,j) -> physical coords & polar indices
                double x_phys = 0.0, y_phys = 0.0;
                double r_phys = 0.0, theta = 0.0;
                int ir = 0, jt = 0;
                if (coordinate_system == "polar") {
                    if (r_orientation == "horizontal") { ir = i; jt = j; }
                    else { ir = j; jt = i; }
                    ir = std::clamp(ir, 0, nr-1);
                    jt = std::clamp(jt, 0, ntheta-1);
                    r_phys = r_coords[ir];

                    // CRITICAL FIX: Account for cumulative rotation from image sliding
                    // theta_offset is calculated at function scope (line 1347-1352)
                    // Physical coordinates include cumulative rotation for correct torque calculation
                    theta = jt * dtheta + theta_offset;
                    x_phys = r_phys * std::cos(theta);
                    y_phys = r_phys * std::sin(theta);
                } else {
                    // for cartesian: i->x, j->y (since we flipped masks, j increases upward)
                    x_phys = static_cast<double>(i) * dx;
                    y_phys = static_cast<double>(j) * dy;
                    r_phys = std::sqrt(x_phys*x_phys + y_phys*y_phys);
                    theta = (r_phys > 0.0) ? std::atan2(y_phys, x_phys) : 0.0;
                }

                // CRITICAL FIX: Convert image-space normal to physical-space normal with proper scaling
                // GPT Review: "Sobel gradient needs physical scaling, especially in polar coordinates"
                double n_phys_x, n_phys_y;
                if (coordinate_system == "polar") {
                    // Polar coordinates: Scale by physical lengths (dr, r*dtheta)
                    // r_orientation=horizontal: i→r direction, j→θ direction
                    // Sobel gives image gradients (gx, gy) = (∂/∂i, ∂/∂j)
                    // Physical gradients: ∂/∂r = (∂/∂i)/dr, ∂/∂θ = (∂/∂j)/(r*dθ)

                    double n_r, n_theta;
                    if (r_orientation == "horizontal") {
                        // i→r, j→θ
                        n_r = n_img_x / dr;
                        n_theta = n_img_y / (r_phys * dtheta + 1e-12);  // Avoid division by zero at r=0
                    } else {
                        // i→θ, j→r
                        n_r = n_img_y / dr;
                        n_theta = n_img_x / (r_phys * dtheta + 1e-12);
                    }

                    // Convert polar normal to Cartesian
                    n_phys_x = n_r * std::cos(theta) - n_theta * std::sin(theta);
                    n_phys_y = n_r * std::sin(theta) + n_theta * std::cos(theta);

                    // Normalize
                    double norm = std::sqrt(n_phys_x * n_phys_x + n_phys_y * n_phys_y);
                    if (norm > EPS_NORMAL) {
                        n_phys_x /= norm;
                        n_phys_y /= norm;
                    }
                } else {
                    // Cartesian coordinates: Scale by dx, dy
                    n_phys_x = n_img_x / dx;
                    n_phys_y = n_img_y / dy;

                    // Normalize
                    double norm = std::sqrt(n_phys_x * n_phys_x + n_phys_y * n_phys_y);
                    if (norm > EPS_NORMAL) {
                        n_phys_x /= norm;
                        n_phys_y /= norm;
                    }
                }

                // basis vectors
                double er_x = std::cos(theta), er_y = std::sin(theta);
                double et_x = -std::sin(theta), et_y = std::cos(theta);

                // tangent vector physical
                double t_phys_x = -n_phys_y;
                double t_phys_y =  n_phys_x;
                double tnorm = std::sqrt(t_phys_x*t_phys_x + t_phys_y*t_phys_y);
                if (tnorm > 0.0) { t_phys_x /= tnorm; t_phys_y /= tnorm; }

                // components of tangent in (er,et)
                double t_r = t_phys_x * er_x + t_phys_y * er_y;
                double t_t = t_phys_x * et_x + t_phys_y * et_y;

                // pixel physical lengths
                double len_r = (coordinate_system == "polar") ? dr : dx;
                double len_t = (coordinate_system == "polar") ? ((r_phys > 0.0) ? (r_phys * dtheta) : 0.0) : dy;

                // ds
                double ds = std::sqrt( (t_r * len_r)*(t_r * len_r) + (t_t * len_t)*(t_t * len_t) );
                if (ds <= 0.0) ds = DS_MIN;

                // CRITICAL FIX: Sample B and μ at SAME physical point using bilinear interpolation
                // Calculate sample point in the SAME coordinate system to avoid atan2 inconsistency
                MagneticFieldAnalyzer::PolarSample sample;

                if (coordinate_system == "polar") {
                    // For polar coordinates: calculate sample point directly in polar coords
                    // This avoids atan2 inconsistency between boundary and sample points
                    double sample_distance = dr;

                    // Decompose normal into polar components
                    double n_r = n_phys_x * std::cos(theta) + n_phys_y * std::sin(theta);
                    double n_theta = -n_phys_x * std::sin(theta) + n_phys_y * std::cos(theta);

                    // Calculate sample point in polar coordinates (physical system with cumulative rotation)
                    double r_sample = r_phys + sample_distance * n_r;
                    double theta_sample_phys = theta + (r_phys > 0.0 ? (sample_distance * n_theta / r_phys) : 0.0);

                    // CRITICAL: Convert physical theta back to image theta for field sampling
                    // Magnetic fields are stored in image coordinate system (without cumulative rotation)
                    // Subtract theta_offset to get image-based theta for correct field lookup
                    double theta_sample_image = theta_sample_phys - theta_offset;

                    // FIX2: Wrap theta to [0, theta_range) for image coordinate system
                    // CRITICAL: Use theta_range (not 2π) to handle sector models correctly
                    while (theta_sample_image < 0.0) theta_sample_image += theta_range;
                    while (theta_sample_image >= theta_range) theta_sample_image -= theta_range;

                    // Sample fields using image coordinates
                    sample = sampleFieldsAtPolarPoint(r_sample, theta_sample_image);

                    // Convert sample coordinates back to physical system for record
                    sample.theta_phys = theta_sample_phys;
                } else {
                    // For Cartesian coordinates: use Cartesian sampling
                    double sample_distance = std::max(dx, dy);
                    double x_sample = x_phys + n_phys_x * sample_distance;
                    double y_sample = y_phys + n_phys_y * sample_distance;
                    sample = sampleFieldsAtPhysicalPoint(x_sample, y_sample);
                }

                // Extract Cartesian B components and μ
                double bx_out = sample.Bx;
                double by_out = sample.By;
                double mu_local = sample.mu;

                // Maxwell stress calculation: Use Cartesian coordinates for BOTH coordinate systems
                // CRITICAL FIX: Avoid basis vector inconsistencies by always computing in Cartesian
                // For polar coordinates, bx_out and by_out have already been computed from Br, Btheta
                // at lines 1484-1485, so we can use them directly.

                // Cartesian Maxwell stress tensor: T = B⊗H - 0.5(B·H)I
                // where H = B/μ (linear material)
                double Hx = bx_out / mu_local;
                double Hy = by_out / mu_local;
                double B_dot_H = bx_out * Hx + by_out * Hy;

                // T_ij = Bi*Hj - 0.5*(B·H)*δij
                double T_xx = bx_out * Hx - 0.5 * B_dot_H;
                double T_yy = by_out * Hy - 0.5 * B_dot_H;
                double T_xy = bx_out * Hy; // = by_out * Hx by symmetry

                // Traction: t = T · n (Cartesian)
                double fx = T_xx * n_phys_x + T_xy * n_phys_y;
                double fy = T_xy * n_phys_x + T_yy * n_phys_y;

                // accumulate
                result.pixel_count++;
                result.force_x += fx * ds;
                result.force_y += fy * ds;

                // Radial force: for polar use T_rr component, for cartesian use magnitude
                if (coordinate_system == "polar") {
                    // Use consistent polar basis at sample point
                    double theta_sample = sample.theta_phys;
                    double er_sample_x = std::cos(theta_sample);
                    double er_sample_y = std::sin(theta_sample);
                    double et_sample_x = -std::sin(theta_sample);
                    double et_sample_y = std::cos(theta_sample);

                    double n_r = n_phys_x * er_sample_x + n_phys_y * er_sample_y;
                    double n_t = n_phys_x * et_sample_x + n_phys_y * et_sample_y;

                    // Use polar B components from sample
                    double B_r = sample.Br;
                    double B_t = sample.Btheta;

                    // Maxwell stress using H = B/μ
                    double H_r = B_r / mu_local;
                    double H_t = B_t / mu_local;
                    double B_dot_H_local = B_r * H_r + B_t * H_t;
                    double T_rr = B_r * H_r - 0.5 * B_dot_H_local;
                    double T_rt = B_r * H_t;
                    double t_r_s = T_rr * n_r + T_rt * n_t;
                    result.force_radial += t_r_s * ds;
                }
                // For cartesian, force_radial is computed as magnitude after loop

                double dFx = fx * ds;
                double dFy = fy * ds;

                // Torque calculation: use Cartesian formula for both coordinate systems
                // IMPORTANT: For consistency and to avoid basis vector issues in polar coordinates,
                // always use the Cartesian formula: τ_z = x * F_y - y * F_x
                // x_phys, y_phys are already computed for both coordinate systems (lines 1413-1418)
                // fx, fy are already converted to Cartesian (lines 1522-1523 or 1567-1568)
                result.torque_origin += x_phys * dFy - y_phys * dFx;

                if (coordinate_system == "polar") {
                    // In polar coordinates centered at origin, torque_center = torque_origin
                    result.torque_center += x_phys * dFy - y_phys * dFx;
                } else {
                    // Cartesian coordinates: use offset from center
                    result.torque_center += (x_phys - cx_physical) * dFy - (y_phys - cy_physical) * dFx;
                }
                result.torque = result.torque_origin;

                // Record boundary stress vector in map (for later full-grid export)
                BoundaryStressPoint stress_point;
                stress_point.i_pixel = i;
                stress_point.j_pixel = j;
                stress_point.x_phys = x_phys;
                stress_point.y_phys = y_phys;
                stress_point.fx = fx;
                stress_point.fy = fy;
                stress_point.ds = ds;
                stress_point.nx = n_phys_x;
                stress_point.ny = n_phys_y;
                stress_point.Bx = sample.Bx;
                stress_point.By = sample.By;
                stress_point.B_magnitude = std::sqrt(sample.Bx * sample.Bx + sample.By * sample.By);
                stress_point.material = name;
                stress_map[std::make_pair(i, j)] = stress_point;
            }
        }

        // Second pass: export all material pixels with stress vectors (0 for non-boundary)
        std::cout << "  Exporting stress vectors for all material pixels..." << std::endl;
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                // Skip if not material pixel
                if (mat_mask_flipped.at<uchar>(j, i) == 0) continue;

                // Check if this pixel has calculated stress (is boundary)
                auto key = std::make_pair(i, j);
                bool has_stress = (stress_map.find(key) != stress_map.end());

                BoundaryStressPoint export_point;
                export_point.i_pixel = i;
                export_point.j_pixel = j;
                export_point.material = name;

                if (has_stress) {
                    // Use calculated stress values
                    export_point = stress_map[key];
                } else {
                    // Zero values for non-boundary pixels
                    // Calculate physical coordinates
                    double x_phys = 0.0, y_phys = 0.0;
                    if (coordinate_system == "polar") {
                        int ir = (r_orientation == "horizontal") ? i : j;
                        int jt = (r_orientation == "horizontal") ? j : i;
                        ir = std::clamp(ir, 0, nr-1);
                        jt = std::clamp(jt, 0, ntheta-1);
                        double r_phys = r_coords[ir];
                        double theta = jt * dtheta;
                        x_phys = r_phys * std::cos(theta);
                        y_phys = r_phys * std::sin(theta);
                    } else {
                        x_phys = static_cast<double>(i) * dx;
                        y_phys = static_cast<double>(j) * dy;
                    }

                    export_point.x_phys = x_phys;
                    export_point.y_phys = y_phys;
                    export_point.fx = 0.0;
                    export_point.fy = 0.0;
                    export_point.ds = 0.0;
                    export_point.nx = 0.0;
                    export_point.ny = 0.0;
                    export_point.Bx = 0.0;
                    export_point.By = 0.0;
                    export_point.B_magnitude = 0.0;
                }

                boundary_stress_vectors.push_back(export_point);
            }
        }

        // Calculate magnetic potential energy for this material: W = ∫ B²/(2μ₀μᵣ) dV
        double mu_r = props["mu_r"].as<double>(1.0);
        double mu_material = mu0 * mu_r;
        double energy = 0.0;

        // Iterate over all pixels of this material to calculate energy
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                // calc all field energy, not just boundary
                // Check if this pixel belongs to the material
                // if (mat_mask_flipped.at<uchar>(j, i) == 0) continue;

                // Get permeability at this pixel (mu_map shape matches mat_mask_flipped shape)
                mu_material = mu_map(j, i); 

                // Get magnetic field at this pixel
                double bx = 0.0, by = 0.0;
                if (coordinate_system == "cartesian") {
                    int bi = std::clamp(i, 0, static_cast<int>(Bx.cols()) - 1);
                    int bj = std::clamp(j, 0, static_cast<int>(Bx.rows()) - 1);
                    bx = Bx(bj, bi);
                    by = By(bj, bi);
                } else {
                    int ir = (r_orientation == "horizontal") ? i : j;
                    int jt = (r_orientation == "horizontal") ? j : i;
                    ir = std::clamp(ir, 0, nr - 1);
                    jt = std::clamp(jt, 0, ntheta - 1);
                    double br_sample = 0.0, bt_sample = 0.0;
                    if (r_orientation == "horizontal") {
                        br_sample = Br(jt, ir);
                        bt_sample = Btheta(jt, ir);
                    } else {
                        br_sample = Br(ir, jt);
                        bt_sample = Btheta(ir, jt);
                    }
                    double theta_sample = jt * dtheta;
                    bx = br_sample * std::cos(theta_sample) - bt_sample * std::sin(theta_sample);
                    by = br_sample * std::sin(theta_sample) + bt_sample * std::cos(theta_sample);
                }

                // Energy density: w = B²/(2μ) [J/m³]
                double B_sq = bx * bx + by * by;
                double w_density = B_sq / (2.0 * mu_material);

                // Volume element (per unit depth): dV = dA * 1 [m³/m] = [m²]
                double dA = 0.0;
                if (coordinate_system == "polar") {
                    int ir = (r_orientation == "horizontal") ? i : j;
                    ir = std::clamp(ir, 0, nr - 1);
                    double r = r_coords[ir];
                    dA = r * dr * dtheta;  // Area element in polar coordinates
                } else {
                    dA = dx * dy;  // Area element in Cartesian coordinates
                }

                energy += w_density * dA;  // [J/m] per unit depth
            }
        }

        result.magnetic_energy = energy;

        // For Cartesian coordinates, set force_radial as total force magnitude
        if (coordinate_system == "cartesian") {
            result.force_radial = std::sqrt(result.force_x * result.force_x +
                                           result.force_y * result.force_y);
        }

        std::cout << "  Material: " << name << std::endl;
        std::cout << "    Boundary pixels: " << result.pixel_count << std::endl;
        std::cout << "    Radial force (outward): " << result.force_radial << " N/m (per unit depth)" << std::endl;
        std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
        std::cout << "    Torque about origin: " << result.torque_origin << " N·m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N·m (per unit depth)" << std::endl;
        std::cout << "    Magnetic energy: " << result.magnetic_energy << " J/m (per unit depth)" << std::endl;

        force_results.push_back(result);
    }

    std::cout << "\nMaxwell stress (polar-aware, Sobel normals) calculation complete!" << std::endl;
}

double MagneticFieldAnalyzer::calculateTotalMagneticEnergy(int step) {
    std::cout << "\n=== Calculating Total Magnetic Energy ===" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    // current_field_step is initialized to -1, so first call always recalculates
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;  // Mark field as calculated for this step
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    // Check mu_map
    if (mu_map.size() == 0) {
        std::cerr << "[ERROR] mu_map is not initialized" << std::endl;
        return 0.0;
    }

    double total_energy = 0.0;

    if (coordinate_system == "cartesian") {
        // Cartesian coordinates: use Eigen array operations for efficiency
        int rows = Bx.rows();
        int cols = Bx.cols();

        // Verify dimensions match
        if (mu_map.rows() != rows || mu_map.cols() != cols) {
            std::cerr << "[ERROR] Dimension mismatch: Bx(" << rows << "x" << cols
                      << ") vs mu_map(" << mu_map.rows() << "x" << mu_map.cols() << ")" << std::endl;
            return 0.0;
        }

        // Calculate B² using Eigen array operations (element-wise)
        Eigen::ArrayXXd B_squared = Bx.array().square() + By.array().square();

        // Calculate energy density: w = B²/(2μ) [J/m³]
        Eigen::ArrayXXd energy_density = B_squared / (2.0 * mu_map.array());

        // Sum all energy densities and multiply by volume element
        double dV = dx * dy;  // [m²] per unit depth
        total_energy = energy_density.sum() * dV;  // [J/m]

        std::cout << "  Grid size: " << rows << " x " << cols << std::endl;
        std::cout << "  dx = " << dx << " m, dy = " << dy << " m" << std::endl;
        std::cout << "  Max energy density: " << energy_density.maxCoeff() << " J/m³" << std::endl;
        std::cout << "  Total energy: " << total_energy << " J/m (per unit depth)" << std::endl;

    } else {
        // Polar coordinates
        int rows = Br.rows();
        int cols = Br.cols();

        if (mu_map.rows() != rows || mu_map.cols() != cols) {
            std::cerr << "[ERROR] Dimension mismatch in polar coordinates" << std::endl;
            return 0.0;
        }

        // Calculate B² = Br² + Bθ²
        Eigen::ArrayXXd B_squared = Br.array().square() + Btheta.array().square();

        // Calculate energy density
        Eigen::ArrayXXd energy_density = B_squared / (2.0 * mu_map.array());

        // Sum with polar volume element: dV = r * dr * dθ
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                int ir = (r_orientation == "horizontal") ? j : i;
                ir = std::clamp(ir, 0, nr - 1);
                double r = r_coords[ir];
                double dV = r * dr * dtheta;
                total_energy += energy_density(i, j) * dV;
            }
        }

        std::cout << "  Grid size (r x θ): " << rows << " x " << cols << std::endl;
        std::cout << "  Total energy: " << total_energy << " J/m (per unit depth)" << std::endl;
    }

    return total_energy;
}


void MagneticFieldAnalyzer::exportForcesToCSV(const std::string& output_path) const {
    if (force_results.empty()) {
        std::cout << "No force results to export" << std::endl;
        return;
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open force output file: " + output_path);
    }

    // CSV header (include both torque measures and magnetic energy)
    file << "Material,RGB_R,RGB_G,RGB_B,Force_X[N/m],Force_Y[N/m],Force_Magnitude[N/m],Force_Radial[N/m],Torque_Origin[N·m],Torque_Center[N·m],Magnetic_Energy[J/m],Boundary_Pixels\n";
    file << "# Note: Forces and energies are per unit depth (2D analysis)\n";
    file << "# Torque_Origin: torque about origin (polar)\n";
    file << "# Torque_Center: torque about image center (x=center_x, y=center_y)\n";
    file << "# Magnetic_Energy: magnetic potential energy W = ∫ B²/(2μ) dV\n";

    file << std::scientific << std::setprecision(6);

    for (const auto& result : force_results) {
        double force_mag = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);

        file << result.material_name << ","
             << static_cast<int>(result.rgb[0]) << ","
             << static_cast<int>(result.rgb[1]) << ","
             << static_cast<int>(result.rgb[2]) << ","
             << result.force_x << ","
             << result.force_y << ","
             << force_mag << ","
             << result.force_radial << ","
             << result.torque_origin << ","
             << result.torque_center << ","
             << result.magnetic_energy << ","
             << result.pixel_count << "\n";
    }

    file.close();
    std::cout << "Force results exported to: " << output_path << std::endl;
}

void MagneticFieldAnalyzer::exportBoundaryStressVectors(const std::string& output_path) const {
    if (boundary_stress_vectors.empty()) {
        std::cout << "No boundary stress vectors to export" << std::endl;
        return;
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open boundary stress vector output file: " + output_path);
    }

    // CSV header
    file << "i_pixel,j_pixel,x[m],y[m],fx[N/m],fy[N/m],ds[m],nx,ny,Bx[T],By[T],B_mag[T],Material\n";
    file << "# Boundary stress vectors for visualization (全材料ピクセル分、境界以外は0)\n";
    file << "# (i_pixel,j_pixel): pixel coordinates in analysis space (flipped, y-up)\n";
    file << "# (x,y): physical coordinates [m]\n";
    file << "# (fx,fy): Maxwell stress force per unit length [N/m] (0 for non-boundary)\n";
    file << "# ds: boundary segment length [m] (0 for non-boundary)\n";
    file << "# (nx,ny): outward unit normal vector (0 for non-boundary)\n";
    file << "# (Bx,By): magnetic field at boundary [T] (0 for non-boundary)\n";
    file << "# B_mag: |B| [T] (0 for non-boundary)\n";

    file << std::scientific << std::setprecision(6);

    for (const auto& point : boundary_stress_vectors) {
        file << point.i_pixel << ","
             << point.j_pixel << ","
             << point.x_phys << ","
             << point.y_phys << ","
             << point.fx << ","
             << point.fy << ","
             << point.ds << ","
             << point.nx << ","
             << point.ny << ","
             << point.Bx << ","
             << point.By << ","
             << point.B_magnitude << ","
             << point.material << "\n";
    }

    file.close();
    std::cout << "Boundary stress vectors exported to: " << output_path << std::endl;
    std::cout << "  Total boundary points: " << boundary_stress_vectors.size() << std::endl;
}

void MagneticFieldAnalyzer::exportBoundaryImage(const std::string& output_path) const {
    if (boundary_image.empty()) {
        std::cerr << "Warning: No boundary image to export" << std::endl;
        return;
    }

    cv::imwrite(output_path, boundary_image);
    std::cout << "Boundary image exported to: " << output_path << std::endl;
}

void MagneticFieldAnalyzer::exportResults(const std::string& base_folder, int step_number) {
    // Create base folder
    std::string mkdir_cmd = "mkdir -p \"" + base_folder + "\"";
    system(mkdir_cmd.c_str());

    // Create subfolders
    std::string az_folder = base_folder + "/Az";
    std::string mu_folder = base_folder + "/Mu";
    std::string jz_folder = base_folder + "/Jz";
    std::string boundary_folder = base_folder + "/BoundaryImg";
    std::string forces_folder = base_folder + "/Forces";
    std::string input_image_folder = base_folder + "/InputImg";
    std::string energy_density_folder = base_folder + "/EnergyDensity";

    system(("mkdir -p \"" + az_folder + "\"").c_str());
    system(("mkdir -p \"" + mu_folder + "\"").c_str());
    system(("mkdir -p \"" + jz_folder + "\"").c_str());
    system(("mkdir -p \"" + boundary_folder + "\"").c_str());
    system(("mkdir -p \"" + forces_folder + "\"").c_str());
    system(("mkdir -p \"" + input_image_folder + "\"").c_str());
    system(("mkdir -p \"" + energy_density_folder + "\"").c_str());

    // Format step number with leading zeros (e.g., step_0001)
    std::ostringstream step_str;
    step_str << "step_" << std::setfill('0') << std::setw(4) << step_number+1;
    std::string step_name = step_str.str();
    cv::Mat image_BGR = image.clone();

    // Export Az
    std::string az_path = az_folder + "/" + step_name + ".csv";
    exportAzToCSV(az_path);

    // Export Mu
    std::string mu_path = mu_folder + "/" + step_name + ".csv";
    exportMuToCSV(mu_path);

    // Export Jz
    std::string jz_path = jz_folder + "/" + step_name + ".csv";
    exportJzToCSV(jz_path);

    // Export boundary image if available
    if (!boundary_image.empty()) {
        std::string boundary_path = boundary_folder + "/" + step_name + ".png";
        exportBoundaryImage(boundary_path);
    }

    // Export input image (current state of material distribution)
    if (!image.empty()) {
        std::string input_image_path = input_image_folder + "/" + step_name + ".png";
        cv::cvtColor(image, image_BGR, cv::COLOR_RGB2BGR); // Convert back to BGR for saving
        cv::imwrite(input_image_path, image_BGR);
        std::cout << "Input image exported to: " << input_image_path << std::endl;
    }

    // Export forces if available
    if (!force_results.empty()) {
        std::string forces_path = forces_folder + "/" + step_name + ".csv";
        exportForcesToCSV(forces_path);
    }

    // Export boundary stress vectors if available
    if (!boundary_stress_vectors.empty()) {
        std::string stress_vectors_folder = base_folder + "/StressVectors";
        system(("mkdir -p \"" + stress_vectors_folder + "\"").c_str());
        std::string stress_vectors_path = stress_vectors_folder + "/" + step_name + ".csv";
        exportBoundaryStressVectors(stress_vectors_path);
    }

    // Export energy density distribution
    if (coordinate_system == "cartesian" && Bx.size() > 0 && By.size() > 0 && mu_map.size() > 0) {
        // Calculate B² and energy density
        Eigen::ArrayXXd B_squared = Bx.array().square() + By.array().square();
        Eigen::ArrayXXd energy_density = B_squared / (2.0 * mu_map.array());

        // Export to CSV
        std::string energy_density_path = energy_density_folder + "/" + step_name + ".csv";
        std::ofstream file(energy_density_path);
        if (file.is_open()) {
            for (int j = 0; j < energy_density.rows(); ++j) {
                for (int i = 0; i < energy_density.cols(); ++i) {
                    file << energy_density(j, i);
                    if (i < energy_density.cols() - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Energy density exported to: " << energy_density_path << std::endl;
        }
    } else if (coordinate_system == "polar" && Br.size() > 0 && Btheta.size() > 0 && mu_map.size() > 0) {
        // Polar coordinates: Calculate B² and energy density
        Eigen::ArrayXXd B_squared = Br.array().square() + Btheta.array().square();
        Eigen::ArrayXXd energy_density = B_squared / (2.0 * mu_map.array());

        // Export to CSV
        std::string energy_density_path = energy_density_folder + "/" + step_name + ".csv";
        std::ofstream file(energy_density_path);
        if (file.is_open()) {
            for (int j = 0; j < energy_density.rows(); ++j) {
                for (int i = 0; i < energy_density.cols(); ++i) {
                    file << energy_density(j, i);
                    if (i < energy_density.cols() - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Energy density (polar) exported to: " << energy_density_path << std::endl;
        }
    }

    std::cout << "\n=== Results exported to folder structure ===" << std::endl;
    std::cout << "Base folder: " << base_folder << std::endl;
    std::cout << "Step number: " << step_number << std::endl;
}

// ============================================================================
// Polar Coordinate Methods
// ============================================================================

double MagneticFieldAnalyzer::getMuAtInterfacePolar(double r_idx, int theta_idx, const std::string& direction) const {
    int r_idx_int = static_cast<int>(r_idx);

    if (direction == "r") {
        // Radial direction interface
        // Use floating-point safe comparison (avoid direct == comparison)
        const double eps = 1e-12;
        if (std::fabs(r_idx - r_idx_int) < eps) {
            // Integer index (or very close to integer)
            if (r_idx_int > 0 && r_idx_int < nr - 1) {
                // Harmonic mean of adjacent cells
                double mu1 = getMuPolar(mu_map, r_idx_int, theta_idx, r_orientation);
                double mu2 = getMuPolar(mu_map, r_idx_int + 1, theta_idx, r_orientation);
                return 2.0 / (1.0/mu1 + 1.0/mu2);
            } else {
                // Boundary: use nearest valid cell
                int r_safe = std::min(std::max(r_idx_int, 0), nr - 1);
                return getMuPolar(mu_map, r_safe, theta_idx, r_orientation);
            }
        } else {
            // Half-integer index (e.g., i ± 0.5 for interface)
            int r_low = static_cast<int>(std::floor(r_idx));
            if (r_low >= 0 && r_low < nr - 1) {
                // Harmonic mean at interface between r_low and r_low+1
                double mu1 = getMuPolar(mu_map, r_low, theta_idx, r_orientation);
                double mu2 = getMuPolar(mu_map, r_low + 1, theta_idx, r_orientation);
                return 2.0 / (1.0/mu1 + 1.0/mu2);
            } else {
                // Boundary or out of range: clamp to valid cell
                int r_safe = std::min(std::max(r_low, 0), nr - 1);
                return getMuPolar(mu_map, r_safe, theta_idx, r_orientation);
            }
        }
    } else {
        // Theta direction interface (for future use with θ-varying μ)
        int theta_next = (theta_idx + 1) % ntheta;
        double mu1 = getMuPolar(mu_map, r_idx_int, theta_idx, r_orientation);
        double mu2 = getMuPolar(mu_map, r_idx_int, theta_next, r_orientation);
        return 2.0 / (1.0/mu1 + 1.0/mu2);
    }
}

void MagneticFieldAnalyzer::buildMatrixPolar(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
    // Build FDM system matrix and right-hand side (Polar coordinates)
    int n = nr * ntheta;
    A.resize(n, n);
    rhs.resize(n);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 7);  // Estimate: 7 non-zeros per row

    // Determine boundary condition type (move outside loop for efficiency)
    // IMPORTANT: In polar coordinates, θ direction is ALWAYS periodic by physical definition
    // θ=0 and θ=theta_range represent the same radial line
    bool is_periodic = true;  // Always periodic in polar coordinates

    // Build equation for each grid point
    for (int i = 0; i < nr; i++) {  // Radial direction
        for (int j = 0; j < ntheta; j++) {  // Angular direction
            int idx = i * ntheta + j;
            double r = r_coords[i];

            // Angular boundary conditions (only for sector domain, not periodic)
            if (!is_periodic) {
                // Sector domain: Dirichlet boundary at θ=0 and θ=theta_range
                if (j == 0 || j == ntheta - 1) {
                    // Dirichlet BC: Az = 0 at sector boundaries
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = 0.0;
                    continue;
                }
            }

            // Radial boundary conditions (Dirichlet only handled here)
            // Neumann boundaries use ghost-elimination in interior stencil below
            if (i == 0 && bc_inner.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_inner.value;
                continue;
            }

            if (i == nr - 1 && bc_outer.type == "dirichlet") {
                triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                rhs(idx) = bc_outer.value;
                continue;
            }

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
            // CRITICAL FIX: Use r-weighted formulation for symmetry
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
                    triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_im_eff + a_ip));
                    coeff_center -= (a_im_eff + a_ip);
                } else {
                    // Standard ghost elimination
                    triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_im + a_ip));
                    coeff_center -= (a_im + a_ip);
                }
            } else if (i == nr - 1 && bc_outer.type == "neumann") {
                // Ghost elimination: Az_{nr} = Az_{nr-2}
                // Stencil: a_im·Az_{nr-2} + a_ip·Az_{nr} → (a_im + a_ip)·Az_{nr-2}
                triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, a_im + a_ip));
                coeff_center -= (a_im + a_ip);
            } else {
                // Standard interior stencil
                // Check if neighbors are Dirichlet boundaries and move bc values to RHS
                bool inner_neighbor_is_dirichlet = (i == 1) && (bc_inner.type == "dirichlet");
                bool outer_neighbor_is_dirichlet = (i == nr - 2) && (bc_outer.type == "dirichlet");

                if (!inner_neighbor_is_dirichlet) {
                    triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, a_im));
                } else {
                    // Inner neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= a_im * bc_inner.value;
                }

                if (!outer_neighbor_is_dirichlet) {
                    triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, a_ip));
                } else {
                    // Outer neighbor is Dirichlet: move bc value to RHS
                    rhs(idx) -= a_ip * bc_outer.value;
                }

                coeff_center -= (a_im + a_ip);
            }

            // Angular second derivative term: (1/r²) · (1/μ) · ∂²Az/∂θ²
            // Simplified using cell-centered μ (for θ-varying μ, use interface values)
            // double coeff_theta = 1.0 / (r * r * mu_current * dtheta * dtheta);

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

            // Check if neighbors are sector Dirichlet boundaries (Az=0)
            // For sector domain: j=0 and j=ntheta-1 have Dirichlet BC
            bool theta_prev_is_dirichlet = (!is_periodic) && (j == 1);  // j_prev_idx = 0
            bool theta_next_is_dirichlet = (!is_periodic) && (j == ntheta - 2);  // j_next_idx = ntheta-1

            // CRITICAL FIX: For periodic boundaries, also check if theta neighbor is on radial Dirichlet boundary
            // This ensures symmetry when periodic wrap connects to Dirichlet points
            if (is_periodic) {
                int idx_theta_prev = i * ntheta + j_prev_idx;
                int idx_theta_next = i * ntheta + j_next_idx;

                // Check if wrapped theta neighbor is on radial Dirichlet boundary
                // (This handles the case where j wraps around and i is on boundary)
                // For now, we don't have this case in our problem, but keep the structure consistent
                // The key is to use the same harmonic mean calculation in both directions
            }

            if (!theta_prev_is_dirichlet) {
                triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_prev_idx, coeff_theta_prev));
            } else {
                // Sector boundary has Az=0, so rhs(idx) -= coeff_theta_prev * 0.0 (no change)
                // Don't add matrix entry to maintain symmetry
            }

            if (!theta_next_is_dirichlet) {
                triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_next_idx, coeff_theta_next));
            } else {
                // Sector boundary has Az=0, so rhs(idx) -= coeff_theta_next * 0.0 (no change)
                // Don't add matrix entry to maintain symmetry
            }

            coeff_center -= (coeff_theta_prev + coeff_theta_next);

            // Center coefficient
            triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

            // Right-hand side (current density)
            // After r-weighting: RHS becomes -r·Jz
            rhs(idx) = -getJzPolar(jz_map, i, j, r_orientation) * r;
        }
    }

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

void MagneticFieldAnalyzer::buildAndSolveSystemPolar() {
    std::cout << "\n=== Building FDM system (Polar coordinates) ===" << std::endl;
    std::cout << "Mesh: r direction " << nr << " points, theta direction " << ntheta << " points" << std::endl;
    std::cout << "Radial range: " << r_start << " ~ " << r_end << " m" << std::endl;
    std::cout << "Boundary conditions: inner=" << bc_inner.type << ", outer=" << bc_outer.type << std::endl;

    int n = nr * ntheta;
    Eigen::SparseMatrix<double> A(n, n);
    Eigen::VectorXd rhs(n);

    // Build matrix using the separated method
    buildMatrixPolar(A, rhs);

    std::cout << "Matrix size: " << A.rows() << "x" << A.cols() << std::endl;
    std::cout << "Non-zero elements: " << A.nonZeros() << std::endl;

    // Check matrix symmetry (important for verifying correct discretization)
    Eigen::SparseMatrix<double> A_T = A.transpose();
    double symmetry_error = (A - A_T).norm();
    double A_norm = A.norm();
    double relative_symmetry_error = (A_norm > 1e-12) ? (symmetry_error / A_norm) : symmetry_error;
    std::cout << "Matrix symmetry: ||A - A^T|| = " << symmetry_error
              << ", relative error = " << relative_symmetry_error << std::endl;
    if (relative_symmetry_error > 1e-8) {
        std::cerr << "Warning: Matrix is not symmetric! Relative error = " << relative_symmetry_error << std::endl;
        std::cerr << "This may indicate incorrect discretization or boundary conditions." << std::endl;
    }

    // Solve linear system
    std::cout << "\n=== Solving linear system ===" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Matrix factorization failed!");
    }

    Eigen::VectorXd Az_flat = solver.solve(rhs);

    // Check residual
    Eigen::VectorXd res = A * Az_flat - rhs;
    double res_norm = res.norm();
    double rhs_norm = rhs.norm();
    std::cout << "[DBG] Residual norm ||A x - b|| = " << res_norm
            << ", relative = " << (rhs_norm>0 ? res_norm / rhs_norm : res_norm) << std::endl;

        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Linear system solve failed!");
        }

    // Reshape solution to matrix form
    // Index mapping: idx = i * ntheta + j, where i is radial, j is angular
    //
    // For consistency with image storage and exportAzToCSV:
    // - r_orientation == "horizontal": image(θ, r), so Az should be (ntheta, nr) = (rows, cols)
    // - r_orientation == "vertical": image(r, θ), so Az should be (nr, ntheta) = (rows, cols)
    //
    // We store Az in image-compatible format: Az(row, col)
    if (r_orientation == "horizontal") {
        // Image is (ntheta, nr), transpose the solution
        Az.resize(ntheta, nr);
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < ntheta; j++) {
                Az(j, i) = Az_flat(i * ntheta + j);  // Transpose: Az(θ, r) from idx(r, θ)
            }
        }
    } else {  // vertical
        // Image is (nr, ntheta), no transpose needed
        Az.resize(nr, ntheta);
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < ntheta; j++) {
                Az(i, j) = Az_flat(i * ntheta + j);
            }
        }
    }

    std::cout << "Solution complete!" << std::endl;

    // Calculate magnetic field
    calculateMagneticFieldPolar();
}

void MagneticFieldAnalyzer::calculateMagneticFieldPolar() {
    std::cout << "Calculating magnetic field (polar coordinates)" << std::endl;

    // Allocate field arrays in image-compatible format
    if (r_orientation == "horizontal") {
        Br = Eigen::MatrixXd::Zero(ntheta, nr);
        Btheta = Eigen::MatrixXd::Zero(ntheta, nr);
    } else {
        Br = Eigen::MatrixXd::Zero(nr, ntheta);
        Btheta = Eigen::MatrixXd::Zero(nr, ntheta);
    }

    // IMPORTANT: In polar coordinates, θ direction is ALWAYS periodic by physical definition
    // θ=0 and θ=theta_range represent the same radial line
    bool is_periodic = true;  // Always periodic in polar coordinates

    // Calculate field components: Br = (1/r)∂Az/∂θ, Btheta = -∂Az/∂r
    for (int i = 0; i < nr; i++) {
        double r = r_coords[i];
        for (int j = 0; j < ntheta; j++) {
            // Get Az values with proper indexing based on r_orientation
            auto getAz = [&](int r_idx, int theta_idx) -> double {
                if (r_orientation == "horizontal") {
                    return Az(theta_idx, r_idx);  // Az(θ, r)
                } else {
                    return Az(r_idx, theta_idx);  // Az(r, θ)
                }
            };

            auto setBr = [&](int r_idx, int theta_idx, double value) {
                if (r_orientation == "horizontal") {
                    Br(theta_idx, r_idx) = value;
                } else {
                    Br(r_idx, theta_idx) = value;
                }
            };

            auto setBtheta = [&](int r_idx, int theta_idx, double value) {
                if (r_orientation == "horizontal") {
                    Btheta(theta_idx, r_idx) = value;
                } else {
                    Btheta(r_idx, theta_idx) = value;
                }
            };

            // Br = (1/r) * dAz/dtheta
            int j_next, j_prev;
            double denom_theta;  // Denominator for finite difference

            if (is_periodic) {
                // Periodic boundary: use modulo wrapping (% ntheta)
                j_prev = (j - 1 + ntheta) % ntheta;
                j_next = (j + 1) % ntheta;
                denom_theta = 2.0 * dtheta;  // Central difference
            } else {
                // Sector domain: use one-sided differences at boundaries
                if (j == 0) {
                    // Forward difference at θ=0
                    j_prev = 0;
                    j_next = 1;
                    denom_theta = dtheta;
                } else if (j == ntheta - 1) {
                    // Backward difference at θ=theta_range
                    j_prev = j - 1;
                    j_next = j;
                    denom_theta = dtheta;
                } else {
                    // Central difference for interior
                    j_prev = j - 1;
                    j_next = j + 1;
                    denom_theta = 2.0 * dtheta;
                }
            }
            double dAz_dtheta = (getAz(i, j_next) - getAz(i, j_prev)) / denom_theta;
            double safe_r = (r > 1e-15) ? r : 1e-15;
            setBr(i, j, dAz_dtheta / safe_r);

            // Btheta = -dAz/dr
            double dAz_dr;
            if (i == 0) {
                dAz_dr = (getAz(1, j) - getAz(0, j)) / dr;
            } else if (i == nr - 1) {
                dAz_dr = (getAz(nr - 1, j) - getAz(nr - 2, j)) / dr;
            } else {
                dAz_dr = (getAz(i + 1, j) - getAz(i - 1, j)) / (2.0 * dr);
            }
            setBtheta(i, j, -dAz_dr);
        }
    }

    // B magnitude diagnostics
    double bmax = 0.0;
    for (int i=0;i<Br.rows();++i){
    for (int j=0;j<Br.cols();++j){
        double valr = Br(i,j);
        double valt = Btheta(i,j);
        double mag = std::sqrt(valr*valr + valt*valt);
        if (mag > bmax) bmax = mag;
    }
    }
    std::cout << "[DBG] Max |B| = " << bmax << std::endl;

    std::cout << "Magnetic field calculated" << std::endl;
}

// ============================================================================
// Dynamic Jz Parsing and Evaluation
// ============================================================================

MagneticFieldAnalyzer::JzValue MagneticFieldAnalyzer::parseJzValue(const YAML::Node& jz_node) {
    JzValue result;

    if (!jz_node) {
        result.type = JzType::STATIC;
        result.static_value = 0.0;
        return result;
    }

    // Check if it's a scalar (number or formula string)
    if (jz_node.IsScalar()) {
        std::string jz_str = jz_node.as<std::string>();

        // Check if it contains formula characters
        if (jz_str.find('$') != std::string::npos ||
            jz_str.find('*') != std::string::npos ||
            jz_str.find('/') != std::string::npos ||
            jz_str.find('+') != std::string::npos ||
            jz_str.find('(') != std::string::npos) {
            // It's a formula
            result.type = JzType::FORMULA;
            result.formula = jz_str;
            // Replace $step with step for tinyexpr
            size_t pos = 0;
            while ((pos = result.formula.find("$step", pos)) != std::string::npos) {
                result.formula.replace(pos, 5, "step");
                pos += 4;
            }
        } else {
            // It's a plain number
            result.type = JzType::STATIC;
            result.static_value = jz_node.as<double>();
        }
    }
    // Check if it's a sequence (array)
    else if (jz_node.IsSequence()) {
        result.type = JzType::ARRAY;
        for (const auto& val : jz_node) {
            result.array.push_back(val.as<double>());
        }
    }
    else {
        throw std::runtime_error("Invalid Jz value format");
    }

    return result;
}

double MagneticFieldAnalyzer::evaluateJz(const JzValue& jz_val, int step) {
    switch (jz_val.type) {
        case JzType::STATIC:
            return jz_val.static_value;

        case JzType::FORMULA: {
            te_parser parser;
            te_variable step_var;
            step_var.m_name = "step";
            step_var.m_value = static_cast<double>(step);
            // m_type defaults to TE_DEFAULT which is correct for variables

            std::set<te_variable> vars = {step_var};
            parser.set_variables_and_functions(vars);

            double result = parser.evaluate(jz_val.formula);
            if (!parser.success()) {
                throw std::runtime_error("Failed to evaluate Jz formula: " + jz_val.formula);
            }
            return result;
        }

        case JzType::ARRAY:
            if (step < 0 || step >= static_cast<int>(jz_val.array.size())) {
                throw std::runtime_error("Step index out of range for Jz array");
            }
            return jz_val.array[step % transient_config.total_steps ];

        default:
            return 0.0;
    }
}

void MagneticFieldAnalyzer::setupMaterialPropertiesForStep(int step) {
    // Update mu_map and jz_map for ALL materials at the given step
    // This function is called at the beginning of each step, after slideImageRegion()
    if (!config["materials"]) {
        return;
    }

    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;

        // Get RGB values
        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Get relative permeability
        double mu_r = props["mu_r"].as<double>(1.0);

        // Evaluate Jz for this step (0.0 if not defined)
        double jz = 0.0;
        if (material_jz.find(name) != material_jz.end()) {
            jz = evaluateJz(material_jz[name], step);
        }

        // Update mu_map and jz_map for ALL matching pixels
        if (coordinate_system == "polar") {
            // For polar coordinates, flip image to match analysis coordinates
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);  // Flip vertically: y down -> y up

            if (r_orientation == "horizontal") {
                // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx) = (j, i)
                for (int j = 0; j < ntheta; j++) {
                    for (int i = 0; i < nr; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            jz_map(j, i) = jz;
                            mu_map(j, i) = mu_r * MU_0;
                        }
                    }
                }
            } else {  // vertical
                // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx) = (j, i)
                for (int j = 0; j < nr; j++) {
                    for (int i = 0; i < ntheta; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            jz_map(j, i) = jz;
                            mu_map(j, i) = mu_r * MU_0;
                        }
                    }
                }
            }
        } else {
            // Cartesian coordinates
            // Create flipped image to match analysis coordinates (y up)
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);  // Flip vertically: y down -> y up

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        jz_map(j, i) = jz;
                        mu_map(j, i) = mu_r * MU_0;
                    }
                }
            }
        }
    }
}

void MagneticFieldAnalyzer::slideImageRegion() {
    int shift = transient_config.slide_pixels_per_step;

    if (transient_config.slide_direction == "vertical") {
        // vertical: slide_region_start <= x <= slide_region_end (column range)
        // slides in y direction (rows)
        int x_start = transient_config.slide_region_start;
        int x_end = transient_config.slide_region_end;

        // Validate region
        if (x_start < 0 || x_end > image.cols || x_start >= x_end) {
            throw std::runtime_error("Invalid slide region for vertical sliding (x range out of bounds)");
        }

        // For each column in the x range, circularly shift all rows
        for (int col = x_start; col < x_end; col++) {
            // Extract the column
            cv::Mat column = image.col(col).clone();

            // Circular shift in y direction (downward)
            // Move bottom 'shift' rows to top
            for (int row = 0; row < image.rows; row++) {
                int src_row = (row + image.rows + shift) % image.rows;
                image.at<cv::Vec3b>(row, col) = column.at<cv::Vec3b>(src_row, 0);
            }
        }

    } else {  // horizontal
        // horizontal: slide_region_start <= y <= slide_region_end (row range)
        // slides in x direction (columns)
        int y_start = transient_config.slide_region_start;
        int y_end = transient_config.slide_region_end;

        // Validate region
        if (y_start < 0 || y_end > image.rows || y_start >= y_end) {
            throw std::runtime_error("Invalid slide region for horizontal sliding (y range out of bounds)");
        }

        // For each row in the y range, circularly shift all columns
        for (int row = y_start; row < y_end; row++) {
            // Extract the row
            cv::Mat row_data = image.row(row).clone();

            // Circular shift in x direction (rightward)
            // Move right 'shift' columns to left
            for (int col = 0; col < image.cols; col++) {
                int src_col = (col + image.cols + shift) % image.cols;
                image.at<cv::Vec3b>(row, col) = row_data.at<cv::Vec3b>(0, src_col);
            }
        }
    }
}

void MagneticFieldAnalyzer::performTransientAnalysis(const std::string& output_dir) {
    if (!transient_config.enabled) {
        std::cout << "Transient analysis is not enabled" << std::endl;
        return;
    }

    std::cout << "\n=== Starting Transient Analysis ===" << std::endl;
    std::cout << "Total steps: " << transient_config.total_steps << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl;

    // Optimization: Reuse matrix pattern (analyzePattern only once) for both coordinate systems
    bool use_optimized_solver = true;
    std::cout << "Using optimized transient solver (pattern reuse for " << coordinate_system << " coordinates)" << std::endl;

    // Start overall timer
    auto analysis_start_time = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < transient_config.total_steps; step++) {
        std::cout << "\n--- Step " << step+1 << " / " << transient_config.total_steps << " ---" << std::endl;

        // Start step timer
        auto step_start_time = std::chrono::high_resolution_clock::now();

        // 1. Update material properties (Jz) for this step
        setupMaterialPropertiesForStep(step);

        // 2. Solve FDM system (optimized for transient analysis)
        auto solve_start = std::chrono::high_resolution_clock::now();

        if (use_optimized_solver) {
            // Optimized path for both coordinate systems
            Eigen::SparseMatrix<double> A;
            Eigen::VectorXd rhs;
            int n;

            // Build matrix based on coordinate system
            if (coordinate_system == "cartesian") {
                n = nx * ny;
                buildMatrix(A, rhs);
            } else {  // polar
                n = nr * ntheta;
                buildMatrixPolar(A, rhs);
            }

            Eigen::VectorXd Az_flat;

            // Solver selection strategy based on problem size and step
            // Performance analysis (n=250k):
            //   - Direct solver: ~5s
            //   - CG + Diagonal: ~18s (converges)
            //   - ICCG: ~100s (fails to converge)
            // For n <= 500k, direct solver is faster. Use iterative for n > 500k (memory constraint)

            bool use_iterative_solver = (step > 0 ) || (n > 100000);  // Temporary: test warm-start correction

            // if (step == 0) {
            //     // First step: use direct solver (SparseLU)
            //     std::cout << "Step 0: Using direct solver (SparseLU)..." << std::endl;
            //     std::cout << "Matrix size: " << A.rows() << "x" << A.cols() << std::endl;
            //     std::cout << "Non-zero elements: " << A.nonZeros() << std::endl;

            //     // Check matrix symmetry (critical for numerical accuracy)
            //     Eigen::SparseMatrix<double> A_T = A.transpose();
            //     Eigen::SparseMatrix<double> D = A - A_T;
            //     double symmetry_error = D.norm();
            //     double A_norm = A.norm();
            //     double relative_symmetry_error = (A_norm > 1e-12) ? (symmetry_error / A_norm) : symmetry_error;

            //     std::cout << "Matrix symmetry check:" << std::endl;
            //     std::cout << "  ||A - A^T|| = " << symmetry_error << std::endl;
            //     std::cout << "  ||A|| = " << A_norm << std::endl;
            //     std::cout << "  Relative error = " << relative_symmetry_error << std::endl;

            //     // Detailed asymmetry diagnosis - find max asymmetric element
            //     double max_asym = 0.0;
            //     int max_i = -1, max_j = -1;
            //     for (int k = 0; k < A.outerSize(); ++k) {
            //         for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it) {
            //             double a_ij = it.value();
            //             double a_ji = A.coeff(it.col(), it.row());
            //             double diff = std::abs(a_ij - a_ji);
            //             if (diff > max_asym) {
            //                 max_asym = diff;
            //                 max_i = it.row();
            //                 max_j = it.col();
            //             }
            //         }
            //     }
            //     std::cout << "  Max asymmetric element: A(" << max_i << "," << max_j
            //               << ") diff = " << max_asym << std::endl;

            //     if (relative_symmetry_error > 1e-8) {
            //         std::cerr << "WARNING: Matrix is not symmetric! Relative error = "
            //                   << relative_symmetry_error << std::endl;
            //         std::cerr << "This may indicate incorrect discretization or mu interface calculation." << std::endl;
            //         std::cerr << "Iterative solvers may struggle to converge." << std::endl;
            //     } else {
            //         std::cout << "  Matrix is symmetric (relative error < 1e-8)" << std::endl;
            //     }

            //     // Use compute() instead of separate analyzePattern + factorize
            //     // This is more robust after matrix operations like transpose
            //     transient_solver.compute(A);
            //     if (transient_solver.info() != Eigen::Success) {
            //         throw std::runtime_error("Matrix decomposition failed");
            //     }

            //     Az_flat = transient_solver.solve(rhs);
            //     if (transient_solver.info() != Eigen::Success) {
            //         throw std::runtime_error("Linear system solving failed");
            //     }

            //     // Save solution, RHS, and matrix for warm start and diagnostics
            //     previous_solution = Az_flat;
            //     previous_rhs = rhs;  // Save for Δb correction in next step
            //     previous_matrix = A;  // Save for ΔA quantification
            //     transient_solver_initialized = true;
            //     transient_matrix_nnz = A.nonZeros();
            // } else 
            if (use_iterative_solver) {
                // Subsequent steps with large problem size: use AMGCL (AMG-preconditioned CG)
                // AMGCL provides dramatic speedup for Poisson-type problems
                std::cout << "Step " << step+1 << ": Using AMGCL (AMG-preconditioned CG)..." << std::endl;
                std::cout << "  (Problem size n=" << n << " > 100k threshold)" << std::endl;

                std::cout << "[AMGCL] Preparing warm-start initial guess..." << std::endl;

                // Start timing
                auto amgcl_total_start = std::chrono::high_resolution_clock::now();

                // Validate previous_solution size
                if (previous_solution.size() != n) {
                    std::cerr << "WARNING: previous_solution size mismatch ("
                              << previous_solution.size() << " vs " << n
                              << "), using zero initial guess" << std::endl;
                    previous_solution = Eigen::VectorXd::Zero(n);
                }

                // Prepare warm-start initial guess with preprocessing
                {
                    // Validate previous_solution size before warm start
                    if (previous_solution.size() != n) {
                        std::cerr << "WARNING: previous_solution size mismatch ("
                                  << previous_solution.size() << " vs " << n
                                  << "), using zero initial guess" << std::endl;
                        previous_solution = Eigen::VectorXd::Zero(n);
                    }

                    // ============================================
                    // DIAGNOSTIC: Permutation verification & matrix change quantification
                    // (GPT-recommended priority #1)
                    // ============================================
                    if (previous_matrix.nonZeros() > 0 && transient_config.enable_sliding && coordinate_system == "polar") {
                        // std::cout << "\n[DIAGNOSTIC] Permutation verification:" << std::endl;

                        int shift_pixels = transient_config.slide_pixels_per_step;

                        // Create permutation matrix from circular shift
                        // Image slides down (θ+ direction): old position jt -> new position (jt + shift_pixels) % ntheta
                        // PermutationMatrix: P.indices()[i] = j means "new position i gets value from old position j"
                        Eigen::PermutationMatrix<Eigen::Dynamic> P(n);
                        for (int ir = 0; ir < nr; ir++) {
                            for (int jt = 0; jt < ntheta; jt++) {
                                int old_idx = ir * ntheta + jt;
                                int new_jt = (jt + shift_pixels) % ntheta;
                                int new_idx = ir * ntheta + new_jt;
                                P.indices()[new_idx] = old_idx;  // New position gets value from old position
                            }
                        }

                        // Test if A_new = P * A_old * P^T
                        Eigen::SparseMatrix<double> A_perm = P * previous_matrix * P.transpose();
                        double A_perm_norm = A_perm.norm();
                        double perm_residual = (A - A_perm).norm() / (A_perm_norm + 1e-30);

                        // Measure matrix changes
                        double A_old_norm = previous_matrix.norm();
                        double matrix_change = (A - previous_matrix).norm() / (A_old_norm + 1e-30);

                        std::cout << "  ||A_new - P*A_old*P^T|| / ||A_perm|| = " << perm_residual << std::endl;
                        std::cout << "  ||A_new - A_old|| / ||A_old|| = " << matrix_change << std::endl;

                        if (perm_residual < 1e-10) {
                            std::cout << "  ✓ Matrix matches permutation (circular shift correct)" << std::endl;
                        } else {
                            std::cout << "  ✗ Matrix does NOT match permutation (shift error or material changes)" << std::endl;
                        }

                        // Additional diagnostic: Quantify ΔA contribution to residual
                        Eigen::SparseMatrix<double> delta_A = A - previous_matrix;
                        Eigen::VectorXd delta_A_x_prev = delta_A * previous_solution;
                        double delta_A_contribution = delta_A_x_prev.norm();
                        double rhs_norm = rhs.norm();
                        double delta_A_ratio = (rhs_norm > 1e-12) ? (delta_A_contribution / rhs_norm) : delta_A_contribution;

                        std::cout << "  ||(A_new - A_old) * x_prev|| / ||b'|| = " << delta_A_ratio << std::endl;
                        std::cout << std::endl;
                    }

                    // ============================================
                    // Warm-start with Δb correction (GPT-recommended approach)
                    // ============================================

                    // Start timing for total iterative solver (includes all preprocessing)
                    auto iterative_total_start = std::chrono::high_resolution_clock::now();

                    // Step 1: Apply circular shift to previous solution (partial-region aware)
                    Eigen::VectorXd x_shifted(n);

                    bool enable_circular_shift = true;  // Enable circular shift for partial-region sliding

                    if (enable_circular_shift && transient_config.enable_sliding && coordinate_system == "polar") {
                        int shift_pixels = transient_config.slide_pixels_per_step;
                        int slide_start = transient_config.slide_region_start;
                        int slide_end = transient_config.slide_region_end;

                        std::cout << "[DEBUG] Applying partial-region circular shift:" << std::endl;
                        std::cout << "  Shift: " << shift_pixels << " pixels (theta direction)" << std::endl;
                        std::cout << "  Region: r index " << slide_start << " to " << slide_end << std::endl;

                        // r_orientation=horizontal: image x-axis (horizontal) -> r direction
                        // Slide region defined by image x-coordinate -> r index (ir)
                        // Image slides down (y+) -> theta+ direction -> positive shift
                        for (int ir = 0; ir < nr; ir++) {
                            if (ir >= slide_start && ir < slide_end) {
                                // Sliding region: apply circular shift in theta direction
                                for (int jt = 0; jt < ntheta; jt++) {
                                    int old_idx = ir * ntheta + jt;
                                    int new_jt = (jt + shift_pixels) % ntheta;
                                    int new_idx = ir * ntheta + new_jt;
                                    x_shifted(new_idx) = previous_solution(old_idx);
                                }
                            } else {
                                // Non-sliding region: keep original indices
                                for (int jt = 0; jt < ntheta; jt++) {
                                    int idx = ir * ntheta + jt;
                                    x_shifted(idx) = previous_solution(idx);
                                }
                            }
                        }
                    } else {
                        std::cout << "[DEBUG] Circular shift disabled" << std::endl;
                        x_shifted = previous_solution;
                    }

                    // Step 1.5: Apply Gaussian smoothing to slide region + boundary buffer
                    // Reverted to full-region smoothing (Phase 1 localization was less effective)
                    if (enable_circular_shift && transient_config.enable_sliding && coordinate_system == "polar") {
                        int slide_start = transient_config.slide_region_start;
                        int slide_end = transient_config.slide_region_end;
                        int seam_ir = slide_end;  // Seam position (for patch later)
                        double sigma = 8.0;  // Gaussian σ (pixels) - optimal value
                        int buffer = 10;  // Buffer beyond slide_end

                        std::cout << "[DEBUG] Applying Gaussian smoothing (r-direction only):" << std::endl;
                        std::cout << "  Region: r=[" << slide_start << ", " << (slide_end + buffer) << "]" << std::endl;
                        std::cout << "  σ=" << sigma << " pixels" << std::endl;

                        // Create Gaussian kernel (1D, r-direction)
                        int kernel_radius = static_cast<int>(3 * sigma);  // 3σ coverage
                        std::vector<double> gaussian_kernel(2 * kernel_radius + 1);
                        double sum = 0.0;
                        for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                            double weight = std::exp(-(k * k) / (2.0 * sigma * sigma));
                            gaussian_kernel[k + kernel_radius] = weight;
                            sum += weight;
                        }
                        // Normalize
                        for (auto& w : gaussian_kernel) {
                            w /= sum;
                        }

                        // Apply Gaussian filter in r-direction only
                        // (theta-direction preserves material boundaries)
                        Eigen::VectorXd x_smoothed = x_shifted;  // Copy
                        int smooth_start = slide_start;
                        int smooth_end = std::min(slide_end + buffer, nr - 1);

                        for (int ir = smooth_start; ir <= smooth_end; ++ir) {
                            for (int jt = 0; jt < ntheta; ++jt) {
                                int idx = ir * ntheta + jt;
                                double smoothed_value = 0.0;

                                for (int k = -kernel_radius; k <= kernel_radius; ++k) {
                                    int ir_neighbor = ir + k;
                                    // Clamp to valid range
                                    if (ir_neighbor < 0) ir_neighbor = 0;
                                    if (ir_neighbor >= nr) ir_neighbor = nr - 1;

                                    int idx_neighbor = ir_neighbor * ntheta + jt;
                                    smoothed_value += gaussian_kernel[k + kernel_radius] * x_shifted(idx_neighbor);
                                }

                                x_smoothed(idx) = smoothed_value;
                            }
                        }

                        x_shifted = x_smoothed;
                    }

                    // Step 2: Calculate Δb and diagnose warm-start viability
                    Eigen::VectorXd delta_b = rhs - previous_rhs;
                    double delta_b_norm = delta_b.norm();
                    double rhs_norm = rhs.norm();
                    double delta_b_ratio = (rhs_norm > 1e-12) ? (delta_b_norm / rhs_norm) : delta_b_norm;

                    // std::cout << "[DEBUG] RHS change analysis:" << std::endl;
                    // std::cout << "  ||Δb|| / ||b'|| = " << delta_b_ratio << std::endl;

                    // Step 3: Decide on correction strategy based on Δb magnitude
                    Eigen::VectorXd x0(n);
                    bool use_correction = false;

                    if (delta_b_ratio < 0.1) {
                        // Small RHS change: simple warm-start works well
                        std::cout << "  Strategy: Simple warm-start (Δb small)" << std::endl;
                        x0 = x_shifted;
                    } else if (delta_b_ratio < 2.0) {
                        // Moderate RHS change: apply diagonal preconditioner correction
                        std::cout << "  Strategy: Diagonal preconditioner correction" << std::endl;
                        use_correction = true;

                        // Cheap diagonal correction: y_i = Δb_i / diag(A)_i
                        Eigen::VectorXd diag = A.diagonal();
                        Eigen::VectorXd y_corr(n);
                        for (int i = 0; i < n; i++) {
                            double d = diag[i];
                            y_corr[i] = (std::abs(d) > 1e-30) ? (delta_b[i] / d) : 0.0;
                        }
                        x0 = x_shifted + y_corr;

                        std::cout << "  Diagonal correction applied: ||y_corr|| = " << y_corr.norm() << std::endl;
                    } else {
                        // Large RHS change: try LU-based correction if available
                        std::cout << "  Strategy: LU-based correction (Δb large)" << std::endl;
                        use_correction = true;

                        if (transient_solver_initialized && A.nonZeros() == transient_matrix_nnz) {
                            // Reuse LU factorization: solve A * y_corr = Δb
                            Eigen::VectorXd y_corr = transient_solver.solve(delta_b);
                            x0 = x_shifted + y_corr;
                            std::cout << "  LU correction applied: ||y_corr|| = " << y_corr.norm() << std::endl;
                        } else {
                            // Fallback to diagonal correction
                            Eigen::VectorXd diag = A.diagonal();
                            Eigen::VectorXd y_corr(n);
                            for (int i = 0; i < n; i++) {
                                double d = diag[i];
                                y_corr[i] = (std::abs(d) > 1e-30) ? (delta_b[i] / d) : 0.0;
                            }
                            x0 = x_shifted + y_corr;
                            std::cout << "  Diagonal correction (fallback): ||y_corr|| = " << y_corr.norm() << std::endl;
                        }
                    }

                    // Step 4: Evaluate corrected warm-start quality
                    Eigen::VectorXd r0 = rhs - A * x0;
                    double r0_norm = r0.norm();
                    double rel_r0 = (rhs_norm > 1e-12) ? (r0_norm / rhs_norm) : r0_norm;

                    std::cout << "[DEBUG] Warm-start quality:" << std::endl;
                    std::cout << "  ||r0|| / ||b'|| = " << rel_r0 << std::endl;

                    // Additional diagnostic: Check residual distribution
                    if (transient_config.enable_sliding && coordinate_system == "polar") {
                        int slide_start = transient_config.slide_region_start;
                        int slide_end = transient_config.slide_region_end;

                        // Calculate residual norms in different regions
                        double r0_slide_norm = 0.0;
                        double r0_nonslide_norm = 0.0;
                        double r0_boundary_norm = 0.0;
                        int count_slide = 0, count_nonslide = 0, count_boundary = 0;

                        for (int ir = 0; ir < nr; ir++) {
                            for (int jt = 0; jt < ntheta; jt++) {
                                int idx = ir * ntheta + jt;
                                double r_val = std::abs(r0[idx]);

                                if (ir >= slide_start && ir < slide_end) {
                                    r0_slide_norm += r_val * r_val;
                                    count_slide++;
                                } else if (std::abs(ir - slide_start) <= 2 || std::abs(ir - slide_end) <= 2) {
                                    r0_boundary_norm += r_val * r_val;
                                    count_boundary++;
                                } else {
                                    r0_nonslide_norm += r_val * r_val;
                                    count_nonslide++;
                                }
                            }
                        }

                        r0_slide_norm = std::sqrt(r0_slide_norm / std::max(count_slide, 1));
                        r0_nonslide_norm = std::sqrt(r0_nonslide_norm / std::max(count_nonslide, 1));
                        r0_boundary_norm = std::sqrt(r0_boundary_norm / std::max(count_boundary, 1));

                        std::cout << "  Residual RMS by region:" << std::endl;
                        std::cout << "    Slide region (r=" << slide_start << "-" << slide_end << "): " << r0_slide_norm << std::endl;
                        std::cout << "    Boundary (±2): " << r0_boundary_norm << std::endl;
                        std::cout << "    Non-slide region: " << r0_nonslide_norm << std::endl;
                    }

                    // ============================================
                    // AMGCL Test: AMG-preconditioned CG (zero initial guess)
                    // ============================================
                    // std::cout << "\n[AMGCL TEST] AMG-preconditioned CG (zero initial guess):" << std::endl;

                    // Convert matrix to RowMajor for AMGCL
                    Eigen::SparseMatrix<double, Eigen::RowMajor> A_rowmajor = A;

                    // Define AMG preconditioner + CG solver with Eigen backend
                    typedef amgcl::make_solver<
                        amgcl::amg<
                            amgcl::backend::eigen<double>,
                            amgcl::coarsening::smoothed_aggregation,
                            amgcl::relaxation::spai0
                        >,
                        amgcl::solver::cg<amgcl::backend::eigen<double>>
                    > AMGSolver;

                    // AMGCL solver parameters
                    AMGSolver::params amg_params;
                    amg_params.solver.tol = SOLVER_TOLERANCE;
                    amg_params.solver.maxiter =  SOLVER_MAX_ITERATIONS;

                    std::cout << "  Building AMG hierarchy..." << std::endl;
                    auto amg_build_start = std::chrono::high_resolution_clock::now();
                    AMGSolver amg_solve(A_rowmajor, amg_params);
                    auto amg_build_end = std::chrono::high_resolution_clock::now();
                    auto amg_build_time = std::chrono::duration_cast<std::chrono::milliseconds>(amg_build_end - amg_build_start);

                    std::cout << "  AMG hierarchy built in " << amg_build_time.count() << " ms" << std::endl;

                    // Solve with AMGCL
                    Eigen::VectorXd amg_solution = Eigen::VectorXd::Zero(n);
                    std::cout << "  Solving with AMG-CG..." << std::endl;
                    auto amg_solve_start = std::chrono::high_resolution_clock::now();
                    int amg_iters;
                    double amg_error;
                    std::tie(amg_iters, amg_error) = amg_solve(rhs, amg_solution);
                    auto amg_solve_end = std::chrono::high_resolution_clock::now();
                    auto amg_solve_time = std::chrono::duration_cast<std::chrono::milliseconds>(amg_solve_end - amg_solve_start);

                    std::cout << "\n  AMGCL Results:" << std::endl;
                    std::cout << "    AMG build time: " << amg_build_time.count() << " ms" << std::endl;
                    std::cout << "    Solver time: " << amg_solve_time.count() << " ms" << std::endl;
                    std::cout << "    Total time: " << (amg_build_time.count() + amg_solve_time.count()) << " ms" << std::endl;
                    std::cout << "    Iterations: " << amg_iters << std::endl;
                    std::cout << "    Residual error: " << amg_error << std::endl;
                    std::cout << "    Status: " << (amg_error < SOLVER_TOLERANCE ? "SUCCESS" : "FAILED") << std::endl;

                    // Use AMGCL solution
                    Az_flat = amg_solution;

                    /* Disabled for AMGCL test - uncomment to use warm-start CG instead
                    // Warm-start iterative solver with extended iterations
                    std::cout << "\n[ITERATIVE SOLVER] Warm-start with preprocessing:" << std::endl;
                    std::cout << "  Initial residual: ||r0|| / ||b'|| = " << rel_r0 << std::endl;
                    std::cout << "  Target: error < SOLVER_TOLERANCE" << std::endl;
                    std::cout << "  Max iterations:  SOLVER_MAX_ITERATIONS" << std::endl;

                    // Set extended iteration limit
                    cg.setMaxIterations( SOLVER_MAX_ITERATIONS);
                    cg.setTolerance(SOLVER_TOLERANCE);

                    // Run warm-start CG to convergence
                    auto cg_start = std::chrono::high_resolution_clock::now();
                    Az_flat = cg.solveWithGuess(rhs, x0);
                    auto cg_end = std::chrono::high_resolution_clock::now();
                    auto cg_time = std::chrono::duration_cast<std::chrono::milliseconds>(cg_end - cg_start);

                    int trial_iters = cg.iterations();
                    double trial_error = cg.error();
                    Eigen::ComputationInfo cg_info = cg.info();
                    auto trial_time = cg_time;

                    std::cout << "\n  Warm-start CG Results:" << std::endl;
                    std::cout << "    Iterations: " << trial_iters << std::endl;
                    std::cout << "    Final error: " << trial_error << std::endl;
                    std::cout << "    Time: " << cg_time.count() << " ms" << std::endl;
                    std::cout << "    Status: ";
                    if (cg_info == Eigen::Success) {
                        std::cout << "SUCCESS - Converged to target!" << std::endl;
                    } else if (cg_info == Eigen::NoConvergence) {
                        std::cout << "NoConvergence (reached max iterations)" << std::endl;
                    } else if (cg_info == Eigen::NumericalIssue) {
                        std::cout << "NumericalIssue" << std::endl;
                    } else {
                        std::cout << "Unknown (code " << cg_info << ")" << std::endl;
                    }

                    // Calculate convergence rate
                    if (trial_iters > 0 && trial_error < rel_r0) {
                        double reduction_factor = trial_error / rel_r0;
                        double avg_reduction_per_iter = std::pow(reduction_factor, 1.0 / trial_iters);
                        std::cout << "    Error reduction: " << rel_r0 << " → " << trial_error
                                  << " (" << (rel_r0 / trial_error) << "x improvement)" << std::endl;
                        std::cout << "    Avg reduction per iteration: " << avg_reduction_per_iter << std::endl;
                    }

                    // Check if iterative solver succeeded
                    if (cg_info != Eigen::Success) {
                        std::cout << "\n[INFO] CG did not converge, falling back to direct solver" << std::endl;

                        // Fallback to direct solver
                        std::cout << "[DEBUG] Using direct solver fallback..." << std::endl;
                        auto fallback_start = std::chrono::high_resolution_clock::now();
                        if (A.nonZeros() != transient_matrix_nnz) {
                            transient_solver.compute(A);
                            transient_matrix_nnz = A.nonZeros();
                        } else {
                            transient_solver.factorize(A);
                        }
                        Az_flat = transient_solver.solve(rhs);
                        auto fallback_end = std::chrono::high_resolution_clock::now();
                        auto fallback_time = std::chrono::duration_cast<std::chrono::milliseconds>(fallback_end - fallback_start);
                        std::cout << "[DEBUG] Direct solver fallback complete, time=" << fallback_time.count() << " ms" << std::endl;
                    }

                    // Print total iterative solver time (including all preprocessing)
                    auto iterative_total_end = std::chrono::high_resolution_clock::now();
                    auto iterative_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(iterative_total_end - iterative_total_start);
                    std::cout << "\n[PERFORMANCE] Total iterative solver time (preprocessing + solve): "
                              << iterative_total_time.count() << " ms" << std::endl;
                    std::cout << "  (includes: circular shift + Gaussian smoothing + patch solve + CG iterations)" << std::endl;
                    */// End of warm-start CG code block
                }

                // Update previous solution, RHS, and matrix for next step
                previous_solution = Az_flat;
                previous_rhs = rhs;
                previous_matrix = A;

            } else {
                // Small/medium problems (n <= 100k): use direct solver with pattern reuse
                std::cout << "Step " << step+1 << ": Using direct solver (SparseLU) with pattern reuse..." << std::endl;
                std::cout << "  (Problem size n=" << n << " <= 100k, direct solver is faster)" << std::endl;

                // Reuse pattern if matrix structure unchanged
                if (A.nonZeros() != transient_matrix_nnz) {
                    std::cout << "  Matrix pattern changed, re-analyzing..." << std::endl;
                    transient_solver.compute(A);
                    transient_matrix_nnz = A.nonZeros();
                } else {
                    // Pattern unchanged, only factorize
                    transient_solver.factorize(A);
                }

                if (transient_solver.info() != Eigen::Success) {
                    throw std::runtime_error("Matrix decomposition failed");
                }

                Az_flat = transient_solver.solve(rhs);
                if (transient_solver.info() != Eigen::Success) {
                    throw std::runtime_error("Linear system solving failed");
                }

                // Save solution, RHS, and matrix for potential future use
                previous_solution = Az_flat;
                previous_rhs = rhs;
                previous_matrix = A;
            }

            // Reshape to 2D matrix based on coordinate system
            if (coordinate_system == "cartesian") {
                Az.resize(ny, nx);
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        Az(j, i) = Az_flat[j * nx + i];
                    }
                }
            } else {  // polar
                // Reshape solution to matrix form
                // Index mapping: idx = i * ntheta + j, where i is radial, j is angular
                if (r_orientation == "horizontal") {
                    // Image is (ntheta, nr), transpose the solution
                    Az.resize(ntheta, nr);
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            Az(j, i) = Az_flat(i * ntheta + j);  // Transpose: Az(θ, r) from idx(r, θ)
                        }
                    }
                } else {  // vertical
                    // Image is (nr, ntheta), no transpose needed
                    Az.resize(nr, ntheta);
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            Az(i, j) = Az_flat(i * ntheta + j);  // Direct: Az(r, θ) from idx(r, θ)
                        }
                    }
                }
            }
        } else {
            // Standard path (fallback - not used)
            solve();
        }

        // 3. Calculate Maxwell stress (pass step number for field caching)
        // CRITICAL: Use step (not step+1) to match theta_offset with actual image shift state
        // step=0: initial state (no shift) → theta_offset=0
        // step=1: 1-pixel shifted → theta_offset=1*dtheta
        calculateMaxwellStress(step);

        // 3.5. Calculate total magnetic energy (reuses calculated field from step)
        double total_energy = calculateTotalMagneticEnergy(step);
        std::cout << "Step " << step+1 << " Total Magnetic Energy: " << total_energy << " J/m" << std::endl;

        // 4. Export results for this step
        exportResults(output_dir, step);

        // Calculate and display step elapsed time
        auto step_end_time = std::chrono::high_resolution_clock::now();
        auto step_duration = std::chrono::duration_cast<std::chrono::milliseconds>(step_end_time - step_start_time);
        auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(step_end_time - analysis_start_time);

        std::cout << "Step " << step+1 << " elapsed time: " << step_duration.count() << " ms" << std::endl;
        std::cout << "Total elapsed time: " << total_elapsed.count() << " s" << std::endl;

        // 5. Slide image for next step (except for last step)
        if (step < transient_config.total_steps - 1 && transient_config.enable_sliding) {
            slideImageRegion();
            // Material properties (mu_map, jz_map) will be updated by setupMaterialPropertiesForStep() at next step
        }
    }

    // Calculate total analysis time
    auto analysis_end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(analysis_end_time - analysis_start_time);
    auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(analysis_end_time - analysis_start_time);

    std::cout << "\n=== Transient Analysis Complete ===" << std::endl;
    std::cout << "Total analysis time: " << total_duration.count() << " s ("
              << total_duration_ms.count() << " ms)" << std::endl;
}
