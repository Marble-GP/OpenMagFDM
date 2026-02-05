#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <chrono>
#include <set>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0755)
#endif

// Cross-platform directory creation (like mkdir -p)
static void createDirectory(const std::string& path) {
#ifdef _WIN32
    system(("mkdir \"" + path + "\" 2>nul").c_str());
#else
    system(("mkdir -p \"" + path + "\"").c_str());
#endif
}

constexpr double MU_0 = 4.0 * M_PI * 1e-7;  // Vacuum permeability [H/m]

MagneticFieldAnalyzer::MagneticFieldAnalyzer(const std::string& config_path,
                                             const std::string& image_path) {
    loadConfig(config_path);
    loadImage(image_path);

    // Initialize flags BEFORE setup methods
    // Transient analysis optimization flags
    transient_solver_initialized = false;
    transient_matrix_nnz = 0;
    boundary_cache_valid = false;
    use_iterative_solver = false;

    // Magnetic field step tracking
    current_field_step = -999;  // Not calculated yet (use sentinel value different from any valid step)
    system_total_energy = 0.0;  // Initialize system total energy

    // Nonlinear solver flags (MUST be initialized before setupMaterialProperties)
    has_nonlinear_materials = false;
    nonlinear_config = NonlinearSolverConfig();  // Default config

    // Load nonlinear solver configuration from YAML (if present)
    if (config["nonlinear_solver"]) {
        auto nl_config = config["nonlinear_solver"];

        // Basic settings
        if (nl_config["enabled"])
            nonlinear_config.enabled = nl_config["enabled"].as<bool>(true);
        if (nl_config["solver_type"])
            nonlinear_config.solver_type = nl_config["solver_type"].as<std::string>("newton-krylov");
        nonlinear_config.max_iterations = nl_config["max_iterations"].as<int>(50);
        nonlinear_config.tolerance = nl_config["tolerance"].as<double>(5e-4);
        nonlinear_config.verbose = nl_config["verbose"].as<bool>(false);
        nonlinear_config.export_convergence = nl_config["export_convergence"].as<bool>(false);

        // Picard specific settings
        nonlinear_config.relaxation = nl_config["relaxation"].as<double>(0.7);

        // Anderson acceleration settings (shared by Picard and Newton-Krylov)
        if (nl_config["anderson"]) {
            auto anderson_cfg = nl_config["anderson"];
            nonlinear_config.anderson.enabled = anderson_cfg["enabled"].as<bool>(false);
            nonlinear_config.anderson.depth = anderson_cfg["depth"].as<int>(5);
            nonlinear_config.anderson.beta = anderson_cfg["beta"].as<double>(1.0);
        }

        // Newton-Krylov specific settings
        nonlinear_config.gmres_restart = nl_config["gmres_restart"].as<int>(30);
        nonlinear_config.line_search_c = nl_config["line_search_c"].as<double>(1e-4);
        nonlinear_config.line_search_alpha_init = nl_config["line_search_alpha_init"].as<double>(1.0);
        nonlinear_config.line_search_alpha_min = nl_config["line_search_alpha_min"].as<double>(1e-3);
        nonlinear_config.line_search_rho = nl_config["line_search_rho"].as<double>(0.65);
        nonlinear_config.line_search_max_trials = nl_config["line_search_max_trials"].as<int>(50);
        nonlinear_config.line_search_adaptive = nl_config["line_search_adaptive"].as<bool>(true);

        // Phase 4: Galerkin coarsening option (for coarsened Newton-Krylov)
        nonlinear_config.use_galerkin_coarsening = nl_config["use_galerkin_coarsening"].as<bool>(true);

        // Phase 5: Matrix-free Jv option (solves oscillation issue with coarsening + nonlinear)
        nonlinear_config.use_matrix_free_jv = nl_config["use_matrix_free_jv"].as<bool>(true);

        // Phase 6: Preconditioned JFNK (uses Galerkin coarse matrix as preconditioner)
        nonlinear_config.use_phase6_precond_jfnk = nl_config["use_phase6_precond_jfnk"].as<bool>(false);
        nonlinear_config.precond_update_frequency = nl_config["precond_update_frequency"].as<int>(1);
        nonlinear_config.precond_verbose = nl_config["precond_verbose"].as<bool>(false);
    }

    // Determine coordinate system
    coordinate_system = config["coordinate_system"].as<std::string>("cartesian");
    std::cout << "Coordinate system: " << coordinate_system << std::endl;

    if (coordinate_system == "polar") {
        setupPolarSystem();
    } else {
        setupCartesianSystem();
    }

    parseUserVariables();  // Parse user-defined variables before material properties
    setupMaterialProperties();  // This may set has_nonlinear_materials = true
    validateBoundaryConditions();

    // Generate adaptive mesh coarsening mask (if any material has coarsening enabled)
    generateCoarseningMask();
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

        // Parse material presets (reusable B-H curves/properties)
        material_presets.clear();
        if (config["material_presets"]) {
            std::cout << "Parsing material presets..." << std::endl;
            for (const auto& preset : config["material_presets"]) {
                std::string preset_name = preset.first.as<std::string>();
                material_presets[preset_name] = YAML::Clone(preset.second);
                std::cout << "  Loaded preset: " << preset_name << std::endl;
            }
            std::cout << "Loaded " << material_presets.size() << " material preset(s)" << std::endl;
        }

        // Parse flux linkage paths
        parseFluxLinkagePaths();

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

    // Set image dimensions (used for coarsening mask and other operations)
    nx = image.cols;
    ny = image.rows;

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
    // Angular spacing: For full rotation (theta_range == 2*pi), θ is periodic
    // θ=0 and θ=2*pi represent the same radial line, so divide by ntheta (NOT ntheta-1)
    // For sector domains (theta_range < 2*pi), θ=0 and θ=theta_range are boundaries (Dirichlet)
    // Still use ntheta division for consistency (boundary points at j=0 and j=ntheta-1)
    dtheta = theta_range / static_cast<double>(ntheta);  // θ sampling: [0, theta_range)

    std::cout << "Polar domain: r = [" << r_start << ", " << r_end << "] m, theta = [0, "
              << theta_range << "] rad (" << (theta_range * 180.0 / M_PI) << " deg)" << std::endl;
    std::cout << "Mesh spacing: dr=" << dr << ", dtheta=" << dtheta << " rad" << std::endl;

    // Check if full rotation or sector domain
    const double TWO_PI = 2.0 * M_PI;
    if (std::abs(theta_range - TWO_PI) < 1e-6) {
        std::cout << "Angular domain: Full rotation (periodic boundary in θ direction)" << std::endl;
    } else {
        std::cout << "Angular domain: Sector (" << (theta_range * 180.0 / M_PI)
                  << " degrees, Dirichlet BC at θ=0 and θ=" << theta_range << ")" << std::endl;
    }

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

        // Helper lambda to parse Robin parameters for polar BC
        auto parsePolarRobinParams = [](const YAML::Node& node, BoundaryCondition& bc_out) {
            bc_out.type = node["type"].as<std::string>("dirichlet");
            bc_out.value = node["value"].as<double>(0.0);

            // Parse Robin BC parameters if type is "robin"
            if (bc_out.type == "robin") {
                bc_out.alpha = node["alpha"].as<double>(1.0);
                bc_out.beta = node["beta"].as<double>(0.0);
                bc_out.gamma = node["gamma"].as<double>(0.0);

                // Validate: alpha and beta cannot both be zero
                if (std::abs(bc_out.alpha) < 1e-15 && std::abs(bc_out.beta) < 1e-15) {
                    throw std::runtime_error("Robin BC: alpha and beta cannot both be zero");
                }
            }
        };

        if (bc_cfg["inner"]) {
            parsePolarRobinParams(bc_cfg["inner"], bc_inner);
        }

        if (bc_cfg["outer"]) {
            parsePolarRobinParams(bc_cfg["outer"], bc_outer);
        }

        // Theta direction boundary conditions (angular boundaries)
        if (bc_cfg["theta_min"]) {
            parsePolarRobinParams(bc_cfg["theta_min"], bc_theta_min);
        }

        if (bc_cfg["theta_max"]) {
            parsePolarRobinParams(bc_cfg["theta_max"], bc_theta_max);
        }
    }

    // Determine if theta is periodic (both theta_min and theta_max must be "periodic")
    // Anti-periodic: type="periodic" with value < 0
    bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool theta_antiperiodic = theta_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);

    std::string theta_bc_str;
    if (theta_antiperiodic) {
        theta_bc_str = "anti-periodic";
    } else if (theta_periodic) {
        theta_bc_str = "periodic";
    } else {
        theta_bc_str = bc_theta_min.type;
    }

    std::cout << "Boundary conditions: inner=" << bc_inner.type
              << ", outer=" << bc_outer.type
              << ", theta=" << theta_bc_str << std::endl;

    // Initialize material property matrices with r_orientation-dependent shape
    // This must match the shape of Br, Btheta, and Az matrices
    if (r_orientation == "horizontal") {
        // Image is (ntheta, nr) = (rows, cols)
        // Matrices: (ntheta, nr) with indexing (theta_idx, r_idx)
        mu_map = Eigen::MatrixXd::Constant(ntheta, nr, MU_0);
        jz_map = Eigen::MatrixXd::Zero(ntheta, nr);
    } else {  // vertical
        // Image is (nr, ntheta) = (rows, cols)
        // Matrices: (nr, ntheta) with indexing (r_idx, theta_idx)
        mu_map = Eigen::MatrixXd::Constant(nr, ntheta, MU_0);
        jz_map = Eigen::MatrixXd::Zero(nr, ntheta);
    }
}

void MagneticFieldAnalyzer::parseUserVariables() {
    // Reserved variable names that cannot be used as user-defined variables
    static const std::set<std::string> reserved_vars = {
        "step",    // Jz: step number
        "H",       // mu_r: magnetic field intensity
        "dx", "dy",        // Cartesian cell size
        "dr", "dtheta",    // Polar cell size
        "N", "A",          // Jz: material pixel count and area
        "pi", "e"          // tinyexpr built-in constants
    };

    if (!config["variables"]) {
        return;  // No user-defined variables
    }

    std::cout << "Parsing user-defined variables:" << std::endl;

    for (const auto& var : config["variables"]) {
        std::string var_name = var.first.as<std::string>();

        // Check for reserved variable names
        if (reserved_vars.count(var_name) > 0) {
            throw std::runtime_error("Variable '" + var_name + "' is a reserved variable name. "
                                     "Reserved variables: step, H, dx, dy, dr, dtheta, N, A, pi, e");
        }

        // Get the value (can be number or formula string)
        double var_value = 0.0;

        if (var.second.IsScalar()) {
            std::string val_str = var.second.as<std::string>();

            // Replace references to already-defined user variables ($varname -> value)
            // Sort by name length descending to avoid partial replacement issues
            // (e.g., $omega shouldn't partially match $o)
            std::vector<std::pair<std::string, double>> sorted_vars(user_variables.begin(), user_variables.end());
            std::sort(sorted_vars.begin(), sorted_vars.end(),
                      [](const auto& a, const auto& b) { return a.first.length() > b.first.length(); });

            for (const auto& [existing_var_name, existing_var_value] : sorted_vars) {
                std::string search_str = "$" + existing_var_name;
                size_t pos = 0;
                while ((pos = val_str.find(search_str, pos)) != std::string::npos) {
                    // Replace $varname with the actual numeric value
                    std::ostringstream oss;
                    oss << std::setprecision(17) << existing_var_value;
                    val_str.replace(pos, search_str.length(), oss.str());
                    pos += oss.str().length();
                }
            }

            // Try to evaluate as formula using tinyexpr
            te_parser parser;
            var_value = parser.evaluate(val_str);

            if (!parser.success()) {
                throw std::runtime_error("Failed to evaluate variable '" + var_name + "' with value: " + val_str +
                                         ". Note: You can reference previously defined variables with $varname syntax.");
            }
        } else {
            throw std::runtime_error("Variable '" + var_name + "' must be a scalar value or formula");
        }

        user_variables[var_name] = var_value;
        std::cout << "  $" << var_name << " = " << var_value << std::endl;
    }
}

void MagneticFieldAnalyzer::setupMaterialProperties() {
    if (!config["materials"]) {
        std::cout << "No materials defined in config" << std::endl;
        return;
    }

    // Flip image once for coordinate system transformation
    cv::Mat image_flipped;
    cv::flip(image, image_flipped, 0);  // Flip vertically: y down -> y up

    // Initialize adaptive mesh coarsening
    coarsening_enabled = false;
    material_coarsen.clear();

    // =========================================================================
    // PASS 1: Count pixels and calculate area for each material
    // This populates material_pixel_info so that $N and $A can be used in formulas
    // =========================================================================
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;

        // Get RGB values
        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        int count = 0;
        double total_area = 0.0;

        if (coordinate_system == "polar") {
            if (r_orientation == "horizontal") {
                for (int j = 0; j < ntheta; j++) {
                    for (int i = 0; i < nr; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            count++;
                            double r_at_cell = r_start + (i + 0.5) * dr;
                            total_area += r_at_cell * dr * dtheta;
                        }
                    }
                }
            } else {  // vertical
                for (int j = 0; j < nr; j++) {
                    for (int i = 0; i < ntheta; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            count++;
                            double r_at_cell = r_start + (j + 0.5) * dr;
                            total_area += r_at_cell * dr * dtheta;
                        }
                    }
                }
            }
        } else {  // Cartesian
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        count++;
                    }
                }
            }
            total_area = count * dx * dy;
        }

        // Store material pixel information
        MaterialPixelInfo pixel_info;
        pixel_info.pixel_count = count;
        pixel_info.area = total_area;
        material_pixel_info[name] = pixel_info;
    }

    // =========================================================================
    // PASS 2: Parse formulas, evaluate properties, and apply to maps
    // Now $N and $A are available for all materials
    // =========================================================================
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        YAML::Node props = material.second;

        // Resolve preset if specified (preset properties are merged, material-specific overrides)
        if (props["preset"]) {
            std::string preset_name = props["preset"].as<std::string>();
            auto preset_it = material_presets.find(preset_name);
            if (preset_it == material_presets.end()) {
                throw std::runtime_error("Material '" + name + "' references unknown preset: " + preset_name);
            }

            // Start with preset properties, then override with material-specific properties
            YAML::Node merged = YAML::Clone(preset_it->second);
            for (auto it = props.begin(); it != props.end(); ++it) {
                std::string key = it->first.as<std::string>();
                if (key != "preset") {  // Don't copy the preset reference itself
                    merged[key] = it->second;
                }
            }
            props = merged;
            std::cout << "Material '" << name << "' using preset '" << preset_name << "'" << std::endl;
        }

        // Get RGB values
        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Parse mu_r value (static, formula, or table) - Nonlinear support
        MuValue mu_value = parseMuValue(props["mu_r"]);

        // Parse dmu_r_extrapolation if specified (for TABLE type only)
        if (props["dmu_r_extrapolation"]) {
            const YAML::Node& extrap_node = props["dmu_r_extrapolation"];

            if (extrap_node.IsScalar()) {
                std::string extrap_str = extrap_node.as<std::string>();

                // Check if it's a formula (contains $H or math operators)
                if (extrap_str.find('$') != std::string::npos ||
                    extrap_str.find('*') != std::string::npos ||
                    extrap_str.find('/') != std::string::npos ||
                    extrap_str.find('(') != std::string::npos) {
                    // Formula-based extrapolation
                    mu_value.has_dmu_extrapolation = true;
                    mu_value.dmu_r_extrap_formula = extrap_str;
                    std::cout << "  dmu_r extrapolation: formula = \"" << extrap_str << "\"" << std::endl;
                } else {
                    // Constant extrapolation
                    mu_value.has_dmu_extrapolation = true;
                    mu_value.dmu_r_extrap_const = std::stod(extrap_str);
                    std::cout << "  dmu_r extrapolation: constant = " << mu_value.dmu_r_extrap_const << std::endl;
                }
            }
        }

        material_mu[name] = mu_value;

        // Check if this material is nonlinear
        if (mu_value.type != MuType::STATIC) {
            has_nonlinear_materials = true;
            std::cout << "Nonlinear material detected: " << name << std::endl;

            // Validate table if present
            if (mu_value.type == MuType::TABLE) {
                validateMuTable(mu_value.H_table, mu_value.mu_table, name);
            }

            // Generate B-H table for nonlinear materials
            generateBHTable(name, mu_value);
        }

        // Evaluate mu_r for initial state (H=0)
        double mu_r = evaluateMu(mu_value, 0.0);

        // Parse antialias flag and add to antialias_materials if enabled
        bool antialias_enabled = props["anti_aliasing"].as<bool>(false);
        if (antialias_enabled) {
            AntialiasableMaterial aa_mat;
            aa_mat.name = name;
            aa_mat.rgb = cv::Vec3b(rgb[0], rgb[1], rgb[2]);
            aa_mat.mu_r = mu_r;
            antialias_materials.push_back(aa_mat);
        }

        // Parse coarsening settings for adaptive mesh
        CoarsenConfig coarsen_cfg;
        // std::cerr << "*** COARSEN DEBUG: " << name << " ***" << std::endl;
        // std::cerr.flush();

        if (props["coarsen"]) {
            coarsen_cfg.enabled = props["coarsen"].as<bool>(false);
            std::cout << "  DEBUG [" << name << "] coarsen parsed: " << (coarsen_cfg.enabled ? "true" : "false") << std::endl;
        }
        if (props["coarsen_ratio"]) {
            coarsen_cfg.ratio = props["coarsen_ratio"].as<int>(2);
        }
        material_coarsen[name] = coarsen_cfg;
        if (coarsen_cfg.enabled) {
            coarsening_enabled = true;
            std::cout << "  [" << name << "] Coarsening enabled: ratio = " << coarsen_cfg.ratio << std::endl;
        }

        // Parse Jz value (static, formula, or array)
        JzValue jz_value = parseJzValue(props["jz"]);
        material_jz[name] = jz_value;

        // Evaluate Jz for step 0 (initial state)
        // Now material_pixel_info is populated, so $N and $A are available
        double jz = evaluateJz(jz_value, 0, name);

        // Apply properties to matching pixels
        if (coordinate_system == "polar") {
            if (r_orientation == "horizontal") {
                // mu_map shape: (ntheta, nr), indexing: (theta_idx, r_idx) = (j, i)
                for (int j = 0; j < ntheta; j++) {
                    for (int i = 0; i < nr; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            mu_map(j, i) = mu_r * MU_0;
                            jz_map(j, i) = jz;
                        }
                    }
                }
            } else {  // vertical
                // mu_map shape: (nr, ntheta), indexing: (r_idx, theta_idx) = (j, i)
                for (int j = 0; j < nr; j++) {
                    for (int i = 0; i < ntheta; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                            mu_map(j, i) = mu_r * MU_0;
                            jz_map(j, i) = jz;
                        }
                    }
                }
            }
        } else {  // Cartesian
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        mu_map(j, i) = mu_r * MU_0;
                        jz_map(j, i) = jz;
                    }
                }
            }
        }

        // Get pixel info for display
        const auto& pix_info = material_pixel_info[name];

        // Display Jz type
        std::string jz_type_str;
        switch (jz_value.type) {
            case JzType::STATIC: jz_type_str = "static"; break;
            case JzType::FORMULA: jz_type_str = "formula"; break;
            case JzType::ARRAY: jz_type_str = "array"; break;
        }

        std::cout << "Material '" << name << "': mu_r=" << mu_r << ", Jz=" << jz << " A/m^2, pixels=" << pix_info.pixel_count
                  << ", area=" << pix_info.area << " m^2"
                  << (antialias_enabled ? ", antialias=on" : "") << std::endl;
    }

    // =========================================================================
    // PASS 3: Apply anti-aliasing interpolation to gradient pixels
    // Only if at least 2 materials have antialias enabled
    // =========================================================================
    if (antialias_materials.size() >= 2) {
        std::cout << "Anti-aliasing: " << antialias_materials.size() << " materials enabled, processing gradient pixels..." << std::endl;

        // Build a set of exact RGB values for quick lookup
        std::set<uint32_t> exact_rgb_set;
        for (const auto& material : config["materials"]) {
            std::vector<int> rgb = material.second["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});
            uint32_t key = (rgb[0] << 16) | (rgb[1] << 8) | rgb[2];
            exact_rgb_set.insert(key);
        }

        int antialias_count = 0;

        if (coordinate_system == "polar") {
            if (r_orientation == "horizontal") {
                for (int j = 0; j < ntheta; j++) {
                    for (int i = 0; i < nr; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        uint32_t pixel_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];

                        // Skip if exact match exists
                        if (exact_rgb_set.count(pixel_key) > 0) {
                            continue;
                        }

                        // Try antialias interpolation
                        double out_mu_r;
                        double mu_interp = interpolateAntialiasedMu(pixel, out_mu_r);
                        if (mu_interp > 0) {
                            mu_map(j, i) = mu_interp;
                            antialias_count++;
                        }
                    }
                }
            } else {  // vertical
                for (int j = 0; j < nr; j++) {
                    for (int i = 0; i < ntheta; i++) {
                        cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                        uint32_t pixel_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];

                        if (exact_rgb_set.count(pixel_key) > 0) {
                            continue;
                        }

                        double out_mu_r;
                        double mu_interp = interpolateAntialiasedMu(pixel, out_mu_r);
                        if (mu_interp > 0) {
                            mu_map(j, i) = mu_interp;
                            antialias_count++;
                        }
                    }
                }
            }
        } else {  // Cartesian
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);
                    uint32_t pixel_key = (pixel[0] << 16) | (pixel[1] << 8) | pixel[2];

                    if (exact_rgb_set.count(pixel_key) > 0) {
                        continue;
                    }

                    double out_mu_r;
                    double mu_interp = interpolateAntialiasedMu(pixel, out_mu_r);
                    if (mu_interp > 0) {
                        mu_map(j, i) = mu_interp;
                        antialias_count++;
                    }
                }
            }
        }

        std::cout << "Anti-aliasing: " << antialias_count << " gradient pixels interpolated" << std::endl;
    }
}

void MagneticFieldAnalyzer::validateBoundaryConditions() {
    if (!config["boundary_conditions"]) {
        std::cout << "Using default Dirichlet boundary conditions (value=0)" << std::endl;
        return;
    }

    auto bc = config["boundary_conditions"];

    // Helper lambda to parse Robin parameters
    auto parseRobinParams = [](const YAML::Node& node, BoundaryCondition& bc_out) {
        bc_out.type = node["type"].as<std::string>("dirichlet");
        bc_out.value = node["value"].as<double>(0.0);

        // Parse Robin BC parameters if type is "robin"
        if (bc_out.type == "robin") {
            bc_out.alpha = node["alpha"].as<double>(1.0);
            bc_out.beta = node["beta"].as<double>(0.0);
            bc_out.gamma = node["gamma"].as<double>(0.0);

            // Validate: alpha and beta cannot both be zero
            if (std::abs(bc_out.alpha) < 1e-15 && std::abs(bc_out.beta) < 1e-15) {
                throw std::runtime_error("Robin BC: alpha and beta cannot both be zero");
            }
        }
    };

    // Left boundary
    if (bc["left"]) {
        parseRobinParams(bc["left"], bc_left);
    }

    // Right boundary
    if (bc["right"]) {
        parseRobinParams(bc["right"], bc_right);
    }

    // Bottom boundary
    if (bc["bottom"]) {
        parseRobinParams(bc["bottom"], bc_bottom);
    }

    // Top boundary
    if (bc["top"]) {
        parseRobinParams(bc["top"], bc_top);
    }

    std::cout << "Boundary conditions:" << std::endl;
    std::cout << "  Left: " << bc_left.type;
    if (bc_left.type == "robin") {
        std::cout << " (alpha=" << bc_left.alpha << ", beta=" << bc_left.beta << ", gamma=" << bc_left.gamma << ")";
    }
    std::cout << std::endl;
    std::cout << "  Right: " << bc_right.type;
    if (bc_right.type == "robin") {
        std::cout << " (alpha=" << bc_right.alpha << ", beta=" << bc_right.beta << ", gamma=" << bc_right.gamma << ")";
    }
    std::cout << std::endl;
    std::cout << "  Bottom: " << bc_bottom.type;
    if (bc_bottom.type == "robin") {
        std::cout << " (alpha=" << bc_bottom.alpha << ", beta=" << bc_bottom.beta << ", gamma=" << bc_bottom.gamma << ")";
    }
    std::cout << std::endl;
    std::cout << "  Top: " << bc_top.type;
    if (bc_top.type == "robin") {
        std::cout << " (alpha=" << bc_top.alpha << ", beta=" << bc_top.beta << ", gamma=" << bc_top.gamma << ")";
    }
    std::cout << std::endl;
}

// ============================================================================
// Anti-aliasing interpolation methods
// ============================================================================

double MagneticFieldAnalyzer::calculateRGBDistance(const cv::Vec3b& a, const cv::Vec3b& b) const {
    double dr = static_cast<double>(a[0]) - static_cast<double>(b[0]);
    double dg = static_cast<double>(a[1]) - static_cast<double>(b[1]);
    double db = static_cast<double>(a[2]) - static_cast<double>(b[2]);
    return std::sqrt(dr * dr + dg * dg + db * db);
}

bool MagneticFieldAnalyzer::isPointOnLineSegment(const cv::Vec3b& pixel,
                                                   const cv::Vec3b& a,
                                                   const cv::Vec3b& b,
                                                   double tolerance) const {
    // Check if pixel lies on the line segment between a and b in RGB space
    // Calculate distance from pixel to line segment AB

    double dist_ab = calculateRGBDistance(a, b);
    if (dist_ab < 1e-6) {
        // a and b are the same color
        return calculateRGBDistance(pixel, a) < tolerance;
    }

    // Calculate distance from pixel to point A and to point B
    double dist_pa = calculateRGBDistance(pixel, a);
    double dist_pb = calculateRGBDistance(pixel, b);

    // If pixel is outside the segment endpoints (with tolerance), reject
    if (dist_pa > dist_ab + tolerance || dist_pb > dist_ab + tolerance) {
        return false;
    }

    // Calculate perpendicular distance from pixel to line AB
    // Using cross product in 3D RGB space: |PA × AB| / |AB|
    cv::Vec3d pa(pixel[0] - a[0], pixel[1] - a[1], pixel[2] - a[2]);
    cv::Vec3d ab(b[0] - a[0], b[1] - a[1], b[2] - a[2]);

    // Cross product PA × AB
    cv::Vec3d cross(
        pa[1] * ab[2] - pa[2] * ab[1],
        pa[2] * ab[0] - pa[0] * ab[2],
        pa[0] * ab[1] - pa[1] * ab[0]
    );

    double cross_mag = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
    double perp_dist = cross_mag / dist_ab;

    return perp_dist < tolerance;
}

double MagneticFieldAnalyzer::interpolateAntialiasedMu(const cv::Vec3b& pixel, double& out_mu_r) const {
    // Find the two closest antialias-enabled materials that the pixel lies between
    // Returns the interpolated mu value (absolute, not relative) or -1 if not applicable

    if (antialias_materials.size() < 2) {
        return -1.0;  // Need at least 2 antialias materials
    }

    // Try all pairs of antialias materials
    double best_mu = -1.0;
    double min_perp_dist = 1e9;

    for (size_t i = 0; i < antialias_materials.size(); ++i) {
        for (size_t j = i + 1; j < antialias_materials.size(); ++j) {
            const auto& mat_a = antialias_materials[i];
            const auto& mat_b = antialias_materials[j];

            // Check if pixel lies on the line segment between mat_a and mat_b
            if (!isPointOnLineSegment(pixel, mat_a.rgb, mat_b.rgb)) {
                continue;
            }

            // Calculate the interpolation ratio (how far along A->B is the pixel)
            double dist_ab = calculateRGBDistance(mat_a.rgb, mat_b.rgb);
            double dist_pa = calculateRGBDistance(pixel, mat_a.rgb);

            // Project pixel onto line AB to get accurate ratio
            cv::Vec3d pa(pixel[0] - mat_a.rgb[0], pixel[1] - mat_a.rgb[1], pixel[2] - mat_a.rgb[2]);
            cv::Vec3d ab(mat_b.rgb[0] - mat_a.rgb[0], mat_b.rgb[1] - mat_a.rgb[1], mat_b.rgb[2] - mat_a.rgb[2]);

            double dot = pa[0] * ab[0] + pa[1] * ab[1] + pa[2] * ab[2];
            double ratio = dot / (dist_ab * dist_ab);

            // Clamp ratio to [0, 1]
            ratio = std::max(0.0, std::min(1.0, ratio));

            // Calculate perpendicular distance for tie-breaking
            cv::Vec3d cross(
                pa[1] * ab[2] - pa[2] * ab[1],
                pa[2] * ab[0] - pa[0] * ab[2],
                pa[0] * ab[1] - pa[1] * ab[0]
            );
            double cross_mag = std::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]);
            double perp_dist = cross_mag / dist_ab;

            // Use the pair with smallest perpendicular distance
            if (perp_dist < min_perp_dist) {
                min_perp_dist = perp_dist;

                // Harmonic mean interpolation (series magnetic circuit model)
                // μ_interp = 1 / ((1-ratio)/μ_A + ratio/μ_B)
                double mu_a = mat_a.mu_r;
                double mu_b = mat_b.mu_r;

                out_mu_r = 1.0 / ((1.0 - ratio) / mu_a + ratio / mu_b);
                best_mu = out_mu_r * MU_0;  // Return absolute permeability
            }
        }
    }

    return best_mu;
}

// ============================================================================
// Flux linkage calculation methods
// ============================================================================

void MagneticFieldAnalyzer::parseFluxLinkagePaths() {
    flux_linkage_paths.clear();
    flux_linkage_results.clear();

    if (!config["flux_linkage"]) {
        return;  // No flux linkage paths defined
    }

    std::cout << "Parsing flux linkage paths..." << std::endl;

    for (const auto& path_node : config["flux_linkage"]) {
        FluxLinkagePath path;

        if (!path_node["name"]) {
            std::cerr << "Warning: flux_linkage entry missing 'name', skipping" << std::endl;
            continue;
        }
        path.name = path_node["name"].as<std::string>();

        if (!path_node["start"] || !path_node["end"]) {
            std::cerr << "Warning: flux_linkage '" << path.name << "' missing start/end, skipping" << std::endl;
            continue;
        }

        auto start = path_node["start"].as<std::vector<double>>();
        auto end = path_node["end"].as<std::vector<double>>();

        if (start.size() < 2 || end.size() < 2) {
            std::cerr << "Warning: flux_linkage '" << path.name << "' invalid start/end format, skipping" << std::endl;
            continue;
        }

        path.x_start = start[0];
        path.y_start = start[1];
        path.x_end = end[0];
        path.y_end = end[1];

        flux_linkage_paths.push_back(path);
        flux_linkage_results[path.name] = std::vector<double>();  // Initialize empty results

        std::cout << "  Flux linkage path '" << path.name << "': ("
                  << path.x_start << ", " << path.y_start << ") -> ("
                  << path.x_end << ", " << path.y_end << ")" << std::endl;
    }

    std::cout << "Loaded " << flux_linkage_paths.size() << " flux linkage path(s)" << std::endl;
}

double MagneticFieldAnalyzer::interpolateAz(double x_phys, double y_phys) const {
    // Bilinear interpolation of Az at physical coordinates
    // x_phys, y_phys are in meters

    // Convert physical coordinates to grid indices (floating point)
    double i_float = x_phys / dx;
    double j_float = y_phys / dy;

    // Get integer grid indices
    int i0 = static_cast<int>(std::floor(i_float));
    int j0 = static_cast<int>(std::floor(j_float));

    // Clamp to valid range
    i0 = std::max(0, std::min(i0, nx - 2));
    j0 = std::max(0, std::min(j0, ny - 2));

    // Fractional parts
    double fx = i_float - i0;
    double fy = j_float - j0;

    // Clamp fractions to [0, 1]
    fx = std::max(0.0, std::min(1.0, fx));
    fy = std::max(0.0, std::min(1.0, fy));

    // Bilinear interpolation
    // Az is stored as Az(j, i) where j=row, i=col
    double Az00 = Az(j0, i0);
    double Az10 = Az(j0, i0 + 1);
    double Az01 = Az(j0 + 1, i0);
    double Az11 = Az(j0 + 1, i0 + 1);

    double Az_interp = (1.0 - fx) * (1.0 - fy) * Az00
                     + fx * (1.0 - fy) * Az10
                     + (1.0 - fx) * fy * Az01
                     + fx * fy * Az11;

    return Az_interp;
}

double MagneticFieldAnalyzer::calculateFluxLinkage(const FluxLinkagePath& path) const {
    // Flux linkage Φ = Az(end) - Az(start) [Wb/m]
    // In 2D analysis, this gives flux per unit depth

    double Az_start = interpolateAz(path.x_start, path.y_start);
    double Az_end = interpolateAz(path.x_end, path.y_end);

    return Az_end - Az_start;
}

void MagneticFieldAnalyzer::calculateAllFluxLinkages(int step) {
    if (flux_linkage_paths.empty()) {
        return;
    }

    for (const auto& path : flux_linkage_paths) {
        double phi = calculateFluxLinkage(path);
        flux_linkage_results[path.name].push_back(phi);

        std::cout << "Flux linkage [" << path.name << "] step " << step
                  << ": " << phi << " Wb/m" << std::endl;
    }
}

void MagneticFieldAnalyzer::exportFluxLinkageCSV(const std::string& output_dir) const {
    if (flux_linkage_paths.empty() || flux_linkage_results.empty()) {
        return;
    }

    // Create FluxLinkage subdirectory
    std::string flux_dir = output_dir + "/FluxLinkage";
    createDirectory(flux_dir);

    std::string csv_path = flux_dir + "/flux_linkage.csv";
    std::ofstream file(csv_path);

    if (!file.is_open()) {
        std::cerr << "Warning: Could not create flux linkage CSV: " << csv_path << std::endl;
        return;
    }

    // Write header
    file << "step";
    for (const auto& path : flux_linkage_paths) {
        file << "," << path.name;
    }
    file << "\n";

    // Determine number of steps from first path's results
    size_t num_steps = 0;
    if (!flux_linkage_paths.empty()) {
        auto it = flux_linkage_results.find(flux_linkage_paths[0].name);
        if (it != flux_linkage_results.end()) {
            num_steps = it->second.size();
        }
    }

    // Write data rows
    for (size_t step = 0; step < num_steps; ++step) {
        file << step;
        for (const auto& path : flux_linkage_paths) {
            auto it = flux_linkage_results.find(path.name);
            if (it != flux_linkage_results.end() && step < it->second.size()) {
                file << "," << std::scientific << std::setprecision(10) << it->second[step];
            } else {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Flux linkage results exported to: " << csv_path << std::endl;
}

// ============================================================================
// Adaptive mesh coarsening methods
// ============================================================================

cv::Mat MagneticFieldAnalyzer::detectMaterialBoundaries() {
    // Detect material boundaries using Canny edge detection
    // Returns a binary mask where boundaries are marked as white (255)

    cv::Mat gray, edges;

    // Convert to grayscale for edge detection
    cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);

    // Apply Canny edge detection
    cv::Canny(gray, edges, 50, 150);

    // Dilate edges to create a protection zone around boundaries
    // This ensures cells near boundaries are not coarsened
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(edges, edges, kernel);

    // Flip to match analysis coordinate system (y up)
    cv::Mat edges_flipped;
    cv::flip(edges, edges_flipped, 0);

    return edges_flipped;
}

void MagneticFieldAnalyzer::calculateOptimalSkipRatios() {
    // Calculate skip_x and skip_y from coarsen_ratio to maintain aspect ratio
    // Goal: Make coarsened cells as square as possible

    for (auto& [mat_name, cfg] : material_coarsen) {
        if (!cfg.enabled || cfg.ratio <= 1) {
            cfg.skip_x = cfg.skip_y = 1;
            continue;
        }

        double aspect;  // Physical cell aspect ratio (x-direction / y-direction)

        if (coordinate_system == "polar") {
            // Polar coordinates: use representative radius for aspect calculation
            double r_mid = (r_start + r_end) / 2.0;
            double physical_dr = dr;
            double physical_dtheta = r_mid * dtheta;
            aspect = physical_dr / physical_dtheta;
        } else {
            // Cartesian coordinates: simple aspect ratio
            aspect = dx / dy;
        }

        // Optimal skip ratios to make coarsened cells square:
        // skip_x × skip_y ≈ ratio (area reduction)
        // (skip_x × dx) / (skip_y × dy) ≈ 1 (square cells)
        //
        // Solution: skip_x = sqrt(ratio / aspect), skip_y = sqrt(ratio * aspect)
        double skip_x_float = std::sqrt(cfg.ratio / aspect);
        double skip_y_float = std::sqrt(cfg.ratio * aspect);

        // Round to nearest integer
        cfg.skip_x = std::max(1, (int)std::round(skip_x_float));
        cfg.skip_y = std::max(1, (int)std::round(skip_y_float));

        // Log actual reduction ratio
        int actual_ratio = cfg.skip_x * cfg.skip_y;
        std::cout << "Material '" << mat_name << "': coarsen_ratio=" << cfg.ratio
                  << " -> skip_x=" << cfg.skip_x << ", skip_y=" << cfg.skip_y
                  << " (actual ratio=" << actual_ratio << ")" << std::endl;
    }
}

void MagneticFieldAnalyzer::generateCoarseningMask() {
    // Generate mask of active cells based on material coarsening settings
    // Cells on or near boundaries are always kept active

    if (!coarsening_enabled) {
        // No coarsening - all cells are active
        if (coordinate_system == "polar") {
            active_cells.resize(ntheta, nr);
            n_active_cells = nr * ntheta;
        } else {
            active_cells.resize(ny, nx);
            n_active_cells = nx * ny;
        }
        active_cells.setConstant(true);
        return;
    }

    std::cout << "Generating coarsening mask..." << std::endl;

    // Calculate optimal skip ratios based on aspect ratio
    calculateOptimalSkipRatios();

    // Detect material boundaries
    cv::Mat boundaries = detectMaterialBoundaries();

    // Generate mask based on coordinate system
    if (coordinate_system == "polar") {
        active_cells.resize(ntheta, nr);
        active_cells.setConstant(true);
        generateCoarseningMaskPolar(boundaries);
    } else {
        active_cells.resize(ny, nx);
        active_cells.setConstant(true);
        generateCoarseningMaskCartesian(boundaries);
    }

    // Build index mappings
    buildCoarseIndexMaps();
}

void MagneticFieldAnalyzer::generateCoarseningMaskCartesian(const cv::Mat& boundaries) {
    // Generate coarsening mask for Cartesian coordinates
    // Uses skip_x and skip_y calculated from aspect ratio

    // Flip image for coordinate matching
    cv::Mat image_flipped;
    cv::flip(image, image_flipped, 0);

    int coarsened_count = 0;

    // Mark cells for coarsening based on material settings
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            // Skip boundary cells - always keep them active
            if (boundaries.at<uchar>(j, i) > 0) {
                continue;
            }

            // Skip cells on domain boundary
            if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
                continue;
            }

            // Find which material this cell belongs to
            cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(j, i);

            // Look up material coarsening settings
            for (const auto& mat_pair : material_coarsen) {
                const auto& cfg = mat_pair.second;
                if (!cfg.enabled) continue;
                if (cfg.skip_x <= 1 && cfg.skip_y <= 1) continue;

                // Check if this material matches by looking up its RGB
                for (const auto& material : config["materials"]) {
                    std::string mat_name = material.first.as<std::string>();
                    if (mat_name != mat_pair.first) continue;

                    auto rgb = material.second["rgb"].as<std::vector<int>>();
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        // This cell belongs to a coarsen-enabled material
                        // Keep only cells at coarsening grid positions
                        bool coarsen_x = (cfg.skip_x > 1 && i % cfg.skip_x != 0);
                        bool coarsen_y = (cfg.skip_y > 1 && j % cfg.skip_y != 0);
                        if (coarsen_x || coarsen_y) {
                            active_cells(j, i) = false;
                            coarsened_count++;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Cartesian coarsening: " << coarsened_count << " cells marked as inactive" << std::endl;
}

void MagneticFieldAnalyzer::generateCoarseningMaskPolar(const cv::Mat& boundaries) {
    // Generate coarsening mask for Polar coordinates
    // Uses skip_x (r-direction) and skip_y (theta-direction) calculated from aspect ratio
    //
    // IMPORTANT: boundaries is already flipped (from detectMaterialBoundaries) to match
    // analysis coordinates (y-up). We must also flip image to ensure coordinate consistency.

    // Flip image to match analysis coordinates (same as boundaries)
    cv::Mat image_flipped;
    cv::flip(image, image_flipped, 0);

    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    int coarsened_count = 0;

    for (int i_r = 0; i_r < nr; i_r++) {
        for (int j_theta = 0; j_theta < ntheta; j_theta++) {
            // Convert polar indices to flipped image coordinates
            // Both boundaries and image_flipped are now in analysis coordinates (y-up)
            int img_i, img_j;
            polarToImageIndices(i_r, j_theta, img_i, img_j);

            // Skip boundary cells - always keep them active
            if (boundaries.at<uchar>(img_j, img_i) > 0) continue;

            // Skip r-direction boundaries
            if (i_r == 0 || i_r == nr - 1) continue;

            // Skip theta-direction boundaries (if not periodic)
            if (!is_periodic && (j_theta == 0 || j_theta == ntheta - 1)) continue;

            // Find which material this cell belongs to
            // Use flipped image to match boundaries coordinate system
            cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(img_j, img_i);

            // Look up material coarsening settings
            for (const auto& mat_pair : material_coarsen) {
                const auto& cfg = mat_pair.second;
                if (!cfg.enabled) continue;
                if (cfg.skip_x <= 1 && cfg.skip_y <= 1) continue;

                // Check if this material matches by looking up its RGB
                for (const auto& material : config["materials"]) {
                    std::string mat_name = material.first.as<std::string>();
                    if (mat_name != mat_pair.first) continue;

                    auto rgb = material.second["rgb"].as<std::vector<int>>();
                    if (pixel[0] == rgb[0] && pixel[1] == rgb[1] && pixel[2] == rgb[2]) {
                        // This cell belongs to a coarsen-enabled material
                        // r-direction: skip_x, theta-direction: skip_y
                        bool coarsen_r = (cfg.skip_x > 1 && i_r % cfg.skip_x != 0);
                        bool coarsen_theta = (cfg.skip_y > 1 && j_theta % cfg.skip_y != 0);
                        if (coarsen_r || coarsen_theta) {
                            active_cells(j_theta, i_r) = false;  // Note: (theta, r) order
                            coarsened_count++;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Polar coarsening: " << coarsened_count << " cells marked as inactive" << std::endl;
}

void MagneticFieldAnalyzer::polarToImageIndices(int i_r, int j_theta, int& img_i, int& img_j) const {
    // Convert polar grid indices (i_r, j_theta) to image pixel coordinates (img_i, img_j)
    // Depends on r_orientation setting

    if (r_orientation == "horizontal") {
        // Image x-axis = r, Image y-axis = theta
        img_i = i_r;
        img_j = j_theta;
    } else {  // "vertical"
        // Image y-axis = r, Image x-axis = theta
        img_i = j_theta;
        img_j = i_r;
    }
}

void MagneticFieldAnalyzer::buildCoarseIndexMaps() {
    // Build mappings between coarse indices and fine grid coordinates

    coarse_to_fine.clear();
    fine_to_coarse.clear();
    n_active_cells = 0;

    if (coordinate_system == "polar") {
        // Polar: iterate over (theta, r), store {i_r, j_theta}
        for (int j_theta = 0; j_theta < ntheta; j_theta++) {
            for (int i_r = 0; i_r < nr; i_r++) {
                if (active_cells(j_theta, i_r)) {
                    coarse_to_fine.push_back({i_r, j_theta});  // {r, theta} order
                    fine_to_coarse[{i_r, j_theta}] = n_active_cells;
                    n_active_cells++;
                }
            }
        }

        std::cout << "Active cells: " << n_active_cells << " / " << (nr * ntheta)
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * n_active_cells / (nr * ntheta)) << "%)" << std::endl;
    } else {
        // Cartesian: iterate over (y, x), store {i, j}
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (active_cells(j, i)) {
                    coarse_to_fine.push_back({i, j});
                    fine_to_coarse[{i, j}] = n_active_cells;
                    n_active_cells++;
                }
            }
        }

        std::cout << "Active cells: " << n_active_cells << " / " << (nx * ny)
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * n_active_cells / (nx * ny)) << "%)" << std::endl;
    }

    // Calculate local mesh spacing for active cells
    calculateLocalMeshSpacing();
}

void MagneticFieldAnalyzer::calculateLocalMeshSpacing() {
    // Calculate local mesh spacing (h_minus, h_plus) for each active cell
    // This is needed for the non-uniform FDM stencil

    if (coordinate_system == "polar") {
        // Polar: local_dx = radial spacing, local_dy = angular spacing
        // Both directions can be non-uniform due to coarsening
        local_dx.resize(ntheta, nr);
        local_dy.resize(ntheta, nr);
        local_dx.setZero();
        local_dy.setZero();

        for (int idx = 0; idx < n_active_cells; idx++) {
            auto [i_r, j_theta] = coarse_to_fine[idx];  // {r, theta} order

            // Find distance to next active cell in +r direction
            int i_next = findNextActiveRadial(i_r, j_theta, +1);
            local_dx(j_theta, i_r) = (i_next - i_r) * dr;

            // Find distance to next active cell in +theta direction
            int j_next = findNextActiveTheta(i_r, j_theta, +1);
            local_dy(j_theta, i_r) = (j_next - j_theta) * dtheta;
        }
    } else {
        // Cartesian: local_dx = x spacing, local_dy = y spacing
        local_dx.resize(ny, nx);
        local_dy.resize(ny, nx);
        local_dx.setZero();
        local_dy.setZero();

        for (int idx = 0; idx < n_active_cells; idx++) {
            auto [i, j] = coarse_to_fine[idx];

            // Find distance to next active cell in +x direction
            int i_next = findNextActiveX(i, j, +1);
            local_dx(j, i) = (i_next - i) * dx;

            // Find distance to next active cell in +y direction
            int j_next = findNextActiveY(i, j, +1);
            local_dy(j, i) = (j_next - j) * dy;
        }
    }
}

int MagneticFieldAnalyzer::findNextActiveX(int i, int j, int direction) const {
    // Find the next active cell in X direction
    // direction: +1 for right, -1 for left

    int i_next = i + direction;
    while (i_next >= 0 && i_next < nx) {
        if (active_cells(j, i_next)) {
            return i_next;
        }
        i_next += direction;
    }
    // Reached boundary
    return (direction > 0) ? nx - 1 : 0;
}

int MagneticFieldAnalyzer::findNextActiveY(int i, int j, int direction) const {
    // Find the next active cell in Y direction
    // direction: +1 for up, -1 for down

    int j_next = j + direction;
    while (j_next >= 0 && j_next < ny) {
        if (active_cells(j_next, i)) {
            return j_next;
        }
        j_next += direction;
    }
    // Reached boundary
    return (direction > 0) ? ny - 1 : 0;
}

int MagneticFieldAnalyzer::findNextActiveRadial(int i_r, int j_theta, int direction) const {
    // Find the next active cell in radial direction (for polar coordinates)
    // direction: +1 for outward, -1 for inward

    int i_next = i_r + direction;
    while (i_next >= 0 && i_next < nr) {
        if (active_cells(j_theta, i_next)) {  // Note: (theta, r) order
            return i_next;
        }
        i_next += direction;
    }
    // Reached boundary
    return (direction > 0) ? nr - 1 : 0;
}

int MagneticFieldAnalyzer::findNextActiveTheta(int i_r, int j_theta, int direction) const {
    // Find the next active cell in theta direction (for polar coordinates)
    // direction: +1 for CCW, -1 for CW
    // Handles periodic boundaries

    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");

    int j_next = j_theta + direction;
    int iterations = 0;

    while (iterations < ntheta) {
        // Handle periodic wrapping
        if (is_periodic) {
            j_next = (j_next + ntheta) % ntheta;
        } else {
            // Non-periodic: clamp at boundaries
            if (j_next < 0) return 0;
            if (j_next >= ntheta) return ntheta - 1;
        }

        if (active_cells(j_next, i_r)) {  // Note: (theta, r) order
            return j_next;
        }

        j_next += direction;
        iterations++;
    }

    // Fallback (shouldn't reach here unless all cells are inactive)
    return j_theta;
}

std::pair<int, int> MagneticFieldAnalyzer::findActiveNeighbor(int i, int j, int di, int dj) const {
    // Find the nearest active neighbor cell in direction (di, dj)
    // Returns (i_neighbor, j_neighbor)

    int i_next = i + di;
    int j_next = j + dj;

    while (i_next >= 0 && i_next < nx && j_next >= 0 && j_next < ny) {
        if (active_cells(j_next, i_next)) {
            return {i_next, j_next};
        }
        i_next += di;
        j_next += dj;
    }

    // Return boundary position if no active cell found
    i_next = std::max(0, std::min(nx - 1, i_next));
    j_next = std::max(0, std::min(ny - 1, j_next));
    return {i_next, j_next};
}

double MagneticFieldAnalyzer::bilinearInterpolateFromCoarse(int i, int j, const Eigen::VectorXd& Az_coarse) const {
    // Bilinear interpolation for inactive cells from surrounding active cells

    // Find surrounding active cells
    int i_left = i, i_right = i, j_bottom = j, j_top = j;

    // Search left
    while (i_left > 0 && !active_cells(j, i_left)) i_left--;
    // Search right
    while (i_right < nx - 1 && !active_cells(j, i_right)) i_right++;
    // Search down
    while (j_bottom > 0 && !active_cells(j_bottom, i)) j_bottom--;
    // Search up
    while (j_top < ny - 1 && !active_cells(j_top, i)) j_top++;

    // If we found active cells on all sides, use bilinear interpolation
    if (active_cells(j, i_left) && active_cells(j, i_right) &&
        active_cells(j_bottom, i) && active_cells(j_top, i)) {

        // Get coarse indices
        auto it_left = fine_to_coarse.find({i_left, j});
        auto it_right = fine_to_coarse.find({i_right, j});
        auto it_bottom = fine_to_coarse.find({i, j_bottom});
        auto it_top = fine_to_coarse.find({i, j_top});

        if (it_left != fine_to_coarse.end() && it_right != fine_to_coarse.end()) {
            // Interpolate in x direction first
            double fx = (i_right > i_left) ? double(i - i_left) / (i_right - i_left) : 0.5;
            double Az_x = (1.0 - fx) * Az_coarse(it_left->second) + fx * Az_coarse(it_right->second);

            if (it_bottom != fine_to_coarse.end() && it_top != fine_to_coarse.end()) {
                double fy = (j_top > j_bottom) ? double(j - j_bottom) / (j_top - j_bottom) : 0.5;
                double Az_y = (1.0 - fy) * Az_coarse(it_bottom->second) + fy * Az_coarse(it_top->second);
                return 0.5 * (Az_x + Az_y);  // Average of x and y interpolations
            }
            return Az_x;
        }
    }

    // Fallback: use nearest active neighbor
    for (int dj = -1; dj <= 1; dj++) {
        for (int di = -1; di <= 1; di++) {
            int ni = i + di, nj = j + dj;
            if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && active_cells(nj, ni)) {
                auto it = fine_to_coarse.find({ni, nj});
                if (it != fine_to_coarse.end()) {
                    return Az_coarse(it->second);
                }
            }
        }
    }

    return 0.0;  // Should not reach here
}

// ============================================================================
// Adaptive mesh coarsened solver (Phase B)
// ============================================================================

void MagneticFieldAnalyzer::buildMatrixCoarsened(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
    // Build FDM system matrix for coarsened mesh with non-uniform stencil
    // Only active cells are included in the system

    A.resize(n_active_cells, n_active_cells);
    rhs.resize(n_active_cells);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(5 * n_active_cells);

    // Check for periodic boundary conditions
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    // Build equation for each active cell
    for (int idx = 0; idx < n_active_cells; idx++) {
        auto [i, j] = coarse_to_fine[idx];

        bool is_left = (i == 0);
        bool is_right = (i == nx - 1);
        bool is_bottom = (j == 0);
        bool is_top = (j == ny - 1);

        // Dirichlet boundary conditions
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

        // Robin boundary conditions
        if (is_left && bc_left.type == "robin") {
            double a = bc_left.alpha;
            double b = bc_left.beta;
            double g = bc_left.gamma;
            // Find next active cell in +x direction for the derivative term
            int i_east = findNextActiveX(i, j, +1);
            double h_east = (i_east - i) * dx;
            triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/h_east));
            auto it_east = fine_to_coarse.find({i_east, j});
            if (it_east != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_east->second, -b/h_east));
            }
            rhs(idx) = g;
            continue;
        } else if (is_right && bc_right.type == "robin") {
            double a = bc_right.alpha;
            double b = bc_right.beta;
            double g = bc_right.gamma;
            int i_west = findNextActiveX(i, j, -1);
            double h_west = (i - i_west) * dx;
            triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/h_west));
            auto it_west = fine_to_coarse.find({i_west, j});
            if (it_west != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_west->second, -b/h_west));
            }
            rhs(idx) = g;
            continue;
        } else if (is_bottom && bc_bottom.type == "robin") {
            double a = bc_bottom.alpha;
            double b = bc_bottom.beta;
            double g = bc_bottom.gamma;
            int j_north = findNextActiveY(i, j, +1);
            double h_north = (j_north - j) * dy;
            triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/h_north));
            auto it_north = fine_to_coarse.find({i, j_north});
            if (it_north != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_north->second, -b/h_north));
            }
            rhs(idx) = g;
            continue;
        } else if (is_top && bc_top.type == "robin") {
            double a = bc_top.alpha;
            double b = bc_top.beta;
            double g = bc_top.gamma;
            int j_south = findNextActiveY(i, j, -1);
            double h_south = (j - j_south) * dy;
            triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/h_south));
            auto it_south = fine_to_coarse.find({i, j_south});
            if (it_south != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_south->second, -b/h_south));
            }
            rhs(idx) = g;
            continue;
        }

        // Interior points - non-uniform FDM stencil
        // For non-uniform mesh: d²u/dx² ≈ 2/(h₋(h₋+h₊))·u_{i-1} - 2/(h₋h₊)·u_i + 2/(h₊(h₋+h₊))·u_{i+1}
        double coeff_center = 0.0;

        // Find neighboring active cells and compute distances
        int i_west = findNextActiveX(i, j, -1);
        int i_east = findNextActiveX(i, j, +1);
        int j_south = findNextActiveY(i, j, -1);
        int j_north = findNextActiveY(i, j, +1);

        // Handle periodic wrapping
        if (x_periodic) {
            if (i == 0 && i_west == 0) i_west = nx - 1;
            if (i == nx - 1 && i_east == nx - 1) i_east = 0;
        }
        if (y_periodic) {
            if (j == 0 && j_south == 0) j_south = ny - 1;
            if (j == ny - 1 && j_north == ny - 1) j_north = 0;
        }

        double h_west = std::abs(i - i_west) * dx;
        double h_east = std::abs(i_east - i) * dx;
        double h_south = std::abs(j - j_south) * dy;
        double h_north = std::abs(j_north - j) * dy;

        // Ensure minimum spacing to avoid division by zero
        if (h_west < 1e-15) h_west = dx;
        if (h_east < 1e-15) h_east = dx;
        if (h_south < 1e-15) h_south = dy;
        if (h_north < 1e-15) h_north = dy;

        // X-direction terms with non-uniform stencil
        // West neighbor
        if (i > 0 || x_periodic) {
            double mu_center = mu_map(j, i);
            double mu_neighbor = mu_map(j, i_west);
            double mu_west = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
            // Non-uniform coefficient: 2 / (h₋ * (h₋ + h₊) * μ)
            double coeff_west = 2.0 / (mu_west * h_west * (h_west + h_east));

            auto it_west = fine_to_coarse.find({i_west, j});
            bool west_is_dirichlet = (i_west == 0) && (bc_left.type == "dirichlet");

            if (!west_is_dirichlet && it_west != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_west->second, coeff_west));
            } else if (west_is_dirichlet) {
                rhs(idx) -= coeff_west * bc_left.value;
            }
            coeff_center -= coeff_west;
        }

        // East neighbor
        if (i < nx - 1 || x_periodic) {
            double mu_center = mu_map(j, i);
            double mu_neighbor = mu_map(j, i_east);
            double mu_east = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
            // Non-uniform coefficient: 2 / (h₊ * (h₋ + h₊) * μ)
            double coeff_east = 2.0 / (mu_east * h_east * (h_west + h_east));

            auto it_east = fine_to_coarse.find({i_east, j});
            bool east_is_dirichlet = (i_east == nx - 1) && (bc_right.type == "dirichlet");

            if (!east_is_dirichlet && it_east != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_east->second, coeff_east));
            } else if (east_is_dirichlet) {
                rhs(idx) -= coeff_east * bc_right.value;
            }
            coeff_center -= coeff_east;
        }

        // Y-direction terms with non-uniform stencil
        // South neighbor
        if (j > 0 || y_periodic) {
            double mu_center = mu_map(j, i);
            double mu_neighbor = mu_map(j_south, i);
            double mu_south = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
            double coeff_south = 2.0 / (mu_south * h_south * (h_south + h_north));

            auto it_south = fine_to_coarse.find({i, j_south});
            bool south_is_dirichlet = (j_south == 0) && (bc_bottom.type == "dirichlet");

            if (!south_is_dirichlet && it_south != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_south->second, coeff_south));
            } else if (south_is_dirichlet) {
                rhs(idx) -= coeff_south * bc_bottom.value;
            }
            coeff_center -= coeff_south;
        }

        // North neighbor
        if (j < ny - 1 || y_periodic) {
            double mu_center = mu_map(j, i);
            double mu_neighbor = mu_map(j_north, i);
            double mu_north = 2.0 / (1.0 / mu_center + 1.0 / mu_neighbor);
            double coeff_north = 2.0 / (mu_north * h_north * (h_south + h_north));

            auto it_north = fine_to_coarse.find({i, j_north});
            bool north_is_dirichlet = (j_north == ny - 1) && (bc_top.type == "dirichlet");

            if (!north_is_dirichlet && it_north != fine_to_coarse.end()) {
                triplets.push_back(Eigen::Triplet<double>(idx, it_north->second, coeff_north));
            } else if (north_is_dirichlet) {
                rhs(idx) -= coeff_north * bc_top.value;
            }
            coeff_center -= coeff_north;
        }

        // Center coefficient
        triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

        // Right-hand side (current density)
        rhs(idx) = -jz_map(j, i);
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
}

void MagneticFieldAnalyzer::interpolateToFullGrid(const Eigen::VectorXd& Az_coarse) {
    // Interpolate coarsened solution back to full grid

    Az.resize(ny, nx);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (active_cells(j, i)) {
                // Active cell - copy directly from coarse solution
                auto it = fine_to_coarse.find({i, j});
                if (it != fine_to_coarse.end()) {
                    Az(j, i) = Az_coarse(it->second);
                }
            } else {
                // Inactive cell - interpolate from surrounding active cells
                Az(j, i) = bilinearInterpolateFromCoarse(i, j, Az_coarse);
            }
        }
    }
}

double MagneticFieldAnalyzer::calculateThetaInterpolationWeight(int j_theta, int j_prev, int j_next) const {
    // Calculate interpolation weight for theta direction (periodic boundary aware)
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");

    if (j_prev == j_next) return 0.5;

    int dist_to_prev, dist_to_next;
    if (is_periodic) {
        dist_to_prev = (j_theta - j_prev + ntheta) % ntheta;
        dist_to_next = (j_next - j_theta + ntheta) % ntheta;
    } else {
        dist_to_prev = j_theta - j_prev;
        dist_to_next = j_next - j_theta;
    }

    return double(dist_to_prev) / double(dist_to_prev + dist_to_next);
}

double MagneticFieldAnalyzer::interpolateFromCoarseGridPolar(int i_r, int j_theta, const Eigen::VectorXd& Az_coarse) const {
    // Bilinear interpolation for inactive cells in polar coordinates
    // Interpolate in both r and theta directions

    // Find surrounding active cells in r direction
    int i_prev = i_r, i_next = i_r;
    while (i_prev > 0 && !active_cells(j_theta, i_prev)) i_prev--;
    while (i_next < nr - 1 && !active_cells(j_theta, i_next)) i_next++;

    // Find surrounding active cells in theta direction (periodic boundary aware)
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    int j_prev = j_theta, j_next = j_theta;

    // Search for j_prev
    int search = j_theta - 1;
    for (int k = 0; k < ntheta; k++) {
        int j_check = is_periodic ? (search + ntheta) % ntheta : std::max(0, search);
        if (!is_periodic && search < 0) break;
        if (active_cells(j_check, i_r)) {
            j_prev = j_check;
            break;
        }
        search--;
    }

    // Search for j_next
    search = j_theta + 1;
    for (int k = 0; k < ntheta; k++) {
        int j_check = is_periodic ? search % ntheta : std::min(ntheta - 1, search);
        if (!is_periodic && search >= ntheta) break;
        if (active_cells(j_check, i_r)) {
            j_next = j_check;
            break;
        }
        search++;
    }

    // Get coarse indices for 4 surrounding points
    auto it_sw = fine_to_coarse.find({i_prev, j_prev});  // (r_low, theta_low)
    auto it_se = fine_to_coarse.find({i_next, j_prev});  // (r_high, theta_low)
    auto it_nw = fine_to_coarse.find({i_prev, j_next});  // (r_low, theta_high)
    auto it_ne = fine_to_coarse.find({i_next, j_next});  // (r_high, theta_high)

    // Calculate interpolation weights
    double wr = (i_next != i_prev) ? double(i_r - i_prev) / double(i_next - i_prev) : 0.5;
    double wt = calculateThetaInterpolationWeight(j_theta, j_prev, j_next);

    // Get values at 4 points
    double v_sw = (it_sw != fine_to_coarse.end()) ? Az_coarse(it_sw->second) : 0.0;
    double v_se = (it_se != fine_to_coarse.end()) ? Az_coarse(it_se->second) : 0.0;
    double v_nw = (it_nw != fine_to_coarse.end()) ? Az_coarse(it_nw->second) : 0.0;
    double v_ne = (it_ne != fine_to_coarse.end()) ? Az_coarse(it_ne->second) : 0.0;

    // Fallback: if no points found, return 0
    int found_count = (it_sw != fine_to_coarse.end()) + (it_se != fine_to_coarse.end()) +
                      (it_nw != fine_to_coarse.end()) + (it_ne != fine_to_coarse.end());
    if (found_count == 0) return 0.0;

    // Bilinear interpolation
    double v_low = (1.0 - wr) * v_sw + wr * v_se;
    double v_high = (1.0 - wr) * v_nw + wr * v_ne;
    return (1.0 - wt) * v_low + wt * v_high;
}

void MagneticFieldAnalyzer::interpolateToFullGridPolar(const Eigen::VectorXd& Az_coarse) {
    // Interpolate coarsened solution back to full polar grid
    // Note: Az is stored in image-compatible format based on r_orientation

    if (r_orientation == "horizontal") {
        Az.resize(ntheta, nr);  // Image is (theta, r)
    } else {
        Az.resize(nr, ntheta);  // Image is (r, theta)
    }

    for (int i_r = 0; i_r < nr; i_r++) {
        for (int j_theta = 0; j_theta < ntheta; j_theta++) {
            if (active_cells(j_theta, i_r)) {  // Note: active_cells is always (theta, r)
                // Active cell - copy directly from coarse solution
                auto it = fine_to_coarse.find({i_r, j_theta});
                if (it != fine_to_coarse.end()) {
                    if (r_orientation == "horizontal") {
                        Az(j_theta, i_r) = Az_coarse(it->second);
                    } else {
                        Az(i_r, j_theta) = Az_coarse(it->second);
                    }
                }
            } else {
                // Inactive cell - interpolate from surrounding active cells
                double interpolated_value = interpolateFromCoarseGridPolar(i_r, j_theta, Az_coarse);
                if (r_orientation == "horizontal") {
                    Az(j_theta, i_r) = interpolated_value;
                } else {
                    Az(i_r, j_theta) = interpolated_value;
                }
            }
        }
    }
}

void MagneticFieldAnalyzer::interpolateInactiveCells(const Eigen::VectorXd& Az_coarse) {
    // Update ONLY inactive cells by interpolation from active cells
    // Active cells in Az are assumed to already have correct values
    // This is a lightweight function for nonlinear iteration (avoids full grid copy)

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (!active_cells(j, i)) {
                // Inactive cell - interpolate from surrounding active cells
                Az(j, i) = bilinearInterpolateFromCoarse(i, j, Az_coarse);
            }
            // Active cells: keep existing values (already updated from Az_coarse)
        }
    }
}

void MagneticFieldAnalyzer::interpolateInactiveCellsPolar(const Eigen::VectorXd& Az_coarse) {
    // Update ONLY inactive cells by interpolation from active cells (Polar version)
    // Active cells in Az are assumed to already have correct values

    for (int i_r = 0; i_r < nr; i_r++) {
        for (int j_theta = 0; j_theta < ntheta; j_theta++) {
            if (!active_cells(j_theta, i_r)) {  // Note: active_cells is (theta, r)
                // Inactive cell - interpolate from surrounding active cells
                double interpolated_value = interpolateFromCoarseGridPolar(i_r, j_theta, Az_coarse);
                if (r_orientation == "horizontal") {
                    Az(j_theta, i_r) = interpolated_value;
                } else {
                    Az(i_r, j_theta) = interpolated_value;
                }
            }
            // Active cells: keep existing values (already updated from Az_coarse)
        }
    }
}

void MagneticFieldAnalyzer::buildAndSolveSystemCoarsened() {
    std::cout << "\n=== Building coarsened FDM system ===" << std::endl;
    std::cout << "Active cells: " << n_active_cells << " / " << (nx * ny)
              << " (reduction: " << std::fixed << std::setprecision(1)
              << (100.0 * (1.0 - double(n_active_cells) / (nx * ny))) << "%)" << std::endl;

    Eigen::SparseMatrix<double> A(n_active_cells, n_active_cells);
    Eigen::VectorXd rhs(n_active_cells);

    // Build coarsened matrix
    buildMatrixCoarsened(A, rhs);

    // Check matrix symmetry
    Eigen::SparseMatrix<double> A_T = A.transpose();
    double symmetry_error = (A - A_T).norm();
    double A_norm = A.norm();
    double relative_symmetry_error = (A_norm > 1e-12) ? (symmetry_error / A_norm) : symmetry_error;
    std::cout << "Coarsened matrix symmetry check:" << std::endl;
    std::cout << "  ||A - A^T|| = " << symmetry_error << std::endl;
    std::cout << "  ||A|| = " << A_norm << std::endl;
    std::cout << "  Relative error = " << relative_symmetry_error << std::endl;
    if (relative_symmetry_error > 1e-8) {
        std::cerr << "WARNING: Coarsened matrix is not symmetric! Relative error = "
                  << relative_symmetry_error << std::endl;
    } else {
        std::cout << "  Matrix is symmetric (relative error < 1e-8)" << std::endl;
    }

    // Solve using SparseLU
    std::cout << "\n=== Solving coarsened linear system ===" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Coarsened matrix decomposition failed");
    }

    Eigen::VectorXd Az_coarse = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Coarsened linear system solving failed");
    }

    // Interpolate back to full grid
    std::cout << "Interpolating to full grid..." << std::endl;
    interpolateToFullGrid(Az_coarse);

    std::cout << "Coarsened solution complete!" << std::endl;
}

void MagneticFieldAnalyzer::exportCoarseningMask(const std::string& output_dir, int step_number) {
    if (!coarsening_enabled) {
        return;  // No coarsening, nothing to export
    }

    // Create binary mask image: same size as input image, single channel
    // Active cells = 255 (white), Inactive/coarsened cells = 0 (black)
    cv::Mat mask_image(ny, nx, CV_8UC1);

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            mask_image.at<uchar>(j, i) = active_cells(j, i) ? 255 : 0;
        }
    }

    // Flip vertically to match image coordinate system (y=0 at top)
    // active_cells is in analysis coordinates (y=0 at bottom)
    cv::Mat mask_flipped;
    cv::flip(mask_image, mask_flipped, 0);

    // Create CoarseningMask subfolder
    std::string mask_folder = output_dir + "/CoarseningMask";
    createDirectory(mask_folder);

    // Save the mask image with step number
    std::ostringstream oss;
    oss << mask_folder << "/step_" << std::setw(4) << std::setfill('0') << (step_number + 1) << ".png";
    std::string output_path = oss.str();

    if (cv::imwrite(output_path, mask_flipped)) {
        std::cout << "Coarsening mask exported to: " << output_path << std::endl;
    } else {
        std::cerr << "Warning: Failed to export coarsening mask to: " << output_path << std::endl;
    }
}

// ============================================================================
// Phase 4: Full-grid residual evaluation for coarsened Newton-Krylov convergence
// ============================================================================

void MagneticFieldAnalyzer::updateFullMatrixCache() {
    // Rebuild full-grid matrix with current mu_map
    // This is called during nonlinear iteration when mu changes
    if (coordinate_system == "cartesian") {
        buildMatrix(A_full_cached, rhs_full_cached);
    } else {
        buildMatrixPolar(A_full_cached, rhs_full_cached);
    }
    full_matrix_cache_valid = true;
}

double MagneticFieldAnalyzer::computeFullGridResidual(double& out_b_norm) {
    // Ensure full matrix cache is valid
    if (!full_matrix_cache_valid) {
        updateFullMatrixCache();
    }

    // Convert current Az matrix to full vector
    int n_full;
    Eigen::VectorXd Az_full;

    if (coordinate_system == "cartesian") {
        n_full = nx * ny;
        Az_full.resize(n_full);
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                Az_full(j * nx + i) = Az(j, i);
            }
        }
    } else {
        // Polar coordinate system
        n_full = nr * ntheta;
        Az_full.resize(n_full);
        for (int i_r = 0; i_r < nr; i_r++) {
            for (int j_theta = 0; j_theta < ntheta; j_theta++) {
                int idx = i_r * ntheta + j_theta;
                if (r_orientation == "horizontal") {
                    // mu_map/Az shape: (ntheta, nr), indexing: (theta_idx, r_idx)
                    Az_full(idx) = Az(j_theta, i_r);
                } else {
                    // mu_map/Az shape: (nr, ntheta), indexing: (r_idx, theta_idx)
                    Az_full(idx) = Az(i_r, j_theta);
                }
            }
        }
    }

    // Compute residual: r = A_f * Az_f - b_f
    Eigen::VectorXd residual = A_full_cached * Az_full - rhs_full_cached;

    out_b_norm = rhs_full_cached.norm();
    return residual.norm();
}

// ============================================================================
// Phase 4: Prolongation/Restriction operators for Galerkin projection
// ============================================================================

void MagneticFieldAnalyzer::buildInterpolationWeights(
    int i, int j, int fine_idx,
    std::vector<Eigen::Triplet<double>>& triplets)
{
    // Find active neighbors in x and y directions for bilinear interpolation
    // This matches the logic in bilinearInterpolateFromCoarse()

    // Search for active neighbors in x direction
    int i_left = i, i_right = i;
    while (i_left > 0 && !active_cells(j, i_left)) i_left--;
    while (i_right < nx - 1 && !active_cells(j, i_right)) i_right++;

    // Search for active neighbors in y direction
    int j_bottom = j, j_top = j;
    while (j_bottom > 0 && !active_cells(j_bottom, i)) j_bottom--;
    while (j_top < ny - 1 && !active_cells(j_top, i)) j_top++;

    // Check if we have valid active neighbors
    auto it_left = fine_to_coarse.find({i_left, j});
    auto it_right = fine_to_coarse.find({i_right, j});
    auto it_bottom = fine_to_coarse.find({i, j_bottom});
    auto it_top = fine_to_coarse.find({i, j_top});

    bool have_x = (it_left != fine_to_coarse.end() && it_right != fine_to_coarse.end() &&
                   active_cells(j, i_left) && active_cells(j, i_right) && i_left != i_right);
    bool have_y = (it_bottom != fine_to_coarse.end() && it_top != fine_to_coarse.end() &&
                   active_cells(j_bottom, i) && active_cells(j_top, i) && j_bottom != j_top);

    if (have_x && have_y) {
        // Full bilinear interpolation: average of x-interp and y-interp
        double fx = double(i - i_left) / double(i_right - i_left);
        double fy = double(j - j_bottom) / double(j_top - j_bottom);

        // X-direction interpolation weights (scaled by 0.5)
        triplets.push_back({fine_idx, it_left->second, 0.5 * (1.0 - fx)});
        triplets.push_back({fine_idx, it_right->second, 0.5 * fx});
        // Y-direction interpolation weights (scaled by 0.5)
        triplets.push_back({fine_idx, it_bottom->second, 0.5 * (1.0 - fy)});
        triplets.push_back({fine_idx, it_top->second, 0.5 * fy});
    } else if (have_x) {
        // X-direction only linear interpolation
        double fx = double(i - i_left) / double(i_right - i_left);
        triplets.push_back({fine_idx, it_left->second, 1.0 - fx});
        triplets.push_back({fine_idx, it_right->second, fx});
    } else if (have_y) {
        // Y-direction only linear interpolation
        double fy = double(j - j_bottom) / double(j_top - j_bottom);
        triplets.push_back({fine_idx, it_bottom->second, 1.0 - fy});
        triplets.push_back({fine_idx, it_top->second, fy});
    } else {
        // Fallback: find nearest active neighbor
        double min_dist = std::numeric_limits<double>::max();
        int nearest_coarse_idx = -1;

        for (int dj = -2; dj <= 2; dj++) {
            for (int di = -2; di <= 2; di++) {
                int ni = i + di;
                int nj = j + dj;
                if (ni >= 0 && ni < nx && nj >= 0 && nj < ny && active_cells(nj, ni)) {
                    double dist = std::sqrt(di * di + dj * dj);
                    if (dist < min_dist) {
                        auto it = fine_to_coarse.find({ni, nj});
                        if (it != fine_to_coarse.end()) {
                            min_dist = dist;
                            nearest_coarse_idx = it->second;
                        }
                    }
                }
            }
        }

        if (nearest_coarse_idx >= 0) {
            triplets.push_back({fine_idx, nearest_coarse_idx, 1.0});
        }
    }
}

void MagneticFieldAnalyzer::buildProlongationMatrixPolar(
    std::vector<Eigen::Triplet<double>>& triplets)
{
    // Polar version of prolongation matrix construction
    // Index mapping: fine_idx = i_r * ntheta + j_theta

    for (int i_r = 0; i_r < nr; i_r++) {
        for (int j_theta = 0; j_theta < ntheta; j_theta++) {
            int fine_idx = i_r * ntheta + j_theta;

            // Check active_cells with orientation-aware indexing
            bool is_active;
            if (r_orientation == "horizontal") {
                is_active = active_cells(j_theta, i_r);
            } else {
                is_active = active_cells(i_r, j_theta);
            }

            if (is_active) {
                // Active cell: injection (weight = 1.0)
                auto it = fine_to_coarse.find({i_r, j_theta});
                if (it != fine_to_coarse.end()) {
                    triplets.push_back({fine_idx, it->second, 1.0});
                }
            } else {
                // Inactive cell: interpolation from surrounding active cells
                // Find neighbors in r and theta directions
                int i_inner = findNextActiveRadial(i_r, j_theta, -1);
                int i_outer = findNextActiveRadial(i_r, j_theta, +1);
                int j_prev = findNextActiveTheta(i_r, j_theta, -1);
                int j_next = findNextActiveTheta(i_r, j_theta, +1);

                auto it_inner = fine_to_coarse.find({i_inner, j_theta});
                auto it_outer = fine_to_coarse.find({i_outer, j_theta});
                auto it_prev = fine_to_coarse.find({i_r, j_prev});
                auto it_next = fine_to_coarse.find({i_r, j_next});

                bool have_r = (it_inner != fine_to_coarse.end() && it_outer != fine_to_coarse.end() &&
                               i_inner != i_outer);
                bool have_theta = (it_prev != fine_to_coarse.end() && it_next != fine_to_coarse.end() &&
                                   j_prev != j_next);

                if (have_r && have_theta) {
                    // Bilinear in r-theta
                    double fr = (i_outer > i_inner) ? double(i_r - i_inner) / double(i_outer - i_inner) : 0.5;
                    double ft = calculateThetaInterpolationWeight(j_theta, j_prev, j_next);

                    triplets.push_back({fine_idx, it_inner->second, 0.5 * (1.0 - fr)});
                    triplets.push_back({fine_idx, it_outer->second, 0.5 * fr});
                    triplets.push_back({fine_idx, it_prev->second, 0.5 * (1.0 - ft)});
                    triplets.push_back({fine_idx, it_next->second, 0.5 * ft});
                } else if (have_r) {
                    double fr = (i_outer > i_inner) ? double(i_r - i_inner) / double(i_outer - i_inner) : 0.5;
                    triplets.push_back({fine_idx, it_inner->second, 1.0 - fr});
                    triplets.push_back({fine_idx, it_outer->second, fr});
                } else if (have_theta) {
                    double ft = calculateThetaInterpolationWeight(j_theta, j_prev, j_next);
                    triplets.push_back({fine_idx, it_prev->second, 1.0 - ft});
                    triplets.push_back({fine_idx, it_next->second, ft});
                } else {
                    // Fallback: nearest active neighbor
                    double min_dist = std::numeric_limits<double>::max();
                    int nearest_coarse_idx = -1;

                    for (int di = -2; di <= 2; di++) {
                        for (int dj = -2; dj <= 2; dj++) {
                            int ni = i_r + di;
                            int nj = (j_theta + dj + ntheta) % ntheta;
                            if (ni >= 0 && ni < nr) {
                                bool neighbor_active = (r_orientation == "horizontal") ?
                                    active_cells(nj, ni) : active_cells(ni, nj);
                                if (neighbor_active) {
                                    double dist = std::sqrt(di * di + dj * dj);
                                    if (dist < min_dist) {
                                        auto it = fine_to_coarse.find({ni, nj});
                                        if (it != fine_to_coarse.end()) {
                                            min_dist = dist;
                                            nearest_coarse_idx = it->second;
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (nearest_coarse_idx >= 0) {
                        triplets.push_back({fine_idx, nearest_coarse_idx, 1.0});
                    }
                }
            }
        }
    }
}

void MagneticFieldAnalyzer::buildProlongationMatrix() {
    if (multigrid_operators_built) return;

    int n_full = (coordinate_system == "cartesian") ? (nx * ny) : (nr * ntheta);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n_full * 4);  // Max 4 weights per inactive cell

    if (coordinate_system == "cartesian") {
        // Cartesian: fine_idx = j * nx + i
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int fine_idx = j * nx + i;

                if (active_cells(j, i)) {
                    // Active cell: injection (weight = 1.0)
                    auto it = fine_to_coarse.find({i, j});
                    if (it != fine_to_coarse.end()) {
                        triplets.push_back({fine_idx, it->second, 1.0});
                    }
                } else {
                    // Inactive cell: bilinear interpolation weights
                    buildInterpolationWeights(i, j, fine_idx, triplets);
                }
            }
        }
    } else {
        // Polar coordinate system
        buildProlongationMatrixPolar(triplets);
    }

    P_prolongation.resize(n_full, n_active_cells);
    P_prolongation.setFromTriplets(triplets.begin(), triplets.end());

    // R = P^T (Galerkin restriction for symmetric A)
    R_restriction = P_prolongation.transpose();

    multigrid_operators_built = true;

    std::cout << "Prolongation matrix built: " << n_full << " x " << n_active_cells
              << " (nnz = " << P_prolongation.nonZeros() << ")" << std::endl;
}

// ============================================================================
// Phase 4: Galerkin coarse matrix (A_c = R * A_f * P)
// ============================================================================

void MagneticFieldAnalyzer::buildMatrixGalerkin(
    Eigen::SparseMatrix<double>& A_coarse,
    Eigen::VectorXd& rhs_coarse)
{
    // Ensure operators are built
    buildProlongationMatrix();

    // Update full matrix cache (rebuilds with current mu_map)
    updateFullMatrixCache();

    // Galerkin projection: A_c = R * A_f * P = P^T * A_f * P
    Eigen::SparseMatrix<double> AP = A_full_cached * P_prolongation;
    A_coarse = R_restriction * AP;

    // RHS: r_c = R * r_f = P^T * r_f
    rhs_coarse = R_restriction * rhs_full_cached;

    // Mark full matrix cache as valid (was just updated)
    full_matrix_cache_valid = true;
}

// ============================================================================
// Phase 5: Matrix-free Jacobian-vector product for coarsened Newton-Krylov
// ============================================================================

/**
 * @brief Compute B, H, μ fields from an Az vector (without modifying member state)
 *
 * This function computes the magnetic field (B), field intensity (H), and
 * permeability (μ) from a given Az vector. Unlike the member functions
 * calculateMagneticField() and updateMuDistribution(), this function
 * does not modify the class member variables, making it safe for use
 * in matrix-free Jacobian computations.
 *
 * @param Az_full_vec  Input Az vector on full grid (size: nx*ny or nr*ntheta)
 * @param mu_out       Output permeability matrix
 * @param Bx_out       Output Bx (or Br for polar) matrix
 * @param By_out       Output By (or Btheta for polar) matrix
 * @param H_out        Output |H| matrix
 */
void MagneticFieldAnalyzer::computeBHmuFromAzVector(
    const Eigen::VectorXd& Az_full_vec,
    Eigen::MatrixXd& mu_out,
    Eigen::MatrixXd& Bx_out,
    Eigen::MatrixXd& By_out,
    Eigen::MatrixXd& H_out)
{
    const bool is_polar = (coordinate_system != "cartesian");

    // Initialize output matrices
    if (is_polar) {
        if (r_orientation == "horizontal") {
            Bx_out.resize(ntheta, nr);  // Br
            By_out.resize(ntheta, nr);  // Btheta
            H_out.resize(ntheta, nr);
            mu_out.resize(ntheta, nr);
        } else {
            Bx_out.resize(nr, ntheta);
            By_out.resize(nr, ntheta);
            H_out.resize(nr, ntheta);
            mu_out.resize(nr, ntheta);
        }
        Bx_out.setZero();
        By_out.setZero();

        // Polar coordinates: B = curl(Az) in cylindrical
        // Br = (1/r) * ∂Az/∂θ
        // Bθ = -∂Az/∂r
        YAML::Node polar_config = config["polar_domain"] ? config["polar_domain"] : config["polar"];
        double r_start_val = polar_config["r_start"].as<double>();
        double r_end_val = polar_config["r_end"].as<double>();
        double dr_val = (r_end_val - r_start_val) / (nr - 1);

        // Parse theta_range (can be string expression or double)
        double theta_range_val;
        try {
            std::string theta_str = polar_config["theta_range"].as<std::string>();
            te_parser parser;
            theta_range_val = parser.evaluate(theta_str);
        } catch (...) {
            theta_range_val = polar_config["theta_range"].as<double>();
        }
        double dtheta_val = theta_range_val / (ntheta - 1);

        for (int i = 0; i < nr; i++) {
            double r = r_start_val + i * dr_val;
            if (r < 1e-10) r = 1e-10;

            for (int j = 0; j < ntheta; j++) {
                int idx = i * ntheta + j;  // Row-major indexing

                // Get Az values for finite differences
                double Az_center = Az_full_vec(idx);

                // ∂Az/∂r (central difference)
                double dAz_dr = 0.0;
                if (i > 0 && i < nr - 1) {
                    int idx_prev = (i - 1) * ntheta + j;
                    int idx_next = (i + 1) * ntheta + j;
                    dAz_dr = (Az_full_vec(idx_next) - Az_full_vec(idx_prev)) / (2.0 * dr_val);
                } else if (i == 0) {
                    int idx_next = (i + 1) * ntheta + j;
                    dAz_dr = (Az_full_vec(idx_next) - Az_center) / dr_val;
                } else {
                    int idx_prev = (i - 1) * ntheta + j;
                    dAz_dr = (Az_center - Az_full_vec(idx_prev)) / dr_val;
                }

                // ∂Az/∂θ (central difference with periodic BC)
                double dAz_dtheta = 0.0;
                int j_prev = (j > 0) ? j - 1 : ntheta - 2;  // Periodic
                int j_next = (j < ntheta - 1) ? j + 1 : 1;  // Periodic
                int idx_theta_prev = i * ntheta + j_prev;
                int idx_theta_next = i * ntheta + j_next;
                dAz_dtheta = (Az_full_vec(idx_theta_next) - Az_full_vec(idx_theta_prev)) / (2.0 * dtheta_val);

                // B components
                double Br_val = dAz_dtheta / r;
                double Btheta_val = -dAz_dr;

                // Store in appropriate layout
                if (r_orientation == "horizontal") {
                    Bx_out(j, i) = Br_val;
                    By_out(j, i) = Btheta_val;
                } else {
                    Bx_out(i, j) = Br_val;
                    By_out(i, j) = Btheta_val;
                }
            }
        }
    } else {
        // Cartesian coordinates
        Bx_out.resize(ny, nx);
        By_out.resize(ny, nx);
        H_out.resize(ny, nx);
        mu_out.resize(ny, nx);
        Bx_out.setZero();
        By_out.setZero();

        // B = curl(Az)
        // Bx = ∂Az/∂y
        // By = -∂Az/∂x
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;

                // ∂Az/∂y
                double dAz_dy = 0.0;
                if (j > 0 && j < ny - 1) {
                    int idx_prev = (j - 1) * nx + i;
                    int idx_next = (j + 1) * nx + i;
                    dAz_dy = (Az_full_vec(idx_next) - Az_full_vec(idx_prev)) / (2.0 * dy);
                } else if (j == 0) {
                    int idx_next = (j + 1) * nx + i;
                    dAz_dy = (Az_full_vec(idx_next) - Az_full_vec(idx)) / dy;
                } else {
                    int idx_prev = (j - 1) * nx + i;
                    dAz_dy = (Az_full_vec(idx) - Az_full_vec(idx_prev)) / dy;
                }

                // ∂Az/∂x
                double dAz_dx = 0.0;
                if (i > 0 && i < nx - 1) {
                    int idx_prev = j * nx + (i - 1);
                    int idx_next = j * nx + (i + 1);
                    dAz_dx = (Az_full_vec(idx_next) - Az_full_vec(idx_prev)) / (2.0 * dx);
                } else if (i == 0) {
                    int idx_next = j * nx + (i + 1);
                    dAz_dx = (Az_full_vec(idx_next) - Az_full_vec(idx)) / dx;
                } else {
                    int idx_prev = j * nx + (i - 1);
                    dAz_dx = (Az_full_vec(idx) - Az_full_vec(idx_prev)) / dx;
                }

                Bx_out(j, i) = dAz_dy;
                By_out(j, i) = -dAz_dx;
            }
        }
    }

    // Compute |H| and μ from |B|
    const double MU_0 = 4.0 * M_PI * 1e-7;
    cv::Mat image_flipped;
    cv::flip(image, image_flipped, 0);

    int rows = is_polar ? (r_orientation == "horizontal" ? ntheta : nr) : ny;
    int cols = is_polar ? (r_orientation == "horizontal" ? nr : ntheta) : nx;

    // Copy initial mu_map as base (for linear materials)
    mu_out = mu_map;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            double B_mag = std::sqrt(Bx_out(row, col) * Bx_out(row, col) +
                                     By_out(row, col) * By_out(row, col));

            // Get pixel for material lookup
            int img_row = row;
            int img_col = col;
            if (is_polar) {
                int i_r, j_theta;
                if (r_orientation == "horizontal") {
                    i_r = col;
                    j_theta = row;
                } else {
                    i_r = row;
                    j_theta = col;
                }
                polarToImageIndices(i_r, j_theta, img_col, img_row);
            }

            if (img_row < 0 || img_row >= image_flipped.rows ||
                img_col < 0 || img_col >= image_flipped.cols) {
                H_out(row, col) = B_mag / (mu_out(row, col) + 1e-20);
                continue;
            }

            cv::Vec3b pixel = image_flipped.at<cv::Vec3b>(img_row, img_col);
            cv::Scalar rgb(pixel[2], pixel[1], pixel[0]);

            // Find material and evaluate μ(H)
            bool found = false;
            for (const auto& mat : config["materials"]) {
                std::string name = mat.first.as<std::string>();
                YAML::Node props = mat.second;
                if (!props["rgb"]) continue;

                cv::Scalar mat_rgb(
                    props["rgb"][0].as<int>(),
                    props["rgb"][1].as<int>(),
                    props["rgb"][2].as<int>()
                );

                if (rgb[0] == mat_rgb[0] && rgb[1] == mat_rgb[1] && rgb[2] == mat_rgb[2]) {
                    auto bh_it = material_bh_tables.find(name);
                    if (bh_it != material_bh_tables.end() && bh_it->second.is_valid) {
                        // Nonlinear material: compute H from B using inverse B-H table
                        double H_val = interpolateH_from_B(bh_it->second, B_mag);
                        H_out(row, col) = H_val;

                        // Update μ = B/H
                        if (H_val > 1e-10) {
                            mu_out(row, col) = B_mag / H_val;
                        }
                    } else {
                        // Linear material: H = B/μ
                        H_out(row, col) = B_mag / (mu_out(row, col) + 1e-20);
                    }
                    found = true;
                    break;
                }
            }

            if (!found) {
                H_out(row, col) = B_mag / (mu_out(row, col) + 1e-20);
            }
        }
    }
}

/**
 * @brief Assemble full-grid residual vector F(Az) = A(μ)*Az - b
 *
 * This function computes the nonlinear residual on the full grid.
 * The key point is that μ is evaluated from the input Az vector,
 * so the residual properly captures the nonlinear dependency.
 *
 * @param Az_full_vec  Input Az vector on full grid
 * @param mu_full      Permeability matrix corresponding to Az_full_vec
 * @return Residual vector F(Az) on full grid
 */
Eigen::VectorXd MagneticFieldAnalyzer::assembleFullGridResidualVector(
    const Eigen::VectorXd& Az_full_vec,
    const Eigen::MatrixXd& mu_full)
{
    // Temporarily swap mu_map to build matrix with given mu
    Eigen::MatrixXd mu_map_backup = mu_map;
    mu_map = mu_full;

    // Build full matrix with the given mu distribution
    Eigen::SparseMatrix<double> A_full;
    Eigen::VectorXd rhs_full;
    if (coordinate_system == "cartesian") {
        buildMatrix(A_full, rhs_full);
    } else {
        buildMatrixPolar(A_full, rhs_full);
    }

    // Restore original mu_map
    mu_map = mu_map_backup;

    // Compute residual: F(Az) = A(μ)*Az - b
    return A_full * Az_full_vec - rhs_full;
}

/**
 * @brief Matrix-free Jacobian-vector product for coarsened system
 *
 * Computes J_c * v using finite differences:
 *   J_c * v ≈ R * (F_full(P*(x + ε*v)) - F_full(P*x)) / ε
 *
 * This approach correctly captures the μ(Az) dependency that is lost
 * when using explicit Jacobian matrices with interpolated Az values.
 *
 * @param x_c  Current solution on coarse grid (size: n_active_cells)
 * @param v_c  Direction vector on coarse grid (size: n_active_cells)
 * @return J_c * v on coarse grid (size: n_active_cells)
 */
Eigen::VectorXd MagneticFieldAnalyzer::matrixFreeJv(
    const Eigen::VectorXd& x_c,
    const Eigen::VectorXd& v_c)
{
    // Ensure prolongation operator is built
    buildProlongationMatrix();

    // Compute optimal epsilon for finite difference
    double x_norm = x_c.norm();
    double v_norm = v_c.norm();
    double eps = std::sqrt(std::numeric_limits<double>::epsilon()) *
                 (1.0 + x_norm) / (v_norm + 1e-12);

    // Prolongate x_c and (x_c + eps*v_c) to full grid
    Eigen::VectorXd x_c_plus = x_c + eps * v_c;
    Eigen::VectorXd Az_full = P_prolongation * x_c;
    Eigen::VectorXd Az_full_eps = P_prolongation * x_c_plus;

    // Compute B/H/μ for both states
    Eigen::MatrixXd mu_full, Bx_full, By_full, H_full;
    Eigen::MatrixXd mu_full_eps, Bx_full_eps, By_full_eps, H_full_eps;

    computeBHmuFromAzVector(Az_full, mu_full, Bx_full, By_full, H_full);
    computeBHmuFromAzVector(Az_full_eps, mu_full_eps, Bx_full_eps, By_full_eps, H_full_eps);

    // Compute full-grid residuals
    Eigen::VectorXd F_full = assembleFullGridResidualVector(Az_full, mu_full);
    Eigen::VectorXd F_full_eps = assembleFullGridResidualVector(Az_full_eps, mu_full_eps);

    // Finite difference approximation of Jacobian action
    Eigen::VectorXd Jv_full = (F_full_eps - F_full) / eps;

    // Restrict to coarse grid: J_c * v = R * (J_full * P * v) ≈ R * Jv_full
    return R_restriction * Jv_full;
}

/**
 * @brief GMRES solver using matrix-free Jacobian-vector product
 *
 * Solves J*delta = rhs where J is defined implicitly through matrixFreeJv().
 * Uses restarted GMRES algorithm for robustness.
 *
 * @param x_c      Current solution on coarse grid (for Jv computation context)
 * @param rhs      Right-hand side vector (-residual)
 * @param max_iter Maximum number of GMRES iterations
 * @param tol      Convergence tolerance
 * @return Solution vector delta
 */
Eigen::VectorXd MagneticFieldAnalyzer::solveWithMatrixFreeGMRES(
    const Eigen::VectorXd& x_c,
    const Eigen::VectorXd& rhs,
    int max_iter,
    double tol)
{
    const int n = rhs.size();
    const int restart = std::min(nonlinear_config.gmres_restart, max_iter);

    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);  // Initial guess
    double rhs_norm = rhs.norm();
    if (rhs_norm < 1e-14) return x;

    // GMRES with restart
    for (int outer = 0; outer < max_iter / restart + 1; outer++) {
        // Compute initial residual: r = rhs - J*x
        Eigen::VectorXd r = rhs;
        if (x.norm() > 1e-14) {
            r = rhs - matrixFreeJv(x_c, x);
        }

        double beta = r.norm();
        if (beta / rhs_norm < tol) return x;

        // Arnoldi process storage
        Eigen::MatrixXd V(n, restart + 1);  // Orthonormal basis
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(restart + 1, restart);  // Hessenberg matrix
        Eigen::VectorXd g = Eigen::VectorXd::Zero(restart + 1);  // Residual in reduced space
        g(0) = beta;

        V.col(0) = r / beta;

        // Persistent Givens rotation coefficients
        Eigen::VectorXd cs(restart);  // cosines
        Eigen::VectorXd sn(restart);  // sines

        int j_final = restart;
        for (int j = 0; j < restart; j++) {
            // Matrix-free Jv: w = J * V(:,j)
            Eigen::VectorXd w = matrixFreeJv(x_c, V.col(j));

            // Check for NaN/Inf in matrix-free Jv result
            if (!w.allFinite()) {
                std::cerr << "WARNING: GMRES matrixFreeJv returned non-finite values at j=" << j << std::endl;
                j_final = (j > 0) ? j : 1;
                break;
            }

            // Modified Gram-Schmidt orthogonalization
            for (int i = 0; i <= j; i++) {
                H(i, j) = w.dot(V.col(i));
                w -= H(i, j) * V.col(i);
            }
            H(j + 1, j) = w.norm();

            if (H(j + 1, j) < 1e-14) {
                j_final = j + 1;
                break;
            }
            V.col(j + 1) = w / H(j + 1, j);

            // Apply previous Givens rotations to new column of H
            for (int i = 0; i < j; i++) {
                double temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
                H(i, j) = temp;
            }

            // Compute new Givens rotation
            double h_jj = H(j, j);
            double h_jp1j = H(j + 1, j);
            double denom = std::sqrt(h_jj * h_jj + h_jp1j * h_jp1j);
            if (denom < 1e-14) {
                j_final = j + 1;
                break;
            }
            cs(j) = h_jj / denom;
            sn(j) = h_jp1j / denom;

            H(j, j) = denom;
            H(j + 1, j) = 0.0;

            // Apply rotation to g
            double g_j = g(j);
            g(j) = cs(j) * g_j;
            g(j + 1) = -sn(j) * g_j;

            // Check convergence
            double res_norm = std::abs(g(j + 1));
            if (res_norm / rhs_norm < tol) {
                j_final = j + 1;
                break;
            }
        }

        // Solve upper triangular system: H(0:j_final, 0:j_final) * y = g(0:j_final)
        Eigen::VectorXd y = Eigen::VectorXd::Zero(j_final);
        bool solve_ok = true;
        for (int i = j_final - 1; i >= 0; i--) {
            y(i) = g(i);
            for (int k = i + 1; k < j_final; k++) {
                y(i) -= H(i, k) * y(k);
            }
            if (std::abs(H(i, i)) < 1e-14) {
                std::cerr << "WARNING: GMRES singular H matrix at diagonal " << i << std::endl;
                solve_ok = false;
                break;
            }
            y(i) /= H(i, i);
        }

        if (!solve_ok || !y.allFinite()) {
            std::cerr << "WARNING: GMRES solve failed, returning current estimate" << std::endl;
            return x;
        }

        // Update solution: x = x + V(:,0:j_final) * y
        x += V.leftCols(j_final) * y;

        // Check for NaN/Inf in solution
        if (!x.allFinite()) {
            std::cerr << "WARNING: GMRES solution became non-finite" << std::endl;
            return Eigen::VectorXd::Zero(n);
        }

        // Check convergence
        Eigen::VectorXd r_new = rhs - matrixFreeJv(x_c, x);
        if (r_new.allFinite() && r_new.norm() / rhs_norm < tol) {
            return x;
        }
    }

    return x;
}

// ============================================================================
// Phase 6: Preconditioned JFNK
// Uses Galerkin coarse matrix A_c as preconditioner for matrix-free GMRES
// ============================================================================

/**
 * @brief Update the Galerkin preconditioner for JFNK
 *
 * Builds A_c = R * A_f * P (Galerkin projection) and computes its LU factorization.
 * The preconditioner is updated based on precond_update_frequency setting.
 *
 * @param newton_iter Current Newton iteration (0-indexed)
 */
void MagneticFieldAnalyzer::updatePreconditioner(int newton_iter) {
    // Check if we need to update
    bool should_update = false;

    if (!precond_factorization_valid) {
        // First call or invalidated
        should_update = true;
    } else if (nonlinear_config.precond_update_frequency <= 0) {
        // Never update after first (lagged preconditioner)
        should_update = false;
    } else if ((newton_iter - precond_newton_iter) >= nonlinear_config.precond_update_frequency) {
        // Time to update
        should_update = true;
    }

    if (!should_update) {
        return;
    }

    // Ensure prolongation/restriction operators are built
    buildProlongationMatrix();

    // Update full matrix cache with current mu
    updateFullMatrixCache();

    // Compute Galerkin coarse matrix: A_c = R * A_f * P = P^T * A_f * P
    Eigen::SparseMatrix<double> AP = A_full_cached * P_prolongation;
    A_coarse_precond = R_restriction * AP;

    // LU factorization
    precond_solver.compute(A_coarse_precond);

    if (precond_solver.info() != Eigen::Success) {
        std::cerr << "WARNING: Preconditioner LU factorization failed" << std::endl;
        precond_factorization_valid = false;
        return;
    }

    precond_factorization_valid = true;
    precond_newton_iter = newton_iter;

    if (nonlinear_config.precond_verbose) {
        std::cout << "Preconditioner updated at Newton iter " << newton_iter
                  << " (A_c: " << A_coarse_precond.rows() << "x" << A_coarse_precond.cols()
                  << ", nnz=" << A_coarse_precond.nonZeros() << ")" << std::endl;
    }
}

/**
 * @brief Apply preconditioner: compute M^{-1} * v = A_c^{-1} * v
 *
 * Uses the cached LU factorization of A_c.
 *
 * @param v_c Input vector (coarse space, size n_active_cells)
 * @return Preconditioned vector A_c^{-1} * v_c
 */
Eigen::VectorXd MagneticFieldAnalyzer::applyPreconditioner(const Eigen::VectorXd& v_c) {
    if (!precond_factorization_valid) {
        // No preconditioner available, return identity
        return v_c;
    }

    Eigen::VectorXd result = precond_solver.solve(v_c);

    if (!result.allFinite()) {
        std::cerr << "WARNING: Preconditioner solve returned non-finite values" << std::endl;
        return v_c;  // Fall back to identity
    }

    return result;
}

/**
 * @brief Right-preconditioned GMRES for Newton step with matrix-free Jacobian
 *
 * Solves J * delta = rhs using right preconditioning:
 *   J * M^{-1} * y = rhs
 *   delta = M^{-1} * y
 *
 * where M = A_c (Galerkin coarse matrix) is used as preconditioner.
 *
 * This combines:
 * - Matrix-free Jv for correctness (captures μ(Az) dependency)
 * - Galerkin preconditioner for efficiency (reduces GMRES iterations)
 *
 * @param x_c Current solution in coarse space
 * @param rhs Right-hand side (typically -residual)
 * @param max_iter Maximum GMRES iterations
 * @param tol Relative tolerance for convergence
 * @return Newton step delta in coarse space
 */
Eigen::VectorXd MagneticFieldAnalyzer::solveWithPreconditionedGMRES(
    const Eigen::VectorXd& x_c, const Eigen::VectorXd& rhs,
    int max_iter, double tol)
{
    const int n = rhs.size();
    const int restart = std::min(max_iter, nonlinear_config.gmres_restart);

    if (n == 0) return Eigen::VectorXd::Zero(0);

    // Track Jv call count for statistics
    int jv_count = 0;

    // Initial guess x = 0
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

    // Right preconditioning: J * M^{-1} * y = rhs, then delta = M^{-1} * y
    // We store Z columns as preconditioned basis vectors: Z(:,j) = M^{-1} * V(:,j)

    double rhs_norm = rhs.norm();
    if (rhs_norm < 1e-15) {
        if (nonlinear_config.precond_verbose) {
            std::cout << "Preconditioned GMRES: rhs norm near zero, returning zero" << std::endl;
        }
        return x;
    }

    // Outer iterations (restarts)
    for (int outer = 0; outer < max_iter / restart + 1; outer++) {
        // Compute r = rhs - J * x
        // For initial x=0: r = rhs
        // Note: x is already δ = M^{-1} * y (built from Z * y where Z = M^{-1} * V)
        // So we compute J * x directly, NOT J * M^{-1} * x
        Eigen::VectorXd r;
        if (x.norm() < 1e-15) {
            r = rhs;
        } else {
            // x is already δ = M^{-1} * y, so compute J * x directly
            r = rhs - matrixFreeJv(x_c, x);
            jv_count++;
        }

        if (!r.allFinite()) {
            std::cerr << "WARNING: Preconditioned GMRES residual is non-finite" << std::endl;
            return x;
        }

        double beta = r.norm();
        if (beta < tol * rhs_norm) {
            if (nonlinear_config.precond_verbose) {
                std::cout << "Preconditioned GMRES converged: " << jv_count << " Jv calls" << std::endl;
            }
            // x = Z * y is already δ = M^{-1} * y for right preconditioning
            return x;
        }

        // Arnoldi iteration with right preconditioning
        Eigen::MatrixXd V(n, restart + 1);  // Orthonormal basis
        Eigen::MatrixXd Z(n, restart);      // Z(:,j) = M^{-1} * V(:,j)
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(restart + 1, restart);  // Upper Hessenberg

        V.col(0) = r / beta;

        // Givens rotation storage
        Eigen::VectorXd cs(restart);  // Cosines
        Eigen::VectorXd sn(restart);  // Sines
        Eigen::VectorXd g(restart + 1);  // RHS in least squares
        g.setZero();
        g(0) = beta;

        int j_last = 0;
        for (int j = 0; j < restart; j++) {
            j_last = j;

            // Right preconditioning: z_j = M^{-1} * v_j
            Z.col(j) = applyPreconditioner(V.col(j));

            // w = J * z_j = J * M^{-1} * v_j
            Eigen::VectorXd w = matrixFreeJv(x_c, Z.col(j));
            jv_count++;

            if (!w.allFinite()) {
                std::cerr << "WARNING: Preconditioned GMRES Jv returned non-finite at j=" << j << std::endl;
                break;
            }

            // Modified Gram-Schmidt orthogonalization
            for (int i = 0; i <= j; i++) {
                H(i, j) = w.dot(V.col(i));
                w -= H(i, j) * V.col(i);
            }

            H(j + 1, j) = w.norm();

            // Check for breakdown
            if (std::abs(H(j + 1, j)) < 1e-14) {
                if (nonlinear_config.precond_verbose) {
                    std::cout << "Preconditioned GMRES breakdown at j=" << j
                              << " (happy breakdown?)" << std::endl;
                }
                // Solve least squares and return
                break;
            }

            V.col(j + 1) = w / H(j + 1, j);

            // Apply previous Givens rotations to new column
            for (int i = 0; i < j; i++) {
                double temp = cs(i) * H(i, j) + sn(i) * H(i + 1, j);
                H(i + 1, j) = -sn(i) * H(i, j) + cs(i) * H(i + 1, j);
                H(i, j) = temp;
            }

            // Compute new Givens rotation
            double h_jj = H(j, j);
            double h_j1j = H(j + 1, j);
            double denom = std::sqrt(h_jj * h_jj + h_j1j * h_j1j);
            if (denom < 1e-15) denom = 1e-15;

            cs(j) = h_jj / denom;
            sn(j) = h_j1j / denom;

            // Apply to H and g
            H(j, j) = cs(j) * h_jj + sn(j) * h_j1j;
            H(j + 1, j) = 0.0;

            double g_old = g(j);
            g(j) = cs(j) * g_old;
            g(j + 1) = -sn(j) * g_old;

            // Check convergence
            double residual_est = std::abs(g(j + 1));
            if (residual_est < tol * rhs_norm) {
                j_last = j;
                break;
            }
        }

        // Solve upper triangular system H*y = g
        Eigen::VectorXd y(j_last + 1);
        for (int i = j_last; i >= 0; i--) {
            y(i) = g(i);
            for (int k = i + 1; k <= j_last; k++) {
                y(i) -= H(i, k) * y(k);
            }
            if (std::abs(H(i, i)) > 1e-15) {
                y(i) /= H(i, i);
            }
        }

        // Update solution: x = x + Z * y (using preconditioned basis)
        for (int i = 0; i <= j_last; i++) {
            x += y(i) * Z.col(i);
        }

        if (!x.allFinite()) {
            std::cerr << "WARNING: Preconditioned GMRES solution became non-finite" << std::endl;
            return Eigen::VectorXd::Zero(n);
        }

        // Check convergence with true residual
        // x = Z * y is already δ = M^{-1} * y, so compute J * x directly
        Eigen::VectorXd r_true = rhs - matrixFreeJv(x_c, x);
        jv_count++;

        if (r_true.allFinite() && r_true.norm() < tol * rhs_norm) {
            if (nonlinear_config.precond_verbose) {
                std::cout << "Preconditioned GMRES converged: " << jv_count << " Jv calls"
                          << ", ||r||/||b|| = " << r_true.norm() / rhs_norm << std::endl;
            }
            return x;  // x is already the Newton step δ
        }
    }

    // Did not converge - return best approximation
    if (nonlinear_config.precond_verbose) {
        std::cout << "Preconditioned GMRES did not converge after " << jv_count << " Jv calls" << std::endl;
    }
    return x;  // x is already the Newton step δ
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
    // A(i,j) == A(j,i) in coefficient matrix

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
    // Check if nonlinear solver is needed
    if (has_nonlinear_materials && nonlinear_config.enabled) {
        std::cout << "\n=== Nonlinear materials detected ===" << std::endl;
        std::cout << "Using nonlinear solver: " << nonlinear_config.solver_type << std::endl;

        // Choose solver based on configuration
        if (nonlinear_config.solver_type == "newton-krylov") {
            solveNonlinearNewtonKrylov();
        } else if (nonlinear_config.solver_type == "anderson") {
            solveNonlinearWithAnderson();
        } else {
            // Default: Picard iteration
            solveNonlinear();
        }
    } else {
        // Standard linear solver
        if (coordinate_system == "polar") {
            // Use coarsened solver if coarsening is enabled and provides benefit
            if (coarsening_enabled && n_active_cells < nr * ntheta) {
                buildAndSolveSystemPolarCoarsened();
            } else {
                buildAndSolveSystemPolar();
            }
        } else {
            // Use coarsened solver if coarsening is enabled and provides benefit
            if (coarsening_enabled && n_active_cells < nx * ny) {
                buildAndSolveSystemCoarsened();
            } else {
                buildAndSolveSystem();
            }
        }
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

    // Periodic boundary detection messages (commented out to reduce log verbosity)
    // if (x_periodic) {
    //     std::cout << "Periodic boundary detected in X direction (left-right)" << std::endl;
    // }
    // if (y_periodic) {
    //     std::cout << "Periodic boundary detected in Y direction (bottom-top)" << std::endl;
    // }

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

            // Robin boundary conditions: alpha*Az + beta*(dAz/dn) = gamma
            // Left boundary (i=0): outward normal is -x direction
            // dAz/dn = -dAz/dx ≈ (Az(0,j) - Az(1,j))/dx
            // => (alpha + beta/dx)*Az(0,j) - (beta/dx)*Az(1,j) = gamma
            if (is_left && bc_left.type == "robin") {
                double a = bc_left.alpha;
                double b = bc_left.beta;
                double g = bc_left.gamma;
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dx));
                triplets.push_back(Eigen::Triplet<double>(idx, idx + 1, -b/dx));
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
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dx));
                triplets.push_back(Eigen::Triplet<double>(idx, idx - 1, -b/dx));
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
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dy));
                triplets.push_back(Eigen::Triplet<double>(idx, idx + nx, -b/dy));
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
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dy));
                triplets.push_back(Eigen::Triplet<double>(idx, idx - nx, -b/dy));
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

    // Check matrix symmetry
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

void MagneticFieldAnalyzer::exportHToCSV(const std::string& output_path) const {
    if (H_map.size() == 0) {
        std::cerr << "Warning: No H-field data to export (H_map is empty). Skipping H export." << std::endl;
        return;
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    file << std::scientific << std::setprecision(16);

    int rows = H_map.rows();
    int cols = H_map.cols();

    for (int j = 0; j < rows; j++) {
        for (int i = 0; i < cols; i++) {
            file << H_map(j, i);
            if (i < cols - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Magnetic field intensity |H| exported to: " << output_path << std::endl;
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

    // std::cout << "Magnetic field calculated" << std::endl;
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
    bool x_periodic = false;
    bool y_periodic = false;

    if (coordinate_system == "polar") {
        // Polar coordinates: check theta boundary conditions
        // theta is periodic only if both theta_min and theta_max are set to "periodic"
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        if (r_orientation == "horizontal") {
            // r = x (columns), theta = y (rows)
            y_periodic = theta_periodic;
        } else {
            // r = y (rows), theta = x (columns)
            x_periodic = theta_periodic;
        }
    } else {
        // Cartesian coordinates: check boundary conditions
        x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
        y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
    }

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
    bool x_periodic = false;
    bool y_periodic = false;

    if (coordinate_system == "polar") {
        // Polar coordinates: check theta boundary conditions
        // theta is periodic only if both theta_min and theta_max are set to "periodic"
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        if (r_orientation == "horizontal") {
            // r = x (columns), theta = y (rows)
            y_periodic = theta_periodic;
        } else {
            // r = y (rows), theta = x (columns)
            x_periodic = theta_periodic;
        }
    } else {
        // Cartesian coordinates: check boundary conditions
        x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
        y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
    }

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
    bool x_periodic = false;
    bool y_periodic = false;

    if (coordinate_system == "polar") {
        // Polar coordinates: check theta boundary conditions
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        if (r_orientation == "horizontal") {
            // r = x (columns), theta = y (rows)
            y_periodic = theta_periodic;
        } else {
            // r = y (rows), theta = x (columns)
            x_periodic = theta_periodic;
        }
    } else {
        // Cartesian coordinates: check boundary conditions
        x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
        y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
    }

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

            // std::cout << "[DEBUG] Recomputing seam at circular shift boundary (row=0)" << std::endl;
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
                // std::cout << "[DEBUG] Seam recomputed: y_range=[" << y_min << ", " << y_max << ")" << std::endl;
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

            // std::cout << "[DEBUG] Recomputing seam at circular shift boundary (col=0)" << std::endl;
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

// Bilinear interpolation for polar coordinates
// Input: physical coordinates (r_phys, theta_phys), theta_periodic/antiperiodic flags
// Output: interpolated field value
inline double bilinearInterpolatePolar(
    const Eigen::MatrixXd& field,
    double r_phys, double theta_phys,
    double r_start, double dr, double dtheta, double theta_range,
    int nr, int ntheta,
    const std::string& r_orientation,
    bool theta_periodic,
    bool theta_antiperiodic = false) {

    // Convert physical coordinates to continuous grid indices
    double r_idx_cont = (r_phys - r_start) / dr;

    double theta_idx_cont;
    if (theta_periodic) {
        // Periodic boundary: normalize theta to [0, theta_range) and wrap
        double theta_norm = std::fmod(theta_phys, theta_range);
        if (theta_norm < 0.0) theta_norm += theta_range;
        theta_idx_cont = theta_norm / dtheta;
    } else {
        // Non-periodic boundary: clamp theta to valid range
        theta_idx_cont = theta_phys / dtheta;
        theta_idx_cont = std::max(0.0, std::min(theta_idx_cont, static_cast<double>(ntheta - 1)));
    }

    // Clamp r index to valid range [0, nr-2] (need r0+1 to be valid)
    r_idx_cont = std::max(0.0, std::min(r_idx_cont, static_cast<double>(nr - 2) + 0.999));

    // Get integer indices (floor)
    int r0 = static_cast<int>(std::floor(r_idx_cont));
    int t0 = static_cast<int>(std::floor(theta_idx_cont));

    // Clamp to ensure valid indices
    r0 = std::max(0, std::min(r0, nr - 2));
    int r1 = r0 + 1;

    // Theta index handling depends on periodicity
    int t1;
    bool crosses_theta_boundary = false;  // For anti-periodic sign flip
    if (theta_periodic) {
        // Theta wraps periodically
        t0 = t0 % ntheta;
        if (t0 < 0) t0 += ntheta;
        t1 = (t0 + 1) % ntheta;
        // Check if interpolation crosses the theta boundary (t1 wraps to 0)
        crosses_theta_boundary = (t1 < t0);
    } else {
        // Theta clamps at boundaries
        t0 = std::max(0, std::min(t0, ntheta - 2));
        t1 = t0 + 1;
    }

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

    // Apply sign flip for anti-periodic BC when crossing the theta boundary
    if (theta_antiperiodic && crosses_theta_boundary) {
        v01 = -v01;
        v11 = -v11;
    }

    // Bilinear interpolation: first interpolate in r, then in theta
    double v0 = (1.0 - wr) * v00 + wr * v10;  // at t0
    double v1 = (1.0 - wr) * v01 + wr * v11;  // at t1
    return (1.0 - wt) * v0 + wt * v1;
}

// Sample B, μ, and coordinates at a physical point with consistent interpolation
//  B and μ are evaluated at the SAME physical location
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

        // Determine if theta is periodic/anti-periodic
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        bool theta_antiperiodic = theta_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);

        // Interpolate Br, Btheta, mu at (r_phys, theta_phys)
        sample.Br = bilinearInterpolatePolar(Br, sample.r_phys, sample.theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);
        sample.Btheta = bilinearInterpolatePolar(Btheta, sample.r_phys, sample.theta_phys,
                                                r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);
        sample.mu = bilinearInterpolatePolar(mu_map, sample.r_phys, sample.theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);

        // Convert to Cartesian using the SAME theta from atan2
        sample.Bx = sample.Br * std::cos(sample.theta_phys) - sample.Btheta * std::sin(sample.theta_phys);
        sample.By = sample.Br * std::sin(sample.theta_phys) + sample.Btheta * std::cos(sample.theta_phys);
    }

    return sample;
}

// Polar coordinate version: directly use r_phys and theta_phys to avoid atan2 inconsistency
//  exact consistency between boundary point and sample point calculations
MagneticFieldAnalyzer::PolarSample MagneticFieldAnalyzer::sampleFieldsAtPolarPoint(double r_phys, double theta_phys) {
    MagneticFieldAnalyzer::PolarSample sample;

    // Convert to Cartesian for the record
    sample.x_phys = r_phys * std::cos(theta_phys);
    sample.y_phys = r_phys * std::sin(theta_phys);
    sample.r_phys = r_phys;
    sample.theta_phys = theta_phys;

    if (coordinate_system == "polar") {
        // Determine if theta is periodic/anti-periodic
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        bool theta_antiperiodic = theta_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);

        // Interpolate Br, Btheta, mu at (r_phys, theta_phys)
        sample.Br = bilinearInterpolatePolar(Br, r_phys, theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);
        sample.Btheta = bilinearInterpolatePolar(Btheta, r_phys, theta_phys,
                                                r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);
        sample.mu = bilinearInterpolatePolar(mu_map, r_phys, theta_phys,
                                            r_start, dr, dtheta, theta_range, nr, ntheta, r_orientation, theta_periodic, theta_antiperiodic);

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
    //  This offset is used for physical coordinate calculation (x_phys, y_phys)
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
        bool x_periodic_loop = false;
        bool y_periodic_loop = false;

        if (coordinate_system == "polar") {
            // Polar coordinates: check theta boundary conditions
            bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
            if (r_orientation == "horizontal") {
                y_periodic_loop = theta_periodic;  // theta = y (rows)
            } else {
                x_periodic_loop = theta_periodic;  // theta = x (columns)
            }
        } else {
            // Cartesian coordinates
            x_periodic_loop = (bc_left.type == "periodic" && bc_right.type == "periodic");
            y_periodic_loop = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
        }

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
                    // Check which boundaries are periodic (handle polar coordinates)
                    bool x_periodic_fallback = false;
                    bool y_periodic_fallback = false;

                    if (coordinate_system == "polar") {
                        // Polar coordinates: check theta boundary conditions
                        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
                        if (r_orientation == "horizontal") {
                            y_periodic_fallback = theta_periodic;
                        } else {
                            x_periodic_fallback = theta_periodic;
                        }
                    } else {
                        // Cartesian coordinates
                        x_periodic_fallback = (bc_left.type == "periodic" && bc_right.type == "periodic");
                        y_periodic_fallback = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
                    }

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

                // Ensure normal points to background (probe along normal)
                // For periodic boundaries, use wrap instead of clamp
                bool x_periodic = false;
                bool y_periodic = false;

                if (coordinate_system == "polar") {
                    // Polar coordinates: check theta boundary conditions
                    bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
                    if (r_orientation == "horizontal") {
                        // r = x (columns), theta = y (rows)
                        y_periodic = theta_periodic;
                    } else {
                        // r = y (rows), theta = x (columns)
                        x_periodic = theta_periodic;
                    }
                } else {
                    // Cartesian coordinates: check boundary conditions
                    x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
                    y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
                }

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

                // Torque calculation coordinates (rotor frame, independent of theta_offset)
                double x_torque = 0.0, y_torque = 0.0;

                if (coordinate_system == "polar") {
                    if (r_orientation == "horizontal") { ir = i; jt = j; }
                    else { ir = j; jt = i; }
                    ir = std::clamp(ir, 0, nr-1);
                    jt = std::clamp(jt, 0, ntheta-1);
                    r_phys = r_coords[ir];

                    //  For field sampling and export, use stator frame (with theta_offset)
                    // theta_offset accounts for cumulative rotation from image sliding
                    theta = jt * dtheta + theta_offset;
                    x_phys = r_phys * std::cos(theta);
                    y_phys = r_phys * std::sin(theta);

                    // Torque calculation in rotor frame (Galilean invariance)
                    double theta_rotor = jt * dtheta;
                    x_torque = r_phys * std::cos(theta_rotor);
                    y_torque = r_phys * std::sin(theta_rotor);
                } else {
                    // for cartesian: i->x, j->y (since we flipped masks, j increases upward)
                    x_phys = static_cast<double>(i) * dx;
                    y_phys = static_cast<double>(j) * dy;
                    r_phys = std::sqrt(x_phys*x_phys + y_phys*y_phys);
                    theta = (r_phys > 0.0) ? std::atan2(y_phys, x_phys) : 0.0;

                    // For Cartesian, torque coordinates are same as physical coordinates
                    x_torque = x_phys;
                    y_torque = y_phys;
                }

                //  Convert image-space normal to physical-space normal with proper scaling
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

                    // Convert polar normal to Cartesian using rotor frame angle (without theta_offset)
                    double theta_rotor = jt * dtheta;
                    n_phys_x = n_r * std::cos(theta_rotor) - n_theta * std::sin(theta_rotor);
                    n_phys_y = n_r * std::sin(theta_rotor) + n_theta * std::cos(theta_rotor);

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

                // Calculate ds (boundary line element length) based on coordinate system
                // ds is the physical length of the boundary segment that this pixel represents
                // The boundary line is perpendicular to the normal vector
                double ds = 0.0;

                if (coordinate_system == "polar") {
                    // Polar coordinates: decompose normal into (r, θ) components
                    // Use rotor frame angle (without theta_offset) for consistency
                    double theta_rotor = jt * dtheta;
                    double n_r = n_phys_x * std::cos(theta_rotor) + n_phys_y * std::sin(theta_rotor);
                    double n_theta = -n_phys_x * std::sin(theta_rotor) + n_phys_y * std::cos(theta_rotor);

                    // Cell dimensions in physical units
                    double len_r = dr;
                    double len_theta = (r_phys > 0.0) ? (r_phys * dtheta) : 0.0;

                    // ds = length of boundary segment perpendicular to normal
                    // If normal is in r-direction, boundary is in θ-direction → ds = r*dθ
                    // If normal is in θ-direction, boundary is in r-direction → ds = dr
                    // General case: ds² = n_θ² * dr² + n_r² * (r*dθ)²
                    ds = std::sqrt(n_theta * n_theta * len_r * len_r +
                                   n_r * n_r * len_theta * len_theta);
                } else {
                    // Cartesian coordinates: normal is (n_phys_x, n_phys_y)
                    // Boundary line is perpendicular to normal
                    // If normal is in x-direction, boundary is in y-direction → ds = dy
                    // If normal is in y-direction, boundary is in x-direction → ds = dx
                    // General case: ds² = ny² * dx² + nx² * dy²
                    ds = std::sqrt(n_phys_y * n_phys_y * dx * dx +
                                   n_phys_x * n_phys_x * dy * dy);
                }

                if (ds <= 0.0) ds = DS_MIN;

                // Basis vectors in rotor frame (used for radial force calculation)
                double theta_for_basis = (coordinate_system == "polar") ? (jt * dtheta) : theta;
                double er_x = std::cos(theta_for_basis), er_y = std::sin(theta_for_basis);
                double et_x = -std::sin(theta_for_basis), et_y = std::cos(theta_for_basis);

                //  Sample B and μ at SAME physical point using bilinear interpolation
                // Calculate sample point in the SAME coordinate system to avoid atan2 inconsistency
                MagneticFieldAnalyzer::PolarSample sample;

                if (coordinate_system == "polar") {
                    // FIX7: Evaluate Maxwell stress ON the boundary, not offset by dr
                    // Force is exerted by the material itself at its surface
                    // Therefore, evaluate B and H at the boundary position, using material properties
                    double sample_distance = 0.0;  // Changed from dr to 0 for on-boundary evaluation

                    // Decompose normal into polar components
                    double n_r = n_phys_x * std::cos(theta) + n_phys_y * std::sin(theta);
                    double n_theta = -n_phys_x * std::sin(theta) + n_phys_y * std::cos(theta);

                    // Calculate sample point in polar coordinates (physical system with cumulative rotation)
                    double r_sample = r_phys + sample_distance * n_r;
                    double theta_sample_phys = theta + (r_phys > 0.0 ? (sample_distance * n_theta / r_phys) : 0.0);

                    //  Convert physical theta back to image theta for field sampling
                    // Magnetic fields are stored in image coordinate system (without cumulative rotation)
                    // Subtract theta_offset to get image-based theta for correct field lookup
                    double theta_sample_image = theta_sample_phys - theta_offset;

                    // FIX2: Wrap theta to [0, theta_range) for image coordinate system
                    //  Use theta_range (not 2π) to handle sector models correctly
                    while (theta_sample_image < 0.0) theta_sample_image += theta_range;
                    while (theta_sample_image >= theta_range) theta_sample_image -= theta_range;

                    // Sample fields using image coordinates
                    sample = sampleFieldsAtPolarPoint(r_sample, theta_sample_image);

                    // Convert sample coordinates back to physical system for record
                    sample.theta_phys = theta_sample_phys;
                } else {
                    // FIX7: For Cartesian coordinates, also evaluate on boundary (sample_distance = 0)
                    double sample_distance = 0.0;  // Changed from std::max(dx, dy) to 0
                    double x_sample = x_phys + n_phys_x * sample_distance;  // = x_phys
                    double y_sample = y_phys + n_phys_y * sample_distance;  // = y_phys
                    sample = sampleFieldsAtPhysicalPoint(x_sample, y_sample);
                }

                // Extract B and μ from boundary pixel directly (not interpolated)
                // Same reasoning as Fix8: boundary is material/air interface
                // Interpolation mixes material and air values → use material side only
                double bx_out = 0.0, by_out = 0.0;
                double mu_boundary = 0.0;

                if (coordinate_system == "polar") {
                    int ir_boundary, jt_boundary;
                    if (r_orientation == "horizontal") {
                        ir_boundary = i;
                        jt_boundary = j;
                    } else {
                        ir_boundary = j;
                        jt_boundary = i;
                    }
                    ir_boundary = std::clamp(ir_boundary, 0, nr - 1);
                    jt_boundary = std::clamp(jt_boundary, 0, ntheta - 1);

                    // Get Br, Btheta, mu from boundary pixel
                    double br_boundary, bt_boundary;
                    if (r_orientation == "horizontal") {
                        br_boundary = Br(jt_boundary, ir_boundary);
                        bt_boundary = Btheta(jt_boundary, ir_boundary);
                        mu_boundary = mu_map(jt_boundary, ir_boundary);
                    } else {
                        br_boundary = Br(ir_boundary, jt_boundary);
                        bt_boundary = Btheta(ir_boundary, jt_boundary);
                        mu_boundary = mu_map(ir_boundary, jt_boundary);
                    }

                    // CRITICAL BUG FIX: Br, Btheta are in image coordinate system (no theta_offset)
                    // Must use image theta (jt * dtheta) not physical theta (jt * dtheta + theta_offset)
                    double theta_image = jt_boundary * dtheta;
                    bx_out = br_boundary * std::cos(theta_image) - bt_boundary * std::sin(theta_image);
                    by_out = br_boundary * std::sin(theta_image) + bt_boundary * std::cos(theta_image);
                } else {
                    int bi = std::clamp(i, 0, static_cast<int>(Bx.cols()) - 1);
                    int bj = std::clamp(j, 0, static_cast<int>(Bx.rows()) - 1);
                    bx_out = Bx(bj, bi);
                    by_out = By(bj, bi);
                    mu_boundary = mu_map(bj, bi);
                }
                double mu_local = mu_boundary;

                // Maxwell stress calculation: Use Cartesian coordinates for BOTH coordinate systems
                //  Avoid basis vector inconsistencies by always computing in Cartesian
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

                // Torque: τ_z = x * F_y - y * F_x (using rotor frame coordinates)
                result.torque_origin += x_torque * dFy - y_torque * dFx;

                if (coordinate_system == "polar") {
                    // In polar coordinates centered at origin, torque_center = torque_origin
                    result.torque_center += x_torque * dFy - y_torque * dFx;
                } else {
                    // Cartesian coordinates: use offset from center
                    result.torque_center += (x_torque - cx_physical) * dFy - (y_torque - cy_physical) * dFx;
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
        std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;
        std::cout << "    Magnetic energy: " << result.magnetic_energy << " J/m (per unit depth)" << std::endl;

        force_results.push_back(result);
    }

    std::cout << "\nMaxwell stress (polar-aware, Sobel normals) calculation complete!" << std::endl;
}

void MagneticFieldAnalyzer::calculateMaxwellStressEdgeBased(int step) {
    std::cout << "\n=== Calculating Maxwell Stress (Edge-Based Integration) ===" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    const double mu0 = MU_0;

    // Physical center for torque calculation
    double cx_physical = (static_cast<double>(image.cols) * dx) / 2.0;
    double cy_physical = (static_cast<double>(image.rows) * dy) / 2.0;

    // Check boundary conditions for periodic handling
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

    // Clear previous results
    force_results.clear();

    // For each material with calc_force=true
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;
        bool calc_force = props["calc_force"].as<bool>(false);
        if (!calc_force) continue;

        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255,255,255});

        // Build material mask from current image (in FDM coordinates, y=0 at bottom)
        // We need to flip the image to match FDM coordinates
        cv::Mat image_flipped;
        cv::flip(image, image_flipped, 0);

        cv::Mat mat_mask(ny, nx, CV_8U, cv::Scalar(0));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                cv::Vec3b px = image_flipped.at<cv::Vec3b>(j, i);
                if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                    mat_mask.at<uchar>(j, i) = 255;
                }
            }
        }

        ForceResult result;
        result.material_name = name;
        result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
        result.force_x = result.force_y = result.force_radial = 0.0;
        result.torque = result.torque_origin = result.torque_center = 0.0;
        result.pixel_count = 0;
        result.magnetic_energy = 0.0;

        std::cout << "\nCalculating force for material: " << name << " (edge-based)" << std::endl;

        // Diagnostic: track contributions from each edge direction
        double fx_right = 0.0, fx_left = 0.0, fx_top = 0.0, fx_bottom = 0.0;
        double fy_right = 0.0, fy_left = 0.0, fy_top = 0.0, fy_bottom = 0.0;
        int edge_right = 0, edge_left = 0, edge_top = 0, edge_bottom = 0;

        // Edge-based integration:
        // For each material pixel, check 4 neighbors (right, left, up, down)
        // If neighbor is non-material, integrate T·n over that edge
        //
        // Edge directions and normals (outward from material):
        // - Right edge (i+0.5, j): normal = (+1, 0), ds = dy
        // - Left edge (i-0.5, j): normal = (-1, 0), ds = dy
        // - Top edge (i, j+0.5): normal = (0, +1), ds = dx
        // - Bottom edge (i, j-0.5): normal = (0, -1), ds = dx

        int edge_count = 0;

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // Skip non-material pixels
                if (mat_mask.at<uchar>(j, i) == 0) continue;

                // Physical coordinates of cell center
                double x_c = (i + 0.5) * dx;
                double y_c = (j + 0.5) * dy;

                // Get B at this cell (use cell center values)
                double bx_c = Bx(j, i);
                double by_c = By(j, i);
                double mu_c = mu_map(j, i);

                // Check each of 4 neighbors
                // Right neighbor (i+1)
                {
                    int i_n = i + 1;
                    bool is_boundary_edge = false;
                    double mu_neighbor = mu_c;  // Default to same as current cell

                    if (i_n >= nx) {
                        if (x_periodic) {
                            i_n = 0;  // Wrap around
                            if (mat_mask.at<uchar>(j, i_n) == 0) {
                                is_boundary_edge = true;
                                mu_neighbor = mu_map(j, i_n);
                            }
                        } else {
                            // Domain boundary - treat as boundary if non-periodic
                            is_boundary_edge = true;
                            // At domain boundary, assume same μ (no force contribution)
                            mu_neighbor = mu_c;
                        }
                    } else {
                        if (mat_mask.at<uchar>(j, i_n) == 0) {
                            is_boundary_edge = true;
                            mu_neighbor = mu_map(j, i_n);
                        }
                    }

                    if (is_boundary_edge) {
                        // Right edge: normal = (+1, 0), ds = dy
                        // Sample B at edge midpoint (average of cell and neighbor)
                        double bx_edge, by_edge, mu_edge;
                        if (i_n < nx && i_n >= 0) {
                            bx_edge = 0.5 * (bx_c + Bx(j, i_n));
                            by_edge = 0.5 * (by_c + By(j, i_n));
                            mu_edge = mu_c;
                        } else {
                            bx_edge = bx_c;
                            by_edge = by_c;
                            mu_edge = mu_c;
                        }

                        // Maxwell stress tensor: T_ij = B_i*H_j - 0.5*(B·H)*δ_ij
                        double Hx = bx_edge / mu_edge;
                        double Hy = by_edge / mu_edge;
                        double B_dot_H = bx_edge * Hx + by_edge * Hy;

                        double T_xx = bx_edge * Hx - 0.5 * B_dot_H;
                        double T_xy = bx_edge * Hy;

                        // Traction: t = T · n, where n = (+1, 0)
                        double fx = T_xx * 1.0 + T_xy * 0.0;
                        double fy = T_xy * 1.0 + (by_edge * Hy - 0.5 * B_dot_H) * 0.0;

                        // Integrate: F += t * ds, where ds = dy
                        result.force_x += fx * dy;
                        result.force_y += fy * dy;
                        fx_right += fx * dy;
                        fy_right += fy * dy;

                        // Torque about center: τ = r × F
                        double x_edge = (i + 1.0) * dx;
                        double y_edge = y_c;
                        result.torque_center += (x_edge - cx_physical) * (fy * dy) - (y_edge - cy_physical) * (fx * dy);
                        result.torque_origin += x_edge * (fy * dy) - y_edge * (fx * dy);

                        edge_count++;
                        edge_right++;
                    }
                }

                // Left neighbor (i-1)
                {
                    int i_n = i - 1;
                    bool is_boundary_edge = false;
                    double mu_neighbor = mu_c;

                    if (i_n < 0) {
                        if (x_periodic) {
                            i_n = nx - 1;
                            if (mat_mask.at<uchar>(j, i_n) == 0) {
                                is_boundary_edge = true;
                                mu_neighbor = mu_map(j, i_n);
                            }
                        } else {
                            is_boundary_edge = true;
                            mu_neighbor = mu_c;  // Domain boundary
                        }
                    } else {
                        if (mat_mask.at<uchar>(j, i_n) == 0) {
                            is_boundary_edge = true;
                            mu_neighbor = mu_map(j, i_n);
                        }
                    }

                    if (is_boundary_edge) {
                        // Left edge: normal = (-1, 0), ds = dy
                        double bx_edge, by_edge, mu_edge;
                        if (i_n >= 0 && i_n < nx) {
                            bx_edge = 0.5 * (bx_c + Bx(j, i_n));
                            by_edge = 0.5 * (by_c + By(j, i_n));
                            mu_edge = mu_c;
                        } else {
                            bx_edge = bx_c;
                            by_edge = by_c;
                            mu_edge = mu_c;
                        }

                        double Hx = bx_edge / mu_edge;
                        double Hy = by_edge / mu_edge;
                        double B_dot_H = bx_edge * Hx + by_edge * Hy;

                        double T_xx = bx_edge * Hx - 0.5 * B_dot_H;
                        double T_xy = bx_edge * Hy;

                        // n = (-1, 0)
                        double fx = T_xx * (-1.0);
                        double fy = T_xy * (-1.0);

                        result.force_x += fx * dy;
                        result.force_y += fy * dy;
                        fx_left += fx * dy;
                        fy_left += fy * dy;

                        double x_edge = i * dx;
                        double y_edge = y_c;
                        result.torque_center += (x_edge - cx_physical) * (fy * dy) - (y_edge - cy_physical) * (fx * dy);
                        result.torque_origin += x_edge * (fy * dy) - y_edge * (fx * dy);

                        edge_count++;
                        edge_left++;
                    }
                }

                // Top neighbor (j+1)
                {
                    int j_n = j + 1;
                    bool is_boundary_edge = false;
                    double mu_neighbor = mu_c;

                    if (j_n >= ny) {
                        if (y_periodic) {
                            j_n = 0;
                            if (mat_mask.at<uchar>(j_n, i) == 0) {
                                is_boundary_edge = true;
                                mu_neighbor = mu_map(j_n, i);
                            }
                        } else {
                            is_boundary_edge = true;
                            mu_neighbor = mu_c;  // Domain boundary
                        }
                    } else {
                        if (mat_mask.at<uchar>(j_n, i) == 0) {
                            is_boundary_edge = true;
                            mu_neighbor = mu_map(j_n, i);
                        }
                    }

                    if (is_boundary_edge) {
                        // Top edge: normal = (0, +1), ds = dx
                        double bx_edge, by_edge, mu_edge;
                        if (j_n >= 0 && j_n < ny) {
                            bx_edge = 0.5 * (bx_c + Bx(j_n, i));
                            by_edge = 0.5 * (by_c + By(j_n, i));
                            mu_edge = mu_c;
                        } else {
                            bx_edge = bx_c;
                            by_edge = by_c;
                            mu_edge = mu_c;
                        }

                        double Hx = bx_edge / mu_edge;
                        double Hy = by_edge / mu_edge;
                        double B_dot_H = bx_edge * Hx + by_edge * Hy;

                        double T_yy = by_edge * Hy - 0.5 * B_dot_H;
                        double T_xy = bx_edge * Hy;

                        // n = (0, +1)
                        double fx = T_xy * 1.0;
                        double fy = T_yy * 1.0;

                        result.force_x += fx * dx;
                        result.force_y += fy * dx;
                        fx_top += fx * dx;
                        fy_top += fy * dx;

                        double x_edge = x_c;
                        double y_edge = (j + 1.0) * dy;
                        result.torque_center += (x_edge - cx_physical) * (fy * dx) - (y_edge - cy_physical) * (fx * dx);
                        result.torque_origin += x_edge * (fy * dx) - y_edge * (fx * dx);

                        edge_count++;
                        edge_top++;
                    }
                }

                // Bottom neighbor (j-1)
                {
                    int j_n = j - 1;
                    bool is_boundary_edge = false;
                    double mu_neighbor = mu_c;

                    if (j_n < 0) {
                        if (y_periodic) {
                            j_n = ny - 1;
                            if (mat_mask.at<uchar>(j_n, i) == 0) {
                                is_boundary_edge = true;
                                mu_neighbor = mu_map(j_n, i);
                            }
                        } else {
                            is_boundary_edge = true;
                            mu_neighbor = mu_c;  // Domain boundary
                        }
                    } else {
                        if (mat_mask.at<uchar>(j_n, i) == 0) {
                            is_boundary_edge = true;
                            mu_neighbor = mu_map(j_n, i);
                        }
                    }

                    if (is_boundary_edge) {
                        // Bottom edge: normal = (0, -1), ds = dx
                        double bx_edge, by_edge, mu_edge;
                        if (j_n >= 0 && j_n < ny) {
                            bx_edge = 0.5 * (bx_c + Bx(j_n, i));
                            by_edge = 0.5 * (by_c + By(j_n, i));
                            mu_edge = mu_c;
                        } else {
                            bx_edge = bx_c;
                            by_edge = by_c;
                            mu_edge = mu_c;
                        }

                        double Hx = bx_edge / mu_edge;
                        double Hy = by_edge / mu_edge;
                        double B_dot_H = bx_edge * Hx + by_edge * Hy;

                        double T_yy = by_edge * Hy - 0.5 * B_dot_H;
                        double T_xy = bx_edge * Hy;

                        // n = (0, -1)
                        double fx = T_xy * (-1.0);
                        double fy = T_yy * (-1.0);

                        result.force_x += fx * dx;
                        result.force_y += fy * dx;
                        fx_bottom += fx * dx;
                        fy_bottom += fy * dx;

                        double x_edge = x_c;
                        double y_edge = j * dy;
                        result.torque_center += (x_edge - cx_physical) * (fy * dx) - (y_edge - cy_physical) * (fx * dx);
                        result.torque_origin += x_edge * (fy * dx) - y_edge * (fx * dx);

                        edge_count++;
                        edge_bottom++;
                    }
                }

                result.pixel_count++;  // Count material pixels
            }
        }

        // Calculate magnetic energy for this material (same as before)
        double energy = 0.0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (mat_mask.at<uchar>(j, i) != 0) {
                    double bx = Bx(j, i);
                    double by = By(j, i);
                    double mu = mu_map(j, i);
                    double B_sq = bx * bx + by * by;
                    energy += 0.5 * B_sq / mu * dx * dy;
                }
            }
        }
        result.magnetic_energy = energy;

        // Set force_radial as magnitude for Cartesian
        result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
        result.torque = result.torque_origin;

        std::cout << "  Material: " << name << std::endl;
        std::cout << "    Material pixels: " << result.pixel_count << std::endl;
        std::cout << "    Boundary edges: " << edge_count << std::endl;
        std::cout << "      Right edges: " << edge_right << ", Left edges: " << edge_left << std::endl;
        std::cout << "      Top edges: " << edge_top << ", Bottom edges: " << edge_bottom << std::endl;
        std::cout << "    --- Edge contribution breakdown ---" << std::endl;
        std::cout << "      Fx_right: " << fx_right << ", Fx_left: " << fx_left << " => sum: " << (fx_right + fx_left) << std::endl;
        std::cout << "      Fx_top: " << fx_top << ", Fx_bottom: " << fx_bottom << " => sum: " << (fx_top + fx_bottom) << std::endl;
        std::cout << "      Fy_right: " << fy_right << ", Fy_left: " << fy_left << " => sum: " << (fy_right + fy_left) << std::endl;
        std::cout << "      Fy_top: " << fy_top << ", Fy_bottom: " << fy_bottom << " => sum: " << (fy_top + fy_bottom) << std::endl;
        std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
        std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
        std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;
        std::cout << "    Magnetic energy: " << result.magnetic_energy << " J/m (per unit depth)" << std::endl;

        force_results.push_back(result);
    }

    std::cout << "\nMaxwell stress (edge-based) calculation complete!" << std::endl;
}

void MagneticFieldAnalyzer::calculateForceVolumeIntegral(int step) {
    std::cout << "\n=== Calculating Force using Volume Integral Method ===" << std::endl;
    std::cout << "  Force density: f = J×B + (M·∇)B" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    const double mu0 = MU_0;

    // Physical center for torque calculation
    double cx_physical = (static_cast<double>(nx) * dx) / 2.0;
    double cy_physical = (static_cast<double>(ny) * dy) / 2.0;

    // Clear previous volume integral results
    force_results_volume.clear();

    // For each material with calc_force=true
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;
        bool calc_force = props["calc_force"].as<bool>(false);
        if (!calc_force) continue;

        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Build material mask from current image (in FDM coordinates, y=0 at bottom)
        cv::Mat image_flipped;
        cv::flip(image, image_flipped, 0);

        cv::Mat mat_mask(ny, nx, CV_8U, cv::Scalar(0));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                cv::Vec3b px = image_flipped.at<cv::Vec3b>(j, i);
                if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                    mat_mask.at<uchar>(j, i) = 255;
                }
            }
        }

        ForceResult result;
        result.material_name = name;
        result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
        result.force_x = result.force_y = result.force_radial = 0.0;
        result.torque = result.torque_origin = result.torque_center = 0.0;
        result.pixel_count = 0;
        result.magnetic_energy = 0.0;

        std::cout << "\nCalculating force for material: " << name << " (volume integral)" << std::endl;

        // Diagnostic counters
        double fx_lorentz = 0.0, fy_lorentz = 0.0;  // J×B contribution
        double fx_magnetization = 0.0, fy_magnetization = 0.0;  // (M·∇)B contribution
        int pixels_with_current = 0;
        int pixels_with_magnetization = 0;

        if (coordinate_system == "cartesian") {
            // Cartesian coordinate system implementation
            for (int j = 1; j < ny - 1; j++) {  // Skip boundary for central difference
                for (int i = 1; i < nx - 1; i++) {
                    // Physical coordinates of cell center
                    double x_c = (i + 0.5) * dx;
                    double y_c = (j + 0.5) * dy;

                    // Get B and μ at this cell
                    double bx = Bx(j, i);
                    double by = By(j, i);
                    double mu = mu_map(j, i);
                    double mu_r = mu / mu0;

                    // Get current density (if any)
                    double jz = jz_map(j, i);

                    // Calculate H = B/μ
                    double Hx = bx / mu;
                    double Hy = by / mu;

                    // Calculate M = (μr - 1) * H
                    double Mx = (mu_r - 1.0) * Hx;
                    double My = (mu_r - 1.0) * Hy;

                    // Calculate ∂B/∂x, ∂B/∂y using central difference
                    double dBx_dx = (Bx(j, i + 1) - Bx(j, i - 1)) / (2.0 * dx);
                    double dBx_dy = (Bx(j + 1, i) - Bx(j - 1, i)) / (2.0 * dy);
                    double dBy_dx = (By(j, i + 1) - By(j, i - 1)) / (2.0 * dx);
                    double dBy_dy = (By(j + 1, i) - By(j - 1, i)) / (2.0 * dy);

                    // Force density: f = J×B + (M·∇)B
                    // J×B: (0, 0, Jz) × (Bx, By, 0) = (-Jz*By, Jz*Bx, 0)
                    // Cross product: (J×B)_x = Jy*Bz - Jz*By = -Jz*By
                    //                (J×B)_y = Jz*Bx - Jx*Bz = Jz*Bx
                    double fx_JxB = -jz * by;
                    double fy_JxB = jz * bx;

                    // (M·∇)B: Mx*∂B/∂x + My*∂B/∂y
                    double fx_MgradB = Mx * dBx_dx + My * dBx_dy;
                    double fy_MgradB = Mx * dBy_dx + My * dBy_dy;

                    // Total force density
                    double fx = fx_JxB + fx_MgradB;
                    double fy = fy_JxB + fy_MgradB;

                    // Volume element (per unit depth)
                    double dA = dx * dy;

                    // Check if this pixel belongs to the target material
                    bool is_material = (mat_mask.at<uchar>(j, i) != 0);
                    if (!is_material) {
                        continue;  // Only integrate within material region
                    }

                    result.pixel_count++;

                    // Accumulate forces within material region only
                    result.force_x += fx * dA;
                    result.force_y += fy * dA;

                    // Diagnostic accumulation
                    if (std::abs(jz) > 1e-10) {
                        fx_lorentz += fx_JxB * dA;
                        fy_lorentz += fy_JxB * dA;
                        pixels_with_current++;
                    }
                    if (std::abs(mu_r - 1.0) > 1e-6) {
                        fx_magnetization += fx_MgradB * dA;
                        fy_magnetization += fy_MgradB * dA;
                        pixels_with_magnetization++;
                    }

                    // Torque about origin: τz = x*fy - y*fx
                    result.torque_origin += x_c * (fy * dA) - y_c * (fx * dA);

                    // Torque about image center
                    result.torque_center += (x_c - cx_physical) * (fy * dA) - (y_c - cy_physical) * (fx * dA);
                }
            }
        } else {
            // Polar coordinate system implementation
            for (int jt = 1; jt < ntheta - 1; jt++) {  // Skip boundary for central difference
                for (int ir = 1; ir < nr - 1; ir++) {
                    // Determine array indices based on r_orientation
                    int i, j;
                    if (r_orientation == "horizontal") {
                        i = ir;  // r along columns (x)
                        j = jt;  // theta along rows (y)
                    } else {
                        i = jt;  // theta along columns (x)
                        j = ir;  // r along rows (y)
                    }

                    // Physical coordinates
                    double r = r_coords[ir];
                    double theta = jt * dtheta;
                    double x_c = r * std::cos(theta);
                    double y_c = r * std::sin(theta);

                    // Get B and μ at this cell
                    double br, bt, mu_val;
                    if (r_orientation == "horizontal") {
                        br = Br(jt, ir);
                        bt = Btheta(jt, ir);
                        mu_val = mu_map(jt, ir);
                    } else {
                        br = Br(ir, jt);
                        bt = Btheta(ir, jt);
                        mu_val = mu_map(ir, jt);
                    }
                    double mu_r = mu_val / mu0;

                    // Convert to Cartesian for force calculation
                    double bx = br * std::cos(theta) - bt * std::sin(theta);
                    double by = br * std::sin(theta) + bt * std::cos(theta);

                    // Get current density
                    double jz;
                    if (r_orientation == "horizontal") {
                        jz = jz_map(jt, ir);
                    } else {
                        jz = jz_map(ir, jt);
                    }

                    // Calculate H = B/μ (in Cartesian)
                    double Hx = bx / mu_val;
                    double Hy = by / mu_val;

                    // Calculate M = (μr - 1) * H
                    double Mx = (mu_r - 1.0) * Hx;
                    double My = (mu_r - 1.0) * Hy;

                    // Calculate ∂B/∂x, ∂B/∂y using central difference in physical coordinates
                    // For polar coordinates, use chain rule: ∂/∂x = cos(θ)∂/∂r - sin(θ)/(r)∂/∂θ
                    //                                        ∂/∂y = sin(θ)∂/∂r + cos(θ)/(r)∂/∂θ
                    double dBr_dr, dBr_dtheta, dBt_dr, dBt_dtheta;
                    if (r_orientation == "horizontal") {
                        dBr_dr = (Br(jt, ir + 1) - Br(jt, ir - 1)) / (2.0 * dr);
                        dBr_dtheta = (Br(jt + 1, ir) - Br(jt - 1, ir)) / (2.0 * dtheta);
                        dBt_dr = (Btheta(jt, ir + 1) - Btheta(jt, ir - 1)) / (2.0 * dr);
                        dBt_dtheta = (Btheta(jt + 1, ir) - Btheta(jt - 1, ir)) / (2.0 * dtheta);
                    } else {
                        dBr_dr = (Br(ir + 1, jt) - Br(ir - 1, jt)) / (2.0 * dr);
                        dBr_dtheta = (Br(ir, jt + 1) - Br(ir, jt - 1)) / (2.0 * dtheta);
                        dBt_dr = (Btheta(ir + 1, jt) - Btheta(ir - 1, jt)) / (2.0 * dr);
                        dBt_dtheta = (Btheta(ir, jt + 1) - Btheta(ir, jt - 1)) / (2.0 * dtheta);
                    }

                    // Transform derivatives to Cartesian coordinates
                    // Bx = Br*cos(θ) - Bt*sin(θ)
                    // By = Br*sin(θ) + Bt*cos(θ)
                    double cos_t = std::cos(theta);
                    double sin_t = std::sin(theta);

                    // ∂Bx/∂r, ∂Bx/∂θ, ∂By/∂r, ∂By/∂θ
                    double dBx_dr = dBr_dr * cos_t - dBt_dr * sin_t;
                    double dBx_dtheta = dBr_dtheta * cos_t - br * sin_t - dBt_dtheta * sin_t - bt * cos_t;
                    double dBy_dr = dBr_dr * sin_t + dBt_dr * cos_t;
                    double dBy_dtheta = dBr_dtheta * sin_t + br * cos_t + dBt_dtheta * cos_t - bt * sin_t;

                    // Convert to ∂/∂x, ∂/∂y
                    double r_safe = std::max(r, 1e-10);
                    double dBx_dx = cos_t * dBx_dr - sin_t / r_safe * dBx_dtheta;
                    double dBx_dy = sin_t * dBx_dr + cos_t / r_safe * dBx_dtheta;
                    double dBy_dx = cos_t * dBy_dr - sin_t / r_safe * dBy_dtheta;
                    double dBy_dy = sin_t * dBy_dr + cos_t / r_safe * dBy_dtheta;

                    // Force density: f = J×B + (M·∇)B
                    // J×B: (0, 0, Jz) × (Bx, By, 0) = (-Jz*By, Jz*Bx, 0)
                    double fx_JxB = -jz * by;
                    double fy_JxB = jz * bx;
                    double fx_MgradB = Mx * dBx_dx + My * dBx_dy;
                    double fy_MgradB = Mx * dBy_dx + My * dBy_dy;

                    double fx = fx_JxB + fx_MgradB;
                    double fy = fy_JxB + fy_MgradB;

                    // Volume element in polar coordinates (per unit depth): r * dr * dθ
                    double dA = r * dr * dtheta;

                    // Check if this pixel belongs to the target material
                    bool is_material = (mat_mask.at<uchar>(j, i) != 0);
                    if (!is_material) {
                        continue;  // Only integrate within material region
                    }

                    result.pixel_count++;

                    // Accumulate forces within material region only
                    result.force_x += fx * dA;
                    result.force_y += fy * dA;

                    // Diagnostic accumulation
                    if (std::abs(jz) > 1e-10) {
                        fx_lorentz += fx_JxB * dA;
                        fy_lorentz += fy_JxB * dA;
                        pixels_with_current++;
                    }
                    if (std::abs(mu_r - 1.0) > 1e-6) {
                        fx_magnetization += fx_MgradB * dA;
                        fy_magnetization += fy_MgradB * dA;
                        pixels_with_magnetization++;
                    }

                    // Torque about origin: τz = x*fy - y*fx
                    result.torque_origin += x_c * (fy * dA) - y_c * (fx * dA);
                    result.torque_center += x_c * (fy * dA) - y_c * (fx * dA);  // Same for polar (center at origin)
                }
            }
        }

        // NOTE: This volume integral computes f = J×B + (M·∇)B
        // This is INCOMPLETE - missing surface bound current term K_b = M×n
        //
        // Full Amperian expression:
        //   F = ∫_V (J_f + ∇×M)×B dV + ∮_S (K_b×B) dS  where K_b = M×n
        //
        // Or from Maxwell stress divergence:
        //   ∇·T = J_f×B + (M·∇)B - (1/2)∇(M·B)
        //   The last term becomes a surface integral: -(1/2)∮_S (M·B)n dS
        //
        // Current implementation provides VOLUME CONTRIBUTION ONLY.
        // For accurate total force, also need:
        // (A) Surface bound current integral (Amperian) - to be implemented
        // (B) Or use Maxwell stress surface integral (existing EdgeBased method)

        // Calculate force magnitude
        result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
        result.torque = result.torque_origin;

        // Output results
        std::cout << "  Material: " << name << std::endl;
        std::cout << "    Material pixels: " << result.pixel_count << std::endl;
        std::cout << "    --- Contribution breakdown ---" << std::endl;
        std::cout << "      Lorentz (J×B): Fx=" << fx_lorentz << ", Fy=" << fy_lorentz
                  << " (" << pixels_with_current << " pixels with current)" << std::endl;
        std::cout << "      Magnetization ((M·∇)B): Fx=" << fx_magnetization << ", Fy=" << fy_magnetization
                  << " (" << pixels_with_magnetization << " pixels with μr≠1)" << std::endl;
        std::cout << "    --- Total (volume contribution only, surface term not included) ---" << std::endl;
        std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
        std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
        std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;

        force_results_volume.push_back(result);
    }

    std::cout << "\nVolume integral force calculation complete!" << std::endl;
}


void MagneticFieldAnalyzer::calculateForceMaxwellStressFaceFlux(int step) {
    std::cout << "\n=== Calculating Force using Face-Flux Method (Maxwell Stress Divergence) ===" << std::endl;
    std::cout << "  Method: F = ∫_V ∇·T dV = Σ_faces (T·n) * A_face" << std::endl;
    std::cout << "  Where T = B⊗H - (1/2)(B·H)I" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    const double mu0 = MU_0;

    // Physical center for torque calculation
    double cx_physical = (static_cast<double>(nx) * dx) / 2.0;
    double cy_physical = (static_cast<double>(ny) * dy) / 2.0;

    // Clear previous flux results
    force_results_flux.clear();

    // For each material with calc_force=true
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;
        bool calc_force = props["calc_force"].as<bool>(false);
        if (!calc_force) continue;

        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Build material mask from current image (in FDM coordinates, y=0 at bottom)
        cv::Mat image_flipped;
        cv::flip(image, image_flipped, 0);

        cv::Mat mat_mask(ny, nx, CV_8U, cv::Scalar(0));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                cv::Vec3b px = image_flipped.at<cv::Vec3b>(j, i);
                if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                    mat_mask.at<uchar>(j, i) = 255;
                }
            }
        }

        ForceResult result;
        result.material_name = name;
        result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
        result.force_x = result.force_y = result.force_radial = 0.0;
        result.torque = result.torque_origin = result.torque_center = 0.0;
        result.pixel_count = 0;
        result.magnetic_energy = 0.0;

        std::cout << "\nCalculating force for material: " << name << " (face-flux method)" << std::endl;

        if (coordinate_system == "cartesian") {
            // Cartesian coordinate system implementation
            // Face-flux method: F_cell = Σ_faces (T·n) * A_face
            //
            // For each cell (i,j), we have 4 faces:
            //   East  (i+1/2, j): n = (+1, 0), area = dy
            //   West  (i-1/2, j): n = (-1, 0), area = dy
            //   North (i, j+1/2): n = (0, +1), area = dx
            //   South (i, j-1/2): n = (0, -1), area = dx
            //
            // Maxwell stress tensor T = B⊗H - (1/2)(B·H)I:
            //   T_xx = Bx*Hx - (1/2)(B·H)
            //   T_xy = Bx*Hy
            //   T_yx = By*Hx
            //   T_yy = By*Hy - (1/2)(B·H)
            //
            // Face fluxes:
            //   East:  T·n = (T_xx, T_yx)
            //   North: T·n = (T_xy, T_yy)

            // Loop over ALL cells including boundary cells
            // For domain boundary faces, use one-sided extrapolation (cell's own values)
            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    // Check if this cell belongs to the target material
                    if (mat_mask.at<uchar>(j, i) == 0) {
                        continue;
                    }

                    result.pixel_count++;

                    // Physical coordinates of cell center
                    double x_c = (i + 0.5) * dx;
                    double y_c = (j + 0.5) * dy;

                    // Compute H at cell center (H = B/μ)
                    double Bx_c = Bx(j, i);
                    double By_c = By(j, i);
                    double Hx_c = Bx_c / mu_map(j, i);
                    double Hy_c = By_c / mu_map(j, i);

                    // --- East face (i+1/2, j) ---
                    double Bx_e, By_e, Hx_e, Hy_e;
                    if (i + 1 < nx) {
                        // Neighbor exists: interpolate
                        double Hx_c_e = Bx(j, i + 1) / mu_map(j, i + 1);
                        double Hy_c_e = By(j, i + 1) / mu_map(j, i + 1);
                        Hx_e = 0.5 * (Hx_c + Hx_c_e);
                        Hy_e = 0.5 * (Hy_c + Hy_c_e);
                        Bx_e = 0.5 * (Bx_c + Bx(j, i + 1));
                        By_e = 0.5 * (By_c + By(j, i + 1));
                    } else {
                        // Domain boundary: use cell's own values
                        Bx_e = Bx_c; By_e = By_c;
                        Hx_e = Hx_c; Hy_e = Hy_c;
                    }
                    double BdotH_e = Bx_e * Hx_e + By_e * Hy_e;
                    double T_xx_e = Bx_e * Hx_e - 0.5 * BdotH_e;
                    double T_yx_e = By_e * Hx_e;

                    // --- West face (i-1/2, j) ---
                    double Bx_w, By_w, Hx_w, Hy_w;
                    if (i - 1 >= 0) {
                        double Hx_c_w = Bx(j, i - 1) / mu_map(j, i - 1);
                        double Hy_c_w = By(j, i - 1) / mu_map(j, i - 1);
                        Hx_w = 0.5 * (Hx_c + Hx_c_w);
                        Hy_w = 0.5 * (Hy_c + Hy_c_w);
                        Bx_w = 0.5 * (Bx_c + Bx(j, i - 1));
                        By_w = 0.5 * (By_c + By(j, i - 1));
                    } else {
                        Bx_w = Bx_c; By_w = By_c;
                        Hx_w = Hx_c; Hy_w = Hy_c;
                    }
                    double BdotH_w = Bx_w * Hx_w + By_w * Hy_w;
                    double T_xx_w = Bx_w * Hx_w - 0.5 * BdotH_w;
                    double T_yx_w = By_w * Hx_w;

                    // --- North face (i, j+1/2) ---
                    double Bx_n, By_n, Hx_n, Hy_n;
                    if (j + 1 < ny) {
                        double Hx_c_n = Bx(j + 1, i) / mu_map(j + 1, i);
                        double Hy_c_n = By(j + 1, i) / mu_map(j + 1, i);
                        Hx_n = 0.5 * (Hx_c + Hx_c_n);
                        Hy_n = 0.5 * (Hy_c + Hy_c_n);
                        Bx_n = 0.5 * (Bx_c + Bx(j + 1, i));
                        By_n = 0.5 * (By_c + By(j + 1, i));
                    } else {
                        Bx_n = Bx_c; By_n = By_c;
                        Hx_n = Hx_c; Hy_n = Hy_c;
                    }
                    double BdotH_n = Bx_n * Hx_n + By_n * Hy_n;
                    double T_xy_n = Bx_n * Hy_n;
                    double T_yy_n = By_n * Hy_n - 0.5 * BdotH_n;

                    // --- South face (i, j-1/2) ---
                    double Bx_s, By_s, Hx_s, Hy_s;
                    if (j - 1 >= 0) {
                        double Hx_c_s = Bx(j - 1, i) / mu_map(j - 1, i);
                        double Hy_c_s = By(j - 1, i) / mu_map(j - 1, i);
                        Hx_s = 0.5 * (Hx_c + Hx_c_s);
                        Hy_s = 0.5 * (Hy_c + Hy_c_s);
                        Bx_s = 0.5 * (Bx_c + Bx(j - 1, i));
                        By_s = 0.5 * (By_c + By(j - 1, i));
                    } else {
                        Bx_s = Bx_c; By_s = By_c;
                        Hx_s = Hx_c; Hy_s = Hy_c;
                    }
                    double BdotH_s = Bx_s * Hx_s + By_s * Hy_s;
                    double T_xy_s = Bx_s * Hy_s;
                    double T_yy_s = By_s * Hy_s - 0.5 * BdotH_s;

                    // --- Discrete divergence: F_cell = Σ_faces (T·n) * A_face ---
                    // Fx = (T_xx_e - T_xx_w) * dy + (T_xy_n - T_xy_s) * dx
                    // Fy = (T_yx_e - T_yx_w) * dy + (T_yy_n - T_yy_s) * dx
                    double fx = (T_xx_e - T_xx_w) * dy + (T_xy_n - T_xy_s) * dx;
                    double fy = (T_yx_e - T_yx_w) * dy + (T_yy_n - T_yy_s) * dx;

                    // Accumulate forces
                    result.force_x += fx;
                    result.force_y += fy;

                    // Torque about origin: τz = x*fy - y*fx
                    result.torque_origin += x_c * fy - y_c * fx;

                    // Torque about image center
                    result.torque_center += (x_c - cx_physical) * fy - (y_c - cy_physical) * fx;
                }
            }
        } else {
            // Polar coordinate system implementation
            // Face-flux in polar: r-faces and θ-faces
            //
            // For cell (ir, jt):
            //   Outer r-face (ir+1/2, jt): n_r = +1, area = r_outer * dθ
            //   Inner r-face (ir-1/2, jt): n_r = -1, area = r_inner * dθ
            //   θ+ face (ir, jt+1/2): n_θ = +1, area = dr
            //   θ- face (ir, jt-1/2): n_θ = -1, area = dr
            //
            // IMPORTANT: Compute H = B/μ at cell centers first, then interpolate to faces
            // This ensures consistency: avg(H) is used, not avg(B)/avg(μ)
            //
            // We work in Cartesian (x,y) for force accumulation

            for (int jt = 1; jt < ntheta - 1; jt++) {
                for (int ir = 1; ir < nr - 1; ir++) {
                    // Determine array indices based on r_orientation
                    int i, j;
                    if (r_orientation == "horizontal") {
                        i = ir;
                        j = jt;
                    } else {
                        i = jt;
                        j = ir;
                    }

                    // Check if this cell belongs to the target material
                    if (mat_mask.at<uchar>(j, i) == 0) {
                        continue;
                    }

                    result.pixel_count++;

                    // Physical coordinates of cell center
                    double r = r_coords[ir];
                    double theta = jt * dtheta;
                    double x_c = r * std::cos(theta);
                    double y_c = r * std::sin(theta);
                    double cos_t = std::cos(theta);
                    double sin_t = std::sin(theta);

                    // ============================================================
                    // Step 1: Compute H at all relevant cell centers in Cartesian
                    // H = B/μ at each cell, then convert to Cartesian coordinates
                    // ============================================================

                    // Helper lambda to get B and H in Cartesian at a given (ir_idx, jt_idx)
                    auto getBH_cartesian = [&](int ir_idx, int jt_idx, double theta_val) {
                        double br_val, bt_val, mu_val;
                        if (r_orientation == "horizontal") {
                            br_val = Br(jt_idx, ir_idx);
                            bt_val = Btheta(jt_idx, ir_idx);
                            mu_val = mu_map(jt_idx, ir_idx);
                        } else {
                            br_val = Br(ir_idx, jt_idx);
                            bt_val = Btheta(ir_idx, jt_idx);
                            mu_val = mu_map(ir_idx, jt_idx);
                        }
                        // H in polar: Hr = Br/μ, Hθ = Bθ/μ
                        double hr_val = br_val / mu_val;
                        double ht_val = bt_val / mu_val;
                        // Convert to Cartesian
                        double cos_th = std::cos(theta_val);
                        double sin_th = std::sin(theta_val);
                        double bx = br_val * cos_th - bt_val * sin_th;
                        double by = br_val * sin_th + bt_val * cos_th;
                        double hx = hr_val * cos_th - ht_val * sin_th;
                        double hy = hr_val * sin_th + ht_val * cos_th;
                        return std::make_tuple(bx, by, hx, hy);
                    };

                    // Current cell (ir, jt)
                    auto [Bx_c, By_c, Hx_c, Hy_c] = getBH_cartesian(ir, jt, theta);

                    // ============================================================
                    // Outer r-face (ir+1/2, jt): between cells (ir, jt) and (ir+1, jt)
                    // ============================================================
                    double r_outer = 0.5 * (r_coords[ir] + r_coords[ir + 1]);
                    auto [Bx_c_ro, By_c_ro, Hx_c_ro, Hy_c_ro] = getBH_cartesian(ir + 1, jt, theta);
                    // Interpolate B and H to face (simple average)
                    double Bx_ro = 0.5 * (Bx_c + Bx_c_ro);
                    double By_ro = 0.5 * (By_c + By_c_ro);
                    double Hx_ro = 0.5 * (Hx_c + Hx_c_ro);
                    double Hy_ro = 0.5 * (Hy_c + Hy_c_ro);
                    // Compute stress tensor T = B⊗H - (1/2)(B·H)I
                    double BdotH_ro = Bx_ro * Hx_ro + By_ro * Hy_ro;
                    double T_xx_ro = Bx_ro * Hx_ro - 0.5 * BdotH_ro;
                    double T_xy_ro = Bx_ro * Hy_ro;
                    double T_yx_ro = By_ro * Hx_ro;
                    double T_yy_ro = By_ro * Hy_ro - 0.5 * BdotH_ro;
                    // Face normal in Cartesian: n = (cos_t, sin_t) for radial outward
                    double flux_x_ro = T_xx_ro * cos_t + T_xy_ro * sin_t;
                    double flux_y_ro = T_yx_ro * cos_t + T_yy_ro * sin_t;
                    double A_ro = r_outer * dtheta;  // Face area

                    // ============================================================
                    // Inner r-face (ir-1/2, jt): between cells (ir, jt) and (ir-1, jt)
                    // ============================================================
                    double r_inner = 0.5 * (r_coords[ir] + r_coords[ir - 1]);
                    auto [Bx_c_ri, By_c_ri, Hx_c_ri, Hy_c_ri] = getBH_cartesian(ir - 1, jt, theta);
                    double Bx_ri = 0.5 * (Bx_c + Bx_c_ri);
                    double By_ri = 0.5 * (By_c + By_c_ri);
                    double Hx_ri = 0.5 * (Hx_c + Hx_c_ri);
                    double Hy_ri = 0.5 * (Hy_c + Hy_c_ri);
                    double BdotH_ri = Bx_ri * Hx_ri + By_ri * Hy_ri;
                    double T_xx_ri = Bx_ri * Hx_ri - 0.5 * BdotH_ri;
                    double T_xy_ri = Bx_ri * Hy_ri;
                    double T_yx_ri = By_ri * Hx_ri;
                    double T_yy_ri = By_ri * Hy_ri - 0.5 * BdotH_ri;
                    // Face normal: n = (cos_t, sin_t) radial outward (but we'll subtract this face)
                    double flux_x_ri = T_xx_ri * cos_t + T_xy_ri * sin_t;
                    double flux_y_ri = T_yx_ri * cos_t + T_yy_ri * sin_t;
                    double A_ri = r_inner * dtheta;  // Face area

                    // ============================================================
                    // θ+ face (ir, jt+1/2): between cells (ir, jt) and (ir, jt+1)
                    // ============================================================
                    double theta_p = (jt + 0.5) * dtheta;  // Face angle
                    double cos_tp = std::cos(theta_p);
                    double sin_tp = std::sin(theta_p);
                    // Convert each cell's field at its own angle for accuracy
                    // Then average the Cartesian components at the face
                    double theta_jt = jt * dtheta;
                    double theta_jt1 = (jt + 1) * dtheta;
                    auto [Bx_c_jt, By_c_jt, Hx_c_jt, Hy_c_jt] = getBH_cartesian(ir, jt, theta_jt);
                    auto [Bx_c_jt1, By_c_jt1, Hx_c_jt1, Hy_c_jt1] = getBH_cartesian(ir, jt + 1, theta_jt1);
                    double Bx_tp = 0.5 * (Bx_c_jt + Bx_c_jt1);
                    double By_tp = 0.5 * (By_c_jt + By_c_jt1);
                    double Hx_tp = 0.5 * (Hx_c_jt + Hx_c_jt1);
                    double Hy_tp = 0.5 * (Hy_c_jt + Hy_c_jt1);
                    double BdotH_tp = Bx_tp * Hx_tp + By_tp * Hy_tp;
                    double T_xx_tp = Bx_tp * Hx_tp - 0.5 * BdotH_tp;
                    double T_xy_tp = Bx_tp * Hy_tp;
                    double T_yx_tp = By_tp * Hx_tp;
                    double T_yy_tp = By_tp * Hy_tp - 0.5 * BdotH_tp;
                    // θ+ direction in Cartesian: n = (-sin_tp, cos_tp)
                    double flux_x_tp = T_xx_tp * (-sin_tp) + T_xy_tp * cos_tp;
                    double flux_y_tp = T_yx_tp * (-sin_tp) + T_yy_tp * cos_tp;
                    double A_tp = dr;  // Face area

                    // ============================================================
                    // θ- face (ir, jt-1/2): between cells (ir, jt) and (ir, jt-1)
                    // ============================================================
                    double theta_m = (jt - 0.5) * dtheta;  // Face angle
                    double cos_tm = std::cos(theta_m);
                    double sin_tm = std::sin(theta_m);
                    // Convert each cell's field at its own angle for accuracy
                    double theta_jtm1 = (jt - 1) * dtheta;
                    auto [Bx_c_jtm1, By_c_jtm1, Hx_c_jtm1, Hy_c_jtm1] = getBH_cartesian(ir, jt - 1, theta_jtm1);
                    // Reuse Bx_c, By_c, Hx_c, Hy_c from current cell (already at theta = jt*dtheta)
                    double Bx_tm = 0.5 * (Bx_c + Bx_c_jtm1);
                    double By_tm = 0.5 * (By_c + By_c_jtm1);
                    double Hx_tm = 0.5 * (Hx_c + Hx_c_jtm1);
                    double Hy_tm = 0.5 * (Hy_c + Hy_c_jtm1);
                    double BdotH_tm = Bx_tm * Hx_tm + By_tm * Hy_tm;
                    double T_xx_tm = Bx_tm * Hx_tm - 0.5 * BdotH_tm;
                    double T_xy_tm = Bx_tm * Hy_tm;
                    double T_yx_tm = By_tm * Hx_tm;
                    double T_yy_tm = By_tm * Hy_tm - 0.5 * BdotH_tm;
                    // θ direction: e_θ = (-sin θ, cos θ). We compute T·e_θ and subtract for inward flux.
                    double flux_x_tm = T_xx_tm * (-sin_tm) + T_xy_tm * cos_tm;
                    double flux_y_tm = T_yx_tm * (-sin_tm) + T_yy_tm * cos_tm;
                    double A_tm = dr;  // Face area

                    // ============================================================
                    // Discrete divergence: sum of face fluxes
                    // F_cell = (flux_outer - flux_inner) + (flux_θ+ - flux_θ-)
                    // ============================================================
                    double fx = flux_x_ro * A_ro - flux_x_ri * A_ri + flux_x_tp * A_tp - flux_x_tm * A_tm;
                    double fy = flux_y_ro * A_ro - flux_y_ri * A_ri + flux_y_tp * A_tp - flux_y_tm * A_tm;

                    // Accumulate forces
                    result.force_x += fx;
                    result.force_y += fy;

                    // Torque about origin: τz = x*fy - y*fx
                    result.torque_origin += x_c * fy - y_c * fx;
                    result.torque_center += x_c * fy - y_c * fx;  // Same for polar (center at origin)
                }
            }
        }

        // Calculate force magnitude
        result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
        result.torque = result.torque_origin;

        // Output results
        std::cout << "  Material: " << name << std::endl;
        std::cout << "    Material pixels: " << result.pixel_count << std::endl;
        std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
        std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
        std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;

        force_results_flux.push_back(result);
    }

    std::cout << "\nFace-flux force calculation complete!" << std::endl;
}


void MagneticFieldAnalyzer::calculateForceShellIntegration(int step, int shell_thickness) {
    std::cout << "\n=== Calculating Force using Shell Volume Integration ===" << std::endl;
    std::cout << "  Method: F = ∫_Ω_shell T · ∇G dS" << std::endl;
    std::cout << "  Shell thickness: " << shell_thickness << " pixels" << std::endl;

    // --- Calculate magnetic field if not already done for this step ---
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    const double mu0 = MU_0;

    // Physical center for torque calculation
    double cx_physical = (static_cast<double>(nx) * dx) / 2.0;
    double cy_physical = (static_cast<double>(ny) * dy) / 2.0;

    // Clear previous shell results
    force_results_shell.clear();

    // ============================================================
    // Detect periodic/anti-periodic boundary conditions
    // ============================================================
    bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
    bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");
    bool x_antiperiodic = x_periodic && (bc_left.value < 0 || bc_right.value < 0);
    bool y_antiperiodic = y_periodic && (bc_bottom.value < 0 || bc_top.value < 0);

    if (x_periodic || y_periodic) {
        std::cout << "  Periodic boundaries detected: "
                  << (x_periodic ? (x_antiperiodic ? "X(anti) " : "X ") : "")
                  << (y_periodic ? (y_antiperiodic ? "Y(anti)" : "Y") : "") << std::endl;
    }

    // Identify air material (mu_r = 1.0 or very close)
    // Build air mask from mu_map
    cv::Mat air_mask(ny, nx, CV_8U, cv::Scalar(0));
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            double mu_r = mu_map(j, i) / mu0;
            if (std::abs(mu_r - 1.0) < 0.01) {  // Air: mu_r ≈ 1.0
                air_mask.at<uchar>(j, i) = 255;
            }
        }
    }

    // For each material with calc_force=true
    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        const auto& props = material.second;
        bool calc_force = props["calc_force"].as<bool>(false);
        if (!calc_force) continue;

        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Build material mask from current image (in FDM coordinates, y=0 at bottom)
        cv::Mat image_flipped;
        cv::flip(image, image_flipped, 0);

        cv::Mat mat_mask(ny, nx, CV_8U, cv::Scalar(0));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                cv::Vec3b px = image_flipped.at<cv::Vec3b>(j, i);
                if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                    mat_mask.at<uchar>(j, i) = 255;
                }
            }
        }

        ForceResult result;
        result.material_name = name;
        result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
        result.force_x = result.force_y = result.force_radial = 0.0;
        result.torque = result.torque_origin = result.torque_center = 0.0;
        result.pixel_count = 0;
        result.magnetic_energy = 0.0;

        std::cout << "\nCalculating force for material: " << name << " (shell integration)" << std::endl;

        // ============================================================
        // Step 1: Create shell region using morphological dilation
        // Handle periodic boundaries by padding/wrapping
        // ============================================================
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat mat_expanded;
        cv::Mat air_expanded;  // Also expand air mask for periodic boundary handling

        int pad = shell_thickness + 1;  // Padding size for periodic wrapping

        if (x_periodic || y_periodic) {
            // Create padded mask for periodic boundary handling
            int pad_top = y_periodic ? pad : 0;
            int pad_bottom = y_periodic ? pad : 0;
            int pad_left = x_periodic ? pad : 0;
            int pad_right = x_periodic ? pad : 0;

            cv::Mat mat_padded, air_padded;
            cv::copyMakeBorder(mat_mask, mat_padded, pad_top, pad_bottom, pad_left, pad_right,
                               cv::BORDER_WRAP);
            cv::copyMakeBorder(air_mask, air_padded, pad_top, pad_bottom, pad_left, pad_right,
                               cv::BORDER_WRAP);

            // Dilate on padded image
            cv::Mat mat_padded_expanded;
            cv::dilate(mat_padded, mat_padded_expanded, kernel, cv::Point(-1, -1), shell_thickness);

            // Extract center region back to original size
            mat_expanded = mat_padded_expanded(cv::Rect(pad_left, pad_top, nx, ny)).clone();
            air_expanded = air_padded(cv::Rect(pad_left, pad_top, nx, ny)).clone();
        } else {
            // Standard dilation without periodic wrapping
            cv::dilate(mat_mask, mat_expanded, kernel, cv::Point(-1, -1), shell_thickness);
            air_expanded = air_mask.clone();
        }

        // Shell = expanded - original
        cv::Mat shell_mask = mat_expanded - mat_mask;

        // ============================================================
        // Step 2: Check for interference and boundary issues
        // ============================================================
        int total_warnings = 0;

        // 2a. Check for non-air material interference
        cv::Mat shell_in_air;
        cv::bitwise_and(shell_mask, air_expanded, shell_in_air);
        int shell_total = cv::countNonZero(shell_mask);
        int shell_air = cv::countNonZero(shell_in_air);

        if (shell_total != shell_air) {
            int interference = shell_total - shell_air;
            std::cerr << "  [WARNING] Shell interferes with non-air material ("
                      << interference << " pixels). Results may be inaccurate." << std::endl;
            shell_mask = shell_in_air;  // Use only air portion
            total_warnings++;
        }

        // 2b. Check for non-periodic image boundary interference
        // Count shell pixels at image edges that are not periodic
        int boundary_interference = 0;
        for (int j = 0; j < ny; j++) {
            // Left boundary (i=0)
            if (!x_periodic && shell_mask.at<uchar>(j, 0) > 0) boundary_interference++;
            // Right boundary (i=nx-1)
            if (!x_periodic && shell_mask.at<uchar>(j, nx - 1) > 0) boundary_interference++;
        }
        for (int i = 0; i < nx; i++) {
            // Bottom boundary (j=0)
            if (!y_periodic && shell_mask.at<uchar>(0, i) > 0) boundary_interference++;
            // Top boundary (j=ny-1)
            if (!y_periodic && shell_mask.at<uchar>(ny - 1, i) > 0) boundary_interference++;
        }

        if (boundary_interference > 0) {
            std::cerr << "  [WARNING] Shell touches non-periodic image boundary ("
                      << boundary_interference << " edge pixels). Results may have boundary noise." << std::endl;
            total_warnings++;
        }

        int shell_pixels = cv::countNonZero(shell_mask);
        std::cout << "  Shell pixels: " << shell_pixels;
        if (total_warnings > 0) {
            std::cout << " (" << total_warnings << " warning(s))";
        }
        std::cout << std::endl;

        if (shell_pixels == 0) {
            std::cout << "  ERROR: No valid shell region found. Skipping." << std::endl;
            force_results_shell.push_back(result);
            continue;
        }

        // ============================================================
        // Step 3: Create weight function G using distance transform
        // G = 1 at material surface, G = 0 at shell outer boundary
        // For periodic boundaries, use padded distance transform
        // ============================================================
        cv::Mat dist_map(ny, nx, CV_32F, cv::Scalar(0.0f));

        if (x_periodic || y_periodic) {
            // For periodic boundaries, compute distance on padded/tiled image
            int pad_top = y_periodic ? pad : 0;
            int pad_bottom = y_periodic ? pad : 0;
            int pad_left = x_periodic ? pad : 0;
            int pad_right = x_periodic ? pad : 0;

            cv::Mat mat_padded;
            cv::copyMakeBorder(mat_mask, mat_padded, pad_top, pad_bottom, pad_left, pad_right,
                               cv::BORDER_WRAP);

            cv::Mat mat_padded_inv;
            cv::bitwise_not(mat_padded, mat_padded_inv);

            cv::Mat dist_padded;
            cv::distanceTransform(mat_padded_inv, dist_padded, cv::DIST_L2, cv::DIST_MASK_PRECISE);

            // Extract center region
            dist_map = dist_padded(cv::Rect(pad_left, pad_top, nx, ny)).clone();
        } else {
            // Standard distance transform
            cv::Mat mat_mask_inv;
            cv::bitwise_not(mat_mask, mat_mask_inv);
            cv::distanceTransform(mat_mask_inv, dist_map, cv::DIST_L2, cv::DIST_MASK_PRECISE);
        }

        // Find maximum distance within shell (= shell thickness in pixels)
        double max_dist = 0;
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (shell_mask.at<uchar>(j, i) > 0) {
                    double d = dist_map.at<float>(j, i);
                    if (d > max_dist) max_dist = d;
                }
            }
        }

        if (max_dist < 1e-10) {
            std::cout << "  ERROR: max_dist is zero. Skipping." << std::endl;
            force_results_shell.push_back(result);
            continue;
        }

        std::cout << "  Max distance in shell: " << max_dist << " pixels" << std::endl;

        // Create G map with CORRECT values:
        // - G = 1 inside material (CRITICAL: needed for correct gradient at interface)
        // - G = 1 - d/max_dist in shell (smooth transition from 1 to 0)
        // - G = 0 outside shell
        cv::Mat G_map(ny, nx, CV_64F, cv::Scalar(0.0));
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (mat_mask.at<uchar>(j, i) > 0) {
                    // Inside material: G = 1
                    G_map.at<double>(j, i) = 1.0;
                } else if (shell_mask.at<uchar>(j, i) > 0) {
                    // In shell: G transitions from 1 to 0
                    double d = dist_map.at<float>(j, i);
                    double g = 1.0 - d / max_dist;
                    G_map.at<double>(j, i) = std::max(0.0, std::min(1.0, g));
                }
                // Outside shell: G = 0 (default)
            }
        }

        // ============================================================
        // Step 4: Compute gradient of G using central differences
        // ∇G = (∂G/∂x, ∂G/∂y) in physical coordinates
        // Using central differences: ∂G/∂x ≈ (G(i+1,j) - G(i-1,j)) / (2*dx)
        // Handle periodic boundaries with wrapped indices.
        // ============================================================
        cv::Mat grad_Gx(ny, nx, CV_64F, cv::Scalar(0.0));
        cv::Mat grad_Gy(ny, nx, CV_64F, cv::Scalar(0.0));

        // Lambda for wrapped index access
        auto getG = [&](int j, int i) -> double {
            // Handle periodic wrapping
            if (x_periodic) {
                i = (i + nx) % nx;
            } else {
                i = std::max(0, std::min(nx - 1, i));
            }
            if (y_periodic) {
                j = (j + ny) % ny;
            } else {
                j = std::max(0, std::min(ny - 1, j));
            }
            return G_map.at<double>(j, i);
        };

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // Skip if not near shell region (optimization)
                // Compute gradient at all points to capture interface effects

                // Determine if we should compute gradient here
                // (only needed where shell_mask is 1 or at its boundary)
                bool compute_here = false;
                if (shell_mask.at<uchar>(j, i) > 0) {
                    compute_here = true;
                } else {
                    // Check if any neighbor is in shell (for interface gradient)
                    for (int dj = -1; dj <= 1; dj++) {
                        for (int di = -1; di <= 1; di++) {
                            int jj = j + dj;
                            int ii = i + di;
                            if (x_periodic) ii = (ii + nx) % nx;
                            if (y_periodic) jj = (jj + ny) % ny;
                            if (jj >= 0 && jj < ny && ii >= 0 && ii < nx) {
                                if (shell_mask.at<uchar>(jj, ii) > 0) {
                                    compute_here = true;
                                    break;
                                }
                            }
                        }
                        if (compute_here) break;
                    }
                }

                if (!compute_here) continue;

                // Central difference with periodic wrapping
                double G_ip1 = getG(j, i + 1);
                double G_im1 = getG(j, i - 1);
                double G_jp1 = getG(j + 1, i);
                double G_jm1 = getG(j - 1, i);

                double dGdx = (G_ip1 - G_im1) / (2.0 * dx);
                double dGdy = (G_jp1 - G_jm1) / (2.0 * dy);

                grad_Gx.at<double>(j, i) = dGdx;
                grad_Gy.at<double>(j, i) = dGdy;
            }
        }

        // ============================================================
        // Step 5: Compute Maxwell stress tensor in air (using μ₀)
        // T_xx = (Bx² - By²) / (2μ₀)
        // T_xy = Bx·By / μ₀
        // T_yy = (By² - Bx²) / (2μ₀)
        // ============================================================
        // Note: In air, H = B/μ₀, so T = B⊗H - (1/2)(B·H)I
        //   T_xx = Bx·Hx - (1/2)(B·H) = Bx²/μ₀ - (Bx² + By²)/(2μ₀) = (Bx² - By²)/(2μ₀)
        //   T_xy = Bx·Hy = Bx·By/μ₀
        //   T_yy = By·Hy - (1/2)(B·H) = By²/μ₀ - (Bx² + By²)/(2μ₀) = (By² - Bx²)/(2μ₀)

        // ============================================================
        // Step 6: Integrate F = ∫_shell T · ∇G dS
        // Fx = ∫ (T_xx · ∂G/∂x + T_xy · ∂G/∂y) dS
        // Fy = ∫ (T_xy · ∂G/∂x + T_yy · ∂G/∂y) dS
        // ============================================================
        double dS = dx * dy;  // Area element

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                if (shell_mask.at<uchar>(j, i) == 0) continue;

                result.pixel_count++;

                double bx = Bx(j, i);
                double by = By(j, i);

                // Maxwell stress tensor components (in air, using μ₀)
                double T_xx = (bx * bx - by * by) / (2.0 * mu0);
                double T_xy = bx * by / mu0;
                double T_yy = (by * by - bx * bx) / (2.0 * mu0);

                // Gradient of G
                double dGdx = grad_Gx.at<double>(j, i);
                double dGdy = grad_Gy.at<double>(j, i);

                // Force contribution: T · ∇G
                double fx = (T_xx * dGdx + T_xy * dGdy) * dS;
                double fy = (T_xy * dGdx + T_yy * dGdy) * dS;

                result.force_x += fx;
                result.force_y += fy;

                // Torque: τ = r × F
                // Physical coordinates of this pixel
                double x_c = (i + 0.5) * dx;
                double y_c = (j + 0.5) * dy;

                result.torque_origin += x_c * fy - y_c * fx;
                result.torque_center += (x_c - cx_physical) * fy - (y_c - cy_physical) * fx;
            }
        }

        // Calculate force magnitude
        result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
        result.torque = result.torque_origin;

        // Output results
        std::cout << "  Material: " << name << std::endl;
        std::cout << "    Shell pixels used: " << result.pixel_count << std::endl;
        std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
        std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
        std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
        std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;

        force_results_shell.push_back(result);
    }

    std::cout << "\nShell volume integration complete!" << std::endl;
}


void MagneticFieldAnalyzer::calculateForceDistributedAmperian(int step, double sigma_smooth) {
    std::cout << "\n=== Calculating Force using Distributed Amperian Method ===" << std::endl;
    std::cout << "  Method: M = B/μ₀ - H, J_b = ∇×M, F = ∫J_b × B dV" << std::endl;
    std::cout << "  Coordinate system: " << coordinate_system << std::endl;
    if (sigma_smooth > 0) {
        std::cout << "  Gaussian smoothing sigma: " << sigma_smooth << " pixels" << std::endl;
    }

    // --- Calculate magnetic field if not already done for this step ---
    if (current_field_step != step) {
        if (coordinate_system == "polar") {
            calculateMagneticFieldPolar();
        } else {
            calculateMagneticField();
        }
        current_field_step = step;
    } else {
        std::cout << "Magnetic field already calculated for step " << step << " (reusing)" << std::endl;
    }

    const double mu0 = MU_0;

    // Clear previous results
    force_results_amperian.clear();

    // ============================================================
    // Branch: Polar vs Cartesian coordinate system
    // ============================================================
    if (coordinate_system == "polar") {
        // ============================================================
        // POLAR COORDINATE IMPLEMENTATION
        // ============================================================
        std::cout << "  Using polar coordinate formulation" << std::endl;
        std::cout << "  r_orientation: " << r_orientation << std::endl;
        std::cout << "  nr=" << nr << ", ntheta=" << ntheta << std::endl;

        // Determine theta periodicity
        bool theta_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
        bool theta_antiperiodic = (bc_theta_min.type == "anti-periodic" || bc_theta_max.type == "anti-periodic");

        // Grid dimensions for matrix storage (depends on r_orientation)
        // r_orientation == "horizontal": rows=ntheta, cols=nr, indexing: mat(jt, ir)
        // r_orientation == "vertical": rows=nr, cols=ntheta, indexing: mat(ir, jt)
        int grid_rows, grid_cols;
        if (r_orientation == "horizontal") {
            grid_rows = ntheta;
            grid_cols = nr;
        } else {  // vertical
            grid_rows = nr;
            grid_cols = ntheta;
        }
        std::cout << "  Grid dimensions: " << grid_rows << " rows x " << grid_cols << " cols" << std::endl;

        // Build r_coords array for physical r values
        std::vector<double> r_coords(nr);
        for (int ir = 0; ir < nr; ir++) {
            r_coords[ir] = r_start + (ir + 0.5) * dr;
        }

        // ============================================================
        // Step 1: Calculate magnetization in polar coordinates
        // M_r = B_r(μ_r - 1)/(μ₀μ_r), M_θ = B_θ(μ_r - 1)/(μ₀μ_r)
        // In air (μ_r = 1): M = 0 exactly (no ghost force)
        // ============================================================
        cv::Mat Mr_map(grid_rows, grid_cols, CV_64F, cv::Scalar(0.0));
        cv::Mat Mt_map(grid_rows, grid_cols, CV_64F, cv::Scalar(0.0));  // M_theta

        for (int j = 0; j < grid_rows; j++) {
            for (int i = 0; i < grid_cols; i++) {
                double mu = mu_map(j, i);
                double mu_r_val = mu / mu0;
                double chi = (mu_r_val - 1.0) / mu_r_val;

                // Get Br, Btheta depending on r_orientation
                // Matrix indexing is always (row, col) = (j, i)
                // For both orientations, Br and Btheta are stored with consistent indexing
                double br_val, bt_val;
                int ir, jt;
                if (r_orientation == "horizontal") {
                    ir = i; jt = j;  // col=r, row=theta
                    br_val = Br(jt, ir);
                    bt_val = Btheta(jt, ir);
                } else {
                    ir = j; jt = i;  // row=r, col=theta
                    br_val = Br(ir, jt);
                    bt_val = Btheta(ir, jt);
                }

                Mr_map.at<double>(j, i) = br_val * chi / mu0;
                Mt_map.at<double>(j, i) = bt_val * chi / mu0;
            }
        }

        // ============================================================
        // Step 2: Optional Gaussian smoothing of M
        // ============================================================
        if (sigma_smooth > 0) {
            int ksize = static_cast<int>(std::ceil(sigma_smooth * 6)) | 1;
            if (ksize < 3) ksize = 3;
            cv::GaussianBlur(Mr_map, Mr_map, cv::Size(ksize, ksize), sigma_smooth, sigma_smooth);
            cv::GaussianBlur(Mt_map, Mt_map, cv::Size(ksize, ksize), sigma_smooth, sigma_smooth);
            std::cout << "  Applied Gaussian smoothing (kernel size: " << ksize << ")" << std::endl;
        }

        // ============================================================
        // Step 3: Compute bound current in polar coordinates
        // J_z = (1/r) ∂(r·M_θ)/∂r - (1/r) ∂M_r/∂θ
        //     = M_θ/r + ∂M_θ/∂r - (1/r) ∂M_r/∂θ
        // ============================================================
        cv::Mat Jz_map(grid_rows, grid_cols, CV_64F, cv::Scalar(0.0));

        // Lambda for wrapped index access with polar boundary handling
        // Note: Mr_map and Mt_map are stored as cv::Mat with (row, col) = (j, i)
        // For horizontal: j=theta, i=r → access as (jt, ir)
        // For vertical: j=r, i=theta → access as (ir, jt)
        auto getMr = [&](int jt, int ir) -> double {
            // Handle r boundary (non-periodic)
            ir = std::max(0, std::min(nr - 1, ir));
            // Handle theta boundary
            if (theta_periodic) {
                jt = (jt + ntheta) % ntheta;
            } else if (theta_antiperiodic) {
                if (jt < 0) {
                    jt = -jt - 1;
                    if (r_orientation == "horizontal") {
                        return -Mr_map.at<double>(jt, ir);
                    } else {
                        return -Mr_map.at<double>(ir, jt);
                    }
                }
                if (jt >= ntheta) {
                    jt = 2 * ntheta - jt - 1;
                    if (r_orientation == "horizontal") {
                        return -Mr_map.at<double>(jt, ir);
                    } else {
                        return -Mr_map.at<double>(ir, jt);
                    }
                }
            } else {
                jt = std::max(0, std::min(ntheta - 1, jt));
            }
            if (r_orientation == "horizontal") {
                return Mr_map.at<double>(jt, ir);
            } else {
                return Mr_map.at<double>(ir, jt);
            }
        };
        auto getMt = [&](int jt, int ir) -> double {
            ir = std::max(0, std::min(nr - 1, ir));
            if (theta_periodic) {
                jt = (jt + ntheta) % ntheta;
            } else if (theta_antiperiodic) {
                if (jt < 0) {
                    jt = -jt - 1;
                    if (r_orientation == "horizontal") {
                        return -Mt_map.at<double>(jt, ir);
                    } else {
                        return -Mt_map.at<double>(ir, jt);
                    }
                }
                if (jt >= ntheta) {
                    jt = 2 * ntheta - jt - 1;
                    if (r_orientation == "horizontal") {
                        return -Mt_map.at<double>(jt, ir);
                    } else {
                        return -Mt_map.at<double>(ir, jt);
                    }
                }
            } else {
                jt = std::max(0, std::min(ntheta - 1, jt));
            }
            if (r_orientation == "horizontal") {
                return Mt_map.at<double>(jt, ir);
            } else {
                return Mt_map.at<double>(ir, jt);
            }
        };

        for (int j = 0; j < grid_rows; j++) {
            for (int i = 0; i < grid_cols; i++) {
                int ir, jt;
                if (r_orientation == "horizontal") {
                    ir = i; jt = j;
                } else {
                    ir = j; jt = i;
                }

                double r = r_coords[ir];
                if (r < 1e-10) {
                    // At r=0, curl is undefined; skip or use limit
                    Jz_map.at<double>(j, i) = 0.0;
                    continue;
                }

                // Central differences for derivatives
                // ∂(r·M_θ)/∂r using central difference
                double rMt_plus = r_coords[std::min(ir + 1, nr - 1)] * getMt(jt, ir + 1);
                double rMt_minus = r_coords[std::max(ir - 1, 0)] * getMt(jt, ir - 1);
                double d_rMt_dr = (rMt_plus - rMt_minus) / (2.0 * dr);

                // ∂M_r/∂θ using central difference
                double dMr_dtheta = (getMr(jt + 1, ir) - getMr(jt - 1, ir)) / (2.0 * dtheta);

                // J_z = (1/r) ∂(r·M_θ)/∂r - (1/r) ∂M_r/∂θ
                double jz = (1.0 / r) * d_rMt_dr - (1.0 / r) * dMr_dtheta;
                Jz_map.at<double>(j, i) = jz;
            }
        }

        // ============================================================
        // Step 4: For each material, integrate F = J_b × B
        // ============================================================
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        int dilation_size = 2;

        for (const auto& material : config["materials"]) {
            std::string name = material.first.as<std::string>();
            const auto& props = material.second;
            bool calc_force = props["calc_force"].as<bool>(false);
            if (!calc_force) continue;

            std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

            // Build material mask
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);

            // Debug: Check image dimensions
            std::cout << "  Image size: " << image_flipped.cols << " x " << image_flipped.rows << std::endl;
            std::cout << "  Creating mask with size: " << grid_cols << " x " << grid_rows << std::endl;

            cv::Mat mat_mask(grid_rows, grid_cols, CV_8U, cv::Scalar(0));
            int match_count = 0;
            for (int jj = 0; jj < grid_rows; jj++) {
                for (int ii = 0; ii < grid_cols; ii++) {
                    cv::Vec3b px = image_flipped.at<cv::Vec3b>(jj, ii);
                    if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                        mat_mask.at<uchar>(jj, ii) = 255;
                        match_count++;
                    }
                }
            }
            std::cout << "  Material pixels found: " << match_count << std::endl;

            // Dilate material mask with periodic handling for theta
            // For polar: theta direction is periodic, r direction is not
            cv::Mat mat_dilated;

            // Check mat_mask validity
            if (mat_mask.empty() || mat_mask.rows == 0 || mat_mask.cols == 0) {
                std::cerr << "  ERROR: mat_mask is empty or invalid!" << std::endl;
                mat_dilated = mat_mask.clone();
            } else if (theta_periodic) {
                int pad = dilation_size + 1;

                // For theta-periodic boundary:
                // r_orientation == "horizontal": rows=ntheta (periodic), cols=nr (non-periodic)
                // r_orientation == "vertical": rows=nr (non-periodic), cols=ntheta (periodic)
                cv::Mat mat_padded;
                if (r_orientation == "horizontal") {
                    // Rows are theta (periodic), cols are r (non-periodic)
                    // Pad top/bottom for theta wrap
                    cv::copyMakeBorder(mat_mask, mat_padded, pad, pad, 0, 0, cv::BORDER_WRAP);
                    cv::Mat mat_padded_dilated;
                    cv::dilate(mat_padded, mat_padded_dilated, kernel, cv::Point(-1, -1), dilation_size);
                    mat_dilated = mat_padded_dilated(cv::Rect(0, pad, grid_cols, grid_rows)).clone();
                } else {
                    // Rows are r (non-periodic), cols are theta (periodic)
                    // Pad left/right for theta wrap
                    cv::copyMakeBorder(mat_mask, mat_padded, 0, 0, pad, pad, cv::BORDER_WRAP);
                    cv::Mat mat_padded_dilated;
                    cv::dilate(mat_padded, mat_padded_dilated, kernel, cv::Point(-1, -1), dilation_size);
                    mat_dilated = mat_padded_dilated(cv::Rect(pad, 0, grid_cols, grid_rows)).clone();
                }
            } else {
                cv::dilate(mat_mask, mat_dilated, kernel, cv::Point(-1, -1), dilation_size);
            }

            ForceResult result;
            result.material_name = name;
            result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
            result.force_x = result.force_y = result.force_radial = 0.0;
            result.torque = result.torque_origin = result.torque_center = 0.0;
            result.pixel_count = 0;
            result.magnetic_energy = 0.0;

            std::cout << "\nCalculating force for material: " << name << " (Amperian, polar)" << std::endl;

            // ============================================================
            // Step 5: Integrate Lorentz force F = J × B in polar coords
            // For J = (0, 0, Jz) and B = (Br, Bθ, 0) in cylindrical:
            //   J × B = (-Jz·Bθ, +Jz·Br, 0) in (r, θ, z) components
            //   => Fr = -Jz·Bθ, Fθ = +Jz·Br
            //
            // Physics note: For constant-current sources (Jz specified),
            // the force is F = +∂W'/∂x where W' is magnetic co-energy.
            // For linear materials, W = W' numerically, so F = +∂W/∂x.
            // The Lorentz formula F = J × B is always valid.
            // ============================================================
            for (int j = 0; j < grid_rows; j++) {
                for (int i = 0; i < grid_cols; i++) {
                    if (mat_dilated.at<uchar>(j, i) == 0) continue;

                    int ir, jt;
                    if (r_orientation == "horizontal") {
                        ir = i; jt = j;
                    } else {
                        ir = j; jt = i;
                    }

                    double r = r_coords[ir];
                    double theta = jt * dtheta;  // Use rotor frame (no theta_offset)

                    // Volume element in polar coordinates: r * dr * dθ
                    double dV = r * dr * dtheta;

                    double jz = Jz_map.at<double>(j, i);

                    // Get Br, Btheta
                    double br_val, bt_val;
                    if (r_orientation == "horizontal") {
                        br_val = Br(jt, ir);
                        bt_val = Btheta(jt, ir);
                    } else {
                        br_val = Br(ir, jt);
                        bt_val = Btheta(ir, jt);
                    }

                    // Force: F = J × B (Lorentz force in polar coordinates)
                    // Fr = (J × B)_r = -Jz·Bθ
                    // Fθ = (J × B)_θ = +Jz·Br
                    double fr = -jz * bt_val * dV;
                    double ft = +jz * br_val * dV;

                    // Convert to Cartesian coordinates
                    double cos_t = std::cos(theta);
                    double sin_t = std::sin(theta);
                    double fx = fr * cos_t - ft * sin_t;
                    double fy = fr * sin_t + ft * cos_t;

                    result.force_x += fx;
                    result.force_y += fy;
                    result.pixel_count++;

                    // Torque: τ = r × F = r · F_θ (in 2D)
                    // For polar centered at origin, torque = r * F_theta
                    result.torque_origin += r * ft;

                    // Physical coordinates for torque calculation
                    double x_c = r * cos_t;
                    double y_c = r * sin_t;
                    result.torque_center += x_c * fy - y_c * fx;
                }
            }

            // Calculate force magnitude
            result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
            result.torque = result.torque_origin;

            // Output results
            std::cout << "  Material: " << name << std::endl;
            std::cout << "    Integration pixels: " << result.pixel_count << std::endl;
            std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
            std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
            std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
            std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;

            force_results_amperian.push_back(result);
        }

    } else {
        // ============================================================
        // CARTESIAN COORDINATE IMPLEMENTATION (original code)
        // ============================================================

        // Physical center for torque calculation
        double cx_physical = (static_cast<double>(nx) * dx) / 2.0;
        double cy_physical = (static_cast<double>(ny) * dy) / 2.0;

        // ============================================================
        // Step 1: Calculate magnetization M = B/μ₀ - H
        // For linear material: H = B/(μ₀μ_r), so M = B/μ₀ - B/(μ₀μ_r) = B(μ_r - 1)/(μ₀μ_r)
        // In air (μ_r = 1): M = 0 exactly (no ghost force)
        // ============================================================
        cv::Mat Mx_map(ny, nx, CV_64F, cv::Scalar(0.0));
        cv::Mat My_map(ny, nx, CV_64F, cv::Scalar(0.0));

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double mu = mu_map(j, i);         // μ = μ₀μ_r [H/m]
                double mu_r = mu / mu0;           // relative permeability
                double bx = Bx(j, i);
                double by = By(j, i);

                // M = B(μ_r - 1)/(μ₀μ_r) = B · chi / μ₀ where chi = (μ_r - 1)/μ_r
                double chi = (mu_r - 1.0) / mu_r;
                Mx_map.at<double>(j, i) = bx * chi / mu0;
                My_map.at<double>(j, i) = by * chi / mu0;
            }
        }

        // ============================================================
        // Step 2: Optional Gaussian smoothing of M
        // This captures surface magnetization current through numerical curl
        // ============================================================
        if (sigma_smooth > 0) {
            int ksize = static_cast<int>(std::ceil(sigma_smooth * 6)) | 1;  // Ensure odd
            if (ksize < 3) ksize = 3;
            cv::GaussianBlur(Mx_map, Mx_map, cv::Size(ksize, ksize), sigma_smooth, sigma_smooth);
            cv::GaussianBlur(My_map, My_map, cv::Size(ksize, ksize), sigma_smooth, sigma_smooth);
            std::cout << "  Applied Gaussian smoothing (kernel size: " << ksize << ")" << std::endl;
        }

        // ============================================================
        // Step 3: Compute bound current J_bz = ∂My/∂x - ∂Mx/∂y
        // Using central differences with periodic boundary handling
        // ============================================================
        bool x_periodic = (bc_left.type == "periodic" && bc_right.type == "periodic");
        bool y_periodic = (bc_bottom.type == "periodic" && bc_top.type == "periodic");

        cv::Mat Jz_map(ny, nx, CV_64F, cv::Scalar(0.0));

        // Lambda for wrapped index access
        auto getMx = [&](int j, int i) -> double {
            if (x_periodic) i = (i + nx) % nx;
            else i = std::max(0, std::min(nx - 1, i));
            if (y_periodic) j = (j + ny) % ny;
            else j = std::max(0, std::min(ny - 1, j));
            return Mx_map.at<double>(j, i);
        };
        auto getMy = [&](int j, int i) -> double {
            if (x_periodic) i = (i + nx) % nx;
            else i = std::max(0, std::min(nx - 1, i));
            if (y_periodic) j = (j + ny) % ny;
            else j = std::max(0, std::min(ny - 1, j));
            return My_map.at<double>(j, i);
        };

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                // Central differences: ∂My/∂x, ∂Mx/∂y
                double dMy_dx = (getMy(j, i + 1) - getMy(j, i - 1)) / (2.0 * dx);
                double dMx_dy = (getMx(j + 1, i) - getMx(j - 1, i)) / (2.0 * dy);

                // J_bz = ∂My/∂x - ∂Mx/∂y (curl of M in 2D)
                Jz_map.at<double>(j, i) = dMy_dx - dMx_dy;
            }
        }

        // ============================================================
        // Step 4: For each material, integrate F = J_b × B over dilated region
        // Dilated region ensures surface current contribution is captured
        // ============================================================
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        int dilation_size = 2;  // Dilate by 2 pixels to capture interface current

        for (const auto& material : config["materials"]) {
            std::string name = material.first.as<std::string>();
            const auto& props = material.second;
            bool calc_force = props["calc_force"].as<bool>(false);
            if (!calc_force) continue;

            std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

            // Build material mask (in FDM coordinates, y=0 at bottom)
            cv::Mat image_flipped;
            cv::flip(image, image_flipped, 0);

            cv::Mat mat_mask(ny, nx, CV_8U, cv::Scalar(0));
            for (int jj = 0; jj < ny; jj++) {
                for (int ii = 0; ii < nx; ii++) {
                    cv::Vec3b px = image_flipped.at<cv::Vec3b>(jj, ii);
                    if (px[0] == rgb[0] && px[1] == rgb[1] && px[2] == rgb[2]) {
                        mat_mask.at<uchar>(jj, ii) = 255;
                    }
                }
            }

            // Dilate material mask to include interface region
            cv::Mat mat_dilated;
            if (x_periodic || y_periodic) {
                int pad = dilation_size + 1;
                int pad_top = y_periodic ? pad : 0;
                int pad_bottom = y_periodic ? pad : 0;
                int pad_left = x_periodic ? pad : 0;
                int pad_right = x_periodic ? pad : 0;

                cv::Mat mat_padded;
                cv::copyMakeBorder(mat_mask, mat_padded, pad_top, pad_bottom, pad_left, pad_right,
                                   cv::BORDER_WRAP);
                cv::Mat mat_padded_dilated;
                cv::dilate(mat_padded, mat_padded_dilated, kernel, cv::Point(-1, -1), dilation_size);
                mat_dilated = mat_padded_dilated(cv::Rect(pad_left, pad_top, nx, ny)).clone();
            } else {
                cv::dilate(mat_mask, mat_dilated, kernel, cv::Point(-1, -1), dilation_size);
            }

            ForceResult result;
            result.material_name = name;
            result.rgb = cv::Scalar(rgb[0], rgb[1], rgb[2]);
            result.force_x = result.force_y = result.force_radial = 0.0;
            result.torque = result.torque_origin = result.torque_center = 0.0;
            result.pixel_count = 0;
            result.magnetic_energy = 0.0;

            std::cout << "\nCalculating force for material: " << name << " (Amperian)" << std::endl;

            // ============================================================
            // Step 5: Integrate Lorentz force F = J × B
            // For 2D with J = (0, 0, Jz) and B = (Bx, By, 0):
            //   J × B = (-Jz·By, +Jz·Bx, 0)
            //   => Fx = -Jz·By, Fy = +Jz·Bx
            //
            // Physics note: For constant-current sources (Jz specified),
            // the force is F = +∂W'/∂x where W' is magnetic co-energy.
            // For linear materials, W = W' numerically, so F = +∂W/∂x.
            // The Lorentz formula F = J × B is always valid.
            // ============================================================
            double dV = dx * dy;  // Volume element (per unit depth)

            for (int j = 0; j < ny; j++) {
                for (int i = 0; i < nx; i++) {
                    if (mat_dilated.at<uchar>(j, i) == 0) continue;

                    double jz = Jz_map.at<double>(j, i);
                    double bx = Bx(j, i);
                    double by = By(j, i);

                    // Force: F = J × B (Lorentz force)
                    // Fx = (J × B)_x = -Jz·By
                    // Fy = (J × B)_y = +Jz·Bx
                    double fx = -jz * by * dV;
                    double fy = +jz * bx * dV;

                    result.force_x += fx;
                    result.force_y += fy;
                    result.pixel_count++;

                    // Torque: τ = r × F
                    double x_c = (i + 0.5) * dx;
                    double y_c = (j + 0.5) * dy;

                    result.torque_origin += x_c * fy - y_c * fx;
                    result.torque_center += (x_c - cx_physical) * fy - (y_c - cy_physical) * fx;
                }
            }

            // Calculate force magnitude
            result.force_radial = std::sqrt(result.force_x * result.force_x + result.force_y * result.force_y);
            result.torque = result.torque_origin;

            // Output results
            std::cout << "  Material: " << name << std::endl;
            std::cout << "    Integration pixels: " << result.pixel_count << std::endl;
            std::cout << "    Fx: " << result.force_x << " N/m, Fy: " << result.force_y << " N/m (per unit depth)" << std::endl;
            std::cout << "    Force magnitude: " << result.force_radial << " N/m (per unit depth)" << std::endl;
            std::cout << "    Torque about origin: " << result.torque_origin << " N*m (per unit depth)" << std::endl;
            std::cout << "    Torque about image center: " << result.torque_center << " N*m (per unit depth)" << std::endl;

            force_results_amperian.push_back(result);
        }
    }

    std::cout << "\nDistributed Amperian force calculation complete!" << std::endl;
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
        // Cartesian coordinates
        int rows = Bx.rows();
        int cols = Bx.cols();

        // Verify dimensions match
        if (mu_map.rows() != rows || mu_map.cols() != cols) {
            std::cerr << "[ERROR] Dimension mismatch: Bx(" << rows << "x" << cols
                      << ") vs mu_map(" << mu_map.rows() << "x" << mu_map.cols() << ")" << std::endl;
            return 0.0;
        }

        // Calculate magnetic co-energy density W' = ∫₀^H B(H') dH'
        // For current-source systems (Jz specified): F = +∂W'/∂x|_I
        // For linear materials: W' = W = B²/(2μ)
        // For nonlinear materials: W' = ∫B dH using Simpson integration
        double max_coenergy_density = 0.0;
        double dV = dx * dy;  // [m²] per unit depth

        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                double B_mag = std::sqrt(Bx(j, i) * Bx(j, i) + By(j, i) * By(j, i));
                double w = calculateCoEnergyDensity(j, i, B_mag);
                total_energy += w * dV;
                if (w > max_coenergy_density) max_coenergy_density = w;
            }
        }

        std::cout << "  Grid size: " << rows << " x " << cols << std::endl;
        std::cout << "  dx = " << dx << " m, dy = " << dy << " m" << std::endl;
        std::cout << "  Max co-energy density: " << max_coenergy_density << " J/m³" << std::endl;
        std::cout << "  Total co-energy: " << total_energy << " J/m (per unit depth)" << std::endl;
        if (has_nonlinear_materials) {
            std::cout << "  (Using magnetic co-energy W' = ∫B dH for nonlinear materials)" << std::endl;
        }

    } else {
        // Polar coordinates
        int rows = Br.rows();
        int cols = Br.cols();

        if (mu_map.rows() != rows || mu_map.cols() != cols) {
            std::cerr << "[ERROR] Dimension mismatch in polar coordinates" << std::endl;
            return 0.0;
        }

        // Calculate magnetic co-energy density W' = ∫₀^H B(H') dH'
        // For current-source systems (Jz specified): F = +∂W'/∂x|_I
        // For linear materials: W' = W = B²/(2μ)
        // For nonlinear materials: W' = ∫B dH using Simpson integration
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < cols; ++i) {
                double B_mag = std::sqrt(Br(j, i) * Br(j, i) + Btheta(j, i) * Btheta(j, i));
                double w = calculateCoEnergyDensity(j, i, B_mag);

                // Polar volume element: dV = r * dr * dθ
                int ir = (r_orientation == "horizontal") ? i : j;
                ir = std::clamp(ir, 0, nr - 1);
                double r = r_coords[ir];
                double dV = r * dr * dtheta;
                total_energy += w * dV;
            }
        }

        std::cout << "  Grid size (r x theta): " << rows << " x " << cols << std::endl;
        std::cout << "  Total co-energy: " << total_energy << " J/m (per unit depth)" << std::endl;
        if (has_nonlinear_materials) {
            std::cout << "  (Using magnetic co-energy W' = ∫B dH for nonlinear materials)" << std::endl;
        }
    }

    // Store the result in the member variable for later export
    system_total_energy = total_energy;

    return total_energy;
}


void MagneticFieldAnalyzer::exportForcesToCSV(const std::string& output_path) const {
    if (force_results_amperian.empty() && system_total_energy == 0.0) {
        std::cout << "No force results to export" << std::endl;
        return;
    }

    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open force output file: " + output_path);
    }

    // CSV header (include both torque measures and magnetic energy)
    file << "Material,RGB_R,RGB_G,RGB_B,Force_X[N/m],Force_Y[N/m],Force_Magnitude[N/m],Force_Radial[N/m],Torque_Origin[N*m],Torque_Center[N*m],Magnetic_Energy[J/m],Boundary_Pixels\n";
    file << "# Note: Forces and energies are per unit depth (2D analysis)\n";
    file << "# Torque_Origin: torque about origin (polar)\n";
    file << "# Torque_Center: torque about image center (x=center_x, y=center_y)\n";
    file << "# Magnetic_Energy: magnetic potential energy W = integral(B^2/(2*mu) dV)\n";
    file << "# _SYSTEM_TOTAL: Total magnetic energy of entire system (for virtual work calculation)\n";
    file << "# Force calculation method: Distributed Amperian Force (J_b = curl(M), F = J_b x B)\n";

    file << std::scientific << std::setprecision(6);

    // Output Distributed Amperian force results (default method)
    // This uses bound current from magnetization: J_b = ∇×M, F = ∫J_b × B dV
    // Key advantage: No ghost force (M = 0 exactly in air where μ_r = 1)
    for (const auto& result : force_results_amperian) {
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

    // Add system total co-energy as a special row
    // This enables virtual work calculation:
    //   - Constant flux (λ): F = -∂W/∂x|_λ (use energy W)
    //   - Constant current (I or Jz): F = +∂W'/∂x|_I (use co-energy W')
    // Since OpenMagFDM uses specified Jz (constant current), we compute co-energy W'.
    // For linear materials: W' = W = B²/(2μ)
    // For nonlinear materials: W' = ∫B dH ≠ W = ∫H dB (Simpson integration used)
    // Virtual Work F = +∂W'/∂x is now accurate for both linear and nonlinear materials.
    file << "_SYSTEM_TOTAL,0,0,0,0,0,0,0,0,0,"
         << system_total_energy << ",0\n";

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
    // Create base folder (cross-platform)
    createDirectory(base_folder);

    // Export coarsening mask (every step, as it may change with sliding)
    if (coarsening_enabled) {
        exportCoarseningMask(base_folder, step_number);
    }

    // Create subfolders
    std::string az_folder = base_folder + "/Az";
    std::string mu_folder = base_folder + "/Mu";
    std::string h_folder = base_folder + "/H";  // Magnetic field intensity |H| [A/m]
    std::string jz_folder = base_folder + "/Jz";
    std::string boundary_folder = base_folder + "/BoundaryImg";
    std::string forces_folder = base_folder + "/Forces";
    std::string input_image_folder = base_folder + "/InputImg";
    std::string energy_density_folder = base_folder + "/EnergyDensity";

    createDirectory(az_folder);
    createDirectory(mu_folder);
    createDirectory(h_folder);
    createDirectory(jz_folder);
    createDirectory(boundary_folder);
    createDirectory(forces_folder);
    createDirectory(input_image_folder);
    createDirectory(energy_density_folder);

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

    // Export H (magnetic field intensity magnitude) if available
    std::string h_path = h_folder + "/" + step_name + ".csv";
    exportHToCSV(h_path);

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

    // Export forces if available (using Amperian method - default)
    if (!force_results_amperian.empty()) {
        std::string forces_path = forces_folder + "/" + step_name + ".csv";
        exportForcesToCSV(forces_path);
    }

    // Export boundary stress vectors (commented out to reduce data size)
    // if (!boundary_stress_vectors.empty()) {
    //     std::string stress_vectors_folder = base_folder + "/StressVectors";
    //     system(("mkdir -p \"" + stress_vectors_folder + "\"").c_str());
    //     std::string stress_vectors_path = stress_vectors_folder + "/" + step_name + ".csv";
    //     exportBoundaryStressVectors(stress_vectors_path);
    // }

    // Export co-energy density distribution (magnetic co-energy W' = ∫B dH)
    // For current-source systems (Jz specified): F = +∂W'/∂x|_I
    if (coordinate_system == "cartesian" && Bx.size() > 0 && By.size() > 0 && mu_map.size() > 0) {
        int rows = Bx.rows();
        int cols = Bx.cols();

        // Export to CSV
        std::string energy_density_path = energy_density_folder + "/" + step_name + ".csv";
        std::ofstream file(energy_density_path);
        if (file.is_open()) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < cols; ++i) {
                    // Calculate co-energy density W' = ∫₀^H B(H') dH'
                    double B_mag = std::sqrt(Bx(j, i) * Bx(j, i) + By(j, i) * By(j, i));
                    double w = calculateCoEnergyDensity(j, i, B_mag);
                    file << w;
                    if (i < cols - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Co-energy density exported to: " << energy_density_path << std::endl;
        }
    } else if (coordinate_system == "polar" && Br.size() > 0 && Btheta.size() > 0 && mu_map.size() > 0) {
        int rows = Br.rows();
        int cols = Br.cols();

        // Export to CSV
        std::string energy_density_path = energy_density_folder + "/" + step_name + ".csv";
        std::ofstream file(energy_density_path);
        if (file.is_open()) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < cols; ++i) {
                    // Calculate co-energy density W' = ∫₀^H B(H') dH'
                    double B_mag = std::sqrt(Br(j, i) * Br(j, i) + Btheta(j, i) * Btheta(j, i));
                    double w = calculateCoEnergyDensity(j, i, B_mag);
                    file << w;
                    if (i < cols - 1) file << ",";
                }
                file << "\n";
            }
            file.close();
            std::cout << "Co-energy density (polar) exported to: " << energy_density_path << std::endl;
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

    // Determine boundary condition type from YAML settings
    // theta is periodic only if both theta_min and theta_max are set to "periodic"
    // Anti-periodic: type="periodic" with value < 0
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    double periodic_sign = is_antiperiodic ? -1.0 : 1.0;  // Sign for coupling across theta boundary

    // Build equation for each grid point
    for (int i = 0; i < nr; i++) {  // Radial direction
        for (int j = 0; j < ntheta; j++) {  // Angular direction
            int idx = i * ntheta + j;
            double r = r_coords[i];

            // Angular boundary conditions (only for non-periodic boundaries)
            if (!is_periodic) {
                // Handle theta_min boundary (j == 0)
                if (j == 0 && bc_theta_min.type == "dirichlet") {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_min.value;
                    continue;
                }
                // Handle theta_max boundary (j == ntheta - 1)
                if (j == ntheta - 1 && bc_theta_max.type == "dirichlet") {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_max.value;
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

            // Robin boundary conditions for radial direction
            // Inner boundary (i=0): outward normal is -r direction
            // dAz/dn = -dAz/dr ≈ (Az(0,j) - Az(1,j))/dr
            // => (alpha + beta/dr)*Az(0,j) - (beta/dr)*Az(1,j) = gamma
            if (i == 0 && bc_inner.type == "robin") {
                double a = bc_inner.alpha;
                double b = bc_inner.beta;
                double g = bc_inner.gamma;
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dr));
                triplets.push_back(Eigen::Triplet<double>(idx, 1 * ntheta + j, -b/dr));
                rhs(idx) = g;
                continue;
            }

            // Outer boundary (i=nr-1): outward normal is +r direction
            // dAz/dn = dAz/dr ≈ (Az(nr-1,j) - Az(nr-2,j))/dr
            // => (alpha + beta/dr)*Az(nr-1,j) - (beta/dr)*Az(nr-2,j) = gamma
            if (i == nr - 1 && bc_outer.type == "robin") {
                double a = bc_outer.alpha;
                double b = bc_outer.beta;
                double g = bc_outer.gamma;
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dr));
                triplets.push_back(Eigen::Triplet<double>(idx, (nr - 2) * ntheta + j, -b/dr));
                rhs(idx) = g;
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
                triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_prev_idx, sign * coeff_theta_prev));
            } else {
                // Theta_min neighbor is Dirichlet: move bc value to RHS
                rhs(idx) -= coeff_theta_prev * bc_theta_min.value;
            }

            if (!theta_next_is_dirichlet) {
                double sign = next_crosses_boundary ? periodic_sign : 1.0;
                triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_next_idx, sign * coeff_theta_next));
            } else {
                // Theta_max neighbor is Dirichlet: move bc value to RHS
                rhs(idx) -= coeff_theta_next * bc_theta_max.value;
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

    // Only show symmetry check in verbose mode, unless there's a problem
    if (nonlinear_config.verbose) {
        std::cout << "Matrix symmetry: ||A - A^T|| = " << symmetry_error
                  << ", relative error = " << relative_symmetry_error << std::endl;
    }

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

    // Check residual (debug output only if verbose)
    if (nonlinear_config.verbose) {
        Eigen::VectorXd res = A * Az_flat - rhs;
        double res_norm = res.norm();
        double rhs_norm = rhs.norm();
        std::cout << "[DBG] Residual norm ||A x - b|| = " << res_norm
                << ", relative = " << (rhs_norm>0 ? res_norm / rhs_norm : res_norm) << std::endl;
    }

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

double MagneticFieldAnalyzer::calculateThetaDistance(int j_from, int j_to) const {
    // Calculate theta-direction distance (periodic boundary aware)
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");

    int diff;
    if (is_periodic) {
        // Periodic: take shortest distance
        diff = j_to - j_from;
        if (diff < -ntheta / 2) diff += ntheta;
        if (diff > ntheta / 2) diff -= ntheta;
        diff = std::abs(diff);
    } else {
        diff = std::abs(j_to - j_from);
    }

    return diff * dtheta;
}

void MagneticFieldAnalyzer::buildMatrixPolarCoarsened(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs) {
    // Build FDM system matrix with coarsening (Polar coordinates, non-uniform stencil)
    A.resize(n_active_cells, n_active_cells);
    rhs.resize(n_active_cells);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(7 * n_active_cells);

    // Determine boundary condition type
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    double periodic_sign = is_antiperiodic ? -1.0 : 1.0;

    // Build equation for each active cell
    for (int idx = 0; idx < n_active_cells; idx++) {
        auto [i_r, j_theta] = coarse_to_fine[idx];  // {r, theta} order
        double r = r_coords[i_r];

        // Angular boundary conditions (non-periodic only)
        if (!is_periodic) {
            if (j_theta == 0 && bc_theta_min.type == "dirichlet") {
                triplets.push_back({idx, idx, 1.0});
                rhs(idx) = bc_theta_min.value;
                continue;
            }
            if (j_theta == ntheta - 1 && bc_theta_max.type == "dirichlet") {
                triplets.push_back({idx, idx, 1.0});
                rhs(idx) = bc_theta_max.value;
                continue;
            }
        }

        // Radial boundary conditions (Dirichlet)
        if (i_r == 0 && bc_inner.type == "dirichlet") {
            triplets.push_back({idx, idx, 1.0});
            rhs(idx) = bc_inner.value;
            continue;
        }
        if (i_r == nr - 1 && bc_outer.type == "dirichlet") {
            triplets.push_back({idx, idx, 1.0});
            rhs(idx) = bc_outer.value;
            continue;
        }

        // Robin boundary conditions for radial direction
        if (i_r == 0 && bc_inner.type == "robin") {
            double a = bc_inner.alpha;
            double b = bc_inner.beta;
            double g = bc_inner.gamma;

            // Find next active cell outward
            int i_next = findNextActiveRadial(i_r, j_theta, +1);
            auto it_next = fine_to_coarse.find({i_next, j_theta});
            if (it_next != fine_to_coarse.end()) {
                double h_plus = (i_next - i_r) * dr;
                triplets.push_back({idx, idx, a + b/h_plus});
                triplets.push_back({idx, it_next->second, -b/h_plus});
            }
            rhs(idx) = g;
            continue;
        }
        if (i_r == nr - 1 && bc_outer.type == "robin") {
            double a = bc_outer.alpha;
            double b = bc_outer.beta;
            double g = bc_outer.gamma;

            // Find next active cell inward
            int i_prev = findNextActiveRadial(i_r, j_theta, -1);
            auto it_prev = fine_to_coarse.find({i_prev, j_theta});
            if (it_prev != fine_to_coarse.end()) {
                double h_minus = (i_r - i_prev) * dr;
                triplets.push_back({idx, idx, a + b/h_minus});
                triplets.push_back({idx, it_prev->second, -b/h_minus});
            }
            rhs(idx) = g;
            continue;
        }

        // Interior points and Neumann boundary points
        // Polar equation: ∂/∂r(r·(1/μ)·∂Az/∂r) + (1/r)·∂/∂θ((1/μ)·∂Az/∂θ) = -r·Jz
        double coeff_center = 0.0;

        // --- Radial direction (non-uniform stencil) ---
        int i_prev = findNextActiveRadial(i_r, j_theta, -1);
        int i_next = findNextActiveRadial(i_r, j_theta, +1);

        double h_minus = (i_r - i_prev) * dr;
        double h_plus = (i_next - i_r) * dr;

        // Prevent division by zero
        if (h_minus < 1e-15) h_minus = dr;
        if (h_plus < 1e-15) h_plus = dr;

        // Interface positions (r-weighting)
        double r_imh = r - h_minus / 2.0;  // r_{i-1/2}
        double r_iph = r + h_plus / 2.0;   // r_{i+1/2}

        // Interface permeabilities
        double mu_inner = getMuAtInterfacePolar(i_r - 0.5, j_theta, "r");
        double mu_outer = getMuAtInterfacePolar(i_r + 0.5, j_theta, "r");

        // Non-uniform + r-weighted coefficients
        double a_im = r_imh / (mu_inner * h_minus * (h_minus + h_plus));
        double a_ip = r_iph / (mu_outer * h_plus * (h_minus + h_plus));

        // Handle Neumann boundaries with ghost elimination
        if (i_r == 0 && bc_inner.type == "neumann") {
            if (r_imh <= 0.0) {
                // Mirror approximation
                std::cerr << "Warning: r_imh <= 0 at inner Neumann BC, using mirror" << std::endl;
                double r_imh_eff = r_iph;
                double a_im_eff = r_imh_eff / (mu_inner * h_plus * (h_minus + h_plus));
                auto it_next = fine_to_coarse.find({i_next, j_theta});
                if (it_next != fine_to_coarse.end()) {
                    triplets.push_back({idx, it_next->second, a_im_eff + a_ip});
                }
                coeff_center -= (a_im_eff + a_ip);
            } else {
                auto it_next = fine_to_coarse.find({i_next, j_theta});
                if (it_next != fine_to_coarse.end()) {
                    triplets.push_back({idx, it_next->second, a_im + a_ip});
                }
                coeff_center -= (a_im + a_ip);
            }
        } else if (i_r == nr - 1 && bc_outer.type == "neumann") {
            auto it_prev = fine_to_coarse.find({i_prev, j_theta});
            if (it_prev != fine_to_coarse.end()) {
                triplets.push_back({idx, it_prev->second, a_im + a_ip});
            }
            coeff_center -= (a_im + a_ip);
        } else {
            // Standard interior stencil
            bool inner_neighbor_is_dirichlet = (i_r == 1) && (bc_inner.type == "dirichlet");
            bool outer_neighbor_is_dirichlet = (i_r == nr - 2) && (bc_outer.type == "dirichlet");

            auto it_prev = fine_to_coarse.find({i_prev, j_theta});
            auto it_next = fine_to_coarse.find({i_next, j_theta});

            if (!inner_neighbor_is_dirichlet && it_prev != fine_to_coarse.end()) {
                triplets.push_back({idx, it_prev->second, a_im});
            } else if (inner_neighbor_is_dirichlet) {
                rhs(idx) -= a_im * bc_inner.value;
            }

            if (!outer_neighbor_is_dirichlet && it_next != fine_to_coarse.end()) {
                triplets.push_back({idx, it_next->second, a_ip});
            } else if (outer_neighbor_is_dirichlet) {
                rhs(idx) -= a_ip * bc_outer.value;
            }

            coeff_center -= (a_im + a_ip);
        }

        // --- Theta direction (non-uniform stencil) ---
        int j_prev_theta = findNextActiveTheta(i_r, j_theta, -1);
        int j_next_theta = findNextActiveTheta(i_r, j_theta, +1);

        double h_theta_minus = calculateThetaDistance(j_theta, j_prev_theta);
        double h_theta_plus = calculateThetaDistance(j_next_theta, j_theta);

        // Prevent division by zero
        if (h_theta_minus < 1e-15) h_theta_minus = dtheta;
        if (h_theta_plus < 1e-15) h_theta_plus = dtheta;

        double mu_theta = getMuPolar(mu_map, i_r, j_theta, r_orientation);

        // Non-uniform theta stencil (1/r weight included)
        double a_theta_m = 1.0 / (r * mu_theta * h_theta_minus * (h_theta_minus + h_theta_plus));
        double a_theta_p = 1.0 / (r * mu_theta * h_theta_plus * (h_theta_minus + h_theta_plus));

        auto it_theta_prev = fine_to_coarse.find({i_r, j_prev_theta});
        auto it_theta_next = fine_to_coarse.find({i_r, j_next_theta});

        if (it_theta_prev != fine_to_coarse.end()) {
            // Check if crossing periodic boundary
            bool crosses_boundary = is_periodic &&
                ((j_theta == 0 && j_prev_theta > j_theta) ||
                 (j_theta > 0 && j_prev_theta > j_theta));
            double sign = crosses_boundary ? periodic_sign : 1.0;
            triplets.push_back({idx, it_theta_prev->second, sign * a_theta_m});
        }
        if (it_theta_next != fine_to_coarse.end()) {
            bool crosses_boundary = is_periodic &&
                ((j_theta == ntheta-1 && j_next_theta < j_theta) ||
                 (j_theta < ntheta-1 && j_next_theta < j_theta));
            double sign = crosses_boundary ? periodic_sign : 1.0;
            triplets.push_back({idx, it_theta_next->second, sign * a_theta_p});
        }
        coeff_center -= (a_theta_m + a_theta_p);

        // Center coefficient
        triplets.push_back({idx, idx, coeff_center});

        // Source term (r-weighted)
        double jz = getJzPolar(jz_map, i_r, j_theta, r_orientation);
        rhs(idx) -= r * jz;
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
}

void MagneticFieldAnalyzer::buildAndSolveSystemPolarCoarsened() {
    std::cout << "\n=== Building polar coarsened FDM system ===" << std::endl;
    std::cout << "Active cells: " << n_active_cells << " / " << (nr * ntheta)
              << " (reduction: " << std::fixed << std::setprecision(1)
              << (100.0 * (1.0 - double(n_active_cells) / (nr * ntheta))) << "%)" << std::endl;

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd rhs;
    buildMatrixPolarCoarsened(A, rhs);

    std::cout << "Coarsened matrix size: " << A.rows() << "x" << A.cols() << std::endl;
    std::cout << "Non-zero elements: " << A.nonZeros() << std::endl;

    // Check matrix symmetry
    Eigen::SparseMatrix<double> A_T = A.transpose();
    double symmetry_error = (A - A_T).norm();
    double A_norm = A.norm();
    double relative_symmetry_error = (A_norm > 1e-12) ? (symmetry_error / A_norm) : symmetry_error;

    std::cout << "Matrix symmetry: relative error = " << relative_symmetry_error << std::endl;
    if (relative_symmetry_error > 1e-8) {
        std::cerr << "Warning: Matrix not symmetric! Relative error = " << relative_symmetry_error << std::endl;
    }

    // Solve linear system
    std::cout << "\n=== Solving coarsened linear system ===" << std::endl;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Polar coarsened matrix decomposition failed!");
    }

    Eigen::VectorXd Az_coarse = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Polar coarsened solve failed!");
    }

    // Check residual
    if (nonlinear_config.verbose) {
        Eigen::VectorXd res = A * Az_coarse - rhs;
        double res_norm = res.norm();
        double rhs_norm = rhs.norm();
        std::cout << "[DBG] Residual norm ||A x - b|| = " << res_norm
                  << ", relative = " << (rhs_norm>0 ? res_norm / rhs_norm : res_norm) << std::endl;
    }

    // Interpolate to full grid
    interpolateToFullGridPolar(Az_coarse);

    std::cout << "Polar coarsened solve complete!" << std::endl;

    // Calculate magnetic field
    calculateMagneticFieldPolar();
}

void MagneticFieldAnalyzer::buildAndSolveCartesianPseudoPolar() {
    // Hybrid initialization for polar Newton-Krylov solver:
    // Interprets polar grid as Cartesian for faster initial guess.
    // Uses r-weighted current density to approximate polar source term.
    // This provides a better starting point than uniform initial guess
    // when r_in is close to r_out (thin annular domain).

    if (nonlinear_config.verbose) {
        std::cout << "\n=== Pseudo-Cartesian solve for polar initial guess ===" << std::endl;
        std::cout << "  r_avg = " << 0.5 * (r_start + r_end) << " m" << std::endl;
        std::cout << "  dx_pseudo = dr = " << dr << " m" << std::endl;
        std::cout << "  dy_pseudo = r_avg*dtheta = " << 0.5 * (r_start + r_end) * dtheta << " m" << std::endl;
    }

    int n = nr * ntheta;
    Eigen::SparseMatrix<double> A(n, n);
    Eigen::VectorXd rhs(n);
    rhs.setZero();

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n * 5);

    // Treat polar grid as Cartesian: r-direction = x, theta-direction = y
    // Use physical arc length for angular spacing: dy = r_avg * dtheta
    double r_avg = 0.5 * (r_start + r_end);
    double dx_pseudo = dr;
    double dy_pseudo = r_avg * dtheta;  // Physical arc length at average radius

    // Determine if theta is periodic
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);
    double periodic_sign = is_antiperiodic ? -1.0 : 1.0;

    // Build Cartesian-style FDM stencil
    for (int i = 0; i < nr; i++) {
        double r = r_coords[i];
        for (int j = 0; j < ntheta; j++) {
            int idx = i * ntheta + j;

            // Handle radial boundaries (Dirichlet)
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

            // Handle radial boundaries (Robin)
            // Inner boundary (i=0): outward normal is -r direction
            if (i == 0 && bc_inner.type == "robin") {
                double a = bc_inner.alpha;
                double b = bc_inner.beta;
                double g = bc_inner.gamma;
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dr));
                triplets.push_back(Eigen::Triplet<double>(idx, 1 * ntheta + j, -b/dr));
                rhs(idx) = g;
                continue;
            }
            // Outer boundary (i=nr-1): outward normal is +r direction
            if (i == nr - 1 && bc_outer.type == "robin") {
                double a = bc_outer.alpha;
                double b = bc_outer.beta;
                double g = bc_outer.gamma;
                triplets.push_back(Eigen::Triplet<double>(idx, idx, a + b/dr));
                triplets.push_back(Eigen::Triplet<double>(idx, (nr - 2) * ntheta + j, -b/dr));
                rhs(idx) = g;
                continue;
            }

            // Handle angular boundaries for non-periodic domains
            if (!is_periodic) {
                if (j == 0 && bc_theta_min.type == "dirichlet") {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_min.value;
                    continue;
                }
                if (j == ntheta - 1 && bc_theta_max.type == "dirichlet") {
                    triplets.push_back(Eigen::Triplet<double>(idx, idx, 1.0));
                    rhs(idx) = bc_theta_max.value;
                    continue;
                }
            }

            // Interior points: Cartesian Laplacian with harmonic-mean permeability
            double coeff_center = 0.0;

            // Get permeability values
            double mu_c = getMuPolar(mu_map, i, j, r_orientation);

            // Radial direction (i ± 1)
            if (i > 0 || bc_inner.type == "neumann") {
                int i_prev = (i > 0) ? i - 1 : i + 1;  // Neumann: mirror
                double mu_prev = getMuPolar(mu_map, i_prev, j, r_orientation);
                double mu_im = 2.0 / (1.0 / mu_c + 1.0 / mu_prev);
                double coeff_im = 1.0 / (mu_im * dx_pseudo * dx_pseudo);

                if (i > 0) {
                    bool inner_is_dirichlet = (i == 1) && (bc_inner.type == "dirichlet");
                    if (!inner_is_dirichlet) {
                        triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, coeff_im));
                    } else {
                        rhs(idx) -= coeff_im * bc_inner.value;
                    }
                } else {
                    // Neumann BC at inner: ghost elimination (Az_{-1} = Az_1)
                    triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, coeff_im));
                }
                coeff_center -= coeff_im;
            }

            if (i < nr - 1 || bc_outer.type == "neumann") {
                int i_next = (i < nr - 1) ? i + 1 : i - 1;  // Neumann: mirror
                double mu_next = getMuPolar(mu_map, i_next, j, r_orientation);
                double mu_ip = 2.0 / (1.0 / mu_c + 1.0 / mu_next);
                double coeff_ip = 1.0 / (mu_ip * dx_pseudo * dx_pseudo);

                if (i < nr - 1) {
                    bool outer_is_dirichlet = (i == nr - 2) && (bc_outer.type == "dirichlet");
                    if (!outer_is_dirichlet) {
                        triplets.push_back(Eigen::Triplet<double>(idx, (i + 1) * ntheta + j, coeff_ip));
                    } else {
                        rhs(idx) -= coeff_ip * bc_outer.value;
                    }
                } else {
                    // Neumann BC at outer: ghost elimination (Az_{nr} = Az_{nr-2})
                    triplets.push_back(Eigen::Triplet<double>(idx, (i - 1) * ntheta + j, coeff_ip));
                }
                coeff_center -= coeff_ip;
            }

            // Angular direction (j ± 1)
            int j_prev = (j - 1 + ntheta) % ntheta;
            int j_next = (j + 1) % ntheta;

            if (!is_periodic) {
                j_prev = j - 1;
                j_next = j + 1;
            }

            double mu_jp = getMuPolar(mu_map, i, j_prev, r_orientation);
            double mu_jn = getMuPolar(mu_map, i, j_next, r_orientation);
            double mu_jm = 2.0 / (1.0 / mu_c + 1.0 / mu_jp);
            double mu_jp2 = 2.0 / (1.0 / mu_c + 1.0 / mu_jn);
            double coeff_jm = 1.0 / (mu_jm * dy_pseudo * dy_pseudo);
            double coeff_jp = 1.0 / (mu_jp2 * dy_pseudo * dy_pseudo);

            bool prev_crosses = is_periodic && (j == 0);
            bool next_crosses = is_periodic && (j == ntheta - 1);

            bool theta_prev_is_dirichlet = (!is_periodic) && (j == 1) && (bc_theta_min.type == "dirichlet");
            bool theta_next_is_dirichlet = (!is_periodic) && (j == ntheta - 2) && (bc_theta_max.type == "dirichlet");

            if (j > 0 || is_periodic) {
                if (!theta_prev_is_dirichlet) {
                    double sign = prev_crosses ? periodic_sign : 1.0;
                    triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_prev, sign * coeff_jm));
                } else {
                    rhs(idx) -= coeff_jm * bc_theta_min.value;
                }
                coeff_center -= coeff_jm;
            }

            if (j < ntheta - 1 || is_periodic) {
                if (!theta_next_is_dirichlet) {
                    double sign = next_crosses ? periodic_sign : 1.0;
                    triplets.push_back(Eigen::Triplet<double>(idx, i * ntheta + j_next, sign * coeff_jp));
                } else {
                    rhs(idx) -= coeff_jp * bc_theta_max.value;
                }
                coeff_center -= coeff_jp;
            }

            // Center coefficient
            triplets.push_back(Eigen::Triplet<double>(idx, idx, coeff_center));

            // RHS: pure Cartesian formulation (no r-weighting)
            // This gives a physically consistent pseudo-Cartesian approximation
            rhs(idx) = -getJzPolar(jz_map, i, j, r_orientation);
        }
    }

    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    // Solve using SparseLU
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Warning: Pseudo-Cartesian factorization failed, using zero initial guess" << std::endl;
        Az = Eigen::MatrixXd::Zero(r_orientation == "horizontal" ? ntheta : nr,
                                    r_orientation == "horizontal" ? nr : ntheta);
        return;
    }

    Eigen::VectorXd Az_flat = solver.solve(rhs);

    if (solver.info() != Eigen::Success) {
        std::cerr << "Warning: Pseudo-Cartesian solve failed, using zero initial guess" << std::endl;
        Az = Eigen::MatrixXd::Zero(r_orientation == "horizontal" ? ntheta : nr,
                                    r_orientation == "horizontal" ? nr : ntheta);
        return;
    }

    // Store result in Az matrix (same format as buildAndSolveSystemPolar)
    if (r_orientation == "horizontal") {
        Az.resize(ntheta, nr);
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < ntheta; j++) {
                Az(j, i) = Az_flat(i * ntheta + j);
            }
        }
    } else {
        Az.resize(nr, ntheta);
        for (int i = 0; i < nr; i++) {
            for (int j = 0; j < ntheta; j++) {
                Az(i, j) = Az_flat(i * ntheta + j);
            }
        }
    }

    if (nonlinear_config.verbose) {
        std::cout << "Pseudo-Cartesian initial guess computed (||Az|| = "
                  << Az.norm() << ")" << std::endl;
    }
}

void MagneticFieldAnalyzer::calculateMagneticFieldPolar() {
    // Removed verbose output - called frequently in nonlinear iterations

    // Allocate field arrays in image-compatible format
    if (r_orientation == "horizontal") {
        Br = Eigen::MatrixXd::Zero(ntheta, nr);
        Btheta = Eigen::MatrixXd::Zero(ntheta, nr);
    } else {
        Br = Eigen::MatrixXd::Zero(nr, ntheta);
        Btheta = Eigen::MatrixXd::Zero(nr, ntheta);
    }

    // Determine if theta is periodic from boundary condition settings
    // theta is periodic only if both theta_min and theta_max are set to "periodic"
    // Anti-periodic: type="periodic" with value < 0
    bool is_periodic = (bc_theta_min.type == "periodic" && bc_theta_max.type == "periodic");
    bool is_antiperiodic = is_periodic && (bc_theta_min.value < 0 || bc_theta_max.value < 0);

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

            // Get Az values at neighbors
            double Az_prev = getAz(i, j_prev);
            double Az_next = getAz(i, j_next);

            // For anti-periodic BC, apply sign flip when crossing the theta boundary
            if (is_antiperiodic) {
                if (j == 0) {
                    // j_prev = ntheta-1 crosses the boundary
                    Az_prev *= -1.0;
                }
                if (j == ntheta - 1) {
                    // j_next = 0 crosses the boundary
                    Az_next *= -1.0;
                }
            }

            double dAz_dtheta = (Az_next - Az_prev) / denom_theta;
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
    // Debug output removed - called frequently in nonlinear iterations
    // if (nonlinear_config.verbose) {
    //     std::cout << "[DBG] Max |B| = " << bmax << std::endl;
    // }
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

            // Validate coordinate system-specific variables
            bool has_dx_dy = (jz_str.find("$dx") != std::string::npos ||
                              jz_str.find("$dy") != std::string::npos);
            bool has_dr_dtheta = (jz_str.find("$dr") != std::string::npos ||
                                  jz_str.find("$dtheta") != std::string::npos);

            if (has_dx_dy && coordinate_system == "polar") {
                throw std::runtime_error("Jz formula error: $dx, $dy can only be used in Cartesian coordinates. "
                                         "Use $dr, $dtheta for polar coordinates.");
            }
            if (has_dr_dtheta && coordinate_system == "cartesian") {
                throw std::runtime_error("Jz formula error: $dr, $dtheta can only be used in polar coordinates. "
                                         "Use $dx, $dy for Cartesian coordinates.");
            }

            // Replace formula variables with tinyexpr-compatible names
            // Order matters: longer names first to avoid partial replacements
            size_t pos = 0;
            while ((pos = result.formula.find("$dtheta", pos)) != std::string::npos) {
                result.formula.replace(pos, 7, "dtheta");
                pos += 6;
            }
            pos = 0;
            while ((pos = result.formula.find("$step", pos)) != std::string::npos) {
                result.formula.replace(pos, 5, "step");
                pos += 4;
            }
            pos = 0;
            while ((pos = result.formula.find("$dx", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dx");
                pos += 2;
            }
            pos = 0;
            while ((pos = result.formula.find("$dy", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dy");
                pos += 2;
            }
            pos = 0;
            while ((pos = result.formula.find("$dr", pos)) != std::string::npos) {
                result.formula.replace(pos, 3, "dr");
                pos += 2;
            }
            pos = 0;
            while ((pos = result.formula.find("$N", pos)) != std::string::npos) {
                result.formula.replace(pos, 2, "N");
                pos += 1;
            }
            pos = 0;
            while ((pos = result.formula.find("$A", pos)) != std::string::npos) {
                result.formula.replace(pos, 2, "A");
                pos += 1;
            }

            // Replace user-defined variables (sorted by name length descending to avoid partial replacements)
            std::vector<std::pair<std::string, double>> sorted_vars(user_variables.begin(), user_variables.end());
            std::sort(sorted_vars.begin(), sorted_vars.end(),
                      [](const auto& a, const auto& b) { return a.first.length() > b.first.length(); });

            for (const auto& [var_name, var_value] : sorted_vars) {
                std::string search_str = "$" + var_name;
                pos = 0;
                while ((pos = result.formula.find(search_str, pos)) != std::string::npos) {
                    result.formula.replace(pos, search_str.length(), var_name);
                    pos += var_name.length();
                }
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

double MagneticFieldAnalyzer::evaluateJz(const JzValue& jz_val, int step, const std::string& material_name) {
    switch (jz_val.type) {
        case JzType::STATIC:
            return jz_val.static_value;

        case JzType::FORMULA: {
            te_parser parser;

            // Prepare variables
            std::set<te_variable> vars;

            // step variable
            te_variable step_var;
            step_var.m_name = "step";
            step_var.m_value = static_cast<double>(step);
            vars.insert(step_var);

            // Coordinate system-specific variables
            if (coordinate_system == "cartesian") {
                te_variable dx_var, dy_var;
                dx_var.m_name = "dx";
                dx_var.m_value = dx;
                dy_var.m_name = "dy";
                dy_var.m_value = dy;
                vars.insert(dx_var);
                vars.insert(dy_var);
            } else {  // polar
                te_variable dr_var, dtheta_var;
                dr_var.m_name = "dr";
                dr_var.m_value = dr;
                dtheta_var.m_name = "dtheta";
                dtheta_var.m_value = dtheta;
                vars.insert(dr_var);
                vars.insert(dtheta_var);
            }

            // Material pixel info variables (N, A)
            te_variable n_var, a_var;
            n_var.m_name = "N";
            a_var.m_name = "A";

            if (!material_name.empty() && material_pixel_info.find(material_name) != material_pixel_info.end()) {
                const auto& pix_info = material_pixel_info.at(material_name);
                n_var.m_value = static_cast<double>(pix_info.pixel_count);
                a_var.m_value = pix_info.area;
            } else {
                // Fallback values if material not found
                n_var.m_value = 0.0;
                a_var.m_value = 0.0;
            }
            vars.insert(n_var);
            vars.insert(a_var);

            // User-defined variables
            for (const auto& [var_name, var_value] : user_variables) {
                te_variable user_var;
                user_var.m_name = var_name.c_str();
                user_var.m_value = var_value;
                vars.insert(user_var);
            }

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

    // IMPORTANT: Reset mu_map and jz_map to default values (air) before applying materials
    // This ensures that pixels that changed material due to sliding get updated correctly
    // Without this reset, old material values would persist even after the image slides
    mu_map.setConstant(MU_0);  // Default: air (mu_r = 1.0)
    jz_map.setZero();          // Default: no current

    for (const auto& material : config["materials"]) {
        std::string name = material.first.as<std::string>();
        YAML::Node props = material.second;

        // Resolve preset if specified (preset properties are merged, material-specific overrides)
        if (props["preset"]) {
            std::string preset_name = props["preset"].as<std::string>();
            auto preset_it = material_presets.find(preset_name);
            if (preset_it != material_presets.end()) {
                YAML::Node merged = YAML::Clone(preset_it->second);
                for (auto it = props.begin(); it != props.end(); ++it) {
                    std::string key = it->first.as<std::string>();
                    if (key != "preset") {
                        merged[key] = it->second;
                    }
                }
                props = merged;
            }
        }

        // Get RGB values
        std::vector<int> rgb = props["rgb"].as<std::vector<int>>(std::vector<int>{255, 255, 255});

        // Get relative permeability
        double mu_r = props["mu_r"].as<double>(1.0);

        // Evaluate Jz for this step (0.0 if not defined)
        double jz = 0.0;
        if (material_jz.find(name) != material_jz.end()) {
            jz = evaluateJz(material_jz[name], step, name);
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

    // Regenerate coarsening mask if sliding is enabled (material boundaries may have moved)
    if (coarsening_enabled && transient_config.enable_sliding && step > 0) {
        generateCoarseningMask();
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

            // Circular shift in y direction (upward in FDM coordinates = positive y)
            // In image coordinates: content moves toward smaller row numbers (top of image)
            // In FDM coordinates (y-flipped): this corresponds to positive y direction
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

            // Circular shift in x direction (rightward = positive x in physics coordinates)
            // Positive shift moves content to the RIGHT (positive x direction)
            for (int col = 0; col < image.cols; col++) {
                int src_col = (col - shift + image.cols) % image.cols;
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

        // 2. Solve FDM system
        auto solve_start = std::chrono::high_resolution_clock::now();

        // Check if nonlinear solver is needed for this step
        if (has_nonlinear_materials && nonlinear_config.enabled) {
            // For nonlinear materials, use standard solve() which includes nonlinear iteration
            // This ensures mu_map is updated based on actual H-field at each transient step
            solve();

            // Update previous_solution for warm start in next step
            int n = (coordinate_system == "cartesian") ? (nx * ny) : (nr * ntheta);
            previous_solution.resize(n);

            if (coordinate_system == "cartesian") {
                for (int j = 0; j < ny; j++) {
                    for (int i = 0; i < nx; i++) {
                        previous_solution[j * nx + i] = Az(j, i);
                    }
                }
            } else {  // polar
                if (r_orientation == "horizontal") {
                    // Az(θ, r) → flat[r * ntheta + θ]
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            previous_solution[i * ntheta + j] = Az(j, i);  // Transpose
                        }
                    }
                } else {  // vertical
                    // Az(r, θ) → flat[r * ntheta + θ]
                    for (int i = 0; i < nr; i++) {
                        for (int j = 0; j < ntheta; j++) {
                            previous_solution[i * ntheta + j] = Az(i, j);
                        }
                    }
                }
            }
        } else if (use_optimized_solver) {
            // Optimized linear path for transient analysis (when no nonlinear materials)
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

        // 3. Calculate force using Distributed Amperian Force method (DEFAULT)
        // This method uses bound current from magnetization: J_b = ∇×M, F = ∫J_b × B dV
        // Key advantage: No ghost force (M = 0 exactly in air where μ_r = 1)
        // This is the recommended and default force calculation method.
        calculateForceDistributedAmperian(step, 0.0);  // No smoothing (sigma = 0)

        // 3.1. Calculate total magnetic energy (reuses calculated field from step)
        double total_energy = calculateTotalMagneticEnergy(step);
        std::cout << "Step " << step+1 << " Total Magnetic Energy: " << total_energy << " J/m" << std::endl;

        // 3.2. Calculate flux linkage for all defined paths
        calculateAllFluxLinkages(step);

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

    // Export flux linkage results to CSV (if any paths were defined)
    exportFluxLinkageCSV(output_dir);
}
