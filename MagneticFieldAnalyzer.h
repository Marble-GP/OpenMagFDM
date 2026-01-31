#ifndef MAGNETICFIELDANALYZER_H
#define MAGNETICFIELDANALYZER_H

// Define _USE_MATH_DEFINES before cmath for M_PI on Windows MSVC
#define _USE_MATH_DEFINES
#include <cmath>

// Define M_PI if not available (Windows MSVC compatibility)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Sparse>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <string>
#include <map>
#include <vector>
#include <variant>
#include <tinyexpr.h>

// AMGCL headers for advanced iterative solvers
#include <amgcl/backend/eigen.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/eigen.hpp>

#define SOLVER_TOLERANCE (1e-6)
#define SOLVER_MAX_ITERATIONS (5000)

/**
 * @brief 2D Magnetic Field Analyzer using Finite Difference Method
 *
 * This class implements magnetic field analysis using vector potential method
 * in Cartesian coordinate system.
 */
class MagneticFieldAnalyzer {
public:
    /**
     * @brief Constructor
     * @param config_path Path to YAML configuration file
     * @param image_path Path to material image file
     */
    MagneticFieldAnalyzer(const std::string& config_path, const std::string& image_path);

    /**
     * @brief Solve the FDM equation system
     */
    void solve();

    /**
     * @brief Export all results to folder structure
     * @param base_folder Base folder name (e.g., "output")
     * @param step_number Step number for transient analysis (default: 0)
     */
    void exportResults(const std::string& base_folder, int step_number = 0);

    /**
     * @brief Perform transient analysis with image sliding
     * @param output_dir Output directory for results
     */
    void performTransientAnalysis(const std::string& output_dir);

    /**
     * @brief Export Az (vector potential) array to CSV file
     * @param output_path Output CSV file path
     */
    void exportAzToCSV(const std::string& output_path) const;

    /**
     * @brief Export permeability distribution to CSV file
     * @param output_path Output CSV file path
     */
    void exportMuToCSV(const std::string& output_path) const;

    /**
     * @brief Export magnetic field intensity |H| distribution to CSV file
     * @param output_path Output CSV file path
     * @note Only available after nonlinear solve (H_map is populated)
     */
    void exportHToCSV(const std::string& output_path) const;

    /**
     * @brief Export current density distribution to CSV file
     * @param output_path Output CSV file path
     */
    void exportJzToCSV(const std::string& output_path) const;

    /**
     * @brief Export boundary detection visualization
     * @param output_path Output image file path
     */
    void exportBoundaryImage(const std::string& output_path) const;

    /**
     * @brief Get the Az matrix (vector potential)
     * @return Eigen matrix containing vector potential values
     */
    const Eigen::MatrixXd& getAz() const { return Az; }

    /**
     * @brief Get the permeability distribution
     * @return Eigen matrix containing permeability values [H/m]
     */
    const Eigen::MatrixXd& getMu() const { return mu_map; }

    /**
     * @brief Calculate Maxwell stress and electromagnetic forces (Sobel-based)
     * @param step Step number for field caching (-1 for static analysis)
     * @deprecated Use calculateForceDistributedAmperian() instead (default method)
     */
    void calculateMaxwellStress(int step = -1);

    /**
     * @brief Calculate Maxwell stress using edge-based integration (more robust)
     * @param step Step number for field caching (-1 for static analysis)
     * @deprecated Use calculateForceDistributedAmperian() instead (default method)
     *
     * This method uses cell-edge-based integration which is more accurate
     * than Sobel-based normal estimation, especially for:
     * - Uniform permeability materials (should give zero force)
     * - Boundaries crossing periodic boundaries
     * - Rectangular/axis-aligned geometries
     */
    void calculateMaxwellStressEdgeBased(int step = -1);

    /**
     * @brief Calculate electromagnetic force using volume integral method
     * @param step Step number for field caching (-1 for static analysis)
     * @deprecated Use calculateForceDistributedAmperian() instead (default method)
     *
     * This method uses volume integral of force density:
     *   f = J×B + (M·∇)B
     * where M = (μr - 1)·H is the magnetization.
     *
     * Advantages over surface integral (Maxwell stress tensor):
     * - No boundary normal vector evaluation required
     * - More robust against jaggy boundaries
     * - Naturally handles permeability discontinuities
     * - Numerically more stable
     *
     * Results are stored in force_results_volume member.
     */
    void calculateForceVolumeIntegral(int step = -1);

    /**
     * @brief Calculate electromagnetic force using face-flux method (Maxwell stress divergence)
     * @param step Step number for field caching (-1 for static analysis)
     * @deprecated Use calculateForceDistributedAmperian() instead (default method)
     *
     * This method uses face-flux discretization of Maxwell stress tensor divergence:
     *   F = ∫_V ∇·T dV = Σ_faces (T·n) * A_face
     * where T = B⊗H - (1/2)(B·H)I is the Maxwell stress tensor.
     *
     * Advantages:
     * - Discrete divergence theorem is satisfied (conservation)
     * - Face-averaged fluxes reduce boundary noise from jaggy geometries
     * - Theoretically consistent with energy method
     * - Automatically zero for uniform field
     *
     * Results are stored in force_results_flux member.
     */
    void calculateForceMaxwellStressFaceFlux(int step = -1);

    /**
     * @brief Calculate electromagnetic force using Shell Volume Integration method
     * @param step Step number for field caching (-1 for static analysis)
     * @param shell_thickness Number of pixels for shell thickness (default: 3)
     * @deprecated Use calculateForceDistributedAmperian() instead (default method)
     *
     * This method uses weighted volume integration in the air shell surrounding
     * the material, avoiding direct boundary calculations:
     *   F = ∫_Ω_shell T · ∇G dS
     * where G is a smooth weight function (1 at material surface, 0 at shell outer edge).
     *
     * Key advantages:
     * - Avoids jaggy boundary normal vector evaluation
     * - All calculations in air (μ₀), avoiding material discontinuities
     * - Spatial averaging reduces numerical noise
     * - Uses image processing (morphology, distance transform) for robust shell generation
     *
     * Mathematically equivalent to surface integral via divergence theorem,
     * but numerically more stable.
     *
     * Results are stored in force_results_shell member.
     */
    void calculateForceShellIntegration(int step = -1, int shell_thickness = 3);

    /**
     * @brief [DEFAULT] Calculate electromagnetic force using Distributed Amperian Force method
     * @param step Step number for field caching (-1 for static analysis)
     * @param sigma_smooth Gaussian smoothing sigma for magnetization (default: 0.0 = no smoothing)
     *
     * THIS IS THE RECOMMENDED AND DEFAULT FORCE CALCULATION METHOD.
     *
     * This method converts magnetization M to equivalent bound current and uses Lorentz force:
     *   M = B/μ₀ - H (magnetization, exactly 0 in air where μ_r = 1)
     *   J_b = ∇ × M (bound current from curl of magnetization)
     *   F = ∫ J_b × B dV (Lorentz force on bound current)
     *
     * Key advantages over other methods:
     * - NO ghost force: M = 0 exactly in air (μ_r = 1), so J_b = 0 in air
     * - Surface magnetization current automatically captured via numerical curl
     * - Optional Gaussian smoothing reduces numerical noise while preserving physics
     * - Avoids jaggy boundary issues inherent to surface integral methods
     * - Robust for complex geometries with multiple materials
     * - Results closely match Virtual Work principle (energy-based) method
     *
     * For 2D (z-invariant): J_bz = ∂My/∂x - ∂Mx/∂y
     *                       F = J × B where J = (0, 0, Jz), B = (Bx, By, 0)
     *                       => Fx = -Jz·By, Fy = +Jz·Bx
     *
     * Physics note: For constant-current sources (Jz specified), the force is
     * F = +∂W'/∂x|_I (co-energy derivative). For linear materials W' = W.
     *
     * Note on nonlinear materials: The Amperian method F = J × B is robust because
     * it only depends on J and B, not on the B(H) constitutive relation. The code
     * uses secant permeability μr := B/(H·μ0), so B = μr·μ0·H at each point.
     * Virtual Work now computes true co-energy W' = ∫B dH for nonlinear materials.
     *
     * Results are stored in force_results_amperian member.
     */
    void calculateForceDistributedAmperian(int step = -1, double sigma_smooth = 0.0);

    /**
     * @brief Calculate total magnetic co-energy of the system
     *
     * For current-source systems (Jz specified): F = +∂W'/∂x|_I
     * Co-energy W' = ∫B dH (for nonlinear materials, Simpson integration)
     * For linear materials: W' = W = B²/(2μ)
     *
     * @param step Step number for field caching (-1 for static analysis)
     * @return Total magnetic co-energy [J/m] (per unit depth)
     */
    double calculateTotalMagneticEnergy(int step = -1);

    /**
     * @brief Export force results to CSV file
     * @param output_path Output CSV file path
     */
    void exportForcesToCSV(const std::string& output_path) const;

    /**
     * @brief Export boundary stress vectors to CSV file for visualization
     * @param output_path Output CSV file path
     */
    void exportBoundaryStressVectors(const std::string& output_path) const;

    // Structure to hold sampled field values at a physical point (public for method return type)
    struct PolarSample {
        double x_phys, y_phys;       // Cartesian physical coordinates
        double r_phys, theta_phys;   // Polar physical coordinates
        double Bx, By;               // Cartesian B components
        double Br, Btheta;           // Polar B components (for reference)
        double mu;                   // Permeability at this point
    };

    // Nonlinear permeability representation (public for Newton-Krylov access)
    enum class MuType {
        STATIC,   // Constant mu_r value (linear material)
        FORMULA,  // Mathematical expression with $H variable (|H| in A/m)
        TABLE     // Table [H_values, mu_r_values] with linear interpolation
    };

    struct MuValue {
        MuType type;
        double static_value;           // For STATIC type (mu_r)
        std::string formula;           // For FORMULA type (mu_r as function of $H)
        std::vector<double> H_table;   // For TABLE type: |H| values [A/m] (must be monotonically increasing)
        std::vector<double> mu_table;  // For TABLE type: mu_r values (recommended monotonically decreasing)

        // Extrapolation for differential permeability dμ_r/dH (outside table domain)
        bool has_dmu_extrapolation;     // True if user specified extrapolation
        double dmu_r_extrap_const;      // Constant extrapolation value (default: 1.0)
        std::string dmu_r_extrap_formula; // Formula for dμ_r/dH(H) extrapolation

        MuValue() : type(MuType::STATIC), static_value(1.0),
                    has_dmu_extrapolation(false), dmu_r_extrap_const(1.0),
                    dmu_r_extrap_formula("") {}
    };

    // B-H relationship tables (generated from mu_r(H))
    struct BHTable {
        std::vector<double> H_values;   // |H| [A/m]
        std::vector<double> B_values;   // |B| [T]
        std::vector<double> mu_values;  // μ [H/m] = μ_r * μ_0

        // Cached for fast interpolation
        bool is_valid;

        BHTable() : is_valid(false) {}
    };

    /**
     * @brief Evaluate effective permeability μ_eff = B/H at given |H| magnitude
     * @param mu_val Nonlinear permeability specification (μ_eff table from catalog)
     * @param H_magnitude Magnetic field intensity |H| [A/m]
     * @return Effective permeability μ_eff = B/H (dimensionless)
     */
    double evaluateMu(const MuValue& mu_val, double H_magnitude);

    /**
     * @brief Evaluate derivative dμ_r/dH at given |H| magnitude
     * @param mu_val Nonlinear permeability specification
     * @param H_magnitude Magnetic field intensity |H| [A/m]
     * @return Derivative dμ_r/dH [m/A] (needed for Newton-Krylov Jacobian)
     */
    double evaluateMuDerivative(const MuValue& mu_val, double H_magnitude);

private:
    // Dynamic current density representation
    enum class JzType {
        STATIC,   // Constant value
        FORMULA,  // Mathematical expression with $step variable
        ARRAY     // Array of values indexed by step
    };

    struct JzValue {
        JzType type;
        double static_value;           // For STATIC type
        std::string formula;           // For FORMULA type
        std::vector<double> array;     // For ARRAY type

        JzValue() : type(JzType::STATIC), static_value(0.0) {}
    };

    // Material pixel information (for formula variables $N, $A)
    struct MaterialPixelInfo {
        int pixel_count;        // Number of pixels (N)
        double area;            // Physical cross-sectional area [m²] (A = N * cell_area)

        MaterialPixelInfo() : pixel_count(0), area(0.0) {}
    };

    // Anti-aliasing material information (for gradient pixel interpolation)
    struct AntialiasableMaterial {
        std::string name;       // Material name
        cv::Vec3b rgb;          // Material RGB color
        double mu_r;            // Relative permeability (linear or last evaluated)
    };

    // Anderson acceleration configuration (shared by Picard and Newton-Krylov)
    struct AndersonConfig {
        bool enabled;       // Enable Anderson acceleration (default: false)
        int depth;          // History depth (default: 5)
        double beta;        // Mixing parameter (default: 1.0)

        AndersonConfig() : enabled(false), depth(5), beta(1.0) {}
    };

    // Nonlinear solver configuration
    struct NonlinearSolverConfig {
        bool enabled;               // Enable nonlinear solver (default: true, used with has_nonlinear_materials)
        std::string solver_type;    // Solver type: "picard", "newton-krylov" (default: "newton-krylov")
        int max_iterations;         // Maximum nonlinear iterations (default: 50)
        double tolerance;           // Convergence tolerance (relative) (default: 5e-4)
        double relaxation;          // Relaxation factor (0.5 ~ 0.8) (default: 0.7) - for Picard
        AndersonConfig anderson;    // Anderson acceleration settings (for Picard and Newton-Krylov)
        int gmres_restart;          // GMRES restart parameter (default: 30) - for Newton-Krylov
        double line_search_c;       // Line search Armijo parameter (default: 1e-4) - for Newton-Krylov
        double line_search_alpha_init;    // Initial step length (default: 1.0) - for Newton-Krylov
        double line_search_alpha_min;     // Minimum step length (default: 1e-3) - for Newton-Krylov
        double line_search_rho;           // Backtracking factor (default: 0.65) - for Newton-Krylov
        int line_search_max_trials;       // Maximum line search trials (default: 50) - for Newton-Krylov
        bool line_search_adaptive;        // Use adaptive initial step length (default: true) - for Newton-Krylov
        bool verbose;               // Print iteration details (default: false)
        bool export_convergence;    // Export convergence history (default: false)

        NonlinearSolverConfig() :
            enabled(true), solver_type("newton-krylov"), max_iterations(50), tolerance(5e-4),
            relaxation(0.7), anderson(), gmres_restart(30), line_search_c(1e-4),
            line_search_alpha_init(1.0), line_search_alpha_min(1e-3), line_search_rho(0.65),
            line_search_max_trials(50), line_search_adaptive(true),
            verbose(false), export_convergence(false) {}
    };

    // Maxwell stress and force calculation
    struct ForceResult {
        std::string material_name;
        cv::Scalar rgb;
        double force_x;        // Force in X direction [N/m]
        double force_y;        // Force in Y direction [N/m]
        double force_radial;   // Radial force (outward) [N/m] - for polar coordinates
        double torque;         // Torque (backward compatibility, equals torque_origin)
        double torque_origin;  // Torque about origin [N] (per unit depth)
        double torque_center;  // Torque about image center [N] (per unit depth)
        int pixel_count;
        double magnetic_energy; // Magnetic potential energy [J/m] (per unit depth)
    };

    // Boundary stress vector for visualization
    struct BoundaryStressPoint {
        int i_pixel;          // Pixel coordinate i (analysis coords, flipped)
        int j_pixel;          // Pixel coordinate j (analysis coords, flipped)
        double x_phys;        // Physical x coordinate [m]
        double y_phys;        // Physical y coordinate [m]
        double fx;            // Force per unit length in x [N/m]
        double fy;            // Force per unit length in y [N/m]
        double ds;            // Boundary segment length [m]
        double nx;            // Normal vector x component (outward)
        double ny;            // Normal vector y component (outward)
        double Bx;            // Magnetic field x component [T]
        double By;            // Magnetic field y component [T]
        double B_magnitude;   // |B| [T]
        std::string material; // Material name
    };

    std::vector<ForceResult> force_results;           // Results from surface integral (Maxwell stress)
    std::vector<ForceResult> force_results_volume;    // Results from volume integral (f = J×B + (M·∇)B)
    std::vector<ForceResult> force_results_flux;      // Results from face-flux method (∇·T with T=B⊗H-(1/2)(B·H)I)
    std::vector<ForceResult> force_results_shell;     // Results from shell volume integration (T·∇G in air shell)
    std::vector<ForceResult> force_results_amperian;  // Results from distributed Amperian force (J_b × B)
    std::vector<BoundaryStressPoint> boundary_stress_vectors;  // Stress vectors at boundaries
    cv::Mat boundary_image;  // Cached boundary detection visualization
    double system_total_energy;  // Total magnetic energy of the entire system [J/m]

    // Flux linkage calculation path
    struct FluxLinkagePath {
        std::string name;           // Path identifier (e.g., "coil_A")
        double x_start, y_start;    // Start point [m] (physical coordinates)
        double x_end, y_end;        // End point [m] (physical coordinates)
    };

    // Flux linkage calculation
    std::vector<FluxLinkagePath> flux_linkage_paths;  // Defined paths for flux linkage
    std::map<std::string, std::vector<double>> flux_linkage_results;  // Results per path per step

    // Boundary detection optimization for transient analysis (incremental update)
    cv::Mat cached_boundaries;  // Cached boundary detection result (binary mask)
    bool boundary_cache_valid;  // Whether the cache is valid

    // Boundary conditions structure
    struct BoundaryCondition {
        std::string type;     // "dirichlet", "neumann", "periodic", or "robin"
        double value;         // Boundary value for Dirichlet

        // Robin BC parameters: alpha*Az + beta*(dAz/dn) = gamma
        double alpha;         // Coefficient for Az (default: 1.0)
        double beta;          // Coefficient for dAz/dn (default: 0.0)
        double gamma;         // RHS value (default: 0.0)

        BoundaryCondition() : type("dirichlet"), value(0.0),
                              alpha(1.0), beta(0.0), gamma(0.0) {}
    };

    // Configuration and input
    YAML::Node config;
    cv::Mat image;
    std::string coordinate_system;  // "cartesian" or "polar"

    // Mesh parameters (Cartesian)
    int nx, ny;        // Number of grid points
    double dx, dy;     // Mesh spacing

    // Polar coordinate parameters
    int nr, ntheta;    // Number of grid points (radial, angular)
    double dr, dtheta; // Mesh spacing
    double r_start, r_end;  // Radial domain
    double theta_range;     // Angular range [rad] (default: 2*pi, sector: pi/2, etc.)
    std::string r_orientation;  // "horizontal" or "vertical"
    std::vector<double> r_coords;  // Radial coordinates

    // Boundary conditions
    BoundaryCondition bc_left, bc_right, bc_bottom, bc_top;  // Cartesian
    BoundaryCondition bc_inner, bc_outer;  // Polar (radial direction)
    BoundaryCondition bc_theta_min, bc_theta_max;  // Polar (angular direction)

    // Transient analysis configuration
    struct TransientConfig {
        bool enabled;
        bool enable_sliding;          // Enable/disable image sliding
        int total_steps;
        std::string slide_direction;  // "vertical" or "horizontal"
        int slide_region_start;       // Pixel position (x for vertical, y for horizontal)
        int slide_region_end;         // Pixel position (x for vertical, y for horizontal)
        int slide_pixels_per_step;    // Pixels to shift per step

        TransientConfig() : enabled(false), enable_sliding(true), total_steps(0),
                           slide_direction("vertical"), slide_region_start(0),
                           slide_region_end(0), slide_pixels_per_step(0) {}
    };

    TransientConfig transient_config;

    // Material properties
    Eigen::MatrixXd mu_map;   // Permeability distribution (updated during nonlinear iteration)
    Eigen::MatrixXd jz_map;   // Current density distribution
    std::map<std::string, JzValue> material_jz;  // Dynamic Jz values per material
    std::map<std::string, MuValue> material_mu;  // Nonlinear permeability values per material
    std::map<std::string, BHTable> material_bh_tables;  // B-H tables per nonlinear material
    std::map<std::string, MaterialPixelInfo> material_pixel_info;  // Pixel count and area per material
    std::vector<AntialiasableMaterial> antialias_materials;  // Materials with antialias enabled
    std::map<std::string, YAML::Node> material_presets;  // Material presets (reusable B-H curves/properties)

    // Adaptive mesh coarsening configuration
    struct CoarsenConfig {
        bool enabled;       // Enable coarsening for this material
        int ratio;          // Coarsening ratio (area reduction factor)
        int skip_x;         // Actual skip ratio in x/r direction (auto-calculated)
        int skip_y;         // Actual skip ratio in y/theta direction (auto-calculated)

        CoarsenConfig() : enabled(false), ratio(1), skip_x(1), skip_y(1) {}
    };
    std::map<std::string, CoarsenConfig> material_coarsen;  // Coarsening config per material

    // Adaptive mesh coarsening data
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> active_cells;  // True if cell is active (not coarsened)
    Eigen::MatrixXd local_dx, local_dy;  // Local mesh spacing at each active cell
    std::vector<std::pair<int, int>> coarse_to_fine;  // coarse_idx -> (i, j) in full grid
    std::map<std::pair<int, int>, int> fine_to_coarse;  // (i, j) -> coarse_idx
    int n_active_cells;  // Number of active cells in coarsened mesh
    bool coarsening_enabled;  // Global flag: true if any material has coarsening enabled

    // User-defined variables (from YAML "variables" section)
    std::map<std::string, double> user_variables;  // Variable name -> evaluated value

    // Nonlinear solver
    NonlinearSolverConfig nonlinear_config;
    bool has_nonlinear_materials;  // Flag to enable nonlinear solver
    Eigen::MatrixXd H_map;  // Magnetic field intensity |H| [A/m] (for nonlinear iteration)

    // Solution
    Eigen::MatrixXd Az;       // Vector potential (z-component)

    // Transient analysis optimization: reuse matrix pattern (direct solver)
    Eigen::SparseLU<Eigen::SparseMatrix<double>> transient_solver;
    bool transient_solver_initialized;
    int transient_matrix_nnz;  // Track non-zero count for pattern verification

    // Transient analysis optimization: warm start (iterative solver)
    Eigen::VectorXd previous_solution;  // Previous step solution for warm start
    Eigen::VectorXd previous_rhs;       // Previous RHS for Δb correction
    Eigen::SparseMatrix<double> previous_matrix;  // Previous matrix for ΔA diagnostic
    bool use_iterative_solver;  // Use iterative solver with warm start (faster for step > 0)

    // Private methods
    void loadConfig(const std::string& config_path);
    void loadImage(const std::string& image_path);
    void parseUserVariables();  // Parse and evaluate user-defined variables from YAML
    void setupCartesianSystem();
    void setupPolarSystem();
    void setupMaterialProperties();
    void setupMaterialPropertiesForStep(int step);  // Update Jz for given step
    void validateBoundaryConditions();

    // Transient analysis methods
    void slideImageRegion();

    // Dynamic Jz evaluation
    JzValue parseJzValue(const YAML::Node& jz_node);
    double evaluateJz(const JzValue& jz_val, int step, const std::string& material_name = "");

    // Nonlinear permeability methods
    MuValue parseMuValue(const YAML::Node& mu_node);
    // Note: evaluateMu() and evaluateMuDerivative() are now public (needed for Newton-Krylov)
    void generateBHTable(const std::string& material_name, const MuValue& mu_val);
    void validateMuTable(const std::vector<double>& H_vals, const std::vector<double>& mu_vals, const std::string& material_name);
    double interpolateH_from_B(const BHTable& table, double B_magnitude);
    double interpolateB_from_H(const BHTable& table, double H_magnitude);
    double integrateMagneticCoEnergy(const BHTable& table, double H_magnitude);  // W' = ∫₀^H B(H') dH'
    double calculateCoEnergyDensity(int j, int i, double B_magnitude);  // Co-energy density w' [J/m³]
    void calculateHField();  // Calculate |H| from Bx, By (or Br, Btheta)
    void updateMuDistribution();  // Update mu_map based on current H_map

    // Anti-aliasing interpolation methods
    double calculateRGBDistance(const cv::Vec3b& a, const cv::Vec3b& b) const;
    bool isPointOnLineSegment(const cv::Vec3b& pixel, const cv::Vec3b& a, const cv::Vec3b& b, double tolerance = 15.0) const;
    double interpolateAntialiasedMu(const cv::Vec3b& pixel, double& out_mu_r) const;

    // Flux linkage calculation methods
    void parseFluxLinkagePaths();           // Parse flux_linkage section from YAML
    double interpolateAz(double x_phys, double y_phys) const;  // Bilinear interpolation of Az
    double calculateFluxLinkage(const FluxLinkagePath& path) const;  // Φ = Az(end) - Az(start)
    void calculateAllFluxLinkages(int step);  // Calculate and store all flux linkages
    void exportFluxLinkageCSV(const std::string& output_dir) const;  // Export to CSV

    // Adaptive mesh coarsening methods
    cv::Mat detectMaterialBoundaries();  // Detect material boundaries using edge detection
    void calculateOptimalSkipRatios();   // Calculate skip_x, skip_y from aspect ratio
    void generateCoarseningMask();       // Generate mask of active/inactive cells
    void generateCoarseningMaskCartesian(const cv::Mat& boundaries);  // Cartesian mask generation
    void generateCoarseningMaskPolar(const cv::Mat& boundaries);      // Polar mask generation
    void polarToImageIndices(int i_r, int j_theta, int& img_i, int& img_j) const;  // Polar->Image coordinate transform
    void buildCoarseIndexMaps();         // Build coarse <-> fine index mappings
    void calculateLocalMeshSpacing();    // Calculate h_minus/h_plus for each active cell
    int findNextActiveX(int i, int j, int direction) const;  // Find next active cell in X
    int findNextActiveY(int i, int j, int direction) const;  // Find next active cell in Y
    int findNextActiveRadial(int i_r, int j_theta, int direction) const;  // Find next active cell in r
    int findNextActiveTheta(int i_r, int j_theta, int direction) const;   // Find next active cell in theta
    std::pair<int, int> findActiveNeighbor(int i, int j, int di, int dj) const;  // Find active neighbor
    double bilinearInterpolateFromCoarse(int i, int j, const Eigen::VectorXd& Az_coarse) const;  // Interpolate inactive cell
    double interpolateFromCoarseGridPolar(int i_r, int j_theta, const Eigen::VectorXd& Az_coarse) const;  // Polar coarse grid interpolation
    double calculateThetaDistance(int j_from, int j_to) const;  // Calculate theta distance (periodic-aware)
    double calculateThetaInterpolationWeight(int j_theta, int j_prev, int j_next) const;  // Theta interpolation weight

    // Nonlinear solver methods
    void solveNonlinear();  // Main nonlinear Picard iteration solver
    void solveNonlinearWithAnderson();  // Picard with Anderson acceleration
    void solveNonlinearNewtonKrylov();  // Newton-Krylov (Jacobian-free GMRES)

    // Unified mu accessors (coordinate-system aware)
    double muAtGrid(int i, int j) const;  // i=col, j=row
    double getMuAtInterfaceSym(int i, int j, const std::string& direction) const;

    // Legacy accessors (will be deprecated)
    double getMuAtInterface(int i, int j, const std::string& direction) const;
    double getMuAtInterfacePolar(double r_idx, int theta_idx, const std::string& direction) const;

    void buildAndSolveSystem();
    void buildAndSolveSystemPolar();
    void buildAndSolveCartesianPseudoPolar();  // Hybrid initialization for polar Newton-Krylov

    // Matrix building methods (without solving) for transient optimization
    void buildMatrix(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);
    void buildMatrixPolar(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);

    // Adaptive mesh coarsening solver methods
    void buildMatrixCoarsened(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);
    void buildMatrixPolarCoarsened(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);
    void buildAndSolveSystemCoarsened();
    void buildAndSolveSystemPolarCoarsened();
    void interpolateToFullGrid(const Eigen::VectorXd& Az_coarse);
    void exportCoarseningMask(const std::string& output_dir, int step_number);  // Export binary mask: active=255, coarsened=0

    // Maxwell stress calculation methods
    cv::Mat detectBoundaries();
    void calculateMagneticField();
    void calculateMagneticFieldPolar();

    // Sampling method for consistent B and μ evaluation at physical points
    PolarSample sampleFieldsAtPhysicalPoint(double x_phys, double y_phys);

    // Overload for polar coordinates: directly use r_phys, theta_phys to avoid atan2 inconsistency
    PolarSample sampleFieldsAtPolarPoint(double r_phys, double theta_phys);

    // Helper method for periodic boundary-aware filtering
    void applyLaplacianWithPeriodicBC(const cv::Mat& src, cv::Mat& dst, int ksize = 3);
    void applySobelWithPeriodicBC(const cv::Mat& src, cv::Mat& dst, int dx, int dy, int ksize = 3);

    // Magnetic field components
    Eigen::MatrixXd Bx, By;  // Magnetic flux density components (Cartesian)
    Eigen::MatrixXd Br, Btheta;  // Magnetic flux density components (Polar)

    // Step tracking for magnetic field calculations
    int current_field_step;  // Current step for which Bx,By (or Br,Btheta) are calculated (-1 = not calculated or static)
};

#endif // MAGNETICFIELDANALYZER_H
