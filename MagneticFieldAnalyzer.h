#ifndef MAGNETICFIELDANALYZER_H
#define MAGNETICFIELDANALYZER_H

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
     * @brief Calculate Maxwell stress and electromagnetic forces
     * @param step Step number for field caching (-1 for static analysis)
     */
    void calculateMaxwellStress(int step = -1);

    /**
     * @brief Calculate total magnetic energy of the system
     * @param step Step number for field caching (-1 for static analysis)
     * @return Total magnetic energy [J/m] (per unit depth)
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

    /**
     * @brief Compute differential permeability dB/dH analytically from effective permeability
     *
     * This function converts effective permeability μ_eff = B/H (catalog data format)
     * to differential permeability dB/dH (needed for accurate Jacobian in Newton-Krylov).
     *
     * Mathematical derivation:
     *   Given: B(H) = μ_eff(H) · μ₀ · H
     *   Then:  dB/dH = d/dH[μ_eff(H) · μ₀ · H]
     *               = μ₀ · [dμ_eff/dH · H + μ_eff · 1]
     *               = μ₀ · (μ_eff + H · dμ_eff/dH)
     *
     * This avoids numerical differentiation errors that plague the naive approach.
     *
     * @param mu_val Effective permeability specification (μ_eff = B/H from YAML)
     * @param H_magnitude Magnetic field intensity |H| [A/m]
     * @return Differential permeability dB/dH [H/m] (absolute permeability units)
     */
    double computeDifferentialPermeability(const MuValue& mu_val, double H_magnitude);

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

    // Nonlinear solver configuration
    struct NonlinearSolverConfig {
        bool enabled;               // Enable nonlinear solver (default: true, used with has_nonlinear_materials)
        std::string solver_type;    // Solver type: "picard", "anderson", "newton-krylov" (default: "newton-krylov")
        int max_iterations;         // Maximum nonlinear iterations (default: 50)
        double tolerance;           // Convergence tolerance (relative) (default: 5e-4)
        double relaxation;          // Relaxation factor (0.5 ~ 0.8) (default: 0.7) - for Picard/Anderson
        int anderson_depth;         // Anderson acceleration depth (default: 5) - for Anderson
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
            relaxation(0.7), anderson_depth(5), gmres_restart(30), line_search_c(1e-4),
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

    std::vector<ForceResult> force_results;
    std::vector<BoundaryStressPoint> boundary_stress_vectors;  // Stress vectors at boundaries
    cv::Mat boundary_image;  // Cached boundary detection visualization

    // Boundary detection optimization for transient analysis (incremental update)
    cv::Mat cached_boundaries;  // Cached boundary detection result (binary mask)
    bool boundary_cache_valid;  // Whether the cache is valid

    // Boundary conditions structure
    struct BoundaryCondition {
        std::string type;     // "dirichlet" or "neumann"
        double value;         // Boundary value for Dirichlet

        BoundaryCondition() : type("dirichlet"), value(0.0) {}
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
    BoundaryCondition bc_inner, bc_outer;  // Polar

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
    void setupCartesianSystem();
    void setupPolarSystem();
    void setupMaterialProperties();
    void setupMaterialPropertiesForStep(int step);  // Update Jz for given step
    void validateBoundaryConditions();

    // Transient analysis methods
    void slideImageRegion();

    // Dynamic Jz evaluation
    JzValue parseJzValue(const YAML::Node& jz_node);
    double evaluateJz(const JzValue& jz_val, int step);

    // Nonlinear permeability methods
    MuValue parseMuValue(const YAML::Node& mu_node);
    // Note: evaluateMu() and evaluateMuDerivative() are now public (needed for Newton-Krylov)
    void generateBHTable(const std::string& material_name, const MuValue& mu_val);
    void validateMuTable(const std::vector<double>& H_vals, const std::vector<double>& mu_vals, const std::string& material_name);
    double interpolateH_from_B(const BHTable& table, double B_magnitude);
    double interpolateB_from_H(const BHTable& table, double H_magnitude);
    void calculateHField();  // Calculate |H| from Bx, By (or Br, Btheta)
    void updateMuDistribution();  // Update mu_map based on current H_map

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

    // Matrix building methods (without solving) for transient optimization
    void buildMatrix(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);
    void buildMatrixPolar(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& rhs);

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
