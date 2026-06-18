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
#include <memory>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>
#include <variant>
#include <tinyexpr.h>

#include "AsyncWriter.h"

// AMGCL headers for advanced iterative solvers
// The `builtin` backend supports OpenMP-parallel SpMV / vector operations
// inside CG and AMG smoothing. We switched from `backend::eigen` because the
// latter is single-threaded. The crs_tuple adapter lets us feed
// (rows, ptr, col, val) tuples derived from Eigen sparse matrices.
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/ilu0.hpp>      // Phase BE candidate: incomplete LU(0)
#include <amgcl/relaxation/gauss_seidel.hpp> // Phase BE candidate: GS (~SSOR)
#include <amgcl/solver/cg.hpp>

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
    void exportActiveOnlyResults(const std::string& base_folder, int step_number = 0);

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
        bool use_galerkin_coarsening;  // Phase 4: Use Galerkin A_c=R*A_f*P instead of geometric coarsening (default: true)
        bool use_matrix_free_jv;       // Phase 5: Use matrix-free Jv for Newton step (solves oscillation issue)

        // Phase 6: Preconditioned JFNK - use Galerkin coarse matrix as preconditioner for matrix-free GMRES
        bool use_phase6_precond_jfnk;     // Enable preconditioned JFNK (default: true)
        int precond_update_frequency;     // How often to update preconditioner: 1=every Newton iter (default: 1)
        bool precond_verbose;             // Print preconditioner statistics (default: false)

        // Fine finishing: full-grid Newton iterations after coarse convergence
        int fine_finishing_iterations;    // Number of full-grid Newton steps after coarse solve (default: 0 = disabled)
        double fine_finishing_tolerance;  // Convergence tolerance for fine finishing (default: -1 = use tolerance)

        // [Phase BJ-5] Strict convergence enforcement.
        // When using Phase 6 + Galerkin coarsening on saturated polar problems
        // (e.g. IEEJ-D IPMSM), the coarse plateau detector at residual ~2-4e-1
        // accepts a solution whose fine residual stays an order of magnitude
        // above TOL and whose iron region is under-saturated by ~50000x.
        // The flux magnitude on such a "converged" solve is ~1/10 of the true
        // (Standard-path) value — see README "適応粗大化が IEEJ-D class motor
        // で有効でない理由" / Phase BJ-5 note. When this flag is true, the
        // post-fine-finishing residual check throws std::runtime_error
        // instead of just warning, so production pipelines can catch the
        // wrong-answer case loudly. Default false to preserve v1.5.0 behaviour
        // — users who haven't seen the new diagnostic see only the warning.
        bool strict_convergence;

        // Phase BC: Eisenstat-Walker inexact-Newton forcing for the inner
        // AMGCL linear solve. When enabled, eta_k = gamma * (||R_k|| / ||R_{k-1}||)^alpha,
        // clipped to [eta_min, eta_max]. Lets CG stop early in iterations
        // where outer Newton residual is still large, avoiding pointless
        // 1e-6 inner accuracy. Disabled by default to preserve existing
        // behaviour for users who haven't opted in.
        bool eisenstat_walker_enabled;
        double eisenstat_walker_gamma;    // EW γ (default 0.9)
        double eisenstat_walker_alpha;    // EW α (default 2.0, Choice 2)
        double eisenstat_walker_eta_min;  // floor (default 1e-6, matches SOLVER_TOLERANCE)
        double eisenstat_walker_eta_max;  // initial / cap (default 0.1)

        NonlinearSolverConfig() :
            enabled(true), solver_type("newton-krylov"), max_iterations(50), tolerance(5e-4),
            relaxation(0.7), anderson(), gmres_restart(30), line_search_c(1e-4),
            line_search_alpha_init(1.0), line_search_alpha_min(1e-3), line_search_rho(0.65),
            line_search_max_trials(50), line_search_adaptive(true),
            verbose(false), export_convergence(false), use_galerkin_coarsening(false),
            use_matrix_free_jv(true),
            use_phase6_precond_jfnk(true), precond_update_frequency(1), precond_verbose(false),
            fine_finishing_iterations(0), fine_finishing_tolerance(-1.0),
            strict_convergence(false),
            eisenstat_walker_enabled(false),
            eisenstat_walker_gamma(0.9), eisenstat_walker_alpha(2.0),
            eisenstat_walker_eta_min(1e-6), eisenstat_walker_eta_max(0.1) {}
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

    // Flux linkage calculation path. Two variants, distinguished by
    // `use_material`:
    //   - use_material == false (default): the existing point-to-point
    //     path. Φ = Az(end) - Az(start) with bilinear interpolation at
    //     the two physical coordinates.
    //   - use_material == true (Phase B.3, v1.5): pixel-region variant
    //     for thick coils. Φ = mean(Az over material A pixels) -
    //     mean(Az over material B pixels). The material name -> RGB key
    //     is resolved at parse time so the per-step computation is just
    //     an image scan.
    struct FluxLinkagePath {
        std::string name;           // Path identifier (e.g., "phase_U")
        bool use_material = false;
        // Path variant
        double x_start = 0.0, y_start = 0.0;
        double x_end   = 0.0, y_end   = 0.0;
        // Material variant
        std::string material_a, material_b;
        int rgb_key_a = -1, rgb_key_b = -1;  // (R<<16)|(G<<8)|B, -1 = unresolved
    };

    // Flux linkage calculation
    std::vector<FluxLinkagePath> flux_linkage_paths;  // Defined paths for flux linkage
    std::map<std::string, std::vector<double>> flux_linkage_results;  // Results per path per step

    // Boundary detection optimization for transient analysis (incremental update)
    cv::Mat cached_boundaries;  // Cached boundary detection result (binary mask)
    bool boundary_cache_valid;  // Whether the cache is valid

    // Phase B.5: per-pixel sign factor for antiperiodic slide wrap.
    // Initialised to +1 on first slide; cells that cross the wrap seam
    // in an antiperiodic-mode slide have their sign flipped, and
    // setupMaterialPropertiesForStep multiplies jz_map (and the
    // magnetisation grids) by this sign so the source term reflects the
    // pole-pair polarity flip after the wrap. CV_8S, same dimensions as
    // `image` (cv::Mat, rows × cols, BGR Y-down).
    cv::Mat slide_sign_map;

    // Phase B.6: per-rectangle-slide cumulative displacement state.
    // dx / dy can be tinyexpr formulas in $step, so we accumulate the
    // float velocity into a float position and take the integer shift
    // per step as the delta between consecutive rounded values. This
    // way fractional velocities ("0.5") still produce coherent pixel
    // motion (alternating 0 / 1 shifts) rather than being silently
    // rounded to zero each step.
    struct RectSlideState {
        double cum_x = 0.0;
        double cum_y = 0.0;
        int    prev_int_x = 0;
        int    prev_int_y = 0;
    };
    std::vector<RectSlideState> rect_slide_states;  // index aligned with transient_config.slides
    int slide_step_counter = 0;  // increments at every slideImageRegion() call

    // Boundary conditions structure
    struct BoundaryCondition {
        std::string type;     // "dirichlet", "neumann", "periodic", or "robin"
        double value;         // Boundary value for Dirichlet

        // Robin BC parameters: alpha*Az + beta*(dAz/dn) = gamma
        double alpha;         // Coefficient for Az (default: 1.0)
        double beta;          // Coefficient for dAz/dn (default: 0.0)
        double gamma;         // RHS value (default: 0.0)

        // v1.6 domain-decomposition (optimized Schwarz): optional per-cell
        // boundary profile that overrides the scalar along the boundary.
        // For a radial boundary (inner/outer) it is indexed by theta-index j
        // and has length ntheta; if empty the scalar value/gamma is used
        // (backward compatible). Loaded from a CSV path in YAML:
        //   dirichlet -> value_profile, robin -> gamma_profile.
        std::vector<double> profile;

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

    // Phase B.2: a single sliding region. Multi-slide support is exposed
    // through the `slides` vector on TransientConfig; for a single slide
    // the loader fills both the vector and the legacy scalar fields from
    // the same source so the polar transient code paths keep working
    // unchanged while the cartesian image-domain slide loops over every
    // entry in the vector.
    //
    // Phase B.5: each slide also picks a wrap_mode that controls what
    // happens to content (and source terms) that crosses the seam:
    //   - "periodic"     : circular shift (current behaviour). The
    //                      material identity wraps unchanged.
    //   - "antiperiodic" : circular shift, AND every pixel that crossed
    //                      the seam gets jz / magnetisation sign-flipped.
    //                      This matches the algebra of an anti-periodic
    //                      theta BC -- the next pole is the opposite
    //                      polarity.
    //   - "vacuum"       : NO wrap. Cells vacated on the inlet side
    //                      get filled with the configured vacuum_rgb
    //                      (default white = air). Matches a Dirichlet
    //                      BC on the wrap axis.
    //   - "auto"         : inspect the corresponding BC type/value and
    //                      pick periodic/antiperiodic/vacuum so the
    //                      slide stays self-consistent with the field
    //                      boundary. Default for new yamls.
    struct SlideRegion {
        std::string name = "slide";
        // Phase B.6: "band" (existing — slides a vertical/horizontal strip with a
        // fixed integer pixels_per_step) or "rectangle" (slides a 2-D rectangular
        // cut-out by per-step (dx, dy), supporting tinyexpr formulas).
        std::string kind = "band";
        // Band variant
        std::string direction = "vertical";  // "vertical" | "horizontal"
        int region_start = 0;
        int region_end = 0;
        int pixels_per_step = 0;
        // [v1.6 Stage 0] Resolution-independent rotor rotation (polar only).
        // If use_angle, the per-step theta shift is computed as
        // round(angle_rad / dtheta) once the polar mesh is known
        // (finalizeSlideRotationForResolution), overwriting pixels_per_step, so
        // the SAME physical rotation is applied at any mesh resolution. This is
        // what makes downsampled / multi-fidelity transient runs comparable.
        bool use_angle = false;
        double angle_rad = 0.0;
        // Common
        std::string wrap_mode = "auto";
        std::vector<int> vacuum_rgb = {255, 255, 255};  // air, used by vacuum mode
        // Phase B.6 rectangle variant — image-coordinate (BGR Y-down) rect.
        // The rectangle's content is cut from this position, vacuum-filled in
        // place, and pasted at (rect + cumulative_displacement). The dx / dy
        // formulas are evaluated PER STEP with $step bound; users can write
        // either a constant ("5") or an expression ("$omega * cos(2*pi*$step/$N)").
        int rect_x_start = 0, rect_x_end = 0;
        int rect_y_start = 0, rect_y_end = 0;
        std::string dx_formula = "0";
        std::string dy_formula = "0";
    };

    // Transient analysis configuration
    struct TransientConfig {
        bool enabled;
        bool enable_sliding;          // Enable/disable image sliding
        int total_steps;

        // Phase B.2: explicit list of sliding regions. The cartesian slide
        // path iterates this vector, applying each region's shift
        // independently to its [region_start, region_end] interval.
        std::vector<SlideRegion> slides;

        // Legacy single-slide fields. Populated from `slides[0]` (if any)
        // during the loader pass and kept around because the polar
        // permutation / Δb / Gaussian-smoothing code paths assume one
        // sliding region. Polar multi-slide is left as a TODO for v1.6.
        std::string slide_direction;  // "vertical" or "horizontal"
        int slide_region_start;       // Pixel position (x for vertical, y for horizontal)
        int slide_region_end;         // Pixel position (x for vertical, y for horizontal)
        int slide_pixels_per_step;    // Pixels to shift per step

        // Output field selection (empty = export all, backward compat).
        // Valid names: "Az", "Mu", "H", "Jz", "InputImg", "BoundaryImg", "Forces", "EnergyDensity"
        std::vector<std::string> export_fields;

        TransientConfig() : enabled(false), enable_sliding(true), total_steps(0),
                           slide_direction("vertical"), slide_region_start(0),
                           slide_region_end(0), slide_pixels_per_step(0) {}
    };

    TransientConfig transient_config;

    // Result export configuration ("how" results are written;
    // the "what" stays in TransientConfig::export_fields).
    //
    // Defaults (v1.4): TIFF + async. Both were verified bit-exact against
    // the legacy CSV path over the Phase 1..5 work. Set `format: both`
    // explicitly in yaml if a downstream tool still consumes CSV directly.
    struct ExportConfig {
        enum class Format { CSV, TIFF, BOTH };
        enum class Precision { F32, F64 };
        Format format = Format::TIFF;
        Precision precision = Precision::F64;
        bool async = true;
        int async_queue_depth = 4;
        int tiff_compression = 8;  // libtiff COMPRESSION_DEFLATE
        int tiff_predictor   = 3;  // floating-point predictor
    };

    ExportConfig export_config;

    // Background writer thread for transient-step output.
    // Created on demand in loadConfig() when ExportConfig::async is true.
    // Drained at performTransientAnalysis() exit so its destructor never
    // runs with pending jobs.
    std::unique_ptr<AsyncWriter> async_writer_;

    // Material properties
    Eigen::MatrixXd mu_map;   // Permeability distribution (updated during nonlinear iteration)
    Eigen::MatrixXd jz_map;   // Current density distribution
    std::map<std::string, JzValue> material_jz;  // Dynamic Jz values per material
    std::map<std::string, MuValue> material_mu;  // Nonlinear permeability values per material
    std::map<std::string, BHTable> material_bh_tables;  // B-H tables per nonlinear material
    std::map<std::string, MaterialPixelInfo> material_pixel_info;  // Pixel count and area per material
    std::vector<AntialiasableMaterial> antialias_materials;  // Materials with antialias enabled
    std::map<std::string, YAML::Node> material_presets;  // Material presets (reusable B-H curves/properties)

    // RGB→material lookup table for O(1) material matching in hot loops (OpenMP-safe)
    struct MaterialLookupEntry {
        std::string name;
    };
    std::unordered_map<int, MaterialLookupEntry> rgb_to_material;  // key: (R<<16)|(G<<8)|B

    // Adaptive mesh coarsening configuration
    struct CoarsenConfig {
        bool enabled;       // Enable coarsening for this material
        int ratio;          // Coarsening ratio (area reduction factor)
        int skip_x;         // Max skip ratio in x/r direction (auto-calculated, power of 2)
        int skip_y;         // Max skip ratio in y/theta direction (auto-calculated, power of 2)
        int max_skip_iso;   // min(skip_x, skip_y) for gradient coarsening levels

        CoarsenConfig() : enabled(false), ratio(1), skip_x(1), skip_y(1), max_skip_iso(1) {}
    };
    std::map<std::string, CoarsenConfig> material_coarsen;  // Coarsening config per material

    // Permanent magnet magnetization model
    struct MagnetizationConfig {
        bool enabled = false;
        double Hc = 0.0;          // Effective magnetization magnitude [A/m] (resolved from Br or Hc in YAML)
        std::string pattern;       // "parallel", "radial", "tangential", "halbach_continuous", "polar_anisotropy", "radial_array", "parallel_array", "custom"
        double angle_deg = 0.0;    // Magnetization angle [deg] (parallel, parallel_array)
        // Phase J: p is the number of POLES (not pole pairs). A 4-pole
        // machine uses p=4. Should be even for a closed NS alternation
        // (odd values are accepted but the M field doesn't close at
        // θ=2π and the user typically wants p even). The halbach
        // formula uses p/2 internally; the array / polar_anisotropy
        // patterns use p sectors / p OJ centres directly.
        int p = 4;
        double cx = 0.0, cy = 0.0; // Rotation center [m]
        double R_pc = 0.0;         // Pitch circle radius [m] (polar_anisotropy)
        // Phase E.4: per-pattern direction sign. +1 = outward / first
        // pole positive (default, backward compat); -1 = inward / first
        // pole negative. Applies to radial / tangential (uniform flip)
        // and radial_array / parallel_array (sets pole 0's sign before
        // alternation).
        double direction_sign = 1.0;
        // Phase D.7: orientation offset for halbach_continuous and
        // polar_anisotropy. Rotates the first pole's orientation centre
        // (OJ in Kano 2025) by this angle around the rotor centre,
        // letting the user align the pole structure with an arbitrary
        // rotor initial angle. Defaults to 0.0 for backward compat.
        // NOTE: distinct from the *transient* theta_offset in
        // applyMaterials() which tracks cumulative sliding rotation
        // between steps.
        double orientation_offset_deg = 0.0;
        std::string Mx_expr, My_expr;  // tinyexpr expressions for Mx, My (parallel/halbach/custom)
    };
    std::map<std::string, MagnetizationConfig> material_magnetization;
    Eigen::MatrixXd Mx_map;      // (ny, nx) or (ntheta, nr) — magnetization x-component
    Eigen::MatrixXd My_map;      // same shape — magnetization y-component
    Eigen::MatrixXd Jz_mag_map;  // equivalent magnetization current (curl of M)

    // Adaptive mesh coarsening data
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> active_cells;  // True if cell is active (not coarsened)
    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> cell_skip_level;  // Per-cell skip level (1,2,4,8...)
    Eigen::MatrixXd local_dx, local_dy;  // Local mesh spacing at each active cell
    std::vector<std::pair<int, int>> coarse_to_fine;  // coarse_idx -> (i, j) in full grid
    std::map<std::pair<int, int>, int> fine_to_coarse;  // (i, j) -> coarse_idx
    int n_active_cells;  // Number of active cells in coarsened mesh
    bool coarsening_enabled;  // Global flag: true if any material has coarsening enabled
    std::vector<bool> active_row_flags;  // True if row has interior active cells (for interpolation)
    int max_coarsen_skip = 1;  // Maximum skip across all coarsened materials (for locality check)
    int coarsen_boundary_shell = 1;      // Edge dilation radius [pixels] for boundary protection (YAML: coarsening.boundary_shell)
    int coarsen_smooth_iterations = 0;   // Post-interpolation Laplacian smoothing iterations (YAML: coarsening.smooth_iterations)
    bool coarsen_auto_bump_skip = false; // [Phase BJ-4] When true, calculateOptimalSkipRatios bumps min(skip_x, skip_y) to 2 if the rounding would have produced 1, preventing silent no-op coarsening at the cost of overshooting the requested ratio. Opt-in only because the Phase 6 + Galerkin path that the bumped mask routes through has a known accuracy regression on saturated nonlinear polar problems (see README "適応粗大化が IEEJ-D class motor で有効でない理由"). YAML: coarsening.auto_bump_skip

    // [v1.6 Stage 2] Field-adaptive material-conforming coarsening (polar).
    // When enabled, the coarse mask is built AFTER an initial full solve from the
    // |B| field: a block is coarsened only if it is material-homogeneous, the
    // material is coarsen-eligible, and the |B| variation within it is below
    // adaptive_field_tol [T] -- so saturated / high-gradient iron + air gap +
    // boundaries stay fine (correct mu) while smooth bulk iron is coarsened.
    bool adaptive_mesh_enabled = false;     // YAML: adaptive_mesh.enabled
    double adaptive_field_tol = 0.1;        // YAML: adaptive_mesh.field_tol  [Tesla]
    int adaptive_coarsen_skip = 2;          // YAML: adaptive_mesh.skip  (block size S)
    void generateAdaptiveCoarseningMask();  // builds active_cells/cell_skip_level from |B| + material
    // [v1.6 Stage 1e / A1] Compute B, H, mu at the ACTIVE cells from the COARSE curl
    // of the coarse Az (findNextActive spacing), writing Br/Btheta/H_map/mu_map there.
    // Replaces the per-iter interpolate-to-full + full-grid field/mu in the coarse NK:
    // makes mu consistent with the coarse operator (correct flux) and is O(n_active)
    // (per-iteration speedup). Polar only; member Az must hold the coarse values at
    // active cells.
    void updateCoarseFieldAndMu();

    // Phase 4: Full-grid residual evaluation cache (for coarsened Newton-Krylov convergence)
    Eigen::SparseMatrix<double> A_full_cached;   // Cached full-grid matrix
    Eigen::VectorXd rhs_full_cached;              // Cached full-grid RHS
    bool full_matrix_cache_valid = false;         // Cache validity flag

    // Phase 4: Multigrid-style operators for Galerkin projection (A_c = R * A_f * P)
    Eigen::SparseMatrix<double> P_prolongation;   // n_full x n_active (coarse -> fine)
    Eigen::SparseMatrix<double> R_restriction;    // n_active x n_full (fine -> coarse, = P^T)
    bool multigrid_operators_built = false;       // Operators built flag

    // CSR (row-major) copies for OpenMP-parallel SpMV (see parallelSpMV)
    // Eigen's default CSC format does not support row-parallel matvec.
    // These are kept in sync with their CSC counterparts and used in the
    // hot path (defect correction residual + line search, ~5 SpMV/Newton iter).
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_full_cached_csr;
    Eigen::SparseMatrix<double, Eigen::RowMajor> P_prolongation_csr;
    Eigen::SparseMatrix<double, Eigen::RowMajor> R_restriction_csr;

    // Phase 7: Hermite interpolation gradients at active cells
    Eigen::MatrixXd dAz_dx_active;  // ∂Az/∂x at active cells (full grid size, zeros at inactive)
    Eigen::MatrixXd dAz_dy_active;  // ∂Az/∂y at active cells

    // Phase 6: Preconditioner cache for Preconditioned JFNK
    Eigen::SparseLU<Eigen::SparseMatrix<double>> precond_solver;  // LU factorization of A_c
    Eigen::SparseMatrix<double> A_coarse_precond;  // Cached coarse matrix for preconditioner
    int precond_newton_iter = -1;                   // Newton iteration at which preconditioner was built
    bool precond_factorization_valid = false;       // Whether preconditioner is ready to use

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
    Eigen::VectorXd previous_solution;  // x_{k-1}: previous step solution
    Eigen::VectorXd previous_previous_solution;  // x_{k-2}: for AR(1) linear extrapolation
    Eigen::VectorXd previous_rhs;       // Previous RHS for Δb correction
    Eigen::SparseMatrix<double> previous_matrix;  // Previous matrix for ΔA diagnostic
    bool use_iterative_solver;  // Use iterative solver with warm start (faster for step > 0)

    // Persistent linear solver for nonlinear iteration pattern reuse (SparseLU path)
    Eigen::SparseLU<Eigen::SparseMatrix<double>> nl_solver_;
    bool nl_solver_pattern_valid_ = false;
    int nl_solver_last_n_ = 0;

    // Adaptive linear solver selection threshold.
    // The original design called for direct (SparseLU) below ~10k DOFs and
    // iterative (AMGCL) above; this restores that policy after the threshold
    // drifted to 30k during the AMGCL migration. SparseLU's O(n^1.5) factor
    // time wins for very small problems where AMG hierarchy build overhead
    // dominates, while AMGCL with OpenMP-parallel builtin backend wins for
    // anything bigger -- including the 250k-DOF Jacobians that show up
    // inside Newton-Krylov outer iterations.
    static constexpr int AMGCL_THRESHOLD = 10000;

    // Export field selection check (used by exportResults). Empty list = export all (backward compat).
    bool shouldExportField(const std::string& field_name) const;

    // Optimized matrix-to-CSV writer: builds the full payload in memory with snprintf
    // and writes it in a single call. ~3-5x faster than per-element operator<< on Windows
    // (where each tiny stdio call incurs significant overhead).
    static void writeMatrixCSV(const Eigen::MatrixXd& m, const std::string& output_path);

    // TIFF writer (IEEE 754 float/double native). Precision is selected by
    // opts.precision (F32 -> CV_32FC1, F64 -> CV_64FC1). NaN/Inf bit patterns
    // pass through unchanged. Compression is libtiff's COMPRESSION_* code.
    static void writeMatrixTIFF(const Eigen::MatrixXd& m, const std::string& output_path,
                                const ExportConfig& opts);

    // Dispatch writer: takes a base path WITHOUT extension and routes to CSV/TIFF
    // writers based on ExportConfig.format.
    void writeMatrix(const Eigen::MatrixXd& m, const std::string& base_path,
                     const ExportConfig& opts) const;

    // Unified linear solver: AMGCL for large problems, SparseLU (with pattern reuse) for small.
    // Phase BC: tolerance > 0 overrides SOLVER_TOLERANCE for this call only. Used by
    // Eisenstat-Walker forcing in the Newton-Krylov outer loop to loosen the inner
    // AMGCL CG tolerance when the outer residual is still large -- no point
    // converging the linear system to 1e-6 when the Newton residual is at 1e+1.
    Eigen::VectorXd solveLinearSystem(const Eigen::SparseMatrix<double>& A,
                                      const Eigen::VectorXd& rhs,
                                      const Eigen::VectorXd& initial_guess = Eigen::VectorXd(),
                                      double tolerance = -1.0);

    // Private methods
    void loadConfig(const std::string& config_path);
    void loadImage(const std::string& image_path);
    void parseUserVariables();  // Parse and evaluate user-defined variables from YAML
    // v1.5 / Phase B.1: walk every YAML node and substitute "$name" tokens
    // with the corresponding value from user_variables. Lets the user write
    // e.g. `mesh: { dx: $cell_size }` or `transient: { total_steps: $N }`
    // without each field having to opt in to substitution. Must be called
    // AFTER parseUserVariables() (otherwise the variable map is empty).
    void expandUserVariablesGlobally();
    void expandUserVariablesInNode(YAML::Node node);
    std::string substituteDollarVarsInString(const std::string& s) const;
    void setupCartesianSystem();
    void setupPolarSystem();
    void setupMaterialProperties();
    void setupMaterialPropertiesForStep(int step);  // Update Jz for given step
    void validateBoundaryConditions();

    // Transient analysis methods
    void slideImageRegion();
    // Phase B.5: resolves SlideRegion.wrap_mode for "auto" by inspecting
    // the field BC perpendicular to the slide axis. Returns one of
    // "periodic", "antiperiodic", "vacuum".
    std::string resolveSlideWrapMode(const SlideRegion& slide) const;
    // Phase B.6: evaluates a rectangle slide's dx / dy formula at the
    // given step. The formula's $name tokens were globally substituted
    // at load time, so only $step is bound here.
    double evaluateSlideFormula(const std::string& formula, int step) const;

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

    // Permanent magnet magnetization model
    void computeMagnetizationGrids(int step = 0);  // Build Mx_map, My_map. Phase AA: step lets transient sliding rotate the magnetisation pattern with the rotor.
    void computeMagnetizationCurl();         // Cartesian: Jz_mag = ∂My/∂x - ∂Mx/∂y
    void computeMagnetizationCurlPolar();    // Polar: Jz_mag = (1/r)∂(r·Mθ)/∂r - (1/r)∂Mr/∂θ

    // Anti-aliasing interpolation methods
    double calculateRGBDistance(const cv::Vec3b& a, const cv::Vec3b& b) const;
    bool isPointOnLineSegment(const cv::Vec3b& pixel, const cv::Vec3b& a, const cv::Vec3b& b, double tolerance = 15.0) const;
    double interpolateAntialiasedMu(const cv::Vec3b& pixel, double& out_mu_r) const;

    // Phase N: parse the transient: block after $name expansion so
    // formulas (mu0 * 1000, ntheta / 2, pi/4, ...) and $variable
    // references in fields like total_steps, slide_pixels_per_step,
    // and the per-slide region / pixels_per_step entries resolve
    // through tinyexpr instead of failing the strict .as<int>() path.
    void parseTransientConfig();
    // [v1.6 Stage 0] Convert any angle-specified rotor rotation (SlideRegion
    // use_angle) into an integer theta-pixel shift now that dtheta is known.
    // Called once after setupPolarSystem(); polar only.
    void finalizeSlideRotationForResolution();
    // Tinyexpr-aware scalar evaluation helpers used by parseTransientConfig.
    double evaluateScalarAsDouble(const YAML::Node& node, double fallback) const;
    int    evaluateScalarAsInt   (const YAML::Node& node, int    fallback) const;

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
    void generateCoarseningMaskCartesian(const cv::Mat& boundaries, const cv::Mat& dist_map);  // Cartesian mask generation
    void generateCoarseningMaskPolar(const cv::Mat& boundaries, const cv::Mat& dist_map);      // Polar mask generation
    void polarToImageIndices(int i_r, int j_theta, int& img_i, int& img_j) const;  // Polar->Image coordinate transform
    void buildCoarseIndexMaps();         // Build coarse <-> fine index mappings
    void calculateLocalMeshSpacing();    // Calculate h_minus/h_plus for each active cell
    int findNextActiveX(int i, int j, int direction) const;  // Find next active cell in X
    int findNextActiveY(int i, int j, int direction) const;  // Find next active cell in Y
    int findNextActiveRadial(int i_r, int j_theta, int direction) const;  // Find next active cell in r
    int findNextActiveTheta(int i_r, int j_theta, int direction) const;   // Find next active cell in theta
    std::pair<int, int> findActiveNeighbor(int i, int j, int di, int dj) const;  // Find active neighbor
    double bilinearInterpolateFromCoarse(int i, int j, const Eigen::VectorXd& Az_coarse) const;  // Interpolate inactive cell
    double bilinearInterpolateFromCoarsePolar(int i_r, int j_theta, const Eigen::VectorXd& Az_coarse) const;  // Polar 4-corner bilinear
    double hermiteInterpolateFromCoarse(int i, int j, const Eigen::VectorXd& Az_coarse) const;  // C^1 Hermite interpolation
    void computeAzGradientsAtActiveCells(const Eigen::VectorXd& Az_coarse);  // Compute ∂Az/∂x, ∂Az/∂y at active cells
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
    void interpolateToFullGridPolar(const Eigen::VectorXd& Az_coarse);
    void smoothInactiveCells(int iterations);  // Post-interpolation Laplacian smoothing at inactive cells
    void interpolateMuToFullGrid();  // IDW harmonic mean μ interpolation at inactive cells
    void interpolateInactiveCells(const Eigen::VectorXd& Az_coarse);  // Update only inactive cells (for nonlinear iteration)
    void interpolateInactiveCellsPolar(const Eigen::VectorXd& Az_coarse);  // Polar version
    void exportCoarseningMask(const std::string& output_dir, int step_number);  // Export binary mask: active=255, coarsened=0

    // Phase 8: Wide-stencil B/H/μ at active cells only (stripe-free nonlinear coarsening)
    // Computes B using only active cell Az values (wide stencil, no inactive cell dependency)
    void calculateBFieldAtActiveCells(const Eigen::VectorXd& Az_coarse,
                                       Eigen::VectorXd& Bx_active,
                                       Eigen::VectorXd& By_active);
    // Computes H from B at active cells via material B-H tables
    void calculateHFieldAtActiveCells(const Eigen::VectorXd& Bx_active,
                                       const Eigen::VectorXd& By_active,
                                       Eigen::VectorXd& H_active);
    // Updates mu_map at active cells only from H (inactive cells unchanged)
    void updateMuAtActiveCells(const Eigen::VectorXd& H_active);
    // Updates mu_map at active cells with differential permeability μ_diff = dB/dH / μ₀
    // Used for Newton correction in Phase 6: A(μ_diff) * δAz = -R(μ_eff)
    void updateMuDiffAtActiveCells(const Eigen::VectorXd& H_active);
    // Full-grid version: updates mu_map for ALL pixels using H_map(j,i).
    // Used for Newton correction in fine finishing after coarse convergence.
    void updateMuDiffDistribution();

    // Phase 4: Full-grid residual evaluation for coarsened Newton-Krylov convergence
    void updateFullMatrixCache();  // Rebuild A_full_cached and rhs_full_cached
    double computeFullGridResidual(double& out_b_norm);  // Compute ||A_f * Az - b_f|| on full grid

    // OpenMP-parallel sparse matrix-vector product y = A * x.
    // Requires a row-major (CSR) matrix; iterates rows in parallel.
    // Used in the defect-correction hot path and line search.
    Eigen::VectorXd parallelSpMV(
        const Eigen::SparseMatrix<double, Eigen::RowMajor>& A,
        const Eigen::VectorXd& x) const;

    // Phase 4: Prolongation/Restriction operators for Galerkin projection
    void buildProlongationMatrix();  // Build P_prolongation (n_full x n_active) and R_restriction (P^T)
    void buildInterpolationWeights(int i, int j, int fine_idx,
        std::vector<Eigen::Triplet<double>>& triplets);  // Helper: Cartesian interpolation weights
    void buildInterpolationWeightsPolar(int i_r, int j_theta, int fine_idx,
        std::vector<Eigen::Triplet<double>>& triplets);  // Helper: Polar interpolation weights
    void buildProlongationMatrixPolar(std::vector<Eigen::Triplet<double>>& triplets);  // Polar version

    // Phase 4: Galerkin coarse matrix (A_c = R * A_f * P)
    void buildMatrixGalerkin(Eigen::SparseMatrix<double>& A_coarse, Eigen::VectorXd& rhs_coarse);

    // Phase 5: Matrix-free Jacobian-vector product for coarsened Newton-Krylov
    // Root solution for oscillating convergence: J*v computed via finite differences
    // J_c*v ≈ R * (F_full(P*(x + ε*v)) - F_full(P*x)) / ε
    Eigen::VectorXd assembleFullGridResidualVector(const Eigen::VectorXd& Az_full_vec,
        const Eigen::MatrixXd& mu_full);  // Compute F(Az) = A(μ)*Az - b on full grid
    void computeBHmuFromAzVector(const Eigen::VectorXd& Az_full_vec,
        Eigen::MatrixXd& mu_out, Eigen::MatrixXd& Bx_out, Eigen::MatrixXd& By_out,
        Eigen::MatrixXd& H_out);  // Compute B, H, μ from Az vector
    Eigen::VectorXd matrixFreeJv(const Eigen::VectorXd& x_c, const Eigen::VectorXd& v_c);
        // Matrix-free J*v for coarsened system
    Eigen::VectorXd solveWithMatrixFreeGMRES(const Eigen::VectorXd& x_c,
        const Eigen::VectorXd& rhs, int max_iter, double tol);
        // GMRES solver using matrix-free Jv

    // Phase 6: Preconditioned JFNK (uses Galerkin coarse matrix as preconditioner)
    // Combines matrix-free Jv correctness with efficient preconditioner
    void updatePreconditioner(int newton_iter);  // Build/update A_c preconditioner
    Eigen::VectorXd applyPreconditioner(const Eigen::VectorXd& v_c);  // Compute A_c^{-1} * v
    Eigen::VectorXd solveWithPreconditionedGMRES(const Eigen::VectorXd& x_c,
        const Eigen::VectorXd& rhs, int max_iter, double tol);
        // Right-preconditioned GMRES: J * M^{-1} * y = rhs, delta = M^{-1} * y

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
