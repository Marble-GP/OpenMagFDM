#include "MagneticFieldAnalyzer.h"
#include "tinyexpr/tinyexpr.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <cmath>
#include <map>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <yaml-cpp/yaml.h>
#include "json.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

using json = nlohmann::json;

namespace {
#ifndef OPENMAGFDM_VERSION_STRING
#define OPENMAGFDM_VERSION_STRING "1.6.1"
#endif
constexpr const char* OPENMAGFDM_VERSION = OPENMAGFDM_VERSION_STRING;
}

/**
 * @brief Custom streambuf that writes to both console and file (tee functionality)
 */
class TeeBuffer : public std::streambuf {
public:
    TeeBuffer(std::streambuf* sb1, std::streambuf* sb2) : sb1_(sb1), sb2_(sb2) {}

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return !EOF;
        }
        if (sb1_->sputc(c) == EOF || sb2_->sputc(c) == EOF) {
            return EOF;
        }
        return c;
    }

    virtual int sync() override {
        int r1 = sb1_->pubsync();
        int r2 = sb2_->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }

private:
    std::streambuf* sb1_;
    std::streambuf* sb2_;
};

/**
 * @brief Generate timestamp-based folder name
 * @return Folder name in format: output_YYYYMMDD_HHMMSS
 */
std::string generateTimestampFolderName() {
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm* tm_now = std::localtime(&time_t_now);

    std::ostringstream oss;
    oss << "output_"
        << std::setfill('0')
        << std::setw(4) << (tm_now->tm_year + 1900)
        << std::setw(2) << (tm_now->tm_mon + 1)
        << std::setw(2) << tm_now->tm_mday
        << "_"
        << std::setw(2) << tm_now->tm_hour
        << std::setw(2) << tm_now->tm_min
        << std::setw(2) << tm_now->tm_sec;

    return oss.str();
}

/**
 * @brief Extract base folder name from output path (remove .csv extension if present)
 * @param output_path Output path provided by user
 * @return Base folder name
 */
std::string getBaseFolderName(const std::string& output_path) {
    // Remove .csv extension if present
    size_t dot_pos = output_path.find_last_of('.');
    if (dot_pos != std::string::npos && output_path.substr(dot_pos) == ".csv") {
        return output_path.substr(0, dot_pos);
    }
    return output_path;
}

/**
 * @brief Export analysis conditions to JSON file
 * @param output_path Output JSON file path
 * @param config_path Path to YAML configuration file
 * @param image_path Path to material image file
 */
// Phase Q: exportConditionsJSON runs BEFORE the analyzer is built, so
// it can't lean on the analyzer's user-variable expansion pass. Resolve
// the variables: block here ourselves and substitute $name in scalar
// fields before evaluating them. The same helper is used for transient
// fields that the WebUI's Insert YAML emits as `$N_step` / `$N_slide`
// references; without this, .as<int>() trips on the literal string and
// the solver aborts before the analyzer ever loads.
static double evalScalarConditions(const YAML::Node& node,
                                   const std::map<std::string, double>& user_vars,
                                   double fallback) {
    if (!node || !node.IsScalar()) return fallback;
    std::string s;
    try { s = node.as<std::string>(); } catch (...) { return fallback; }
    if (s.empty()) return fallback;
    // $name substitution, longest names first so $omega doesn't partial-
    // match $o.
    std::vector<std::pair<std::string, double>> sorted(user_vars.begin(), user_vars.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
    for (const auto& [name, value] : sorted) {
        const std::string needle = "$" + name;
        std::size_t pos = 0;
        while ((pos = s.find(needle, pos)) != std::string::npos) {
            std::ostringstream oss;
            oss << std::setprecision(17) << value;
            const std::string repl = oss.str();
            s.replace(pos, needle.size(), repl);
            pos += repl.size();
        }
    }
    // Fast path: plain numeric literal.
    try {
        std::size_t consumed = 0;
        const double v = std::stod(s, &consumed);
        while (consumed < s.size()
               && std::isspace(static_cast<unsigned char>(s[consumed]))) ++consumed;
        if (consumed == s.size()) return v;
    } catch (...) { /* fall through */ }
    // tinyexpr: pi / e are built-in; mu0 is injected so users can write
    // `mu0 * 1000`. $step survives — it's a per-step variable evaluated
    // by the analyzer at runtime, not here.
    te_parser parser;
    {
        te_variable mu0_var; mu0_var.m_name = "mu0";
        constexpr double MU_0 = 4.0 * 3.14159265358979323846 * 1e-7;
        mu0_var.m_value = MU_0;
        parser.set_variables_and_functions({mu0_var});
    }
    const double v = parser.evaluate(s);
    if (parser.success() && std::isfinite(v)) return v;
    return fallback;
}

static int evalScalarConditionsAsInt(const YAML::Node& node,
                                     const std::map<std::string, double>& user_vars,
                                     int fallback) {
    return static_cast<int>(std::lround(evalScalarConditions(
        node, user_vars, static_cast<double>(fallback))));
}

static bool scalarUsesPhysicalMetres(
        const YAML::Node& node,
        const std::map<std::string, double>& user_vars) {
    if (!node || !node.IsScalar()) return false;
    std::string s = node.Scalar();
    if (s.find('$') != std::string::npos) {
        // Match the analyzer's global $variable expansion: substitute the
        // numeric value at 17-digit precision before looking for the decimal
        // unit marker. A variable resolving to 50 stays pixels; one resolving
        // to 0.05 becomes metres.
        std::vector<std::pair<std::string, double>> sorted(user_vars.begin(), user_vars.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });
        for (const auto& [name, value] : sorted) {
            const std::string needle = "$" + name;
            std::size_t pos = 0;
            while ((pos = s.find(needle, pos)) != std::string::npos) {
                std::ostringstream oss;
                oss << std::setprecision(17) << value;
                const std::string replacement = oss.str();
                s.replace(pos, needle.size(), replacement);
                pos += replacement.size();
            }
        }
    }
    return s.find('.') != std::string::npos
        || s.find('e') != std::string::npos
        || s.find('E') != std::string::npos;
}

void exportConditionsJSON(const std::string& output_path,
                          const std::string& config_path,
                          const std::string& image_path) {
    // Load YAML configuration
    YAML::Node config = YAML::LoadFile(config_path);

    // Phase Q: resolve user variables up-front so transient and other
    // formula-bearing fields can use $name references.
    std::map<std::string, double> user_vars;
    {
        constexpr double MU_0 = 4.0 * 3.14159265358979323846 * 1e-7;
        user_vars["pi"]  = 3.14159265358979323846;
        user_vars["e"]   = 2.71828182845904523536;
        user_vars["mu0"] = MU_0;
        if (config["variables"] && config["variables"].IsMap()) {
            for (const auto& var : config["variables"]) {
                const std::string name = var.first.as<std::string>("");
                if (name.empty()) continue;
                // Recurse: evaluate each variable with the already-resolved
                // ones so `omega: "2*pi*60"` and similar formulas resolve.
                user_vars[name] = evalScalarConditions(var.second, user_vars, 0.0);
            }
        }
    }

    // Load image to get dimensions
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Create JSON object
    json j;
    j["openmagfdm_version"] = OPENMAGFDM_VERSION;

    // Coordinate system
    std::string coord_system = config["coordinate_system"]
        ? config["coordinate_system"].as<std::string>() : "cartesian";
    j["coordinate_system"] = coord_system;

    // Image dimensions
    j["image_width"] = image.cols;
    j["image_height"] = image.rows;

    // Mesh spacing
    if (coord_system == "cartesian") {
        double dx = 0.001;
        double dy = 0.001;
        // Try mesh section first, then fallback to top-level
        if (config["mesh"] && config["mesh"]["dx"]) {
            dx = config["mesh"]["dx"].as<double>();
        } else if (config["dx"]) {
            dx = config["dx"].as<double>();
        }
        if (config["mesh"] && config["mesh"]["dy"]) {
            dy = config["mesh"]["dy"].as<double>();
        } else if (config["dy"]) {
            dy = config["dy"].as<double>();
        }
        j["dx"] = dx;
        j["dy"] = dy;
    } else if (coord_system == "polar") {
        // Try polar_domain first, then polar
        YAML::Node polar_section;
        if (config["polar_domain"]) {
            polar_section = config["polar_domain"];
        } else if (config["polar"]) {
            polar_section = config["polar"];
        }

        double r_start = polar_section["r_start"].as<double>();
        double r_end = polar_section["r_end"].as<double>();
        int nr = image.cols;
        int ntheta = image.rows;
        double dr = (r_end - r_start) / (nr - 1);

        // Parse theta_range (supports tinyexpr formula like "pi/2", "pi/3", or numeric values)
        double theta_range;
        if (polar_section["theta_range"]) {
            std::string theta_str;
            try {
                // Try to read as string first (for formulas like "pi/2", "2*pi/12")
                theta_str = polar_section["theta_range"].as<std::string>();
            } catch (...) {
                // If that fails, try to read as double and convert to string
                try {
                    double val = polar_section["theta_range"].as<double>();
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

        // CRITICAL FIX: Periodic boundary condition requires dtheta = theta_range / ntheta
        // NOT theta_range / (ntheta-1) which is for non-periodic grids
        double dtheta = theta_range / static_cast<double>(ntheta);

        j["dr"] = dr;
        j["dtheta"] = dtheta;

        // Polar parameters
        std::string r_orientation = polar_section["r_orientation"]
            ? polar_section["r_orientation"].as<std::string>() : "horizontal";
        j["polar"] = {
            {"r_start", r_start},
            {"r_end", r_end},
            {"theta_range", theta_range},
            {"r_orientation", r_orientation}
        };
    }

    // Boundary conditions
    auto getBoundary = [](const YAML::Node& bc_node) -> json {
        std::string type = "dirichlet";
        double value = 0.0;
        if (bc_node) {
            if (bc_node["type"]) type = bc_node["type"].as<std::string>();
            if (bc_node["value"]) value = bc_node["value"].as<double>();
        }
        return {{"type", type}, {"value", value}};
    };

    // Try boundary_conditions first, then boundary
    YAML::Node bc_section;
    if (config["boundary_conditions"]) {
        bc_section = config["boundary_conditions"];
    } else if (config["boundary"]) {
        bc_section = config["boundary"];
    }

    if (coord_system == "cartesian") {
        j["boundary_conditions"] = {
            {"left", getBoundary(bc_section["left"])},
            {"right", getBoundary(bc_section["right"])},
            {"bottom", getBoundary(bc_section["bottom"])},
            {"top", getBoundary(bc_section["top"])}
        };
    } else if (coord_system == "polar") {
        // Try polar_boundary_conditions first, then boundary
        YAML::Node polar_bc_section;
        if (config["polar_boundary_conditions"]) {
            polar_bc_section = config["polar_boundary_conditions"];
        } else if (config["boundary"]) {
            polar_bc_section = config["boundary"];
        }

        j["boundary_conditions"] = {
            {"inner", getBoundary(polar_bc_section["inner"])},
            {"outer", getBoundary(polar_bc_section["outer"])},
            {"theta_min", getBoundary(polar_bc_section["theta_min"])},
            {"theta_max", getBoundary(polar_bc_section["theta_max"])}
        };
    }

    // Transient configuration
    j["transient"] = json::object();
    if (config["transient"] && config["transient"]["enabled"]) {
        const YAML::Node transient = config["transient"];
        bool enabled = config["transient"]["enabled"].as<bool>();
        j["transient"]["enabled"] = enabled;

        if (enabled) {
            bool enable_sliding = transient["enable_sliding"]
                ? transient["enable_sliding"].as<bool>() : true;
            // Phase Q: tinyexpr-evaluated with $var substitution so
            // `total_steps: $N_step` resolves correctly.
            int total_steps = evalScalarConditionsAsInt(
                transient["total_steps"], user_vars, 0);

            j["transient"]["enable_sliding"] = enable_sliding;
            j["transient"]["total_steps"] = total_steps;
            j["transient"]["parallel_chunks"] = evalScalarConditionsAsInt(
                transient["parallel_chunks"], user_vars, 1);

            if (enable_sliding) {
                const bool has_slides = transient["slides"]
                    && transient["slides"].IsSequence() && transient["slides"].size() > 0;
                YAML::Node primary_slide = has_slides
                    ? transient["slides"][0] : YAML::Node();
                const std::string primary_kind = has_slides
                    ? primary_slide["kind"].as<std::string>("band") : "band";
                std::string slide_direction = has_slides && primary_slide["direction"]
                    ? primary_slide["direction"].as<std::string>("vertical")
                    : transient["slide_direction"].as<std::string>("vertical");
                YAML::Node start_node = has_slides && primary_slide["region_start"]
                    ? primary_slide["region_start"] : transient["slide_region_start"];
                YAML::Node end_node = has_slides && primary_slide["region_end"]
                    ? primary_slide["region_end"] : transient["slide_region_end"];
                YAML::Node pixels_node = has_slides && primary_slide["pixels_per_step"]
                    ? primary_slide["pixels_per_step"] : transient["slide_pixels_per_step"];
                const bool bounds_in_metres =
                    scalarUsesPhysicalMetres(start_node, user_vars)
                    || scalarUsesPhysicalMetres(end_node, user_vars);
                double slide_region_start = evalScalarConditions(
                    start_node, user_vars, 0.0);
                double slide_region_end = evalScalarConditions(
                    end_node, user_vars, 0.0);
                int slide_pixels_per_step = evalScalarConditionsAsInt(
                    pixels_node, user_vars, 0);

                j["transient"]["slide_kind"] = primary_kind;
                if (primary_kind == "band") {
                    j["transient"]["slide_direction"] = slide_direction;
                    if (bounds_in_metres) {
                        j["transient"]["slide_region_start"] = slide_region_start;
                        j["transient"]["slide_region_end"] = slide_region_end;
                        j["transient"]["slide_region_units"] = "m";
                    } else {
                        j["transient"]["slide_region_start"] = static_cast<int>(std::lround(slide_region_start));
                        j["transient"]["slide_region_end"] = static_cast<int>(std::lround(slide_region_end));
                        j["transient"]["slide_region_units"] = "pixel";
                    }
                    j["transient"]["slide_pixels_per_step"] = slide_pixels_per_step;
                }

                // Preserve the complete multi-slide declaration as metadata.
                // Bounds remain in their authored units; the analyzer logs the
                // resolved pixel range after the mesh has been initialized.
                if (transient["slides"] && transient["slides"].IsSequence()) {
                    j["transient"]["slides"] = json::array();
                    int slide_index = 0;
                    for (const auto& slide : transient["slides"]) {
                        json out;
                        out["name"] = slide["name"].as<std::string>(
                            "slide_" + std::to_string(slide_index));
                        const std::string kind = slide["kind"].as<std::string>("band");
                        out["kind"] = kind;
                        out["direction"] = slide["direction"].as<std::string>("vertical");
                        if (kind == "rectangle") {
                            if (slide["rect"]) out["rect"] = slide["rect"].as<std::vector<int>>();
                            out["dx"] = slide["dx"].as<std::string>("0");
                            out["dy"] = slide["dy"].as<std::string>("0");
                        } else {
                            const bool metres = scalarUsesPhysicalMetres(slide["region_start"], user_vars)
                                || scalarUsesPhysicalMetres(slide["region_end"], user_vars);
                            const double lo = evalScalarConditions(slide["region_start"], user_vars, 0.0);
                            const double hi = evalScalarConditions(slide["region_end"], user_vars, 0.0);
                            out["region_start"] = metres ? json(lo) : json(static_cast<int>(std::lround(lo)));
                            out["region_end"] = metres ? json(hi) : json(static_cast<int>(std::lround(hi)));
                            out["region_units"] = metres ? "m" : "pixel";
                            out["pixels_per_step"] = evalScalarConditionsAsInt(
                                slide["pixels_per_step"], user_vars, 0);
                            if (slide["angle_rad"]) {
                                out["angle_rad"] = evalScalarConditions(slide["angle_rad"], user_vars, 0.0);
                            } else if (slide["angle_deg"]) {
                                out["angle_deg"] = evalScalarConditions(slide["angle_deg"], user_vars, 0.0);
                            }
                        }
                        out["wrap_mode"] = slide["wrap_mode"].as<std::string>("auto");
                        if (slide["vacuum_rgb"]) {
                            out["vacuum_rgb"] = slide["vacuum_rgb"].as<std::vector<int>>();
                        }
                        j["transient"]["slides"].push_back(std::move(out));
                        ++slide_index;
                    }
                }
            }
        }
    } else {
        j["transient"]["enabled"] = false;
    }

    // Nonlinear materials detection. Phase T: previously this only
    // looked at mu_r, ignored B-H entirely, and never resolved preset
    // references -- so a config that referenced pure_iron_model via
    // `preset: pure_iron_model` (where the preset carries a B-H formula
    // and no mu_r override) reported has_nonlinear_materials = false
    // in conditions.json, even though the analyzer's own classification
    // in setupMaterialProperties correctly detected the B-H formula and
    // ran the nonlinear solver.
    bool has_nonlinear_materials = false;
    if (config["materials"]) {
        // Build a quick map of presets so we can read inherited fields.
        std::map<std::string, YAML::Node> presets;
        if (config["material_presets"]) {
            for (const auto& p : config["material_presets"]) {
                presets[p.first.as<std::string>("")] = p.second;
            }
        }
        auto fieldOf = [&](const YAML::Node& props, const std::string& key) -> YAML::Node {
            if (props[key]) return props[key];
            if (props["preset"]) {
                const std::string pn = props["preset"].as<std::string>("");
                auto it = presets.find(pn);
                if (it != presets.end() && it->second[key]) return it->second[key];
            }
            return YAML::Node();
        };
        auto looksLikeFormula = [](const std::string& s) {
            return s.find('$') != std::string::npos
                || s.find('*') != std::string::npos
                || s.find('/') != std::string::npos
                || s.find('+') != std::string::npos
                || s.find('(') != std::string::npos
                || s.find("exp") != std::string::npos
                || s.find("tanh") != std::string::npos;
        };
        for (const auto& material : config["materials"]) {
            const auto& props = material.second;
            // B-H trumps mu_r: any defined B-H (string formula or 2-array
            // table) is nonlinear.
            YAML::Node bh = fieldOf(props, "B-H");
            if (bh && bh.IsScalar()) {
                has_nonlinear_materials = true; break;
            }
            if (bh && bh.IsSequence() && bh.size() == 2
                && bh[0].IsSequence() && bh[1].IsSequence()) {
                has_nonlinear_materials = true; break;
            }
            YAML::Node mu = fieldOf(props, "mu_r");
            if (!mu) continue;
            if (mu.IsScalar()) {
                std::string mu_str = mu.as<std::string>("");
                if (looksLikeFormula(mu_str)) {
                    has_nonlinear_materials = true; break;
                }
            } else if (mu.IsSequence() && mu.size() == 2) {
                has_nonlinear_materials = true; break;
            }
        }
    }

    // Nonlinear solver configuration
    j["nonlinear_solver"] = json::object();
    j["nonlinear_solver"]["has_nonlinear_materials"] = has_nonlinear_materials;

    if (config["nonlinear_solver"]) {
        bool nl_enabled = false;
        std::string solver_type = "picard";

        if (config["nonlinear_solver"]["enabled"]) {
            nl_enabled = config["nonlinear_solver"]["enabled"].as<bool>();
        }
        if (config["nonlinear_solver"]["solver_type"]) {
            solver_type = config["nonlinear_solver"]["solver_type"].as<std::string>();
        }

        j["nonlinear_solver"]["enabled"] = nl_enabled;
        j["nonlinear_solver"]["solver_type"] = solver_type;
    }

    // Export configuration (which writers were active for this run; lets the
    // WebUI show "Source: TIFF (double)" badges and pick the right reader).
    // Defaults here must match ExportConfig in MagneticFieldAnalyzer.h.
    j["export"] = json::object();
    j["export"]["format"]    = "tiff";    // v1.4 default
    j["export"]["precision"] = "double";
    j["export"]["async"]     = true;
    if (config["export"]) {
        auto exp = config["export"];
        if (exp["format"])    j["export"]["format"]    = exp["format"].as<std::string>("tiff");
        if (exp["precision"]) j["export"]["precision"] = exp["precision"].as<std::string>("double");
        if (exp["async"])     j["export"]["async"]     = exp["async"].as<bool>(true);
        if (exp["tiff"]) {
            j["export"]["tiff"] = json::object();
            if (exp["tiff"]["compression"]) j["export"]["tiff"]["compression"] = exp["tiff"]["compression"].as<std::string>("deflate");
            if (exp["tiff"]["predictor"])   j["export"]["tiff"]["predictor"]   = exp["tiff"]["predictor"].as<int>(3);
        }
    }

    // Write to file with proper indentation
    std::ofstream json_file(output_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to create JSON file: " + output_path);
    }
    json_file << j.dump(2) << std::endl;  // indent with 2 spaces
    json_file.close();
}

int main(int argc, char* argv[]) {
#ifdef _WIN32
    // Set console output to UTF-8 for proper encoding in Web UI
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    if (argc == 2 && std::string(argv[1]) == "--version") {
        std::cout << "OpenMagFDM " << OPENMAGFDM_VERSION << std::endl;
        return 0;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "2D Magnetic Field Analyzer (FDM)" << std::endl;
    std::cout << "Cartesian / Polar Coordinate Systems" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check command-line arguments
    if (argc < 3) {
        std::cerr << "\nUsage: " << argv[0] << " <config.yaml> <image.png> [output_folder]" << std::endl;
        std::cerr << "  config.yaml    : YAML configuration file" << std::endl;
        std::cerr << "  image.png      : Material distribution image (RGB)" << std::endl;
        std::cerr << "  output_folder  : Output folder name (optional, default: timestamped)" << std::endl;
        std::cerr << "\nOutput structure:" << std::endl;
        std::cerr << "  output_folder/" << std::endl;
        std::cerr << "    Az/step_0001.tiff" << std::endl;
        std::cerr << "    Mu/step_0001.tiff" << std::endl;
        std::cerr << "    BoundaryImg/step_0001.png" << std::endl;
        std::cerr << "    Forces/step_0001.csv" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::string image_path = argv[2];
    std::string base_folder;

    // Determine output folder
    if (argc >= 4) {
        base_folder = getBaseFolderName(argv[3]);
    } else {
        base_folder = generateTimestampFolderName();
        std::cout << "\nOutput folder not specified. Using timestamp: " << base_folder << std::endl;
    }

    // Create the output folder before exporting conditions.json.  Native
    // filesystem APIs handle spaces and nested user paths consistently on
    // Windows, macOS, and Linux (the old shell mkdir could fail silently).
    std::error_code mkdir_error;
    std::filesystem::create_directories(base_folder, mkdir_error);
    if (mkdir_error) {
        std::cerr << "Failed to create output folder '" << base_folder
                  << "': " << mkdir_error.message() << std::endl;
        return 1;
    }

    // Setup log file output (tee to both console and file)
    std::ofstream log_file(base_folder + "/log.txt");
    std::streambuf* cout_original = std::cout.rdbuf();
    TeeBuffer tee_buffer(cout_original, log_file.rdbuf());
    std::cout.rdbuf(&tee_buffer);

    try {
        // Export analysis conditions to JSON (before initialization)
        std::cout << "\n=== Exporting Conditions ===" << std::endl;
        exportConditionsJSON(base_folder + "/conditions.json", config_path, image_path);
        std::cout << "Conditions saved to: " << base_folder << "/conditions.json" << std::endl;
        // Initialize analyzer
        std::cout << "\n=== Initialization ===" << std::endl;
        MagneticFieldAnalyzer analyzer(config_path, image_path);

        // Check if transient analysis is enabled
        // We need to check this via a simple YAML read (analyzer doesn't expose config)
        YAML::Node config = YAML::LoadFile(config_path);
        bool transient_enabled = false;
        if (config["transient"] && config["transient"]["enabled"]) {
            transient_enabled = config["transient"]["enabled"].as<bool>();
        }

        if (transient_enabled) {
            // Perform transient analysis (includes solve, stress calc, and export for all steps)
            analyzer.performTransientAnalysis(base_folder);
        } else {
            // Static analysis: single solve
            analyzer.solve();

            // Use the same robust default as transient analysis. The legacy
            // edge-based Maxwell stress path is retained as an explicit
            // research API, but its result container is not the default
            // Forces export consumed by the WebUI.
            analyzer.calculateForceDistributedAmperian(0, 0.0);

            // Calculate total magnetic energy
            double total_energy = analyzer.calculateTotalMagneticEnergy();
            std::cout << "Total Magnetic Energy: " << total_energy << " J/m" << std::endl;

            // Export all results to folder structure
            std::cout << "\n=== Exporting Results ===" << std::endl;
            analyzer.exportResults(base_folder, 0);
            analyzer.exportActiveOnlyResults(base_folder, 0);
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Analysis completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        // Restore original cout buffer and close log file
        std::cout.rdbuf(cout_original);
        log_file.close();
        std::cout << "\nLog file saved to: " << base_folder << "/log.txt" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n!!! Error occurred !!!" << std::endl;
        std::cerr << e.what() << std::endl;

        // Restore original cout buffer and close log file
        std::cout.rdbuf(cout_original);
        log_file.close();

        return 1;
    }
}
