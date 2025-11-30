#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include "json.hpp"

using json = nlohmann::json;

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
void exportConditionsJSON(const std::string& output_path,
                          const std::string& config_path,
                          const std::string& image_path) {
    // Load YAML configuration
    YAML::Node config = YAML::LoadFile(config_path);

    // Load image to get dimensions
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Create JSON object
    json j;

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
        bool enabled = config["transient"]["enabled"].as<bool>();
        j["transient"]["enabled"] = enabled;

        if (enabled) {
            bool enable_sliding = config["transient"]["enable_sliding"]
                ? config["transient"]["enable_sliding"].as<bool>() : true;
            int total_steps = config["transient"]["total_steps"].as<int>();

            j["transient"]["enable_sliding"] = enable_sliding;
            j["transient"]["total_steps"] = total_steps;

            if (enable_sliding) {
                std::string slide_direction = config["transient"]["slide_direction"].as<std::string>();
                int slide_region_start = config["transient"]["slide_region_start"].as<int>();
                int slide_region_end = config["transient"]["slide_region_end"].as<int>();
                int slide_pixels_per_step = config["transient"]["slide_pixels_per_step"].as<int>();

                j["transient"]["slide_direction"] = slide_direction;
                j["transient"]["slide_region_start"] = slide_region_start;
                j["transient"]["slide_region_end"] = slide_region_end;
                j["transient"]["slide_pixels_per_step"] = slide_pixels_per_step;
            }
        }
    } else {
        j["transient"]["enabled"] = false;
    }

    // Nonlinear materials detection
    bool has_nonlinear_materials = false;
    if (config["materials"]) {
        for (const auto& material : config["materials"]) {
            const auto& props = material.second;
            if (!props["mu_r"]) continue;

            // Check if mu_r is nonlinear (formula or table)
            if (props["mu_r"].IsScalar()) {
                std::string mu_str = props["mu_r"].as<std::string>();
                // Check for formula characters
                if (mu_str.find('$') != std::string::npos ||
                    mu_str.find('*') != std::string::npos ||
                    mu_str.find('/') != std::string::npos ||
                    mu_str.find('+') != std::string::npos ||
                    mu_str.find('(') != std::string::npos ||
                    mu_str.find("exp") != std::string::npos ||
                    mu_str.find("tanh") != std::string::npos) {
                    has_nonlinear_materials = true;
                    break;
                }
            } else if (props["mu_r"].IsSequence() && props["mu_r"].size() == 2) {
                // B-H table format
                has_nonlinear_materials = true;
                break;
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

    // Write to file with proper indentation
    std::ofstream json_file(output_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to create JSON file: " + output_path);
    }
    json_file << j.dump(2) << std::endl;  // indent with 2 spaces
    json_file.close();
}

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "2D Magnetic Field Analyzer (FDM)" << std::endl;
    std::cout << "Cartesian Coordinate System" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check command-line arguments
    if (argc < 3) {
        std::cerr << "\nUsage: " << argv[0] << " <config.yaml> <image.png> [output_folder]" << std::endl;
        std::cerr << "  config.yaml    : YAML configuration file" << std::endl;
        std::cerr << "  image.png      : Material distribution image (RGB)" << std::endl;
        std::cerr << "  output_folder  : Output folder name (optional, default: timestamped)" << std::endl;
        std::cerr << "\nOutput structure:" << std::endl;
        std::cerr << "  output_folder/" << std::endl;
        std::cerr << "    Az/step_0000.csv" << std::endl;
        std::cerr << "    Mu/step_0000.csv" << std::endl;
        std::cerr << "    BoundaryImg/step_0000.png" << std::endl;
        std::cerr << "    Forces/step_0000.csv" << std::endl;
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

    // Create output folder if it doesn't exist
    system(("mkdir -p \"" + base_folder + "\"").c_str());

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

            // Calculate Maxwell stress and forces
            analyzer.calculateMaxwellStress();

            // Calculate total magnetic energy
            double total_energy = analyzer.calculateTotalMagneticEnergy();
            std::cout << "Total Magnetic Energy: " << total_energy << " J/m" << std::endl;

            // Export all results to folder structure
            std::cout << "\n=== Exporting Results ===" << std::endl;
            analyzer.exportResults(base_folder, 0);
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
