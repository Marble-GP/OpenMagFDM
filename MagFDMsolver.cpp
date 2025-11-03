#include "MagneticFieldAnalyzer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>

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

    // Open output file
    std::ofstream json_file(output_path);
    if (!json_file.is_open()) {
        throw std::runtime_error("Failed to create JSON file: " + output_path);
    }

    json_file << "{\n";

    // Coordinate system
    std::string coord_system = config["coordinate_system"]
        ? config["coordinate_system"].as<std::string>() : "cartesian";
    json_file << "  \"coordinate_system\": \"" << coord_system << "\",\n";

    // Image dimensions
    json_file << "  \"image_width\": " << image.cols << ",\n";
    json_file << "  \"image_height\": " << image.rows << ",\n";

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
        json_file << "  \"dx\": " << dx << ",\n";
        json_file << "  \"dy\": " << dy << ",\n";
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

        double theta_range_deg = polar_section["theta_range"]
            ? polar_section["theta_range"].as<double>() : 360.0;
        double theta_range_rad = theta_range_deg * M_PI / 180.0;
        double dtheta = theta_range_rad / (ntheta - 1);

        json_file << "  \"dr\": " << dr << ",\n";
        json_file << "  \"dtheta\": " << dtheta << ",\n";

        // Polar parameters
        json_file << "  \"polar\": {\n";
        json_file << "    \"r_start\": " << r_start << ",\n";
        json_file << "    \"r_end\": " << r_end << ",\n";
        json_file << "    \"theta_range\": " << theta_range_rad << ",\n";
        std::string r_orientation = polar_section["r_orientation"]
            ? polar_section["r_orientation"].as<std::string>() : "horizontal";
        json_file << "    \"r_orientation\": \"" << r_orientation << "\"\n";
        json_file << "  },\n";
    }

    // Boundary conditions
    json_file << "  \"boundary_conditions\": {\n";

    auto writeBoundary = [&](const std::string& name, const YAML::Node& bc_node) {
        std::string type = "dirichlet";
        double value = 0.0;
        if (bc_node) {
            if (bc_node["type"]) type = bc_node["type"].as<std::string>();
            if (bc_node["value"]) value = bc_node["value"].as<double>();
        }
        json_file << "    \"" << name << "\": {\"type\": \"" << type
                  << "\", \"value\": " << value << "}";
    };

    // Try boundary_conditions first, then boundary
    YAML::Node bc_section;
    if (config["boundary_conditions"]) {
        bc_section = config["boundary_conditions"];
    } else if (config["boundary"]) {
        bc_section = config["boundary"];
    }

    if (coord_system == "cartesian") {
        writeBoundary("left", bc_section["left"]);
        json_file << ",\n";
        writeBoundary("right", bc_section["right"]);
        json_file << ",\n";
        writeBoundary("bottom", bc_section["bottom"]);
        json_file << ",\n";
        writeBoundary("top", bc_section["top"]);
        json_file << "\n";
    } else if (coord_system == "polar") {
        // Try polar_boundary_conditions first, then boundary
        YAML::Node polar_bc_section;
        if (config["polar_boundary_conditions"]) {
            polar_bc_section = config["polar_boundary_conditions"];
        } else if (config["boundary"]) {
            polar_bc_section = config["boundary"];
        }

        writeBoundary("inner", polar_bc_section["inner"]);
        json_file << ",\n";
        writeBoundary("outer", polar_bc_section["outer"]);
        json_file << ",\n";
        writeBoundary("theta_min", polar_bc_section["theta_min"]);
        json_file << ",\n";
        writeBoundary("theta_max", polar_bc_section["theta_max"]);
        json_file << "\n";
    }

    json_file << "  },\n";

    // Transient configuration
    json_file << "  \"transient\": {\n";
    if (config["transient"] && config["transient"]["enabled"]) {
        bool enabled = config["transient"]["enabled"].as<bool>();
        json_file << "    \"enabled\": " << (enabled ? "true" : "false") << ",\n";

        if (enabled) {
            bool enable_sliding = config["transient"]["enable_sliding"]
                ? config["transient"]["enable_sliding"].as<bool>() : true;
            int total_steps = config["transient"]["total_steps"].as<int>();

            json_file << "    \"enable_sliding\": " << (enable_sliding ? "true" : "false") << ",\n";
            json_file << "    \"total_steps\": " << total_steps << ",\n";

            if (enable_sliding) {
                std::string slide_direction = config["transient"]["slide_direction"].as<std::string>();
                int slide_region_start = config["transient"]["slide_region_start"].as<int>();
                int slide_region_end = config["transient"]["slide_region_end"].as<int>();
                int slide_pixels_per_step = config["transient"]["slide_pixels_per_step"].as<int>();

                json_file << "    \"slide_direction\": \"" << slide_direction << "\",\n";
                json_file << "    \"slide_region_start\": " << slide_region_start << ",\n";
                json_file << "    \"slide_region_end\": " << slide_region_end << ",\n";
                json_file << "    \"slide_pixels_per_step\": " << slide_pixels_per_step << "\n";
            }
        }
    } else {
        json_file << "    \"enabled\": false\n";
    }
    json_file << "  }\n";

    json_file << "}\n";
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
