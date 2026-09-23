#include "MagneticFieldAnalyzer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    try {
        const auto dir = std::filesystem::path(argc > 1 ? argv[1] : "chunk_threads_fixture");
        std::filesystem::create_directories(dir);
        const auto yaml = dir/"chunks.yaml", image = dir/"chunks.png";
        std::ofstream f(yaml);
        f << "omp: {threads: 4}\ncoordinate_system: cartesian\nmesh: {dx: 0.001, dy: 0.001}\n"
             "materials:\n  air: {rgb: [255,255,255], mu_r: 1, jz: 0}\n"
             "boundary_conditions:\n  left: {type: dirichlet, value: 0}\n"
             "  right: {type: dirichlet, value: 0}\n  bottom: {type: dirichlet, value: 0}\n"
             "  top: {type: dirichlet, value: 0}\n"
             "transient: {enabled: true, total_steps: 4, enable_sliding: false, parallel_chunks: 2, export_fields: [Az]}\n"
             "export: {async: false}\n";
        f.close();
        if (!cv::imwrite(image.string(),cv::Mat(10,12,CV_8UC3,cv::Scalar(255,255,255))))
            throw std::runtime_error("fixture image write failed");
        MagneticFieldAnalyzer analyzer(yaml.string(),image.string());
#ifdef _OPENMP
        const int before = omp_get_max_threads();
        const int dynamic = omp_get_dynamic();
#endif
        analyzer.performTransientAnalysis((dir/"results").string());
#ifdef _OPENMP
        if (omp_get_max_threads() != before || omp_get_dynamic() != dynamic)
            throw std::runtime_error("parent OpenMP configuration not restored");
        if (analyzer.getLastChunkOpenMPTeams().size() != 2)
            throw std::runtime_error("missing chunk team observations");
        for (int k = 0; k < 2; ++k)
            if (analyzer.getLastChunkOpenMPTeams()[k] != 2)
                throw std::runtime_error("chunk team differs from its configured share");
#endif
        std::cerr << "Chunk team size and parent restoration passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
