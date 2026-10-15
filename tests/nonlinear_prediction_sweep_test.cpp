#include "MagneticFieldAnalyzer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

static void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    if (*value) setenv(key,value,1); else unsetenv(key);
#endif
}
static int count(const std::string& log, const std::string& marker) {
    int n=0; std::size_t pos=0;
    while ((pos=log.find(marker,pos))!=std::string::npos) {++n; pos+=marker.size();}
    return n;
}
int main(int argc, char** argv) {
    try {
        const auto root=std::filesystem::path(argc>1?argv[1]:"prediction_sweep_fixture");
        std::filesystem::create_directories(root);
        env("NK_PROF","1"); env("OMFDM_NK_INITIALIZATION","predict");
        env("OMFDM_TANGENT_OPTIMIZATIONS","6"); env("OMFDM_TANGENT_PRECONDITIONER","tangent-amg");
        for (bool horizontal : {true,false}) for (bool anderson : {false,true}) {
            const auto name=std::string(horizontal?"horizontal":"vertical")+(anderson?"_aa":"_plain");
            const auto yaml=root/(name+".yaml"), png=root/(name+".png");
            std::ofstream f(yaml);
            f << "omp: {threads: 2}\ncoordinate_system: polar\npolar_domain:\n"
              << "  r_start: 0.01\n  r_end: 0.05\n  theta_range: 2*pi\n  r_orientation: " << (horizontal?"horizontal":"vertical")
              << "\npolar_boundary_conditions:\n  inner: {type: dirichlet, value: 0}\n  outer: {type: dirichlet, value: 0}\n"
              << "  theta_min: {type: periodic, value: 1}\n  theta_max: {type: periodic, value: 1}\nmaterials:\n"
              << "  air: {rgb: [255,255,255], mu_r: 1}\n  coil: {rgb: [255,0,0], mu_r: 1, jz: 1000}\n"
              << "  iron: {rgb: [128,128,128], B-H: '$H/(40+0.52*$H)+4*pi*1e-7*$H'}\n"
              << "transient:\n  enabled: true\n  total_steps: 4\n  enable_sliding: true\n  parallel_chunks: 1\n"
              << "  slide_direction: " << (horizontal?"vertical":"horizontal") << "\n"
              << "  slide_region_start: 0\n  slide_region_end: 8\n  slide_pixels_per_step: 1\n  export_fields: [Az]\n"
              << "nonlinear_solver:\n  enabled: true\n  solver_type: newton-krylov\n  jacobian: tangent\n"
              << "  line_search_objective: energy\n  tolerance: 1e-4\n  max_iterations: 200\n"
              << "  anderson: {enabled: " << (anderson?"true":"false") << ", depth: 3, beta: 0.3}\nexport: {async: false}\n";
            f.close();
            cv::Mat image(24,16,CV_8UC3,cv::Scalar(255,255,255));
            image(cv::Rect(4,3,3,5)).setTo(cv::Scalar(0,0,255));
            image(cv::Rect(9,0,3,24)).setTo(cv::Scalar(128,128,128));
            if(!horizontal) cv::transpose(image,image);
            if(!cv::imwrite(png.string(),image)) throw std::runtime_error("image write failed");
            MagneticFieldAnalyzer analyzer(yaml.string(),png.string());
            for(int repeat=0;repeat<2;++repeat) {
                std::ostringstream output;
                const auto old=std::cout.rdbuf(output.rdbuf());
                try {analyzer.performTransientAnalysis((root/(name+std::to_string(repeat))).string());}
                catch(...) {std::cout.rdbuf(old);throw;}
                std::cout.rdbuf(old);
                const auto log=output.str();
                if(count(log,"[NKINIT]")!=4 || count(log,"prediction_eligible=1")!=2 || count(log,"direct_used=1")!=3)
                    throw std::runtime_error("sweep/repeated-run eligibility failed: "+name+"\n"+log);
                if(log.find("DID NOT CONVERGE")!=std::string::npos || !analyzer.lastNonlinearConverged() || !analyzer.analysisConverged())
                    throw std::runtime_error("small prediction sweep not converged: "+name+"\n"+log);
            }
        }
        std::cout << "Actual sliding and repeated sweeps: both polar layouts, AA OFF/ON passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n'; return 1;}
}
