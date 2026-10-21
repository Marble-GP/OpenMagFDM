#include "MagneticFieldAnalyzer.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

int main(int argc,char** argv){try{
    namespace fs=std::filesystem;
    const auto root=fs::path(argc>1?argv[1]:"dimension-compat")/std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());fs::create_directories(root);
    cv::Mat image(20,16,CV_8UC3,cv::Scalar(255,255,255));image(cv::Rect(4,3,3,5)).setTo(cv::Scalar(0,0,255));image(cv::Rect(10,0,2,20)).setTo(cv::Scalar(128,128,128));auto png=root/"model.png";if(!cv::imwrite(png.string(),image))throw std::runtime_error("Image write failed");
    for(int mode=0;mode<3;++mode){
        std::string text="omp: {threads: 2}\nvariables: {Ipeak: 1000}\nmaterial_presets:\n  winding: {mu_r: 1, jz: '$Ipeak'}\nmaterials:\n  air: {rgb: [255,255,255], mu_r: 1}\n  coil: {rgb: [255,0,0], preset: winding}\n  iron: {rgb: [128,128,128], mu_r: 100}\nexport: {async: false}\n";
        if(mode==0)text+="coordinate_system: cartesian\nmesh: {dx: 0.001, dy: 0.002}\nboundary_conditions:\n  left: {type: dirichlet, value: 0}\n  right: {type: dirichlet, value: 0}\n  bottom: {type: dirichlet, value: 0}\n  top: {type: dirichlet, value: 0}\n";
        else text+="coordinate_system: polar\npolar_domain: {r_start: 0.01, r_end: 0.05, theta_range: '2*pi', r_orientation: horizontal}\npolar_boundary_conditions:\n  inner: {type: dirichlet, value: 0}\n  outer: {type: dirichlet, value: 0}\n  theta_min: {type: periodic, value: 1}\n  theta_max: {type: periodic, value: 1}\n";
        auto yaml=YAML::Load(text);
        if(mode==2){yaml["materials"]["iron"].remove("mu_r");yaml["materials"]["iron"]["B-H"]="$H/(40+0.52*$H)+4*pi*1e-7*$H";yaml["nonlinear_solver"]=YAML::Load("{enabled: true, solver_type: newton-krylov, max_iterations: 200, tolerance: 0.001}");yaml["transient"]=YAML::Load("{enabled: true, total_steps: 3, enable_sliding: true, parallel_chunks: 1, slide_direction: vertical, slide_region_start: 0, slide_region_end: 8, slide_pixels_per_step: 1, export_fields: [Az]}");}
        auto legacy=root/("legacy"+std::to_string(mode)+".yaml"),extended=root/("extended"+std::to_string(mode)+".yaml");{std::ofstream o(legacy);o<<yaml;}
        yaml["materials"]["coil"]["conductivity"]=5.8e7;yaml["variables"]["coil_turns"]=100;
        yaml["model_3d"]=YAML::Load("schema_version: 1\ngeometry: {voxels: missing.vti, manifest: missing.json}\nmaterials:\n  only_3d: {mu_r: 99999, conductivity: 10}\ninitial_az_path: missing-inactive-reference.bin\n");{std::ofstream o(extended);o<<yaml;}
        MagneticFieldAnalyzer a(legacy.string(),png.string()),b(extended.string(),png.string());
        if(mode==2){a.performTransientAnalysis((root/"legacy-sweep").string());b.performTransientAnalysis((root/"extended-sweep").string());}
        else{a.solve();b.solve();}
        if(!a.analysisConverged()||!b.analysisConverged()||a.getAz().norm()==0||(a.getAz()-b.getAz()).norm()>1e-12*std::max(1.,a.getAz().norm()))throw std::runtime_error("3D extension changed 2D result in mode "+std::to_string(mode));
    }
    std::cout<<"Unchanged Cartesian, polar and legacy sliding/B-H YAML results with inactive 3D extension\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
