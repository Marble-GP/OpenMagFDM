#include "InputSnapshot.h"
#include <chrono>
#include <iostream>
int main(int argc, char** argv) {
    namespace fs=std::filesystem;
    try {
        if(argc!=2)throw std::runtime_error("Expected test artifact directory");
        const auto root=fs::absolute(argv[1])/std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        if(!fs::create_directories(root))throw std::runtime_error("Test directory already exists");
        const auto original=root/"original";fs::create_directory(original);const auto out=root/"result";fs::create_directory(out);
        {std::ofstream file(original/"field.bin",std::ios::binary);file<<"abc";}
        if(omfdm::inputDigest(original/"field.bin")!="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")throw std::runtime_error("SHA256 mismatch");
        {std::ofstream file(original/"image.png");file<<"initial image";}
        YAML::Node config;config["coordinate_system"]="cartesian";config["initial_az_path"]=(original/"field.bin").string();
        config["model_3d"]["geometry"]["voxels"]="missing-3d.vti";
        config["model_3d"]["initial_az_path"]="missing-inactive-reference.bin";
        {std::ofstream file(original/"config.yaml");file<<config;}
        std::string yaml=(original/"config.yaml").string(),image=(original/"image.png").string();
        omfdm::saveInputSnapshot(out.string(),yaml,image,"test-version");
        if(YAML::LoadFile(yaml)["model_3d"]["initial_az_path"].as<std::string>()!="missing-inactive-reference.bin")throw std::runtime_error("Inactive 3D section was modified");
        {std::ofstream file(original/"image.png");file<<"changed image";}
        std::ifstream manifestFile(out/"run-inputs.json");nlohmann::json manifest;manifestFile>>manifest;
        if(manifest["solverVersion"]!="test-version"||manifest["files"].size()!=3)throw std::runtime_error("Manifest incomplete");
        if(omfdm::inputDigest(image)==omfdm::inputDigest(original/"image.png"))throw std::runtime_error("Snapshot was not frozen");
        omfdm::saveInputSnapshot(out.string(),yaml,image,"test-version");
        const auto migrated=root/"migrated";
        fs::copy(out,migrated,fs::copy_options::recursive);
        yaml=(migrated/".inputs"/"config.yaml").string();image=(migrated/".inputs"/"model.png").string();
        const auto reference=YAML::LoadFile(yaml)["initial_az_path"].as<std::string>();
        if(fs::path(reference).is_absolute()||omfdm::inputDigest(migrated/reference)!=omfdm::inputDigest(original/"field.bin"))throw std::runtime_error("Migrated reference is not portable");
        omfdm::saveInputSnapshot(migrated.string(),yaml,image,"test-version");
        {std::ofstream file(image);file<<"tampered";}
        bool rejected=false;try{omfdm::saveInputSnapshot(migrated.string(),yaml,image,"test-version");}catch(...){rejected=true;}
        if(!rejected)throw std::runtime_error("Tampered snapshot accepted");
        std::cout<<"Input snapshot checks passed; artifacts: "<<root<<std::endl;return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
}
