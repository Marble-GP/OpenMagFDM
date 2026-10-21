#include "core.h"
#include <iostream>
#include <sstream>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace omfdm::d3;
int main(int argc,char** argv){try{
    if(argc<2||std::string(argv[1])=="--help"){
        std::cout<<"MagFDM3D 0.1.0 - experimental current-source preprocessor (no magnetic solve)\n"
        "  validate --config FILE [--step N]\n  prepare-current --config FILE --output NEW_DIRECTORY [--step N]\n"
        "  select-faces --config FILE --coil NAME --output NEW_JSON --point X,Y,Z --normal X,Y,Z\n"
        "    --min X,Y,Z --max X,Y,Z --tolerance METRES [--cut]\n"
        "Face selection uses face centers within plane tolerance and an axis-aligned bounding box.\n";return 0;}
    std::string command=argv[1];require(command=="validate"||command=="prepare-current"||command=="select-faces","Unknown command; magnetic solve is not implemented");
    std::map<std::string,std::string> args;bool cut=false;
    for(int i=2;i<argc;++i){std::string key=argv[i];if(key=="--cut"){require(!cut,"Duplicate --cut");cut=true;continue;}require(i+1<argc&&key.rfind("--",0)==0&&!args.count(key),"Invalid/duplicate option");args[key]=argv[++i];}
    std::set<std::string> allowed={"--config","--step"};if(command=="prepare-current")allowed.insert("--output");if(command=="select-faces")for(auto key:{"--coil","--output","--point","--normal","--min","--max","--tolerance"})allowed.insert(key);
    for(auto entry:args)require(allowed.count(entry.first),"Unknown option: "+entry.first);require(!cut||command=="select-faces","--cut only applies to select-faces");
    auto get=[&](const std::string& key){require(args.count(key),"Missing "+key);return args.at(key);};
    int step=0;if(args.count("--step")){size_t used=0;step=std::stoi(args.at("--step"),&used);require(used==args.at("--step").size()&&step>=0,"Invalid step");}
    Config c=readConfig(fs::u8path(get("--config")),step);
#ifdef _OPENMP
    omp_set_num_threads(c.threads>0?c.threads:std::min(8,omp_get_max_threads()));
#endif
    fs::path folder;if(command=="prepare-current"){folder=fs::absolute(fs::u8path(get("--output")));c=snapshot(c,folder);}
    auto grid=readGrid(c.voxels,c.maxCells);auto manifest=readJson(c.manifest);require(manifest.is_object(),"Invalid voxel manifest");
    if(command=="select-faces"){
        auto name=get("--coil");auto coil=std::find_if(c.coils.begin(),c.coils.end(),[&](const Coil& v){return v.name==name;});require(coil!=c.coils.end(),"Unknown coil");
        auto vector=[&](const std::string& key){auto value=get(key);std::replace(value.begin(),value.end(),',',' ');std::istringstream in(value);Vec result;for(double& n:result)require(bool(in>>n)&&std::isfinite(n),"Expected three finite numbers for "+key);std::string extra;require(!(in>>extra),"Extra vector component");return result;};
        auto path=fs::u8path(get("--output")),surface=path;surface.replace_extension("vtp");require(!fs::exists(path)&&!fs::exists(surface),"Face output exists");
        auto faces=selectFaces(c,grid,*coil,vector("--point"),vector("--normal"),vector("--min"),vector("--max"),std::stod(get("--tolerance")),cut);writeFaces(path,grid,faces);writeSurface(surface,grid,faces);std::cout<<"Selected "<<faces.size()<<" faces\n";return 0;
    }
    bool accepted=true;Json summary={{"schema_version",1},{"stage","current_source_only"},{"coils",Json::array()}};
    for(const auto& coil:c.coils){auto domain=makeDomain(c,grid,coil);if(command=="validate"){std::cout<<coil.name<<": "<<domain.cells.size()<<" cells; geometry and terminals valid\n";continue;}
        auto result=prepareCurrent(grid,domain,coil);saveResult(folder/coil.name,grid,domain,result);accepted=accepted&&result.accepted;summary["coils"].push_back({{"name",coil.name},{"accepted",result.accepted}});std::cout<<coil.name<<": "<<result.diagnostics.dump()<<'\n';}
    if(command=="prepare-current"){summary["accepted"]=accepted;writeJson(folder/"result.json",summary);}return accepted?0:2;
}catch(const std::exception& e){std::cerr<<"MagFDM3D: "<<e.what()<<'\n';return 1;}}
