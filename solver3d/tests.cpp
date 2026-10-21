#include "core.h"
#include "operators.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace omfdm::d3;
static void check(bool ok,const std::string& label){require(ok,"TEST FAILED: "+label);}
static void near(double a,double b,double tolerance,const std::string& label){check(std::abs(a-b)<=tolerance*std::max(1.,std::abs(b)),label);}
template<class F>static void rejects(F f,const std::string& label){bool threw=false;try{f();}catch(const std::exception&){threw=true;}check(threw,label);}
static void rawGrid(const fs::path& path,const Grid& g){
    std::ofstream o(path,std::ios::binary);o<<"<VTKFile type=\"ImageData\" byte_order=\"LittleEndian\" header_type=\"UInt64\"><ImageData WholeExtent=\"0 "<<g.size[0]<<" 0 "<<g.size[1]<<" 0 "<<g.size[2]<<"\" Origin=\"0 0 0\" Spacing=\""<<g.spacing[0]<<' '<<g.spacing[1]<<' '<<g.spacing[2]<<"\"><Piece Extent=\"0 "<<g.size[0]<<" 0 "<<g.size[1]<<" 0 "<<g.size[2]<<"\"><CellData><DataArray type=\"UInt32\" Name=\"material_id\" format=\"appended\" offset=\"0\"/><DataArray type=\"UInt32\" Name=\"solid_id\" format=\"appended\" offset=\""<<g.count()*4+8<<"\"/></CellData></Piece></ImageData><AppendedData encoding=\"raw\">_";
    auto integer=[&](uint64_t v,int n){for(int i=0;i<n;++i)o.put(char(v>>(8*i)));};for(auto array:{g.material,g.solid}){integer(g.count()*4,8);for(auto v:array)integer(v,4);}o<<"</AppendedData></VTKFile>";
}
static Config base(const fs::path& root,const Grid& g,const std::string& name){
    Config c;c.source=root/(name+".yaml");c.voxels=root/(name+".vti");c.manifest=root/(name+"-manifest.json");rawGrid(c.voxels,g);writeJson(c.manifest,{{"units","m"}});
    c.regions["wire"]={{1},"copper"};c.materials["copper"]=YAML::Load("{conductivity: 10, mu_r: 1}");Coil coil;coil.name="wire";coil.model="bulk_dc";coil.regions={"wire"};coil.current=2;coil.inlet=root/(name+"-in.json");coil.outlet=root/(name+"-out.json");c.coils={coil};return c;
}
static void terminals(Config& c,Grid& g,int axis){
    g.hash=digest(c.voxels);std::vector<SelectedFace> in,out;Index n=g.size;n[axis]=1;
    CubicalComplex::each(n,[&](Index p){in.push_back({{axis,p},-1});p[axis]=g.size[axis];out.push_back({{axis,p},1});});writeFaces(c.coils[0].inlet,g,in);writeFaces(c.coils[0].outlet,g,out);
}
int main(int argc,char** argv){try{
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    fs::path root=fs::path(argc>1?argv[1]:"test-artifacts")/std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count());fs::create_directories(root);
    CubicalComplex complex{{3,4,2}};std::vector<double> scalar(CubicalComplex::count(complex.nodes()));for(size_t i=0;i<scalar.size();++i)scalar[i]=std::sin(double(i));
    for(auto values:complex.curl(complex.gradient(scalar)))for(double v:values)near(v,0,1e-14,"curl grad / gauge nullspace");
    CubicalComplex::Field edges;for(int a=0;a<3;++a){edges[a].resize(CubicalComplex::count(complex.edges(a)));for(size_t i=0;i<edges[a].size();++i)edges[a][i]=std::cos(double(i+a));}
    for(double v:complex.divergence(complex.curl(edges)))near(v,0,1e-14,"div curl");
    for(int axis=0;axis<3;++axis){
        Grid g;g.size={4,4,4};g.size[axis]=8;g.spacing={.01,.02,.03};g.material.assign(g.count(),1);g.solid=g.material;auto c=base(root,g,"rod"+std::to_string(axis));terminals(c,g,axis);
        auto read=readGrid(c.voxels);check(read.material==g.material&&read.size==g.size,"raw VTI round trip");rejects([&]{readGrid(c.voxels,2);},"allocation guard");
        auto d=makeDomain(c,g,c.coils[0]);auto result=prepareCurrent(g,d,c.coils[0]);check(result.accepted,"straight bulk accepted");auto reference=prepareCurrent(g,d,c.coils[0],true);
        for(size_t i=0;i<result.flux.size();++i)near(result.flux[i],reference.flux[i],1e-10,"AMGCL/direct parity");
        double area=g.size[(axis+1)%3]*g.spacing[(axis+1)%3]*g.size[(axis+2)%3]*g.spacing[(axis+2)%3];double resistance=g.size[axis]*g.spacing[axis]/(10*area);
        near(result.diagnostics["resistance_ohm"],resistance,1e-10,"R=L/sigma S");for(auto j:result.density)near(j[axis],2/area,1e-9,"J=I/S");
        auto coil=c.coils[0];coil.model="stranded_uniform";coil.turns=30;auto sd=makeDomain(c,g,coil);auto stranded=prepareCurrent(g,sd,coil);check(stranded.accepted,"uniform straight bundle");for(auto j:stranded.density)near(j[axis],60/area,1e-9,"J=NI/S");
        coil.current=-2;auto negative=prepareCurrent(g,sd,coil);for(size_t i=0;i<stranded.flux.size();++i)near(negative.flux[i],-stranded.flux[i],1e-10,"negative excitation");coil.current=0;auto zero=prepareCurrent(g,sd,coil);for(auto j:zero.density)for(double v:j)near(v,0,0,"zero excitation");
        saveResult(root/("result"+std::to_string(axis)),g,d,result);
        if(axis==0){
            // Series materials: harmonic interface averaging must reproduce the exact DC resistance.
            for(size_t i=0;i<g.count();++i)if(g.position(i)[0]>=4){g.solid[i]=2;g.material[i]=2;}
            rawGrid(c.voxels,g);terminals(c,g,axis);c.regions["second"]={{2},"other"};c.materials["other"]=YAML::Load("{conductivity: 20}");c.coils[0].regions.push_back("second");auto series=prepareCurrent(g,makeDomain(c,g,c.coils[0]),c.coils[0]);near(series.diagnostics["resistance_ohm"],resistance*.75,1e-10,"harmonic interface conductance");
            auto faces=readJson(c.coils[0].inlet);faces["grid_sha256"]="stale";writeJson(root/"stale.json",faces);rejects([&]{readFaces(root/"stale.json",g);},"stale face hash");faces["grid_sha256"]=g.hash;faces["faces"].push_back(faces["faces"][0]);writeJson(root/"duplicate.json",faces);rejects([&]{readFaces(root/"duplicate.json",g);},"duplicate faces");
        }
    }
    // Closed rectangular ring: a potential jump drives a continuous current without a physical gap.
    Grid ring;ring.size={16,16,2};ring.spacing={.01,.01,.02};ring.material.resize(ring.count());ring.solid.resize(ring.count());
    for(size_t i=0;i<ring.count();++i){auto p=ring.position(i);if(p[0]<3||p[0]>=13||p[1]<3||p[1]>=13)ring.material[i]=ring.solid[i]=1;}
    auto rc=base(root,ring,"ring");ring.hash=digest(rc.voxels);rc.coils[0].cut=root/"cut.json";rc.coils[0].inlet.clear();rc.coils[0].outlet.clear();std::vector<SelectedFace> cut;for(int z=0;z<2;++z)for(int y=0;y<3;++y)cut.push_back({{0,{8,y,z}},1});writeFaces(rc.coils[0].cut,ring,cut);
    auto closed=prepareCurrent(ring,makeDomain(rc,ring,rc.coils[0]),rc.coils[0]);check(closed.accepted&&closed.diagnostics["source_status"]=="closed_cut","closed cut continuity");saveResult(root/"closed",ring,makeDomain(rc,ring,rc.coils[0]),closed);
    // YAML sharing: unused 2D reserved expressions and paths do not constrain 3D.
    auto yaml=root/"shared.yaml";{std::ofstream o(yaml);o<<"variables:\n  excitation: 2\n  unused_2d: $A*$N\nmaterial_presets:\n  copper: {conductivity: 10, magnetization: {Br: 1, angle: 0}}\nmaterials:\n  copper: {preset: copper, rgb: [255,0,0], jz: '$A', magnetization: {angle: 1}}\nmodel_3d:\n  schema_version: 1\n  geometry: {voxels: ring.vti, manifest: ring-manifest.json}\n  regions:\n    wire: {solid_ids: [1], material: copper}\n  coils:\n    - name: wire\n      regions: [wire]\n      model: bulk_dc\n      current: $excitation*cos(pi*$step)\n      cut: {faces: cut.json}\n";}
    auto config=readConfig(yaml,1);near(config.coils[0].current,-2,1e-12,"shared expression and step");near(config.materials.at("copper")["magnetization"]["Br"].as<double>(),1,0,"preset deep merge");near(config.materials.at("copper")["magnetization"]["angle"].as<double>(),1,0,"preset override");
    auto frozen=snapshot(config,root/"snapshot");check(frozen.voxels!=config.voxels&&digest(frozen.voxels)==digest(config.voxels),"input snapshot");
    auto mutate=[&](const std::string& name,const std::function<void(YAML::Node)>& edit){auto node=YAML::LoadFile(yaml.string());edit(node);auto file=root/name;{std::ofstream o(file);o<<node;}rejects([&]{readConfig(file);},name);};
    mutate("bad-turns.yaml",[](YAML::Node n){n["model_3d"]["coils"][0]["turns"]=2;});
    mutate("bad-variable.yaml",[](YAML::Node n){n["model_3d"]["coils"][0]["current"]="$A";});
    mutate("bad-material.yaml",[](YAML::Node n){n["model_3d"]["materials"]["copper"]["conductivity"]=10;});
    mutate("bad-conductivity.yaml",[](YAML::Node n){n["materials"]["copper"]["conductivity"]=0;});
    std::cout<<"3D foundation checks passed; artifacts: "<<root<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
