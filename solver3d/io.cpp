#include "core.h"
#include <fstream>
#include <sstream>
#include <regex>
#include <cmath>
#include <iomanip>
#include <limits>

namespace omfdm::d3 {
static std::string attribute(const std::string& tag,const std::string& name){
    std::smatch m;require(std::regex_search(tag,m,std::regex(name+"=\"([^\"]*)\"")),"Missing VTI attribute: "+name);return m[1];
}
Grid readGrid(const fs::path& path,size_t maxCells){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Cannot read VTI");
    std::string header;char ch;bool marker=false;
    while(in.get(ch)) {header+=ch;if(header.size()>1024*1024)throw std::runtime_error("VTI header too large");if(ch=='_'&&header.find("<AppendedData")!=std::string::npos){marker=true;break;}}
    require(marker,"Expected raw-appended VTI");auto start=in.tellg();
    require(header.find("compressor=")==std::string::npos&&attribute(header,"type")=="ImageData"&&attribute(header,"byte_order")=="LittleEndian"&&attribute(header,"header_type")=="UInt64", "Only uncompressed LittleEndian UInt64 ImageData is supported");
    auto app=header.substr(header.find("<AppendedData"));require(attribute(app,"encoding")=="raw","Expected raw encoding");
    std::smatch match;require(std::regex_search(header,match,std::regex("<ImageData[^>]*>")),"Missing ImageData");auto image=match.str();
    Grid g;Index lower;std::istringstream ext(attribute(image,"WholeExtent")),origin(attribute(image,"Origin")),spacing(attribute(image,"Spacing"));
    size_t cells=1;
    for(int a=0;a<3;++a){int upper;require(bool(ext>>lower[a]>>upper)&&lower[a]==0&&upper>0,"Invalid/unsupported VTI extent");g.size[a]=upper;require(size_t(upper)<=maxCells/cells,"Grid exceeds max_cells");cells*=upper;require(bool(origin>>g.origin[a])&&std::isfinite(g.origin[a]),"Invalid VTI origin");require(bool(spacing>>g.spacing[a])&&std::isfinite(g.spacing[a])&&g.spacing[a]>0,"Invalid VTI spacing");}
    require(std::regex_search(header,match,std::regex("<Piece[^>]*>"))&&attribute(match.str(),"Extent")==attribute(image,"WholeExtent"),"Only a single whole-grid VTI piece is supported");
    auto cellStart=header.find("<CellData"),cellEnd=header.find("</CellData>");require(cellStart!=std::string::npos&&cellEnd>cellStart,"Missing cell data");auto section=header.substr(cellStart,cellEnd-cellStart);
    auto readArray=[&](const std::string& name){
        std::regex re("<DataArray[^>]*>");std::string tag;
        for(auto it=std::sregex_iterator(section.begin(),section.end(),re);it!=std::sregex_iterator();++it)if(attribute(it->str(),"Name")==name){require(tag.empty(),"Duplicate VTI array");tag=it->str();}
        require(!tag.empty()&&attribute(tag,"type")=="UInt32"&&attribute(tag,"format")=="appended","Missing/invalid UInt32 "+name);
        if(tag.find("NumberOfComponents=")!=std::string::npos)require(attribute(tag,"NumberOfComponents")=="1","Expected scalar IDs");
        uint64_t offset=std::stoull(attribute(tag,"offset"));require(offset<=uint64_t(std::numeric_limits<std::streamoff>::max())-uint64_t(start),"Invalid VTI offset");
        in.clear();in.seekg(start+std::streamoff(offset));unsigned char bytes[8]{};in.read(reinterpret_cast<char*>(bytes),8);require(bool(in),"Truncated VTI header");uint64_t length=0;for(int b=0;b<8;++b)length|=uint64_t(bytes[b])<<(b*8);
        require(length==cells*4,"VTI cell array length mismatch");std::vector<uint32_t> values(cells);
        for(auto& value:values){in.read(reinterpret_cast<char*>(bytes),4);require(bool(in),"Truncated VTI payload");value=0;for(int b=0;b<4;++b)value|=uint32_t(bytes[b])<<(b*8);}return values;
    };
    g.material=readArray("material_id");g.solid=readArray("solid_id");
    for(size_t i=0;i<cells;++i)require((g.material[i]==0)==(g.solid[i]==0),"Air/material/solid IDs disagree");
    g.hash=digest(path);return g;
}
static std::ofstream output(const fs::path& path){std::ofstream out(path,std::ios::binary);out.exceptions(std::ios::badbit|std::ios::failbit);out.imbue(std::locale::classic());out<<std::setprecision(17);return out;}
void writeGrid(const fs::path& path,const Grid& g,const std::map<std::string,std::vector<double>>& scalars,const std::vector<Vec>& vectors){
    auto out=output(path);std::ostringstream extent;for(int a=0;a<3;++a)extent<<"0 "<<g.size[a]<<' ';
    out<<"<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\"><ImageData WholeExtent=\""<<extent.str()<<"\" Origin=\"";
    for(double v:g.origin)out<<v<<' ';out<<"\" Spacing=\"";for(double v:g.spacing)out<<v<<' ';out<<"\"><Piece Extent=\""<<extent.str()<<"\"><PointData/><CellData>\n";
    auto array=[&](const std::string& name,const auto& values,const char* type){require(values.size()==g.count(),"Output array size mismatch");out<<"<DataArray type=\""<<type<<"\" Name=\""<<name<<"\" format=\"ascii\">\n";for(auto v:values)out<<v<<' ';out<<"\n</DataArray>\n";};
    array("material_id",g.material,"UInt32");array("solid_id",g.solid,"UInt32");for(const auto& entry:scalars)array(entry.first,entry.second,"Float64");
    if(!vectors.empty()){require(vectors.size()==g.count(),"Vector array size mismatch");out<<"<DataArray type=\"Float64\" Name=\"current_density_visualization\" NumberOfComponents=\"3\" format=\"ascii\">\n";for(auto v:vectors)for(double x:v)out<<x<<' ';out<<"\n</DataArray>\n";}
    out<<"</CellData></Piece></ImageData></VTKFile>\n";
}
std::vector<SelectedFace> readFaces(const fs::path& path,const Grid& grid){
    auto j=readJson(path);require(j.at("schema_version")==1&&j.at("grid_sha256")==grid.hash,"Face list version or grid hash mismatch");
    std::vector<SelectedFace> faces;std::set<FaceKey> seen;
    for(const auto& n:j.at("faces")){
        SelectedFace f;f.key.axis=n.at("axis").get<int>();f.key.index=n.at("index").get<Index>();f.sign=n.at("sign").get<int>();require(f.key.axis>=0&&f.key.axis<3&&(f.sign==1||f.sign==-1),"Invalid oriented face");
        for(int a=0;a<3;++a)require(f.key.index[a]>=0&&f.key.index[a]<grid.size[a]+(a==f.key.axis),"Face index outside grid");
        require(seen.insert(f.key).second,"Duplicate face");faces.push_back(f);
    }
    require(!faces.empty(),"Empty face list");return faces;
}
void writeFaces(const fs::path& path,const Grid& g,const std::vector<SelectedFace>& faces){
    Json j={{"schema_version",1},{"grid_sha256",g.hash},{"faces",Json::array()}};
    for(auto f:faces)j["faces"].push_back({{"axis",f.key.axis},{"index",f.key.index},{"sign",f.sign}});writeJson(path,j);
}
void writeSurface(const fs::path& path,const Grid& g,const std::vector<SelectedFace>& faces){
    auto out=output(path);size_t count=faces.size();out<<"<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\"><PolyData><Piece NumberOfPoints=\""<<count*4<<"\" NumberOfPolys=\""<<count<<"\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\"><Points><DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for(auto f:faces){int u=(f.key.axis+1)%3,v=(f.key.axis+2)%3;const int corners[4][2]={{0,0},{1,0},{1,1},{0,1}};for(int j=0;j<4;++j){int k=f.sign>0?j:3-j;for(int a=0;a<3;++a)out<<g.origin[a]+(f.key.index[a]+(a==u?corners[k][0]:a==v?corners[k][1]:0))*g.spacing[a]<<' ';}}
    out<<"</DataArray></Points><Polys><DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">";for(size_t i=0;i<count*4;++i)out<<i<<' ';out<<"</DataArray><DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">";for(size_t i=1;i<=count;++i)out<<i*4<<' ';out<<"</DataArray></Polys></Piece></PolyData></VTKFile>\n";
}
Config snapshot(const Config& c,const fs::path& folder){
    require(!fs::exists(folder),"Output already exists; choose a new directory");require(fs::create_directories(folder/".inputs"),"Cannot create snapshot");
    Json manifest={{"version",1},{"solverVersion","3d-preprocessor-0.1.0"},{"step",c.step},{"files",Json::array()}};
    auto copy=[&](const fs::path& source,const std::string& name){auto target=folder/".inputs"/name;auto hash=digest(source);fs::copy_file(source,target);require(digest(target)==hash&&digest(source)==hash,"Input changed during snapshot");manifest["files"].push_back({{"path",".inputs/"+name},{"sha256",hash},{"source",source.u8string()}});return name;};
    copy(c.source,"source-config.yaml");auto yaml=YAML::LoadFile((folder/".inputs/source-config.yaml").string());auto m=yaml["model_3d"];
    m["geometry"]["voxels"]=copy(c.voxels,"model.vti");m["geometry"]["manifest"]=copy(c.manifest,"voxel-manifest.json");
    for(size_t i=0;i<c.coils.size();++i){const auto& coil=c.coils[i];auto n=m["coils"][i];if(!coil.cut.empty())n["cut"]["faces"]=copy(coil.cut,coil.name+"-cut.json");else{n["terminals"]["inlet"]["faces"]=copy(coil.inlet,coil.name+"-inlet.json");n["terminals"]["outlet"]["faces"]=copy(coil.outlet,coil.name+"-outlet.json");}}
    auto config=folder/".inputs/config.yaml";{auto out=output(config);out<<yaml;}
    manifest["effectiveConfigHash"]=digest(config);writeJson(folder/"run-inputs.json",manifest);return readConfig(config,c.step);
}
void saveResult(const fs::path& folder,const Grid& g,const Domain& d,const Result& r){
    require(fs::create_directory(folder),"Result coil directory already exists");
    std::vector<double> potential(g.count(),0),mask(g.count(),0);std::vector<Vec> density(g.count());
    for(size_t i=0;i<d.cells.size();++i){potential[d.cells[i]]=r.potential[i];mask[d.cells[i]]=1;density[d.cells[i]]=r.density[i];}
    writeGrid(folder/"current.vti",g,{{"auxiliary_potential",potential},{"coil_mask",mask}},density);
    auto out=output(folder/"face-currents.json");out<<"{\"schema_version\":1,\"units\":\"A\",\"orientation\":\"positive_axis\",\"grid_sha256\":\""<<g.hash<<"\",\"faces\":[\n";
    for(size_t i=0;i<d.faces.size();++i){const auto& f=d.faces[i];if(i)out<<",\n";out<<Json({{"axis",f.key.axis},{"index",f.key.index},{"current",r.flux[i]}}).dump();}out<<"]}\n";out.close();
    writeSurface(folder/"selected-faces.vtp",g,d.selected);writeJson(folder/"diagnostics.json",r.diagnostics);
}
}
