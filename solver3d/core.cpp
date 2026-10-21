#include "core.h"
#include "tinyexpr/tinyexpr.h"
#include "InputSnapshot.h"
#include <cmath>
#include <fstream>
#include <queue>
#include <regex>
#include <sstream>
#include <algorithm>

namespace omfdm::d3 {
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string digest(const fs::path& path) { return omfdm::inputDigest(path); }
Json readJson(const fs::path& path) { std::ifstream in(path); require(bool(in), "Cannot read " + path.string()); Json j; in >> j; return j; }
void writeJson(const fs::path& path, const Json& j) { std::ofstream out(path); out << j.dump(2) << '\n'; require(bool(out), "Cannot write " + path.string()); }
size_t Grid::count() const { return size_t(size[0]) * size[1] * size[2]; }
size_t Grid::index(int x, int y, int z) const { return size_t(x) + size_t(size[0]) * (y + size_t(size[1]) * z); }
Index Grid::position(size_t i) const { return {int(i % size[0]), int((i / size[0]) % size[1]), int(i / (size_t(size[0]) * size[1]))}; }
bool FaceKey::operator<(const FaceKey& o) const { return axis != o.axis ? axis < o.axis : index < o.index; }
static YAML::Node merge(const YAML::Node& base, const YAML::Node& over) {
    auto result = YAML::Clone(base);
    if (!result || !result.IsMap()) result = YAML::Node(YAML::NodeType::Map);
    for (const auto& p : over) {
        auto key = p.first.as<std::string>();
        result[key] = p.second.IsMap() && result[key].IsMap() ? merge(result[key], p.second) : YAML::Clone(p.second);
    }
    return result;
}
double Config::number(const YAML::Node& node) const {
    require(node && node.IsScalar(), "Expected a scalar number or expression");
    std::string expression = node.as<std::string>();
    std::regex variable(R"(\$([A-Za-z_][A-Za-z_0-9]*))");
    std::string resolved; size_t last = 0;
    for (auto it = std::sregex_iterator(expression.begin(), expression.end(), variable); it != std::sregex_iterator(); ++it) {
        auto name = (*it)[1].str(); double value;
        if (name == "step") value = step;
        else {
            auto found = variables.find(name);
            require(found != variables.end(), "Undefined or 2D-only variable $" + name + "; specify a 3D value explicitly");
            value = found->second;
        }
        resolved += expression.substr(last, size_t(it->position()) - last);
        std::ostringstream v; v.precision(17); v << '(' << value << ')'; resolved += v.str();
        last = size_t(it->position() + it->length());
    }
    resolved += expression.substr(last);
    te_parser parser;
    require(parser.compile(resolved), "Invalid expression: " + expression);
    double value = parser.evaluate();
    require(std::isfinite(value), "Non-finite expression: " + expression);
    return value;
}
Config readConfig(const fs::path& path, int step) {
    Config c; c.source = fs::absolute(path); c.step = step; require(step >= 0, "step must be nonnegative");
    c.variables = {{"pi",std::acos(-1.)},{"e",std::exp(1.)},{"mu0",4e-7*std::acos(-1.)}};
    c.yaml = YAML::LoadFile(c.source.string());
    auto m = c.yaml["model_3d"];
    require(m && m.IsMap() && m["schema_version"].as<int>(0) == 1, "model_3d.schema_version: 1 is required");
    // Resolve only variables needed by the 3D section: unrelated 2D runtime variables remain untouched.
    auto vars = c.yaml["variables"];
    std::set<std::string> resolving;
    std::function<void(const std::string&)> resolve = [&](const std::string& name) {
        if (name == "step" || c.variables.count(name)) return;
        require(name != "A" && name != "N" && name != "H" && name != "dx" && name != "dy" && name != "dr" && name != "dtheta", "2D-only reserved variable $" + name);
        require(vars && vars[name] && resolving.insert(name).second, "Undefined or cyclic variable $" + name);
        std::string value = vars[name].as<std::string>(); std::regex re(R"(\$([A-Za-z_][A-Za-z_0-9]*))");
        for (auto i = std::sregex_iterator(value.begin(), value.end(), re); i != std::sregex_iterator(); ++i) resolve((*i)[1]);
        c.variables[name] = c.number(vars[name]); resolving.erase(name);
    };
    auto eval = [&](const YAML::Node& value) {
        require(value && value.IsScalar(), "Missing numeric setting");
        auto text = value.as<std::string>(); std::regex re(R"(\$([A-Za-z_][A-Za-z_0-9]*))");
        for (auto i = std::sregex_iterator(text.begin(), text.end(), re); i != std::sregex_iterator(); ++i) resolve((*i)[1]);
        return c.number(value);
    };
    auto file = [&](const YAML::Node& n) { require(n && n.IsScalar(), "Missing input path"); return (c.source.parent_path() / fs::u8path(n.as<std::string>())).lexically_normal(); };
    c.voxels = file(m["geometry"]["voxels"]); c.manifest = file(m["geometry"]["manifest"]);
    if (m["max_cells"]) { auto n = eval(m["max_cells"]); require(n >= 1 && n <= 20000000 && n == std::floor(n), "max_cells must be an integer in [1,20000000]"); c.maxCells = size_t(n); }
    if (c.yaml["omp"] && c.yaml["omp"]["threads"]) { auto n = eval(c.yaml["omp"]["threads"]); require(n >= 0 && n <= 1024 && n == std::floor(n), "Invalid omp.threads"); c.threads = int(n); }
    auto addMaterials = [&](const YAML::Node& materials) {
        if (!materials) return;
        require(materials.IsMap(), "materials must be a map");
        for (const auto& p : materials) {
            auto name = p.first.as<std::string>(); require(!c.materials.count(name), "Duplicate common/3D material: " + name);
            auto v = YAML::Clone(p.second); require(v.IsMap(), "Material must be a map");
            if (v["preset"]) { auto preset = c.yaml["material_presets"][v["preset"].as<std::string>()]; require(preset && preset.IsMap(), "Unknown material preset"); v = merge(preset, v); }
            c.materials.emplace(name, v);
        }
    };
    addMaterials(c.yaml["materials"]); addMaterials(m["materials"]);
    require(m["regions"] && m["regions"].IsMap(), "model_3d.regions is required");
    std::set<uint32_t> assigned;
    for (auto p : m["regions"]) {
        Region r; r.material = p.second["material"].as<std::string>(); require(c.materials.count(r.material), "Unknown material " + r.material);
        require(p.second["solid_ids"].IsSequence(), "solid_ids must be a list");
        for (auto id : p.second["solid_ids"]) { auto n = id.as<uint32_t>(); require(n > 0 && assigned.insert(n).second, "Solid IDs must be positive and assigned once"); r.solids.insert(n); }
        require(!r.solids.empty(), "Empty region"); c.regions.emplace(p.first.as<std::string>(), r);
    }
    require(m["coils"] && m["coils"].IsSequence() && m["coils"].size() > 0, "At least one coil is required");
    std::set<std::string> names, usedRegions;
    for (auto n : m["coils"]) {
        Coil coil; coil.name = n["name"].as<std::string>();
        require(std::regex_match(coil.name, std::regex("[A-Za-z0-9_-]+")) && names.insert(coil.name).second, "Invalid or duplicate coil name");
        coil.model = n["model"].as<std::string>(); require(coil.model == "stranded_uniform" || coil.model == "bulk_dc", "Unknown coil model");
        coil.current = eval(n["current"]);
        require(n["regions"] && n["regions"].IsSequence(), "Coil regions must be a list");
        for (auto rn : n["regions"]) {
            auto name = rn.as<std::string>(); require(c.regions.count(name) && usedRegions.insert(name).second, "Unknown, duplicate or overlapping coil region: " + name); coil.regions.push_back(name);
            if (coil.model == "bulk_dc") {
                auto mat = c.materials.at(c.regions.at(name).material);
                double sigma = eval(mat["conductivity"]); require(sigma > 0, "Bulk conductivity must be positive"); mat["conductivity"] = sigma;
            }
        }
        require(!coil.regions.empty(), "Coil has no regions");
        if (coil.model == "bulk_dc") require(!n["turns"], "bulk_dc does not accept turns");
        else { coil.turns = eval(n["turns"]); require(coil.turns >= 1 && coil.turns == std::floor(coil.turns), "turns must be a positive integer"); }
        require(std::isfinite(coil.current * coil.turns), "Current times turns overflow");
        if (n["uniformity_tolerance"]) coil.uniformityTolerance = eval(n["uniformity_tolerance"]);
        require(coil.uniformityTolerance > 0 && coil.uniformityTolerance <= 1, "uniformity_tolerance must be in (0,1]");
        require(bool(n["terminals"]) != bool(n["cut"]), "Specify terminals OR a virtual cut");
        if (n["cut"]) coil.cut = file(n["cut"]["faces"]);
        else { coil.inlet = file(n["terminals"]["inlet"]["faces"]); coil.outlet = file(n["terminals"]["outlet"]["faces"]); }
        c.coils.push_back(coil);
    }
    return c;
}
static Domain topology(const Config& config, const Grid& g, const Coil& coil) {
    std::map<uint32_t, double> sigma;
    for (auto name : coil.regions) {
        const auto& r = config.regions.at(name);
        double s = coil.model == "bulk_dc" ? config.number(config.materials.at(r.material)["conductivity"]) : 1;
        for (auto id : r.solids) sigma[id] = s;
    }
    Domain d; std::vector<int> local(g.count(), -1); std::set<uint32_t> found;
    for (size_t i = 0; i < g.count(); ++i) if (sigma.count(g.solid[i])) { local[i] = int(d.cells.size()); d.cells.push_back(i); d.sigma.push_back(sigma.at(g.solid[i])); found.insert(g.solid[i]); }
    require(found.size() == sigma.size(), "A selected solid is absent from the voxel grid");
    require(!d.cells.empty(), "Empty coil");
    for (int axis = 0; axis < 3; ++axis) {
        auto n = g.size; ++n[axis];
        for (int z = 0; z < n[2]; ++z) for (int y = 0; y < n[1]; ++y) for (int x = 0; x < n[0]; ++x) {
            Index p{x,y,z}, lo = p; --lo[axis]; int a = -1, b = -1;
            if (p[axis] > 0) a = local[g.index(lo[0],lo[1],lo[2])];
            if (p[axis] < g.size[axis]) b = local[g.index(x,y,z)];
            if (a < 0 && b < 0) continue;
            Face f; f.key = {axis,p}; f.a = a; f.b = b; f.area = g.spacing[(axis+1)%3] * g.spacing[(axis+2)%3];
            f.distance = g.spacing[axis] * ((a >= 0 && b >= 0) ? 1 : .5);
            double s = a < 0 ? d.sigma[b] : b < 0 ? d.sigma[a] : 2 / (1/d.sigma[a] + 1/d.sigma[b]);
            f.conductance = s * f.area / f.distance; require(std::isfinite(f.conductance) && f.conductance > 0, "Invalid face conductance"); d.faces.push_back(f);
        }
    }
    return d;
}
Domain makeDomain(const Config& config, const Grid& g, const Coil& coil) {
    Domain d = topology(config,g,coil); d.closed = !coil.cut.empty();
    std::map<FaceKey,size_t> lookup; for (size_t i=0; i<d.faces.size(); ++i) lookup[d.faces[i].key]=i;
    auto apply = [&](const fs::path& path, int terminal) {
        for (auto selected : readFaces(path,g)) {
            require(lookup.count(selected.key), "Selected face is not on this coil"); auto& f = d.faces[lookup.at(selected.key)];
            require(!f.terminal && !f.cutSign, "Overlapping terminal/cut faces");
            if (terminal) { require((f.a < 0) != (f.b < 0), "Terminal must be an exterior coil face"); require(selected.sign == (f.a < 0 ? -1 : 1), "Terminal face sign must be outward"); f.terminal=terminal; }
            else { require(f.a >= 0 && f.b >= 0, "Cut must be an interior coil face"); f.cutSign=selected.sign; }
            d.selected.push_back(selected);
        }
    };
    if (d.closed) apply(coil.cut,0); else { apply(coil.inlet,1); apply(coil.outlet,-1); }
    // Removing a virtual cut must leave a connected path: no isolated pieces or accidental separating cuts.
    std::vector<std::vector<int>> adj(d.cells.size());
    for (const auto& f:d.faces) if(f.a>=0 && f.b>=0 && !f.cutSign) {adj[f.a].push_back(f.b);adj[f.b].push_back(f.a);}
    std::vector<bool> seen(d.cells.size()); std::queue<int> todo; todo.push(0);seen[0]=true;size_t count=0;
    while(!todo.empty()){int a=todo.front();todo.pop();++count;for(int b:adj[a])if(!seen[b]){seen[b]=true;todo.push(b);}}
    require(count==d.cells.size(), "Disconnected coil or separating/invalid virtual cut");
    return d;
}
std::vector<SelectedFace> selectFaces(const Config& c,const Grid& g,const Coil& coil,const Vec& point,const Vec& normal,const Vec& lower,const Vec& upper,double tolerance,bool cut){
    double norm=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);require(std::isfinite(norm)&&norm>0&&tolerance>=0,"Invalid plane");
    for(int a=0;a<3;++a)require(std::isfinite(point[a])&&std::isfinite(lower[a])&&std::isfinite(upper[a])&&lower[a]<=upper[a],"Invalid plane bounds");
    auto d=topology(c,g,coil);std::vector<SelectedFace> result;
    for(const auto& f:d.faces){
        bool interior=f.a>=0&&f.b>=0;if(interior!=cut)continue;
        Vec center;double distance=0;bool inside=true;
        for(int a=0;a<3;++a){center[a]=g.origin[a]+(f.key.index[a]+(a==f.key.axis?0:.5))*g.spacing[a];distance+=(center[a]-point[a])*normal[a]/norm;inside=inside&&center[a]>=lower[a]&&center[a]<=upper[a];}
        int sign=cut?(normal[f.key.axis]>0?1:-1):(f.a<0?-1:1);
        if(inside&&std::abs(distance)<=tolerance&&sign*normal[f.key.axis]/norm>1e-8)result.push_back({f.key,sign});
    }
    require(!result.empty(),"Plane selected no faces; check bounds, normal and tolerance");return result;
}
} // namespace
