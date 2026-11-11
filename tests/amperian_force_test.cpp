#include "MagneticFieldAnalyzer.h"
#include "AmperianForce.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
constexpr double pi = 3.14159265358979323846, mu0 = 4e-7*pi;
using Row = std::vector<double>;
using Loads = std::map<std::string,Row>;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(double actual, double expected, double tol, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tol) {
        std::cerr << message << ": actual=" << actual << ", expected=" << expected << '\n';
        throw std::runtime_error(message);
    }
}
Loads readLoads(const fs::path& file) {
    std::ifstream input(file); std::string line; Loads loads;
    while (std::getline(input,line)) {
        if (line.empty() || line[0]=='#' || line.rfind("Material,",0)==0) continue;
        std::istringstream stream(line); std::string name,value;
        std::getline(stream,name,','); Row row;
        while (std::getline(stream,value,',')) row.push_back(std::stod(value));
        loads[name]=row;
    }
    return loads;
}
Loads evaluate(MagneticFieldAnalyzer& a, const fs::path& dir) {
    const Eigen::MatrixXd before = a.getAz();
    a.calculateForceDistributedAmperian(); // Exercise the public default step=-1.
    check((a.getAz()-before).norm()==0,"Force calculation changed the solved potential");
    const auto file=dir/"forces.csv";
    a.exportForcesToCSV(file.string());
    return readLoads(file);
}

void cartesian(const fs::path& root) {
    Row joined,splitA,splitB;
    Eigen::MatrixXd joinedAz;
    for (const std::string name : {"coil","pm","joined","split","marker"}) {
        const auto dir=root/name; fs::create_directories(dir);
        cv::Mat im(64,64,CV_8UC3,cv::Scalar(255,255,255));
        const bool uniform=name=="coil" || name=="pm";
        for (int y=0;y<64;++y) for(int x=0;x<64;++x) {
            if (uniform && x>=26 && x<38 && y>=26 && y<38) im.at<cv::Vec3b>(63-y,x)={0,0,255};
            if (!uniform) {
                if (x>=38&&x<46&&y>=24&&y<40) im.at<cv::Vec3b>(63-y,x)={0,0,255};
                if (x>=24&&x<26&&y>=20&&y<44) {
                    const unsigned char c=(name=="split" && x==25)?64:128;
                    im.at<cv::Vec3b>(63-y,x)={c,c,c};
                }
                if (name=="marker"&&x>=28&&x<34&&y>=24&&y<40) im.at<cv::Vec3b>(63-y,x)={64,64,64};
            }
        }
        cv::imwrite((dir/"input.png").string(),im);
        std::ofstream yaml(dir/"input.yaml");
        yaml << "coordinate_system: cartesian\nmesh: {dx: .001, dy: .001}\nboundary_conditions:\n"
             << "  left: {type: dirichlet, value: 0}\n  right: {type: dirichlet, value: " << (uniform?-.0063:0) << "}\n"
             << "  bottom: {type: " << (uniform?"neumann":"dirichlet") << ", value: 0}\n"
             << "  top: {type: " << (uniform?"neumann":"dirichlet") << ", value: 0}\nmaterials:\n"
             << "  air: {rgb: [255,255,255], mu_r: 1, jz: 0}\n";
        if (name=="pm") yaml << "  pm: {rgb: [255,0,0], mu_r: 1, calc_force: true, magnetization: {Br: 1, pattern: parallel, angle: 0}}\n";
        else yaml << "  coil: {rgb: [255,0,0], mu_r: 1, jz: 1e6, calc_force: true}\n";
        if (!uniform) yaml << "  body: {rgb: [128,128,128], mu_r: 1000, calc_force: true}\n";
        if (name=="split" || name=="marker") yaml << "  part: {rgb: [64,64,64], mu_r: " << (name=="split"?1000:1) << ", calc_force: true}\n";
        yaml.close();
        MagneticFieldAnalyzer a((dir/"input.yaml").string(),(dir/"input.png").string());
        a.solve(); const auto loads=evaluate(a,dir);
        // CSV numeric row indices: RGB(0..2), Fx(3), Fy(4), torque-origin(7), centre(8).
        if (name=="coil") {
            near(loads.at("coil")[3],-14.4,1e-8,"Free-current force J x B");
            near(loads.at("coil")[8],0,1e-10,"Uniform-field centred coil torque");
        }
        if (name=="pm") {
            // Independent interaction-energy derivative: U=-m dot B,
            // -dU/dangle at angle=0 is m*By. No displaced solve is used.
            near(loads.at("pm")[8],144e-6*.1/mu0,1e-8,"PM torque vs interaction energy");
            near(loads.at("pm")[3],0,1e-8,"PM translational self force");
        }
        if (name=="joined") { joined=loads.at("body"); joinedAz=a.getAz(); }
        if (name=="split") {
            splitA=loads.at("body"); splitB=loads.at("part");
            near((a.getAz()-joinedAz).norm(),0,1e-14,"Material recolouring changed field");
            for (const int k : {3,4,7,8}) near(splitA[k]+splitB[k],joined[k],1e-12,"Overlapping current ownership");
        }
        if (name=="marker") for (const int k : {3,4,7,8})
            near(loads.at("part")[k],0,0,"Inert marker ghost force");
    }
}

void transient(const fs::path& root) {
    const auto dir=root/"transient"; fs::create_directories(dir);
    std::ifstream original(root/"coil/input.yaml");
    std::string yaml((std::istreambuf_iterator<char>(original)),{});
    const auto position=yaml.find("jz: 1e6");
    check(position!=std::string::npos,"Missing transient fixture current");
    yaml.replace(position,7,"jz: [0, 1e6, -1e6]");
    yaml+="\ntransient: {enabled: true, total_steps: 3, enable_sliding: false, parallel_chunks: 1, export_fields: [Forces]}\nexport: {format: csv, async: false}\n";
    {std::ofstream out(dir/"input.yaml");out<<yaml;}
    MagneticFieldAnalyzer a((dir/"input.yaml").string(),(root/"coil/input.png").string());
    a.performTransientAnalysis((dir/"out").string());
    for(int step=0;step<3;++step) {
        const auto loads=readLoads(dir/"out/Forces"/("step_000"+std::to_string(step+1)+".csv"));
        const double expected=step==0?0:step==1?-14.4:14.4;
        // The transient path uses its iterative warm start; allow its field
        // residual (observed ~4e-7 N/m), rather than the static direct-solve floor.
        near(loads.at("coil")[3],expected,1e-5,"Transient current/field cache mismatch");
    }
}

struct PolarResult { double torque, arkkio, energy, referenceError, fx, fy; Eigen::MatrixXd az; };
PolarResult polar(const fs::path& root, int refinement, bool half, bool vertical, double offset=0, bool nonlinear=false) {
    const int nr=39*refinement+1, full=64*refinement, nt=half?full/2:full;
    const double dr=.001/refinement, dt=2*pi/full, rs=.01;
    const auto dir=root/("polar_"+std::to_string(refinement)+(half?"half":"full")+(vertical?"v":"h")+std::to_string(offset)+(nonlinear?"nl":""));
    fs::create_directories(dir);
    cv::Mat normalized(nt,nr,CV_8UC3,cv::Scalar(255,255,255));
    for(int j=0;j<nt;++j)for(int i=0;i<nr;++i)
        if(i>=4*refinement && i<refinement*(17+3*std::cos(2*j*dt-.7)))
            normalized.at<cv::Vec3b>(j,i)={128,128,128};
    cv::Mat im;
    if (vertical) cv::transpose(normalized,im); else im=normalized;
    cv::flip(im,im,0); cv::imwrite((dir/"input.png").string(),im);
    const auto profile=dir/"outer.txt";
    { std::ofstream out(profile); out << std::setprecision(17);
      for(int j=0;j<nt;++j) out << .001*std::cos(j*dt-.4) << '\n'; }
    std::ofstream yaml(dir/"input.yaml");
    yaml << std::setprecision(17) << "coordinate_system: polar\npolar_domain: {r_start: .01, r_end: .049, r_orientation: "
         << (vertical?"vertical":"horizontal") << ", theta_range: " << (half?"pi":"2*pi") << ", theta_offset: " << offset << "}\n"
         << "polar_boundary_conditions:\n  inner: {type: dirichlet, value: 0}\n  outer: {type: dirichlet, value: 0, value_profile: '"
         << profile.generic_string() << "'}\n  theta_min: {type: periodic, value: " << (half?-1:1)
         << "}\n  theta_max: {type: periodic, value: " << (half?-1:1) << "}\nmaterials:\n"
         << "  air: {rgb: [255,255,255], mu_r: 1}\n  rotor: {rgb: [128,128,128], "
         << (nonlinear?"B-H: '$H/(40+0.52*$H)+4*pi*1e-7*$H'":"mu_r: 1000") << ", calc_force: true}\n";
    if(nonlinear) yaml << "nonlinear_solver: {enabled: true, solver_type: newton-krylov, jacobian: tangent, tolerance: 1e-7, max_iterations: 200}\n";
    yaml.close();
    MagneticFieldAnalyzer a((dir/"input.yaml").string(),(dir/"input.png").string());
    a.solve(); const auto loads=evaluate(a,dir);
    Eigen::MatrixXd az=a.getAz(); if(vertical) az.transposeInPlace();
    auto value=[&](int j,int i) {return az((j+nt)%nt,i)*(half&&(j<0||j>=nt)?-1:1);};
    double ark=0,wp=0,wm=0;
    const int first=24*refinement,last=30*refinement;
    // Separate reference discretizations in verified source-free air:
    // Arkkio averages nodal circle tractions with trapezoidal radial weights.
    // Virtual work shears theta by epsilon*g(r) and evaluates face energy at
    // fixed A. Only the air annulus deforms; g'=-1/width. No field re-solve.
    const double width=(last-first)*dr, eps=1e-5, gp=-1/width;
    for(int j=0;j<nt;++j) for(int i=first;i<=last;++i) {
        const double r=rs+i*dr;
        const double br=(value(j+1,i)-value(j-1,i))/(2*dt*r);
        const double bt=-(value(j,i+1)-value(j,i-1))/(2*dr);
        ark+=(i==first||i==last?.5:1)*r*r*br*bt*dr*dt/(mu0*width);
        if(i==last)continue;
        const double rm=r+dr/2, ar=(value(j,i+1)-value(j,i))/dr;
        const double at=(value(j+1,i)+value(j+1,i+1)-value(j-1,i)-value(j-1,i+1))/(4*dt);
        wp+=.5/mu0*rm*dr*dt*(std::pow(ar-eps*gp*at,2)+std::pow(at/rm,2));
        wm+=.5/mu0*rm*dr*dt*(std::pow(ar+eps*gp*at,2)+std::pow(at/rm,2));
    }
    const double energy=-(wp-wm)/(2*eps), torque=loads.at("rotor")[7];
    const double error=std::abs(torque/ark-1);
    check(error<3e-4,"Amperian/Arkkio discrepancy on polar rotor");
    near(torque,energy,std::abs(torque)*3e-4,"Amperian/air virtual-work discrepancy");
    std::cout << std::setprecision(12) << "REFERENCE " << dir.filename().string() << " ampere=" << torque
              << " arkkio=" << ark << " energy=" << energy << " relative=" << error << '\n';
    return {torque,ark,energy,error,loads.at("rotor")[3],loads.at("rotor")[4],az};
}

void kernelChecks() {
    using namespace openmagfdm::detail;
    const Eigen::MatrixXd zero=Eigen::MatrixXd::Zero(8,8);
    auto dn=zero; dn(0,2)=1;
    bool rejected=false;
    try { amperianCurrent(zero,dn,zero,zero,zero,AmperianGrid{}); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"Truncated magnetized body accepted");
    auto invalid=zero; invalid(3,3)=std::numeric_limits<double>::quiet_NaN();
    rejected=false;
    try { amperianCurrent(invalid,zero,zero,zero,zero,AmperianGrid{}); }
    catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"Nonfinite potential accepted in Release build");
    AmperianGrid g; g.h1=.002; g.h2=.003;
    Eigen::MatrixXd a=zero,mx=zero,j=zero;
    const Eigen::MatrixXd bx=Eigen::MatrixXd::Constant(8,8,.07);
    const Eigen::MatrixXd by=Eigen::MatrixXd::Constant(8,8,.11);
    for(int y=0;y<8;++y)for(int x=0;x<8;++x) {
        a(y,x)=.07*y*g.h2-.11*x*g.h1;
        if(x>=2&&x<6&&y>=2&&y<6){mx(y,x)=1/mu0;j(y,x)=2e6;}
    }
    const auto loads=integrateAmperianCurrent(amperianCurrent(a,zero,mx,zero,j,g),bx,by,g);
    const double area=16*g.h1*g.h2;
    near(loads.fx,-2e6*.11*area,1e-10,"Anisotropic free-current force");
    near(loads.fy,2e6*.07*area,1e-10,"Anisotropic free-current force y");
    near(loads.torqueCenter,.11*area/mu0,1e-10,"Anisotropic combined PM/current torque");
}

int main(int argc,char**argv) {
    try {
#ifdef _OPENMP
        omp_set_num_threads(2);
#endif
        const auto root=fs::absolute(argc>1?argv[1]:"amperian_force_fixture");
        fs::create_directories(root);
        cartesian(root); transient(root); kernelChecks();
        const auto full=polar(root,1,false,false), half=polar(root,1,true,false);
        near(half.torque*2,full.torque,1e-10,"Anti-periodic sector torque mismatch");
        near((full.az.topRows(half.az.rows())-half.az).norm(),0,1e-13,"Full/half field mismatch");
        const auto vertical=polar(root,1,false,true);
        near(vertical.torque,full.torque,1e-10,"Polar storage orientation mismatch");
        const auto rotated=polar(root,1,true,false,.37);
        near(rotated.torque,half.torque,1e-10,"Coordinate-frame torque mismatch");
        near(rotated.fx,half.fx*std::cos(.37)-half.fy*std::sin(.37),1e-9,"Global force x basis mismatch");
        near(rotated.fy,half.fx*std::sin(.37)+half.fy*std::cos(.37),1e-9,"Global force y basis mismatch");
        const auto fine=polar(root,2,false,false);
        check(fine.referenceError<full.referenceError*.35,"Reference discrepancy did not converge on refinement");
        polar(root,1,false,false,0,true);
        std::cout << "Amperian force physics and integration tests passed\n";
        return 0;
    } catch(const std::exception& e) {std::cerr << e.what() << '\n';return 1;}
}
