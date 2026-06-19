// v1.6 — C++ in-solver multi-patch domain-decomposition WALL-TIME BENCHMARK (optimized Schwarz).
//
// Goal: quantify the speedup of an in-process C++ DD vs the monolithic solve (Python-overhead-free).
// Reports, every outer iteration: aggregate patch DOF, outer-iter count, wall time, flux error.
// Reuses MagneticFieldAnalyzer as persistent in-process patch solvers (no subprocess/TIFF overhead).
//
// Stage C1 (this build): Nr x Nth FULL-RES overlapping patches + symmetric Robin transmission +
// multiplicative Schwarz. Coarse space + r/theta coarsening (cf 2/4/8) + mortar: added next, benchmarked.
//
// Usage: dd_bench <base_config.yaml> <image.png> [Nr Nth p ov max_outer tol]
#include "MagneticFieldAnalyzer.h"
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>

using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b){ return std::chrono::duration<double>(b-a).count(); }
static std::vector<std::pair<int,int>> split(int n,int k){
    std::vector<std::pair<int,int>> o;
    for(int s=0;s<k;++s) o.push_back({(int)std::llround((double)s*n/k),(int)std::llround((double)(s+1)*n/k)});
    return o;
}
static int rgbkey(int r,int g,int b){ return (r<<16)|(g<<8)|b; }

int main(int argc,char**argv){
    if(argc<3){ std::cerr<<"usage: dd_bench <cfg.yaml> <img.png> [Nr Nth p ov max_outer tol]\n"; return 1; }
    std::string base_cfg=argv[1], img_path=argv[2];
    int   Nr  =argc>3?std::atoi(argv[3]):1;
    int   Nth =argc>4?std::atoi(argv[4]):4;
    double p  =argc>5?std::atof(argv[5]):0.3;
    int   ov  =argc>6?std::atoi(argv[6]):4;
    int   maxo=argc>7?std::atoi(argv[7]):20;
    double tol=argc>8?std::atof(argv[8]):2e-3;
    const double PR=40.0;   // radial Robin p ~ 40x theta (PoC tuning)

    // ---------- monolithic reference (in-process) ----------
    std::cerr<<"=== monolithic reference ===\n";
    auto t0=Clock::now();
    MagneticFieldAnalyzer full(base_cfg,img_path);
    full.solve();
    double t_full=secs(t0,Clock::now());
    const int NTH=full.getNtheta(), NR=full.getNr();
    const double DR=full.getDr(), DTH=full.getDtheta(), RS=full.getRStart();
    Eigen::MatrixXd Gref=full.getAz();
    double denom=std::max(Gref.cwiseAbs().maxCoeff(),1e-30);
    struct Ph{const char*n;int a,b;} ph[3]={
        {"A",rgbkey(208,83,58),rgbkey(112,48,37)},
        {"B",rgbkey(58,208,82),rgbkey(37,112,48)},
        {"C",rgbkey(58,108,207),rgbkey(37,63,112)}};
    double fref[3]; for(int k=0;k<3;k++) fref[k]=full.fluxLinkageMaterialPair(ph[k].a,ph[k].b);
    std::cerr<<"monolithic DOF="<<NTH*NR<<" wall="<<t_full<<"s  Phi=["<<fref[0]<<","<<fref[1]<<","<<fref[2]<<"]\n";

    // ---------- load image (RGB) + base config ----------
    cv::Mat bgr=cv::imread(img_path,cv::IMREAD_COLOR), img; cv::cvtColor(bgr,img,cv::COLOR_BGR2RGB);
    YAML::Node base=YAML::LoadFile(base_cfg);
    auto rcores=split(NR,Nr), tcores=split(NTH,Nth);
    std::cerr<<"=== DD Nr="<<Nr<<" Nth="<<Nth<<" ov="<<ov<<" p="<<p<<" ("<<Nr*Nth<<" patches) ===\n";

    struct Patch{ int er0,er1,et0,et1,cr0,cr1,ct0,ct1; std::vector<int> tr; std::unique_ptr<MagneticFieldAnalyzer> an; };
    std::vector<Patch> P;
    system("mkdir -p dd_tmp");
    long agg_dof=0; int pid=0;
    for(auto[ra0,ra1]:rcores){
        int er0=std::max(0,ra0-ov), er1=std::min(NR,ra1+ov);
        for(auto[tb0,tb1]:tcores){
            Patch q; q.er0=er0;q.er1=er1;q.cr0=ra0;q.cr1=ra1;q.ct0=tb0;q.ct1=tb1;
            if(Nth==1){q.et0=0;q.et1=NTH;} else {q.et0=tb0-ov;q.et1=tb1+ov;}
            int nthx=q.et1-q.et0, nrx=er1-er0;
            for(int m=0;m<nthx;m++) q.tr.push_back((((q.et0+m)%NTH)+NTH)%NTH);
            agg_dof += (long)nthx*nrx;
            cv::Mat crop(nthx,nrx,CV_8UC3);
            for(int m=0;m<nthx;m++){ int g=q.tr[nthx-1-m]; int ir=((NTH-1-g)%NTH+NTH)%NTH;
                for(int c=0;c<nrx;c++) crop.at<cv::Vec3b>(m,c)=img.at<cv::Vec3b>(ir,er0+c); }
            cv::Mat cb; cv::cvtColor(crop,cb,cv::COLOR_RGB2BGR);
            std::string png="dd_tmp/p"+std::to_string(pid)+".png"; cv::imwrite(png,cb);
            YAML::Node cfg=YAML::Clone(base);
            cfg["polar_domain"]["r_start"]=RS+er0*DR;
            cfg["polar_domain"]["r_end"]  =RS+(er1-1)*DR;
            cfg["polar_domain"]["theta_range"]=nthx*DTH;
            cfg["polar_domain"]["theta_offset"]=q.et0*DTH;
            if(cfg["transient"]) cfg["transient"]["enabled"]=false;
            if(cfg["nonlinear_solver"]) cfg["nonlinear_solver"]["verbose"]=false;
            YAML::Node bc(YAML::NodeType::Map);
            auto edge=[&](const char*e,const char*t,double a,double b,double g){
                YAML::Node n(YAML::NodeType::Map); n["type"]=std::string(t);
                if(std::string(t)=="robin"){n["alpha"]=a;n["beta"]=b;n["gamma"]=g;}
                else if(std::string(t)=="dirichlet")n["value"]=0;
                else if(std::string(t)=="periodic")n["value"]=1;
                bc[e]=n; };
            edge("inner", er0==0?"dirichlet":"robin", p*PR,1.0,0.0);
            edge("outer", er1-1==NR-1?"dirichlet":"robin", p*PR,1.0,0.0);
            if(Nth==1){edge("theta_min","periodic",0,0,0);edge("theta_max","periodic",0,0,0);}
            else{edge("theta_min","robin",p,1.0,0.0);edge("theta_max","robin",p,1.0,0.0);}
            cfg["polar_boundary_conditions"]=bc;
            std::string yp="dd_tmp/p"+std::to_string(pid)+".yaml"; std::ofstream(yp)<<cfg;
            q.an=std::make_unique<MagneticFieldAnalyzer>(yp,png);
            q.an->setDDWarmStart(true);   // NK warm-starts from the orchestrator-set Az each sweep
            P.push_back(std::move(q)); pid++;
        }
    }
    std::cerr<<"built "<<P.size()<<" patches, aggregate DOF="<<agg_dof
             <<" ("<<100.0*agg_dof/(NTH*NR)<<"% of monolithic, includes overlap)\n";

    Eigen::MatrixXd G=Eigen::MatrixXd::Zero(NTH,NR);
    auto Gat=[&](int gtr,int c)->double{ return G(((gtr%NTH)+NTH)%NTH,c); };
    auto tloop=Clock::now(); int it=0; double err=1.0;
    for(it=1; it<=maxo; ++it){
        for(auto&q:P){
            int nthx=q.et1-q.et0, nrx=q.er1-q.er0;
            std::vector<double> gin(nthx,0),gout(nthx,0),gtmin(nrx,0),gtmax(nrx,0);
            for(int m=0;m<nthx;m++){ int gt=q.tr[m];
                if(q.er0!=0)      gin[m] =p*PR*Gat(gt,q.er0)   -(Gat(gt,q.er0)-Gat(gt,q.er0-1))/DR;
                if(q.er1-1!=NR-1) gout[m]=p*PR*Gat(gt,q.er1-1) +(Gat(gt,q.er1)-Gat(gt,q.er1-1))/DR; }
            if(Nth>1){ int g0=q.tr.front(),g1=q.tr.back(); int g0m=((g0-1)%NTH+NTH)%NTH,g1p=(g1+1)%NTH;
                for(int c=0;c<nrx;c++){ gtmin[c]=p*Gat(g0,q.er0+c)-(Gat(g0,q.er0+c)-Gat(g0m,q.er0+c))/DTH;
                    gtmax[c]=p*Gat(g1,q.er0+c)+(Gat(g1p,q.er0+c)-Gat(g1,q.er0+c))/DTH; } }
            if(q.er0!=0)      q.an->setBoundaryProfile("inner",gin);
            if(q.er1-1!=NR-1) q.an->setBoundaryProfile("outer",gout);
            if(Nth>1){ q.an->setBoundaryProfile("theta_min",gtmin); q.an->setBoundaryProfile("theta_max",gtmax); }
            Eigen::MatrixXd pAz(nthx,nrx);
            for(int m=0;m<nthx;m++)for(int c=0;c<nrx;c++) pAz(m,c)=Gat(q.tr[m],q.er0+c);
            q.an->setAz(pAz); q.an->solve();
            const Eigen::MatrixXd& sol=q.an->getAz();
            for(int m=0;m<nthx;m++){ int gug=q.et0+m; int gt=q.tr[m];
                bool core_t=(Nth==1)||(q.ct0<=gug&&gug<q.ct1);
                if(!core_t) continue;
                for(int c=0;c<nrx;c++){ int gc=q.er0+c;
                    if(q.cr0<=gc&&gc<q.cr1) G(((gt%NTH)+NTH)%NTH,gc)=sol(m,c); } }
        }
        err=(G-Gref).cwiseAbs().maxCoeff()/denom;
        // flux from G via the full analyzer's material map
        full.setAz(G);
        double fa=full.fluxLinkageMaterialPair(ph[0].a,ph[0].b);
        double ea=std::abs((fa-fref[0])/ (std::abs(fref[0])+1e-30));
        std::cerr<<"outer "<<it<<": err_vs_mono="<<err<<"  PhiA_err="<<ea
                 <<"  cum_wall="<<secs(tloop,Clock::now())<<"s\n";
        if(err<tol) break;
    }
    double t_loop=secs(tloop,Clock::now());
    full.setAz(G);
    std::cerr<<"=== DONE outer="<<it<<" err="<<err<<" ===\n";
    std::cerr<<"flux: ";
    for(int k=0;k<3;k++){ double f=full.fluxLinkageMaterialPair(ph[k].a,ph[k].b);
        std::cerr<<ph[k].n<<"="<<f<<"(err "<<std::abs((f-fref[k])/(std::abs(fref[k])+1e-30))*100<<"%) "; }
    std::cerr<<"\nWALL: monolithic="<<t_full<<"s  DD_loop="<<t_loop<<"s  speedup="<<t_full/t_loop<<"x"
             <<"  aggDOF="<<100.0*agg_dof/(NTH*NR)<<"%\n";
    return 0;
}
