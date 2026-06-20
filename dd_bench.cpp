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
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <array>
#include <cstdio>
#include <cstdlib>

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
    long mono_li=full.getTotalLinearIters(); int mono_ns=full.getNumLinearSolves();
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
    std::cerr<<"monolithic CG-iters="<<mono_li<<" over "<<mono_ns<<" linear solves (avg "
             <<(mono_ns?mono_li/mono_ns:0)<<" CG/solve = global conditioning)\n";

    // ---------- load image (RGB) + base config ----------
    cv::Mat bgr=cv::imread(img_path,cv::IMREAD_COLOR), img; cv::cvtColor(bgr,img,cv::COLOR_BGR2RGB);
    YAML::Node base=YAML::LoadFile(base_cfg);

    // ===================== WEDGE FIXED-POINT TEST (DD_WEDGE="t0:t1:r0:r1:cf") =====================
    // Stability check for coarsening the ACTIVE region as a theta-sector wedge: coarsen ONE wedge
    // (grid-theta [t0,t1) x r-cols [r0,r1) by cf), pin all 4 edges to the MONOLITHIC trace (Dirichlet
    // value_profile), init from the (downsampled) monolithic, solve once, compare to monolithic in the
    // interior. Reproduces (small err) -> the coarse active-wedge is self-consistent = STABLE. Drifts
    // away / NK fails -> the active region cannot be coarsened even as a conforming wedge.
    if(const char* we=getenv("DD_WEDGE")){
        int t0,t1,r0,r1,cf;
        if(sscanf(we,"%d:%d:%d:%d:%d",&t0,&t1,&r0,&r1,&cf)!=5){ std::cerr<<"DD_WEDGE=t0:t1:r0:r1:cf\n"; return 1; }
        int nthc=std::max(2,(t1-t0)/cf), nrc=std::max(2,(r1-r0)/cf);
        auto flip=[&](int gt){ return ((NTH-1-(gt%NTH))%NTH+NTH)%NTH; };
        cv::Mat crop(nthc,nrc,CV_8UC3);
        for(int r=0;r<nthc;r++){ int gt=t0+(nthc-1-r)*cf+cf/2; int ir=flip(gt);
            for(int cc=0;cc<nrc;cc++){ int gc=std::min(r0+cc*cf+cf/2,r1-1); crop.at<cv::Vec3b>(r,cc)=img.at<cv::Vec3b>(ir,gc); } }
        cv::Mat cb; cv::cvtColor(crop,cb,cv::COLOR_RGB2BGR);
        system("mkdir -p dd_tmp"); std::string png="dd_tmp/wedge.png"; cv::imwrite(png,cb);
        YAML::Node cfg=YAML::Clone(base);
        cfg["polar_domain"]["r_start"]=RS+r0*DR; cfg["polar_domain"]["r_end"]=RS+(r1-1)*DR;
        cfg["polar_domain"]["theta_range"]=(t1-t0)*DTH; cfg["polar_domain"]["theta_offset"]=t0*DTH;
        if(cfg["transient"]) cfg["transient"]["enabled"]=false;
        if(cfg["nonlinear_solver"]) cfg["nonlinear_solver"]["verbose"]=false;
        YAML::Node bc(YAML::NodeType::Map);
        for(const char* e:{"inner","outer","theta_min","theta_max"}){
            YAML::Node n(YAML::NodeType::Map); n["type"]="dirichlet"; n["value"]=0; bc[e]=n; }
        cfg["polar_boundary_conditions"]=bc;
        std::string yp="dd_tmp/wedge.yaml"; std::ofstream(yp)<<cfg;
        MagneticFieldAnalyzer w(yp,png);
        std::vector<double> pin(nthc),pout(nthc),ptmin(nrc),ptmax(nrc);
        for(int j=0;j<nthc;j++){ int gt=((t0+j*cf+cf/2)%NTH+NTH)%NTH;
            pin[j]=Gref(gt,r0); pout[j]=Gref(gt,std::min(r1-1,NR-1)); }
        for(int i=0;i<nrc;i++){ int gc=std::min(r0+i*cf+cf/2,r1-1);
            ptmin[i]=Gref((t0%NTH+NTH)%NTH,gc); ptmax[i]=Gref(((t1-1)%NTH+NTH)%NTH,gc); }
        w.setBoundaryProfile("inner",pin);     w.setBoundaryProfile("outer",pout);
        w.setBoundaryProfile("theta_min",ptmin); w.setBoundaryProfile("theta_max",ptmax);
        Eigen::MatrixXd w0(nthc,nrc);
        for(int j=0;j<nthc;j++)for(int i=0;i<nrc;i++){ int gt=((t0+j*cf+cf/2)%NTH+NTH)%NTH;
            int gc=std::min(r0+i*cf+cf/2,r1-1); w0(j,i)=Gref(gt,gc); }
        w.setAz(w0); w.setDDWarmStart(true);
        auto tw=Clock::now(); w.solve(); double twall=secs(tw,Clock::now());
        const Eigen::MatrixXd& sol=w.getAz();
        double num=0,den=0,mx=0;
        for(int j=0;j<nthc;j++)for(int i=0;i<nrc;i++){ int gt=((t0+j*cf+cf/2)%NTH+NTH)%NTH;
            int gc=std::min(r0+i*cf+cf/2,r1-1); double d=sol(j,i)-Gref(gt,gc);
            num+=d*d; den+=Gref(gt,gc)*Gref(gt,gc); mx=std::max(mx,std::abs(d)); }
        double dn=std::max(Gref.cwiseAbs().maxCoeff(),1e-30);
        std::cerr<<"=== WEDGE FIXED-POINT t["<<t0<<","<<t1<<") r["<<r0<<","<<r1<<") cf="<<cf
                 <<" grid "<<nthc<<"x"<<nrc<<" ===\n"
                 <<"L2rel(vs mono)="<<std::sqrt(num/std::max(den,1e-30))
                 <<"  Linf/||G||="<<mx/dn<<"  solve="<<twall<<"s\n";
        return 0;
    }

    // ===================== BANDED COARSENING MODE (radial bands, theta periodic) =====================
    // DD_BANDS="c0:c1:cf,c0:c1:cf,..." -> material-aligned radial bands; smooth bands solved on a cf x
    // downsampled own grid; radial Robin transmission with mortar (theta/r up/down-sample across the
    // resolution jump). Nth ignored (theta periodic per band). This is the DOF-reduction lever.
    const char* be = getenv("DD_BANDS");
    bool dd_yaml = base["domain_decomposition"] && base["domain_decomposition"]["enabled"]
                   && base["domain_decomposition"]["enabled"].as<bool>(false);
    if (be || dd_yaml) {
        // bands: {c0,c1,cf_r,cf_theta}. cf_theta=1 -> r-ONLY coarsening (keeps full theta -> preserves
        // theta-varying structure: magnets, slots -> stable for the active region). cf_theta=cf_r ->
        // 2D coarsening (theta-uniform regions: bore, yoke).
        std::vector<std::array<int,4>> bands;
        if (dd_yaml && base["domain_decomposition"]["bands"]) {
            for (auto bn : base["domain_decomposition"]["bands"])
                bands.push_back({bn[0].as<int>(), bn[1].as<int>(), bn[2].as<int>(),
                                 bn.size()>3 ? bn[3].as<int>() : bn[2].as<int>()});
            auto dd = base["domain_decomposition"];
            if (dd["robin_p"])   p   = dd["robin_p"].as<double>() / PR;   // radial Robin uses p*PR
            if (dd["overlap"])   ov  = dd["overlap"].as<int>();
            if (dd["max_outer"]) maxo= dd["max_outer"].as<int>();
            if (dd["tol"])       tol = dd["tol"].as<double>();
        } else {
            std::string s(be?be:""); size_t i=0; while(i<s.size()){ int c0,c1,cfr,cft=-1;
                int got=sscanf(s.c_str()+i,"%d:%d:%d:%d",&c0,&c1,&cfr,&cft); if(got<4)cft=cfr;
                bands.push_back({c0,c1,cfr,cft}); size_t n=s.find(',',i); if(n==std::string::npos)break; i=n+1; }
        }
        struct Band{ int c0,c1,cfr,cft,er0,er1,nth,nrb; std::unique_ptr<MagneticFieldAnalyzer> an; };
        std::vector<Band> B; system("mkdir -p dd_tmp"); long agg=0; int bid=0;
        for(auto&bd:bands){ Band b; b.c0=bd[0];b.c1=bd[1];b.cfr=bd[2];b.cft=bd[3];
            b.er0=std::max(0,bd[0]-ov*b.cfr); b.er1=std::min(NR,bd[1]+ov*b.cfr);   // overlap in fine cols
            b.nth=NTH/b.cft; b.nrb=std::max(2,(b.er1-b.er0)/b.cfr);
            agg += (long)b.nth*b.nrb;
            cv::Mat crop(NTH, b.er1-b.er0, CV_8UC3);   // full theta, ext cols (RGB)
            // NO theta pre-flip: the band keeps full theta; the solver's internal grid<->image flip
            // (grid g <-> image row NTH-1-g) already aligns band grid g with the monolithic grid g.
            for(int j=0;j<NTH;j++) for(int c=0;c<b.er1-b.er0;c++) crop.at<cv::Vec3b>(j,c)=img.at<cv::Vec3b>(j,b.er0+c);
            cv::Mat cs; cv::resize(crop, cs, cv::Size(b.nrb,b.nth), 0,0, cv::INTER_NEAREST);
            cv::Mat cb; cv::cvtColor(cs,cb,cv::COLOR_RGB2BGR);
            std::string png="dd_tmp/b"+std::to_string(bid)+".png"; cv::imwrite(png,cb);
            YAML::Node cfg=YAML::Clone(base);
            cfg["polar_domain"]["r_start"]=RS+b.er0*DR; cfg["polar_domain"]["r_end"]=RS+(b.er1-1)*DR;
            cfg["polar_domain"]["theta_range"]="2*pi"; cfg["polar_domain"]["theta_offset"]=0.0;
            if(cfg["transient"])cfg["transient"]["enabled"]=false;
            if(cfg["nonlinear_solver"])cfg["nonlinear_solver"]["verbose"]=false;
            YAML::Node bc(YAML::NodeType::Map);
            auto ed=[&](const char*e,const char*t,double a){ YAML::Node n(YAML::NodeType::Map); n["type"]=std::string(t);
                if(std::string(t)=="robin"){n["alpha"]=a;n["beta"]=1.0;n["gamma"]=0.0;} else if(std::string(t)=="dirichlet")n["value"]=0; else n["value"]=1; bc[e]=n; };
            ed("inner", b.er0==0?"dirichlet":"robin", p*PR);
            ed("outer", b.er1-1==NR-1?"dirichlet":"robin", p*PR);
            ed("theta_min","periodic",0); ed("theta_max","periodic",0);
            cfg["polar_boundary_conditions"]=bc;
            std::string yp="dd_tmp/b"+std::to_string(bid)+".yaml"; std::ofstream(yp)<<cfg;
            b.an=std::make_unique<MagneticFieldAnalyzer>(yp,png); b.an->setDDWarmStart(true);
            B.push_back(std::move(b)); bid++;
        }
        std::cerr<<"banded: "<<B.size()<<" bands, aggregate DOF="<<agg<<" ("<<100.0*agg/(NTH*NR)<<"% of monolithic)\n";
        Eigen::MatrixXd G=Eigen::MatrixXd::Zero(NTH,NR);
        // ---- 2-level coarse space (DD_COARSE="CR:CTH"; DD_CS_NONLIN=1 rebuilds A each iter) ----
        int CR=0,CTH=0; bool use_cs=false, cs_nonlin=(getenv("DD_CS_NONLIN")!=nullptr);
        Eigen::SparseMatrix<double> Pcs, Afull; Eigen::VectorXd bvec;
        Eigen::SparseLU<Eigen::SparseMatrix<double>> Aclu;
        if(const char* cs=getenv("DD_COARSE")){ sscanf(cs,"%d:%d",&CR,&CTH); use_cs=(CR>0&&CTH>0); }
        if(use_cs){
            int nrc=NR/CR, nthc=NTH/CTH; long Nc=(long)nrc*nthc;
            std::vector<Eigen::Triplet<double>> tp;
            for(int i=0;i<NR;i++){ double fic=(double)i/CR; int ic0=std::min((int)fic,nrc-1),ic1=std::min(ic0+1,nrc-1); double wr=fic-(int)fic;
                for(int j=0;j<NTH;j++){ double fjc=(double)j/CTH; int jc0=((int)fjc)%nthc,jc1=(jc0+1)%nthc; double wt=fjc-(int)fjc; int f=i*NTH+j;
                    tp.push_back({f,(int)(ic0*nthc+jc0),(1-wr)*(1-wt)}); tp.push_back({f,(int)(ic0*nthc+jc1),(1-wr)*wt});
                    tp.push_back({f,(int)(ic1*nthc+jc0),wr*(1-wt)});     tp.push_back({f,(int)(ic1*nthc+jc1),wr*wt}); } }
            Pcs.resize((long)NTH*NR,Nc); Pcs.setFromTriplets(tp.begin(),tp.end());
            full.setAz(G); full.buildPolarOperator(Afull,bvec);
            Eigen::SparseMatrix<double> Ac=(Eigen::SparseMatrix<double>(Pcs.transpose())*Afull*Pcs).pruned();
            Aclu.analyzePattern(Ac); Aclu.factorize(Ac);
            std::cerr<<"coarse space: CR="<<CR<<" CTH="<<CTH<<" coarseDOF="<<Nc<<(cs_nonlin?" (rebuilt/iter)":"")<<"\n";
        }
        auto tl=Clock::now(); int it=0; double err=1.0;
        for(it=1; it<=maxo; ++it){
            for(auto&b:B){
                // ---- mortar gamma: downsample fine G edge traces to band theta ----
                std::vector<double> gin(b.nth,0),gout(b.nth,0);
                for(int mb=0;mb<b.nth;mb++){ double ui=0,di=0,uo=0,doo=0; int n=b.cft;
                    for(int t=0;t<b.cft;t++){ int j=mb*b.cft+t;
                        if(b.er0!=0){ ui+=G(j,b.er0); di+=(G(j,b.er0)-G(j,b.er0-1))/DR; }
                        if(b.er1-1!=NR-1){ uo+=G(j,b.er1-1); doo+=(G(j,b.er1)-G(j,b.er1-1))/DR; } }
                    if(b.er0!=0)      gin[mb]=p*PR*(ui/n)-(di/n);
                    if(b.er1-1!=NR-1) gout[mb]=p*PR*(uo/n)+(doo/n); }
                if(b.er0!=0)      b.an->setBoundaryProfile("inner",gin);
                if(b.er1-1!=NR-1) b.an->setBoundaryProfile("outer",gout);
                // ---- warm start: sample G into band grid ----
                Eigen::MatrixXd pAz(b.nth,b.nrb);
                for(int mb=0;mb<b.nth;mb++)for(int kb=0;kb<b.nrb;kb++){
                    int j=mb*b.cft+b.cft/2; int c=b.er0+(int)std::llround((double)kb*(b.er1-1-b.er0)/(b.nrb-1));
                    pAz(mb,kb)=G(std::min(j,NTH-1),std::min(c,NR-1)); }
                b.an->setAz(pAz); b.an->solve();
                const Eigen::MatrixXd& sol=b.an->getAz();
                // ---- write CORE back to fine G (nearest-neighbor upsample) ----
                for(int j=0;j<NTH;j++){ int mb=j/b.cft; if(mb>=b.nth)mb=b.nth-1;
                    for(int c=b.c0;c<b.c1;c++){ int kb=(int)std::llround((double)(c-b.er0)*(b.nrb-1)/(b.er1-1-b.er0));
                        kb=std::max(0,std::min(kb,b.nrb-1)); G(j,c)=sol(mb,kb); } }
            }
            // ---- 2-level coarse correction: G += P * Ac^-1 * P^T (b - A*G) ----
            if(use_cs){
                if(cs_nonlin){ full.setAz(G); full.buildPolarOperator(Afull,bvec);
                    Eigen::SparseMatrix<double> Ac=(Eigen::SparseMatrix<double>(Pcs.transpose())*Afull*Pcs).pruned();
                    Aclu.factorize(Ac); }
                Eigen::VectorXd Gv((long)NTH*NR);
                for(int i=0;i<NR;i++)for(int j=0;j<NTH;j++) Gv[(long)i*NTH+j]=G(j,i);
                Eigen::VectorXd R=bvec-Afull*Gv;
                Eigen::VectorXd d=Pcs*Aclu.solve(Eigen::VectorXd(Pcs.transpose()*R));
                for(int i=0;i<NR;i++)for(int j=0;j<NTH;j++) G(j,i)+=d[(long)i*NTH+j];
            }
            err=(G-Gref).cwiseAbs().maxCoeff()/denom; full.setAz(G);
            double ea=std::abs((full.fluxLinkageMaterialPair(ph[0].a,ph[0].b)-fref[0])/(std::abs(fref[0])+1e-30));
            std::cerr<<"outer "<<it<<": err_vs_mono="<<err<<" PhiA_err="<<ea<<" cum_wall="<<secs(tl,Clock::now())<<"s\n";
            if(err<tol) break;
        }
        double tloopb=secs(tl,Clock::now()); full.setAz(G);
        std::cerr<<"=== BANDED DONE outer="<<it<<" err="<<err<<" ===\nflux: ";
        for(int k=0;k<3;k++){ double f=full.fluxLinkageMaterialPair(ph[k].a,ph[k].b);
            std::cerr<<ph[k].n<<"="<<f<<"(err "<<std::abs((f-fref[k])/(std::abs(fref[k])+1e-30))*100<<"%) "; }
        std::cerr<<"\nWALL: monolithic="<<t_full<<"s DD_loop="<<tloopb<<"s speedup="<<t_full/tloopb<<"x aggDOF="<<100.0*agg/(NTH*NR)<<"%\n";
        return 0;
    }

    auto rcores=split(NR,Nr), tcores=split(NTH,Nth);
    std::cerr<<"=== DD Nr="<<Nr<<" Nth="<<Nth<<" ov="<<ov<<" p="<<p<<" ("<<Nr*Nth<<" patches) ===\n";

    // per-patch coarsening: cf>1 for patches NOT overlapping the gap/coil radial cols (env DD_GRID_CF,
    // DD_FINE_LO:DD_FINE_HI keep-fine col range). 2D mortar (theta+r) down/up-sample.
    int GCF = getenv("DD_GRID_CF") ? atoi(getenv("DD_GRID_CF")) : 1;
    int finelo=200, finehi=330; if(getenv("DD_FINE")) sscanf(getenv("DD_FINE"),"%d:%d",&finelo,&finehi);
    struct Patch{ int er0,er1,et0,et1,cr0,cr1,ct0,ct1,cf,nthc,nrc; bool ft; std::unique_ptr<MagneticFieldAnalyzer> an; };
    std::vector<Patch> P;
    system("mkdir -p dd_tmp");
    long agg_dof=0; int pid=0;
    auto flip=[&](int gt){ return ((NTH-1-(gt%NTH))%NTH+NTH)%NTH; };
    // Radial band layout. DD_RBANDS="c0:c1:cf:nth,..." gives explicit forbidden-band control: each band
    // has its r-range, coarsening cf, and theta-sector count nth (nth<=1 => FULL-THETA fine/coarse ring,
    // no theta cut -> for the slide-gap +/-a and any no-cut band). Else: uniform Nr bands with auto gap
    // detection (the gap-overlapping band is forced full-theta fine).
    struct RB{int c0,c1,cf,nth;}; std::vector<RB> rblist;
    if(const char* rbe=getenv("DD_RBANDS")){ std::string s(rbe); size_t i=0;
        while(i<s.size()){ int c0,c1,cf2,nth2; if(sscanf(s.c_str()+i,"%d:%d:%d:%d",&c0,&c1,&cf2,&nth2)==4)
            rblist.push_back({c0,c1,std::max(1,cf2),std::max(1,nth2)});
            size_t n=s.find(',',i); if(n==std::string::npos)break; i=n+1; }
        std::cerr<<"DD_RBANDS: "<<rblist.size()<<" custom radial bands\n";
    } else { for(auto[ra0,ra1]:rcores){ bool gb=(ra1>finelo&&ra0<finehi);
        rblist.push_back({ra0,ra1, gb?1:std::max(1,GCF), (gb||Nth==1)?1:Nth}); } }
    // MATERIAL-CONFORMING theta cuts (e.g. tooth-center iron positions) from DD_TCUTS="t0,t1,..." (sorted):
    // sectored bands cut at these theta indices so cut lines pass through iron, NOT through coils/magnets
    // (a cut straddling a different material corrupts its flux + stiffens the interface). Else uniform.
    std::vector<int> tcuts;
    if(const char* tce=getenv("DD_TCUTS")){ std::string s(tce); size_t i=0;
        while(i<s.size()){ int t; if(sscanf(s.c_str()+i,"%d",&t)==1) tcuts.push_back(t);
            size_t n=s.find(',',i); if(n==std::string::npos)break; i=n+1; }
        std::cerr<<"DD_TCUTS: "<<tcuts.size()<<" material-conforming theta cuts\n"; }
    for(auto&rbnd:rblist){
        int ra0=rbnd.c0, ra1=rbnd.c1;
        int er0=std::max(0,ra0-ov), er1=std::min(NR,ra1+ov);
        int cf=rbnd.cf;
        std::vector<std::pair<int,int>> sectors;
        if(rbnd.nth<=1) sectors={{0,NTH}};
        else if(!tcuts.empty()){ for(size_t k=0;k<tcuts.size();k++){
            int a=tcuts[k], b=(k+1<tcuts.size()? tcuts[k+1] : tcuts[0]+NTH); sectors.push_back({a,b}); } }
        else { auto tc=split(NTH,rbnd.nth); for(auto&pr:tc) sectors.push_back(pr); }
        for(auto[tb0,tb1]:sectors){
            Patch q; q.er0=er0;q.er1=er1;q.cr0=ra0;q.cr1=ra1;q.ct0=tb0;q.ct1=tb1; q.cf=cf;
            q.ft=(tb0==0 && tb1==NTH);   // full-theta (periodic) patch: forbidden-band ring or Nth==1
            if(q.ft){q.et0=0;q.et1=NTH;} else {q.et0=tb0-ov;q.et1=tb1+ov;}
            int nthx=q.et1-q.et0, nrx=er1-er0;
            q.nthc=std::max(2,nthx/cf); q.nrc=std::max(2,nrx/cf);
            agg_dof += (long)q.nthc*q.nrc;
            // clean coarse crop: solver grid (mc,cc) <-> global theta [et0+mc*cf..], col [er0+cc*cf..];
            // bake in the solver's grid<->image flip. cf=1 reduces exactly to the C1 crop.
            cv::Mat crop(q.nthc,q.nrc,CV_8UC3);
            for(int r=0;r<q.nthc;r++){ int gt=q.et0+(q.nthc-1-r)*cf+cf/2; int ir=flip(gt);
                for(int cc=0;cc<q.nrc;cc++){ int gc=std::min(er0+cc*cf+cf/2,er1-1); crop.at<cv::Vec3b>(r,cc)=img.at<cv::Vec3b>(ir,gc); } }
            cv::Mat cb; cv::cvtColor(crop,cb,cv::COLOR_RGB2BGR);
            std::string png="dd_tmp/p"+std::to_string(pid)+".png"; cv::imwrite(png,cb);
            YAML::Node cfg=YAML::Clone(base);
            cfg["polar_domain"]["r_start"]=RS+er0*DR;
            cfg["polar_domain"]["r_end"]  =RS+(er1-1)*DR;
            if(q.ft) cfg["polar_domain"]["theta_range"]="2*pi"; else cfg["polar_domain"]["theta_range"]=nthx*DTH;
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
            if(q.ft){edge("theta_min","periodic",0,0,0);edge("theta_max","periodic",0,0,0);}
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
    // ---- optional 2-level theta+r coarse space (DD_COARSE="CR:CTH") to bound outer iters with many
    //      theta-wedges (1-level Schwarz residual grows with sector count; the coarse space fixes it). ----
    int CR=0,CTH=0; bool use_cs=false, cs_nonlin=(getenv("DD_CS_NONLIN")!=nullptr);
    Eigen::SparseMatrix<double> Pcs, Afull; Eigen::VectorXd bvec;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> Aclu;
    if(const char* cs=getenv("DD_COARSE")){ sscanf(cs,"%d:%d",&CR,&CTH); use_cs=(CR>0&&CTH>0); }
    if(use_cs){
        int nrcs=NR/CR, nthcs=NTH/CTH; long Nc=(long)nrcs*nthcs;
        std::vector<Eigen::Triplet<double>> tp;
        for(int i=0;i<NR;i++){ double fic=(double)i/CR; int ic0=std::min((int)fic,nrcs-1),ic1=std::min(ic0+1,nrcs-1); double wr=fic-(int)fic;
            for(int j=0;j<NTH;j++){ double fjc=(double)j/CTH; int jc0=((int)fjc)%nthcs,jc1=(jc0+1)%nthcs; double wt=fjc-(int)fjc; int f=i*NTH+j;
                tp.push_back({f,(int)(ic0*nthcs+jc0),(1-wr)*(1-wt)}); tp.push_back({f,(int)(ic0*nthcs+jc1),(1-wr)*wt});
                tp.push_back({f,(int)(ic1*nthcs+jc0),wr*(1-wt)});     tp.push_back({f,(int)(ic1*nthcs+jc1),wr*wt}); } }
        Pcs.resize((long)NTH*NR,Nc); Pcs.setFromTriplets(tp.begin(),tp.end());
        full.setAz(G); full.buildPolarOperator(Afull,bvec);
        Eigen::SparseMatrix<double> Ac=(Eigen::SparseMatrix<double>(Pcs.transpose())*Afull*Pcs).pruned();
        Aclu.analyzePattern(Ac); Aclu.factorize(Ac);
        std::cerr<<"grid coarse space: CR="<<CR<<" CTH="<<CTH<<" coarseDOF="<<Nc<<(cs_nonlin?" (rebuilt/iter)":"")<<"\n";
    }
    double relax = getenv("DD_RELAX") ? atof(getenv("DD_RELAX")) : 1.0;   // under-relaxation omega
    if(relax!=1.0) std::cerr<<"under-relaxation omega="<<relax<<"\n";
    int mAA = getenv("DD_ANDERSON") ? atoi(getenv("DD_ANDERSON")) : 0;    // Anderson depth (PoC-2)
    std::vector<Eigen::VectorXd> Fhist, Ghist;
    if(mAA>0) std::cerr<<"Anderson acceleration depth m="<<mAA<<"\n";
    // DD PARALLELISM/CONDITIONING profiling: per-patch wall T_i and AMGCL-CG iters per sweep ->
    // serial sum(T_i) vs PARALLEL max(T_i); local vs global CG-iter conditioning.
    double par_wall=0.0, ser_wall=0.0; long dd_li_total=0;
    // PLATEAU DETECTION + best-iterate (DD_PLATEAU): Anderson can drift past its optimum; track the
    // lowest-Schwarz-residual iterate (practical criterion, no Gref) and stop when it stalls/rises.
    Eigen::MatrixXd Gbest=G; double best_res=1e30, prev_res=1e30; int best_it=0, plateau=0;
    bool use_plateau=(getenv("DD_PLATEAU")!=nullptr);
    auto tloop=Clock::now(); int it=0; double err=1.0;
    for(it=1; it<=maxo; ++it){
        Eigen::MatrixXd Gprev=G;
        double sumT=0,maxT=0; long sumLI=0,maxLI=0; int maxp=-1, pidx=0;
        for(auto&q:P){
            int cf=q.cf;
            // theta-avg over coarse cell j's fine global theta range; r-avg over coarse cell i's cols
            auto Gth=[&](int j,int c)->double{ double s=0; for(int t=0;t<cf;t++) s+=Gat(q.et0+j*cf+t,c); return s/cf; };
            std::vector<double> gin(q.nthc,0),gout(q.nthc,0),gtmin(q.nrc,0),gtmax(q.nrc,0);
            for(int j=0;j<q.nthc;j++){
                if(q.er0!=0)      gin[j] =p*PR*Gth(j,q.er0)   -(Gth(j,q.er0)-Gth(j,q.er0-1))/DR;
                if(q.er1-1!=NR-1) gout[j]=p*PR*Gth(j,q.er1-1) +(Gth(j,q.er1)-Gth(j,q.er1-1))/DR; }
            if(!q.ft){ int t0=q.et0, t1=q.et0+(q.nthc-1)*cf+cf/2;  // boundary cell-center global thetas
                // PoC-3b conservative restriction: r-AVERAGE the theta-interface trace over each coarse
                // cell (consistent with bilinear prolongation), not a point r-sample.
                auto Grav=[&](int gth,int i)->double{ double s=0; int n=0;
                    for(int k=0;k<cf;k++){ int c=q.er0+i*cf+k; if(c<q.er1){ s+=Gat(gth,c); n++; } } return s/std::max(n,1); };
                for(int i=0;i<q.nrc;i++){
                    gtmin[i]=p*Grav(t0,i)   -(Grav(t0,i)-Grav(t0-cf,i))/(cf*DTH);
                    gtmax[i]=p*Grav(t1,i)   +(Grav(t1+cf,i)-Grav(t1,i))/(cf*DTH); } }
            if(q.er0!=0)      q.an->setBoundaryProfile("inner",gin);
            if(q.er1-1!=NR-1) q.an->setBoundaryProfile("outer",gout);
            if(!q.ft){ q.an->setBoundaryProfile("theta_min",gtmin); q.an->setBoundaryProfile("theta_max",gtmax); }
            Eigen::MatrixXd pAz(q.nthc,q.nrc);
            for(int j=0;j<q.nthc;j++)for(int i=0;i<q.nrc;i++) pAz(j,i)=Gat(q.et0+j*cf+cf/2,std::min(q.er0+i*cf+cf/2,q.er1-1));
            q.an->setAz(pAz);
            long li0=q.an->getTotalLinearIters(); auto ts=Clock::now();
            q.an->solve();
            double Ti=secs(ts,Clock::now()); long Li=q.an->getTotalLinearIters()-li0;
            sumT+=Ti; sumLI+=Li; if(Ti>maxT){maxT=Ti;maxp=pidx;} if(Li>maxLI)maxLI=Li; pidx++;
            const Eigen::MatrixXd& sol=q.an->getAz();
            // write CORE back to fine G (PoC-3 conservative mortar: BILINEAR prolongation, not
            // piecewise-constant. A smooth field means neighbour Robin gradients across a coarse cell
            // are correct -> removes the piecewise-constant flux bias. cf=1 reduces to exact sol.)
            auto Sc=[&](int jj,int ii)->double{ jj=std::max(0,std::min(jj,q.nthc-1));
                ii=std::max(0,std::min(ii,q.nrc-1)); return sol(jj,ii); };
            for(int j=0;j<q.nthc;j++) for(int t=0;t<cf;t++){ int gug=q.et0+j*cf+t; int gt=((gug%NTH)+NTH)%NTH;
                if(!q.ft && !(q.ct0<=gug&&gug<q.ct1)) continue;
                double fj=((double)(gug-q.et0-cf/2))/cf; int j0=(int)std::floor(fj); double wj=fj-j0;
                for(int i=0;i<q.nrc;i++) for(int s=0;s<cf;s++){ int gc=q.er0+i*cf+s;
                    if(gc>=q.er1) continue; if(!(q.cr0<=gc&&gc<q.cr1)) continue;
                    double fi=((double)(gc-q.er0-cf/2))/cf; int i0=(int)std::floor(fi); double wi=fi-i0;
                    G(gt,gc)=(1-wj)*(1-wi)*Sc(j0,i0)+(1-wj)*wi*Sc(j0,i0+1)
                            +wj*(1-wi)*Sc(j0+1,i0)+wj*wi*Sc(j0+1,i0+1); } }
        }
        par_wall+=maxT; ser_wall+=sumT; dd_li_total+=sumLI;
        std::cerr<<"  [profile it"<<it<<"] serial_sumT="<<sumT<<"s PARALLEL_maxT="<<maxT<<"s(patch"<<maxp
                 <<") CG sum="<<sumLI<<" max="<<maxLI<<" mean="<<(P.size()?sumLI/(long)P.size():0)<<"\n";
        if(relax!=1.0) G = Gprev + relax*(G-Gprev);   // PoC-1: under-relax the composite update
        // ---- PoC-2: Anderson acceleration on the fixed point G = H(G) (multi-secant quasi-Newton on
        //      the residual f=H(G)-G; converges divergent fixed points without a fine smoother) ----
        if(mAA>0){
            const long Nn=(long)NTH*NR;
            Eigen::VectorXd xk=Eigen::Map<Eigen::VectorXd>(Gprev.data(),Nn);
            Eigen::VectorXd gk=Eigen::Map<Eigen::VectorXd>(G.data(),Nn);
            Eigen::VectorXd fk=gk-xk;
            Fhist.push_back(fk); Ghist.push_back(gk);
            if((int)Fhist.size()>mAA+1){ Fhist.erase(Fhist.begin()); Ghist.erase(Ghist.begin()); }
            int m=(int)Fhist.size()-1;
            if(m>=1){
                Eigen::MatrixXd dF(Nn,m),dG(Nn,m);
                for(int i=0;i<m;i++){ dF.col(i)=Fhist[i+1]-Fhist[i]; dG.col(i)=Ghist[i+1]-Ghist[i]; }
                Eigen::MatrixXd AtA=dF.transpose()*dF;
                AtA.diagonal().array()+=1e-10*(AtA.diagonal().maxCoeff()+1e-30);
                Eigen::VectorXd gamma=AtA.ldlt().solve(dF.transpose()*fk);
                Eigen::VectorXd xnew=gk-dG*gamma;
                Eigen::Map<Eigen::VectorXd>(G.data(),Nn)=xnew;
            }
        }
        // ---- 2-level coarse correction: G += P * Ac^-1 * P^T (b - A*G) ----
        if(use_cs){
            if(cs_nonlin){ full.setAz(G); full.buildPolarOperator(Afull,bvec);
                Eigen::SparseMatrix<double> Ac=(Eigen::SparseMatrix<double>(Pcs.transpose())*Afull*Pcs).pruned();
                Aclu.factorize(Ac); }
            Eigen::VectorXd Gv((long)NTH*NR);
            for(int i=0;i<NR;i++)for(int j=0;j<NTH;j++) Gv[(long)i*NTH+j]=G(j,i);
            Eigen::VectorXd R=bvec-Afull*Gv;
            Eigen::VectorXd d=Pcs*Aclu.solve(Eigen::VectorXd(Pcs.transpose()*R));
            for(int i=0;i<NR;i++)for(int j=0;j<NTH;j++) G(j,i)+=d[(long)i*NTH+j];
        }
        err=(G-Gref).cwiseAbs().maxCoeff()/denom;
        // flux from G via the full analyzer's material map
        full.setAz(G);
        double fa=full.fluxLinkageMaterialPair(ph[0].a,ph[0].b);
        double ea=std::abs((fa-fref[0])/ (std::abs(fref[0])+1e-30));
        double res=(G-Gprev).norm()/(G.norm()+1e-30);   // Schwarz residual (practical, no Gref)
        if(res<best_res){ best_res=res; Gbest=G; best_it=it; }
        std::cerr<<"outer "<<it<<": err_vs_mono="<<err<<"  PhiA_err="<<ea
                 <<"  schwarz_res="<<res<<"  cum_wall="<<secs(tloop,Clock::now())<<"s\n";
        if(err<tol) break;
        if(use_plateau){ if(res>0.9*prev_res){ if(++plateau>=2){
            std::cerr<<"  [plateau: Schwarz residual stalled -> stop at best iter "<<best_it<<"]\n"; break; } }
            else plateau=0; }
        prev_res=res;
    }
    double t_loop=secs(tloop,Clock::now());
    if(use_plateau){ G=Gbest; std::cerr<<"[plateau: using best iterate it="<<best_it<<" schwarz_res="<<best_res<<"]\n"; }
    full.setAz(G);
    std::cerr<<"=== DONE outer="<<it<<" err="<<err<<" ===\n";
    std::cerr<<"flux: ";
    for(int k=0;k<3;k++){ double f=full.fluxLinkageMaterialPair(ph[k].a,ph[k].b);
        std::cerr<<ph[k].n<<"="<<f<<"(err "<<std::abs((f-fref[k])/(std::abs(fref[k])+1e-30))*100<<"%) "; }
    std::cerr<<"\nWALL: monolithic="<<t_full<<"s  DD_loop="<<t_loop<<"s  speedup="<<t_full/t_loop<<"x"
             <<"  aggDOF="<<100.0*agg_dof/(NTH*NR)<<"%\n";
    std::cerr<<"\n=== DD PARALLELISM / CONDITIONING (re-eval beyond DOF) ===\n";
    std::cerr<<"serial sum_sweeps(sum_i T_i)="<<ser_wall<<"s ;  IDEAL-PARALLEL sum_sweeps(max_i T_i)="<<par_wall<<"s\n";
    std::cerr<<"  intra-sweep parallel speedup (sum/max avg)="<<(par_wall>0?ser_wall/par_wall:0)<<"x ("
             <<P.size()<<" patches, "<<Nth<<" theta x "<<Nr<<" r)\n";
    std::cerr<<"  PARALLEL DD vs monolithic: "<<t_full<<"s / "<<par_wall<<"s = "<<(par_wall>0?t_full/par_wall:0)<<"x\n";
    std::cerr<<"CONDITIONING: monolithic avg "<<(mono_ns?mono_li/mono_ns:0)<<" CG/solve (global) vs DD per-patch "
             <<"total "<<dd_li_total<<" CG over all patches&sweeps; per-patch avg/solve printed per sweep above.\n";
    return 0;
}
