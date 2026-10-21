#include "core.h"
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <queue>

namespace omfdm::d3 {
namespace {
using Matrix=Eigen::SparseMatrix<double,Eigen::RowMajor>;
std::vector<double> solve(const Domain& d,const std::vector<double>& weights,const std::vector<double>& rhs,bool pin,bool direct){
    int n=int(d.cells.size());std::vector<Eigen::Triplet<double>> triplets;std::vector<double> diagonal(n,0);
    for(size_t i=0;i<d.faces.size();++i){auto f=d.faces[i];double w=weights[i];if(!w)continue;
        if(f.a>=0)diagonal[f.a]+=w;if(f.b>=0)diagonal[f.b]+=w;
        if(f.a>=0&&f.b>=0&&(!pin||(f.a!=0&&f.b!=0))){triplets.emplace_back(f.a,f.b,-w);triplets.emplace_back(f.b,f.a,-w);}}
    for(int i=0;i<n;++i){require(diagonal[i]>0||n==1,"Unconstrained cell");triplets.emplace_back(i,i,pin&&i==0?1:diagonal[i]);}
    Matrix A(n,n);A.setFromTriplets(triplets.begin(),triplets.end());A.makeCompressed();
    std::vector<double> b=rhs,x(n,0);if(pin)b[0]=0;
    double norm=std::sqrt(std::inner_product(b.begin(),b.end(),b.begin(),0.));if(norm==0)return x;
    for(auto& value:b)value/=norm; // avoid AMGCL absolute stopping on a roundoff-sized correction RHS
    if(direct){Eigen::SimplicialLDLT<Matrix> factor;factor.compute(A);require(factor.info()==Eigen::Success,"Direct factorization failed");Eigen::VectorXd v=factor.solve(Eigen::Map<Eigen::VectorXd>(b.data(),n));require(factor.info()==Eigen::Success,"Direct solve failed");std::copy(v.data(),v.data()+n,x.begin());}
    else{
        using Solver=amgcl::make_solver<amgcl::amg<amgcl::backend::builtin<double>,amgcl::coarsening::smoothed_aggregation,amgcl::relaxation::spai0>,amgcl::solver::cg<amgcl::backend::builtin<double>>>;
        std::vector<int> ptr(A.outerIndexPtr(),A.outerIndexPtr()+n+1),col(A.innerIndexPtr(),A.innerIndexPtr()+A.nonZeros());std::vector<double> val(A.valuePtr(),A.valuePtr()+A.nonZeros());
        Solver::params prm;prm.solver.tol=1e-12;prm.solver.maxiter=3000;Solver solver(std::tie(n,ptr,col,val),prm);solver(b,x);
    }
    Eigen::VectorXd residual=A*Eigen::Map<Eigen::VectorXd>(x.data(),n)-Eigen::Map<Eigen::VectorXd>(b.data(),n);
    require(residual.allFinite()&&residual.norm()<=1e-10,"Scalar solve did not converge; normalized true residual="+std::to_string(residual.norm()));
    for(auto& value:x)value*=norm;
    return x;
}
void accumulate(const Face& f,double q,std::vector<double>& out){if(f.a>=0)out[f.a]+=q;if(f.b>=0)out[f.b]-=q;}
std::vector<Vec> density(const Domain& d,const std::vector<double>& q){
    std::vector<Vec> value(d.cells.size());std::vector<std::array<int,3>> count(d.cells.size());
    for(size_t i=0;i<d.faces.size();++i){auto f=d.faces[i];for(int c:{f.a,f.b})if(c>=0){value[c][f.key.axis]+=q[i]/f.area;++count[c][f.key.axis];}}
    for(size_t c=0;c<value.size();++c)for(int a=0;a<3;++a)if(count[c][a])value[c][a]/=count[c][a];return value;
}
double magnitude(const Vec& v){return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);}
// Voxel faces crossing a potential level approximate its oriented surface.
// Projected areas remove the staircase-area bias for oblique sections.
struct Section{double level=0,area=0;std::vector<std::pair<size_t,double>> faces;};
std::vector<Section> sections(const Domain& d,const std::vector<double>& p,const std::vector<Vec>& tangent){
    auto mm=std::minmax_element(p.begin(),p.end());double low=*mm.first,high=*mm.second;require(high>low,"No resolved current path; use more cells along the coil");
    std::vector<Section> result;
    for(int k=1;k<=32;++k){Section s;s.level=low+(high-low)*k/33.;
        for(size_t i=0;i<d.faces.size();++i){const auto& f=d.faces[i];if(f.a<0||f.b<0||f.cutSign)continue;
            if((p[f.a]<s.level)==(p[f.b]<s.level))continue;
            Vec t;for(int a=0;a<3;++a)t[a]=tangent[f.a][a]+tangent[f.b][a];double norm=magnitude(t);require(norm>1e-12,"Ambiguous opposing path directions");double projection=std::abs(t[f.key.axis]/norm);double area=f.area*projection;
            if(area>0){s.faces.push_back({i,area});s.area+=area;}}
        require(s.area>0&&!s.faces.empty(),"Vanishing winding cross section");
        // Disjoint isopotential sections indicate parallel branches, not one winding bundle.
        std::map<Index,std::vector<size_t>> vertices;
        for(size_t j=0;j<s.faces.size();++j){auto f=d.faces[s.faces[j].first];int u=(f.key.axis+1)%3,v=(f.key.axis+2)%3;for(int x=0;x<2;++x)for(int y=0;y<2;++y){auto pos=f.key.index;pos[u]+=x;pos[v]+=y;vertices[pos].push_back(j);}}
        std::vector<bool> seen(s.faces.size());std::queue<size_t> todo;todo.push(0);seen[0]=true;size_t reached=0;
        while(!todo.empty()){auto j=todo.front();todo.pop();++reached;auto f=d.faces[s.faces[j].first];int u=(f.key.axis+1)%3,v=(f.key.axis+2)%3;for(int x=0;x<2;++x)for(int y=0;y<2;++y){auto pos=f.key.index;pos[u]+=x;pos[v]+=y;for(auto next:vertices[pos])if(!seen[next]){seen[next]=true;todo.push(next);}}}
        require(reached==s.faces.size(),"Branched winding: disconnected perpendicular sections");result.push_back(std::move(s));
    }
    return result;
}
}
Result prepareCurrent(const Grid&,const Domain& d,const Coil& coil,bool direct){
    const size_t n=d.cells.size(),nf=d.faces.size();std::vector<double> weights(nf),drive(nf),rhs(n,0);
    for(size_t i=0;i<nf;++i){const auto& f=d.faces[i];if(f.a>=0&&f.b>=0)weights[i]=f.conductance;else if(f.terminal)weights[i]=f.conductance;
        if(f.terminal==1)drive[i]=f.a<0?1:-1;if(f.cutSign)drive[i]=f.cutSign;accumulate(f,-weights[i]*drive[i],rhs);}
    Result r;r.potential=solve(d,weights,rhs,d.closed,direct);r.flux.resize(nf);
    double unitCurrent=0;
    for(size_t i=0;i<nf;++i){const auto& f=d.faces[i];r.flux[i]=weights[i]*((f.a>=0?r.potential[f.a]:0)-(f.b>=0?r.potential[f.b]:0)+drive[i]);
        if(d.closed&&f.cutSign)unitCurrent+=f.cutSign*r.flux[i];else if(!d.closed&&f.terminal==1)unitCurrent+=(f.a<0?1:-1)*r.flux[i];}
    require(std::isfinite(unitCurrent)&&unitCurrent>0,"No conducting terminal-to-terminal path");
    const double resistance=1/unitCurrent;for(auto& q:r.flux)q/=unitCurrent;
    double worstUniformity=0;Json sectionReports=Json::array();
    if(coil.model=="stranded_uniform"){
        auto tangent=density(d,r.flux);double maxMagnitude=0;for(auto t:tangent)maxMagnitude=std::max(maxMagnitude,magnitude(t));
        for(auto& t:tangent){double norm=magnitude(t);require(norm>maxMagnitude*1e-10,"Dead end/stagnant or unresolved winding path");for(auto& v:t)v/=norm;}
        auto cross=sections(d,r.potential,tangent);
        auto areaAt=[&](double p){auto it=std::lower_bound(cross.begin(),cross.end(),p,[](const Section& s,double v){return s.level<v;});if(it==cross.begin())return it->area;if(it==cross.end())return cross.back().area;auto before=it-1;double t=(p-before->level)/(it->level-before->level);return before->area*(1-t)+it->area*t;};
        double inletArea=0,outletArea=0,cutArea=0;for(auto f:d.faces){if(f.terminal==1)inletArea+=f.area;if(f.terminal==-1)outletArea+=f.area;if(f.cutSign)cutArea+=f.area;}
        std::vector<double> target(nf),freeWeights(nf),correctionRhs(n,0);
        for(size_t i=0;i<nf;++i){auto f=d.faces[i];
            if(f.cutSign)target[i]=f.cutSign*f.area/cutArea;
            else if(f.terminal)target[i]=(f.a<0?1:-1)*(f.terminal==1?1:-1)*f.area/(f.terminal==1?inletArea:outletArea);
            else if(f.a>=0&&f.b>=0){Vec t;for(int a=0;a<3;++a)t[a]=tangent[f.a][a]+tangent[f.b][a];double norm=magnitude(t);require(norm>1e-12,"Ambiguous winding tangent");target[i]=f.area*t[f.key.axis]/norm/areaAt((r.potential[f.a]+r.potential[f.b])*.5);freeWeights[i]=f.area/f.distance;}
            accumulate(f,-target[i],correctionRhs);
        }
        auto correction=solve(d,freeWeights,correctionRhs,true,direct);
        for(size_t i=0;i<nf;++i){auto f=d.faces[i];r.flux[i]=target[i]+freeWeights[i]*((f.a>=0?correction[f.a]:0)-(f.b>=0?correction[f.b]:0));}
        for(const auto& s:cross){double square=0,total=0;
            for(auto entry:s.faces){auto f=d.faces[entry.first];double projectedArea=entry.second;double flow=r.flux[entry.first]*(r.potential[f.a]>r.potential[f.b]?1:-1);double deviation=flow/projectedArea*s.area-1;square+=projectedArea*deviation*deviation;total+=flow;}
            double deviation=std::sqrt(square/s.area);worstUniformity=std::max(worstUniformity,deviation);sectionReports.push_back({{"potential_level",s.level},{"projected_area_m2",s.area},{"relative_rms_deviation",deviation},{"unit_current",total}});
        }
    }
    // Verify on the unit solution as well: zero excitation cannot hide a defective source shape.
    std::vector<double> balance(n,0);double measured=0,wallLeak=0;
    for(size_t i=0;i<nf;++i){auto f=d.faces[i];accumulate(f,r.flux[i],balance);if(d.closed&&f.cutSign)measured+=f.cutSign*r.flux[i];else if(!d.closed&&f.terminal==1)measured+=(f.a<0?1:-1)*r.flux[i];if((f.a<0||f.b<0)&&!f.terminal)wallLeak=std::max(wallLeak,std::abs(r.flux[i]));}
    double divergence=0;for(double b:balance)divergence=std::max(divergence,std::abs(b));double currentError=std::abs(measured-1);
    const double requested=coil.current*coil.turns;for(auto& q:r.flux)q*=requested;r.density=density(d,r.flux);
    r.accepted=divergence<=1e-8&&currentError<=1e-8&&wallLeak<=1e-8&&(coil.model!="stranded_uniform"||worstUniformity<=coil.uniformityTolerance);
    r.diagnostics={{"schema_version",1},{"model",coil.model},{"accepted",r.accepted},{"source_status",d.closed?"closed_cut":"open_terminals"},{"magnetic_solver_validated",false},{"current_A",coil.current},{"effective_ampere_turns",requested},{"max_cell_balance_relative",divergence},{"total_current_relative_error",currentError},{"insulating_wall_leak_relative",wallLeak},{"max_section_relative_rms_deviation",worstUniformity},{"uniformity_tolerance",coil.uniformityTolerance},{"sections",sectionReports},{"potential_kind","auxiliary_unit_drive"},{"linear_solver",direct?"Eigen LDLT":"AMGCL CG"}};
    if(coil.model=="bulk_dc"){r.diagnostics["resistance_ohm"]=resistance;r.diagnostics["voltage_V"]=coil.current*resistance;r.diagnostics["joule_power_W"]=coil.current*coil.current*resistance;}
    return r;
}
}
