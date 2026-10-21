#pragma once
#include "core.h"

namespace omfdm::d3 {
// Oriented integral cochains: node scalar, edge integral of A, face flux of B.
// Incidence contains topology only; material/metric Hodge operators are separate.
// A -> A + gradient(psi) is the gauge nullspace, not three scalar Poisson solves.
struct CubicalComplex {
    Index cells;
    using Field = std::array<std::vector<double>,3>;
    static size_t index(const Index& p,const Index& n){return p[0]+size_t(n[0])*(p[1]+size_t(n[1])*p[2]);}
    static size_t count(const Index& n){return size_t(n[0])*n[1]*n[2];}
    Index nodes()const{return {cells[0]+1,cells[1]+1,cells[2]+1};}
    Index edges(int a)const{auto n=nodes();--n[a];return n;}
    Index faces(int a)const{auto n=cells;++n[a];return n;}
    template<class F>static void each(const Index& n,F f){for(int z=0;z<n[2];++z)for(int y=0;y<n[1];++y)for(int x=0;x<n[0];++x)f(Index{x,y,z});}
    Field gradient(const std::vector<double>& v)const{
        require(v.size()==count(nodes()),"Node cochain size mismatch");Field out;
        for(int a=0;a<3;++a){auto n=edges(a);out[a].resize(count(n));each(n,[&](Index p){auto q=p;++q[a];out[a][index(p,n)]=v[index(q,nodes())]-v[index(p,nodes())];});}return out;
    }
    Field curl(const Field& v)const{
        for(int a=0;a<3;++a)require(v[a].size()==count(edges(a)),"Edge cochain size mismatch");Field out;
        for(int a=0;a<3;++a){int u=(a+1)%3,w=(a+2)%3;auto n=faces(a);out[a].resize(count(n));each(n,[&](Index p){auto pu=p,pw=p;++pu[u];++pw[w];out[a][index(p,n)]=v[w][index(pu,edges(w))]-v[w][index(p,edges(w))]-v[u][index(pw,edges(u))]+v[u][index(p,edges(u))];});}return out;
    }
    std::vector<double> divergence(const Field& v)const{
        for(int a=0;a<3;++a)require(v[a].size()==count(faces(a)),"Face cochain size mismatch");std::vector<double> out(count(cells));
        each(cells,[&](Index p){double sum=0;for(int a=0;a<3;++a){auto q=p;++q[a];sum+=v[a][index(q,faces(a))]-v[a][index(p,faces(a))];}out[index(p,cells)]=sum;});return out;
    }
};
}
