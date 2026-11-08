// Offline accuracy audit using the production FixBackground point-query API.
// Link against the same CPU SPARTA library as the benchmark. Run a setup-only
// deck (-in ... with run 0) on one MPI rank. This never replaces physics data.
#include "mpi.h"
#include "sparta.h"
#include "input.h"
#include "modify.h"
#include "fix_background.h"
#include "compute_nearest_surf_grid.h"
#include "grid.h"
#include "particle.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace SPARTA_NS;
constexpr int NF = 10;
using Fields = std::array<double,NF>;
const char *names[NF] = {"Te","Ti","ne","ni","Bx","By","Bz","Ex","Ey","Ez"};
Fields sample(FixBackground *pd, const double x[3], int cell) {
  PlasmaPointSample p;
  pd->sample_point(x,p,cell,-1,PLASMA_NEED_THERMO|PLASMA_NEED_B|PLASMA_NEED_E);
  return {p.te,p.ti,p.ne,p.ni,p.b[0],p.b[1],p.b[2],p.e[0],p.e[1],p.e[2]};
}
struct Errors {
  std::vector<double> relative;
  double sum2=0, ref2=0, maxabs=0;
  long negative=0;
  void add(double pred,double ref,double floor) {
    const double e=pred-ref;
    relative.push_back(std::abs(e)/std::max(std::abs(ref),floor));
    sum2+=e*e; ref2+=ref*ref; maxabs=std::max(maxabs,std::abs(e));
    if (pred<0) negative++;
  }
};
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  int size=0; MPI_Comm_size(MPI_COMM_WORLD,&size);
  if (size!=1) { fprintf(stderr,"Audit requires one rank\n"); MPI_Abort(MPI_COMM_WORLD,2); }
  {
    SPARTA s(argc,argv,MPI_COMM_WORLD);
    s.input->file();
    auto *pd=dynamic_cast<FixBackground *>(s.modify->fix[s.modify->find_fix("pd")]);
    auto *geom=dynamic_cast<ComputeNearestSurfGrid *>(s.modify->compute[s.modify->find_compute("cgeom")]);
    if (!pd || !geom) MPI_Abort(MPI_COMM_WORLD,3);
    geom->compute_per_grid();
    // Component relative errors use explicit SI floors. NRMSE does not use a floor.
    const Fields floors={1e-3,1e-3,1e10,1e10,1e-6,1e-6,1e-6,1e-3,1e-3,1e-3};
    Errors errors[2][2][NF]; // center/linear, all/near-wall cell, field
    long ncells=0, nearcells=0;
    const double begin=MPI_Wtime();
    for (int c=0;c<s.grid->nlocal;c++) {
      const auto &cell=s.grid->cells[c];
      if (cell.nsplit<1 || s.grid->cinfo[c].volume<=0) continue;
      // Near-wall is a cell-centroid distance bin, not a point/sheath mask.
      const bool near=geom->array_grid[c][0]<=0.005;
      ncells++; nearcells+=near;
      double center[3],width[3];
      for(int d=0;d<3;d++) { center[d]=(cell.lo[d]+cell.hi[d])*0.5; width[d]=cell.hi[d]-cell.lo[d]; }
      const Fields constant=sample(pd,center,c);
      Fields mean={},slopes[3]={};
      // Symmetric 3^3 least-squares training stencil, independent holdout below.
      for(int i=-1;i<=1;i++) for(int j=-1;j<=1;j++) for(int k=-1;k<=1;k++) {
        const double u[3]={i/3.0,j/3.0,k/3.0};
        double x[3]; for(int d=0;d<3;d++) x[d]=center[d]+u[d]*width[d];
        const Fields f=sample(pd,x,c);
        for(int a=0;a<NF;a++) { mean[a]+=f[a]/27; for(int d=0;d<3;d++) slopes[d][a]+=u[d]*f[a]/2; }
      }
      constexpr double holdout[3]={-0.42,-0.13,0.27};
      for(double ux:holdout) for(double uy:holdout) for(double uz:holdout) {
        const double u[3]={ux,uy,uz};
        double x[3]; for(int d=0;d<3;d++) x[d]=center[d]+u[d]*width[d];
        const Fields exact=sample(pd,x,c);
        for(int a=0;a<NF;a++) {
          double linear=mean[a]; for(int d=0;d<3;d++) linear+=u[d]*slopes[d][a];
          for(int r=0;r<=(near?1:0);r++) {
            errors[0][r][a].add(constant[a],exact[a],floors[a]);
            errors[1][r][a].add(linear,exact[a],floors[a]);
          }
        }
      }
    }
    const char *path=getenv("OE_AUDIT_CSV");
    FILE *fp=path?fopen(path,"w"):stdout;
    fprintf(fp,"model,region,field,n,p50_relative,p95_relative,p99_relative,max_relative,nrmse,max_absolute,negative_predictions\n");
    for(int m=0;m<2;m++) for(int r=0;r<2;r++) for(int a=0;a<NF;a++) {
      auto &e=errors[m][r][a]; auto &v=e.relative; std::sort(v.begin(),v.end());
      if(v.empty()) continue;
      fprintf(fp,"%s,%s,%s,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%ld\n",
        m?"linear":"center",r?"near_wall_5mm":"all",names[a],v.size(),
        v[size_t(.50*(v.size()-1))],v[size_t(.95*(v.size()-1))],v[size_t(.99*(v.size()-1))],
        v.back(),e.ref2>0?std::sqrt(e.sum2/e.ref2):0,e.maxabs,e.negative);
    }
    if(fp!=stdout) fclose(fp);
    printf("OE_AUDIT cells=%ld near_cells=%ld seconds=%.6g OnePart_bytes=%zu\n",ncells,nearcells,MPI_Wtime()-begin,sizeof(Particle::OnePart));
  }
  MPI_Finalize();
}
