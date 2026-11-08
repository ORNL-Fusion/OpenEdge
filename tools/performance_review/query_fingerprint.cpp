// Exact query-sequence comparison, including shared edges and intermediate
// pusher positions. Run with the audit setup deck and one MPI rank.
#include "mpi.h"
#include "sparta.h"
#include "input.h"
#include "modify.h"
#include "fix_background.h"
#include "compute_plasma_fields.h"
#include "particle.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace SPARTA_NS;
uint64_t digest=14695981039346656037ULL;
void add(double x) {
  unsigned char bytes[sizeof(x)]; std::memcpy(bytes,&x,sizeof(x));
  for(unsigned char b:bytes) {digest^=b; digest*=1099511628211ULL;}
}
void point(FixBackground *pd,double x[3]) {
  PlasmaPointSample p;
  // Preserve a scalar memo while B/E sample intermediate pusher positions.
  pd->sample_point(x,p,0,0,PLASMA_NEED_THERMO);
  add(p.te);add(p.ti);add(p.ne);add(p.ni);
  double br,bz,bt;
  pd->bfield_at_xyz(x,br,bz,bt,0,0);add(br);add(bz);add(bt);
  double shifted[3]={x[0]+1e-8,x[1],x[2]-1e-8};
  const auto b=pd->query_bfield_at_point(shifted,0,0,false);
  add(b.br);add(b.bz);add(b.bt);
  double er,ez,et;
  pd->query_efield_at_point(x,er,ez,et,0,0);add(er);add(ez);add(et);
  pd->sample_point(x,p,0,0,PLASMA_NEED_THERMO|PLASMA_NEED_B|PLASMA_NEED_E);
  add(p.te);add(p.ti);add(p.ne);add(p.ni);
  for(int d=0;d<3;d++) {add(p.b[d]);add(p.e[d]);}
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  int size;MPI_Comm_size(MPI_COMM_WORLD,&size);
  if(size!=1) MPI_Abort(MPI_COMM_WORLD,2);
  {
    SPARTA s(argc,argv,MPI_COMM_WORLD);s.input->file();
    const int idx=s.modify->find_fix("pd");
    if(idx<0) MPI_Abort(MPI_COMM_WORLD,3);
    auto *pd=dynamic_cast<FixBackground *>(s.modify->fix[idx]);
    double x[3]={2.1,0,-.7},v[3]={};
    s.particle->add_particle(1,0,0,x,v,0,0);
    long points=0;
    for(int t=0;t<pd->mesh_ntri;t++) {
      double r[3],z[3];
      for(int d=0;d<3;d++) {int iv=pd->mesh_tri[3*t+d];r[d]=pd->mesh_vtx_r[iv];z[d]=pd->mesh_vtx_z[iv];}
      for(int d=0;d<3;d++) {
        // Vertices and edge midpoints exercise the original hash tie order.
        x[0]=r[d];x[2]=z[d];point(pd,x);points++;
        x[0]=.5*(r[d]+r[(d+1)%3]);x[2]=.5*(z[d]+z[(d+1)%3]);point(pd,x);points++;
      }
      x[0]=(r[0]+r[1]+r[2])/3;x[2]=(z[0]+z[1]+z[2])/3;point(pd,x);points++;
    }
    const char *path=getenv("OE_FINGERPRINT_OUTPUT");
    FILE *fp=path?fopen(path,"w"):stdout;
    fprintf(fp,"points=%ld fnv1a64=%016llx\n",points,(unsigned long long)digest);
    if(fp!=stdout) fclose(fp);
  }
  MPI_Finalize();
}
