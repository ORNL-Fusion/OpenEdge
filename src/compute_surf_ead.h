/* ----------------------------------------------------------------------
   OpenEdge: per-surface Energy-Angle diagnostic
   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
     - 42d
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surf/ead,ComputeSurfEAD)

#else

#ifndef SPARTA_COMPUTE_SURF_EAD_H
#define SPARTA_COMPUTE_SURF_EAD_H

#include "compute.h"
#include "surf.h"
#include "hash3.h"

namespace SPARTA_NS {

class ComputeSurfEAD : public Compute {
 public:
  ComputeSurfEAD(class SPARTA *, int, char **);
  ~ComputeSurfEAD();
  void init();
  void compute_per_surf();
  void clear();
  void surf_tally(double, int, int, int, Particle::OnePart *,
                  Particle::OnePart *, Particle::OnePart *);
  int tallyinfo(surfint *&);
  void post_process_surf();
  void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix;
  int dim;
  int bins_e,bins_a,bins_total,ncols;
  double emax,amax,de,da;
  int q_sumw,q_sumn,q_hist;
  int idx_sumw,idx_sumn,idx_hist;

  int ntally;
  int maxtally;
  surfint *tally2surf;
  int maxsurf;
  int combined;

  int weightflag;
  double weight;

#ifdef SPARTA_MAP
  typedef std::map<surfint,int> MyHash;
#elif defined SPARTA_UNORDERED_MAP
  typedef std::unordered_map<surfint,int> MyHash;
#else
  typedef std::tr1::unordered_map<surfint,int> MyHash;
#endif

  MyHash *hash;

  Surf::Line *lines;
  Surf::Tri *tris;

  void grow_tally();
};

}

#endif
#endif
