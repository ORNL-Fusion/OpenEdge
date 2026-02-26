/* ----------------------------------------------------------------------
   OpenEdge: aggregate Ion Energy-Angle Distribution (IEAD)
   Accumulates a single 2-D histogram (energy × angle) across all
   surfaces in a group.  Output is a global array (bins_a × bins_e)
   suitable for fix ave/time.
   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(iead,ComputeIEAD)

#else

#ifndef SPARTA_COMPUTE_IEAD_H
#define SPARTA_COMPUTE_IEAD_H

#include "compute.h"
#include "surf.h"

namespace SPARTA_NS {

class ComputeIEAD : public Compute {
 public:
  ComputeIEAD(class SPARTA *, int, char **);
  ~ComputeIEAD();
  void init();
  void compute_array();
  void clear();
  void surf_tally(double, int, int, int, Particle::OnePart *,
                  Particle::OnePart *, Particle::OnePart *);
  int tallyinfo(surfint *&);
  int matchstep(bigint) { return 1; }   // always tally, every timestep
  bigint memory_usage();

 protected:
  int groupbit,imix;
  int dim;
  int bins_e,bins_a;
  double emax,amax,de,da;

  double *hist_local;     // local rank accumulator [bins_a * bins_e]
  double *hist_global;    // MPI-reduced result     [bins_a * bins_e]
  double **hist_2d;       // pointer into hist_global for array_array interface

  bigint last_reduced;    // timestep of last MPI_Allreduce

  Surf::Line *lines;
  Surf::Tri *tris;

  int weightflag;
};

}

#endif
#endif
