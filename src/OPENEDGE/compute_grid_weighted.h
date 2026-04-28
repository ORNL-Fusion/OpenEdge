/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(grid/weighted,ComputeGridWeighted)

#else

#ifndef SPARTA_COMPUTE_GRID_WEIGHTED_H
#define SPARTA_COMPUTE_GRID_WEIGHTED_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeGridWeighted : public Compute {
 public:
  ComputeGridWeighted(class SPARTA *, int, char **);
  ~ComputeGridWeighted();
  void init();
  void compute_per_grid();
  int query_tally_grid(int, double **&, int *&);
  void post_process_grid(int, int, double **, int *, double *, int);
  void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,nvalue,ngroup;

  int *value;                // keyword for each user requested value
  int *unique;               // unique keywords for tally
  int npergroup;             // # of unique tally quantities per group
  int ntotal;                // total # of columns in tally array
  int nglocal;               // # of owned grid cells

  int *nmap;                 // # of tally quantities each user value uses
  int **map;                 // which tally columns each output value uses
  double **tally;            // array of tally quantities

  int pweight_index;         // index of pweight custom attribute
  int pweight_ewhich;        // index into edvec

  int cellcount_w,cellmass_w; // 1 if total cell weight/mass is tallied
                              // (for nfrac_w / massfrac_w)

  double eprefactor;         // velocity^2 to energy
  double tprefactor;         // KE to temperature
  double rvprefactor;        // rot/vib energy to temperature

  void set_map(int, int);
  void reset_map();
};

}

#endif
#endif
