/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(eflux/grid,ComputeEFluxGrid)

#else

#ifndef SPARTA_COMPUTE_EFLUX_GRID_H
#define SPARTA_COMPUTE_EFLUX_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeEFluxGrid : public Compute {
 public:
  ComputeEFluxGrid(class SPARTA *, int, char **);
  ~ComputeEFluxGrid();
  void init();
  void compute_per_grid();
  int query_tally_grid(int, double **&, int *&);
  void post_process_grid(int, int, double **, int *, double *, int);
  void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,nvalue,ngroup;

  int *value;                // keyword for each user requested value
  int *unique;               // unique keywords for tally, len = npergroup
  int npergroup;             // # of unique tally quantities per group
  int ntotal;                // total # of columns in tally array
  int nglocal;               // # of owned grid cells

  int *nmap;                 // # of tally quantities each user value uses
  int **map;                 // which tally columns each output value uses
  double **tally;            // array of tally quantities, cells by ntotal

  void set_map(int, int);
  void reset_map();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Compute grid mixture ID does not exist

Self-explanatory.

E: Number of groups in compute grid mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Invalid call to ComputeGrid::post_process_grid()

This indicates a coding error.  Please report the issue to the SPARTA
developers.

*/
