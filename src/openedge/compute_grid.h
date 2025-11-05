/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(grid,ComputeGrid)

#else

#ifndef SPARTA_COMPUTE_GRID_H
#define SPARTA_COMPUTE_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeGrid : public Compute {
 public:
  ComputeGrid(class SPARTA *, int, char **);
  ~ComputeGrid();
  void init();
  virtual void compute_per_grid();
  virtual int query_tally_grid(int, double **&, int *&);
  virtual void post_process_grid(int, int, double **, int *, double *, int);
  virtual void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,nvalue,ngroup,tvib_flag;

  int *value;                // keyword for each user requested value
  int *unique;               // unique keywords for tally, len = npergroup
  int npergroup;             // # of unique tally quantities per group
  int cellcount,cellmass;    // 1 if total cell count/mass is tallied
  int ntotal;                // total # of columns in tally array
  int nglocal;               // # of owned grid cells

  int *nmap;                 // # of tally quantities each user value uses
  int **map;                 // which tally columns each output value uses
  double **tally;            // array of tally quantities, cells by ntotal

  double eprefactor;         // conversion from velocity^2 to energy
  double tprefactor;         // conversion from KE to temperature
  double rvprefactor;        // conversion from rot/vib E to temperature

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

W: Using compute grid tvib with fix vibmode may give incorrect temperature,
use compute tvib/grid instead

Self-explanatory.

E: Number of groups in compute grid mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Invalid call to ComputeGrid::post_process_grid()

This indicates a coding error.  Please report the issue to the SPARTA
developers.

*/
