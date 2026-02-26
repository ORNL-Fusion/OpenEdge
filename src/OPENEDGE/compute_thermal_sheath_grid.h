/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(thermal/sheath/grid,ComputeThermalSheathGrid)

#else

#ifndef SPARTA_COMPUTE_THERMAL_SHEATH_GRID_H
#define SPARTA_COMPUTE_THERMAL_SHEATH_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeThermalSheathGrid : public Compute {
 public:
  ComputeThermalSheathGrid(class SPARTA *, int, char **);
  ~ComputeThermalSheathGrid();
  void init();
  virtual void compute_per_grid();
  virtual int query_tally_grid(int, double **&, int *&);
  virtual void post_process_grid(int, int, double **, int *, double *, int);
  virtual void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,nvalue;
  int ngroup;

  int *value;                // keyword for each user requested value
  int npergroup;             // # of unique tally quantities per group
  int ntotal;                // total # of columns in tally array
  int nglocal;               // # of owned grid cells

  int *nmap;                 // # of tally quantities each user value uses
  int **map;                 // which tally columns each output value uses
  double **tally;            // array of tally quantities, cells by ntotal

  double tprefactor;         // conversion from KE to temperature
  double pprefactor;         // conversion from KE to pressure
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Compute sonine/grid mixture ID does not exist

Self-explanatory.

E: Number of groups in compute sonine/grid mixture has changed

This mixture property cannot be changed after this compute command is
issued.

E: Invalid call to ComputeSonineGrid::post_process_grid()

This indicates a coding error.  Please report the issue to the SPARTA
developers.

*/
