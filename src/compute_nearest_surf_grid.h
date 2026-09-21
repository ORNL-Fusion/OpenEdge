/* ----------------------------------------------------------------------
   OpenEdge: nearest wall surface per grid cell.
   Outputs distance / surface index / outward normal for the closest
   member of a surface group, for each cell in a grid group. Static
   geometry — computed once and cached.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(nearest_surf/grid,ComputeNearestSurfGrid)

#else

#ifndef SPARTA_COMPUTE_NEAREST_SURF_GRID_H
#define SPARTA_COMPUTE_NEAREST_SURF_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeNearestSurfGrid : public Compute {
 public:
  ComputeNearestSurfGrid(class SPARTA *, int, char **);
  ~ComputeNearestSurfGrid();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 public:
  int *midx_grid;           // nearest-surface array index per cell
  int sgroupbit;            // surface group bitmask (for particle-level refinement)
  int nglocal;              // midx_grid extent; consumers must bounds-check
                            // cell indices that may be stale after a rebalance

 protected:
  int groupbit;
  int nvalue;
  int *value;
  int computed_once;           // 1 after first compute (static geometry cache)
  cellint stamp_id0_ = -1;     // first-cell id at last reallocate

  enum {DIST,SURFID,NX,NY,NZ,SURFIDX};
};

}

#endif
#endif

