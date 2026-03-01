/* ----------------------------------------------------------------------
   OpenEdge: nearest wall geometry per grid cell for sheath workflows
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(sheath/geometry/grid,ComputeSheathGeometryGrid)

#else

#ifndef SPARTA_COMPUTE_SHEATH_GEOMETRY_GRID_H
#define SPARTA_COMPUTE_SHEATH_GEOMETRY_GRID_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeSheathGeometryGrid : public Compute {
 public:
  ComputeSheathGeometryGrid(class SPARTA *, int, char **);
  ~ComputeSheathGeometryGrid();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 public:
  int *midx_grid;           // nearest-surface array index per cell
  int sgroupbit;            // surface group bitmask (for particle-level refinement)

 protected:
  int nglocal;
  int groupbit;
  int nvalue;
  int *value;
  int computed_once;           // 1 after first compute (static geometry cache)

  enum {DIST,SURFID,NX,NY,NZ,SURFIDX};
};

}

#endif
#endif

