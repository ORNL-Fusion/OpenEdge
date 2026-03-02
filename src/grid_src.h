/* ----------------------------------------------------------------------
    OpenEdge:
    Shared per-grid source descriptor used by multiple fixes
    (coll/nanbu, drag, viscous, droplet/charge).

    CollGridSrc maps a per-grid quantity to either:
      - a SPARTA compute  (c_ID[col])
      - a SPARTA variable (v_name)
------------------------------------------------------------------------- */

#ifndef SPARTA_GRID_SRC_H
#define SPARTA_GRID_SRC_H

namespace SPARTA_NS {

enum CollSrcKind { COLL_SRC_NONE, COLL_SRC_VAR, COLL_SRC_COMP };

struct CollGridSrc {
  CollSrcKind kind = COLL_SRC_NONE;
  // VAR path
  char *vname = nullptr;  int varid = -1;
  // COMP path
  char *cid   = nullptr;  int icompute = -1;  int col = 0; // 1-based

  // --- cache for per-timestep fast access ---
  double **arr_cache = nullptr; // c->array_grid
  int      src_index = -1;      // mapped column index
  int      cache_ts  = -1;      // update->ntimestep when filled
};

}  // namespace SPARTA_NS

#endif
