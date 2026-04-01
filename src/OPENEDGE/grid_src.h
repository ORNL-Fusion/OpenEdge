/* ----------------------------------------------------------------------
    OpenEdge:
    Shared per-grid source descriptor used by multiple fixes
    (coll/nanbu, drag, viscous, droplet/charge).

    CollGridSrc maps a source quantity to either:
      - a SPARTA compute  (c_ID[col])
      - a SPARTA variable (v_name)
      - a particle custom vector (p_name)
------------------------------------------------------------------------- */

#ifndef SPARTA_GRID_SRC_H
#define SPARTA_GRID_SRC_H

namespace SPARTA_NS {

enum CollSrcKind { COLL_SRC_NONE, COLL_SRC_VAR, COLL_SRC_COMP, COLL_SRC_PCUSTOM };

struct CollGridSrc {
  CollSrcKind kind = COLL_SRC_NONE;
  // VAR path
  char *vname = nullptr;  int varid = -1;
  // COMP path
  char *cid   = nullptr;  int icompute = -1;  int col = 0; // 1-based
  // per-particle custom vector path
  char *pname = nullptr;  int ipcustom = -1;  int ipwhich = -1;

  // --- cache for per-timestep fast access ---
  double **arr_cache = nullptr; // c->array_grid
  double  *pvec_cache = nullptr; // particle->edvec[ipwhich]
  int      src_index = -1;      // mapped column index
  int      cache_ts  = -1;      // update->ntimestep when filled
};

}  // namespace SPARTA_NS

#endif
