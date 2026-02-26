/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix coll/nanbu: binary Coulomb collisions via the Nanbu (1997) /
    Takizuka-Abe (1977) algorithm.  Coexists with collide vss for
    neutral-neutral collisions.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coll/nanbu,FixCollNanbu)

#else

#ifndef SPARTA_FIX_COLL_NANBU_H
#define SPARTA_FIX_COLL_NANBU_H

#include "fix.h"
#include "nanbu_scatter_table.h"

namespace SPARTA_NS {

class RanKnuth;

/* ------------------------------------------------------------------
   Lightweight source descriptor for per-grid compute arrays.
   Duplicated here so that coll/nanbu is self-contained and does not
   depend on fix_coll_background.h.
------------------------------------------------------------------ */

#ifndef NANBU_GRID_SRC_DEFINED
#define NANBU_GRID_SRC_DEFINED
enum NanbuSrcKind { NANBU_SRC_NONE, NANBU_SRC_COMP };

struct NanbuGridSrc {
  NanbuSrcKind kind = NANBU_SRC_NONE;
  char *cid   = nullptr;
  int icompute = -1;
  int col      = 0;        // 1-based column in compute array

  // per-timestep cache
  double **arr_cache = nullptr;
  int      src_index = -1;
  int      cache_ts  = -1;
};
#endif

class FixCollNanbu : public Fix {
 public:
  FixCollNanbu(class SPARTA *, int, char **);
  ~FixCollNanbu();
  int  setmask();
  void init();
  void end_of_step();
  double memory_usage();

 private:
  NanbuScatterTable scatter_table_;
  RanKnuth *rng_;

  // plasma sources for Coulomb logarithm (Te, Ne)
  NanbuGridSrc srcTe_, srcNe_;

  // scratch particle list for one cell
  int npmax_;
  int *plist_;

  // helper methods
  void refresh_compute_src(NanbuGridSrc &S);
  double read_cell_src(const NanbuGridSrc &S, int icell);
  void parse_compute_src(const char *tok, NanbuGridSrc &dst, const char *label);

  // core Nanbu algorithm
  void nanbu_collisions_cell(int icell, int np);
  double compute_coulomb_log(double ne, double Te_eV);
};

}  // namespace SPARTA_NS

#endif
#endif
