/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix coll/nanbu: binary Coulomb collisions via the Nanbu (1997) /
    Takizuka-Abe (1977) algorithm.  Coexists with collide vss for
    neutral-neutral collisions.

    Optional background mode: each charged simulation particle also
    collides with a virtual partner sampled from a prescribed
    Maxwellian background plasma (Ti, Ni, Vpar, B-field direction).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coll/nanbu,FixCollNanbu)

#else

#ifndef SPARTA_FIX_COLL_NANBU_H
#define SPARTA_FIX_COLL_NANBU_H

#include "fix.h"
#include "grid_src.h"
#include "nanbu_scatter_table.h"

namespace SPARTA_NS {

class RanKnuth;

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

  // binary particle-particle collisions (0 = off via nobinary keyword)
  int do_binary_;

  // plasma sources for Coulomb logarithm (Te, Ne)
  CollGridSrc srcTe_, srcNe_;

  // background collision mode
  int have_background_;
  double A_bg_, Z_bg_, m_bg_, q_bg_;
  CollGridSrc srcTi_bg_, srcNi_bg_, srcVpar_bg_;
  CollGridSrc srcBx_, srcBy_, srcBz_;

  // scratch particle list for one cell
  int npmax_;
  int *plist_;

  // helper methods
  void refresh_compute_src(CollGridSrc &S);
  double read_cell_src(const CollGridSrc &S, int icell);
  void parse_compute_src(const char *tok, CollGridSrc &dst, const char *label);

  // core Nanbu algorithm
  void nanbu_collisions_cell(int icell, int np);
  void nanbu_background_cell(int icell, int np);
  double compute_coulomb_log(double ne, double Te_eV);
  double box_muller();
};

}  // namespace SPARTA_NS

#endif
#endif
