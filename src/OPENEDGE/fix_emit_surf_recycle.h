/* ----------------------------------------------------------------------
    OpenEdge: fix emit/surf/recycle
    Wall-recycling neutral source emitter.

    Per wall segment, queries (ne, Te, Ti) at the adjacent SOLPS sheath-edge
    plasma cell (cached at init) via a fix plasma/data, computes the Bohm
    wall flux
       Gamma = n_i * c_s * sin(alpha_B)         c_s = sqrt((Te+Ti)/m_ion)
    and emits the mixture at rate
       dot{N} = 0.5 * R * Gamma * area
    where R is the total recycling coefficient (1 - pumping fraction).
    The factor 1/2 mass-balances D+ -> D2 recombination at the wall.
    Mixture fractions control the atom/molecule split.

    Stage 1: thermal-only (Maxwellian flux at twall along inward normal).
             No TRIM fast-reflection channel yet.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(emit/surf/recycle,FixEmitSurfRecycle)

#else

#ifndef SPARTA_FIX_EMIT_SURF_RECYCLE_H
#define SPARTA_FIX_EMIT_SURF_RECYCLE_H

#include "fix_emit.h"
#include "surf.h"
#include "grid.h"

namespace SPARTA_NS {

class FixEmitSurfRecycle : public FixEmit {
 public:
  FixEmitSurfRecycle(class SPARTA *, int, char **);
  ~FixEmitSurfRecycle();
  void init();

  void grid_changed();
  void custom_surf_changed() { grid_changed(); }

 private:
  int imix, groupbit;

  // plasma data source
  int ifix_plasma;
  class FixPlasmaData *plasma;

  // recycling parameters
  double mass_amu;       // main ion mass [amu], for c_s
  double R_recycle;      // total recycling coefficient (1 - pumping frac)
  double twall;          // wall temperature [K]

  // copies of data from other classes
  int dimension, nspecies;
  double fnum;
  double *fraction, *cummulative;

  class Cut2d *cut2d;
  class Cut3d *cut3d;

  struct Task {
    double area;                // surf/cell overlap area
    double ntarget;             // # of mols to insert this step (all species)
    double tan1[3], tan2[3];    // tangent vectors
    double vscale_molec;        // sqrt(2 k twall / m) for Maxwellian flux
    double *path;               // overlap polygon
    double *fracarea;           // per-sub-tri fractional area
    int    icell;
    surfint isurf;
    int    pcell;
    int    npoint;
    double rmid, zmid;          // segment midpoint (for plasma query)
    double inward[3];           // unit inward normal (flip of outward)
    int    plasma_cell;         // cached SOLPS cell index (-1 if none)
    double area_share;           // task area / (sum of task areas mapped
                                 // to the same plasma_cell), for
                                 // area-weighted distribution of the
                                 // per-cell B2 wall face area budget.
  };

  Task *tasks;
  int ntaskmax;

 protected:
  virtual void perform_task();

 private:
  void create_task(int);
  void grow_task();
  int  option(int, char **);
  double emission_rate_per_surface(int itask);
};

}

#endif
#endif
