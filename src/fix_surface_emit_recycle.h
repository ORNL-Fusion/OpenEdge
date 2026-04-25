/* ----------------------------------------------------------------------
    OpenEdge: fix surface/emit/recycle
    Wall-recycling neutral source emitter.

    Per wall segment, queries (ne, Te, Ti) at the adjacent SOLPS sheath-edge
    plasma cell (cached at init) via a fix background, computes the Bohm
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

FixStyle(surface/emit/recycle,FixSurfaceEmitRecycle)

#else

#ifndef SPARTA_FIX_SURFACE_EMIT_RECYCLE_H
#define SPARTA_FIX_SURFACE_EMIT_RECYCLE_H

#include "fix_emit.h"
#include "surf.h"
#include "grid.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class FixSurfaceEmitRecycle : public FixEmit {
 public:
  FixSurfaceEmitRecycle(class SPARTA *, int, char **);
  ~FixSurfaceEmitRecycle();
  void init();

  void grid_changed();
  void custom_surf_changed() { grid_changed(); }

 private:
  int imix, groupbit;

  // plasma data source
  int ifix_plasma;
  class FixBackground *plasma;

  // recycling parameters
  double mass_amu;       // main ion mass [amu], for c_s
  double R_recycle;      // total recycling coefficient (1 - pumping frac)
  double twall;          // default emission temperature [K]
  // Optional per-species overrides. Parsed from `twall_species <sp> <T> ...`
  // in options(); resolved to mixture-local species indices in init() and
  // materialised as twall_by_species (size = mixture nspecies, falling back
  // to the scalar twall for any species not explicitly named).
  std::vector<std::string> twall_species_names;
  std::vector<double>      twall_species_values;
  std::vector<double>      twall_by_species;

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
  int diag_printed;          // 1 once init diagnostic has been printed,
                             // suppresses re-print on every rebalance.

  // Track pd->generation so tasks[].plasma_cell and the cell-centroid
  // cache below are rebuilt when plasma.h5 reloads (dynamic plasma, SOLPS
  // coupling). -1 means "no cache built yet".
  int plasma_generation_;

  // Instance-scoped cache for the B2-cell centroid fallback search in
  // create_task. Replaces the old static locals so (a) a reload bumps
  // plasma_generation_ and forces a rebuild, and (b) multiple fix
  // emit/surf/recycle instances can coexist without sharing state.
  std::vector<double> plasma_cell_r_;
  std::vector<double> plasma_cell_z_;
  int plasma_cell_centroid_gen_;  // pd generation when cell_r/z was built

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
