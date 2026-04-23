/* ----------------------------------------------------------------------
    OpenEdge: fix emit/surf/puff
    Cold-gas puff emission from a surface group (EIRENE-style injection).
    No PMI compute is required; each timestep injects `n` particles per
    rank distributed across active surf-cell tasks proportional to surf
    area, using a cosine-law angular distribution from the surface normal
    and Maxwellian velocity at the mixture thermal temperature.

    Derived from fix_surface_emit_sputter but stripped of:
      iflux / flux_index / flux_thresh / source_thresh
      nlaunch / nlaunch_total modes (pweight-weighted launches)
      ComputePMISurfData dependency
    Use fix_surface_emit_sputter when your emission is driven by a per-surf PMI
    flux compute; use this fix when you want a fixed puff rate (in particles
    per step) from a predefined puff surface.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(emit/surf/puff,FixEmitSurfPuff)

#else

#ifndef SPARTA_FIX_EMIT_SURF_PUFF_H
#define SPARTA_FIX_EMIT_SURF_PUFF_H

#include "fix_emit.h"
#include "surf.h"
#include "grid.h"

namespace SPARTA_NS {

class FixEmitSurfPuff : public FixEmit {
 public:
  FixEmitSurfPuff(class SPARTA *, int, char **);
  ~FixEmitSurfPuff();
  void init();

  void grid_changed();
  void custom_surf_changed() { grid_changed(); }

 private:
  int imix, groupbit, normalflag;

  // Per-step emission count (CONSTANT mode). `n 0` means FLOW mode:
  // use mixture nrho / temp and surf-inflow formula, same as plain emit/surf.
  int npmode, np;   // npmode = FLOW, CONSTANT
  char *npstr;      // unused; reserved for future VARIABLE mode

  // Optional emit-count cap: stops emitting exactly after `stop_at_np`
  // particles have been launched (summed across ranks and steps). Uses the
  // cumulative `ntotal` from FixEmit, NOT the live nlocal, so chemistry
  // byproducts (dissoc daughters etc) do not falsely inflate the cap.
  // 0 = disabled (continuous emission). Latches once hit (stop_latched = 1)
  // so the fix never re-emits after the batch is complete. Per-task ninsert
  // is clamped against the remaining global quota so final count = N_cap
  // within +/- (nprocs - 1) particles (and exactly N for single-rank emit).
  //
  // This is the "EIRENE batch" semantics: N = number of MC trajectories.
  // If you care about current-Np instead, wrap the fix with a SPARTA
  // variable+jump loop in the input deck.
  bigint stop_at_np;
  int    stop_latched;

  // copies of data from other classes

  int dimension, nspecies;
  double fnum;
  double nrho, temp_thermal, temp_rot, temp_vib;
  double *fraction, *cummulative;

  class Cut2d *cut2d;
  class Cut3d *cut3d;

  // one insertion task for a cell / surf pair

  struct Task {
    double area;                // area of overlap of surf with cell
    double ntarget;             // # of mols to insert for all species
    double tan1[3], tan2[3];    // 2 normalized tangent vectors to surf normal
    double nrho;                // from mixture
    double temp_thermal;        // from mixture
    double temp_rot;            // from mixture
    double temp_vib;            // from mixture
    double magvstream;          // from mixture
    double vstream[3];          // from mixture

    double *ntargetsp;          // # of mols to insert for each species
                                //   only defined for PERSPECIES
    double *vscale;             // vscale per species
    double *path;               // path of points for overlap of surf with cell
    double *fracarea;           // fractional area for each sub-tri in path

    int icell;                  // associated cell index, unsplit or split cell
    surfint isurf;              // surf index, sometimes a surf ID
    int pcell;                  // associated cell index for particles
                                //   unsplit or sub cell (not split cell)
    int npoint;                 // # of points in path
  };

                         // ntask = # of tasks is stored by parent class
  Task *tasks;           // list of particle insertion tasks
  int ntaskmax;          // max # of tasks allocated

  double magvstream;       // magnitude of mixture vstream
  double norm_vstream[3];  // direction of mixture vstream

 protected:
  virtual void perform_task();

 private:
  void create_task(int);
  void grow_task();
  int option(int, char **);
};

}  // namespace SPARTA_NS

#endif
#endif
