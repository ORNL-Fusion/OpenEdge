/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(emit/surf/pmi,FixEmitSurfPmi)

#else

#ifndef SPARTA_FIX_EMIT_SURF_PMI_H
#define SPARTA_FIX_EMIT_SURF_PMI_H

#include "fix_emit.h"
#include "surf.h"
#include "grid.h"
#include <vector>

namespace SPARTA_NS {

class FixEmitSurfPmi : public FixEmit {
 public:
  FixEmitSurfPmi(class SPARTA *, int, char **);
  ~FixEmitSurfPmi();
  void init();

  void grid_changed();
  void custom_surf_changed() { grid_changed(); }

 private:
  int imix,groupbit,normalflag;

  int npmode,np;    // npmode = FLOW,CONSTANT
  char *npstr;
  int iflux,flux_index;

  int nlaunch_mode;           // 1 if per-task nlaunch weighted mode is active
  int nlaunch_per_surf;       // # of particles to launch per emitting task
  int nlaunch_total_mode;     // 1 if global-budget weighted mode is active
  int nlaunch_total;          // expected total # of particles launched per step
  double flux_thresh;         // minimum flux to emit (skip tasks below this)
  double source_thresh;       // minimum source strength flux*area*dt to emit
  int pweight_index;          // index of pweight custom attribute (-1 if none)
  int pweight_ewhich;         // index into edvec for pweight

  // copies of data from other classes

  int dimension,nspecies;
  double fnum;
  double nrho,temp_thermal,temp_rot,temp_vib;
  double *fraction,*cummulative;

  class Cut2d *cut2d;
  class Cut3d *cut3d;

  // one insertion task for a cell and a surf

  struct Task {
    double area;                // area of overlap of surf with cell
    double ntarget;             // # of mols to insert for all species
    double tan1[3],tan2[3];     // 2 normalized tangent vectors to surf normal
    double nrho;                // from mixture or adjacent subsonic cell
    double temp_thermal;        // from mixture or adjacent subsonic cell
    double temp_rot;            // from mixture or subsonic temp_thermal
    double temp_vib;            // from mixture or subsonic temp_thermal
    double magvstream;          // from mixture
    double vstream[3];          // from mixture or adjacent subsonic cell
    double *ntargetsp;          // # of mols to insert for each species,
                                //   only defined for PERSPECIES
    double *vscale;             // vscale for each species,
                                //   only defined for subsonic_style PONLY

    double *path;               // path of points for overlap of surf with cell
    double *fracarea;           // fractional area for each sub tri in path

    int icell;                  // associated cell index, unsplit or split cell
    surfint isurf;              // surf index, sometimes a surf ID
    int pcell;                  // associated cell index for particles
                                // unsplit or sub cell (not split cell)
    int npoint;                 // # of points in path
  };

                         // ntask = # of tasks is stored by parent class
  Task *tasks;           // list of particle insertion tasks
  int ntaskmax;          // max # of tasks allocated

  double magvstream;       // magnitude of mixture vstream
  double norm_vstream[3];  // direction of mixture vstream

  // Cached per-task source strengths for nlaunch_total mode.
  // When the upstream pmi/surf/data compute is in static_cache mode,
  // task_source[] and source_total never change, so we build them once
  // and reuse them across timesteps.
  std::vector<double> cached_task_source;
  double cached_source_total;
  int task_source_cached;        // 0 = stale, 1 = valid

  // custom options for per-surf emission properties

 protected:
  virtual void perform_task();

 private:
  void create_task(int);
  void grow_task();
  int local_isurf_index(surfint) const;
  double flux_for_surface(surfint);

  int option(int, char **);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

E: Fix emit/face mixture ID does not exist

Self-explanatory.

E: Cannot use fix emit/face in z dimension for 2d simulation

Self-explanatory.

E: Cannot use fix emit/face in y dimension for axisymmetric

This is because the y dimension boundaries cannot be
emit/face boundaries for an axisymmetric model.

E: Cannot use fix emit/face n > 0 with perspecies yes

This is because the perspecies option calculates the
number of particles to insert itself.

E: Cannot use fix emit/face on periodic boundary

Self-explanatory.

E: Fix emit/face used on outflow boundary

Self-explanatory.

*/
