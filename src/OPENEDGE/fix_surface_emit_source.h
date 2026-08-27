/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(surface/emit/source,FixSurfaceEmitSource)

#else

#ifndef SPARTA_FIX_SURFACE_EMIT_SOURCE_H
#define SPARTA_FIX_SURFACE_EMIT_SOURCE_H

#include "fix_emit.h"
#include "surf.h"
#include "grid.h"
#include <vector>

namespace SPARTA_NS {

class FixSurfaceEmitSource : public FixEmit {
 public:
  FixSurfaceEmitSource(class SPARTA *, int, char **);
  ~FixSurfaceEmitSource();
  void init();

  void grid_changed();
  void custom_surf_changed() { grid_changed(); }

 public:
  enum EmitModel { MODEL_THERMAL, MODEL_THERMAL_TSURF, MODEL_THOMPSON,
                   MODEL_FIXED_ENERGY };

 private:
  int imix,groupbit,normalflag;

  int npmode,np;    // npmode = FLOW,CONSTANT
  char *npstr;
  int iflux,flux_index;

  // outgoing-particle sampling model
  int emit_model;            // one of EmitModel
  double model_Ub;           // surface binding energy [eV] (thompson)
  double model_cos_n;        // angular cos^n exponent (thompson; default 1)
  double model_Emax;         // Thompson high-E cutoff [eV] (0 = unbounded)
  double model_E_fixed;      // fixed emission energy [eV] (fixed_energy)
  char *model_tsurf_name;    // custom-attribute name (thermal_tsurf)
  int model_tsurf_index;     // surf->find_custom(...) index (resolved in init)

  int nlaunch_mode;           // 1 if per-task nlaunch weighted mode is active
  int nlaunch_per_surf;       // # of particles to launch per emitting task
  int nlaunch_total_mode;     // 1 if global-budget weighted mode is active
  int nlaunch_total;          // expected total # of particles launched per step
  double flux_thresh;         // minimum flux to emit (skip tasks below this)
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
  // When the upstream surface/physical/sputter compute is in static_cache mode,
  // task_source[] and source_total never change, so we build them once
  // and reuse them across timesteps.
  std::vector<double> cached_task_source;
  double cached_source_total;
  int task_source_cached;        // 0 = stale, 1 = valid

  // Per-surface flux replicated across ranks (canonical SPARTA pattern: the
  // upstream compute writes array_surf only at this rank's owned surfaces, so
  // we spread owned->local+ghost via Surf::spread_own2local() before the task
  // loop reads it. Without this, a task's cell owner and surface owner can
  // disagree under MPI and flux_for_surface() returns 0.)
  double *flux_localghost;
  int     flux_n_localghost;
  int     flux_spread_valid;   // 1 = spread flux is current (static upstream)
  std::vector<double> flux_owned_buf;

  // optional per-surf concentration multiplier on the flux (mixed layers):
  // conc_scale <custom-array-name> <col>
  char *conc_attr;
  int conc_col, conc_index;

  // custom options for per-surf emission properties

  // ---- File-driven flux mode (alternative to compute-driven) ----
  // When file_mode = 1, per-surf nrho/vstream/temp_thermal come from a
  // SPARTA inflow-file (IMESH/JMESH/VALUES) bilinear-interpolated at each
  // surf centroid. The file format is the same as fix emit/face/file:
  //   <SECTION>             # e.g. ZLO  (declares which face axes are i,j)
  //   NIJ Ni Nj
  //   NV Nv
  //   VALUES name1 name2 ...   # any of: nrho, vx, vy, vz, temp, gamma
  //   IMESH x_0 ... x_{Ni-1}
  //   JMESH y_0 ... y_{Nj-1}
  //
  //   i j v1 v2 ... vNv
  //   ...
  // Section name picks the axis pair: ZLO/ZHI -> (xc,yc), XLO/XHI -> (yc,zc),
  // YLO/YHI -> (xc,zc). gamma column is read but ignored (redundant with
  // nrho * vz). Set face_axis_idx accordingly in load_file().
  int  file_mode;
  int  const_mode;               // 1 if uniform constant flux mode is active
  double const_flux;             // particles/m^2/s applied uniformly to group
  char *file_path;
  char *file_section;
  int  face_axis_idx;            // 0,1,2 of (X,Y,Z); selects (i,j) axes
  struct InflowMesh {
    int ni, nj, nvalues;
    std::vector<double> imesh, jmesh;        // (ni), (nj)
    std::vector<int>    which;               // (nvalues) -- IM_* enum values
    std::vector<double> values;              // (ni*nj*nvalues), [j*ni+i]*nv+m
    double lo[2], hi[2];
  } fmesh;
  void file_load();
  void file_bcast();
  void file_lookup(const double *xyz, double &nrho_out,
                   double *vstream_out, double &T_out) const;

 protected:
  virtual void perform_task();

 private:
  void create_task(int);
  void grow_task();
  double flux_for_surface(surfint);
  void spread_flux(class Compute *);

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
