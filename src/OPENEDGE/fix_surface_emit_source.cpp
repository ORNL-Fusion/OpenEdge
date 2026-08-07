/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_surface_emit_source.h"
#include "update.h"
#include "compute.h"
#include "domain.h"
#include "region.h"
#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "modify.h"
#include "cut2d.h"
#include "cut3d.h"
#include "input.h"
#include "comm.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "compute_surface_physical_sputter.h"

#include <cmath>
#include <vector>

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};
enum{FLOW,CONSTANT};
enum{INT,DOUBLE};                       // type tag for Surf::spread_own2local
// File-mode column kinds; matches the layout in wip/fix_emit_face_file.cpp
// minus species fractions (we only consume scalar inflow fields).
enum IM_Kind { IM_NRHO = 1, IM_VX, IM_VY, IM_VZ, IM_TEMP_THERMAL, IM_GAMMA };

#define DELTATASK 256
#define MAXLINE 16384

namespace {
  // Boltzmann constant in eV/K
  constexpr double KB_EV_PER_K = 8.617333262e-5;
  // 1 eV in joules
  constexpr double EV_TO_J = 1.602176634e-19;

  // Thompson sputtered-atom energy distribution, truncated form of
  // Mellet PPCF 59 (2017) eq. (10) / Guterl NME 27 (2021) eq. (2.2):
  //   f(E) ~ E/(E+U_b)^3 * [1 - sqrt((E+U_b)/(Emax+U_b))],  E in [0,Emax]
  // U_b is the SURFACE BINDING energy (W: 8.68 eV; spectrum peaks at
  // U_b/2). Emax is the kinematic cutoff (Guterl: ~2*Te for low-Z
  // projectiles; Mellet/Mousel: gamma*<E_imp>*((m1+2m2)/(2m1+m2))^6).
  // Emax<=0 recovers the classic unbounded Thompson, whose ~1/E^2 tail
  // puts ~1e-4 of atoms above 100*U_b (keV-scale ballistic flyers) --
  // production decks should set emax.
  // Sampling: f(E) = 2*E*U_b/(E+U_b)^3 has CDF F(E) = (E/(E+U_b))^2,
  // so the exact inverse transform is E = U_b*sqrt(v)/(1-sqrt(v)) with
  // v ~ U(0,1) (v ~ U(0,vmax), vmax = (Emax/(Emax+U_b))^2 for the
  // sharp-truncated proposal). NOTE: the previous transform
  // E = U_b*(1/sqrt(u)-1) inverted 1-F of f ~ 1/(E+U_b)^3 -- it
  // dropped the leading factor E and sampled a much softer spectrum
  // (median 0.41*U_b instead of 2.41*U_b). Rejection weight
  // 1-sqrt((E+U_b)/(Emax+U_b)) is the Mellet smooth-cutoff factor.
  inline double sample_thompson_energy(double Ub_eV, double Emax_eV,
                                       RanKnuth *rng) {
    if (Emax_eV <= 0.0) {
      double v = rng->uniform();
      if (v > 1.0 - 1.0e-12) v = 1.0 - 1.0e-12;   // tail guard
      const double s = std::sqrt(v);
      return Ub_eV * s / (1.0 - s);
    }
    const double rmax = Emax_eV / (Emax_eV + Ub_eV);
    const double vmax = rmax * rmax;
    for (int it = 0; it < 1000; it++) {
      const double s = std::sqrt(rng->uniform() * vmax);
      const double E = Ub_eV * s / (1.0 - s);
      const double w = 1.0 - std::sqrt((E + Ub_eV) / (Emax_eV + Ub_eV));
      if (rng->uniform() < w) return E;
    }
    return 0.5 * Ub_eV;                   // unreachable in practice
  }

  // Thermal flux energy: cos-weighted Maxwell flux through a surface gives
  // f(E) = (E/(kT)^2) exp(-E/kT) — Gamma(2,kT) — sampled as
  //   E = -kT · ln(R1·R2)
  inline double sample_thermal_flux_energy(double T_K, RanKnuth *rng) {
    double r1 = rng->uniform(), r2 = rng->uniform();
    if (r1 <= 0.0) r1 = 1.0e-12;
    if (r2 <= 0.0) r2 = 1.0e-12;
    return -KB_EV_PER_K * T_K * std::log(r1 * r2);
  }

  // cos^n angular sampling: cos(theta) = xi^(1/(n+1)),  xi ~ U(0,1].
  // Returns cos_th, sin_th by reference. n=1 is Knudsen cosine.
  inline void sample_cos_n(double n, RanKnuth *rng,
                           double &cos_th, double &sin_th) {
    double xi = rng->uniform();
    if (xi <= 0.0) xi = 1.0e-12;
    cos_th = std::pow(xi, 1.0 / (n + 1.0));
    double s2 = 1.0 - cos_th * cos_th;
    sin_th = (s2 > 0.0) ? std::sqrt(s2) : 0.0;
  }
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitSource::FixSurfaceEmitSource(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  // Emission tasks cache per-grid pcell indices; gridmigrate=1 puts this
  // fix in Modify::list_pergrid so Grid::notify_changed() invokes
  // grid_changed() -> create_tasks() after mid-run balance/adapt
  // migrations. Without it, stale task pcell values made post-balance
  // emissions insert particles with old (wrong or ghost/out-of-range)
  // cell indices -> silent mis-binning and downstream segfaults.
  gridmigrate = 1;

  if (narg < 6) error->all(FLERR,"Illegal fix surface/emit/source command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix surface/emit/source mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix surface/emit/source group ID does not exist");
  groupbit = surf->bitmask[igroup];

  // arg[4] is either a compute ID (default, compute-driven flux) or the
  // literal keyword "file" (file-driven flux from an inflow .dat).
  file_mode = 0;
  const_mode = 0;
  const_flux = 0.0;
  file_path = nullptr;
  file_section = nullptr;
  face_axis_idx = 2;        // ZLO/ZHI default
  iflux = -1;
  flux_index = 0;
  int iarg = 5;
  if (strcmp(arg[4],"constant") == 0) {
    const_mode = 1;
    if (narg < 6)
      error->all(FLERR,"Fix surface/emit/source: 'constant' needs a flux value");
    const_flux = atof(arg[5]);
    if (!std::isfinite(const_flux) || const_flux <= 0.0)
      error->all(FLERR,"Fix surface/emit/source: 'constant' flux must be > 0");
    iarg = 6;
  } else if (strcmp(arg[4],"file") == 0) {
    file_mode = 1;
    if (narg < 6)
      error->all(FLERR,"Fix surface/emit/source: 'file' needs a path");
    int n = strlen(arg[5]) + 1;
    file_path = new char[n];
    memcpy(file_path, arg[5], n);
    iarg = 6;
    // optional 'section <NAME>' (default ZLO)
    if (iarg + 1 < narg && strcmp(arg[iarg],"section") == 0) {
      int sn = strlen(arg[iarg+1]) + 1;
      file_section = new char[sn];
      memcpy(file_section, arg[iarg+1], sn);
      iarg += 2;
    } else {
      file_section = new char[4];
      memcpy(file_section, "ZLO", 4);
    }
  } else {
    iflux = modify->find_compute(arg[4]);
    if (iflux < 0)
      error->all(FLERR,"Fix surface/emit/source compute ID does not exist");
    if (strcmp(arg[iarg],"flux_index") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR,"Illegal fix surface/emit/source command");
      flux_index = atoi(arg[iarg+1]);
      if (flux_index <= 0)
        error->all(FLERR,"Fix surface/emit/source flux_index must be >= 1");
      iarg += 2;
    }
  }

  np = 0;
  npmode = FLOW;
  npstr = NULL;
  normalflag = 0;
  nlaunch_mode = 0;
  nlaunch_per_surf = 0;
  nlaunch_total_mode = 0;
  nlaunch_total = 0;
  flux_thresh = 0.0;
  pweight_index = -1;
  pweight_ewhich = -1;
  conc_attr = NULL;
  conc_col = 0;
  conc_index = -1;

  emit_model = MODEL_THERMAL;
  model_Ub = 0.0;
  model_Emax = 0.0;
  model_cos_n = 1.0;
  model_E_fixed = 0.0;
  model_tsurf_name = NULL;
  model_tsurf_index = -1;

  options(narg-iarg,&arg[iarg]);

  if (!surf->exist)
    error->all(FLERR,"Fix surface/emit/source requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix surface/emit/source not allowed for implicit surfaces");
  if (npmode == CONSTANT && perspecies)
    error->all(FLERR,"Cannot use fix surface/emit/source n > 0 with perspecies yes");
  if (nlaunch_total_mode && perspecies)
    error->all(FLERR,
      "Cannot use fix surface/emit/source nlaunch_total with perspecies yes");
  if (npmode == CONSTANT && nlaunch_total_mode)
    error->all(FLERR,
      "Cannot use fix surface/emit/source nlaunch_total with n > 0");

  tasks = NULL;
  ntask = ntaskmax = 0;

  cached_source_total = 0.0;
  task_source_cached = 0;

  flux_localghost = NULL;
  flux_n_localghost = 0;

  dimension = domain->dimension;
  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta,domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitSource::~FixSurfaceEmitSource()
{
  if (copymode) return;

  delete [] npstr;
  delete [] model_tsurf_name;
  delete [] file_path;
  delete [] file_section;

  for (int i = 0; i < ntaskmax; i++) {
    delete [] tasks[i].ntargetsp;
    delete [] tasks[i].vscale;
    delete [] tasks[i].path;
    delete [] tasks[i].fracarea;
  }
  memory->sfree(tasks);

  memory->destroy(flux_localghost);

  if (dimension == 3) delete cut3d;
  else delete cut2d;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSource::init()
{
  FixEmit::init();

  // bind the optional concentration-multiplier custom array (created by
  // surf_react surface/pwi at its init; both run before the first step)
  if (conc_attr) {
    conc_index = surf->find_custom(conc_attr);
    if (conc_index < 0)
      error->all(FLERR,"Fix surface/emit/source conc_scale: custom attribute not found");
    if (surf->etype[conc_index] != 1)
      error->all(FLERR,"Fix surface/emit/source conc_scale: attribute must be DOUBLE");
    if (surf->esize[conc_index] < conc_col)
      error->all(FLERR,"Fix surface/emit/source conc_scale: column out of range");
  }

  if (file_mode) {
    // File mode: no compute, no flux_index, no nlaunch* paths.
    if (nlaunch_mode || nlaunch_total_mode)
      error->all(FLERR,"Fix surface/emit/source: file mode is incompatible "
                       "with nlaunch / nlaunch_total");
    if (fmesh.values.empty()) {
      if (comm->me == 0) file_load();
      file_bcast();
      // resolve face_axis_idx from section name
      const char *s = file_section ? file_section : "ZLO";
      if      (!strcmp(s,"XLO") || !strcmp(s,"XHI")) face_axis_idx = 0;
      else if (!strcmp(s,"YLO") || !strcmp(s,"YHI")) face_axis_idx = 1;
      else if (!strcmp(s,"ZLO") || !strcmp(s,"ZHI")) face_axis_idx = 2;
      else error->all(FLERR,"Fix surface/emit/source: section must be one of "
                            "XLO/XHI/YLO/YHI/ZLO/ZHI");
    }
  } else if (const_mode) {
    // Constant-flux mode: no compute, uniform per-surf flux from input deck.
  } else {
    if (iflux < 0 || iflux >= modify->ncompute)
      error->all(FLERR,"Fix surface/emit/source compute ID no longer exists");
    Compute *c = modify->compute[iflux];
    if (!c->per_surf_flag)
      error->all(FLERR,"Fix surface/emit/source compute must provide per-surf values");
  }

  fnum = update->fnum;

  // look up pweight custom attribute if any weighted launch mode is active
  if (nlaunch_mode || nlaunch_total_mode) {
    pweight_index = particle->find_custom((char *) "pweight");
    if (pweight_index < 0)
      error->all(FLERR,
        "Fix surface/emit/source weighted launch modes require fix particle/weight");
    pweight_ewhich = particle->ewhich[pweight_index];
  }

  nspecies = particle->mixture[imix]->nspecies;
  fraction = particle->mixture[imix]->fraction;
  cummulative = particle->mixture[imix]->cummulative;

  // magvstream = magnitude of mixture vstream vector
  // norm_vstream = unit vector in stream direction
  double *vstream = particle->mixture[imix]->vstream;
  magvstream = MathExtra::len3(vstream);
  norm_vstream[0] = vstream[0];
  norm_vstream[1] = vstream[1];
  norm_vstream[2] = vstream[2];
  if (norm_vstream[0] != 0.0 || norm_vstream[1] != 0.0 || norm_vstream[2] != 0.0)
    MathExtra::norm3(norm_vstream);

  if (perspecies) {
    for (int i = 0; i < ntask; i++) {
      delete [] tasks[i].ntargetsp;
      tasks[i].ntargetsp = new double[nspecies];
    }
  }

  // resolve per-surf Tsurf custom attribute (thermal_tsurf model only)
  if (emit_model == MODEL_THERMAL_TSURF) {
    if (!model_tsurf_name)
      error->all(FLERR,"Fix surface/emit/source thermal_tsurf needs custom name");
    model_tsurf_index = surf->find_custom(model_tsurf_name);
    if (model_tsurf_index < 0) {
      char msg[256];
      snprintf(msg,sizeof(msg),
        "Fix surface/emit/source: surf custom attribute '%s' not found "
        "(produced by fix surface/state/lm)", model_tsurf_name);
      error->all(FLERR,msg);
    }
  }

  grid_changed();
}

/* ---------------------------------------------------------------------- */

int FixSurfaceEmitSource::local_isurf_index(surfint isurf) const
{
  if (surf->distributed) {
    if (isurf < 0 || isurf >= surf->nown) return -1;
    return static_cast<int>(isurf);
  }

  int me = comm->me;
  int nprocs = comm->nprocs;
  if ((isurf % nprocs) != me) return -1;
  int ilocal = static_cast<int>(isurf / nprocs);
  if (ilocal < 0 || ilocal >= surf->nown) return -1;
  return ilocal;
}

/* ---------------------------------------------------------------------- */

double FixSurfaceEmitSource::flux_for_surface(surfint isurf)
{
  // Flux is read from the globally-replicated flux_localghost[] vector built
  // in spread_flux() via the canonical SPARTA Surf::spread_own2local() path.
  // This is indexed by the global isurf (local+ghost layout), so the cell-
  // owning rank can read any surface its tasks touch — no rank ownership
  // mismatch even when the upstream compute writes only at this rank's
  // owned surfaces.
  if (!flux_localghost) return 0.0;
  if (isurf < 0 || isurf >= flux_n_localghost) return 0.0;
  double f = flux_localghost[isurf];
  if (conc_index >= 0)
    f *= surf->edarray_local[surf->ewhich[conc_index]][isurf][conc_col-1];
  return f;
}

/* ----------------------------------------------------------------------
   build/refresh flux_localghost[] from the upstream per-surf flux compute
   uses Surf::spread_own2local() — same pattern as SurfCollide VARSURF Tsurf
------------------------------------------------------------------------- */

void FixSurfaceEmitSource::spread_flux(Compute *c)
{
  const int nown = surf->nown;
  const int nlg  = surf->nlocal + surf->nghost;

  // (re)allocate replicated buffer when local+ghost size changes
  if (flux_n_localghost != nlg) {
    memory->destroy(flux_localghost);
    flux_n_localghost = nlg;
    if (nlg > 0)
      memory->create(flux_localghost,nlg,"surface/emit/source:flux_localghost");
    else
      flux_localghost = NULL;
  }
  if (nlg == 0) return;

  // pack per-rank-owned flux into a contiguous vector of size nown
  flux_owned_buf.assign(nown,0.0);
  if (c->size_per_surf_cols == 0) {
    if (c->vector_surf) {
      for (int i = 0; i < nown; i++) flux_owned_buf[i] = c->vector_surf[i];
    }
  } else if (c->array_surf) {
    const int icol = (flux_index > 0) ? flux_index-1 : 0;
    if (icol >= 0 && icol < c->size_per_surf_cols) {
      for (int i = 0; i < nown; i++) flux_owned_buf[i] = c->array_surf[i][icol];
    }
  }

  // spread owned -> local+ghost (Allreduce for non-distributed,
  // rendezvous for distributed; handled by Surf)
  surf->spread_own2local(1, DOUBLE,
                         flux_owned_buf.data(), flux_localghost);
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSource::grid_changed()
{
  create_tasks();

  // task layout changed: invalidate cached source strengths
  task_source_cached = 0;
  cached_task_source.clear();
  cached_source_total = 0.0;

  // for mode CONSTANT, set per-task ntarget to area fraction
  if (npmode != FLOW) {
    double areasum_me = 0.0;
    for (int i = 0; i < ntask; i++) areasum_me += tasks[i].area;

    double areasum = 0.0;
    MPI_Allreduce(&areasum_me,&areasum,1,MPI_DOUBLE,MPI_SUM,world);

    if (areasum > 0.0) {
      for (int i = 0; i < ntask; i++)
        tasks[i].ntarget = tasks[i].area / areasum;
    } else {
      for (int i = 0; i < ntask; i++)
        tasks[i].ntarget = 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   create task for one grid cell
   add them to tasks list and increment ntasks
------------------------------------------------------------------------- */

void FixSurfaceEmitSource::create_task(int icell)
{
  int i,m,isurf,npoint,isplit,subcell;
  double indot,area,areaone;
  double *normal,*p1,*p2,*p3,*path;
  double cpath[36],delta[3],e1[3],e2[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  // no tasks if no surfs in cell
  if (cells[icell].nsurf == 0) return;
  // no tasks if cell is outside flow volume
  if (cinfo[icell].volume == 0.0) return;

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  double nrho = particle->mixture[imix]->nrho;
  double *vstream = particle->mixture[imix]->vstream;
  double temp_thermal = particle->mixture[imix]->temp_thermal;

  double *lo = cells[icell].lo;
  double *hi = cells[icell].hi;
  surfint *csurfs = cells[icell].csurfs;
  int nsurf = cells[icell].nsurf;

  for (i = 0; i < nsurf; i++) {
    isurf = csurfs[i];

    if (dimension == 2) {
      if (!(lines[isurf].mask & groupbit)) continue;
    } else {
      if (!(tris[isurf].mask & groupbit)) continue;
    }

    // File mode: replace the mixture defaults with per-surf values from the
    // inflow file, looked up at the surf centroid via bilinear interpolation.
    double per_nrho = nrho;
    double per_T    = temp_thermal;
    double per_v[3] = {vstream[0], vstream[1], vstream[2]};
    if (file_mode) {
      double xc[3] = {0.0, 0.0, 0.0};
      if (dimension == 2) {
        xc[0] = 0.5 * (lines[isurf].p1[0] + lines[isurf].p2[0]);
        xc[1] = 0.5 * (lines[isurf].p1[1] + lines[isurf].p2[1]);
      } else {
        xc[0] = (tris[isurf].p1[0] + tris[isurf].p2[0] + tris[isurf].p3[0]) / 3.0;
        xc[1] = (tris[isurf].p1[1] + tris[isurf].p2[1] + tris[isurf].p3[1]) / 3.0;
        xc[2] = (tris[isurf].p1[2] + tris[isurf].p2[2] + tris[isurf].p3[2]) / 3.0;
      }
      file_lookup(xc, per_nrho, per_v, per_T);
    }

    if (ntask == ntaskmax) grow_task();

    tasks[ntask].icell = icell;
    tasks[ntask].isurf = isurf;
    if (cells[icell].nsplit == 1) tasks[ntask].pcell = icell;
    else {
      isplit = cells[icell].isplit;
      subcell = sinfo[isplit].csplits[i];
      tasks[ntask].pcell = sinfo[isplit].csubs[subcell];
    }

    if (dimension == 2) {
      normal = lines[isurf].norm;
      indot = magvstream;

      p1 = lines[isurf].p1;
      p2 = lines[isurf].p2;
      npoint = cut2d->clip_external(p1,p2,lo,hi,cpath);
      if (npoint < 2) continue;

      tasks[ntask].npoint = 2;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[6];
      path = tasks[ntask].path;
      path[0] = cpath[0];
      path[1] = cpath[1];
      path[2] = 0.0;
      path[3] = cpath[2];
      path[4] = cpath[3];
      path[5] = 0.0;

      if (domain->axisymmetric) {
        double sqrtarg = (path[1]-path[4])*(path[1]-path[4]) +
                         (path[0]-path[3])*(path[0]-path[3]);
        area = MY_PI * (path[1]+path[4]) * sqrt(sqrtarg);
      } else {
        MathExtra::sub3(&path[0],&path[3],delta);
        area = MathExtra::len3(delta);
      }
      tasks[ntask].area = area;

      tasks[ntask].tan1[0] = normal[1];
      tasks[ntask].tan1[1] = -normal[0];
      tasks[ntask].tan1[2] = 0.0;
      tasks[ntask].tan2[0] = 0.0;
      tasks[ntask].tan2[1] = 0.0;
      tasks[ntask].tan2[2] = 1.0;

    } else {
      normal = tris[isurf].norm;
      if (normalflag) indot = magvstream;
      else indot = vstream[0]*normal[0] + vstream[1]*normal[1] +
                   vstream[2]*normal[2];

      p1 = tris[isurf].p1;
      p2 = tris[isurf].p2;
      p3 = tris[isurf].p3;
      npoint = cut3d->clip_external(p1,p2,p3,lo,hi,cpath);
      if (npoint < 3) continue;

      tasks[ntask].npoint = npoint;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[npoint*3];
      path = tasks[ntask].path;
      memcpy(path,cpath,npoint*3*sizeof(double));
      delete [] tasks[ntask].fracarea;
      tasks[ntask].fracarea = new double[npoint-2];

      area = 0.0;
      p1 = &path[0];
      for (m = 0; m < npoint-2; m++) {
        p2 = &path[3*(m+1)];
        p3 = &path[3*(m+2)];
        MathExtra::sub3(p2,p1,e1);
        MathExtra::sub3(p3,p1,e2);
        MathExtra::cross3(e1,e2,delta);
        areaone = fabs(0.5*MathExtra::len3(delta));
        area += areaone;
        tasks[ntask].fracarea[m] = area;
      }
      tasks[ntask].area = area;
      for (m = 0; m < npoint-2; m++) tasks[ntask].fracarea[m] /= area;

      delta[0] = random->uniform();
      delta[1] = random->uniform();
      delta[2] = random->uniform();
      MathExtra::cross3(tris[isurf].norm,delta,tasks[ntask].tan1);
      MathExtra::norm3(tasks[ntask].tan1);
      MathExtra::cross3(tris[isurf].norm,tasks[ntask].tan1,tasks[ntask].tan2);
      MathExtra::norm3(tasks[ntask].tan2);
    }

    tasks[ntask].ntarget = 0.0;
    if (perspecies) {
      for (int isp = 0; isp < nspecies; isp++) tasks[ntask].ntargetsp[isp] = 0.0;
    }

    tasks[ntask].nrho = per_nrho;
    tasks[ntask].temp_thermal = per_T;
    tasks[ntask].temp_rot = particle->mixture[imix]->temp_rot;
    tasks[ntask].temp_vib = particle->mixture[imix]->temp_vib;
    tasks[ntask].magvstream =
      std::sqrt(per_v[0]*per_v[0] + per_v[1]*per_v[1] + per_v[2]*per_v[2]);
    tasks[ntask].vstream[0] = per_v[0];
    tasks[ntask].vstream[1] = per_v[1];
    tasks[ntask].vstream[2] = per_v[2];

    ntask++;
  }
}

/* ----------------------------------------------------------------------
   insert particles in grid cells with emitting surface elements
------------------------------------------------------------------------- */

void FixSurfaceEmitSource::perform_task()
{
  int i,m,n,pcell,isurf,ninsert,nactual,isp,ispecies,ntri,id;
  double indot,scosine,rn,ntarget,vr,alpha,beta,w_emit,source_strength;
  double beta_un,normalized_distbn_fn,theta,erot,evib;
  double vnmag,vamag,vbmag;
  double *normal,*p1,*p2,*p3,*atan,*btan,*vstream,*vscale;
  double x[3],v[3],e1[3],e2[3];
  Particle::OnePart *p;

  // multiply by nevery so source_strength integrates the full inter-firing
  // interval, preserving total flux when nevery > 1
  const double dt = update->dt * nevery;
  int *species = particle->mixture[imix]->species;

  // evaluate requested per-surf flux once per timestep
  // (skipped in file_mode and const_mode: per-task flux is static)
  Compute *c = NULL;
  if (!file_mode && !const_mode) {
    c = modify->compute[iflux];
    c->compute_per_surf();
    // spread per-rank-owned flux into a globally-replicated vector indexed
    // by global isurf — canonical SPARTA pattern (Surf::spread_own2local)
    spread_flux(c);
  }

  // resolve per-surf Tsurf vector for thermal_tsurf model
  // canonical SPARTA pattern (mirrors SurfCollide CUSTOM mode at
  // surf_collide.cpp:225-233): ensure custom owned values are spread to
  // local+ghost, then read tsurf_vec[isurf] using the global surf index.
  double *tsurf_vec = NULL;
  if (emit_model == MODEL_THERMAL_TSURF && model_tsurf_index >= 0) {
    if (surf->estatus[model_tsurf_index] == 0)
      surf->spread_custom(model_tsurf_index);
    tsurf_vec = surf->edvec_local[surf->ewhich[model_tsurf_index]];
  }

  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;

  int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;
  int nfix_update_custom = modify->n_update_custom;
  const int weighted_launch_mode = nlaunch_mode || nlaunch_total_mode;

  // pweight pointer (may be reallocated each step)
  double *pweight_dvec = NULL;
  if (weighted_launch_mode && pweight_ewhich >= 0)
    pweight_dvec = particle->edvec[pweight_ewhich];

  std::vector<double> task_source;
  const double *task_source_ptr = NULL;   // points to active source array
  double source_total = 0.0;
  if (npmode == FLOW && nlaunch_total_mode) {
    // Fast path: when the upstream compute reports a frozen static cache,
    // reuse cached_task_source / cached_source_total instead of rebuilding.
    // The cached size must match ntask (grid_changed() invalidates).
    auto *cpmi = (file_mode || const_mode) ? NULL
                                            : dynamic_cast<ComputeSurfacePhysicalSputter *>(c);
    const bool upstream_static = file_mode || const_mode ||
                                 (cpmi && cpmi->is_static_cached());

    if (upstream_static && task_source_cached &&
        static_cast<int>(cached_task_source.size()) == ntask) {
      // reuse — flux and dt are constant for static sources
      if (cached_source_total <= 0.0) return;
      source_total = cached_source_total;
      task_source_ptr = cached_task_source.data();
    } else {
      task_source.assign(ntask,0.0);

      double source_total_me = 0.0;
      for (i = 0; i < ntask; i++) {
        source_strength = 0.0;
        isurf = tasks[i].isurf;

        double flux;
        if (file_mode) {
          double *vs = tasks[i].vstream;
          double *nrm = (dimension == 2)
                        ? surf->lines[isurf].norm
                        : surf->tris[isurf].norm;
          double vn = vs[0]*nrm[0] + vs[1]*nrm[1] + vs[2]*nrm[2];
          flux = tasks[i].nrho * (vn > 0.0 ? vn : -vn);
        } else if (const_mode) {
          flux = const_flux;
        } else {
          flux = flux_for_surface(isurf);
        }
        if (!std::isfinite(flux) || flux <= 0.0) continue;
        if (flux < flux_thresh) continue;

        source_strength = flux * tasks[i].area * dt;
        if (!std::isfinite(source_strength) || source_strength <= 0.0) continue;

        task_source[i] = source_strength;
        source_total_me += source_strength;
      }

      MPI_Allreduce(&source_total_me,&source_total,1,MPI_DOUBLE,MPI_SUM,world);

      if (upstream_static) {
        cached_task_source = task_source;
        cached_source_total = source_total;
        task_source_cached = 1;
      }

      if (source_total <= 0.0) return;
      task_source_ptr = task_source.data();
    }
  }

  for (i = 0; i < ntask; i++) {
    pcell = tasks[i].pcell;
    isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"BAD surf index");

    if (dimension == 2) normal = lines[isurf].norm;
    else normal = tris[isurf].norm;
    atan = tasks[i].tan1;
    btan = tasks[i].tan2;

    temp_thermal = tasks[i].temp_thermal;
    temp_rot = tasks[i].temp_rot;
    temp_vib = tasks[i].temp_vib;
    magvstream = tasks[i].magvstream;
    vstream = tasks[i].vstream;
    vscale = particle->mixture[imix]->vscale;

    // per-task model state (THERMAL_TSURF reads Tsurf [°C] from custom)
    // tsurf_vec is the local+ghost view from spread_custom(), indexed by isurf
    double Tsurf_K_task = -1.0;
    if (emit_model == MODEL_THERMAL_TSURF) {
      if (tsurf_vec == NULL) continue;
      double T_K = tsurf_vec[isurf] + 273.15;
      if (!std::isfinite(T_K) || T_K <= 0.0) continue;
      Tsurf_K_task = T_K;
      // override mixture rot/vib so internal energies match wall temperature
      temp_thermal = T_K;
      temp_rot = T_K;
      temp_vib = T_K;
    }

    if (normalflag) indot = magvstream;
    else indot = vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];

    w_emit = fnum;  // default weight = fnum (backward compatible)
    source_strength = 0.0;
    ninsert = -1;

    if (npmode == FLOW) {
      if (nlaunch_total_mode) {
        source_strength = task_source_ptr[i];
        if (source_strength <= 0.0) continue;

        ntarget = static_cast<double>(nlaunch_total) * source_strength / source_total;
        ninsert = static_cast<int>(ntarget + random->uniform());
        if (ninsert <= 0) continue;

        // Stratified source conservation. Elements resolved by the budget
        // (ntarget >= 1) emit exactly their own source each event. Elements
        // below one particle per event fire with probability ntarget; the
        // unbiased weight is then s_i/ntarget = S_total/N, the global
        // per-particle quantum (w = s_i/ninsert would under-emit weak
        // elements by the factor ntarget in expectation).
        if (ntarget >= 1.0) w_emit = source_strength / ninsert;
        else w_emit = source_total / static_cast<double>(nlaunch_total);
      } else {
        double flux;
        if (file_mode) {
          double vn = vstream[0]*normal[0] + vstream[1]*normal[1] +
                      vstream[2]*normal[2];
          flux = tasks[i].nrho * (vn > 0.0 ? vn : -vn);
        } else if (const_mode) {
          flux = const_flux;
        } else {
          flux = flux_for_surface(isurf);
        }
        if (!std::isfinite(flux) || flux <= 0.0) continue;
        if (flux < flux_thresh) continue;

        source_strength = flux * tasks[i].area * dt;
        if (!std::isfinite(source_strength) || source_strength <= 0.0) continue;

        if (nlaunch_mode) {
          // per-particle weight mode: fixed number of particles per task
          w_emit = source_strength / nlaunch_per_surf;
          ntarget = static_cast<double>(nlaunch_per_surf);
        } else {
          ntarget = source_strength / fnum;
        }
      }
    } else {
      ntarget = np * tasks[i].ntarget;
    }
    if (ntarget <= 0.0) continue;

    if (perspecies) {
      for (isp = 0; isp < nspecies; isp++) {
        ispecies = species[isp];
        double ntarget_sp = ntarget * fraction[isp];
        ninsert = static_cast<int>(ntarget_sp + random->uniform());
        if (ninsert <= 0) continue;
        scosine = indot / vscale[isp];

        nactual = 0;
        for (m = 0; m < ninsert; m++) {
          if (dimension == 2) {
            rn = random->uniform();
            p1 = &tasks[i].path[0];
            p2 = &tasks[i].path[3];
            x[0] = p1[0] + rn * (p2[0]-p1[0]);
            x[1] = p1[1] + rn * (p2[1]-p1[1]);
            x[2] = 0.0;
          } else {
            rn = random->uniform();
            ntri = tasks[i].npoint - 2;
            for (n = 0; n < ntri; n++)
              if (rn < tasks[i].fracarea[n]) break;
            p1 = &tasks[i].path[0];
            p2 = &tasks[i].path[3*(n+1)];
            p3 = &tasks[i].path[3*(n+2)];
            MathExtra::sub3(p2,p1,e1);
            MathExtra::sub3(p3,p1,e2);
            alpha = random->uniform();
            beta = random->uniform();
            if (alpha+beta > 1.0) {
              alpha = 1.0 - alpha;
              beta = 1.0 - beta;
            }
            x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
            x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
            x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
          }

          if (region && !region->match(x)) continue;

          if (emit_model == MODEL_THERMAL) {
            do {
              do beta_un = (6.0*random->uniform() - 3.0);
              while (beta_un + scosine < 0.0);
              normalized_distbn_fn = 2.0 * (beta_un + scosine) /
                (scosine + sqrt(scosine*scosine + 2.0)) *
                exp(0.5 + (0.5*scosine)*(scosine-sqrt(scosine*scosine + 2.0)) -
                    beta_un*beta_un);
            } while (normalized_distbn_fn < random->uniform());

            if (normalflag) vnmag = beta_un*vscale[isp] + magvstream;
            else vnmag = beta_un*vscale[isp] + indot;

            theta = MY_2PI * random->uniform();
            vr = vscale[isp] * sqrt(-log(random->uniform()));
            if (normalflag) {
              vamag = vr * sin(theta);
              vbmag = vr * cos(theta);
            } else {
              vamag = vr * sin(theta) + MathExtra::dot3(vstream,atan);
              vbmag = vr * cos(theta) + MathExtra::dot3(vstream,btan);
            }
          } else {
            // energy + cos^n direction in the surface frame (n=1 cosine default)
            double E_eV;
            double cos_n_local = 1.0;
            if (emit_model == MODEL_THOMPSON) {
              E_eV = sample_thompson_energy(model_Ub, model_Emax, random);
              cos_n_local = model_cos_n;
            } else if (emit_model == MODEL_FIXED_ENERGY) {
              E_eV = model_E_fixed;
            } else { // MODEL_THERMAL_TSURF
              E_eV = sample_thermal_flux_energy(Tsurf_K_task, random);
            }
            double cos_th, sin_th;
            sample_cos_n(cos_n_local, random, cos_th, sin_th);
            double phi_az = MY_2PI * random->uniform();
            double mass_sp = particle->species[ispecies].mass;
            double speed = (mass_sp > 0.0)
              ? std::sqrt(2.0 * E_eV * EV_TO_J / mass_sp) : 0.0;
            vnmag = speed * cos_th;
            vamag = speed * sin_th * std::cos(phi_az);
            vbmag = speed * sin_th * std::sin(phi_az);
            vnmag += vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];
            vamag += vstream[0]*atan[0]   + vstream[1]*atan[1]   + vstream[2]*atan[2];
            vbmag += vstream[0]*btan[0]   + vstream[1]*btan[1]   + vstream[2]*btan[2];
          }

          v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
          v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
          v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

          erot = particle->erot(ispecies,temp_rot,random);
          evib = particle->evib(ispecies,temp_vib,random);
          id = MAXSMALLINT*random->uniform();

          particle->add_particle(id,ispecies,pcell,x,v,erot,evib);
          nactual++;

          p = &particle->particles[particle->nlocal-1];
          p->flag = PSURF + 1 + isurf;
          p->dtremain = dt * random->uniform();

          if (nlaunch_mode) {
            pweight_dvec = particle->edvec[pweight_ewhich];
            pweight_dvec[particle->nlocal-1] = w_emit;
          }

          if (nsurf_tally)
            for (int k = 0; k < nsurf_tally; k++)
              slist_active[k]->surf_tally(p->dtremain,isurf,pcell,0,NULL,p,NULL);

          if (nfix_update_custom)
            modify->update_custom(particle->nlocal-1,temp_thermal,
                                  temp_rot,temp_vib,vstream);
        }
        nsingle += nactual;
      }

    } else {
      if (ninsert < 0) {
        ninsert = static_cast<int>(ntarget + random->uniform());
        if (ninsert <= 0) continue;
      }

      nactual = 0;
      for (m = 0; m < ninsert; m++) {
        rn = random->uniform();
        isp = 0;
        while (cummulative[isp] < rn) isp++;
        ispecies = species[isp];
        scosine = indot / vscale[isp];

        if (dimension == 2) {
          rn = random->uniform();
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3];
          x[0] = p1[0] + rn * (p2[0]-p1[0]);
          x[1] = p1[1] + rn * (p2[1]-p1[1]);
          x[2] = 0.0;
        } else {
          rn = random->uniform();
          ntri = tasks[i].npoint - 2;
          for (n = 0; n < ntri; n++)
            if (rn < tasks[i].fracarea[n]) break;
          p1 = &tasks[i].path[0];
          p2 = &tasks[i].path[3*(n+1)];
          p3 = &tasks[i].path[3*(n+2)];
          MathExtra::sub3(p2,p1,e1);
          MathExtra::sub3(p3,p1,e2);
          alpha = random->uniform();
          beta = random->uniform();
          if (alpha+beta > 1.0) {
            alpha = 1.0 - alpha;
            beta = 1.0 - beta;
          }
          x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
          x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
          x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
        }

        if (region && !region->match(x)) continue;

        if (emit_model == MODEL_THERMAL) {
          do {
            do beta_un = (6.0*random->uniform() - 3.0);
            while (beta_un + scosine < 0.0);
            normalized_distbn_fn = 2.0 * (beta_un + scosine) /
              (scosine + sqrt(scosine*scosine + 2.0)) *
              exp(0.5 + (0.5*scosine)*(scosine-sqrt(scosine*scosine + 2.0)) -
                  beta_un*beta_un);
          } while (normalized_distbn_fn < random->uniform());

          if (normalflag) vnmag = beta_un*vscale[isp] + magvstream;
          else vnmag = beta_un*vscale[isp] + indot;

          theta = MY_2PI * random->uniform();
          vr = vscale[isp] * sqrt(-log(random->uniform()));
          if (normalflag) {
            vamag = vr * sin(theta);
            vbmag = vr * cos(theta);
          } else {
            vamag = vr * sin(theta) + MathExtra::dot3(vstream,atan);
            vbmag = vr * cos(theta) + MathExtra::dot3(vstream,btan);
          }
        } else {
          double E_eV;
          double cos_n_local = 1.0;
          if (emit_model == MODEL_THOMPSON) {
            E_eV = sample_thompson_energy(model_Ub, model_Emax, random);
            cos_n_local = model_cos_n;
          } else if (emit_model == MODEL_FIXED_ENERGY) {
            E_eV = model_E_fixed;
          } else { // MODEL_THERMAL_TSURF
            E_eV = sample_thermal_flux_energy(Tsurf_K_task, random);
          }
          double cos_th, sin_th;
          sample_cos_n(cos_n_local, random, cos_th, sin_th);
          double phi_az = MY_2PI * random->uniform();
          double mass_sp = particle->species[ispecies].mass;
          double speed = (mass_sp > 0.0)
            ? std::sqrt(2.0 * E_eV * EV_TO_J / mass_sp) : 0.0;
          vnmag = speed * cos_th;
          vamag = speed * sin_th * std::cos(phi_az);
          vbmag = speed * sin_th * std::sin(phi_az);
          vnmag += vstream[0]*normal[0] + vstream[1]*normal[1] + vstream[2]*normal[2];
          vamag += vstream[0]*atan[0]   + vstream[1]*atan[1]   + vstream[2]*atan[2];
          vbmag += vstream[0]*btan[0]   + vstream[1]*btan[1]   + vstream[2]*btan[2];
        }

        v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
        v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
        v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

        erot = particle->erot(ispecies,temp_rot,random);
        evib = particle->evib(ispecies,temp_vib,random);
        id = MAXSMALLINT*random->uniform();

        particle->add_particle(id,ispecies,pcell,x,v,erot,evib);
        nactual++;

        p = &particle->particles[particle->nlocal-1];
        p->flag = PSURF + 1 + isurf;
        p->dtremain = dt * random->uniform();

        if (weighted_launch_mode) {
          pweight_dvec = particle->edvec[pweight_ewhich];
          pweight_dvec[particle->nlocal-1] = w_emit;
        }

        if (nsurf_tally)
          for (int k = 0; k < nsurf_tally; k++)
            slist_active[k]->surf_tally(p->dtremain,isurf,pcell,0,NULL,p,NULL);

        if (nfix_update_custom)
          modify->update_custom(particle->nlocal-1,temp_thermal,
                                temp_rot,temp_vib,vstream);
      }
      nsingle += nactual;
    }
  }
}

/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixSurfaceEmitSource::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks,ntaskmax*sizeof(Task),
                                    "surface/emit/source:tasks");

  memset(&tasks[oldmax],0,(ntaskmax-oldmax)*sizeof(Task));

  for (int i = oldmax; i < ntaskmax; i++) {
    if (perspecies) tasks[i].ntargetsp = new double[nspecies];
    else tasks[i].ntargetsp = NULL;
    tasks[i].vscale = NULL;
    tasks[i].path = NULL;
    tasks[i].fracarea = NULL;
  }
}

/* ----------------------------------------------------------------------
   process keywords specific to this class
------------------------------------------------------------------------- */

int FixSurfaceEmitSource::option(int narg, char **arg)
{
  if (strcmp(arg[0],"conc_scale") == 0) {
    // multiply the per-surf flux by column <col> of a per-surf custom
    // DOUBLE array (e.g. the surface/pwi mixing-zone concentrations):
    // emitted flux of this material = erosion flux * local concentration
    if (3 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    delete [] conc_attr;
    int n = strlen(arg[1]) + 1;
    conc_attr = new char[n];
    strcpy(conc_attr, arg[1]);
    conc_col = atoi(arg[2]);
    if (conc_col < 1)
      error->all(FLERR,"Fix surface/emit/source conc_scale column must be >= 1");
    return 3;
  }

  if (strcmp(arg[0],"n") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    np = atoi(arg[1]);
    if (np <= 0) npmode = FLOW;
    else npmode = CONSTANT;
    return 2;
  }

  if (strcmp(arg[0],"normal") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    if (strcmp(arg[1],"yes") == 0) normalflag = 1;
    else if (strcmp(arg[1],"no") == 0) normalflag = 0;
    else error->all(FLERR,"Illegal fix surface/emit/source command");
    return 2;
  }

  if (strcmp(arg[0],"nlaunch") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    if (nlaunch_total_mode)
      error->all(FLERR,
        "Cannot use fix surface/emit/source nlaunch with nlaunch_total");
    nlaunch_per_surf = atoi(arg[1]);
    if (nlaunch_per_surf <= 0)
      error->all(FLERR,"Fix surface/emit/source nlaunch must be > 0");
    nlaunch_mode = 1;
    return 2;
  }

  if (strcmp(arg[0],"nlaunch_total") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    if (nlaunch_mode)
      error->all(FLERR,
        "Cannot use fix surface/emit/source nlaunch_total with nlaunch");
    nlaunch_total = atoi(arg[1]);
    if (nlaunch_total <= 0)
      error->all(FLERR,"Fix surface/emit/source nlaunch_total must be > 0");
    nlaunch_total_mode = 1;
    return 2;
  }

  if (strcmp(arg[0],"flux_thresh") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    flux_thresh = atof(arg[1]);
    if (flux_thresh < 0.0)
      error->all(FLERR,"Fix surface/emit/source flux_thresh must be >= 0");
    return 2;
  }

  if (strcmp(arg[0],"source_thresh") == 0) {
    error->all(FLERR,"fix surface/emit/source: source_thresh was removed "
               "(it censored real source and made the total emission depend "
               "on the cutoff). Use nlaunch_total N for a fixed particle "
               "budget with flux-carrying weights, or nlaunch N per element.");
    return 2;
  }

  if (strcmp(arg[0],"model") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/source command");
    if (strcmp(arg[1],"thermal") == 0) {
      emit_model = MODEL_THERMAL;
      return 2;
    }
    if (strcmp(arg[1],"thermal_tsurf") == 0) {
      if (3 > narg)
        error->all(FLERR,"Fix surface/emit/source: model thermal_tsurf needs custom name");
      delete [] model_tsurf_name;
      int n = strlen(arg[2]) + 1;
      model_tsurf_name = new char[n];
      strcpy(model_tsurf_name, arg[2]);
      emit_model = MODEL_THERMAL_TSURF;
      return 3;
    }
    if (strcmp(arg[1],"thompson") == 0) {
      if (3 > narg)
        error->all(FLERR,"Fix surface/emit/source: model thompson needs Ub_eV");
      model_Ub = atof(arg[2]);
      if (model_Ub <= 0.0)
        error->all(FLERR,"Fix surface/emit/source: model thompson Ub must be > 0");
      emit_model = MODEL_THOMPSON;
      int consumed = 3;
      // optional keywords in any order: cos_n <n>, emax <eV>
      while (consumed + 1 < narg) {
        if (strcmp(arg[consumed],"cos_n") == 0) {
          model_cos_n = atof(arg[consumed+1]);
          if (model_cos_n <= 0.0)
            error->all(FLERR,"Fix surface/emit/source: model thompson cos_n must be > 0");
          consumed += 2;
        } else if (strcmp(arg[consumed],"emax") == 0) {
          model_Emax = atof(arg[consumed+1]);
          if (model_Emax <= model_Ub)
            error->all(FLERR,"Fix surface/emit/source: model thompson emax must be > Ub");
          consumed += 2;
        } else break;
      }
      return consumed;
    }
    if (strcmp(arg[1],"fixed_energy") == 0) {
      if (3 > narg)
        error->all(FLERR,"Fix surface/emit/source: model fixed_energy needs E_eV");
      model_E_fixed = atof(arg[2]);
      if (model_E_fixed <= 0.0)
        error->all(FLERR,"Fix surface/emit/source: model fixed_energy E must be > 0");
      emit_model = MODEL_FIXED_ENERGY;
      return 3;
    }
    error->all(FLERR,"Fix surface/emit/source: unknown model "
                     "(thermal | thermal_tsurf | thompson | fixed_energy)");
  }

  error->all(FLERR,"Illegal fix surface/emit/source command");
  return 0;
}

/* ----------------------------------------------------------------------
   File-driven flux mode helpers (cribbed from wip/fix_emit_face_file).
   read_file -> file_load: rank 0 parses the SPARTA inflow .dat
   bcast_mesh -> file_bcast: broadcast to all ranks
   file_lookup: bilinear-interp at a surf centroid
------------------------------------------------------------------------- */

void FixSurfaceEmitSource::file_load()
{
  FILE *fp = fopen(file_path,"r");
  if (!fp) {
    char msg[256];
    snprintf(msg,sizeof(msg),
             "Fix surface/emit/source: cannot open file %s", file_path);
    error->one(FLERR,msg);
  }

  char line[MAXLINE];
  char *word, *tmp;

  // scan to the requested section keyword
  while (1) {
    if (fgets(line,MAXLINE,fp) == NULL)
      error->one(FLERR,"Fix surface/emit/source: section not found in file");
    if (strspn(line," \t\n\r") == strlen(line)) continue;
    if (line[0] == '#') continue;
    word = strtok(line," \t\n\r");
    if (word && strcmp(word,file_section) == 0) break;
  }

  // NIJ Ni Nj
  tmp = fgets(line,MAXLINE,fp);
  word = strtok(line," \t\n\r");
  if (!word || strcmp(word,"NIJ") != 0)
    error->one(FLERR,"Fix surface/emit/source: expected NIJ");
  fmesh.ni = atoi(strtok(NULL," \t\n\r"));
  fmesh.nj = atoi(strtok(NULL," \t\n\r"));
  if (fmesh.ni < 2 || fmesh.nj < 2)
    error->one(FLERR,"Fix surface/emit/source: NIJ too small");

  // NV Nv
  tmp = fgets(line,MAXLINE,fp);
  word = strtok(line," \t\n\r");
  if (!word || strcmp(word,"NV") != 0)
    error->one(FLERR,"Fix surface/emit/source: expected NV");
  fmesh.nvalues = atoi(strtok(NULL," \t\n\r"));
  if (fmesh.nvalues <= 0)
    error->one(FLERR,"Fix surface/emit/source: bad NV");

  // VALUES name1 ...
  tmp = fgets(line,MAXLINE,fp);
  word = strtok(line," \t\n\r");
  if (!word || strcmp(word,"VALUES") != 0)
    error->one(FLERR,"Fix surface/emit/source: expected VALUES");
  fmesh.which.assign(fmesh.nvalues, 0);
  for (int m = 0; m < fmesh.nvalues; m++) {
    word = strtok(NULL," \t\n\r");
    if      (!strcmp(word,"nrho"))  fmesh.which[m] = IM_NRHO;
    else if (!strcmp(word,"vx"))    fmesh.which[m] = IM_VX;
    else if (!strcmp(word,"vy"))    fmesh.which[m] = IM_VY;
    else if (!strcmp(word,"vz"))    fmesh.which[m] = IM_VZ;
    else if (!strcmp(word,"temp"))  fmesh.which[m] = IM_TEMP_THERMAL;
    else if (!strcmp(word,"gamma")) fmesh.which[m] = IM_GAMMA;
    else error->one(FLERR,"Fix surface/emit/source: unknown VALUES entry "
                          "(nrho/vx/vy/vz/temp/gamma)");
  }

  // IMESH x_0 ...
  tmp = fgets(line,MAXLINE,fp);
  word = strtok(line," \t\n\r");
  if (!word || strcmp(word,"IMESH") != 0)
    error->one(FLERR,"Fix surface/emit/source: expected IMESH");
  fmesh.imesh.assign(fmesh.ni, 0.0);
  for (int i = 0; i < fmesh.ni; i++)
    fmesh.imesh[i] = atof(strtok(NULL," \t\n\r"));

  // JMESH y_0 ...
  tmp = fgets(line,MAXLINE,fp);
  word = strtok(line," \t\n\r");
  if (!word || strcmp(word,"JMESH") != 0)
    error->one(FLERR,"Fix surface/emit/source: expected JMESH");
  fmesh.jmesh.assign(fmesh.nj, 0.0);
  for (int j = 0; j < fmesh.nj; j++)
    fmesh.jmesh[j] = atof(strtok(NULL," \t\n\r"));

  fmesh.lo[0] = fmesh.imesh[0];
  fmesh.hi[0] = fmesh.imesh[fmesh.ni-1];
  fmesh.lo[1] = fmesh.jmesh[0];
  fmesh.hi[1] = fmesh.jmesh[fmesh.nj-1];

  // Skip blank line, then read Ni*Nj rows: i j v1 v2 ...
  size_t n = static_cast<size_t>(fmesh.ni) * fmesh.nj * fmesh.nvalues;
  fmesh.values.assign(n, 0.0);
  // read until we have Ni*Nj data rows; tolerate blank lines in between
  long count = 0;
  long target = static_cast<long>(fmesh.ni) * fmesh.nj;
  while (count < target && fgets(line,MAXLINE,fp)) {
    if (strspn(line," \t\n\r") == strlen(line)) continue;
    if (line[0] == '#') continue;
    word = strtok(line," \t\n\r");
    int ii = atoi(word);
    int jj = atoi(strtok(NULL," \t\n\r"));
    if (ii < 1 || ii > fmesh.ni || jj < 1 || jj > fmesh.nj)
      error->one(FLERR,"Fix surface/emit/source: row indices out of range");
    long off = (static_cast<long>(jj-1) * fmesh.ni + (ii-1)) * fmesh.nvalues;
    for (int m = 0; m < fmesh.nvalues; m++)
      fmesh.values[off + m] = atof(strtok(NULL," \t\n\r"));
    count++;
  }
  fclose(fp);
  if (count != target)
    error->one(FLERR,"Fix surface/emit/source: not enough data rows in file");
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSource::file_bcast()
{
  MPI_Bcast(&fmesh.ni,      1, MPI_INT,    0, world);
  MPI_Bcast(&fmesh.nj,      1, MPI_INT,    0, world);
  MPI_Bcast(&fmesh.nvalues, 1, MPI_INT,    0, world);
  MPI_Bcast(fmesh.lo,       2, MPI_DOUBLE, 0, world);
  MPI_Bcast(fmesh.hi,       2, MPI_DOUBLE, 0, world);

  if (comm->me) {
    fmesh.imesh.assign(fmesh.ni, 0.0);
    fmesh.jmesh.assign(fmesh.nj, 0.0);
    fmesh.which.assign(fmesh.nvalues, 0);
    fmesh.values.assign(static_cast<size_t>(fmesh.ni) * fmesh.nj * fmesh.nvalues, 0.0);
  }
  MPI_Bcast(fmesh.imesh.data(),  fmesh.ni,      MPI_DOUBLE, 0, world);
  MPI_Bcast(fmesh.jmesh.data(),  fmesh.nj,      MPI_DOUBLE, 0, world);
  MPI_Bcast(fmesh.which.data(),  fmesh.nvalues, MPI_INT,    0, world);
  MPI_Bcast(fmesh.values.data(),
            static_cast<int>(fmesh.values.size()), MPI_DOUBLE, 0, world);
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitSource::file_lookup(const double *xyz, double &nrho_out,
                                       double *vstream_out, double &T_out) const
{
  // Pick the (i, j) coordinates of the surf centroid based on the section's
  // face axis. The face axis itself is the "perpendicular" coordinate.
  int i_axis = (face_axis_idx == 0) ? 1 : 0;     // skip face axis
  int j_axis = (face_axis_idx == 2) ? 1 : 2;
  if (face_axis_idx == 1) { i_axis = 0; j_axis = 2; }

  double xq = xyz[i_axis];
  double yq = xyz[j_axis];

  // Bilinear lookup, edge-clamped
  if (xq < fmesh.lo[0]) xq = fmesh.lo[0];
  if (xq > fmesh.hi[0]) xq = fmesh.hi[0];
  if (yq < fmesh.lo[1]) yq = fmesh.lo[1];
  if (yq > fmesh.hi[1]) yq = fmesh.hi[1];

  // find lower index in each axis
  int i = 0;
  while (i < fmesh.ni - 2 && fmesh.imesh[i+1] < xq) i++;
  int j = 0;
  while (j < fmesh.nj - 2 && fmesh.jmesh[j+1] < yq) j++;

  double dxi = fmesh.imesh[i+1] - fmesh.imesh[i];
  double dyj = fmesh.jmesh[j+1] - fmesh.jmesh[j];
  double fx = (dxi > 0.0) ? (xq - fmesh.imesh[i]) / dxi : 0.0;
  double fy = (dyj > 0.0) ? (yq - fmesh.jmesh[j]) / dyj : 0.0;

  auto cell_off = [&](int ii, int jj) {
    return (static_cast<long>(jj) * fmesh.ni + ii) * fmesh.nvalues;
  };
  long o00 = cell_off(i,     j);
  long o10 = cell_off(i + 1, j);
  long o01 = cell_off(i,     j + 1);
  long o11 = cell_off(i + 1, j + 1);

  // walk the value columns; only override outputs for fields that appear
  for (int m = 0; m < fmesh.nvalues; m++) {
    double v = (1.0 - fx) * (1.0 - fy) * fmesh.values[o00 + m]
             +        fx  * (1.0 - fy) * fmesh.values[o10 + m]
             + (1.0 - fx) *        fy  * fmesh.values[o01 + m]
             +        fx  *        fy  * fmesh.values[o11 + m];
    switch (fmesh.which[m]) {
      case IM_NRHO:           nrho_out      = v; break;
      case IM_VX:             vstream_out[0] = v; break;
      case IM_VY:             vstream_out[1] = v; break;
      case IM_VZ:             vstream_out[2] = v; break;
      case IM_TEMP_THERMAL:   T_out         = v; break;
      case IM_GAMMA:          /* redundant with nrho * vz; ignore */ break;
      default: break;
    }
  }
}
