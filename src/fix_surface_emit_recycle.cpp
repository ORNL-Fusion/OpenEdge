/* ----------------------------------------------------------------------
    OpenEdge: fix surface/emit/recycle
    Wall-recycling neutral source driven by the local plasma Bohm flux.
    See header for algorithm summary.
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_surface_emit_recycle.h"
#include "fix_background.h"
#include "surf_react_surface_pwi.h"
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
#include "openedge_geom.h"
#include "error.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

using namespace SPARTA_NS;
using namespace MathConst;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};

#define DELTATASK 256

namespace {
  constexpr double QE    = 1.602176634e-19;   // C
  constexpr double AMU   = 1.66053906660e-27; // kg
  constexpr double KB    = 1.380649e-23;      // J/K
  constexpr double ME    = 9.1093837015e-31;  // kg
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitRecycle::FixSurfaceEmitRecycle(SPARTA *sparta, int narg, char **arg) :
  FixEmit(sparta, narg, arg)
{
  // Emission tasks cache per-grid pcell indices; gridmigrate=1 puts this
  // fix in Modify::list_pergrid so Grid::notify_changed() invokes
  // grid_changed() -> create_tasks() after mid-run balance/adapt
  // migrations. Without it, stale task pcell values made post-balance
  // emissions insert particles with old (wrong or ghost/out-of-range)
  // cell indices -> silent mis-binning and downstream segfaults.
  gridmigrate = 1;

  // Usage: fix ID emit/surf/recycle mix group plasma_fix_ID
  //              [mass <amu>] [R <val>] [twall <K>]
  if (narg < 5) error->all(FLERR,"Illegal fix surface/emit/recycle command");

  imix = particle->find_mixture(arg[2]);
  if (imix < 0)
    error->all(FLERR,"Fix emit/surf/recycle mixture ID does not exist");

  int igroup = surf->find_group(arg[3]);
  if (igroup < 0)
    error->all(FLERR,"Fix emit/surf/recycle group ID does not exist");
  groupbit = surf->bitmask[igroup];

  ifix_plasma = modify->find_fix(arg[4]);
  if (ifix_plasma < 0)
    error->all(FLERR,"Fix emit/surf/recycle plasma fix ID does not exist");
  plasma = dynamic_cast<FixBackground *>(modify->fix[ifix_plasma]);
  if (!plasma)
    error->all(FLERR,"Fix emit/surf/recycle requires a fix background");

  // no physics defaults: mass, and R/twall without pwi, must be given;
  // the wall-flux search radius is derived from the sample spacing
  mass_amu   = 0.0;
  R_recycle  = -1.0;
  twall      = 0.0;
  wall_flux_cutoff = 0.0;
  impact_model = IMPACT_MAXWELL;
  wall_flux_ispec  = 0;
  pwi_id = NULL;
  pwi = NULL;
  pwi_species = -1;
  ion_charge = 1.0;

  int iarg = 5;
  options(narg-iarg, &arg[iarg]);

  if (!surf->exist)
    error->all(FLERR,"Fix emit/surf/recycle requires surface elements");
  if (surf->implicit)
    error->all(FLERR,"Fix emit/surf/recycle not allowed for implicit surfaces");

  tasks = NULL;
  ntask = ntaskmax = 0;
  diag_printed = 0;

  // Plasma-generation tracking for dynamic reloads. Bumped every time
  // FixBackground::reload() runs; we compare here and rebuild tasks +
  // centroid cache if it changed.
  plasma_generation_ = -1;
  plasma_cell_centroid_gen_ = -1;

  dimension = domain->dimension;
  if (dimension == 3) cut3d = new Cut3d(sparta);
  else cut2d = new Cut2d(sparta, domain->axisymmetric);
}

/* ---------------------------------------------------------------------- */

FixSurfaceEmitRecycle::~FixSurfaceEmitRecycle()
{
  if (copymode) return;

  for (int i = 0; i < ntaskmax; i++) {
    delete [] tasks[i].path;
    delete [] tasks[i].fracarea;
  }
  memory->sfree(tasks);
  delete [] pwi_id;

  if (dimension == 3) delete cut3d;
  else delete cut2d;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitRecycle::init()
{
  FixEmit::init();

  fnum = update->fnum;

  nspecies    = particle->mixture[imix]->nspecies;
  fraction    = particle->mixture[imix]->fraction;
  cummulative = particle->mixture[imix]->cummulative;

  if (ifix_plasma < 0 || ifix_plasma >= modify->nfix)
    error->all(FLERR,"Fix emit/surf/recycle plasma fix ID no longer exists");
  plasma = dynamic_cast<FixBackground *>(modify->fix[ifix_plasma]);
  // reads mesh_te/mesh_ne by SOLPS cell and bfield_at(R,Z): mesh-only
  if (plasma && plasma->is_zones3d())
    error->all(FLERR, "fix surface/emit/recycle: requires an R-Z mesh "
               "plasma background; native zones3d backgrounds are not "
               "supported by this fix");
  if (!plasma)
    error->all(FLERR,"Fix emit/surf/recycle requires a fix background");

  // Resolve per-species temperature overrides into a mixture-local table.
  // Unmatched mixture species fall back to the scalar twall. A name listed
  // in twall_species_names that is not present in this mixture is an error
  // (fail loudly rather than silently ignoring).
  if (mass_amu <= 0.0)
    error->all(FLERR,"fix surface/emit/recycle: mass <amu> is required");
  if (!pwi_id && R_recycle < 0.0)
    error->all(FLERR,"fix surface/emit/recycle: R <coefficient> is required without pwi");
  if (!pwi_id && twall <= 0.0)
    error->all(FLERR,"fix surface/emit/recycle: twall <K> is required without pwi");
  if (wall_flux_cutoff <= 0.0 && plasma->has_wall_flux()) {
    // twice the largest nearest-neighbour spacing of the wall samples,
    // so every wall element reaches at least one sample
    const FixBackground::WallFluxData &wf = plasma->wall_flux;
    double dmax = 0.0;
    for (int i = 0; i < wf.n; i++) {
      double dmin = 1.0e300;
      for (int j = 0; j < wf.n; j++) {
        if (j == i) continue;
        const double d = std::hypot(wf.r[i]-wf.r[j], wf.z[i]-wf.z[j]);
        if (d < dmin) dmin = d;
      }
      if (dmin < 1.0e300 && dmin > dmax) dmax = dmin;
    }
    wall_flux_cutoff = 2.0 * dmax;
    if (comm->me == 0) {
      if (screen)  fprintf(screen,  "fix surface/emit/recycle: wall_flux_cutoff %g m from the sample spacing\n", wall_flux_cutoff);
      if (logfile) fprintf(logfile, "fix surface/emit/recycle: wall_flux_cutoff %g m from the sample spacing\n", wall_flux_cutoff);
    }
  }
  twall_by_species.assign(nspecies, twall);
  const int *mix_species = particle->mixture[imix]->species;
  for (size_t k = 0; k < twall_species_names.size(); k++) {
    int global_sp = particle->find_species(
                      const_cast<char *>(twall_species_names[k].c_str()));
    if (global_sp < 0) {
      std::string msg = "fix surface/emit/recycle twall_species: unknown species '"
                        + twall_species_names[k] + "'";
      error->all(FLERR, msg.c_str());
    }
    int slot = -1;
    for (int isp = 0; isp < nspecies; isp++)
      if (mix_species[isp] == global_sp) { slot = isp; break; }
    if (slot < 0) {
      std::string msg = "fix surface/emit/recycle twall_species: species '"
                        + twall_species_names[k] + "' is not in mixture";
      error->all(FLERR, msg.c_str());
    }
    twall_by_species[slot] = twall_species_values[k];
  }

  // pwi mode: the mixture names the incident ion; the surf_react does the rest
  pwi = NULL;
  if (pwi_id) {
    int isr = surf->find_react(pwi_id);
    if (isr < 0)
      error->all(FLERR,"fix surface/emit/recycle pwi: surf_react ID does not exist");
    pwi = dynamic_cast<SurfReactSurfacePWI *>(surf->sr[isr]);
    if (!pwi)
      error->all(FLERR,"fix surface/emit/recycle pwi: surf_react must be surface/pwi");
    if (nspecies != 1)
      error->all(FLERR,"fix surface/emit/recycle pwi: the mixture must hold "
                 "exactly one species, the incident ion");
    pwi_species = mix_species[0];
    ion_charge = particle->species[pwi_species].charge;
    if (ion_charge <= 0.0)
      error->all(FLERR,"fix surface/emit/recycle pwi: the mixture species must be an ion");
  }

  grid_changed();
}

/* ----------------------------------------------------------------------
   mean ion impact energy: Chankin 2014 eq. (4), the floating sheath of the
   main ion (mass_amu) plus 2 Ti, as in compute surface/physical/sputter
------------------------------------------------------------------------- */

double FixSurfaceEmitRecycle::impact_energy_eV(double te_eV, double ti_eV) const
{
  if (!(te_eV > 0.0)) return 2.0 * std::max(ti_eV, 0.0);
  const double mi_over_me = (mass_amu * AMU) / ME;
  const double psi_over_te = 0.5 * std::log(mi_over_me / (2.0 * MY_PI * (1.0 + ti_eV / te_eV)));
  return 2.0 * std::max(ti_eV, 0.0) + ion_charge * std::max(psi_over_te, 0.0) * te_eV;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceEmitRecycle::grid_changed()
{
  create_tasks();

  // Per-task area share within its parent wall isurf: when adapt_grid
  // refines a wall segment into multiple cells, each refined task gets a
  // fraction = task.area / (sum of task areas with the same isurf). With
  // mesh/wall_surf_area[isurf] = aggregated B2 face area (set by the
  // converter, possibly summing over multiple B2 cells that all map to
  // this segment), the per-task emission rate is
  //    Gamma * mesh_wall_surf_area[isurf] * area_share
  // so the per-isurf total is exactly Gamma * surf_area[isurf] regardless
  // of how many tasks share the isurf.
  const int nsurf_global = surf->nsurf;
  std::vector<double> area_sum_me(nsurf_global, 0.0);
  for (int i = 0; i < ntask; i++) {
    const int s = static_cast<int>(tasks[i].isurf);
    if (s >= 0 && s < nsurf_global) area_sum_me[s] += tasks[i].area;
  }
  std::vector<double> area_sum(nsurf_global, 0.0);
  MPI_Allreduce(area_sum_me.data(), area_sum.data(), nsurf_global,
                MPI_DOUBLE, MPI_SUM, world);
  for (int i = 0; i < ntask; i++) {
    const int s = static_cast<int>(tasks[i].isurf);
    tasks[i].area_share = (s >= 0 && s < nsurf_global && area_sum[s] > 0.0) ?
                          tasks[i].area / area_sum[s] : 0.0;
  }
}

/* ----------------------------------------------------------------------
   create task for one grid cell (called by FixEmit::create_tasks)
------------------------------------------------------------------------- */

void FixSurfaceEmitRecycle::create_task(int icell)
{
  int i, m, isurf, npoint, isplit, subcell;
  double area, areaone;
  double *normal, *p1, *p2, *p3, *path;
  double cpath[36], delta[3], e1[3], e2[3];

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Grid::SplitInfo *sinfo = grid->sinfo;

  if (cells[icell].nsurf == 0) return;
  if (cinfo[icell].volume == 0.0) return;

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

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

    if (ntask == ntaskmax) grow_task();

    tasks[ntask].icell = icell;
    tasks[ntask].isurf = isurf;
    if (cells[icell].nsplit == 1) tasks[ntask].pcell = icell;
    else {
      isplit = cells[icell].isplit;
      subcell = sinfo[isplit].csplits[i];
      // subcell = -1 if the surf did not end up in a polygon (see
      // create_surfmap() in cut2d/cut3d): skip it, as fix emit/surf does
      if (subcell < 0) continue;
      tasks[ntask].pcell = sinfo[isplit].csubs[subcell];
    }

    if (dimension == 2) {
      normal = lines[isurf].norm;

      p1 = lines[isurf].p1;
      p2 = lines[isurf].p2;
      npoint = cut2d->clip_external(p1, p2, lo, hi, cpath);
      if (npoint < 2) continue;

      tasks[ntask].npoint = 2;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[6];
      path = tasks[ntask].path;
      path[0] = cpath[0]; path[1] = cpath[1]; path[2] = 0.0;
      path[3] = cpath[2]; path[4] = cpath[3]; path[5] = 0.0;

      if (domain->axisymmetric) {
        double sqrtarg = (path[1]-path[4])*(path[1]-path[4]) +
                         (path[0]-path[3])*(path[0]-path[3]);
        area = MY_PI * (path[1]+path[4]) * sqrt(sqrtarg);
      } else {
        MathExtra::sub3(&path[0], &path[3], delta);
        area = MathExtra::len3(delta);
      }
      tasks[ntask].area = area;

      tasks[ntask].tan1[0] =  normal[1];
      tasks[ntask].tan1[1] = -normal[0];
      tasks[ntask].tan1[2] =  0.0;
      tasks[ntask].tan2[0] =  0.0;
      tasks[ntask].tan2[1] =  0.0;
      tasks[ntask].tan2[2] =  1.0;

      // Segment midpoint in SPARTA coords -> physical (R, Z) for plasma
      // lookup and B2-cell mapping. SPARTA axi mode stores x=Z, y=R; legacy
      // 2D Cartesian stores x=R, y=Z. Helper picks the right slot mapping.
      const double xyz_mid[3] = { 0.5 * (path[0] + path[3]),
                                  0.5 * (path[1] + path[4]),
                                  0.0 };
      OpenEdge::sparta_to_RZ(xyz_mid, dimension, domain->axisymmetric,
                              tasks[ntask].rmid, tasks[ntask].zmid,
                              plasma->column_x0, plasma->column_y0);

      // normal[] points INTO the fluid (SPARTA canonical).
      tasks[ntask].inward[0] = normal[0];
      tasks[ntask].inward[1] = normal[1];
      tasks[ntask].inward[2] = 0.0;
      tasks[ntask].phimid = 0.0;

    } else {
      normal = tris[isurf].norm;

      p1 = tris[isurf].p1;
      p2 = tris[isurf].p2;
      p3 = tris[isurf].p3;
      npoint = cut3d->clip_external(p1, p2, p3, lo, hi, cpath);
      if (npoint < 3) continue;

      tasks[ntask].npoint = npoint;
      delete [] tasks[ntask].path;
      tasks[ntask].path = new double[npoint*3];
      path = tasks[ntask].path;
      memcpy(path, cpath, npoint*3*sizeof(double));
      delete [] tasks[ntask].fracarea;
      tasks[ntask].fracarea = new double[npoint-2];

      area = 0.0;
      p1 = &path[0];
      for (m = 0; m < npoint-2; m++) {
        p2 = &path[3*(m+1)];
        p3 = &path[3*(m+2)];
        MathExtra::sub3(p2, p1, e1);
        MathExtra::sub3(p3, p1, e2);
        MathExtra::cross3(e1, e2, delta);
        areaone = fabs(0.5 * MathExtra::len3(delta));
        area += areaone;
        tasks[ntask].fracarea[m] = area;
      }
      tasks[ntask].area = area;
      for (m = 0; m < npoint-2; m++) tasks[ntask].fracarea[m] /= area;

      delta[0] = random->uniform();
      delta[1] = random->uniform();
      delta[2] = random->uniform();
      MathExtra::cross3(tris[isurf].norm, delta, tasks[ntask].tan1);
      MathExtra::norm3(tasks[ntask].tan1);
      MathExtra::cross3(tris[isurf].norm, tasks[ntask].tan1, tasks[ntask].tan2);
      MathExtra::norm3(tasks[ntask].tan2);

      double cx = 0.0, cy = 0.0, cz = 0.0;
      for (int k = 0; k < npoint; k++) {
        cx += path[3*k + 0];
        cy += path[3*k + 1];
        cz += path[3*k + 2];
      }
      cx /= npoint; cy /= npoint; cz /= npoint;
      const double cxyz[3] = {cx, cy, cz};
      OpenEdge::sparta_to_RZ(cxyz, dimension, domain->axisymmetric,
                             tasks[ntask].rmid, tasks[ntask].zmid,
                             plasma->column_x0, plasma->column_y0);
      tasks[ntask].phimid = atan2(cy - plasma->column_y0, cx - plasma->column_x0);

      tasks[ntask].inward[0] = normal[0];
      tasks[ntask].inward[1] = normal[1];
      tasks[ntask].inward[2] = normal[2];
    }

    tasks[ntask].vscale_molec = 0.0;
    tasks[ntask].ntarget = 0.0;
    tasks[ntask].sin_alpha = 1.0;
    tasks[ntask].sin_alpha_generation = -1;
    tasks[ntask].wf_generation = -1;

    // Topological wall->plasma-cell lookup:
    // Owning B2 cell for this SPARTA wall surface. Three strategies,
    // tried in order:
    //  (1) Topological: plasma.h5 has an explicit mesh/wall_surf_cell
    //      mapping (written by the SOLPS converter when wall_b2.surf is
    //      generated from B2 boundary faces). Direct index lookup.
    //  (2) Nearest B2 cell centroid (restricted to boundary cells with
    //      wall_face_area > 0). Used when (1) is absent.
    //  (3) Nearest B2 cell centroid (all cells). Used when (2) can't be
    //      applied (no wall_face_area).
    tasks[ntask].plasma_cell = -1;
    if (plasma && plasma->has_mesh_wall_surf_cell &&
        isurf >= 0 &&
        isurf < static_cast<int>(plasma->mesh_wall_surf_cell.size())) {
      // (1) topological map from wall.surf. Only accept cells with a
      // positive face_area — otherwise the segment is on a non-emitting
      // boundary (e.g. core-side) and should NOT be mapped.
      const int c = plasma->mesh_wall_surf_cell[isurf];
      if (c >= 0 && c < plasma->mesh_ncell &&
          plasma->has_mesh_wall_face_area &&
          c < static_cast<int>(plasma->mesh_wall_face_area.size()) &&
          plasma->mesh_wall_face_area[c] > 0.0) {
        tasks[ntask].plasma_cell = c;
      }
    }
    else if (plasma && plasma->has_mesh && plasma->has_mesh_wall_face_area &&
        plasma->mesh_ntri > 0) {
      const double rm = tasks[ntask].rmid;
      const double zm = tasks[ntask].zmid;
      const int ncell = plasma->mesh_ncell;
      const int ntri = plasma->mesh_ntri;
      const int *ci = plasma->mesh_cell_idx.data();
      const int *tr = plasma->mesh_tri.data();
      const double *vr = plasma->mesh_vtx_r.data();
      const double *vz = plasma->mesh_vtx_z.data();
      const double *fa = plasma->mesh_wall_face_area.data();

      // Per-cell centroid cache (mean of the cell's B2 triangles). Keyed
      // on pd generation + ncell so a plasma reload forces a rebuild, and
      // per-instance (no static locals) so multiple fix surface/emit/recycle
      // instances don't clobber each other.
      const int pd_gen = plasma ? plasma->generation : -1;
      if (plasma_cell_centroid_gen_ != pd_gen ||
          static_cast<int>(plasma_cell_r_.size()) != ncell) {
        plasma_cell_r_.assign(ncell, 0.0);
        plasma_cell_z_.assign(ncell, 0.0);
        std::vector<int> ncount(ncell, 0);
        for (int t = 0; t < ntri; t++) {
          const int c = ci[t];
          if (c < 0 || fa[c] <= 0.0) continue;
          const int v0 = tr[3*t+0], v1 = tr[3*t+1], v2 = tr[3*t+2];
          plasma_cell_r_[c] += (vr[v0] + vr[v1] + vr[v2]) / 3.0;
          plasma_cell_z_[c] += (vz[v0] + vz[v1] + vz[v2]) / 3.0;
          ncount[c]++;
        }
        for (int c = 0; c < ncell; c++) {
          if (ncount[c] > 0) {
            plasma_cell_r_[c] /= ncount[c];
            plasma_cell_z_[c] /= ncount[c];
          }
        }
        plasma_cell_centroid_gen_ = pd_gen;
      }

      double dmin2 = std::numeric_limits<double>::infinity();
      int best = -1;
      for (int c = 0; c < ncell; c++) {
        if (fa[c] <= 0.0) continue;
        const double dr = plasma_cell_r_[c] - rm;
        const double dz = plasma_cell_z_[c] - zm;
        const double d2 = dr*dr + dz*dz;
        if (d2 < dmin2) { dmin2 = d2; best = c; }
      }
      if (best >= 0) tasks[ntask].plasma_cell = best;
    }

    ntask++;
  }
}

/* ----------------------------------------------------------------------
   per-surface emission rate (particles / step) from local Bohm flux
------------------------------------------------------------------------- */

double FixSurfaceEmitRecycle::emission_rate_per_surface(int itask)
{
  Task &t = tasks[itask];
  const double R = t.rmid;
  const double Z = t.zmid;

  // |B.n|/|B| at the wall midpoint, cached per plasma generation (also the
  // impact angle handed to the pwi model)
  if (t.sin_alpha_generation != plasma->generation) {
    double sin_alpha = 1.0;
    if (plasma->has_bfield) {
      double Br, Bz, Bt;
      plasma->bfield_at(R, Z, Br, Bz, Bt, t.icell);
      const double Bmag = std::sqrt(Br*Br + Bz*Bz + Bt*Bt);
      if (Bmag > 0.0) {
        // (Br,Bz,Bt) into the SPARTA slots at the element (axi: x=Z, y=R;
        // 3-D: rotated by the element azimuth), then dot the inward normal
        double bx, by, bz;
        OpenEdge::RZphi_force_to_sparta(Br, Bz, Bt, domain->dimension,
                                        domain->axisymmetric, t.phimid, bx, by, bz);
        const double proj = bx * t.inward[0] + by * t.inward[1] + bz * t.inward[2];
        sin_alpha = std::fabs(proj) / Bmag;
        if (sin_alpha > 1.0) sin_alpha = 1.0;
      }
    }
    t.sin_alpha = sin_alpha;
    t.sin_alpha_generation = plasma->generation;
  }

  double gamma = 0.0;        // ion flux density into the wall [m^-2 s^-1]
  double gamma_area = 0.0;   // area it applies to [m^2]
  bool have_gamma = false;

  // (a) plasma-code wall tallies: /wall_flux/ samples of the background,
  //     3-point inverse-distance weighted within wall_flux_cutoff; the
  //     flux is a density, so it multiplies the geometric element area
  if (plasma->has_wall_flux()) {
    if (t.wf_generation != plasma->generation) {
      t.wf_gamma = plasma->query_wall_flux_at_point(R, Z, wall_flux_ispec,
                                                    wall_flux_cutoff);
      double te = 0.0, ti = 0.0;
      if (!plasma->query_wall_te_ti_at_point(R, Z, te, ti, wall_flux_cutoff))
        te = ti = 0.0;
      t.wf_te = te;
      t.wf_ti = ti;
      t.wf_generation = plasma->generation;
    }
    if (t.wf_gamma > 0.0) {
      gamma = t.wf_gamma;
      gamma_area = t.area;
      t.te_imp = t.wf_te;
      t.ti_imp = t.wf_ti;
      have_gamma = true;
    }
  }

  // (b) Bohm reconstruction from the wall-adjacent plasma cell:
  //     Gamma = ne cs sin(alpha_B), cs = sqrt((Te+Ti)/mi) (Stangeby ch. 2),
  //     over the plasma-code wall face area when the converter wrote one
  if (!have_gamma) {
    const int cell = t.plasma_cell;
    if (cell < 0) return 0.0;
    // Defensive bounds check: a plasma reload with a smaller mesh between
    // grid-triggered task rebuilds could leave cell >= mesh_ncell here.
    if (cell >= plasma->mesh_ncell) return 0.0;

    const double ne = plasma->mesh_ne[cell];
    const double te = plasma->mesh_te[cell];
    const double ti = plasma->mesh_ti.empty() ? te : plasma->mesh_ti[cell];
    if (!std::isfinite(ne) || !std::isfinite(te)) return 0.0;
    if (ne <= 0.0 || te <= 0.0) return 0.0;
    const double ti_eff = std::isfinite(ti) && ti > 0.0 ? ti : te;
    const double cs_arg = (te + ti_eff) * QE / (mass_amu * AMU);
    if (cs_arg <= 0.0) return 0.0;
    const double cs = std::sqrt(cs_arg);
    t.te_imp = te;
    t.ti_imp = ti_eff;

    // Surface area used in the flux calculation, in preference order:
    //  (1) Per-wall-segment aggregated area mesh/wall_surf_area[isurf] -
    //      sums the B2 face area of every B2 boundary face that chose this
    //      wall segment as its nearest. Preserves the full SOLPS wall flux
    //      budget when the SPARTA wall is coarser than the B2 grid (multiple
    //      B2 cells per wall segment). Multiplied by area_share so refined
    //      sub-segments split it correctly.
    //  (2) Per-cell B2 face area (legacy: when only mesh_wall_face_area
    //      is available, before the converter started writing surf_area).
    //  (3) Raw SPARTA surface area (approximate, for non-SOLPS tests).
    const int isurf_i = static_cast<int>(t.isurf);
    if (plasma->has_mesh_wall_surf_cell &&
        isurf_i >= 0 &&
        isurf_i < static_cast<int>(plasma->mesh_wall_surf_area.size()) &&
        plasma->mesh_wall_surf_area[isurf_i] > 0.0) {
      gamma_area = plasma->mesh_wall_surf_area[isurf_i] * t.area_share;
    } else if (plasma->has_mesh_wall_face_area &&
               cell < static_cast<int>(plasma->mesh_wall_face_area.size()) &&
               plasma->mesh_wall_face_area[cell] > 0.0) {
      gamma_area = plasma->mesh_wall_face_area[cell] * t.area_share;
    } else {
      gamma_area = t.area;
    }
    gamma = ne * cs * t.sin_alpha;
  }

  // thermal mixture emission: 0.5 R Gamma A (two atoms make one molecule);
  // pwi mode: the full ion flux, the surf_react applies R and the 2 -> 1
  const double dot_N = pwi ? gamma * gamma_area
                           : 0.5 * R_recycle * gamma * gamma_area;

  // Per-cell particle weighting (stock fix_emit_surf does the same -
  // line 573 of src/fix_emit_surf.cpp). With `global weight cell` in
  // axisymmetric mode or any user-imposed variable cell weight, a cell
  // with weight w represents w physical particles per sim particle, so
  // sim-particle injection rate must divide by w.
  const int icell = t.icell;
  const double wcell = (icell >= 0) ? grid->cinfo[icell].weight : 1.0;
  const double inv_wcell = (wcell > 0.0) ? 1.0 / wcell : 1.0;
  // scale by nevery: firing every N steps must inject N steps of flux
  // (matches fix_surface_emit_source; was under-counting by 1/nevery)
  return dot_N * update->dt * nevery * inv_wcell / fnum;
}

/* ----------------------------------------------------------------------
   perform one step's insertion for every task
------------------------------------------------------------------------- */

void FixSurfaceEmitRecycle::perform_task()
{
  // Rebuild tasks if the plasma mesh reloaded since we built them
  // (pd->generation bumps on every FixBackground::reload). For DIII-D
  // with `static yes` this check is a single int compare per step and
  // never triggers a rebuild. For dynamic plasma (SOLPS coupling) it's
  // what keeps tasks[].plasma_cell valid across reloads.
  if (plasma && plasma->generation != plasma_generation_) {
    grid_changed();
    plasma_generation_ = plasma->generation;
  }

  int i, m, n, pcell, isurf, ninsert, nactual, isp, ispecies, ntri, id;
  double rn, ntarget, vr, alpha, beta, theta, erot, evib;
  double vnmag, vamag, vbmag;
  double *normal, *p1, *p2, *p3, *atan, *btan;
  double x[3], v[3], e1[3], e2[3];
  Particle::OnePart *p;

  const double dt = update->dt;
  int *species = particle->mixture[imix]->species;

  Surf::Line *lines = surf->lines;
  Surf::Tri  *tris  = surf->tris;

  int nsurf_tally = update->nsurf_tally;
  Compute **slist_active = update->slist_active;
  int nfix_update_custom = modify->n_update_custom;

  // pwi mode: ions retained by the wall are removed after the loop
  int *dellist = NULL;
  int ndel = 0, maxdel = 0;

  for (i = 0; i < ntask; i++) {
    pcell = tasks[i].pcell;
    isurf = tasks[i].isurf;
    if (isurf >= surf->nlocal) error->one(FLERR,"Fix surface/emit/recycle: task surf index out of range");

    if (dimension == 2) normal = lines[isurf].norm;
    else normal = tris[isurf].norm;
    atan = tasks[i].tan1;
    btan = tasks[i].tan2;

    ntarget = emission_rate_per_surface(i);
    if (ntarget <= 0.0) continue;

    ninsert = static_cast<int>(ntarget + random->uniform());
    // Bootstrap: at step 1 the halt-on-zero-particles check in update.cpp
    if (ninsert <= 0) continue;

    nactual = 0;
    for (m = 0; m < ninsert; m++) {
      rn = random->uniform();
      isp = 0;
      while (isp < nspecies - 1 && cummulative[isp] < rn) isp++;
      ispecies = species[isp];

      if (dimension == 2) {
        rn = random->uniform();
        p1 = &tasks[i].path[0];
        p2 = &tasks[i].path[3];
        x[0] = p1[0] + rn * (p2[0] - p1[0]);
        x[1] = p1[1] + rn * (p2[1] - p1[1]);
        x[2] = 0.0;
      } else {
        rn = random->uniform();
        ntri = tasks[i].npoint - 2;
        for (n = 0; n < ntri; n++)
          if (rn < tasks[i].fracarea[n]) break;
        p1 = &tasks[i].path[0];
        p2 = &tasks[i].path[3*(n+1)];
        p3 = &tasks[i].path[3*(n+2)];
        MathExtra::sub3(p2, p1, e1);
        MathExtra::sub3(p3, p1, e2);
        alpha = random->uniform();
        beta  = random->uniform();
        if (alpha + beta > 1.0) { alpha = 1.0 - alpha; beta = 1.0 - beta; }
        x[0] = p1[0] + alpha*e1[0] + beta*e2[0];
        x[1] = p1[1] + alpha*e1[1] + beta*e2[1];
        x[2] = p1[2] + alpha*e1[2] + beta*e2[2];
      }

      if (region && !region->match(x)) continue;

      if (pwi) {
        // incident ion at the sheath impact energy, coming in along the
        // field line (angle from the normal = 90 - alpha_B, within the
        // TRIM table range), handed to the surface/pwi lottery: reflected
        // as a fast atom, re-emitted thermally with R, or retained
        const double mion = particle->species[pwi_species].mass;
        const double E_th = 2.0 * std::max(tasks[i].ti_imp, 0.0);
        const double E_sh = impact_energy_eV(tasks[i].te_imp, tasks[i].ti_imp) - E_th;
        const double phi = MY_2PI * random->uniform();
        const double ca = std::cos(phi), sa = std::sin(phi);
        double sin_a = std::min(1.0, std::max(tasks[i].sin_alpha, 0.0));
        const double cos_a = std::sqrt(std::max(0.0, 1.0 - sin_a*sin_a));
        if (impact_model == IMPACT_MAXWELL) {
          // Maxwellian flux at Ti drifting at the isothermal sound speed
          // along the field line into the wall; the floating-sheath drop
          // of that flux, Te/2 ln(mi/(pi me (1+Ti/Te))), is added to the
          // normal velocity component only
          const double ti = std::max(tasks[i].ti_imp, 1.0e-3);
          const double te = std::max(tasks[i].te_imp, 0.0);
          const double sig = std::sqrt(ti * QE / mion);
          const double c_s = std::sqrt((te + ti) * QE / mion);
          double e_sh = 0.0;
          if (te > 0.0) {
            const double arg = (mion / ME) / (MY_PI * (1.0 + ti / te));
            e_sh = ion_charge * 0.5 * std::log(arg) * te;
            if (e_sh < 0.0) e_sh = 0.0;
          }
          const double u_n = c_s * sin_a;            // drift into the wall
          const double u_t = c_s * cos_a;            // drift along the wall
          const double vn_max = u_n + 4.0 * sig;
          double vn = 0.0;
          for (int it = 0; it < 200; it++) {
            const double g = random->gaussian();
            vn = u_n + sig * g;
            if (vn > 0.0 && random->uniform() * vn_max < vn) break;
            vn = 0.0;
          }
          if (vn <= 0.0) vn = std::max(u_n, sig);
          vn = std::sqrt(vn*vn + 2.0 * e_sh * QE / mion);
          const double vt1 = u_t + sig * random->gaussian();
          const double vt2 = sig * random->gaussian();
          for (int k = 0; k < 3; k++)
            v[k] = -(vn * normal[k] + vt1 * (ca*atan[k] + sa*btan[k])
                     + vt2 * (-sa*atan[k] + ca*btan[k]));
        } else {
          // thermal part 2 Ti along the field line (normal component
          // sin_alpha), sheath part Z e phi_sh added to the normal motion:
          // total energy = impact_energy_eV, angle bent toward the normal
          const double v_th = std::sqrt(2.0 * std::max(E_th, 1.0e-3) * QE / mion);
          const double v_sh2 = 2.0 * std::max(E_sh, 0.0) * QE / mion;
          const double v_tan = v_th * cos_a;
          const double v_nrm = std::sqrt(v_th*sin_a*v_th*sin_a + v_sh2);
          for (int k = 0; k < 3; k++)
            v[k] = -(v_nrm * normal[k] + v_tan * (ca*atan[k] + sa*btan[k]));
        }
        id = MAXSMALLINT * random->uniform();
        particle->add_particle(id, pwi_species, pcell, x, v, 0.0, 0.0);
        const int ipart = particle->nlocal - 1;
        p = &particle->particles[ipart];
        p->flag = PSURF + 1 + isurf;
        p->dtremain = dt * random->uniform();
        if (nfix_update_custom)
          modify->update_custom(ipart, tasks[i].te_imp, tasks[i].te_imp,
                                tasks[i].te_imp, v);

        Particle::OnePart *ip = p;
        Particle::OnePart *jp = NULL;
        int velreset = 0;
        pwi->react(ip, isurf, normal, jp, velreset);
        if (ip == NULL) {
          if (ndel == maxdel) {
            maxdel += 1024;
            memory->grow(dellist, maxdel, "surface/emit/recycle:dellist");
          }
          dellist[ndel++] = ipart;
        } else {
          nactual++;
          if (nsurf_tally)
            for (int k = 0; k < nsurf_tally; k++)
              slist_active[k]->surf_tally(ip->dtremain, isurf, pcell, 0, NULL, ip, NULL);
        }
        continue;
      }

      // Maxwellian flux velocity at per-species twall, inward along normal.
      // v_perp ~ Rayleigh ~ sqrt(-ln U) * vscale
      // v_parallel ~ two independent Gaussians (tangential)
      const double mspec = particle->species[ispecies].mass;
      const double t_emit = twall_by_species[isp];
      const double vscale_twall = std::sqrt(2.0 * KB * t_emit / mspec);

      vnmag = vscale_twall * std::sqrt(-std::log(random->uniform()));

      theta = MY_2PI * random->uniform();
      vr = vscale_twall * std::sqrt(-std::log(random->uniform()));
      vamag = vr * std::sin(theta);
      vbmag = vr * std::cos(theta);

      // SPARTA canonical: normal[] points INTO the fluid (matches
      // stock fix_emit_surf, surf_collide_diffuse, surf_react surface/pwi).
      // Emission into the fluid = +vnmag * normal.
      v[0] = vnmag*normal[0] + vamag*atan[0] + vbmag*btan[0];
      v[1] = vnmag*normal[1] + vamag*atan[1] + vbmag*btan[1];
      v[2] = vnmag*normal[2] + vamag*atan[2] + vbmag*btan[2];

      erot = particle->erot(ispecies, t_emit, random);
      evib = particle->evib(ispecies, t_emit, random);
      id = MAXSMALLINT * random->uniform();

      particle->add_particle(id, ispecies, pcell, x, v, erot, evib);
      nactual++;

      p = &particle->particles[particle->nlocal-1];
      p->flag = PSURF + 1 + isurf;
      p->dtremain = dt * random->uniform();

      if (nsurf_tally)
        for (int k = 0; k < nsurf_tally; k++)
          slist_active[k]->surf_tally(p->dtremain, isurf, pcell, 0, NULL, p, NULL);

      if (nfix_update_custom)
        modify->update_custom(particle->nlocal-1, t_emit, t_emit, t_emit, v);
    }
    nsingle += nactual;
  }

  if (ndel) particle->compress_reactions(ndel, dellist);
  memory->destroy(dellist);
}

/* ----------------------------------------------------------------------
   grow task list
------------------------------------------------------------------------- */

void FixSurfaceEmitRecycle::grow_task()
{
  int oldmax = ntaskmax;
  ntaskmax += DELTATASK;
  tasks = (Task *) memory->srealloc(tasks, ntaskmax*sizeof(Task),
                                    "surface/emit/recycle:tasks");
  memset(&tasks[oldmax], 0, (ntaskmax-oldmax)*sizeof(Task));

  for (int i = oldmax; i < ntaskmax; i++) {
    tasks[i].path = NULL;
    tasks[i].fracarea = NULL;
    tasks[i].wf_generation = -1;
    tasks[i].sin_alpha_generation = -1;
  }
}

/* ----------------------------------------------------------------------
   keyword options
------------------------------------------------------------------------- */

int FixSurfaceEmitRecycle::option(int narg, char **arg)
{
  if (strcmp(arg[0], "mass") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    mass_amu = atof(arg[1]);
    if (mass_amu <= 0.0)
      error->all(FLERR,"fix surface/emit/recycle mass must be > 0");
    return 2;
  }
  if (strcmp(arg[0], "R") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    R_recycle = atof(arg[1]);
    if (R_recycle < 0.0 || R_recycle > 1.0)
      error->all(FLERR,"fix surface/emit/recycle R must be in [0,1]");
    return 2;
  }
  if (strcmp(arg[0], "twall") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    twall = atof(arg[1]);
    if (twall <= 0.0)
      error->all(FLERR,"fix surface/emit/recycle twall must be > 0");
    return 2;
  }
  if (strcmp(arg[0], "pwi") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    delete [] pwi_id;
    int n = strlen(arg[1]) + 1;
    pwi_id = new char[n];
    strcpy(pwi_id, arg[1]);
    return 2;
  }
  if (strcmp(arg[0], "impact") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    if (strcmp(arg[1], "sheath") == 0) impact_model = IMPACT_SHEATH;
    else if (strcmp(arg[1], "maxwell") == 0) impact_model = IMPACT_MAXWELL;
    else error->all(FLERR,"fix surface/emit/recycle impact must be sheath|maxwell");
    return 2;
  }
  if (strcmp(arg[0], "wall_flux_cutoff") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    wall_flux_cutoff = atof(arg[1]);
    if (wall_flux_cutoff <= 0.0)
      error->all(FLERR,"fix surface/emit/recycle wall_flux_cutoff must be > 0");
    return 2;
  }
  if (strcmp(arg[0], "wall_flux_species") == 0) {
    if (2 > narg) error->all(FLERR,"Illegal fix surface/emit/recycle command");
    wall_flux_ispec = atoi(arg[1]) - 1;
    if (wall_flux_ispec < 0)
      error->all(FLERR,"fix surface/emit/recycle wall_flux_species must be >= 1");
    return 2;
  }
  if (strcmp(arg[0], "twall_species") == 0) {
    // twall_species <sp1> <T1> [<sp2> <T2> ...]
    // Consume pairs until end of args or a non-pair (next keyword) shows up.
    // Each Ti replaces the scalar twall for species name spi in the emission
    // sampler. Unnamed species fall back to the scalar twall.
    int consumed = 1;
    while (consumed + 1 < narg) {
      const char *name = arg[consumed];
      // stop if name is actually a keyword this fix would recognise next
      if (strcmp(name, "mass") == 0 || strcmp(name, "R") == 0 ||
          strcmp(name, "twall") == 0 || strcmp(name, "twall_species") == 0 ||
          strcmp(name, "pwi") == 0 || strcmp(name, "wall_flux_cutoff") == 0 ||
          strcmp(name, "wall_flux_species") == 0 || strcmp(name, "nevery") == 0 ||
          strcmp(name, "region") == 0)
        break;
      const double t = atof(arg[consumed + 1]);
      if (t <= 0.0)
        error->all(FLERR,"fix surface/emit/recycle twall_species T must be > 0");
      twall_species_names.push_back(std::string(name));
      twall_species_values.push_back(t);
      consumed += 2;
    }
    if (consumed == 1)
      error->all(FLERR,"fix surface/emit/recycle twall_species needs at least one <sp> <T> pair");
    return consumed;
  }

  error->all(FLERR,"Illegal fix surface/emit/recycle command");
  return 0;
}
