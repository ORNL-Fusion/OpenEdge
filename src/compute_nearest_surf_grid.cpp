/* ----------------------------------------------------------------------
   OpenEdge: nearest wall surface per grid cell.
   For each cell in <grid-group>, finds the closest surface in
   <surf-group> and exposes distance / surface ID / outward normal
   components. Used by sheath models, near-wall diagnostics, and any
   compute or fix that needs cell-local wall geometry.
------------------------------------------------------------------------- */

#include <cmath>
#include "mpi.h"
#include "compute_nearest_surf_grid.h"

#include <cstring>

#include "update.h"
#include "grid.h"
#include "surf.h"
#include "domain.h"
#include "geometry.h"
#include "input.h"
#include "math_extra.h"
#include "memory.h"
#include "comm.h"
#include "error.h"

using namespace SPARTA_NS;

namespace {
constexpr double DIST_BIG = 1.0e20;
}

ComputeNearestSurfGrid::ComputeNearestSurfGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal compute nearest_surf/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute nearest_surf/grid grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  igroup = surf->find_group(arg[3]);
  if (igroup < 0) error->all(FLERR,"Compute nearest_surf/grid surface group ID does not exist");
  sgroupbit = surf->bitmask[igroup];

  nvalue = narg - 4;
  value = new int[nvalue];
  int iarg = 4;
  int iv = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"dist") == 0) value[iv] = DIST;
    else if (strcmp(arg[iarg],"surfid") == 0) value[iv] = SURFID;
    else if (strcmp(arg[iarg],"nx") == 0) value[iv] = NX;
    else if (strcmp(arg[iarg],"ny") == 0) value[iv] = NY;
    else if (strcmp(arg[iarg],"nz") == 0) value[iv] = NZ;
    else if (strcmp(arg[iarg],"surfidx") == 0) value[iv] = SURFIDX;
    else error->all(FLERR,"Illegal compute nearest_surf/grid command");
    ++iarg;
    ++iv;
  }

  per_grid_flag = 1;
  size_per_grid_cols = (nvalue == 1) ? 0 : nvalue;
  nglocal = 0;
  vector_grid = nullptr;
  array_grid = nullptr;
  midx_grid = nullptr;
  computed_once = 0;
}

ComputeNearestSurfGrid::~ComputeNearestSurfGrid()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  memory->destroy(midx_grid);
}

void ComputeNearestSurfGrid::init()
{
  reallocate();
  ensure_customs();
}

/* grid custom attributes carrying the per-cell result (created once; a
   restart file restores them, find_custom then succeeds) */

void ComputeNearestSurfGrid::ensure_customs()
{
  if (cidx_dist_ >= 0) return;
  cidx_dist_ = grid->find_custom((char *) "nsg_dist");
  if (cidx_dist_ < 0) cidx_dist_ = grid->add_custom((char *) "nsg_dist",1,0);   // DOUBLE
  cidx_n_ = grid->find_custom((char *) "nsg_n");
  if (cidx_n_ < 0) cidx_n_ = grid->add_custom((char *) "nsg_n",1,3);
  cidx_midx_ = grid->find_custom((char *) "nsg_midx");
  if (cidx_midx_ < 0) cidx_midx_ = grid->add_custom((char *) "nsg_midx",0,0);    // INT
  cidx_id_ = grid->find_custom((char *) "nsg_id");
  if (cidx_id_ < 0) cidx_id_ = grid->add_custom((char *) "nsg_id",1,0);        // cell id as double (exact to 2^53)
}

void ComputeNearestSurfGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  // geometry is static: grid and surface don't change, so cache after first eval
  if (computed_once) return;
  const int dim = domain->dimension;
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  Surf::Line *lines = surf->lines;
  Surf::Tri *tris = surf->tris;
  const int nsurf_all = surf->nsurf;

  // cached results travel with the cells (grid customs); usable when the
  // surf array index is global (surfs not distributed)
  ensure_customs();
  custom_sync_host();
  double *c_dist = grid->edvec[grid->ewhich[cidx_dist_]];
  double **c_n = grid->edarray[grid->ewhich[cidx_n_]];
  int *c_midx = grid->eivec[grid->ewhich[cidx_midx_]];
  double *c_id = grid->edvec[grid->ewhich[cidx_id_]];
  const int cache_ok = surf->distributed ? 0 : 1;
  bigint ncached = 0, ncomputed = 0;

  int *eligible = nullptr;
  memory->create(eligible,nsurf_all,"nearest_surf/grid:eligible");
  int neligible = 0;
  for (int i = 0; i < nsurf_all; ++i) {
    int ok = 0;
    if (dim == 2) ok = (lines[i].mask & sgroupbit) ? 1 : 0;
    else ok = (tris[i].mask & sgroupbit) ? 1 : 0;
    eligible[i] = ok;
    if (ok) ++neligible;
  }

  for (int icell = 0; icell < nglocal; ++icell) {
    if (!(cinfo[icell].mask & groupbit) || cells[icell].nsplit < 1) {
      if (nvalue == 1) vector_grid[icell] = 0.0;
      else for (int j = 0; j < nvalue; ++j) array_grid[icell][j] = 0.0;
      midx_grid[icell] = -1;
      continue;
    }

    double mind = DIST_BIG;
    int midx = -1;
    double nx = 0.0, ny = 0.0, nz = 0.0;
    double sid = -1.0;
    const bool cached = cache_ok && c_id[icell] == (double) cells[icell].id &&
                        c_midx[icell] >= -1 && c_midx[icell] < nsurf_all;
    if (cached) {
      mind = c_dist[icell]; midx = c_midx[icell];
      nx = c_n[icell][0]; ny = c_n[icell][1]; nz = c_n[icell][2];
      if (midx >= 0) sid = (dim == 2) ? (double) lines[midx].id : (double) tris[midx].id;
      ncached++;
    } else {
      ncomputed++;

    const double *lo = cells[icell].lo;
    const double *hi = cells[icell].hi;
    const double ctr[3] = {
      0.5 * (lo[0] + hi[0]),
      0.5 * (lo[1] + hi[1]),
      (dim == 3) ? 0.5 * (lo[2] + hi[2]) : 0.0
    };

    // For sheath workflows, use perpendicular distance from cell center to the
    // triangle plane.  This correctly selects the plasma-facing surface even when
    // multiple triangles (top, bottom, side faces of a slab) have similar
    // bounding-box distances — which happens whenever a cell is much larger than
    // the surface feature (e.g. 1×1×N grid with a thin slab surface).
    for (int m = 0; m < nsurf_all; ++m) {
      if (!eligible[m]) continue;
      double d = DIST_BIG;
      if (dim == 2) {
        // 2D: perpendicular distance from cell center to the line
        const double lnx = lines[m].norm[0];
        const double lny = lines[m].norm[1];
        d = std::fabs((ctr[0] - lines[m].p1[0]) * lnx +
                      (ctr[1] - lines[m].p1[1]) * lny);
      } else {
        // 3D: perpendicular distance from cell center to the triangle plane
        const double *tn = tris[m].norm;
        d = std::fabs((ctr[0] - tris[m].p1[0]) * tn[0] +
                      (ctr[1] - tris[m].p1[1]) * tn[1] +
                      (ctr[2] - tris[m].p1[2]) * tn[2]);
      }
      if (d < mind) {
        mind = d;
        midx = m;
      }
    }

    if (midx >= 0) {
      if (dim == 2) {
        nx = lines[midx].norm[0];
        ny = lines[midx].norm[1];
        nz = 0.0;
        sid = static_cast<double>(lines[midx].id);
      } else {
        nx = tris[midx].norm[0];
        ny = tris[midx].norm[1];
        nz = tris[midx].norm[2];
        sid = static_cast<double>(tris[midx].id);
      }
      // orient normal toward cell center for consistent sign.
      double sctr[3] = {0.0, 0.0, 0.0};
      if (dim == 2) {
        sctr[0] = 0.5 * (lines[midx].p1[0] + lines[midx].p2[0]);
        sctr[1] = 0.5 * (lines[midx].p1[1] + lines[midx].p2[1]);
      } else {
        sctr[0] = (tris[midx].p1[0] + tris[midx].p2[0] + tris[midx].p3[0]) / 3.0;
        sctr[1] = (tris[midx].p1[1] + tris[midx].p2[1] + tris[midx].p3[1]) / 3.0;
        sctr[2] = (tris[midx].p1[2] + tris[midx].p2[2] + tris[midx].p3[2]) / 3.0;
      }
      double v[3] = {ctr[0]-sctr[0], ctr[1]-sctr[1], ctr[2]-sctr[2]};
      double nvec[3] = {nx, ny, nz};
      if (MathExtra::dot3(v, nvec) < 0.0) { nx = -nx; ny = -ny; nz = -nz; }
    }
    c_dist[icell] = mind; c_midx[icell] = midx;
    c_n[icell][0] = nx; c_n[icell][1] = ny; c_n[icell][2] = nz;
    c_id[icell] = (double) cells[icell].id;
    }   // computed

    midx_grid[icell] = midx;

    if (nvalue == 1) {
      if (value[0] == DIST) vector_grid[icell] = (midx >= 0) ? mind : DIST_BIG;
      else if (value[0] == SURFID) vector_grid[icell] = sid;
      else if (value[0] == NX) vector_grid[icell] = nx;
      else if (value[0] == NY) vector_grid[icell] = ny;
      else if (value[0] == NZ) vector_grid[icell] = nz;
      else if (value[0] == SURFIDX) vector_grid[icell] = static_cast<double>(midx);
    } else {
      for (int j = 0; j < nvalue; ++j) {
        if (value[j] == DIST) array_grid[icell][j] = (midx >= 0) ? mind : DIST_BIG;
        else if (value[j] == SURFID) array_grid[icell][j] = sid;
        else if (value[j] == NX) array_grid[icell][j] = nx;
        else if (value[j] == NY) array_grid[icell][j] = ny;
        else if (value[j] == NZ) array_grid[icell][j] = nz;
        else if (value[j] == SURFIDX) array_grid[icell][j] = static_cast<double>(midx);
      }
    }
  }

  memory->destroy(eligible);
  custom_modify_host();
  computed_once = 1;
  // no collective here: this compute runs only on the ranks whose cells changed
  // (reallocate), so a reduction would mismatch other ranks' collectives
  // (Cray MPICH aborted with "message sizes do not match" at a rebalance)
  if (comm->me == 0 && screen && ncomputed > 0)
    fprintf(screen,"  nearest_surf/grid (rank 0): " BIGINT_FORMAT " cells from the migrated cache, "
            BIGINT_FORMAT " computed\n",ncached,ncomputed);
}

void ComputeNearestSurfGrid::reallocate()
{
  // Change detection: cell count AND first-cell id (a balance/migration
  // can relabel cells while keeping nlocal equal on a rank).
  const cellint id0 = (grid->nlocal > 0 && grid->cells) ? grid->cells[0].id : -1;
  if (grid->nlocal == nglocal && id0 == stamp_id0_) return;
  stamp_id0_ = id0;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  memory->destroy(midx_grid);
  nglocal = grid->nlocal;
  if (nvalue == 1) memory->create(vector_grid,nglocal,"nearest_surf/grid:vector");
  else memory->create(array_grid,nglocal,nvalue,"nearest_surf/grid:array");
  memory->create(midx_grid,nglocal,"nearest_surf/grid:midx");
  computed_once = 0;
  if (nvalue == 1 && vector_grid) {
    for (int i = 0; i < nglocal; i++) vector_grid[i] = 0.0;
  } else if (array_grid) {
    for (int i = 0; i < nglocal; i++)
      for (int j = 0; j < nvalue; j++) array_grid[i][j] = 0.0;
  }
  if (midx_grid) {
    for (int i = 0; i < nglocal; i++) midx_grid[i] = -1;
  }
}

bigint ComputeNearestSurfGrid::memory_usage()
{
  bigint bytes = 0;
  if (nvalue == 1) bytes += nglocal * sizeof(double);
  else bytes += static_cast<bigint>(nglocal) * nvalue * sizeof(double);
  bytes += static_cast<bigint>(nglocal) * sizeof(int);
  return bytes;
}
