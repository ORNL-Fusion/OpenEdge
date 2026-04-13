/* ----------------------------------------------------------------------
   OpenEdge: nearest wall geometry per grid cell for sheath workflows
------------------------------------------------------------------------- */

#include "compute_sheath_geometry_grid.h"

#include <cstring>

#include "update.h"
#include "grid.h"
#include "surf.h"
#include "domain.h"
#include "geometry.h"
#include "input.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

namespace {
constexpr double DIST_BIG = 1.0e20;
}

ComputeSheathGeometryGrid::ComputeSheathGeometryGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal compute sheath/geometry/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute sheath/geometry/grid grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  igroup = surf->find_group(arg[3]);
  if (igroup < 0) error->all(FLERR,"Compute sheath/geometry/grid surface group ID does not exist");
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
    else error->all(FLERR,"Illegal compute sheath/geometry/grid command");
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

ComputeSheathGeometryGrid::~ComputeSheathGeometryGrid()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  memory->destroy(midx_grid);
}

void ComputeSheathGeometryGrid::init()
{
  reallocate();
}

void ComputeSheathGeometryGrid::compute_per_grid()
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

  int *eligible = nullptr;
  memory->create(eligible,nsurf_all,"sheath/geometry/grid:eligible");
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

    const double *lo = cells[icell].lo;
    const double *hi = cells[icell].hi;
    double lom[3] = {lo[0], lo[1], lo[2]};
    double him[3] = {hi[0], hi[1], hi[2]};
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
    double mind = DIST_BIG;
    int midx = -1;
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

    double nx = 0.0, ny = 0.0, nz = 0.0;
    double sid = -1.0;
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
  computed_once = 1;
}

void ComputeSheathGeometryGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  memory->destroy(midx_grid);
  nglocal = grid->nlocal;
  if (nvalue == 1) memory->create(vector_grid,nglocal,"sheath/geometry/grid:vector");
  else memory->create(array_grid,nglocal,nvalue,"sheath/geometry/grid:array");
  memory->create(midx_grid,nglocal,"sheath/geometry/grid:midx");
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

bigint ComputeSheathGeometryGrid::memory_usage()
{
  bigint bytes = 0;
  if (nvalue == 1) bytes += nglocal * sizeof(double);
  else bytes += static_cast<bigint>(nglocal) * nvalue * sizeof(double);
  bytes += static_cast<bigint>(nglocal) * sizeof(int);
  return bytes;
}
