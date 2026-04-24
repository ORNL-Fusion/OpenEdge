/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute grid/weighted — like compute grid but uses per-particle
    pweight custom attribute instead of global fnum.

    Supported values:
      nrho_w    — weighted number density   = sum(pweight) * cellweight / V
      massrho_w — weighted mass density     = sum(pweight*mass) * cellweight / V
      n_w       — weighted count            = sum(pweight)
      pxrho_w   — weighted x-momentum density
      pyrho_w   — weighted y-momentum density
      pzrho_w   — weighted z-momentum density
      kerho_w   — weighted kinetic energy density
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_grid_weighted.h"
#include "particle.h"
#include "mixture.h"
#include "grid.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

// user keywords

enum{N_W,NRHO_W,MASSRHO_W,PXRHO_W,PYRHO_W,PZRHO_W,KERHO_W};

// internal accumulators

enum{WCOUNT,WMASSSUM,WMVX,WMVY,WMVZ,WMVSQ,LASTSIZE};

#define MAXACCUMULATE 2

/* ---------------------------------------------------------------------- */

ComputeGridWeighted::ComputeGridWeighted(SPARTA *sparta, int narg,
                                         char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5) error->all(FLERR,"Illegal compute grid/weighted command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute grid/weighted group ID does not exist");
  groupbit = grid->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0)
    error->all(FLERR,"Compute grid/weighted mixture ID does not exist");
  ngroup = particle->mixture[imix]->ngroup;

  nvalue = narg - 4;
  value = new int[nvalue];

  npergroup = 0;
  unique = new int[LASTSIZE];
  nmap = new int[nvalue];
  memory->create(map,ngroup*nvalue,MAXACCUMULATE,"grid_weighted:map");
  for (int i = 0; i < nvalue; i++) nmap[i] = 0;

  int ivalue = 0;
  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"n_w") == 0) {
      value[ivalue] = N_W;
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"nrho_w") == 0) {
      value[ivalue] = NRHO_W;
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"massrho_w") == 0) {
      value[ivalue] = MASSRHO_W;
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"pxrho_w") == 0) {
      value[ivalue] = PXRHO_W;
      set_map(ivalue,WMVX);
    } else if (strcmp(arg[iarg],"pyrho_w") == 0) {
      value[ivalue] = PYRHO_W;
      set_map(ivalue,WMVY);
    } else if (strcmp(arg[iarg],"pzrho_w") == 0) {
      value[ivalue] = PZRHO_W;
      set_map(ivalue,WMVZ);
    } else if (strcmp(arg[iarg],"kerho_w") == 0) {
      value[ivalue] = KERHO_W;
      set_map(ivalue,WMVSQ);
    } else error->all(FLERR,"Illegal compute grid/weighted command");

    ivalue++;
    iarg++;
  }

  // setup output
  // ngroup*nvalue columns, matching compute_grid convention
  // post_process_grid_flag tells fix ave/grid to call post_process_grid()

  per_grid_flag = 1;
  ntotal = ngroup * npergroup;
  size_per_grid_cols = ngroup * nvalue;
  post_process_grid_flag = 1;

  reset_map();

  nglocal = 0;
  vector_grid = NULL;
  tally = NULL;

  pweight_index = -1;
  pweight_ewhich = -1;
  eprefactor = 0.5*update->mvv2e;
}

/* ---------------------------------------------------------------------- */

ComputeGridWeighted::~ComputeGridWeighted()
{
  if (copymode) return;
  delete [] value;
  delete [] unique;
  delete [] nmap;
  memory->destroy(map);
  memory->destroy(vector_grid);
  memory->destroy(tally);
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::init()
{
  // verify pweight custom attribute exists

  pweight_index = particle->find_custom((char *) "pweight");
  if (pweight_index < 0)
    error->all(FLERR,
      "Compute grid/weighted requires fix particle/weight");
  pweight_ewhich = particle->ewhich[pweight_index];

  // verify mixture ngroup is unchanged

  if (ngroup != particle->mixture[imix]->ngroup)
    error->all(FLERR,
      "Number of groups in compute grid/weighted mixture has changed");

  eprefactor = 0.5*update->mvv2e;
  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  Grid::ChildInfo *cinfo = grid->cinfo;
  Particle::Species *species = particle->species;
  Particle::OnePart *particles = particle->particles;
  int *s2g = particle->mixture[imix]->species2group;
  int nlocal = particle->nlocal;

  // refresh edvec pointer (may have been reallocated)
  pweight_ewhich = particle->ewhich[pweight_index];
  double *pweight_dvec = particle->edvec[pweight_ewhich];

  int i,j,k,m,ispecies,igroup,icell;
  double mass,pw;
  double *v,*vec;

  for (i = 0; i < nglocal; i++)
    for (j = 0; j < ntotal; j++)
      tally[i][j] = 0.0;

  for (i = 0; i < nlocal; i++) {
    ispecies = particles[i].ispecies;
    igroup = s2g[ispecies];
    if (igroup < 0) continue;
    icell = particles[i].icell;
    // Guard against stale icell after fix balance / fix adapt (same pattern
    // as compute_grid.cpp fix in 96de7c5). ASan caught this as heap-UAF
    // when cinfo[icell].mask reads past the freshly-resized buffer.
    if (icell < 0 || icell >= nglocal) continue;
    if (!(cinfo[icell].mask & groupbit)) continue;

    mass = species[ispecies].mass;
    v = particles[i].v;
    pw = pweight_dvec[i];

    vec = tally[icell];
    k = igroup*npergroup;

    for (m = 0; m < npergroup; m++) {
      switch (unique[m]) {
      case WCOUNT:
        vec[k++] += pw;
        break;
      case WMASSSUM:
        vec[k++] += pw * mass;
        break;
      case WMVX:
        vec[k++] += pw * mass * v[0];
        break;
      case WMVY:
        vec[k++] += pw * mass * v[1];
        break;
      case WMVZ:
        vec[k++] += pw * mass * v[2];
        break;
      case WMVSQ:
        vec[k++] += pw * mass * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        break;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeGridWeighted::query_tally_grid(int index, double **&array,
                                          int *&cols)
{
  index--;
  int ivalue = index % nvalue;
  array = tally;
  cols = map[index];
  return nmap[ivalue];
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::post_process_grid(int index, int nsample,
                                            double **etally, int *emap,
                                            double *vec, int nstride)
{
  index--;
  int ivalue = index % nvalue;

  int lo = 0;
  int hi = nglocal;
  int k = 0;

  if (!etally) {
    nsample = 1;
    etally = tally;
    emap = map[index];
    vec = vector_grid;
    nstride = 1;
  }

  switch (value[ivalue]) {

  case N_W:
    {
      int wcount = emap[0];
      for (int icell = lo; icell < hi; icell++) {
        vec[k] = etally[icell][wcount] / nsample;
        k += nstride;
      }
      break;
    }

  case NRHO_W:
    {
      // nrho = sum(pweight) * cellweight / volume / nsample
      // no fnum multiplication — pweight already carries real-particle count
      Grid::ChildInfo *cinfo = grid->cinfo;
      int wcount = emap[0];
      for (int icell = lo; icell < hi; icell++) {
        double vol = cinfo[icell].volume;
        if (vol == 0.0) vec[k] = 0.0;
        else {
          double wt = cinfo[icell].weight / vol;
          vec[k] = wt * etally[icell][wcount] / nsample;
        }
        k += nstride;
      }
      break;
    }

  case MASSRHO_W:
    {
      Grid::ChildInfo *cinfo = grid->cinfo;
      int wmass = emap[0];
      for (int icell = lo; icell < hi; icell++) {
        double vol = cinfo[icell].volume;
        if (vol == 0.0) vec[k] = 0.0;
        else {
          double wt = cinfo[icell].weight / vol;
          vec[k] = wt * etally[icell][wmass] / nsample;
        }
        k += nstride;
      }
      break;
    }

  case PXRHO_W:
  case PYRHO_W:
  case PZRHO_W:
    {
      Grid::ChildInfo *cinfo = grid->cinfo;
      int wmom = emap[0];
      for (int icell = lo; icell < hi; icell++) {
        double vol = cinfo[icell].volume;
        if (vol == 0.0) vec[k] = 0.0;
        else {
          double wt = cinfo[icell].weight / vol;
          vec[k] = wt * etally[icell][wmom] / nsample;
        }
        k += nstride;
      }
      break;
    }

  case KERHO_W:
    {
      Grid::ChildInfo *cinfo = grid->cinfo;
      int wke = emap[0];
      for (int icell = lo; icell < hi; icell++) {
        double vol = cinfo[icell].volume;
        if (vol == 0.0) vec[k] = 0.0;
        else {
          double wt = cinfo[icell].weight / vol;
          vec[k] = eprefactor * wt * etally[icell][wke] / nsample;
        }
        k += nstride;
      }
      break;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::set_map(int ivalue, int name)
{
  int index = 0;
  for (index = 0; index < npergroup; index++)
    if (unique[index] == name) break;

  if (index == npergroup) {
    unique[npergroup++] = name;
  }

  for (int igroup = 0; igroup < ngroup; igroup++)
    map[igroup*nvalue+ivalue][nmap[ivalue]] = index;
  nmap[ivalue]++;
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::reset_map()
{
  for (int i = 0; i < ngroup*nvalue; i++) {
    int igroup = i / nvalue;
    int ivalue = i % nvalue;
    for (int k = 0; k < nmap[ivalue]; k++)
      map[i][k] += igroup*npergroup;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeighted::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memory->destroy(vector_grid);
  memory->destroy(tally);
  nglocal = grid->nlocal;
  memory->create(vector_grid,nglocal,"grid_weighted:vector_grid");
  memory->create(tally,nglocal,ntotal,"grid_weighted:tally");
}

/* ---------------------------------------------------------------------- */

bigint ComputeGridWeighted::memory_usage()
{
  bigint bytes;
  bytes = nglocal * sizeof(double);
  bytes += ntotal*nglocal * sizeof(double);
  return bytes;
}
