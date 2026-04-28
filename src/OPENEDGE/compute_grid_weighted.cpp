/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute grid/weighted — like compute grid but uses per-particle
    pweight custom attribute instead of global fnum.

    Supported values (full upstream compute-grid keyword set, with `_w`
    suffix to make the weighting explicit):

      density-like (extensive, normalised by cell volume):
        n_w        Sum(pweight)
        nrho_w     Sum(pweight) * cellweight / V
        massrho_w  Sum(pweight*m) * cellweight / V
        pxrho_w    Sum(pweight*m*v_x) * cellweight / V        (idem y, z)
        pyrho_w    "
        pzrho_w    "
        kerho_w    Sum(0.5*pweight*m*v^2) * cellweight / V

      mean / intensive (per-particle):
        mass_w     Sum(pweight*m) / Sum(pweight)
        u_w        Sum(pweight*m*v_x) / Sum(pweight*m)        (idem y, z)
        v_w        "
        w_w        "
        usq_w      Sum(pweight*m*v_x^2) / Sum(pweight*m)      (idem y, z)
        vsq_w      "
        wsq_w      "
        ke_w       0.5 * mvv2e * Sum(pweight*m*v^2) / Sum(pweight)
        temp_w     mvv2e/(3*kB) * Sum(pweight*m*v^2) / Sum(pweight)
        erot_w     Sum(pweight*erot) / Sum(pweight)
        evib_w     Sum(pweight*evib) / Sum(pweight)
        trot_w     2*mvv2e/kB * Sum(pweight*erot) / Sum(pweight*rotdof)
        tvib_w     2*mvv2e/kB * Sum(pweight*evib) / Sum(pweight*vibdof)

      group fractions (per-cell denominator across all groups):
        nfrac_w     Sum(pweight)_group   / Sum(pweight)_cell
        massfrac_w  Sum(pweight*m)_group / Sum(pweight*m)_cell

    Requires `fix particle/weight` to provide the `pweight` custom.
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

enum{N_W,NRHO_W,MASSRHO_W,PXRHO_W,PYRHO_W,PZRHO_W,KERHO_W,
     MASS_W,NFRAC_W,MASSFRAC_W,
     U_W,V_W,W_W,USQ_W,VSQ_W,WSQ_W,
     KE_W,TEMP_W,EROT_W,EVIB_W,TROT_W,TVIB_W};

// internal accumulators (per-group tally quantities)

enum{WCOUNT,WMASSSUM,WMVX,WMVY,WMVZ,WMVSQ,
     WMVXSQ,WMVYSQ,WMVZSQ,
     WEROT,WEVIB,WDOFROT,WDOFVIB,LASTSIZE};

// sentinels for cell-level (not per-group) tallies, replaced with
// real column indices in reset_map()

#define WCELLCOUNT_TAG  -101
#define WCELLMASS_TAG   -102

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
  cellcount_w = cellmass_w = 0;
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
    } else if (strcmp(arg[iarg],"mass_w") == 0) {
      value[ivalue] = MASS_W;
      set_map(ivalue,WMASSSUM);
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"massrho_w") == 0) {
      value[ivalue] = MASSRHO_W;
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"nfrac_w") == 0) {
      value[ivalue] = NFRAC_W;
      set_map(ivalue,WCOUNT);
      set_map(ivalue,WCELLCOUNT_TAG);
      cellcount_w = 1;
    } else if (strcmp(arg[iarg],"massfrac_w") == 0) {
      value[ivalue] = MASSFRAC_W;
      set_map(ivalue,WMASSSUM);
      set_map(ivalue,WCELLMASS_TAG);
      cellmass_w = 1;

    } else if (strcmp(arg[iarg],"u_w") == 0) {
      value[ivalue] = U_W;
      set_map(ivalue,WMVX);
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"v_w") == 0) {
      value[ivalue] = V_W;
      set_map(ivalue,WMVY);
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"w_w") == 0) {
      value[ivalue] = W_W;
      set_map(ivalue,WMVZ);
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"usq_w") == 0) {
      value[ivalue] = USQ_W;
      set_map(ivalue,WMVXSQ);
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"vsq_w") == 0) {
      value[ivalue] = VSQ_W;
      set_map(ivalue,WMVYSQ);
      set_map(ivalue,WMASSSUM);
    } else if (strcmp(arg[iarg],"wsq_w") == 0) {
      value[ivalue] = WSQ_W;
      set_map(ivalue,WMVZSQ);
      set_map(ivalue,WMASSSUM);

    } else if (strcmp(arg[iarg],"ke_w") == 0) {
      value[ivalue] = KE_W;
      set_map(ivalue,WMVSQ);
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"temp_w") == 0) {
      value[ivalue] = TEMP_W;
      set_map(ivalue,WMVSQ);
      set_map(ivalue,WCOUNT);

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

    } else if (strcmp(arg[iarg],"erot_w") == 0) {
      value[ivalue] = EROT_W;
      set_map(ivalue,WEROT);
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"evib_w") == 0) {
      value[ivalue] = EVIB_W;
      set_map(ivalue,WEVIB);
      set_map(ivalue,WCOUNT);
    } else if (strcmp(arg[iarg],"trot_w") == 0) {
      value[ivalue] = TROT_W;
      set_map(ivalue,WEROT);
      set_map(ivalue,WDOFROT);
    } else if (strcmp(arg[iarg],"tvib_w") == 0) {
      value[ivalue] = TVIB_W;
      set_map(ivalue,WEVIB);
      set_map(ivalue,WDOFVIB);

    } else error->all(FLERR,"Illegal compute grid/weighted command");

    ivalue++;
    iarg++;
  }

  // setup output: ngroup*nvalue logical columns; cell-level extras (if
  // requested) are appended in reset_map().

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
  eprefactor = 0.5 * update->mvv2e;
  tprefactor = update->mvv2e / (3.0 * update->boltz);
  rvprefactor = 2.0 * update->mvv2e / update->boltz;
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

  eprefactor  = 0.5 * update->mvv2e;
  tprefactor  = update->mvv2e / (3.0 * update->boltz);
  rvprefactor = 2.0 * update->mvv2e / update->boltz;
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
    if (cellmass_w) vec[cellmass_w] += pw * mass;
    if (cellcount_w) vec[cellcount_w] += pw;
    k = igroup * npergroup;

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
      case WMVXSQ:
        vec[k++] += pw * mass * v[0]*v[0];
        break;
      case WMVYSQ:
        vec[k++] += pw * mass * v[1]*v[1];
        break;
      case WMVZSQ:
        vec[k++] += pw * mass * v[2]*v[2];
        break;
      case WEROT:
        vec[k++] += pw * particles[i].erot;
        break;
      case WEVIB:
        vec[k++] += pw * particles[i].evib;
        break;
      case WDOFROT:
        vec[k++] += pw * species[ispecies].rotdof;
        break;
      case WDOFVIB:
        vec[k++] += pw * species[ispecies].vibdof;
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

  Grid::ChildInfo *cinfo = grid->cinfo;

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

  case MASS_W:
    {
      int wmass = emap[0];
      int wcount = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wcount];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = etally[icell][wmass] / norm;
        k += nstride;
      }
      break;
    }

  case NRHO_W:
    {
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

  case NFRAC_W:
  case MASSFRAC_W:
    {
      int numer = emap[0];
      int denom = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][denom];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = etally[icell][numer] / norm;
        k += nstride;
      }
      break;
    }

  case U_W:
  case V_W:
  case W_W:
  case USQ_W:
  case VSQ_W:
  case WSQ_W:
    {
      int velocity = emap[0];
      int wmass = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wmass];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = etally[icell][velocity] / norm;
        k += nstride;
      }
      break;
    }

  case KE_W:
    {
      int wmvsq = emap[0];
      int wcount = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wcount];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = eprefactor * etally[icell][wmvsq] / norm;
        k += nstride;
      }
      break;
    }

  case TEMP_W:
    {
      int wmvsq = emap[0];
      int wcount = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wcount];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = tprefactor * etally[icell][wmvsq] / norm;
        k += nstride;
      }
      break;
    }

  case EROT_W:
  case EVIB_W:
    {
      int eng = emap[0];
      int wcount = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wcount];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = etally[icell][eng] / norm;
        k += nstride;
      }
      break;
    }

  case TROT_W:
  case TVIB_W:
    {
      int eng = emap[0];
      int wdof = emap[1];
      for (int icell = lo; icell < hi; icell++) {
        double norm = etally[icell][wdof];
        if (norm == 0.0) vec[k] = 0.0;
        else vec[k] = rvprefactor * etally[icell][eng] / norm;
        k += nstride;
      }
      break;
    }

  case PXRHO_W:
  case PYRHO_W:
  case PZRHO_W:
    {
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
  // CELLCOUNT_W / CELLMASS_W are cell-level (not per-group). Stash the
  // sentinel; reset_map() rewrites it to the real ntotal index.
  if (name == WCELLCOUNT_TAG || name == WCELLMASS_TAG) {
    for (int igroup = 0; igroup < ngroup; igroup++)
      map[igroup*nvalue+ivalue][nmap[ivalue]] = name;
    nmap[ivalue]++;
    return;
  }

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
  // First, shift per-group indices into their final tally column.
  // Then, append cell-level columns (cellcount_w / cellmass_w) and
  // rewrite the sentinel-tagged map entries to point at them.

  for (int i = 0; i < ngroup*nvalue; i++) {
    int igroup = i / nvalue;
    int ivalue = i % nvalue;
    for (int kk = 0; kk < nmap[ivalue]; kk++) {
      int v = map[i][kk];
      if (v == WCELLCOUNT_TAG || v == WCELLMASS_TAG) continue;
      map[i][kk] += igroup*npergroup;
    }
  }

  if (cellcount_w) cellcount_w = ntotal++;
  if (cellmass_w)  cellmass_w  = ntotal++;

  for (int i = 0; i < ngroup*nvalue; i++) {
    int ivalue = i % nvalue;
    for (int kk = 0; kk < nmap[ivalue]; kk++) {
      if (map[i][kk] == WCELLCOUNT_TAG) map[i][kk] = cellcount_w;
      else if (map[i][kk] == WCELLMASS_TAG) map[i][kk] = cellmass_w;
    }
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
