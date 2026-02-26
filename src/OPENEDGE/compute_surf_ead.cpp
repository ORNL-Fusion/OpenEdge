/* ----------------------------------------------------------------------
   OpenEdge: per-surface Energy-Angle diagnostic
   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
     - 42d
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_surf_ead.h"

#include "particle.h"
#include "mixture.h"
#include "surf.h"
#include "grid.h"
#include "update.h"
#include "modify.h"
#include "input.h"
#include "domain.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

#include <cmath>

using namespace SPARTA_NS;

#define DELTA 4096

/* ---------------------------------------------------------------------- */

ComputeSurfEAD::ComputeSurfEAD(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  // compute ID surf/ead group-ID mix-ID Emax NbE Amax NbA [quantities ...]
  if (narg < 8) error->all(FLERR,"Illegal compute surf/ead command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute surf/ead group ID does not exist");
  groupbit = surf->bitmask[igroup];

  imix = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Compute surf/ead mixture ID does not exist");

  emax = input->numeric(FLERR,arg[4]);
  bins_e = input->inumeric(FLERR,arg[5]);
  amax = input->numeric(FLERR,arg[6]);
  bins_a = input->inumeric(FLERR,arg[7]);
  if (emax <= 0.0 || amax <= 0.0 || bins_e <= 0 || bins_a <= 0)
    error->all(FLERR,"Illegal compute surf/ead command");

  bins_total = bins_e * bins_a;
  de = emax / bins_e;
  da = amax / bins_a;

  // default output: all
  q_sumw = 1;
  q_sumn = 1;
  q_hist = 1;

  if (narg > 8) {
    if (strcmp(arg[8],"quantities") != 0)
      error->all(FLERR,"Illegal compute surf/ead command");
    if (narg == 9)
      error->all(FLERR,"Illegal compute surf/ead command");
    q_sumw = q_sumn = q_hist = 0;
    for (int i = 9; i < narg; i++) {
      if (strcmp(arg[i],"sum_weight") == 0) q_sumw = 1;
      else if (strcmp(arg[i],"sum_particles") == 0) q_sumn = 1;
      else if (strcmp(arg[i],"hist") == 0) q_hist = 1;
      else error->all(FLERR,"Illegal compute surf/ead command");
    }
    if (!q_sumw && !q_sumn && !q_hist)
      error->all(FLERR,"Illegal compute surf/ead command");
  }

  ncols = 0;
  idx_sumw = idx_sumn = idx_hist = -1;
  if (q_sumw) idx_sumw = ncols++;
  if (q_sumn) idx_sumn = ncols++;
  if (q_hist) {
    idx_hist = ncols;
    ncols += bins_total;
  }

  per_surf_flag = 1;
  size_per_surf_cols = ncols;
  surf_tally_flag = 1;
  timeflag = 1;

  ntally = maxtally = 0;
  tally2surf = NULL;
  array_surf_tally = NULL;

  maxsurf = 0;
  array_surf = NULL;

  hash = new MyHash;

  dim = domain->dimension;
  lines = NULL;
  tris = NULL;

  weightflag = 0;
  weight = 1.0;

  combined = 0;
}

/* ---------------------------------------------------------------------- */

ComputeSurfEAD::~ComputeSurfEAD()
{
  if (copy || copymode) return;

  delete hash;
  memory->destroy(array_surf_tally);
  memory->destroy(tally2surf);
  memory->destroy(array_surf);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute surf/ead when surfs do not exist");
  if (surf->implicit)
    error->all(FLERR,"Cannot use compute surf/ead with implicit surfs");

  if (grid->cellweightflag) weightflag = 1;
  else weightflag = 0;

  // initialize tally storage in case queried before first tally step
  clear();
  combined = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::compute_per_surf()
{
  invoked_per_surf = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::clear()
{
  lines = surf->lines;
  tris = surf->tris;
  hash->clear();
  ntally = 0;
  combined = 0;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::surf_tally(double /*dtremain*/, int isurf, int /*icell*/, int /*reaction*/,
                                Particle::OnePart *iorig,
                                Particle::OnePart * /*ip*/, Particle::OnePart * /*jp*/)
{
  if (!iorig) return;

  // skip if isurf not in surface group
  if (dim == 2) {
    if (!(lines[isurf].mask & groupbit)) return;
  } else {
    if (!(tris[isurf].mask & groupbit)) return;
  }

  // skip if particle species not in mixture group
  int origspecies = iorig->ispecies;
  int igroup = particle->mixture[imix]->species2group[origspecies];
  if (igroup < 0) return;

  surfint surfID;
  double *norm;
  if (dim == 2) {
    surfID = lines[isurf].id;
    norm = lines[isurf].norm;
  } else {
    surfID = tris[isurf].id;
    norm = tris[isurf].norm;
  }

  int itally;
  if (hash->find(surfID) != hash->end()) itally = (*hash)[surfID];
  else {
    if (ntally == maxtally) grow_tally();
    itally = ntally;
    (*hash)[surfID] = itally;
    tally2surf[itally] = surfID;
    double *vec0 = array_surf_tally[itally];
    for (int i = 0; i < ncols; i++) vec0[i] = 0.0;
    ntally++;
  }

  double *v = iorig->v;
  double vx = v[0], vy = v[1], vz = v[2];
  double v2 = vx*vx + vy*vy + vz*vz;
  if (v2 <= 0.0) return;

  // Use per-particle mass when present, fall back to species mass.
  double mass = (iorig->mass > 0.0) ? iorig->mass : particle->species[origspecies].mass;
  if (mass <= 0.0) return;

  if (weightflag) weight = iorig->weight;
  else weight = 1.0;

  // In SI units, 0.5*mvv2e*m*v^2 is Joules; convert to eV for EAD bins.
  // In non-SI styles keep legacy behavior (no extra conversion).
  double e_ev = 0.5 * update->mvv2e * mass * v2;
  if (strcmp(update->unit_style,"si") == 0) e_ev *= update->joule2ev;
  if (e_ev < 0.0 || e_ev > emax) return;
  const double nx = norm[0];
  const double ny = norm[1];
  const double nz = (dim == 3) ? norm[2] : 0.0;
  const double n2 = nx*nx + ny*ny + nz*nz;
  if (n2 <= 0.0) return;

  double cosang = (vx*nx + vy*ny + vz*nz) / sqrt(v2*n2);
  if (cosang < -1.0) cosang = -1.0;
  if (cosang >  1.0) cosang =  1.0;
  double ang = acos(fabs(cosang)) * 180.0 / MathConst::MY_PI;
  if (ang < 0.0 || ang > amax) return;

  int ie = static_cast<int>(e_ev / de);
  int ia = static_cast<int>(ang / da);
  if (ie >= bins_e) ie = bins_e - 1;
  if (ia >= bins_a) ia = bins_a - 1;
  if (ie < 0 || ia < 0) return;

  const int ibin = ia * bins_e + ie;

  double *vec = array_surf_tally[itally];
  if (q_sumw) vec[idx_sumw] += weight;
  if (q_sumn) vec[idx_sumn] += 1.0;
  if (q_hist) vec[idx_hist + ibin] += weight;
  combined = 0;
}

/* ---------------------------------------------------------------------- */

int ComputeSurfEAD::tallyinfo(surfint *&ptr)
{
  ptr = tally2surf;
  return ntally;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::post_process_surf()
{
  if (combined) return;
  combined = 1;

  int nown = surf->nown;
  if (nown > maxsurf) {
    memory->destroy(array_surf);
    maxsurf = nown;
    memory->create(array_surf,maxsurf,ncols,"surf/ead:array_surf");
  }

  surf->collate_array(ntally,ncols,tally2surf,array_surf_tally,array_surf);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::grow_tally()
{
  maxtally += DELTA;
  memory->grow(tally2surf,maxtally,"surf/ead:tally2surf");
  memory->grow(array_surf_tally,maxtally,ncols,"surf/ead:array_surf_tally");
}

/* ---------------------------------------------------------------------- */

void ComputeSurfEAD::reallocate()
{
  // no-op; nothing depends on per-surf geometric normalization
}

/* ---------------------------------------------------------------------- */

bigint ComputeSurfEAD::memory_usage()
{
  bigint bytes = 0;
  bytes += static_cast<bigint>(ncols) * maxtally * sizeof(double);
  bytes += static_cast<bigint>(maxtally) * sizeof(surfint);
  return bytes;
}
