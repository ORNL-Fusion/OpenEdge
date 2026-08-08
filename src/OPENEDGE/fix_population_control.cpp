/* ----------------------------------------------------------------------
   fix population/control — see header for the contract.
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "fix_population_control.h"
#include "particle.h"
#include "update.h"
#include "comm.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

#include <unordered_map>
#include <vector>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixPopulationControl::FixPopulationControl(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 4)
    error->all(FLERR,"Illegal fix population/control command: "
               "fix ID population/control Nevery Nmax");

  nevery = atoi(arg[2]);
  nmax_  = atoi(arg[3]);
  if (nevery <= 0 || nmax_ <= 0)
    error->all(FLERR,"fix population/control: Nevery and Nmax must be > 0");

  pweight_index_ = -1;
  pweight_ewhich_ = -1;
  rng_ = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  rng_->reset(seed, comm->me, 100);
  ndeleted_total_ = 0;
}

/* ---------------------------------------------------------------------- */

FixPopulationControl::~FixPopulationControl()
{
  delete rng_;
}

/* ---------------------------------------------------------------------- */

int FixPopulationControl::setmask()
{
  return END_OF_STEP;
}

/* ---------------------------------------------------------------------- */

void FixPopulationControl::init()
{
  pweight_index_ = particle->find_custom((char *) "pweight");
  if (pweight_index_ < 0)
    error->all(FLERR,"fix population/control requires fix particle/weight "
               "(pweight custom attribute)");
  pweight_ewhich_ = particle->ewhich[pweight_index_];
}

/* ---------------------------------------------------------------------- */

void FixPopulationControl::end_of_step()
{
  const int nlocal = particle->nlocal;
  if (nlocal == 0) return;

  Particle::OnePart *particles = particle->particles;
  double *pw = particle->edvec[pweight_ewhich_];

  // bucket by (cell, species)
  std::unordered_map<long long, std::vector<int>> buckets;
  buckets.reserve(nlocal/4 + 16);
  for (int i = 0; i < nlocal; i++) {
    const long long key =
      (static_cast<long long>(particles[i].icell) << 10) | particles[i].ispecies;
    buckets[key].push_back(i);
  }

  int *dellist = NULL;
  int ndelete = 0, maxdel = 0;

  for (auto &kv : buckets) {
    std::vector<int> &idx = kv.second;
    const int n = static_cast<int>(idx.size());
    if (n <= nmax_) continue;

    // partial Fisher-Yates: first nmax_ entries become the survivors
    for (int k = 0; k < nmax_; k++) {
      const int j = k + static_cast<int>(rng_->uniform() * (n - k));
      const int jj = (j < n) ? j : n-1;
      const int tmp = idx[k]; idx[k] = idx[jj]; idx[jj] = tmp;
    }

    double w_all = 0.0, w_kept = 0.0;
    for (int k = 0; k < n; k++)      w_all  += pw[idx[k]];
    for (int k = 0; k < nmax_; k++)  w_kept += pw[idx[k]];
    if (w_kept <= 0.0) continue;    // degenerate; leave bucket untouched

    // exact weight conservation: survivors absorb the deleted weight
    const double scale = w_all / w_kept;
    for (int k = 0; k < nmax_; k++) pw[idx[k]] *= scale;

    if (ndelete + (n - nmax_) > maxdel) {
      maxdel = ndelete + (n - nmax_) + 1024;
      memory->grow(dellist, maxdel, "population/control:dellist");
    }
    for (int k = nmax_; k < n; k++) dellist[ndelete++] = idx[k];
  }

  if (ndelete) {
    particle->compress_reactions(ndelete, dellist);
    particle->sorted = 0;
    ndeleted_total_ += ndelete;
  }
  memory->destroy(dellist);
}
