/* ----------------------------------------------------------------------
   fix population/control — see header for the contract.
------------------------------------------------------------------------- */

#include "string.h"
#include "stdlib.h"
#include "fix_population_control.h"
#include "particle.h"
#include "grid.h"
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
  if (narg < 4)
    error->all(FLERR,"Illegal fix population/control command: "
               "fix ID population/control Nevery Nmax [wratio R] [split S]");

  nevery = atoi(arg[2]);
  nmax_  = atoi(arg[3]);
  if (nevery <= 0 || nmax_ <= 0)
    error->all(FLERR,"fix population/control: Nevery and Nmax must be > 0");
  wratio_ = 4.0; split_ = 10.0;
  int iarg = 4;
  while (iarg < narg) {
    if (iarg+1 >= narg) error->all(FLERR,"fix population/control: keyword needs a value");
    if (strcmp(arg[iarg],"wratio") == 0) wratio_ = atof(arg[iarg+1]);
    else if (strcmp(arg[iarg],"split") == 0) split_ = atof(arg[iarg+1]);
    else error->all(FLERR,"fix population/control: unknown keyword (wratio, split)");
    iarg += 2;
  }
  if ((wratio_ != 0.0 && wratio_ < 1.0) || (split_ != 0.0 && split_ < 2.0))
    error->all(FLERR,"fix population/control: wratio must be 0 or >= 1, split 0 or >= 2");
  vector_flag = 1; size_vector = 4; global_freq = nevery;
  for (int k = 0; k < 6; k++) tally_loc_[k] = 0.0;

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
  if (copymode) return;   // Kokkos functor copies share rng_
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
  for (int k = 0; k < 6; k++) tally_loc_[k] = 0.0;
  const int nlocal0 = particle->nlocal;
  if (nlocal0 == 0) return;

  Particle::OnePart *particles = particle->particles;
  double *pw = particle->edvec[pweight_ewhich_];

  // bucket by (cell, species)
  std::unordered_map<long long, std::vector<int>> buckets;
  buckets.reserve(nlocal0/4 + 16);
  for (int i = 0; i < nlocal0; i++) {
    const long long key =
      (static_cast<long long>(particles[i].icell) << 10) | particles[i].ispecies;
    buckets[key].push_back(i);
  }

  // 1. resampling inside the buckets. The marker count of a bucket never
  // grows here, so the copies of split markers go into the slots of the
  // markers that were not selected: no reallocation, indices stay valid.
  std::vector<int> dellist, mult, dead;
  bigint ncopies = 0;
  for (auto &kv : buckets) {
    std::vector<int> &idx = kv.second;
    const int n = static_cast<int>(idx.size());
    if (n < 2) continue;
    double wsum = 0.0, wmax = 0.0;
    for (int k = 0; k < n; k++) { const double w = pw[idx[k]]; wsum += w; if (w > wmax) wmax = w; }
    if (!(wsum > 0.0)) continue;                  // degenerate: leave the bucket untouched
    const bool over = n > nmax_;
    const bool uneven = wratio_ > 0.0 && wmax*n > wratio_*wsum;
    if (!over && !uneven) continue;

    // systematic resampling: m equally spaced points on the cumulative
    // weight, one random offset; marker k gets as many copies as points
    // fall in its interval (expected value m w_k / W)
    const int m = over ? nmax_ : n;
    const double d = wsum/m;
    double u = rng_->uniform()*d;
    mult.assign(n,0);
    int k = 0;
    double cum = pw[idx[0]];
    for (int j = 0; j < m; j++) {
      const double t = u + j*d;
      while (t > cum && k < n-1) { k++; cum += pw[idx[k]]; }
      mult[k]++;
    }

    dead.clear();
    for (int q = 0; q < n; q++) if (mult[q] == 0) dead.push_back(idx[q]);
    size_t nd = 0;
    for (int q = 0; q < n; q++) {
      if (mult[q] == 0) continue;
      const int i = idx[q];
      pw[i] = d;
      for (int c = 1; c < mult[q]; c++) {          // copies of a split marker
        const int j = dead[nd++];
        memcpy(&particles[j],&particles[i],sizeof(Particle::OnePart));
        particle->copy_custom(j,i);
        particles[j].id = static_cast<int>(MAXSMALLINT*rng_->uniform());
        ncopies++;
      }
    }
    for (; nd < dead.size(); nd++) dellist.push_back(dead[nd]);
  }
  buckets.clear();

  const int ndelete = static_cast<int>(dellist.size());
  if (ndelete) {
    particle->compress_reactions(ndelete,dellist.data());
    ndeleted_total_ += ndelete;
  }

  // 2. split markers that are heavy for their cell (mean over all species)
  if (split_ > 0.0) {
    const int nl = particle->nlocal;
    const int ncell = grid->nlocal;
    std::vector<double> cw(ncell,0.0);
    std::vector<int> cn(ncell,0);
    particles = particle->particles;
    pw = particle->edvec[pweight_ewhich_];
    for (int i = 0; i < nl; i++) {
      const int ic = particles[i].icell;
      if (ic < 0 || ic >= ncell) continue;
      cw[ic] += pw[i]; cn[ic]++;
    }
    for (int i = 0; i < nl; i++) {
      const int ic = particle->particles[i].icell;   // not the cached pointer: cloning may reallocate
      if (ic < 0 || ic >= ncell || cn[ic] < 2) continue;
      const double mean = cw[ic]/cn[ic];
      const double w = particle->edvec[pweight_ewhich_][i];
      if (!(w > split_*mean)) continue;
      int kc = static_cast<int>(w/mean + 0.999999);
      if (kc > 8) kc = 8;
      particle->edvec[pweight_ewhich_][i] = w/kc;
      for (int c = 1; c < kc; c++) {
        particle->clone_particle(i);               // may reallocate: pointers are re-read each time
        particle->particles[particle->nlocal-1].id = static_cast<int>(MAXSMALLINT*rng_->uniform());
        ncopies++;
      }
    }
  }

  if (ndelete || ncopies) particle->sorted = 0;

  // diagnostics: effective sample size of all markers and of the tagged ones
  {
    const int nl = particle->nlocal;
    const double *w = particle->edvec[pweight_ewhich_];
    const int *tag = (update->tag_index >= 0) ? particle->eivec[particle->ewhich[update->tag_index]] : NULL;
    double s1 = 0.0, s2 = 0.0, t1 = 0.0, t2 = 0.0;
    for (int i = 0; i < nl; i++) {
      s1 += w[i]; s2 += w[i]*w[i];
      if (tag && tag[i]) { t1 += w[i]; t2 += w[i]*w[i]; }
    }
    tally_loc_[0] = ndelete; tally_loc_[1] = static_cast<double>(ncopies);
    tally_loc_[2] = s1; tally_loc_[3] = s2; tally_loc_[4] = t1; tally_loc_[5] = t2;
  }
}

/* ----------------------------------------------------------------------
   [1] deleted, [2] copies (last pass), [3] effective sample size of all
   markers, [4] of the transit-tagged markers (after the last pass)
------------------------------------------------------------------------- */

double FixPopulationControl::compute_vector(int i)
{
  double all[6];
  MPI_Allreduce(tally_loc_,all,6,MPI_DOUBLE,MPI_SUM,world);
  if (i == 0) return all[0];
  if (i == 1) return all[1];
  if (i == 2) return (all[3] > 0.0) ? all[2]*all[2]/all[3] : 0.0;
  return (all[5] > 0.0) ? all[4]*all[4]/all[5] : 0.0;
}
