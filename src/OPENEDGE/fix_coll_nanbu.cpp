/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix coll/nanbu: binary Coulomb collisions via the Nanbu (1997) /
    Takizuka-Abe (1977) algorithm.

    Syntax:
      fix ID coll/nanbu Nevery plasma <TeSrc> <NeSrc>

    Example:
      fix nanbu coll/nanbu 1 plasma c_cplasma[7] c_cplasma[10]

    Te (eV) and Ne (m^-3) from per-grid computes are used for the
    Coulomb logarithm.  The particle density entering the scattering-
    parameter s is computed from the actual charged simulation particles
    in each cell: n_partner = N_charged * fnum / V_cell.
------------------------------------------------------------------------- */

#include "fix_coll_nanbu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "comm.h"
#include "compute.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "particle.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "update.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16
#define DELTAPART 128

/* ---------------------------------------------------------------------- */

FixCollNanbu::FixCollNanbu(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  rng_(nullptr),
  npmax_(0),
  plist_(nullptr)
{
  // Syntax: fix ID coll/nanbu Nevery plasma TeSrc NeSrc
  //   arg[0] = fix ID
  //   arg[1] = style = "coll/nanbu"
  //   arg[2] = Nevery
  //   arg[3] = "plasma"
  //   arg[4] = Te source (c_ID[col])
  //   arg[5] = Ne source (c_ID[col])

  if (narg < 6)
    error->all(FLERR,
      "Illegal fix coll/nanbu command (need: nevery plasma TeSrc NeSrc)");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg++], "plasma") != 0)
    error->all(FLERR, "fix coll/nanbu: missing 'plasma' keyword");

  parse_compute_src(arg[iarg++], srcTe_, "Te");
  parse_compute_src(arg[iarg++], srcNe_, "Ne");
}

/* ---------------------------------------------------------------------- */

FixCollNanbu::~FixCollNanbu()
{
  if (copymode) return;
  delete rng_;
  rng_ = nullptr;
  memory->destroy(plist_);
  delete[] srcTe_.cid;
  delete[] srcNe_.cid;
}

/* ---------------------------------------------------------------------- */

int FixCollNanbu::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::init()
{
  // seed RNG from ranmaster (same pattern as collide.cpp)
  if (!rng_) {
    rng_ = new RanKnuth(update->ranmaster->uniform());
    double seed = update->ranmaster->uniform();
    rng_->reset(seed, comm->me, 100);
  }

  // build scatter table
  scatter_table_.initialize();

  // resolve compute sources
  auto bind_compute = [&](NanbuGridSrc &S, const char *label) {
    if (S.kind != NANBU_SRC_COMP) return;
    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix coll/nanbu: compute '%s' for %s not found", S.cid, label);
      error->all(FLERR, msg);
    }
    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR, "fix coll/nanbu: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR, "fix coll/nanbu: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix coll/nanbu: column %d for compute '%s' (%s) out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR, msg);
    }
  };

  bind_compute(srcTe_, "Te");
  bind_compute(srcNe_, "Ne");
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  if (!particle->sorted) particle->sort();

  // refresh compute caches for this timestep
  refresh_compute_src(srcTe_);
  refresh_compute_src(srcNe_);

  if (grid->nlocal == 0) return;
  if (particle->nlocal == 0) return;

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  int               *next      = particle->next;
  Grid::ChildInfo   *cinfo     = grid->cinfo;
  const int nglocal = grid->nlocal;

  for (int icell = 0; icell < nglocal; icell++) {
    int np = cinfo[icell].count;
    if (np <= 1) continue;

    // build particle list for this cell (same pattern as collide.cpp)
    if (np > npmax_) {
      while (np > npmax_) npmax_ += DELTAPART;
      memory->destroy(plist_);
      memory->create(plist_, npmax_, "coll/nanbu:plist");
    }

    int n = 0;
    int ip = cinfo[icell].first;
    while (ip >= 0) {
      plist_[n++] = ip;
      ip = next[ip];
    }

    nanbu_collisions_cell(icell, n);
  }
}

/* ----------------------------------------------------------------------
   Nanbu binary Coulomb collisions within a single grid cell.
   1) Collect charged particles
   2) Shuffle (Fisher-Yates)
   3) Pair sequentially and scatter
------------------------------------------------------------------------- */

void FixCollNanbu::nanbu_collisions_cell(int icell, int np)
{
  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  Grid::ChildInfo   *cinfo     = grid->cinfo;

  // collect indices of charged particles from plist_
  // reuse tail of plist_ for the charged sub-list
  // (charged_list is a separate local array to avoid aliasing)

  static std::vector<int> charged;
  charged.clear();
  charged.reserve(np);

  for (int i = 0; i < np; i++) {
    int idx = plist_[i];
    int isp = particles[idx].ispecies;
    if (species[isp].charge != 0.0)
      charged.push_back(idx);
  }

  int nc = static_cast<int>(charged.size());
  if (nc < 2) return;

  // cell volume (accounting for cell-weight, same as collide.cpp)
  double volume = cinfo[icell].volume / cinfo[icell].weight;
  if (volume <= 0.0) return;

  // plasma parameters for Coulomb logarithm
  double Te_eV = std::max(read_cell_src(srcTe_, icell), 0.0);
  double ne    = std::max(read_cell_src(srcNe_, icell), 0.0);
  double lnLambda = compute_coulomb_log(ne, Te_eV);

  // physical constants from update
  const double echarge   = update->echarge;     // C
  const double epsilon_0 = update->epsilon_0;   // F/m
  const double fnum      = update->fnum;
  const double dt        = update->dt * nevery;

  // effective density of collision partners (from simulation particles)
  double n_partner = static_cast<double>(nc) * fnum / volume;

  // Fisher-Yates shuffle of the charged list
  for (int i = nc - 1; i > 0; i--) {
    int j = static_cast<int>(rng_->uniform() * (i + 1));
    if (j > i) j = i;   // guard against rng returning exactly 1.0
    std::swap(charged[i], charged[j]);
  }

  // pair sequentially: (0,1), (2,3), ...
  // if odd count, last particle pairs with a random earlier partner
  int npairs = nc / 2;
  if (nc % 2 == 1) {
    int rand_partner = static_cast<int>(rng_->uniform() * (nc - 1));
    if (rand_partner >= nc - 1) rand_partner = nc - 2;
    charged.push_back(charged[rand_partner]);
    npairs = (nc + 1) / 2;
  }

  // scatter each pair
  for (int ipair = 0; ipair < npairs; ipair++) {
    int idxA = charged[2 * ipair];
    int idxB = charged[2 * ipair + 1];

    Particle::OnePart &pA = particles[idxA];
    Particle::OnePart &pB = particles[idxB];
    int ispA = pA.ispecies;
    int ispB = pB.ispecies;

    double *vA = pA.v;
    double *vB = pB.v;

    // masses (kg)
    double mA = species[ispA].mass;
    double mB = species[ispB].mass;
    double total_mass = mA + mB;
    double mu = mA * mB / total_mass;   // reduced mass

    // charges (C)
    double qA = species[ispA].charge * echarge;
    double qB = species[ispB].charge * echarge;

    // relative velocity
    double g[3];
    g[0] = vB[0] - vA[0];
    g[1] = vB[1] - vA[1];
    g[2] = vB[2] - vA[2];
    double g_mag = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

    if (g_mag == 0.0) continue;

    // Nanbu scattering parameter s
    // s = (qA*qB)^2 * n_partner * ln(Lambda) * dt / (8*pi*eps0^2 * mu^2 * g^3)
    double g_mag3 = g_mag * g_mag * g_mag;
    double qq = qA * qB;
    double s_factor = qq * qq * n_partner * lnLambda * dt /
                      (8.0 * M_PI * epsilon_0 * epsilon_0 * mu * mu);

    // compute s_ab; guard against division by zero for g_mag3
    double s_ab;

    // check isotropic limit before dividing (avoids overflow)
    if (6.0 * g_mag3 < s_factor) {
      s_ab = 7.0;  // will trigger isotropic branch below
    } else {
      s_ab = s_factor / g_mag3;
    }

    // sample cos(chi) from Nanbu distribution
    double cos_chi;

    if (s_ab > 6.0) {
      // isotropic scattering
      cos_chi = 2.0 * rng_->uniform() - 1.0;
    } else if (s_ab > 0.01) {
      // preferentially forward scattering
      double A = scatter_table_.get_A(s_ab);
      double U = rng_->uniform();
      cos_chi = (1.0 / A) * std::log(std::exp(-A) + 2.0 * std::sinh(A) * U);
    } else {
      // small-angle scattering
      double U = rng_->uniform();
      if (U < 1.0e-30) U = 1.0e-30;  // prevent log(0)
      cos_chi = 1.0 + s_ab * std::log(U);
    }

    // clamp cos_chi to [-1, 1]
    if (cos_chi > 1.0)  cos_chi = 1.0;
    if (cos_chi < -1.0) cos_chi = -1.0;

    double one_minus_cos = 1.0 - cos_chi;
    double sin_chi = std::sqrt(1.0 - cos_chi * cos_chi);

    // random azimuthal angle
    double eps_angle = 2.0 * M_PI * rng_->uniform();
    double cos_eps = std::cos(eps_angle);
    double sin_eps = std::sin(eps_angle);

    // perpendicular component of g for rotation
    double g_perp = std::sqrt(g[1]*g[1] + g[2]*g[2]);
    double h[3];
    if (g_perp > 1.0e-12 * g_mag) {
      h[0] =  g_perp * cos_eps;
      h[1] = -(g[0]*g[1]*cos_eps + g_mag*g[2]*sin_eps) / g_perp;
      h[2] = -(g[0]*g[2]*cos_eps - g_mag*g[1]*sin_eps) / g_perp;
    } else {
      // g is nearly along x-axis
      h[0] = 0.0;
      h[1] = -g_mag * cos_eps;
      h[2] = -g_mag * sin_eps;
    }

    // velocity update (preserves momentum exactly)
    double mB_frac = mB / total_mass;
    double mA_frac = mA / total_mass;

    vA[0] -= mB_frac * (one_minus_cos * g[0] + sin_chi * h[0]);
    vA[1] -= mB_frac * (one_minus_cos * g[1] + sin_chi * h[1]);
    vA[2] -= mB_frac * (one_minus_cos * g[2] + sin_chi * h[2]);

    vB[0] += mA_frac * (one_minus_cos * g[0] + sin_chi * h[0]);
    vB[1] += mA_frac * (one_minus_cos * g[1] + sin_chi * h[1]);
    vB[2] += mA_frac * (one_minus_cos * g[2] + sin_chi * h[2]);
  }
}

/* ----------------------------------------------------------------------
   Coulomb logarithm from Debye-length formulation
   ln(Lambda) = max(2, ln(12*pi * ne * lambda_D^3))
   lambda_D = sqrt(epsilon_0 * Te / (ne * e^2))
------------------------------------------------------------------------- */

double FixCollNanbu::compute_coulomb_log(double ne, double Te_eV)
{
  if (ne <= 0.0 || Te_eV <= 0.0) return 2.0;

  const double echarge   = update->echarge;     // C
  const double epsilon_0 = update->epsilon_0;   // F/m
  const double Te_J      = Te_eV * echarge;     // convert eV to Joules

  double lambda_D = std::sqrt(epsilon_0 * Te_J / (ne * echarge * echarge));
  double arg = 12.0 * M_PI * ne * lambda_D * lambda_D * lambda_D;

  if (arg <= 1.0) return 2.0;
  double lnL = std::log(arg);
  return std::max(2.0, lnL);
}

/* ---------------------------------------------------------------------- */

double FixCollNanbu::memory_usage()
{
  return static_cast<double>(npmax_) * sizeof(int);
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::parse_compute_src(const char *tok, NanbuGridSrc &dst,
                                     const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "fix coll/nanbu: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) != 0)
    error->all(FLERR,
      "fix coll/nanbu: source must be a compute (c_ID[col])");

  dst.kind = NANBU_SRC_COMP;
  const char *name = tok + 2;
  const char *lb   = strchr(name, '[');
  const char *rb   = lb ? strrchr(name, ']') : nullptr;

  if (!lb || !rb || rb <= lb + 1)
    error->all(FLERR,
      "fix coll/nanbu: use c_ID[col] syntax for compute sources");

  int idlen = static_cast<int>(lb - name);
  dst.cid = new char[idlen + 1];
  strncpy(dst.cid, name, idlen);
  dst.cid[idlen] = '\0';
  dst.col = atoi(lb + 1);   // 1-based

  if (dst.col <= 0)
    error->all(FLERR,
      "fix coll/nanbu: compute column must be >= 1");
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::refresh_compute_src(NanbuGridSrc &S)
{
  if (S.kind != NANBU_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr;
  int    *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);

  if (nmap <= 0 || !arr) {
    S.arr_cache = nullptr;
    S.src_index = -1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  S.arr_cache = arr;
  S.src_index = cols ? cols[0] : (S.col - 1);
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double FixCollNanbu::read_cell_src(const NanbuGridSrc &S, int icell)
{
  if (S.kind != NANBU_SRC_COMP) return 0.0;
  if (!S.arr_cache || S.src_index < 0) return 0.0;
  return S.arr_cache[icell][S.src_index];
}
