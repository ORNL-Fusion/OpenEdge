/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix coll/nanbu: binary Coulomb collisions via the Nanbu (1997) /
    Takizuka-Abe (1977) algorithm.

    Syntax:
      fix ID coll/nanbu Nevery plasma TeSrc NeSrc \
          [nobinary] \
          [background A_bg Z_bg TiSrc NiSrc VparSrc BxSrc BySrc BzSrc]

    Example (binary only):
      fix nanbu coll/nanbu 1 plasma c_cplasma[7] c_cplasma[10]

    Example (binary + background D+ at Z=1, A=2):
      fix nanbu coll/nanbu 1 plasma c_cplasma[7] c_cplasma[10] \
          background 2.0 1.0 c_cplasma[8] c_cplasma[10] c_cplasma[11] \
          c_cplasma[1] c_cplasma[2] c_cplasma[3]

    Example (background-only electron collisions, no binary):
      fix nanbu_e coll/nanbu 1 plasma c_cplasma[7] c_cplasma[10] \
          nobinary \
          background 5.486e-4 -1.0 c_cplasma[7] c_cplasma[10] \
          c_cplasma[11] c_cplasma[1] c_cplasma[2] c_cplasma[3]

    Te (eV) and Ne (m^-3) from per-grid computes are used for the
    Coulomb logarithm.  The particle density entering the scattering-
    parameter s is computed from the actual charged simulation particles
    in each cell: n_partner = N_charged * fnum / V_cell.

    When the background keyword is present, each charged particle also
    undergoes a Nanbu collision with a virtual partner sampled from a
    Maxwellian at the prescribed Ti, Ni, Vpar along the local B-field.
------------------------------------------------------------------------- */

#include "fix_coll_nanbu.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "particle.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "update.h"
#include "fix_plasma_data.h"
#include "openedge_geom.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16
#define DELTAPART 128
enum { INT, DOUBLE };

/* ---------------------------------------------------------------------- */

FixCollNanbu::FixCollNanbu(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  rng_(nullptr),
  use_plasma_data_(0),
  plasma_fix_id_(),
  pd_(nullptr),
  do_binary_(1),
  have_background_(0),
  A_bg_(0.0), Z_bg_(0.0), m_bg_(0.0), q_bg_(0.0),
  npmax_(0),
  plist_(nullptr)
{
  // Syntax: fix ID coll/nanbu Nevery {plasma TeSrc NeSrc | plasma_data FIXID}
  //         [nobinary]
  //         [background A_bg Z_bg [TiSrc NiSrc VparSrc BxSrc BySrc BzSrc]]

  if (narg < 5)
    error->all(FLERR,
      "Illegal fix coll/nanbu command "
      "(need: nevery {plasma TeSrc NeSrc | plasma_data FIXID})");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg], "plasma_data") == 0) {
    iarg++;
    if (iarg >= narg)
      error->all(FLERR, "fix coll/nanbu: plasma_data needs a fix ID");
    use_plasma_data_ = 1;
    plasma_fix_id_ = arg[iarg++];
  } else {
    if (strcmp(arg[iarg++], "plasma") != 0)
      error->all(FLERR, "fix coll/nanbu: missing 'plasma' keyword");
    if (iarg + 2 > narg)
      error->all(FLERR,
        "Illegal fix coll/nanbu command (need: nevery plasma TeSrc NeSrc)");
    parse_compute_src(arg[iarg++], srcTe_, "Te");
    parse_compute_src(arg[iarg++], srcNe_, "Ne");
  }

  // optional nobinary keyword
  if (iarg < narg && strcmp(arg[iarg], "nobinary") == 0) {
    do_binary_ = 0;
    iarg++;
  }

  // optional background keyword
  if (iarg < narg && strcmp(arg[iarg], "background") == 0) {
    iarg++;
    if (iarg + 2 > narg)
      error->all(FLERR, "fix coll/nanbu background: need A_bg Z_bg");

    A_bg_ = input->numeric(FLERR, arg[iarg++]);
    Z_bg_ = input->numeric(FLERR, arg[iarg++]);

    if (!use_plasma_data_) {
      if (iarg + 6 > narg)
        error->all(FLERR,
          "fix coll/nanbu background: need A_bg Z_bg TiSrc NiSrc VparSrc BxSrc BySrc BzSrc");
      parse_compute_src(arg[iarg++], srcTi_bg_,  "Ti_bg");
      parse_compute_src(arg[iarg++], srcNi_bg_,  "Ni_bg");
      parse_compute_src(arg[iarg++], srcVpar_bg_, "Vpar_bg");
      parse_compute_src(arg[iarg++], srcBx_,     "Bx");
      parse_compute_src(arg[iarg++], srcBy_,     "By");
      parse_compute_src(arg[iarg++], srcBz_,     "Bz");
    }

    // precompute background mass and charge
    const double amu = 1.66053906660e-27;  // kg
    m_bg_ = A_bg_ * amu;
    q_bg_ = Z_bg_ * update->echarge;

    have_background_ = 1;
  }
}

/* ---------------------------------------------------------------------- */

FixCollNanbu::~FixCollNanbu()
{
  if (copymode) return;
  delete rng_;
  rng_ = nullptr;
  memory->destroy(plist_);
  delete[] srcTe_.cid;
  delete[] srcTe_.pname;
  delete[] srcNe_.cid;
  delete[] srcNe_.pname;
  if (have_background_) {
    delete[] srcTi_bg_.cid;
    delete[] srcTi_bg_.pname;
    delete[] srcNi_bg_.cid;
    delete[] srcNi_bg_.pname;
    delete[] srcVpar_bg_.cid;
    delete[] srcVpar_bg_.pname;
    delete[] srcBx_.cid;
    delete[] srcBx_.pname;
    delete[] srcBy_.cid;
    delete[] srcBy_.pname;
    delete[] srcBz_.cid;
    delete[] srcBz_.pname;
  }
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
  auto bind_compute = [&](CollGridSrc &S, const char *label) {
    if (S.kind == COLL_SRC_COMP) {
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
      return;
    }
    if (S.kind == COLL_SRC_PCUSTOM) {
      S.ipcustom = particle->find_custom(S.pname);
      if (S.ipcustom < 0) {
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "fix coll/nanbu: particle custom '%s' for %s not found",
                 S.pname, label);
        error->all(FLERR, msg);
      }
      if (particle->etype[S.ipcustom] != DOUBLE)
        error->all(FLERR,
          "fix coll/nanbu: particle custom source must be floating point");
      if (particle->esize[S.ipcustom] != 0)
        error->all(FLERR,
          "fix coll/nanbu: particle custom source must be a vector");
      S.ipwhich = particle->ewhich[S.ipcustom];
    }
  };

  if (use_plasma_data_) {
    const int ifix = modify->find_fix(plasma_fix_id_.c_str());
    if (ifix < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix coll/nanbu: plasma_data fix '%s' not found",
               plasma_fix_id_.c_str());
      error->all(FLERR, msg);
    }
    pd_ = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
    if (!pd_)
      error->all(FLERR,
        "fix coll/nanbu: plasma_data fix must be style plasma/data");
    pd_->init();
  } else {
    bind_compute(srcTe_, "Te");
    bind_compute(srcNe_, "Ne");
  }

  if (have_background_ && !use_plasma_data_) {
    bind_compute(srcTi_bg_,  "Ti_bg");
    bind_compute(srcNi_bg_,  "Ni_bg");
    bind_compute(srcVpar_bg_, "Vpar_bg");
    bind_compute(srcBx_,     "Bx");
    bind_compute(srcBy_,     "By");
    bind_compute(srcBz_,     "Bz");
  }
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  if (!particle->sorted) particle->sort();

  // refresh compute caches for this timestep
  if (!use_plasma_data_) {
    refresh_compute_src(srcTe_);
    refresh_compute_src(srcNe_);
  }

  if (have_background_ && !use_plasma_data_) {
    refresh_compute_src(srcTi_bg_);
    refresh_compute_src(srcNi_bg_);
    refresh_compute_src(srcVpar_bg_);
    refresh_compute_src(srcBx_);
    refresh_compute_src(srcBy_);
    refresh_compute_src(srcBz_);
  }

  if (grid->nlocal == 0) return;
  if (particle->nlocal == 0) return;

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  int               *next      = particle->next;
  Grid::ChildInfo   *cinfo     = grid->cinfo;
  const int nglocal = grid->nlocal;

  for (int icell = 0; icell < nglocal; icell++) {
    int np = cinfo[icell].count;
    if (np < 1) continue;

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

    // binary particle-particle collisions (need >= 2 particles)
    if (do_binary_ && n >= 2) nanbu_collisions_cell(icell, n);

    // background collisions (each particle vs virtual partner)
    if (have_background_ && n >= 1) nanbu_background_cell(icell, n);
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
    const double Te_eV = std::max(
      use_plasma_data_
        ? 0.5 * (pd_interp(pd_->temp_e, pA) + pd_interp(pd_->temp_e, pB))
        : 0.5 * (read_src(srcTe_, idxA, icell) + read_src(srcTe_, idxB, icell)),
      0.0);
    const double ne = std::max(
      use_plasma_data_
        ? 0.5 * (pd_interp(pd_->dens_e, pA) + pd_interp(pd_->dens_e, pB))
        : 0.5 * (read_src(srcNe_, idxA, icell) + read_src(srcNe_, idxB, icell)),
      0.0);
    const double lnLambda = compute_coulomb_log(ne, Te_eV);

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
    // s = (qA*qB)^2 * n_partner * ln(Lambda) * dt / (4*pi*eps0^2 * mu^2 * g^3)
    double g_mag3 = g_mag * g_mag * g_mag;
    double qq = qA * qB;
    double s_factor = qq * qq * n_partner * lnLambda * dt /
                      (4.0 * M_PI * epsilon_0 * epsilon_0 * mu * mu);

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

    // velocity update: Δg = g' - g = -(1-cos_chi)*g + sin_chi*h
    // preserves both momentum and kinetic energy exactly
    double mB_frac = mB / total_mass;
    double mA_frac = mA / total_mass;

    double dg0 = sin_chi * h[0] - one_minus_cos * g[0];
    double dg1 = sin_chi * h[1] - one_minus_cos * g[1];
    double dg2 = sin_chi * h[2] - one_minus_cos * g[2];

    vA[0] -= mB_frac * dg0;
    vA[1] -= mB_frac * dg1;
    vA[2] -= mB_frac * dg2;

    vB[0] += mA_frac * dg0;
    vB[1] += mA_frac * dg1;
    vB[2] += mA_frac * dg2;
  }
}

/* ----------------------------------------------------------------------
   Background Nanbu collisions: each charged particle collides with a
   virtual partner sampled from a Maxwellian at the local background
   plasma conditions (Ti, Ni, Vpar along B).
   Only the simulation particle velocity is updated.
------------------------------------------------------------------------- */

void FixCollNanbu::nanbu_background_cell(int icell, int np)
{
  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  Grid::ChildInfo   *cinfo     = grid->cinfo;

  // cell volume
  double volume = cinfo[icell].volume / cinfo[icell].weight;
  if (volume <= 0.0) return;

  const double echarge   = update->echarge;
  const double epsilon_0 = update->epsilon_0;
  const double dt        = update->dt * nevery;

  // process each charged particle
  for (int i = 0; i < np; i++) {
    int idx = plist_[i];
    int isp = particles[idx].ispecies;
    if (species[isp].charge == 0.0) continue;

    const Particle::OnePart &part = particles[idx];
    double Te_eV   = use_plasma_data_ ? std::max(pd_interp(pd_->temp_e, part), 0.0)
                                      : std::max(read_src(srcTe_, idx, icell), 0.0);
    double ne      = use_plasma_data_ ? std::max(pd_interp(pd_->dens_e, part), 0.0)
                                      : std::max(read_src(srcNe_, idx, icell), 0.0);
    double Ti_eV   = use_plasma_data_ ? std::max(pd_interp(pd_->temp_i, part), 0.0)
                                      : std::max(read_src(srcTi_bg_, idx, icell), 0.0);
    double Ni_bg   = use_plasma_data_ ? std::max(pd_interp(pd_->dens_i, part), 0.0)
                                      : std::max(read_src(srcNi_bg_, idx, icell), 0.0);
    double Vpar_bg = use_plasma_data_ ? pd_interp(pd_->parr_flow, part)
                                      : read_src(srcVpar_bg_, idx, icell);
    double Bx = 0.0, By = 0.0, Bz = 0.0;
    if (use_plasma_data_) pd_bfield_sparta(part, Bx, By, Bz);
    else {
      Bx = read_src(srcBx_, idx, icell);
      By = read_src(srcBy_, idx, icell);
      Bz = read_src(srcBz_, idx, icell);
    }

    if (Ni_bg <= 0.0 || Ti_eV <= 0.0) continue;

    double lnLambda = compute_coulomb_log(ne, Te_eV);
    double Bmag = std::sqrt(Bx*Bx + By*By + Bz*Bz);
    double bhat[3], e1[3], e2[3];

    if (Bmag > 0.0) {
      bhat[0] = Bx / Bmag;
      bhat[1] = By / Bmag;
      bhat[2] = Bz / Bmag;
    } else {
      bhat[0] = 0.0; bhat[1] = 0.0; bhat[2] = 1.0;
    }

    double ax[3] = {1.0, 0.0, 0.0};
    if (std::abs(bhat[0]) > 0.9) { ax[0] = 0.0; ax[1] = 1.0; }
    double dot = ax[0]*bhat[0] + ax[1]*bhat[1] + ax[2]*bhat[2];
    e1[0] = ax[0] - dot * bhat[0];
    e1[1] = ax[1] - dot * bhat[1];
    e1[2] = ax[2] - dot * bhat[2];
    double e1mag = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
    if (e1mag <= 0.0) continue;
    e1[0] /= e1mag; e1[1] /= e1mag; e1[2] /= e1mag;
    e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
    e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
    e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];

    double v_thermal = std::sqrt(Ti_eV * echarge / m_bg_);
    double *v = particles[idx].v;
    double m_test = species[isp].mass;
    double q_test = species[isp].charge * echarge;

    // sample virtual partner velocity from Maxwellian
    double vpar  = Vpar_bg + box_muller() * v_thermal;
    double vp1   = box_muller() * v_thermal;
    double vp2   = box_muller() * v_thermal;

    double v_bg[3];
    v_bg[0] = vpar * bhat[0] + vp1 * e1[0] + vp2 * e2[0];
    v_bg[1] = vpar * bhat[1] + vp1 * e1[1] + vp2 * e2[1];
    v_bg[2] = vpar * bhat[2] + vp1 * e1[2] + vp2 * e2[2];

    // relative velocity
    double g[3];
    g[0] = v_bg[0] - v[0];
    g[1] = v_bg[1] - v[1];
    g[2] = v_bg[2] - v[2];
    double g_mag = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

    if (g_mag == 0.0) continue;

    // reduced mass
    double total_mass = m_test + m_bg_;
    double mu = m_test * m_bg_ / total_mass;

    // scattering parameter s using background density
    double g_mag3 = g_mag * g_mag * g_mag;
    double qq = q_test * q_bg_;
    double s_factor = qq * qq * Ni_bg * lnLambda * dt /
                      (4.0 * M_PI * epsilon_0 * epsilon_0 * mu * mu);

    double s_ab;
    if (6.0 * g_mag3 < s_factor) {
      s_ab = 7.0;
    } else {
      s_ab = s_factor / g_mag3;
    }

    // sample cos(chi)
    double cos_chi;
    if (s_ab > 6.0) {
      cos_chi = 2.0 * rng_->uniform() - 1.0;
    } else if (s_ab > 0.01) {
      double A = scatter_table_.get_A(s_ab);
      double U = rng_->uniform();
      cos_chi = (1.0 / A) * std::log(std::exp(-A) + 2.0 * std::sinh(A) * U);
    } else {
      double U = rng_->uniform();
      if (U < 1.0e-30) U = 1.0e-30;
      cos_chi = 1.0 + s_ab * std::log(U);
    }

    if (cos_chi > 1.0)  cos_chi = 1.0;
    if (cos_chi < -1.0) cos_chi = -1.0;

    double one_minus_cos = 1.0 - cos_chi;
    double sin_chi = std::sqrt(1.0 - cos_chi * cos_chi);

    // random azimuthal angle
    double eps_angle = 2.0 * M_PI * rng_->uniform();
    double cos_eps = std::cos(eps_angle);
    double sin_eps = std::sin(eps_angle);

    // perpendicular component for rotation
    double g_perp = std::sqrt(g[1]*g[1] + g[2]*g[2]);
    double h[3];
    if (g_perp > 1.0e-12 * g_mag) {
      h[0] =  g_perp * cos_eps;
      h[1] = -(g[0]*g[1]*cos_eps + g_mag*g[2]*sin_eps) / g_perp;
      h[2] = -(g[0]*g[2]*cos_eps - g_mag*g[1]*sin_eps) / g_perp;
    } else {
      h[0] = 0.0;
      h[1] = -g_mag * cos_eps;
      h[2] = -g_mag * sin_eps;
    }

    // velocity update: only test particle is modified
    double dg0 = sin_chi * h[0] - one_minus_cos * g[0];
    double dg1 = sin_chi * h[1] - one_minus_cos * g[1];
    double dg2 = sin_chi * h[2] - one_minus_cos * g[2];

    double m_bg_frac = m_bg_ / total_mass;
    v[0] -= m_bg_frac * dg0;
    v[1] -= m_bg_frac * dg1;
    v[2] -= m_bg_frac * dg2;
  }
}

/* ----------------------------------------------------------------------
   Box-Muller transform: generate a standard normal random variate
------------------------------------------------------------------------- */

double FixCollNanbu::box_muller()
{
  double u1 = rng_->uniform();
  double u2 = rng_->uniform();
  if (u1 < 1.0e-30) u1 = 1.0e-30;
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
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

void FixCollNanbu::parse_compute_src(const char *tok, CollGridSrc &dst,
                                     const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "fix coll/nanbu: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) == 0) {
    dst.kind = COLL_SRC_COMP;
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
    return;
  }

  if (strncmp(tok, "p_", 2) == 0) {
    dst.kind = COLL_SRC_PCUSTOM;
    const char *name = tok + 2;
    dst.pname = new char[strlen(name) + 1];
    strcpy(dst.pname, name);
    return;
  }

  error->all(FLERR,
    "fix coll/nanbu: source must be c_ID[col] or p_name");
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind == COLL_SRC_PCUSTOM) {
    if (S.cache_ts == update->ntimestep) return;
    S.pvec_cache = nullptr;
    if (S.ipcustom >= 0) {
      S.ipwhich = particle->ewhich[S.ipcustom];
      if (S.ipwhich >= 0) S.pvec_cache = particle->edvec[S.ipwhich];
    }
    S.cache_ts = update->ntimestep;
    return;
  }
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  // Try tally-style mapped access first (e.g. compute grid)
  double **arr = nullptr;
  int    *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts  = update->ntimestep;
    return;
  }

  // Fallback: standard per-grid array (e.g. compute plasma/fields)
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  // Source is empty for this step
  S.arr_cache = nullptr;
  S.pvec_cache = nullptr;
  S.src_index = -1;
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double FixCollNanbu::read_src(const CollGridSrc &S, int ip, int icell) const
{
  if (S.kind == COLL_SRC_PCUSTOM) {
    if (!S.pvec_cache) return 0.0;
    return S.pvec_cache[ip];
  }
  if (S.kind != COLL_SRC_COMP) return 0.0;
  if (!S.arr_cache || S.src_index < 0) return 0.0;
  return S.arr_cache[icell][S.src_index];
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::particle_rz(const Particle::OnePart &p,
                               double &R, double &Z) const
{
  OpenEdge::sparta_to_RZ(p.x, domain->dimension, domain->axisymmetric, R, Z);
}

/* ---------------------------------------------------------------------- */

double FixCollNanbu::pd_interp(const std::vector<double> &field,
                               const Particle::OnePart &p) const
{
  if (!pd_) return 0.0;
  double R, Z;
  particle_rz(p, R, Z);
  return pd_->interp2D(field, R, Z, p.icell);
}

/* ---------------------------------------------------------------------- */

void FixCollNanbu::pd_bfield_sparta(const Particle::OnePart &p,
                                    double &Bx, double &By, double &Bz) const
{
  Bx = By = Bz = 0.0;
  if (!pd_ || !pd_->has_bfield) return;

  double R, Z;
  particle_rz(p, R, Z);

  double Br = 0.0, Bz_cyl = 0.0, Bt = 0.0;
  pd_->bfield_at(R, Z, Br, Bz_cyl, Bt, p.icell);

  double phi = 0.0;
  if (domain->dimension == 3) phi = std::atan2(p.x[1], p.x[0]);
  OpenEdge::RZphi_force_to_sparta(Br, Bz_cyl, Bt, domain->dimension,
                                   domain->axisymmetric, phi, Bx, By, Bz);
}
