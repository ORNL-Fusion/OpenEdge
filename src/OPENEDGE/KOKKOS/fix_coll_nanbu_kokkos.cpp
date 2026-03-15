/* ----------------------------------------------------------------------
   OpenEdge: Nanbu Coulomb collisions — Kokkos device kernel.
   Background-mode collisions run entirely on device.
   Binary particle-particle collisions fall back to host.
------------------------------------------------------------------------- */

#include "fix_coll_nanbu_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "compute.h"
#include "modify.h"
#include "update.h"
#include "comm.h"
#include "kokkos_base.h"
#include "sparta_masks.h"
#include "memory.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixCollNanbuKokkos::FixCollNanbuKokkos(SPARTA *sparta, int narg, char **arg) :
  FixCollNanbu(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK | CINFO_MASK;
  datamask_modify = PARTICLE_MASK;

  scatter_npts = 0;
}

FixCollNanbuKokkos::~FixCollNanbuKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

int FixCollNanbuKokkos::resolve_column(const CollGridSrc &S)
{
  // S.col is 1-based column index into the compute's array_grid
  if (S.kind != COLL_SRC_COMP) return -1;
  return S.col - 1;  // convert to 0-based
}

/* ---------------------------------------------------------------------- */

void FixCollNanbuKokkos::init()
{
  FixCollNanbu::init();

  // copy scatter table to device
  scatter_table_.initialize();

  // access private vectors via the public get_A interface
  // rebuild the table on host then copy
  scatter_npts = 801;
  auto d_s = t_double_1d(Kokkos::view_alloc("nanbu:s_table"), scatter_npts);
  auto d_A = t_double_1d(Kokkos::view_alloc("nanbu:A_table"), scatter_npts);
  auto h_s = Kokkos::create_mirror_view(d_s);
  auto h_A = Kokkos::create_mirror_view(d_A);

  double A_power = 5.0;
  for (int i = 0; i < scatter_npts; i++) {
    double A = pow(10.0, A_power);
    double s = -log(1.0 / tanh(A) - 1.0 / A);
    h_s(i) = s;
    h_A(i) = A;
    A_power -= 0.01;
  }
  Kokkos::deep_copy(d_s, h_s);
  Kokkos::deep_copy(d_A, h_A);
  d_s_table = d_s;
  d_A_table = d_A;

  // resolve column indices from the CollGridSrc structs
  col_Te = resolve_column(srcTe_);
  col_Ne = resolve_column(srcNe_);

  if (have_background_) {
    col_Ti_bg  = resolve_column(srcTi_bg_);
    col_Ni_bg  = resolve_column(srcNi_bg_);
    col_Vpar_bg = resolve_column(srcVpar_bg_);
    col_Bx     = resolve_column(srcBx_);
    col_By     = resolve_column(srcBy_);
    col_Bz     = resolve_column(srcBz_);
  }

  // cache background species constants
  kk_A_bg = A_bg_;
  kk_Z_bg = Z_bg_;
  kk_m_bg = m_bg_;
  kk_q_bg = q_bg_;

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(rng_);
#endif
}

/* ---------------------------------------------------------------------- */

void FixCollNanbuKokkos::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  GridKokkos *grid_kk = (GridKokkos *) grid;

  // sort particles into cells (populates d_plist, d_cellcount)
  if (!particle_kk->sorted_kk) particle_kk->sort_kokkos();

  // refresh compute caches on host and ensure compute has run
  refresh_compute_src(srcTe_);
  refresh_compute_src(srcNe_);
  if (have_background_) {
    refresh_compute_src(srcTi_bg_);
    refresh_compute_src(srcNi_bg_);
    refresh_compute_src(srcVpar_bg_);
    refresh_compute_src(srcBx_);
    refresh_compute_src(srcBy_);
    refresh_compute_src(srcBz_);
  }

  if (grid->nlocal == 0 || particle->nlocal == 0) return;

  // ----- binary collisions: fall back to host -----

  if (do_binary_) {
    particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);
    // use host-side cell lists
    if (!particle->sorted) particle->sort();

    Particle::OnePart *particles = particle->particles;
    Particle::Species *species = particle->species;
    int *next = particle->next;
    Grid::ChildInfo *cinfo = grid->cinfo;
    const int nglocal = grid->nlocal;

    for (int icell = 0; icell < nglocal; icell++) {
      int np = cinfo[icell].count;
      if (np < 2) continue;
      if (np > npmax_) {
        while (np > npmax_) npmax_ += 256;
        memory->destroy(plist_);
        memory->create(plist_, npmax_, "coll/nanbu:plist");
      }
      int n = 0, ip = cinfo[icell].first;
      while (ip >= 0) { plist_[n++] = ip; ip = next[ip]; }
      nanbu_collisions_cell(icell, n);
    }
    particle_kk->modify(Host, PARTICLE_MASK);
  }

  // ----- background collisions: device kernel -----

  if (!have_background_) return;

  // get device views
  particle_kk->sync(Device, PARTICLE_MASK | SPECIES_MASK);
  grid_kk->sync(Device, CINFO_MASK);

  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  d_cellcount = grid_kk->d_cellcount;
  d_plist = grid_kk->d_plist;

  // get plasma compute device view
  // All srcTe_, srcNe_, srcTi_bg_, etc. reference the same compute
  // (plasma/fields), so grab its d_array_grid once
  Compute *cp = modify->compute[srcTe_.icompute];
  KokkosBase *kk_cp = dynamic_cast<KokkosBase*>(cp);
  if (!kk_cp) {
    // compute doesn't have Kokkos views — fall back to host
    particle_kk->sync(Host, PARTICLE_MASK | SPECIES_MASK);
    if (!particle->sorted) particle->sort();
    Particle::OnePart *particles_h = particle->particles;
    Grid::ChildInfo *cinfo = grid->cinfo;
    int *next = particle->next;
    for (int icell = 0; icell < grid->nlocal; icell++) {
      int np = cinfo[icell].count;
      if (np < 1) continue;
      if (np > npmax_) {
        while (np > npmax_) npmax_ += 256;
        memory->destroy(plist_);
        memory->create(plist_, npmax_, "coll/nanbu:plist");
      }
      int n = 0, ip = cinfo[icell].first;
      while (ip >= 0) { plist_[n++] = ip; ip = next[ip]; }
      nanbu_background_cell(icell, n);
    }
    particle_kk->modify(Host, PARTICLE_MASK);
    return;
  }

  d_plasma = kk_cp->d_array_grid;

  // cache scalar constants
  kk_echarge = update->echarge;
  kk_epsilon_0 = update->epsilon_0;
  kk_fnum = update->fnum;
  kk_dt = update->dt * nevery;

  // launch device kernel over all cells
  const int nglocal = grid->nlocal;
  copymode = 1;
  Kokkos::parallel_for(nglocal, *this);
  copymode = 0;

  // mark particles modified on device
  particle_kk->modify(Device, PARTICLE_MASK);
}

/* ----------------------------------------------------------------------
   Device kernel: background Nanbu collisions for one cell.
   Each charged particle collides with a virtual Maxwellian partner.
   Only modifies particle velocities (no species change, no create/destroy).
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixCollNanbuKokkos::background_cell_kernel(int icell) const
{
  int np = d_cellcount(icell);
  if (np < 1) return;

  // read plasma fields for this cell
  double Te_eV   = d_plasma(icell, col_Te);
  double ne      = d_plasma(icell, col_Ne);
  double Ti_eV   = d_plasma(icell, col_Ti_bg);
  double Ni_bg   = d_plasma(icell, col_Ni_bg);
  double Vpar_bg = d_plasma(icell, col_Vpar_bg);
  double Bx      = d_plasma(icell, col_Bx);
  double By      = d_plasma(icell, col_By);
  double Bz      = d_plasma(icell, col_Bz);

  if (Te_eV < 0.0) Te_eV = 0.0;
  if (ne < 0.0) ne = 0.0;
  if (Ti_eV < 0.0) Ti_eV = 0.0;
  if (Ni_bg <= 0.0 || Ti_eV <= 0.0) return;

  double lnLambda = kk_coulomb_log(ne, Te_eV);

  // B-field unit vector and orthonormal frame
  double Bmag = sqrt(Bx*Bx + By*By + Bz*Bz);
  double bhat[3], e1[3], e2[3];

  if (Bmag > 0.0) {
    bhat[0] = Bx / Bmag; bhat[1] = By / Bmag; bhat[2] = Bz / Bmag;
  } else {
    bhat[0] = 0.0; bhat[1] = 0.0; bhat[2] = 1.0;
  }

  // build e1 perpendicular to bhat
  double ax[3] = {1.0, 0.0, 0.0};
  if (fabs(bhat[0]) > 0.9) { ax[0] = 0.0; ax[1] = 1.0; }
  double dot = ax[0]*bhat[0] + ax[1]*bhat[1] + ax[2]*bhat[2];
  e1[0] = ax[0] - dot*bhat[0];
  e1[1] = ax[1] - dot*bhat[1];
  e1[2] = ax[2] - dot*bhat[2];
  double e1mag = sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
  if (e1mag > 0.0) { e1[0] /= e1mag; e1[1] /= e1mag; e1[2] /= e1mag; }
  // e2 = bhat x e1
  e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
  e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
  e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];

  // thermal speed of background species
  double v_thermal = sqrt(Ti_eV * kk_echarge / kk_m_bg);

  rand_type rng = rand_pool.get_state();

  for (int j = 0; j < np; j++) {
    int idx = d_plist(icell, j);
    int isp = d_particles(idx).ispecies;
    double charge = d_species[isp].charge;
    if (charge == 0.0) continue;

    double *v = d_particles(idx).v;
    double m_test = d_species[isp].mass;
    double q_test = charge * kk_echarge;

    // sample virtual partner velocity from Maxwellian
    double vpar  = Vpar_bg + kk_box_muller(rng) * v_thermal;
    double vp1   = kk_box_muller(rng) * v_thermal;
    double vp2   = kk_box_muller(rng) * v_thermal;

    double v_bg[3];
    v_bg[0] = vpar * bhat[0] + vp1 * e1[0] + vp2 * e2[0];
    v_bg[1] = vpar * bhat[1] + vp1 * e1[1] + vp2 * e2[1];
    v_bg[2] = vpar * bhat[2] + vp1 * e1[2] + vp2 * e2[2];

    // relative velocity
    double g[3];
    g[0] = v_bg[0] - v[0];
    g[1] = v_bg[1] - v[1];
    g[2] = v_bg[2] - v[2];
    double g_mag = sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
    if (g_mag == 0.0) continue;

    // reduced mass
    double total_mass = m_test + kk_m_bg;
    double mu = m_test * kk_m_bg / total_mass;

    // scattering parameter s
    double g_mag3 = g_mag * g_mag * g_mag;
    double qq = q_test * kk_q_bg;
    double s_factor = qq * qq * Ni_bg * lnLambda * kk_dt /
                      (4.0 * MathConst::MY_PI * kk_epsilon_0 * kk_epsilon_0 * mu * mu);

    double s_ab;
    if (6.0 * g_mag3 < s_factor) {
      s_ab = 7.0;
    } else {
      s_ab = s_factor / g_mag3;
    }

    // sample cos(chi) from Nanbu distribution
    double cos_chi;
    if (s_ab > 6.0) {
      cos_chi = 2.0 * rng.drand() - 1.0;
    } else if (s_ab > 0.01) {
      double A = kk_get_A(s_ab);
      double U = rng.drand();
      cos_chi = (1.0 / A) * log(exp(-A) + 2.0 * sinh(A) * U);
    } else {
      double U = rng.drand();
      if (U < 1.0e-30) U = 1.0e-30;
      cos_chi = 1.0 + s_ab * log(U);
    }

    if (cos_chi > 1.0)  cos_chi = 1.0;
    if (cos_chi < -1.0) cos_chi = -1.0;

    double one_minus_cos = 1.0 - cos_chi;
    double sin_chi = sqrt(1.0 - cos_chi * cos_chi);

    // random azimuthal angle
    double eps_angle = 2.0 * MathConst::MY_PI * rng.drand();
    double cos_eps = cos(eps_angle);
    double sin_eps = sin(eps_angle);

    // rotation
    double g_perp = sqrt(g[1]*g[1] + g[2]*g[2]);
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

    // velocity update: only test particle
    double m_bg_frac = kk_m_bg / total_mass;
    v[0] -= m_bg_frac * (sin_chi * h[0] - one_minus_cos * g[0]);
    v[1] -= m_bg_frac * (sin_chi * h[1] - one_minus_cos * g[1]);
    v[2] -= m_bg_frac * (sin_chi * h[2] - one_minus_cos * g[2]);
  }

  rand_pool.free_state(rng);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double FixCollNanbuKokkos::kk_get_A(double s) const
{
  if (s < 2.0e-3) return 1.0 / s;
  if (s > 3.2)    return 3.0 * exp(-s);

  // binary search on d_s_table (monotonically increasing)
  int lo = 0, hi = scatter_npts - 1;
  while (lo < hi - 1) {
    int mid = (lo + hi) / 2;
    if (d_s_table(mid) <= s) lo = mid;
    else hi = mid;
  }

  double ds = d_s_table(hi) - d_s_table(lo);
  if (ds < 1.0e-30) return d_A_table(lo);
  double frac = (s - d_s_table(lo)) / ds;
  return d_A_table(lo) + frac * (d_A_table(hi) - d_A_table(lo));
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double FixCollNanbuKokkos::kk_box_muller(rand_type &rng) const
{
  double u1 = rng.drand();
  double u2 = rng.drand();
  if (u1 < 1.0e-30) u1 = 1.0e-30;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * MathConst::MY_PI * u2);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double FixCollNanbuKokkos::kk_coulomb_log(double ne, double Te_eV) const
{
  if (ne <= 0.0 || Te_eV <= 0.0) return 2.0;

  double Te_J = Te_eV * kk_echarge;
  double lambda_D = sqrt(kk_epsilon_0 * Te_J / (ne * kk_echarge * kk_echarge));
  double arg = 12.0 * MathConst::MY_PI * ne * lambda_D * lambda_D * lambda_D;

  if (arg <= 1.0) return 2.0;
  double lnL = log(arg);
  return (lnL > 2.0) ? lnL : 2.0;
}
