/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/background — Kokkos backend (gate 9a).
   See the header for scope. Kernel is a line-for-line device port of
   FixCoulombBase::nanbu_background_cell's per-particle body (background
   branch, Boris mode), with plasma sampled from the per-tri mesh views.
------------------------------------------------------------------------- */

#include "fix_coulomb_background_kokkos.h"

#include <cstdlib>

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_background.h"
#include "grid_kokkos.h"
#include "particle_kokkos.h"
#include "pusher_kokkos.h"
#include "sparta.h"
#include "sparta_masks.h"
#include "update.h"
#include "update_kokkos.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixCoulombBackgroundKokkos::FixCoulombBackgroundKokkos(SPARTA *sparta,
                                                       int narg, char **arg) :
  FixCoulombBackground(sparta, narg, arg),
  // distinct base seed per OpenEdge pool (chem uses 42345): a shared
  // base made per-thread streams identical across fixes
  rand_pool(32345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(rng_);
#endif
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK;
  datamask_modify = PARTICLE_MASK;

  device_ok = 0;
  warned_fallback = 0;
  ntab_ = 0;
}

/* ---------------------------------------------------------------------- */

FixCoulombBackgroundKokkos::~FixCoulombBackgroundKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixCoulombBackgroundKokkos::init()
{
  FixCoulombBackground::init();

  dim_ = domain->dimension;
  axisym_ = domain->axisymmetric;

  device_ok = 1;
  const char *why = nullptr;
  if (getenv("OE_COUL_HOST")) {
    device_ok = 0; why = "OE_COUL_HOST env override";
  } else if (!use_background_) {
    device_ok = 0; why = "compute-source plasma (device is background-mesh only)";
  } else if (do_binary_) {
    device_ok = 0; why = "binary mode (device is background-drag only)";
  } else if (pd_ && pd_->has_const_bfield()) {
    device_ok = 0; why = "constant-B branch (device B chain is mesh/equilibrium only)";
  } else if (pd_ && (!pd_->dens_e.empty() || !pd_->dens_i.empty())) {
    // old plasma.h5 with a regular (R,Z) raster (or constant mode): the
    // CPU falls back to bilinear raster interp outside the mesh
    // footprint; the device zeroes there -> stay on the host
    device_ok = 0; why = "structured-grid plasma raster (device is mesh-only)";
  }

  UpdateKokkos *update_kk = dynamic_cast<UpdateKokkos *>(update);
  if (device_ok && !update_kk) {
    device_ok = 0; why = "no UpdateKokkos (host run)";
  }

  if (!device_ok) {
    if (comm->me == 0 && screen && !warned_fallback)
      fprintf(screen,"fix coulomb/background/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    return;
  }

  // Nanbu A(s) table -> device (same construction as the host table)
  scatter_table_.initialize();
  {
    const int npoints = 801;
    ntab_ = npoints;
    d_s_tab = DAT::t_float_1d("coul:s_tab",npoints);
    d_A_tab = DAT::t_float_1d("coul:A_tab",npoints);
    auto h_s = Kokkos::create_mirror_view(d_s_tab);
    auto h_A = Kokkos::create_mirror_view(d_A_tab);
    double A_power = 5.0;
    for (int i = 0; i < npoints; i++) {
      double A = pow(10.0, A_power);
      double s = -log(1.0 / tanh(A) - 1.0 / A);
      h_s(i) = s;
      h_A(i) = A;
      A_power -= 0.01;
    }
    Kokkos::deep_copy(d_s_tab,h_s);
    Kokkos::deep_copy(d_A_tab,h_A);
  }
}

/* ---------------------------------------------------------------------- */

void FixCoulombBackgroundKokkos::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  UpdateKokkos *update_kk = dynamic_cast<UpdateKokkos *>(update);

  // the mesh views are built at run() setup; check availability here
  // (not in init, which runs before UpdateKokkos::run's setup)
  int dev = device_ok;
  const char *why = nullptr;
  if (dev && (!update_kk->oe_has_mesh_b || !update_kk->oe_has_mesh_plasma)) {
    dev = 0; why = "device mesh B/plasma views not built";
  }
  if (dev && !update_kk->oe_has_mesh_drag) {
    dev = 0; why = "plasma file lacks mesh ni/upar fields";
  }

  if (!dev) {
    if (!warned_fallback && comm->me == 0 && screen && why)
      fprintf(screen,"fix coulomb/background/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK);
    FixCoulombBackground::end_of_step();
    particle_kk->modify(Host,PARTICLE_MASK);
    particle_kk->sync(Device,PARTICLE_MASK);
    return;
  }

  if (particle->nlocal == 0 || grid->nlocal == 0) return;

  // bind views
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  GridKokkos *grid_kk = (GridKokkos *) grid;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  grid_kk->sync(Device,CINFO_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species   = particle_kk->k_species.d_view;
  d_cinfo     = grid_kk->k_cinfo.d_view;

  col_x0_ = pd_ ? pd_->column_x0 : 0.0;
  col_y0_ = pd_ ? pd_->column_y0 : 0.0;

  d_vtx_r    = update_kk->d_oe_mesh_vtx_r;
  d_vtx_z    = update_kk->d_oe_mesh_vtx_z;
  d_tri      = update_kk->d_oe_mesh_tri;
  d_tri_te   = update_kk->d_oe_mesh_tri_te;
  d_tri_ti   = update_kk->d_oe_mesh_tri_ti;
  d_tri_ne   = update_kk->d_oe_mesh_tri_ne;
  d_tri_ni   = update_kk->d_oe_mesh_tri_ni;
  d_tri_upar = update_kk->d_oe_mesh_tri_upar;
  d_tri_br   = update_kk->d_oe_mesh_tri_br;
  d_tri_bz   = update_kk->d_oe_mesh_tri_bz;
  d_tri_bt   = update_kk->d_oe_mesh_tri_bt;
  d_tri_rmin = update_kk->d_oe_mesh_tri_rmin;
  d_tri_rmax = update_kk->d_oe_mesh_tri_rmax;
  d_tri_zmin = update_kk->d_oe_mesh_tri_zmin;
  d_tri_zmax = update_kk->d_oe_mesh_tri_zmax;
  d_hash_off = update_kk->d_oe_hash_offset;
  d_hash_ent = update_kk->d_oe_hash_entries;
  hash_rmin_ = update_kk->oe_mesh_hash_rmin;
  hash_zmin_ = update_kk->oe_mesh_hash_zmin;
  hash_dr_   = update_kk->oe_mesh_hash_dr;
  hash_dz_   = update_kk->oe_mesh_hash_dz;
  hash_nr_   = update_kk->oe_mesh_hash_nr;
  hash_nz_   = update_kk->oe_mesh_hash_nz;
  ntri_      = update_kk->oe_mesh_ntri;

  has_equ_ = update_kk->oe_has_equilibrium;
  if (has_equ_) {
    d_equ_r   = update_kk->d_oe_equ_r;
    d_equ_z   = update_kk->d_oe_equ_z;
    d_equ_psi = update_kk->d_oe_equ_psi;
    equ_btf_  = update_kk->oe_equ_btf;
    equ_rtf_  = update_kk->oe_equ_rtf;
    equ_jm_   = update_kk->oe_equ_jm;
    equ_km_   = update_kk->oe_equ_km;
    has_equ_bmaps_ = update_kk->oe_has_equ_bmaps;
    if (has_equ_bmaps_) {
      d_equ_br = update_kk->d_oe_equ_br;
      d_equ_bt = update_kk->d_oe_equ_bt;
      d_equ_bz = update_kk->d_oe_equ_bz;
    }
  }

  dtc_     = update->dt * nevery;
  echarge_ = update->echarge;
  eps0_    = update->epsilon_0;
  mbg_     = m_bg_;
  qbg_     = q_bg_;

  const int nlocal = particle->nlocal;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixCoulombBg>(0,nlocal),*this);
  Kokkos::fence();
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);
}

/* ----------------------------------------------------------------------
   per-particle Nanbu background scatter (device twin of the CPU
   nanbu_background_cell body, background branch, Boris mode)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixCoulombBackgroundKokkos::operator()(TagFixCoulombBg,
                                            const int &i) const
{
  Particle::OnePart &p = d_particles(i);
  const int isp = p.ispecies;
  const double Zq = d_species(isp).charge;
  if (Zq == 0.0) return;

  // CPU guard: whole cell skipped when volume <= 0
  const int icell = p.icell;
  if (icell < 0 || icell >= (int) d_cinfo.extent(0)) return;  // defensive
  const double volume = d_cinfo(icell).volume / d_cinfo(icell).weight;
  if (volume <= 0.0) return;

  // column-axis offset (3D linear-device decks, 'column_axis x0 y0'):
  // CPU sparta_to_RZ subtracts it in every point query; shift once so
  // R and the cyl->Cartesian rotation both see column coordinates
  double xq[3] = {p.x[0], p.x[1], p.x[2]};
  if (dim_ == 3 && !axisym_) { xq[0] -= col_x0_; xq[1] -= col_y0_; }

  // plasma at the particle position (tri-constant, mesh branch of
  // interp2D; a miss = CPU's empty-structured-grid fallback = zeros)
  const int tri = MeshKokkos::locate_tri_at_point(
      xq, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_);

  double Te_eV = 0.0, ne = 0.0, Ti_eV = 0.0, Ni_bg = 0.0, Vpar_bg = 0.0;
  if (tri >= 0) {
    Te_eV   = d_tri_te(tri) > 0.0 ? d_tri_te(tri) : 0.0;
    ne      = d_tri_ne(tri) > 0.0 ? d_tri_ne(tri) : 0.0;
    Ti_eV   = d_tri_ti(tri) > 0.0 ? d_tri_ti(tri) : 0.0;
    Ni_bg   = d_tri_ni(tri) > 0.0 ? d_tri_ni(tri) : 0.0;
    Vpar_bg = d_tri_upar(tri);
  }

  double B[3] = {0.0,0.0,0.0};
  const bool gotB = MeshKokkos::query_bfield_at_point(
      xq, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_tri_br, d_tri_bz, d_tri_bt,
      d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_, B);
  if (!gotB && has_equ_) {
    // native maps preferred; no psi fallback when they exist (matches
    // the CPU equ_bfield_at chain, slag b05b4687)
    if (has_equ_bmaps_)
      EquilibriumKokkos::query_bfield_native_maps(
          xq, dim_, axisym_, d_equ_r, d_equ_z,
          d_equ_br, d_equ_bt, d_equ_bz, equ_jm_, equ_km_, B);
    else
      EquilibriumKokkos::query_bfield_at_point(
          xq, dim_, axisym_, d_equ_r, d_equ_z, d_equ_psi,
          equ_btf_, equ_rtf_, equ_jm_, equ_km_, B);
  }
  const double Bx = B[0], By = B[1], Bz = B[2];

  if (Ni_bg <= 0.0 || Ti_eV <= 0.0) return;

  // Coulomb log (CPU compute_coulomb_log)
  double lnLambda = 2.0;
  if (ne > 0.0 && Te_eV > 0.0) {
    const double Te_J = Te_eV * echarge_;
    const double lambda_D =
        Kokkos::sqrt(eps0_ * Te_J / (ne * echarge_ * echarge_));
    const double arg = 12.0 * M_PI * ne * lambda_D * lambda_D * lambda_D;
    if (arg > 1.0) {
      const double lnL = Kokkos::log(arg);
      lnLambda = lnL > 2.0 ? lnL : 2.0;
    }
  }

  const double Bmag = Kokkos::sqrt(Bx*Bx + By*By + Bz*Bz);
  double bhat[3], e1[3], e2[3];
  if (Bmag > 0.0) {
    bhat[0] = Bx / Bmag; bhat[1] = By / Bmag; bhat[2] = Bz / Bmag;
  } else {
    bhat[0] = 0.0; bhat[1] = 0.0; bhat[2] = 1.0;
  }

  double ax[3] = {1.0, 0.0, 0.0};
  if (Kokkos::fabs(bhat[0]) > 0.9) { ax[0] = 0.0; ax[1] = 1.0; }
  const double dot = ax[0]*bhat[0] + ax[1]*bhat[1] + ax[2]*bhat[2];
  e1[0] = ax[0] - dot * bhat[0];
  e1[1] = ax[1] - dot * bhat[1];
  e1[2] = ax[2] - dot * bhat[2];
  const double e1mag =
      Kokkos::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
  if (e1mag <= 0.0) return;
  e1[0] /= e1mag; e1[1] /= e1mag; e1[2] /= e1mag;
  e2[0] = bhat[1]*e1[2] - bhat[2]*e1[1];
  e2[1] = bhat[2]*e1[0] - bhat[0]*e1[2];
  e2[2] = bhat[0]*e1[1] - bhat[1]*e1[0];

  const double v_thermal = Kokkos::sqrt(Ti_eV * echarge_ / mbg_);
  double *v = p.v;
  const double m_test = d_species(isp).mass;
  const double q_test = Zq * echarge_;

  rand_type rand_gen = rand_pool.get_state();

  // virtual Maxwellian partner
  const double vpar = Vpar_bg + rand_gen.normal() * v_thermal;
  const double vp1  = rand_gen.normal() * v_thermal;
  const double vp2  = rand_gen.normal() * v_thermal;

  double v_bg[3];
  v_bg[0] = vpar * bhat[0] + vp1 * e1[0] + vp2 * e2[0];
  v_bg[1] = vpar * bhat[1] + vp1 * e1[1] + vp2 * e2[1];
  v_bg[2] = vpar * bhat[2] + vp1 * e1[2] + vp2 * e2[2];

  double g[3];
  g[0] = v_bg[0] - v[0];
  g[1] = v_bg[1] - v[1];
  g[2] = v_bg[2] - v[2];
  const double g_mag = Kokkos::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
  if (g_mag == 0.0) { rand_pool.free_state(rand_gen); return; }

  const double total_mass = m_test + mbg_;
  const double mu = m_test * mbg_ / total_mass;

  const double g_mag3 = g_mag * g_mag * g_mag;
  const double qq = q_test * qbg_;
  const double s_factor = qq * qq * Ni_bg * lnLambda * dtc_ /
                          (4.0 * M_PI * eps0_ * eps0_ * mu * mu);

  double s_ab;
  if (6.0 * g_mag3 < s_factor) s_ab = 7.0;
  else s_ab = s_factor / g_mag3;

  double cos_chi;
  if (s_ab > 6.0) {
    cos_chi = 2.0 * rand_gen.drand() - 1.0;
  } else if (s_ab > 0.01) {
    // A(s): CPU NanbuScatterTable::get_A — asymptotics + binary search
    double A;
    if (s_ab < 2.0e-3) A = 1.0 / s_ab;
    else if (s_ab > d_s_tab(ntab_ - 1)) A = 3.0 * Kokkos::exp(-s_ab);
    else {
      int lo = 0, hi = ntab_ - 1;
      while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (d_s_tab(mid) <= s_ab) lo = mid;
        else hi = mid;
      }
      const double ds = d_s_tab(hi) - d_s_tab(lo);
      if (ds < 1.0e-30) A = d_A_tab(lo);
      else {
        const double frac = (s_ab - d_s_tab(lo)) / ds;
        A = d_A_tab(lo) + frac * (d_A_tab(hi) - d_A_tab(lo));
      }
    }
    const double U = rand_gen.drand();
    cos_chi = (1.0 / A) *
        Kokkos::log(Kokkos::exp(-A) + 2.0 * Kokkos::sinh(A) * U);
  } else {
    double U = rand_gen.drand();
    if (U < 1.0e-30) U = 1.0e-30;
    cos_chi = 1.0 + s_ab * Kokkos::log(U);
  }

  if (cos_chi > 1.0)  cos_chi = 1.0;
  if (cos_chi < -1.0) cos_chi = -1.0;

  const double one_minus_cos = 1.0 - cos_chi;
  const double sin_chi = Kokkos::sqrt(1.0 - cos_chi * cos_chi);

  const double eps_angle = 2.0 * M_PI * rand_gen.drand();
  const double cos_eps = Kokkos::cos(eps_angle);
  const double sin_eps = Kokkos::sin(eps_angle);

  const double g_perp = Kokkos::sqrt(g[1]*g[1] + g[2]*g[2]);
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

  const double dg0 = sin_chi * h[0] - one_minus_cos * g[0];
  const double dg1 = sin_chi * h[1] - one_minus_cos * g[1];
  const double dg2 = sin_chi * h[2] - one_minus_cos * g[2];

  const double m_bg_frac = mbg_ / total_mass;
  v[0] -= m_bg_frac * dg0;
  v[1] -= m_bg_frac * dg1;
  v[2] -= m_bg_frac * dg2;

  rand_pool.free_state(rand_gen);
}
