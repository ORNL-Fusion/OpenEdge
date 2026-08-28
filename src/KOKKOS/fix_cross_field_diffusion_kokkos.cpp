/* ----------------------------------------------------------------------
   OpenEdge: fix cross_field_diffusion — Kokkos backend.
   See the header for scope. Kernel is a device twin of the CPU
   start_of_step displacement loop (background branch, 3D).
------------------------------------------------------------------------- */

#include "fix_cross_field_diffusion_kokkos.h"

#include <cstdlib>

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_background.h"
#include "particle_kokkos.h"
#include "pusher_kokkos.h"
#include "sparta.h"
#include "sparta_masks.h"
#include "update.h"
#include "update_kokkos.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixCrossFieldDiffusionKokkos::FixCrossFieldDiffusionKokkos(SPARTA *sparta,
                                                           int narg,
                                                           char **arg) :
  FixCrossFieldDiffusion(sparta, narg, arg),
  // distinct base seed per OpenEdge pool (PWI 22345, coulomb 32345,
  // chem 42345)
  rand_pool(52345 + comm->me
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
  datamask_modify = 0;

  device_ok = 0;
  warned_fallback = 0;
}

/* ---------------------------------------------------------------------- */

FixCrossFieldDiffusionKokkos::~FixCrossFieldDiffusionKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixCrossFieldDiffusionKokkos::init()
{
  FixCrossFieldDiffusion::init();

  dim_ = domain->dimension;
  axisym_ = domain->axisymmetric;

  device_ok = 1;
  const char *why = nullptr;
  if (getenv("OE_CD_HOST")) {
    device_ok = 0; why = "OE_CD_HOST env override";
  } else if (!use_background_) {
    device_ok = 0; why = "compute-source / constant-B fields (device is background-mesh only)";
  } else if (dim_ != 3) {
    device_ok = 0; why = "2D/axisymmetric (device fill is 3D-only)";
  } else if (!dynamic_cast<UpdateKokkos *>(update)) {
    device_ok = 0; why = "no UpdateKokkos (host run)";
  } else if (pd_ && pd_->has_const_bfield()) {
    device_ok = 0; why = "constant-B branch (device B chain is mesh/equilibrium only)";
  } else if (have_grad_pinch_ && pd_ &&
             pd_->rvals.size() >= 2 && !pd_->dens_e.empty()) {
    // the CPU does per-particle FD on the structured raster there; on
    // mesh-only files the gradient is exactly zero and the device path
    // can skip the term with identical physics
    device_ok = 0; why = "gradient_pinch with a structured raster (host FD)";
  }

  if (!device_ok) {
    if (comm->me == 0 && screen && !warned_fallback)
      fprintf(screen,"fix cross_field_diffusion/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    return;
  }

  // psi-pinch gradient inputs (device twin of psi_norm_gradient_at);
  // replicate the CPU validity gates so an unusable equilibrium makes
  // the term vanish exactly as the CPU's `return false` does
  psi_ok_ = 0;
  psi_denom_ = dr_eq_ = dz_eq_ = 0.0;
  r_front_ = r_back_ = z_front_ = z_back_ = 0.0;
  if (have_psi_pinch_ && pd_ && pd_->has_equ &&
      pd_->equ_jm >= 2 && pd_->equ_km >= 2 &&
      pd_->equ_r.size() >= 2 && pd_->equ_z.size() >= 2 &&
      pd_->psirz.size() >= (size_t)(pd_->equ_jm * pd_->equ_km)) {
    dr_eq_ = pd_->equ_r[1] - pd_->equ_r[0];
    dz_eq_ = pd_->equ_z[1] - pd_->equ_z[0];
    psi_denom_ = pd_->psib - pd_->psi_axis;
    r_front_ = pd_->equ_r.front();  r_back_ = pd_->equ_r.back();
    z_front_ = pd_->equ_z.front();  z_back_ = pd_->equ_z.back();
    if (std::fabs(dr_eq_) >= 1.0e-30 && std::fabs(dz_eq_) >= 1.0e-30 &&
        std::fabs(psi_denom_) >= 1.0e-30)
      psi_ok_ = 1;
  }

  col_x0_ = pd_ ? pd_->column_x0 : 0.0;
  col_y0_ = pd_ ? pd_->column_y0 : 0.0;

  if (comm->me == 0 && screen)
    fprintf(screen,"fix cross_field_diffusion/kk: device fill active%s\n",
            (have_psi_pinch_ && psi_ok_) ? " (psi pinch on device)" : "");
}

/* ---------------------------------------------------------------------- */

void FixCrossFieldDiffusionKokkos::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) {
    update->cd_flag = 0;
    return;
  }

  UpdateKokkos *update_kk = dynamic_cast<UpdateKokkos *>(update);

  // the device mesh views are built at run() setup; check here
  int dev = device_ok;
  const char *why = nullptr;
  if (dev && !update_kk->oe_has_mesh_b) {
    dev = 0; why = "device mesh B views not built";
  }
  if (dev && diff_model_ == 2 && !update_kk->oe_has_mesh_plasma) {
    dev = 0; why = "mesh te view absent (bohm model)";
  }
  if (dev && have_psi_pinch_ && psi_ok_ && !update_kk->oe_has_equilibrium) {
    dev = 0; why = "equilibrium psi map not on device";
  }

  if (!dev) {
    if (!warned_fallback && comm->me == 0 && screen && why)
      fprintf(screen,"fix cross_field_diffusion/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    if (update_kk) update_kk->oe_cd_dev = 0;   // run() uploads host dx_cd
    ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK);
    FixCrossFieldDiffusion::start_of_step();
    return;
  }

  const int nlocal = particle->nlocal;
  if (nlocal == 0) {
    update->cd_flag = 0;
    return;
  }

  // device displacement buffer, headroom like the CPU (the mover guards
  // reads with i < cd_nmax; PINSERT indices past nlocal read the zeros)
  if (nlocal > update->cd_nmax)
    update->cd_nmax = nlocal + nlocal/10 + 1;
  if ((int) update_kk->d_dx_cd.extent(0) < update->cd_nmax)
    update_kk->d_dx_cd =
        DAT::t_float_2d_lr("update:dx_cd",update->cd_nmax,3);
  Kokkos::deep_copy(update_kk->d_dx_cd,0.0);
  update->cd_flag = 1;
  update_kk->oe_cd_dev = 1;   // run() skips its host->device upload

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species   = particle_kk->k_species.d_view;
  d_dx        = update_kk->d_dx_cd;

  d_vtx_r    = update_kk->d_oe_mesh_vtx_r;
  d_vtx_z    = update_kk->d_oe_mesh_vtx_z;
  d_tri      = update_kk->d_oe_mesh_tri;
  d_tri_br   = update_kk->d_oe_mesh_tri_br;
  d_tri_bz   = update_kk->d_oe_mesh_tri_bz;
  d_tri_bt   = update_kk->d_oe_mesh_tri_bt;
  d_tri_rmin = update_kk->d_oe_mesh_tri_rmin;
  d_tri_rmax = update_kk->d_oe_mesh_tri_rmax;
  d_tri_zmin = update_kk->d_oe_mesh_tri_zmin;
  d_tri_zmax = update_kk->d_oe_mesh_tri_zmax;
  if (update_kk->oe_has_mesh_plasma)
    d_tri_te = update_kk->d_oe_mesh_tri_te;
  d_hash_off = update_kk->d_oe_hash_offset;
  d_hash_ent = update_kk->d_oe_hash_entries;
  hash_rmin_ = update_kk->oe_mesh_hash_rmin;
  hash_zmin_ = update_kk->oe_mesh_hash_zmin;
  hash_dr_   = update_kk->oe_mesh_hash_dr;
  hash_dz_   = update_kk->oe_mesh_hash_dz;
  hash_nr_   = update_kk->oe_mesh_hash_nr;
  hash_nz_   = update_kk->oe_mesh_hash_nz;
  ntri_      = update_kk->oe_mesh_ntri;

  has_equ_       = update_kk->oe_has_equilibrium;
  has_equ_bmaps_ = update_kk->oe_has_equ_bmaps;
  if (has_equ_) {
    d_equ_r   = update_kk->d_oe_equ_r;
    d_equ_z   = update_kk->d_oe_equ_z;
    d_equ_psi = update_kk->d_oe_equ_psi;
    equ_btf_  = update_kk->oe_equ_btf;
    equ_rtf_  = update_kk->oe_equ_rtf;
    equ_jm_   = update_kk->oe_equ_jm;
    equ_km_   = update_kk->oe_equ_km;
    if (has_equ_bmaps_) {
      d_equ_br = update_kk->d_oe_equ_br;
      d_equ_bt = update_kk->d_oe_equ_bt;
      d_equ_bz = update_kk->d_oe_equ_bz;
    }
  }

  dt_eff_ = update->dt * nevery;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixCrossFieldDiffusion>(0,nlocal),
      *this);
  DeviceType().fence();
  copymode = 0;
}

/* ----------------------------------------------------------------------
   per-particle displacement (device twin of the CPU loop, 3D background)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixCrossFieldDiffusionKokkos::operator()(TagFixCrossFieldDiffusion,
                                              const int &i) const
{
  Particle::OnePart &p = d_particles(i);
  if (d_species(p.ispecies).charge == 0.0) return;   // skip neutrals

  // column-axis shift for the (R,Z) queries (3D linear-device decks)
  double xq[3] = {p.x[0] - col_x0_, p.x[1] - col_y0_, p.x[2]};

  // B at the particle: mesh -> equilibrium chain (CPU pd_bfield_sparta)
  double B[3] = {0.0, 0.0, 0.0};
  bool got_B = MeshKokkos::query_bfield_at_point(
      xq, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_tri_br, d_tri_bz, d_tri_bt,
      d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_, B);
  if (!got_B && has_equ_) {
    if (has_equ_bmaps_)
      EquilibriumKokkos::query_bfield_native_maps(
          xq, dim_, axisym_, d_equ_r, d_equ_z,
          d_equ_br, d_equ_bt, d_equ_bz, equ_jm_, equ_km_, B);
    else
      EquilibriumKokkos::query_bfield_at_point(
          xq, dim_, axisym_, d_equ_r, d_equ_z, d_equ_psi,
          equ_btf_, equ_rtf_, equ_jm_, equ_km_, B);
  }
  const double Bmag = Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag < 1.0e-20) return;   // CPU skips everything, pinches included

  // local D_perp
  double D_local = 0.0;
  if (diff_model_ == 1) {
    D_local = D_perp_;
  } else if (diff_model_ == 2) {
    double te = 0.0;
    const int tri = MeshKokkos::locate_tri_at_point(
        xq, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
        d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
        hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_);
    if (tri >= 0) te = d_tri_te(tri);
    if (te < 0.0) te = 0.0;
    D_local = bohm_scale_ * te / (16.0 * Bmag);
  }

  double ddx0 = 0.0, ddx1 = 0.0, ddx2 = 0.0;

  // stochastic perpendicular displacement (3D basis, two Gaussians)
  if (D_local > 0.0) {
    const double sigma = Kokkos::sqrt(2.0 * D_local * dt_eff_);
    const double inv_Bmag = 1.0 / Bmag;
    const double bhat0 = B[0] * inv_Bmag;
    const double bhat1 = B[1] * inv_Bmag;
    const double bhat2 = B[2] * inv_Bmag;

    double ax0 = 1.0, ax1 = 0.0, ax2 = 0.0;
    if (Kokkos::fabs(bhat0) > 0.9) { ax0 = 0.0; ax1 = 1.0; }
    const double dot = ax0*bhat0 + ax1*bhat1 + ax2*bhat2;
    double e1_0 = ax0 - dot*bhat0;
    double e1_1 = ax1 - dot*bhat1;
    double e1_2 = ax2 - dot*bhat2;
    const double e1mag =
        Kokkos::sqrt(e1_0*e1_0 + e1_1*e1_1 + e1_2*e1_2);
    e1_0 /= e1mag; e1_1 /= e1mag; e1_2 /= e1mag;

    const double e2_0 = bhat1*e1_2 - bhat2*e1_1;
    const double e2_1 = bhat2*e1_0 - bhat0*e1_2;
    const double e2_2 = bhat0*e1_1 - bhat1*e1_0;

    rand_type rand_gen = rand_pool.get_state();
    const double xi1 = rand_gen.normal();
    const double xi2 = rand_gen.normal();
    rand_pool.free_state(rand_gen);

    ddx0 += sigma * (xi1*e1_0 + xi2*e2_0);
    ddx1 += sigma * (xi1*e1_1 + xi2*e2_1);
    ddx2 += sigma * (xi1*e1_2 + xi2*e2_2);
  }

  // constant pinch (cylindrical R,Z about the column axis)
  const double rx = p.x[0] - col_x0_;
  const double ry = p.x[1] - col_y0_;
  const double R = Kokkos::sqrt(rx*rx + ry*ry);
  if (have_pinch_) {
    if (R > 1.0e-20) {
      const double cphi = rx / R, sphi = ry / R;
      ddx0 += v_pinch_R_ * cphi * dt_eff_;
      ddx1 += v_pinch_R_ * sphi * dt_eff_;
    }
    ddx2 += v_pinch_Z_ * dt_eff_;
  }

  // flux-surface-normal psi pinch: v * grad(psiN)/|grad(psiN)| (device
  // twin of pd->psi_norm_gradient_at — bilinear cell derivative with
  // the CPU's clamp-to-zero outside the equilibrium rectangle)
  if (have_psi_pinch_ && psi_ok_) {
    const double Rp = R, Zp = p.x[2];
    const bool clamp_R = (Rp < r_front_ || Rp > r_back_);
    const bool clamp_Z = (Zp < z_front_ || Zp > z_back_);
    const double Rc = Kokkos::fmin(Kokkos::fmax(Rp, r_front_), r_back_);
    const double Zc = Kokkos::fmin(Kokkos::fmax(Zp, z_front_), z_back_);
    const double fi = (Rc - r_front_) / dr_eq_;
    const double fj = (Zc - z_front_) / dz_eq_;
    int i0 = (int) fi; if (i0 < 0) i0 = 0; if (i0 > equ_jm_-2) i0 = equ_jm_-2;
    int j0 = (int) fj; if (j0 < 0) j0 = 0; if (j0 > equ_km_-2) j0 = equ_km_-2;
    double s = fi - i0; s = Kokkos::fmin(Kokkos::fmax(s, 0.0), 1.0);
    double t = fj - j0; t = Kokkos::fmin(Kokkos::fmax(t, 0.0), 1.0);
    const double p00 = d_equ_psi(j0,   i0);
    const double p10 = d_equ_psi(j0,   i0+1);
    const double p01 = d_equ_psi(j0+1, i0);
    const double p11 = d_equ_psi(j0+1, i0+1);
    double gR = 0.0, gZ = 0.0;
    if (!clamp_R)
      gR = ((1.0-t)*(p10-p00) + t*(p11-p01)) / (dr_eq_ * psi_denom_);
    if (!clamp_Z)
      gZ = ((1.0-s)*(p01-p00) + s*(p11-p10)) / (dz_eq_ * psi_denom_);
    const double gm = Kokkos::sqrt(gR*gR + gZ*gZ);
    if (gm > 1.0e-12) {
      const double vr = v_pinch_psi_ * gR / gm;
      const double vz = v_pinch_psi_ * gZ / gm;
      if (R > 1.0e-20) {
        const double cphi = rx / R, sphi = ry / R;
        ddx0 += vr * cphi * dt_eff_;
        ddx1 += vr * sphi * dt_eff_;
      }
      ddx2 += vz * dt_eff_;
    }
  }

  // gradient_pinch: mesh-only files carry no structured grad(ne) -> the
  // CPU term is exactly zero there (raster decks are host-gated at init)

  d_dx(i,0) = ddx0;
  d_dx(i,1) = ddx1;
  d_dx(i,2) = ddx2;
}
