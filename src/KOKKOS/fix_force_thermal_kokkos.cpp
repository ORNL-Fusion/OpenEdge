/* ----------------------------------------------------------------------
   OpenEdge: fix force/thermal — Kokkos backend (gate 9a).
   See the header for scope. Kernel is a line-for-line device port of
   FixForceThermal::kick_half (background branch, Boris mode: the
   apply_parallel_impulse GC shortcut refuses for Boris particles and
   GCA is blocked under Kokkos, so the plain kick is the whole story).
------------------------------------------------------------------------- */

#include "fix_force_thermal_kokkos.h"

#include <cstdlib>

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "openedge_geom.h"
#include "particle_kokkos.h"
#include "pusher_kokkos.h"
#include "sparta.h"
#include "sparta_masks.h"
#include "update.h"
#include "update_kokkos.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixForceThermalKokkos::FixForceThermalKokkos(SPARTA *sparta, int narg,
                                             char **arg) :
  FixForceThermal(sparta, narg, arg)
{
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK;
  datamask_modify = PARTICLE_MASK;

  device_ok = 0;
  warned_fallback = 0;
}

/* ---------------------------------------------------------------------- */

FixForceThermalKokkos::~FixForceThermalKokkos() {}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::init()
{
  FixForceThermal::init();

  dim_ = domain->dimension;
  axisym_ = domain->axisymmetric;

  device_ok = 1;
  const char *why = nullptr;
  if (getenv("OE_FTH_HOST")) {
    device_ok = 0; why = "OE_FTH_HOST env override";
  } else if (!use_background_) {
    device_ok = 0; why = "compute-source fields (device is background-mesh only)";
  } else if (!dynamic_cast<UpdateKokkos *>(update)) {
    device_ok = 0; why = "no UpdateKokkos (host run)";
  }

  if (!device_ok) {
    if (comm->me == 0 && screen && !warned_fallback)
      fprintf(screen,"fix force/thermal/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    return;
  }
}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  kick_device(0.5 * update->dt * nevery);
}

void FixForceThermalKokkos::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  kick_device(0.5 * update->dt * nevery);
}

/* ---------------------------------------------------------------------- */

void FixForceThermalKokkos::kick_device(double dt_half)
{
  UpdateKokkos *update_kk = dynamic_cast<UpdateKokkos *>(update);

  int dev = device_ok;
  const char *why = nullptr;
  if (dev && !update_kk->oe_has_mesh_b) {
    dev = 0; why = "device mesh B views not built";
  }

  if (!dev) {
    if (!warned_fallback && comm->me == 0 && screen && why)
      fprintf(screen,"fix force/thermal/kk: HOST fallback (%s)\n",why);
    warned_fallback = 1;
    ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK);
    FixForceThermal::kick_half(dt_half);
    particle_kk->modify(Host,PARTICLE_MASK);
    particle_kk->sync(Device,PARTICLE_MASK);
    return;
  }

  if (particle->nlocal == 0) return;

  // the CPU contributes 0 for a gradient family whose mesh arrays are
  // absent (empty structured fallback); mirror with per-family flags
  use_gradte_ = have_elec_thermal_ && update_kk->oe_has_mesh_gradte;
  use_gradti_ = have_ion_thermal_  && update_kk->oe_has_mesh_gradti;
  if (!use_gradte_ && !use_gradti_) return;   // nothing to kick

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  d_particles = particle_kk->k_particles.view_device();
  d_species   = particle_kk->k_species.d_view;

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
  if (use_gradte_) {
    d_tri_gter = update_kk->d_oe_mesh_tri_gter;
    d_tri_gtez = update_kk->d_oe_mesh_tri_gtez;
  }
  if (use_gradti_) {
    d_tri_gtir = update_kk->d_oe_mesh_tri_gtir;
    d_tri_gtiz = update_kk->d_oe_mesh_tri_gtiz;
  }
  d_hash_off = update_kk->d_oe_hash_offset;
  d_hash_ent = update_kk->d_oe_hash_entries;
  hash_rmin_ = update_kk->oe_mesh_hash_rmin;
  hash_zmin_ = update_kk->oe_mesh_hash_zmin;
  hash_dr_   = update_kk->oe_mesh_hash_dr;
  hash_dz_   = update_kk->oe_mesh_hash_dz;
  hash_nr_   = update_kk->oe_mesh_hash_nr;
  hash_nz_   = update_kk->oe_mesh_hash_nz;
  ntri_      = update_kk->oe_mesh_ntri;

  dt_half_   = dt_half;
  echarge_   = update->echarge;
  alpha_e_k_ = alpha_e_;
  beta_i_k_  = beta_i_;

  const int nlocal = particle->nlocal;

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixForceThermal>(0,nlocal),*this);
  Kokkos::fence();
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);
}

/* ----------------------------------------------------------------------
   per-particle Braginskii parallel half-kick (device twin of the CPU
   FixForceThermal::kick_half body, background branch, Boris mode)
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixForceThermalKokkos::operator()(TagFixForceThermal,
                                       const int &i) const
{
  Particle::OnePart &p = d_particles(i);
  const int isp = p.ispecies;
  const double Z = d_species(isp).charge;
  if (Z == 0.0) return;

  const double m_Z = d_species(isp).mass;
  if (m_Z <= 0.0) return;

  double B[3] = {0.0,0.0,0.0};
  MeshKokkos::query_bfield_at_point(
      p.x, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_tri_br, d_tri_bz, d_tri_bt,
      d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_, B);
  const double Bmag =
      Kokkos::sqrt(B[0]*B[0] + B[1]*B[1] + B[2]*B[2]);
  if (Bmag < 1.0e-20) return;

  const double inv_Bmag = 1.0 / Bmag;
  const double bhat0 = B[0] * inv_Bmag;
  const double bhat1 = B[1] * inv_Bmag;
  const double bhat2 = B[2] * inv_Bmag;

  // bhat in the cylindrical frame for the gradient dot product
  const double bhat_sparta[3] = {bhat0, bhat1, bhat2};
  const double phi_p = (dim_ == 3)
      ? Kokkos::atan2(p.x[1], p.x[0]) : 0.0;
  double bhat_R_cyl, bhat_Z_cyl, bhat_phi_unused;
  OpenEdge::sparta_v_to_RZphi(bhat_sparta, dim_, axisym_ != 0,
                              phi_p, bhat_R_cyl, bhat_Z_cyl,
                              bhat_phi_unused);

  // gradients at the particle position (tri-constant; a miss = 0,
  // matching the CPU's empty-structured-grid fallback)
  const int tri = MeshKokkos::locate_tri_at_point(
      p.x, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_);

  double a_par = 0.0;
  const double Z2 = Z * Z;

  if (use_gradti_ && tri >= 0) {
    const double grad_par_Ti =
        d_tri_gtir(tri) * bhat_R_cyl + d_tri_gtiz(tri) * bhat_Z_cyl;
    a_par += beta_i_k_ * Z2 * echarge_ * grad_par_Ti / m_Z;
  }
  if (use_gradte_ && tri >= 0) {
    const double grad_par_Te =
        d_tri_gter(tri) * bhat_R_cyl + d_tri_gtez(tri) * bhat_Z_cyl;
    a_par += alpha_e_k_ * Z2 * echarge_ * grad_par_Te / m_Z;
  }

  if (a_par == 0.0) return;

  p.v[0] += a_par * bhat0 * dt_half_;
  p.v[1] += a_par * bhat1 * dt_half_;
  p.v[2] += a_par * bhat2 * dt_half_;
}
