/* ----------------------------------------------------------------------
   OpenEdge: fix force/thermal — Kokkos backend (gate 9a).
   See the header for scope. Kernel is a line-for-line device port of
   FixForceThermal::kick_half (background branch, Boris mode: the
   apply_parallel_impulse GC shortcut refuses for Boris particles and
   GCA is blocked under Kokkos, so the plain kick is the whole story).
------------------------------------------------------------------------- */

#include "fix_force_thermal_kokkos.h"

#include <cstdlib>
#include <cmath>
#include <vector>

#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix_background.h"
#include "grid.h"
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
  cmc_stamp_n = -1;
  cmc_stamp_id = (cellint) -1;
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
    d_gter_cell = update_kk->d_oe_meshcell_gter;
    d_gtez_cell = update_kk->d_oe_meshcell_gtez;
  }
  if (use_gradti_) {
    d_gtir_cell = update_kk->d_oe_meshcell_gtir;
    d_gtiz_cell = update_kk->d_oe_meshcell_gtiz;
  }

  // SPARTA-cell -> mesh-cell map (host pd_grad's cell_mesh_cell):
  // refresh the host cache and re-upload when the decomposition stamps
  // change. Per-rank stamp check is SAFE here — the upload contains no
  // collectives (unlike build_oe_sheath_cache; see gate-6 Bug F).
  {
    const int nglocal = grid->nlocal;
    const cellint fid = (nglocal > 0 && grid->cells)
      ? grid->cells[0].id : (cellint) -1;
    if (nglocal != cmc_stamp_n || fid != cmc_stamp_id) {
      if ((int) pd_->cell_mesh_cell.size() != nglocal)
        pd_->build_cell_mesh_index();
      d_cell_mesh_cell = DAT::t_int_1d(
          Kokkos::view_alloc("fth:cell_mesh_cell",
                             Kokkos::WithoutInitializing),
          nglocal > 0 ? nglocal : 1);
      auto h = Kokkos::create_mirror_view(d_cell_mesh_cell);
      for (int c = 0; c < nglocal; c++)
        h(c) = (c < (int) pd_->cell_mesh_cell.size())
                 ? pd_->cell_mesh_cell[c] : -1;
      Kokkos::deep_copy(d_cell_mesh_cell,h);
      cmc_stamp_n = nglocal;
      cmc_stamp_id = fid;
    }
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

  has_equ_ = update_kk->oe_has_equilibrium;
  if (has_equ_) {
    d_equ_r   = update_kk->d_oe_equ_r;
    d_equ_z   = update_kk->d_oe_equ_z;
    d_equ_psi = update_kk->d_oe_equ_psi;
    equ_btf_  = update_kk->oe_equ_btf;
    equ_rtf_  = update_kk->oe_equ_rtf;
    equ_jm_   = update_kk->oe_equ_jm;
    equ_km_   = update_kk->oe_equ_km;
  }

  dt_half_   = dt_half;
  echarge_   = update->echarge;
  alpha_e_k_ = alpha_e_;
  beta_i_k_  = beta_i_;

  const int nlocal = particle->nlocal;

  // OE_FTH_COMPARE: parity microscope — snapshot v, run the device
  // kernel, then recompute the host kick from the snapshot and print
  // the first mismatching particles with their inputs. Host-visible
  // OpenMP backend only; strictly a debugging aid.
  const bool compare = getenv("OE_FTH_COMPARE") != nullptr;
  std::vector<double> vpre;
  if (compare) {
    particle_kk->sync(Host,PARTICLE_MASK);
    vpre.resize(3*(size_t)nlocal);
    for (int i = 0; i < nlocal; i++)
      for (int c = 0; c < 3; c++)
        vpre[3*(size_t)i+c] = particle->particles[i].v[c];
  }

  copymode = 1;
  Kokkos::parallel_for(
      Kokkos::RangePolicy<DeviceType,TagFixForceThermal>(0,nlocal),*this);
  Kokkos::fence();
  copymode = 0;

  particle_kk->modify(Device,PARTICLE_MASK);

  if (compare) {
    particle_kk->sync(Host,PARTICLE_MASK);
    Particle::OnePart *parts = particle->particles;
    Particle::Species *specs = particle->species;
    static int nprinted = 0;
    for (int i = 0; i < nlocal && nprinted < 12; i++) {
      const int isp = parts[i].ispecies;
      const double Z = specs[isp].charge;
      if (Z == 0.0) continue;
      const double m_Z = specs[isp].mass;
      if (m_Z <= 0.0) continue;
      // host-side kick from the snapshot (mirror of the CPU kick_half)
      double B0,B1,B2;
      pd_bfield_sparta(parts[i], i, B0, B1, B2);
      const double Bmag = sqrt(B0*B0+B1*B1+B2*B2);
      double vh[3] = {vpre[3*(size_t)i], vpre[3*(size_t)i+1],
                      vpre[3*(size_t)i+2]};
      if (Bmag >= 1.0e-20) {
        const double ib = 1.0/Bmag;
        const double bh[3] = {B0*ib, B1*ib, B2*ib};
        const double phi_p = (domain->dimension == 3)
            ? atan2(parts[i].x[1], parts[i].x[0]) : 0.0;
        double bR, bZc, bphi;
        OpenEdge::sparta_v_to_RZphi(bh, domain->dimension,
                                    domain->axisymmetric, phi_p,
                                    bR, bZc, bphi);
        double a_par = 0.0;
        const double Z2 = Z*Z;
        if (have_ion_thermal_) {
          const double gr = pd_grad(pd_->mesh_grad_ti_r, pd_->grad_ti_r, parts[i]);
          const double gz = pd_grad(pd_->mesh_grad_ti_z, pd_->grad_ti_z, parts[i]);
          a_par += beta_i_ * Z2 * update->echarge * (gr*bR + gz*bZc) / m_Z;
        }
        if (have_elec_thermal_) {
          const double gr = pd_grad(pd_->mesh_grad_te_r, pd_->grad_te_r, parts[i]);
          const double gz = pd_grad(pd_->mesh_grad_te_z, pd_->grad_te_z, parts[i]);
          a_par += alpha_e_ * Z2 * update->echarge * (gr*bR + gz*bZc) / m_Z;
        }
        if (a_par != 0.0)
          for (int c = 0; c < 3; c++) vh[c] += a_par * bh[c] * dt_half;
      }
      double dmax = 0.0;
      for (int c = 0; c < 3; c++) {
        const double d = fabs(parts[i].v[c] - vh[c]);
        if (d > dmax) dmax = d;
      }
      const double vscale = fabs(vh[0])+fabs(vh[1])+fabs(vh[2])+1.0;
      if (dmax > 1e-9 * vscale) {
        const int mc_dbg = (parts[i].icell >= 0 &&
            parts[i].icell < (int) pd_->cell_mesh_cell.size())
              ? pd_->cell_mesh_cell[parts[i].icell] : -2;
        fprintf(screen,
          "[fthcmp] step " BIGINT_FORMAT " i=%d id=%d icell=%d mc=%d "
          "x=(%.6g,%.6g,%.6g) Bhost=(%.6g,%.6g,%.6g) dmax=%.3e "
          "vdev=(%.9g,%.9g,%.9g) vhost=(%.9g,%.9g,%.9g)\n",
          update->ntimestep, i, parts[i].id, parts[i].icell, mc_dbg,
          parts[i].x[0], parts[i].x[1], parts[i].x[2],
          B0, B1, B2, dmax,
          parts[i].v[0], parts[i].v[1], parts[i].v[2],
          vh[0], vh[1], vh[2]);
        nprinted++;
      }
    }
  }
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
  const bool gotB = MeshKokkos::query_bfield_at_point(
      p.x, dim_, axisym_, d_vtx_r, d_vtx_z, d_tri,
      d_tri_br, d_tri_bz, d_tri_bt,
      d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax,
      d_hash_off, d_hash_ent, hash_rmin_, hash_zmin_,
      hash_dr_, hash_dz_, hash_nr_, hash_nz_, ntri_, B);
  if (!gotB && has_equ_)
    EquilibriumKokkos::query_bfield_at_point(
        p.x, dim_, axisym_, d_equ_r, d_equ_z, d_equ_psi,
        equ_btf_, equ_rtf_, equ_jm_, equ_km_, B);
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

  // gradients: exact host pd_grad semantics — the SPARTA cell's
  // centroid-mapped mesh cell (constant per SPARTA cell; -1 = centroid
  // outside the mesh footprint = 0, the empty-structured fallback)
  const int mc = (p.icell >= 0 && p.icell < (int) d_cell_mesh_cell.extent(0))
                   ? d_cell_mesh_cell(p.icell) : -1;

  double a_par = 0.0;
  const double Z2 = Z * Z;

  if (use_gradti_ && mc >= 0 && mc < (int) d_gtir_cell.extent(0)) {
    const double grad_par_Ti =
        d_gtir_cell(mc) * bhat_R_cyl + d_gtiz_cell(mc) * bhat_Z_cyl;
    a_par += beta_i_k_ * Z2 * echarge_ * grad_par_Ti / m_Z;
  }
  if (use_gradte_ && mc >= 0 && mc < (int) d_gter_cell.extent(0)) {
    const double grad_par_Te =
        d_gter_cell(mc) * bhat_R_cyl + d_gtez_cell(mc) * bhat_Z_cyl;
    a_par += alpha_e_k_ * Z2 * echarge_ * grad_par_Te / m_Z;
  }

  if (a_par == 0.0) return;

  p.v[0] += a_par * bhat0 * dt_half_;
  p.v[1] += a_par * bhat1 * dt_half_;
  p.v[2] += a_par * bhat2 * dt_half_;
}
