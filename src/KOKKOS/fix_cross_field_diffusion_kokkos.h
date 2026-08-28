/* ----------------------------------------------------------------------
   OpenEdge: fix cross_field_diffusion — Kokkos backend.

   Device port of the per-particle anomalous cross-field random walk.
   Each charged particle samples B at its position from the device mesh
   views (mesh -> equilibrium chain), builds the perpendicular basis,
   draws the Gaussian step, and adds the deterministic pinches
   (constant R/Z and the SOLEDGE3X flux-surface-normal psi pinch, whose
   bilinear psi_norm gradient is evaluated on the device equ map — the
   same pd->psirz raster the CPU uses). Displacements land directly in
   UpdateKokkos::d_dx_cd, retiring the per-step host fill + H2D upload.

   SUPPORTED on device: 3D background mode with device mesh B (Bohm
   model additionally needs the mesh te view; psi pinch needs the equ
   map). HOST FALLBACK otherwise, and with OE_CD_HOST. gradient_pinch
   with a structured raster stays on the host (per-particle FD there);
   on mesh-only files the CPU gradient is exactly zero, so the device
   path skips the term with identical physics.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(cross_field_diffusion/kk,FixCrossFieldDiffusionKokkos)

#else

#ifndef SPARTA_FIX_CROSS_FIELD_DIFFUSION_KOKKOS_H
#define SPARTA_FIX_CROSS_FIELD_DIFFUSION_KOKKOS_H

#include "fix_cross_field_diffusion.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

struct TagFixCrossFieldDiffusion {};

class FixCrossFieldDiffusionKokkos : public FixCrossFieldDiffusion,
                                     public KokkosBase {
 public:
#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  FixCrossFieldDiffusionKokkos(class SPARTA *, int, char **);
  ~FixCrossFieldDiffusionKokkos();
  void init() override;
  void start_of_step() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixCrossFieldDiffusion, const int &i) const;

 private:
  int device_ok;
  int warned_fallback;

  // bound per step from UpdateKokkos (friend) + ParticleKokkos
  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_float_2d_lr d_dx;                 // UpdateKokkos::d_dx_cd
  DAT::t_float_1d d_vtx_r, d_vtx_z;
  DAT::t_int_1d d_tri;
  DAT::t_float_1d d_tri_br, d_tri_bz, d_tri_bt;
  DAT::t_float_1d d_tri_te;
  DAT::t_float_1d d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax;
  DAT::t_int_1d d_hash_off, d_hash_ent;
  double hash_rmin_, hash_zmin_, hash_dr_, hash_dz_;
  int hash_nr_, hash_nz_, ntri_;
  int has_equ_, has_equ_bmaps_;
  DAT::t_float_1d d_equ_r, d_equ_z;
  DAT::t_float_2d_lr d_equ_psi;
  DAT::t_float_2d_lr d_equ_br, d_equ_bt, d_equ_bz;
  double equ_btf_, equ_rtf_;
  int equ_jm_, equ_km_;

  // scalars for the kernel
  int dim_, axisym_;
  double col_x0_, col_y0_;
  double dt_eff_;                          // dt * nevery
  // psi-pinch gradient (device twin of pd->psi_norm_gradient_at)
  int psi_ok_;
  double psi_denom_, dr_eq_, dz_eq_;
  double r_front_, r_back_, z_front_, z_back_;
};

}  // namespace SPARTA_NS

#endif
#endif
