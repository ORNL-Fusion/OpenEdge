/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/background — Kokkos backend (gate 9a).

   Device port of the Nanbu test-particle-vs-Maxwellian-background drag.
   Each charged particle independently samples the background plasma
   (te/ne/ti/ni/upar) at its CURRENT position from the per-tri flattened
   FixBackground mesh views (exact device twin of interp2D's mesh
   branch), draws a virtual Maxwellian partner, and scatters — identical
   sampling semantics to the CPU path. RNG uses a rank-offset Kokkos
   pool, so parity with the CPU is STATISTICAL (validated the gate-6
   way). No sort needed (flat per-particle kernel); only the test
   particle's velocity is modified.

   SUPPORTED on device (device_ok, checked in init):
     - background mode with plasma from `background <fixID>`
       (use_background_), Boris push mode (hybrid/GCA is blocked under
       Kokkos anyway), mesh-resident plasma with ni + upar fields.
   HOST FALLBACK (base end_of_step) otherwise, and with OE_COUL_HOST.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coulomb/background/kk,FixCoulombBackgroundKokkos)

#else

#ifndef SPARTA_FIX_COULOMB_BACKGROUND_KOKKOS_H
#define SPARTA_FIX_COULOMB_BACKGROUND_KOKKOS_H

#include "fix_coulomb_background.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

struct TagFixCoulombBg {};

class FixCoulombBackgroundKokkos : public FixCoulombBackground,
                                   public KokkosBase {
 public:
#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  FixCoulombBackgroundKokkos(class SPARTA *, int, char **);
  ~FixCoulombBackgroundKokkos();
  void init() override;
  void end_of_step() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixCoulombBg, const int &i) const;

 private:
  int device_ok;
  int warned_fallback;

  // Nanbu A(s) table (uploaded once in init)
  DAT::t_float_1d d_s_tab, d_A_tab;
  int ntab_;

  // bound per step from UpdateKokkos (friend) + ParticleKokkos
  t_particle_1d d_particles;
  t_species_1d d_species;
  t_cinfo_1d d_cinfo;
  DAT::t_float_1d d_vtx_r, d_vtx_z;
  DAT::t_int_1d d_tri;
  DAT::t_float_1d d_tri_te, d_tri_ti, d_tri_ne, d_tri_ni, d_tri_upar;
  DAT::t_float_1d d_tri_br, d_tri_bz, d_tri_bt;
  DAT::t_float_1d d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax;
  DAT::t_int_1d d_hash_off, d_hash_ent;
  double hash_rmin_, hash_zmin_, hash_dr_, hash_dz_;
  int hash_nr_, hash_nz_, ntri_;
  int dim_, axisym_;
  // equilibrium fallback for mesh-miss B (matches host bfield_at chain)
  int has_equ_;
  DAT::t_float_1d d_equ_r, d_equ_z;
  DAT::t_float_2d_lr d_equ_psi;
  double equ_btf_, equ_rtf_;
  int equ_jm_, equ_km_;

  // scalars for the kernel
  double dtc_, echarge_, eps0_, mbg_, qbg_;

  void run_device_kernel();
};

}  // namespace SPARTA_NS

#endif
#endif
