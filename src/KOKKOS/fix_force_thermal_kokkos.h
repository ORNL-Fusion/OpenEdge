/* ----------------------------------------------------------------------
   OpenEdge: fix force/thermal — Kokkos backend (gate 9a).

   Device port of the Braginskii thermal-force half-kicks. Each charged
   particle independently samples B and grad(Te)/grad(Ti) at its CURRENT
   position from the per-tri flattened FixBackground mesh views (exact
   device twin of the CPU's pd_bfield_sparta / pd_grad mesh branch) and
   receives the parallel half-kick — deterministic, velocity-only.
   Both hooks (start_of_step and end_of_step, the leapfrog halves) run
   the same kernel.

   SUPPORTED on device (checked at bind time): background mode with the
   device mesh views built (mesh B + whichever of gradTe/gradTi the
   plasma file carries; a missing gradient family contributes 0 exactly
   like the CPU's empty-structured-grid fallback). HOST FALLBACK
   otherwise, and with OE_FTH_HOST.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(force/thermal/kk,FixForceThermalKokkos)

#else

#ifndef SPARTA_FIX_FORCE_THERMAL_KOKKOS_H
#define SPARTA_FIX_FORCE_THERMAL_KOKKOS_H

#include "fix_force_thermal.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"

namespace SPARTA_NS {

struct TagFixForceThermal {};

class FixForceThermalKokkos : public FixForceThermal, public KokkosBase {
 public:
  FixForceThermalKokkos(class SPARTA *, int, char **);
  ~FixForceThermalKokkos();
  void init() override;
  void start_of_step() override;
  void end_of_step() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixForceThermal, const int &i) const;

 private:
  int device_ok;
  int warned_fallback;

  // bound per kick from UpdateKokkos (friend) + ParticleKokkos
  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_float_1d d_vtx_r, d_vtx_z;
  DAT::t_int_1d d_tri;
  DAT::t_float_1d d_tri_br, d_tri_bz, d_tri_bt;
  DAT::t_float_1d d_tri_rmin, d_tri_rmax, d_tri_zmin, d_tri_zmax;
  // gradients: PER MESH CELL, looked up via the SPARTA-cell -> mesh-cell
  // map (exact host pd_grad semantics: cell-centroid sampling, 0 when
  // the centroid falls outside the mesh footprint)
  DAT::t_float_1d d_gter_cell, d_gtez_cell, d_gtir_cell, d_gtiz_cell;
  DAT::t_int_1d d_cell_mesh_cell;
  int cmc_stamp_n;
  cellint cmc_stamp_id;
  DAT::t_int_1d d_hash_off, d_hash_ent;
  double hash_rmin_, hash_zmin_, hash_dr_, hash_dz_;
  int hash_nr_, hash_nz_, ntri_;
  int dim_, axisym_;
  int use_gradte_, use_gradti_;
  // equilibrium fallback for mesh-miss B (matches host bfield_at chain)
  int has_equ_;
  DAT::t_float_1d d_equ_r, d_equ_z;
  DAT::t_float_2d_lr d_equ_psi;
  double equ_btf_, equ_rtf_;
  int equ_jm_, equ_km_;

  // kick scalars
  double dt_half_, echarge_, alpha_e_k_, beta_i_k_;

  void kick_device(double dt_half);
};

}  // namespace SPARTA_NS

#endif
#endif
