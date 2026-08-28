/* ----------------------------------------------------------------------
   OpenEdge: fix surface/emit/source — Kokkos backend (gate 10).

   Device port of the task-based surface emission for the production
   configuration: 3D, FLOW mode with the nlaunch_total weighted budget
   and a STATIC per-task source (frozen upstream flux, file, or const
   mode — the slag monoblock case), Thompson or fixed-energy emission,
   monatomic products. The kernel runs one thread per task, draws the
   stratified ninsert, samples positions on the clipped surf-cell
   polygon (CSR-flattened path), and creates particles in-kernel with
   the A1b machinery (pre-grown storage + atomic append). Newborn
   customs: zero_all + pweight (w_emit / fnum default) — same
   approximation as the PWI device newborns.

   HOST FALLBACK (loud, per step where needed): 2D, perspecies, region,
   thermal / thermal_tsurf models, non-FLOW np mode, polyatomic mixture
   species, active surf tallies, the not-yet-frozen first step(s), and
   OE_EMIT_HOST.

   Contributors:
     - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(surface/emit/source/kk,FixSurfaceEmitSourceKokkos)

#else

#ifndef SPARTA_FIX_SURFACE_EMIT_SOURCE_KOKKOS_H
#define SPARTA_FIX_SURFACE_EMIT_SOURCE_KOKKOS_H

#include "fix_surface_emit_source.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

struct TagFixSurfEmitSource {};

class FixSurfaceEmitSourceKokkos : public FixSurfaceEmitSource,
                                   public KokkosBase {
 public:
#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  FixSurfaceEmitSourceKokkos(class SPARTA *, int, char **);
  ~FixSurfaceEmitSourceKokkos();
  void init() override;
  void grid_changed() override;
  void perform_task() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixSurfEmitSource, const int &i) const;

 private:
  int device_ok;
  int warned_fallback;
  int tasks_uploaded_;            // device task views current

  void upload_tasks();

  // ---- static task views (uploaded per task build) ----
  DAT::t_int_1d d_t_pcell, d_t_isurf, d_t_npoint;
  DAT::t_float_2d_lr d_t_tan1, d_t_tan2, d_t_vstream;
  DAT::t_int_1d d_t_poff;         // CSR offsets into path/frac
  DAT::t_float_1d d_t_path;       // 3 doubles per point
  DAT::t_float_1d d_t_frac;       // npoint-2 per task
  DAT::t_float_1d d_t_src;        // per-task source strength (static)
  DAT::t_float_1d d_cumm;         // mixture cumulative fractions
  DAT::t_int_1d d_mix_sp;         // mixture species ids

  // ---- per-call bindings ----
  t_particle_1d d_particles;
  t_species_1d d_species;
  t_tri_1d d_tris;
  ParticleKokkos::DeviceCustom custom_;
  int pw_slot_;
  Kokkos::View<int, DeviceType> d_new_count;
  Kokkos::View<int, DeviceType> d_nsingle;

  // ---- kernel scalars ----
  int ntask_, nspecies_mix_;
  int model_;                     // EmitModel copy (THOMPSON/FIXED)
  int weighted_;                  // pweight = w_emit vs fnum default
  double dt_eff_, fnum_, src_total_;
  double ub_, emax_, cosn_, efixed_;
  int nlaunch_total_;
};

}  // namespace SPARTA_NS

#endif
#endif
