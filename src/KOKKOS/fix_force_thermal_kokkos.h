/* ----------------------------------------------------------------------
   OpenEdge: fix force/thermal — Kokkos backend (skeleton).

   Status: FILE REGISTERED, runtime falls back to host kick_half().
   The class hierarchy + style registration are in place so `-sf kk`
   decks don't crash; the actual GPU kernel for the per-particle
   Braginskii force is left as a follow-on (see TODO in .cpp).

   Plan for the kernel:
     - Read B-field and grad(Te), grad(Ti) from the per-particle pcache
       custom DOUBLE attributes (pc_bx, pc_by, pc_bz, pc_grad_te_r, ...).
       Those are populated each pcache_nevery steps by update.cpp from
       the registered FixBackground / compute plasma/fields.
     - Per-particle parallel_for over particle->nlocal.
     - For `use_background_`: read directly from pcache (device-side).
     - For legacy `!use_background_` (CollGridSrc compute/fix sources):
       keep the host fallback. Modern decks use background mode.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(force/thermal/kk,FixForceThermalKokkos)

#else

#ifndef SPARTA_FIX_FORCE_THERMAL_KOKKOS_H
#define SPARTA_FIX_FORCE_THERMAL_KOKKOS_H

#include "fix_force_thermal.h"

namespace SPARTA_NS {

class FixForceThermalKokkos : public FixForceThermal {
 public:
  FixForceThermalKokkos(class SPARTA *, int, char **);
  ~FixForceThermalKokkos();
  void init();
  void start_of_step();
  void end_of_step();
};

}  // namespace SPARTA_NS

#endif
#endif
