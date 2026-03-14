/* ----------------------------------------------------------------------
   OpenEdge: per-particle electric field fix — Kokkos version.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(efield/particle/kk,FixEfieldParticleKokkos)

#else

#ifndef SPARTA_FIX_EFIELD_PARTICLE_KOKKOS_H
#define SPARTA_FIX_EFIELD_PARTICLE_KOKKOS_H

#include "fix_efield_particle.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixEfieldParticleKokkos : public FixEfieldParticle, public KokkosBase {
 public:
  FixEfieldParticleKokkos(class SPARTA *, int, char **);
  ~FixEfieldParticleKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_particle;
  int maxpart_kk;
};

}

#endif
#endif
