/* ----------------------------------------------------------------------
   OpenEdge: generic per-particle field fix — Kokkos version.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(field/particle/kk,FixFieldParticleKokkos)

#else

#ifndef SPARTA_FIX_FIELD_PARTICLE_KOKKOS_H
#define SPARTA_FIX_FIELD_PARTICLE_KOKKOS_H

#include "fix_field_particle.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixFieldParticleKokkos : public FixFieldParticle, public KokkosBase {
 public:
  FixFieldParticleKokkos(class SPARTA *, int, char **);
  ~FixFieldParticleKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_particle;
  int maxpart_kk;
};

}

#endif
#endif
