/* ----------------------------------------------------------------------
   OpenEdge: per-particle magnetic field fix — Kokkos version.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(bfield/particle/kk,FixBfieldParticleKokkos)

#else

#ifndef SPARTA_FIX_BFIELD_PARTICLE_KOKKOS_H
#define SPARTA_FIX_BFIELD_PARTICLE_KOKKOS_H

#include "fix_bfield_particle.h"
#include "kokkos_type.h"
#include "kokkos_base.h"

namespace SPARTA_NS {

class FixBfieldParticleKokkos : public FixBfieldParticle, public KokkosBase {
 public:
  FixBfieldParticleKokkos(class SPARTA *, int, char **);
  ~FixBfieldParticleKokkos();
  void init();
  void compute_field();

 private:
  DAT::tdual_float_2d_lr k_array_particle;
  int maxpart_kk;
};

}

#endif
#endif
