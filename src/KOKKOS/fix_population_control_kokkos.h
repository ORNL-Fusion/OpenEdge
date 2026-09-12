/* ----------------------------------------------------------------------
    OpenEdge: fix population/control — Kokkos backend.
    Device twin of the roulette down-sampler: particles are bucketed per
    (cell, species) with the Kokkos cell sort; every bucket over Nmax keeps
    Nmax uniformly random survivors (selection sampling, one thread per
    cell), their pweights are rescaled by bucket weight / kept weight, and
    the rest are deleted with the device compaction. Total weight per
    bucket is conserved exactly, as on the CPU; which markers survive is
    random on both sides (statistical parity only).
    HOST FALLBACK (base end_of_step): more than MAXSP species.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(population/control/kk,FixPopulationControlKokkos)

#else

#ifndef SPARTA_FIX_POPULATION_CONTROL_KOKKOS_H
#define SPARTA_FIX_POPULATION_CONTROL_KOKKOS_H

#include "fix_population_control.h"
#include "kokkos_base.h"
#include "kokkos_type.h"
#include "particle_kokkos.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"

namespace SPARTA_NS {

struct TagFixPopControl_cells {};
struct TagFixPopControl_scan {};

class FixPopulationControlKokkos : public FixPopulationControl, public KokkosBase {
 public:
  enum { MAXSP = 64 };   // per-thread species counter (registers/local memory)

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  FixPopulationControlKokkos(class SPARTA *, int, char **);
  ~FixPopulationControlKokkos();
  void end_of_step() override;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPopControl_cells, const int &icell, int &ndel) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixPopControl_scan, const int &i, int &offset, const bool final) const;

 private:
  int nmax_kk;
  t_particle_1d d_particles;
  DAT::t_float_1d d_pw;
  DAT::t_int_2d d_plist;
  DAT::t_int_1d d_cellcount;
  DAT::t_int_1d d_del;        // 1 = delete particle i
  DAT::t_int_1d d_dellist;    // compacted indices
  int warned_fallback;
};

}

#endif
#endif
