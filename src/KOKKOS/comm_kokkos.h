/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifndef SPARTA_COMM_KOKKOS_H
#define SPARTA_COMM_KOKKOS_H

#include "comm.h"
#include "grid.h"
#include "particle_kokkos.h"
#include "kokkos_type.h"
#include "kokkos_copy.h"

namespace SPARTA_NS {

template<int NEED_ATOMICS, int HAVE_CUSTOM>
struct TagCommMigrateParticles{};

template<int HAVE_CUSTOM>
struct TagCommMigrateUnpackParticles{};

class CommKokkos : public Comm {
 public:

  CommKokkos(class SPARTA *);
  ~CommKokkos();
  // entryexit_in / any_entryexit_out (OpenEdge 2026-09-14): the mover's per-pass
  // entry/exit flag rides in the plan's MPI_Alltoall; the reduction result comes
  // back through the pointer. With a null pointer the plain migration runs.
  int migrate_particles(int, int*, const DAT::t_int_1d &, int entryexit_in = 0, int *any_entryexit_out = nullptr);
  void migrate_cells(int);
  int cell_migration_device() const;   // 1 = migrate_cells moves particles on the device
  // OE_COMM_TIMING=<nsteps>: per-section wall time of migrate_particles, printed
  // every nsteps (rank 0 / max over ranks). Sections: 0 pack (sync + kernel +
  // pproc D2H), 1 compress, 2 plan (create/augment_data_uniform), 3 grow + sync,
  // 4 exchange_uniform, 5 unpack kernel, 6 total
  int oe_comm_timing_every, oe_ct_calls;
  double oe_ct[7];
  bigint oe_ct_last, oe_nsend_sum, oe_nrecv_sum;
  void oe_comm_timing_report();

  template<int NEED_ATOMICS, int HAVE_CUSTOM>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCommMigrateParticles<NEED_ATOMICS,HAVE_CUSTOM>, const int&) const;

  template<int HAVE_CUSTOM>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagCommMigrateUnpackParticles<HAVE_CUSTOM>, const int&) const;

 private:
  int nlocal;


  typedef Kokkos::
    DualView<Grid::ChildCell*, Kokkos::LayoutRight, DeviceType> tdual_cell_1d;
  typedef tdual_cell_1d::t_dev t_cell_1d;
  t_cell_1d d_cells;

  typedef Kokkos::
    DualView<Particle::OnePart*, Kokkos::LayoutRight, DeviceType> tdual_particle_1d;
  typedef tdual_particle_1d::t_dev t_particle_1d;
  t_particle_1d d_particles;

  DAT::t_int_1d d_plist;
  // [0] = packed-particle counter, [1+k] = destination proc of packed particle k;
  // one D2H copy of nsend+1 ints replaces the pproc copy + the counter sync
  DAT::tdual_int_1d k_pmeta;
  DAT::t_int_1d d_pmeta;
  HAT::t_int_1d h_pmeta;
  DAT::t_int_1d d_cellmig_plist;       // particles in migrating cells (device cell migration)
  class Irregular *ibalance;           // own comm plan for the rebalance migration (never the per-step neighbor plan)
  void migrate_cells_only(int, int *);
  DAT::t_char_1d d_sbuf;
  DAT::t_char_1d d_rbuf;

  int nbytes_particle,nbytes_total;

  KKCopy<ParticleKokkos> particle_kk_copy;
};

}

#endif

/* ERROR/WARNING messages:

*/
