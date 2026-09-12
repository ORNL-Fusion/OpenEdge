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

#include "string.h"
#include "stdlib.h"
#include "fix_balance_kokkos.h"
#include "balance_grid.h"
#include "update.h"
#include "grid_kokkos.h"
#include "particle_kokkos.h"
#include "surf_kokkos.h"
#include "comm.h"
#include "comm_kokkos.h"
#include "rcb.h"
#include "modify.h"
#include "compute.h"
#include "output.h"
#include "dump.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"
#include "kokkos.h"
#include "update.h"
#include <cstdio>
#include <cstdlib>

using namespace SPARTA_NS;

enum{RANDOM,PROC,BISECTION};
enum{CELL,PARTICLE};

#define ZEROPARTICLE 0.1

/* ---------------------------------------------------------------------- */

FixBalanceKokkos::FixBalanceKokkos(SPARTA *sparta, int narg, char **arg) :
  FixBalance(sparta, narg, arg)
{
  kokkos_flag = 0; // need auto sync
  execution_space = Host;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ----------------------------------------------------------------------
   perform dynamic load balancing
------------------------------------------------------------------------- */

void FixBalanceKokkos::end_of_step()
{
  GridKokkos* grid_kk = (GridKokkos*) grid;
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  SurfKokkos* surf_kk = (SurfKokkos*) surf;
  CommKokkos* comm_kk = (CommKokkos*) comm;

  // OpenEdge (2026-09-12): with the device cell migration the particles
  // never visit the host here. RCB by particle count needs the per-cell
  // counts: take them from the device sort and leave the host particle
  // lists empty (cinfo.first = -1) so FixBalance skips its host sort and
  // Grid::compress's repoint loop does nothing; CommKokkos::migrate_cells
  // remaps the cell indices on the device.
  const int devmig = comm_kk->cell_migration_device();
  static int diag = -1;
  if (diag < 0) { const char *e = getenv("OE_CELLMIG_DIAG"); diag = e ? atoi(e) : 0; }
  const double t0 = MPI_Wtime();

  grid_kk->sync(Host,CELL_MASK|CINFO_MASK|SINFO_MASK|PCELL_MASK);
  surf_kk->sync(Host,ALL_MASK);
  if (!devmig) particle_kk->sync(Host,PARTICLE_MASK);
  else if (bstyle == BISECTION && rcbwt == PARTICLE) {
    // device-only particle work: no auto_sync (its blanket modify(Host)
    // would push the stale host mirror over the device particles)
    const int as = sparta->kokkos->auto_sync;
    sparta->kokkos->auto_sync = 0;
    particle_kk->sync(Device,PARTICLE_MASK);
    if (!particle_kk->sorted_kk) particle_kk->sort_kokkos();
    sparta->kokkos->auto_sync = as;
    auto h_count = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       grid_kk->d_cellcount);
    Grid::ChildInfo *cinfo = grid->cinfo;
    const int nglocal = grid->nlocal;
    for (int i = 0; i < nglocal; i++) {
      cinfo[i].count = (i < (int) h_count.extent(0)) ? h_count(i) : 0;
      cinfo[i].first = -1;
    }
    particle->sorted = 1;
  }

  const double t1 = MPI_Wtime();
  // With the device migration nothing touches the particles on the host,
  // but ModifyKokkos runs this (non-Kokkos) fix with auto_sync = 1, under
  // which ParticleKokkos::grow()'s sync(Device) does a blanket modify(Host)
  // and pushes the stale host mirror over the live device particles when a
  // receiving rank has to grow. Run the balance with auto_sync off.
  // (auto_sync stays on for FixBalance::end_of_step: the host cell work,
  // GridKokkos::grow_cells included, relies on it; CommKokkos::migrate_cells
  // switches it off around its device particle phases only)
  FixBalance::end_of_step();
  const double t2 = MPI_Wtime();

  grid_kk->modify(Host,CELL_MASK|CINFO_MASK|SINFO_MASK|PCELL_MASK);
  if (!devmig) particle_kk->modify(Host,PARTICLE_MASK);
  surf_kk->modify(Host,ALL_MASK);
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  grid_kk->wrap_kokkos_graphs();
  const double t3 = MPI_Wtime();
  grid_kk->update_hash();
  const double t4 = MPI_Wtime();
  if (diag) {
    printf("OE_BALANCE_KK rank=%d step=%ld t(sync+counts %.3f, FixBalance::end_of_step %.3f, wrap %.3f, hash %.3f) s\n",
           comm->me,(long)update->ntimestep,t1-t0,t2-t1,t3-t2,t4-t3);
    fflush(stdout);
  }
}

