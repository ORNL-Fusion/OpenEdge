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

#include "stdio.h"
#include "string.h"
#include "modify_kokkos.h"
#include "domain.h"
#include "update.h"
#include "compute.h"
#include "fix.h"
#include "style_compute.h"
#include "style_fix.h"
#include "memory_kokkos.h"
#include "error.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "kokkos.h"
#include "comm.h"
#include "timer.h"
#include <stdlib.h>
#include <mpi.h>

using namespace SPARTA_NS;

#define DELTA 4

// OpenEdge timing buckets (see modify.cpp): coulomb/* -> Coll,
// volume/chem/* -> Chem, rest -> Modify. Device fixes are fenced before
// the stamp so their asynchronous kernels are charged to their own bucket
// instead of the next synchronization point.
static inline void stamp_fix_bucket_kk(Timer *timer, Fix *f, bool detail)
{
  int which = TIME_MODIFY;
  if (strstr(f->style,"coulomb")) which = TIME_COLLIDE;
  else if (strstr(f->style,"volume/chem")) which = TIME_CHEM;
  // perf: the attribution fence is only worth its cost when per-fix timing
  // detail is requested (OE_FIX_TIMING); otherwise the async kernel is
  // charged to the next synchronization point
  if (detail && which != TIME_MODIFY && f->kokkos_flag) Kokkos::fence();
  timer->stamp(which);
}

// mask settings - same as in fix.cpp

#define START_OF_STEP  1
#define END_OF_STEP    2

/* ---------------------------------------------------------------------- */

ModifyKokkos::ModifyKokkos(SPARTA *sparta) : Modify(sparta)
{
  particle_kk = (ParticleKokkos*) particle;
  grid_kk = (GridKokkos*) grid;

  fix_timing_every = 0;
  fix_drain_start = fix_drain_end = 0.0;
  fix_wall_start = fix_wall_end = 0.0;
  if (const char *e = getenv("OE_FIX_TIMING")) {
    fix_timing_every = atoi(e);
    if (fix_timing_every <= 0) fix_timing_every = 100;
    if (comm->me == 0 && screen)
      fprintf(screen,"ModifyKokkos: per-fix timing enabled "
              "(OE_FIX_TIMING, report every %d steps)\n",fix_timing_every);
  }
}

/* ---------------------------------------------------------------------- */

ModifyKokkos::~ModifyKokkos()
{

}

/* ----------------------------------------------------------------------
   start-of-timestep call, only for relevant fixes
------------------------------------------------------------------------- */

void ModifyKokkos::start_of_step()
{
  const bool ft = fix_timing_every > 0;
  const double tw = ft ? MPI_Wtime() : 0.0;
  if (ft && (int) fix_time_start.size() < nfix) {
    fix_time_start.resize(nfix,0.0); fix_time_end.resize(nfix,0.0);
    fix_calls_start.resize(nfix,0);   fix_calls_end.resize(nfix,0);
  }
  for (int i = 0; i < n_start_of_step; i++) {
    int j = list_start_of_step[i];
    double t0 = 0.0;
    if (ft) { double tf = MPI_Wtime(); Kokkos::fence(); t0 = MPI_Wtime();
              fix_drain_start += t0 - tf; }
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[list_start_of_step[i]]->start_of_step();

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
    if (ft) { Kokkos::fence(); fix_time_start[j] += MPI_Wtime() - t0;
              fix_calls_start[j]++; }
    stamp_fix_bucket_kk(timer,fix[j],ft);
  }
  if (ft) fix_wall_start += MPI_Wtime() - tw;
}

/* ----------------------------------------------------------------------
   end-of-timestep call, only for relevant fixes
   only call fix->end_of_step() on timesteps that are multiples of nevery
------------------------------------------------------------------------- */

void ModifyKokkos::end_of_step()
{
  const bool ft = fix_timing_every > 0;
  const double tw = ft ? MPI_Wtime() : 0.0;
  if (ft && (int) fix_time_end.size() < nfix) {
    fix_time_start.resize(nfix,0.0); fix_time_end.resize(nfix,0.0);
    fix_calls_start.resize(nfix,0);   fix_calls_end.resize(nfix,0);
  }
  for (int i = 0; i < n_end_of_step; i++)
    if (update->ntimestep % end_of_step_every[i] == 0) {
      int j = list_end_of_step[i];
      double t0 = 0.0;
      if (ft) { double tf = MPI_Wtime(); Kokkos::fence(); t0 = MPI_Wtime();
                fix_drain_end += t0 - tf; }
      particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
      int prev_auto_sync = sparta->kokkos->auto_sync;
      if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

      fix[list_end_of_step[i]]->end_of_step();

      sparta->kokkos->auto_sync = prev_auto_sync;
      particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
      if (ft) { Kokkos::fence(); fix_time_end[j] += MPI_Wtime() - t0;
                fix_calls_end[j]++; }
      stamp_fix_bucket_kk(timer,fix[j],ft);
    }
  if (ft) fix_wall_end += MPI_Wtime() - tw;
  if (ft && update->ntimestep % fix_timing_every == 0) fix_timing_report();
}

/* ----------------------------------------------------------------------
   OE_FIX_TIMING report: cumulative per-fix seconds (rank 0 values plus
   the max over ranks), start_of_step and end_of_step separately
------------------------------------------------------------------------- */

void ModifyKokkos::fix_timing_report()
{
  const int n = nfix;
  std::vector<double> mx_start(n,0.0), mx_end(n,0.0);
  MPI_Reduce(fix_time_start.data(),mx_start.data(),n,MPI_DOUBLE,MPI_MAX,0,world);
  MPI_Reduce(fix_time_end.data(),mx_end.data(),n,MPI_DOUBLE,MPI_MAX,0,world);
  if (comm->me != 0) return;
  FILE *outs[2] = {screen,logfile};
  for (FILE *out : outs) {
    if (!out) continue;
    fprintf(out,"[fix-timing] step " BIGINT_FORMAT
            " cumulative seconds (rank0 / max-rank), calls\n",update->ntimestep);
    fprintf(out,"  %-8s %-32s start %9.3f             end %9.3f   "
            "(leading-fence wait: async kernels from earlier phases)\n",
            "-","async-drain",fix_drain_start,fix_drain_end);
    fprintf(out,"  %-8s %-32s start %9.3f             end %9.3f   "
            "(entry->exit wall of ModifyKokkos loops; TIME_MODIFY bucket so far %.3f)\n",
            "-","loop-wall",fix_wall_start,fix_wall_end,timer->array[TIME_MODIFY]);
    for (int j = 0; j < n; j++) {
      if (fix_calls_start[j] == 0 && fix_calls_end[j] == 0) continue;
      fprintf(out,"  %-8s %-32s start %9.3f / %9.3f (%ld)   end %9.3f / %9.3f (%ld)\n",
              fix[j]->id,fix[j]->style,
              fix_time_start[j],mx_start[j],fix_calls_start[j],
              fix_time_end[j],mx_end[j],fix_calls_end[j]);
    }
    fflush(out);
  }
}

/* ----------------------------------------------------------------------
   pack_grid_one call, only for relevant fixes
   invoked by load balancer when grid cells migrate
------------------------------------------------------------------------- */

int ModifyKokkos::pack_grid_one(int icell, char *buf, int memflag)
{
  char *ptr = buf;
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    ptr += fix[list_pergrid[i]]->pack_grid_one(icell,ptr,memflag);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
  return ptr-buf;
}

/* ----------------------------------------------------------------------
   unpack_grid_one call, only for relevant fixes
   invoked by load balancer when grid cells migrate
------------------------------------------------------------------------- */

int ModifyKokkos::unpack_grid_one(int icell, char *buf)
{
  char *ptr = buf;
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    ptr += fix[list_pergrid[i]]->unpack_grid_one(icell,ptr);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
  return ptr-buf;
}

/* ----------------------------------------------------------------------
   copy_grid call, only for relevant fixes
   invoked when a grod cell is removed
------------------------------------------------------------------------- */

void ModifyKokkos::copy_grid_one(int icell, int jcell)
{
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[j]->copy_grid_one(icell,jcell);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   add_grid_one call, only for relevant fixes
   invoked by adapt_grid and fix adapt when new child cells are created
------------------------------------------------------------------------- */

void ModifyKokkos::add_grid_one()
{
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[j]->add_grid_one();

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   reset_grid call, only for relevant fixes
   invoked after all grid cell removals
------------------------------------------------------------------------- */

void ModifyKokkos::reset_grid_count(int nlocal)
{
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[j]->reset_grid_count(nlocal);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   grid_changed call, only for relevant fixes
   invoked after per-processor list of grid cells has changed
------------------------------------------------------------------------- */

void ModifyKokkos::grid_changed()
{
  for (int i = 0; i < n_pergrid; i++) {
    int j = list_pergrid[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[j]->grid_changed();

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   custom_surf_changed call, only for relevant fixes
   invoked after per-surf custom values have changed
------------------------------------------------------------------------- */

void ModifyKokkos::custom_surf_changed()
{
  for (int i = 0; i < n_custom_surf_changed; i++) {
    int j = list_custom_surf_changed[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[j]->custom_surf_changed();

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   invoke update_custom() method, only for relevant fixes
------------------------------------------------------------------------- */

void ModifyKokkos::update_custom(int index, double temp_thermal,
                                 double temp_rot, double temp_vib, double *vstream)
{
  for (int i = 0; i < n_update_custom; i++) {
    int j = list_update_custom[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[list_update_custom[i]]->update_custom(index,temp_thermal,temp_rot,
                                            temp_vib,vstream);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   invoke gas_react() method, only for relevant fixes
------------------------------------------------------------------------- */

void ModifyKokkos::gas_react(int index)
{
  for (int i = 0; i < n_gas_react; i++) {
    int j = list_gas_react[i];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[list_gas_react[i]]->gas_react(index);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}

/* ----------------------------------------------------------------------
   invoke surf_react() method, only for relevant fixes
------------------------------------------------------------------------- */

void ModifyKokkos::surf_react(Particle::OnePart *iorig, int &i, int &)
{
  for (int m = 0; m < n_surf_react; m++) {
    int j = list_surf_react[m];
    particle_kk->sync(fix[j]->execution_space,fix[j]->datamask_read);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    if (!fix[j]->kokkos_flag) sparta->kokkos->auto_sync = 1;

    fix[list_surf_react[m]]->surf_react(iorig,i,j);

    sparta->kokkos->auto_sync = prev_auto_sync;
    particle_kk->modify(fix[j]->execution_space,fix[j]->datamask_modify);
  }
}
