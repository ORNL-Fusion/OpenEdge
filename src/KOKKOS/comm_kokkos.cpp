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

#include "mpi.h"
#include "stdlib.h"
#include "string.h"
#include "comm_kokkos.h"
#include "collide_vss_kokkos.h"
#include "irregular_kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "update.h"
//#include "adapt_grid.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"
#include "sparta_masks.h"
#include "surf.h"
#include "domain.h"
#include "irregular.h"

#define OE_A2A_MAXPROCS 256   // Alltoall migration plan up to this many ranks (see migrate_particles)
#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace SPARTA_NS;

enum{PKEEP,PINSERT,PDONE,PDISCARD,PENTRY,PEXIT,PSURF};   // several files

/* ---------------------------------------------------------------------- */

CommKokkos::CommKokkos(SPARTA *sparta) : Comm(sparta),
  particle_kk_copy(sparta)
{
  delete iparticle;
  iparticle = new IrregularKokkos(sparta);
  ibalance = NULL;

  k_pmeta = DAT::tdual_int_1d("comm:pmeta",1);
  d_pmeta = k_pmeta.view_device();
  h_pmeta = k_pmeta.view_host();

  oe_comm_timing_every = 0; oe_ct_calls = 0; oe_ct_last = -1;
  oe_nsend_sum = oe_nrecv_sum = 0;
  for (int i = 0; i < 7; i++) oe_ct[i] = 0.0;
  if (const char *e = getenv("OE_COMM_TIMING")) oe_comm_timing_every = atoi(e);
}

/* OE_COMM_TIMING report: cumulative seconds per section of migrate_particles
   (rank 0 / max over ranks) plus the exchange_uniform breakdown; collective */

void CommKokkos::oe_comm_timing_report()
{
  IrregularKokkos* ik = (IrregularKokkos*) iparticle;
  double loc[13], mx[13];
  for (int i = 0; i < 7; i++) loc[i] = oe_ct[i];
  for (int i = 0; i < 6; i++) loc[7+i] = ik->oe_xt[i];
  MPI_Allreduce(loc,mx,13,MPI_DOUBLE,MPI_MAX,world);
  bigint sums[2] = {oe_nsend_sum,oe_nrecv_sum}, gsum[2];
  MPI_Allreduce(sums,gsum,2,MPI_SPARTA_BIGINT,MPI_SUM,world);
  if (me != 0) return;
  const char *nm[7] = {"pack","compress","plan","grow+sync","exchange","unpack","total"};
  const char *xn[6] = {"irecv","packbuf","send","self","waitall","h2d"};
  FILE *outs[2] = {screen,logfile};
  for (int o = 0; o < 2; o++) {
    FILE *out = outs[o]; if (!out) continue;
    fprintf(out,"[comm-timing] step " BIGINT_FORMAT " cumulative seconds (rank0 / max-rank), "
            "%d calls, particles sent " BIGINT_FORMAT " recv " BIGINT_FORMAT " (all ranks)\n",
            update->ntimestep,oe_ct_calls,gsum[0],gsum[1]);
    for (int i = 0; i < 7; i++)
      fprintf(out,"  %-10s %9.3f / %9.3f\n",nm[i],loc[i],mx[i]);
    fprintf(out,"  exchange_uniform:");
    for (int i = 0; i < 6; i++) fprintf(out,"  %s %.3f/%.3f",xn[i],loc[7+i],mx[7+i]);
    fprintf(out,"\n");
  }
}

/* ---------------------------------------------------------------------- */

CommKokkos::~CommKokkos()
{
  if (copymode) return;

  particle_kk_copy.uncopy();
  delete ibalance;

  if (!sparta->kokkos->comm_serial) {
    pproc = NULL;
  }
}

/* ----------------------------------------------------------------------
   migrate particles to new procs after particle move
   return particle nlocal after compression,
     so Update can iterate on particle move
------------------------------------------------------------------------- */

int CommKokkos::migrate_particles(int nmigrate, int *plist, const DAT::t_int_1d &d_plist_in,
                                  int entryexit_in, int *any_entryexit_out)
{
  // Plan choice (uniform across ranks): the one-Alltoall plan whenever the rank
  // count is small enough for a 2*nprocs-int Alltoall to be trivial, even with
  // the neighbor plan enabled (gridcut >= 0 makes neighflag = 1 on every
  // production deck). Above OE_A2A_MAXPROCS the neighbor plan (augment) keeps
  // its point-to-point counts and the separate flag reduction. The Alltoall
  // plan excludes empty pairs, so an all-empty pass can return before the
  // exchange; the neighbor plan exchanges zero-length messages with every
  // plan neighbor and must always run exchange_uniform.
  const int use_a2a = (!neighflag || nprocs <= OE_A2A_MAXPROCS) && !sparta->kokkos->comm_serial;
  if (any_entryexit_out && !use_a2a) {
    MPI_Allreduce(&entryexit_in,any_entryexit_out,1,MPI_INT,MPI_MAX,world);
    if (!*any_entryexit_out) return particle->nlocal;
    any_entryexit_out = nullptr;
  }
  GridKokkos* grid_kk = (GridKokkos*) grid;
  ParticleKokkos* particle_kk = ((ParticleKokkos*)particle);
  particle_kk->update_class_variables();
  particle_kk_copy.copy(particle_kk);

  const int gpu_aware_flag = sparta->kokkos->gpu_aware_flag;
  const int need_atomics = sparta->kokkos->need_atomics;

  if (sparta->kokkos->comm_serial) {
    particle_kk->sync(Host,ALL_MASK);
    //grid_kk->sync(Host,ALL_MASK);
    int prev_auto_sync = sparta->kokkos->auto_sync;
    sparta->kokkos->auto_sync = 1;

    int ncompress = Comm::migrate_particles(nmigrate,plist);

    particle_kk->sync(Device,ALL_MASK);
    //grid_kk->sync(Device,ALL_MASK);
    sparta->kokkos->auto_sync = prev_auto_sync;

    return ncompress;
  }

  // int i,j;

  d_plist = d_plist_in;
  double oe_t0 = 0.0, oe_t = 0.0;
  if (oe_comm_timing_every) oe_t0 = oe_t = MPI_Wtime();

  int ncustom = particle->ncustom;
  nbytes_particle = sizeof(Particle::OnePart);
  int nbytes_custom = particle->sizeof_custom();
  nbytes_total = nbytes_particle + nbytes_custom;

  // Kokkos
  // memory access: cells, particles
  // local views: pproc, sbuf, rbuf
  // functions: particle->compress_migrate, grow,
  //  iparticle->augment_data_uniform, iparticle->create_data_uniform,
  //  iparticle->exchange_uniform
  // parallel_for: loop over nmigrate to pack buffer,
  //  loop over nrecv to unpack buffer
  // atomic variables: ?

  // grow pproc and sbuf if necessary

  if (nmigrate > maxpproc) {
    maxpproc = nmigrate;
    k_pmeta = DAT::tdual_int_1d(Kokkos::view_alloc("comm:pmeta",Kokkos::WithoutInitializing),maxpproc+1);
    d_pmeta = k_pmeta.view_device();
    h_pmeta = k_pmeta.view_host();
  }
  pproc = h_pmeta.data()+1;
  //if (maxsendbuf == 0 || nmigrate*nbytes_total > maxsendbuf) { // this doesn't work, not sure why

    bigint maxsendbuf = (bigint)nmigrate*nbytes_total;
    // OpenEdge perf: grow-only (upstream re-allocated every call without
    // GPU-aware MPI: 1 cudaMalloc+cudaFree per migrate = per step)
    if (maxsendbuf > bigint(d_sbuf.extent(0)))
      d_sbuf = DAT::t_char_1d(Kokkos::view_alloc("comm:sbuf",Kokkos::WithoutInitializing),maxsendbuf);
  //}

  // fill proclist with procs to send to
  // pack sbuf with particles to migrate
  // if flag == PDISCARD, particle is deleted but not sent
  // change icell of migrated particle to owning cell on receiving proc
  // nsend = particles that actually migrate
  // if no custom attributes, pack particles directly via memcpy()
  // else pack_custom() performs packing into sbuf

  int nsend = 0;
  //int offset = 0;

  // OpenEdge: pack_custom_kokkos reads the custom device views — sync
  // them too, not just the particle structs
  if (ncustom)
    particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
  else
    particle_kk->sync(Device,PARTICLE_MASK);
  grid_kk->sync(Device,CELL_MASK);

  d_cells = grid_kk->k_cells.view_device();
  d_particles = particle_kk->k_particles.view_device();

  // OpenEdge perf (2026-09-14): counter + destination list in one view, zeroed on
  // the device and read back with one D2H copy of nmigrate+1 ints (was: counter
  // H2D, kernel, fence, full pproc D2H, counter D2H). Nothing to pack: no launches.
  if (nmigrate) Kokkos::deep_copy(Kokkos::subview(d_pmeta,0),0);

  copymode = 1;
  if (!nmigrate) {
    // no migrating particle on this rank: skip the pack kernel
  } else if (!ncustom) {

    if (need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<1,0> >(0,nmigrate),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<0,0> >(0,nmigrate),*this);

  } else {

    if (need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<1,1> >(0,nmigrate),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateParticles<0,1> >(0,nmigrate),*this);

  }
  copymode = 0;

  if (nmigrate) {
    particle_kk->modify(Device,PARTICLE_MASK);
    const auto n1 = std::make_pair(0,nmigrate+1);
    Kokkos::deep_copy(Kokkos::subview(h_pmeta,n1),Kokkos::subview(d_pmeta,n1));   // fences
    nsend = h_pmeta(0);
  }
  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[0] += t - oe_t; oe_t = t; }

  // compress my list of particles

  if (nmigrate) particle_kk->compress_migrate_kokkos(nmigrate,d_plist);   // device: no host mlist needed
  int ncompress = particle->nlocal;
  if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[1] += t - oe_t; oe_t = t; }

  // create or augment irregular communication plan
  // nrecv = # of incoming particles

  IrregularKokkos* iparticle_kk = (IrregularKokkos*) iparticle;

  int nrecv;
  if (!use_a2a)
    nrecv = iparticle_kk->augment_data_uniform(nsend,pproc);
  else {
    // one MPI_Alltoall: counts + the mover's entry/exit flag (replaces the
    // Reduce_scatter, count Send/Recv, Barrier and the separate flag Allreduce)
    int any_flag = 0;
    nrecv = iparticle_kk->create_data_uniform_flag(nsend,pproc,entryexit_in,any_flag);
    if (any_entryexit_out) *any_entryexit_out = any_flag;
  }
  if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[2] += t - oe_t; oe_t = t; }

  if (use_a2a && nsend == 0 && nrecv == 0) {   // Alltoall plan, empty on every side: nothing to exchange
    d_plist = {};
    if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[6] += t - oe_t0; oe_ct_calls++;
      if (update->ntimestep % oe_comm_timing_every == 0 && update->ntimestep != oe_ct_last) { oe_ct_last = update->ntimestep; oe_comm_timing_report(); } }
    return ncompress;
  }

  // extend particle list if necessary

  particle->grow(nrecv);

  // OpenEdge BUGFIX (2026-08-26, gate-6 sheath parity): grow() above can
  // REALLOCATE the particle and custom views, but particle_kk_copy (whose
  // captured views unpack_custom_kokkos writes through) was copied at the
  // top of this function. The unpack kernel then wrote the received
  // particles' custom attributes into the orphaned old allocations and
  // the live arrays kept grow's zero-fill — migrating particles lost all
  // persistent custom state (the spatial-sheath phiprev/bank ledgers,
  // plasma cache, pweight) whenever a grow coincided with a migration.
  // Refresh the functor copy after grow so it captures the live views.
  if (ncustom) {
    particle_kk->update_class_variables();
    particle_kk_copy.copy(particle_kk);
  }

  // perform irregular communication
  // if no custom attributes, append recv particles directly to particle list
  // else receive into rbuf, unpack particles one by one via unpack_custom()

  if (ncustom)
    particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
  else
    particle_kk->sync(Device,PARTICLE_MASK);
  d_particles = particle_kk->k_particles.view_device();
  if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[3] += t - oe_t; oe_t = t; }

  if (gpu_aware_flag && !ncustom) {
    iparticle_kk->
      exchange_uniform(d_sbuf,nbytes_total,
                       (char *) (d_particles.data()+particle->nlocal),d_rbuf);
    if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[4] += t - oe_t; oe_t = t; }
  } else {

    // allocate exact buffer size to reduce GPU <--> CPU memory transfer

    bigint maxrecvbuf = (bigint)nrecv*nbytes_total;
    if (maxrecvbuf > bigint(d_rbuf.extent(0)))   // OpenEdge perf: grow-only
      d_rbuf = DAT::t_char_1d(Kokkos::view_alloc("comm:rbuf",Kokkos::WithoutInitializing),maxrecvbuf);

    nlocal = particle->nlocal;
    iparticle_kk->exchange_uniform(d_sbuf,nbytes_total,(char *)d_rbuf.data(),d_rbuf);
    if (oe_comm_timing_every) { double t = MPI_Wtime(); oe_ct[4] += t - oe_t; oe_t = t; }

    copymode = 1;
    if (!ncustom) {

      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateUnpackParticles<0> >(0,nrecv),*this);
      DeviceType().fence();
      copymode = 0;

    } else {

      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCommMigrateUnpackParticles<1> >(0,nrecv),*this);
      DeviceType().fence();
      copymode = 0;
    }

  }

  // OpenEdge: the unpack kernel also wrote the received particles'
  // custom attributes on the device — mark them modified too
  if (ncustom)
    particle_kk->modify(Device,PARTICLE_MASK|CUSTOM_MASK);
  else
    particle_kk->modify(Device,PARTICLE_MASK);
  d_particles = t_particle_1d(); // destroy reference to reduce memory use
  d_plist = {};

  particle->nlocal += nrecv;
  ncomm += nsend;
  if (oe_comm_timing_every) {
    double t = MPI_Wtime(); oe_ct[5] += t - oe_t; oe_ct[6] += t - oe_t0;
    oe_ct_calls++; oe_nsend_sum += nsend; oe_nrecv_sum += nrecv;
    if (update->ntimestep % oe_comm_timing_every == 0 && update->ntimestep != oe_ct_last) {
      oe_ct_last = update->ntimestep;
      oe_comm_timing_report();
    }
  }
  return ncompress;
}

template<int NEED_ATOMICS, int HAVE_CUSTOM>
KOKKOS_INLINE_FUNCTION
void CommKokkos::operator()(TagCommMigrateParticles<NEED_ATOMICS, HAVE_CUSTOM>, const int &i) const {
  const int j = d_plist[i];
  if (d_particles[j].flag == PDISCARD) return;
  int nsend;
  if (NEED_ATOMICS)
    nsend = Kokkos::atomic_fetch_add(&d_pmeta(0),1);
  else {
    nsend = d_pmeta(0);
    d_pmeta(0)++;
  }
  d_pmeta(1+nsend) = d_cells[d_particles[j].icell].proc;
  d_particles[j].icell = d_cells[d_particles[j].icell].ilocal;
  const bigint offset = (bigint)nsend*nbytes_total;
  memcpy(&d_sbuf[offset],&d_particles[j],nbytes_particle);
  if (HAVE_CUSTOM)
    particle_kk_copy.obj.pack_custom_kokkos(j,(char*)(d_sbuf.data()+offset+nbytes_particle));
}

template<int HAVE_CUSTOM>
KOKKOS_INLINE_FUNCTION
void CommKokkos::operator()(TagCommMigrateUnpackParticles<HAVE_CUSTOM>, const int &irecv) const {
  const int i = nlocal + irecv;
  const bigint offset = (bigint)irecv*nbytes_total;
  memcpy(&d_particles[i],&d_rbuf[offset],nbytes_particle);
  if (HAVE_CUSTOM)
    particle_kk_copy.obj.unpack_custom_kokkos((char*)(d_rbuf.data()+offset+nbytes_particle),i);
}

/* ----------------------------------------------------------------------
   migrate grid cells with their particles to new procs
   called from BalanceGrid and FixBalance
------------------------------------------------------------------------- */

int CommKokkos::cell_migration_device() const
{
  // host path during setup (balance_grid before the Kokkos views are
  // wrapped), with a memory limit, or with serial comm
  return !(update->have_mem_limit() || sparta->kokkos->comm_serial ||
           sparta->kokkos->prewrap);
}

/* ----------------------------------------------------------------------
   migrate grid cells with their particles to new procs
   called from BalanceGrid and FixBalance

   OpenEdge (2026-09-12): the particles of migrating cells travel through
   the device pack/unpack of migrate_particles (custom attributes included);
   only the cell metadata goes through the host. The previous host path
   copied every particle and custom vector to the host, packed/unpacked
   them one by one and pushed everything back (~7 us per marker, 11-25 s
   per rebalance at 3.5 M markers on 4 A100 in the RFPIE case).

   Receiver-side index of a migrated cell: Grid::unpack_one appends each
   received top-level cell at grid->nlocal, immediately followed by its
   nsplit sub cells (csubs order); datums from one source arrive in the
   sender's ascending icell order (Irregular preserves it); sources are
   unpacked in the plan's receive order. So the sender encodes, per
   destination, a running slot count and the receiver adds the first slot
   of that source. Kept cells are renumbered exactly like Grid::compress
   (ascending, all kinds together).
------------------------------------------------------------------------- */

void CommKokkos::migrate_cells(int nmigrate)
{
  CollideVSSKokkos* collide_kk = (CollideVSSKokkos*) collide;
  if (collide)
    collide_kk->sync(Host,ALL_MASK);

  if (!cell_migration_device()) {
    Comm::migrate_cells(nmigrate);
    if (collide)
      collide_kk->modified(Host,ALL_MASK);
    return;
  }

  GridKokkos *grid_kk = (GridKokkos *) grid;
  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int nglocal = grid->nlocal;

  static int diag = -1;
  if (diag < 0) { const char *e = getenv("OE_CELLMIG_DIAG"); diag = e ? atoi(e) : 0; }
  static int announced = 0;
  if (!announced && me == 0 && screen) {
    fprintf(screen,"CommKokkos::migrate_cells: device particle migration active\n");
    announced = 1;
  }
  double t0 = MPI_Wtime(), t1 = t0, t2 = t0, t3 = t0, t4 = t0;

  // 1. host: destination proc and encoded index of every local cell

  std::vector<int> enc(nglocal > 0 ? nglocal : 1, 0);
  std::vector<int> dest(nglocal > 0 ? nglocal : 1, me);
  std::vector<long long> kcount(nprocs, 0);
  int nkept = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    int top = icell;
    if (cells[icell].nsplit <= 0) top = sinfo[cells[icell].isplit].icell;
    const int p = cells[top].proc;
    dest[icell] = p;
    if (p == me) enc[icell] = nkept++;
    else if (cells[icell].nsplit >= 1) {
      enc[icell] = (int) kcount[p];
      kcount[p] += 1 + (cells[icell].nsplit > 1 ? cells[icell].nsplit : 0);
    } else enc[icell] = enc[top] + 1 + (-cells[icell].nsplit);
  }
  for (int icell = 0; icell < nglocal; icell++)
    if (dest[icell] != me) {
      cells[icell].proc = dest[icell];      // sub cells: same proc as their split cell
      cells[icell].ilocal = enc[icell];     // read by the migrate_particles pack kernel
    }
  grid_kk->modify(Host,CELL_MASK);
  grid_kk->sync(Device,CELL_MASK);
  // device particle phases run without auto_sync: under auto_sync every
  // sync(Device) starts with a blanket modify(Host) that would push the
  // stale host particle mirror over the live device particles (the host
  // cell work in step 4 needs auto_sync on: GridKokkos::grow_cells keeps
  // the host copy authoritative through it)
  const int auto_sync_save = sparta->kokkos->auto_sync;
  sparta->kokkos->auto_sync = 0;
  particle_kk->sync(Device,PARTICLE_MASK);

  // 2. device: particles that sit in migrating cells

  const int nlocal_old = particle->nlocal;
  auto d_cells_v = grid_kk->k_cells.view_device();
  auto d_part_v = particle_kk->k_particles.view_device();
  if (nlocal_old > (int) d_cellmig_plist.extent(0))
    d_cellmig_plist = DAT::t_int_1d(Kokkos::view_alloc("comm:cellmig_plist",
                        Kokkos::WithoutInitializing), nlocal_old);
  auto d_pl = d_cellmig_plist;
  const int me_ = me;
  int npmig = 0;
  Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType>(0,nlocal_old),
    KOKKOS_LAMBDA(const int i, int &upd, const bool final) {
      const int ic = d_part_v(i).icell;
      const bool mig = (ic >= 0 && ic < nglocal && d_cells_v(ic).proc != me_);
      if (final && mig) d_pl(upd) = i;
      if (mig) upd++;
    }, npmig);
  HAT::t_int_1d h_pl(Kokkos::view_alloc("comm:cellmig_plist_h",
                     Kokkos::WithoutInitializing), npmig > 0 ? npmig : 1);
  if (npmig > 0)
    Kokkos::deep_copy(h_pl, Kokkos::subview(d_pl, std::make_pair(0,npmig)));
  if (diag >= 2) {
    // zero records (x == 0, id == 0) already present on the sender: in the
    // migrating set and in the whole local array
    int nz_mig = 0, nz_all = 0;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,npmig), KOKKOS_LAMBDA(const int i, int &z) {
        const Particle::OnePart &q = d_part_v(d_pl(i));
        if (q.id == 0 && q.x[0] == 0.0 && q.x[1] == 0.0 && q.x[2] == 0.0) z++; }, nz_mig);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nlocal_old), KOKKOS_LAMBDA(const int i, int &z) {
        const Particle::OnePart &q = d_part_v(i);
        if (q.id == 0 && q.x[0] == 0.0 && q.x[1] == 0.0 && q.x[2] == 0.0) z++; }, nz_all);
    printf("OE_CELLMIG_PRE rank=%d step=%ld zero records before migration: %d in the migrating set (%d), %d in all %d local particles\n",
           me,(long)update->ntimestep,nz_mig,npmig,nz_all,nlocal_old);
    fflush(stdout);
  }

  t1 = MPI_Wtime();

  // 3. device particle migration to arbitrary procs, through a communicator
  //    of its own: the per-step iparticle keeps its neighbor plan and the
  //    device-side state augment_data_uniform relies on

  if (!ibalance) ibalance = new IrregularKokkos(sparta);
  Irregular *iparticle_save = iparticle;
  iparticle = ibalance;
  const int neigh_save = neighflag;
  neighflag = 0;
  const int ncompress = migrate_particles(npmig, h_pl.data(), d_pl);
  neighflag = neigh_save;
  iparticle = iparticle_save;
  const int nprecv = particle->nlocal - ncompress;
  sparta->kokkos->auto_sync = auto_sync_save;
  t2 = MPI_Wtime();

  // 4. host: the cells themselves, without particles.
  //    Grid::compress repoints particles through cinfo.first/next lists;
  //    empty them so it does nothing (the device remap below does it).

  for (int icell = 0; icell < nglocal; icell++) cinfo[icell].first = -1;
  std::vector<int> cellbase(nprocs, -1);
  migrate_cells_only(nmigrate, cellbase.data());
  t3 = MPI_Wtime();

  // 5. device: cell index remap
  //    kept particles [0,ncompress): old index -> compressed index
  //    received [ncompress,nlocal): encoded slot + first slot of the source

  DAT::t_int_1d d_enc(Kokkos::view_alloc("comm:cellmig_enc",Kokkos::WithoutInitializing),
                      nglocal > 0 ? nglocal : 1);
  {
    auto h_enc = Kokkos::create_mirror_view(d_enc);
    for (int i = 0; i < nglocal; i++) h_enc(i) = enc[i];
    Kokkos::deep_copy(d_enc, h_enc);
  }
  const int nrp = ibalance->recv_nprocs();
  const int *rp = ibalance->recv_procs();
  const int *rn = ibalance->recv_nums();
  if (ibalance->self_count() != 0)
    error->one(FLERR,"CommKokkos::migrate_cells: unexpected self-send in the particle plan");
  std::vector<int> pstart(nrp + 1, 0), pbase(nrp > 0 ? nrp : 1, 0);
  for (int r = 0; r < nrp; r++) {
    pbase[r] = cellbase[rp[r]];
    if (pbase[r] < 0)
      error->one(FLERR,"CommKokkos::migrate_cells: particles received from a proc that sent no cells");
    pstart[r+1] = pstart[r] + rn[r];
  }
  if (pstart[nrp] != nprecv)
    error->one(FLERR,"CommKokkos::migrate_cells: particle receive plan does not match the received count");
  DAT::t_int_1d d_pstart(Kokkos::view_alloc("comm:cellmig_pstart",Kokkos::WithoutInitializing), nrp + 1);
  DAT::t_int_1d d_pbase(Kokkos::view_alloc("comm:cellmig_pbase",Kokkos::WithoutInitializing), nrp > 0 ? nrp : 1);
  {
    auto h1 = Kokkos::create_mirror_view(d_pstart); for (int r = 0; r <= nrp; r++) h1(r) = pstart[r];
    auto h2 = Kokkos::create_mirror_view(d_pbase);  for (int r = 0; r < nrp; r++) h2(r) = pbase[r];
    Kokkos::deep_copy(d_pstart, h1); Kokkos::deep_copy(d_pbase, h2);
  }
  sparta->kokkos->auto_sync = 0;
  particle_kk->sync(Device,PARTICLE_MASK);
  auto d_part2 = particle_kk->k_particles.view_device();
  const int nl = particle->nlocal;
  const int nglocal_new = grid->nlocal;
  int nbad = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType>(0,nl),
    KOKKOS_LAMBDA(const int i, int &bad) {
      int ic;
      if (i < ncompress) ic = d_enc(d_part2(i).icell);
      else {
        const int k = i - ncompress;
        int r = 0;
        while (r < nrp - 1 && k >= d_pstart(r+1)) r++;
        ic = d_pbase(r) + d_part2(i).icell;
      }
      d_part2(i).icell = ic;
      if (ic < 0 || ic >= nglocal_new) bad++;
    }, nbad);
  if (nbad)
    error->one(FLERR,"CommKokkos::migrate_cells: particle cell index out of range after rebalance");
  particle_kk->modify(Device,PARTICLE_MASK);
  sparta->kokkos->auto_sync = auto_sync_save;
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;
  t4 = MPI_Wtime();

  // OE_CELLMIG_DIAG >= 2: verify on the host that every particle lies inside
  // the bounds of the cell it is now assigned to (sub cells share their split
  // cell's box); reports the first offenders per category
  if (diag >= 2) {
    sparta->kokkos->auto_sync = 0;
    particle_kk->sync(Host,PARTICLE_MASK);
    Particle::OnePart *hp = particle->particles;
    Grid::ChildCell *hc = grid->cells;
    Grid::SplitInfo *hs = grid->sinfo;
    const int ng = grid->nlocal;
    int nbad_kept = 0, nbad_recv = 0, nbad_sub = 0, nshown = 0;
    for (int i = 0; i < nl; i++) {
      int ic = hp[i].icell;
      if (ic < 0 || ic >= ng) { nbad_kept++; continue; }
      int box = ic;
      const bool sub = hc[ic].nsplit <= 0;
      if (sub) box = hs[hc[ic].isplit].icell;
      const double *lo = hc[box].lo, *hi = hc[box].hi;
      const double *x = hp[i].x;
      const bool inside = x[0] >= lo[0] && x[0] <= hi[0] && x[1] >= lo[1] && x[1] <= hi[1] &&
                          (domain->dimension == 2 || (x[2] >= lo[2] && x[2] <= hi[2]));
      if (inside) continue;
      if (i < ncompress) nbad_kept++; else nbad_recv++;
      if (sub) nbad_sub++;
      if (nshown < 3) {
        nshown++;
        // which local top-level cell actually contains x?
        int actual = -1;
        for (int c = 0; c < ng && actual < 0; c++) {
          if (hc[c].nsplit <= 0) continue;
          const double *l = hc[c].lo, *h = hc[c].hi;
          if (x[0] >= l[0] && x[0] <= h[0] && x[1] >= l[1] && x[1] <= h[1] && x[2] >= l[2] && x[2] <= h[2]) actual = c;
        }
        printf("OE_CELLMIG_BAD rank=%d step=%ld particle %d (%s) icell=%d nsplit=%d isplit=%d parent=%d | actual containing top-level cell=%d (nsplit %d) | x=(%.5f,%.5f,%.5f) assigned box lo=(%.5f,%.5f,%.5f) hi=(%.5f,%.5f,%.5f)\n",
               me,(long)update->ntimestep,i,(i < ncompress) ? "kept" : "received",ic,hc[ic].nsplit,hc[ic].isplit,box,
               actual,(actual >= 0) ? hc[actual].nsplit : 0,x[0],x[1],x[2],lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]);
      }
    }
    printf("OE_CELLMIG_CHECK rank=%d step=%ld particles outside their cell: kept %d, received %d (in sub cells %d) of %d | ncompress %d, source ranges:",
           me,(long)update->ntimestep,nbad_kept,nbad_recv,nbad_sub,nl,ncompress);
    for (int r = 0; r < nrp; r++) printf(" p%d:[%d,%d)",rp[r],ncompress+pstart[r],ncompress+pstart[r+1]);
    printf(" | bad ranges:");
    {
      int start = -1, prev = -2, nr = 0;
      for (int i = 0; i < nl && nr < 12; i++) {
        const int ic = hp[i].icell; bool bad = (ic < 0 || ic >= ng);
        if (!bad) { int box = ic; if (hc[ic].nsplit <= 0) box = hs[hc[ic].isplit].icell;
          const double *lo = hc[box].lo, *hi = hc[box].hi, *x = hp[i].x;
          bad = !(x[0] >= lo[0] && x[0] <= hi[0] && x[1] >= lo[1] && x[1] <= hi[1] && (domain->dimension == 2 || (x[2] >= lo[2] && x[2] <= hi[2]))); }
        if (bad) { if (start < 0) start = i; else if (i != prev + 1) { printf(" [%d,%d]",start,prev); nr++; start = i; } prev = i; }
      }
      if (start >= 0 && nr < 12) printf(" [%d,%d]",start,prev);
    }
    printf("\n");
    fflush(stdout);
    particle_kk->modify(Device,PARTICLE_MASK);   // host copy untouched; keep device authoritative
    sparta->kokkos->auto_sync = auto_sync_save;
  }
  if (diag) {
    printf("OE_CELLMIG rank=%d step=%ld cells: nglocal %d -> %d, sent %d, kept %d | particles: %d -> compress %d + recv %d (mig %d) | sources %d | t(host prep %.3f, dev migrate %.3f, cells %.3f, remap %.3f) s\n",
           me,(long)update->ntimestep,nglocal,grid->nlocal,nmigrate,nkept,nlocal_old,ncompress,nprecv,npmig,nrp,t1-t0,t2-t1,t3-t2,t4-t3);
    fflush(stdout);
  }

  if (collide)
    collide_kk->modified(Host,ALL_MASK);
}

/* ----------------------------------------------------------------------
   Comm::migrate_cells without the particles (partflag = 0), and without
   the host particle compress. cellbase[p] = first local index of the
   cells received from proc p (-1 if none), in unpack order.
------------------------------------------------------------------------- */

void CommKokkos::migrate_cells_only(int nmigrate, int *cellbase)
{
  Grid::ChildCell *cells = grid->cells;
  int nglocal = grid->nlocal;
  static int diag = -1;
  if (diag < 0) { const char *e = getenv("OE_CELLMIG_DIAG"); diag = e ? atoi(e) : 0; }
  double tt[8]; int nt = 0; tt[nt++] = MPI_Wtime();

  if (nmigrate > maxgproc) {
    maxgproc = nmigrate;
    memory->destroy(gproc);
    memory->destroy(gsize);
    memory->create(gproc,maxgproc,"comm:gproc");
    memory->create(gsize,maxgproc,"comm:gsize");
  }

  int nsend = 0;
  bigint offset = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].proc == me) continue;
    gproc[nsend] = cells[icell].proc;
    int n = grid->pack_one(icell,NULL,1,0,1,0);
    gsize[nsend++] = n;
    offset += n;
  }

  if (offset > maxsendbuf) {
    memory->sfree(sbuf);
    maxsendbuf = offset;
    sbuf = (char *) memory->smalloc(maxsendbuf,"comm:sbuf");
    memset(sbuf,0,maxsendbuf);
  }

  offset = 0;
  for (int icell = 0; icell < nglocal; icell++) {
    if (cells[icell].nsplit <= 0) continue;
    if (cells[icell].proc == me) continue;
    offset += grid->pack_one(icell,&sbuf[offset],1,0,1,1);
  }

  tt[nt++] = MPI_Wtime();   // pack
  if (nmigrate) {
    if (surf->implicit) surf->compress_implicit();
    grid->compress();
    if (surf->distributed && !surf->implicit) surf->compress_explicit();
  }
  particle->sorted = 0;
  tt[nt++] = MPI_Wtime();   // compress

  if (!igrid) igrid = new Irregular(sparta);
  bigint recvsize;
  igrid->create_data_variable(nmigrate,gproc,gsize,recvsize,commsortflag);

  if (recvsize > maxrecvbuf) {
    memory->sfree(rbuf);
    maxrecvbuf = recvsize;
    rbuf = (char *) memory->smalloc(maxrecvbuf,"comm:rbuf");
    memset(rbuf,0,maxrecvbuf);
  }

  tt[nt++] = MPI_Wtime();   // plan
  igrid->exchange_variable(sbuf,gsize,rbuf);
  tt[nt++] = MPI_Wtime();   // exchange

  // pre-grow the cell arrays once for the incoming top-level cells (sub
  // cells add a little more); otherwise unpack_one grows them chunk by chunk
  {
    int nin = igrid->self_count();
    const int nrp0 = igrid->recv_nprocs(); const int *rn0 = igrid->recv_nums();
    for (int r = 0; r < nrp0; r++) nin += rn0[r];
    if (nin > 0) grid->grow_cells(nin, nin);
  }
  tt[nt++] = MPI_Wtime();   // pregrow

  offset = 0;
  const int nself = igrid->self_count();
  for (int i = 0; i < nself; i++) offset += grid->unpack_one(&rbuf[offset],1,0,1);
  const int nrp = igrid->recv_nprocs();
  const int *rp = igrid->recv_procs();
  const int *rn = igrid->recv_nums();
  for (int r = 0; r < nrp; r++) {
    cellbase[rp[r]] = grid->nlocal;
    for (int k = 0; k < rn[r]; k++) offset += grid->unpack_one(&rbuf[offset],1,0,1);
  }
  tt[nt++] = MPI_Wtime();   // unpack
  if (diag) {
    const char *nm[] = {"pack","compress","plan","exchange","pregrow","unpack"};
    char line[512]; int n = snprintf(line,sizeof(line),"OE_CELLMIG_CELLS rank=%d nlocal %d -> %d t(",me,nglocal,grid->nlocal);
    for (int i = 1; i < nt; i++) n += snprintf(line+n,sizeof(line)-n,"%s %.3f%s",nm[i-1],tt[i]-tt[i-1],i<nt-1?", ":") s\n");
    fputs(line,screen ? screen : stderr); if (logfile) fputs(line,logfile);
  }
}
