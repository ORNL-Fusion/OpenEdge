/* ----------------------------------------------------------------------
   fix population/control/kk — see header.
------------------------------------------------------------------------- */

#include <cstdlib>
#include <cstdio>
#include "fix_population_control_kokkos.h"
#include "kokkos.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "update.h"
#include "comm.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixPopulationControlKokkos::FixPopulationControlKokkos(SPARTA *sparta, int narg, char **arg) :
  FixPopulationControl(sparta, narg, arg),
  rand_pool(72345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(rng_);
#endif
  kokkos_flag = 1;
  execution_space = Device;
  datamask_read = PARTICLE_MASK | SPECIES_MASK | CUSTOM_MASK;
  datamask_modify = PARTICLE_MASK | CUSTOM_MASK;
  nmax_kk = nmax_;
  warned_fallback = 0;
  pop_debug_ = getenv("OE_POP_DEBUG") ? atoi(getenv("OE_POP_DEBUG")) : 0;
}

/* ---------------------------------------------------------------------- */

FixPopulationControlKokkos::~FixPopulationControlKokkos()
{
  if (copymode) return;
#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
#endif
}

/* ---------------------------------------------------------------------- */

void FixPopulationControlKokkos::end_of_step()
{
  const int nlocal = particle->nlocal;
  if (nlocal == 0) return;

  ParticleKokkos *particle_kk = (ParticleKokkos *) particle;
  GridKokkos *grid_kk = (GridKokkos *) grid;

  if (particle->nspecies > MAXSP)
    error->all(FLERR,"fix population/control/kk: more species than the per-cell "
               "species mask holds (raise MAXSP_WORDS in fix_population_control_kokkos.h)");

  // bucket by cell with the Kokkos sort (species handled inside the kernel)
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  particle_kk->sort_kokkos();
  d_particles = particle_kk->k_particles.view_device();
  d_pw = particle_kk->k_edvec.h_view[pweight_ewhich_].k_view.view_device();
  d_plist = grid_kk->d_plist;
  d_cellcount = grid_kk->d_cellcount;
  if (pop_debug_) { DeviceType().fence(); fprintf(stderr,"[popctl] rank %d step " BIGINT_FORMAT " sorted, nlocal %d plist %d x %d cells %d\n",comm->me,update->ntimestep,nlocal,(int) d_plist.extent(0),(int) d_plist.extent(1),grid->nlocal); }

  if ((int) d_del.extent(0) < nlocal) {
    d_del = DAT::t_int_1d(Kokkos::view_alloc("popcontrol:del",Kokkos::WithoutInitializing),nlocal);
    d_dellist = DAT::t_int_1d(Kokkos::view_alloc("popcontrol:dellist",Kokkos::WithoutInitializing),nlocal);
  }
  Kokkos::deep_copy(Kokkos::subview(d_del,std::make_pair(0,nlocal)),0);

  int ndelete = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType,TagFixPopControl_cells>(0,grid->nlocal),*this,ndelete);
  copymode = 0;

  particle_kk->modify(Device,CUSTOM_MASK);   // rescaled pweights live on the device
  if (pop_debug_) { DeviceType().fence(); fprintf(stderr,"[popctl] rank %d step " BIGINT_FORMAT " cells kernel done, ndelete %d\n",comm->me,update->ntimestep,ndelete); }

  if (ndelete) {
    copymode = 1;
    Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType,TagFixPopControl_scan>(0,nlocal),*this);
    copymode = 0;
    if (pop_debug_) { DeviceType().fence(); fprintf(stderr,"[popctl] rank %d step " BIGINT_FORMAT " scan done\n",comm->me,update->ntimestep); }
    particle_kk->compress_migrate_kokkos(ndelete,d_dellist);   // device compaction from the device list (ascending), marks modified
    if (pop_debug_) { DeviceType().fence(); fprintf(stderr,"[popctl] rank %d step " BIGINT_FORMAT " compress done\n",comm->me,update->ntimestep); }
    ndeleted_total_ += ndelete;
  }
  particle->sorted = 0;
  particle_kk->sorted_kk = 0;

  d_particles = t_particle_1d();
}

/* ----------------------------------------------------------------------
   one thread per cell: count per species, then for every species over
   Nmax select Nmax survivors uniformly (selection sampling), rescale their
   pweights by w_all/w_kept and flag the others for deletion
------------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixPopulationControlKokkos::operator()(TagFixPopControl_cells, const int &icell, int &ndel) const
{
  const int np = d_cellcount(icell);
  if (np <= nmax_kk) return;

  // species already handled in this cell (first occurrence drives the work)
  unsigned long long seen[MAXSP_WORDS];
  for (int w = 0; w < MAXSP_WORDS; w++) seen[w] = 0ULL;

  rand_type rand_gen = rand_pool.get_state();
  for (int n0 = 0; n0 < np; n0++) {
    const int s = d_particles[d_plist(icell,n0)].ispecies;
    if (s < 0 || s >= MAXSP) continue;
    const unsigned long long bit = 1ULL << (s & 63);
    if (seen[s >> 6] & bit) continue;
    seen[s >> 6] |= bit;

    int ns = 0;
    for (int n = n0; n < np; n++)
      if (d_particles[d_plist(icell,n)].ispecies == s) ns++;
    if (ns <= nmax_kk) continue;

    int seen_s = 0, kept = 0;
    double w_all = 0.0, w_kept = 0.0;
    for (int n = n0; n < np; n++) {
      const int i = d_plist(icell,n);
      if (d_particles[i].ispecies != s) continue;
      const double w = d_pw(i);
      w_all += w;
      // keep with probability (nmax-kept)/(ns-seen): uniform random subset
      const double u = rand_gen.drand();
      if (u * (ns - seen_s) < (double) (nmax_kk - kept)) { kept++; w_kept += w; }
      else d_del(i) = 1;
      seen_s++;
    }
    if (w_kept <= 0.0) {          // degenerate: leave the bucket untouched (CPU)
      for (int n = n0; n < np; n++) {
        const int i = d_plist(icell,n);
        if (d_particles[i].ispecies == s) d_del(i) = 0;
      }
      continue;
    }
    const double scale = w_all / w_kept;
    for (int n = n0; n < np; n++) {
      const int i = d_plist(icell,n);
      if (d_particles[i].ispecies == s && !d_del(i)) d_pw(i) *= scale;
    }
    ndel += ns - kept;
  }
  rand_pool.free_state(rand_gen);
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void FixPopulationControlKokkos::operator()(TagFixPopControl_scan, const int &i, int &offset, const bool final) const
{
  if (d_del(i)) {
    if (final) d_dellist(offset) = i;
    offset++;
  }
}
