/* ----------------------------------------------------------------------
   fix population/control/kk — see header.
------------------------------------------------------------------------- */

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

  if (particle->nspecies > MAXSP) {
    if (!warned_fallback && comm->me == 0 && screen)
      fprintf(screen,"fix population/control/kk: HOST fallback (more than %d species)\n",MAXSP);
    warned_fallback = 1;
    sparta->kokkos->note_fallback("fix population/control/kk","more than MAXSP species");
    particle_kk->sync(Host,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
    particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    FixPopulationControl::end_of_step();
    particle_kk->modify(Host,PARTICLE_MASK|CUSTOM_MASK);
    particle_kk->sync(Device,PARTICLE_MASK|CUSTOM_MASK);
    return;
  }

  // bucket by cell with the Kokkos sort (species handled inside the kernel)
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  particle_kk->sort_kokkos();
  d_particles = particle_kk->k_particles.view_device();
  d_pw = particle_kk->k_edvec.h_view[pweight_ewhich_].k_view.view_device();
  d_plist = grid_kk->d_plist;
  d_cellcount = grid_kk->d_cellcount;

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

  if (ndelete) {
    copymode = 1;
    Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType,TagFixPopControl_scan>(0,nlocal),*this);
    copymode = 0;
    auto h_dellist = Kokkos::create_mirror_view(Kokkos::subview(d_dellist,std::make_pair(0,ndelete)));
    Kokkos::deep_copy(h_dellist,Kokkos::subview(d_dellist,std::make_pair(0,ndelete)));
    particle_kk->compress_migrate(ndelete,h_dellist.data());   // device compaction, marks modified
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

  int cnt[MAXSP];
  for (int s = 0; s < MAXSP; s++) cnt[s] = 0;
  for (int n = 0; n < np; n++) cnt[d_particles[d_plist(icell,n)].ispecies]++;

  rand_type rand_gen = rand_pool.get_state();
  for (int s = 0; s < MAXSP; s++) {
    const int ns = cnt[s];
    if (ns <= nmax_kk) continue;

    int seen = 0, kept = 0;
    double w_all = 0.0, w_kept = 0.0;
    for (int n = 0; n < np; n++) {
      const int i = d_plist(icell,n);
      if (d_particles[i].ispecies != s) continue;
      const double w = d_pw(i);
      w_all += w;
      // keep with probability (nmax-kept)/(ns-seen): uniform random subset
      const double u = rand_gen.drand();
      if (u * (ns - seen) < (double) (nmax_kk - kept)) { kept++; w_kept += w; }
      else d_del(i) = 1;
      seen++;
    }
    if (w_kept <= 0.0) {          // degenerate: leave the bucket untouched (CPU)
      for (int n = 0; n < np; n++) {
        const int i = d_plist(icell,n);
        if (d_particles[i].ispecies == s) d_del(i) = 0;
      }
      continue;
    }
    const double scale = w_all / w_kept;
    for (int n = 0; n < np; n++) {
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
