/* ----------------------------------------------------------------------
   OpenEdge: PMI surface reaction — Kokkos implementation.
------------------------------------------------------------------------- */

#include "surf_react_pmi_kokkos.h"
#include "update.h"
#include "comm.h"
#include "particle_kokkos.h"
#include "sparta_masks.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfReactPMIKokkos::SurfReactPMIKokkos(SPARTA *sparta, int narg, char **arg) :
  SurfReactPMI(sparta, narg, arg),
  rand_pool(12345 + comm->me
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  kokkosable = 1;
  random_backup = NULL;
}

SurfReactPMIKokkos::SurfReactPMIKokkos(SPARTA *sparta) :
  SurfReactPMI(sparta),
  rand_pool(12345
#ifdef SPARTA_KOKKOS_EXACT
            , sparta
#endif
            )
{
  copy = 1;
}

/* ---------------------------------------------------------------------- */

SurfReactPMIKokkos::~SurfReactPMIKokkos()
{
  if (copy) return;

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.destroy();
  if (random_backup)
    delete random_backup;
#endif
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::init()
{
  SurfReactPMI::init();

  // cache physics constants
  kk_mvv2e = update->mvv2e;
  kk_joule2ev = update->joule2ev;
  kk_si_units = (strcmp(update->unit_style,"si") == 0) ? 1 : 0;

  // allocate device tally: [0]=nsingle, [1..nlist]=tally_single
  int ntally = 1 + nlist;
  d_tally_all = DAT::t_int_1d("surf_react_pmi:tally", ntally);
  h_tally_all = HAT::t_int_1d("surf_react_pmi:tally_mirror", ntally);
  d_nsingle = Kokkos::subview(d_tally_all, 0);
  d_tally_single = Kokkos::subview(d_tally_all, std::make_pair(1, ntally));

  Kokkos::deep_copy(d_tally_all, 0);

#ifdef SPARTA_KOKKOS_EXACT
  rand_pool.init(random);
#endif
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::tally_reset()
{
  SurfReact::tally_reset();
  Kokkos::deep_copy(d_tally_all, 0);
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::tally_update()
{
  Kokkos::deep_copy(h_tally_all, d_tally_all);
  ntotal += h_tally_all(0);
  for (int i = 0; i < nlist; i++)
    tally_total[i] += h_tally_all(1 + i);
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::pre_react()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device, PARTICLE_MASK | SPECIES_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::backup()
{
  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  d_particles = particle_kk->k_particles.d_view;

#ifdef SPARTA_KOKKOS_EXACT
  if (!random_backup)
    random_backup = new RanKnuth(12345 + comm->me);
  memcpy(random_backup, random, sizeof(RanKnuth));
#endif
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIKokkos::restore()
{
  Kokkos::deep_copy(d_tally_all, 0);

#ifdef SPARTA_KOKKOS_EXACT
  memcpy(random, random_backup, sizeof(RanKnuth));
#endif
}
