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

// helper: copy std::vector<double> to a new device view
static Kokkos::View<double*, DeviceType>
vec_to_device(const std::vector<double> &v, const std::string &label)
{
  size_t n = v.size();
  Kokkos::View<double*, DeviceType> d(Kokkos::view_alloc(label), n);
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < n; i++) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

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

  // cache scalar constants for device access
  kk_mvv2e = update->mvv2e;
  kk_joule2ev = update->joule2ev;
  kk_si_units = (strcmp(update->unit_style,"si") == 0) ? 1 : 0;
  kk_mode = mode;
  kk_nE = nE;
  kk_nA = nA;
  kk_nb_cdf = nb_cdf;
  kk_has_cdf_Y = has_cdf_Y;
  kk_Ebind = Ebind;
  kk_const_RN = const_RN;
  kk_const_RE = const_RE;
  kk_const_Y = const_Y;
  kk_const_Ebind = const_Ebind;

  // allocate device tally: [0]=nsingle, [1..nlist]=tally_single
  int ntally = 1 + nlist;
  d_tally_all = DAT::t_int_1d("surf_react_pmi:tally", ntally);
  h_tally_all = HAT::t_int_1d("surf_react_pmi:tally_mirror", ntally);
  d_nsingle = Kokkos::subview(d_tally_all, 0);
  d_tally_single = Kokkos::subview(d_tally_all, std::make_pair(1, ntally));
  Kokkos::deep_copy(d_tally_all, 0);

  // ---------- copy table data to device views ----------

  if (mode == 1) {
    d_RN_table = vec_to_device(RN_table, "pmi:RN_table");
    d_RE_table = vec_to_device(RE_table, "pmi:RE_table");
    d_spyld_table = vec_to_device(spyld_table, "pmi:spyld_table");
    d_E_axis = vec_to_device(E_axis, "pmi:E_axis");
    d_A_axis = vec_to_device(A_axis, "pmi:A_axis");

    if (has_cdf_Y)
      d_edist_Y = vec_to_device(edist_Y, "pmi:edist_Y");
  }

  // ---------- copy per-species reaction lookup ----------

  int nsp = particle->nspecies;
  d_species_reactions = Kokkos::View<SpeciesReactionsKK*, DeviceType>(
    Kokkos::view_alloc("pmi:species_reactions"), nsp);
  auto h_sr = Kokkos::create_mirror_view(d_species_reactions);
  for (int i = 0; i < nsp; i++) {
    if (species_reactions) {
      h_sr(i).ireflect = species_reactions[i].ireflect;
      h_sr(i).isputter = species_reactions[i].isputter;
      h_sr(i).iabsorb  = species_reactions[i].iabsorb;
    } else {
      h_sr(i).ireflect = h_sr(i).isputter = h_sr(i).iabsorb = -1;
    }
  }
  Kokkos::deep_copy(d_species_reactions, h_sr);

  // ---------- copy first product species for each reaction ----------

  {
    DAT::tdual_int_1d k_rp("pmi:react_product0", nlist);
    auto h_rp = k_rp.h_view;
    for (int i = 0; i < nlist; i++)
      h_rp(i) = rlist[i].products[0];
    k_rp.modify_host();
    k_rp.sync_device();
    d_react_product0 = k_rp.d_view;
  }

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
