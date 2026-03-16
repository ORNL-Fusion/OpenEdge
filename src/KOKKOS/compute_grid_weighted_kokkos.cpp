/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    Kokkos-enabled version of compute grid/weighted.
    Uses per-particle pweight custom attribute instead of global fnum.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_grid_weighted_kokkos.h"
#include "particle_kokkos.h"
#include "mixture.h"
#include "grid_kokkos.h"
#include "update.h"
#include "modify.h"
#include "memory_kokkos.h"
#include "error.h"
#include "sparta_masks.h"
#include "kokkos.h"

using namespace SPARTA_NS;

// user keywords (must match compute_grid_weighted.cpp)

enum{N_W,NRHO_W,MASSRHO_W,PXRHO_W,PYRHO_W,PZRHO_W,KERHO_W};

// internal accumulators (must match compute_grid_weighted.cpp)

enum{WCOUNT,WMASSSUM,WMVX,WMVY,WMVZ,WMVSQ,LASTSIZE};

/* ---------------------------------------------------------------------- */

ComputeGridWeightedKokkos::ComputeGridWeightedKokkos(SPARTA *sparta,
                                                     int narg, char **arg) :
  ComputeGridWeighted(sparta, narg, arg)
{
  kokkos_flag = 1;

  k_unique = DAT::tdual_int_1d("compute/grid_weighted:unique",npergroup);
  for (int m = 0; m < npergroup; m++)
    k_unique.h_view(m) = unique[m];
  k_unique.modify_host();
  k_unique.sync_device();
  d_unique = k_unique.d_view;
}

/* ---------------------------------------------------------------------- */

ComputeGridWeightedKokkos::~ComputeGridWeightedKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_vector_grid,vector_grid);
  memoryKK->destroy_kokkos(k_tally,tally);
  vector_grid = NULL;
  tally = NULL;
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeightedKokkos::compute_per_grid()
{
  if (sparta->kokkos->prewrap) {
    ComputeGridWeighted::compute_per_grid();
  } else {
    compute_per_grid_kokkos();
    k_tally.modify_device();
    k_tally.sync_host();
  }
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeightedKokkos::compute_per_grid_kokkos()
{
  invoked_per_grid = update->ntimestep;

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK|CUSTOM_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;

  // get device view of pweight custom dvec
  pweight_ewhich = particle->ewhich[pweight_index];
  d_pweight = particle_kk->k_edvec.h_view[pweight_ewhich].k_view.d_view;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  d_cellcount = grid_kk->d_cellcount;
  d_plist = grid_kk->d_plist;
  grid_kk->sync(Device,CINFO_MASK);
  d_cinfo = grid_kk->k_cinfo.d_view;

  d_s2g = particle_kk->k_species2group.d_view;
  int nlocal = particle->nlocal;

  // zero all accumulators

  Kokkos::deep_copy(d_tally,0.0);

  // scatter view setup

  need_dup = sparta->kokkos->need_dup<DeviceType>();
  if (particle_kk->sorted_kk && sparta->kokkos->need_atomics && !sparta->kokkos->atomic_reduction)
    need_dup = 0;

  if (need_dup)
    dup_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterDuplicated>(d_tally);
  else
    ndup_tally = Kokkos::Experimental::create_scatter_view<typename Kokkos::Experimental::ScatterSum, typename Kokkos::Experimental::ScatterNonDuplicated>(d_tally);

  copymode = 1;
  if (particle_kk->sorted_kk && sparta->kokkos->need_atomics && !sparta->kokkos->atomic_reduction)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_compute_per_grid>(0,nglocal),*this);
  else {
    if (sparta->kokkos->need_atomics)
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_compute_per_grid_atomic<1> >(0,nlocal),*this);
    else
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_compute_per_grid_atomic<0> >(0,nlocal),*this);
  }
  copymode = 0;

  if (need_dup) {
    Kokkos::Experimental::contribute(d_tally, dup_tally);
    dup_tally = {};
  }

  d_particles = t_particle_1d();
  d_plist = DAT::t_int_2d();
}

/* ---------------------------------------------------------------------- */

template<int NEED_ATOMICS>
KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_compute_per_grid_atomic<NEED_ATOMICS>, const int &i) const {

  auto v_tally = ScatterViewHelper<typename NeedDup<NEED_ATOMICS,DeviceType>::value,decltype(dup_tally),decltype(ndup_tally)>::get(dup_tally,ndup_tally);
  auto a_tally = v_tally.template access<typename AtomicDup<NEED_ATOMICS,DeviceType>::value>();

  const int ispecies = d_particles[i].ispecies;
  const int igroup = d_s2g(imix,ispecies);
  if (igroup < 0) return;

  const int icell = d_particles[i].icell;
  if (!(d_cinfo[icell].mask & groupbit)) return;

  const double mass = d_species[ispecies].mass;
  double* v = d_particles[i].v;
  const double pw = d_pweight[i];

  int k = igroup*npergroup;

  for (int m = 0; m < npergroup; m++) {
    switch (d_unique[m]) {
    case WCOUNT:
      a_tally(icell,k++) += pw;
      break;
    case WMASSSUM:
      a_tally(icell,k++) += pw * mass;
      break;
    case WMVX:
      a_tally(icell,k++) += pw * mass * v[0];
      break;
    case WMVY:
      a_tally(icell,k++) += pw * mass * v[1];
      break;
    case WMVZ:
      a_tally(icell,k++) += pw * mass * v[2];
      break;
    case WMVSQ:
      a_tally(icell,k++) += pw * mass * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
      break;
    }
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_compute_per_grid, const int &icell) const {
  if (!(d_cinfo[icell].mask & groupbit)) return;

  const int np = d_cellcount[icell];

  for (int n = 0; n < np; n++) {

    const int i = d_plist(icell,n);

    const int ispecies = d_particles[i].ispecies;
    const int igroup = d_s2g(imix,ispecies);
    if (igroup < 0) return;

    const double mass = d_species[ispecies].mass;
    double* v = d_particles[i].v;
    const double pw = d_pweight[i];

    int k = igroup*npergroup;

    for (int m = 0; m < npergroup; m++) {
      switch (d_unique[m]) {
      case WCOUNT:
        d_tally(icell,k++) += pw;
        break;
      case WMASSSUM:
        d_tally(icell,k++) += pw * mass;
        break;
      case WMVX:
        d_tally(icell,k++) += pw * mass * v[0];
        break;
      case WMVY:
        d_tally(icell,k++) += pw * mass * v[1];
        break;
      case WMVZ:
        d_tally(icell,k++) += pw * mass * v[2];
        break;
      case WMVSQ:
        d_tally(icell,k++) += pw * mass * (v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
        break;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeGridWeightedKokkos::query_tally_grid_kokkos(DAT::t_float_2d_lr &d_array)
{
  d_array = d_tally;
  return 0;
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeightedKokkos::post_process_grid_kokkos(int index, int nsample,
                                      DAT::t_float_2d_lr d_etally, int *emap,
                                      DAT::t_float_1d_strided d_vec)
{
  index--;
  int ivalue = index % nvalue;

  int lo = 0;
  int hi = nglocal;

  if (!d_etally.data()) {
    nsample = 1;
    d_etally = d_tally;
    emap = map[index];
    d_vec = d_vector_grid;
    nstride = 1;
  }

  this->nsample = nsample;
  this->d_etally = d_etally;
  this->d_vec = d_vec;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  grid_kk->sync(Device,CINFO_MASK);
  d_cinfo = grid_kk->k_cinfo.d_view;

  copymode = 1;

  switch (value[ivalue]) {

  case N_W:
    {
      wcount_col = emap[0];
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_N_W>(lo,hi),*this);
      break;
    }

  case NRHO_W:
    {
      wcount_col = emap[0];
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_NRHO_W>(lo,hi),*this);
      break;
    }

  case MASSRHO_W:
    {
      wmass_col = emap[0];
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_MASSRHO_W>(lo,hi),*this);
      break;
    }

  case PXRHO_W:
  case PYRHO_W:
  case PZRHO_W:
    {
      wmom_col = emap[0];
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_PXRHO_W>(lo,hi),*this);
      break;
    }

  case KERHO_W:
    {
      wke_col = emap[0];
      Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagComputeGridWeighted_KERHO_W>(lo,hi),*this);
      break;
    }
  }
  copymode = 0;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_N_W, const int &icell) const {
  d_vec[icell] = d_etally(icell,wcount_col) / nsample;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_NRHO_W, const int &icell) const {
  const double vol = d_cinfo[icell].volume;
  if (vol == 0.0) d_vec[icell] = 0.0;
  else {
    const double wt = d_cinfo[icell].weight / vol;
    d_vec[icell] = wt * d_etally(icell,wcount_col) / nsample;
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_MASSRHO_W, const int &icell) const {
  const double vol = d_cinfo[icell].volume;
  if (vol == 0.0) d_vec[icell] = 0.0;
  else {
    const double wt = d_cinfo[icell].weight / vol;
    d_vec[icell] = wt * d_etally(icell,wmass_col) / nsample;
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_PXRHO_W, const int &icell) const {
  const double vol = d_cinfo[icell].volume;
  if (vol == 0.0) d_vec[icell] = 0.0;
  else {
    const double wt = d_cinfo[icell].weight / vol;
    d_vec[icell] = wt * d_etally(icell,wmom_col) / nsample;
  }
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void ComputeGridWeightedKokkos::operator()(TagComputeGridWeighted_KERHO_W, const int &icell) const {
  const double vol = d_cinfo[icell].volume;
  if (vol == 0.0) d_vec[icell] = 0.0;
  else {
    const double wt = d_cinfo[icell].weight / vol;
    d_vec[icell] = eprefactor * wt * d_etally(icell,wke_col) / nsample;
  }
}

/* ---------------------------------------------------------------------- */

void ComputeGridWeightedKokkos::reallocate()
{
  if (grid->nlocal == nglocal) return;

  memoryKK->destroy_kokkos(k_vector_grid,vector_grid);
  memoryKK->destroy_kokkos(k_tally,tally);
  nglocal = grid->nlocal;
  memoryKK->create_kokkos(k_vector_grid,vector_grid,nglocal,"compute_grid_weighted:vector_grid");
  d_vector_grid = k_vector_grid.d_view;
  memoryKK->create_kokkos(k_tally,tally,nglocal,ntotal,"compute_grid_weighted:tally");
  d_tally = k_tally.d_view;
}
