/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    Kokkos-enabled version of compute grid/weighted.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(grid/weighted/kk,ComputeGridWeightedKokkos)

#else

#ifndef SPARTA_COMPUTE_GRID_WEIGHTED_KOKKOS_H
#define SPARTA_COMPUTE_GRID_WEIGHTED_KOKKOS_H

#include "compute_grid_weighted.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace SPARTA_NS {

template<int NEED_ATOMICS>
struct TagComputeGridWeighted_compute_per_grid_atomic{};

struct TagComputeGridWeighted_compute_per_grid{};
struct TagComputeGridWeighted_N_W{};
struct TagComputeGridWeighted_NRHO_W{};
struct TagComputeGridWeighted_MASSRHO_W{};
struct TagComputeGridWeighted_PXRHO_W{};
struct TagComputeGridWeighted_KERHO_W{};

class ComputeGridWeightedKokkos : public ComputeGridWeighted, public KokkosBase {
 public:
  ComputeGridWeightedKokkos(class SPARTA *, int, char **);
  ~ComputeGridWeightedKokkos();
  void compute_per_grid();
  void compute_per_grid_kokkos();
  int query_tally_grid_kokkos(DAT::t_float_2d_lr &);
  void post_process_grid_kokkos(int, int, DAT::t_float_2d_lr, int *,
                                  DAT::t_float_1d_strided);
  void reallocate();

  template<int NEED_ATOMICS>
  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_compute_per_grid_atomic<NEED_ATOMICS>, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_compute_per_grid, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_N_W, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_NRHO_W, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_MASSRHO_W, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_PXRHO_W, const int&) const;

  KOKKOS_INLINE_FUNCTION
  void operator()(TagComputeGridWeighted_KERHO_W, const int&) const;

  DAT::tdual_float_1d k_vector_grid;

 private:
  DAT::tdual_float_2d_lr k_tally;
  DAT::t_float_2d_lr d_tally;
  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterDuplicated> dup_tally;
  Kokkos::Experimental::ScatterView<F_FLOAT**, typename DAT::t_float_2d_lr::array_layout,DeviceType,typename Kokkos::Experimental::ScatterSum,typename Kokkos::Experimental::ScatterNonDuplicated> ndup_tally;

  DAT::t_float_2d_lr d_etally;
  DAT::t_float_1d_strided d_vec;

  t_cinfo_1d d_cinfo;
  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_int_2d d_s2g;

  DAT::t_int_1d d_cellcount;
  DAT::t_int_2d d_plist;

  DAT::tdual_int_1d k_unique;
  DAT::t_int_1d d_unique;

  // device-accessible pweight array
  DAT::t_float_1d d_pweight;

  int wcount_col,wmass_col,wmom_col,wke_col;
  int nsample,nstride;
};

}

#endif
#endif
