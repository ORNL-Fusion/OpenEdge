/* ----------------------------------------------------------------------
   OpenEdge: Nanbu Coulomb collisions — Kokkos device kernel.
   Background-mode collisions run as a parallel_for over grid cells.
   Binary particle-particle collisions fall back to host.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coll/nanbu/kk,FixCollNanbuKokkos)

#else

#ifndef SPARTA_FIX_COLL_NANBU_KOKKOS_H
#define SPARTA_FIX_COLL_NANBU_KOKKOS_H

#include "fix_coll_nanbu.h"
#include "kokkos_type.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"
#include "math_const.h"

namespace SPARTA_NS {

class FixCollNanbuKokkos : public FixCollNanbu {
 public:
  FixCollNanbuKokkos(class SPARTA *, int, char **);
  ~FixCollNanbuKokkos();
  void init();
  void end_of_step();

 private:

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  // device views for scatter table (801 points)
  typedef Kokkos::View<double*, DeviceType> t_double_1d;
  t_double_1d d_s_table;
  t_double_1d d_A_table;
  int scatter_npts;

  // column indices into the plasma compute's d_array_grid
  int col_Te, col_Ne;
  int col_Ti_bg, col_Ni_bg, col_Vpar_bg;
  int col_Bx, col_By, col_Bz;

  // cached device views (set in end_of_step before kernel launch)
  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_float_2d_lr d_plasma;  // compute plasma/fields device array
  DAT::t_int_1d d_cellcount;
  DAT::t_int_2d d_plist;

  // cached scalar constants
  double kk_echarge, kk_epsilon_0, kk_fnum, kk_dt;
  double kk_A_bg, kk_Z_bg, kk_m_bg, kk_q_bg;

  // resolve compute column index from CollGridSrc
  int resolve_column(const CollGridSrc &S);

  // device kernel: background collisions for one cell
  KOKKOS_INLINE_FUNCTION
  void background_cell_kernel(int icell) const;

  // device-side helpers
  KOKKOS_INLINE_FUNCTION
  double kk_get_A(double s) const;

  KOKKOS_INLINE_FUNCTION
  double kk_box_muller(rand_type &rng) const;

  KOKKOS_INLINE_FUNCTION
  double kk_coulomb_log(double ne, double Te_eV) const;

 public:
  // Kokkos functor operator
  KOKKOS_INLINE_FUNCTION
  void operator()(int icell) const {
    background_cell_kernel(icell);
  }
};

}

#endif
#endif
