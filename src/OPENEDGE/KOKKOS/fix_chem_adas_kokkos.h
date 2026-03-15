/* ----------------------------------------------------------------------
   OpenEdge: ADAS ionization/recombination — Kokkos device kernel.
   Rate table lookups and Poisson draws run on device.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(chem/adas/kk,FixChemAdasKokkos)

#else

#ifndef SPARTA_FIX_CHEM_ADAS_KOKKOS_H
#define SPARTA_FIX_CHEM_ADAS_KOKKOS_H

#include "fix_chem_adas.h"
#include "kokkos_type.h"
#include "Kokkos_Random.hpp"
#include "rand_pool_wrap.h"
#include "particle_kokkos.h"
#include "grid_kokkos.h"

namespace SPARTA_NS {

class FixChemAdasKokkos : public FixChemAdas {
 public:
  FixChemAdasKokkos(class SPARTA *, int, char **);
  ~FixChemAdasKokkos();
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

  typedef Kokkos::View<double*, DeviceType> t_double_1d;
  typedef Kokkos::View<int*, DeviceType> t_int_1d_kk;

  // device views for rate tables
  t_double_1d d_ion_coeff, d_rec_coeff;
  t_double_1d d_gridT_ion, d_gridD_ion, d_gridT_rec, d_gridD_rec;
  int kk_ion_nQ, kk_ion_nT, kk_ion_nD;
  int kk_rec_nQ, kk_rec_nT, kk_rec_nD;

  // per-species reaction info on device
  t_int_1d_kk d_rxn_offset;    // start index in d_rxn_list per species
  t_int_1d_kk d_rxn_count;     // reaction count per species
  t_int_1d_kk d_rxn_list;      // flat list of reaction indices
  t_int_1d_kk d_rxn_type;      // 0=ionization, 1=recombination per reaction
  t_int_1d_kk d_rxn_product0;  // products[0] per reaction

  // column indices into plasma compute d_array_grid
  int col_Te, col_Ne;

  // cached device views (set each end_of_step)
  t_particle_1d d_particles;
  t_species_1d d_species;
  DAT::t_float_2d_lr d_plasma;
  int kk_nlocal;

  // cached scalars
  double kk_dt_chem;
  int kk_atomic_number;
  int kk_nlist;

  // device tally
  t_int_1d_kk d_tally_reactions;
  DAT::t_int_scalar d_nreact_one;

  int resolve_column(const GridSrc &S);

 public:
  KOKKOS_INLINE_FUNCTION
  void operator()(int i) const;

 private:
  KOKKOS_INLINE_FUNCTION
  int kk_attempt(int idx, double Te_eV, double ne_m3, rand_type &rng) const;

  KOKKOS_INLINE_FUNCTION
  double kk_interp_rate(int is_rec, int charge_idx,
                        double logTe, double logne_cm) const;

  KOKKOS_INLINE_FUNCTION
  double kk_bilinear(double x0, double x1, double y0, double y1,
                     double f00, double f01, double f10, double f11,
                     double x, double y) const;

  KOKKOS_INLINE_FUNCTION
  int kk_bracket(const t_double_1d &grid, int n, double x,
                 int &ilo, int &ihi) const;
};

}

#endif
#endif
