/* ----------------------------------------------------------------------
   OpenEdge: PMI surface reaction — Kokkos implementation.
   Provides react_kokkos() for thread-safe execution in Kokkos parallel
   regions.  All table data is stored in Kokkos device views so this
   works on both OpenMP and CUDA backends.
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(pmi/kk,SurfReactPMIKokkos)

#else

#ifndef SPARTA_SURF_REACT_PMI_KOKKOS_H
#define SPARTA_SURF_REACT_PMI_KOKKOS_H

#include "surf_react_pmi.h"
#include "kokkos_type.h"
#include "rand_pool_wrap.h"
#include "Kokkos_Random.hpp"
#include "particle_kokkos.h"
#include "math_const.h"

namespace SPARTA_NS {

// device-friendly version of SpeciesReactions (no pointers)
struct SpeciesReactionsKK {
  int ireflect, isputter, iabsorb;
};

class SurfReactPMIKokkos : public SurfReactPMI {
 public:
  SurfReactPMIKokkos(class SPARTA *, int, char **);
  SurfReactPMIKokkos(class SPARTA *);
  ~SurfReactPMIKokkos();
  void init();
  void tally_reset();
  void tally_update();

  void pre_react();
  void backup();
  void restore();

#ifndef SPARTA_KOKKOS_EXACT
  Kokkos::Random_XorShift64_Pool<DeviceType> rand_pool;
  typedef typename Kokkos::Random_XorShift64_Pool<DeviceType>::generator_type rand_type;
#else
  RandPoolWrap rand_pool;
  typedef RandWrap rand_type;
#endif

  RanKnuth* random_backup;

  t_particle_1d d_particles;
  t_species_1d d_species;

  // device tally storage: [0] = nsingle, [1..nlist] = tally_single
  DAT::t_int_1d d_tally_all;
  HAT::t_int_1d h_tally_all;
  DAT::t_int_scalar d_nsingle;
  DAT::t_int_1d d_tally_single;

  // cached constants for device access
  double kk_mvv2e, kk_joule2ev;
  int kk_si_units;
  int kk_mode;              // copy of mode (0=constant, 1=file)
  int kk_nE, kk_nA;         // grid dimensions
  int kk_nb_cdf;
  int kk_has_cdf_Y;
  double kk_Ebind;
  double kk_const_RN, kk_const_RE, kk_const_Y, kk_const_Ebind;

  // ---------- device views for table data ----------

  typedef Kokkos::View<double*, DeviceType> t_double_1d;

  // PMI coefficient tables [nE*nA]
  t_double_1d d_RN_table;
  t_double_1d d_RE_table;
  t_double_1d d_spyld_table;

  // energy/angle axes
  t_double_1d d_E_axis;
  t_double_1d d_A_axis;

  // optional CDF for sputter energy [nE*nA*nb_cdf]
  t_double_1d d_edist_Y;

  // per-species reaction lookup
  Kokkos::View<SpeciesReactionsKK*, DeviceType> d_species_reactions;

  // first product species index for each reaction
  DAT::t_int_1d d_react_product0;

 public:

  /* ---------------------------------------------------------------------- */

  template<int ATOMIC_REDUCTION>
  KOKKOS_INLINE_FUNCTION
  int react_kokkos(Particle::OnePart *&ip, int isurf, const double *norm,
                   Particle::OnePart *&jp, int &velreset,
                   const DAT::t_int_scalar &d_retry,
                   const DAT::t_int_scalar &d_nlocal) const
  {
    jp = NULL;

    int ispecies = ip->ispecies;
    SpeciesReactionsKK sr = d_species_reactions(ispecies);

    if (sr.ireflect < 0 && sr.isputter < 0 && sr.iabsorb < 0) return 0;

    // incident energy (eV) and angle (degrees)

    double *v = ip->v;
    double vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
    double vlen = sqrt(vsq);

    double mass = d_species[ispecies].mass;
    double E_eV = 0.5 * kk_mvv2e * mass * vsq;
    if (kk_si_units) E_eV *= kk_joule2ev;

    double theta_deg = 0.0;
    if (vlen > 0.0) {
      double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
      if (nlen > 0.0) {
        double dot = v[0]*norm[0] + v[1]*norm[1] + v[2]*norm[2];
        double cos_theta = -dot / (vlen * nlen);
        if (cos_theta < -1.0) cos_theta = -1.0;
        if (cos_theta > 1.0) cos_theta = 1.0;
        theta_deg = acos(cos_theta) * 180.0 / MathConst::MY_PI;
      }
    }

    // look up coefficients

    double RN, RE, Y, Eb;

    if (kk_mode == 0) {
      RN = kk_const_RN;
      RE = kk_const_RE;
      Y  = kk_const_Y;
      Eb = kk_const_Ebind;
    } else {
      RN = kk_interp_table(d_RN_table, E_eV, theta_deg);
      RE = kk_interp_table(d_RE_table, E_eV, theta_deg);
      Y  = kk_interp_table(d_spyld_table, E_eV, theta_deg);
      Eb = kk_Ebind;
    }

    if (RN < 0.0) RN = 0.0;
    if (RN > 1.0) RN = 1.0;
    if (RE < 0.0) RE = 0.0;
    if (RE > 1.0) RE = 1.0;
    if (Y < 0.0) Y = 0.0;

    double sum = RN + Y;
    if (sum > 1.0) { RN /= sum; Y /= sum; }

    // stochastic decision

    rand_type rand_gen = rand_pool.get_state();
    double u = rand_gen.drand();
    velreset = 1;

    if (u < RN && sr.ireflect >= 0) {
      // REFLECT

      int irxn = sr.ireflect;
      ip->ispecies = d_react_product0(irxn);

      double E_out = RE * E_eV;
      double mass_out = d_species[ip->ispecies].mass;
      double E_out_J = E_out / kk_joule2ev / kk_mvv2e;
      double vmag_out = sqrt(2.0 * E_out_J / mass_out);

      kk_sample_cosine(norm, ip->v, vmag_out, rand_gen);

      ip->erot = 0.0;
      ip->evib = 0.0;

      if (ATOMIC_REDUCTION == 0) {
        d_nsingle()++;
        d_tally_single(irxn)++;
      } else {
        Kokkos::atomic_inc(&d_nsingle());
        Kokkos::atomic_inc(&d_tally_single(irxn));
      }
      rand_pool.free_state(rand_gen);
      return irxn + 1;

    } else if (u < RN + Y && sr.isputter >= 0) {
      // SPUTTER

      int irxn = sr.isputter;

      double x[3];
      memcpy(x, ip->x, 3*sizeof(double));
      int icell = ip->icell;

      ip = NULL;

      int sput_species = d_react_product0(irxn);
      double mass_sput = d_species[sput_species].mass;

      double E_sput;
      if (kk_has_cdf_Y) {
        int iE = 0, iA = 0;
        if (kk_nE > 0) {
          iE = kk_lower_bound(d_E_axis, kk_nE, E_eV);
          if (iE >= kk_nE) iE = kk_nE - 1;
        }
        if (kk_nA > 0) {
          iA = kk_lower_bound(d_A_axis, kk_nA, theta_deg);
          if (iA >= kk_nA) iA = kk_nA - 1;
        }
        E_sput = kk_sample_from_cdf(d_edist_Y, iE, iA, rand_gen);
      } else {
        E_sput = kk_sample_thompson(Eb / 2.0, E_eV, rand_gen);
      }
      if (E_sput < 0.0) E_sput = 0.0;

      double E_sput_J = E_sput / kk_joule2ev / kk_mvv2e;
      double vmag_sput = sqrt(2.0 * E_sput_J / mass_sput);

      double v_sput[3];
      kk_sample_cosine(norm, v_sput, vmag_sput, rand_gen);

      int id = MAXSMALLINT * rand_gen.drand();

      int index;
      if (ATOMIC_REDUCTION == 0) {
        index = d_nlocal();
        d_nlocal()++;
      } else
        index = Kokkos::atomic_fetch_add(&d_nlocal(), 1);

      int reallocflag =
        ParticleKokkos::add_particle_kokkos(d_particles, index, id,
                                            sput_species, icell,
                                            x, v_sput, 0.0, 0.0);
      if (reallocflag) {
        d_retry() = 1;
        rand_pool.free_state(rand_gen);
        return 0;
      }
      jp = &d_particles[index];

      if (ATOMIC_REDUCTION == 0) {
        d_nsingle()++;
        d_tally_single(irxn)++;
      } else {
        Kokkos::atomic_inc(&d_nsingle());
        Kokkos::atomic_inc(&d_tally_single(irxn));
      }
      rand_pool.free_state(rand_gen);
      return irxn + 1;

    } else if (sr.iabsorb >= 0) {
      // ABSORB

      int irxn = sr.iabsorb;
      ip = NULL;

      if (ATOMIC_REDUCTION == 0) {
        d_nsingle()++;
        d_tally_single(irxn)++;
      } else {
        Kokkos::atomic_inc(&d_nsingle());
        Kokkos::atomic_inc(&d_tally_single(irxn));
      }
      rand_pool.free_state(rand_gen);
      return irxn + 1;
    }

    rand_pool.free_state(rand_gen);
    return 0;
  }

 private:

  /* ---------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  double kk_interp_table(const t_double_1d &table,
                         double e_eV, double a_deg) const
  {
    if (kk_nE < 2 || kk_nA < 2) return 0.0;

    double ec = e_eV, ac = a_deg;
    if (ec < d_E_axis(0)) ec = d_E_axis(0);
    if (ec > d_E_axis(kk_nE-1)) ec = d_E_axis(kk_nE-1);
    if (ac < d_A_axis(0)) ac = d_A_axis(0);
    if (ac > d_A_axis(kk_nA-1)) ac = d_A_axis(kk_nA-1);

    int iE2 = 1;
    while (iE2 < kk_nE-1 && d_E_axis(iE2) < ec) iE2++;
    int iE1 = iE2 - 1;

    int iA2 = 1;
    while (iA2 < kk_nA-1 && d_A_axis(iA2) < ac) iA2++;
    int iA1 = iA2 - 1;

    double tE = (d_E_axis(iE2) != d_E_axis(iE1)) ?
      (ec - d_E_axis(iE1)) / (d_E_axis(iE2) - d_E_axis(iE1)) : 0.0;
    double tA = (d_A_axis(iA2) != d_A_axis(iA1)) ?
      (ac - d_A_axis(iA1)) / (d_A_axis(iA2) - d_A_axis(iA1)) : 0.0;

    double w11 = (1.0 - tE) * (1.0 - tA);
    double w21 = tE * (1.0 - tA);
    double w12 = (1.0 - tE) * tA;
    double w22 = tE * tA;

    double val = w11*table(iE1*kk_nA + iA1) + w21*table(iE2*kk_nA + iA1) +
                 w12*table(iE1*kk_nA + iA2) + w22*table(iE2*kk_nA + iA2);

    return val;
  }

  /* ---------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  double kk_sample_thompson(double Eb, double Emax, rand_type &rng) const
  {
    if (Eb <= 0.0 || Emax <= 0.0) return 0.0;

    double emu = 1.0 / (Emax / Eb + 1.0);
    double betad2 = 1.0 / (emu * emu - 2.0 * emu + 1.0);

    double r = rng.drand();
    double arg = r / betad2;
    double energy = Eb / (1.0 - sqrt(arg)) - Eb;

    if (energy < 0.0) energy = 0.0;
    if (energy > Emax) energy = Emax;
    return energy;
  }

  /* ---------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  void kk_sample_cosine(const double *norm, double *dir,
                        double vmag, rand_type &rng) const
  {
    double tangent1[3], tangent2[3];

    double arb[3];
    if (fabs(norm[0]) < 0.9) {
      arb[0] = 1.0; arb[1] = 0.0; arb[2] = 0.0;
    } else {
      arb[0] = 0.0; arb[1] = 1.0; arb[2] = 0.0;
    }

    tangent1[0] = norm[1]*arb[2] - norm[2]*arb[1];
    tangent1[1] = norm[2]*arb[0] - norm[0]*arb[2];
    tangent1[2] = norm[0]*arb[1] - norm[1]*arb[0];

    double len = sqrt(tangent1[0]*tangent1[0] + tangent1[1]*tangent1[1] +
                      tangent1[2]*tangent1[2]);
    if (len > 0.0) {
      tangent1[0] /= len; tangent1[1] /= len; tangent1[2] /= len;
    }

    tangent2[0] = norm[1]*tangent1[2] - norm[2]*tangent1[1];
    tangent2[1] = norm[2]*tangent1[0] - norm[0]*tangent1[2];
    tangent2[2] = norm[0]*tangent1[1] - norm[1]*tangent1[0];

    double xi1 = rng.drand();
    double xi2 = rng.drand();

    double cosTheta = sqrt(xi1);
    double sinTheta = sqrt(1.0 - xi1);
    double phi = MathConst::MY_2PI * xi2;
    double cosPhi = cos(phi);
    double sinPhi = sin(phi);

    dir[0] = sinTheta * cosPhi * tangent1[0] +
             sinTheta * sinPhi * tangent2[0] +
             cosTheta * norm[0];
    dir[1] = sinTheta * cosPhi * tangent1[1] +
             sinTheta * sinPhi * tangent2[1] +
             cosTheta * norm[1];
    dir[2] = sinTheta * cosPhi * tangent1[2] +
             sinTheta * sinPhi * tangent2[2] +
             cosTheta * norm[2];

    len = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (len > 0.0) {
      dir[0] = dir[0]/len * vmag;
      dir[1] = dir[1]/len * vmag;
      dir[2] = dir[2]/len * vmag;
    }
  }

  /* ---------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  double kk_sample_from_cdf(const t_double_1d &cdf, int iE, int iA,
                            rand_type &rng) const
  {
    if (cdf.extent(0) == 0 || kk_nb_cdf <= 0) return 0.0;

    double u = rng.drand();

    int base = iE * kk_nA * kk_nb_cdf + iA * kk_nb_cdf;

    int lo = 0, hi = kk_nb_cdf - 1;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (cdf(base + mid) < u)
        lo = mid + 1;
      else
        hi = mid;
    }

    double cdf_lo = (lo > 0) ? cdf(base + lo - 1) : 0.0;
    double cdf_hi = cdf(base + lo);
    double frac = (cdf_hi > cdf_lo) ? (u - cdf_lo) / (cdf_hi - cdf_lo) : 0.5;

    return (lo + frac) / kk_nb_cdf;
  }

  /* ---------------------------------------------------------------------- */

  KOKKOS_INLINE_FUNCTION
  int kk_lower_bound(const t_double_1d &arr, int n, double val) const
  {
    int lo = 0, hi = n;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (arr(mid) < val) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
};

}

#endif
#endif
