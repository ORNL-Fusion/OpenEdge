/* ----------------------------------------------------------------------
   Inline ion energy-angle distribution at a wall element: test ions are
   launched at the sheath edge (Maxwellian at Ti drifting along B into the
   wall) and followed with the pusher's Boris rotation plus the spatial
   Borodkina sheath impulse until they hit the wall. The impacts are
   binned in (E, theta from the normal); <Y> = sum f Y replaces the
   SHEATH-built database lookup of compute surface/physical/sputter.
   Histograms are cached on the exact local conditions.
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_IEAD_H
#define OPENEDGE_SHEATH_IEAD_H

#include <map>
#include <vector>

namespace SPARTA_NS {

class RanKnuth;

class SheathIead {
 public:
  // conditions are quantized before lookup: Te, Ti, ne, |B|, drift to the
  // nearest 1 % (log steps), the angle to 0.25 deg, so neighbouring elements
  // of one plasma cell share a histogram (per-triangle B interpolation would
  // otherwise make every triangle unique)
  struct Key {
    double te, ti, ne, bmag, alpha_deg, mass, upar;
    int Z;
    bool operator<(const Key &o) const;
  };
  struct Hist {
    int nE = 0, nT = 0;
    double emax = 0.0;                 // bins: E in [0, emax], theta in [0, 90]
    std::vector<double> f;             // nE*nT, sums to 1 over the impacts
    double escape_frac = 0.0;          // launched ions that left outward
    double d_launch = 0.0, dt = 0.0;   // launch distance and substep used
    long nstep_max = 0;
    double e_mean = 0.0, th_mean = 0.0;
    long nimpact = 0;
    double ec(int ie) const { return (ie + 0.5) * emax / nE; }
    double thc(int it) const { return (it + 0.5) * 90.0 / nT; }
  };

  // drift: 0 = sqrt(e(Te+Ti)/(2 mD)) (SHEATH tracker / database convention),
  //        1 = sqrt(e(Te+Ti)/mD) (Bohm-Chodura), 2 = the value passed per call
  //        (slot flow), falling back to 1 when it is <= 0
  SheathIead(int nmarkers, int nE, int nT, double mD_amu, int drift_mode,
             int seed = 20261110);
  ~SheathIead();

  // b, n in any common frame; only |b.n| matters. upar is used in drift
  // mode 2 (m/s, magnitude; falls back to mode 1 when <= 0).
  const Hist &get(double te_eV, double ti_eV, double ne_m3, const double b[3],
                  const double n[3], double mass_kg, int Z, double upar = 0.0);

  template <class F> double convolve(const Hist &h, F yield) const {
    double y = 0.0;
    for (int ie = 0; ie < h.nE; ie++)
      for (int it = 0; it < h.nT; it++) {
        const double w = h.f[ie * h.nT + it];
        if (w > 0.0) y += w * yield(h.ec(ie), h.thc(it));
      }
    return y;
  }

  size_t cache_size() const { return cache.size(); }
  // every cached histogram: a "# ..." header line then nE rows of nT weights
  void dump(const char *path) const;
  double seconds() const { return cpu_seconds; }
  double dmax_override = 0.0;          // global pusher ... dmax, else mps_end_m

 private:
  int nmarkers, nE, nT, drift_mode, seed;
  double mD_amu;
  RanKnuth *random;
  std::map<Key, Hist> cache;
  double cpu_seconds = 0.0;
  Hist compute(const Key &k);
};

}  // namespace SPARTA_NS

#endif
