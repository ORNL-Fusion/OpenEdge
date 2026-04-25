/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    FixCoulombBase -- shared Nanbu (1997) / Takizuka-Abe (1977) Coulomb
    scatter infrastructure. Internal base class; no FixStyle. Two
    user-facing fixes derive from it:

      fix coulomb/binary       binary kinetic-kinetic Coulomb pairs
      fix coulomb/background   kinetic vs. Maxwellian fluid background
------------------------------------------------------------------------- */

#ifndef SPARTA_FIX_COULOMB_BASE_H
#define SPARTA_FIX_COULOMB_BASE_H

#include "fix.h"
#include "grid_src.h"
#include <cmath>
#include <string>
#include <vector>

namespace SPARTA_NS {

// Tabulated A(s) for the Nanbu (1997) Coulomb collision algorithm.
// 801-point table with A from 10^5 down to 10^-3. Linear interpolation
// inside; asymptotic forms outside (1/s for small s, 3*exp(-s) for large
// s -- both <0.10% error at the boundaries). Inlined here because
// fix coulomb/base is the sole consumer.
class NanbuScatterTable {
 public:
  NanbuScatterTable() : initialized_(false) {}

  void initialize() {
    if (initialized_) return;
    const int npoints = 801;
    s_table_.resize(npoints);
    A_table_.resize(npoints);
    // s = -ln( coth(A) - 1/A )  (Nanbu 1997 Eq. 10), monotonically
    // increasing as A decreases.
    double A_power = 5.0;
    for (int i = 0; i < npoints; i++) {
      double A = std::pow(10.0, A_power);
      double s = -std::log(1.0 / std::tanh(A) - 1.0 / A);
      s_table_[i] = s;
      A_table_[i] = A;
      A_power -= 0.01;
    }
    initialized_ = true;
  }

  double get_A(double s) const {
    if (s < 2.0e-3) return 1.0 / s;
    if (s > 3.2)    return 3.0 * std::exp(-s);
    int lo = 0;
    int hi = static_cast<int>(s_table_.size()) - 1;
    while (lo < hi - 1) {
      int mid = (lo + hi) / 2;
      if (s_table_[mid] <= s) lo = mid;
      else hi = mid;
    }
    double ds = s_table_[hi] - s_table_[lo];
    if (ds < 1.0e-30) return A_table_[lo];
    double frac = (s - s_table_[lo]) / ds;
    return A_table_[lo] + frac * (A_table_[hi] - A_table_[lo]);
  }

 private:
  bool initialized_;
  std::vector<double> s_table_;
  std::vector<double> A_table_;
};

class RanKnuth;
class FixBackground;

class FixCoulombBase : public Fix {
 public:
  FixCoulombBase(class SPARTA *, int, char **);
  ~FixCoulombBase();
  int  setmask();
  void init();
  void end_of_step();
  double memory_usage();

  // True only if this fix reads from the per-particle plasma cache. In
  // background mode the fix interpolates directly from FixBackground and
  // does not touch the cache, so Update::init() can drop the corresponding
  // mask bits and skip the writes.
  bool needs_pcache() const { return !use_background_; }

 protected:
  NanbuScatterTable scatter_table_;
  RanKnuth *rng_;
  int use_background_;
  std::string plasma_fix_id_;
  FixBackground *pd_;

  // mode flags set by subclass ctors
  int do_binary_;        // FixCoulombBinary sets to 1
  int have_background_;  // FixCoulombBackground sets to 1

  // index into argv after FixCoulombBase parsed nevery + plasma block.
  // Subclasses continue parsing their mode-specific args from here.
  int iarg_after_common_;

  // plasma sources for Coulomb logarithm (Te, Ne)
  CollGridSrc srcTe_, srcNe_;

  // background collision parameters (set by FixCoulombBackground subclass)
  double A_bg_, Z_bg_, m_bg_, q_bg_;
  CollGridSrc srcTi_bg_, srcNi_bg_, srcVpar_bg_;
  CollGridSrc srcBx_, srcBy_, srcBz_;

  // scratch particle list for one cell
  int npmax_;
  int *plist_;

  // helper methods
  void refresh_compute_src(CollGridSrc &S);
  double read_src(const CollGridSrc &S, int ip, int icell) const;
  void parse_compute_src(const char *tok, CollGridSrc &dst, const char *label);
  void particle_rz(const class Particle::OnePart &p, double &R, double &Z) const;
  void pd_bfield_sparta(const class Particle::OnePart &p,
                        double &Bx, double &By, double &Bz) const;
  double pd_interp(const std::vector<double> &field,
                   const class Particle::OnePart &p) const;

  // core Nanbu algorithm
  void nanbu_collisions_cell(int icell, int np);
  void nanbu_background_cell(int icell, int np);
  double compute_coulomb_log(double ne, double Te_eV);
  double box_muller();
};

}  // namespace SPARTA_NS

#endif
