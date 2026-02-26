/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    Nanbu scatter table: tabulated A(s) for the Nanbu (1997) Coulomb
    collision algorithm.  801-point table with A from 10^5 down to 10^-3.
    Linear interpolation inside the table; asymptotic forms outside.
------------------------------------------------------------------------- */

#ifndef SPARTA_NANBU_SCATTER_TABLE_H
#define SPARTA_NANBU_SCATTER_TABLE_H

#include <cmath>
#include <vector>

namespace SPARTA_NS {

class NanbuScatterTable {
 public:
  NanbuScatterTable() : initialized_(false) {}

  void initialize() {
    if (initialized_) return;

    const int npoints = 801;
    s_table_.resize(npoints);
    A_table_.resize(npoints);

    // Build table: A sweeps from 10^5 down to 10^-3
    // s = -ln( coth(A) - 1/A )          (Nanbu 1997 Eq. 10)
    // As A decreases, s increases monotonically

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

  /* ------------------------------------------------------------------
     get_A: return A given s, using table interpolation.
     Asymptotic limits outside the tabulated range:
       s < 0.002  ->  A = 1/s   (small-angle limit,  0.10% error)
       s > 3.2    ->  A = 3*exp(-s)  (isotropic limit, 0.10% error)
  ------------------------------------------------------------------ */

  double get_A(double s) const {
    if (s < 2.0e-3) return 1.0 / s;
    if (s > 3.2)    return 3.0 * std::exp(-s);

    // Binary search: s_table_ is monotonically increasing
    int lo = 0;
    int hi = static_cast<int>(s_table_.size()) - 1;
    while (lo < hi - 1) {
      int mid = (lo + hi) / 2;
      if (s_table_[mid] <= s) lo = mid;
      else hi = mid;
    }

    // Linear interpolation between bracketing points
    double ds = s_table_[hi] - s_table_[lo];
    if (ds < 1.0e-30) return A_table_[lo];
    double frac = (s - s_table_[lo]) / ds;
    return A_table_[lo] + frac * (A_table_[hi] - A_table_[lo]);
  }

 private:
  bool initialized_;
  std::vector<double> s_table_;   // s values (increasing)
  std::vector<double> A_table_;   // corresponding A values (decreasing)
};

}  // namespace SPARTA_NS

#endif
