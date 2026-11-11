/* ----------------------------------------------------------------------
   OpenEdge: 1-D flux-surface-averaged impurity transport (core model)
   Linder finite-volume scheme (Linder et al., NF 60 (2020) 096031):
   cylindrical geometry in the volume radius, implicit, two half steps
   sweeping the charge states up then down, one tridiagonal solve per
   state and half step. Units are the caller's (consistent length/time),
   densities per volume.
------------------------------------------------------------------------- */

#ifndef OPENEDGE_CORE_TRANSPORT_1D_H
#define OPENEDGE_CORE_TRANSPORT_1D_H

#include <vector>

namespace SPARTA_NS {

class CoreTransport1D {
 public:
  CoreTransport1D(int ir, int nion);

  // cell-centre radii, increasing, rr[0] = axis cell; sets the cell volumes
  void set_grid(const double *rr);

  // one step of length dt. Arrays [nion][ir] row-major: diff, conv (per
  // state), s_rates (ionization z -> z+1, 1/s), r_rates (recombination
  // z+1 -> z, stored at index z, 1/s), n in/out. Optional (may be null):
  //   n0        [ir]      evolve_neutrals = false: n[0][:] := n0, a fixed
  //                       neutral density (recombination to neutrals is
  //                       then a sink); true: volume source rate of the
  //                       transported neutrals (conservative)
  //   src       [nion][ir] volume source rate added to every state
  //   par_loss  [ir]      parallel loss frequency (1/s), cells 1..ir-1
  // dlen > 0: decay length at the outer boundary (outward flux D n/dlen
  // plus outward advection); dlen <= 0: closed outer boundary.
  void step(double dt, const double *diff, const double *conv,
            const double *s_rates, const double *r_rates,
            const double *n0, const double *src, const double *par_loss,
            double dlen, bool evolve_neutrals, double *n, bool check = false);

  // Conservative, positivity-limited substeps. Fluxes are time averages
  // over the entire requested interval. Throws on invalid inputs/results.
  void advance(double dt, const double *diff, const double *conv,
               const double *s_rates, const double *r_rates,
               const double *n0, const double *src, const double *par_loss,
               double dlen, bool evolve_neutrals, double *n);
  int substeps() const { return substeps_; }

  // Implicit first-order split step(s): per state a backward-Euler
  // monotone transport solve (sources added there), then per cell a
  // backward-Euler solve of the charge chain. Unconditionally positive
  // and conservative; accuracy is first order in dt/nsub. Neutrals are
  // transported (evolve mode). Throws on invalid inputs/results.
  void advance_split(double dt, const double *diff, const double *conv,
                     const double *s_rates, const double *r_rates,
                     const double *src, const double *par_loss,
                     double dlen, double *n, int nsub = 1);

  // bookkeeping of the last step, per state, per unit axial length
  // (multiply by 2 pi R_axis for a torus): particles/time
  double outflux(int z) const { return outflux_[z]; }    // through the outer face
  double parloss(int z) const { return parloss_[z]; }    // parallel loss sink
  // content of one state, per unit axial length
  double content(int z, const double *n) const;
  double cell_volume(int i) const { return vol_[i]; }
  int nr() const { return ir_; }
  int nstates() const { return nion_; }

 private:
  int ir_, nion_;
  std::vector<double> rr_, vol_;
  std::vector<double> a_, b_, c_, d_, outflux_, parloss_;
  // outer-face coefficient of the last assembled matrix: flux per unit
  // length = 2 pi edge_coef_ * (n_new + n_old)/2 at the outer cell
  double edge_coef_;
  int substeps_ = 1;

  void assemble(const double *rnt, const double *dimp, const double *vimp,
                double dt, double dlen, const double *par_loss);
  void tdma(double *sol);
  void tally(int z, const double *n_old, const double *n_new, double dt,
             double dlen, const double *par_loss);
};

}

#endif
