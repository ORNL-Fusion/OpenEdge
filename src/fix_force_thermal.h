/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix thermal_force: Braginskii thermal forces on impurity ions.

    Applies per-particle half-kick acceleration from:
      1) Ion thermal force:       F_iT = beta_i * Z^2 * e * grad_par(Ti)
      2) Electron thermal force:  F_eT = alpha_e * Z^2 * e * grad_par(Te)

    where grad_par = (grad . bhat) is the parallel component of the
    temperature gradient.  Both forces push impurities toward higher
    temperature (toward the core).

    The parallel E-field (ambipolar) is NOT included here — it is a
    Lorentz force and should be applied via the Boris pusher through
    the existing global efield mechanism.

    B-field sources must be in SPARTA coordinate order (bx, by, bz),
    which in 2D maps to (B_R, B_Z, B_toroidal) and in 3D to Cartesian.
    Temperature gradient sources must be cylindrical (grad_T_R, grad_T_Z).

    Syntax:
      fix ID thermal_force Nevery \
          {bfield BxSRC BySRC BzSRC | background FIXID} \
          [ion_thermal yes|no [gradTiR_SRC gradTiZ_SRC in source-token mode]] \
          [elec_thermal yes|no [gradTeR_SRC gradTeZ_SRC in source-token mode]]

    Example (2D WEST, direct background path):
      fix ftf thermal_force 1 \
          background pd \
          ion_thermal yes \
          elec_thermal yes

    Example (legacy explicit-source path):
      fix ftf thermal_force 1 \
          bfield c_cwest[1] c_cwest[2] c_cwest[3] \
          ion_thermal yes c_cwest[12] c_cwest[13] \
          elec_thermal yes c_cwest[10] c_cwest[11]

    Default coefficients:
      beta_i  = 2.6  (Neu 1974 heavy-impurity limit, hard-coded)
      alpha_e = 0.71 (Braginskii Z_eff=1 limit, hard-coded)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(force/thermal,FixForceThermal)

#else

#ifndef SPARTA_FIX_FORCE_THERMAL_H
#define SPARTA_FIX_FORCE_THERMAL_H

#include "fix.h"
#include "grid_src.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class FixBackground;

class FixForceThermal : public Fix {
 public:
  FixForceThermal(class SPARTA *, int, char **);
  ~FixForceThermal();
  int  setmask();
  void init();
  void start_of_step();
  void end_of_step();

  // True only if this fix reads from the per-particle plasma cache. In
  // background mode the fix interpolates directly from FixBackground and
  // does not touch the cache, so Update::init() can drop the corresponding
  // mask bits and skip the writes.
  bool needs_pcache() const { return !use_background_; }

 protected:
  int use_background_;
  std::string plasma_fix_id_;
  FixBackground *pd_;

  // B-field sources in SPARTA coordinate order (bx, by, bz)
  CollGridSrc srcBx_, srcBy_, srcBz_;

  // ion thermal force
  int have_ion_thermal_;
  CollGridSrc srcGradTiR_, srcGradTiZ_;
  double beta_i_;  // coefficient (default 2.6)

  // electron thermal force
  int have_elec_thermal_;
  CollGridSrc srcGradTeR_, srcGradTeZ_;
  double alpha_e_;  // coefficient (default 0.71)

  // helper methods
  void parse_compute_src(const char *tok, CollGridSrc &dst, const char *label);
  void refresh_compute_src(CollGridSrc &S);
  double read_src(const CollGridSrc &S, int ip, int icell) const;
  void particle_rz(const class Particle::OnePart &p, double &R, double &Z) const;
  void pd_bfield_sparta(const class Particle::OnePart &p,
                        double &B0, double &B1, double &B2) const;
  double pd_interp(const std::vector<double> &field,
                   const class Particle::OnePart &p) const;
  double pd_grad(const std::vector<double> &mesh_grad,
                 const std::vector<double> &regular_grad,
                 const class Particle::OnePart &p) const;
  void kick_half(double dt_half);
};

}  // namespace SPARTA_NS

#endif
#endif
