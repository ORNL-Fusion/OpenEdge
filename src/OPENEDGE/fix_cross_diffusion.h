/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix cross_diffusion: anomalous perpendicular diffusion and
    convective pinch for impurity ions.

    Applies per-particle position displacement perpendicular to B:
      1) Stochastic diffusion:  dx_perp = sqrt(2 * D_perp * dt) * xi
      2) Deterministic pinch:   dx += V_pinch * dt

    D_perp can be constant or Bohm-like: D_Bohm = Te / (16 * e * |B|).
    V_pinch can be constant (R, Z) or gradient-driven:
      V = C_p * D_perp * (grad_perp(ne) / ne)

    B-field sources must be in SPARTA coordinate order (bx, by, bz).

    Syntax:
      fix ID cross_diffusion Nevery \
          {bfield BxSRC BySRC BzSRC | plasma_data FIXID} \
          [D_perp VAL | bohm [TeSRC in source-token mode] [scale VAL]] \
          [pinch Vr Vz] \
          [gradient_pinch Cp [neSRC gradNeR_SRC gradNeZ_SRC in source-token mode]]

    Example (direct plasma/data path):
      fix fcd cross_diffusion 100 \
          plasma_data pd \
          D_perp 0.1 \
          gradient_pinch 2.0

    Example (constant D + constant pinch):
      fix fcd cross_diffusion 1 \
          bfield c_cwest[1] c_cwest[2] c_cwest[3] \
          D_perp 1.0 \
          pinch -50.0 0.0

    Example (constant D + gradient-driven pinch):
      fix fcd cross_diffusion 1 \
          bfield c_cwest[1] c_cwest[2] c_cwest[3] \
          D_perp 1.0 \
          gradient_pinch 2.0 c_cwest[6] c_cwest[18] c_cwest[19]

    For Nevery > 1, skipped timesteps leave cross-diffusion inactive and
    do not reuse stale displacement buffers.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(cross_diffusion,FixCrossDiffusion)

#else

#ifndef SPARTA_FIX_CROSS_DIFFUSION_H
#define SPARTA_FIX_CROSS_DIFFUSION_H

#include "fix.h"
#include "grid_src.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class RanKnuth;
class FixPlasmaData;

class FixCrossDiffusion : public Fix {
 public:
  FixCrossDiffusion(class SPARTA *, int, char **);
  ~FixCrossDiffusion();
  int  setmask();
  void init();
  void start_of_step();

 protected:
  RanKnuth *rng_;
  int use_plasma_data_;
  std::string plasma_fix_id_;
  FixPlasmaData *pd_;

  // B-field sources in SPARTA coordinate order
  CollGridSrc srcBx_, srcBy_, srcBz_;

  // diffusion model
  int diff_model_;       // 0=none, 1=constant, 2=bohm
  double D_perp_;        // constant D_perp [m^2/s]
  CollGridSrc srcTe_;    // Te source for Bohm model
  double bohm_scale_;    // scale factor for Bohm (default 1.0)

  // constant pinch velocity (cylindrical R, Z components) [m/s]
  int have_pinch_;
  double v_pinch_R_;
  double v_pinch_Z_;

  // gradient-driven pinch: V = C_p * D_perp * grad_perp(ne) / ne
  int have_grad_pinch_;
  double C_p_;
  CollGridSrc srcNe_;
  CollGridSrc srcGradNeR_, srcGradNeZ_;

  // helper methods
  void parse_compute_src(const char *tok, CollGridSrc &dst, const char *label);
  void refresh_compute_src(CollGridSrc &S);
  double read_src(const CollGridSrc &S, int ip, int icell) const;
  void particle_rz(const class Particle::OnePart &p, double &R, double &Z) const;
  void pd_bfield_sparta(const class Particle::OnePart &p,
                        double &B0, double &B1, double &B2) const;
  double pd_interp(const std::vector<double> &field,
                   const class Particle::OnePart &p) const;
};

}  // namespace SPARTA_NS

#endif
#endif
