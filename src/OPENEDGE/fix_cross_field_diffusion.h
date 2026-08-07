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
          {bfield BxSRC BySRC BzSRC | background FIXID | bfield_const BX BY BZ} \
          [D_perp VAL | bohm [TeSRC in source-token mode] [scale VAL]] \
          [pinch Vr Vz] \
          [pinch_psi V] \
          [gradient_pinch Cp [neSRC gradNeR_SRC gradNeZ_SRC in source-token mode]]

    Example (direct background path):
      fix fcd cross_diffusion 100 \
          background pd \
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

FixStyle(cross_field_diffusion,FixCrossFieldDiffusion)

#else

#ifndef SPARTA_FIX_CROSS_FIELD_DIFFUSION_H
#define SPARTA_FIX_CROSS_FIELD_DIFFUSION_H

#include "fix.h"
#include "grid_src.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class RanKnuth;
class FixBackground;

class FixCrossFieldDiffusion : public Fix {
 public:
  FixCrossFieldDiffusion(class SPARTA *, int, char **);
  ~FixCrossFieldDiffusion();
  int  setmask();
  void init();
  void start_of_step();

  // True iff this fix actually consumes grad(ne) at runtime (gradient_pinch).
  // Used by Update::init() to keep PCACHE_GRAD_NE off when not needed,
  // which avoids 4 extra bilinear-interp queries per particle per step.
  bool needs_grad_ne() const { return have_grad_pinch_ != 0; }

  // True only if this fix reads from the per-particle plasma cache. In
  // background mode the fix interpolates directly from FixBackground and
  // does not touch the cache, so Update::init() can drop the corresponding
  // mask bits and skip the writes.
  bool needs_pcache() const { return !use_background_; }

 protected:
  RanKnuth *rng_;
  int use_background_;
  std::string plasma_fix_id_;
  FixBackground *pd_;

  // B-field sources in SPARTA coordinate order
  CollGridSrc srcBx_, srcBy_, srcBz_;

  // uniform B in SPARTA slots (bfield_const Bx By Bz) — for analytic
  // verification cases (e.g. MSD tests) with no plasma file or compute
  int use_const_;
  double Bconst_[3];

  // diffusion model
  int diff_model_;       // 0=none, 1=constant, 2=bohm
  double D_perp_;        // constant D_perp [m^2/s]
  CollGridSrc srcTe_;    // Te source for Bohm model
  double bohm_scale_;    // scale factor for Bohm (default 1.0)

  // constant pinch velocity (cylindrical R, Z components) [m/s]
  int have_pinch_;
  int have_psi_pinch_ = 0;
  double v_pinch_psi_ = 0.0;   // flux-surface-normal pinch [m/s], <0 = inward
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
  void pd_bfield_sparta(const class Particle::OnePart &p, int iparticle,
                        double &B0, double &B1, double &B2) const;
  double pd_interp(const std::vector<double> &field, int iparticle,
                   const class Particle::OnePart &p) const;
};

}  // namespace SPARTA_NS

#endif
#endif
