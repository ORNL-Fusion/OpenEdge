/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(neutral/collide/background,FixNeutralCollideBackground)

#else

#ifndef SPARTA_FIX_NEUTRAL_COLLIDE_BACKGROUND_H
#define SPARTA_FIX_NEUTRAL_COLLIDE_BACKGROUND_H

#include "fix.h"
#include <vector>

namespace SPARTA_NS {

class FixBackground;
class RanKnuth;

/* Elastic collisions of KINETIC NEUTRAL particles against the FLUID
   plasma background (test-particle DSMC / BGK-like): per neutral, a
   virtual partner is sampled from the local background Maxwellian
   (ion channel: dens_i/temp_i/parr_flow; neutral channel:
   dens_n/temp_n from fluid-neutral runs), the collision fires with
   P = 1 - exp(-n_b sigma(g) g dt), and the neutral is scattered
   isotropically in the COM frame (partner is a bath: not updated).
   Cross-section is the VSS/VHS power-law sigma(g) =
   sigma_ref * (g_ref/g)^(2 omega - 1): omega = 0.5 is hard sphere,
   omega = 1 is the Langevin (polarization capture) 1/g limit for
   ion-neutral channels. Complements `collide vss` (kinetic-kinetic,
   Bird) and `fix coulomb/background` (Nanbu, charged). */

class FixNeutralCollideBackground : public Fix {
 public:
  FixNeutralCollideBackground(class SPARTA *, int, char **);
  ~FixNeutralCollideBackground();
  int setmask();
  void init();
  void end_of_step();
  double compute_scalar();      // cumulative collision count

 private:
  struct Channel {
    int active = 0;
    double sigma_ref = 0.0;     // [m^2] at g_ref
    double g_ref = 1.0e4;       // [m/s]
    double omega = 0.5;         // VSS exponent (0.5 = hard sphere)
    double mass_b = 0.0;        // partner mass [kg]
  };
  Channel chan_ion, chan_neut;

  char *pd_id;
  FixBackground *pd;
  RanKnuth *rng;
  bigint ncollide;
  int dim, axi;
  double mass_p_current;

  void collide_channel(const Channel &c, double *v, double nb,
                       double Tb_eV, const double ub[3]);
};

}

#endif
#endif
