/* ----------------------------------------------------------------------
    OpenEdge:
    fix evaporation — droplet evaporation driven by the plasma heat-flux
    vector (q_par, q_perp) carried by fix background.

    Syntax:
      fix ID evaporation Nevery MIXTURE background PD \
          [mass M] [radius R] [temp T] [heatflux/scale S] [rocket_eta E]

    background PD is required: the fix pulls q_par / q_perp and
    grad_Te_{R,Z} from that fix background at the droplet position.
    If the plasma.h5 does not carry q_par/q_perp, fix background prints
    a one-time warning at init and this fix reads back the built-in
    defaults (default_q_par / default_q_perp on pd).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(droplet/evaporate,FixDropletEvaporate)

#else

#ifndef SPARTA_FIX_EVAP_H
#define SPARTA_FIX_EVAP_H

#include "fix.h"
#include <string>

namespace SPARTA_NS {

class FixBackground;

class FixDropletEvaporate : public Fix {
 public:
  FixDropletEvaporate(class SPARTA*, int, char**);
  ~FixDropletEvaporate() override;
  int setmask() override;
  void init() override;
  double memory_usage() override;

  double heatflux_scale;  // multiplier on |q| (default 1.0)
  double rocket_eta;      // asymmetry parameter for rocket force [0,1]

 protected:
  int imix;
  void end_of_step() override;
  void start_of_step() override;

  std::string plasma_fix_id_;
  FixBackground *pd_;

  void droplet_evaporation_model(Particle::OnePart *ip, double dt_half);
  double set_mass   = -1.0;
  double set_temp   = -1.0;
  double set_radius = -1.0;

  void evap_half(double dt_half);
};

}

#endif
#endif
