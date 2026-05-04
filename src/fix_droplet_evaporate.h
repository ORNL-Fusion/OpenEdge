/* ----------------------------------------------------------------------
    OpenEdge:
    fix evaporation — droplet evaporation driven by the plasma heat-flux
    vector (q_par, q_perp) carried by fix background.

    Syntax:
      fix ID evaporation Nevery MIXTURE background PD \
          [heatflux/scale S] [rocket_eta E] [emit_into MIXTURE_ID]

    Per-particle mass, radius, and bulk temperature are taken from the
    species file (extended 12-column format).

    emit_into MIX_ID:
      Optional — when set, evaporated atoms are spawned as new particles
      at each droplet's current cell. Number per call is Poisson with
      lambda = 4*pi*R^2 * Gevap_atoms * dt_half / fnum. Velocity is
      isotropic Maxwellian at droplet T. Species drawn from MIX_ID's
      fraction-weighted distribution.

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
class RanKnuth;

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

  // Optional volumetric Li source on droplet evaporation.
  // When emit_imix >= 0, each droplet spawns Maxwellian atoms in its
  // cell each evap call.  Mixture defines the species menu.
  int       emit_imix;
  RanKnuth *random;

  void droplet_evaporation_model(int idrop, double dt_half);
  void spawn_evap_atoms(int idrop, double area, double Gevap_atoms,
                         double TK, double dt_half);

  void evap_half(double dt_half);
};

}

#endif
#endif
