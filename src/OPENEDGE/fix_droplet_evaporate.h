/* ----------------------------------------------------------------------
    OpenEdge:
    fix evaporation — droplet evaporation driven by the plasma heat-flux
    vector (q_par, q_perp) carried by fix background.

    Syntax:
      fix ID evaporation Nevery MIXTURE background PD \
          [heatflux/scale S] [rocket_eta E] [emit_into MIXTURE_ID] \
          [model dustt2005|dis2021] \
          [termination dis|dustt] [dis_mass_tol F]

    Per-particle mass, radius, and bulk temperature are taken from the
    species file (extended 12-column format).

    emit_into MIX_ID:
      Optional — when set, evaporated atoms are spawned as new particles
      at each droplet's current cell. Number per call is Poisson with
      lambda = 4*pi*R^2 * Gevap_atoms * dt_half / fnum. Velocity is
      isotropic Maxwellian at droplet T. Species drawn from MIX_ID's
      fraction-weighted distribution.

    model dis2021 (canonical termination):
      Follow the continuous DIS evaporation equation down to the relative
      mass tolerance dis_mass_tol (default 1e-12), then account the tiny
      numerical remainder as evaporated and remove the grain.  The emitted
      atom tally therefore closes exactly against the initial grain mass.

    model dustt2005 (default; canonical termination):
      Reproduce the DUSTT trajectory cutoff R/R0 = 0.1 (Pigarov et al.,
      Phys. Plasmas 12, 122508, 2005).  This is a breakup/termination rule,
      not complete evaporation: 99.9% of a spherical grain has evaporated
      and the remaining 0.1% is reported separately as a terminal remainder.

    termination dis|dustt explicitly overrides the model's termination rule
    for diagnostic A/B tests; production decks should normally omit it.

    background PD is required. Prescribed-flux and auto heating pull
    q_par / q_perp from that background and fail at init when neither
    component is present. OML heating does not request prescribed heat
    flux. There is deliberately no synthetic heat-flux fallback.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(particulate/thermal,FixDropletEvaporate)

#else

#ifndef SPARTA_FIX_EVAP_H
#define SPARTA_FIX_EVAP_H

#include "fix.h"
#include "particulate_model_kernels.h"
#include <string>

namespace SPARTA_NS {

class FixBackground;
class FixDropletCharge;
class RanKnuth;

class FixDropletEvaporate : public Fix {
 public:
  FixDropletEvaporate(class SPARTA*, int, char**);
  ~FixDropletEvaporate() override;
  int setmask() override;
  void init() override;
  double memory_usage() override;
  double compute_scalar() override;
  double compute_vector(int) override;

  double heatflux_scale;  // multiplier on |q| (default 1.0)
  double rocket_eta;      // asymmetry parameter for rocket force [0,1]
  double alpha_e_ = 1.0;   // HKL evaporation/accommodation coefficient

 protected:
  int imix;

  // Grain material (registry in grain_material.h). Resolved in init() so
  // `material` commands later in the deck still apply. Default "Li"
  // reproduces the original hardcoded model.
  char mat_name_[16];
  const struct GrainMaterial *mat_;
  double twall_K_;        // wall temperature for radiative cooling [K]

  // Heating model: 0 = prescribed |q|, 1 = OML collection, 2 = adaptive.
  int heating_mode_;
  double ion_mass_amu_;   // background ion mass for the OML ion flux
  double ion_charge_state_;
  ParticulateModel::PhysicsModel physics_model_;
  enum TerminationMode { TERMINATION_DIS = 0, TERMINATION_DUSTT = 1 };
  int termination_mode_;
  int termination_explicit_;
  double dis_mass_tol_;   // numerical completion tolerance m/m0 for DIS
  int dq_custom_;         // particulate_charge custom index (-1 = none)
  int nw_custom_;         // grain_nweight custom index (-1 = none)
  int heating_q_custom_;
  int debye_ratio_custom_;
  int oml_weight_custom_;
  int dis_q_ion_custom_;
  int dis_q_electron_custom_;
  int dis_q_neutralization_custom_;
  int dis_q_sheath_custom_;
  int dis_q_thermionic_custom_;
  int dis_q_secondary_custom_;
  void end_of_step() override;
  void start_of_step() override;

  // Cumulative real material atoms evaporated by all grains, on this rank.
  // compute_scalar() MPI-reduces across ranks on demand.
  double evap_atoms_local_;
  // DUSTT-only mass removed at R/R0=0.1, expressed as an equivalent atom
  // count.  It is deliberately not included in the evaporation/source tally.
  double terminal_remainder_atoms_local_;

  std::string plasma_fix_id_;
  FixBackground *pd_;
  std::string charge_fix_id_;
  FixDropletCharge *charge_fix_;

  // Optional volumetric material source on grain evaporation.
  // When emit_imix >= 0, each grain spawns Maxwellian atoms in its
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
