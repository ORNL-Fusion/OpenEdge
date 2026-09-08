/* ----------------------------------------------------------------------
   Pure particulate-model kernels shared by charge, drag, and thermal fixes.

   These functions contain no SPARTA state.  Keeping the algebra in this
   h/cpp pair makes the public fixes explicit dispatchers
   (`model dustt2005|dis2021`) and allows direct golden-vector testing.

   DIS conventions (Nespoli et al., Phys. Plasmas 28, 073704, 2021):
     chi = -e Phi_d / T_e       (>0 for a negatively charged grain)
     U   = |u_i-v_d| / sqrt(2 T_i/m_i)
     X   = Z_i chi / (T_i/T_e)
------------------------------------------------------------------------- */

#ifndef SPARTA_PARTICULATE_MODEL_KERNELS_H
#define SPARTA_PARTICULATE_MODEL_KERNELS_H

namespace SPARTA_NS {
namespace ParticulateModel {

enum PhysicsModel { DUSTT2005 = 0, DIS2021 = 1 };

// Secondary-electron closures are named separately from the transport
// model.  In particular, the Young--Dekker closure printed in DIS cannot be
// evaluated without beta and F_YD from Ref. 17; callers must reject that
// option until those data are supplied rather than silently substituting a
// different yield law.
enum SecondaryEmissionModel {
  SECONDARY_NONE = 0,
  SECONDARY_DUSTT2005 = 1,
  SECONDARY_KOLLATH_SMIRNOV2007 = 2,
  SECONDARY_YOUNG_DEKKER = 3
};

struct DISCurrentParameters {
  double Te_eV;
  double Ti_eV;
  double ne_m3;
  double ni_m3;
  double Td_K;
  double radius_m;
  double relative_mach;
  double ion_mass_kg;
  double ion_charge_state;
  double work_function_eV;
  double richardson_A;
  double see_delta_m;
  double see_E_m_eV;
  double echarge;
  double electron_mass;
  double boltz;
  int thermionic_on;
  SecondaryEmissionModel secondary_model;
  int see_angular_correction;
};

struct DISCurrentState {
  double chi;
  double phi_V;
  double ion_A;
  double electron_A;
  double thermionic_unsuppressed_A;
  double thermionic_A;
  double secondary_A;
  double secondary_yield;
  double secondary_escape_fraction;
  double residual_A;
};

struct DISHeatFluxState {
  double ion_kinetic_W_m2;
  double electron_kinetic_W_m2;
  double ion_neutralization_W_m2;
  double sheath_W_m2;
  double thermionic_cooling_W_m2;
  double secondary_cooling_W_m2;
  double net_W_m2;
};

double dis_ion_current_factor(double U, double X);
double dis_electron_current_factor(double chi);
double dis_thermionic_recollection_factor(double chi, double Te_eV,
                                           double Td_K, double echarge,
                                           double boltz);
double dis_ion_energy_factor(double U, double X);
double dis_electron_energy_factor(double chi);

// Smirnov Sec. 3.1 / Kollath normal-incidence secondary-electron model.
// The flux average is evaluated analytically after reducing the Maxwellian
// integral to incomplete Gaussian moments; no per-particle quadrature or
// yield clipping is used.  The optional angular correction is the cosine-
// incidence average of (cos alpha)^(-0.4), namely 1.25.
double dis_kollath_yield(double incident_energy_eV, double delta_m,
                         double E_m_eV);
double dis_kollath_flux_averaged_yield(double Te_eV, double chi,
                                       double delta_m, double E_m_eV,
                                       int angular_correction);
double dis_secondary_escape_fraction(double barrier_over_work_function);
double dis_secondary_energy_per_electron(double barrier_over_work_function,
                                          double work_function_eV);
double dis_thermionic_energy_escape_fraction(double barrier_over_kTd);

// Pure, shared DIS current and energy balances.  Charge and thermal fixes
// call these same routines so an emission current cannot affect the floating
// potential without its corresponding cooling term (or vice versa).
bool dis_current_state_at_chi(const DISCurrentParameters &parameters,
                              double chi, DISCurrentState &state);
bool dis_solve_current_balance(const DISCurrentParameters &parameters,
                               DISCurrentState &state);
bool dis_heat_flux(const DISCurrentParameters &parameters,
                   const DISCurrentState &current,
                   double ion_neutralization_eV,
                   DISHeatFluxState &heat);

// Smirnov et al. (PPCF 49, 347, 2007) Eqs. (5)/(6), the primary source for
// DIS A2/A4.  Eq. (6) corrects the sign error in printed DIS A4 and is
// verified against the exact OML moment for positive grains (X<0).
bool dis_ion_drag_collection_factor(double U, double X, double &factor);
double dis_ion_drag_scattering_factor(double U, double X, double lnLambda);
bool dis_ion_drag_factor(double U, double X, double lnLambda, double &factor);

} // namespace ParticulateModel
} // namespace SPARTA_NS

#endif
