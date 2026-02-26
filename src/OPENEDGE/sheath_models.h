/* ----------------------------------------------------------------------
    OpenEdge sheath helper models (EIRENE-style + Chodura metrics)
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_MODELS_H
#define OPENEDGE_SHEATH_MODELS_H

#include <vector>

namespace SPARTA_NS {
namespace SheathModels {

struct EireneSheathResult {
  double esheath_eV = 0.0;   // positive energy gain Ze*DeltaPhi in eV for Z=1
  double arg = 0.0;          // log() argument before clamp/fallback
  int fallback = 0;          // 1 if fallback 2.8*Te was used
};

struct BorodkinaSheathResult {
  double esheath_eV = 0.0;  // local positive drop: -(phi-phi_p)/e in eV
  double emag_vpm = 0.0;    // local |E| in V/m
  double fd = 0.0;          // DS blending fraction
  double lmps_m = 0.0;      // magnetic pre-sheath scale
  double lambdaD_m = 0.0;   // Debye scale at sheath entrance
  double rho_i_m = 0.0;     // ion gyro-radius scale
  double phi_ds_eV = 0.0;   // Debye-sheath contribution at sheath entrance
  double phi_cs_eV = 0.0;   // Chodura/magnetic pre-sheath contribution at sheath entrance
};

// c_s = sqrt((Te + Ti) * e / (2*mD))
double sound_speed_d(double te_eV, double ti_eV, double mD_amu);

// EIRENE-like sheath drop from multi-ion moments.
// dens_m3, upar_ms, charge_state_z must have same size.
EireneSheathResult eirene_sheath_ev(double te_eV,
                                    const std::vector<double> &dens_m3,
                                    const std::vector<double> &upar_ms,
                                    const std::vector<int> &charge_state_z,
                                    double gamma_see = 0.0,
                                    double cur_A_m2 = 0.0);

struct ChoduraMetrics {
  double bdotn = 0.0;      // Bhat dot n
  double alpha_deg = 90.0; // angle between B and normal
  double mach_par = 0.0;   // |u_par|/cs
  double mach_n = 0.0;     // |u_par*bdotn|/cs
  double u_n = 0.0;        // |u_par*bdotn|
};

// n and b can be non-unit vectors.
ChoduraMetrics chodura_metrics(double upar_ms,
                               double cs_ms,
                               const double b[3],
                               const double n[3]);

// Borodkina-style DS+CS profile at local distance-to-wall.
// alpha_deg = angle between B and wall normal (0 = normal, 90 = tangent).
// te_eV, ti_eV, ne_m3 are sheath-entrance values.
BorodkinaSheathResult borodkina_sheath_at_distance(double dist_m,
                                                   double te_eV,
                                                   double ti_eV,
                                                   double ne_m3,
                                                   double bmag_T,
                                                   double alpha_deg,
                                                   double mD_amu,
                                                   double pot_mult = 2.5);

// Stangeby-style Debye sheath (DS) + Chodura sheath (CS) profile:
// - alpha_deg = angle between B and wall normal (0 = normal, 90 = tangent)
// - CS drop: -Te*ln(cos(alpha)), clipped to phi_total
// - DS drop: phi_total - phi_cs
// - Decay scales: 2*lambdaD (DS), rho_i/cos(alpha) (CS)
BorodkinaSheathResult stangeby_sheath_at_distance(double dist_m,
                                                  double te_eV,
                                                  double ti_eV,
                                                  double ne_m3,
                                                  double bmag_T,
                                                  double alpha_deg,
                                                  double mD_amu,
                                                  double pot_mult = 0.0);

}  // namespace SheathModels
}  // namespace SPARTA_NS

#endif
