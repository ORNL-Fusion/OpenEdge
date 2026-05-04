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

// 1D effective thermal speed sqrt((Te+Ti)*e / (2*mD)) used for sheath-scale
// estimates (lambda_D, rho_i, L_MPS). NOT the Bohm sound speed; for Bohm
// flux Gamma = n*c_s*sin(alpha_B) use cs = sqrt((Te+Ti)*e/mD) directly.
double vth_d_eff(double te_eV, double ti_eV, double mD_amu);

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

// Borodkina-style sheath profile at local distance-to-wall.
// alpha_deg is angle between B and wall normal (deg).
// te_eV, ti_eV, ne_m3 are sheath-entrance values.
BorodkinaSheathResult borodkina_sheath_at_distance(double dist_m,
                                                   double te_eV,
                                                   double ti_eV,
                                                   double ne_m3,
                                                   double bmag_T,
                                                   double alpha_deg,
                                                   double mD_amu,
                                                   double pot_mult = 2.5);

// Coulette-Manfredi kinetic PIC sheath model (two-exponential fit)
// Covers full CS+DS transition; slow component scaled by rho_i/lambdaD.
BorodkinaSheathResult coulette_manfredi_sheath_at_distance(double dist_m,
                                                           double te_eV,
                                                           double ti_eV,
                                                           double ne_m3,
                                                           double bmag_T,
                                                           double alpha_deg,
                                                           double mD_amu,
                                                           double pot_mult = 0.0);

// ----------------------------------------------------------------------
// Fast-evaluation API (prepare + emag/phi-at-distance).
//
// The subcycle loop in the Boris pusher evaluates E(d) on every step,
// but Te, ne, B, alpha and the derived scales (lambda_D, rho_i, L_MPS,
// fd, phi0, fit coefficients) are constant over one Boris call.
// `sheath_prepare_*` runs the expensive transcendentals once; the
// per-subcycle `sheath_emag_at_distance`/`sheath_phi_at_distance` are
// reduced to 2-4 exp() and a handful of multiplies.
// ----------------------------------------------------------------------

enum SheathKind { SHEATH_BORODKINA = 0, SHEATH_COULETTE_MANFREDI = 1 };

struct SheathEmagCoeffs {
  int kind = SHEATH_BORODKINA;

  // Physical scales (exposed for diagnostics).
  double lambdaD_m = 0.0;
  double lmps_m    = 0.0;
  double rho_i_m   = 0.0;
  double phi_total_eV = 0.0;
  double fd        = 0.0;

  // Borodkina: E(d) = amp_ds·exp(-d·inv_2lD) + amp_mps·exp(-d·inv_lmps)
  double amp_ds_vpm  = 0.0;
  double amp_mps_vpm = 0.0;
  double inv_2lD     = 0.0;
  double inv_lmps    = 0.0;
  double phi0_ds_eV  = 0.0;   // phi0 * fd
  double phi0_mps_eV = 0.0;   // phi0 * (1 - fd)

  // Coulette-Manfredi: two-exponential fit in s = d/lambdaD, with a
  // blended Borodkina-style Chodura tail for s >= s_blend_start.
  double inv_lD           = 0.0;
  double K1_scaled        = 0.0;
  double K2               = 0.0;
  double amp_slow_vpm     = 0.0;  // phi_cm_slow * K1_scaled / lambdaD
  double amp_fast_vpm     = 0.0;  // phi_cm_fast * K2        / lambdaD
  double phi_cm_slow_eV   = 0.0;
  double phi_cm_fast_eV   = 0.0;
  double s_blend_start    = 60.0;
  double s_blend_end      = 120.0;
  double s_blend_width_inv = 1.0 / 60.0;
  double e_slow_at_anchor_vpm = 0.0;  // amp_slow_vpm · exp(-K1_scaled·s_start)
};

SheathEmagCoeffs sheath_prepare_borodkina(double te_eV, double ti_eV,
                                          double ne_m3, double bmag_T,
                                          double alpha_deg, double mD_amu,
                                          double pot_mult = 2.5);

SheathEmagCoeffs sheath_prepare_coulette_manfredi(double te_eV, double ti_eV,
                                                  double ne_m3, double bmag_T,
                                                  double alpha_deg, double mD_amu,
                                                  double pot_mult = 0.0);

// Per-subcycle evaluation. `dist_m` must be >= 0; negative (overshoot)
// callers should gate upstream.
double sheath_emag_at_distance(const SheathEmagCoeffs &c, double dist_m);
double sheath_phi_at_distance (const SheathEmagCoeffs &c, double dist_m);

}  // namespace SheathModels
}  // namespace SPARTA_NS

#endif
