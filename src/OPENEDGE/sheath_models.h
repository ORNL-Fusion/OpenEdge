/* ----------------------------------------------------------------------
    OpenEdge sheath helper models (Coulette-Manfredi profile + Chodura metrics)
------------------------------------------------------------------------- */

#ifndef OPENEDGE_SHEATH_MODELS_H
#define OPENEDGE_SHEATH_MODELS_H

#include <vector>

namespace SPARTA_NS {
namespace SheathModels {

// Shared sheath-profile result (name retained from the removed
// Borodkina model; today filled by the Coulette-Manfredi profile).
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

enum SheathKind { SHEATH_COULETTE_MANFREDI = 1 };

struct SheathEmagCoeffs {
  int kind = SHEATH_COULETTE_MANFREDI;

  // Physical scales (exposed for diagnostics).
  double lambdaD_m = 0.0;
  double lmps_m    = 0.0;
  double rho_i_m   = 0.0;
  double phi_total_eV = 0.0;
  double fd        = 0.0;

  // magnetic-pre-sheath tail scale (used by the CM blend tail)
  double inv_lmps  = 0.0;

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
  double e_slow_at_anchor_vpm = 0.0;  // MPS tail E amp: phi_slow(anchor)/L_mps (phi-consistent)
};

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
