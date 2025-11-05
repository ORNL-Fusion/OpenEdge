// #ifndef SHETH_MODEL_BROOKSLIKE_H
// #define SHETH_MODEL_BROOKSLIKE_H

// #include <cmath>

// namespace Sheath {

// /**
//  * Parameterization:
//  *   φ/Te = -[ B1(α) e^{-k1(α) s} + B2(α) e^{-k2(α) s} ],  s = x / λ_D
//  *   Ê(s) = +[ B1 k1 e^{-k1 s} + B2 k2 e^{-k2 s} ]  where Ê = E / (Te/λ_D)
//  * α in degrees. Te in eV when converting to physical units.
//  */

// // Coefficient getters (α in degrees)
// double B1_of_alpha(double alpha_deg);
// double B2_of_alpha(double alpha_deg);
// double k1_of_alpha(double alpha_deg);
// double k2_of_alpha(double alpha_deg);

// // Dimensionless profiles
// double sheath_phi_over_Te(double alpha_deg, double s_norm);
// double sheath_E_over_Te_per_lambdaD(double alpha_deg, double s_norm);

// // Physical-unit helpers
// // Potential in volts (Te_eV in eV)
// inline double sheath_phi_V(double alpha_deg, double s_norm, double Te_eV) {
//     return Te_eV * sheath_phi_over_Te(alpha_deg, s_norm);
// }

// // Electric field in V/m (Te_eV in eV, lambda_D_m in meters)
// inline double sheath_E_Vpm(double alpha_deg, double s_norm,
//                            double Te_eV, double lambda_D_m) {
//     // E = (Te/λ_D) * Ê  ; with Te in eV → numeric factor is just Te_eV (since eΦ/Te_eV is dimensionless)
//     // Using Te in eV: Te_eV / λ_D [V/m] times dimensionless Ê
//     return (Te_eV / lambda_D_m) * sheath_E_over_Te_per_lambdaD(alpha_deg, s_norm);
// }

// } // namespace Sheath

// #endif // SHETH_MODEL_BROOKSLIKE_H
