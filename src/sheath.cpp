// // -----------------------------------------------------------------------------
// // DS + MPS composite sheath (GITRm-style model 1)
// //   Phi(d)  = - pot * [ fd * exp(-d / (2*lambdaD)) + (1-fd) * exp(-d / L_MPS) ]
// //   |E|(d)  =   pot * [ fd/(2*lambdaD) * exp(-d / (2*lambdaD))
// //                     + (1-fd)/L_MPS   * exp(-d / L_MPS) ]
// //   ne(d)   =   ne0 * exp( Phi(d) / Te_eV )
// // where: lambdaD = sqrt( eps0 * Te_eV / (ne0 * qe) )
// //        cs      = sqrt( (Te_eV + Ti_eV) * qe / mi_kg )
// //        rho_i   = cs / (qe*B_T/mi_kg)
// //        L_MPS   = rho_i
// //        fd(α)   = polynomial (α in degrees, clamped to [0,1])
// //        pot     = pot_mult * Te_eV   (default 3*Te)
// // -----------------------------------------------------------------------------

// #include <cmath>
// #include <algorithm>

// namespace sheath {

// // physical constants (SI)
// inline constexpr double QE   = 1.602176634e-19;     // C
// inline constexpr double EPS0 = 8.8541878128e-12;    // F/m
// inline constexpr double MP   = 1.67262192369e-27;   // kg

// struct Out {
//   double phi;     // V  (negative near wall)
//   double E_mag;   // V/m (magnitude along -n)
//   double ne;      // m^-3
// };

// // angle polynomial fd(α in deg) from GITRm
// inline double fd_poly_deg(double a_deg) {
//   const double a = a_deg;
//   double fd =
//       0.98992
//     + 5.1220e-3  * a
//     - 7.0040e-4  * a*a
//     + 3.3591e-5  * a*a*a
//     - 8.2917e-7  * std::pow(a,4)
//     + 9.5856e-9  * std::pow(a,5)
//     - 4.2682e-11 * std::pow(a,6);
//   return std::min(1.0, std::max(0.0, fd));
// }

// // core evaluator
// inline Out eval_ds_mps(double d_m,        // distance from wall (m), d >= 0
//                        double Te_eV,      // electron temperature (eV)
//                        double Ti_eV,      // ion temperature (eV)
//                        double ne0_m3,     // upstream electron density (m^-3)
//                        double B_T,        // magnetic field magnitude (T)
//                        double alpha_deg,  // grazing angle in degrees
//                        double mi_kg = 2.0*MP, // ion mass (kg), default D+
//                        double pot_mult = 3.0) // pot = pot_mult * Te_eV
// {
//   Out out{0.0, 0.0, ne0_m3};

//   // basic guards
//   if (d_m < 0.0) d_m = 0.0;
//   if (Te_eV <= 0.0 || ne0_m3 <= 0.0 || mi_kg <= 0.0) return out;

//   const double fd = fd_poly_deg(alpha_deg);
//   const double pot = pot_mult * Te_eV; // volts (1 eV ≡ 1 V)

//   // Debye length (Te in eV): lambdaD = sqrt( EPS0 * Te / (ne0 * QE) )
//   const double lambdaD = std::sqrt(EPS0 * Te_eV / (ne0_m3 * QE));

//   // ion-sound speed and ion gyro radius
//   const double cs       = std::sqrt(std::max(Te_eV + Ti_eV, 0.0) * QE / mi_kg);
//   const double omega_ci = (B_T > 0.0) ? (QE * B_T / mi_kg) : 0.0;
//   const double rho_i    = (omega_ci > 0.0) ? (cs / omega_ci) : 1e300; // large if B=0
//   const double L_MPS    = rho_i;

//   // exponentials (avoid overflow)
//   const double e_DS  = std::exp(- d_m / (2.0 * lambdaD));
//   const double e_MPS = std::exp(- d_m / std::max(L_MPS, 1e-300));

//   // potential (negative toward the wall)
//   const double phi = - pot * ( fd * e_DS + (1.0 - fd) * e_MPS );

//   // field magnitude (positive)
//   const double E_DS  =  pot * ( fd / (2.0 * lambdaD) ) * e_DS;
//   const double E_MPS =  pot * ( (1.0 - fd) / std::max(L_MPS, 1e-300) ) * e_MPS;
//   const double E_mag =  std::abs(E_DS + E_MPS);

//   // Boltzmann electrons (clamp exponent for safety)
//   double x = phi / std::max(Te_eV, 1e-300);
//   x = std::max(-100.0, std::min(50.0, x));
//   const double ne = ne0_m3 * std::exp(x);

//   out.phi   = phi;
//   out.E_mag = E_mag;
//   out.ne    = ne;
//   return out;
// }

// // convenience: project E along a supplied outward normal (length need not be 1)
// inline void field_vector_along_minus_n(const Out& o,
//                                        const double n[3], // surface normal
//                                        double& Ex, double& Ey, double& Ez)
// {
//   // normalize n
//   const double nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
//   if (nn <= 0.0) { Ex = Ey = Ez = 0.0; return; }
//   const double nx = n[0]/nn, ny = n[1]/nn, nz = n[2]/nn;

//   // accelerate ions toward the wall: E = -|E| * n_hat
//   Ex = - o.E_mag * nx;
//   Ey = - o.E_mag * ny;
//   Ez = - o.E_mag * nz;
// }

// } // namespace sheath
