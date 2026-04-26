/* ----------------------------------------------------------------------
   OpenEdge: EIRENE-equivalent analytic sputter formulas and shorthand
   lookup, used by compute surface/physical/sputter (CPU and Kokkos).

   Host/device header shared by the CPU and GPU react() paths.  All
   math is marked with a HOST_DEVICE macro that expands to
   KOKKOS_INLINE_FUNCTION when Kokkos is present or plain `inline`
   otherwise - the SAME code runs in both.

   ------------------------------------------------------------------
   Sputter model
   ------------------------------------------------------------------
   Formula and parameters match EIRENE verbatim - this is the same
   physical-sputter path that SOLPS-ITER / SOLEDGE3X use (modpys=2 in
   eirmod_sputer.f):

       Y(E, 0) = Q * (1 - (Eth/E)^(2/3)) * (1 - Eth/E)^2
                   * s_n^{KrC}(E / ETF)

       Y(E, theta) = Y(E, 0) * cos(theta)^(-f)
                             * exp(f * (1 - 1/cos(theta)) * cos(theta_opt))

       s_n^{KrC}(eps) = 0.5 * ln(1 + 1.2288*eps)
                        -------------------------------------------------
                        eps + 0.1728*sqrt(eps) + 0.008*eps^0.1504

   with f = 2.0 and cos(theta_opt) = 0.26 hard-coded for every
   (projectile, target) pair, matching eirmod_sputer.f lines 660-670.

   The fit parameters (Q, Eth, ETF, Es) for 286 (projectile, target)
   pairs are in eckstein_sputter_data.h and are parsed directly from
   EIRENE's SPUTER database file:

       W. Eckstein, C. Garcia-Rosales, J. Roth, W. Ottenberger,
       "Sputtering Data", IPP 9/82, Feb 1993, Table 3, page 335.

   Regenerate with:
       python3 tools/sputter/parse_eirene_sputer.py

   ------------------------------------------------------------------
   Reflection model
   ------------------------------------------------------------------
   EIRENE does NOT use an analytic fit for reflection - it reads per-
   combination TRIM tables (12 energies x 7 angles of R_N plus quantile
   distributions, see eirmod_sputer.f / rdtrim.f).  Mapping those
   tables into a simple analytic form is not trivial and would diverge
   from EIRENE.  For now this header provides a small placeholder fit
   (Thomas/Eckstein universal form) so the existing pmi "R" branch has
   numbers to work with; by default R_N for W->W is tiny (a few %) and
   very weakly E-dependent, which is already a good approximation for
   test_west_axi.  Follow-up work can port the real EIRENE TRIM tables.
------------------------------------------------------------------------- */

#ifndef SPARTA_ECKSTEIN_SPUTTER_H
#define SPARTA_ECKSTEIN_SPUTTER_H

#include "eckstein_sputter_data.h"

#include <cmath>
#include <cstring>

#ifdef KOKKOS_INLINE_FUNCTION
#define ECK_HOST_DEVICE KOKKOS_INLINE_FUNCTION
#else
#define ECK_HOST_DEVICE inline
#endif

namespace SPARTA_NS {
namespace Eckstein {

// ---- constants ----
static constexpr double ECK_PI = 3.14159265358979323846;

// Yamamura angular parameters (eirmod_sputer.f lines 665-666).
static constexpr double YAMAMURA_F       = 2.0;
static constexpr double YAMAMURA_COS_OPT = 0.26;

// Reflection model: keep the simple universal form, as "reflection
// from EIRENE" is a tables problem we'll tackle separately.  See
// note at top of file.
static constexpr int NCOEFF_ECKSTEIN = 12;


// ---- parameter structs ----

// Sputter parameters pulled from EIRENE's SPUTER file.  Layout matches
// eckstein_sputter_data.h SputterDataEntry.
struct SputterParams {
  double Z1, M1, Z2, M2;
  double Es;       // target surface binding energy [eV]
  double Eth;      // threshold energy [eV]
  double Q;        // Bohdansky yield prefactor [atoms/ion]
  double ETF;      // Thomas-Fermi reference energy [eV]
};

// Reflection parameters: thin placeholder for a Thomas-style fit.
// Replaced in a follow-up when we import EIRENE TRIM tables.
struct ReflectParams {
  double Z1, M1, Z2, M2;
  double a1, a2, a3, a4, a5, a6;
  double re_frac;  // mean <E_out>/<E_in> fraction for reflected particles
  double c_ang;    // cos(theta)^(-c_ang) angular factor
};


// ---- sputter: EIRENE Bohdansky-1993 + Yamamura angular ----

ECK_HOST_DEVICE
double sn_KrC(double eps)
{
  if (eps <= 0.0) return 0.0;
  double num = 0.5 * log(1.0 + 1.2288 * eps);
  double den = eps + 0.1728 * sqrt(eps) + 0.008 * pow(eps, 0.1504);
  return num / den;
}

// eirmod_sputer.f lines 637-658: Bohdansky-1993 (IPP 9/82) yield at
// normal incidence.  eps = E0/ETF is the reduced energy in Thomas-Fermi
// units, not Lindhard units.
ECK_HOST_DEVICE
double sputter_yield_normal(double E_eV, const SputterParams &p)
{
  if (E_eV <= p.Eth) return 0.0;
  double ethe0 = p.Eth / E_eV;
  double onemx = 1.0 - ethe0;
  double f1    = p.Q * (1.0 - pow(ethe0, 2.0 / 3.0)) * onemx * onemx;
  double eps   = E_eV / p.ETF;
  double se    = sn_KrC(eps);
  double y     = f1 * se;
  return (y > 0.0) ? y : 0.0;
}

// eirmod_sputer.f lines 659-670: Yamamura angular factor, f=2,
// cos(theta_opt)=0.26, same for every combination.
ECK_HOST_DEVICE
double yamamura_angular_factor(double theta_deg)
{
  double th = theta_deg;
  if (th < 0.0)  th = 0.0;
  if (th > 89.5) th = 89.5;
  double c = cos(th * ECK_PI / 180.0);
  if (c < 1e-3) c = 1e-3;
  double arg = -YAMAMURA_F * log(c)
             + YAMAMURA_F * (1.0 - 1.0 / c) * YAMAMURA_COS_OPT;
  if (arg < -500.0) arg = -500.0;   // same clamp as EIRENE
  return exp(arg);
}

ECK_HOST_DEVICE
double sputter_yield(double E_eV, double theta_deg, const SputterParams &p)
{
  double Y0 = sputter_yield_normal(E_eV, p);
  if (Y0 <= 0.0) return 0.0;
  double ang = yamamura_angular_factor(theta_deg);
  double y = Y0 * ang;
  return (y > 0.0) ? y : 0.0;
}


// ---- reflection (placeholder analytic) ----

ECK_HOST_DEVICE
double lindhard_a_L(double Z1, double Z2)
{
  double z = pow(Z1, 2.0/3.0) + pow(Z2, 2.0/3.0);
  return 0.8853 * 0.5292 / sqrt(z);
}

ECK_HOST_DEVICE
double reduced_energy_L(double E_eV, double Z1, double Z2,
                        double M1, double M2)
{
  double aL = lindhard_a_L(Z1, Z2);
  return E_eV * (M2 / (M1 + M2)) * aL / (Z1 * Z2 * 14.3996);
}

ECK_HOST_DEVICE
double reflection_RN_normal(double E_eV, const ReflectParams &p)
{
  double eps = reduced_energy_L(E_eV, p.Z1, p.Z2, p.M1, p.M2);
  if (eps <= 0.0) return 0.0;
  double num = p.a1 * log(p.a2 * eps + 2.71828182845904523536);
  double den = 1.0 + p.a3 * pow(eps, p.a4) + p.a5 * pow(eps, p.a6);
  double r = num / den;
  if (r < 0.0) r = 0.0;
  if (r > 1.0) r = 1.0;
  return r;
}

ECK_HOST_DEVICE
double reflection_RN(double E_eV, double theta_deg, const ReflectParams &p)
{
  double R0 = reflection_RN_normal(E_eV, p);
  if (R0 <= 0.0) return 0.0;
  double th = theta_deg;
  if (th < 0.0)  th = 0.0;
  if (th > 89.5) th = 89.5;
  double c = cos(th * ECK_PI / 180.0);
  if (c < 1e-3) c = 1e-3;
  double r = R0 * pow(c, -p.c_ang);
  if (r > 1.0) r = 1.0;
  return r;
}


// ---- shorthand lookups ----

inline bool lookup_sputter(const char *name, SputterParams &out)
{
  for (int i = 0; i < N_EIRENE_SPUTTER; ++i) {
    if (std::strcmp(name, EIRENE_SPUTTER_TABLE[i].name) == 0) {
      const SputterDataEntry &e = EIRENE_SPUTTER_TABLE[i];
      out.Z1 = e.Z1; out.M1 = e.M1; out.Z2 = e.Z2; out.M2 = e.M2;
      out.Es = e.Es; out.Eth = e.Eth; out.Q = e.Q; out.ETF = e.ETF;
      return true;
    }
  }
  return false;
}

// W -> W reflection placeholder: small, nearly E-independent.  These
// values are intentionally conservative pending a real TRIM-table port.
// They're hardcoded only for W->W since that is the test_west_axi case
// the user asked to ship.  Any other combination will require the
// follow-up TRIM table import.
inline bool lookup_reflect(const char *name, ReflectParams &out)
{
  if (std::strcmp(name, "W_on_W") == 0) {
    out.Z1 = 74.0; out.M1 = 183.84; out.Z2 = 74.0; out.M2 = 183.84;
    out.a1 = 0.02; out.a2 = 1.0; out.a3 = 0.0; out.a4 = 1.0;
    out.a5 = 0.0;  out.a6 = 1.0;
    out.re_frac = 0.3;
    out.c_ang   = 0.5;
    return true;
  }
  return false;
}


// ---- pack/unpack into OneReaction::coeff[NCOEFF_ECKSTEIN] ----

inline void pack_sputter(const SputterParams &p, double *c)
{
  c[0]  = p.Z1;  c[1]  = p.M1;   c[2]  = p.Z2;  c[3]  = p.M2;
  c[4]  = p.Es;  c[5]  = p.Eth;  c[6]  = p.Q;   c[7]  = p.ETF;
  c[8]  = 0.0;   c[9]  = 0.0;    c[10] = 0.0;   c[11] = 0.0;
}

ECK_HOST_DEVICE
void unpack_sputter(const double *c, SputterParams &p)
{
  p.Z1  = c[0];  p.M1  = c[1];   p.Z2 = c[2];  p.M2 = c[3];
  p.Es  = c[4];  p.Eth = c[5];   p.Q  = c[6];  p.ETF = c[7];
}

inline void pack_reflect(const ReflectParams &p, double *c)
{
  c[0]  = p.Z1;  c[1]  = p.M1;   c[2]  = p.Z2;  c[3]  = p.M2;
  c[4]  = p.a1;  c[5]  = p.a2;   c[6]  = p.a3;  c[7]  = p.a4;
  c[8]  = p.a5;  c[9]  = p.a6;
  c[10] = p.re_frac;
  c[11] = p.c_ang;
}

ECK_HOST_DEVICE
void unpack_reflect(const double *c, ReflectParams &p)
{
  p.Z1 = c[0];  p.M1 = c[1];  p.Z2 = c[2];  p.M2 = c[3];
  p.a1 = c[4];  p.a2 = c[5];  p.a3 = c[6];  p.a4 = c[7];
  p.a5 = c[8];  p.a6 = c[9];
  p.re_frac = c[10];
  p.c_ang   = c[11];
}


}  // namespace Eckstein
}  // namespace SPARTA_NS

#undef ECK_HOST_DEVICE

#endif
