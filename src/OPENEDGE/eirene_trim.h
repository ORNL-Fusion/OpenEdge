/* ----------------------------------------------------------------------
   OpenEdge: EIRENE TRIM reflection tables - host/device sampling kernels.

   Port of EIRENE's quantile-based reflection model (see
   modules/Eirene/.../eirmod_reflec.f, subroutine EIRENE_REFLEC, lines
   614-870).  Each (projectile, target) combination is represented by a
   stack of arrays on an (E_in, theta_in) incident grid:

       R_N(E, theta)                           particle reflection prob.
       E_out   as 5 quantiles of the conditional distribution
       cos(polar_out)  as 5 quantiles, conditioned on E_out bin
       cos(azim_out)   as 5 quantiles, conditioned on (E_out, polar) bin

   Quantile levels are RAAR = [0.1, 0.3, 0.5, 0.7, 0.9] - the midpoints of
   five equal-width probability bins, identical to EIRENE (rdtrim.f line 146).

   Sampling (matches eirmod_reflec.f):
       1) u0 ~ U[0,1], compare to bilinear-interp R_N(E, theta)
       2) u1 ~ U[0,1], quantile-invert + trilinear interp -> E_out
       3) u2 ~ U[0,1], quantile-invert + 4-linear interp -> cos(polar)
       4) u3 ~ U[0,1], quantile-invert + 5-linear interp -> cos(azim)

   The caller supplies the 4 random numbers, so the CPU path can use its
   RanKnuth and the Kokkos path can use the Kokkos random pool; the math
   stays identical.
------------------------------------------------------------------------- */

#ifndef SPARTA_EIRENE_TRIM_H
#define SPARTA_EIRENE_TRIM_H

#include <cmath>
#include <cstddef>

#ifdef KOKKOS_INLINE_FUNCTION
#define TRIM_HD KOKKOS_INLINE_FUNCTION
#else
#define TRIM_HD inline
#endif

namespace SPARTA_NS {
namespace EireneTrim {

// Dimensions are fixed by the EIRENE TRIM format (rdtrim.f lines 51-53).
static constexpr int TRIM_NE = 12;
static constexpr int TRIM_NW = 7;
static constexpr int TRIM_NR = 5;

static constexpr double TRIM_PI = 3.14159265358979323846;

// Flat index helpers (row-major, matching how we pack the data).
TRIM_HD int idx2(int i, int j, int nj)             { return i * nj + j; }
TRIM_HD int idx3(int i, int j, int k, int nj, int nk)
{ return (i * nj + j) * nk + k; }
TRIM_HD int idx4(int i, int j, int k, int l, int nj, int nk, int nl)
{ return ((i * nj + j) * nk + k) * nl + l; }
TRIM_HD int idx5(int i, int j, int k, int l, int m,
                 int nj, int nk, int nl, int nm)
{ return (((i * nj + j) * nk + k) * nl + l) * nm + m; }

// Pointer view over one combination's tables.  Both the CPU path and the
// Kokkos device path build one of these and pass it to the sampler.
struct TrimView {
  const double *E;            // [nE]  incident energies [eV]
  const double *theta_deg;    // [nW]  incident angles   [deg]
  const double *raar;         // [nR]  quantile levels
  const double *R_N;          // [nE*nW]
  const double *Eout_q;       // [nE*nW*nR]
  const double *Eout_min;     // [nE*nW]
  const double *Eout_max;     // [nE*nW]
  const double *cos_polar_q;  // [nE*nW*nR*nR]
  const double *cos_azim_q;   // [nE*nW*nR*nR*nR]
};


// ------------- helpers: locate bin and fractional coord --------------

// Find largest index iE in [0, nE-2] with E_grid[iE] <= E (clamped to grid).
TRIM_HD void locate(const double *grid, int n, double x,
                    int &i_lo, int &i_hi, double &frac)
{
  if (n < 2) { i_lo = 0; i_hi = 0; frac = 0.0; return; }
  if (x <= grid[0])     { i_lo = 0;     i_hi = 1;     frac = 0.0; return; }
  if (x >= grid[n - 1]) { i_lo = n - 2; i_hi = n - 1; frac = 1.0; return; }
  int lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (grid[mid] <= x) lo = mid; else hi = mid;
  }
  i_lo = lo; i_hi = lo + 1;
  double dx = grid[i_hi] - grid[i_lo];
  frac = (dx > 0.0) ? (x - grid[i_lo]) / dx : 0.0;
}

// Quantile locator: given quantile levels RAAR[nR] = [0.1,0.3,0.5,0.7,0.9]
// and random number u, find bracketing indices and fractional position.
// Matches eirmod_reflec.f lines 676-683 (uses RAAR segments with linear
// extrapolation past the ends).
TRIM_HD void locate_quantile(const double *raar, int nR, double u,
                             int &i_lo, int &i_hi, double &frac)
{
  if (nR < 2) { i_lo = 0; i_hi = 0; frac = 0.0; return; }
  // find first i with raar[i] >= u; clamp to range
  int i = 1;
  while (i < nR - 1 && raar[i] < u) i++;
  // EIRENE keeps i_hi >= 1, so extrapolates past the 0.1 and 0.9 tails
  i_hi = i;
  i_lo = i_hi - 1;
  double d = raar[i_hi] - raar[i_lo];
  frac = (d > 0.0) ? (u - raar[i_lo]) / d : 0.0;
}


// ------------- R_N(E, theta) bilinear lookup --------------

TRIM_HD double R_N_interp(const TrimView &t, double E_eV, double theta_deg)
{
  int iE_lo, iE_hi, iW_lo, iW_hi;
  double fE, fW;
  locate(t.E,         TRIM_NE, E_eV,      iE_lo, iE_hi, fE);
  locate(t.theta_deg, TRIM_NW, theta_deg, iW_lo, iW_hi, fW);

  double r00 = t.R_N[idx2(iE_lo, iW_lo, TRIM_NW)];
  double r01 = t.R_N[idx2(iE_lo, iW_hi, TRIM_NW)];
  double r10 = t.R_N[idx2(iE_hi, iW_lo, TRIM_NW)];
  double r11 = t.R_N[idx2(iE_hi, iW_hi, TRIM_NW)];

  double r0 = r00 + fE * (r10 - r00);
  double r1 = r01 + fE * (r11 - r01);
  double r  = r0  + fW * (r1  - r0);

  if (r < 0.0) r = 0.0;
  if (r > 1.0) r = 1.0;
  return r;
}


// ------------- conditional E_out sampler (HFTR1) --------------

TRIM_HD double sample_E_out(const TrimView &t,
                            int iE_lo, int iE_hi, double fE,
                            int iW_lo, int iW_hi, double fW,
                            double u1,
                            int &iQ1_lo, int &iQ1_hi, double &fQ1)
{
  locate_quantile(t.raar, TRIM_NR, u1, iQ1_lo, iQ1_hi, fQ1);

  // quantile iQ1_lo
  double e00 = t.Eout_q[idx3(iE_lo, iW_lo, iQ1_lo, TRIM_NW, TRIM_NR)];
  double e10 = t.Eout_q[idx3(iE_hi, iW_lo, iQ1_lo, TRIM_NW, TRIM_NR)];
  double e01 = t.Eout_q[idx3(iE_lo, iW_hi, iQ1_lo, TRIM_NW, TRIM_NR)];
  double e11 = t.Eout_q[idx3(iE_hi, iW_hi, iQ1_lo, TRIM_NW, TRIM_NR)];
  double rff1 = (e00 + fE * (e10 - e00))
              + fW * ((e01 + fE * (e11 - e01)) - (e00 + fE * (e10 - e00)));

  // quantile iQ1_hi
  double f00 = t.Eout_q[idx3(iE_lo, iW_lo, iQ1_hi, TRIM_NW, TRIM_NR)];
  double f10 = t.Eout_q[idx3(iE_hi, iW_lo, iQ1_hi, TRIM_NW, TRIM_NR)];
  double f01 = t.Eout_q[idx3(iE_lo, iW_hi, iQ1_hi, TRIM_NW, TRIM_NR)];
  double f11 = t.Eout_q[idx3(iE_hi, iW_hi, iQ1_hi, TRIM_NW, TRIM_NR)];
  double rff2 = (f00 + fE * (f10 - f00))
              + fW * ((f01 + fE * (f11 - f01)) - (f00 + fE * (f10 - f00)));

  double E_out = rff1 + fQ1 * (rff2 - rff1);

  // clamp to Eout_min/Eout_max at the (iE_lo, iW_lo) bin (same as EIRENE)
  double emin = t.Eout_min[idx2(iE_lo, iW_lo, TRIM_NW)];
  double emax = t.Eout_max[idx2(iE_lo, iW_lo, TRIM_NW)];
  if (emax <= 0.0) emax = 1.0e30;
  if (E_out < emin) E_out = emin;
  if (E_out > emax) E_out = emax;
  if (E_out < 0.0)  E_out = 0.0;
  return E_out;
}


// ------------- conditional cos(polar) sampler (HFTR2) --------------

TRIM_HD double sample_cos_polar(const TrimView &t,
                                int iE_lo, int iE_hi, double fE,
                                int iW_lo, int iW_hi, double fW,
                                int iQ1_lo, int iQ1_hi, double fQ1,
                                double u2,
                                int &iQ2_lo, int &iQ2_hi, double &fQ2)
{
  locate_quantile(t.raar, TRIM_NR, u2, iQ2_lo, iQ2_hi, fQ2);

  // 4-linear interp: (E, theta, Q1, Q2); 16 corners.
  double c = 0.0;
  for (int di = 0; di < 2; ++di) {
    int ie = (di == 0) ? iE_lo : iE_hi;
    double we = (di == 0) ? (1.0 - fE) : fE;
    for (int dj = 0; dj < 2; ++dj) {
      int iw = (dj == 0) ? iW_lo : iW_hi;
      double ww = (dj == 0) ? (1.0 - fW) : fW;
      for (int dk = 0; dk < 2; ++dk) {
        int iq1 = (dk == 0) ? iQ1_lo : iQ1_hi;
        double wq1 = (dk == 0) ? (1.0 - fQ1) : fQ1;
        for (int dl = 0; dl < 2; ++dl) {
          int iq2 = (dl == 0) ? iQ2_lo : iQ2_hi;
          double wq2 = (dl == 0) ? (1.0 - fQ2) : fQ2;
          double val = t.cos_polar_q[idx4(ie, iw, iq1, iq2,
                                          TRIM_NW, TRIM_NR, TRIM_NR)];
          c += we * ww * wq1 * wq2 * val;
        }
      }
    }
  }
  // Clamp as EIRENE does (eirmod_reflec.f line 812): max 85 deg from normal.
  if (c < 0.08716)   c = 0.08716;
  if (c > 0.999999)  c = 0.999999;
  return c;
}


// ------------- conditional cos(azim) sampler (HFTR3) --------------

TRIM_HD double sample_cos_azim(const TrimView &t,
                               int iE_lo, int iE_hi, double fE,
                               int iW_lo, int iW_hi, double fW,
                               int iQ1_lo, int iQ1_hi, double fQ1,
                               int iQ2_lo, int iQ2_hi, double fQ2,
                               double u3)
{
  int iQ3_lo, iQ3_hi;
  double fQ3;
  locate_quantile(t.raar, TRIM_NR, u3, iQ3_lo, iQ3_hi, fQ3);

  double c = 0.0;
  for (int di = 0; di < 2; ++di) {
    int ie = (di == 0) ? iE_lo : iE_hi;
    double we = (di == 0) ? (1.0 - fE) : fE;
    for (int dj = 0; dj < 2; ++dj) {
      int iw = (dj == 0) ? iW_lo : iW_hi;
      double ww = (dj == 0) ? (1.0 - fW) : fW;
      for (int dk = 0; dk < 2; ++dk) {
        int iq1 = (dk == 0) ? iQ1_lo : iQ1_hi;
        double wq1 = (dk == 0) ? (1.0 - fQ1) : fQ1;
        for (int dl = 0; dl < 2; ++dl) {
          int iq2 = (dl == 0) ? iQ2_lo : iQ2_hi;
          double wq2 = (dl == 0) ? (1.0 - fQ2) : fQ2;
          for (int dm = 0; dm < 2; ++dm) {
            int iq3 = (dm == 0) ? iQ3_lo : iQ3_hi;
            double wq3 = (dm == 0) ? (1.0 - fQ3) : fQ3;
            double val = t.cos_azim_q[idx5(ie, iw, iq1, iq2, iq3,
                                           TRIM_NW, TRIM_NR, TRIM_NR, TRIM_NR)];
            c += we * ww * wq1 * wq2 * wq3 * val;
          }
        }
      }
    }
  }
  if (c < -1.0) c = -1.0;
  if (c >  1.0) c =  1.0;
  return c;
}


// ------------- top-level: draw a reflection event --------------
//
// Given incident (E, theta_deg) and 3 independent uniform random draws
// u1,u2,u3, write out the reflected energy and outgoing direction in
// (cos_polar, cos_azim) form.  cos_polar is relative to the outward
// surface normal; cos_azim is the cosine of the azimuthal angle relative
// to the plane containing incident velocity and normal.  Caller must
// already have decided this is a reflection (e.g. via R_N_interp).
TRIM_HD void sample_reflection(const TrimView &t,
                               double E_in, double theta_in_deg,
                               double u1, double u2, double u3,
                               double *E_out, double *cos_polar,
                               double *cos_azim)
{
  int iE_lo, iE_hi, iW_lo, iW_hi;
  double fE, fW;
  locate(t.E,         TRIM_NE, E_in,         iE_lo, iE_hi, fE);
  locate(t.theta_deg, TRIM_NW, theta_in_deg, iW_lo, iW_hi, fW);

  int iQ1_lo, iQ1_hi, iQ2_lo, iQ2_hi;
  double fQ1, fQ2;

  *E_out     = sample_E_out(t, iE_lo, iE_hi, fE, iW_lo, iW_hi, fW,
                            u1, iQ1_lo, iQ1_hi, fQ1);
  *cos_polar = sample_cos_polar(t, iE_lo, iE_hi, fE, iW_lo, iW_hi, fW,
                                iQ1_lo, iQ1_hi, fQ1,
                                u2, iQ2_lo, iQ2_hi, fQ2);
  *cos_azim  = sample_cos_azim(t, iE_lo, iE_hi, fE, iW_lo, iW_hi, fW,
                               iQ1_lo, iQ1_hi, fQ1,
                               iQ2_lo, iQ2_hi, fQ2,
                               u3);
}

}  // namespace EireneTrim
}  // namespace SPARTA_NS

#undef TRIM_HD

#endif
