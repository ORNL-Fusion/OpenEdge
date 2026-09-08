/* ----------------------------------------------------------------------
   OpenEdge - Plasma-edge particle transport code
   https://github.com/ORNL-Fusion/OpenEdge

   Liquid metal MHD film strip solver.
   Ported from Sergey Smolentsev's Fortran code (main.for / MYGTRI.FOR).

   Solves coupled shallow-water MHD + heat transfer for a liquid Li film
   on an inclined divertor surface:
     1) Momentum:      DU/Dt + U*DU/Dx + V*DU/Dy = gravity - MHD_drag*U
     2) Continuity:    DU/Dx + DV/Dy = 0
     3) Free surface:  Dh/Dt + Us*Dh/Dx = Vs - V_evap
     4) Heat transfer: DT/Dt + U*DT/Dx + V*DT/Dy = (1/Re/Pr)*D2T/Dy2
     5) Evaporation:   Antoine vapor pressure + Hertz-Knudsen flux

   Reference: Smolentsev et al., Nucl. Fusion (2021).
------------------------------------------------------------------------- */

#ifndef LIQUID_METAL_STRIP_H
#define LIQUID_METAL_STRIP_H

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace LiquidMetal {

// ---------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------

static const double KB  = 1.380649e-23;       // Boltzmann [J/K]
static const double MLI = 1.1526e-26;         // Li-7 atom mass [kg]
static const double ATM_TO_PA = 101325.0;     // 1 atm in Pa
static const double NA  = 6.02214076e23;      // Avogadro
static const double EV  = 1.602176634e-19;    // J per eV
static const double PI  = 3.14159265358979323846;

// ---------------------------------------------------------------
// Antoine vapor pressure fit for Li
// log10(P_atm) = ANTOINE_A - ANTOINE_B / T_K
// Fitted from NIST thermodynamic data (298-1600 K range)
// ---------------------------------------------------------------

static const double ANTOINE_A = 5.66797;
static const double ANTOINE_B = 8310.41;

// ---------------------------------------------------------------
// Li evaporation flux: Antoine + Hertz-Knudsen
//   P_vapor = 10^(A - B/T_K) * 101325  [Pa]
//   Gamma_evap = alpha * P / sqrt(2*pi*m*kB*T)  [atoms/m²/s]
// ---------------------------------------------------------------

double li_evap_flux(double T_C, double sticking = 1.0);

// ---------------------------------------------------------------
// Evaporative cooling power [W/m²] from evaporation flux
//   Q_vapor = H_vap [J/mol] * FLUX [atoms/m²/s] / N_A
// ---------------------------------------------------------------

double li_evap_cooling(double flux, double H_vap = 145920.0);

// ---------------------------------------------------------------
// Li ad-atom flux
//   Gamma_ad = f_ad * (Yad/Yps) / (1 + A*exp(E_eff / E_surf))
//              * Yad_D_Li * Gamma_D+
//
//   T_C:       surface temperature [°C]
//   Gamma_Dp:  incident D+ flux [ions/m²/s]
//   Yad_D_Li:  ad-atom yield for D on Li (default 1e-3)
//   Yad_Yps:   ratio Yad/Yps (default 1.0)
//   f_ad:      neutral fraction of ad-atoms (default 1.0)
//   A_arr:     Arrhenius pre-factor (default 1e-7)
//   E_eff:     effective binding energy [eV] (default 0.9)
// ---------------------------------------------------------------

double li_adatom_flux(double T_C, double Gamma_Dp,
                       double Yad_D_Li = 1e-3,
                       double Yad_Yps = 1.0,
                       double f_ad = 1.0,
                       double A_arr = 1e-7,
                       double E_eff = 0.9);

// Li physical properties (default values at ~350 C)

struct LiProperties {
  double vis;       // kinematic viscosity [m^2/s]
  double rho;       // density [kg/m^3]
  double sigma_e;   // electrical conductivity [S/m]
  double Cp;        // specific heat [J/kg-K]
  double k_th;      // thermal conductivity [W/m-K]
  double VM;        // Li atom mass [kg]
  double H_vap;     // latent heat of vaporization [J/mol]

  LiProperties() :
    vis(8.5e-7), rho(485.0), sigma_e(3.09e6),
    Cp(4200.0), k_th(49.6),
    VM(1.17e-26), H_vap(145920.0) {}
};

// ------------------------------------------------------------------
// Thomas algorithm (tridiagonal solver)
// Ported from MYGTRI.FOR by Sergey Smolentsev.
//
// Solves A(m)*W(m+1) + B(m)*W(m) + C(m)*W(m-1) = D(m) for m in [n1,n2]
// with boundary conditions:
//   L=1: Dirichlet W = val
//   L=2: Neumann   dW/dy = Q/(2*hy)   (second-order)
//   L=3: Robin     dW/dy + P*W = Q/(2*hy)
//
// Left BC:  type L1, params A1, Q1
// Right BC: type LM, params AM, QM
// ------------------------------------------------------------------

void thomas_solve(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N1, int N2, double *W,
    int L1, double A1, double Q1,
    int LM, double AM, double QM);

// simplified wrapper: N1=1, results in W[1..N2]
void thomas_solve_simple(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N, double *W,
    int L1, double A1, double Q1,
    int LM, double AM, double QM);

// ------------------------------------------------------------------
// Strip solver: owns 1D mesh + state arrays, solves to steady state
// ------------------------------------------------------------------

struct Strip {
  // grid
  int Nx, Ny;
  double hx, hy;         // dimensionless grid spacing

  // 1D arrays [1..Nx] (0-indexed but we use 1-based internally)
  std::vector<double> X;       // dimensionless x-coordinates
  std::vector<double> Ho, Hn;  // film thickness (old, new)
  std::vector<double> Qs, Qs0; // surface heat flux (current, initial) [dimensionless]

  // 2D arrays [n][m] stored as (Nx+1)*(Ny+1) with 1-based indexing
  std::vector<double> Uo, Vo, Un, Vn;  // velocity fields
  std::vector<double> T1, T2;          // temperature fields (old, new)

  // output arrays [1..Nx] dimensional
  std::vector<double> Tsurf_dim;   // surface temperature [C]
  std::vector<double> evap_flux;   // evaporation flux [atoms/m^2-s]
  std::vector<double> Q_net;       // net heat flux [W/m^2]
  std::vector<double> h_dim;       // film thickness [m]

  // physical / geometry parameters
  double h0, U0, Bs, Bw;
  double alpha_deg, width;
  double Tin;              // inlet temperature [C]
  LiProperties li;

  // wall conductance
  double sigma_w, tw;

  // dimensionless groups (computed in init)
  double Re, Fr, Ha_s, Ha_w, Be, Cw, Rtor, Pr, Tscale;
  double Al;               // inclination angle [rad]
  double Xl;               // dimensionless length
  double qss;              // heat flux scale [W/m^2]

  // solver parameters
  double dt_pseudo;        // pseudo-time step (dimensionless)
  int max_iter;            // max pseudo-time iterations
  double eps_conv;         // convergence criterion
  double relax;            // relaxation parameter
  int ncase;               // 1=sidewall, 2=axisymmetric

  // evaporation toggle
  int evap_on;

  // divergence diagnostics — set by solve_steady() if numerical guards trip.
  // Callers (SPARTA fix or standalone driver) should inspect both fields
  // and abort or fall back rather than trust output silently.
  bool diverged;                   // true if any guard tripped
  std::string diverged_reason;     // human-readable message
  double Tdim_max_seen;            // peak |T_dimless| reached during the run
  int    Qs_clamped_count;         // # of times Qs hit the evap-cooling clamp

  // Thresholds for the runtime guards (override before solve_steady()).
  // Defaults are calibrated so the validated ITER-scale baseline runs
  // clean and the thin-film/slow-flow runaway gets caught.
  double Tdim_diverge;             // |T_dimless| above this -> diverged (default 50)
  double Qs_clamp_factor;          // |Qs| capped at this * |Qs0| (default 5)
  int    diverge_check_every;      // check every N pseudo-time iters (default 25)

  // inline 2D indexing
  inline int idx(int n, int m) const { return n * (Ny + 1) + m; }

  // ------------------------------------------------------------------
  // Initialize: compute dimensionless groups, allocate arrays
  // ------------------------------------------------------------------

  void init();

  // ------------------------------------------------------------------
  // Set surface heat flux from external source [W/m^2] at Nx stations
  // x_phys[1..Nx] = physical x-coordinates [m]
  // q_phys[1..Nx] = heat flux [W/m^2]
  // ------------------------------------------------------------------

  void set_heat_flux(const double *x_phys, const double *q_phys, int npts);

  // ------------------------------------------------------------------
  // Set uniform heat flux (for testing with Gaussian, etc.)
  // Xlength = physical length [m], q_func = dimensionless Qs0(x_dimless)
  // ------------------------------------------------------------------

  void set_heat_flux_uniform(double Xlength);

  // ------------------------------------------------------------------
  // Solve to steady state (port of main.for lines 272-401)
  // ------------------------------------------------------------------

  void solve_steady();

  // ------------------------------------------------------------------
  // Set defaults
  // ------------------------------------------------------------------

  Strip() :
    Nx(201), Ny(51),
    h0(0.005), U0(8.0), Bs(5.0), Bw(0.0),
    alpha_deg(43.0), width(1.67), Tin(350.0),
    sigma_w(0.0), tw(25e-6),
    qss(1.0e6),
    dt_pseudo(0.5), max_iter(3000), eps_conv(5e-8),
    relax(1.0), ncase(1), evap_on(1),
    diverged(false), Tdim_max_seen(0.0), Qs_clamped_count(0),
    Tdim_diverge(50.0), Qs_clamp_factor(5.0), diverge_check_every(25) {}
};

}  // namespace LiquidMetal

#endif
