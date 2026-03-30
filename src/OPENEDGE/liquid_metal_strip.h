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
     5) Evaporation:   FLUX = 7e-15*(T_surf - 55)^13.7

   Reference: Smolentsev et al., Nucl. Fusion (2021).
------------------------------------------------------------------------- */

#ifndef LIQUID_METAL_STRIP_H
#define LIQUID_METAL_STRIP_H

#include <cmath>
#include <cstring>
#include <vector>

namespace LiquidMetal {

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

inline void thomas_solve(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N1, int N2,
    double *W,
    int L1, double A1, double Q1,
    int LM, double AM, double QM)
{
  const int MD = N2 - N1 + 1;
  const int MM = MD - 1;

  // work arrays (stack-allocated for typical sizes <= 1001)
  double A[1024], B[1024], C[1024], D[1024], E[1024], F[1024];

  // copy interior coefficients (shift to 1-based indexing in local arrays)
  for (int m = 2; m <= MM; m++) {
    int nm = m + N1 - 1;
    A[m] = AH[nm];
    B[m] = BH[nm];
    C[m] = CH[nm];
    D[m] = DH[nm];
  }

  // left boundary
  if (L1 == 1) {
    E[1] = 0.0;
    F[1] = A1;
  } else {
    double Z = A1 - 3.0 + C[2] / A[2];
    E[1] = (B[2] / A[2] - 4.0) / Z;
    F[1] = (Q1 - D[2] / A[2]) / Z;
  }

  // forward sweep
  for (int m = 2; m <= MM; m++) {
    double den = B[m] - C[m] * E[m - 1];
    E[m] = A[m] / den;
    F[m] = (D[m] + C[m] * F[m - 1]) / den;
  }

  // right boundary
  if (LM == 1) {
    W[MD] = AM;
  } else {
    W[MD] = (QM - F[MD - 2] + F[MM] * (4.0 - E[MD - 2])) /
            (AM + 3.0 - E[MM] * (4.0 - E[MD - 2]));
  }

  // back substitution
  for (int mk = 1; mk <= MM; mk++) {
    int m = MD - mk;
    W[m] = E[m] * W[m + 1] + F[m];
  }

  // shift result back to original indexing [N1..N2]
  for (int m = N2; m >= N1; m--)
    W[m] = W[MD + N1 - (N2 + N1 - m)];
  // equivalent to Fortran: W(N2+N1-M) = W(MD+N1-M)
  // rewritten: copy W[MD..1] -> W[N2..N1]
}

// simplified wrapper: N1=1, results in W[1..N2]
inline void thomas_solve_simple(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N,
    double *W,
    int L1, double A1, double Q1,
    int LM, double AM, double QM)
{
  thomas_solve(AH, BH, CH, DH, 1, N, W, L1, A1, Q1, LM, AM, QM);
}

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

  // inline 2D indexing
  inline int idx(int n, int m) const { return n * (Ny + 1) + m; }

  // ------------------------------------------------------------------
  // Initialize: compute dimensionless groups, allocate arrays
  // ------------------------------------------------------------------

  void init() {
    // dimensionless groups
    Re = U0 * h0 / li.vis;
    Fr = U0 * U0 / (9.81 * h0);
    double b = 0.5 * width;
    Ha_s = Bs * b * std::sqrt(li.sigma_e / (li.vis * li.rho));
    Ha_w = Bw * h0 * std::sqrt(li.sigma_e / (li.vis * li.rho));
    Be = h0 / b;
    Cw = (sigma_w > 0.0) ? sigma_w * tw / (li.sigma_e * b) : 0.0;
    Pr = li.vis * li.rho * li.Cp / li.k_th;

    if (ncase == 1)
      Rtor = Be * Be * Ha_s / Re +
             Be * Be * (Cw / (1.0 + Cw)) * Ha_s * Ha_s / Re;
    else
      Rtor = 0.0;  // ncase=2 uses Ha_w in the solver loop

    Al = alpha_deg * M_PI / 180.0;
    double Xlength = 0.0;  // will be set from Qs0 or computed from Nx*hx

    // temperature scale
    Tscale = qss * h0 / li.k_th;

    // grid
    Xl = 0.0;  // set after heat flux is loaded
    hx = 0.0;
    hy = 1.0 / (Ny - 1);

    // allocate arrays (1-based, so size = max_index + 1)
    int sz1 = Nx + 2;
    int sz2 = (Nx + 2) * (Ny + 2);

    X.assign(sz1, 0.0);
    Ho.assign(sz1, 1.0);
    Hn.assign(sz1, 1.0);
    Qs.assign(sz1, 0.0);
    Qs0.assign(sz1, 0.0);

    Uo.assign(sz2, 1.0);
    Vo.assign(sz2, 0.0);
    Un.assign(sz2, 1.0);
    Vn.assign(sz2, 0.0);
    T1.assign(sz2, 0.0);
    T2.assign(sz2, 0.0);

    Tsurf_dim.assign(sz1, Tin);
    evap_flux.assign(sz1, 0.0);
    Q_net.assign(sz1, 0.0);
    h_dim.assign(sz1, h0);
  }

  // ------------------------------------------------------------------
  // Set surface heat flux from external source [W/m^2] at Nx stations
  // x_phys[1..Nx] = physical x-coordinates [m]
  // q_phys[1..Nx] = heat flux [W/m^2]
  // ------------------------------------------------------------------

  void set_heat_flux(const double *x_phys, const double *q_phys, int npts) {
    // determine physical length from x-coordinates
    double Xin = x_phys[1];
    double Xout = x_phys[npts];
    double Xlength = Xout - Xin;
    Xl = Xlength / h0;
    hx = Xl / (Nx - 1);

    // temperature scale
    Tscale = qss * h0 / li.k_th;

    // build x-mesh and interpolate heat flux
    for (int n = 1; n <= Nx; n++) {
      X[n] = (n - 1) * hx;
      double coord = Xin + X[n] * h0;

      // linear interpolation from input data
      double q_interp = q_phys[1];  // default to first value
      for (int k = 1; k < npts; k++) {
        if (coord >= x_phys[k] && coord <= x_phys[k + 1]) {
          double frac = (coord - x_phys[k]) / (x_phys[k + 1] - x_phys[k]);
          q_interp = q_phys[k] + frac * (q_phys[k + 1] - q_phys[k]);
          break;
        }
      }
      if (coord > x_phys[npts]) q_interp = q_phys[npts];

      Qs0[n] = q_interp / qss;  // dimensionless
      Qs[n] = Qs0[n];
    }
  }

  // ------------------------------------------------------------------
  // Set uniform heat flux (for testing with Gaussian, etc.)
  // Xlength = physical length [m], q_func = dimensionless Qs0(x_dimless)
  // ------------------------------------------------------------------

  void set_heat_flux_uniform(double Xlength) {
    Xl = Xlength / h0;
    hx = Xl / (Nx - 1);
    Tscale = qss * h0 / li.k_th;

    for (int n = 1; n <= Nx; n++) {
      X[n] = (n - 1) * hx;
      // Qs0 must be set externally before calling this
      Qs[n] = Qs0[n];
    }
  }

  // ------------------------------------------------------------------
  // Solve to steady state (port of main.for lines 272-401)
  // ------------------------------------------------------------------

  void solve_steady() {
    // work arrays for Thomas solver (1-based, size Ny+2)
    std::vector<double> A_th(Ny + 2), B_th(Ny + 2), C_th(Ny + 2), D_th(Ny + 2);
    std::vector<double> Wt(Ny + 2);

    // Y-mesh
    std::vector<double> Y(Ny + 2, 0.0);
    for (int m = 1; m <= Ny; m++)
      Y[m] = (m - 1) * hy;

    // initial conditions
    for (int n = 1; n <= Nx; n++) {
      Ho[n] = 1.0;
      Hn[n] = 1.0;
      for (int m = 1; m <= Ny; m++) {
        Uo[idx(n, m)] = 1.0;
        Vo[idx(n, m)] = 0.0;
        Un[idx(n, m)] = 1.0;
        Vn[idx(n, m)] = 0.0;
        T1[idx(n, m)] = 0.0;
        T2[idx(n, m)] = 0.0;
      }
    }
    for (int n = 1; n <= Nx; n++)
      Qs[n] = Qs0[n];

    double t = 0.0;
    int k = 0;

    // pseudo-time iteration
    while (true) {
      k++;
      t = (k - 1) * dt_pseudo;

      // march in x from n=2 to Nx
      for (int n = 2; n <= Nx; n++) {

        // surface temperature and evaporation
        double TMPRDIM = Tin + T2[idx(n, Ny)] * Tscale;
        double V1 = 0.0;
        if (evap_on && TMPRDIM > 55.0) {
          double FLUX = 7.0e-15 * std::pow(TMPRDIM - 55.0, 13.7);
          V1 = -FLUX * li.VM / li.rho / U0;
        }

        // film thickness (Eq. 3a)
        if (evap_on) {
          Hn[n] = (Ho[n] / dt_pseudo +
                   Uo[idx(n, Ny)] * Hn[n - 1] / hx +
                   Vo[idx(n, Ny)] + V1) /
                  (1.0 / dt_pseudo + Uo[idx(n, Ny)] / hx);
        } else {
          Hn[n] = (Ho[n] / dt_pseudo +
                   Uo[idx(n, Ny)] * Hn[n - 1] / hx +
                   Vo[idx(n, Ny)]) /
                  (1.0 / dt_pseudo + Uo[idx(n, Ny)] / hx);
        }

        double dhdt = (Hn[n] - Ho[n]) / dt_pseudo;
        double dhdx = (Hn[n] - Hn[n - 1]) / hx;

        // --- Solve for U (Thomas algorithm) ---
        for (int m = 2; m <= Ny - 1; m++) {
          double P = (Vo[idx(n, m)] -
                      Y[m] * (dhdt + Uo[idx(n, m)] * dhdx)) /
                     Hn[n];
          A_th[m] = -P / (2.0 * hy) + 1.0 / (Re * Hn[n] * Hn[n] * hy * hy);
          C_th[m] = P / (2.0 * hy) + 1.0 / (Re * Hn[n] * Hn[n] * hy * hy);

          if (ncase == 1)
            B_th[m] = 1.0 / dt_pseudo + A_th[m] + C_th[m] + Rtor +
                      Uo[idx(n, m)] / hx;
          else
            B_th[m] = 1.0 / dt_pseudo + A_th[m] + C_th[m] +
                      Ha_w * Ha_w / Re + Uo[idx(n, m)] / hx;

          D_th[m] = Uo[idx(n, m)] / dt_pseudo +
                    Uo[idx(n, m)] * Un[idx(n - 1, m)] / hx -
                    std::cos(Al) / Fr * dhdx +
                    std::sin(Al) / Fr;
        }

        thomas_solve(A_th.data(), B_th.data(), C_th.data(), D_th.data(),
                     1, Ny, Wt.data(),
                     1, 0.0, 0.0,   // left BC: U=0 at wall (Dirichlet)
                     2, 0.0, 0.0);  // right BC: dU/dy=0 at surface (Neumann)

        for (int m = 1; m <= Ny; m++)
          Un[idx(n, m)] = Wt[m];

        // --- Compute V from continuity ---
        for (int m = 2; m <= Ny - 1; m++) {
          Vn[idx(n, m)] = Vn[idx(n, m - 1)] -
                          Hn[n] * hy *
                              ((Un[idx(n, m)] - Un[idx(n - 1, m)]) / hx -
                               Y[m] / Hn[n] *
                                   (Un[idx(n, m + 1)] - Un[idx(n, m - 1)]) /
                                   (2.0 * hy) * dhdx);
        }
        Vn[idx(n, Ny)] = Vn[idx(n, Ny - 1)] -
                          Hn[n] * hy *
                              ((Un[idx(n, Ny)] - Un[idx(n - 1, Ny)]) / hx);

        // --- Solve for T (Thomas algorithm) ---
        for (int m = 2; m <= Ny - 1; m++) {
          double P = (Vo[idx(n, m)] -
                      Y[m] * (dhdt + Uo[idx(n, m)] * dhdx)) /
                     Hn[n];
          A_th[m] = -P / (2.0 * hy) +
                    1.0 / (Re * Pr * Hn[n] * Hn[n] * hy * hy);
          C_th[m] = P / (2.0 * hy) +
                    1.0 / (Re * Pr * Hn[n] * Hn[n] * hy * hy);
          B_th[m] = 1.0 / dt_pseudo + A_th[m] + C_th[m] +
                    Uo[idx(n, m)] / hx;
          D_th[m] = T1[idx(n, m)] / dt_pseudo +
                    Uo[idx(n, m)] * T2[idx(n - 1, m)] / hx;
        }

        thomas_solve(A_th.data(), B_th.data(), C_th.data(), D_th.data(),
                     1, Ny, Wt.data(),
                     2, 0.0, 0.0,                      // left BC: dT/dy=0
                     2, 0.0, 2.0 * hy * Hn[n] * Qs[n]); // right BC: dT/dy=Qs

        for (int m = 1; m <= Ny; m++)
          T2[idx(n, m)] = Wt[m];

      }  // end x-march

      // convergence check
      double DeltaH = 0.0;
      for (int n = 1; n <= Nx; n++) {
        double del = std::fabs(Ho[n] - Hn[n]);
        if (del > DeltaH) DeltaH = del;
      }

      // relaxation and reassignment
      for (int n = 1; n <= Nx; n++) {
        for (int m = 1; m <= Ny; m++) {
          Un[idx(n, m)] = relax * Un[idx(n, m)] +
                          (1.0 - relax) * Uo[idx(n, m)];
          Vn[idx(n, m)] = relax * Vn[idx(n, m)] +
                          (1.0 - relax) * Vo[idx(n, m)];
        }
        Hn[n] = relax * Hn[n] + (1.0 - relax) * Ho[n];
      }

      // copy new -> old
      Uo = Un;
      Vo = Vn;
      Ho = Hn;
      T1 = T2;

      // update surface heat flux with evaporative cooling
      if (evap_on) {
        for (int n = 1; n <= Nx; n++) {
          double TMPRDIM = Tin + T2[idx(n, Ny)] * Tscale;
          if (TMPRDIM > 55.0) {
            double EFN = 7.0e-15 * std::pow(TMPRDIM - 55.0, 13.7);
            double Qvapor = li.H_vap * EFN / 6.022e23;
            Qs[n] = Qs0[n] - Qvapor / qss;
          } else {
            Qs[n] = Qs0[n];
          }
        }
      }

      // check stopping criteria
      if (DeltaH < eps_conv && k > 5) break;
      if (t >= (double)max_iter * dt_pseudo) break;
    }

    // extract dimensional outputs
    for (int n = 1; n <= Nx; n++) {
      Tsurf_dim[n] = Tin + T2[idx(n, Ny)] * Tscale;
      h_dim[n] = Hn[n] * h0;

      if (evap_on && Tsurf_dim[n] > 55.0) {
        evap_flux[n] = 7.0e-15 * std::pow(Tsurf_dim[n] - 55.0, 13.7);
        Q_net[n] = Qs[n] * qss;
      } else {
        evap_flux[n] = 0.0;
        Q_net[n] = Qs0[n] * qss;
      }
    }
  }

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
    relax(1.0), ncase(1), evap_on(1) {}
};

}  // namespace LiquidMetal

#endif
