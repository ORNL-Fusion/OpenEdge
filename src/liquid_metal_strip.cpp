/* ----------------------------------------------------------------------
   OpenEdge liquid-metal MHD strip solver implementation.
------------------------------------------------------------------------- */

#include "liquid_metal_strip.h"

namespace LiquidMetal {

double li_evap_flux(double T_C, double sticking)
{
  double T_K = T_C + 273.15;
  if (T_K < 298.15) return 0.0;
  double P_Pa = std::pow(10.0, ANTOINE_A - ANTOINE_B / T_K) * ATM_TO_PA;
  return sticking * P_Pa / std::sqrt(2.0 * PI * MLI * KB * T_K);
}

double li_evap_cooling(double flux, double H_vap)
{
  return H_vap * flux / NA;
}

double li_adatom_flux(double T_C, double Gamma_Dp,
                       double Yad_D_Li, double Yad_Yps,
                       double f_ad, double A_arr, double E_eff)
{
  double T_K = T_C + 273.15;
  if (T_K < 1.0) return 0.0;
  double E_surf_eV = KB * T_K / EV;
  double denom = 1.0 + A_arr * std::exp(E_eff / E_surf_eV);
  return f_ad * (Yad_Yps / denom) * Yad_D_Li * Gamma_Dp;
}

void thomas_solve(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N1, int N2, double *W,
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

void thomas_solve_simple(
    const double *AH, const double *BH, const double *CH, const double *DH,
    int N, double *W,
    int L1, double A1, double Q1,
    int LM, double AM, double QM)
{
  thomas_solve(AH, BH, CH, DH, 1, N, W, L1, A1, Q1, LM, AM, QM);
}

void Strip::init()
{
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

    // reset diagnostics
    diverged = false;
    diverged_reason.clear();
    Tdim_max_seen = 0.0;
    Qs_clamped_count = 0;

    // Validated baseline (Smolentsev ITER-scale) lives at:
    //   Re ~ 5e4, Ha_s ~ 4e5, Pr ~ 0.035, Xl/Nx-spacing modest.
    // Warn if we're far from that envelope. These are SOFT warnings —
    // the run continues, but the user has been told.
    if (Re < 1.0e3)
      std::fprintf(stderr,
        "[liquid_metal_strip] WARN: Re = %.2e is well below the "
        "validated range (~1e4-1e5). Convective sink for surface "
        "heating will be weak; expect slow convergence or runaway.\n", Re);
    if (Ha_s < 5.0e4)
      std::fprintf(stderr,
        "[liquid_metal_strip] WARN: Ha_s = %.2e is below the validated "
        "range (~1e5-1e6). MHD drag may not bound U; expect divergence "
        "if Re is also small. Consider increasing Bs or width.\n", Ha_s);
    if (Pr < 1.0e-3 || Pr > 1.0e2)
      std::fprintf(stderr,
        "[liquid_metal_strip] WARN: Pr = %.2e is outside the liquid-"
        "metal physical range (~1e-3 .. 1e1).\n", Pr);
  }

void Strip::set_heat_flux(const double *x_phys, const double *q_phys, int npts)
{
    // determine physical length from x-coordinates
    double Xin = x_phys[1];
    double Xout = x_phys[npts];
    double Xlength = Xout - Xin;
    Xl = Xlength / h0;
    hx = Xl / (Nx - 1);

    if (Xl > 300.0)
      std::fprintf(stderr,
        "[liquid_metal_strip] WARN: Xl = L/h0 = %.1f is well above the "
        "validated range (~50-200). The marching x-solve may not have "
        "enough pseudo-time to convect heat downstream; consider thicker "
        "h0 or truncating to the wetted target region.\n", Xl);

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

void Strip::set_heat_flux_uniform(double Xlength)
{
    Xl = Xlength / h0;
    hx = Xl / (Nx - 1);
    Tscale = qss * h0 / li.k_th;

    for (int n = 1; n <= Nx; n++) {
      X[n] = (n - 1) * hx;
      // Qs0 must be set externally before calling this
      Qs[n] = Qs0[n];
    }
  }

void Strip::solve_steady()
{
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
        if (evap_on && TMPRDIM > 25.0) {
          double FLUX = li_evap_flux(TMPRDIM);
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

      // update surface heat flux with evaporative cooling.
      // Clamp |Qs| against |Qs0| * Qs_clamp_factor so a single overshoot
      // in T cannot drive Qs hugely negative and oscillate the next sweep.
      if (evap_on) {
        const double clamp = Qs_clamp_factor;
        for (int n = 1; n <= Nx; n++) {
          double TMPRDIM = Tin + T2[idx(n, Ny)] * Tscale;
          double EFN = li_evap_flux(TMPRDIM);
          double q_new;
          if (EFN > 0.0) {
            double Qvapor = li_evap_cooling(EFN, li.H_vap);
            q_new = Qs0[n] - Qvapor / qss;
          } else {
            q_new = Qs0[n];
          }
          double q_bound = clamp * std::fabs(Qs0[n]);
          if (q_new >  q_bound) { q_new =  q_bound; ++Qs_clamped_count; }
          if (q_new < -q_bound) { q_new = -q_bound; ++Qs_clamped_count; }
          Qs[n] = q_new;
        }
      }

      // runtime divergence guard: every diverge_check_every iterations,
      // scan T2 for NaN or runaway. Sets diverged + reason and breaks.
      if (k % diverge_check_every == 0) {
        double Tmax_iter = 0.0;
        bool nanseen = false;
        for (int n = 1; n <= Nx && !nanseen; n++) {
          for (int m = 1; m <= Ny && !nanseen; m++) {
            double v = T2[idx(n, m)];
            if (!std::isfinite(v)) { nanseen = true; break; }
            double a = std::fabs(v);
            if (a > Tmax_iter) Tmax_iter = a;
          }
        }
        if (Tmax_iter > Tdim_max_seen) Tdim_max_seen = Tmax_iter;
        if (nanseen) {
          diverged = true;
          diverged_reason = "non-finite T encountered (NaN or Inf)";
          break;
        }
        if (Tmax_iter > Tdim_diverge) {
          diverged = true;
          char buf[256];
          std::snprintf(buf, sizeof(buf),
            "max |T_dimless| = %.2e > Tdim_diverge = %.2e (~%.0f K dim) "
            "at pseudo-time iter %d; check Re, Ha_s, Xl",
            Tmax_iter, Tdim_diverge, Tmax_iter * Tscale, k);
          diverged_reason = buf;
          break;
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
      evap_flux[n] = li_evap_flux(Tsurf_dim[n]);
      Q_net[n] = (evap_flux[n] > 0.0) ? Qs[n] * qss : Qs0[n] * qss;
    }
  }

} // namespace LiquidMetal
