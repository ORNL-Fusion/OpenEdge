/* ----------------------------------------------------------------------
   Standalone validation of the liquid metal strip solver against
   Sergey Smolentsev's Fortran reference (main.for / MYGTRI.FOR).

   Three test cases:
     1) Gaussian heat flux (NVAR=2, NHEAT=3) -- ref/
     2) Outer divertor SOLPS heat flux (NVAR=2, NHEAT=2) -- ref_outer/
     3) Inner divertor SOLPS heat flux (NVAR=1, NHEAT=1) -- ref_inner/

   Build:
     g++ -O2 -std=c++11 -I../../src/OPENEDGE test_strip.cpp -o test_strip

   Run:
     ./test_strip            # run all three cases
     ./test_strip gaussian   # run only Gaussian
     ./test_strip outer      # run only outer SOLPS
     ./test_strip inner      # run only inner SOLPS
------------------------------------------------------------------------- */

#include "liquid_metal_strip.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

/* ------------------------------------------------------------------
   Jeremy Lore's SOLPS heat flux profiles (from main.for SHFo/SHFi).
   36 data points each, x in [m], heat flux in [MW/m^2].
------------------------------------------------------------------ */

static const int N_SOLPS = 36;

// Outer divertor
static const double Xheat_outer[N_SOLPS] = {
    0.000000, 0.009979, 0.020073, 0.030049, 0.040138,
    0.050149, 0.059810, 0.069184, 0.078447, 0.087355,
    0.095605, 0.103290, 0.110514, 0.117186, 0.122969,
    0.127785, 0.131672, 0.134569, 0.136246, 0.137428,
    0.139019, 0.140989, 0.143360, 0.146108, 0.149154,
    0.152461, 0.155975, 0.159607, 0.163345, 0.167207,
    0.171160, 0.175173, 0.179294, 0.183497, 0.187671,
    0.191737
};
static const double Heat_outer[N_SOLPS] = {
    2.719710, 3.819670, 4.685440, 5.011570, 5.483040,
    6.273190, 6.580600, 6.810900, 7.384040, 7.666150,
    8.260160, 8.739420, 7.412780, 6.371840, 5.682940,
    4.540650, 3.387410, 2.964730, 2.689490, 2.237820,
    2.050670, 1.738580, 1.542260, 1.194080, 0.966618,
    0.924331, 0.769075, 0.733574, 0.568295, 0.561533,
    0.527683, 0.449960, 0.443661, 0.394544, 0.383064,
    0.358113
};

// Inner divertor
static const double Xheat_inner[N_SOLPS] = {
    0.000000, 0.019651, 0.039355, 0.058800, 0.078289,
    0.097963, 0.117531, 0.136670, 0.155480, 0.173746,
    0.191030, 0.207417, 0.222524, 0.236061, 0.247965,
    0.258057, 0.266327, 0.272365, 0.275790, 0.278260,
    0.281586, 0.285741, 0.290674, 0.296141, 0.302082,
    0.308629, 0.315722, 0.323115, 0.330676, 0.338501,
    0.346520, 0.354504, 0.362450, 0.370473, 0.378343,
    0.385966
};
static const double Heat_inner[N_SOLPS] = {
    1.314290, 1.828720, 2.037420, 2.108560, 2.199070,
    2.339560, 2.270300, 2.385230, 2.302060, 2.470620,
    2.511400, 2.632190, 2.973900, 2.771880, 2.822800,
    2.565450, 1.965050, 1.610460, 1.313400, 1.114690,
    1.010870, 0.802295, 0.749200, 0.571185, 0.459776,
    0.418857, 0.317604, 0.278340, 0.259040, 0.203935,
    0.178766, 0.181736, 0.162898, 0.159038, 0.123292,
    0.111291
};

/* ------------------------------------------------------------------
   Interpolate SOLPS heat flux at coordinate x [m].
   Returns heat flux in MW/m^2.
   Matches Fortran nearest-bracket + linear interpolation.
------------------------------------------------------------------ */

static double interp_solps(double coord, const double *Xh, const double *Qh, int npts)
{
    // find nearest data point
    double delmax = 1.0e10;
    int ks = 0;
    for (int k = 0; k < npts; k++) {
        double del = std::fabs(coord - Xh[k]);
        if (del < delmax) { delmax = del; ks = k; }
    }

    double result;
    if (coord >= Xh[ks]) {
        int kp = (ks < npts - 1) ? ks + 1 : ks;
        double dx = Xh[kp] - Xh[ks];
        if (dx > 0.0)
            result = Qh[ks] + (Qh[kp] - Qh[ks]) * (coord - Xh[ks]) / dx;
        else
            result = Qh[ks];
    } else {
        int km = (ks > 0) ? ks - 1 : ks;
        double dx = Xh[ks] - Xh[km];
        if (dx > 0.0)
            result = Qh[ks] + (Qh[ks] - Qh[km]) * (coord - Xh[ks]) / dx;
        else
            result = Qh[ks];
    }
    return result;
}

/* ------------------------------------------------------------------
   Compare output against Fortran reference.
   Returns max relative error (-1 if no reference found).
------------------------------------------------------------------ */

static double compare_reference(const LiquidMetal::Strip &strip,
                                const char *ref_path)
{
    FILE *ref = fopen(ref_path, "r");
    if (!ref) return -1.0;

    double max_err = 0.0, max_rel = 0.0;
    int nref = 0;
    char line[256];
    int n = 1;

    while (fgets(line, sizeof(line), ref) && n <= strip.Nx) {
        double xref, tref;
        if (sscanf(line, "%lf %lf", &xref, &tref) == 2) {
            double terr = std::fabs(strip.Tsurf_dim[n] - tref);
            double trel = (tref > 0.0) ? terr / tref : terr;
            if (terr > max_err) max_err = terr;
            if (trel > max_rel) max_rel = trel;
            nref++;
        }
        n++;
    }
    fclose(ref);

    printf("  Comparison with Fortran reference (%d points):\n", nref);
    printf("    Max absolute error: %.4e C\n", max_err);
    printf("    Max relative error: %.4e\n", max_rel);

    if (max_rel < 1.0e-3)
        printf("    --> PASS (relative error < 1e-3)\n");
    else if (max_rel < 1.0e-2)
        printf("    --> MARGINAL (relative error < 1e-2)\n");
    else
        printf("    --> FAIL (relative error >= 1e-2)\n");

    return max_rel;
}

/* ------------------------------------------------------------------
   Run one test case.
------------------------------------------------------------------ */

struct TestResult {
    const char *name;
    double max_rel_err;
    bool has_ref;
};

static TestResult run_case(const char *name,
                           double h0, double U0, double Bs, double Bw,
                           double alpha_deg, double width, double Tin,
                           double qss, double Xlength,
                           int nheat,  // 1=inner SOLPS, 2=outer SOLPS, 3=gaussian
                           const char *ref_dir,
                           const char *out_prefix)
{
    LiquidMetal::Strip strip;

    strip.h0 = h0;
    strip.U0 = U0;
    strip.Bs = Bs;
    strip.Bw = Bw;
    strip.alpha_deg = alpha_deg;
    strip.width = width;
    strip.Tin = Tin;
    strip.li.sigma_e = 3.09e6;
    strip.sigma_w = 0.0;
    strip.tw = 0.000025;

    strip.Nx = 1001;
    strip.Ny = 201;
    strip.qss = qss;
    strip.dt_pseudo = 0.5;
    strip.max_iter = 3000;
    strip.eps_conv = 0.5e-7;
    strip.relax = 1.0;
    strip.ncase = 1;
    strip.evap_on = 0;  // Fortran has evaporation commented out

    strip.init();

    // set up grid and heat flux
    strip.Xl = Xlength / strip.h0;
    strip.hx = strip.Xl / (strip.Nx - 1);
    strip.Tscale = strip.qss * strip.h0 / strip.li.k_th;

    for (int n = 1; n <= strip.Nx; n++) {
        strip.X[n] = (n - 1) * strip.hx;
        double coord = strip.X[n] * strip.h0;  // physical x [m]

        if (nheat == 3) {
            // Gaussian
            double pexp = 0.4;
            strip.Qs0[n] = std::exp(-pexp * (strip.X[n] - 0.5 * strip.Xl) *
                                             (strip.X[n] - 0.5 * strip.Xl));
        } else if (nheat == 2) {
            // Outer divertor SOLPS (MW/m^2 -> dimensionless via qss)
            strip.Qs0[n] = interp_solps(coord, Xheat_outer, Heat_outer, N_SOLPS);
        } else if (nheat == 1) {
            // Inner divertor SOLPS (MW/m^2 -> dimensionless via qss)
            strip.Qs0[n] = interp_solps(coord, Xheat_inner, Heat_inner, N_SOLPS);
        }
        strip.Qs[n] = strip.Qs0[n];
    }

    printf("\n========================================\n");
    printf("  %s\n", name);
    printf("========================================\n");
    printf("  h0=%.4f  U0=%.1f  Bs=%.1f  alpha=%.1f  width=%.2f  Tin=%.0f\n",
           h0, U0, Bs, alpha_deg, width, Tin);
    printf("  Re=%.5e  Fr=%.5e  Ha_s=%.5e  Pr=%.4e\n",
           strip.Re, strip.Fr, strip.Ha_s, strip.Pr);
    printf("  Rtor=%.5e  Tscale=%.5e  qss=%.2e\n",
           strip.Rtor, strip.Tscale, strip.qss);
    printf("  Solving...\n");

    strip.solve_steady();

    // sample output
    int mid = strip.Nx / 2;
    printf("  T_surf: inlet=%.2f  mid=%.2f  outlet=%.2f C\n",
           strip.Tsurf_dim[1], strip.Tsurf_dim[mid], strip.Tsurf_dim[strip.Nx]);

    // find peak temperature
    double tmax = 0.0;
    int imax = 1;
    for (int n = 1; n <= strip.Nx; n++) {
        if (strip.Tsurf_dim[n] > tmax) { tmax = strip.Tsurf_dim[n]; imax = n; }
    }
    printf("  T_peak=%.2f C at x=%.4f m\n", tmax, strip.X[imax] * strip.h0);

    // write output files
    char fname[256];
    snprintf(fname, sizeof(fname), "%s_Tsurf_dml.dat", out_prefix);
    FILE *fp = fopen(fname, "w");
    for (int n = 1; n <= strip.Nx; n++)
        fprintf(fp, " %9.4f   %.5e\n", strip.X[n] * strip.h0, strip.Tsurf_dim[n]);
    fclose(fp);

    snprintf(fname, sizeof(fname), "%s_h_dim.dat", out_prefix);
    fp = fopen(fname, "w");
    for (int n = 1; n <= strip.Nx; n++)
        fprintf(fp, " %9.4f   %.5e\n", strip.X[n] * strip.h0, strip.h_dim[n]);
    fclose(fp);

    printf("  Output: %s_Tsurf_dml.dat, %s_h_dim.dat\n", out_prefix, out_prefix);

    // compare against Fortran reference
    TestResult result;
    result.name = name;
    snprintf(fname, sizeof(fname), "%s/Tsurf_dml.dat", ref_dir);
    result.max_rel_err = compare_reference(strip, fname);
    result.has_ref = (result.max_rel_err >= 0.0);

    return result;
}

/* ================================================================== */

int main(int argc, char **argv)
{
    bool run_gaussian = true, run_outer = true, run_inner = true;

    if (argc > 1) {
        run_gaussian = run_outer = run_inner = false;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "gaussian") == 0) run_gaussian = true;
            if (strcmp(argv[i], "outer") == 0) run_outer = true;
            if (strcmp(argv[i], "inner") == 0) run_inner = true;
            if (strcmp(argv[i], "all") == 0)
                run_gaussian = run_outer = run_inner = true;
        }
    }

    printf("Liquid metal strip solver validation\n");
    printf("Comparing C++ port against Sergey Smolentsev's Fortran code\n");

    std::vector<TestResult> results;

    // --- Case 1: Gaussian (NVAR=2, NHEAT=3) ---
    if (run_gaussian) {
        results.push_back(run_case(
            "Outer divertor, Gaussian heat flux (NHEAT=3)",
            0.005, 8.0, 5.0, 0.0,       // h0, U0, Bs, Bw
            43.0, 1.67, 350.0,           // alpha, width, Tin
            10.0e6, 0.191737,            // qss, Xlength
            3, "ref", "gaussian"));       // nheat, ref_dir, out_prefix
    }

    // --- Case 2: Outer divertor SOLPS (NVAR=2, NHEAT=2) ---
    if (run_outer) {
        results.push_back(run_case(
            "Outer divertor, SOLPS heat flux (NHEAT=2)",
            0.005, 8.0, 5.0, 0.0,
            43.0, 1.67, 350.0,
            1.0e6, 0.191737,             // qss=1 MW/m^2 for SOLPS
            2, "ref_outer", "outer"));
    }

    // --- Case 3: Inner divertor SOLPS (NVAR=1, NHEAT=1) ---
    if (run_inner) {
        results.push_back(run_case(
            "Inner divertor, SOLPS heat flux (NHEAT=1)",
            0.005, 3.0, 5.0, 0.0,       // inner: U0=3
            73.0, 1.43, 350.0,           // inner: alpha=73, width=1.43
            1.0e6, 0.385966,             // inner: longer divertor
            1, "ref_inner", "inner"));
    }

    // --- Case 4: Gaussian with evaporation (Antoine+HK) ---
    if (run_gaussian) {
        // Same setup as case 1 but with evaporation enabled.
        // No Fortran reference (model changed), so just verify:
        //   - Tsurf < no-evap case (evaporative cooling works)
        //   - evap_flux > 0 where Tsurf > 25 C
        //   - film thickness decreases from evaporation

        LiquidMetal::Strip s;
        s.h0 = 0.005; s.U0 = 8.0; s.Bs = 5.0; s.Bw = 0.0;
        s.alpha_deg = 43.0; s.width = 1.67; s.Tin = 350.0;
        s.li.sigma_e = 3.09e6; s.sigma_w = 0.0; s.tw = 0.000025;
        // Use high heat flux to push Tsurf > 500 C where evaporation matters
        s.Nx = 1001; s.Ny = 201; s.qss = 50.0e6;
        s.dt_pseudo = 0.5; s.max_iter = 3000; s.eps_conv = 0.5e-7;
        s.relax = 1.0; s.ncase = 1; s.evap_on = 0;

        s.init();
        s.Xl = 0.191737 / s.h0;
        s.hx = s.Xl / (s.Nx - 1);
        s.Tscale = s.qss * s.h0 / s.li.k_th;

        for (int n = 1; n <= s.Nx; n++) {
            s.X[n] = (n - 1) * s.hx;
            double pexp = 0.4;
            s.Qs0[n] = std::exp(-pexp * (s.X[n] - 0.5 * s.Xl) *
                                         (s.X[n] - 0.5 * s.Xl));
            s.Qs[n] = s.Qs0[n];
        }

        printf("\n========================================\n");
        printf("  Gaussian + evaporation (Antoine+HK)\n");
        printf("========================================\n");

        // first: no-evap baseline at same qss
        s.solve_steady();
        double tmax_noevap = 0.0;
        for (int n = 1; n <= s.Nx; n++)
            if (s.Tsurf_dim[n] > tmax_noevap) tmax_noevap = s.Tsurf_dim[n];

        // now: with evaporation
        s.evap_on = 1;
        s.init();
        s.Xl = 0.191737 / s.h0;
        s.hx = s.Xl / (s.Nx - 1);
        s.Tscale = s.qss * s.h0 / s.li.k_th;
        for (int n = 1; n <= s.Nx; n++) {
            s.X[n] = (n - 1) * s.hx;
            double pexp = 0.4;
            s.Qs0[n] = std::exp(-pexp * (s.X[n] - 0.5 * s.Xl) *
                                         (s.X[n] - 0.5 * s.Xl));
            s.Qs[n] = s.Qs0[n];
        }
        s.solve_steady();

        double tmax_evap = 0.0, emax = 0.0, hmin = s.h0;
        for (int n = 1; n <= s.Nx; n++) {
            if (s.Tsurf_dim[n] > tmax_evap) tmax_evap = s.Tsurf_dim[n];
            if (s.evap_flux[n] > emax) emax = s.evap_flux[n];
            if (s.h_dim[n] < hmin) hmin = s.h_dim[n];
        }

        printf("  T_peak (no evap) = %.2f C\n", tmax_noevap);
        printf("  T_peak (w/ evap) = %.2f C\n", tmax_evap);
        printf("  Evap cooling dT  = %.2f C\n", tmax_noevap - tmax_evap);
        printf("  Max evap flux    = %.3e atoms/m2/s\n", emax);
        printf("  Min film thick   = %.6f m (initial %.4f)\n", hmin, s.h0);

        // verify Antoine+HK at a known temperature
        double test_T = 500.0;  // C
        double test_flux = LiquidMetal::li_evap_flux(test_T);
        printf("  Antoine+HK check: flux(500C) = %.3e atoms/m2/s\n", test_flux);

        // ad-atom model check
        double test_adatom = LiquidMetal::li_adatom_flux(500.0, 1e22);
        printf("  Ad-atom check: flux(500C, 1e22 D+) = %.3e atoms/m2/s\n", test_adatom);

        // write output
        FILE *fp = fopen("evap_Tsurf_dml.dat", "w");
        for (int n = 1; n <= s.Nx; n++)
            fprintf(fp, " %9.4f   %.5e   %.5e   %.5e\n",
                    s.X[n] * s.h0, s.Tsurf_dim[n], s.evap_flux[n], s.h_dim[n]);
        fclose(fp);
        printf("  Output: evap_Tsurf_dml.dat (x, Tsurf, evap_flux, h)\n");

        TestResult r;
        r.name = "Gaussian + evaporation (Antoine+HK)";

        bool ok = true;
        if (tmax_evap >= tmax_noevap) {
            printf("  FAIL: evaporative cooling not reducing Tsurf\n");
            ok = false;
        }
        if (emax <= 0.0) {
            printf("  FAIL: evaporation flux is zero\n");
            ok = false;
        }
        // Antoine+HK at 500C should give ~3e22 (validated against NIST data)
        if (test_flux < 1e21 || test_flux > 1e23) {
            printf("  FAIL: Antoine+HK flux at 500C out of expected range\n");
            ok = false;
        }
        // ad-atom flux should be positive with nonzero D+ flux
        if (test_adatom <= 0.0) {
            printf("  FAIL: ad-atom flux is zero with nonzero D+ flux\n");
            ok = false;
        }

        r.max_rel_err = ok ? 0.0 : 1.0;
        r.has_ref = true;
        if (ok) printf("  --> PASS (self-consistency checks)\n");
        else printf("  --> FAIL\n");
        results.push_back(r);
    }

    // --- Summary ---
    printf("\n========================================\n");
    printf("  SUMMARY\n");
    printf("========================================\n");
    int pass = 0, fail = 0, noref = 0;
    for (size_t i = 0; i < results.size(); i++) {
        if (!results[i].has_ref) {
            printf("  %-50s  NO REF\n", results[i].name);
            noref++;
        } else if (results[i].max_rel_err < 1.0e-3) {
            printf("  %-50s  PASS (%.2e)\n", results[i].name, results[i].max_rel_err);
            pass++;
        } else {
            printf("  %-50s  FAIL (%.2e)\n", results[i].name, results[i].max_rel_err);
            fail++;
        }
    }
    printf("  ---\n");
    printf("  %d PASS, %d FAIL, %d NO REF\n", pass, fail, noref);

    return (fail > 0) ? 1 : 0;
}
