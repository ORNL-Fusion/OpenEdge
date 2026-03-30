/* ----------------------------------------------------------------------
   Standalone validation of the liquid metal strip solver against
   Sergey Smolentsev's Fortran reference (NVAR=2, NHEAT=3, Ncase=1).

   Outer divertor: h0=5mm, U0=8m/s, Bs=5T, alpha=43deg, Width=1.67m
   Gaussian heat flux: qss=10 MW/m^2, Qs0 = exp(-0.4*(x - 0.5*Xl)^2)
   No evaporation (matches Fortran: evaporation line is commented out).

   Build:
     g++ -O2 -std=c++11 -I../../src/OPENEDGE test_strip.cpp -o test_strip

   Run:
     ./test_strip

   Compare: Tsurf_dml.dat and h_dim.dat against Fortran reference in ref/
------------------------------------------------------------------------- */

#include "liquid_metal_strip.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

int main(int argc, char **argv)
{
    LiquidMetal::Strip strip;

    // match Fortran NVAR=2 (outer divertor)
    strip.h0 = 0.005;
    strip.U0 = 8.0;
    strip.Bs = 5.0;
    strip.Bw = 0.0;
    strip.alpha_deg = 43.0;
    strip.width = 1.67;
    strip.Tin = 350.0;

    // wall conductance
    strip.sigma_w = 0.0;
    strip.tw = 0.000025;

    // Fortran uses sigma_e = 3.09e6 (overridden from default 1.1e6)
    strip.li.sigma_e = 3.09e6;

    // mesh: Fortran uses Nx=1001, Ny=201
    strip.Nx = 1001;
    strip.Ny = 201;

    // solver parameters (match Fortran)
    strip.dt_pseudo = 0.5;
    strip.max_iter = 3000;
    strip.eps_conv = 0.5e-7;
    strip.relax = 1.0;
    strip.ncase = 1;

    // NHEAT=3: Gaussian heat flux, qss = 10 MW/m^2
    strip.qss = 10.0e6;

    // no evaporation (Fortran has this line commented out)
    strip.evap_on = 0;

    // initialize
    strip.init();

    // set up Gaussian heat flux: Qs0(x) = exp(-0.4*(x - 0.5*Xl)^2)
    double Xlength = 0.191737;  // outer divertor length [m]
    strip.Xl = Xlength / strip.h0;
    strip.hx = strip.Xl / (strip.Nx - 1);
    strip.Tscale = strip.qss * strip.h0 / strip.li.k_th;

    for (int n = 1; n <= strip.Nx; n++) {
        strip.X[n] = (n - 1) * strip.hx;
        double pexp = 0.4;
        strip.Qs0[n] = std::exp(-pexp * (strip.X[n] - 0.5 * strip.Xl) *
                                         (strip.X[n] - 0.5 * strip.Xl));
        strip.Qs[n] = strip.Qs0[n];
    }

    printf("Strip solver validation (outer divertor, Gaussian heat flux)\n");
    printf("  Nx=%d  Ny=%d  h0=%.4f  U0=%.1f  Bs=%.1f\n",
           strip.Nx, strip.Ny, strip.h0, strip.U0, strip.Bs);
    printf("  Re=%.5e  Fr=%.5e  Ha_s=%.5e\n", strip.Re, strip.Fr, strip.Ha_s);
    printf("  Be=%.5e  Cw=%.5e  Rtor=%.5e  Pr=%.5e\n",
           strip.Be, strip.Cw, strip.Rtor, strip.Pr);
    printf("  Tscale=%.5e  qss=%.2e\n", strip.Tscale, strip.qss);
    printf("Solving to steady state...\n");

    strip.solve_steady();

    // write Tsurf_dml.dat
    FILE *fp = fopen("Tsurf_dml.dat", "w");
    for (int n = 1; n <= strip.Nx; n++) {
        double xdim = strip.X[n] * strip.h0;
        fprintf(fp, " %9.4f   %.5e\n", xdim, strip.Tsurf_dim[n]);
    }
    fclose(fp);

    // write h_dim.dat
    fp = fopen("h_dim.dat", "w");
    for (int n = 1; n <= strip.Nx; n++) {
        double xdim = strip.X[n] * strip.h0;
        fprintf(fp, " %9.4f   %.5e\n", xdim, strip.h_dim[n]);
    }
    fclose(fp);

    // write Q_surface.dat (input heat flux)
    fp = fopen("Q_surface.dat", "w");
    for (int n = 1; n <= strip.Nx; n++) {
        double xdim = strip.X[n] * strip.h0;
        fprintf(fp, " %12.5e  %12.5e\n", xdim, strip.Qs0[n] * strip.qss);
    }
    fclose(fp);

    printf("Output written: Tsurf_dml.dat, h_dim.dat, Q_surface.dat\n");

    // --- Compare against Fortran reference if available ---
    FILE *ref = fopen("ref/Tsurf_dml.dat", "r");
    if (ref) {
        double max_err = 0.0;
        double max_rel = 0.0;
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

        printf("\nComparison with Fortran reference (%d points):\n", nref);
        printf("  Max absolute error in Tsurf: %.4e C\n", max_err);
        printf("  Max relative error in Tsurf: %.4e\n", max_rel);

        if (max_rel < 1.0e-3)
            printf("  PASS (relative error < 1e-3)\n");
        else if (max_rel < 1.0e-2)
            printf("  MARGINAL (relative error < 1e-2)\n");
        else
            printf("  FAIL (relative error >= 1e-2)\n");
    } else {
        printf("\nNo Fortran reference found at ref/Tsurf_dml.dat\n");
        printf("Copy from /home/cloud/mhd/DATA/ to run comparison.\n");
    }

    // print a few sample points
    printf("\nSample T_surf values:\n");
    printf("  x=0.000 m: T=%.2f C\n", strip.Tsurf_dim[1]);
    int mid = strip.Nx / 2;
    printf("  x=%.3f m: T=%.2f C (midpoint)\n",
           strip.X[mid] * strip.h0, strip.Tsurf_dim[mid]);
    printf("  x=%.3f m: T=%.2f C (outlet)\n",
           strip.X[strip.Nx] * strip.h0, strip.Tsurf_dim[strip.Nx]);

    return 0;
}
