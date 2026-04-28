// Standalone driver for LiquidMetal::Strip — verifies the solver against
// SOLPS-ITER target loading data without going through SPARTA.
//
// Reads ld_tg_o.dat (or ld_tg_i.dat) from a SOLPS b2pl.exe.dir output and
// drives the strip solver to steady state. Writes per-station CSV with
// Tsurf, film thickness, evaporation flux, and adatom flux.
//
// Usage:
//   ./run_strip ld_tg_o.dat out.csv [key=value ...]
//
// Knobs (key=value, all optional; defaults match the test_solps_coupling deck):
//   h0=1e-3     U0=0.5      Bs=0.5      alpha=5        width=0.05
//   Tin=600     Nx=201      Ny=51       evap=1
//   m_ion_amu=2.014   (deuterium plasma; for adatom Bohm flux)
//
// Build:
//   g++ -O2 -std=c++11 -I../../src/OPENEDGE run_strip.cpp -o run_strip

#include "liquid_metal_strip.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
  double x;       // arc length along target [m]; <0 in PFR
  double ne;      // electron density [1/m^3]
  double Te;      // electron temperature [eV]
  double Ti;      // ion temperature [eV]
  double Wtot;    // total power flux [W/m^2]
  double Rclfc;   // radial coord on target [m] (col 16, for toroidal weighting)
};

// SOLPS ld_tg_*.dat parser. We need x, ne, Te, Ti, Wtot and Rclfc.
// Column order (per the header): 1=x  2=ne  3=Te  4=Ti  5=Wtot
//   6..13 = Wpart..Wptpl   14=xMP   15=Rclfc   16=Zclfc   ...
// Read 1..5, skip 9 (6..14), then read 15 = Rclfc.
std::vector<Row> read_ld_tg(const std::string &path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "run_strip: cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<Row> rows;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    Row r;
    if (!(is >> r.x >> r.ne >> r.Te >> r.Ti >> r.Wtot)) continue;
    double skip;
    bool ok = true;
    for (int k = 0; k < 9 && ok; ++k) ok = static_cast<bool>(is >> skip);
    if (!(ok && (is >> r.Rclfc))) r.Rclfc = 0.0;  // fall back: no toroidal weight
    rows.push_back(r);
  }
  if (rows.size() < 4) {
    std::fprintf(stderr, "run_strip: too few data rows in %s (%zu)\n",
                 path.c_str(), rows.size());
    std::exit(1);
  }
  return rows;
}

double get_kv(int argc, char **argv, const char *key, double dflt) {
  std::string pat = std::string(key) + "=";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.size() > pat.size() && a.compare(0, pat.size(), pat) == 0)
      return std::atof(a.c_str() + pat.size());
  }
  return dflt;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
        "Usage: %s <ld_tg_*.dat> <out.csv> [key=value ...]\n", argv[0]);
    return 1;
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];

  // SOLPS target loading
  auto rows = read_ld_tg(in_path);
  const int N = static_cast<int>(rows.size());

  // Strip parameters (defaults match the SPARTA deck; override via key=value)
  LiquidMetal::Strip s;
  s.h0        = get_kv(argc, argv, "h0",     1.0e-3);
  s.U0        = get_kv(argc, argv, "U0",     0.5);
  s.Bs        = get_kv(argc, argv, "Bs",     0.5);
  s.alpha_deg = get_kv(argc, argv, "alpha",  5.0);
  s.width     = get_kv(argc, argv, "width",  0.05);
  s.Tin       = get_kv(argc, argv, "Tin",    600.0);
  s.Nx        = static_cast<int>(get_kv(argc, argv, "Nx", 201));
  s.Ny        = static_cast<int>(get_kv(argc, argv, "Ny", 51));
  s.evap_on   = static_cast<int>(get_kv(argc, argv, "evap", 1));

  const double m_ion_amu = get_kv(argc, argv, "m_ion_amu", 2.014);
  const double m_ion_kg  = m_ion_amu * 1.66053906660e-27;

  // qss = peak Wtot rounds the dimensionless temperature scale; using
  // the actual peak from the data matches the solver's own scaling.
  double q_peak = 0.0;
  for (const auto &r : rows) q_peak = std::max(q_peak, r.Wtot);
  s.qss = (q_peak > 0.0) ? q_peak : 1.0e6;

  s.init();

  // Strip is 1-indexed; pack x and Wtot with size N+2 so [1..N] is filled.
  std::vector<double> x_phys(N + 2, 0.0), q_phys(N + 2, 0.0);
  for (int i = 0; i < N; ++i) {
    x_phys[i + 1] = rows[i].x;
    q_phys[i + 1] = rows[i].Wtot;
  }
  s.set_heat_flux(x_phys.data(), q_phys.data(), N);

  std::printf("run_strip: %s -> %s\n", in_path.c_str(), out_path.c_str());
  std::printf("  N_solps=%d  qss=%.3e W/m^2  Xlength=%.3f m\n",
              N, s.qss, x_phys[N] - x_phys[1]);
  std::printf("  Re=%.3e Fr=%.3e Ha_s=%.3e Pr=%.3e\n",
              s.Re, s.Fr, s.Ha_s, s.Pr);

  s.solve_steady();

  // Per-strip-station outputs. The strip mesh is uniform in dimensionless
  // x; map back to physical arc length using the inlet x and h0.
  const double Xin = x_phys[1];
  std::ofstream out(out_path);
  out << "# x_m,Tsurf_C,h_film_m,Qnet_Wm2,evap_flux,adatom_flux,ne,Te,Ti,Wtot\n";
  out.setf(std::ios::scientific);
  out.precision(6);

  // For adatom: Bohm flux Gamma = ne * sqrt(e(Te+Ti)/m_ion), interpolated
  // back from the SOLPS data at each strip station.
  auto interp_solps = [&](double xq, int col) -> double {
    if (xq <= rows.front().x) return col == 0 ? rows.front().ne :
                                      col == 1 ? rows.front().Te :
                                      rows.front().Ti;
    if (xq >= rows.back().x)  return col == 0 ? rows.back().ne :
                                      col == 1 ? rows.back().Te :
                                      rows.back().Ti;
    for (int k = 0; k + 1 < N; ++k) {
      if (xq >= rows[k].x && xq <= rows[k + 1].x) {
        double f = (xq - rows[k].x) / (rows[k + 1].x - rows[k].x);
        double a = col == 0 ? rows[k].ne : col == 1 ? rows[k].Te : rows[k].Ti;
        double b = col == 0 ? rows[k + 1].ne : col == 1 ? rows[k + 1].Te : rows[k + 1].Ti;
        return a + f * (b - a);
      }
    }
    return 0.0;
  };

  for (int n = 1; n <= s.Nx; ++n) {
    double xq = Xin + s.X[n] * s.h0;
    double ne = interp_solps(xq, 0);
    double Te = interp_solps(xq, 1);
    double Ti = interp_solps(xq, 2);
    double cs = (Te + Ti > 0.0)
                ? std::sqrt(LiquidMetal::EV * (Te + Ti) / m_ion_kg) : 0.0;
    double gamma_Dp = ne * cs;
    double adat = LiquidMetal::li_adatom_flux(s.Tsurf_dim[n], gamma_Dp);
    double Wtot = s.Q_net[n];

    out << xq << ',' << s.Tsurf_dim[n] << ',' << s.h_dim[n] << ','
        << s.Q_net[n] << ',' << s.evap_flux[n] << ',' << adat << ','
        << ne << ',' << Te << ',' << Ti << ',' << Wtot << '\n';
  }

  // Sanity numbers for terminal cross-check
  double Tsurf_max = 0.0, evap_max = 0.0;
  int n_at_peak = 1;
  for (int n = 1; n <= s.Nx; ++n) {
    if (s.Tsurf_dim[n] > Tsurf_max) { Tsurf_max = s.Tsurf_dim[n]; n_at_peak = n; }
    if (s.evap_flux[n] > evap_max)  evap_max = s.evap_flux[n];
  }
  std::printf("  Tsurf peak: %.1f C at x=%.3f m\n",
              Tsurf_max, Xin + s.X[n_at_peak] * s.h0);
  std::printf("  evap_flux peak: %.3e atoms/m^2/s\n", evap_max);

  // Integrated input power along the strip.
  // Per-toroidal-meter (matches strip's 2D framing) AND full toroidal
  // (matches SOLPS peak_power_w int_tot — uses Rclfc(x) from the data).
  auto interp_R = [&](double xq) -> double {
    if (xq <= rows.front().x) return rows.front().Rclfc;
    if (xq >= rows.back().x)  return rows.back().Rclfc;
    for (int k = 0; k + 1 < N; ++k) {
      if (xq >= rows[k].x && xq <= rows[k + 1].x) {
        double f = (xq - rows[k].x) / (rows[k + 1].x - rows[k].x);
        return rows[k].Rclfc + f * (rows[k + 1].Rclfc - rows[k].Rclfc);
      }
    }
    return 0.0;
  };

  double P_per_m = 0.0;       // ∫ Wtot dx          [W per toroidal m]
  double P_torus = 0.0;       // ∫ Wtot · 2πR dx    [W, full torus]
  for (int n = 1; n < s.Nx; ++n) {
    double xn  = Xin + s.X[n] * s.h0;
    double xn1 = Xin + s.X[n + 1] * s.h0;
    double dx  = xn1 - xn;
    double q   = 0.5 * (s.Qs0[n] + s.Qs0[n + 1]) * s.qss;
    double Rmid = 0.5 * (interp_R(xn) + interp_R(xn1));
    P_per_m += q * dx;
    P_torus += q * dx * 2.0 * LiquidMetal::PI * Rmid;
  }
  std::printf("  integrated power: %.3e W/m  (toroidal: %.3e W)\n",
              P_per_m, P_torus);
  if (s.diverged) {
    std::printf("  *** SOLVER DIVERGED: %s\n", s.diverged_reason.c_str());
  }

  return 0;
}
