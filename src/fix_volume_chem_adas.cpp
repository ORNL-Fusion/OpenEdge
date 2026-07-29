/* ----------------------------------------------------------------------
    OpenEdge: ADAS ionization/recombination chemistry fix
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - 42d
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include <cmath>
#include <algorithm>
#include "stdlib.h"
#include "string.h"
#include <unistd.h>
#include "fix_volume_chem_adas.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "math.h"
#include "react_bird.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "random_knuth.h"
#include "math_const.h"
#include <filesystem>
#include <vector>
#include "math_extra.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "compute.h"
#include "compute_plasma_fields.h"
#include "fix_background.h"
#include "domain.h"
#include "openedge_geom.h"
#include "database_paths.h"
#include "process_library.h"

namespace fs = std::filesystem;
using namespace SPARTA_NS;
using MathConst::MY_2PI;
enum{IONIZATION,RECOMBINATION,EXCHANGE,DISSOCIATION};  // file-local reaction types
enum{IONIZATIONRATE, RECOMBINATIONRATE};               // other files
enum{ADAS,JANEV};                                      // rate styles


#define MAXREACTANT 2
#define MAXPRODUCT 3

// HYDHEL H.1 3.1.8: H + H+ -> H+ + H charge exchange cross-section.
// Janev/Smirnov fit. Natural-log polynomial:
//   ln(sigma[cm^2]) = sum_{n=0..8} a_n * (ln E_lab[eV])^n
// Valid range: 0.1 eV <= E_lab <= 5e5 eV.
// Returns sigma in m^2; E_lab is the projectile-rest-frame energy in eV.
static inline double sigma_cx_hh_m2(double E_lab_eV) {
  static const double a[9] = {
    -3.274123792568e+01, -8.916456579806e-02, -3.016990732025e-02,
     9.205482406462e-03,  2.400266568315e-03, -1.927122311323e-03,
     3.654750340106e-04, -2.788866460622e-05,  7.422296363524e-07
  };
  const double E = (E_lab_eV < 0.1) ? 0.1
                : (E_lab_eV > 5e5) ? 5e5
                : E_lab_eV;
  const double aL = std::log(E);
  double s = a[8];
  for (int n = 7; n >= 0; --n) s = s * aL + a[n];
  return std::exp(s) * 1.0e-4;   // cm^2 -> m^2
}

// AMJUEL H.4 2.1.5: e + D -> D+ + 2e effective ionization rate (Sawada/Fujimoto).
// Double-polynomial in (ln Te, ln(ne_cm3/1e8)):
//   ln(<sigma v>[cm^3/s]) = sum_{i,j=0..8} c[i,j] * (ln Te[eV])^i * (ln(ne[cm^-3]/1e8))^j
// Valid: Te in [0.1, 2e4] eV, ne in [1e8, 1e16] cm^-3.
// Returns log10(<sv> [cm^3/s]). Replaces ADAS SCD89 (~2x lower across 1-100 eV).
static inline double log10_sigmav_ioniz_amjuel_cm3s(double Te_eV, double ne_m3) {
  static const double c[9][9] = {
    {-3.248025330340e+01, -5.440669186583e-02,  9.048888225109e-02,
     -4.054078993576e-02,  8.976513750477e-03, -1.060334011186e-03,
      6.846238436472e-05, -2.242955329604e-06,  2.890437688072e-08},
    { 1.425332391510e+01, -3.594347160760e-02, -2.014729121556e-02,
      1.039773615730e-02, -1.771792153042e-03,  1.237467264294e-04,
     -3.130184159149e-06, -3.051994601527e-08,  1.888148175469e-09},
    {-6.632235026785e+00,  9.255558353174e-02, -5.580210154625e-03,
     -5.902218748238e-03,  1.295609806553e-03, -1.056721622588e-04,
      4.646310029498e-06, -1.479612391848e-07,  2.852251258320e-09},
    { 2.059544135448e+00, -7.562462086943e-02,  1.519595967433e-02,
      5.803498098354e-04, -3.527285012725e-04,  3.201533740322e-05,
     -1.835196889733e-06,  9.474014343303e-08, -2.342505583774e-09},
    {-4.425370331410e-01,  2.882634019199e-02, -7.285771485050e-03,
      4.643389885987e-04,  1.145700685235e-06,  8.493662724988e-07,
     -1.001032516512e-08, -1.476839184318e-08,  6.047700368169e-10},
    { 6.309381861496e-02, -5.788686535780e-03,  1.507382955250e-03,
     -1.201550548662e-04,  6.574487543511e-06, -9.678782818849e-07,
      5.176265845225e-08,  1.291551676860e-09, -9.685157340473e-11},
    {-5.620091829261e-03,  6.329105568040e-04, -1.527777697951e-04,
      8.270124691336e-06,  3.224101773605e-08,  4.377402649057e-08,
     -2.622921686955e-09, -2.259663431436e-10,  1.161438990709e-11},
    { 2.812016578355e-04, -3.564132950345e-05,  7.222726811078e-06,
      1.433018694347e-07, -1.097431215601e-07,  7.789031791949e-09,
     -4.197728680251e-10,  3.032260338723e-11, -8.911076930014e-13},
    {-6.011143453374e-06,  8.089651265488e-07, -1.186212683668e-07,
     -2.381080756307e-08,  6.271173694534e-09, -5.483010244930e-10,
      3.064611702159e-11, -1.355903284487e-12,  2.935080031599e-14}
  };
  const double Te = (Te_eV < 0.1) ? 0.1 : (Te_eV > 2e4 ? 2e4 : Te_eV);
  const double ne_cm3 = ne_m3 * 1.0e-6;
  const double ne_norm = ne_cm3 / 1.0e8;
  const double ne_c = (ne_norm < 1.0) ? 1.0 : (ne_norm > 1.0e8 ? 1.0e8 : ne_norm);
  const double lT = std::log(Te);
  const double lD = std::log(ne_c);
  double sum = 0.0;
  for (int i = 8; i >= 0; --i) {
    double row = c[i][8];
    for (int j = 7; j >= 0; --j) row = row * lD + c[i][j];
    sum = sum * lT + row;
  }
  return sum / 2.302585092994046;
}

// HYDHEL H.3 3.1.8: D + D+ -> D+ + D charge exchange RATE coefficient.
// Double-polynomial:
//   ln(<sigma v>[cm^3/s]) = sum_{i,j=0..8} c[i,j] * (ln Ti[eV])^i * (ln E_atom[eV])^j
// Replaces ADAS CCD89 (which is for impurity-hydrogen CX, not H-H+ resonant)
// for the D--D+ EXCHANGE channel. ADAS CCD is 10-20x lower across 1-200 eV.
// Returns log10(<sv> [cm^3/s]) for compatibility with computeReactionLambda.
static inline double log10_sigmav_cx_hh_cm3s(double Ti_eV, double E_atom_eV) {
  static const double c[9][9] = {
    {-1.831670498376e+01,  1.650239332070e-01,  5.025740610454e-02,
      5.288358515136e-03, -2.437122342843e-03, -4.461891214720e-04,
      1.731631548110e-04, -1.588434781959e-05,  4.482291414386e-07},
    { 2.143624996483e-01, -1.067658289373e-01, -5.304993033743e-03,
      8.289383645942e-03, -9.698773663345e-05, -4.470180279338e-04,
      7.944326905066e-05, -5.303688417551e-06,  1.235167254501e-07},
    { 5.139117192662e-02,  9.536923957409e-03, -1.306075129405e-02,
     -1.033166370333e-03,  1.280464204775e-03, -8.453294908907e-05,
     -3.040874906105e-05,  4.747888095498e-06, -1.923953750574e-07},
    {-9.896180369559e-04,  6.315097684976e-03,  2.655464630308e-03,
     -1.365781346175e-03, -1.859939123743e-04,  1.237942304972e-04,
     -1.588253432932e-05,  6.603560345800e-07, -1.970606344918e-09},
    {-2.495327546080e-03, -1.265503371044e-03,  7.569269700468e-04,
      2.756946036257e-04, -1.107375149384e-04, -7.217379426085e-06,
      5.769971321188e-06, -6.717311113584e-07,  2.440961351104e-08},
    {-2.417046684097e-05, -6.945512319613e-05, -2.956984088728e-04,
      2.318277483195e-05,  3.704494397140e-05, -6.066558692480e-06,
     -4.951573401626e-07,  1.437520597154e-07, -6.998724470004e-09},
    { 1.177406072793e-04,  3.698501620365e-05,  3.424317896619e-05,
     -9.815693511794e-06, -4.285719813022e-06,  1.169257650609e-06,
     -4.968953461875e-10, -1.618948982477e-08,  9.440094842562e-10},
    {-1.483036457978e-05, -3.348172574417e-06, -1.527018819072e-06,
      8.362050692462e-07,  2.058392726953e-07, -7.463594884928e-08,
      5.924370389093e-10,  1.078208689229e-09, -6.619767848464e-11},
    { 5.351909441226e-07,  9.728230870242e-08,  1.676354786072e-08,
     -2.237567830699e-08, -3.081685803820e-09,  1.450862501121e-09,
      4.434231893204e-11, -3.324377862622e-11,  1.935019679501e-12}
  };
  const double Ti  = (Ti_eV     < 0.1) ? 0.1 : (Ti_eV     > 1e5 ? 1e5 : Ti_eV);
  const double Ea  = (E_atom_eV < 0.1) ? 0.1 : (E_atom_eV > 1e5 ? 1e5 : E_atom_eV);
  const double lT  = std::log(Ti);
  const double lE  = std::log(Ea);
  double sum = 0.0;
  for (int i = 8; i >= 0; --i) {
    double row = c[i][8];
    for (int j = 7; j >= 0; --j) row = row * lE + c[i][j];
    sum = sum * lT + row;
  }
  return sum / 2.302585092994046;
}
#define MAXCOEFF 10              // Janev polynomials use up to 9 coeffs (b0..b8)
#define MAXLINE 1024
#define DELTALIST 16
/* ---------------------------------------------------------------------- */

FixVolumeChemAdas::FixVolumeChemAdas(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
    // fix ID volume/chem/adas <nevery> <species|Z> <reactions_file> [plasma <TeVar> <NeVar>]
    //   <species|Z>  element symbol (e.g. "C", "W") OR numeric atomic number.
    //                When given as a symbol, the ADAS file is auto-located
    //                under ${OPENEDGE_ROOT}/database/adas/ADAS_Rates_<Z>.h5.

  if (narg < 5)     error->all(FLERR,"Illegal fix volume/chem/adas command (need: nevery <species|Z> reactions_file [plasma TeVar NeVar])");
    nevery = atoi(arg[2]);

    // Accept arg[3] as either a numeric Z or an element symbol.
    {
      const std::string tok(arg[3]);
      bool numeric = !tok.empty();
      for (char c : tok) if (c < '0' || c > '9') { numeric = false; break; }
      if (numeric) {
        atomic_number = atoi(arg[3]);
      } else {
        atomic_number = element_to_z(tok);
        if (atomic_number < 0) {
          std::string msg = "fix volume/chem/adas: unknown element '" + tok + "'";
          error->all(FLERR, msg.c_str());
        }
      }
    }

    // per-cell array for aveflag = 1 case

    nlist = maxlist = 0;
    rlist = NULL;

    // Reactions-list argument can be:
    //   * literal path (contains '/' or ends in .reactions) -> used as-is
    //   * element symbol (e.g. "D", "C") -> first try processes.h5
    //     /volume/reactions/<elem>/catalog, fall back to
    //     ${OPENEDGE_ROOT}/database/adas/reactions/<elem>.reactions
    //   * "auto" sentinel -> same as passing the arg[3] element here
    std::string reactions_path;
    {
      const std::string tok4(arg[4]);
      std::string elem_for_catalog;
      bool is_literal_path = tok4.find('/') != std::string::npos ||
          (tok4.size() >= 10 &&
           tok4.compare(tok4.size() - 10, 10, ".reactions") == 0);

      if (!is_literal_path) {
        elem_for_catalog = (tok4 == "auto") ? std::string(arg[3]) : tok4;
        // Normalise isotopes: processes.h5 stores hydrogen as "d" key.
        // element_to_z already maps D/T/H -> 1; reuse for the lookup.
        int zc = element_to_z(elem_for_catalog);
        if (zc > 0) {
          static const std::map<int, std::string> z_to_sym = {
            {1,"d"}, {2,"he"}, {3,"li"}, {4,"be"}, {5,"b"}, {6,"c"},
            {7,"n"}, {8,"o"}, {10,"ne"}, {18,"ar"}, {26,"fe"}, {36,"kr"},
            {42,"mo"}, {54,"xe"}, {73,"ta"}, {74,"w"},
          };
          auto it = z_to_sym.find(zc);
          if (it != z_to_sym.end()) elem_for_catalog = it->second;
          else elem_for_catalog.clear();
        } else elem_for_catalog.clear();
      }

      bool loaded_from_processes = false;
      if (!elem_for_catalog.empty()) {
        std::string processes_path = resolve_processes_file();
        if (!processes_path.empty()) {
          ProcessLibrary lib_r;
          lib_r.open(processes_path, world, error);
          std::string catalog;
          if (lib_r.is_open() &&
              lib_r.load_reactions_catalog(elem_for_catalog, catalog)) {
            // Write the catalog to a rank-0 tempfile, then readfile() it
            // so every rank parses the same content through the existing
            // text parser. The tempfile is removed after readfile() returns.
            char tmpl[] = "/tmp/openedge_reactions_XXXXXX";
            int fd = -1;
            if (comm->me == 0) {
              fd = mkstemp(tmpl);
              if (fd >= 0) {
                ssize_t nw = write(fd, catalog.data(), catalog.size());
                (void) nw;
                close(fd);
              }
            }
            MPI_Bcast(tmpl, sizeof(tmpl), MPI_CHAR, 0, world);
            std::vector<char> buf(tmpl, tmpl + strlen(tmpl) + 1);
            readfile(buf.data());
            if (comm->me == 0) unlink(tmpl);
            loaded_from_processes = true;
            if (comm->me == 0 && screen)
              fprintf(screen,
                "  reactions catalog: %s (processes.h5 "
                "/volume/reactions/%s/catalog)\n",
                elem_for_catalog.c_str(), elem_for_catalog.c_str());
          }
        }
      }

      if (!loaded_from_processes) {
        const std::string tok = (tok4 == "auto") ? std::string(arg[3]) : tok4;
        reactions_path = resolve_reactions_file(tok, error);
        std::vector<char> buf(reactions_path.begin(), reactions_path.end());
        buf.push_back('\0');
        readfile(buf.data());
      }
    }
    check_duplicate();

    // Read ADAS rate data from the consolidated database/processes.h5
    // under /volume/rates/<cls>/<elem>/ + /volume/thresholds/.  The
    // legacy ADAS_Rates_<Z>.h5 fallback was removed in Phase 2b; if
    // processes.h5 is missing or doesn't contain the requested element,
    // volume/chem/adas errors out at init.
    int iarg = 5;
    {
      // Derive the canonical lowercase element symbol for processes.h5
      // group lookup by going through Z (element_to_z already normalizes
      // D/T -> 1, so we reuse its output unconditionally).  This avoids
      // the isotope/element mismatch between deck input ("D") and the
      // processes.h5 key ("h").
      std::string elem_sym;
      {
        static const std::map<int, std::string> z_to_sym = {
          {1,"h"}, {2,"he"}, {3,"li"}, {4,"be"}, {5,"b"}, {6,"c"},
          {7,"n"}, {8,"o"}, {10,"ne"}, {18,"ar"}, {26,"fe"}, {36,"kr"},
          {42,"mo"}, {54,"xe"}, {73,"ta"}, {74,"w"},
        };
        auto it = z_to_sym.find(atomic_number);
        if (it != z_to_sym.end()) elem_sym = it->second;
      }

      bool loaded_from_processes = false;
      std::string processes_path = resolve_processes_file();
      if (!elem_sym.empty() && !processes_path.empty()) {
        ProcessLibrary lib;
        lib.open(processes_path, world, error);
        if (lib.is_open()) {
          RateData &rd = materials_rate_data[atomic_number];
          auto try_cls = [&](const char *cls,
                             std::vector<double> &coef,
                             std::vector<double> &gT,
                             std::vector<double> &gD,
                             int &nQ, int &nT, int &nD) -> bool {
            return lib.load_rate(cls, elem_sym, coef, gT, gD, nQ, nT, nD);
          };
          bool scd = try_cls("scd", rd.ion_coeff, rd.gridT_ion, rd.gridD_ion,
                             rd.ion_nQ, rd.ion_nT, rd.ion_nD);
          bool acd = try_cls("acd", rd.rec_coeff, rd.gridT_rec, rd.gridD_rec,
                             rd.rec_nQ, rd.rec_nT, rd.rec_nD);
          try_cls("ccd", rd.cx_coeff, rd.gridT_cx, rd.gridD_cx,
                  rd.cx_nQ, rd.cx_nT, rd.cx_nD);
          try_cls("plt", rd.plt_coeff, rd.gridT_plt, rd.gridD_plt,
                  rd.plt_nQ, rd.plt_nT, rd.plt_nD);
          try_cls("prb", rd.prb_coeff, rd.gridT_prb, rd.gridD_prb,
                  rd.prb_nQ, rd.prb_nT, rd.prb_nD);
          lib.load_ionization_potential(elem_sym, rd.ion_potential);
          loaded_from_processes = scd && acd;
          if (loaded_from_processes) {
            rd.Atomic_Number = std::vector<double>(1, (double)atomic_number);
            if (comm->me == 0)
              printf("Reading ADAS data for %s (Z=%d) from %s "
                     "(/volume/rates/)\n",
                     elem_sym.c_str(), atomic_number,
                     processes_path.c_str());
          }
        }
      }

      if (!loaded_from_processes) {
        std::string msg;
        if (processes_path.empty()) {
          msg = "fix volume/chem/adas: database/processes.h5 not found. "
                "Build it with database/ingest/build_processes_h5.py.";
        } else {
          msg = "fix volume/chem/adas: element '" + elem_sym +
                "' (Z=" + std::to_string(atomic_number) + ") not found in "
                + processes_path + " under /volume/rates/";
        }
        error->all(FLERR, msg.c_str());
      }
    }

    // //
    maxgrid = 0;
    reactions = NULL;
    list_ij = NULL;

    tally_reactions = new bigint[nlist];
    tally_reactions_all = new bigint[nlist];
    tally_flag = 0;
    nreact_one = nreact_running = 0;
    rng_adas = nullptr;

  // Te/ne come from the per-particle plasma cache (populated by
  // update.cpp from a configured plasma_compute / plasma_fix). For a
  // uniform synthetic plasma in test cases, use `fix background
  // constant ...` — the pcache picks it up automatically and the
  // sheath Boltzmann correction (when active) flows through.

  // --- Optional Mode A (EIRENE-semantics) keywords ---
  //   mode neutral                         -> delete particle on ionization
  //   source_species <sp1> [sp2] ...       -> species to count for exhaustion
  //   stop_on_exhaust yes|no               -> halt run when source species = 0
  while (iarg < narg) {
    if (strcmp(arg[iarg], "mode") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: mode requires a value (kinetic|neutral)");
      if (strcmp(arg[iarg+1], "neutral") == 0) eirene_mode = 1;
      else if (strcmp(arg[iarg+1], "kinetic") == 0) eirene_mode = 0;
      else error->all(FLERR, "fix volume/chem/adas: unknown mode (use kinetic|neutral)");
      iarg += 2;
    } else if (strcmp(arg[iarg], "rate_cache") == 0) {
      // rate_cache cell|particle
      //   particle (default) — recompute reaction rates for every particle.
      //   cell                — compute rates once per (cell, species) using
      //                         the first encountered particle's pcache and
      //                         reuse for all subsequent particles of that
      //                         species in the same cell. ~10x cheaper for
      //                         dense impurities; identical physics in cells
      //                         where Te/ne don't vary across particles
      //                         (i.e., away from sheath cells).
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: rate_cache requires cell|particle");
      if (strcmp(arg[iarg+1], "cell") == 0) rate_cache_mode = 1;
      else if (strcmp(arg[iarg+1], "particle") == 0) rate_cache_mode = 0;
      else error->all(FLERR, "fix volume/chem/adas: rate_cache must be cell|particle");
      iarg += 2;
    } else if (strcmp(arg[iarg], "source_species") == 0) {
      // consume species names until the next keyword or end of args
      int j = iarg + 1;
      while (j < narg &&
             strcmp(arg[j], "mode") != 0 &&
             strcmp(arg[j], "rate_cache") != 0 &&
             strcmp(arg[j], "stop_on_exhaust") != 0 &&
             strcmp(arg[j], "source_species") != 0 &&
             strcmp(arg[j], "units") != 0 &&
             strcmp(arg[j], "output") != 0 &&
             strcmp(arg[j], "exhaust_threshold") != 0 &&
             strcmp(arg[j], "ionization") != 0 &&
             strcmp(arg[j], "recombination") != 0 &&
             strcmp(arg[j], "cx") != 0 &&
             strcmp(arg[j], "dissociation") != 0 &&
             strcmp(arg[j], "volume_source") != 0) j++;
      nsrc_species = j - (iarg + 1);
      if (nsrc_species <= 0)
        error->all(FLERR, "fix volume/chem/adas: source_species needs >=1 species name");
      src_species_names = new char*[nsrc_species];
      for (int k = 0; k < nsrc_species; k++) {
        int n = strlen(arg[iarg + 1 + k]) + 1;
        src_species_names[k] = new char[n];
        strcpy(src_species_names[k], arg[iarg + 1 + k]);
      }
      iarg = j;
    } else if (strcmp(arg[iarg], "stop_on_exhaust") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: stop_on_exhaust requires yes|no");
      if (strcmp(arg[iarg+1], "yes") == 0) stop_on_exhaust = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) stop_on_exhaust = 0;
      else error->all(FLERR, "fix volume/chem/adas: stop_on_exhaust must be yes|no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "exhaust_threshold") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: exhaust_threshold requires <N>");
      exhaust_threshold = ATOBIGINT(arg[iarg+1]);
      if (exhaust_threshold < 0)
        error->all(FLERR, "fix volume/chem/adas: exhaust_threshold must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: units requires counts|rate|eirene");
      if (strcmp(arg[iarg+1], "counts") == 0) {
        tally_units = TALLY_COUNTS;
        iarg += 2;
      } else if (strcmp(arg[iarg+1], "rate") == 0) {
        tally_units = TALLY_RATE;
        iarg += 2;
      } else if (strcmp(arg[iarg+1], "batch") == 0) {
        // Batch (EIRENE-style MC): `units batch <N_trajectories> <R_puff>`
        if (iarg + 3 >= narg)
          error->all(FLERR, "fix volume/chem/adas: units batch requires N R_puff");
        batch_N      = atoi(arg[iarg+2]);
        batch_R_puff = atof(arg[iarg+3]);
        if (batch_N <= 0 || batch_R_puff <= 0.0)
          error->all(FLERR, "fix volume/chem/adas: units batch needs N>0 and R_puff>0");
        tally_units = TALLY_BATCH;
        iarg += 4;
      } else if (strcmp(arg[iarg+1], "batch_fix") == 0) {
        // EIRENE-batch auto-scaled from a paired emit fix:
        //   `units batch_fix <emit_fix_id> <R_puff>`
        if (iarg + 3 >= narg)
          error->all(FLERR, "fix volume/chem/adas: units batch_fix requires <fix_id> R_puff");
        int n = strlen(arg[iarg+2]) + 1;
        batch_fix_id = new char[n];
        strcpy(batch_fix_id, arg[iarg+2]);
        batch_R_puff = atof(arg[iarg+3]);
        if (batch_R_puff <= 0.0)
          error->all(FLERR, "fix volume/chem/adas: units batch_fix needs R_puff > 0");
        tally_units = TALLY_BATCH_FIX;
        iarg += 4;
      } else {
        error->all(FLERR, "fix volume/chem/adas: units must be counts|rate|batch|batch_fix");
      }
    } else if (strcmp(arg[iarg], "output") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: output requires summary|detailed");
      if (strcmp(arg[iarg+1], "summary") == 0) output_mode = OUT_SUMMARY;
      else if (strcmp(arg[iarg+1], "detailed") == 0) output_mode = OUT_DETAILED;
      else error->all(FLERR, "fix volume/chem/adas: output must be summary|detailed");
      iarg += 2;
    } else if (strcmp(arg[iarg], "volume_source") == 0) {
      // volume_source <id>
      //   id = compute or fix that exposes cell-indexed plasma fields.
      //   Resolved at init() to either a ComputePlasmaFields (for the
      //   `compute … plasma/fields …` path) or a FixBackground (for the
      //   `fix … background …` path). Activates volume recombination
      //   spawning in end_of_step_no_average. Required to make Mode A
      //   neutral runs see recombination at all -- attempt() can't fire
      //   it (no kinetic D+ in the cell list).
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix volume/chem/adas: volume_source requires <id>");
      const int n = strlen(arg[iarg+1]) + 1;
      volume_source_id = new char[n];
      strcpy(volume_source_id, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ionization") == 0 ||
               strcmp(arg[iarg], "recombination") == 0 ||
               strcmp(arg[iarg], "cx") == 0 ||
               strcmp(arg[iarg], "dissociation") == 0) {
      if (iarg + 1 >= narg) {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "fix volume/chem/adas: %s requires yes|no", arg[iarg]);
        error->all(FLERR, msg);
      }
      int v;
      if (strcmp(arg[iarg+1], "yes") == 0) v = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) v = 0;
      else {
        char msg[160];
        snprintf(msg, sizeof(msg),
                 "fix volume/chem/adas: %s must be yes|no", arg[iarg]);
        error->all(FLERR, msg);
      }
      if (strcmp(arg[iarg], "ionization") == 0)        chan_ionization = v;
      else if (strcmp(arg[iarg], "recombination") == 0) chan_recombination = v;
      else if (strcmp(arg[iarg], "cx") == 0)           chan_cx = v;
      else                                              chan_dissociation = v;
      iarg += 2;
    } else {
      char msg[160];
      snprintf(msg, sizeof(msg), "fix volume/chem/adas: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  // Per-grid source tally layout depends on `output` keyword:
  //   OUT_SUMMARY  (default): 6 cols {Sp, Sm_x, Sm_y, Sm_z, Qe, Qi},
  //                 signed plasma-frame moments, ready for fluid coupling.
  //   OUT_DETAILED (diagnostic): 20 cols = {count, m vx, m vy, m vz, 0.5 m v^2}
  //                 x {ioniz, recomb, CX, dissoc}, unsigned raw event tally
  //                 (legacy layout for per-reaction debugging).
  // See docs/neutral_plasma_coupling/main.tex for per-reaction formulas.
  per_grid_flag      = 1;
  size_per_grid_cols = (output_mode == OUT_SUMMARY) ? 6 : 20;
  per_grid_freq      = 1;     // tally is updated every step; dump_grid
                              //   does  (nevery % per_grid_freq)  so this
                              //   MUST be non-zero (SIGFPE otherwise).
  array_grid         = NULL;
  maxgrid_src        = 0;

}

/* ---------------------------------------------------------------------- */

FixVolumeChemAdas::~FixVolumeChemAdas()
{
  if (copymode) return;


  delete [] tally_reactions;
  delete [] tally_reactions_all;

  if (rlist) {
    for (int i = 0; i < maxlist; i++) {
      for (int j = 0; j < rlist[i].nreactant; j++)
        delete [] rlist[i].id_reactants[j];
      for (int j = 0; j < rlist[i].nproduct; j++)
        delete [] rlist[i].id_products[j];
      delete [] rlist[i].id_reactants;
      delete [] rlist[i].id_products;
      delete [] rlist[i].reactants;
      delete [] rlist[i].products;
      delete [] rlist[i].coeff;
      delete [] rlist[i].id;
    }
  }
  memory->destroy(rlist);

  memory->destroy(reactions);
  memory->destroy(list_ij);

memory->destroy(array_grid);
delete rng_adas;

if (src_species_names) {
  for (int k = 0; k < nsrc_species; k++) delete [] src_species_names[k];
  delete [] src_species_names;
}

delete [] batch_fix_id;
delete [] volume_source_id;



}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdas::reset_tally()
{
  if (array_grid && maxgrid_src > 0)
    memset(&array_grid[0][0], 0,
           sizeof(double) * maxgrid_src * size_per_grid_cols);
}

/* ---------------------------------------------------------------------- */

int FixVolumeChemAdas::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdas::init()
{

  tally_flag = 0;
  nreact_one = nreact_running = 0;
  for (int i = 0; i < nlist; i++) tally_reactions[i] = 0;

  // convert species IDs to species indices
  // flag reactions as active/inactive depending on whether all species exist
  // mark recombination reactions inactive if recombflag_user = 0

  // Per-channel tally so the init printout shows exactly which reactions
  // survived species-matching + channel-toggle filtering.
  int nI_active = 0, nI_total = 0;
  int nR_active = 0, nR_total = 0;
  int nE_active = 0, nE_total = 0;
  int nD_active = 0, nD_total = 0;
  // Up to 8 "missing species" tags kept for the diagnostic print.
  std::vector<std::string> missing_species_samples;
  missing_species_samples.reserve(8);
  auto note_missing = [&](const char *name) {
    if (missing_species_samples.size() >= 8) return;
    for (const auto &s : missing_species_samples) if (s == name) return;
    missing_species_samples.emplace_back(name);
  };

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    r->active = 1;

    for (int i = 0; i < r->nreactant; i++) {
      r->reactants[i] = particle->find_species(r->id_reactants[i]);
      if (r->reactants[i] < 0) {
        r->active = 0;
        note_missing(r->id_reactants[i]);
      }
    }

    for (int i = 0; i < r->nproduct; i++) {
      r->products[i] = particle->find_species(r->id_products[i]);
      if (r->products[i] < 0) {
        r->active = 0;
        note_missing(r->id_products[i]);
      }
    }

    // Channel-toggle filter: disable otherwise-valid reactions whose type
    // the user turned off via `cx no`, `recombination no`, etc.
    if (r->active) {
      if      (r->type == IONIZATION    && !chan_ionization)    r->active = 0;
      else if (r->type == RECOMBINATION && !chan_recombination) r->active = 0;
      else if (r->type == EXCHANGE      && !chan_cx)            r->active = 0;
      else if (r->type == DISSOCIATION  && !chan_dissociation)  r->active = 0;
    }

    switch (r->type) {
      case IONIZATION:    nI_total++; if (r->active) nI_active++; break;
      case RECOMBINATION: nR_total++; if (r->active) nR_active++; break;
      case EXCHANGE:      nE_total++; if (r->active) nE_active++; break;
      case DISSOCIATION:  nD_total++; if (r->active) nD_active++; break;
      default: break;
    }
  }

  if (comm->me == 0) {
    auto report = [&](const char *label, int active, int total, int enabled) {
      if (total == 0) return;
      const int skipped = total - active;
      if (!enabled)
        printf("[volume/chem/adas] %-14s: disabled by toggle (%d in file)\n",
               label, total);
      else
        printf("[volume/chem/adas] %-14s: %d active, %d skipped (of %d)\n",
               label, active, skipped, total);
    };
    report("ionization",    nI_active, nI_total, chan_ionization);
    report("recombination", nR_active, nR_total, chan_recombination);
    report("cx",            nE_active, nE_total, chan_cx);
    report("dissociation",  nD_active, nD_total, chan_dissociation);
    if (!missing_species_samples.empty()) {
      printf("[volume/chem/adas] missing species (first %zu): ",
             missing_species_samples.size());
      for (size_t k = 0; k < missing_species_samples.size(); k++)
        printf("%s%s", k ? "," : "", missing_species_samples[k].c_str());
      printf("\n");
    }
    auto it = materials_rate_data.find(atomic_number);
    const bool have = (it != materials_rate_data.end());
    printf("[volume/chem/adas] output %-8s (%d cols)  "
           "ADAS tables: %s %s %s %s %s %s\n",
           output_mode == OUT_SUMMARY ? "summary" : "detailed",
           size_per_grid_cols,
           (have && it->second.ion_nQ > 0) ? "SCD" : "-",
           (have && it->second.rec_nQ > 0) ? "ACD" : "-",
           (have && it->second.cx_nQ  > 0) ? "CCD" : "-",
           (have && it->second.plt_nQ > 0) ? "PLT" : "-",
           (have && it->second.prb_nQ > 0) ? "PRB" : "-",
           (have && !it->second.ion_potential.empty()) ? "IP" : "-");
  }

  // count possible active reactions for each species pair
  // include J,I reactions in I,J list and vice versa
  // this allows collision pair I,J to be in either order in Collide

  memory->destroy(reactions);
  int nspecies = particle->nspecies;
  reactions = memory->create(reactions,nspecies,
                             "react/bird:reactions");

  for (int i = 0; i < nspecies; i++)
      reactions[i].n = 0;

  int n = 0;
  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].n++;
    n++;
  }

  // allocate list_IJ = contiguous list of reactions for each IJ pair

  memory->destroy(list_ij);
  memory->create(list_ij,n,"react/bird:list_ij");

  // reactions[i][j].list = pointer into full list_ij vector

  int offset = 0;
  for (int i = 0; i < nspecies; i++){
      reactions[i].list = &list_ij[offset];
      offset += reactions[i].n;
  }
    

  // reactions[i][j].list = indices of reactions for each species pair
  // include J,I reactions in I,J list and vice versa

  for (int i = 0; i < nspecies; i++)
      reactions[i].n = 0;

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].list[reactions[i].n++] = m;
  }

  // modify Arrhenius coefficients for TCE model
  // C1,C2 Bird 94, p 127
  // initflag logic insures only done once per reaction

  Particle::Species *species = particle->species;

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    if (r->initflag) continue;
    r->initflag = 1;

    int isp = r->reactants[0];
  }

// Plasma cache is the only source path; the gate check fires at the
// first end_of_step (here is too early — update->init() runs after).

// Allocate per-cell source-tally output (array_grid, 6 or 20 cols depending
// on output_mode) up front so dumps that fire at step 0
// (dump_modify ... first yes) see a valid buffer.
if (grid->maxlocal > maxgrid_src) {
  maxgrid_src = grid->maxlocal;
  memory->grow(array_grid, maxgrid_src, size_per_grid_cols,
               "volume/chem/adas:array_grid(src)");
}
if (maxgrid_src > 0) {
  memset(&array_grid[0][0], 0,
         sizeof(double) * maxgrid_src * size_per_grid_cols);
}

// initialize SPARTA RNG (same pattern as fix_coll_nanbu)
if (!rng_adas) {
  rng_adas = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  rng_adas->reset(seed, comm->me, 100);
}

// TALLY_BATCH_FIX: resolve the emit-fix ID to a modify->fix index so we can
// query its cumulative ntotal at tally time. We only need the index; the fix
// pointer is looked up per-step via modify->fix[idx] (the index is stable
// for the lifetime of the run once both fixes are defined).
if (tally_units == TALLY_BATCH_FIX) {
  batch_fix_idx = modify->find_fix(batch_fix_id);
  if (batch_fix_idx < 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "fix volume/chem/adas: units batch_fix source fix '%s' not found",
             batch_fix_id);
    error->all(FLERR, msg);
  }
}

// Mode A: resolve source_species names to species indices.
// Runs here (not ctor) because `species` commands are parsed before `fix`
// in the usual input deck but find_species must be callable either way.
source_species.clear();
for (int k = 0; k < nsrc_species; k++) {
  int sp = particle->find_species(src_species_names[k]);
  if (sp < 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "fix volume/chem/adas: source_species '%s' not found",
             src_species_names[k]);
    error->all(FLERR, msg);
  }
  source_species.push_back(sp);
}
if (stop_on_exhaust && source_species.empty() && eirene_mode && comm->me == 0) {
  error->warning(FLERR,
    "fix volume/chem/adas: stop_on_exhaust requested without source_species; "
    "run will rely on SPARTA's built-in nglobal==0 termination");
}

// Volume recombination: resolve plasma source handle and build the list
// of active recombination reactions whose product is a defined species.
// Inactive in non-Mode-A runs (attempt() handles kinetic-ion recomb) and
// when chan_recombination = no.
volume_source_cidx = -1;
volume_source_fidx = -1;
rec_ridx.clear();
rec_product_isp.clear();
nrec_active = 0;
if (volume_source_id) {
  volume_source_cidx = modify->find_compute(volume_source_id);
  if (volume_source_cidx < 0) {
    volume_source_fidx = modify->find_fix(volume_source_id);
    if (volume_source_fidx < 0) {
      char msg[160];
      snprintf(msg, sizeof(msg),
               "fix volume/chem/adas: volume_source '%s' not found "
               "(expected compute plasma/fields or fix background)",
               volume_source_id);
      error->all(FLERR, msg);
    }
  }
  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    if (r->type != RECOMBINATION) continue;
    if (r->nproduct < 1) continue;
    rec_ridx.push_back(m);
    rec_product_isp.push_back(r->products[0]);
  }
  nrec_active = static_cast<int>(rec_ridx.size());
  if (eirene_mode && nrec_active == 0 && chan_recombination && comm->me == 0) {
    error->warning(FLERR,
      "fix volume/chem/adas: volume_source set but no active recombination "
      "reaction found in the reactions file -- volume source is a no-op");
  }
}

}

/* ---------------------------------------------------------------------- */

void FixVolumeChemAdas::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // Require the per-particle plasma cache. Activated by any one of:
  // sheath, GCA, or `global bfield_compute`. For uniform test cases
  // use `fix background constant ...` plus one of those activators.
  if (!update->plasma_cache_flag) {
    error->all(FLERR,
      "fix volume/chem/adas: per-particle plasma cache not active — "
      "configure sheath / GCA / global bfield_compute so update.cpp "
      "populates Te/ne at particle positions");
  }

  nreact_one = 0;
  if (!particle->sorted) particle->sort();
  end_of_step_no_average();
  nreact_running += nreact_one;
}

/* ----------------------------------------------------------------------
   end-of-run summary: cumulative per-type reaction tally, mirrors the
   surf_react tally block printed at the end of each `run`.
---------------------------------------------------------------------- */

void FixVolumeChemAdas::post_run()
{
  std::vector<bigint> local_tally(nlist, 0), global_tally(nlist, 0);
  for (int i = 0; i < nlist; i++) local_tally[i] = tally_reactions[i];
  MPI_Allreduce(local_tally.data(), global_tally.data(), nlist,
                MPI_SPARTA_BIGINT, MPI_SUM, world);

  if (comm->me != 0) return;

  bigint nI = 0, nR = 0, nE = 0, nD = 0, nall = 0;
  for (int i = 0; i < nlist; i++) {
    nall += global_tally[i];
    if (rlist[i].type == IONIZATION) nI += global_tally[i];
    else if (rlist[i].type == RECOMBINATION) nR += global_tally[i];
    else if (rlist[i].type == EXCHANGE) nE += global_tally[i];
    else if (rlist[i].type == DISSOCIATION) nD += global_tally[i];
  }

  auto print_block = [&](FILE *fp) {
    if (!fp) return;
    fprintf(fp,
      "Chem/ADAS reaction tallies:\n"
      "  id %s Z=%d #-of-reactions %d\n"
      "    reaction all: " BIGINT_FORMAT "\n"
      "    ioniz: " BIGINT_FORMAT "\n"
      "    recomb: " BIGINT_FORMAT "\n"
      "    CX: " BIGINT_FORMAT "\n"
      "    dissoc: " BIGINT_FORMAT "\n",
      id, atomic_number, nlist, nall, nI, nR, nE, nD);
  };
  print_block(screen);
  print_block(logfile);
}


/* ----------------------------------------------------------------------
   current thermal temperature is calculated on a per-cell basis
---------------------------------------------------------------------- */

void FixVolumeChemAdas::end_of_step_no_average()
{
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  deferred_particles.clear();

  // TALLY_BATCH_FIX: pull current cumulative N from the paired emit fix.
  // One allreduce per chem nevery step (compute_vector does the reduce
  // internally). Cached N_batch stays constant within this call so every
  // event tallied in attempt() uses the same weight.
  if (tally_units == TALLY_BATCH_FIX && batch_fix_idx >= 0) {
    Fix *fe = modify->fix[batch_fix_idx];
    double n_global = fe->compute_vector(1);   // FixEmit ntotal (global sum)
    batch_N_cached = static_cast<bigint>(n_global);
  }

  // Refresh the per-particle weight handle (fix particle/weight). In RATE
  // mode attempt() weights each event by this real-particle weight instead
  // of the global fnum that the RATE normalization re-applies, so that
  // flux-scaled emission (fix surface/emit/source nlaunch) produces the
  // correct source magnitude. ewhich can change as custom attrs realloc, so
  // look it up fresh each step. -1 => attribute absent (no-op weighting).
  pweight_index  = particle->find_custom((char *) "pweight");
  pweight_ewhich = (pweight_index >= 0) ? particle->ewhich[pweight_index] : -1;

  // Ensure per-cell source-tally array (20 columns) tracks current grid size.
  // In TALLY_COUNTS mode this grows but never zeros across steps (cumulative).
  // In TALLY_RATE mode we additionally zero the whole buffer each call so the
  // subsequent accumulate + normalize yields an instantaneous rate over the
  // current `nevery` window.
  const int ncols = size_per_grid_cols;  // 6 (summary) or 20 (detailed)
  if (grid->maxlocal > maxgrid_src) {
    const int oldmax = maxgrid_src;
    maxgrid_src = grid->maxlocal;
    memory->grow(array_grid, maxgrid_src, ncols, "volume/chem/adas:array_grid(src)");
    // zero the freshly-added rows (count mode relies on this for fresh cells)
    if (maxgrid_src > oldmax) {
      memset(&array_grid[oldmax][0], 0,
             sizeof(double) * (maxgrid_src - oldmax) * ncols);
    }
  }
  if (tally_units == TALLY_RATE && maxgrid_src > 0) {
    memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * ncols);
  }

  // Read Te/ne/Ti/vpar/B from per-particle plasma cache. The cache is
  // populated by update.cpp from the configured plasma provider; the
  // sheath Boltzmann correction (when active) is already folded in,
  // so near-wall ne depletion shows up here automatically.
  {
    double *te_vec = particle->edvec[particle->ewhich[update->pc_te_custom]];
    double *ne_vec = particle->edvec[particle->ewhich[update->pc_ne_custom]];
    double *ti_vec = (update->pc_ti_custom >= 0 && particle->ewhich[update->pc_ti_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_ti_custom]] : nullptr;
    double *vpar_vec = (update->pc_vpar_custom >= 0 && particle->ewhich[update->pc_vpar_custom] >= 0)
                       ? particle->edvec[particle->ewhich[update->pc_vpar_custom]] : nullptr;
    double *bx_vec = (update->pc_bx_custom >= 0 && particle->ewhich[update->pc_bx_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_bx_custom]] : nullptr;
    double *by_vec = (update->pc_by_custom >= 0 && particle->ewhich[update->pc_by_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_by_custom]] : nullptr;
    double *bz_vec = (update->pc_bz_custom >= 0 && particle->ewhich[update->pc_bz_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_bz_custom]] : nullptr;

    if (rate_cache_mode == 1) {
      // Cell mode: compute λ_i once per (cell, species) using the first
      // encountered particle's pcache values, then reuse for every
      // subsequent particle of that species in the cell. Identical
      // physics to the particle path away from sheath cells where
      // Te/ne don't vary across particles within a cell. ~10x cheaper
      // for dense impurities (W chain in this case).
      const int nspec = particle->nspecies;
      // Per-species scratch buffers, refilled per cell.
      // -1 in lambda_total marks "not yet computed for this cell".
      std::vector<double> sp_lambda(nspec * 16, 0.0);
      std::vector<int>    sp_ridx_map(nspec * 16, 0);
      std::vector<int>    sp_nchan(nspec, -1);
      std::vector<double> sp_lambda_total(nspec, 0.0);

      for (int icell = 0; icell < nglocal; icell++) {
        if (cinfo[icell].count == 0) continue;
        // Reset species cache for this cell. Only species that actually
        // appear get their slot recomputed below.
        for (int s = 0; s < nspec; ++s) sp_nchan[s] = -1;

        int ip = cinfo[icell].first;
        while (ip >= 0) {
          const int isp = particles[ip].ispecies;
          const double Te_eV = std::max(te_vec[ip], 1e-6);
          const double ne_m3 = std::max(ne_vec[ip], 0.0);
          const double Ti_eV = ti_vec ? std::max(ti_vec[ip], 0.0) : 0.0;
          const double vp = vpar_vec ? vpar_vec[ip] : 0.0;
          const double Bx = bx_vec ? bx_vec[ip] : 0.0;
          const double By = by_vec ? by_vec[ip] : 0.0;
          const double Bz = bz_vec ? bz_vec[ip] : 0.0;

          // First particle of this species in this cell? Compute lambdas.
          if (sp_nchan[isp] < 0) {
            int nchan_loc = 0;
            double ltot = 0.0;
            compute_species_lambdas(isp, Te_eV, ne_m3, Ti_eV, icell,
                                    &sp_lambda[isp * 16],
                                    &sp_ridx_map[isp * 16],
                                    nchan_loc, ltot);
            sp_nchan[isp] = nchan_loc;
            sp_lambda_total[isp] = ltot;
          }

          attempt(&particles[ip], ip, Te_eV, ne_m3, Ti_eV, vp, Bx, By, Bz,
                  &sp_lambda[isp * 16],
                  &sp_ridx_map[isp * 16],
                  sp_nchan[isp],
                  sp_lambda_total[isp]);
          ip = next[ip];
        }
      }
    } else {
      // Particle mode (default): rates recomputed per particle.
      for (int icell = 0; icell < nglocal; icell++) {
        if (cinfo[icell].count == 0) continue;
        int ip = cinfo[icell].first;
        while (ip >= 0) {
          const double Te_eV = std::max(te_vec[ip], 1e-6);
          const double ne_m3 = std::max(ne_vec[ip], 0.0);
          const double Ti_eV = ti_vec ? std::max(ti_vec[ip], 0.0) : 0.0;
          const double vp = vpar_vec ? vpar_vec[ip] : 0.0;
          const double Bx = bx_vec ? bx_vec[ip] : 0.0;
          const double By = by_vec ? by_vec[ip] : 0.0;
          const double Bz = bz_vec ? bz_vec[ip] : 0.0;
          attempt(&particles[ip], ip, Te_eV, ne_m3, Ti_eV, vp, Bx, By, Bz);
          ip = next[ip];
        }
      }
    }
  }

  // Volume recombination spawn: background-D+ → kinetic-D macroparticles.
  // Adds raw events into array_grid before the RATE normalization below so
  // the same fnum/(vol*window) scaling applies. Spawned particles go onto
  // deferred_particles and get created by the existing add_particle loop.
  if (eirene_mode && nrec_active > 0 &&
      (volume_source_cidx >= 0 || volume_source_fidx >= 0)) {
    spawn_volume_recombination();
  }

  // Rate-mode normalization. Tally was zeroed at the start of this call and
  // just accumulated raw {count, m*v, 0.5 m v^2} over `nevery` steps. Convert
  // each column to a volumetric rate in SI units:
  //    S_n [m^-3 s^-1]    = count * fnum / (vol * window)
  //    S_m [kg m^-2 s^-2] = sum(m v) * fnum / (vol * window)
  //    S_E [W m^-3]       = sum(0.5 m v^2) * fnum / (vol * window)
  // Cells with zero flow volume (outside the domain / fully covered by solid)
  // are set to 0 to avoid divide-by-zero rather than NaN so downstream
  // `fix ave/grid` can aggregate without masking.
  if (tally_units == TALLY_RATE && maxgrid_src > 0) {
    const double window_s = nevery * update->dt;
    const double fnum     = update->fnum;
    if (window_s > 0.0 && fnum > 0.0) {
      const double num = fnum / window_s;
      for (int icell = 0; icell < nglocal; icell++) {
        const double vol = cinfo[icell].volume;
        if (vol <= 0.0) {
          double *row = array_grid[icell];
          for (int c = 0; c < ncols; c++) row[c] = 0.0;
          continue;
        }
        const double scale = num / vol;
        double *row = array_grid[icell];
        for (int c = 0; c < ncols; c++) row[c] *= scale;
      }
    }
  }

  // Mode A: remove particles flagged for deletion on ionization.
  // Do this BEFORE spawning deferred (dissociation) products so the new
  // particles get stable indices at the end of the array.
  if (!dellist.empty()) {
    particle->compress_reactions(static_cast<int>(dellist.size()), dellist.data());
    particle->sorted = 0;   // linked list is now stale; next use will re-sort
    dellist.clear();
  }

  // Create deferred particles from dissociation reactions.
  // Notify update_custom subscribers so per-particle attributes (pweight
  // via fix particle/weight, etc.) get initialized — otherwise spawned
  // products carry the zero defaults from zero_custom() and downstream
  // weighted diagnostics tally zero on them.
  const int nfix_update_custom_local = modify->n_update_custom;
  double zero_v[3] = {0.0, 0.0, 0.0};
  for (size_t i = 0; i < deferred_particles.size(); i++) {
    DeferredParticle &dp = deferred_particles[i];
    int id = MAXSMALLINT * rng_adas->uniform();
    particle->add_particle(id, dp.species, dp.icell,
                           dp.x, dp.v, 0.0, 0.0);
    if (nfix_update_custom_local)
      modify->update_custom(particle->nlocal - 1, 0.0, 0.0, 0.0, zero_v);
  }

  // Mode A: optional stop-when-exhausted across the global source-species pool.
  // Only required when non-source-species particles remain (e.g. impurities);
  // the pure-neutral case is already handled by Update::run's nglobal==0 break.
  if (eirene_mode && stop_on_exhaust && !source_species.empty()) {
    bigint alive_local = 0;
    Particle::OnePart *pp = particle->particles;
    const int np = particle->nlocal;
    for (int i = 0; i < np; i++) {
      const int sp = pp[i].ispecies;
      for (size_t k = 0; k < source_species.size(); k++) {
        if (sp == source_species[k]) { alive_local++; break; }
      }
    }
    bigint alive_global = 0;
    MPI_Allreduce(&alive_local, &alive_global, 1, MPI_SPARTA_BIGINT,
                  MPI_SUM, world);
    // Arm the check only after the batch has ramped above the threshold --
    // otherwise we'd trigger early-exit at step 1 while the emit fix is
    // still filling the initial population.
    if (!exhaust_armed && alive_global > exhaust_threshold) exhaust_armed = 1;

    if (exhaust_armed && alive_global <= exhaust_threshold) {
      // Flag picked up by Update::run on the next iteration; triggers a
      // clean break after final output is written. See update.cpp.
      // Default threshold 0 = wait until population is fully drained.
      // Nonzero threshold = skip the slow fat tail for large speedups with
      // <1% impact on rate moments.
      update->early_exit_requested = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   spawn_volume_recombination

   Volume recombination is a *source* (no kinetic reactant): background D+
   plus a free electron recombines into a neutral D atom. In Mode A there
   are no kinetic D+ particles, so the per-particle attempt() loop never
   sees this channel — it has to be sampled per-cell.

   Per-cell physical event rate:
       R_phys = ne * ni_D+ * <σv>_ACD(Te,ne)              [m^-3 s^-1]
   Events per cell per chem step:
       N_phys = R_phys * V_cell * dt_chem
   Macroparticles to spawn (mean of Poisson):
       λ_macro = N_phys / fnum

   For each spawned macroparticle:
     - position = cell centroid (bbox midpoint)
     - velocity = v_drift + v_thermal
                  v_drift  = vpar * b_hat   (fluid parallel flow)
                  v_thermal ~ Maxwellian at Ti, isotropic 3D
     - tally into array_grid:  Sp += -1, Sm += -m*v_drift,
                               Qe += -E_rad (PRB/ACD), Qi += -(½m·v_drift² + 1.5 kTi)

   `scale` matches attempt()'s convention (COUNTS / RATE / BATCH /
   BATCH_FIX) so this path drops into the same `array_grid` row layout
   used by the per-particle channels and ave/grid normalization.
------------------------------------------------------------------------- */

void FixVolumeChemAdas::spawn_volume_recombination()
{
  if (nrec_active == 0) return;
  if (volume_source_cidx < 0 && volume_source_fidx < 0) return;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int nglocal = grid->nlocal;
  const int dim     = domain->dimension;
  const bool axi    = (domain->axisymmetric != 0);
  const double dt_chem = nevery * update->dt;
  const double fnum    = update->fnum;
  if (dt_chem <= 0.0 || fnum <= 0.0) return;

  // Resolve plasma source: prefer compute (point-query API), fall back to
  // FixBackground (R,Z stencil interp). Both expose ne/Te/ni/Ti/vpar/B.
  ComputePlasmaFields *cp = nullptr;
  FixBackground       *pd = nullptr;
  int gen_now = 0;
  if (volume_source_cidx >= 0) {
    cp = dynamic_cast<ComputePlasmaFields *>(modify->compute[volume_source_cidx]);
    // Make sure the per-cell plasma_arr is current (cheap if already invoked
    // this step by another consumer).
    if (cp && !(cp->invoked_flag & 16)) {
      cp->compute_per_grid();
      cp->invoked_flag |= 16;
    }
  }
  if (!cp && volume_source_fidx >= 0) {
    pd = dynamic_cast<FixBackground *>(modify->fix[volume_source_fidx]);
    if (pd) gen_now = pd->generation;
  }
  if (!cp && !pd) return;

  // Single-charge convention: only the q=1 -> 0 transition is supported
  // here (the only one ADAS lists for D, and the only one this fix's
  // attempt() path uses). Use rec_ridx[0]; if a future deck adds another
  // recomb reaction with a different reactant Z we fall through cleanly.
  const int   ridx_rec = rec_ridx[0];
  const int   sp_prod  = rec_product_isp[0];
  const double m_prod  = particle->species[sp_prod].mass;

  constexpr double eV_to_J = 1.602176634e-19;
  constexpr double kB      = 1.380649e-23;

  auto mit = materials_rate_data.find(atomic_number);
  const bool have_tables = (mit != materials_rate_data.end());
  const RateData *rd = have_tables ? &mit->second : nullptr;
  const bool have_rad = rd && rd->prb_nQ > 0 && rd->rec_nQ > 0;

  // Build / refresh per-cell cache. Plasma query + ACD lookup is the hot
  // path (10s of us per cell × thousands of cells × every step). For static
  // plasma the cached μ/v_drift/v_th/Ti/E_rad are reused across all steps;
  // we only redo the work on grid changes (nlocal / first-cell ID) or on
  // plasma reload (FixBackground::generation bump).
  const bigint first_id =
      (nglocal > 0 && cells) ? cells[0].id : -1;
  const bool need_rebuild =
      (static_cast<int>(rec_cache.size()) != nglocal) ||
      (rec_cache_nlocal != nglocal) ||
      (rec_cache_first_id != first_id) ||
      (pd && rec_cache_generation != gen_now);

  if (need_rebuild) {
    rec_cache.assign(nglocal, RecCellCache{});
    rec_cache_nlocal = nglocal;
    rec_cache_first_id = first_id;
    rec_cache_generation = gen_now;

    for (int icell = 0; icell < nglocal; icell++) {
      RecCellCache &c = rec_cache[icell];
      c.mu = 0.0;
      const double vol = cinfo[icell].volume;
      if (vol <= 0.0) continue;
      if (cells[icell].nsplit > 1) continue;

      const double xc[3] = {
        0.5 * (cells[icell].lo[0] + cells[icell].hi[0]),
        0.5 * (cells[icell].lo[1] + cells[icell].hi[1]),
        (dim == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]) : 0.0
      };

      double Te_eV = 0.0, ne_m3 = 0.0, ni_m3 = 0.0, Ti_eV = 0.0, vpar = 0.0;
      double bx = 0.0, by = 0.0, bz = 0.0;
      if (cp) {
        PlasmaFileParams pf = cp->query_plasma_at_point(xc);
        Te_eV = pf.temp_e;
        ne_m3 = pf.dens_e;
        ni_m3 = (pf.dens_i > 0.0) ? pf.dens_i : pf.dens_e;
        Ti_eV = pf.temp_i;
        vpar  = pf.parr_flow;
        MagneticFieldFileDataParams bf = cp->query_bfield_at_point(xc);
        bx = bf.br;  by = bf.bt;  bz = bf.bz;
      } else {
        double R = 0.0, Z = 0.0;
        OpenEdge::sparta_to_RZ(xc, dim, axi, R, Z,
                               pd->column_x0, pd->column_y0);
        Te_eV = pd->interp2D(pd->temp_e, R, Z, icell);
        ne_m3 = pd->interp2D(pd->dens_e, R, Z, icell);
        ni_m3 = pd->dens_i.empty() ? ne_m3 : pd->interp2D(pd->dens_i, R, Z, icell);
        Ti_eV = pd->temp_i.empty() ? Te_eV : pd->interp2D(pd->temp_i, R, Z, icell);
        vpar  = pd->parr_flow.empty() ? 0.0
                                      : pd->interp2D(pd->parr_flow, R, Z, icell);
        if (pd->has_bfield) {
          double Br = 0.0, Bz_ = 0.0, Bt = 0.0;
          pd->bfield_at(R, Z, Br, Bz_, Bt, icell);
          bx = Br; by = Bt; bz = Bz_;
        }
      }
      if (Te_eV <= 0.0 || ne_m3 <= 0.0 || ni_m3 <= 0.0) continue;

      const double logTe    = std::log10(std::max(Te_eV, 1e-6));
      const double logne_cm = std::log10(std::max(ne_m3 * 1e-6, 1e-99));
      double acd_log10 = -INFINITY;
      interpolateRateData(atomic_number, /*q-1*/ 0, icell, logTe, logne_cm,
                          acd_log10, ReactionType::Recombination);
      if (!std::isfinite(acd_log10)) continue;

      const double sigv_m3s = std::pow(10.0, acd_log10) * 1e-6;
      const double mu = ne_m3 * ni_m3 * sigv_m3s * vol * dt_chem / fnum;
      if (mu <= 0.0 || !std::isfinite(mu)) continue;

      const double Bmag = std::sqrt(bx*bx + by*by + bz*bz);
      double vix = 0.0, viy = 0.0, viz = 0.0;
      if (Bmag > 1e-30) {
        const double invB = 1.0 / Bmag;
        vix = vpar * bx * invB;
        viy = vpar * by * invB;
        viz = vpar * bz * invB;
      }
      const double Ti_J = std::max(Ti_eV, 0.0) * eV_to_J;
      const double Ti_K = Ti_J / kB;
      const double v_th = (m_prod > 0.0) ? std::sqrt(kB * Ti_K / m_prod) : 0.0;

      double E_rad_J = 0.0;
      if (have_rad) {
        double prb_log10 = -INFINITY;
        interpolateRateData(atomic_number, 0, icell, logTe, logne_cm,
                            prb_log10, ReactionType::RecombRadiation);
        if (std::isfinite(prb_log10)) {
          E_rad_J = std::pow(10.0, prb_log10 - acd_log10);
        }
      }

      c.mu      = mu;
      c.vix     = vix;
      c.viy     = viy;
      c.viz     = viz;
      c.v_th    = v_th;
      c.Ti_J    = Ti_J;
      c.dQe_per = -E_rad_J;
    }
  }

  const int nfix_update_custom_local = modify->n_update_custom;
  double zero_v[3] = {0.0, 0.0, 0.0};

  bigint nevents_local = 0;

  for (int icell = 0; icell < nglocal; icell++) {
    const RecCellCache &c = rec_cache[icell];
    if (c.mu <= 0.0) continue;

    const double vol = cinfo[icell].volume;

    // Poisson(mu) sampler: small-mu uses inverse-CDF (Knuth);
    // large-mu uses Gaussian(mu, sqrt(mu)) clamped at 0.
    const double mu = c.mu;
    int N_macro = 0;
    if (mu < 30.0) {
      const double L = std::exp(-mu);
      double p = 1.0;
      int k = 0;
      while (true) {
        p *= rng_adas->uniform();
        if (p < L) break;
        k++;
        if (k > 10000) break;   // runaway guard
      }
      N_macro = k;
    } else {
      const double u1 = std::max(rng_adas->uniform(), 1e-30);
      const double u2 = rng_adas->uniform();
      const double g  = std::sqrt(-2.0 * std::log(u1)) * std::cos(MY_2PI * u2);
      const double sample = mu + std::sqrt(mu) * g;
      N_macro = (sample > 0.0) ? static_cast<int>(sample + 0.5) : 0;
    }
    if (N_macro <= 0) continue;

    const double xc[3] = {
      0.5 * (cells[icell].lo[0] + cells[icell].hi[0]),
      0.5 * (cells[icell].lo[1] + cells[icell].hi[1]),
      (dim == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]) : 0.0
    };

    const double vix = c.vix, viy = c.viy, viz = c.viz;
    const double v_th = c.v_th;

    double scale = 1.0;
    if (tally_units == TALLY_BATCH) {
      const double w = batch_R_puff / static_cast<double>(batch_N);
      scale = w / vol;
    } else if (tally_units == TALLY_BATCH_FIX && batch_N_cached > 0) {
      const double w = batch_R_puff / static_cast<double>(batch_N_cached);
      scale = w / vol;
    }

    const double vi2 = vix*vix + viy*viy + viz*viz;
    const double dQi_per = -(0.5 * m_prod * vi2 + 1.5 * c.Ti_J);
    const double dQe_per = c.dQe_per;

    for (int s = 0; s < N_macro; s++) {
      // Sample shifted-Maxwellian velocity (3 i.i.d. Gaussians).
      const double u1 = std::max(rng_adas->uniform(), 1e-30);
      const double u2 = rng_adas->uniform();
      const double u3 = rng_adas->uniform();
      const double u4 = std::max(rng_adas->uniform(), 1e-30);
      const double g1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(MY_2PI * u2);
      const double g2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(MY_2PI * u2);
      const double g3 = std::sqrt(-2.0 * std::log(u4)) * std::cos(MY_2PI * u3);

      DeferredParticle dp;
      dp.x[0] = xc[0]; dp.x[1] = xc[1]; dp.x[2] = xc[2];
      dp.v[0] = vix + v_th * g1;
      dp.v[1] = viy + v_th * g2;
      dp.v[2] = viz + v_th * g3;
      dp.species = sp_prod;
      dp.icell = icell;
      deferred_particles.push_back(dp);

      // Tally into array_grid in the same row layout as attempt().
      if (array_grid && icell < maxgrid_src) {
        double *row = array_grid[icell];
        if (output_mode == OUT_DETAILED) {
          // Recombination column = index 1 in the {ioniz, recomb, CX, dissoc}
          // bucket layout (col 0..3 count, 4..7 mvx, 8..11 mvy, 12..15 mvz,
          // 16..19 ½mv²). Use product-particle drift velocity as the event
          // velocity, matching the per-particle attempt() detailed path.
          const double vx = dp.v[0], vy = dp.v[1], vz = dp.v[2];
          const double ke = 0.5 * m_prod * (vx*vx + vy*vy + vz*vz);
          row[1]    += scale;
          row[5]    += m_prod * vx * scale;
          row[9]    += m_prod * vy * scale;
          row[13]   += m_prod * vz * scale;
          row[17]   += ke * scale;
        } else {
          row[0] += -1.0 * scale;        // Sp
          row[1] += -m_prod * vix * scale; // Sm_x
          row[2] += -m_prod * viy * scale; // Sm_y
          row[3] += -m_prod * viz * scale; // Sm_z
          row[4] += dQe_per * scale;     // Qe
          row[5] += dQi_per * scale;     // Qi
        }
      }
      nevents_local++;
    }
  }

  // Bump the per-reaction event tally (used by post_run summary).
  if (nevents_local > 0) {
    tally_reactions[ridx_rec] += nevents_local;
    nreact_one += nevents_local;
  }

  (void) nfix_update_custom_local;  // creation/update_custom done by the
                                    // shared deferred_particles loop below.
  (void) zero_v;
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixVolumeChemAdas::memory_usage()
{
  double bytes = 0.0;
  bytes += maxgrid*3 * sizeof(double);    // vcom
  return bytes;
}

/* ----------------------------------------------------------------------
   attempt a reaction for a single particle
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   compute_species_lambdas: per-channel Poisson rates λ_i = k_i * ne * dt
   for one (species, Te, ne, icell). Mirrors the rate-table block of
   attempt() so the cell-mode caller in end_of_step_no_average can
   precompute lambdas once per (cell, species) and reuse them across all
   particles of that species in the cell.
------------------------------------------------------------------------- */
void FixVolumeChemAdas::compute_species_lambdas(int isp, double Te_eV, double ne_m3,
                                                  double Ti_eV, int icell,
                                                  double *lambda_out, int *ridx_map_out,
                                                  int &nchan_out, double &lambda_total_out)
{
  nchan_out = 0;
  lambda_total_out = 0.0;
  if (Te_eV <= 0.0 || ne_m3 <= 0.0) return;
  if (isp < 0 || reactions[isp].n == 0) return;

  Particle::Species *species = particle->species;
  const double logTe    = std::log10(Te_eV);
  const double logne_cm = std::log10(std::max(ne_m3 * 1e-6, 1e-99));
  const double dt_chem  = nevery * update->dt;

  const int n = reactions[isp].n;
  for (int i = 0; i < n && nchan_out < 16; ++i) {
    const int ridx = reactions[isp].list[i];
    OneReaction *r = &rlist[ridx];
    const size_t q = static_cast<size_t>(std::max(0.0, species[isp].charge));

    double rate_log10_cm3s = -INFINITY;
    if (r->type == IONIZATION) {
      if (q >= static_cast<size_t>(atomic_number)) continue;
      if (atomic_number == 1 && q == 0) {
        // AMJUEL H.4 2.1.5 (D ionization). Replaces ADAS SCD89.
        rate_log10_cm3s = log10_sigmav_ioniz_amjuel_cm3s(Te_eV, ne_m3);
      } else {
        interpolateRateData(atomic_number, q,   icell, logTe, logne_cm,
                            rate_log10_cm3s, ReactionType::Ionization);
      }
    } else if (r->type == RECOMBINATION) {
      if (q == 0) continue;
      interpolateRateData(atomic_number, q-1, icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::Recombination);
    } else if (r->type == EXCHANGE) {
      if (atomic_number == 1 && Ti_eV > 0.0) {
        // HYDHEL H.3 3.1.8 (D-D+ resonant CX).
        rate_log10_cm3s = log10_sigmav_cx_hh_cm3s(Ti_eV, 1.5 * Ti_eV);
      } else {
        // Impurity-H CX: keep ADAS CCD.
        const size_t cx_row = (q > 0) ? (q - 1) : 0;
        interpolateRateData(atomic_number, cx_row, icell, logTe, logne_cm,
                            rate_log10_cm3s, ReactionType::ChargeExchange);
      }
    } else if (r->type == DISSOCIATION && r->style == JANEV) {
      const double lnT = std::log(Te_eV);
      double lnsv = r->coeff[0];
      double lnTn = 1.0;
      for (int k = 1; k < r->ncoeff; k++) {
        lnTn *= lnT;
        lnsv += r->coeff[k] * lnTn;
      }
      rate_log10_cm3s = lnsv / 2.302585092994046;
    } else {
      continue;
    }

    if (!std::isfinite(rate_log10_cm3s)) continue;
    const double lam = computeReactionLambda(rate_log10_cm3s, dt_chem, ne_m3);
    if (lam <= 0.0) continue;

    lambda_out[nchan_out] = lam;
    ridx_map_out[nchan_out] = ridx;
    lambda_total_out += lam;
    nchan_out++;
  }
}

/* ---------------------------------------------------------------------- */

int FixVolumeChemAdas::attempt(Particle::OnePart *ip, int ip_index,
                         double Te_eV, double ne_m3,
                         double Ti_eV, double vpar, double bx, double by, double bz,
                         const double *cached_lambda,
                         const int *cached_ridx_map,
                         int cached_nchan,
                         double cached_lambda_total)
{
  Particle::Species *species = particle->species;

  const int isp0 = ip->ispecies;
  if (reactions[isp0].n == 0) return 0;

  const int icell = ip->icell;
  if (icell < 0 || icell >= grid->nlocal) return 0;

  if (Te_eV <= 0.0 || ne_m3 <= 0.0) return 0;

  // logTe/logne_cm are needed by the per-event tally branch (PLT/PRB
  // interpolations on lines ~1130-1138, 1149+) regardless of whether the
  // per-channel lambdas were precomputed by the caller, so always compute
  // them here.
  const double logTe    = std::log10(Te_eV);
  const double logne_cm = std::log10(std::max(ne_m3 * 1e-6, 1e-99));

  int isp = ip->ispecies;
  const int n = reactions[isp].n;
  if (n == 0) return 0;

  // Poisson competing-channel selection:
  //   λ_i = k_i * ne * dt_chem   for each channel i
  //   λ_total = Σ λ_i
  //   P(any event) = 1 - exp(-λ_total)
  //   channel selected proportional to λ_i / λ_total

  const double dt_chem = nevery * update->dt;

  double lambda[16];   // per-channel Poisson rates (max 16 channels)
  int    ridx_map[16]; // reaction index for each channel
  int    nchan = 0;
  double lambda_total = 0.0;

  if (cached_nchan >= 0 && cached_lambda && cached_ridx_map) {
    // Fast path: caller (rate_cache=cell mode in end_of_step_no_average)
    // already computed lambdas for this (cell, species) using the first
    // encountered particle's pcache values. Reuse them — saves the
    // per-particle log10 + bracket_index + bilinearInterpolate work.
    nchan = cached_nchan;
    lambda_total = cached_lambda_total;
    for (int i = 0; i < nchan; ++i) {
      lambda[i] = cached_lambda[i];
      ridx_map[i] = cached_ridx_map[i];
    }
  } else {
    for (int i = 0; i < n && nchan < 16; ++i) {
      const int ridx = reactions[isp].list[i];
      OneReaction *r = &rlist[ridx];

      const size_t q = static_cast<size_t>(std::max(0.0, species[isp].charge));

      double rate_log10_cm3s = -INFINITY;

      if (r->type == IONIZATION) {
        if (q >= static_cast<size_t>(atomic_number)) continue;
        if (atomic_number == 1 && q == 0) {
          // AMJUEL H.4 2.1.5 (D ionization, Sawada/Fujimoto).
          rate_log10_cm3s = log10_sigmav_ioniz_amjuel_cm3s(Te_eV, ne_m3);
        } else {
          interpolateRateData(atomic_number, q,   icell, logTe, logne_cm,
                              rate_log10_cm3s, ReactionType::Ionization);
        }
      } else if (r->type == RECOMBINATION) {
        if (q == 0) continue;
        interpolateRateData(atomic_number, q-1, icell, logTe, logne_cm,
                            rate_log10_cm3s, ReactionType::Recombination);
      } else if (r->type == EXCHANGE) {
        if (atomic_number == 1 && Ti_eV > 0.0) {
          // HYDHEL H.3 3.1.8 (D-D+ resonant CX). Per-particle E_atom.
          const double m_atom = particle->species[isp].mass;
          const double vx0p = ip->v[0], vy0p = ip->v[1], vz0p = ip->v[2];
          const double v2p  = vx0p*vx0p + vy0p*vy0p + vz0p*vz0p;
          constexpr double eV_to_J_local = 1.602176634e-19;
          const double E_atom_eV = 0.5 * m_atom * v2p / eV_to_J_local;
          rate_log10_cm3s = log10_sigmav_cx_hh_cm3s(Ti_eV, E_atom_eV);
        } else {
          // Impurity-H CX: keep ADAS CCD.
          const size_t cx_row = (q > 0) ? (q - 1) : 0;
          interpolateRateData(atomic_number, cx_row, icell, logTe, logne_cm,
                              rate_log10_cm3s, ReactionType::ChargeExchange);
        }
      } else if (r->type == DISSOCIATION && r->style == JANEV) {
        // Janev polynomial: ln<sv> = sum_n b_n (ln Te)^n, Te in eV
        const double lnT = std::log(Te_eV);
        double lnsv = r->coeff[0];
        double lnTn = 1.0;
        for (int k = 1; k < r->ncoeff; k++) {
          lnTn *= lnT;
          lnsv += r->coeff[k] * lnTn;
        }
        // Convert from ln(cm3/s) to log10(cm3/s)
        rate_log10_cm3s = lnsv / 2.302585092994046;
      } else {
        continue;
      }

      if (!std::isfinite(rate_log10_cm3s)) continue;

      const double lam = computeReactionLambda(rate_log10_cm3s, dt_chem, ne_m3);
      if (lam <= 0.0) continue;

      lambda[nchan] = lam;
      ridx_map[nchan] = ridx;
      lambda_total += lam;
      nchan++;
    }
  }

  if (nchan == 0 || lambda_total <= 0.0) return 0;

  // P(at least one event in dt_chem) = 1 - exp(-λ_total)
  const double P_any = -std::expm1(-lambda_total);
  const double u = rng_adas->uniform();
  if (u > P_any) return 0;

  // select channel proportional to λ_i / λ_total
  int chosen = 0;
  if (nchan > 1) {
    const double v = rng_adas->uniform() * lambda_total;
    double cumsum = 0.0;
    for (int i = 0; i < nchan; i++) {
      cumsum += lambda[i];
      if (v <= cumsum) { chosen = i; break; }
    }
  }

  const int best_idx = ridx_map[chosen];
  OneReaction *rchosen = &rlist[best_idx];
  tally_reactions[best_idx]++;
  nreact_one++;

  // Per-cell source-term tally. Layout depends on output_mode; see
  // docs/neutral_plasma_coupling/main.tex for the full physics.
  //
  //   OUT_SUMMARY  (6 cols, signed plasma-frame moments):
  //     col 0 : Sp   (particle source,     +1 ioniz, -1 recomb, 0 CX/diss)
  //     col 1 : Sm_x (momentum source, kg*m/s)
  //     col 2 : Sm_y
  //     col 3 : Sm_z
  //     col 4 : Qe   (electron energy source, J)
  //     col 5 : Qi   (ion energy source, J)
  //
  //   OUT_DETAILED (20 cols, unsigned per-reaction raw tally):
  //     col  0.. 3 = count  (ioniz, recomb, CX, dissoc)
  //     col  4.. 7 = sum(m*vx)
  //     col  8..11 = sum(m*vy)
  //     col 12..15 = sum(m*vz)
  //     col 16..19 = sum(0.5*m*|v|^2) [J]
  //
  // Weight `scale` is shared by both modes and implements
  // COUNTS / RATE / BATCH / BATCH_FIX unit conventions (see `units` keyword).
  if (array_grid && icell >= 0 && icell < maxgrid_src) {
    double scale = 1.0;
    if (tally_units == TALLY_BATCH) {
      const double vol = grid->cinfo[icell].volume;
      const double w   = batch_R_puff / static_cast<double>(batch_N);
      scale = (vol > 0.0) ? (w / vol) : 0.0;
    } else if (tally_units == TALLY_BATCH_FIX && batch_N_cached > 0) {
      const double vol = grid->cinfo[icell].volume;
      const double w   = batch_R_puff / static_cast<double>(batch_N_cached);
      scale = (vol > 0.0) ? (w / vol) : 0.0;
    } else if (tally_units == TALLY_RATE && pweight_ewhich >= 0 &&
               ip_index >= 0) {
      // Weight the event by the reacting particle's real-particle count
      // (pweight) rather than the global fnum the RATE normalization
      // re-applies below. pweight defaults to fnum, so unweighted particles
      // give scale = 1 (identical to the previous behaviour); flux-scaled
      // emission (fix surface/emit/source nlaunch) is now tracked correctly.
      const double fnum = update->fnum;
      if (fnum > 0.0)
        scale = particle->edvec[pweight_ewhich][ip_index] / fnum;
    }

    const double m   = particle->species[isp0].mass;
    const double vx0 = ip->v[0];
    const double vy0 = ip->v[1];
    const double vz0 = ip->v[2];

    if (output_mode == OUT_DETAILED) {
      int rtype_off = -1;
      switch (rchosen->type) {
        case IONIZATION:     rtype_off = 0; break;
        case RECOMBINATION:  rtype_off = 1; break;
        case EXCHANGE:       rtype_off = 2; break;
        case DISSOCIATION:   rtype_off = 3; break;
      }
      if (rtype_off >= 0) {
        const double ke  = 0.5 * m * (vx0*vx0 + vy0*vy0 + vz0*vz0);
        double *row = array_grid[icell];
        row[rtype_off]         += scale;
        row[4  + rtype_off]    += m * vx0 * scale;
        row[8  + rtype_off]    += m * vy0 * scale;
        row[12 + rtype_off]    += m * vz0 * scale;
        row[16 + rtype_off]    += ke        * scale;
      }
    } else {
      // OUT_SUMMARY: signed plasma-frame moments.
      constexpr double eV_to_J = 1.602176634e-19;
      constexpr double E_diss_D2_eV = 4.478;    // bond energy, D2 -> 2D

      // Ion velocity approximation: fluid parallel flow v_par * b_hat.
      const double Bmag = std::sqrt(bx*bx + by*by + bz*bz);
      double vix = 0.0, viy = 0.0, viz = 0.0;
      if (Bmag > 1e-30) {
        const double invB = 1.0 / Bmag;
        vix = vpar * bx * invB;
        viy = vpar * by * invB;
        viz = vpar * bz * invB;
      }
      const double v2  = vx0*vx0 + vy0*vy0 + vz0*vz0;
      const double vi2 = vix*vix + viy*viy + viz*viz;

      // Reactant charge state (neutral=0 for ionization, q=charge for recomb).
      const size_t q = static_cast<size_t>(
                         std::max(0.0, particle->species[isp0].charge));
      const double Ti_J = Ti_eV * eV_to_J;

      // Rate tables live in materials_rate_data[atomic_number], not the
      // unused default `rate_data` member.  The loader writes there (see
      // readRateDataParallel() call at constructor time), and
      // interpolateRateData() internally reads from there too -- use the
      // same struct for the presence guards so we don't silently skip
      // PLT/PRB/IP contributions.
      auto mit = materials_rate_data.find(atomic_number);
      const bool have_tables = (mit != materials_rate_data.end());
      const RateData *rd = have_tables ? &mit->second : nullptr;

      double dSp = 0.0;
      double dSmx = 0.0, dSmy = 0.0, dSmz = 0.0;
      double dQe = 0.0, dQi = 0.0;

      switch (rchosen->type) {
      case IONIZATION: {
        // Effective cost per ionization: E_eff = E_ion + PLT/SCD [eV].
        // Falls back to E_ion only when PLT table absent, and to 0 when IP
        // table is also absent.
        double E_eff_eV = 0.0;
        if (rd && !rd->ion_potential.empty() &&
            q < rd->ion_potential.size()) {
          E_eff_eV = rd->ion_potential[q];
        }
        if (rd && rd->plt_nQ > 0 && rd->ion_nQ > 0) {
          double plt_log10 = -INFINITY, scd_log10 = -INFINITY;
          interpolateRateData(atomic_number, q, icell, logTe, logne_cm,
                              plt_log10, ReactionType::LineRadiation);
          interpolateRateData(atomic_number, q, icell, logTe, logne_cm,
                              scd_log10, ReactionType::Ionization);
          if (std::isfinite(plt_log10) && std::isfinite(scd_log10)) {
            // PLT [log10 W cm^3] - SCD [log10 cm^3/s] = log10 (W s) = log10 J
            const double J_per_event = std::pow(10.0, plt_log10 - scd_log10);
            E_eff_eV += J_per_event / eV_to_J;
          }
        }
        dSp  = +1.0;
        dSmx = +m * vx0;
        dSmy = +m * vy0;
        dSmz = +m * vz0;
        dQe  = -E_eff_eV * eV_to_J;
        // EIRENE convention: ionization adds the neutral's KE to the ion
        // fluid. Thermalisation to local Ti happens later as a fluid
        // process, not from EIRENE. No thermal subtraction here.
        dQi  = +0.5 * m * v2;
        break;
      }
      case RECOMBINATION: {
        // Qe: use PRB/ACD as the total radiated power per recombination event.
        // Falls back to 0 when either table is absent (no electron cooling
        // contribution), keeping Qi correct regardless.
        double E_rec_loss_eV = 0.0;
        if (rd && rd->prb_nQ > 0 && rd->rec_nQ > 0 && q > 0) {
          const size_t qrow = q - 1;    // ACD/PRB row indexed by q_ion - 1
          double prb_log10 = -INFINITY, acd_log10 = -INFINITY;
          interpolateRateData(atomic_number, qrow, icell, logTe, logne_cm,
                              prb_log10, ReactionType::RecombRadiation);
          interpolateRateData(atomic_number, qrow, icell, logTe, logne_cm,
                              acd_log10, ReactionType::Recombination);
          if (std::isfinite(prb_log10) && std::isfinite(acd_log10)) {
            const double J_per_event = std::pow(10.0, prb_log10 - acd_log10);
            E_rec_loss_eV = J_per_event / eV_to_J;
          }
        }
        dSp  = -1.0;
        dSmx = -m * vix;
        dSmy = -m * viy;
        dSmz = -m * viz;
        dQe  = -E_rec_loss_eV * eV_to_J;
        dQi  = -(0.5 * m * vi2 + 1.5 * Ti_J);   // drift KE + thermal (3/2 kB Ti)
        break;
      }
      case EXCHANGE: {
        dSp  = 0.0;
        dSmx = +m * (vx0 - vix);
        dSmy = +m * (vy0 - viy);
        dSmz = +m * (vz0 - viz);
        dQe  = 0.0;
        // Model 1 (analytical mean) for all CX channels: dQi = drift KE
        // swap minus thermal 3/2 Ti. Previous Model 2 (sigma*v-sampled
        // v_post) over-weighted high v_rel and inflated per-event |dQi|
        // by ~2x at SOL temperatures.
        dQi = +0.5 * m * (v2 - vi2) - 1.5 * Ti_J;
        break;
      }
      case DISSOCIATION: {
        // Only D2 / H2 / T2 currently supported (4.478 eV).  Other molecular
        // species would need a species-specific bond energy; for now use this
        // single constant and accept that any future addition requires a
        // local edit here.  See LaTeX doc Sec. 4.4 for discussion.
        dSp  = 0.0;
        dSmx = dSmy = dSmz = 0.0;
        dQe  = -E_diss_D2_eV * eV_to_J;
        dQi  = 0.0;
        break;
      }
      }

      double *row = array_grid[icell];
      row[0] += dSp  * scale;
      row[1] += dSmx * scale;
      row[2] += dSmy * scale;
      row[3] += dSmz * scale;
      row[4] += dQe  * scale;
      row[5] += dQi  * scale;
    }
  }

  // Mode A (EIRENE semantics): neutral is consumed on ionization.
  // Defer deletion -- we are iterating the cell's linked list via Particle::next,
  // so actually removing ip now would invalidate that traversal. The caller
  // compresses the particle array after the cell loop completes.
  // NB: only IONIZATION terminates; CX/DISSOCIATION still run their species
  // swap + velocity re-sampling below so neutral-neutral chains keep working.
  if (eirene_mode && rchosen->type == IONIZATION) {
    dellist.push_back(ip_index);
    return 1;
  }

  // Assign first product
  ip->ispecies = rchosen->products[0];

  // CX products: shifted Maxwellian at local Ti + bulk flow (EIRENE-like).
  // Dissociation products: isotropic Franck-Condon kick in the parent CM
  // frame. The molecular potential releases ~3 eV/atom for D2 -> 2D,
  // independent of plasma Ti; resampling dissociation at Ti would give
  // products 3-10x too energetic in a hot SOL.
  const double vpx_parent = ip->v[0];
  const double vpy_parent = ip->v[1];
  const double vpz_parent = ip->v[2];

  double fc_kick_x = 0.0, fc_kick_y = 0.0, fc_kick_z = 0.0;
  const bool do_cx     = (rchosen->type == EXCHANGE) && Ti_eV > 0.0;
  const bool do_dissoc = (rchosen->type == DISSOCIATION);

  if (do_cx) {
    const double kB = 1.380649e-23;
    const double eV_to_J = 1.602176634e-19;
    const double Ti_K = Ti_eV * eV_to_J / kB;
    const double m_prod = particle->species[rchosen->products[0]].mass;
    const double v_th = (m_prod > 0.0) ? std::sqrt(kB * Ti_K / m_prod) : 0.0;

    const double Bmag = std::sqrt(bx*bx + by*by + bz*bz);
    double vfx = 0.0, vfy = 0.0, vfz = 0.0;
    if (Bmag > 1e-30) {
      const double invB = 1.0 / Bmag;
      vfx = vpar * bx * invB;
      vfy = vpar * by * invB;
      vfz = vpar * bz * invB;
    }

    if (atomic_number == 1) {
      // Hydrogen: cross-section-weighted post-CX velocity sampling. The
      // bulk-ion partner velocity is rejection-sampled from the drifting
      // Maxwellian weighted by sigma_CX(E_rel)*v_rel. Post-CX neutral
      // takes the sampled ion velocity. SGCVMX is a conservative upper
      // bound on sigma*v across the relevant E_rel range.
      const double E_lab_factor = 0.5 * m_prod / eV_to_J;
      constexpr double SGCVMX = 1.2e-13;
      double v_post_x = 0.0, v_post_y = 0.0, v_post_z = 0.0;
      int icount = 0;
      while (true) {
        const double u1 = std::max(rng_adas->uniform(), 1e-30);
        const double u2 = rng_adas->uniform();
        const double u3 = rng_adas->uniform();
        const double u4 = std::max(rng_adas->uniform(), 1e-30);
        const double g1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(MY_2PI * u2);
        const double g2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(MY_2PI * u2);
        const double g3 = std::sqrt(-2.0 * std::log(u4)) * std::cos(MY_2PI * u3);
        v_post_x = vfx + v_th * g1;
        v_post_y = vfy + v_th * g2;
        v_post_z = vfz + v_th * g3;

        const double dvx = v_post_x - vpx_parent;
        const double dvy = v_post_y - vpy_parent;
        const double dvz = v_post_z - vpz_parent;
        const double vrel_sq = dvx*dvx + dvy*dvy + dvz*dvz;
        const double vrel    = std::sqrt(vrel_sq);
        const double E_lab   = E_lab_factor * vrel_sq;
        const double sigma   = sigma_cx_hh_m2(E_lab);
        const double sgv     = sigma * vrel;

        if (rng_adas->uniform() * SGCVMX < sgv) break;
        if (++icount >= 500) break;
      }
      ip->v[0] = v_post_x;
      ip->v[1] = v_post_y;
      ip->v[2] = v_post_z;
      // Per-event dQi tallied above via Model 1 in the EXCHANGE case.
    } else {
      // Impurity-H CX (Z>=2): simple shifted-Maxwellian draw. dQi is
      // tallied in the per-event block above using the analytical mean.
      const double u1 = std::max(rng_adas->uniform(), 1e-30);
      const double u2 = rng_adas->uniform();
      const double u3 = rng_adas->uniform();
      const double u4 = std::max(rng_adas->uniform(), 1e-30);
      const double g1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(MY_2PI * u2);
      const double g2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(MY_2PI * u2);
      const double g3 = std::sqrt(-2.0 * std::log(u4)) * std::cos(MY_2PI * u3);
      ip->v[0] = vfx + v_th * g1;
      ip->v[1] = vfy + v_th * g2;
      ip->v[2] = vfz + v_th * g3;
    }
  }
  else if (do_dissoc) {
    const double E_FC_eV = 3.0;
    const double eV_to_J = 1.602176634e-19;
    const double m_prod = particle->species[rchosen->products[0]].mass;
    const double v_FC = (m_prod > 0.0)
                        ? std::sqrt(2.0 * E_FC_eV * eV_to_J / m_prod) : 0.0;

    const double u1 = rng_adas->uniform();
    const double u2 = rng_adas->uniform();
    const double costh = 1.0 - 2.0 * u1;
    const double sinth = std::sqrt(std::max(0.0, 1.0 - costh*costh));
    const double phi   = MY_2PI * u2;
    fc_kick_x = v_FC * sinth * std::cos(phi);
    fc_kick_y = v_FC * sinth * std::sin(phi);
    fc_kick_z = v_FC * costh;

    ip->v[0] = vpx_parent + fc_kick_x;
    ip->v[1] = vpy_parent + fc_kick_y;
    ip->v[2] = vpz_parent + fc_kick_z;
  }

  // For dissociation with 2 products: defer creation of second particle.
  // Back-to-back kick in the CM frame keeps p1+p2 = m_parent * v_parent.
  if (rchosen->nproduct == 2) {
    DeferredParticle dp;
    dp.x[0] = ip->x[0]; dp.x[1] = ip->x[1]; dp.x[2] = ip->x[2];

    if (do_dissoc) {
      dp.v[0] = vpx_parent - fc_kick_x;
      dp.v[1] = vpy_parent - fc_kick_y;
      dp.v[2] = vpz_parent - fc_kick_z;
    } else {
      dp.v[0] = ip->v[0]; dp.v[1] = ip->v[1]; dp.v[2] = ip->v[2];
    }

    dp.species = rchosen->products[1];
    dp.icell = ip->icell;
    deferred_particles.push_back(dp);
  }

  return 1;
}

void FixVolumeChemAdas::readfile(char *fname)
{
  int n,n1,n2,eof;
  char line1[MAXLINE],line2[MAXLINE];
  char copy1[MAXLINE],copy2[MAXLINE];
  char *word;
  OneReaction *r;

  // proc 0 opens file

  if (comm->me == 0) {
    fp = fopen(fname,"r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open reaction file %s",fname);
      error->one(FLERR,str);
    }
  }

  // read reactions one at a time and store their info in rlist

  while (1) {
    if (comm->me == 0) eof = readone(line1,line2,n1,n2);
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;

    MPI_Bcast(&n1,1,MPI_INT,0,world);
    MPI_Bcast(&n2,1,MPI_INT,0,world);
    MPI_Bcast(line1,n1,MPI_CHAR,0,world);
    MPI_Bcast(line2,n2,MPI_CHAR,0,world);

    if (nlist == maxlist) {
      maxlist += DELTALIST;
      rlist = (OneReaction *)
        memory->srealloc(rlist,maxlist*sizeof(OneReaction),"react/adas:rlist");
      for (int i = nlist; i < maxlist; i++) {
        r = &rlist[i];
        r->nreactant = r->nproduct = 0;
        r->id_reactants = new char*[MAXREACTANT];
        r->id_products = new char*[MAXPRODUCT];
        r->reactants = new int[MAXREACTANT];
        r->products = new int[MAXPRODUCT];
        r->coeff = new double[MAXCOEFF];
        r->id = NULL;
      }
    }

    strcpy(copy1,line1);
    strcpy(copy2,line2);

    r = &rlist[nlist];
    r->initflag = 0;

    int side = 0;
    int species = 1;

    n = strlen(line1) - 1;
    r->id = new char[n+1];
    strncpy(r->id,line1,n);
    r->id[n] = '\0';

    word = strtok(line1," \t\n\r");

    while (1) {
      if (!word) {
        if (side == 0) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        if (species) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        break;
      }
      if (species) {
        species = 0;
        if (side == 0) {
          if (r->nreactant == MAXREACTANT) {
            print_reaction(copy1,copy2);
            error->all(FLERR,"Too many reactants in a reaction formula");
          }
          n = strlen(word) + 1;
          r->id_reactants[r->nreactant] = new char[n];
          strcpy(r->id_reactants[r->nreactant],word);
          r->nreactant++;
        } else {
          if (r->nreactant == MAXPRODUCT) {
            print_reaction(copy1,copy2);
            error->all(FLERR,"Too many products in a reaction formula");
          }
          n = strlen(word) + 1;
          r->id_products[r->nproduct] = new char[n];
          strcpy(r->id_products[r->nproduct],word);
          r->nproduct++;
        }
      } else {
        species = 1;
        if (strcmp(word,"+") == 0) {
          word = strtok(NULL," \t\n\r");
          continue;
        }
        if (strcmp(word,"-->") != 0) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        side = 1;
      }
      word = strtok(NULL," \t\n\r");
    }

    word = strtok(line2," \t\n\r");
    if (!word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction type in file");
    }
    if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    else if (word[0] == 'I' || word[0] == 'i') r->type = IONIZATION;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
    else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    else {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction type in file");
    }

    // check that reactant/product counts are consistent with type

   if (r->type == IONIZATION) {
      if (r->nreactant != 1 || (r->nproduct != 1 && r->nproduct != 1)) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid ionization reaction");
      }
    } else if (r->type == RECOMBINATION) {
      if (r->nreactant != 1 || r->nproduct != 1) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid recombination reaction");
      }
    } else if (r->type == EXCHANGE) {
      if (r->nreactant != 1 || r->nproduct != 1) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid charge exchange reaction");
      }
    } else if (r->type == DISSOCIATION) {
      if (r->nreactant != 1 || (r->nproduct != 1 && r->nproduct != 2)) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid dissociation reaction");
      }
    }

    word = strtok(NULL," \t\n\r");
    if (!word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (word[0] == 'A' || word[0] == 'a') r->style = ADAS;
    else if (word[0] == 'J' || word[0] == 'j') r->style = JANEV;
    else {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (r->style == ADAS) r->ncoeff = 5;
    else if (r->style == JANEV) r->ncoeff = 9;  // b0..b8

    for (int i = 0; i < r->ncoeff; i++) {
      word = strtok(NULL," \t\n\r");
      if (!word) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid reaction coefficients in file");
      }
      r->coeff[i] = input->numeric(FLERR,word);
    }

    word = strtok(NULL," \t\n\r");
    if (word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Too many coefficients in a reaction formula");
    }

    nlist++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   print reaction as read from file
   only proc 0 performs output
------------------------------------------------------------------------- */

void FixVolumeChemAdas::print_reaction(char *line1, char *line2)
{
  if (comm->me) return;
  printf("Bad reaction format:\n");
  printf("%s\n%s\n",line1,line2);
};

/* ----------------------------------------------------------------------
   print reaction as stored in rlist
   only proc 0 performs output
------------------------------------------------------------------------- */

void FixVolumeChemAdas::print_reaction(OneReaction *r)
{
  if (comm->me) return;
  printf("Bad reaction:\n");
  char type;
  if (r->type == IONIZATION) type = 'I';
  else if (r->type == RECOMBINATION) type = 'R';
  else if (r->type == EXCHANGE) type = 'E';
  else if (r->type == DISSOCIATION) type = 'D';
  else type = '?';

  char style;
  if (r->style == ADAS) style = 'A';
  else if (r->style == JANEV) style = 'J';
  else style = '?';

  if (r->nproduct == 1)
    printf("  %c %c: %s + %s --> %s\n",type,style,
           r->id_reactants[0],r->id_reactants[1],
           r->id_products[0]);
};

/* ----------------------------------------------------------------------
   read one reaction from file
   reaction = 2 lines
   return 1 if end-of-file, else return 0
------------------------------------------------------------------------- */

int FixVolumeChemAdas::readone(char *line1, char *line2, int &n1, int &n2)
{
  char *eof;
  while ((eof = fgets(line1,MAXLINE,fp))) {
    size_t pre = strspn(line1," \t\n\r");
    if (pre == strlen(line1) || line1[pre] == '#') continue;
    eof = fgets(line2,MAXLINE,fp);
    if (!eof) break;
    n1 = strlen(line1) + 1;
    n2 = strlen(line2) + 1;
    return 0;
  }

  return 1;
}


/* ----------------------------------------------------------------------
   check for duplicates in list of reactions read from file
   error if any exist
------------------------------------------------------------------------- */

void FixVolumeChemAdas::check_duplicate()
{
  OneReaction *r,*s;

  for (int i = 0; i < nlist; i++) {
    r = &rlist[i];

    for (int j = i+1; j < nlist; j++) {
      s = &rlist[j];

      if (r->type != s->type) continue;
      if (r->style != s->style) continue;
      if (r->nreactant != s->nreactant) continue;
      if (r->nproduct != s->nproduct) continue;

      int reactant_match = 0;
      if (strcmp(r->id_reactants[0],s->id_reactants[0]) == 0)
        reactant_match = 1;
      if (!reactant_match) continue;

      int product_match = 0;
      if (r->nproduct == 1) {
        if (strcmp(r->id_products[0],s->id_products[0]) == 0)
          product_match = 1;
      } 
      if (!product_match) continue;

      if (comm->me == 0) {
        printf("MATCH %d %d %d: %d\n",i,j,nlist,product_match);
        printf("MATCH %d %d\n",
               r->products[0],s->products[0]);
      }
      print_reaction(r);
      print_reaction(s);
      error->all(FLERR,"Duplicate reactions in reaction file");
    }
  }
}

/* ----------------------------------------------------------------------
   Compute Poisson rate λ = k * ne * dt for a single reaction channel.
   Returns λ (dimensionless).  Caller sums λ across channels and draws
   from Poisson: P(≥1 event) = 1 - exp(-λ_total).
------------------------------------------------------------------------- */
double FixVolumeChemAdas::computeReactionLambda(double rate_log10_cm3s, // log10(k [cm^3/s])
                                          double dt,              // [s]
                                          double ne_m3)           // [m^-3]
{
  if (!(dt > 0.0) || !(ne_m3 > 0.0) || !std::isfinite(rate_log10_cm3s))
    return 0.0;

  // k in cm^3/s -> m^3/s  (exp is ~5x faster than pow(10,x))
  const double k_cm3s = std::exp(rate_log10_cm3s * 2.302585092994046);
  const double k_m3s  = std::max(0.0, k_cm3s) * 1e-6;

  // λ = k [m^3/s] * n_e [m^-3] * dt [s]
  double lambda = k_m3s * ne_m3 * dt;

  if (!std::isfinite(lambda)) return 50.0;
  return std::min(lambda, 50.0);
}


void FixVolumeChemAdas::interpolateRateData(int atomic_number, double charge, int /*icell*/, double te, double ne, double& rate_final, ReactionType reactionType) {

  size_t charge_idx = static_cast<size_t>(charge);

  double x0, x1, y0, y1;
  double f00, f01, f10, f11;
  bool success = setupInterpolation(reactionType, atomic_number, charge_idx, te, ne, x0, x1, y0, y1, f00, f01, f10, f11);
  if (!success) {
      rate_final = 0.0;
      return;
  }

  MathExtra::bilinearInterpolate(x0, x1, y0, y1, f00, f01, f10, f11, te, ne, rate_final);
}


/*----------------------------------------------------------------------
   find indices for bilinear interpolation
------------------------------------------------------------------------- */

bool FixVolumeChemAdas::setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx, double te, double ne, double& x0, double& x1, double& y0, double& y1, double& f00, double& f01, double& f10, double& f11) {
  auto& rd = materials_rate_data[atomic_number];

  auto bracket_index = [](const std::vector<double>& grid, double x,
                          size_t &ilo, size_t &ihi) -> bool {
    const size_t n = grid.size();
    if (n < 2) return false;
    if (x <= grid[0])   { ilo = 0; ihi = 1; return true; }
    if (x >= grid[n-1]) { ihi = n-1; ilo = ihi-1; return true; }
    const size_t hi = static_cast<size_t>(
        std::lower_bound(grid.begin(), grid.end(), x) - grid.begin());
    if (hi == 0 || hi >= n) return false;
    ihi = hi;
    ilo = hi - 1;
    return true;
  };

  size_t tlo, thi, nlo, nhi;

  if (reactionType == ReactionType::Recombination) {
      if (static_cast<int>(charge_idx) >= rd.rec_nQ) return false;
      if (!bracket_index(rd.gridT_rec, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_rec, ne, nlo, nhi)) return false;

      x0 = rd.gridT_rec[tlo];  x1 = rd.gridT_rec[thi];
      y0 = rd.gridD_rec[nlo];  y1 = rd.gridD_rec[nhi];

      // flat layout: [charge][temperature][density]
      f00 = rd.rec_at(charge_idx, tlo, nlo);
      f01 = rd.rec_at(charge_idx, tlo, nhi);
      f10 = rd.rec_at(charge_idx, thi, nlo);
      f11 = rd.rec_at(charge_idx, thi, nhi);

  } else if (reactionType == ReactionType::ChargeExchange) {
      if (static_cast<int>(charge_idx) >= rd.cx_nQ) return false;
      if (!bracket_index(rd.gridT_cx, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_cx, ne, nlo, nhi)) return false;

      x0 = rd.gridT_cx[tlo];  x1 = rd.gridT_cx[thi];
      y0 = rd.gridD_cx[nlo];  y1 = rd.gridD_cx[nhi];

      f00 = rd.cx_at(charge_idx, tlo, nlo);
      f01 = rd.cx_at(charge_idx, tlo, nhi);
      f10 = rd.cx_at(charge_idx, thi, nlo);
      f11 = rd.cx_at(charge_idx, thi, nhi);

  } else if (reactionType == ReactionType::LineRadiation) {
      if (static_cast<int>(charge_idx) >= rd.plt_nQ) return false;
      if (!bracket_index(rd.gridT_plt, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_plt, ne, nlo, nhi)) return false;

      x0 = rd.gridT_plt[tlo];  x1 = rd.gridT_plt[thi];
      y0 = rd.gridD_plt[nlo];  y1 = rd.gridD_plt[nhi];

      f00 = rd.plt_at(charge_idx, tlo, nlo);
      f01 = rd.plt_at(charge_idx, tlo, nhi);
      f10 = rd.plt_at(charge_idx, thi, nlo);
      f11 = rd.plt_at(charge_idx, thi, nhi);

  } else if (reactionType == ReactionType::RecombRadiation) {
      if (static_cast<int>(charge_idx) >= rd.prb_nQ) return false;
      if (!bracket_index(rd.gridT_prb, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_prb, ne, nlo, nhi)) return false;

      x0 = rd.gridT_prb[tlo];  x1 = rd.gridT_prb[thi];
      y0 = rd.gridD_prb[nlo];  y1 = rd.gridD_prb[nhi];

      f00 = rd.prb_at(charge_idx, tlo, nlo);
      f01 = rd.prb_at(charge_idx, tlo, nhi);
      f10 = rd.prb_at(charge_idx, thi, nlo);
      f11 = rd.prb_at(charge_idx, thi, nhi);

  } else {
      if (static_cast<int>(charge_idx) >= rd.ion_nQ) return false;
      if (!bracket_index(rd.gridT_ion, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_ion, ne, nlo, nhi)) return false;

      x0 = rd.gridT_ion[tlo];  x1 = rd.gridT_ion[thi];
      y0 = rd.gridD_ion[nlo];  y1 = rd.gridD_ion[nhi];

      f00 = rd.ion_at(charge_idx, tlo, nlo);
      f01 = rd.ion_at(charge_idx, tlo, nhi);
      f10 = rd.ion_at(charge_idx, thi, nlo);
      f11 = rd.ion_at(charge_idx, thi, nhi);
  }

  return true;
}



