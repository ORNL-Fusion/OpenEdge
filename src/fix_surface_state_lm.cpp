/* ----------------------------------------------------------------------
   OpenEdge - Plasma-edge particle transport code
   https://github.com/ORNL-Fusion/OpenEdge

   fix liquid_metal: MHD liquid metal film model for divertor surfaces.
   Solves Smolentsev shallow-water MHD + heat transfer equations on a
   1D strip along the divertor, computes surface temperature,
   Li evaporation flux (Antoine+HK), ad-atom flux, and film thickness
   as per-surf custom attributes.

   Syntax:
     fix ID liquid_metal group Nevery hf_source \
         h0 VAL U0 VAL Bs VAL alpha VAL width VAL Tin VAL \
         [dp_flux SOURCE] [Yad VAL] [E_eff VAL] [A_arr VAL] \
         [evap yes/no] [hf_scale VAL] [keywords ...]

   hf_source can be:
     - c_compute[col] : per-surf compute
     - f_fix[col]     : per-surf fix
     - CONSTANT       : uniform heat flux [W/m²]
     - plasma ID      : point-query from compute plasma/fields
                        (also provides D+ flux = ni * u_parallel)

   Per-surf output columns (via f_ID[i][1..4]):
     1: Tsurf [C]
     2: evap_flux [atoms/m²/s]
     3: adatom_flux [atoms/m²/s]
     4: h_film [m]

   Contributing author: Abdou Diaw (ORNL)
   Based on Fortran code by Sergey Smolentsev (UCLA)
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "math.h"
#include "fix_surface_state_lm.h"
#include "domain.h"
#include "comm.h"
#include "surf.h"
#include "modify.h"
#include "compute.h"
#include "fix.h"
#include "input.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "compute_plasma_fields.h"
#include "fix_background.h"
#include "openedge_geom.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <hdf5.h>

using namespace SPARTA_NS;

enum { INT, DOUBLE };
enum { COMPUTE, FIX, CONSTANT, PLASMA, TARGET, BACKGROUND, SOLPS_B2PL };

/* ---------------------------------------------------------------------- */

FixSurfaceStateLm::FixSurfaceStateLm(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg < 14)
    error->all(FLERR,
      "Illegal fix liquid_metal command: not enough arguments\n"
      "Usage: fix ID liquid_metal group Nevery hf_source "
      "h0 VAL U0 VAL Bs VAL alpha VAL width VAL Tin VAL [keywords]");

  if (surf->implicit)
    error->all(FLERR, "Cannot use fix liquid_metal with implicit surfs");

  // per-surf array output: 4 columns (Tsurf, evap, adatom, h)
  per_surf_flag = 1;
  size_per_surf_cols = 4;

  // surface group
  int igroup = surf->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR, "Fix liquid_metal group ID does not exist");
  groupbit = surf->bitmask[igroup];

  // frequency
  nevery = atoi(arg[3]);

  // heat flux source
  id_hf = NULL;
  chf = NULL;
  fhf = NULL;
  hf_index = 0;
  hf_constant = 0.0;

  if (strncmp(arg[4], "c_", 2) == 0) {
    hf_source = COMPUTE;
    int n = strlen(arg[4]);
    id_hf = new char[n];
    strcpy(id_hf, &arg[4][2]);

    char *ptr = strchr(id_hf, '[');
    if (ptr) {
      if (id_hf[strlen(id_hf) - 1] != ']')
        error->all(FLERR, "Invalid heat flux source in fix liquid_metal");
      hf_index = atoi(ptr + 1);
      *ptr = '\0';
    }

    int icompute = modify->find_compute(id_hf);
    if (icompute < 0)
      error->all(FLERR, "Fix liquid_metal compute ID not found");
    chf = modify->compute[icompute];
    if (chf->per_surf_flag == 0)
      error->all(FLERR,
        "Fix liquid_metal compute does not produce per-surf info");

  } else if (strncmp(arg[4], "f_", 2) == 0) {
    hf_source = FIX;
    int n = strlen(arg[4]);
    id_hf = new char[n];
    strcpy(id_hf, &arg[4][2]);

    char *ptr = strchr(id_hf, '[');
    if (ptr) {
      if (id_hf[strlen(id_hf) - 1] != ']')
        error->all(FLERR, "Invalid heat flux source in fix liquid_metal");
      hf_index = atoi(ptr + 1);
      *ptr = '\0';
    }

    int ifix = modify->find_fix(id_hf);
    if (ifix < 0)
      error->all(FLERR, "Fix liquid_metal fix ID not found");
    fhf = modify->fix[ifix];
    if (fhf->per_surf_flag == 0)
      error->all(FLERR,
        "Fix liquid_metal fix does not produce per-surf info");

  } else if (strcmp(arg[4], "plasma") == 0) {
    hf_source = PLASMA;
    if (narg < 7)
      error->all(FLERR, "Fix liquid_metal: 'plasma' requires compute ID");
    int n = strlen(arg[5]) + 1;
    id_plasma = new char[n];
    strcpy(id_plasma, arg[5]);
  } else if (strcmp(arg[4], "background") == 0) {
    hf_source = BACKGROUND;
    if (narg < 7)
      error->all(FLERR, "Fix liquid_metal: 'background' requires fix ID");
    int n = strlen(arg[5]) + 1;
    id_background = new char[n];
    strcpy(id_background, arg[5]);
  } else if (strcmp(arg[4], "target") == 0) {
    hf_source = TARGET;
    if (narg < 7)
      error->all(FLERR, "Fix liquid_metal: 'target' requires FILE and LEG (outer/inner)");
    int n = strlen(arg[5]) + 1;
    target_file = new char[n];
    strcpy(target_file, arg[5]);
    n = strlen(arg[6]) + 1;
    target_leg = new char[n];
    strcpy(target_leg, arg[6]);
  } else if (strcmp(arg[4], "solps_b2pl") == 0) {
    hf_source = SOLPS_B2PL;
    if (narg < 6)
      error->all(FLERR,
        "Fix liquid_metal: 'solps_b2pl' requires path to ld_tg_*.dat");
    int n = strlen(arg[5]) + 1;
    solps_b2pl_path = new char[n];
    strcpy(solps_b2pl_path, arg[5]);
  } else {
    hf_source = CONSTANT;
    hf_constant = input->numeric(FLERR, arg[4]);
  }

  // defaults
  id_custom_t = NULL;
  id_custom_h = NULL;
  id_custom_g = NULL;
  gindex = -1;

  // plasma compute pointer (for PLASMA mode)
  cp_plasma = NULL;
  if (hf_source != PLASMA) id_plasma = NULL;
  hf_scale = 1.0;

  // fix background pointer (for BACKGROUND mode)
  fbg = NULL;
  if (hf_source != BACKGROUND) id_background = NULL;

  // target file defaults
  if (hf_source != TARGET) {
    target_file = NULL;
    target_leg = NULL;
  }

  // solps_b2pl defaults
  if (hf_source != SOLPS_B2PL) solps_b2pl_path = NULL;
  solps_m_ion_amu = 2.014;          // D+ default; allow override via keyword

  // static-mode defaults (off; user opts in with `static yes`)
  static_mode = 0;
  frozen_ = 0;

  // parse keyword/value pairs (start after hf_source args)
  int iarg = 5;
  if (hf_source == PLASMA) iarg = 6;
  if (hf_source == TARGET) iarg = 7;
  if (hf_source == BACKGROUND) iarg = 6;
  if (hf_source == SOLPS_B2PL) iarg = 6;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "h0") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for h0");
      strip.h0 = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "U0") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for U0");
      strip.U0 = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Bs") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Bs");
      strip.Bs = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Bw") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Bw");
      strip.Bw = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "alpha") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for alpha");
      strip.alpha_deg = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "width") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for width");
      strip.width = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "csv") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for csv");
      csv_path = std::string(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Tin") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Tin");
      strip.Tin = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Nx") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Nx");
      strip.Nx = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Ny") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Ny");
      strip.Ny = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "sigma_w") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for sigma_w");
      strip.sigma_w = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "tw") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for tw");
      strip.tw = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "qss") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for qss");
      strip.qss = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ncase") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for ncase");
      strip.ncase = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "max_iter") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for max_iter");
      strip.max_iter = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "eps") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for eps");
      strip.eps_conv = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "relax") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for relax");
      strip.relax = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dt") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for dt");
      strip.dt_pseudo = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "evap") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for evap");
      if (strcmp(arg[iarg + 1], "yes") == 0) strip.evap_on = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) strip.evap_on = 0;
      else error->all(FLERR, "Invalid evap value: use yes or no");
      iarg += 2;

    } else if (strcmp(arg[iarg], "hf_scale") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for hf_scale");
      hf_scale = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "static") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for static");
      if (strcmp(arg[iarg + 1], "yes") == 0) static_mode = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) static_mode = 0;
      else error->all(FLERR, "Invalid static value: use yes or no");
      iarg += 2;

    } else if (strcmp(arg[iarg], "custom") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for custom");
      int n = strlen(arg[iarg + 1]) + 1;
      id_custom_t = new char[n];
      strcpy(id_custom_t, arg[iarg + 1]);
      iarg += 2;
    } else {
      char msg[256];
      snprintf(msg, 256, "Unknown keyword in fix liquid_metal: %s", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  // set default custom attribute names. Tsurf and h_film are the
  // model's persistent state. Evaporation and adatom fluxes are
  // computed from these by `compute surface/chemical/evaporation`
  // and `compute surface/chemical/adatom` respectively.
  if (!id_custom_t) {
    id_custom_t = new char[16];
    strcpy(id_custom_t, "Tsurf_lm");
  }
  id_custom_h = new char[16];
  strcpy(id_custom_h, "h_lm");
  id_custom_g = new char[16];
  strcpy(id_custom_g, "Gamma_D_lm");

  // create custom per-surf attributes
  tindex = surf->find_custom(id_custom_t);
  if (tindex < 0) tindex = surf->add_custom(id_custom_t, DOUBLE, 0);

  hindex = surf->find_custom(id_custom_h);
  if (hindex < 0) hindex = surf->add_custom(id_custom_h, DOUBLE, 0);

  gindex = surf->find_custom(id_custom_g);
  if (gindex < 0) gindex = surf->add_custom(id_custom_g, DOUBLE, 0);

  firstflag = 1;
}

/* ---------------------------------------------------------------------- */

FixSurfaceStateLm::~FixSurfaceStateLm()
{
  delete[] id_hf;
  delete[] id_plasma;
  delete[] target_file;
  delete[] target_leg;
  delete[] solps_b2pl_path;
  delete[] id_custom_t;
  delete[] id_custom_h;
  delete[] id_custom_g;
  surf->remove_custom(tindex);
  surf->remove_custom(hindex);
  if (gindex >= 0) surf->remove_custom(gindex);
}

/* ---------------------------------------------------------------------- */

int FixSurfaceStateLm::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceStateLm::init()
{
  // SPARTA calls init() before every run command. strip.init() below
  // re-allocates Tsurf_dim to Tin, so we must let end_of_step re-solve;
  // otherwise a stale frozen_ leaves the wall stuck at the inlet temp.
  if (static_mode) frozen_ = 0;

  // load target heat flux profiles from HDF5
  if (hf_source == TARGET && target_file) {
    load_target_heatflux();
  }
  // load SOLPS b2pl text-file profile (parses on rank 0, broadcasts).
  if (hf_source == SOLPS_B2PL && solps_b2pl_path) {
    load_solps_b2pl();
  }

  // resolve plasma compute for point-query mode
  if (hf_source == PLASMA && id_plasma) {
    int ic = modify->find_compute(id_plasma);
    if (ic < 0) error->all(FLERR, "Fix surface/state/liquid_metal: plasma compute ID not found");
    cp_plasma = dynamic_cast<ComputePlasmaFields*>(modify->compute[ic]);
    if (!cp_plasma)
      error->all(FLERR, "Fix surface/state/liquid_metal: plasma source must be a plasma/fields compute");
  }

  // resolve fix background for BACKGROUND mode (heat flux from plasma.h5)
  if (hf_source == BACKGROUND && id_background) {
    int ifix = modify->find_fix(id_background);
    if (ifix < 0)
      error->all(FLERR,
        "Fix surface/state/liquid_metal: background fix ID not found");
    fbg = dynamic_cast<FixBackground*>(modify->fix[ifix]);
    if (!fbg)
      error->all(FLERR,
        "Fix surface/state/liquid_metal: 'background' arg must reference a fix background");
  }

  // initialize strip solver
  strip.init();

  // build geometry mapping from SPARTA surfaces to strip x-stations
  build_geometry_map();

  // set initial values for THIS fix's surface group only.
  // (When several fix surface/state/lm instances share Tsurf_lm via the
  // same custom, an unfiltered loop would clobber the other fix's surfs.)
  if (firstflag) {
    firstflag = 0;

    double *tvec = surf->edvec[surf->ewhich[tindex]];
    double *hvec = surf->edvec[surf->ewhich[hindex]];
    double *gvec = surf->edvec[surf->ewhich[gindex]];
    int nsown = surf->nown;

    int me_init = comm->me;
    int nprocs_init = comm->nprocs;
    int distributed_init = surf->distributed;
    int dim_init = domain->dimension;
    Surf::Line *lines_init = distributed_init ? surf->mylines : surf->lines;
    Surf::Tri  *tris_init  = distributed_init ? surf->mytris  : surf->tris;

    for (int i = 0; i < nsown; i++) {
      int m = distributed_init ? i : me_init + i * nprocs_init;
      int mask = (dim_init == 2) ? lines_init[m].mask : tris_init[m].mask;
      if (!(mask & groupbit)) continue;
      tvec[i] = strip.Tin;
      hvec[i] = strip.h0;
      gvec[i] = 0.0;
    }
  }

  // set up strip grid from wall arc length.
  // SOLPS-anchored mode: stretch the strip to cover the FULL SOLPS arc
  // range — this is what surf_to_strip indexes into, and what set_heat_flux
  // / TARGET-branch interpolation uses to evaluate Wtot at strip stations.
  double Xlength = 0.0;
  if (hf_source == SOLPS_B2PL && !tgt_s.empty()) {
    Xlength = tgt_s.back();          // tgt_s is shifted so [0]=0
  } else if (!surf_arc_len.empty()) {
    Xlength = *std::max_element(surf_arc_len.begin(), surf_arc_len.end());
  }
  if (Xlength <= 0.0) Xlength = strip.h0 * 100.0;

  strip.Xl = Xlength / strip.h0;
  strip.hx = strip.Xl / (strip.Nx - 1);
  strip.Tscale = strip.qss * strip.h0 / strip.li.k_th;

  for (int n = 1; n <= strip.Nx; n++) {
    strip.X[n] = (n - 1) * strip.hx;
    if (hf_source == CONSTANT) {
      strip.Qs0[n] = hf_constant / strip.qss;
      strip.Qs[n] = strip.Qs0[n];
    }
  }

  // static-mode: solve the strip once now so Tsurf_lm is populated at
  // step 0 (before any run). Subsequent end_of_step calls return early.
  if (static_mode) {
    end_of_step();
    frozen_ = 1;
  }

  // Optional smooth-strip CSV dump (Sergey's solver output, 201 stations).
  // Useful for paper-quality plots that bypass the SPARTA per-surf jagginess.
  if (!csv_path.empty() && comm->me == 0) write_strip_csv();
}

/* ----------------------------------------------------------------------
   Dump strip's smooth Smolentsev profile to CSV.  One row per strip
   station (n=1..Nx).  Columns: s_phys, R, Z, Tsurf, h, Wtot, Gamma_D,
   Gamma_evap.  R/Z/Wtot/Gamma_D are linearly interpolated from the
   SOLPS source data at each strip arc position.
------------------------------------------------------------------------- */
void FixSurfaceStateLm::write_strip_csv()
{
  FILE *fp = fopen(csv_path.c_str(), "w");
  if (!fp) {
    if (comm->me == 0)
      error->warning(FLERR, "fix surface/state/lm: cannot open csv path");
    return;
  }
  fprintf(fp, "s_m,R_m,Z_m,Tsurf_C,h_m,Wtot_Wm2,Gamma_D_m2s,Gamma_evap_m2s\n");

  const int Nx_sm = strip.Nx;
  const int ntgt = static_cast<int>(tgt_s.size());

  auto interp_at_s = [&](const std::vector<double> &arr, double s) -> double {
    if (ntgt == 0 || arr.empty()) return 0.0;
    if (s <= tgt_s[0]) return arr[0];
    if (s >= tgt_s[ntgt - 1]) return arr[ntgt - 1];
    for (int k = 0; k + 1 < ntgt; k++) {
      if (s >= tgt_s[k] && s <= tgt_s[k + 1]) {
        const double dx = tgt_s[k + 1] - tgt_s[k];
        const double f = (dx > 0.0) ? (s - tgt_s[k]) / dx : 0.0;
        return arr[k] + f * (arr[k + 1] - arr[k]);
      }
    }
    return arr[ntgt - 1];
  };

  for (int n = 1; n <= Nx_sm; n++) {
    const double s_phys = strip.X[n] * strip.h0;
    const double Rn = interp_at_s(tgt_R, s_phys);
    const double Zn = interp_at_s(tgt_Z, s_phys);
    const double Wn = interp_at_s(tgt_q, s_phys);
    const double Gn = interp_at_s(tgt_gamma, s_phys);
    const double Tn = strip.Tsurf_dim[n];
    const double hn = strip.h_dim[n];
    const double En = strip.evap_flux[n];
    fprintf(fp, "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
            s_phys, Rn, Zn, Tn, hn, Wn, Gn, En);
  }
  fclose(fp);
  if (comm->me == 0)
    fprintf(screen ? screen : stdout,
            "  fix surface/state/lm: wrote strip profile -> %s (Nx=%d)\n",
            csv_path.c_str(), Nx_sm);
}

/* ---------------------------------------------------------------------- */

void FixSurfaceStateLm::build_geometry_map()
{
  int dimension = domain->dimension;
  int distributed = surf->distributed;

  Surf::Line *lines;
  Surf::Tri *tris;

  if (distributed) {
    lines = surf->mylines;
    tris = surf->mytris;
  } else {
    lines = surf->lines;
    tris = surf->tris;
  }

  int me = comm->me;
  int nprocs = comm->nprocs;
  int nsown = surf->nown;

  struct SurfPt {
    double r, z;
    int local_idx;
  };
  std::vector<SurfPt> pts;

  for (int i = 0; i < nsown; i++) {
    int m;
    int mask;
    if (!distributed) m = me + i * nprocs;
    else m = i;

    if (dimension == 2) {
      mask = lines[m].mask;
      if (!(mask & groupbit)) continue;
      double *p1 = lines[m].p1;
      double *p2 = lines[m].p2;
      SurfPt sp;
      sp.r = 0.5 * (p1[0] + p2[0]);
      sp.z = 0.5 * (p1[1] + p2[1]);
      sp.local_idx = i;
      pts.push_back(sp);
    } else {
      mask = tris[m].mask;
      if (!(mask & groupbit)) continue;
      double *p1 = tris[m].p1;
      double *p2 = tris[m].p2;
      double *p3 = tris[m].p3;
      SurfPt sp;
      sp.r = (p1[0] + p2[0] + p3[0]) / 3.0;
      sp.z = (p1[1] + p2[1] + p3[1]) / 3.0;
      sp.local_idx = i;
      pts.push_back(sp);
    }
  }

  surf_to_strip.resize(nsown, -1);
  surf_arc_len.resize(nsown, 0.0);

  if (pts.empty()) return;

  // SOLPS-anchored mapping: when a SOLPS b2pl ld_tg file is loaded,
  // each SPARTA surf is positioned at the arc length of its (R, Z)-
  // nearest SOLPS data point. This makes SPARTA's strip arc axis
  // coincide with SOLPS's, so the standalone solver and SPARTA see the
  // same Wtot at each surf — independent of SPARTA's mesh resolution.
  if (hf_source == SOLPS_B2PL && !tgt_R.empty() && !tgt_Z.empty()) {
    const int npts_solps = (int)tgt_R.size();
    const double s_max = tgt_s[npts_solps - 1];
    if (s_max <= 0.0) return;
    const bool axi_local = (domain->axisymmetric != 0);
    for (size_t k = 0; k < pts.size(); k++) {
      // pts[k].r/.z are SPARTA-x/-y slot midpoints. The mapping to physical
      // (R, Z) depends on the box convention:
      //   axi      : SPARTA x = Z, y = R
      //   Cartesian: SPARTA x = R, y = Z
      double R_surf, Z_surf;
      if (axi_local) { Z_surf = pts[k].r; R_surf = pts[k].z; }
      else           { R_surf = pts[k].r; Z_surf = pts[k].z; }
      int kbest = 0;
      double dbest = 1.0e30;
      for (int j = 0; j < npts_solps; j++) {
        double dR = R_surf - tgt_R[j];
        double dZ = Z_surf - tgt_Z[j];
        double d2 = dR * dR + dZ * dZ;
        if (d2 < dbest) { dbest = d2; kbest = j; }
      }
      double s_solps = tgt_s[kbest];
      double frac = s_solps / s_max;
      int ix = 1 + (int)(frac * (strip.Nx - 1) + 0.5);
      if (ix < 1) ix = 1;
      if (ix > strip.Nx) ix = strip.Nx;
      surf_to_strip[pts[k].local_idx] = ix;
      surf_arc_len[pts[k].local_idx] = s_solps;
    }
    return;
  }

  // Fallback (non-SOLPS sources): sort surfs by (Z, R) and use cumulative
  // segment length within the SPARTA group as the strip arc-length axis.
  std::sort(pts.begin(), pts.end(),
    [](const SurfPt &a, const SurfPt &b) {
      if (std::fabs(a.z - b.z) > 1e-6) return a.z < b.z;
      return a.r < b.r;
    });

  double arc = 0.0;
  std::vector<double> arc_at_pt(pts.size(), 0.0);

  for (size_t k = 1; k < pts.size(); k++) {
    double dr = pts[k].r - pts[k - 1].r;
    double dz = pts[k].z - pts[k - 1].z;
    arc += std::sqrt(dr * dr + dz * dz);
    arc_at_pt[k] = arc;
  }
  double arc_max = arc;

  for (size_t k = 0; k < pts.size(); k++) {
    double frac = (arc_max > 0.0) ? arc_at_pt[k] / arc_max : 0.0;
    int ix = 1 + (int)(frac * (strip.Nx - 1) + 0.5);
    if (ix < 1) ix = 1;
    if (ix > strip.Nx) ix = strip.Nx;
    surf_to_strip[pts[k].local_idx] = ix;
    surf_arc_len[pts[k].local_idx] = arc_at_pt[k];
  }
}

/* ---------------------------------------------------------------------- */

void FixSurfaceStateLm::gather_heat_flux()
{
  if (hf_source == CONSTANT) return;

  int me = comm->me;
  int nprocs = comm->nprocs;
  int nsown = surf->nown;
  int distributed = surf->distributed;
  int dimension = domain->dimension;

  Surf::Line *lines;
  Surf::Tri *tris;
  if (distributed) {
    lines = surf->mylines;
    tris = surf->mytris;
  } else {
    lines = surf->lines;
    tris = surf->tris;
  }

  for (int n = 1; n <= strip.Nx; n++)
    strip.Qs0[n] = 0.0;
  std::vector<int> count(strip.Nx + 2, 0);

  double *qwvector = NULL;
  double **qwarray = NULL;

  if (hf_source == COMPUTE) {
    chf->post_process_surf();
    if (hf_index == 0) qwvector = chf->vector_surf;
    else qwarray = chf->array_surf;
  } else {
    if (hf_index == 0) qwvector = fhf->vector_surf;
    else qwarray = fhf->array_surf;
  }

  int icol = (hf_index > 0) ? hf_index - 1 : 0;

  for (int i = 0; i < nsown; i++) {
    int m;
    int mask;
    if (!distributed) m = me + i * nprocs;
    else m = i;

    if (dimension == 2) mask = lines[m].mask;
    else mask = tris[m].mask;
    if (!(mask & groupbit)) continue;

    int ix = surf_to_strip[i];
    if (ix < 1 || ix > strip.Nx) continue;

    double qw;
    if (hf_index == 0) qw = qwvector[i];
    else qw = qwarray[i][icol];

    strip.Qs0[ix] += qw / strip.qss;
    count[ix]++;
  }

  for (int n = 1; n <= strip.Nx; n++) {
    if (count[n] > 1) strip.Qs0[n] /= count[n];
    strip.Qs[n] = strip.Qs0[n];
  }
}

/* ---------------------------------------------------------------------- */

void FixSurfaceStateLm::end_of_step()
{
  if (frozen_) return;                                  // static_mode: solved once at init
  if (!static_mode && update->ntimestep % nevery) return;

  int me = comm->me;
  int nprocs = comm->nprocs;
  int nsown = surf->nown;
  int distributed = surf->distributed;
  int dimension = domain->dimension;

  Surf::Line *lines;
  Surf::Tri *tris;
  if (distributed) {
    lines = surf->mylines;
    tris = surf->mytris;
  } else {
    lines = surf->lines;
    tris = surf->tris;
  }

  // --- Gather heat flux per strip station ---
  if (hf_source == PLASMA && cp_plasma) {
    // Point-query plasma at each surface element midpoint; q_mag gives heat flux.
    for (int n = 1; n <= strip.Nx; n++)
      strip.Qs0[n] = 0.0;
    std::vector<int> count(strip.Nx + 2, 0);

    for (int i = 0; i < nsown; i++) {
      int m;
      int mask;
      if (!distributed) m = me + i * nprocs;
      else m = i;

      if (dimension == 2) mask = lines[m].mask;
      else mask = tris[m].mask;
      if (!(mask & groupbit)) continue;

      // surface element midpoint
      double xmid[3];
      if (dimension == 2) {
        xmid[0] = 0.5 * (lines[m].p1[0] + lines[m].p2[0]);
        xmid[1] = 0.5 * (lines[m].p1[1] + lines[m].p2[1]);
        xmid[2] = 0.0;
      } else {
        xmid[0] = (tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0;
        xmid[1] = (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0;
        xmid[2] = (tris[m].p1[2] + tris[m].p2[2] + tris[m].p3[2]) / 3.0;
      }

      PlasmaFileParams pp = cp_plasma->query_plasma_at_point(xmid);

      // heat flux from q_mag with optional scaling
      double qw = pp.q_mag * hf_scale;
      if (!std::isfinite(qw) || qw < 0.0) qw = 0.0;

      int ix = surf_to_strip[i];
      if (ix >= 1 && ix <= strip.Nx) {
        strip.Qs0[ix] += qw / strip.qss;
        count[ix]++;
      }
    }

    for (int n = 1; n <= strip.Nx; n++) {
      if (count[n] > 1) strip.Qs0[n] /= count[n];
      strip.Qs[n] = strip.Qs0[n];
    }

  } else if (hf_source == BACKGROUND && fbg) {
    // Heat flux from fix background's mesh/q_par + mesh/q_perp arrays in
    // plasma.h5. Sample at each surface centroid via fix background's
    // public interp2D (mesh-cell aware when icell is found, falls back
    // to regular-grid bilinear). Average over surfaces sharing each
    // strip station, same as the PLASMA branch above.
    for (int n = 1; n <= strip.Nx; n++) strip.Qs0[n] = 0.0;
    std::vector<int> count(strip.Nx + 2, 0);

    for (int i = 0; i < nsown; i++) {
      int m, mask;
      if (!distributed) m = me + i * nprocs;
      else m = i;
      if (dimension == 2) mask = lines[m].mask;
      else mask = tris[m].mask;
      if (!(mask & groupbit)) continue;

      double xmid[3];
      if (dimension == 2) {
        xmid[0] = 0.5 * (lines[m].p1[0] + lines[m].p2[0]);
        xmid[1] = 0.5 * (lines[m].p1[1] + lines[m].p2[1]);
        xmid[2] = 0.0;
      } else {
        xmid[0] = (tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0;
        xmid[1] = (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0;
        xmid[2] = (tris[m].p1[2] + tris[m].p2[2] + tris[m].p3[2]) / 3.0;
      }
      double R = 0.0, Z = 0.0;
      OpenEdge::sparta_to_RZ(xmid, dimension, domain->axisymmetric, R, Z);

      double q_par  = fbg->default_q_par;
      double q_perp = fbg->default_q_perp;
      if (fbg->has_qheatflux) {
        // Prefer the wall_surf_cell map: identifies the SOLPS cell adjacent
        // to each wall segment, no bilinear extrapolation across the SOLPS
        // mesh boundary (which can yield spurious heat flux at PFR/wall-gap
        // segments and overheat Tsurf there). Fall back to interp2D only
        // when the map isn't populated.
        int wcell = -1;
        if (fbg->has_mesh_wall_surf_cell) {
          const int sid = (dimension == 2)
            ? static_cast<int>(lines[m].id)
            : static_cast<int>(tris[m].id);
          const int wid = sid - 1;
          if (wid >= 0 && wid <
              static_cast<int>(fbg->mesh_wall_surf_cell.size())) {
            wcell = fbg->mesh_wall_surf_cell[wid];
          }
        }
        if (wcell >= 0 && !fbg->mesh_q_par.empty()
            && wcell < static_cast<int>(fbg->mesh_q_par.size())) {
          q_par = fbg->mesh_q_par[wcell];
          if (!fbg->mesh_q_perp.empty()
              && wcell < static_cast<int>(fbg->mesh_q_perp.size()))
            q_perp = fbg->mesh_q_perp[wcell];
        } else {
          if (!fbg->mesh_q_par.empty())
            q_par = fbg->interp2D(fbg->mesh_q_par, R, Z);
          else if (!fbg->q_par.empty())
            q_par = fbg->interp2D(fbg->q_par, R, Z);
          if (!fbg->mesh_q_perp.empty())
            q_perp = fbg->interp2D(fbg->mesh_q_perp, R, Z);
          else if (!fbg->q_perp.empty())
            q_perp = fbg->interp2D(fbg->q_perp, R, Z);
        }
      }
      double qw = std::sqrt(q_par*q_par + q_perp*q_perp) * hf_scale;
      if (!std::isfinite(qw) || qw < 0.0) qw = 0.0;

      int ix = surf_to_strip[i];
      if (ix >= 1 && ix <= strip.Nx) {
        strip.Qs0[ix] += qw / strip.qss;
        count[ix]++;
      }
    }
    for (int n = 1; n <= strip.Nx; n++) {
      if (count[n] > 1) strip.Qs0[n] /= count[n];
      strip.Qs[n] = strip.Qs0[n];
    }

  } else if ((hf_source == TARGET || hf_source == SOLPS_B2PL)
             && !tgt_s.empty()) {
    // Interpolate target heat flux profile onto strip stations.
    int ntgt = (int)tgt_s.size();
    double s_max = tgt_s[ntgt - 1];

    for (int n = 1; n <= strip.Nx; n++) {
      double s_phys = strip.X[n] * strip.h0;
      double qw = 0.0;
      if (s_phys <= tgt_s[0]) qw = tgt_q[0];
      else if (s_phys >= s_max) qw = tgt_q[ntgt - 1];
      else {
        for (int k = 0; k < ntgt - 1; k++) {
          if (s_phys >= tgt_s[k] && s_phys <= tgt_s[k + 1]) {
            double frac = (s_phys - tgt_s[k]) / (tgt_s[k + 1] - tgt_s[k]);
            qw = tgt_q[k] + frac * (tgt_q[k + 1] - tgt_q[k]);
            break;
          }
        }
      }
      strip.Qs0[n] = qw * hf_scale / strip.qss;
      strip.Qs[n] = strip.Qs0[n];
    }

  } else {
    // COMPUTE/FIX/CONSTANT path
    gather_heat_flux();
  }

  // solve strip MHD model to steady state
  strip.solve_steady();

  // map strip state outputs (Tsurf, h_film, Gamma_D) to per-surf customs.
  // Evaporation and adatom fluxes are produced by dedicated SPARTA computes.
  // Gamma_D is non-zero only when the heat-flux source carries it
  // (currently SOLPS_B2PL → tgt_gamma); zero otherwise.
  double *tvec = surf->edvec[surf->ewhich[tindex]];
  double *hvec = surf->edvec[surf->ewhich[hindex]];
  double *gvec = surf->edvec[surf->ewhich[gindex]];

  const bool have_gamma =
    (hf_source == SOLPS_B2PL || hf_source == TARGET) && !tgt_gamma.empty();
  const int ntgt = (int)tgt_s.size();
  const double s_max = ntgt ? tgt_s[ntgt - 1] : 0.0;

  for (int i = 0; i < nsown; i++) {
    int m;
    int mask;
    if (!distributed) m = me + i * nprocs;
    else m = i;

    if (dimension == 2) mask = lines[m].mask;
    else mask = tris[m].mask;
    // Skip surfs not in this fix's group without touching gvec — another
    // fix surface/state/lm instance (e.g. inner vs outer leg) sharing the
    // same Gamma_D_lm custom may have written this slot already.
    if (!(mask & groupbit)) continue;

    int ix = surf_to_strip[i];
    if (ix < 1 || ix > strip.Nx) {
      gvec[i] = 0.0;
      continue;
    }

    tvec[i] = strip.Tsurf_dim[ix];
    hvec[i] = strip.h_dim[ix];

    if (have_gamma) {
      double s_phys = strip.X[ix] * strip.h0;
      double g = 0.0;
      if (s_phys <= tgt_s[0]) g = tgt_gamma[0];
      else if (s_phys >= s_max) g = tgt_gamma[ntgt - 1];
      else {
        for (int k = 0; k < ntgt - 1; k++) {
          if (s_phys >= tgt_s[k] && s_phys <= tgt_s[k + 1]) {
            double frac = (s_phys - tgt_s[k]) / (tgt_s[k + 1] - tgt_s[k]);
            g = tgt_gamma[k] + frac * (tgt_gamma[k + 1] - tgt_gamma[k]);
            break;
          }
        }
      }
      gvec[i] = g;
    } else {
      gvec[i] = 0.0;
    }
  }

  surf->estatus[tindex] = 0;
  surf->estatus[hindex] = 0;
  surf->estatus[gindex] = 0;
}

/* ---------------------------------------------------------------------- */

void FixSurfaceStateLm::load_target_heatflux()
{
  // Read target heat flux and D+ flux profiles from HDF5
  // Format: group "outer" or "inner" containing datasets s, q_total, gamma_D

  if (!target_file || !target_leg) return;

  if (comm->me == 0) {
    hid_t file_id = H5Fopen(target_file, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
      error->one(FLERR, "Fix liquid_metal: cannot open target file");

    hid_t grp = H5Gopen2(file_id, target_leg, H5P_DEFAULT);
    if (grp < 0) {
      H5Fclose(file_id);
      char msg[256];
      snprintf(msg, 256, "Fix liquid_metal: group '%s' not found in %s",
               target_leg, target_file);
      error->one(FLERR, msg);
    }

    // read s dataset to get size
    hid_t ds_s = H5Dopen2(grp, "s", H5P_DEFAULT);
    hid_t space = H5Dget_space(ds_s);
    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, NULL);
    int npts = (int)dims[0];
    H5Sclose(space);

    tgt_s.resize(npts);
    tgt_q.resize(npts);
    tgt_gamma.resize(npts);

    H5Dread(ds_s, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tgt_s.data());
    H5Dclose(ds_s);

    hid_t ds_q = H5Dopen2(grp, "q_total", H5P_DEFAULT);
    H5Dread(ds_q, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tgt_q.data());
    H5Dclose(ds_q);

    hid_t ds_g = H5Dopen2(grp, "gamma_D", H5P_DEFAULT);
    H5Dread(ds_g, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, tgt_gamma.data());
    H5Dclose(ds_g);

    H5Gclose(grp);
    H5Fclose(file_id);

    printf("  fix liquid_metal: loaded %s target from %s (%d points)\n",
           target_leg, target_file, npts);
    printf("    q_total range: [%.2e, %.2e] W/m²\n",
           *std::min_element(tgt_q.begin(), tgt_q.end()),
           *std::max_element(tgt_q.begin(), tgt_q.end()));
    printf("    gamma_D range: [%.2e, %.2e] m⁻²s⁻¹\n",
           *std::min_element(tgt_gamma.begin(), tgt_gamma.end()),
           *std::max_element(tgt_gamma.begin(), tgt_gamma.end()));
  }

  // broadcast to all procs
  int npts = (int)tgt_s.size();
  MPI_Bcast(&npts, 1, MPI_INT, 0, world);
  if (comm->me != 0) {
    tgt_s.resize(npts);
    tgt_q.resize(npts);
    tgt_gamma.resize(npts);
  }
  MPI_Bcast(tgt_s.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_q.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_gamma.data(), npts, MPI_DOUBLE, 0, world);
}

/* ----------------------------------------------------------------------
   Read SOLPS b2pl ld_tg_*.dat:
     col 1 = x [m] (signed arc length, <0 in PFR)
     col 2 = ne [1/m^3]
     col 3 = Te [eV]
     col 4 = Ti [eV]
     col 5 = Wtot [W/m^2]
   We compute Bohm flux from cols 2-4 and store as tgt_gamma. tgt_s is
   shifted so its first entry is 0, matching the strip's relative-arc
   convention used by the existing TARGET interpolation in heat-flux.
------------------------------------------------------------------------- */

void FixSurfaceStateLm::load_solps_b2pl()
{
  if (comm->me == 0) {
    FILE *fp = fopen(solps_b2pl_path, "r");
    if (!fp) {
      char msg[512];
      snprintf(msg, sizeof(msg),
        "Fix liquid_metal: cannot open SOLPS b2pl file '%s'",
        solps_b2pl_path);
      error->one(FLERR, msg);
    }

    // ld_tg_*.dat columns:
    //   1=x  2=ne  3=Te  4=Ti  5=Wtot
    //   6..13 = Wpart..Wptpl   14=xMP   15=Rclfc   16=Zclfc   ...
    // We use cols 1..5 plus 15..16 (Rclfc, Zclfc) for (R, Z) anchoring.
    std::vector<double> xs, ws, gammas, Rs, Zs;
    char buf[1024];
    const double m_ion_kg = solps_m_ion_amu * 1.66053906660e-27;
    const double e_J = 1.602176634e-19;
    while (fgets(buf, sizeof(buf), fp)) {
      char *p = buf;
      while (*p == ' ' || *p == '\t') p++;
      if (*p == '#' || *p == '\n' || *p == '\0') continue;
      double cols[21];
      int n_read = 0;
      const char *cp = buf;
      char *endp;
      while (n_read < 21) {
        double v = strtod(cp, &endp);
        if (endp == cp) break;
        cols[n_read++] = v;
        cp = endp;
      }
      if (n_read < 16) continue;     // need through Rclfc / Zclfc
      double x_v   = cols[0];
      double ne_v  = cols[1];
      double Te_v  = cols[2];
      double Ti_v  = cols[3];
      double W_v   = cols[4];
      double R_v   = cols[14];
      double Z_v   = cols[15];
      double cs = (Te_v + Ti_v > 0.0) ?
        sqrt(e_J * (Te_v + Ti_v) / m_ion_kg) : 0.0;
      xs.push_back(x_v);
      ws.push_back(W_v);
      gammas.push_back(ne_v * cs);
      Rs.push_back(R_v);
      Zs.push_back(Z_v);
    }
    fclose(fp);

    if (xs.size() < 4) {
      char msg[512];
      snprintf(msg, sizeof(msg),
        "Fix liquid_metal: SOLPS b2pl file '%s' has only %zu data rows",
        solps_b2pl_path, xs.size());
      error->one(FLERR, msg);
    }

    // shift x so tgt_s[0] = 0 (matches strip's relative-arc convention)
    double xmin = xs.front();
    int npts = (int)xs.size();
    tgt_s.resize(npts);
    tgt_q.resize(npts);
    tgt_gamma.resize(npts);
    tgt_R.resize(npts);
    tgt_Z.resize(npts);
    for (int i = 0; i < npts; ++i) {
      tgt_s[i] = xs[i] - xmin;
      tgt_q[i] = ws[i];
      tgt_gamma[i] = gammas[i];
      tgt_R[i] = Rs[i];
      tgt_Z[i] = Zs[i];
    }

    printf("  fix liquid_metal: loaded SOLPS b2pl from %s (%d points)\n",
           solps_b2pl_path, npts);
    printf("    arc length: [%.3f, %.3f] m\n", tgt_s.front(), tgt_s.back());
    printf("    R range: [%.3f, %.3f] m   Z range: [%.3f, %.3f] m\n",
           *std::min_element(tgt_R.begin(), tgt_R.end()),
           *std::max_element(tgt_R.begin(), tgt_R.end()),
           *std::min_element(tgt_Z.begin(), tgt_Z.end()),
           *std::max_element(tgt_Z.begin(), tgt_Z.end()));
    printf("    Wtot range: [%.2e, %.2e] W/m^2\n",
           *std::min_element(tgt_q.begin(), tgt_q.end()),
           *std::max_element(tgt_q.begin(), tgt_q.end()));
    printf("    Bohm flux range: [%.2e, %.2e] m^-2 s^-1\n",
           *std::min_element(tgt_gamma.begin(), tgt_gamma.end()),
           *std::max_element(tgt_gamma.begin(), tgt_gamma.end()));
  }

  // broadcast tables to all ranks (same pattern as load_target_heatflux)
  int npts = (int)tgt_s.size();
  MPI_Bcast(&npts, 1, MPI_INT, 0, world);
  if (comm->me != 0) {
    tgt_s.resize(npts);
    tgt_q.resize(npts);
    tgt_gamma.resize(npts);
    tgt_R.resize(npts);
    tgt_Z.resize(npts);
  }
  MPI_Bcast(tgt_s.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_q.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_gamma.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_R.data(), npts, MPI_DOUBLE, 0, world);
  MPI_Bcast(tgt_Z.data(), npts, MPI_DOUBLE, 0, world);
}
