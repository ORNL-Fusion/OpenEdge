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
#include "fix_liquid_metal.h"
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
#include <cmath>
#include <algorithm>
#include <vector>
#include <hdf5.h>

using namespace SPARTA_NS;

enum { INT, DOUBLE };
enum { COMPUTE, FIX, CONSTANT, PLASMA, TARGET };

/* ---------------------------------------------------------------------- */

FixLiquidMetal::FixLiquidMetal(SPARTA *sparta, int narg, char **arg) :
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
  } else {
    hf_source = CONSTANT;
    hf_constant = input->numeric(FLERR, arg[4]);
  }

  // defaults
  id_custom_t = NULL;
  id_custom_evap = NULL;
  id_custom_adatom = NULL;
  id_custom_h = NULL;

  // plasma compute pointer (for PLASMA mode)
  cp_plasma = NULL;
  if (hf_source != PLASMA) id_plasma = NULL;
  hf_scale = 1.0;

  // target file defaults
  if (hf_source != TARGET) {
    target_file = NULL;
    target_leg = NULL;
  }

  // D+ flux source defaults
  id_dp = NULL;
  cdp = NULL;
  fdp = NULL;
  dp_index = 0;
  dp_source = CONSTANT;
  dp_constant = 0.0;

  // ad-atom model defaults
  Yad_D_Li = 1e-3;
  Yad_Yps = 1.0;
  f_ad_neutral = 1.0;
  A_arrhenius = 1e-7;
  E_eff_eV = 0.9;

  // parse keyword/value pairs (start after hf_source args)
  int iarg = 5;
  if (hf_source == PLASMA) iarg = 6;
  if (hf_source == TARGET) iarg = 7;

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

    // --- D+ flux source for ad-atom model ---
    } else if (strcmp(arg[iarg], "dp_flux") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Missing value for dp_flux");
      if (strncmp(arg[iarg + 1], "c_", 2) == 0) {
        dp_source = COMPUTE;
        int n = strlen(arg[iarg + 1]);
        id_dp = new char[n];
        strcpy(id_dp, &arg[iarg + 1][2]);
        char *ptr = strchr(id_dp, '[');
        if (ptr) {
          dp_index = atoi(ptr + 1);
          *ptr = '\0';
        }
      } else if (strncmp(arg[iarg + 1], "f_", 2) == 0) {
        dp_source = FIX;
        int n = strlen(arg[iarg + 1]);
        id_dp = new char[n];
        strcpy(id_dp, &arg[iarg + 1][2]);
        char *ptr = strchr(id_dp, '[');
        if (ptr) {
          dp_index = atoi(ptr + 1);
          *ptr = '\0';
        }
      } else {
        dp_source = CONSTANT;
        dp_constant = input->numeric(FLERR, arg[iarg + 1]);
      }
      iarg += 2;

    // --- ad-atom model parameters ---
    } else if (strcmp(arg[iarg], "Yad") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Yad");
      Yad_D_Li = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "Yad_Yps") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for Yad_Yps");
      Yad_Yps = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "E_eff") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for E_eff");
      E_eff_eV = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "A_arr") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for A_arr");
      A_arrhenius = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "hf_scale") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "Missing value for hf_scale");
      hf_scale = input->numeric(FLERR, arg[iarg + 1]);
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

  // set default custom attribute names
  if (!id_custom_t) {
    id_custom_t = new char[16];
    strcpy(id_custom_t, "Tsurf_lm");
  }
  id_custom_evap = new char[16];
  strcpy(id_custom_evap, "evap_lm");
  id_custom_adatom = new char[16];
  strcpy(id_custom_adatom, "adatom_lm");
  id_custom_h = new char[16];
  strcpy(id_custom_h, "h_lm");

  // create custom per-surf attributes
  tindex = surf->find_custom(id_custom_t);
  if (tindex < 0) tindex = surf->add_custom(id_custom_t, DOUBLE, 0);

  evap_index = surf->find_custom(id_custom_evap);
  if (evap_index < 0) evap_index = surf->add_custom(id_custom_evap, DOUBLE, 0);

  adatom_index = surf->find_custom(id_custom_adatom);
  if (adatom_index < 0) adatom_index = surf->add_custom(id_custom_adatom, DOUBLE, 0);

  hindex = surf->find_custom(id_custom_h);
  if (hindex < 0) hindex = surf->add_custom(id_custom_h, DOUBLE, 0);

  firstflag = 1;
}

/* ---------------------------------------------------------------------- */

FixLiquidMetal::~FixLiquidMetal()
{
  delete[] id_hf;
  delete[] id_dp;
  delete[] id_plasma;
  delete[] target_file;
  delete[] target_leg;
  delete[] id_custom_t;
  delete[] id_custom_evap;
  delete[] id_custom_adatom;
  delete[] id_custom_h;
  surf->remove_custom(tindex);
  surf->remove_custom(evap_index);
  surf->remove_custom(adatom_index);
  surf->remove_custom(hindex);
}

/* ---------------------------------------------------------------------- */

int FixLiquidMetal::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixLiquidMetal::init()
{
  // load target heat flux profiles from HDF5
  if (hf_source == TARGET && target_file) {
    load_target_heatflux();
    dp_source = TARGET;  // D+ flux also from target file
  }

  // resolve plasma compute for point-query mode
  if (hf_source == PLASMA && id_plasma) {
    int ic = modify->find_compute(id_plasma);
    if (ic < 0) error->all(FLERR, "Fix liquid_metal: plasma compute ID not found");
    cp_plasma = dynamic_cast<ComputePlasmaFields*>(modify->compute[ic]);
    if (!cp_plasma)
      error->all(FLERR, "Fix liquid_metal: plasma source must be a plasma/fields compute");
    // in PLASMA mode, D+ flux also comes from the plasma compute
    dp_source = PLASMA;
  }

  // resolve D+ flux source if specified by name
  if (dp_source == COMPUTE && id_dp) {
    int ic = modify->find_compute(id_dp);
    if (ic < 0) error->all(FLERR, "Fix liquid_metal dp_flux compute not found");
    cdp = modify->compute[ic];
  } else if (dp_source == FIX && id_dp) {
    int ifx = modify->find_fix(id_dp);
    if (ifx < 0) error->all(FLERR, "Fix liquid_metal dp_flux fix not found");
    fdp = modify->fix[ifx];
  }

  // initialize strip solver
  strip.init();

  // build geometry mapping from SPARTA surfaces to strip x-stations
  build_geometry_map();

  // set initial values for all surfaces
  if (firstflag) {
    firstflag = 0;

    double *tvec = surf->edvec[surf->ewhich[tindex]];
    double *evapvec = surf->edvec[surf->ewhich[evap_index]];
    double *adatomvec = surf->edvec[surf->ewhich[adatom_index]];
    double *hvec = surf->edvec[surf->ewhich[hindex]];
    int nsown = surf->nown;

    for (int i = 0; i < nsown; i++) {
      tvec[i] = strip.Tin;
      evapvec[i] = 0.0;
      adatomvec[i] = 0.0;
      hvec[i] = strip.h0;
    }
  }

  // set up strip grid from wall arc length
  double Xlength = 0.0;
  if (!surf_arc_len.empty())
    Xlength = *std::max_element(surf_arc_len.begin(), surf_arc_len.end());
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
}

/* ---------------------------------------------------------------------- */

void FixLiquidMetal::build_geometry_map()
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

  std::sort(pts.begin(), pts.end(),
    [](const SurfPt &a, const SurfPt &b) {
      if (std::fabs(a.z - b.z) > 1e-6) return a.z < b.z;
      return a.r < b.r;
    });

  surf_to_strip.resize(nsown, -1);
  surf_arc_len.resize(nsown, 0.0);

  if (pts.empty()) return;

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

void FixLiquidMetal::gather_heat_flux()
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

void FixLiquidMetal::gather_dp_flux(std::vector<double> &dp_per_surf)
{
  int nsown = surf->nown;
  dp_per_surf.assign(nsown, dp_constant);

  if (dp_source == CONSTANT) return;

  double *dpvector = NULL;
  double **dparray = NULL;

  if (dp_source == COMPUTE) {
    cdp->post_process_surf();
    if (dp_index == 0) dpvector = cdp->vector_surf;
    else dparray = cdp->array_surf;
  } else {
    if (dp_index == 0) dpvector = fdp->vector_surf;
    else dparray = fdp->array_surf;
  }

  int icol = (dp_index > 0) ? dp_index - 1 : 0;

  for (int i = 0; i < nsown; i++) {
    if (dp_index == 0) dp_per_surf[i] = dpvector[i];
    else dp_per_surf[i] = dparray[i][icol];
  }
}

/* ---------------------------------------------------------------------- */

void FixLiquidMetal::end_of_step()
{
  if (update->ntimestep % nevery) return;

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

  // --- Gather heat flux and D+ flux ---
  std::vector<double> dp_per_surf(nsown, dp_constant);

  if (hf_source == PLASMA && cp_plasma) {
    // Point-query plasma at each surface element midpoint
    // Gets q_mag for heat flux and ni*u_parallel for D+ flux
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

      // D+ flux = ni * |u_parallel|
      double gamma_dp = pp.dens_i * std::fabs(pp.parr_flow);
      if (!std::isfinite(gamma_dp)) gamma_dp = 0.0;
      dp_per_surf[i] = gamma_dp;
    }

    for (int n = 1; n <= strip.Nx; n++) {
      if (count[n] > 1) strip.Qs0[n] /= count[n];
      strip.Qs[n] = strip.Qs0[n];
    }

  } else if (hf_source == TARGET && !tgt_s.empty()) {
    // Interpolate target heat flux and D+ flux profiles onto strip stations
    int ntgt = (int)tgt_s.size();
    double s_max = tgt_s[ntgt - 1];

    for (int n = 1; n <= strip.Nx; n++) {
      // map strip station to physical arc length
      double s_phys = strip.X[n] * strip.h0;
      // linear interpolation from target profile
      double qw = 0.0, gd = 0.0;
      if (s_phys <= tgt_s[0]) {
        qw = tgt_q[0];
        gd = tgt_gamma[0];
      } else if (s_phys >= s_max) {
        qw = tgt_q[ntgt - 1];
        gd = tgt_gamma[ntgt - 1];
      } else {
        for (int k = 0; k < ntgt - 1; k++) {
          if (s_phys >= tgt_s[k] && s_phys <= tgt_s[k + 1]) {
            double frac = (s_phys - tgt_s[k]) / (tgt_s[k + 1] - tgt_s[k]);
            qw = tgt_q[k] + frac * (tgt_q[k + 1] - tgt_q[k]);
            gd = tgt_gamma[k] + frac * (tgt_gamma[k + 1] - tgt_gamma[k]);
            break;
          }
        }
      }
      strip.Qs0[n] = qw * hf_scale / strip.qss;
      strip.Qs[n] = strip.Qs0[n];
    }

    // map D+ flux to per-surface via arc length
    for (int i = 0; i < nsown; i++) {
      if (surf_to_strip[i] < 1) continue;
      double s_phys = surf_arc_len[i];
      double gd = 0.0;
      if (s_phys <= tgt_s[0]) gd = tgt_gamma[0];
      else if (s_phys >= s_max) gd = tgt_gamma[ntgt - 1];
      else {
        for (int k = 0; k < ntgt - 1; k++) {
          if (s_phys >= tgt_s[k] && s_phys <= tgt_s[k + 1]) {
            double frac = (s_phys - tgt_s[k]) / (tgt_s[k + 1] - tgt_s[k]);
            gd = tgt_gamma[k] + frac * (tgt_gamma[k + 1] - tgt_gamma[k]);
            break;
          }
        }
      }
      dp_per_surf[i] = gd;
    }

  } else {
    // COMPUTE/FIX/CONSTANT path
    gather_heat_flux();
    gather_dp_flux(dp_per_surf);
  }

  // solve strip MHD model to steady state
  strip.solve_steady();

  // map strip outputs to per-surf custom attributes
  double *tvec = surf->edvec[surf->ewhich[tindex]];
  double *evapvec = surf->edvec[surf->ewhich[evap_index]];
  double *adatomvec = surf->edvec[surf->ewhich[adatom_index]];
  double *hvec = surf->edvec[surf->ewhich[hindex]];

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

    double Tsurf = strip.Tsurf_dim[ix];
    tvec[i] = Tsurf;
    evapvec[i] = strip.evap_flux[ix];
    adatomvec[i] = LiquidMetal::li_adatom_flux(
        Tsurf, dp_per_surf[i],
        Yad_D_Li, Yad_Yps, f_ad_neutral, A_arrhenius, E_eff_eV);
    hvec[i] = strip.h_dim[ix];
  }

  surf->estatus[tindex] = 0;
  surf->estatus[evap_index] = 0;
  surf->estatus[adatom_index] = 0;
  surf->estatus[hindex] = 0;
}

/* ---------------------------------------------------------------------- */

void FixLiquidMetal::load_target_heatflux()
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
