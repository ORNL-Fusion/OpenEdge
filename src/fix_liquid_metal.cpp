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
         [evap yes/no] [keywords ...]

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
#include <cmath>
#include <algorithm>
#include <vector>

using namespace SPARTA_NS;

enum { INT, DOUBLE };
enum { COMPUTE, FIX, CONSTANT };

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

  } else {
    hf_source = CONSTANT;
    hf_constant = input->numeric(FLERR, arg[4]);
  }

  // defaults
  id_custom_t = NULL;
  id_custom_evap = NULL;
  id_custom_adatom = NULL;
  id_custom_h = NULL;

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

  // parse keyword/value pairs from arg[5] onward
  int iarg = 5;

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

  // set heat flux on the strip for constant source
  if (hf_source == CONSTANT) {
    double Xlength = 0.0;
    if (!surf_arc_len.empty())
      Xlength = *std::max_element(surf_arc_len.begin(), surf_arc_len.end());
    if (Xlength <= 0.0) Xlength = strip.h0 * 100.0;

    strip.Xl = Xlength / strip.h0;
    strip.hx = strip.Xl / (strip.Nx - 1);
    strip.Tscale = strip.qss * strip.h0 / strip.li.k_th;

    for (int n = 1; n <= strip.Nx; n++) {
      strip.X[n] = (n - 1) * strip.hx;
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

  // gather heat flux and solve strip MHD model
  gather_heat_flux();
  strip.solve_steady();

  // gather D+ flux for ad-atom calculation
  std::vector<double> dp_per_surf;
  gather_dp_flux(dp_per_surf);

  // map outputs to per-surf custom attributes
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
