/* ----------------------------------------------------------------------
   OpenEdge:
   Per-surface adatom desorption flux. Arrhenius (D-on-Li Yad model).
------------------------------------------------------------------------- */

#include "compute_surface_chemical_adatom.h"

#include <cstring>
#include <cstdlib>

#include "surf.h"
#include "domain.h"
#include "input.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "modify.h"
#include "fix.h"
#include "liquid_metal_strip.h"

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputeSurfaceChemicalAdatom::ComputeSurfaceChemicalAdatom(
    SPARTA *sparta, int narg, char **arg) : Compute(sparta, narg, arg)
{
  if (narg < 7)
    error->all(FLERR,
      "compute surface/chemical/adatom: "
      "Usage: ID surfgroup tsurf <attr> dp_flux <c_|f_|c_[col]|f_[col]> "
      "[Yad VAL] [Yad_Yps VAL] [f_ad VAL] [A_arr VAL] [E_eff VAL]");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,
      "compute surface/chemical/adatom: surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  tsurf_name = nullptr;
  dp_id = nullptr;
  dp_kind = SRC_NONE;
  dp_col = 0;
  dp_index = -1;

  // Defaults from liquid_metal_strip.h li_adatom_flux signature.
  Yad_D_Li = 1e-3;
  Yad_Yps  = 1.0;
  f_ad     = 1.0;
  A_arr    = 1e-7;
  E_eff    = 0.9;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "tsurf") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute surface/chemical/adatom: tsurf needs a value");
      const int n = strlen(arg[iarg+1]) + 1;
      delete [] tsurf_name;
      tsurf_name = new char[n];
      strcpy(tsurf_name, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dp_flux") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "compute surface/chemical/adatom: dp_flux needs a value");
      const char *tok = arg[iarg+1];
      if (strncmp(tok, "c_", 2) == 0)      dp_kind = SRC_COMPUTE;
      else if (strncmp(tok, "f_", 2) == 0) dp_kind = SRC_FIX;
      else error->all(FLERR,
        "compute surface/chemical/adatom: dp_flux must be c_<id>[col] or f_<id>[col]");
      const char *body = tok + 2;
      const char *lb = strchr(body, '[');
      int idlen;
      if (lb) {
        if (tok[strlen(tok)-1] != ']')
          error->all(FLERR, "compute surface/chemical/adatom: bad dp_flux token");
        idlen = lb - body;
        dp_col = atoi(lb + 1);
        if (dp_col <= 0)
          error->all(FLERR, "compute surface/chemical/adatom: dp_flux column must be >=1");
      } else {
        idlen = strlen(body);
        dp_col = 0;
      }
      delete [] dp_id;
      dp_id = new char[idlen + 1];
      strncpy(dp_id, body, idlen);
      dp_id[idlen] = '\0';
      iarg += 2;
    } else if (strcmp(arg[iarg], "Yad") == 0) {
      Yad_D_Li = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "Yad_Yps") == 0) {
      Yad_Yps = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "f_ad") == 0) {
      f_ad = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "A_arr") == 0) {
      A_arr = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else if (strcmp(arg[iarg], "E_eff") == 0) {
      E_eff = input->numeric(FLERR, arg[iarg+1]); iarg += 2;
    } else {
      error->all(FLERR, "compute surface/chemical/adatom: unknown keyword");
    }
  }

  if (!tsurf_name)
    error->all(FLERR, "compute surface/chemical/adatom: tsurf <attr> required");
  if (dp_kind == SRC_NONE || !dp_id)
    error->all(FLERR, "compute surface/chemical/adatom: dp_flux <source> required");

  per_surf_flag = 1;
  size_per_surf_cols = 0;
  vector_surf = nullptr;
  nslocal = 0;
  tsurf_index = -1;
}

/* ---------------------------------------------------------------------- */

ComputeSurfaceChemicalAdatom::~ComputeSurfaceChemicalAdatom()
{
  if (copymode) return;
  delete [] tsurf_name;
  delete [] dp_id;
  memory->destroy(vector_surf);
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalAdatom::init()
{
  tsurf_index = surf->find_custom(tsurf_name);
  if (tsurf_index < 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
      "compute surface/chemical/adatom: surf custom '%s' not found", tsurf_name);
    error->all(FLERR, msg);
  }

  if (dp_kind == SRC_COMPUTE) {
    dp_index = modify->find_compute(dp_id);
    if (dp_index < 0)
      error->all(FLERR, "compute surface/chemical/adatom: dp_flux compute ID not found");
    if (!modify->compute[dp_index]->per_surf_flag)
      error->all(FLERR, "compute surface/chemical/adatom: dp_flux compute is not per-surf");
  } else {
    dp_index = modify->find_fix(dp_id);
    if (dp_index < 0)
      error->all(FLERR, "compute surface/chemical/adatom: dp_flux fix ID not found");
    if (!modify->fix[dp_index]->per_surf_flag)
      error->all(FLERR, "compute surface/chemical/adatom: dp_flux fix is not per-surf");
  }

  reallocate();
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalAdatom::reallocate()
{
  const int n = surf->nown;
  if (n == nslocal && vector_surf) return;
  memory->destroy(vector_surf);
  nslocal = n;
  memory->create(vector_surf, nslocal, "surface/chemical/adatom:vector");
  for (int i = 0; i < nslocal; i++) vector_surf[i] = 0.0;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfaceChemicalAdatom::compute_per_surf()
{
  invoked_per_surf = update->ntimestep;
  reallocate();

  // Pull dp_flux per-surf from the bound compute or fix.
  double *dp_vec = nullptr;
  double **dp_arr = nullptr;
  if (dp_kind == SRC_COMPUTE) {
    Compute *c = modify->compute[dp_index];
    if (!(c->invoked_flag & 32)) {  // INVOKED_PER_SURF
      c->compute_per_surf();
      c->invoked_flag |= 32;
    }
    if (dp_col == 0) dp_vec = c->vector_surf;
    else             dp_arr = c->array_surf;
  } else {
    Fix *f = modify->fix[dp_index];
    if (dp_col == 0) dp_vec = f->vector_surf;
    else             dp_arr = f->array_surf;
  }

  const int dim = domain->dimension;
  const int distributed = surf->distributed;
  Surf::Line *lines = distributed ? surf->mylines : surf->lines;
  Surf::Tri  *tris  = distributed ? surf->mytris  : surf->tris;
  const int me = comm->me, nprocs = comm->nprocs;

  double *tvec = surf->edvec[surf->ewhich[tsurf_index]];

  for (int i = 0; i < nslocal; i++) {
    const int m = distributed ? i : me + i * nprocs;
    int mask;
    if (dim == 2) mask = lines[m].mask;
    else          mask = tris[m].mask;

    if (!(mask & groupbit)) {
      vector_surf[i] = 0.0;
      continue;
    }

    double dp = 0.0;
    if (dp_vec) dp = dp_vec[i];
    else if (dp_arr) dp = dp_arr[i][dp_col - 1];

    vector_surf[i] = LiquidMetal::li_adatom_flux(
        tvec[i], dp, Yad_D_Li, Yad_Yps, f_ad, A_arr, E_eff);
  }
}
