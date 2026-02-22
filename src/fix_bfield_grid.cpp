/* ----------------------------------------------------------------------
    OpenEdge: grid magnetic field fix
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - 42d
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_bfield_grid.h"
#include "grid.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "compute.h"
#include "update.h"

#define INVOKED_PER_GRID 16

using namespace SPARTA_NS;


/* ---------------------------------------------------------------------- */

FixBfieldGrid::FixBfieldGrid(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 5 && narg != 7) error->all(FLERR,"Illegal fix bfield/grid command");

  nevery_field = 1;
  if (narg == 7) {
    if (strcmp(arg[5],"every") != 0)
      error->all(FLERR,"Illegal fix bfield/grid command");
    nevery_field = input->inumeric(FLERR,arg[6]);
    if (nevery_field <= 0)
      error->all(FLERR,"Illegal fix bfield/grid command");
  }

  int ncols = 0;

  if (strcmp(arg[2],"NULL") == 0) axstr = NULL;
  else {
    int n = strlen(arg[2]) + 1;
    axstr = new char[n];
    strcpy(axstr,arg[2]);
    ncols++;
  }
  if (strcmp(arg[3],"NULL") == 0) aystr = NULL;
  else {
    int n = strlen(arg[3]) + 1;
    aystr = new char[n];
    strcpy(aystr,arg[3]);
    ncols++;
  }
  if (strcmp(arg[4],"NULL") == 0) azstr = NULL;
  else {
    int n = strlen(arg[4]) + 1;
    azstr = new char[n];
    strcpy(azstr,arg[4]);
    ncols++;
  }

  // fix settings

  per_grid_flag = 1;
  size_per_grid_cols = ncols;
  per_grid_freq = 1;
  per_grid_field = 1;

  field_active[0] = field_active[1] = field_active[2] = 0;
  if (axstr) field_active[0] = 1;
  if (aystr) field_active[1] = 1;
  if (azstr) field_active[2] = 1;

  // per-grid memory initialization

  maxgrid = 0;
  array_grid = NULL;
  axvar = ayvar = azvar = -1;  // legacy path default
  sx.kind = sy.kind = sz.kind = SRC_NONE;

  last_compute_timestep = -1;

}
/* ---------------------------------------------------------------------- */

FixBfieldGrid::~FixBfieldGrid()
{
  delete [] axstr;
  delete [] aystr;
  delete [] azstr;
  delete [] sx.cid;
  delete [] sy.cid;
  delete [] sz.cid;

  memory->destroy(array_grid);
}

/* ---------------------------------------------------------------------- */

int FixBfieldGrid::setmask()
{
  int mask = 0;
  return mask;
}
/* ---------------------------------------------------------------------- */

void FixBfieldGrid::init()
{
  // check if all variables exist and are grid-style vars

  if (axstr) parse_src_token(axstr, sx, "Bx");
  if (aystr) parse_src_token(aystr, sy, "By");
  if (azstr) parse_src_token(azstr, sz, "Bz");

  // resolve legacy variables
  if (sx.kind == SRC_VAR) {
    axvar = input->variable->find(axstr);
    if (axvar < 0 || !input->variable->grid_style(axvar))
      error->all(FLERR,"Bx arg for fix bfield/grid must be grid-style var or c_ID[idx]");
    sx.varid = axvar;
  }
  if (sy.kind == SRC_VAR) {
    ayvar = input->variable->find(aystr);
    if (ayvar < 0 || !input->variable->grid_style(ayvar))
      error->all(FLERR,"By arg for fix bfield/grid must be grid-style var or c_ID[idx]");
    sy.varid = ayvar;
  }
  if (sz.kind == SRC_VAR) {
    azvar = input->variable->find(azstr);
    if (azvar < 0 || !input->variable->grid_style(azvar))
      error->all(FLERR,"Bz arg for fix bfield/grid must be grid-style var or c_ID[idx]");
    sz.varid = azvar;
  }

  // bind compute sources
  if (sx.kind == SRC_COMP) bind_compute(sx, "Bx");
  if (sy.kind == SRC_COMP) bind_compute(sy, "By");
  if (sz.kind == SRC_COMP) bind_compute(sz, "Bz");

  // set initial grid values to zero in case dump is performed at step 0

  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid,maxgrid,size_per_grid_cols,"array_grid");
  }

  bigint nbytes = (bigint) grid->nlocal * size_per_grid_cols;
  if (nbytes) memset(&array_grid[0][0],0,nbytes*sizeof(double));
}

/* ---------------------------------------------------------------------- */

void FixBfieldGrid::compute_field()
{
  if (!grid->nlocal) return;

  // Optional cadence for expensive sources, similar to fix ablate nevery.
  if (nevery_field > 1 && last_compute_timestep >= 0 &&
      (update->ntimestep % nevery_field)) return;

  // Guard against duplicate invocation within same timestep.
  if (last_compute_timestep == update->ntimestep) return;
  last_compute_timestep = update->ntimestep;

  // reallocate array_grid if necessary
  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid,maxgrid,size_per_grid_cols,"array_grid");
  }

  int stride = size_per_grid_cols;
  int icol = 0;

  struct RouteEnt { const GridSrc *S; int outcol; };
  RouteEnt ents[3];
  int nent = 0;

  auto push_ent = [&](const GridSrc &S) {
    ents[nent].S = &S;
    ents[nent].outcol = icol;
    nent++;
    icol++;
  };

  if (field_active[0]) push_ent(sx);
  if (field_active[1]) push_ent(sy);
  if (field_active[2]) push_ent(sz);

  // Fast path: all active components from same compute -> single compute call,
  // single cell loop writing all requested columns.
  bool all_comp = (nent > 0);
  int icompute = -1;
  for (int i = 0; i < nent; i++) {
    if (ents[i].S->kind != SRC_COMP) { all_comp = false; break; }
    if (i == 0) icompute = ents[i].S->icompute;
    else if (ents[i].S->icompute != icompute) { all_comp = false; break; }
  }

  if (all_comp) {
    Compute *c = modify->compute[icompute];
    if (!(c->invoked_flag & INVOKED_PER_GRID)) {
      c->compute_per_grid();
      c->invoked_flag |= INVOKED_PER_GRID;
    }

    const int ng = grid->nlocal;
    if (c->size_per_grid_cols == 0) {
      if (c->vector_grid == NULL)
        error->all(FLERR,"fix bfield/grid: compute has no per-grid vector");
      for (int i = 0; i < nent; i++)
        if (ents[i].S->col != 1)
          error->all(FLERR,"fix bfield/grid: column for vector source must be 1");

      for (int icell = 0; icell < ng; ++icell) {
        const double v = c->vector_grid[icell];
        for (int i = 0; i < nent; i++)
          array_grid[icell][ents[i].outcol] = v;
      }
    } else {
      if (c->array_grid == NULL)
        error->all(FLERR,"fix bfield/grid: compute has no per-grid array");
      int srccol[3];
      for (int i = 0; i < nent; i++) {
        if (ents[i].S->col < 1 || ents[i].S->col > c->size_per_grid_cols)
          error->all(FLERR,"fix bfield/grid: column out of range");
        srccol[i] = ents[i].S->col - 1;
      }

      for (int icell = 0; icell < ng; ++icell)
        for (int i = 0; i < nent; i++)
          array_grid[icell][ents[i].outcol] = c->array_grid[icell][srccol[i]];
    }
    return;
  }

  // Generic path.
  icol = 0;
  auto route = [&](const GridSrc &S){
    if (S.kind == SRC_VAR) {
      input->variable->compute_grid(S.varid,&array_grid[0][icol],stride,0);
    } else if (S.kind == SRC_COMP) {
      fill_from_compute(S, icol);
    }
    icol++;
  };
  if (field_active[0]) route(sx);
  if (field_active[1]) route(sy);
  if (field_active[2]) route(sz);
 }


 /* ---------------------------------------------------------------------- 
 Parse one token: either grid-var name, or "c_ID[idx]"
 ------------------------------------------------------------------------- */
void FixBfieldGrid::parse_src_token(const char *tok, GridSrc &dst, const char *label)
{
  if (!tok) { dst.kind = SRC_NONE; return; }
  if (strncmp(tok,"c_",2)==0) {
    dst.kind = SRC_COMP;
    const char *name = tok + 2;
    const char *lb   = strchr(name,'[');
    if (!lb || tok[strlen(tok)-1] != ']') {
      char msg[160];
      snprintf(msg,sizeof(msg),"fix bfield/grid: bad %s token (use c_ID[idx])",label);
      error->all(FLERR,msg);
    }
    int idlen = lb - name;
    dst.cid = new char[idlen+1];
    strncpy(dst.cid,name,idlen);
    dst.cid[idlen] = '\0';
    dst.col = atoi(lb+1);  // 1-based
    if (dst.col <= 0) {
      char msg[160];
      snprintf(msg,sizeof(msg),"fix bfield/grid: %s column must be >=1",label);
      error->all(FLERR,msg);
    }
  } else {
    dst.kind = SRC_VAR;
  }
}

/* ---------------------------------------------------------------------- 
  Bind compute source: find compute index, check validity
  ------------------------------------------------------------------------- */

  void FixBfieldGrid::bind_compute(GridSrc &S, const char *label)
{
  S.icompute = modify->find_compute(S.cid);
  if (S.icompute < 0) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix bfield/grid: compute ID for %s not found",label);
    error->all(FLERR,msg);
  }
  Compute *c = modify->compute[S.icompute];
  if (c->per_grid_flag == 0) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix bfield/grid: compute for %s is not per-grid",label);
    error->all(FLERR,msg);
  }
  if (c->size_per_grid_cols == 0) {
    if (S.col != 1) {
      char msg[160];
      snprintf(msg,sizeof(msg),"fix bfield/grid: column for %s must be 1 for vector source",label);
      error->all(FLERR,msg);
    }
  } else if (S.col < 1 || S.col > c->size_per_grid_cols) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix bfield/grid: column for %s out of range",label);
    error->all(FLERR,msg);
  }
} // <-- this brace was missing


void FixBfieldGrid::fill_from_compute(const GridSrc &S, int icol)
{
  Compute *c = modify->compute[S.icompute];

  if (!(c->invoked_flag & INVOKED_PER_GRID)) {
    c->compute_per_grid();
    c->invoked_flag |= INVOKED_PER_GRID;
  }

  const int ng = grid->nlocal;
  if (c->size_per_grid_cols == 0) {
    // Single per-grid value is exposed as vector_grid (column index must be 1).
    if (S.col != 1 || c->vector_grid == NULL)
      error->all(FLERR,"fix bfield/grid: compute has no per-grid vector");
    for (int icell = 0; icell < ng; ++icell)
      array_grid[icell][icol] = c->vector_grid[icell];
  } else {
    if (S.col < 1 || S.col > c->size_per_grid_cols || c->array_grid == NULL)
      error->all(FLERR,"fix bfield/grid: compute has no per-grid array");
    const int src = S.col - 1;
    for (int icell = 0; icell < ng; ++icell)
      array_grid[icell][icol] = c->array_grid[icell][src];
  }
}
