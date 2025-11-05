/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_efield_grid.h"
#include "grid.h"
#include "input.h"
#include "variable.h"
#include "memory.h"
#include "error.h"
#include "modify.h"
#include "compute.h"

#define INVOKED_PER_GRID 16

using namespace SPARTA_NS;


/* ---------------------------------------------------------------------- */

FixEfieldGrid::FixEfieldGrid(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  if (narg != 5) error->all(FLERR,"Illegal fix efield/grid command");

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
  axvar = ayvar = azvar = -1;  
  sx.kind = sy.kind = sz.kind = SRC_NONE;

}

/* ---------------------------------------------------------------------- */

FixEfieldGrid::~FixEfieldGrid()
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

int FixEfieldGrid::setmask()
{
  int mask = 0;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixEfieldGrid::init()
{
  // check if all variables exist and are grid-style vars

  // parse each token: either grid variable name or c_ID[idx]
  if (axstr) parse_src_token(axstr, sx, "Ex");
  if (aystr) parse_src_token(aystr, sy, "Ey");
  if (azstr) parse_src_token(azstr, sz, "Ez");

  // resolve legacy variables
  if (sx.kind == SRC_VAR) {
    axvar = input->variable->find(axstr);
    if (axvar < 0 || !input->variable->grid_style(axvar))
      error->all(FLERR,"Ex arg for fix efield/grid must be grid-style var or c_ID[idx]");
    sx.varid = axvar;
  }
  if (sy.kind == SRC_VAR) {
    ayvar = input->variable->find(aystr);
    if (ayvar < 0 || !input->variable->grid_style(ayvar))
      error->all(FLERR,"Ey arg for fix efield/grid must be grid-style var or c_ID[idx]");
    sy.varid = ayvar;
  }
  if (sz.kind == SRC_VAR) {
    azvar = input->variable->find(azstr);
    if (azvar < 0 || !input->variable->grid_style(azvar))
      error->all(FLERR,"Ez arg for fix efield/grid must be grid-style var or c_ID[idx]");
    sz.varid = azvar;
  }

  // bind compute sources
  if (sx.kind == SRC_COMP) bind_compute(sx, "Ex");
  if (sy.kind == SRC_COMP) bind_compute(sy, "Ey");
  if (sz.kind == SRC_COMP) bind_compute(sz, "Ez");

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

void FixEfieldGrid::compute_field()
{
  if (!grid->nlocal) return;

  // reallocate array_grid if necessary

  if (grid->nlocal > maxgrid) {
    maxgrid = grid->maxlocal;
    memory->destroy(array_grid);
    memory->create(array_grid,maxgrid,size_per_grid_cols,"array_grid");
  }

  // evaluate each grid-style variable
  // results are put into strided array_grid

  int stride = size_per_grid_cols;
  int icol = 0;

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
void FixEfieldGrid::parse_src_token(const char *tok, GridSrc &dst, const char *label)
{
  if (!tok) { dst.kind = SRC_NONE; return; }
  if (strncmp(tok,"c_",2)==0) {
    dst.kind = SRC_COMP;
    const char *name = tok + 2;
    const char *lb   = strchr(name,'[');
    if (!lb || tok[strlen(tok)-1] != ']') {
      char msg[160];
      snprintf(msg,sizeof(msg),"fix efield/grid: bad %s token (use c_ID[idx])",label);
      error->all(FLERR,msg);
    }
    int idlen = lb - name;
    dst.cid = new char[idlen+1];
    strncpy(dst.cid,name,idlen);
    dst.cid[idlen] = '\0';
    dst.col = atoi(lb+1);  // 1-based
    if (dst.col <= 0) {
      char msg[160];
      snprintf(msg,sizeof(msg),"fix efield/grid: %s column must be >=1",label);
      error->all(FLERR,msg);
    }
  } else {
    dst.kind = SRC_VAR;
  }
}

/* ---------------------------------------------------------------------- 
  Bind compute source: find compute index, check validity
  ------------------------------------------------------------------------- */

  void FixEfieldGrid::bind_compute(GridSrc &S, const char *label)
{
  S.icompute = modify->find_compute(S.cid);
  if (S.icompute < 0) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix efield/grid: compute ID for %s not found",label);
    error->all(FLERR,msg);
  }
  Compute *c = modify->compute[S.icompute];
  if (c->per_grid_flag == 0) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix efield/grid: compute for %s is not per-grid",label);
    error->all(FLERR,msg);
  }
  if (c->size_per_grid_cols == 0) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix efield/grid: compute for %s has no per-grid array",label);
    error->all(FLERR,msg);
  }
  if (S.col < 1 || S.col > c->size_per_grid_cols) {
    char msg[160];
    snprintf(msg,sizeof(msg),"fix efield/grid: column for %s out of range",label);
    error->all(FLERR,msg);
  }
} 


void FixEfieldGrid::fill_from_compute(const GridSrc &S, int icol)
{
  Compute *c = modify->compute[S.icompute];

  if (!(c->invoked_flag & INVOKED_PER_GRID)) {
    c->compute_per_grid();
    c->invoked_flag |= INVOKED_PER_GRID;
  }

  double **arr = nullptr; 
  int *cols = nullptr;

  // If your Compute class exposes a query method:
  int ok = c->query_tally_grid(S.col, arr, cols);
  if (ok <= 0 || !arr) error->all(FLERR,"fix efield/grid: compute has no per-grid data");

  const int src = cols ? cols[0] : (S.col - 1);
  const int ng = grid->nlocal;

  for (int icell = 0; icell < ng; ++icell)
    array_grid[icell][icol] = arr[icell][src];
}
