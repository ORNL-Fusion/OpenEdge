/* ----------------------------------------------------------------------
   OpenEdge: sheath electric field per grid cell from plasma + wall geometry
------------------------------------------------------------------------- */

#include "compute_sheath_fields_grid.h"

#include <cmath>
#include <cstring>

#include "compute_plasma_fields.h"
#include "compute_sheath_geometry_grid.h"
#include "sheath_models.h"

#include "update.h"
#include "grid.h"
#include "domain.h"
#include "modify.h"
#include "input.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

namespace {
constexpr double BIGD = 1.0e20;
}

ComputeSheathFieldsGrid::ComputeSheathFieldsGrid(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal compute sheath/fields/grid command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute sheath/fields/grid grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  iplasma = modify->find_compute(arg[3]);
  if (iplasma < 0) error->all(FLERR,"Compute sheath/fields/grid plasma compute ID does not exist");
  igeom = modify->find_compute(arg[4]);
  if (igeom < 0) error->all(FLERR,"Compute sheath/fields/grid geometry compute ID does not exist");

  dmax = 0.02;
  mach_min = 0.8;
  mach_max = 1.2;
  gamma_see = 0.0;
  cur_A_m2 = 0.0;
  mD_amu = 2.01410177811;
  emax_vpm = 5.0e5;
  borodkina_pot_mult = 2.5;
  sheath_model = MODEL_BORODKINA;

  int iarg = 5;
  while (iarg < narg && strcmp(arg[iarg],"values") != 0) {
    if (iarg + 1 >= narg) error->all(FLERR,"Illegal compute sheath/fields/grid command");
    if (strcmp(arg[iarg],"dmax") == 0) dmax = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"mach_min") == 0) mach_min = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"mach_max") == 0) mach_max = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"gamma_see") == 0) gamma_see = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"cur_A_m2") == 0) cur_A_m2 = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"mD_amu") == 0) mD_amu = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"emax_vpm") == 0) emax_vpm = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"pot_mult") == 0) borodkina_pot_mult = input->numeric(FLERR,arg[iarg+1]);
    else if (strcmp(arg[iarg],"model") == 0) {
      if (strcmp(arg[iarg+1],"eirene") == 0) sheath_model = MODEL_EIRENE;
      else if (strcmp(arg[iarg+1],"borodkina") == 0) sheath_model = MODEL_BORODKINA;
      else error->all(FLERR,"compute sheath/fields/grid model must be eirene or borodkina");
    }
    else error->all(FLERR,"Illegal compute sheath/fields/grid command");
    iarg += 2;
  }
  if (iarg >= narg || strcmp(arg[iarg],"values") != 0)
    error->all(FLERR,"compute sheath/fields/grid requires values keyword");
  ++iarg;
  if (iarg >= narg) error->all(FLERR,"compute sheath/fields/grid missing values list");

  nvalue = narg - iarg;
  value = new int[nvalue];
  int iv = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"ex") == 0) value[iv] = EX;
    else if (strcmp(arg[iarg],"ey") == 0) value[iv] = EY;
    else if (strcmp(arg[iarg],"ez") == 0) value[iv] = EZ;
    else if (strcmp(arg[iarg],"esheath") == 0) value[iv] = ESHEATH;
    else if (strcmp(arg[iarg],"mach_par") == 0) value[iv] = MACH_PAR;
    else if (strcmp(arg[iarg],"mach_n") == 0) value[iv] = MACH_N;
    else if (strcmp(arg[iarg],"alpha") == 0) value[iv] = ALPHA;
    else if (strcmp(arg[iarg],"active") == 0) value[iv] = ACTIVE;
    else if (strcmp(arg[iarg],"bdotn") == 0) value[iv] = BDOTN;
    else if (strcmp(arg[iarg],"dist") == 0) value[iv] = DIST;
    else if (strcmp(arg[iarg],"surfid") == 0) value[iv] = SURFID;
    else error->all(FLERR,"Illegal compute sheath/fields/grid values entry");
    ++iv;
    ++iarg;
  }

  per_grid_flag = 1;
  size_per_grid_cols = (nvalue == 1) ? 0 : nvalue;
  nglocal = 0;
  vector_grid = nullptr;
  array_grid = nullptr;
}

ComputeSheathFieldsGrid::~ComputeSheathFieldsGrid()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
}

void ComputeSheathFieldsGrid::init()
{
  Compute *cp = modify->compute[iplasma];
  Compute *cg = modify->compute[igeom];
  if (!cp->per_grid_flag || !cg->per_grid_flag)
    error->all(FLERR,"compute sheath/fields/grid dependencies must be per-grid computes");
  reallocate();
}

void ComputeSheathFieldsGrid::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;
  Compute *cp_base = modify->compute[iplasma];
  Compute *cg_base = modify->compute[igeom];
  if (cp_base->invoked_per_grid != update->ntimestep) cp_base->compute_per_grid();
  if (cg_base->invoked_per_grid != update->ntimestep) cg_base->compute_per_grid();

  auto *cp = dynamic_cast<ComputePlasmaFields *>(cp_base);
  if (!cp) error->all(FLERR,"compute sheath/fields/grid requires plasma/fields compute as first dependency");

  const double **garr = nullptr;
  if (cg_base->size_per_grid_cols <= 0)
    error->all(FLERR,"compute sheath/fields/grid requires geometry compute with array output");
  garr = const_cast<const double **>(cg_base->array_grid);
  // expected order from compute_sheath_geometry_grid command: dist surfid nx ny nz
  const int IDIST = 0, ISID = 1, INX = 2, INY = 3, INZ = 4;
  if (cg_base->size_per_grid_cols < 5)
    error->all(FLERR,"compute sheath/fields/grid expects geometry array with at least 5 columns: dist surfid nx ny nz");

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  const int nspec = cp->nion_species;
  const bool has_multi = (nspec > 0 &&
                          static_cast<int>(cp->ion_dens_grid.size()) == grid->nlocal * nspec &&
                          static_cast<int>(cp->ion_parr_flow_grid.size()) == grid->nlocal * nspec &&
                          static_cast<int>(cp->ion_charge_state_z.size()) == nspec);

  std::vector<double> dens_i;
  std::vector<double> upar_i;
  std::vector<int> z_i;

  for (int icell = 0; icell < nglocal; ++icell) {
    double ex = 0.0, ey = 0.0, ez = 0.0;
    double esh = 0.0, mpar = 0.0, mn = 0.0, alpha = 90.0, active = 0.0, bdotn = 0.0;
    double dist = BIGD, sid = -1.0;

    if ((cinfo[icell].mask & groupbit) && cells[icell].nsplit >= 1) {
      dist = garr[icell][IDIST];
      sid  = garr[icell][ISID];
      double nx = garr[icell][INX];
      double ny = garr[icell][INY];
      double nz = garr[icell][INZ];
      double nvec[3] = {nx, ny, nz};
      double bvec[3] = {0.0, 0.0, 0.0};
      const double br = cp->mag_arr[icell].br;
      const double bt = cp->mag_arr[icell].bt;
      const double bz = cp->mag_arr[icell].bz;
      if (domain->dimension == 2) {
        // 2D OpenEdge WEST setup uses in-plane (R,Z) => (x,y) here.
        // Toroidal Bt is out-of-plane and should not project onto wall normal.
        bvec[0] = br;
        bvec[1] = bz;
        bvec[2] = bt;
      } else {
        // 3D Cartesian: convert cylindrical (Br,Bt,Bz) to (Bx,By,Bz)
        const double x = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
        const double y = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
        const double r = std::sqrt(x*x + y*y);
        const double cphi = (r > 1.0e-20) ? x / r : 1.0;
        const double sphi = (r > 1.0e-20) ? y / r : 0.0;
        bvec[0] = br * cphi - bt * sphi;
        bvec[1] = br * sphi + bt * cphi;
        bvec[2] = bz;
      }

      const double te = cp->plasma_arr[icell].temp_e;
      const double ti = cp->plasma_arr[icell].temp_i;
      const double ne = cp->plasma_arr[icell].dens_e;
      const double upar = cp->plasma_arr[icell].parr_flow;
      const double cs = SheathModels::sound_speed_d(te, ti, mD_amu);
      const SheathModels::ChoduraMetrics cm = SheathModels::chodura_metrics(upar, cs, bvec, nvec);
      mpar = cm.mach_par;
      mn = cm.mach_n;
      alpha = cm.alpha_deg;
      bdotn = cm.bdotn;

      const bool near = (dist <= dmax);
      const bool inband = (mpar >= mach_min && mpar <= mach_max);
      // Do not gate by incoming u_par*(b.n); allow tangential/near-tangential
      // wall regions (e.g., limiter) to activate based on near-wall + Mach band.
      const bool active_sheath = near && inband && te > 0.0 && ne > 0.0;

      if (active_sheath) {
        if (has_multi) {
          dens_i.assign(nspec, 0.0);
          upar_i.assign(nspec, 0.0);
          z_i = cp->ion_charge_state_z;
          const int base = icell * nspec;
          for (int is = 0; is < nspec; ++is) {
            dens_i[is] = cp->ion_dens_grid[base + is];
            upar_i[is] = cp->ion_parr_flow_grid[base + is];
          }
        } else {
          dens_i.assign(1, cp->plasma_arr[icell].dens_i);
          upar_i.assign(1, cp->plasma_arr[icell].parr_flow);
          z_i.assign(1, 1);
        }

        double emag = 0.0;
        if (sheath_model == MODEL_EIRENE) {
          const SheathModels::EireneSheathResult sr =
            SheathModels::eirene_sheath_ev(te, dens_i, upar_i, z_i, gamma_see, cur_A_m2);
          esh = sr.esheath_eV;
          const double scale = std::max(dist, 1.0e-5);
          emag = esh / scale;
        } else {
          const double bmag = std::sqrt(bvec[0]*bvec[0] + bvec[1]*bvec[1] + bvec[2]*bvec[2]);
          const SheathModels::BorodkinaSheathResult br =
            SheathModels::borodkina_sheath_at_distance(dist, te, ti, ne, bmag, alpha, mD_amu, borodkina_pot_mult);
          esh = br.esheath_eV;
          emag = br.emag_vpm;
        }
        emag = std::min(emag, emax_vpm);
        // E toward wall (opposite to n)
        ex = -emag * nx;
        ey = -emag * ny;
        ez = -emag * nz;
        active = 1.0;
      }
    }

    if (nvalue == 1) {
      if (value[0] == EX) vector_grid[icell] = ex;
      else if (value[0] == EY) vector_grid[icell] = ey;
      else if (value[0] == EZ) vector_grid[icell] = ez;
      else if (value[0] == ESHEATH) vector_grid[icell] = esh;
      else if (value[0] == MACH_PAR) vector_grid[icell] = mpar;
      else if (value[0] == MACH_N) vector_grid[icell] = mn;
      else if (value[0] == ALPHA) vector_grid[icell] = alpha;
      else if (value[0] == ACTIVE) vector_grid[icell] = active;
      else if (value[0] == BDOTN) vector_grid[icell] = bdotn;
      else if (value[0] == DIST) vector_grid[icell] = dist;
      else if (value[0] == SURFID) vector_grid[icell] = sid;
    } else {
      for (int j = 0; j < nvalue; ++j) {
        if (value[j] == EX) array_grid[icell][j] = ex;
        else if (value[j] == EY) array_grid[icell][j] = ey;
        else if (value[j] == EZ) array_grid[icell][j] = ez;
        else if (value[j] == ESHEATH) array_grid[icell][j] = esh;
        else if (value[j] == MACH_PAR) array_grid[icell][j] = mpar;
        else if (value[j] == MACH_N) array_grid[icell][j] = mn;
        else if (value[j] == ALPHA) array_grid[icell][j] = alpha;
        else if (value[j] == ACTIVE) array_grid[icell][j] = active;
        else if (value[j] == BDOTN) array_grid[icell][j] = bdotn;
        else if (value[j] == DIST) array_grid[icell][j] = dist;
        else if (value[j] == SURFID) array_grid[icell][j] = sid;
      }
    }
  }
}

void ComputeSheathFieldsGrid::reallocate()
{
  if (grid->nlocal == nglocal) return;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  nglocal = grid->nlocal;
  if (nvalue == 1) memory->create(vector_grid,nglocal,"sheath/fields/grid:vector");
  else memory->create(array_grid,nglocal,nvalue,"sheath/fields/grid:array");
}

bigint ComputeSheathFieldsGrid::memory_usage()
{
  bigint bytes = 0;
  if (nvalue == 1) bytes += nglocal * sizeof(double);
  else bytes += static_cast<bigint>(nglocal) * nvalue * sizeof(double);
  return bytes;
}
