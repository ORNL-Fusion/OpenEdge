/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include <cmath>
#include <algorithm>
#include "string.h"
#include "compute_plasma_fields.h"
#include "fix_background.h"
#include "update.h"
#include "grid.h"
#include "domain.h"
#include "input.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "openedge_geom.h"

#include <tuple>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <H5Cpp.h>
#include <hdf5.h>
#include <stdexcept>



using namespace SPARTA_NS;

// user keywords
enum {
  BR, BT, BZ, BX, BY,
  ER, ET, EZ, EX, EY,
  VR, VT, VZ, VX, VY,
  TI, TE, NI, NE, PARRFLOW, EPAR,
  GRAD_TE_R, GRAD_TE_T, GRAD_TE_Z, GRAD_TI_R, GRAD_TI_T, GRAD_TI_Z,
  GRAD_NE_R, GRAD_NE_Z,
  GRAD_TE_MAG, SEPDIST
};

/* ---------------------------------------------------------------------- */

ComputePlasmaFields::
ComputePlasmaFields(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 5)
    error->all(FLERR,"Illegal compute plasma/fields command");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute plasma/fields grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  // defaults
  bconst[0] = bconst[1] = bconst[2] = 0.0;
  econst[0] = econst[1] = econst[2] = 0.0;
  teconst = ticonst = niconst = neconst = parrflowconst = 0.0;
  analytic_r0 = 0.02;
  analytic_ne0 = 1.0e19;
  analytic_te0 = 1.0;
  analytic_te1 = 4.0;
  analytic_ti0 = 1.0;
  analytic_eps = 1.0e-10;
  analytic_x0 = 0.0;
  analytic_y0 = 0.0;
  analytic_use_x0 = 0;
  analytic_use_y0 = 0;

  // parse:
  // compute ... plasma/fields ggroup background <fix_id> ...
  // compute ... plasma/fields ggroup constant [const args] ...
  // compute ... plasma/fields ggroup analytic [analytic args] ...
  //
  // File-based plasma input is now routed through fix background
  // (which owns the single-source-of-truth plasma.h5 + mesh/equilibrium
  // data). `compute plasma/fields ... file plasma.h5 ...` is DEPRECATED
  // and rejected below with a pointer to the migration. Declare
  //   fix pd background file plasma.h5
  //   compute cplasma plasma/fields all background pd ...
  // instead.
  int iarg = 3;
  if (iarg >= narg)
    error->all(FLERR,
      "compute plasma/fields requires mode: background, constant, or analytic");
  if (strcmp(arg[iarg],"file") == 0) {
    error->all(FLERR,
      "compute plasma/fields: the 'file' mode has been removed. "
      "Declare a `fix background file plasma.h5` instance and use "
      "`compute plasma/fields ... background <fix_id> ...` instead. "
      "This routes per-cell / per-particle plasma lookups through the "
      "EIRENE triangulation owned by fix background (single source of "
      "truth, no duplicate file reads, no regular-grid interpolation).");
    // keep compiler happy — MODE_FILE code paths are unreachable now
    input_mode = MODE_BACKGROUND;
  } else if (strcmp(arg[iarg],"background") == 0) {
    input_mode = MODE_BACKGROUND;
    iarg++;
    if (iarg >= narg)
      error->all(FLERR,"compute plasma/fields background mode needs fix ID");
    background_fix_id = std::string(arg[iarg++]);
  } else if (strcmp(arg[iarg],"constant") == 0) {
    input_mode = MODE_CONSTANT;
    iarg++;
  } else if (strcmp(arg[iarg],"analytic") == 0) {
    input_mode = MODE_ANALYTIC;
    iarg++;
  } else {
    error->all(FLERR,
      "compute plasma/fields mode must be 'background', 'constant', or 'analytic'");
  }

  // constant/analytic mode options
  if (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC) {
    while (iarg < narg) {
      if (strcmp(arg[iarg],"values")==0) { iarg++; break; }
      if (iarg + 3 < narg && strcmp(arg[iarg],"magnetic_field")==0) {
        bconst[0] = input->numeric(FLERR,arg[iarg+1]);
        bconst[1] = input->numeric(FLERR,arg[iarg+2]);
        bconst[2] = input->numeric(FLERR,arg[iarg+3]);
        iarg += 4;
      } else if (iarg + 3 < narg && strcmp(arg[iarg],"electric_field")==0) {
        econst[0] = input->numeric(FLERR,arg[iarg+1]);
        econst[1] = input->numeric(FLERR,arg[iarg+2]);
        econst[2] = input->numeric(FLERR,arg[iarg+3]);
        iarg += 4;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"temp_e")==0) {
        teconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"temp_i")==0) {
        ticonst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"dens_e")==0) {
        neconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"dens_i")==0) {
        niconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"parrflow")==0) {
        parrflowconst = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"r0")==0) {
        analytic_r0 = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"ne0")==0) {
        analytic_ne0 = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"te0")==0) {
        analytic_te0 = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"te1")==0) {
        analytic_te1 = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"ti0")==0) {
        analytic_ti0 = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"eps")==0) {
        analytic_eps = input->numeric(FLERR,arg[iarg+1]);
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"x0")==0) {
        analytic_x0 = input->numeric(FLERR,arg[iarg+1]);
        analytic_use_x0 = 1;
        iarg += 2;
      } else if (iarg + 1 < narg && strcmp(arg[iarg],"y0")==0) {
        analytic_y0 = input->numeric(FLERR,arg[iarg+1]);
        analytic_use_y0 = 1;
        iarg += 2;
      } else break;
    }
  }

  // Optional equilibrium keyword (before value keywords)
  has_equilibrium = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"equilibrium") == 0) {
      // external-equilibrium files went away with the removed file mode:
      // the equilibrium now arrives through fix background (either an
      // /equilibrium group in plasma.h5 or native B maps)
      error->all(FLERR,"compute plasma/fields: the 'equilibrium <file>' "
                 "keyword has been removed - embed the equilibrium in the "
                 "plasma file loaded by fix background instead");
    } else break;
  }

  // Note: when no bfield.h5 and no equilibrium file are passed, we will
  // try to read B from the plasma.h5 (br/bt/bz datasets) at init time.
  // The error here is deferred until after plasma.h5 is read.

  if (iarg >= narg)
    error->all(FLERR,"plasma/fields needs values (br/bt/bz, er/et/ez, vr/vt/vz, ...)");

  // collect value keywords
  nvalue = narg - iarg;
  value = new int[nvalue];
  for (int iv = 0; iv < nvalue; ++iv, ++iarg) {
    if      (strcmp(arg[iarg],"br")==0)      value[iv] = BR;
    else if (strcmp(arg[iarg],"bt")==0)      value[iv] = BT;
    else if (strcmp(arg[iarg],"bz")==0)      value[iv] = BZ;
    else if (strcmp(arg[iarg],"bx")==0)      value[iv] = BX; // cartesian alias
    else if (strcmp(arg[iarg],"by")==0)      value[iv] = BY; // cartesian alias
    else if (strcmp(arg[iarg],"er")==0)      value[iv] = ER;
    else if (strcmp(arg[iarg],"et")==0)      value[iv] = ET;
    else if (strcmp(arg[iarg],"ez")==0)      value[iv] = EZ;
    else if (strcmp(arg[iarg],"ex")==0)      value[iv] = EX; // cartesian alias
    else if (strcmp(arg[iarg],"ey")==0)      value[iv] = EY; // cartesian alias
    else if (strcmp(arg[iarg],"vr")==0)      value[iv] = VR;
    else if (strcmp(arg[iarg],"vt")==0)      value[iv] = VT;
    else if (strcmp(arg[iarg],"vz")==0)      value[iv] = VZ;
    else if (strcmp(arg[iarg],"vx")==0)      value[iv] = VX; // cartesian alias
    else if (strcmp(arg[iarg],"vy")==0)      value[iv] = VY; // cartesian alias
    else if (strcmp(arg[iarg],"temp_i")==0)  value[iv] = TI;
    else if (strcmp(arg[iarg],"temp_e")==0)  value[iv] = TE;
    else if (strcmp(arg[iarg],"dens_i")==0)  value[iv] = NI;
    else if (strcmp(arg[iarg],"dens_e")==0)  value[iv] = NE;
    else if (strcmp(arg[iarg],"parrflow")==0) value[iv] = PARRFLOW;
    else if (strcmp(arg[iarg],"epar")==0)    value[iv] = EPAR;
    else if (strcmp(arg[iarg],"grad_te_r")==0) value[iv] = GRAD_TE_R;
    else if (strcmp(arg[iarg],"grad_te_t")==0) value[iv] = GRAD_TE_T;
    else if (strcmp(arg[iarg],"grad_te_z")==0) value[iv] = GRAD_TE_Z;
    else if (strcmp(arg[iarg],"grad_ti_r")==0) value[iv] = GRAD_TI_R;
    else if (strcmp(arg[iarg],"grad_ti_t")==0) value[iv] = GRAD_TI_T;
    else if (strcmp(arg[iarg],"grad_ti_z")==0) value[iv] = GRAD_TI_Z;
    else if (strcmp(arg[iarg],"grad_ne_r")==0) value[iv] = GRAD_NE_R;
    else if (strcmp(arg[iarg],"grad_ne_z")==0) value[iv] = GRAD_NE_Z;
    else if (strcmp(arg[iarg],"grad_te_mag")==0) value[iv] = GRAD_TE_MAG;
    else if (strcmp(arg[iarg],"sepdist")==0)   value[iv] = SEPDIST;
    else error->all(FLERR,"Illegal plasma/fields value");
  }

  per_grid_flag = 1;
  if (nvalue == 1) size_per_grid_cols = 0;
  else size_per_grid_cols = nvalue;
  post_process_grid_flag = 0;

  nglocal = 0;
  vector_grid = NULL;
  array_grid = NULL;

  plasma_arr = NULL;
  mag_arr    = NULL;
  sample_stale = 0;


}

/* ---------------------------------------------------------------------- */

ComputePlasmaFields::~ComputePlasmaFields()
{
  if (copymode) return;
  delete [] value;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);

  if (plasma_arr) {
    memory->destroy(plasma_arr);
    plasma_arr = NULL;
  }
  if (mag_arr) {
    memory->destroy(mag_arr);
    mag_arr = NULL;
  }
}

/* ---------------------------------------------------------------------- */

// psi at the magnetic axis = grid extremum farthest from the boundary value.
// File-mode readers only store the psi grid (EFIT's simag is not carried),
// so derive it; background mode copies the fix's psi_axis directly.
static void derive_psi_axis(EquilibriumData &e)
{
  if (e.jm < 2 || e.km < 2 || e.psi.empty()) return;
  double best = e.psib, bestd = 0.0;
  for (int k = 0; k < e.km; k++)
    for (int j = 0; j < e.jm; j++) {
      const double d = std::fabs(e.psi[k][j] - e.psib);
      if (d > bestd) { bestd = d; best = e.psi[k][j]; }
    }
  e.psi_axis = best;
}

struct EquilibriumMapStencil {
  int j = 0, k = 0;
  double fr = 0.0, fz = 0.0;
  double dr = 0.0, dz = 0.0;
};

static bool equilibrium_map_shape_ok(
    const std::vector<std::vector<double>> &field, int km, int jm)
{
  if (static_cast<int>(field.size()) != km) return false;
  for (int k = 0; k < km; ++k)
    if (static_cast<int>(field[k].size()) != jm) return false;
  return true;
}

static bool equilibrium_has_native_b(const EquilibriumData &equ)
{
  return equ.jm >= 2 && equ.km >= 2 &&
         equilibrium_map_shape_ok(equ.br, equ.km, equ.jm) &&
         equilibrium_map_shape_ok(equ.bt, equ.km, equ.jm) &&
         equilibrium_map_shape_ok(equ.bz, equ.km, equ.jm);
}

static bool make_equilibrium_map_stencil(const EquilibriumData &equ,
                                         double R, double Z,
                                         EquilibriumMapStencil &s)
{
  if (equ.r.size() < 2 || equ.z.size() < 2 ||
      equ.r.back() <= equ.r.front() || equ.z.back() <= equ.z.front())
    return false;
  const double Rc = std::min(std::max(R, equ.r.front()), equ.r.back());
  const double Zc = std::min(std::max(Z, equ.z.front()), equ.z.back());
  auto ir = std::upper_bound(equ.r.begin(), equ.r.end(), Rc);
  auto iz = std::upper_bound(equ.z.begin(), equ.z.end(), Zc);
  s.j = std::max(0, std::min(static_cast<int>(ir - equ.r.begin()) - 1,
                             equ.jm - 2));
  s.k = std::max(0, std::min(static_cast<int>(iz - equ.z.begin()) - 1,
                             equ.km - 2));
  s.dr = equ.r[s.j + 1] - equ.r[s.j];
  s.dz = equ.z[s.k + 1] - equ.z[s.k];
  if (s.dr <= 0.0 || s.dz <= 0.0) return false;
  s.fr = (Rc - equ.r[s.j]) / s.dr;
  s.fz = (Zc - equ.z[s.k]) / s.dz;
  return true;
}

static void sample_equilibrium_map(
    const std::vector<std::vector<double>> &field,
    const EquilibriumMapStencil &s,
    double &value, double &d_dR, double &d_dZ)
{
  const double f00 = field[s.k][s.j];
  const double f10 = field[s.k][s.j + 1];
  const double f01 = field[s.k + 1][s.j];
  const double f11 = field[s.k + 1][s.j + 1];
  const double lower = (1.0 - s.fr) * f00 + s.fr * f10;
  const double upper = (1.0 - s.fr) * f01 + s.fr * f11;
  value = (1.0 - s.fz) * lower + s.fz * upper;
  d_dR = ((1.0 - s.fz) * (f10 - f00) +
          s.fz * (f11 - f01)) / s.dr;
  d_dZ = ((1.0 - s.fr) * (f01 - f00) +
          s.fr * (f11 - f10)) / s.dz;
}

static bool sample_native_equilibrium_b(const EquilibriumData &equ,
                                        double R, double Z,
                                        MagneticFieldFileDataParams &B)
{
  if (!equilibrium_has_native_b(equ)) return false;
  EquilibriumMapStencil stencil;
  if (!make_equilibrium_map_stencil(equ, R, Z, stencil)) return false;
  B.r = R;
  B.z = Z;
  sample_equilibrium_map(equ.br, stencil, B.br, B.dBr_dr, B.dBr_dz);
  sample_equilibrium_map(equ.bt, stencil, B.bt, B.dBt_dr, B.dBt_dz);
  sample_equilibrium_map(equ.bz, stencil, B.bz, B.dBz_dr, B.dBz_dz);
  B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
  if (B.Bmag > 0.0) {
    B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr +
                  B.bz*B.dBz_dr) / B.Bmag;
    B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz +
                  B.bz*B.dBz_dz) / B.Bmag;
  }
  B.derivatives_valid = true;
  B.axisymmetric_source = true;
  return true;
}

void ComputePlasmaFields::init()
{
  reallocate();

  const int me     = comm->me;
  const int ncells = grid->nlocal;

  // clean any old pre-sampled field arrays
  if (plasma_arr) {
    memory->destroy(plasma_arr);
    plasma_arr = NULL;
  }
  if (mag_arr) {
    memory->destroy(mag_arr);
    mag_arr = NULL;
  }
  nion_species = 0;
  ion_spec_index.clear();
  ion_charge_state_z.clear();
  ion_mass_amu.clear();
  ion_names.clear();


  // --- 1) load field source (file or constants) ------------------------------
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  memory->create(plasma_arr, ncells, "plasma/fields:plasma_arr");
  memory->create(mag_arr,    ncells, "plasma/fields:mag_arr");

  if (input_mode == MODE_BACKGROUND) {
    // Pull plasma data from fix background — no file reads
    int ifix = modify->find_fix(background_fix_id.c_str());
    if (ifix < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "compute plasma/fields: fix '%s' not found",
               background_fix_id.c_str());
      error->all(FLERR, msg);
    }
    auto *pd = dynamic_cast<FixBackground*>(modify->fix[ifix]);
    bg_fix_ = pd;
    if (!pd)
      error->all(FLERR,
        "compute plasma/fields: background fix must be style background");

    // Build PlasmaFileData from the fix's flat arrays
    // (convert flat vectors to vector<vector<double>> format)
    plasma_data.column_x0 = pd->column_x0;
    plasma_data.column_y0 = pd->column_y0;
    plasma_data.r.assign(pd->rvals.begin(), pd->rvals.end());
    plasma_data.z.assign(pd->zvals.begin(), pd->zvals.end());
    int pnr = pd->nr, pnz = pd->nz;
    auto flat2vv = [&](const std::vector<double> &flat) {
      std::vector<std::vector<double>> vv(pnz, std::vector<double>(pnr, 0.0));
      if (flat.empty()) return vv;
      for (int iz = 0; iz < pnz; iz++)
        for (int ir = 0; ir < pnr; ir++)
          vv[iz][ir] = flat[iz * pnr + ir];
      return vv;
    };
    plasma_data.dens_e = flat2vv(pd->dens_e);
    plasma_data.temp_e = flat2vv(pd->temp_e);
    plasma_data.dens_i = flat2vv(pd->dens_i);
    plasma_data.temp_i = flat2vv(pd->temp_i);
    plasma_data.parr_flow = flat2vv(pd->parr_flow);
    plasma_data.parr_flow_r = flat2vv(pd->parr_flow_r);
    plasma_data.parr_flow_t = flat2vv(pd->parr_flow_t);
    plasma_data.parr_flow_z = flat2vv(pd->parr_flow_z);
    plasma_data.grad_temp_e_r = flat2vv(pd->grad_te_r);
    plasma_data.grad_temp_e_t = flat2vv(pd->grad_te_t);
    plasma_data.grad_temp_e_z = flat2vv(pd->grad_te_z);
    plasma_data.grad_temp_i_r = flat2vv(pd->grad_ti_r);
    plasma_data.grad_temp_i_t = flat2vv(pd->grad_ti_t);
    plasma_data.grad_temp_i_z = flat2vv(pd->grad_ti_z);

    // B comes from the fix's mesh/equilibrium/constant sources at query
    // time — the legacy regular-grid raster copy is gone.

    // Mesh data from fix
    if (pd->has_mesh) {
      plasma_data.has_mesh = true;
      plasma_data.mesh_nvtx = pd->mesh_nvtx;
      plasma_data.mesh_ntri = pd->mesh_ntri;
      plasma_data.mesh_ncell = pd->mesh_ncell;
      plasma_data.mesh_vtx_r = pd->mesh_vtx_r;
      plasma_data.mesh_vtx_z = pd->mesh_vtx_z;
      plasma_data.mesh_tri = pd->mesh_tri;
      plasma_data.mesh_cell_idx = pd->mesh_cell_idx;
      plasma_data.mesh_ne = pd->mesh_ne;
      plasma_data.mesh_te = pd->mesh_te;
      plasma_data.mesh_ti = pd->mesh_ti;
      plasma_data.mesh_ni = pd->mesh_ni;
      plasma_data.mesh_upar = pd->mesh_upar;
      plasma_data.mesh_e_r = pd->mesh_e_r;
      plasma_data.mesh_e_z = pd->mesh_e_z;
      plasma_data.mesh_e_t = pd->mesh_e_t;
      // Per-triangle mesh B (vertex-averaged). Empty on rank 0 =>
      // mesh-B not loaded from plasma.h5; leave empty so the legacy
      // regular-grid / equilibrium branches handle the query.
      plasma_data.mesh_tri_br = pd->mesh_tri_br;
      plasma_data.mesh_tri_bz = pd->mesh_tri_bz;
      plasma_data.mesh_tri_bt = pd->mesh_tri_bt;
      // Build bounding boxes, mapped centroids, and spatial hash
      plasma_data.mesh_tri_rmin.resize(pd->mesh_ntri);
      plasma_data.mesh_tri_rmax.resize(pd->mesh_ntri);
      plasma_data.mesh_tri_zmin.resize(pd->mesh_ntri);
      plasma_data.mesh_tri_zmax.resize(pd->mesh_ntri);
      for (int t = 0; t < pd->mesh_ntri; t++) {
        const int v0 = pd->mesh_tri[t*3], v1 = pd->mesh_tri[t*3+1], v2 = pd->mesh_tri[t*3+2];
        const double r0 = pd->mesh_vtx_r[v0], r1 = pd->mesh_vtx_r[v1], r2 = pd->mesh_vtx_r[v2];
        const double z0 = pd->mesh_vtx_z[v0], z1 = pd->mesh_vtx_z[v1], z2 = pd->mesh_vtx_z[v2];
        plasma_data.mesh_tri_rmin[t] = std::min({r0,r1,r2});
        plasma_data.mesh_tri_rmax[t] = std::max({r0,r1,r2});
        plasma_data.mesh_tri_zmin[t] = std::min({z0,z1,z2});
        plasma_data.mesh_tri_zmax[t] = std::max({z0,z1,z2});
      }
      for (int t = 0; t < pd->mesh_ntri; t++) {
        if (pd->mesh_cell_idx[t] < 0) continue;
        const int v0 = pd->mesh_tri[t*3], v1 = pd->mesh_tri[t*3+1], v2 = pd->mesh_tri[t*3+2];
        plasma_data.mapped_cr.push_back((pd->mesh_vtx_r[v0]+pd->mesh_vtx_r[v1]+pd->mesh_vtx_r[v2])/3.0);
        plasma_data.mapped_cz.push_back((pd->mesh_vtx_z[v0]+pd->mesh_vtx_z[v1]+pd->mesh_vtx_z[v2])/3.0);
        plasma_data.mapped_idx.push_back(t);
      }
      plasma_data.buildSpatialHash();
    }

    // Multi-ion species
    if (pd->nion > 0) {
      plasma_data.ions_nspec = pd->nion;
      plasma_data.ions_nz = pnz;
      plasma_data.ions_nr = pnr;
      plasma_data.ion_charge_state_z = pd->ion_charge_z;
      plasma_data.ion_mass_amu = pd->ion_mass_amu;
      plasma_data.ions_dens = pd->ions_dens;
      plasma_data.ions_temp = pd->ions_temp;
      plasma_data.ions_parr_flow = pd->ions_upar;
    }

    // Equilibrium from fix
    if (pd->has_equ) {
      has_equilibrium = 1;
      equ_data.jm = pd->equ_jm;
      equ_data.km = pd->equ_km;
      equ_data.btf = pd->btf;
      equ_data.rtf = pd->rtf;
      equ_data.psib = pd->psib;
      equ_data.psi_axis = pd->psi_axis;
      equ_data.r = pd->equ_r;
      equ_data.z = pd->equ_z;
      // Convert flat psirz to 2D
      equ_data.psi.resize(pd->equ_km, std::vector<double>(pd->equ_jm));
      for (int k = 0; k < pd->equ_km; k++)
        for (int j = 0; j < pd->equ_jm; j++)
          equ_data.psi[k][j] = pd->psirz[k * pd->equ_jm + j];
      const size_t equ_n = static_cast<size_t>(pd->equ_jm) * pd->equ_km;
      if (pd->equ_br.size() == equ_n && pd->equ_bt.size() == equ_n &&
          pd->equ_bz.size() == equ_n) {
        equ_data.br.assign(pd->equ_km,
                           std::vector<double>(pd->equ_jm, 0.0));
        equ_data.bt.assign(pd->equ_km,
                           std::vector<double>(pd->equ_jm, 0.0));
        equ_data.bz.assign(pd->equ_km,
                           std::vector<double>(pd->equ_jm, 0.0));
        for (int k = 0; k < pd->equ_km; ++k)
          for (int j = 0; j < pd->equ_jm; ++j) {
            const size_t idx = static_cast<size_t>(k) * pd->equ_jm + j;
            equ_data.br[k][j] = pd->equ_br[idx];
            equ_data.bt[k][j] = pd->equ_bt[idx];
            equ_data.bz[k][j] = pd->equ_bz[idx];
          }
      }
    }

    if (me == 0 && screen)
      fprintf(screen,
        "compute plasma/fields: using data from fix '%s' (gen=%d)\n",
        background_fix_id.c_str(), pd->generation);

  } else if (input_mode == MODE_FILE) {
    // file mode was removed (constructor rejects it); defensive guard
    error->all(FLERR,"compute plasma/fields: file mode is no longer supported "
               "- load the plasma via fix background");
  }

  // --- Stencil computation and per-cell interpolation ---
  if (input_mode == MODE_BACKGROUND) {
    precomputeStencils(plasma_data.r, plasma_data.z, plasma_stencil);

    for (int icell = 0; icell < ncells; ++icell) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit < 1)         continue;
      plasma_arr[icell] = bilinearInterpolationPlasma(icell, plasma_data);

      if (has_equilibrium) {
        double xyz[3] = {
          0.5 * (cells[icell].lo[0] + cells[icell].hi[0]),
          0.5 * (cells[icell].lo[1] + cells[icell].hi[1]),
          (domain->dimension == 3)
            ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2])
            : 0.5 * (cells[icell].lo[1] + cells[icell].hi[1])
        };
        mag_arr[icell] = query_bfield_at_point(xyz);
      }
    }

  } else if (input_mode == MODE_CONSTANT) {
    plasma_stencil.clear();
    for (int icell = 0; icell < ncells; ++icell) {
      PlasmaFileParams p{};
      p.dens_e = neconst;
      p.temp_e = teconst;
      p.dens_i = niconst;
      p.temp_i = ticonst;
      p.grad_dens_e_r = 0.0;
      p.grad_dens_e_t = 0.0;
      p.grad_dens_e_z = 0.0;
      p.parr_flow = parrflowconst;
      p.parr_flow_r = 0.0;
      p.parr_flow_t = 0.0;
      p.parr_flow_z = 0.0;
      p.grad_temp_e_r = 0.0;
      p.grad_temp_e_t = 0.0;
      p.grad_temp_e_z = 0.0;
      p.grad_temp_i_r = 0.0;
      p.grad_temp_i_t = 0.0;
      p.grad_temp_i_z = 0.0;
      plasma_arr[icell] = p;

      MagneticFieldFileDataParams b{};
      b.br = bconst[0];
      b.bt = bconst[1];
      b.bz = bconst[2];
      b.r = 0.0;
      b.z = 0.0;
      mag_arr[icell] = b;
    }
  } else {
    plasma_stencil.clear();
    const double x0_use = analytic_use_x0 ? analytic_x0 : 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
    const double y0_use = analytic_use_y0 ? analytic_y0 : 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
    const double r0_use = (analytic_r0 > 0.0) ? analytic_r0 : 1.0e-12;
    const double eps_use = (analytic_eps > 0.0) ? analytic_eps : 0.0;
    for (int icell = 0; icell < ncells; ++icell) {
      const double x = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
      const double y = (domain->dimension == 3) ? 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]) : 0.0;
      const double dx = x - x0_use;
      const double dy = (domain->dimension == 3) ? (y - y0_use) : 0.0;
      const double r = std::sqrt(dx*dx + dy*dy + eps_use);
      const double rr = r / r0_use;
      const double expsg = std::exp(-std::pow(rr,12.0));
      double dexp_dr = 0.0;
      if (r > 0.0) dexp_dr = expsg * (-12.0 * std::pow(rr,11.0) / r0_use);

      PlasmaFileParams p{};
      p.dens_e = analytic_ne0 * expsg;
      p.temp_e = analytic_te0 + analytic_te1 * expsg;
      p.dens_i = p.dens_e;
      p.temp_i = analytic_ti0;
      p.grad_dens_e_r = analytic_ne0 * dexp_dr;
      p.grad_dens_e_t = 0.0;
      p.grad_dens_e_z = 0.0;
      p.parr_flow = parrflowconst;
      p.parr_flow_r = 0.0;
      p.parr_flow_t = 0.0;
      p.parr_flow_z = 0.0;
      p.grad_temp_e_r = analytic_te1 * dexp_dr;
      p.grad_temp_e_t = 0.0;
      p.grad_temp_e_z = 0.0;
      p.grad_temp_i_r = 0.0;
      p.grad_temp_i_t = 0.0;
      p.grad_temp_i_z = 0.0;
      plasma_arr[icell] = p;

      MagneticFieldFileDataParams b{};
      b.br = bconst[0];
      b.bt = bconst[1];
      b.bz = bconst[2];
      b.r = r;
      b.z = (domain->dimension == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]) : 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
      mag_arr[icell] = b;
    }
  }

  // sepdist needs a psi map: fail loudly at init instead of silently
  // reporting 0 (would blanket-refine under adapt_grid `thresh less`)
  for (int iv = 0; iv < nvalue; ++iv) {
    if (value[iv] == SEPDIST &&
        (!has_equilibrium || equ_data.jm < 2 || equ_data.km < 2 ||
         std::fabs(equ_data.psib - equ_data.psi_axis) < 1e-30))
      error->all(FLERR,
        "compute plasma/fields: sepdist requires an equilibrium psi map — "
        "add `equilibrium <file>` or use a background/plasma.h5 with an "
        "embedded equilibrium group");
  }

  sample_stale = 0;
}


/* ----------------------------------------------------------------------
   Reload plasma background from the current plasmaStatePath.
   Re-reads HDF5, broadcasts, re-interpolates onto all grid cells.
   Called by the coupling driver between time chunks to refresh the
   background plasma without reinitializing the entire compute.
------------------------------------------------------------------------- */

void ComputePlasmaFields::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  // grid changed (adapt refine / balance) since the per-cell arrays were
  // sampled: re-run init() to rebuild + resample before indexing them.
  // init() clears the flag; its reallocate() call no-ops (sizes now match).
  if (sample_stale) init();

  constexpr double eQ = 1.602176634e-19;
  constexpr double tiny = 1.0e-30;

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int dim = domain->dimension;

  for (int icell = 0; icell < nglocal; icell++) {
    double *row = (nvalue == 1) ? NULL : array_grid[icell];
    if (!(cinfo[icell].mask & groupbit) || cells[icell].nsplit < 1) {
      if (nvalue == 1) vector_grid[icell] = 0.0;
      else for (int iv = 0; iv < nvalue; ++iv) row[iv] = 0.0;
      continue;
    }

    const PlasmaFileParams &P = plasma_arr[icell];
    const MagneticFieldFileDataParams &B = mag_arr[icell];

    // cell centroid in SPARTA coords and cylindrical (R,Z) — shared by the
    // mesh-E lookup, 3D toroidal rotation, and sepdist below
    const Grid::ChildCell &cell = cells[icell];
    const double xyz_c[3] = {
      0.5 * (cell.lo[0] + cell.hi[0]),
      0.5 * (cell.lo[1] + cell.hi[1]),
      (dim == 3) ? 0.5 * (cell.lo[2] + cell.hi[2]) : 0.0
    };
    double cell_Rc, cell_Zc;
    OpenEdge::sparta_to_RZ(xyz_c, dim, domain->axisymmetric, cell_Rc, cell_Zc,
                           plasma_data.column_x0, plasma_data.column_y0);

    // Base components are cylindrical (r, t, z).
    const double Br = B.br;
    const double Bt = B.bt;
    const double Bzv = B.bz;
    // Electric field: sourced from the plasma code (SOLPS po, SOLEDGE3X
    // phi, OEDGE epara) at converter time and embedded in plasma.h5
    // /mesh/e_{r,z,t}. For file-mode input the MODE_FILE path is gone;
    // for constant/analytic modes we use the econst[] vector.
    double Er  = (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC)
                  ? econst[0] : 0.0;
    double Et  = (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC)
                  ? econst[1] : 0.0;
    double Ezv = (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC)
                  ? econst[2] : 0.0;
    if (input_mode == MODE_BACKGROUND && plasma_data.has_mesh &&
        !plasma_data.mesh_e_r.empty()) {
      // Look up the mesh cell for this SPARTA cell centroid and read
      // stored E components directly.
      const int tri = findNearestMappedTriangle(plasma_data, cell_Rc, cell_Zc, 0.1);
      int mcell = -1;
      if (tri >= 0 && tri < static_cast<int>(plasma_data.mesh_cell_idx.size()))
        mcell = plasma_data.mesh_cell_idx[tri];
      if (mcell >= 0 &&
          mcell < static_cast<int>(plasma_data.mesh_e_r.size())) {
        Er  = plasma_data.mesh_e_r[mcell];
        Ezv = plasma_data.mesh_e_z[mcell];
        Et  = plasma_data.mesh_e_t[mcell];
      }
    }
    const double Vr = P.parr_flow_r;
    const double Vt = P.parr_flow_t;
    const double Vzv = P.parr_flow_z;

    // Cylindrical -> Cartesian transform in 3D Cartesian geometry.
    // Project cylindrical (r, z, t) field components onto SPARTA's
    // (x, y, z) slot order. The helper handles all three modes:
    //   2D Cart (legacy):  x=R, y=Z, z=phi  -> Bx=Br, By=Bz, Bz=Bt
    //   2D axi (native):   x=Z, y=R, z=phi  -> Bx=Bz, By=Br, Bz=Bt
    //   3D Cartesian:      Br/Bt rotated by phi at this cell's centroid
    double phi = 0.0;
    if (dim == 3) phi = std::atan2(xyz_c[1], xyz_c[0]);
    double Bx, By, Bzz, Ex, Ey, Ezz, Vx, Vy, Vzz;
    OpenEdge::RZphi_force_to_sparta(Br, Bzv, Bt, dim, domain->axisymmetric,
                                     phi, Bx, By, Bzz);
    OpenEdge::RZphi_force_to_sparta(Er, Ezv, Et, dim, domain->axisymmetric,
                                     phi, Ex, Ey, Ezz);
    OpenEdge::RZphi_force_to_sparta(Vr, Vzv, Vt, dim, domain->axisymmetric,
                                     phi, Vx, Vy, Vzz);
    // Parallel E for diagnostic output: just the dot product of the
    // mesh-stored E vector with b-hat. No pressure-balance approximation.
    const double Bmag = std::sqrt(Br*Br + Bt*Bt + Bzv*Bzv);
    const double invBmag = (Bmag > tiny) ? 1.0 / Bmag : 0.0;
    const double bhat_r = Br * invBmag;
    const double bhat_t = Bt * invBmag;
    const double bhat_z = Bzv * invBmag;
    const double epar = (Bmag > tiny)
      ? (Er * bhat_r + Et * bhat_t + Ezv * bhat_z) : 0.0;

    // Refresh SPARTA-slot E in case mesh E overrode the zero defaults.
    OpenEdge::RZphi_force_to_sparta(Er, Ezv, Et, dim, domain->axisymmetric,
                                     phi, Ex, Ey, Ezz);

    for (int iv = 0; iv < nvalue; ++iv) {
      double vout = 0.0;
      switch (value[iv]) {
        case BR:        vout = Br; break;
        case BT:        vout = Bt; break;
        case BZ:        vout = (dim == 2) ? Bzz : Bzv; break;
        case BX:        vout = Bx; break;
        case BY:        vout = By; break;
        case ER:        vout = Er; break;
        case ET:        vout = Et; break;
        case EZ:        vout = (dim == 2) ? Ezz : Ezv; break;
        case EX:        vout = Ex; break;
        case EY:        vout = Ey; break;
        case VR:        vout = Vr; break;
        case VT:        vout = Vt; break;
        case VZ:        vout = (dim == 2) ? Vzz : Vzv; break;
        case VX:        vout = Vx; break;
        case VY:        vout = Vy; break;
        case TI:        vout = P.temp_i; break;
        case TE:        vout = P.temp_e; break;
        case NI:        vout = P.dens_i; break;
        case NE:        vout = P.dens_e; break;
        case PARRFLOW:  vout = P.parr_flow; break;
        case EPAR:      vout = epar; break;
        case GRAD_TE_R: vout = P.grad_temp_e_r; break;
        case GRAD_TE_T: vout = P.grad_temp_e_t; break;
        case GRAD_TE_Z: vout = P.grad_temp_e_z; break;
        case GRAD_TI_R: vout = P.grad_temp_i_r; break;
        case GRAD_TI_T: vout = P.grad_temp_i_t; break;
        case GRAD_TI_Z: vout = P.grad_temp_i_z; break;
        case GRAD_NE_R: vout = P.grad_dens_e_r; break;
        case GRAD_NE_Z: vout = P.grad_dens_e_z; break;
        case GRAD_TE_MAG:
          // poloidal-plane |grad Te| by design (refinement criterion);
          // the toroidal component grad_te_t is deliberately excluded
          vout = std::sqrt(P.grad_temp_e_r*P.grad_temp_e_r +
                           P.grad_temp_e_z*P.grad_temp_e_z);
          break;
        case SEPDIST:
          // |psiN - 1| at the cell center: adapt_grid `thresh less` on this
          // refines a separatrix-following band. init() guarantees a psi map.
          vout = std::fabs(psi_norm_from_equ(cell_Rc, cell_Zc) - 1.0);
          break;
        default:        vout = 0.0; break;
      }
      if (nvalue == 1) vector_grid[icell] = vout;
      else row[iv] = vout;
    }
  }
}


/* ----------------------------------------------------------------------
   reallocate vector if nglocal has changed
   called by init() and load balancer
------------------------------------------------------------------------- */

void ComputePlasmaFields::reallocate() {
  if (grid->nlocal == nglocal) return;
  memory->destroy(vector_grid);
  memory->destroy(array_grid);
  nglocal = grid->nlocal;
  // per-cell sampled arrays (plasma_arr/mag_arr) are still sized
  // for the old cell set. Re-sample immediately: the pusher reads mag_arr
  // every step, so deferring to the next compute_per_grid() leaves an
  // out-of-bounds window after adapt/balance. init()'s own reallocate()
  // call no-ops (sizes match once nglocal is updated below).
  sample_stale = 1;
  if (nvalue == 1) {
    memory->create(vector_grid, nglocal, "plasma/fields:vector_grid");
  } else {
    memory->create(array_grid, nglocal, nvalue, "plasma/fields:array_grid");
  }
  // plasma_arr non-NULL means we were fully initialized before this grid
  // change — re-sample now. NULL means we're inside the first init()'s own
  // reallocate() call; that init() finishes the sampling itself.
  if (plasma_arr) init();
}


/* ----------------------------------------------------------------------
   normalized psi from this compute's own equilibrium copy (any input mode)
------------------------------------------------------------------------- */

double ComputePlasmaFields::psi_norm_from_equ(double R, double Z) const
{
  if (!has_equilibrium || equ_data.jm < 2 || equ_data.km < 2) return 1.0e30;
  const double denom = equ_data.psib - equ_data.psi_axis;
  if (std::fabs(denom) < 1e-30) return 1.0e30;

  const double dr = equ_data.r[1] - equ_data.r[0];
  const double dz = equ_data.z[1] - equ_data.z[0];
  if (std::fabs(dr) < 1e-30 || std::fabs(dz) < 1e-30) return 1.0e30;

  const double Rc = std::min(std::max(R, equ_data.r.front()), equ_data.r.back());
  const double Zc = std::min(std::max(Z, equ_data.z.front()), equ_data.z.back());
  const double fi = (Rc - equ_data.r.front()) / dr;
  const double fj = (Zc - equ_data.z.front()) / dz;
  const int i0 = std::max(0, std::min((int)fi, equ_data.jm - 2));
  const int j0 = std::max(0, std::min((int)fj, equ_data.km - 2));
  const double s = std::max(0.0, std::min(1.0, fi - i0));
  const double t = std::max(0.0, std::min(1.0, fj - j0));

  const double psi =
      (1-s)*(1-t)*equ_data.psi[j0][i0]   + s*(1-t)*equ_data.psi[j0][i0+1]
    + (1-s)*t*equ_data.psi[j0+1][i0]     + s*t*equ_data.psi[j0+1][i0+1];
  return (psi - equ_data.psi_axis) / denom;
}

/* ----------------------------------------------------------------------
   memory usage of local grid-based array
------------------------------------------------------------------------- */

bigint ComputePlasmaFields::memory_usage()
{
  bigint bytes = (bigint) nglocal * nvalue * sizeof(double);
  return bytes;
}


/* ----------------------------------------------------------------------
  read magnetic field data from file
------------------------------------------------------------------------- */
/* ---------------------------------------------------------------------- */

void PlasmaFileData::buildSpatialHash(int nr_bins, int nz_bins)
{
  if (!has_mesh || mesh_ntri == 0) return;

  hash_rmin = *std::min_element(mesh_tri_rmin.begin(), mesh_tri_rmin.end());
  double rmax = *std::max_element(mesh_tri_rmax.begin(), mesh_tri_rmax.end());
  hash_zmin = *std::min_element(mesh_tri_zmin.begin(), mesh_tri_zmin.end());
  double zmax = *std::max_element(mesh_tri_zmax.begin(), mesh_tri_zmax.end());

  hash_nr = nr_bins;
  hash_nz = nz_bins;
  hash_dr = (rmax - hash_rmin) / nr_bins + 1e-12;
  hash_dz = (zmax - hash_zmin) / nz_bins + 1e-12;

  hash_grid.assign(static_cast<size_t>(hash_nr) * hash_nz, std::vector<int>());

  for (int t = 0; t < mesh_ntri; t++) {
    int ir0 = std::max(0, (int)((mesh_tri_rmin[t] - hash_rmin) / hash_dr));
    int ir1 = std::min(hash_nr - 1, (int)((mesh_tri_rmax[t] - hash_rmin) / hash_dr));
    int iz0 = std::max(0, (int)((mesh_tri_zmin[t] - hash_zmin) / hash_dz));
    int iz1 = std::min(hash_nz - 1, (int)((mesh_tri_zmax[t] - hash_zmin) / hash_dz));
    for (int iz = iz0; iz <= iz1; iz++)
      for (int ir = ir0; ir <= ir1; ir++)
        hash_grid[iz * hash_nr + ir].push_back(t);
  }
}


/*----------------------------------------------------------------------
   broadcast plasma data
------------------------------------------------------------------------- */
/*----------------------------------------------------------------------
   bilinear interpolation of plasma data at cell center
------------------------------------------------------------------------- */

PlasmaFileParams ComputePlasmaFields::bilinearInterpolationPlasma(
    int icell, const PlasmaFileData &data)
{
  PlasmaFileParams P{};  // default all zeros

  // Mesh-first: for mesh-only plasma.h5 (the current converter output),
  // the EIRENE triangulation provides per-cell plasma directly.
  if (data.has_mesh) {
    meshLookupPlasma(icell, data, P);
    // Gradients (grad_te_{r,z}, grad_ne_{r,z}, etc.) on the mesh are
    // not yet implemented; leave them zero. Consumers that need them
    // (fix thermal_force, fix cross_diffusion) should migrate to a
    // per-particle finite-difference scheme.
    return P;
  }

  // Legacy regular-grid fallback — used only if plasma.h5 still ships
  // the top-level r/z + dens_e/temp_e/etc. arrays. Will go away once
  // the last test migrates.
  if (icell < 0 || icell >= static_cast<int>(plasma_stencil.size())) return P;
  const BilinearStencil &s = plasma_stencil[icell];
  if (!s.valid) return P;
  P.temp_e = interpField2D(data.temp_e, s);
  P.dens_e = interpField2D(data.dens_e, s);
  P.temp_i = interpField2D(data.temp_i, s);
  P.dens_i = interpField2D(data.dens_i, s);

  gradField2D(data.dens_e, s, P.grad_dens_e_r, P.grad_dens_e_z);
  P.grad_dens_e_t = 0.0;
  P.grad_temp_e_r = interpField2D(data.grad_temp_e_r, s);
  P.grad_temp_e_t = interpField2D(data.grad_temp_e_t, s);
  P.grad_temp_e_z = interpField2D(data.grad_temp_e_z, s);
  P.grad_temp_i_r = interpField2D(data.grad_temp_i_r, s);
  P.grad_temp_i_t = interpField2D(data.grad_temp_i_t, s);
  P.grad_temp_i_z = interpField2D(data.grad_temp_i_z, s);
  P.parr_flow_r   = interpField2D(data.parr_flow_r, s);
  P.parr_flow_t   = interpField2D(data.parr_flow_t, s);
  P.parr_flow_z   = interpField2D(data.parr_flow_z, s);
  P.parr_flow     = interpField2D(data.parr_flow, s);
  P.q_mag = data.has_qmag ? interpField2D(data.q_mag, s) : 0.0;

  return P;
}


/* ---------------------------------------------------------------------- */

int ComputePlasmaFields::findMeshTriangle(
    const PlasmaFileData &data, double r, double z) const
{
  // Use spatial hash if available (O(1) instead of O(N))
  if (data.hash_nr > 0 && !data.hash_grid.empty()) {
    int ir = (int)((r - data.hash_rmin) / data.hash_dr);
    int iz = (int)((z - data.hash_zmin) / data.hash_dz);
    if (ir < 0 || ir >= data.hash_nr || iz < 0 || iz >= data.hash_nz)
      return -1;
    const auto &candidates = data.hash_grid[iz * data.hash_nr + ir];
    for (int t : candidates) {
      const int v0 = data.mesh_tri[t*3+0];
      const int v1 = data.mesh_tri[t*3+1];
      const int v2 = data.mesh_tri[t*3+2];
      const double r0 = data.mesh_vtx_r[v0], z0 = data.mesh_vtx_z[v0];
      const double r1 = data.mesh_vtx_r[v1], z1 = data.mesh_vtx_z[v1];
      const double r2 = data.mesh_vtx_r[v2], z2 = data.mesh_vtx_z[v2];
      const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
      if (std::fabs(d) < 1e-30) continue;
      const double a = ((r-r0)*(z2-z0) - (r2-r0)*(z-z0)) / d;
      const double b = ((r1-r0)*(z-z0) - (r-r0)*(z1-z0)) / d;
      if (a >= -1e-10 && b >= -1e-10 && (a+b) <= 1.0+1e-10) return t;
    }
    return -1;
  }

  // Fallback: brute force scan
  for (int t = 0; t < data.mesh_ntri; t++) {
    if (r < data.mesh_tri_rmin[t] || r > data.mesh_tri_rmax[t]) continue;
    if (z < data.mesh_tri_zmin[t] || z > data.mesh_tri_zmax[t]) continue;
    const int v0 = data.mesh_tri[t*3+0];
    const int v1 = data.mesh_tri[t*3+1];
    const int v2 = data.mesh_tri[t*3+2];
    const double r0 = data.mesh_vtx_r[v0], z0 = data.mesh_vtx_z[v0];
    const double r1 = data.mesh_vtx_r[v1], z1 = data.mesh_vtx_z[v1];
    const double r2 = data.mesh_vtx_r[v2], z2 = data.mesh_vtx_z[v2];
    const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
    if (std::fabs(d) < 1e-30) continue;
    const double a = ((r-r0)*(z2-z0) - (r2-r0)*(z-z0)) / d;
    const double b = ((r1-r0)*(z-z0) - (r-r0)*(z1-z0)) / d;
    if (a >= -1e-10 && b >= -1e-10 && (a+b) <= 1.0+1e-10) return t;
  }
  return -1;
}

int ComputePlasmaFields::findNearestMappedTriangle(
    const PlasmaFileData &data, double r, double z, double max_dist) const
{
  double best_d2 = max_dist * max_dist;
  int best = -1;
  for (int i = 0; i < static_cast<int>(data.mapped_idx.size()); i++) {
    double dr = data.mapped_cr[i] - r;
    double dz = data.mapped_cz[i] - z;
    double d2 = dr*dr + dz*dz;
    if (d2 < best_d2) { best_d2 = d2; best = i; }
  }
  return (best >= 0) ? data.mapped_idx[best] : -1;
}

bool ComputePlasmaFields::meshLookupPlasmaAtPoint(
    const PlasmaFileData &data, double r, double z, PlasmaFileParams &P) const
{
  int tri_idx = findMeshTriangle(data, r, z);
  if (tri_idx < 0 || data.mesh_cell_idx[tri_idx] < 0)
    tri_idx = findNearestMappedTriangle(data, r, z, 0.05);
  if (tri_idx < 0) return false;

  int cell = data.mesh_cell_idx[tri_idx];
  if (cell < 0 || cell >= data.mesh_ncell) return false;

  P.temp_e = data.mesh_te[cell];
  P.dens_e = data.mesh_ne[cell];
  P.temp_i = data.mesh_ti[cell];
  P.dens_i = data.mesh_ni[cell];
  P.parr_flow = (!data.mesh_upar.empty()) ? data.mesh_upar[cell] : 0.0;
  return true;
}

bool ComputePlasmaFields::meshLookupPlasma(
    int icell, const PlasmaFileData &data, PlasmaFileParams &P) const
{
  Grid::ChildCell *cells = grid->cells;
  const int dim = domain->dimension;
  const double xc[3] = {
      0.5 * (cells[icell].lo[0] + cells[icell].hi[0]),
      0.5 * (cells[icell].lo[1] + cells[icell].hi[1]),
      (dim == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]) : 0.0};
  double r, z;
  OpenEdge::sparta_to_RZ(xc, dim, domain->axisymmetric, r, z,
                         data.column_x0, data.column_y0);
  return meshLookupPlasmaAtPoint(data, r, z, P);
}

double ComputePlasmaFields::interpField3DFlat(
    const std::vector<double> &field, int ispec, int nz, int nr,
    const BilinearStencil &s) const
{
  if (!s.valid || field.empty() || nz <= 0 || nr <= 0 || ispec < 0) return 0.0;
  const size_t nslice = static_cast<size_t>(nz) * static_cast<size_t>(nr);
  const size_t off = static_cast<size_t>(ispec) * nslice;
  if (off + nslice > field.size()) return 0.0;

  if (s.iz1 < 0 || s.iz2 >= nz || s.ir1 < 0 || s.ir2 >= nr) return 0.0;
  const size_t i11 = off + static_cast<size_t>(s.iz1) * nr + static_cast<size_t>(s.ir1);
  const size_t i21 = off + static_cast<size_t>(s.iz1) * nr + static_cast<size_t>(s.ir2);
  const size_t i12 = off + static_cast<size_t>(s.iz2) * nr + static_cast<size_t>(s.ir1);
  const size_t i22 = off + static_cast<size_t>(s.iz2) * nr + static_cast<size_t>(s.ir2);

  const double q11 = field[i11];
  const double q21 = field[i21];
  const double q12 = field[i12];
  const double q22 = field[i22];
  return s.w11 * q11 + s.w21 * q21 + s.w12 * q12 + s.w22 * q22;
}

/*----------------------------------------------------------------------
   Build a bilinear stencil at an arbitrary (x,y,z) position.
   This is the factored-out core of precomputeStencils().
------------------------------------------------------------------------- */

ComputePlasmaFields::BilinearStencil
ComputePlasmaFields::makeStencilAtPoint(
    const double xyz[3],
    const std::vector<double> &r_vals,
    const std::vector<double> &z_vals) const
{
  BilinearStencil s{};
  if (r_vals.size() < 2 || z_vals.size() < 2) return s;

  const int dim = domain->dimension;
  double r, z;
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, r, z,
                         plasma_data.column_x0, plasma_data.column_y0);

  const int nr = static_cast<int>(r_vals.size());
  const int nz = static_cast<int>(z_vals.size());
  const double r_clamp = std::min(std::max(r, r_vals.front()), r_vals.back());
  const double z_clamp = std::min(std::max(z, z_vals.front()), z_vals.back());

  auto r_it = std::lower_bound(r_vals.begin(), r_vals.end(), r_clamp);
  auto z_it = std::lower_bound(z_vals.begin(), z_vals.end(), z_clamp);

  int ir2 = static_cast<int>(r_it - r_vals.begin());
  int iz2 = static_cast<int>(z_it - z_vals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (ir2 >= nr) ir2 = nr - 1;
  if (iz2 <= 0) iz2 = 1;
  if (iz2 >= nz) iz2 = nz - 1;

  s.ir1 = ir2 - 1;
  s.ir2 = ir2;
  s.iz1 = iz2 - 1;
  s.iz2 = iz2;

  const double R1 = r_vals[s.ir1];
  const double R2 = r_vals[s.ir2];
  const double Z1 = z_vals[s.iz1];
  const double Z2 = z_vals[s.iz2];
  const double denomR = R2 - R1;
  const double denomZ = Z2 - Z1;

  if (denomR == 0.0 || denomZ == 0.0) {
    s.w11 = 1.0;
    s.w21 = s.w12 = s.w22 = 0.0;
    s.valid = 1;
    return s;
  }

  const double t = (r_clamp - R1) / denomR;
  const double u = (z_clamp - Z1) / denomZ;
  s.t = t;
  s.u = u;
  s.inv_dR = 1.0 / denomR;
  s.inv_dZ = 1.0 / denomZ;
  s.w11 = (1.0 - t) * (1.0 - u);
  s.w21 = t * (1.0 - u);
  s.w12 = (1.0 - t) * u;
  s.w22 = t * u;
  s.valid = 1;
  return s;
}

/*----------------------------------------------------------------------
   Precompute stencils for all cell centers (delegates to makeStencilAtPoint)
------------------------------------------------------------------------- */

void ComputePlasmaFields::precomputeStencils(
    const std::vector<double> &r_vals,
    const std::vector<double> &z_vals,
    std::vector<BilinearStencil> &stencil)
{
  const int ncells = grid->nlocal;
  stencil.clear();
  stencil.resize(ncells);

  const int dim = domain->dimension;

  for (int icell = 0; icell < ncells; ++icell) {
    Grid::ChildCell *cell = &grid->cells[icell];
    double cc[3];
    cc[0] = 0.5 * (cell->lo[0] + cell->hi[0]);
    cc[1] = 0.5 * (cell->lo[1] + cell->hi[1]);
    cc[2] = (dim == 3)
            ? 0.5 * (cell->lo[2] + cell->hi[2])
            : cc[1];
    stencil[icell] = makeStencilAtPoint(cc, r_vals, z_vals);
  }
}

double ComputePlasmaFields::interpField2D(
    const std::vector<std::vector<double>> &field,
    const BilinearStencil &s) const
{
  if (!s.valid || field.empty()) return 0.0;
  if (s.iz1 < 0 || s.iz2 >= static_cast<int>(field.size())) return 0.0;
  if (field[s.iz1].empty() || field[s.iz2].empty()) return 0.0;
  if (s.ir1 < 0 || s.ir2 >= static_cast<int>(field[s.iz1].size())) return 0.0;
  if (s.ir2 >= static_cast<int>(field[s.iz2].size())) return 0.0;

  const double q11 = field[s.iz1][s.ir1];
  const double q21 = field[s.iz1][s.ir2];
  const double q12 = field[s.iz2][s.ir1];
  const double q22 = field[s.iz2][s.ir2];
  return s.w11 * q11 + s.w21 * q21 + s.w12 * q12 + s.w22 * q22;
}

void ComputePlasmaFields::gradField2D(
    const std::vector<std::vector<double>> &field,
    const BilinearStencil &s,
    double &grad_r,
    double &grad_z) const
{
  grad_r = 0.0;
  grad_z = 0.0;
  if (!s.valid || field.empty()) return;
  if (s.iz1 < 0 || s.iz2 >= static_cast<int>(field.size())) return;
  if (field[s.iz1].empty() || field[s.iz2].empty()) return;
  if (s.ir1 < 0 || s.ir2 >= static_cast<int>(field[s.iz1].size())) return;
  if (s.ir2 >= static_cast<int>(field[s.iz2].size())) return;
  if (s.inv_dR <= 0.0 || s.inv_dZ <= 0.0) return;

  const double q11 = field[s.iz1][s.ir1];
  const double q21 = field[s.iz1][s.ir2];
  const double q12 = field[s.iz2][s.ir1];
  const double q22 = field[s.iz2][s.ir2];

  grad_r = ((1.0 - s.u) * (q21 - q11) + s.u * (q22 - q12)) * s.inv_dR;
  grad_z = ((1.0 - s.t) * (q12 - q11) + s.t * (q22 - q21)) * s.inv_dZ;
}

/*----------------------------------------------------------------------
   Point-query: interpolate plasma fields at arbitrary (x,y,z)
------------------------------------------------------------------------- */

PlasmaFileParams ComputePlasmaFields::query_plasma_at_point(
    const double xyz[3]) const
{
  PlasmaFileParams P{};

  if (input_mode == MODE_CONSTANT) {
    P.dens_e = neconst;
    P.temp_e = teconst;
    P.dens_i = niconst;
    P.temp_i = ticonst;
    P.parr_flow = parrflowconst;
    return P;
  }

  if (input_mode == MODE_ANALYTIC) {
    const double x0_use = analytic_use_x0 ? analytic_x0
                          : 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
    const double y0_use = analytic_use_y0 ? analytic_y0
                          : 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
    const double r0_use = (analytic_r0 > 0.0) ? analytic_r0 : 1.0e-12;
    const double eps_use = (analytic_eps > 0.0) ? analytic_eps : 0.0;

    const double dx = xyz[0] - x0_use;
    const double dy = (domain->dimension == 3) ? (xyz[1] - y0_use) : 0.0;
    const double r = std::sqrt(dx*dx + dy*dy + eps_use);
    const double rr = r / r0_use;
    const double expsg = std::exp(-std::pow(rr, 12.0));
    double dexp_dr = 0.0;
    if (r > 0.0) dexp_dr = expsg * (-12.0 * std::pow(rr, 11.0) / r0_use);

    P.dens_e = analytic_ne0 * expsg;
    P.temp_e = analytic_te0 + analytic_te1 * expsg;
    P.dens_i = P.dens_e;
    P.temp_i = analytic_ti0;
    P.grad_dens_e_r = analytic_ne0 * dexp_dr;
    P.parr_flow = parrflowconst;
    P.grad_temp_e_r = analytic_te1 * dexp_dr;
    return P;
  }

  const int dim = domain->dimension;
  double r, z;
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, r, z,
                         plasma_data.column_x0, plasma_data.column_y0);

  bool used_mesh = false;
  if (plasma_data.has_mesh)
    used_mesh = meshLookupPlasmaAtPoint(plasma_data, r, z, P);

  // regular-grid fallback: build stencil on-the-fly and interpolate
  BilinearStencil s = makeStencilAtPoint(xyz, plasma_data.r, plasma_data.z);
  if (!used_mesh) {
    if (!s.valid) return P;
    P.temp_e = interpField2D(plasma_data.temp_e, s);
    P.dens_e = interpField2D(plasma_data.dens_e, s);
    P.temp_i = interpField2D(plasma_data.temp_i, s);
    P.dens_i = interpField2D(plasma_data.dens_i, s);
  } else if (!s.valid) {
    return P;
  }

  gradField2D(plasma_data.dens_e, s, P.grad_dens_e_r, P.grad_dens_e_z);
  P.grad_dens_e_t = 0.0;
  P.grad_temp_e_r = interpField2D(plasma_data.grad_temp_e_r, s);
  P.grad_temp_e_t = interpField2D(plasma_data.grad_temp_e_t, s);
  P.grad_temp_e_z = interpField2D(plasma_data.grad_temp_e_z, s);
  P.grad_temp_i_r = interpField2D(plasma_data.grad_temp_i_r, s);
  P.grad_temp_i_t = interpField2D(plasma_data.grad_temp_i_t, s);
  P.grad_temp_i_z = interpField2D(plasma_data.grad_temp_i_z, s);
  P.parr_flow_r   = interpField2D(plasma_data.parr_flow_r, s);
  P.parr_flow_t   = interpField2D(plasma_data.parr_flow_t, s);
  P.parr_flow_z   = interpField2D(plasma_data.parr_flow_z, s);
  P.parr_flow     = interpField2D(plasma_data.parr_flow, s);
  P.q_mag = plasma_data.has_qmag ?
            interpField2D(plasma_data.q_mag, s) : 0.0;

  // Parallel E for per-particle query: use mesh-stored E vector if
  // available. No pressure-balance approximation.
  if (plasma_data.has_mesh && !plasma_data.mesh_e_r.empty()) {
    double R, Z;
    OpenEdge::sparta_to_RZ(xyz, domain->dimension, domain->axisymmetric, R, Z,
                           plasma_data.column_x0, plasma_data.column_y0);
    const int tri = findNearestMappedTriangle(plasma_data, R, Z, 0.1);
    int mcell = -1;
    if (tri >= 0 && tri < static_cast<int>(plasma_data.mesh_cell_idx.size()))
      mcell = plasma_data.mesh_cell_idx[tri];
    if (mcell >= 0 &&
        mcell < static_cast<int>(plasma_data.mesh_e_r.size())) {
      MagneticFieldFileDataParams bf = query_bfield_at_point(xyz);
      const double Bmag = std::sqrt(bf.br*bf.br + bf.bt*bf.bt + bf.bz*bf.bz);
      if (Bmag > 1.0e-30) {
        const double invB = 1.0 / Bmag;
        P.epar = (plasma_data.mesh_e_r[mcell] * bf.br +
                  plasma_data.mesh_e_t[mcell] * bf.bt +
                  plasma_data.mesh_e_z[mcell] * bf.bz) * invB;
      }
    }
  }

  return P;
}

/*----------------------------------------------------------------------
   Point-query: interpolate magnetic field at arbitrary (x,y,z)
------------------------------------------------------------------------- */

MagneticFieldFileDataParams ComputePlasmaFields::query_bfield_at_point(
    const double xyz[3], bool prefer_equilibrium) const
{
  MagneticFieldFileDataParams B{};

  if (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC) {
    B.derivatives_valid = (input_mode == MODE_CONSTANT);  // uniform: grads exactly 0
    B.axisymmetric_source = true;   // cylindrical constants
    B.br = bconst[0];
    B.bt = bconst[1];
    B.bz = bconst[2];
    if (input_mode == MODE_ANALYTIC) {
      const double x0_use = analytic_use_x0 ? analytic_x0
                            : 0.5 * (domain->boxlo[0] + domain->boxhi[0]);
      const double y0_use = analytic_use_y0 ? analytic_y0
                            : 0.5 * (domain->boxlo[1] + domain->boxhi[1]);
      const double eps_use = (analytic_eps > 0.0) ? analytic_eps : 0.0;
      const double dx = xyz[0] - x0_use;
      const double dy = (domain->dimension == 3) ? (xyz[1] - y0_use) : 0.0;
      B.r = std::sqrt(dx*dx + dy*dy + eps_use);
      B.z = (domain->dimension == 3) ? xyz[2] : xyz[1];
      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
    }
    return B;
  }

  // Mesh-native B (per-triangle vertex average). Takes precedence when
  // populated; gradient fields are not computed here since the mesh
  // carries point values only. Consumers that need dB/dR, dB/dZ should
  // fall through to the regular-grid or equilibrium branch instead.
  if (!prefer_equilibrium && !plasma_data.mesh_tri_br.empty()) {
    double R, Z;
    OpenEdge::sparta_to_RZ(xyz, domain->dimension, domain->axisymmetric, R, Z,
                           plasma_data.column_x0, plasma_data.column_y0);
    const int tri = findMeshTriangle(plasma_data, R, Z);
    if (tri >= 0 && tri < static_cast<int>(plasma_data.mesh_tri_br.size())) {
      B.br = plasma_data.mesh_tri_br[tri];
      B.bz = plasma_data.mesh_tri_bz[tri];
      B.bt = plasma_data.mesh_tri_bt[tri];
      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
      B.axisymmetric_source = true;   // mirror FixBackground mesh branch
      return B;
    }
    // outside mesh footprint: fall through to grid / equ branches.
  }

  // Smooth equilibrium B with derivatives. Native maps preserve the source
  // convention; legacy files fall back to psi reconstruction. Footprint-checked:
  // points outside the equilibrium rectangle must not use edge-extrapolated
  // psi stencils — they fall back to the mesh (GCA) or zero instead.
  const int dim = domain->dimension;
  bool equ_usable = has_equilibrium && equ_data.jm >= 3 && equ_data.km >= 3;
  double R = 0.0, Z = 0.0;
  if (equ_usable) {
    OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, R, Z,
                           plasma_data.column_x0, plasma_data.column_y0);
    equ_usable = R >= 1.0e-10 &&
                 R >= equ_data.r.front() && R <= equ_data.r.back() &&
                 Z >= equ_data.z.front() && Z <= equ_data.z.back();
  }
  if (!equ_usable) {
    if (prefer_equilibrium) {
      // GCA caller with no usable smooth field here: retry the mesh /
      // constant path, whose B carries no spatial derivatives — grad-B
      // and mirror terms are zero there. Warn once.
      static int warned = 0;
      if (!warned && comm->me == 0) {
        warned = 1;
        error->warning(FLERR,
          "compute plasma/fields: GCA requested equilibrium B derivatives "
          "but none are usable at some query points — grad-B/mirror terms "
          "are zero where mesh/constant B is used instead");
      }
      return query_bfield_at_point(xyz, false);
    }
    return B;
  }

  const EquilibriumData &equ = equ_data;
  if (sample_native_equilibrium_b(equ, R, Z, B)) return B;

  const int jm = equ.jm;
  const int km = equ.km;
  const double dr = equ.r[1] - equ.r[0];
  const double dz = equ.z[1] - equ.z[0];
  if (dr <= 0.0 || dz <= 0.0)
    return prefer_equilibrium ? query_bfield_at_point(xyz, false) : B;

  double fj = (R - equ.r[0]) / dr;
  double fk = (Z - equ.z[0]) / dz;
  int jc = static_cast<int>(std::round(fj));
  int kc = static_cast<int>(std::round(fk));
  jc = std::max(1, std::min(jc, jm - 2));
  kc = std::max(1, std::min(kc, km - 2));

  const double dR = equ.r[jc+1] - equ.r[jc-1];
  const double dZ = equ.z[kc+1] - equ.z[kc-1];
  const double dpsi_dR = (equ.psi[kc][jc+1] - equ.psi[kc][jc-1]) / dR;
  const double dpsi_dZ = (equ.psi[kc+1][jc] - equ.psi[kc-1][jc]) / dZ;

  const double dR1 = equ.r[jc+1] - equ.r[jc];
  const double dR0 = equ.r[jc] - equ.r[jc-1];
  const double dZ1 = equ.z[kc+1] - equ.z[kc];
  const double dZ0 = equ.z[kc] - equ.z[kc-1];

  const double d2psi_dR2 = 2.0 * (equ.psi[kc][jc+1] / (dR1*(dR1+dR0))
                                 - equ.psi[kc][jc] / (dR1*dR0)
                                 + equ.psi[kc][jc-1] / (dR0*(dR1+dR0)));
  const double d2psi_dZ2 = 2.0 * (equ.psi[kc+1][jc] / (dZ1*(dZ1+dZ0))
                                 - equ.psi[kc][jc] / (dZ1*dZ0)
                                 + equ.psi[kc-1][jc] / (dZ0*(dZ1+dZ0)));
  const double d2psi_dRdZ = (equ.psi[kc+1][jc+1] - equ.psi[kc+1][jc-1]
                            - equ.psi[kc-1][jc+1] + equ.psi[kc-1][jc-1]) / (dR * dZ);

  const double invR = 1.0 / R;
  const double invR2 = invR * invR;

  B.derivatives_valid = true;   // legacy equilibrium psi-map derivatives
  B.axisymmetric_source = true;
  B.r = R;
  B.z = Z;
  B.br = -dpsi_dZ * invR;
  B.bz = dpsi_dR * invR;
  B.bt = equ.btf * equ.rtf * invR;

  B.dBr_dr = dpsi_dZ * invR2 - d2psi_dRdZ * invR;
  B.dBr_dz = -d2psi_dZ2 * invR;
  B.dBz_dr = -dpsi_dR * invR2 + d2psi_dR2 * invR;
  B.dBz_dz = d2psi_dRdZ * invR;
  B.dBt_dr = -equ.btf * equ.rtf * invR2;
  B.dBt_dz = 0.0;

  B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
  if (B.Bmag > 0.0) {
    B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr + B.bz*B.dBz_dr) / B.Bmag;
    B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz + B.bz*B.dBz_dz) / B.Bmag;
  }
  return B;
}

