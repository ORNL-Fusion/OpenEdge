/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_plasma_fields.h"
#include "fix_plasma_data.h"
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
  GRAD_NE_R, GRAD_NE_Z
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
  // compute ... plasma/fields ggroup plasma_data <fix_id> ...
  // compute ... plasma/fields ggroup constant [const args] ...
  // compute ... plasma/fields ggroup analytic [analytic args] ...
  //
  // File-based plasma input is now routed through fix plasma/data
  // (which owns the single-source-of-truth plasma.h5 + mesh/equilibrium
  // data). `compute plasma/fields ... file plasma.h5 ...` is DEPRECATED
  // and rejected below with a pointer to the migration. Declare
  //   fix pd plasma/data file plasma.h5
  //   compute cplasma plasma/fields all plasma_data pd ...
  // instead.
  int iarg = 3;
  if (iarg >= narg)
    error->all(FLERR,
      "compute plasma/fields requires mode: plasma_data, constant, or analytic");
  if (strcmp(arg[iarg],"file") == 0) {
    error->all(FLERR,
      "compute plasma/fields: the 'file' mode has been removed. "
      "Declare a `fix plasma/data file plasma.h5` instance and use "
      "`compute plasma/fields ... plasma_data <fix_id> ...` instead. "
      "This routes per-cell / per-particle plasma lookups through the "
      "EIRENE triangulation owned by fix plasma/data (single source of "
      "truth, no duplicate file reads, no regular-grid interpolation).");
    // keep compiler happy — MODE_FILE code paths are unreachable now
    input_mode = MODE_PLASMA_DATA;
  } else if (strcmp(arg[iarg],"plasma_data") == 0) {
    input_mode = MODE_PLASMA_DATA;
    iarg++;
    if (iarg >= narg)
      error->all(FLERR,"compute plasma/fields plasma_data mode needs fix ID");
    plasma_data_fix_id = std::string(arg[iarg++]);
  } else if (strcmp(arg[iarg],"constant") == 0) {
    input_mode = MODE_CONSTANT;
    iarg++;
  } else if (strcmp(arg[iarg],"analytic") == 0) {
    input_mode = MODE_ANALYTIC;
    iarg++;
  } else {
    error->all(FLERR,
      "compute plasma/fields mode must be 'plasma_data', 'constant', or 'analytic'");
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
  geom_arr = NULL;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"equilibrium") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"compute plasma/fields equilibrium requires a file path");
      equilibriumPath = std::string(arg[iarg+1]);
      has_equilibrium = 1;
      iarg += 2;
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
  if (geom_arr) {
    memory->destroy(geom_arr);
    geom_arr = NULL;
  }
}

/* ---------------------------------------------------------------------- */

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
  if (geom_arr) {
    memory->destroy(geom_arr);
    geom_arr = NULL;
  }
  nion_species = 0;
  ion_spec_index.clear();
  ion_charge_state_z.clear();
  ion_mass_amu.clear();
  ion_names.clear();
  ion_dens_grid.clear();
  ion_temp_grid.clear();
  ion_parr_flow_grid.clear();
  ion_parr_flow_r_grid.clear();
  ion_parr_flow_t_grid.clear();
  ion_parr_flow_z_grid.clear();


  // --- 1) load field source (file or constants) ------------------------------
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  memory->create(plasma_arr, ncells, "plasma/fields:plasma_arr");
  memory->create(mag_arr,    ncells, "plasma/fields:mag_arr");

  if (input_mode == MODE_PLASMA_DATA) {
    // Pull plasma data from fix plasma/data — no file reads
    int ifix = modify->find_fix(plasma_data_fix_id.c_str());
    if (ifix < 0) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "compute plasma/fields: fix '%s' not found",
               plasma_data_fix_id.c_str());
      error->all(FLERR, msg);
    }
    auto *pd = dynamic_cast<FixPlasmaData*>(modify->fix[ifix]);
    if (!pd)
      error->all(FLERR,
        "compute plasma/fields: plasma_data fix must be style plasma/data");

    // Build PlasmaFileData from the fix's flat arrays
    // (convert flat vectors to vector<vector<double>> format)
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

    // B-field from fix
    if (pd->has_bfield) {
      plasma_data.br = flat2vv(pd->br);
      plasma_data.bz = flat2vv(pd->bz);
      plasma_data.bt = flat2vv(pd->bt);
      plasma_data.has_bfield = true;
    }

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
      equ_data.r = pd->equ_r;
      equ_data.z = pd->equ_z;
      // Convert flat psirz to 2D
      equ_data.psi.resize(pd->equ_km, std::vector<double>(pd->equ_jm));
      for (int k = 0; k < pd->equ_km; k++)
        for (int j = 0; j < pd->equ_jm; j++)
          equ_data.psi[k][j] = pd->psirz[k * pd->equ_jm + j];
    }

    if (me == 0 && screen)
      fprintf(screen,
        "compute plasma/fields: using data from fix '%s' (gen=%d)\n",
        plasma_data_fix_id.c_str(), pd->generation);

  } else if (input_mode == MODE_FILE) {
    if (me == 0) {
      plasma_data = readPlasmaFileData(plasmaStatePath);
      if (!magneticFieldsPath.empty())
        magnetic_data = readMagneticFieldFileData(magneticFieldsPath);
      if (has_equilibrium) {
        equ_data = readEquilibriumFile(equilibriumPath);
      } else {
        // No explicit equilibrium <file> keyword — try to pick up an
        // embedded /equilibrium group from plasma.h5.
        if (readEquilibriumFromPlasmaH5(plasmaStatePath, equ_data)) {
          has_equilibrium = 1;
          if (screen)
            fprintf(screen,
              "compute plasma/fields: loaded embedded equilibrium "
              "from %s (%d x %d psi grid)\n",
              plasmaStatePath.c_str(), equ_data.jm, equ_data.km);
        }
      }
    }
    // Broadcast has_equilibrium so other ranks know whether to expect
    // broadcastEquilibriumData below.
    MPI_Bcast(&has_equilibrium, 1, MPI_INT, 0, world);
    broadcastPlasmaData(plasma_data);
    if (!magneticFieldsPath.empty())
      broadcastMagneticData(magnetic_data);
    if (has_equilibrium)
      broadcastEquilibriumData(equ_data);

    // Populate magnetic_data from the B-field embedded in plasma.h5
    // (br/bt/bz datasets the SOLPS / SOLEDGE3X / OEDGE converters write
    // directly into plasma.h5). Same r/z grid as plasma_data, so the
    // bilinear stencils line up.
    const bool plasma_has_B =
      !plasma_data.br.empty() && !plasma_data.bt.empty() &&
      !plasma_data.bz.empty();
    if (plasma_has_B) {
      magnetic_data.r = plasma_data.r;
      magnetic_data.z = plasma_data.z;
      magnetic_data.br = plasma_data.br;
      magnetic_data.bt = plasma_data.bt;
      magnetic_data.bz = plasma_data.bz;
    } else if (!has_equilibrium) {
      // No B-field source available — hard error so we don't run with
      // a silently-zero B field. Typical fix: regenerate plasma.h5 with
      // the current converter (which embeds br/bt/bz), or pass
      // `equilibrium <file.equ>` on the compute plasma/fields line.
      char msg[512];
      snprintf(msg, sizeof(msg),
        "compute plasma/fields: %s has no br/bt/bz datasets and no "
        "equilibrium file was provided. Re-run the converter to embed "
        "the B-field, or add `equilibrium <file>` to the compute line.",
        plasmaStatePath.c_str());
      error->all(FLERR, msg);
    } else if (comm->me == 0) {
      error->warning(FLERR,
        "compute plasma/fields: plasma.h5 has no embedded B-field; "
        "falling back to equilibrium-derived B. For best accuracy, "
        "regenerate plasma.h5 with a converter that writes br/bt/bz.");
    }
  }

  // --- Stencil computation and per-cell interpolation ---
  // (shared by MODE_FILE and MODE_PLASMA_DATA)
  if (input_mode == MODE_FILE || input_mode == MODE_PLASMA_DATA) {
    precomputeStencils(plasma_data.r, plasma_data.z, plasma_stencil);
    if (!magnetic_data.r.empty())
      precomputeStencils(magnetic_data.r, magnetic_data.z, magnetic_stencil);
    else
      magnetic_stencil.clear();

    for (int icell = 0; icell < ncells; ++icell) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit < 1)         continue;
      plasma_arr[icell] = bilinearInterpolationPlasma(icell, plasma_data);

      if (!magnetic_data.r.empty()) {
        mag_arr[icell] = bilinearInterpolationMagneticField(icell, magnetic_data);
      } else if (has_equilibrium) {
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

    // Optional species-resolved storage for later sputtering/emission models
    nion_species = plasma_data.ions_nspec;
    ion_spec_index = plasma_data.ion_spec_index;
    ion_charge_state_z = plasma_data.ion_charge_state_z;
    ion_mass_amu = plasma_data.ion_mass_amu;
    ion_names = plasma_data.ion_names;
    if (nion_species > 0) {
      const int nstore = ncells * nion_species;
      ion_dens_grid.assign(nstore, 0.0);
      ion_temp_grid.assign(nstore, 0.0);
      ion_parr_flow_grid.assign(nstore, 0.0);
      ion_parr_flow_r_grid.assign(nstore, 0.0);
      ion_parr_flow_t_grid.assign(nstore, 0.0);
      ion_parr_flow_z_grid.assign(nstore, 0.0);

      for (int icell = 0; icell < ncells; ++icell) {
        if (!(cinfo[icell].mask & groupbit)) continue;
        if (cells[icell].nsplit < 1)         continue;
        if (icell < 0 || icell >= static_cast<int>(plasma_stencil.size())) continue;
        const BilinearStencil &s = plasma_stencil[icell];
        if (!s.valid) continue;
        const int base = icell * nion_species;
        for (int is = 0; is < nion_species; ++is) {
          ion_dens_grid[base + is] =
            interpField3DFlat(plasma_data.ions_dens, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
          ion_temp_grid[base + is] =
            interpField3DFlat(plasma_data.ions_temp, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
          ion_parr_flow_grid[base + is] =
            interpField3DFlat(plasma_data.ions_parr_flow, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
          ion_parr_flow_r_grid[base + is] =
            interpField3DFlat(plasma_data.ions_parr_flow_r, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
          ion_parr_flow_t_grid[base + is] =
            interpField3DFlat(plasma_data.ions_parr_flow_t, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
          ion_parr_flow_z_grid[base + is] =
            interpField3DFlat(plasma_data.ions_parr_flow_z, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        }
      }
    }

    // --- Equilibrium-based magnetic geometry ---
    if (has_equilibrium) {
      // Only re-read if in file mode and not already loaded
      if (input_mode == MODE_FILE && !equilibriumPath.empty()) {
        if (me == 0)
          equ_data = readEquilibriumFile(equilibriumPath);
        broadcastEquilibriumData(equ_data);
      }
      memory->create(geom_arr, ncells, "plasma/fields:geom_arr");
      for (int icell = 0; icell < ncells; ++icell) {
        MagneticGeometry g{};
        g.kappa[0] = g.kappa[1] = g.kappa[2] = 0.0;
        g.curl_b[0] = g.curl_b[1] = g.curl_b[2] = 0.0;
        g.gradBmag[0] = g.gradBmag[1] = g.gradBmag[2] = 0.0;
        g.Bmag = 0.0;
        if (!(cinfo[icell].mask & groupbit)) { geom_arr[icell] = g; continue; }
        if (cells[icell].nsplit < 1)         { geom_arr[icell] = g; continue; }
        computeMagneticGeometry(icell, equ_data, g);
        geom_arr[icell] = g;
      }
      if (me == 0)
        printf("  Equilibrium magnetic geometry precomputed for %d cells\n", ncells);
    }

  } else if (input_mode == MODE_CONSTANT) {
    plasma_stencil.clear();
    magnetic_stencil.clear();
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
    magnetic_stencil.clear();
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
}


/* ----------------------------------------------------------------------
   Reload plasma background from the current plasmaStatePath.
   Re-reads HDF5, broadcasts, re-interpolates onto all grid cells.
   Called by the coupling driver between time chunks to refresh the
   background plasma without reinitializing the entire compute.
------------------------------------------------------------------------- */

void ComputePlasmaFields::reload_plasma()
{
  if (input_mode != MODE_FILE) return;

  const int me     = comm->me;
  const int ncells = grid->nlocal;
  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;

  // Re-read plasma file
  if (me == 0) plasma_data = readPlasmaFileData(plasmaStatePath);
  broadcastPlasmaData(plasma_data);

  // Re-read magnetic field if provided separately (legacy bfield.h5 path).
  if (!magneticFieldsPath.empty()) {
    if (me == 0) magnetic_data = readMagneticFieldFileData(magneticFieldsPath);
    broadcastMagneticData(magnetic_data);
  } else if (!plasma_data.br.empty() && !plasma_data.bt.empty() &&
             !plasma_data.bz.empty()) {
    // Refresh embedded B-field from the (possibly new) plasma.h5.
    magnetic_data.r = plasma_data.r;
    magnetic_data.z = plasma_data.z;
    magnetic_data.br = plasma_data.br;
    magnetic_data.bt = plasma_data.bt;
    magnetic_data.bz = plasma_data.bz;
  }

  // Recompute stencils (grid may not have changed, but R/Z grids in file might)
  precomputeStencils(plasma_data.r, plasma_data.z, plasma_stencil);
  if (!magnetic_data.r.empty())
    precomputeStencils(magnetic_data.r, magnetic_data.z, magnetic_stencil);

  // Re-interpolate onto grid cells
  for (int icell = 0; icell < ncells; ++icell) {
    if (!(cinfo[icell].mask & groupbit)) continue;
    if (cells[icell].nsplit < 1)         continue;
    plasma_arr[icell] = bilinearInterpolationPlasma(icell, plasma_data);

    if (!magnetic_data.r.empty())
      mag_arr[icell] = bilinearInterpolationMagneticField(icell, magnetic_data);
  }

  // Update multi-ion species data
  nion_species = plasma_data.ions_nspec;
  ion_spec_index = plasma_data.ion_spec_index;
  ion_charge_state_z = plasma_data.ion_charge_state_z;
  ion_mass_amu = plasma_data.ion_mass_amu;
  ion_names = plasma_data.ion_names;
  if (nion_species > 0) {
    const int nstore = ncells * nion_species;
    ion_dens_grid.assign(nstore, 0.0);
    ion_temp_grid.assign(nstore, 0.0);
    ion_parr_flow_grid.assign(nstore, 0.0);
    ion_parr_flow_r_grid.assign(nstore, 0.0);
    ion_parr_flow_t_grid.assign(nstore, 0.0);
    ion_parr_flow_z_grid.assign(nstore, 0.0);

    for (int icell = 0; icell < ncells; ++icell) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit < 1)         continue;
      if (icell < 0 || icell >= static_cast<int>(plasma_stencil.size())) continue;
      const BilinearStencil &s = plasma_stencil[icell];
      if (!s.valid) continue;
      const int base = icell * nion_species;
      for (int is = 0; is < nion_species; ++is) {
        ion_dens_grid[base + is] =
          interpField3DFlat(plasma_data.ions_dens, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        ion_temp_grid[base + is] =
          interpField3DFlat(plasma_data.ions_temp, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        ion_parr_flow_grid[base + is] =
          interpField3DFlat(plasma_data.ions_parr_flow, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        ion_parr_flow_r_grid[base + is] =
          interpField3DFlat(plasma_data.ions_parr_flow_r, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        ion_parr_flow_t_grid[base + is] =
          interpField3DFlat(plasma_data.ions_parr_flow_t, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
        ion_parr_flow_z_grid[base + is] =
          interpField3DFlat(plasma_data.ions_parr_flow_z, is, plasma_data.ions_nz, plasma_data.ions_nr, s);
      }
    }
  }

  if (me == 0)
    printf("  Plasma background reloaded from %s (%d cells)\n",
           plasmaStatePath.c_str(), ncells);
}

/* ----------------------------------------------------------------------
   Reload from a new file path (updates plasmaStatePath before reloading)
------------------------------------------------------------------------- */

void ComputePlasmaFields::reload_plasma(const std::string &new_plasma_path)
{
  plasmaStatePath = new_plasma_path;
  reload_plasma();
}

void ComputePlasmaFields::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;
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
    if (input_mode == MODE_PLASMA_DATA && plasma_data.has_mesh &&
        !plasma_data.mesh_e_r.empty()) {
      // Look up the mesh cell for this SPARTA cell centroid and read
      // stored E components directly.
      const Grid::ChildCell &cell = cells[icell];
      const double xyz_c[3] = {
        0.5 * (cell.lo[0] + cell.hi[0]),
        0.5 * (cell.lo[1] + cell.hi[1]),
        (dim == 3) ? 0.5 * (cell.lo[2] + cell.hi[2]) : 0.0
      };
      double Rc, Zc;
      OpenEdge::sparta_to_RZ(xyz_c, dim, domain->axisymmetric, Rc, Zc);
      const int tri = findNearestMappedTriangle(plasma_data, Rc, Zc, 0.1);
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
    if (dim == 3) {
      const Grid::ChildCell &cell = cells[icell];
      const double x = 0.5 * (cell.lo[0] + cell.hi[0]);
      const double y = 0.5 * (cell.lo[1] + cell.hi[1]);
      phi = std::atan2(y, x);
    }
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
  if (nvalue == 1) {
    memory->create(vector_grid, nglocal, "plasma/fields:vector_grid");
  } else {
    memory->create(array_grid, nglocal, nvalue, "plasma/fields:array_grid");
  }
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
/* ----------------------------------------------------------------------
   Read plasma data from HDF5 file
------------------------------------------------------------------------- */
PlasmaFileData ComputePlasmaFields::readPlasmaFileData(const std::string& filePath) {
    printf("Reading plasma data from file: %s\n", filePath.c_str());
    PlasmaFileData data;

    try {
        // Keep HDF5 from printing internal diagnostics to stderr.
        // We handle errors explicitly via exceptions and return codes.
        H5::Exception::dontPrint();
        H5::H5File file(filePath, H5F_ACC_RDONLY);

        // Utility to read 1D dataset
        auto read1D = [&](const std::string& name) -> std::vector<double> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dim;
            space.getSimpleExtentDims(&dim);
            std::vector<double> vec(dim);
            ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
            return vec;
        };

        // Utility to test whether dataset/path exists
        auto hasDataset = [&](const std::string& name) -> bool {
            htri_t exists = 0;
            H5E_BEGIN_TRY {
              exists = H5Oexists_by_name(file.getId(), name.c_str(), H5P_DEFAULT);
            } H5E_END_TRY;
            return exists > 0;
        };

        // Optional regular (R, Z) grid. Current SOLPS/SOLEDGE3X/OEDGE
        // converters emit mesh-only plasma.h5 — the regular grid at top
        // level is absent. Keep this path alive for older plasma.h5
        // files; consumers that need per-cell plasma (compute grid
        // nrho, fix volume/chem/adas) route through mesh_cell lookup instead.
        size_t nr = 0;
        size_t nz = 0;
        if (hasDataset("r") && hasDataset("z")) {
            data.r = read1D("r");
            data.z = read1D("z");
            nr = data.r.size();
            nz = data.z.size();
        }

        // Utility to read 2D dataset with shape validation
        auto read2D = [&](const std::string& name) -> std::vector<std::vector<double>> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dims[2];
            space.getSimpleExtentDims(dims);

            if (dims[0] != nz || dims[1] != nr) {
                throw std::runtime_error("Dataset '" + name + "' shape mismatch: expected " +
                                         std::to_string(nz) + " x " + std::to_string(nr) +
                                         ", got " + std::to_string(dims[0]) + " x " + std::to_string(dims[1]));
            }

            std::vector<double> raw(dims[0] * dims[1]);
            ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);

            std::vector<std::vector<double>> grid(nz, std::vector<double>(nr));
            for (size_t i = 0; i < nz; ++i) {
                for (size_t j = 0; j < nr; ++j) {
                    grid[i][j] = raw[i * nr + j];
                }
            }
            return grid;
        };

        auto read1DInt = [&](const std::string& name) -> std::vector<int> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dim;
            space.getSimpleExtentDims(&dim);
            std::vector<int> vec(dim);
            ds.read(vec.data(), H5::PredType::NATIVE_INT);
            return vec;
        };

        auto read1DString = [&](const std::string& name) -> std::vector<std::string> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dim;
            space.getSimpleExtentDims(&dim);
            std::vector<std::string> vec(dim);
            H5::StrType stype = ds.getStrType();
            std::vector<char*> rdata(dim, nullptr);
            ds.read(rdata.data(), stype);
            for (hsize_t i = 0; i < dim; ++i) {
              if (rdata[i]) vec[i] = std::string(rdata[i]);
            }
            return vec;
        };

        auto read3D = [&](const std::string& name, int &ns_out, int &nz_out, int &nr_out) -> std::vector<double> {
            H5::DataSet ds = file.openDataSet(name);
            H5::DataSpace space = ds.getSpace();
            hsize_t dims[3];
            space.getSimpleExtentDims(dims);

            const size_t ns = dims[0];
            if (dims[1] != nz || dims[2] != nr) {
                throw std::runtime_error("Dataset '" + name + "' shape mismatch: expected (*, " +
                                         std::to_string(nz) + ", " + std::to_string(nr) +
                                         "), got (" + std::to_string(dims[0]) + ", " +
                                         std::to_string(dims[1]) + ", " + std::to_string(dims[2]) + ")");
            }

            std::vector<double> raw(ns * nz * nr);
            ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);
            ns_out = static_cast<int>(ns);
            nz_out = static_cast<int>(nz);
            nr_out = static_cast<int>(nr);
            return raw;
        };

        // Legacy regular-grid plasma fields. Loaded only if the file
        // still ships them (older plasma.h5 without mesh-only layout).
        // New-style plasma.h5 from convert_solps_plasma.py et al. skips
        // this block entirely; per-cell plasma is read from /mesh/*.
        auto read2D_optional = [&](const std::string &name) {
          return hasDataset(name) ? read2D(name)
                                   : std::vector<std::vector<double>>();
        };
        if (nr > 0 && nz > 0) {
            data.dens_e        = read2D_optional("dens_e");
            data.temp_e        = read2D_optional("temp_e");
            data.dens_i        = read2D_optional("dens_i");
            data.temp_i        = read2D_optional("temp_i");

            data.parr_flow     = read2D_optional("parr_flow");
            data.parr_flow_r   = read2D_optional("parr_flow_r");
            data.parr_flow_t   = read2D_optional("parr_flow_t");
            data.parr_flow_z   = read2D_optional("parr_flow_z");

            data.grad_temp_e_r = read2D_optional("grad_te_r");
            data.grad_temp_e_t = read2D_optional("grad_te_t");
            data.grad_temp_e_z = read2D_optional("grad_te_z");

            data.grad_temp_i_r = read2D_optional("grad_ti_r");
            data.grad_temp_i_t = read2D_optional("grad_ti_t");
            data.grad_temp_i_z = read2D_optional("grad_ti_z");

            // Optional heat flux
            if (hasDataset("q_mag")) {
              data.q_mag = read2D("q_mag");
              data.has_qmag = true;
              printf("  Loaded q_mag from %s\n", filePath.c_str());
            }
        }

        // Optional multi-ion extension
        if (hasDataset("ion_species/spec_index")) {
          data.ion_spec_index = read1DInt("ion_species/spec_index");
        }
        if (hasDataset("ion_species/charge_state_z")) {
          data.ion_charge_state_z = read1DInt("ion_species/charge_state_z");
        }
        if (hasDataset("ion_species/mass_amu")) {
          data.ion_mass_amu = read1D("ion_species/mass_amu");
        }
        if (hasDataset("ion_species/names")) {
          data.ion_names = read1DString("ion_species/names");
        }

        // Legacy per-species 3D regular-grid arrays. Only load when the
        // file carries the regular grid (older plasma.h5). Mesh-only
        // files store per-species plasma under /mesh/ions/.
        if (nr > 0 && nz > 0) {
          if (hasDataset("ions/dens")) {
            data.ions_dens = read3D("ions/dens", data.ions_nspec, data.ions_nz, data.ions_nr);
          }
          auto check3DShape = [&](int ns, int nzf, int nrf, const std::string &name) {
            if (data.ions_nspec == 0) {
              data.ions_nspec = ns;
              data.ions_nz = nzf;
              data.ions_nr = nrf;
              return;
            }
            if (ns != data.ions_nspec || nzf != data.ions_nz || nrf != data.ions_nr) {
              throw std::runtime_error("Dataset '" + name + "' 3D shape mismatch with ions/dens");
            }
          };
          if (hasDataset("ions/temp")) {
            int ns = 0, nzf = 0, nrf = 0;
            data.ions_temp = read3D("ions/temp", ns, nzf, nrf);
            check3DShape(ns, nzf, nrf, "ions/temp");
          }
          if (hasDataset("ions/parr_flow")) {
            int ns = 0, nzf = 0, nrf = 0;
            data.ions_parr_flow = read3D("ions/parr_flow", ns, nzf, nrf);
            check3DShape(ns, nzf, nrf, "ions/parr_flow");
          }
          if (hasDataset("ions/parr_flow_r")) {
            int ns = 0, nzf = 0, nrf = 0;
            data.ions_parr_flow_r = read3D("ions/parr_flow_r", ns, nzf, nrf);
            check3DShape(ns, nzf, nrf, "ions/parr_flow_r");
          }
          if (hasDataset("ions/parr_flow_t")) {
            int ns = 0, nzf = 0, nrf = 0;
            data.ions_parr_flow_t = read3D("ions/parr_flow_t", ns, nzf, nrf);
            check3DShape(ns, nzf, nrf, "ions/parr_flow_t");
          }
          if (hasDataset("ions/parr_flow_z")) {
            int ns = 0, nzf = 0, nrf = 0;
            data.ions_parr_flow_z = read3D("ions/parr_flow_z", ns, nzf, nrf);
            check3DShape(ns, nzf, nrf, "ions/parr_flow_z");
          }
        }

        // Optional mesh triangulation (SOLPS cell data)
        if (hasDataset("mesh/triangles") && hasDataset("mesh/vtx_r")) {
          data.mesh_vtx_r = read1D("mesh/vtx_r");
          data.mesh_vtx_z = read1D("mesh/vtx_z");
          data.mesh_nvtx = static_cast<int>(data.mesh_vtx_r.size());

          {
            H5::DataSet ds = file.openDataSet("mesh/triangles");
            H5::DataSpace sp = ds.getSpace();
            hsize_t dims[2]; sp.getSimpleExtentDims(dims);
            data.mesh_ntri = static_cast<int>(dims[0]);
            data.mesh_tri.resize(data.mesh_ntri * 3);
            ds.read(data.mesh_tri.data(), H5::PredType::NATIVE_INT);
          }
          {
            H5::DataSet ds = file.openDataSet("mesh/cell_index");
            H5::DataSpace sp = ds.getSpace();
            hsize_t dim; sp.getSimpleExtentDims(&dim);
            data.mesh_cell_idx.resize(dim);
            ds.read(data.mesh_cell_idx.data(), H5::PredType::NATIVE_INT);
          }

          auto read1D_mesh = [&](const std::string &name) -> std::vector<double> {
            if (!hasDataset(name)) return {};
            return read1D(name);
          };
          data.mesh_ne = read1D_mesh("mesh/dens_e");
          data.mesh_te = read1D_mesh("mesh/temp_e");
          data.mesh_ti = read1D_mesh("mesh/temp_i");
          data.mesh_ni = read1D_mesh("mesh/dens_i");
          data.mesh_upar = read1D_mesh("mesh/parr_flow");
          data.mesh_ncell = data.mesh_ne.empty() ? 0 : static_cast<int>(data.mesh_ne.size());

          // Build bounding boxes for triangle search
          data.mesh_tri_rmin.resize(data.mesh_ntri);
          data.mesh_tri_rmax.resize(data.mesh_ntri);
          data.mesh_tri_zmin.resize(data.mesh_ntri);
          data.mesh_tri_zmax.resize(data.mesh_ntri);
          for (int t = 0; t < data.mesh_ntri; t++) {
            const int v0 = data.mesh_tri[t*3+0], v1 = data.mesh_tri[t*3+1], v2 = data.mesh_tri[t*3+2];
            const double r0 = data.mesh_vtx_r[v0], r1 = data.mesh_vtx_r[v1], r2 = data.mesh_vtx_r[v2];
            const double z0 = data.mesh_vtx_z[v0], z1 = data.mesh_vtx_z[v1], z2 = data.mesh_vtx_z[v2];
            data.mesh_tri_rmin[t] = std::min({r0,r1,r2});
            data.mesh_tri_rmax[t] = std::max({r0,r1,r2});
            data.mesh_tri_zmin[t] = std::min({z0,z1,z2});
            data.mesh_tri_zmax[t] = std::max({z0,z1,z2});
          }

          // Precompute centroids for nearest-neighbor fallback
          for (int t = 0; t < data.mesh_ntri; t++) {
            if (data.mesh_cell_idx[t] < 0) continue;
            const int v0 = data.mesh_tri[t*3+0], v1 = data.mesh_tri[t*3+1], v2 = data.mesh_tri[t*3+2];
            data.mapped_cr.push_back((data.mesh_vtx_r[v0]+data.mesh_vtx_r[v1]+data.mesh_vtx_r[v2])/3.0);
            data.mapped_cz.push_back((data.mesh_vtx_z[v0]+data.mesh_vtx_z[v1]+data.mesh_vtx_z[v2])/3.0);
            data.mapped_idx.push_back(t);
          }

          data.has_mesh = true;
          data.buildSpatialHash();
          printf("  Loaded mesh: %d triangles, %d vertices, %d cells (%d mapped)\n",
                 data.mesh_ntri, data.mesh_nvtx, data.mesh_ncell,
                 static_cast<int>(data.mapped_idx.size()));
        }

    } catch (const H5::Exception& e) {
        fprintf(stderr, "HDF5 error: %s\n", e.getCDetailMsg());
        throw;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        throw;
    }

    return data;
}

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
void ComputePlasmaFields::broadcastPlasmaData(PlasmaFileData& data) {
    int me = comm->me;

    // Broadcast sizes of 1D vectors (r and z)
    int r_size = data.r.size();
    int z_size = data.z.size();
    MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
    MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

    // Resize vectors on non-root processes
    if (me != 0) {
        data.r.resize(r_size);
        data.z.resize(z_size);
    }

    // Broadcast 1D vector data
    MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
    MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

    // Broadcast 2D vectors (dens_e, temp_e, dens_i, temp_i, parr_flow, grad_temp_e, grad_temp_i)
    auto broadcast2DVector = [&](std::vector<std::vector<double>>& vec) {
        int dim1 = vec.size();
        int dim2 = dim1 ? vec[0].size() : 0;

        MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
        MPI_Bcast(&dim2, 1, MPI_INT, 0, world);

        // Resize the outer vector and each inner vector on non-root processes
        if (me != 0) {
            vec.resize(dim1, std::vector<double>(dim2));
        }

        for (int i = 0; i < dim1; ++i) {
            MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
        }
    };

    broadcast2DVector(data.dens_e);
    broadcast2DVector(data.temp_e);
    broadcast2DVector(data.dens_i);
    broadcast2DVector(data.temp_i);

    broadcast2DVector(data.parr_flow);
    broadcast2DVector(data.parr_flow_r);
    broadcast2DVector(data.parr_flow_t);
    broadcast2DVector(data.parr_flow_z);

    broadcast2DVector(data.grad_temp_e_r);
    broadcast2DVector(data.grad_temp_e_t);
    broadcast2DVector(data.grad_temp_e_z);

    broadcast2DVector(data.grad_temp_i_r);
    broadcast2DVector(data.grad_temp_i_t);
    broadcast2DVector(data.grad_temp_i_z);

    MPI_Bcast(&data.has_qmag, 1, MPI_C_BOOL, 0, world);
    if (data.has_qmag) broadcast2DVector(data.q_mag);

    auto broadcast1DInt = [&](std::vector<int>& vec) {
      int n = static_cast<int>(vec.size());
      MPI_Bcast(&n, 1, MPI_INT, 0, world);
      if (me != 0) vec.resize(n);
      if (n > 0) MPI_Bcast(vec.data(), n, MPI_INT, 0, world);
    };
    auto broadcast1DDouble = [&](std::vector<double>& vec) {
      int n = static_cast<int>(vec.size());
      MPI_Bcast(&n, 1, MPI_INT, 0, world);
      if (me != 0) vec.resize(n);
      if (n > 0) MPI_Bcast(vec.data(), n, MPI_DOUBLE, 0, world);
    };
    auto broadcast1DString = [&](std::vector<std::string>& vec) {
      int n = static_cast<int>(vec.size());
      MPI_Bcast(&n, 1, MPI_INT, 0, world);
      if (me != 0) vec.resize(n);
      for (int i = 0; i < n; ++i) {
        int len = me == 0 ? static_cast<int>(vec[i].size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, world);
        if (me != 0) vec[i].assign(len, '\0');
        if (len > 0) MPI_Bcast(vec[i].data(), len, MPI_CHAR, 0, world);
      }
    };
    auto broadcast3DFlat = [&](std::vector<double>& vec, int &d0, int &d1, int &d2) {
      MPI_Bcast(&d0, 1, MPI_INT, 0, world);
      MPI_Bcast(&d1, 1, MPI_INT, 0, world);
      MPI_Bcast(&d2, 1, MPI_INT, 0, world);
      const int n = d0 * d1 * d2;
      if (me != 0) vec.resize(n);
      if (n > 0) MPI_Bcast(vec.data(), n, MPI_DOUBLE, 0, world);
    };

    broadcast1DInt(data.ion_spec_index);
    broadcast1DInt(data.ion_charge_state_z);
    broadcast1DDouble(data.ion_mass_amu);
    broadcast1DString(data.ion_names);
    broadcast3DFlat(data.ions_dens, data.ions_nspec, data.ions_nz, data.ions_nr);
    {
      int ns = data.ions_nspec, nz = data.ions_nz, nr = data.ions_nr;
      broadcast3DFlat(data.ions_temp, ns, nz, nr);
      broadcast3DFlat(data.ions_parr_flow, ns, nz, nr);
      broadcast3DFlat(data.ions_parr_flow_r, ns, nz, nr);
      broadcast3DFlat(data.ions_parr_flow_t, ns, nz, nr);
      broadcast3DFlat(data.ions_parr_flow_z, ns, nz, nr);
    }

    // Broadcast mesh data
    int hm = data.has_mesh ? 1 : 0;
    MPI_Bcast(&hm, 1, MPI_INT, 0, world);
    data.has_mesh = (hm != 0);
    if (data.has_mesh) {
      MPI_Bcast(&data.mesh_nvtx, 1, MPI_INT, 0, world);
      MPI_Bcast(&data.mesh_ntri, 1, MPI_INT, 0, world);
      MPI_Bcast(&data.mesh_ncell, 1, MPI_INT, 0, world);
      int nmapped = static_cast<int>(data.mapped_idx.size());
      MPI_Bcast(&nmapped, 1, MPI_INT, 0, world);
      if (me != 0) {
        data.mesh_vtx_r.resize(data.mesh_nvtx);
        data.mesh_vtx_z.resize(data.mesh_nvtx);
        data.mesh_tri.resize(data.mesh_ntri * 3);
        data.mesh_cell_idx.resize(data.mesh_ntri);
        data.mesh_ne.resize(data.mesh_ncell);
        data.mesh_te.resize(data.mesh_ncell);
        data.mesh_ti.resize(data.mesh_ncell);
        data.mesh_ni.resize(data.mesh_ncell);
        data.mesh_upar.resize(data.mesh_ncell);
        data.mesh_tri_rmin.resize(data.mesh_ntri);
        data.mesh_tri_rmax.resize(data.mesh_ntri);
        data.mesh_tri_zmin.resize(data.mesh_ntri);
        data.mesh_tri_zmax.resize(data.mesh_ntri);
        data.mapped_cr.resize(nmapped);
        data.mapped_cz.resize(nmapped);
        data.mapped_idx.resize(nmapped);
      }
      MPI_Bcast(data.mesh_vtx_r.data(), data.mesh_nvtx, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_vtx_z.data(), data.mesh_nvtx, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_tri.data(), data.mesh_ntri*3, MPI_INT, 0, world);
      MPI_Bcast(data.mesh_cell_idx.data(), data.mesh_ntri, MPI_INT, 0, world);
      MPI_Bcast(data.mesh_ne.data(), data.mesh_ncell, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_te.data(), data.mesh_ncell, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_ti.data(), data.mesh_ncell, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_ni.data(), data.mesh_ncell, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_upar.data(), data.mesh_ncell, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_tri_rmin.data(), data.mesh_ntri, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_tri_rmax.data(), data.mesh_ntri, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_tri_zmin.data(), data.mesh_ntri, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mesh_tri_zmax.data(), data.mesh_ntri, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mapped_cr.data(), nmapped, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mapped_cz.data(), nmapped, MPI_DOUBLE, 0, world);
      MPI_Bcast(data.mapped_idx.data(), nmapped, MPI_INT, 0, world);
      // Rebuild spatial hash on non-root ranks
      if (me != 0) data.buildSpatialHash();
    }
}


/*----------------------------------------------------------------------
   broadcast magnetic field data
------------------------------------------------------------------------- */

void ComputePlasmaFields::broadcastMagneticData(MagneticFieldFileData& data) {
  int me = comm->me;

  // Broadcast sizes of 1D vectors (e.g., r and z for the magnetic field)
  int r_size = data.r.size();
  int z_size = data.z.size();
  MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

  // Resize vectors on non-root processes
  if (me != 0) {
      data.r.resize(r_size);
      data.z.resize(z_size);
  }

  // Broadcast 1D vector data (r and z)
  MPI_Bcast(data.r.data(), r_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.z.data(), z_size, MPI_DOUBLE, 0, world);

  // Broadcast 2D vectors (e.g., br, bt, bz)
  auto broadcast2DVector = [&](std::vector<std::vector<double>>& vec) {
      int dim1 = vec.size();
      int dim2 = dim1 ? vec[0].size() : 0;

      MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
      MPI_Bcast(&dim2, 1, MPI_INT, 0, world);

      // Resize the outer vector and each inner vector on non-root processes
      if (me != 0) {
          vec.resize(dim1, std::vector<double>(dim2));
      }

      for (int i = 0; i < dim1; ++i) {
          MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
      }
  };

  // Broadcast the magnetic field components
  broadcast2DVector(data.br);
  broadcast2DVector(data.bt);
  broadcast2DVector(data.bz);
}

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
  OpenEdge::sparta_to_RZ(xc, dim, domain->axisymmetric, r, z);
  return meshLookupPlasmaAtPoint(data, r, z, P);
}


/*---------------------------------
  Bilinear interpolation plasma
-----------------------------------*/

MagneticFieldFileDataParams
ComputePlasmaFields::bilinearInterpolationMagneticField(
    int icell, const MagneticFieldFileData &data)
{
  MagneticFieldFileDataParams B{};  // all zeros
  if (icell < 0 || icell >= static_cast<int>(magnetic_stencil.size())) return B;
  const BilinearStencil &s = magnetic_stencil[icell];
  if (!s.valid) return B;

  B.br = interpField2D(data.br, s);
  B.bt = interpField2D(data.bt, s);
  B.bz = interpField2D(data.bz, s);

  // Compute gradients of each B component using the existing stencil
  gradField2D(data.br, s, B.dBr_dr, B.dBr_dz);
  gradField2D(data.bt, s, B.dBt_dr, B.dBt_dz);
  gradField2D(data.bz, s, B.dBz_dr, B.dBz_dz);

  // |B| and grad(|B|) via chain rule
  B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
  if (B.Bmag > 0.0) {
    B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr + B.bz*B.dBz_dr) / B.Bmag;
    B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz + B.bz*B.dBz_dz) / B.Bmag;
  } else {
    B.dBmag_dr = 0.0;
    B.dBmag_dz = 0.0;
  }
  return B;
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
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, r, z);

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
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, r, z);

  bool used_mesh = false;
  if (plasma_data.has_mesh)
    used_mesh = meshLookupPlasmaAtPoint(plasma_data, r, z, P);

  // MODE_FILE: build stencil on-the-fly and interpolate
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
    OpenEdge::sparta_to_RZ(xyz, domain->dimension, domain->axisymmetric, R, Z);
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
    const double xyz[3]) const
{
  MagneticFieldFileDataParams B{};

  if (input_mode == MODE_CONSTANT || input_mode == MODE_ANALYTIC) {
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
  if (!plasma_data.mesh_tri_br.empty()) {
    double R, Z;
    OpenEdge::sparta_to_RZ(xyz, domain->dimension, domain->axisymmetric, R, Z);
    const int tri = findMeshTriangle(plasma_data, R, Z);
    if (tri >= 0 && tri < static_cast<int>(plasma_data.mesh_tri_br.size())) {
      B.br = plasma_data.mesh_tri_br[tri];
      B.bz = plasma_data.mesh_tri_bz[tri];
      B.bt = plasma_data.mesh_tri_bt[tri];
      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
      return B;
    }
    // outside mesh footprint: fall through to grid / equ branches.
  }

  if (!magnetic_data.r.empty() && !magnetic_data.z.empty()) {
    BilinearStencil s = makeStencilAtPoint(xyz, magnetic_data.r, magnetic_data.z);
    if (!s.valid) return B;

    B.br = interpField2D(magnetic_data.br, s);
    B.bt = interpField2D(magnetic_data.bt, s);
    B.bz = interpField2D(magnetic_data.bz, s);
    gradField2D(magnetic_data.br, s, B.dBr_dr, B.dBr_dz);
    gradField2D(magnetic_data.bt, s, B.dBt_dr, B.dBt_dz);
    gradField2D(magnetic_data.bz, s, B.dBz_dr, B.dBz_dz);
    B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
    if (B.Bmag > 0.0) {
      B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr + B.bz*B.dBz_dr) / B.Bmag;
      B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz + B.bz*B.dBz_dz) / B.Bmag;
    }
    return B;
  }

  if (!has_equilibrium || equ_data.jm < 3 || equ_data.km < 3) return B;

  const int dim = domain->dimension;
  double R, Z;
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, R, Z);
  if (R < 1.0e-10) return B;

  const EquilibriumData &equ = equ_data;
  const int jm = equ.jm;
  const int km = equ.km;
  const double dr = equ.r[1] - equ.r[0];
  const double dz = equ.z[1] - equ.z[0];
  if (dr <= 0.0 || dz <= 0.0) return B;

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

/* ----------------------------------------------------------------------
   read magnetic field data from file
------------------------------------------------------------------------- */
MagneticFieldFileData ComputePlasmaFields::readMagneticFieldFileData(const std::string& filePath) {
  printf("Reading magnetic field data from file: %s\n", filePath.c_str());
    MagneticFieldFileData data; // Initialize an empty MagneticFieldFileData struct
    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        
        auto read1DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(1);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<double> data(dims[0]);
            ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
            return data;
        };
        
        auto read2DDataSet = [&file](const std::string& datasetPath) {
            H5::DataSet ds = file.openDataSet(datasetPath);
            H5::DataSpace space = ds.getSpace();
            std::vector<hsize_t> dims(2);
            space.getSimpleExtentDims(dims.data(), NULL);
            
            std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
            std::vector<double> rawData(dims[0] * dims[1]);
            ds.read(rawData.data(), H5::PredType::NATIVE_DOUBLE);
            
            for (hsize_t i = 0; i < dims[0]; ++i) {
                for (hsize_t j = 0; j < dims[1]; ++j) {
                    data[i][j] = rawData[i * dims[1] + j];
                }
            }
            return data;
        };
        
        // Read the required datasets
        data.r = read1DDataSet("r");
        data.z = read1DDataSet("z");
        data.br = read2DDataSet("br");
        data.bz = read2DDataSet("bz");
        data.bt = read2DDataSet("bt");
        file.close();
    } catch (const H5::Exception& e) {
        printf("Error reading magnetic field file file: %s\n", e.getCDetailMsg());
        throw;  // Re-throw the exception to handle it outside
    } catch (const std::exception& e) {
        printf("Error: %s\n", e.what());
        throw;
    }
    printf("Finished reading magnetic field data from file: %s\n", filePath.c_str());
    return data;
}

/* ----------------------------------------------------------------------
   Read equilibrium (.equ) file: SOLPS format with ψ(R,Z) grid
------------------------------------------------------------------------- */

EquilibriumData ComputePlasmaFields::readEquilibriumFile(const std::string &path) {
  printf("Reading equilibrium data from file: %s\n", path.c_str());
  EquilibriumData equ;

  std::ifstream fin(path);
  if (!fin.is_open()) {
    char msg[256];
    snprintf(msg, sizeof(msg), "Cannot open equilibrium file: %s", path.c_str());
    error->one(FLERR, msg);
  }

  auto readDoubles = [&](int count) -> std::vector<double> {
    std::vector<double> vals;
    vals.reserve(count);
    std::string line;
    while (static_cast<int>(vals.size()) < count && std::getline(fin, line)) {
      std::istringstream iss(line);
      double v;
      while (iss >> v) {
        vals.push_back(v);
        if (static_cast<int>(vals.size()) >= count) break;
      }
    }
    return vals;
  };

  std::string line;
  while (std::getline(fin, line)) {
    if (line.find("jm") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ.jm;
    }
    if (line.find("km") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      if (line.find("km") < line.find("=")) {
        std::istringstream iss(line.substr(line.find("=") + 1));
        iss >> equ.km;
      }
    }
    if (line.find("psib") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ.psib;
    }
    if (line.find("btf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ.btf;
    }
    if (line.find("rtf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ.rtf;
    }
    if (line.find("r(1:jm)") != std::string::npos) break;
  }

  if (equ.jm <= 0 || equ.km <= 0)
    error->one(FLERR, "Equilibrium file: failed to parse jm/km");

  equ.r = readDoubles(equ.jm);
  if (static_cast<int>(equ.r.size()) != equ.jm)
    error->one(FLERR, "Equilibrium file: incomplete r array");

  while (std::getline(fin, line)) {
    if (line.find("z(1:km)") != std::string::npos) break;
  }

  equ.z = readDoubles(equ.km);
  if (static_cast<int>(equ.z.size()) != equ.km)
    error->one(FLERR, "Equilibrium file: incomplete z array");

  while (std::getline(fin, line)) {
    if (line.find("psi(j,k)") != std::string::npos) break;
  }

  int total = equ.jm * equ.km;
  std::vector<double> psi_flat = readDoubles(total);
  if (static_cast<int>(psi_flat.size()) != total)
    error->one(FLERR, "Equilibrium file: incomplete psi array");

  equ.psi.resize(equ.km, std::vector<double>(equ.jm));
  int idx = 0;
  for (int k = 0; k < equ.km; ++k)
    for (int j = 0; j < equ.jm; ++j)
      equ.psi[k][j] = psi_flat[idx++] + equ.psib;

  printf("  Equilibrium: jm=%d km=%d btf=%.3f rtf=%.3f psib=%.6e\n",
         equ.jm, equ.km, equ.btf, equ.rtf, equ.psib);
  printf("  R range: [%.4f, %.4f] m, Z range: [%.4f, %.4f] m\n",
         equ.r.front(), equ.r.back(), equ.z.front(), equ.z.back());

  fin.close();
  return equ;
}

bool ComputePlasmaFields::readEquilibriumFromPlasmaH5(
    const std::string &plasma_h5_path, EquilibriumData &data)
{
  try {
    H5::Exception::dontPrint();
    H5::H5File file(plasma_h5_path, H5F_ACC_RDONLY);

    auto hasGroup = [&](const std::string &name) -> bool {
      htri_t exists = 0;
      H5E_BEGIN_TRY {
        exists = H5Oexists_by_name(file.getId(), name.c_str(), H5P_DEFAULT);
      } H5E_END_TRY;
      return exists > 0;
    };
    if (!hasGroup("equilibrium") || !hasGroup("equilibrium/psi"))
      return false;

    auto read1D = [&](const std::string &name) -> std::vector<double> {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim = 0;
      sp.getSimpleExtentDims(&dim);
      std::vector<double> vec(dim);
      ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
      return vec;
    };
    auto readScalar = [&](const std::string &name) -> double {
      if (!hasGroup(name)) return 0.0;
      H5::DataSet ds = file.openDataSet(name);
      double v = 0.0;
      ds.read(&v, H5::PredType::NATIVE_DOUBLE);
      return v;
    };

    data.r = read1D("equilibrium/r");
    data.z = read1D("equilibrium/z");
    data.jm = static_cast<int>(data.r.size());
    data.km = static_cast<int>(data.z.size());

    H5::DataSet dspsi = file.openDataSet("equilibrium/psi");
    H5::DataSpace sp = dspsi.getSpace();
    hsize_t dims[2] = {0, 0};
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != data.km ||
        static_cast<int>(dims[1]) != data.jm) {
      fprintf(stderr,
        "readEquilibriumFromPlasmaH5: psi shape (%lu,%lu) does not match "
        "(km=%d, jm=%d) from equilibrium/r,z — ignoring embedded equilibrium\n",
        (unsigned long)dims[0], (unsigned long)dims[1], data.km, data.jm);
      return false;
    }
    std::vector<double> flat(dims[0] * dims[1]);
    dspsi.read(flat.data(), H5::PredType::NATIVE_DOUBLE);
    data.psi.assign(data.km, std::vector<double>(data.jm, 0.0));
    for (int k = 0; k < data.km; ++k)
      for (int j = 0; j < data.jm; ++j)
        data.psi[k][j] = flat[k * data.jm + j];

    data.btf  = readScalar("equilibrium/btf");
    data.rtf  = readScalar("equilibrium/rtf");
    data.psib = readScalar("equilibrium/psib");
    return true;
  } catch (H5::Exception &) {
    return false;
  }
}

void ComputePlasmaFields::broadcastEquilibriumData(EquilibriumData &data) {
  int me = comm->me;
  MPI_Bcast(&data.jm, 1, MPI_INT, 0, world);
  MPI_Bcast(&data.km, 1, MPI_INT, 0, world);
  MPI_Bcast(&data.btf, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&data.rtf, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&data.psib, 1, MPI_DOUBLE, 0, world);
  if (me != 0) {
    data.r.resize(data.jm);
    data.z.resize(data.km);
  }
  MPI_Bcast(data.r.data(), data.jm, MPI_DOUBLE, 0, world);
  MPI_Bcast(data.z.data(), data.km, MPI_DOUBLE, 0, world);
  if (me != 0) {
    data.psi.resize(data.km, std::vector<double>(data.jm));
  }
  for (int k = 0; k < data.km; ++k)
    MPI_Bcast(data.psi[k].data(), data.jm, MPI_DOUBLE, 0, world);
}

void ComputePlasmaFields::computeMagneticGeometry(
    int icell, const EquilibriumData &equ, MagneticGeometry &geom)
{
  Grid::ChildCell *cells = grid->cells;
  const int dim = domain->dimension;
  const double xyz[3] = {
    0.5 * (cells[icell].lo[0] + cells[icell].hi[0]),
    0.5 * (cells[icell].lo[1] + cells[icell].hi[1]),
    (dim == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2]) : 0.0
  };
  double R, Z;
  OpenEdge::sparta_to_RZ(xyz, dim, domain->axisymmetric, R, Z);
  if (R < 1.0e-10) return;

  const int jm = equ.jm;
  const int km = equ.km;
  if (jm < 3 || km < 3) return;

  const double dr = equ.r[1] - equ.r[0];
  const double dz = equ.z[1] - equ.z[0];
  if (dr <= 0.0 || dz <= 0.0) return;

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
  const double BR = -dpsi_dZ * invR;
  const double BZ = dpsi_dR * invR;
  const double Bphi = equ.btf * equ.rtf * invR;

  const double Bmag = std::sqrt(BR*BR + Bphi*Bphi + BZ*BZ);
  if (Bmag <= 0.0) return;
  geom.Bmag = Bmag;

  const double invBmag = 1.0 / Bmag;
  const double bR = BR * invBmag;
  const double bphi = Bphi * invBmag;
  const double bZ = BZ * invBmag;

  const double invR2 = invR * invR;
  const double dBR_dR2 = dpsi_dZ * invR2 - d2psi_dRdZ * invR;
  const double dBR_dZ2 = -d2psi_dZ2 * invR;
  const double dBZ_dR2 = -dpsi_dR * invR2 + d2psi_dR2 * invR;
  const double dBZ_dZ2 = d2psi_dRdZ * invR;
  const double dBphi_dR = -equ.btf * equ.rtf * invR2;
  const double dBphi_dZ = 0.0;

  geom.gradBmag[0] = (BR * dBR_dR2 + Bphi * dBphi_dR + BZ * dBZ_dR2) * invBmag;
  geom.gradBmag[1] = 0.0;
  geom.gradBmag[2] = (BR * dBR_dZ2 + Bphi * dBphi_dZ + BZ * dBZ_dZ2) * invBmag;

  const double dbR_dR = invBmag * (dBR_dR2 - bR * geom.gradBmag[0]);
  const double dbR_dZ = invBmag * (dBR_dZ2 - bR * geom.gradBmag[2]);
  const double dbphi_dR = invBmag * (dBphi_dR - bphi * geom.gradBmag[0]);
  const double dbphi_dZ = invBmag * (dBphi_dZ - bphi * geom.gradBmag[2]);
  const double dbZ_dR = invBmag * (dBZ_dR2 - bZ * geom.gradBmag[0]);
  const double dbZ_dZ = invBmag * (dBZ_dZ2 - bZ * geom.gradBmag[2]);

  geom.kappa[0] = bR * dbR_dR + bZ * dbR_dZ - bphi * bphi * invR;
  geom.kappa[1] = bR * dbphi_dR + bZ * dbphi_dZ + bR * bphi * invR;
  geom.kappa[2] = bR * dbZ_dR + bZ * dbZ_dZ;

  geom.curl_b[0] = -dbphi_dZ;
  geom.curl_b[1] = dbR_dZ - dbZ_dR;
  geom.curl_b[2] = bphi * invR + dbphi_dR;
}
