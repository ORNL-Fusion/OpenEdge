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
#include "update.h"
#include "grid.h"
#include "domain.h"
#include "input.h"
#include "memory.h"
#include "error.h"
#include "comm.h"

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
  GRAD_TE_R, GRAD_TE_T, GRAD_TE_Z, GRAD_TI_R, GRAD_TI_T, GRAD_TI_Z
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
  // compute ... plasma/fields ggroup file plasma.h5 bfield.h5 ...
  // compute ... plasma/fields ggroup constant [const args] ...
  // compute ... plasma/fields ggroup analytic [analytic args] ...
  int iarg = 3;
  if (iarg >= narg)
    error->all(FLERR,"compute plasma/fields requires mode: file, constant, or analytic");
  if (strcmp(arg[iarg],"file") == 0) {
    input_mode = MODE_FILE;
    iarg++;
    if (iarg + 1 >= narg)
      error->all(FLERR,"compute plasma/fields file mode needs: plasma.h5 bfield.h5");
    plasmaStatePath = std::string(arg[iarg++]);
    magneticFieldsPath = std::string(arg[iarg++]);
  } else if (strcmp(arg[iarg],"constant") == 0) {
    input_mode = MODE_CONSTANT;
    iarg++;
  } else if (strcmp(arg[iarg],"analytic") == 0) {
    input_mode = MODE_ANALYTIC;
    iarg++;
  } else {
    error->all(FLERR,"compute plasma/fields mode must be 'file', 'constant', or 'analytic'");
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

  if (input_mode == MODE_FILE) {
    if (me == 0) {
      plasma_data   = readPlasmaFileData(plasmaStatePath);
      magnetic_data = readMagneticFieldFileData(magneticFieldsPath);
    }
    broadcastPlasmaData(plasma_data);
    broadcastMagneticData(magnetic_data);
    precomputeStencils(plasma_data.r, plasma_data.z, plasma_stencil);
    precomputeStencils(magnetic_data.r, magnetic_data.z, magnetic_stencil);

    for (int icell = 0; icell < ncells; ++icell) {
      if (!(cinfo[icell].mask & groupbit)) continue;
      if (cells[icell].nsplit < 1)         continue;
      plasma_arr[icell] = bilinearInterpolationPlasma(icell, plasma_data);
      mag_arr[icell]    = bilinearInterpolationMagneticField(icell, magnetic_data);
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
      if (me == 0) {
        equ_data = readEquilibriumFile(equilibriumPath);
      }
      broadcastEquilibriumData(equ_data);
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
    const double Er = (input_mode != MODE_FILE) ? econst[0] : 0.0;
    const double Et = (input_mode != MODE_FILE) ? econst[1] : 0.0;
    const double Ezv = (input_mode != MODE_FILE) ? econst[2] : 0.0;
    const double Vr = P.parr_flow_r;
    const double Vt = P.parr_flow_t;
    const double Vzv = P.parr_flow_z;

    // Cylindrical -> Cartesian transform in 3D Cartesian geometry.
    // In 2D (r,z), cartesian aliases map directly to cylindrical components.
    double cphi = 1.0, sphi = 0.0;
    if (dim == 3) {
      const Grid::ChildCell &cell = cells[icell];
      const double x = 0.5 * (cell.lo[0] + cell.hi[0]);
      const double y = 0.5 * (cell.lo[1] + cell.hi[1]);
      const double rxy = std::sqrt(x*x + y*y);
      if (rxy > 0.0) {
        cphi = x / rxy;
        sphi = y / rxy;
      }
    }

    // Component mapping:
    // - 3D Cartesian: convert (r,t,z) -> (x,y,z) using local phi.
    // - 2D R-Z SPARTA plane (x=R, y=Z, z=out-of-plane toroidal):
    //     bx=Br, by=Bz, bz=Bt
    //     ex=Er, ey=Ez, ez=Et
    //     vx=Vr, vy=Vz, vz=Vt
    double Bx, By, Bzz;
    double Ex, Ey, Ezz;
    double Vx, Vy, Vzz;
    if (dim == 2) {
      Bx = Br;   By = Bzv;  Bzz = Bt;
      Ex = Er;   Ey = Ezv;  Ezz = Et;
      Vx = Vr;   Vy = Vzv;  Vzz = Vt;
    } else {
      Bx = Br*cphi - Bt*sphi;
      By = Br*sphi + Bt*cphi;
      Bzz = Bzv;
      Ex = Er*cphi - Et*sphi;
      Ey = Er*sphi + Et*cphi;
      Ezz = Ezv;
      Vx = Vr*cphi - Vt*sphi;
      Vy = Vr*sphi + Vt*cphi;
      Vzz = Vzv;
    }
    const double ne = std::max(P.dens_e, tiny);
    // WEST/SOLPS convention used here:
    //   Te in eV, ne in 1/m^3, grad(Te) in eV/m, grad(ne) in 1/m^4.
    // Then electron pressure is:
    //   pe [Pa] = e * ne * Te[eV]
    // and
    //   grad(pe) [Pa/m] = e * (Te*grad(ne) + ne*grad(Te)).
    const double gradPe_r = eQ * (P.temp_e * P.grad_dens_e_r + ne * P.grad_temp_e_r);
    const double gradPe_t = eQ * (P.temp_e * P.grad_dens_e_t + ne * P.grad_temp_e_t);
    const double gradPe_z = eQ * (P.temp_e * P.grad_dens_e_z + ne * P.grad_temp_e_z);
    const double Bmag = std::sqrt(Br*Br + Bt*Bt + Bzv*Bzv);
    const double invBmag = (Bmag > tiny) ? 1.0 / Bmag : 0.0;
    const double bhat_r = Br * invBmag;
    const double bhat_t = Bt * invBmag;
    const double bhat_z = Bzv * invBmag;
    const double epar =
      (Bmag > tiny) ? (-(gradPe_r*bhat_r + gradPe_t*bhat_t + gradPe_z*bhat_z) / (ne * eQ)) : 0.0;

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

        // First read coordinates
        data.r = read1D("r");
        data.z = read1D("z");
        size_t nr = data.r.size();
        size_t nz = data.z.size();

        // Utility to test whether dataset/path exists
        auto hasDataset = [&](const std::string& name) -> bool {
            htri_t exists = 0;
            H5E_BEGIN_TRY {
              exists = H5Oexists_by_name(file.getId(), name.c_str(), H5P_DEFAULT);
            } H5E_END_TRY;
            return exists > 0;
        };

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

        // Load 2D fields with strict shape check
        data.dens_e        = read2D("dens_e");
        data.temp_e        = read2D("temp_e");
        data.dens_i        = read2D("dens_i");
        data.temp_i        = read2D("temp_i");

        data.parr_flow     = read2D("parr_flow");
        data.parr_flow_r   = read2D("parr_flow_r");
        data.parr_flow_t   = read2D("parr_flow_t");
        data.parr_flow_z   = read2D("parr_flow_z");

        data.grad_temp_e_r = read2D("grad_te_r");
        data.grad_temp_e_t = read2D("grad_te_t");
        data.grad_temp_e_z = read2D("grad_te_z");

        data.grad_temp_i_r = read2D("grad_ti_r");
        data.grad_temp_i_t = read2D("grad_ti_t");
        data.grad_temp_i_z = read2D("grad_ti_z");

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

    } catch (const H5::Exception& e) {
        fprintf(stderr, "HDF5 error: %s\n", e.getCDetailMsg());
        throw;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        throw;
    }

    return data;
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

  return P;
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
  if (dim == 2) {
    r = xyz[0];
    z = xyz[1];
  } else {
    r = std::sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1]);
    z = xyz[2];
  }

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

  // MODE_FILE: build stencil on-the-fly and interpolate
  BilinearStencil s = makeStencilAtPoint(xyz, plasma_data.r, plasma_data.z);
  if (!s.valid) return P;

  P.temp_e = interpField2D(plasma_data.temp_e, s);
  P.dens_e = interpField2D(plasma_data.dens_e, s);
  P.temp_i = interpField2D(plasma_data.temp_i, s);
  P.dens_i = interpField2D(plasma_data.dens_i, s);
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
    }
    return B;
  }

  // MODE_FILE: build stencil on-the-fly and interpolate
  BilinearStencil s = makeStencilAtPoint(xyz, magnetic_data.r, magnetic_data.z);
  if (!s.valid) return B;

  B.br = interpField2D(magnetic_data.br, s);
  B.bt = interpField2D(magnetic_data.bt, s);
  B.bz = interpField2D(magnetic_data.bz, s);
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
  const double x = 0.5 * (cells[icell].lo[0] + cells[icell].hi[0]);
  const double y = 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
  const double zc = (dim == 3) ? 0.5 * (cells[icell].lo[2] + cells[icell].hi[2])
                                : 0.5 * (cells[icell].lo[1] + cells[icell].hi[1]);
  double R, Z;
  if (dim == 2) { R = x; Z = y; }
  else { R = std::sqrt(x*x + y*y); Z = zc; }
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
  const double BR = dpsi_dZ * invR;
  const double BZ = -dpsi_dR * invR;
  const double Bphi = equ.btf * equ.rtf * invR;

  const double Bmag = std::sqrt(BR*BR + Bphi*Bphi + BZ*BZ);
  if (Bmag <= 0.0) return;
  geom.Bmag = Bmag;

  const double invBmag = 1.0 / Bmag;
  const double bR = BR * invBmag;
  const double bphi = Bphi * invBmag;
  const double bZ = BZ * invBmag;

  const double invR2 = invR * invR;
  const double dBR_dR2 = -dpsi_dZ * invR2 + d2psi_dRdZ * invR;
  const double dBR_dZ2 = d2psi_dZ2 * invR;
  const double dBZ_dR2 = dpsi_dR * invR2 - d2psi_dR2 * invR;
  const double dBZ_dZ2 = -d2psi_dRdZ * invR;
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
