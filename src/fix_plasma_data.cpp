/* ----------------------------------------------------------------------
   OpenEdge fix plasma/data
   Centralised plasma + equilibrium data store.
   Read once, shared by all fixes/computes that need background plasma.

   Usage:
     fix ID plasma/data file plasma.h5 [equilibrium file.equ] [static yes]
     fix ID plasma/data constant [r_bounds rmin rmax] [z_bounds zmin zmax]
                         [ne val] [te val] [ni val] [ti val]
                         [parr_flow val] [parr_flow_r val] [parr_flow_t val]
                         [parr_flow_z val] [grad_te_r val] [grad_te_t val]
                         [grad_te_z val] [grad_ti_r val] [grad_ti_t val]
                         [grad_ti_z val] [epar val] [br val] [bz val] [bt val]
                         [equilibrium file.equ] [static yes]

   Other fixes/computes access this via:
     int ifix = modify->find_fix("ID");
     auto *pd = dynamic_cast<FixPlasmaData*>(modify->fix[ifix]);
     double te = pd->interp2D(pd->temp_e, R, Z);
------------------------------------------------------------------------- */

#include "string.h"
#include "fix_plasma_data.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "modify.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace SPARTA_NS;

namespace {
enum { PLASMA_SOURCE_FILE = 0, PLASMA_SOURCE_CONSTANT = 1 };
}

/* ---------------------------------------------------------------------- */

FixPlasmaData::FixPlasmaData(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // fix ID plasma/data file PATH [equilibrium PATH] [static yes/no]
  // fix ID plasma/data constant ...
  if (narg < 3)
    error->all(FLERR, "Illegal fix plasma/data command");

  nr = nz = 0;
  nion = 0;
  has_bfield = 0;
  has_equ = 0;
  has_mesh = 0;
  is_static = 0;
  source_mode = -1;
  generation = 0;
  equ_jm = equ_km = 0;
  btf = rtf = psib = psi_axis = 0.0;
  mesh_nvtx = mesh_ntri = mesh_ncell = 0;
  const_has_r_bounds = const_has_z_bounds = 0;
  const_rmin = 0.0; const_rmax = 1.0;
  const_zmin = 0.0; const_zmax = 1.0;
  const_dens_e = const_temp_e = 0.0;
  const_dens_i = const_temp_i = 0.0;
  const_has_dens_i = const_has_temp_i = 0;
  const_parr_flow = const_parr_flow_r = const_parr_flow_t = const_parr_flow_z = 0.0;
  const_grad_te_r = const_grad_te_t = const_grad_te_z = 0.0;
  const_grad_ti_r = const_grad_ti_t = const_grad_ti_z = 0.0;
  const_epar = 0.0;
  const_br = const_bz = const_bt = 0.0;
  const_has_bfield = 0;

  int iarg = 2;
  auto parse_scalar = [&](double &dst, const char *label) {
    if (iarg + 1 >= narg) {
      char msg[256];
      snprintf(msg, sizeof(msg), "fix plasma/data: %s needs a numeric value", label);
      error->all(FLERR, msg);
    }
    dst = input->numeric(FLERR, arg[iarg + 1]);
    iarg += 2;
  };
  while (iarg < narg) {
    if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix plasma/data: file needs path");
      if (source_mode == PLASMA_SOURCE_CONSTANT)
        error->all(FLERR, "fix plasma/data: choose either file or constant mode");
      source_mode = PLASMA_SOURCE_FILE;
      plasma_path = std::string(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "constant") == 0) {
      if (source_mode == PLASMA_SOURCE_FILE)
        error->all(FLERR, "fix plasma/data: choose either file or constant mode");
      source_mode = PLASMA_SOURCE_CONSTANT;
      iarg += 1;
    } else if (strcmp(arg[iarg], "equilibrium") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix plasma/data: equilibrium needs path");
      equ_path = std::string(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "static") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix plasma/data: static needs yes/no");
      if (strcmp(arg[iarg + 1], "yes") == 0) is_static = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) is_static = 0;
      else error->all(FLERR, "fix plasma/data: static must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "r_bounds") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "fix plasma/data: r_bounds needs rmin rmax");
      const_rmin = input->numeric(FLERR, arg[iarg + 1]);
      const_rmax = input->numeric(FLERR, arg[iarg + 2]);
      const_has_r_bounds = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "z_bounds") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "fix plasma/data: z_bounds needs zmin zmax");
      const_zmin = input->numeric(FLERR, arg[iarg + 1]);
      const_zmax = input->numeric(FLERR, arg[iarg + 2]);
      const_has_z_bounds = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "ne") == 0 || strcmp(arg[iarg], "dens_e") == 0) {
      parse_scalar(const_dens_e, arg[iarg]);
    } else if (strcmp(arg[iarg], "te") == 0 || strcmp(arg[iarg], "temp_e") == 0) {
      parse_scalar(const_temp_e, arg[iarg]);
    } else if (strcmp(arg[iarg], "ni") == 0 || strcmp(arg[iarg], "dens_i") == 0) {
      parse_scalar(const_dens_i, arg[iarg]);
      const_has_dens_i = 1;
    } else if (strcmp(arg[iarg], "ti") == 0 || strcmp(arg[iarg], "temp_i") == 0) {
      parse_scalar(const_temp_i, arg[iarg]);
      const_has_temp_i = 1;
    } else if (strcmp(arg[iarg], "parr_flow") == 0 || strcmp(arg[iarg], "upar") == 0) {
      parse_scalar(const_parr_flow, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_r") == 0 || strcmp(arg[iarg], "upar_r") == 0) {
      parse_scalar(const_parr_flow_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_t") == 0 || strcmp(arg[iarg], "upar_t") == 0) {
      parse_scalar(const_parr_flow_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_z") == 0 || strcmp(arg[iarg], "upar_z") == 0) {
      parse_scalar(const_parr_flow_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_r") == 0) {
      parse_scalar(const_grad_te_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_t") == 0) {
      parse_scalar(const_grad_te_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_z") == 0) {
      parse_scalar(const_grad_te_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_r") == 0) {
      parse_scalar(const_grad_ti_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_t") == 0) {
      parse_scalar(const_grad_ti_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_z") == 0) {
      parse_scalar(const_grad_ti_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "epar") == 0) {
      parse_scalar(const_epar, arg[iarg]);
    } else if (strcmp(arg[iarg], "br") == 0) {
      parse_scalar(const_br, arg[iarg]);
      const_has_bfield = 1;
    } else if (strcmp(arg[iarg], "bz") == 0) {
      parse_scalar(const_bz, arg[iarg]);
      const_has_bfield = 1;
    } else if (strcmp(arg[iarg], "bt") == 0) {
      parse_scalar(const_bt, arg[iarg]);
      const_has_bfield = 1;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix plasma/data: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (source_mode < 0)
    error->all(FLERR, "fix plasma/data: must specify either 'file PATH' or 'constant'");
  if (source_mode == PLASMA_SOURCE_FILE && plasma_path.empty())
    error->all(FLERR, "fix plasma/data: must specify 'file PATH'");
  if (const_has_r_bounds && const_rmax <= const_rmin)
    error->all(FLERR, "fix plasma/data: r_bounds requires rmax > rmin");
  if (const_has_z_bounds && const_zmax <= const_zmin)
    error->all(FLERR, "fix plasma/data: z_bounds requires zmax > zmin");
}

/* ---------------------------------------------------------------------- */

FixPlasmaData::~FixPlasmaData() {}

/* ---------------------------------------------------------------------- */

int FixPlasmaData::setmask()
{
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::init()
{
  // Only load on first init, or if not static
  if (generation > 0 && is_static) return;

  reload();
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::reload()
{
  clear_loaded_data();

  try {
    if (comm->me == 0) {
      if (source_mode == PLASMA_SOURCE_FILE) load_plasma_h5();
      else load_constant_profile();
    }
  } catch (const std::exception &e) {
    error->one(FLERR, e.what());
  }

  // Broadcast plasma grid dimensions
  MPI_Bcast(&nr, 1, MPI_INT, 0, world);
  MPI_Bcast(&nz, 1, MPI_INT, 0, world);
  MPI_Bcast(&has_bfield, 1, MPI_INT, 0, world);
  MPI_Bcast(&nion, 1, MPI_INT, 0, world);
  MPI_Bcast(&has_mesh, 1, MPI_INT, 0, world);

  size_t grid_n = static_cast<size_t>(nz) * nr;

  // Resize on non-root
  if (comm->me != 0) {
    rvals.resize(nr);
    zvals.resize(nz);
    dens_e.resize(grid_n);
    temp_e.resize(grid_n);
    dens_i.resize(grid_n);
    temp_i.resize(grid_n);
    parr_flow.resize(grid_n);
    parr_flow_r.resize(grid_n);
    parr_flow_t.resize(grid_n);
    parr_flow_z.resize(grid_n);
    grad_te_r.resize(grid_n);
    grad_te_t.resize(grid_n);
    grad_te_z.resize(grid_n);
    grad_ti_r.resize(grid_n);
    grad_ti_t.resize(grid_n);
    grad_ti_z.resize(grid_n);
    epar.resize(grid_n);
    if (has_bfield) {
      br.resize(grid_n);
      bz.resize(grid_n);
      bt.resize(grid_n);
    }
    if (nion > 0) {
      ion_charge_z.resize(nion);
      ion_mass_amu.resize(nion);
      size_t ion_n = static_cast<size_t>(nion) * grid_n;
      ions_dens.resize(ion_n);
      ions_temp.resize(ion_n);
      ions_upar.resize(ion_n);
    }
  }

  // Broadcast all arrays
  MPI_Bcast(rvals.data(), nr, MPI_DOUBLE, 0, world);
  MPI_Bcast(zvals.data(), nz, MPI_DOUBLE, 0, world);
  MPI_Bcast(dens_e.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(temp_e.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(dens_i.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(temp_i.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(epar.data(), grid_n, MPI_DOUBLE, 0, world);

  if (has_bfield) {
    MPI_Bcast(br.data(), grid_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(bz.data(), grid_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(bt.data(), grid_n, MPI_DOUBLE, 0, world);
  }

  if (nion > 0) {
    MPI_Bcast(ion_charge_z.data(), nion, MPI_INT, 0, world);
    MPI_Bcast(ion_mass_amu.data(), nion, MPI_DOUBLE, 0, world);
    size_t ion_n = static_cast<size_t>(nion) * grid_n;
    MPI_Bcast(ions_dens.data(), ion_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(ions_temp.data(), ion_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(ions_upar.data(), ion_n, MPI_DOUBLE, 0, world);
  }

  // Load equilibrium (all ranks, text file is cheap)
  if (!equ_path.empty()) {
    try {
      load_equilibrium();
    } catch (const std::exception &e) {
      error->all(FLERR, e.what());
    }
    // Derive B-field if not already in plasma.h5
    if (!has_bfield) {
      derive_bfield_from_equ();
    }
  }

  generation++;

  if (comm->me == 0) {
    if (screen) {
      fprintf(screen,
        "[plasma/data] Loaded: %d x %d grid, %d ion species, "
        "bfield=%s, equ=%s, gen=%d\n",
        nr, nz, nion, has_bfield ? "yes" : "no",
        has_equ ? "yes" : "no", generation);
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::clear_loaded_data()
{
  nr = nz = 0;
  nion = 0;
  has_bfield = 0;
  has_equ = 0;
  has_mesh = 0;
  equ_jm = equ_km = 0;
  btf = rtf = psib = psi_axis = 0.0;
  mesh_nvtx = mesh_ntri = mesh_ncell = 0;

  rvals.clear();
  zvals.clear();
  dens_e.clear();
  temp_e.clear();
  dens_i.clear();
  temp_i.clear();
  parr_flow.clear();
  parr_flow_r.clear();
  parr_flow_t.clear();
  parr_flow_z.clear();
  grad_te_r.clear();
  grad_te_t.clear();
  grad_te_z.clear();
  grad_ti_r.clear();
  grad_ti_t.clear();
  grad_ti_z.clear();
  epar.clear();
  br.clear();
  bz.clear();
  bt.clear();
  ion_charge_z.clear();
  ion_mass_amu.clear();
  ion_names.clear();
  ions_dens.clear();
  ions_temp.clear();
  ions_upar.clear();
  equ_r.clear();
  equ_z.clear();
  psirz.clear();
  mesh_vtx_r.clear();
  mesh_vtx_z.clear();
  mesh_tri.clear();
  mesh_cell_idx.clear();
  mesh_ne.clear();
  mesh_te.clear();
  mesh_ti.clear();
  mesh_ni.clear();
  mesh_upar.clear();
  valid_mask.clear();
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::reload_plasma(const std::string &path)
{
  plasma_path = path;
  source_mode = PLASMA_SOURCE_FILE;
  reload();
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::load_constant_profile()
{
  double rmin = const_rmin, rmax = const_rmax;
  double zmin = const_zmin, zmax = const_zmax;

  if (!const_has_r_bounds || !const_has_z_bounds) {
    if (domain->dimension == 2) {
      if (!const_has_r_bounds) { rmin = domain->boxlo[0]; rmax = domain->boxhi[0]; }
      if (!const_has_z_bounds) { zmin = domain->boxlo[1]; zmax = domain->boxhi[1]; }
    } else {
      const double r00 = std::hypot(domain->boxlo[0], domain->boxlo[1]);
      const double r01 = std::hypot(domain->boxlo[0], domain->boxhi[1]);
      const double r10 = std::hypot(domain->boxhi[0], domain->boxlo[1]);
      const double r11 = std::hypot(domain->boxhi[0], domain->boxhi[1]);
      if (!const_has_r_bounds) {
        rmin = 0.0;
        rmax = std::max(std::max(r00, r01), std::max(r10, r11));
      }
      if (!const_has_z_bounds) { zmin = domain->boxlo[2]; zmax = domain->boxhi[2]; }
    }
  }

  if (rmax <= rmin) rmax = rmin + 1.0;
  if (zmax <= zmin) zmax = zmin + 1.0;

  nr = 2;
  nz = 2;
  rvals = {rmin, rmax};
  zvals = {zmin, zmax};

  const size_t n = 4;
  const double ni_uniform = const_has_dens_i ? const_dens_i : const_dens_e;
  const double ti_uniform = const_has_temp_i ? const_temp_i : const_temp_e;

  dens_e.assign(n, const_dens_e);
  temp_e.assign(n, const_temp_e);
  dens_i.assign(n, ni_uniform);
  temp_i.assign(n, ti_uniform);
  parr_flow.assign(n, const_parr_flow);
  parr_flow_r.assign(n, const_parr_flow_r);
  parr_flow_t.assign(n, const_parr_flow_t);
  parr_flow_z.assign(n, const_parr_flow_z);
  grad_te_r.assign(n, const_grad_te_r);
  grad_te_t.assign(n, const_grad_te_t);
  grad_te_z.assign(n, const_grad_te_z);
  grad_ti_r.assign(n, const_grad_ti_r);
  grad_ti_t.assign(n, const_grad_ti_t);
  grad_ti_z.assign(n, const_grad_ti_z);
  epar.assign(n, const_epar);

  has_bfield = const_has_bfield;
  if (has_bfield) {
    br.assign(n, const_br);
    bz.assign(n, const_bz);
    bt.assign(n, const_bt);
  }

  if (screen) {
    fprintf(screen,
            "[plasma/data] Using inline constant fields on synthetic %d x %d grid"
            " (R:[%.6g,%.6g] Z:[%.6g,%.6g])\n",
            nr, nz, rmin, rmax, zmin, zmax);
  }
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::load_plasma_h5()
{
  if (screen)
    fprintf(screen, "[plasma/data] Reading %s\n", plasma_path.c_str());

  H5::Exception::dontPrint();
  H5::H5File file(plasma_path, H5F_ACC_RDONLY);

  auto read1D = [&](const std::string &name, std::vector<double> &out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dim;
    sp.getSimpleExtentDims(&dim);
    out.resize(dim);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  auto hasDataset = [&](const std::string &name) -> bool {
    htri_t exists = 0;
    H5E_BEGIN_TRY { exists = H5Oexists_by_name(file.getId(), name.c_str(), H5P_DEFAULT); }
    H5E_END_TRY;
    return exists > 0;
  };

  // Grid coordinates
  read1D("r", rvals);
  read1D("z", zvals);
  nr = static_cast<int>(rvals.size());
  nz = static_cast<int>(zvals.size());
  size_t n = static_cast<size_t>(nz) * nr;

  // Read 2D field into flat vector [iz * nr + ir]
  auto read2D = [&](const std::string &name, std::vector<double> &out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != nz || static_cast<int>(dims[1]) != nr)
      throw std::runtime_error("Shape mismatch in " + name);
    out.resize(n);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  auto read2D_optional = [&](const std::string &name, std::vector<double> &out) {
    if (hasDataset(name)) read2D(name, out);
    else { out.assign(n, 0.0); }
  };

  read2D("dens_e", dens_e);
  read2D("temp_e", temp_e);
  read2D("dens_i", dens_i);
  read2D("temp_i", temp_i);
  read2D_optional("parr_flow", parr_flow);
  read2D_optional("parr_flow_r", parr_flow_r);
  read2D_optional("parr_flow_t", parr_flow_t);
  read2D_optional("parr_flow_z", parr_flow_z);
  read2D_optional("grad_te_r", grad_te_r);
  read2D_optional("grad_te_t", grad_te_t);
  read2D_optional("grad_te_z", grad_te_z);
  read2D_optional("grad_ti_r", grad_ti_r);
  read2D_optional("grad_ti_t", grad_ti_t);
  read2D_optional("grad_ti_z", grad_ti_z);
  read2D_optional("epar", epar);

  // Optional B-field in plasma HDF5
  if (hasDataset("br") && hasDataset("bz")) {
    read2D("br", br);
    read2D("bz", bz);
    if (hasDataset("bt")) read2D("bt", bt);
    else bt.assign(n, 0.0);
    has_bfield = 1;
  }

  // Multi-ion species
  nion = 0;
  if (hasDataset("ion_species/charge_state_z")) {
    H5::DataSet ds = file.openDataSet("ion_species/charge_state_z");
    H5::DataSpace sp = ds.getSpace();
    hsize_t dim;
    sp.getSimpleExtentDims(&dim);
    nion = static_cast<int>(dim);
    ion_charge_z.resize(nion);
    ds.read(ion_charge_z.data(), H5::PredType::NATIVE_INT);
  }
  if (hasDataset("ion_species/mass_amu")) {
    H5::DataSet ds = file.openDataSet("ion_species/mass_amu");
    ion_mass_amu.resize(nion);
    ds.read(ion_mass_amu.data(), H5::PredType::NATIVE_DOUBLE);
  }

  // 3D ion fields: (nion, nz, nr) -> flat
  auto read3D = [&](const std::string &name, std::vector<double> &out) {
    if (!hasDataset(name)) { out.assign(static_cast<size_t>(nion) * n, 0.0); return; }
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    out.resize(dims[0] * dims[1] * dims[2]);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  if (nion > 0) {
    read3D("ions/dens", ions_dens);
    read3D("ions/temp", ions_temp);
    read3D("ions/parr_flow", ions_upar);
  }

  // Mesh triangulation (optional)
  has_mesh = 0;
  if (hasDataset("mesh/vtx_r")) {
    has_mesh = 1;
    // Read mesh data
    auto read1Dint = [&](const std::string &name, std::vector<int> &out) {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      out.resize(dim);
      ds.read(out.data(), H5::PredType::NATIVE_INT);
    };
    read1D("mesh/vtx_r", mesh_vtx_r);
    read1D("mesh/vtx_z", mesh_vtx_z);
    mesh_nvtx = static_cast<int>(mesh_vtx_r.size());

    if (hasDataset("mesh/triangles")) {
      H5::DataSet ds = file.openDataSet("mesh/triangles");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2];
      sp.getSimpleExtentDims(dims);
      mesh_ntri = static_cast<int>(dims[0]);
      mesh_tri.resize(mesh_ntri * 3);
      ds.read(mesh_tri.data(), H5::PredType::NATIVE_INT);
    }
    if (hasDataset("mesh/cell_index")) {
      read1Dint("mesh/cell_index", mesh_cell_idx);
    }

    auto read1D_mesh = [&](const std::string &name, std::vector<double> &out) {
      if (!hasDataset(name)) return;
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      out.resize(dim);
      ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };
    read1D_mesh("mesh/dens_e", mesh_ne);
    read1D_mesh("mesh/temp_e", mesh_te);
    read1D_mesh("mesh/temp_i", mesh_ti);
    read1D_mesh("mesh/dens_i", mesh_ni);
    read1D_mesh("mesh/parr_flow", mesh_upar);
    mesh_ncell = mesh_ne.empty() ? 0 : static_cast<int>(mesh_ne.size());
  }
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::load_equilibrium()
{
  if (comm->me == 0 && screen)
    fprintf(screen, "[plasma/data] Reading equilibrium %s\n", equ_path.c_str());

  std::ifstream fin(equ_path);
  if (!fin.is_open())
    throw std::runtime_error("Cannot open equilibrium file: " + equ_path);

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

  equ_jm = equ_km = 0;
  btf = rtf = psib = 0.0;

  std::string line;
  while (std::getline(fin, line)) {
    if (line.find("jm") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ_jm;
    }
    if (line.find("km") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      if (line.find("km") < line.find("=")) {
        std::istringstream iss(line.substr(line.find("=") + 1));
        iss >> equ_km;
      }
    }
    if (line.find("psib") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> psib;
    }
    if (line.find("btf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> btf;
    }
    if (line.find("rtf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> rtf;
    }
    if (line.find("r(1:jm)") != std::string::npos) break;
  }

  if (equ_jm < 2 || equ_km < 2)
    throw std::runtime_error("Equilibrium: failed to parse jm/km");

  equ_r = readDoubles(equ_jm);
  if (static_cast<int>(equ_r.size()) != equ_jm)
    throw std::runtime_error("Equilibrium: incomplete r array");

  while (std::getline(fin, line)) {
    if (line.find("z(1:km)") != std::string::npos) break;
  }

  equ_z = readDoubles(equ_km);
  if (static_cast<int>(equ_z.size()) != equ_km)
    throw std::runtime_error("Equilibrium: incomplete z array");

  while (std::getline(fin, line)) {
    if (line.find("psi(j,k)") != std::string::npos) break;
  }

  int total = equ_jm * equ_km;
  std::vector<double> psi_flat = readDoubles(total);
  if (static_cast<int>(psi_flat.size()) != total)
    throw std::runtime_error("Equilibrium: incomplete psi array");

  psirz.resize(total);
  for (int i = 0; i < total; i++)
    psirz[i] = psi_flat[i] + psib;

  // Find psi_axis (minimum psi in central region)
  psi_axis = 1e30;
  int j0 = equ_km / 4, j1 = 3 * equ_km / 4;
  int i0 = equ_jm / 4, i1 = 3 * equ_jm / 4;
  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++) {
      double p = psirz[j * equ_jm + i];
      if (p < psi_axis) psi_axis = p;
    }

  has_equ = 1;
  fin.close();

  if (comm->me == 0 && screen)
    fprintf(screen,
      "  Equilibrium: jm=%d km=%d btf=%.3f rtf=%.3f psib=%.6e psi_axis=%.6e\n",
      equ_jm, equ_km, btf, rtf, psib, psi_axis);
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::derive_bfield_from_equ()
{
  if (!has_equ || nr == 0 || nz == 0) return;

  size_t n = static_cast<size_t>(nz) * nr;
  br.resize(n);
  bz.resize(n);
  bt.resize(n);

  double dr_eq = equ_r[1] - equ_r[0];
  double dz_eq = equ_z[1] - equ_z[0];

  auto interp_psi = [&](double R, double Z) -> double {
    double fi = (R - equ_r.front()) / dr_eq;
    double fj = (Z - equ_z.front()) / dz_eq;
    int i0 = std::max(0, std::min((int)fi, equ_jm - 2));
    int j0 = std::max(0, std::min((int)fj, equ_km - 2));
    double s = std::max(0.0, std::min(1.0, fi - i0));
    double t = std::max(0.0, std::min(1.0, fj - j0));
    return (1-s)*(1-t)*psirz[j0*equ_jm+i0] + s*(1-t)*psirz[j0*equ_jm+i0+1]
         + (1-s)*t*psirz[(j0+1)*equ_jm+i0] + s*t*psirz[(j0+1)*equ_jm+i0+1];
  };

  double eps = 1e-4;
  for (int iz = 0; iz < nz; iz++) {
    for (int ir = 0; ir < nr; ir++) {
      double R = rvals[ir];
      double Z = zvals[iz];
      size_t idx = static_cast<size_t>(iz) * nr + ir;

      if (R < 0.01) { br[idx] = bz[idx] = bt[idx] = 0.0; continue; }

      double dpsi_dz = (interp_psi(R, Z + eps) - interp_psi(R, Z - eps)) / (2.0 * eps);
      br[idx] = -dpsi_dz / R;

      double dpsi_dr = (interp_psi(R + eps, Z) - interp_psi(R - eps, Z)) / (2.0 * eps);
      bz[idx] = dpsi_dr / R;

      bt[idx] = btf * rtf / R;
    }
  }

  has_bfield = 1;

  if (comm->me == 0 && screen)
    fprintf(screen, "  B-field derived from equilibrium on %d x %d grid\n", nr, nz);
}

/* ---------------------------------------------------------------------- */

double FixPlasmaData::interp2D(const std::vector<double> &field,
                                double R, double Z) const
{
  if (field.empty() || nr < 2 || nz < 2) return 0.0;

  double Rc = std::min(std::max(R, rvals.front()), rvals.back());
  double Zc = std::min(std::max(Z, zvals.front()), zvals.back());

  double dr = rvals[1] - rvals[0];
  double dz = zvals[1] - zvals[0];
  double fi = (Rc - rvals.front()) / dr;
  double fj = (Zc - zvals.front()) / dz;

  int ir0 = std::max(0, std::min((int)fi, nr - 2));
  int iz0 = std::max(0, std::min((int)fj, nz - 2));
  double s = std::max(0.0, std::min(1.0, fi - ir0));
  double t = std::max(0.0, std::min(1.0, fj - iz0));

  return (1-s)*(1-t)*field[iz0*nr+ir0] + s*(1-t)*field[iz0*nr+ir0+1]
       + (1-s)*t*field[(iz0+1)*nr+ir0] + s*t*field[(iz0+1)*nr+ir0+1];
}

/* ---------------------------------------------------------------------- */

void FixPlasmaData::bfield_at(double R, double Z,
                               double &Br_out, double &Bz_out,
                               double &Bt_out) const
{
  Br_out = interp2D(br, R, Z);
  Bz_out = interp2D(bz, R, Z);
  Bt_out = interp2D(bt, R, Z);
}

/* ---------------------------------------------------------------------- */

double FixPlasmaData::psi_norm_at(double R, double Z) const
{
  if (!has_equ || psirz.empty()) return 1.0;

  double dr_eq = equ_r[1] - equ_r[0];
  double dz_eq = equ_z[1] - equ_z[0];

  double Rc = std::min(std::max(R, equ_r.front()), equ_r.back());
  double Zc = std::min(std::max(Z, equ_z.front()), equ_z.back());

  double fi = (Rc - equ_r.front()) / dr_eq;
  double fj = (Zc - equ_z.front()) / dz_eq;
  int i0 = std::max(0, std::min((int)fi, equ_jm - 2));
  int j0 = std::max(0, std::min((int)fj, equ_km - 2));
  double s = std::max(0.0, std::min(1.0, fi - i0));
  double t = std::max(0.0, std::min(1.0, fj - j0));

  double psi = (1-s)*(1-t)*psirz[j0*equ_jm+i0] + s*(1-t)*psirz[j0*equ_jm+i0+1]
             + (1-s)*t*psirz[(j0+1)*equ_jm+i0] + s*t*psirz[(j0+1)*equ_jm+i0+1];

  double denom = psib - psi_axis;
  if (std::abs(denom) < 1e-30) return 1.0;
  return (psi - psi_axis) / denom;
}
