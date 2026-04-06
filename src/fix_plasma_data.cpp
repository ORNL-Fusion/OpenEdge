/* ----------------------------------------------------------------------
   OpenEdge fix plasma/data
   Centralised plasma + equilibrium data store.
   Read once, shared by all fixes/computes that need background plasma.

   Usage:
     fix ID plasma/data file plasma.h5 [equilibrium file.equ] [static yes]

   Other fixes/computes access this via:
     int ifix = modify->find_fix("ID");
     auto *pd = dynamic_cast<FixPlasmaData*>(modify->fix[ifix]);
     double te = pd->interp2D(pd->temp_e, R, Z);
------------------------------------------------------------------------- */

#include "string.h"
#include "fix_plasma_data.h"
#include "comm.h"
#include "error.h"
#include "modify.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

FixPlasmaData::FixPlasmaData(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // fix ID plasma/data file PATH [equilibrium PATH] [static yes/no]
  if (narg < 4)
    error->all(FLERR, "Illegal fix plasma/data command: need at least 'file PATH'");

  nr = nz = 0;
  nion = 0;
  has_bfield = 0;
  has_equ = 0;
  has_mesh = 0;
  is_static = 0;
  generation = 0;
  equ_jm = equ_km = 0;
  btf = rtf = psib = psi_axis = 0.0;
  mesh_nvtx = mesh_ntri = mesh_ncell = 0;

  int iarg = 2;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix plasma/data: file needs path");
      plasma_path = std::string(arg[iarg + 1]);
      iarg += 2;
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
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix plasma/data: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (plasma_path.empty())
    error->all(FLERR, "fix plasma/data: must specify 'file PATH'");
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
  try {
    if (comm->me == 0) load_plasma_h5();
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

void FixPlasmaData::reload_plasma(const std::string &path)
{
  plasma_path = path;
  reload();
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
