/* ----------------------------------------------------------------------
    OpenEdge: time-dependent plasma field compute
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See compute_plasma_timedep.h for full documentation.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_plasma_timedep.h"
#include "update.h"
#include "grid.h"
#include "domain.h"
#include "input.h"
#include "memory.h"
#include "error.h"
#include "comm.h"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <H5Cpp.h>
#include <hdf5.h>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

ComputePlasmaTimedep::
ComputePlasmaTimedep(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  // compute ID plasma/timedep group file_list manifest.txt
  //   [nevery N] [interp linear|step] values val1 val2 ...
  if (narg < 5)
    error->all(FLERR,"Illegal compute plasma/timedep command: need at least "
               "group, file_list, manifest path, and values");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute plasma/timedep grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  // Parse required argument: file_list <manifest>
  int iarg = 3;
  if (strcmp(arg[iarg], "file_list") != 0)
    error->all(FLERR,"Compute plasma/timedep: expected 'file_list' keyword");
  iarg++;
  if (iarg >= narg)
    error->all(FLERR,"Compute plasma/timedep: missing manifest file path");
  std::string manifest_path = std::string(arg[iarg++]);

  // Defaults
  nevery = 1;
  interp_mode = 1;  // linear

  // Parse optional keywords before 'values'
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nevery") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Compute plasma/timedep: missing nevery value");
      nevery = input->inumeric(FLERR, arg[iarg + 1]);
      if (nevery <= 0)
        error->all(FLERR,"Compute plasma/timedep: nevery must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "interp") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Compute plasma/timedep: missing interp value");
      if (strcmp(arg[iarg + 1], "linear") == 0) interp_mode = 1;
      else if (strcmp(arg[iarg + 1], "step") == 0) interp_mode = 0;
      else error->all(FLERR,"Compute plasma/timedep: interp must be 'linear' or 'step'");
      iarg += 2;
    } else if (strcmp(arg[iarg], "values") == 0) {
      iarg++;
      break;
    } else {
      error->all(FLERR,"Compute plasma/timedep: unknown keyword before 'values'");
    }
  }

  // Parse value keywords
  nvalues = 0;
  while (iarg < narg) {
    int vid = -1;
    if      (strcmp(arg[iarg], "dens_e") == 0)     vid = DENS_E;
    else if (strcmp(arg[iarg], "temp_e") == 0)     vid = TEMP_E;
    else if (strcmp(arg[iarg], "dens_i") == 0)     vid = DENS_I;
    else if (strcmp(arg[iarg], "temp_i") == 0)     vid = TEMP_I;
    else if (strcmp(arg[iarg], "parrflow") == 0)   vid = PARRFLOW;
    else if (strcmp(arg[iarg], "parrflow_r") == 0) vid = PARRFLOW_R;
    else if (strcmp(arg[iarg], "parrflow_t") == 0) vid = PARRFLOW_T;
    else if (strcmp(arg[iarg], "parrflow_z") == 0) vid = PARRFLOW_Z;
    else if (strcmp(arg[iarg], "grad_te_r") == 0)  vid = GRAD_TE_R;
    else if (strcmp(arg[iarg], "grad_te_t") == 0)  vid = GRAD_TE_T;
    else if (strcmp(arg[iarg], "grad_te_z") == 0)  vid = GRAD_TE_Z;
    else if (strcmp(arg[iarg], "grad_ti_r") == 0)  vid = GRAD_TI_R;
    else if (strcmp(arg[iarg], "grad_ti_t") == 0)  vid = GRAD_TI_T;
    else if (strcmp(arg[iarg], "grad_ti_z") == 0)  vid = GRAD_TI_Z;
    else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Compute plasma/timedep: unknown value keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
    value_ids.push_back(vid);
    nvalues++;
    iarg++;
  }

  if (nvalues == 0)
    error->all(FLERR,"Compute plasma/timedep: no output values specified");

  // Parse the manifest file
  parseManifest(manifest_path);

  if (nsnaps < 1)
    error->all(FLERR,"Compute plasma/timedep: manifest must have at least 1 snapshot");

  for (int i = 1; i < nsnaps; i++) {
    if (snap_times[i] < snap_times[i - 1])
      error->all(FLERR,"Compute plasma/timedep: snapshot times must be non-decreasing");
  }

  // Output nvalues columns
  per_grid_flag = 1;
  size_per_grid_cols = nvalues;
  post_process_grid_flag = 0;

  nglocal = 0;
  array_grid = NULL;

  idx_lo = -1;
  idx_hi = -1;
}

/* ---------------------------------------------------------------------- */

ComputePlasmaTimedep::~ComputePlasmaTimedep()
{
  if (copymode) return;
  memory->destroy(array_grid);
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::init()
{
  reallocate();

  // Load initial bracketing snapshots
  if (nsnaps == 1) {
    loadSnapshot(0, snap_lo, stencil_lo);
    snap_hi = snap_lo;
    stencil_hi = stencil_lo;
    idx_lo = 0;
    idx_hi = 0;
  } else {
    loadSnapshot(0, snap_lo, stencil_lo);
    loadSnapshot(1, snap_hi, stencil_hi);
    idx_lo = 0;
    idx_hi = 1;
  }

  if (comm->me == 0) {
    printf("  compute plasma/timedep: %d snapshots, %d values, nevery=%d, interp=%s\n",
           nsnaps, nvalues, nevery, interp_mode ? "linear" : "step");
    printf("  Time range: [%.4f, %.4f] s\n",
           snap_times.front(), snap_times.back());
  }
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  // Respect nevery cadence
  if (nevery > 1 && update->ntimestep > 0 &&
      (update->ntimestep % nevery) != 0) {
    return;
  }

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int ncells = grid->nlocal;

  const double t = update->time;

  // --- Find bracketing snapshots ---
  double alpha = 0.0;

  if (nsnaps == 1) {
    alpha = 0.0;
  } else if (t <= snap_times[0]) {
    if (idx_lo != 0 || idx_hi != 0) {
      loadSnapshot(0, snap_lo, stencil_lo);
      snap_hi = snap_lo;
      stencil_hi = stencil_lo;
      idx_lo = 0;
      idx_hi = 0;
    }
    alpha = 0.0;
  } else if (t >= snap_times[nsnaps - 1]) {
    int last = nsnaps - 1;
    if (idx_lo != last || idx_hi != last) {
      loadSnapshot(last, snap_lo, stencil_lo);
      snap_hi = snap_lo;
      stencil_hi = stencil_lo;
      idx_lo = last;
      idx_hi = last;
    }
    alpha = 0.0;
  } else {
    int new_lo = 0;
    for (int i = nsnaps - 2; i >= 0; i--) {
      if (snap_times[i] <= t) {
        new_lo = i;
        break;
      }
    }
    int new_hi = new_lo + 1;

    if (new_lo != idx_lo) {
      if (new_lo == idx_hi) {
        std::swap(snap_lo, snap_hi);
        std::swap(stencil_lo, stencil_hi);
        idx_lo = new_lo;
      } else {
        loadSnapshot(new_lo, snap_lo, stencil_lo);
        idx_lo = new_lo;
      }
    }
    if (new_hi != idx_hi) {
      loadSnapshot(new_hi, snap_hi, stencil_hi);
      idx_hi = new_hi;
    }

    double dt_snap = snap_times[idx_hi] - snap_times[idx_lo];
    if (dt_snap > 0.0) {
      alpha = (t - snap_times[idx_lo]) / dt_snap;
      alpha = std::min(std::max(alpha, 0.0), 1.0);
    }
  }

  // For step interpolation, snap to nearest
  if (interp_mode == 0) {
    alpha = (alpha < 0.5) ? 0.0 : 1.0;
  }

  const double w_lo = 1.0 - alpha;
  const double w_hi = alpha;

  // --- Interpolate plasma fields to each cell center ---
  for (int icell = 0; icell < ncells; icell++) {
    if (!(cinfo[icell].mask & groupbit) || cells[icell].nsplit < 1) {
      for (int iv = 0; iv < nvalues; iv++)
        array_grid[icell][iv] = 0.0;
      continue;
    }

    for (int iv = 0; iv < nvalues; iv++) {
      const auto &field_lo = getField(snap_lo, value_ids[iv]);

      double val_lo = 0.0;
      if (icell < static_cast<int>(stencil_lo.size()) && stencil_lo[icell].valid) {
        val_lo = interpField2D(field_lo, stencil_lo[icell]);
      }

      if (idx_lo == idx_hi || alpha == 0.0) {
        array_grid[icell][iv] = val_lo;
      } else {
        const auto &field_hi = getField(snap_hi, value_ids[iv]);
        double val_hi = 0.0;
        if (icell < static_cast<int>(stencil_hi.size()) && stencil_hi[icell].valid) {
          val_hi = interpField2D(field_hi, stencil_hi[icell]);
        }
        array_grid[icell][iv] = w_lo * val_lo + w_hi * val_hi;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::reallocate()
{
  if (grid->nlocal == nglocal) return;

  nglocal = grid->nlocal;
  memory->destroy(array_grid);
  memory->create(array_grid, nglocal, nvalues, "plasma/timedep:array_grid");

  for (int i = 0; i < nglocal; i++)
    for (int j = 0; j < nvalues; j++)
      array_grid[i][j] = 0.0;
}

/* ---------------------------------------------------------------------- */

bigint ComputePlasmaTimedep::memory_usage()
{
  bigint bytes = (bigint) nglocal * nvalues * sizeof(double);
  // Two snapshot storage estimates
  auto snap_bytes = [](const PlasmaSnapshot &s) -> bigint {
    bigint b = s.r.size() * sizeof(double);
    b += s.z.size() * sizeof(double);
    auto add2d = [&b](const std::vector<std::vector<double>> &v) {
      for (auto &row : v) b += row.size() * sizeof(double);
    };
    add2d(s.dens_e); add2d(s.temp_e);
    add2d(s.dens_i); add2d(s.temp_i);
    add2d(s.parr_flow);
    add2d(s.parr_flow_r); add2d(s.parr_flow_t); add2d(s.parr_flow_z);
    add2d(s.grad_te_r); add2d(s.grad_te_t); add2d(s.grad_te_z);
    add2d(s.grad_ti_r); add2d(s.grad_ti_t); add2d(s.grad_ti_z);
    return b;
  };
  bytes += snap_bytes(snap_lo) + snap_bytes(snap_hi);
  bytes += (stencil_lo.size() + stencil_hi.size()) * sizeof(PlasmaStencil);
  return bytes;
}

/* ----------------------------------------------------------------------
   Parse manifest file: each line has "time  filename"
------------------------------------------------------------------------- */

void ComputePlasmaTimedep::parseManifest(const std::string &path)
{
  int me = comm->me;

  size_t last_sep = path.find_last_of("/\\");
  manifest_dir = (last_sep != std::string::npos) ? path.substr(0, last_sep + 1) : "";

  int count = 0;
  if (me == 0) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Compute plasma/timedep: cannot open manifest file: %s",
               path.c_str());
      error->one(FLERR, msg);
    }

    std::string line;
    while (std::getline(fin, line)) {
      size_t first = line.find_first_not_of(" \t");
      if (first == std::string::npos) continue;
      if (line[first] == '#') continue;

      std::istringstream iss(line);
      double t;
      std::string fname;
      if (!(iss >> t >> fname)) continue;

      snap_times.push_back(t);
      if (fname[0] != '/' && !manifest_dir.empty())
        snap_files.push_back(manifest_dir + fname);
      else
        snap_files.push_back(fname);
    }
    count = static_cast<int>(snap_times.size());
  }

  MPI_Bcast(&count, 1, MPI_INT, 0, world);
  nsnaps = count;

  if (me != 0) snap_times.resize(nsnaps);
  MPI_Bcast(snap_times.data(), nsnaps, MPI_DOUBLE, 0, world);

  if (me != 0) snap_files.resize(nsnaps);
  for (int i = 0; i < nsnaps; i++) {
    int slen = 0;
    if (me == 0) slen = static_cast<int>(snap_files[i].size());
    MPI_Bcast(&slen, 1, MPI_INT, 0, world);
    if (me != 0) snap_files[i].resize(slen);
    MPI_Bcast(&snap_files[i][0], slen, MPI_CHAR, 0, world);
  }
}

/* ----------------------------------------------------------------------
   Read a plasma HDF5 file
------------------------------------------------------------------------- */

PlasmaSnapshot ComputePlasmaTimedep::readPlasmaH5(const std::string &path)
{
  PlasmaSnapshot snap;
  if (comm->me == 0)
    printf("  plasma/timedep: reading %s\n", path.c_str());

  try {
    H5::H5File file(path, H5F_ACC_RDONLY);

    auto read1D = [&file](const std::string &name) {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace space = ds.getSpace();
      std::vector<hsize_t> dims(1);
      space.getSimpleExtentDims(dims.data(), NULL);
      std::vector<double> data(dims[0]);
      ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
      return data;
    };

    auto read2D = [&file](const std::string &name) {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace space = ds.getSpace();
      std::vector<hsize_t> dims(2);
      space.getSimpleExtentDims(dims.data(), NULL);
      std::vector<double> raw(dims[0] * dims[1]);
      ds.read(raw.data(), H5::PredType::NATIVE_DOUBLE);
      std::vector<std::vector<double>> data(dims[0],
                                             std::vector<double>(dims[1]));
      for (hsize_t i = 0; i < dims[0]; i++)
        for (hsize_t j = 0; j < dims[1]; j++)
          data[i][j] = raw[i * dims[1] + j];
      return data;
    };

    // Helper: read 2D if dataset exists, else return empty
    auto has_dataset = [&file](const std::string &name) -> bool {
      return H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT) > 0;
    };

    auto read2D_opt = [&](const std::string &name)
        -> std::vector<std::vector<double>> {
      if (has_dataset(name)) return read2D(name);
      return {};
    };

    snap.r = read1D("r");
    snap.z = read1D("z");

    snap.dens_e = read2D("dens_e");
    snap.temp_e = read2D("temp_e");
    snap.dens_i = read2D("dens_i");
    snap.temp_i = read2D("temp_i");
    snap.parr_flow = read2D_opt("parr_flow");
    snap.parr_flow_r = read2D_opt("parr_flow_r");
    snap.parr_flow_t = read2D_opt("parr_flow_t");
    snap.parr_flow_z = read2D_opt("parr_flow_z");
    snap.grad_te_r = read2D_opt("grad_te_r");
    snap.grad_te_t = read2D_opt("grad_te_t");
    snap.grad_te_z = read2D_opt("grad_te_z");
    snap.grad_ti_r = read2D_opt("grad_ti_r");
    snap.grad_ti_t = read2D_opt("grad_ti_t");
    snap.grad_ti_z = read2D_opt("grad_ti_z");

    file.close();
  } catch (const H5::Exception &e) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "plasma/timedep: error reading HDF5 file %s: %s",
             path.c_str(), e.getCDetailMsg());
    error->one(FLERR, msg);
  }

  return snap;
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::broadcastSnapshot(PlasmaSnapshot &snap)
{
  int me = comm->me;

  int r_size = snap.r.size();
  int z_size = snap.z.size();
  MPI_Bcast(&r_size, 1, MPI_INT, 0, world);
  MPI_Bcast(&z_size, 1, MPI_INT, 0, world);

  if (me != 0) {
    snap.r.resize(r_size);
    snap.z.resize(z_size);
  }
  MPI_Bcast(snap.r.data(), r_size, MPI_DOUBLE, 0, world);
  MPI_Bcast(snap.z.data(), z_size, MPI_DOUBLE, 0, world);

  auto bcast2D = [&](std::vector<std::vector<double>> &vec) {
    int dim1 = vec.size();
    int dim2 = dim1 ? static_cast<int>(vec[0].size()) : 0;
    MPI_Bcast(&dim1, 1, MPI_INT, 0, world);
    MPI_Bcast(&dim2, 1, MPI_INT, 0, world);
    if (dim1 == 0) {
      if (me != 0) vec.clear();
      return;
    }
    if (me != 0) vec.resize(dim1, std::vector<double>(dim2));
    for (int i = 0; i < dim1; i++)
      MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
  };

  bcast2D(snap.dens_e);
  bcast2D(snap.temp_e);
  bcast2D(snap.dens_i);
  bcast2D(snap.temp_i);
  bcast2D(snap.parr_flow);
  bcast2D(snap.parr_flow_r);
  bcast2D(snap.parr_flow_t);
  bcast2D(snap.parr_flow_z);
  bcast2D(snap.grad_te_r);
  bcast2D(snap.grad_te_t);
  bcast2D(snap.grad_te_z);
  bcast2D(snap.grad_ti_r);
  bcast2D(snap.grad_ti_t);
  bcast2D(snap.grad_ti_z);
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::loadSnapshot(
    int idx, PlasmaSnapshot &snap, std::vector<PlasmaStencil> &stencils)
{
  if (idx < 0 || idx >= nsnaps)
    error->all(FLERR, "plasma/timedep: snapshot index out of range");

  if (comm->me == 0)
    snap = readPlasmaH5(snap_files[idx]);

  broadcastSnapshot(snap);
  precomputeStencils(snap, stencils);
}

/* ---------------------------------------------------------------------- */

void ComputePlasmaTimedep::precomputeStencils(
    const PlasmaSnapshot &snap, std::vector<PlasmaStencil> &stencils)
{
  const int ncells = grid->nlocal;
  stencils.clear();
  stencils.resize(ncells);

  const int dim = domain->dimension;

  for (int icell = 0; icell < ncells; icell++) {
    Grid::ChildCell *cell = &grid->cells[icell];
    double cc[3];
    cc[0] = 0.5 * (cell->lo[0] + cell->hi[0]);
    cc[1] = 0.5 * (cell->lo[1] + cell->hi[1]);
    cc[2] = (dim == 3)
            ? 0.5 * (cell->lo[2] + cell->hi[2])
            : cc[1];
    stencils[icell] = makeStencilAtPoint(cc, snap.r, snap.z);
  }
}

/* ---------------------------------------------------------------------- */

PlasmaStencil ComputePlasmaTimedep::makeStencilAtPoint(
    const double xyz[3],
    const std::vector<double> &r_vals,
    const std::vector<double> &z_vals) const
{
  PlasmaStencil s{};
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

  const double tt = (r_clamp - R1) / denomR;
  const double uu = (z_clamp - Z1) / denomZ;
  s.t = tt;
  s.u = uu;
  s.w11 = (1.0 - tt) * (1.0 - uu);
  s.w21 = tt * (1.0 - uu);
  s.w12 = (1.0 - tt) * uu;
  s.w22 = tt * uu;
  s.valid = 1;
  return s;
}

/* ---------------------------------------------------------------------- */

double ComputePlasmaTimedep::interpField2D(
    const std::vector<std::vector<double>> &field,
    const PlasmaStencil &s) const
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

/* ----------------------------------------------------------------------
   Map value enum to the corresponding 2D field in a snapshot.
   Returns empty vector ref for missing optional fields.
------------------------------------------------------------------------- */

const std::vector<std::vector<double>> &
ComputePlasmaTimedep::getField(const PlasmaSnapshot &snap, int val_id) const
{
  switch (val_id) {
    case DENS_E:     return snap.dens_e;
    case TEMP_E:     return snap.temp_e;
    case DENS_I:     return snap.dens_i;
    case TEMP_I:     return snap.temp_i;
    case PARRFLOW:   return snap.parr_flow;
    case PARRFLOW_R: return snap.parr_flow_r;
    case PARRFLOW_T: return snap.parr_flow_t;
    case PARRFLOW_Z: return snap.parr_flow_z;
    case GRAD_TE_R:  return snap.grad_te_r;
    case GRAD_TE_T:  return snap.grad_te_t;
    case GRAD_TE_Z:  return snap.grad_te_z;
    case GRAD_TI_R:  return snap.grad_ti_r;
    case GRAD_TI_T:  return snap.grad_ti_t;
    case GRAD_TI_Z:  return snap.grad_ti_z;
    default:         return snap.dens_e;  // fallback
  }
}
