/* ----------------------------------------------------------------------
    OpenEdge: time-dependent magnetic field compute
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See compute_bfield_timedep.h for full documentation.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_bfield_timedep.h"
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

ComputeBfieldTimedep::
ComputeBfieldTimedep(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  // compute ID bfield/timedep group file_list manifest.txt [nevery N] [interp linear|step]
  if (narg < 5)
    error->all(FLERR,"Illegal compute bfield/timedep command: need at least "
               "group, file_list, and manifest path");

  int igroup = grid->find_group(arg[2]);
  if (igroup < 0)
    error->all(FLERR,"Compute bfield/timedep grid group ID does not exist");
  groupbit = grid->bitmask[igroup];

  // Parse required argument: file_list <manifest>
  int iarg = 3;
  if (strcmp(arg[iarg], "file_list") != 0)
    error->all(FLERR,"Compute bfield/timedep: expected 'file_list' keyword");
  iarg++;
  if (iarg >= narg)
    error->all(FLERR,"Compute bfield/timedep: missing manifest file path");
  std::string manifest_path = std::string(arg[iarg++]);

  // Defaults
  nevery = 1;
  interp_mode = 1;  // linear

  // Parse optional keywords
  while (iarg < narg) {
    if (strcmp(arg[iarg], "nevery") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Compute bfield/timedep: missing nevery value");
      nevery = input->inumeric(FLERR, arg[iarg + 1]);
      if (nevery <= 0)
        error->all(FLERR,"Compute bfield/timedep: nevery must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "interp") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"Compute bfield/timedep: missing interp value");
      if (strcmp(arg[iarg + 1], "linear") == 0) interp_mode = 1;
      else if (strcmp(arg[iarg + 1], "step") == 0) interp_mode = 0;
      else error->all(FLERR,"Compute bfield/timedep: interp must be 'linear' or 'step'");
      iarg += 2;
    } else {
      error->all(FLERR,"Compute bfield/timedep: unknown keyword");
    }
  }

  // Parse the manifest file (rank 0 reads, then broadcast file list)
  parseManifest(manifest_path);

  if (nsnaps < 1)
    error->all(FLERR,"Compute bfield/timedep: manifest must have at least 1 snapshot");

  // Verify times are monotonically non-decreasing
  for (int i = 1; i < nsnaps; i++) {
    if (snap_times[i] < snap_times[i - 1])
      error->all(FLERR,"Compute bfield/timedep: snapshot times must be non-decreasing");
  }

  // Output 3 columns: br, bt, bz
  per_grid_flag = 1;
  size_per_grid_cols = 3;
  post_process_grid_flag = 0;

  nglocal = 0;
  array_grid = NULL;

  idx_lo = -1;
  idx_hi = -1;
}

/* ---------------------------------------------------------------------- */

ComputeBfieldTimedep::~ComputeBfieldTimedep()
{
  if (copymode) return;
  memory->destroy(array_grid);
}

/* ---------------------------------------------------------------------- */

void ComputeBfieldTimedep::init()
{
  reallocate();

  // Load initial bracketing snapshots
  // Start with idx_lo = 0
  if (nsnaps == 1) {
    // Single snapshot: load it as both lo and hi
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
    printf("  compute bfield/timedep: %d snapshots, nevery=%d, interp=%s\n",
           nsnaps, nevery, interp_mode ? "linear" : "step");
    printf("  Time range: [%.4f, %.4f] s\n",
           snap_times.front(), snap_times.back());
  }
}

/* ---------------------------------------------------------------------- */

void ComputeBfieldTimedep::compute_per_grid()
{
  invoked_per_grid = update->ntimestep;

  // Respect nevery cadence: skip if not on cadence step
  // (But always compute on first invocation)
  if (nevery > 1 && update->ntimestep > 0 &&
      (update->ntimestep % nevery) != 0) {
    // Keep previous values in array_grid
    return;
  }

  Grid::ChildCell *cells = grid->cells;
  Grid::ChildInfo *cinfo = grid->cinfo;
  const int ncells = grid->nlocal;

  // Get current simulation time
  const double t = update->time;

  // --- Find bracketing snapshots ---
  // Clamp to available range
  double alpha = 0.0;

  if (nsnaps == 1) {
    // Single snapshot, no interpolation needed
    alpha = 0.0;
  } else if (t <= snap_times[0]) {
    // Before first snapshot: use first
    if (idx_lo != 0 || idx_hi != 0) {
      loadSnapshot(0, snap_lo, stencil_lo);
      snap_hi = snap_lo;
      stencil_hi = stencil_lo;
      idx_lo = 0;
      idx_hi = 0;
    }
    alpha = 0.0;
  } else if (t >= snap_times[nsnaps - 1]) {
    // After last snapshot: use last
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
    // Find bracket: snap_times[new_lo] <= t < snap_times[new_hi]
    int new_lo = 0;
    for (int i = nsnaps - 2; i >= 0; i--) {
      if (snap_times[i] <= t) {
        new_lo = i;
        break;
      }
    }
    int new_hi = new_lo + 1;

    // Load new snapshots if bracket changed
    if (new_lo != idx_lo) {
      // Check if we can reuse snap_hi as the new snap_lo
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

  // --- Interpolate B-field to each cell center ---
  for (int icell = 0; icell < ncells; icell++) {
    if (!(cinfo[icell].mask & groupbit) || cells[icell].nsplit < 1) {
      array_grid[icell][0] = 0.0;
      array_grid[icell][1] = 0.0;
      array_grid[icell][2] = 0.0;
      continue;
    }

    // Interpolate from lo snapshot
    double br_lo = 0.0, bt_lo = 0.0, bz_lo = 0.0;
    if (icell < static_cast<int>(stencil_lo.size()) && stencil_lo[icell].valid) {
      br_lo = interpField2D(snap_lo.br, stencil_lo[icell]);
      bt_lo = interpField2D(snap_lo.bt, stencil_lo[icell]);
      bz_lo = interpField2D(snap_lo.bz, stencil_lo[icell]);
    }

    if (idx_lo == idx_hi || alpha == 0.0) {
      // No interpolation needed
      array_grid[icell][0] = br_lo;
      array_grid[icell][1] = bt_lo;
      array_grid[icell][2] = bz_lo;
    } else {
      // Interpolate from hi snapshot
      double br_hi = 0.0, bt_hi = 0.0, bz_hi = 0.0;
      if (icell < static_cast<int>(stencil_hi.size()) && stencil_hi[icell].valid) {
        br_hi = interpField2D(snap_hi.br, stencil_hi[icell]);
        bt_hi = interpField2D(snap_hi.bt, stencil_hi[icell]);
        bz_hi = interpField2D(snap_hi.bz, stencil_hi[icell]);
      }

      array_grid[icell][0] = w_lo * br_lo + w_hi * br_hi;
      array_grid[icell][1] = w_lo * bt_lo + w_hi * bt_hi;
      array_grid[icell][2] = w_lo * bz_lo + w_hi * bz_hi;
    }
  }
}

/* ---------------------------------------------------------------------- */

void ComputeBfieldTimedep::reallocate()
{
  if (grid->nlocal == nglocal) return;

  nglocal = grid->nlocal;
  memory->destroy(array_grid);
  memory->create(array_grid, nglocal, 3, "bfield/timedep:array_grid");

  // Zero-initialize
  for (int i = 0; i < nglocal; i++)
    array_grid[i][0] = array_grid[i][1] = array_grid[i][2] = 0.0;
}

/* ---------------------------------------------------------------------- */

bigint ComputeBfieldTimedep::memory_usage()
{
  bigint bytes = (bigint) nglocal * 3 * sizeof(double);
  // Add snapshot storage (two snapshots)
  auto snap_bytes = [](const BfieldSnapshot &s) -> bigint {
    bigint b = s.r.size() * sizeof(double);
    b += s.z.size() * sizeof(double);
    for (auto &row : s.br) b += row.size() * sizeof(double);
    for (auto &row : s.bt) b += row.size() * sizeof(double);
    for (auto &row : s.bz) b += row.size() * sizeof(double);
    return b;
  };
  bytes += snap_bytes(snap_lo) + snap_bytes(snap_hi);
  bytes += (stencil_lo.size() + stencil_hi.size()) * sizeof(BilinStencil);
  return bytes;
}

/* ----------------------------------------------------------------------
   Parse manifest file: each line has "time  filename"
   Lines starting with # are comments
------------------------------------------------------------------------- */

void ComputeBfieldTimedep::parseManifest(const std::string &path)
{
  int me = comm->me;

  // Extract directory from manifest path for relative file resolution
  size_t last_sep = path.find_last_of("/\\");
  manifest_dir = (last_sep != std::string::npos) ? path.substr(0, last_sep + 1) : "";

  // Rank 0 reads the manifest
  int count = 0;
  if (me == 0) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Compute bfield/timedep: cannot open manifest file: %s",
               path.c_str());
      error->one(FLERR, msg);
    }

    std::string line;
    while (std::getline(fin, line)) {
      // Skip comments and empty lines
      size_t first = line.find_first_not_of(" \t");
      if (first == std::string::npos) continue;
      if (line[first] == '#') continue;

      std::istringstream iss(line);
      double t;
      std::string fname;
      if (!(iss >> t >> fname)) continue;

      snap_times.push_back(t);
      // Resolve relative paths against manifest directory
      if (fname[0] != '/' && !manifest_dir.empty())
        snap_files.push_back(manifest_dir + fname);
      else
        snap_files.push_back(fname);
    }
    count = static_cast<int>(snap_times.size());
  }

  // Broadcast count
  MPI_Bcast(&count, 1, MPI_INT, 0, world);
  nsnaps = count;

  // Broadcast times
  if (me != 0) snap_times.resize(nsnaps);
  MPI_Bcast(snap_times.data(), nsnaps, MPI_DOUBLE, 0, world);

  // Broadcast file paths (pack as length-prefixed strings)
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
   Read a B-field HDF5 file: datasets r(nr), z(nz), br(nz,nr), bt, bz
------------------------------------------------------------------------- */

BfieldSnapshot ComputeBfieldTimedep::readBfieldH5(const std::string &path)
{
  BfieldSnapshot snap;
  if (comm->me == 0)
    printf("  bfield/timedep: reading %s\n", path.c_str());

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

    snap.r = read1D("r");
    snap.z = read1D("z");
    snap.br = read2D("br");
    snap.bt = read2D("bt");
    snap.bz = read2D("bz");
    file.close();
  } catch (const H5::Exception &e) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "bfield/timedep: error reading HDF5 file %s: %s",
             path.c_str(), e.getCDetailMsg());
    error->one(FLERR, msg);
  }

  return snap;
}

/* ---------------------------------------------------------------------- */

void ComputeBfieldTimedep::broadcastSnapshot(BfieldSnapshot &snap)
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
    if (me != 0) vec.resize(dim1, std::vector<double>(dim2));
    for (int i = 0; i < dim1; i++)
      MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
  };

  bcast2D(snap.br);
  bcast2D(snap.bt);
  bcast2D(snap.bz);
}

/* ----------------------------------------------------------------------
   Load one snapshot: rank 0 reads HDF5, broadcast, precompute stencils
------------------------------------------------------------------------- */

void ComputeBfieldTimedep::loadSnapshot(
    int idx, BfieldSnapshot &snap, std::vector<BilinStencil> &stencils)
{
  if (idx < 0 || idx >= nsnaps)
    error->all(FLERR, "bfield/timedep: snapshot index out of range");

  if (comm->me == 0)
    snap = readBfieldH5(snap_files[idx]);

  broadcastSnapshot(snap);
  precomputeStencils(snap, stencils);
}

/* ----------------------------------------------------------------------
   Precompute bilinear stencils for all cell centers
------------------------------------------------------------------------- */

void ComputeBfieldTimedep::precomputeStencils(
    const BfieldSnapshot &snap, std::vector<BilinStencil> &stencils)
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

BilinStencil ComputeBfieldTimedep::makeStencilAtPoint(
    const double xyz[3],
    const std::vector<double> &r_vals,
    const std::vector<double> &z_vals) const
{
  BilinStencil s{};
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

double ComputeBfieldTimedep::interpField2D(
    const std::vector<std::vector<double>> &field,
    const BilinStencil &s) const
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
