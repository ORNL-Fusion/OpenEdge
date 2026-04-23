/* ----------------------------------------------------------------------
   OpenEdge — Neutral and charge-state transport with plasma-wall interactions
   Built on SPARTA (sparta.github.io; Plimpton et al., Sandia National Labs).

   Author: Abdourahmane Diaw <diawa@ornl.gov>
           Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge

   Distributed under the GNU General Public License, same as SPARTA.
   See the LICENSE file at the top of the repository.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   ProcessLibrary implementation.

   HDF5 access follows the OpenEdge rank-0-read + MPI_Bcast idiom used
   throughout the package (see docs/mpi_audit_2026_04_21 notes).  Every
   public method is a collective: all ranks MUST call it, all ranks
   receive the same data.
------------------------------------------------------------------------- */

#include "process_library.h"

#include <H5Cpp.h>
#include <cstring>
#include <stdexcept>

#include "error.h"

namespace SPARTA_NS {

// ------------------------------------------------------------------------
// open()
// ------------------------------------------------------------------------
void ProcessLibrary::open(const std::string &path, MPI_Comm comm,
                          Error *error)
{
  comm_ = comm;
  MPI_Comm_rank(comm_, &me_);

  // Broadcast the path length + chars so every rank knows whether a
  // valid file was provided.
  int len = static_cast<int>(path.size());
  MPI_Bcast(&len, 1, MPI_INT, 0, comm_);
  std::vector<char> buf(len + 1, 0);
  if (me_ == 0 && len > 0)
    std::memcpy(buf.data(), path.data(), len);
  MPI_Bcast(buf.data(), len + 1, MPI_CHAR, 0, comm_);
  path_ = std::string(buf.data(), len);

  if (path_.empty()) {
    opened_ = false;
    return;
  }

  // Rank 0 tries to open the file; on failure, broadcast the error and
  // abort collectively.
  int ok = 1;
  if (me_ == 0) {
    try {
      H5::H5File f(path_, H5F_ACC_RDONLY);
      (void)f;
    } catch (const H5::Exception &) {
      ok = 0;
    }
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, comm_);
  if (!ok) {
    if (error)
      error->all(FLERR,
        ("ProcessLibrary: failed to open " + path_).c_str());
    opened_ = false;
    return;
  }
  opened_ = true;
}

// ------------------------------------------------------------------------
// load_rate()
// ------------------------------------------------------------------------
bool ProcessLibrary::load_rate(const std::string &cls,
                               const std::string &elem,
                               std::vector<double> &coef,
                               std::vector<double> &logT,
                               std::vector<double> &logN,
                               int &nQ, int &nT, int &nD)
{
  if (!opened_) return false;

  const std::string key = cls + "/" + elem;

  auto it = rate_cache_.find(key);
  if (it == rate_cache_.end()) {
    RateCacheEntry &entry = rate_cache_[key];

    // /volume/rates/<cls>/<elem>/ for sigma-v tables;
    // /volume/radiation/<cls>/<elem>/ for power coefficients.
    std::string top = "/volume/rates/";
    if (cls == "plt" || cls == "prb") top = "/volume/radiation/";
    const std::string group_path = top + cls + "/" + elem;

    Error *err = nullptr;  // warnings only; fetch_rate returns false on miss
    fetch_rate(group_path, entry, err);
    it = rate_cache_.find(key);
  }

  const RateCacheEntry &entry = it->second;
  if (!entry.present) return false;

  coef = entry.coef;
  logT = entry.logT;
  logN = entry.logN;
  nQ = entry.nQ;
  nT = entry.nT;
  nD = entry.nD;
  return true;
}

// ------------------------------------------------------------------------
// load_ionization_potential()
// ------------------------------------------------------------------------
bool ProcessLibrary::load_ionization_potential(const std::string &elem,
                                               std::vector<double> &ip_eV)
{
  if (!opened_) return false;

  auto it = ip_cache_.find(elem);
  if (it == ip_cache_.end()) {
    std::vector<double> &slot = ip_cache_[elem];  // default-constructs empty
    const std::string ds_path =
      "/volume/thresholds/ionization/" + elem + "/energy";
    fetch_vector(ds_path, slot, nullptr);
    it = ip_cache_.find(elem);
  }
  if (it->second.empty()) return false;
  ip_eV = it->second;
  return true;
}

// ------------------------------------------------------------------------
// fetch_rate(): rank-0 reads /<group>/coefficient + temperature +
// density, broadcasts to all ranks.
// ------------------------------------------------------------------------
bool ProcessLibrary::fetch_rate(const std::string &group_path,
                                RateCacheEntry &out, Error * /*error*/)
{
  // Header: presence flag + three dim counts.
  int present = 0;

  if (me_ == 0) {
    try {
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), group_path.c_str(), H5P_DEFAULT) > 0) {
        const std::string coef_path = group_path + "/coefficient";
        const std::string temp_path = group_path + "/temperature";
        const std::string dens_path = group_path + "/density";
        if (H5Lexists(f.getId(), coef_path.c_str(), H5P_DEFAULT) > 0 &&
            H5Lexists(f.getId(), temp_path.c_str(), H5P_DEFAULT) > 0 &&
            H5Lexists(f.getId(), dens_path.c_str(), H5P_DEFAULT) > 0) {
          H5::DataSet ds = f.openDataSet(coef_path);
          H5::DataSpace sp = ds.getSpace();
          hsize_t dims[3] = {0, 0, 0};
          sp.getSimpleExtentDims(dims, nullptr);
          out.nQ = static_cast<int>(dims[0]);
          out.nT = static_cast<int>(dims[1]);
          out.nD = static_cast<int>(dims[2]);
          out.coef.resize(dims[0] * dims[1] * dims[2]);
          ds.read(out.coef.data(), H5::PredType::NATIVE_DOUBLE);

          H5::DataSet dsT = f.openDataSet(temp_path);
          hsize_t tdim[1] = {0};
          dsT.getSpace().getSimpleExtentDims(tdim, nullptr);
          out.logT.resize(tdim[0]);
          dsT.read(out.logT.data(), H5::PredType::NATIVE_DOUBLE);

          H5::DataSet dsN = f.openDataSet(dens_path);
          hsize_t ndim[1] = {0};
          dsN.getSpace().getSimpleExtentDims(ndim, nullptr);
          out.logN.resize(ndim[0]);
          dsN.read(out.logN.data(), H5::PredType::NATIVE_DOUBLE);

          present = 1;
        }
      }
    } catch (const H5::Exception &) {
      present = 0;
    }
  }

  MPI_Bcast(&present, 1, MPI_INT, 0, comm_);
  if (!present) {
    out.present = false;
    return false;
  }

  // Broadcast dims, then data.
  int dims3[3] = {out.nQ, out.nT, out.nD};
  MPI_Bcast(dims3, 3, MPI_INT, 0, comm_);
  out.nQ = dims3[0]; out.nT = dims3[1]; out.nD = dims3[2];
  const size_t n = static_cast<size_t>(out.nQ) * out.nT * out.nD;
  if (me_ != 0) out.coef.resize(n);
  if (n > 0) MPI_Bcast(out.coef.data(), n, MPI_DOUBLE, 0, comm_);

  int nt_len = static_cast<int>(out.logT.size());
  MPI_Bcast(&nt_len, 1, MPI_INT, 0, comm_);
  if (me_ != 0) out.logT.resize(nt_len);
  if (nt_len > 0) MPI_Bcast(out.logT.data(), nt_len, MPI_DOUBLE, 0, comm_);

  int nn_len = static_cast<int>(out.logN.size());
  MPI_Bcast(&nn_len, 1, MPI_INT, 0, comm_);
  if (me_ != 0) out.logN.resize(nn_len);
  if (nn_len > 0) MPI_Bcast(out.logN.data(), nn_len, MPI_DOUBLE, 0, comm_);

  out.present = true;
  return true;
}

// ------------------------------------------------------------------------
// fetch_vector(): rank-0 reads a 1-D dataset, broadcasts.
// ------------------------------------------------------------------------
bool ProcessLibrary::fetch_vector(const std::string &dataset_path,
                                  std::vector<double> &out,
                                  Error * /*error*/)
{
  int n = 0;
  if (me_ == 0) {
    try {
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), dataset_path.c_str(), H5P_DEFAULT) > 0) {
        H5::DataSet ds = f.openDataSet(dataset_path);
        hsize_t dim[1] = {0};
        ds.getSpace().getSimpleExtentDims(dim, nullptr);
        n = static_cast<int>(dim[0]);
        out.resize(n);
        if (n > 0) ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
      }
    } catch (const H5::Exception &) {
      n = 0;
    }
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, comm_);
  if (n == 0) {
    out.clear();
    return false;
  }
  if (me_ != 0) out.resize(n);
  MPI_Bcast(out.data(), n, MPI_DOUBLE, 0, comm_);
  return true;
}

}  // namespace SPARTA_NS
