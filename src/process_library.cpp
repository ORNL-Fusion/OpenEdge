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

#include <cmath>
#include <algorithm>
#include "process_library.h"

#include <H5Cpp.h>
#include <cstring>
#include <stdexcept>

#include "error.h"

namespace SPARTA_NS {

namespace {
// RAII guard that silences the HDF5 error stack for the lifetime of the
// guard, then restores it. Use inside rank-0 blocks that probe optional
// datasets with H5Lexists — without this the HDF5 library dumps its
// error stack to stderr every time an optional path is absent.
struct H5ErrGuard {
  H5E_auto2_t fn;
  void *data;
  H5ErrGuard() {
    H5Eget_auto2(H5E_DEFAULT, &fn, &data);
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  }
  ~H5ErrGuard() { H5Eset_auto2(H5E_DEFAULT, fn, data); }
};
}

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
// load_trim_reflection()
// ------------------------------------------------------------------------
bool ProcessLibrary::load_trim_reflection(const std::string &pair,
                                          TrimReflectionTable &out)
{
  if (!opened_) return false;

  out = TrimReflectionTable{};  // reset

  int present = 0;
  const std::string grp = "/surface/reflection/" + pair;

  if (me_ == 0) {
    try {
      H5ErrGuard _h5err;
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), grp.c_str(), H5P_DEFAULT) > 0) {
        auto read1 = [&](const std::string &p, std::vector<double> &v,
                         int &dim_out) {
          H5::DataSet d = f.openDataSet(p);
          hsize_t dim[1] = {0};
          d.getSpace().getSimpleExtentDims(dim, nullptr);
          dim_out = static_cast<int>(dim[0]);
          v.resize(dim[0]);
          d.read(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        auto readN = [&](const std::string &p, std::vector<double> &v) {
          H5::DataSet d = f.openDataSet(p);
          H5::DataSpace sp = d.getSpace();
          int rnk = sp.getSimpleExtentNdims();
          std::vector<hsize_t> dims(rnk);
          sp.getSimpleExtentDims(dims.data());
          size_t total = 1;
          for (int r = 0; r < rnk; r++) total *= dims[r];
          v.resize(total);
          d.read(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        read1(grp + "/E",     out.E,     out.NE);
        read1(grp + "/theta", out.theta, out.NTHETA);
        read1(grp + "/raar",  out.raar,  out.NQ);
        readN(grp + "/R_N",          out.R_N);
        readN(grp + "/Eout_min",     out.Eout_min);
        readN(grp + "/Eout_max",     out.Eout_max);
        readN(grp + "/Eout_q",       out.Eout_q);
        readN(grp + "/polar_min",    out.polar_min);
        readN(grp + "/polar_max",    out.polar_max);
        readN(grp + "/cos_polar_q",  out.cos_polar_q);
        readN(grp + "/cos_azim_q",   out.cos_azim_q);

        H5::Group g = f.openGroup(grp);
        auto readAttrDouble = [&](const std::string &a, double &v) {
          if (g.attrExists(a)) {
            H5::Attribute att = g.openAttribute(a);
            att.read(H5::PredType::NATIVE_DOUBLE, &v);
          }
        };
        readAttrDouble("Z1", out.Z1);
        readAttrDouble("M1", out.M1);
        readAttrDouble("Z2", out.Z2);
        readAttrDouble("M2", out.M2);

        present = 1;
      }
    } catch (const H5::Exception &) {
      present = 0;
    }
  }

  MPI_Bcast(&present, 1, MPI_INT, 0, comm_);
  if (!present) return false;

  // Broadcast scalars then arrays.
  int dims3[3] = {out.NE, out.NTHETA, out.NQ};
  MPI_Bcast(dims3, 3, MPI_INT, 0, comm_);
  out.NE = dims3[0]; out.NTHETA = dims3[1]; out.NQ = dims3[2];
  double four[4] = {out.Z1, out.M1, out.Z2, out.M2};
  MPI_Bcast(four, 4, MPI_DOUBLE, 0, comm_);
  out.Z1 = four[0]; out.M1 = four[1]; out.Z2 = four[2]; out.M2 = four[3];

  auto bcast_vec = [&](std::vector<double> &v) {
    int n = static_cast<int>(v.size());
    MPI_Bcast(&n, 1, MPI_INT, 0, comm_);
    if (me_ != 0) v.resize(n);
    if (n > 0) MPI_Bcast(v.data(), n, MPI_DOUBLE, 0, comm_);
  };
  bcast_vec(out.E);
  bcast_vec(out.theta);
  bcast_vec(out.raar);
  bcast_vec(out.R_N);
  bcast_vec(out.Eout_min);
  bcast_vec(out.Eout_max);
  bcast_vec(out.Eout_q);
  bcast_vec(out.polar_min);
  bcast_vec(out.polar_max);
  bcast_vec(out.cos_polar_q);
  bcast_vec(out.cos_azim_q);
  return true;
}

// ------------------------------------------------------------------------
// load_trim_sputter()
// ------------------------------------------------------------------------
double ProcessLibrary::TrimSputterTable::slice_yield(int ic, double E_eV,
                                                     double theta_deg) const
{
  // precondition: caller has checked valid() -- both public yield()
  // overloads do; this private path assumes loaded tables
  // Below the table's lowest energy = below threshold -> no sputtering
  // (clamping upward would fabricate yield near/below threshold for
  // sparse tables, e.g. d_on_w starting at 250 eV).
  if (E_eV < E.front()) return 0.0;
  const double le = std::log(std::min(E_eV, E.back()));
  const double a  = std::min(std::max(theta_deg, theta.front()), theta.back());
  int ie = int(std::lower_bound(E.begin(), E.end(), std::exp(le)) - E.begin());
  int ia = int(std::lower_bound(theta.begin(), theta.end(), a) - theta.begin());
  if (ie <= 0) ie = 1;
  if (ie >= NE) ie = NE - 1;
  if (ia <= 0) ia = 1;
  if (ia >= NTHETA) ia = NTHETA - 1;
  const double le1 = std::log(E[ie-1]), le2 = std::log(E[ie]);
  const double a1 = theta[ia-1], a2 = theta[ia];
  const double te = (le2 != le1) ? (le - le1) / (le2 - le1) : 0.0;
  const double ta = (a2 != a1) ? (a - a1) / (a2 - a1) : 0.0;
  const size_t off = size_t(ic) * NE * NTHETA;
  auto at = [&](int i, int j) { return Y[off + size_t(i) * NTHETA + j]; };
  double y = (1-te)*(1-ta)*at(ie-1,ia-1) + te*(1-ta)*at(ie,ia-1)
           + (1-te)*ta*at(ie-1,ia) + te*ta*at(ie,ia);
  return (std::isfinite(y) && y > 0.0) ? y : 0.0;
}

double ProcessLibrary::TrimSputterTable::yield(double E_eV,
                                               double theta_deg) const
{
  if (!valid()) return 0.0;
  return slice_yield(0, E_eV, theta_deg);
}

double ProcessLibrary::TrimSputterTable::yield(double E_eV, double theta_deg,
                                               double c) const
{
  if (!valid()) return 0.0;
  if (NC <= 1) return slice_yield(0, E_eV, theta_deg);
  const double cc = std::min(std::max(c, C.front()), C.back());
  int ic = int(std::lower_bound(C.begin(), C.end(), cc) - C.begin());
  if (ic <= 0) ic = 1;
  if (ic >= NC) ic = NC - 1;
  const double c1 = C[ic-1], c2 = C[ic];
  const double tc = (c2 != c1) ? (cc - c1) / (c2 - c1) : 0.0;
  return (1.0 - tc) * slice_yield(ic-1, E_eV, theta_deg)
       + tc * slice_yield(ic, E_eV, theta_deg);
}

bool ProcessLibrary::load_trim_sputter(const std::string &pair,
                                       TrimSputterTable &out)
{
  if (!opened_) return false;
  out = TrimSputterTable{};

  int present = 0;
  const std::string grp = "/surface/sputter/" + pair;
  if (me_ == 0) {
    try {
      H5ErrGuard _h5err;
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), grp.c_str(), H5P_DEFAULT) > 0 &&
          H5Lexists(f.getId(), (grp + "/Y").c_str(), H5P_DEFAULT) > 0) {
        auto read1 = [&](const std::string &p, std::vector<double> &v,
                         int &dim_out) {
          H5::DataSet d = f.openDataSet(p);
          hsize_t dim[1] = {0};
          d.getSpace().getSimpleExtentDims(dim, nullptr);
          dim_out = static_cast<int>(dim[0]);
          v.resize(dim[0]);
          d.read(v.data(), H5::PredType::NATIVE_DOUBLE);
        };
        read1(grp + "/E", out.E, out.NE);
        read1(grp + "/theta", out.theta, out.NTHETA);
        // optional composition axis -> Y is [NC x NE x NTHETA]
        if (H5Lexists(f.getId(), (grp + "/C").c_str(), H5P_DEFAULT) > 0)
          read1(grp + "/C", out.C, out.NC);
        H5::DataSet d = f.openDataSet(grp + "/Y");
        if (out.NC > 0) {
          hsize_t dims[3] = {0, 0, 0};
          d.getSpace().getSimpleExtentDims(dims, nullptr);
          if (static_cast<int>(dims[0]) == out.NC &&
              static_cast<int>(dims[1]) == out.NE &&
              static_cast<int>(dims[2]) == out.NTHETA) {
            out.Y.resize(size_t(out.NC) * out.NE * out.NTHETA);
            d.read(out.Y.data(), H5::PredType::NATIVE_DOUBLE);
            present = 1;
          }
        } else {
          hsize_t dims[2] = {0, 0};
          d.getSpace().getSimpleExtentDims(dims, nullptr);
          if (static_cast<int>(dims[0]) == out.NE &&
              static_cast<int>(dims[1]) == out.NTHETA) {
            out.Y.resize(size_t(out.NE) * out.NTHETA);
            d.read(out.Y.data(), H5::PredType::NATIVE_DOUBLE);
            present = 1;
          }
        }
        if (present) {
          H5::Group g = f.openGroup(grp);
          if (g.attrExists("Es_eV")) {
            H5::Attribute a = g.openAttribute("Es_eV");
            a.read(H5::PredType::NATIVE_DOUBLE, &out.Es);
          }
        }
      }
    } catch (const H5::Exception &) {
      present = 0;
    }
  }

  MPI_Bcast(&present, 1, MPI_INT, 0, comm_);
  if (!present) return false;
  int dims3[3] = {out.NE, out.NTHETA, out.NC};
  MPI_Bcast(dims3, 3, MPI_INT, 0, comm_);
  out.NE = dims3[0]; out.NTHETA = dims3[1]; out.NC = dims3[2];
  MPI_Bcast(&out.Es, 1, MPI_DOUBLE, 0, comm_);
  auto bcast_vec = [&](std::vector<double> &v, size_t n) {
    if (me_ != 0) v.resize(n);
    if (n > 0) MPI_Bcast(v.data(), static_cast<int>(n), MPI_DOUBLE, 0, comm_);
  };
  bcast_vec(out.E, out.NE);
  bcast_vec(out.theta, out.NTHETA);
  if (out.NC > 0) bcast_vec(out.C, out.NC);
  bcast_vec(out.Y, size_t(out.NC > 0 ? out.NC : 1) * out.NE * out.NTHETA);
  return true;
}

// ------------------------------------------------------------------------
// load_pec_line()
// ------------------------------------------------------------------------
bool ProcessLibrary::load_pec_line(const std::string &elem,
                                   const std::string &pec_id,
                                   const std::string &line_key,
                                   PecTable &out)
{
  if (!opened_) return false;
  out = PecTable{};

  const std::string grp = "/volume/pec/" + elem + "/" + pec_id +
                          "/" + line_key;
  int present = 0;
  if (me_ == 0) {
    try {
      H5ErrGuard _h5err;
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), grp.c_str(), H5P_DEFAULT) > 0) {
        H5::DataSet dc = f.openDataSet(grp + "/coefficient");
        H5::DataSpace spc = dc.getSpace();
        hsize_t dims2[2] = {0, 0};
        spc.getSimpleExtentDims(dims2, nullptr);
        out.nT  = static_cast<int>(dims2[0]);
        out.nNe = static_cast<int>(dims2[1]);
        out.coef.resize(dims2[0] * dims2[1]);
        dc.read(out.coef.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet dt = f.openDataSet(grp + "/temperature");
        hsize_t dim1[1] = {0};
        dt.getSpace().getSimpleExtentDims(dim1, nullptr);
        out.logT.resize(dim1[0]);
        dt.read(out.logT.data(), H5::PredType::NATIVE_DOUBLE);

        H5::DataSet dn = f.openDataSet(grp + "/density");
        dn.getSpace().getSimpleExtentDims(dim1, nullptr);
        out.logN.resize(dim1[0]);
        dn.read(out.logN.data(), H5::PredType::NATIVE_DOUBLE);

        present = 1;
      }
    } catch (const H5::Exception &) {
      present = 0;
    }
  }

  MPI_Bcast(&present, 1, MPI_INT, 0, comm_);
  if (!present) return false;

  int dims2[2] = {out.nT, out.nNe};
  MPI_Bcast(dims2, 2, MPI_INT, 0, comm_);
  out.nT = dims2[0]; out.nNe = dims2[1];
  const size_t n = static_cast<size_t>(out.nT) * out.nNe;
  if (me_ != 0) out.coef.resize(n);
  if (n > 0) MPI_Bcast(out.coef.data(), n, MPI_DOUBLE, 0, comm_);

  int nt = static_cast<int>(out.logT.size());
  MPI_Bcast(&nt, 1, MPI_INT, 0, comm_);
  if (me_ != 0) out.logT.resize(nt);
  if (nt > 0) MPI_Bcast(out.logT.data(), nt, MPI_DOUBLE, 0, comm_);

  int nn = static_cast<int>(out.logN.size());
  MPI_Bcast(&nn, 1, MPI_INT, 0, comm_);
  if (me_ != 0) out.logN.resize(nn);
  if (nn > 0) MPI_Bcast(out.logN.data(), nn, MPI_DOUBLE, 0, comm_);
  return true;
}

// ------------------------------------------------------------------------
// load_reactions_catalog()
// ------------------------------------------------------------------------
bool ProcessLibrary::load_reactions_catalog(const std::string &elem,
                                            std::string &out)
{
  if (!opened_) return false;
  out.clear();

  const std::string grp = "/volume/reactions/" + elem + "/catalog";
  int present = 0;
  int len = 0;
  std::string local_out;

  if (me_ == 0) {
    try {
      H5ErrGuard _h5err;
      H5::H5File f(path_, H5F_ACC_RDONLY);
      if (H5Lexists(f.getId(), grp.c_str(), H5P_DEFAULT) > 0) {
        H5::DataSet ds = f.openDataSet(grp);
        // Stored by build_processes_h5.py as a scalar opaque string.
        H5::StrType stype = ds.getStrType();
        H5::DataSpace sp  = ds.getSpace();
        if (sp.getSimpleExtentNdims() == 0) {
          std::string tmp;
          ds.read(tmp, stype);
          local_out = std::move(tmp);
          present = 1;
          len = static_cast<int>(local_out.size());
        }
      }
    } catch (const H5::Exception &) {
      present = 0;
    }
  }

  MPI_Bcast(&present, 1, MPI_INT, 0, comm_);
  if (!present) return false;
  MPI_Bcast(&len, 1, MPI_INT, 0, comm_);
  std::vector<char> buf(len + 1, 0);
  if (me_ == 0 && len > 0) std::memcpy(buf.data(), local_out.data(), len);
  MPI_Bcast(buf.data(), len + 1, MPI_CHAR, 0, comm_);
  out.assign(buf.data(), len);
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
      H5ErrGuard _h5err;
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
      H5ErrGuard _h5err;
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
