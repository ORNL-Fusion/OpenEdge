/* ----------------------------------------------------------------------
   OpenEdge IEAD lookup table — implementation.
------------------------------------------------------------------------- */

#include "iead_table.h"

#include <H5Cpp.h>
#include <algorithm>
#include <stdexcept>

namespace SPARTA_NS {

namespace {

void read_1d_double(H5::H5File &f, const std::string &name,
                    std::vector<double> &out)
{
  H5::DataSet ds = f.openDataSet(name);
  H5::DataSpace sp = ds.getSpace();
  hsize_t d0 = 0;
  sp.getSimpleExtentDims(&d0);
  out.resize(static_cast<size_t>(d0));
  ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
}

void read_1d_int(H5::H5File &f, const std::string &name,
                 std::vector<int> &out)
{
  H5::DataSet ds = f.openDataSet(name);
  H5::DataSpace sp = ds.getSpace();
  hsize_t d0 = 0;
  sp.getSimpleExtentDims(&d0);
  out.resize(static_cast<size_t>(d0));
  // Tolerate either int32 or int64 storage on disk.
  H5::DataType dt = ds.getDataType();
  if (dt.getSize() == sizeof(long long)) {
    std::vector<long long> tmp(d0);
    ds.read(tmp.data(), H5::PredType::NATIVE_LLONG);
    for (size_t i = 0; i < out.size(); ++i) out[i] = static_cast<int>(tmp[i]);
  } else {
    ds.read(out.data(), H5::PredType::NATIVE_INT);
  }
}

double read_scalar_attr(H5::H5File &f, const std::string &name, double fallback)
{
  if (!f.attrExists(name)) return fallback;
  H5::Attribute a = f.openAttribute(name);
  double v = fallback;
  a.read(H5::PredType::NATIVE_DOUBLE, &v);
  return v;
}

}  // namespace

void IeadTable::load(const std::string &path)
{
  H5::H5File file(path, H5F_ACC_RDONLY);

  read_1d_double(file, "tau_grid",            tau_grid);
  read_1d_double(file, "psi_grid_deg",        psi_grid_deg);
  read_1d_int   (file, "Z_grid",              Z_grid);
  read_1d_double(file, "Etilde_bin_edges",    Etilde_edges);
  read_1d_double(file, "theta_bin_edges_deg", theta_edges_deg);

  n_tau   = static_cast<int>(tau_grid.size());
  n_psi   = static_cast<int>(psi_grid_deg.size());
  n_Z     = static_cast<int>(Z_grid.size());
  n_E     = static_cast<int>(Etilde_edges.size()) - 1;
  n_theta = static_cast<int>(theta_edges_deg.size()) - 1;

  if (n_tau < 2 || n_psi < 2 || n_Z < 1 || n_E < 1 || n_theta < 1)
    throw std::runtime_error("IeadTable: degenerate axes in " + path);

  // f shape: (n_tau, n_psi, n_Z, n_E, n_theta)
  H5::DataSet ds = file.openDataSet("f");
  H5::DataSpace sp = ds.getSpace();
  hsize_t dims[5] = {0,0,0,0,0};
  if (sp.getSimpleExtentNdims() != 5)
    throw std::runtime_error("IeadTable: f must be 5-D in " + path);
  sp.getSimpleExtentDims(dims);
  if (static_cast<int>(dims[0]) != n_tau ||
      static_cast<int>(dims[1]) != n_psi ||
      static_cast<int>(dims[2]) != n_Z   ||
      static_cast<int>(dims[3]) != n_E   ||
      static_cast<int>(dims[4]) != n_theta)
    throw std::runtime_error("IeadTable: f shape inconsistent with axes in " + path);

  size_t ntot = size_t(n_tau) * n_psi * n_Z * n_E * n_theta;
  f.resize(ntot);
  ds.read(f.data(), H5::PredType::NATIVE_FLOAT);

  Te_eV_sweep = read_scalar_attr(file, "Te_eV", 0.0);
  mi_bg_amu   = read_scalar_attr(file, "mi_bg_amu", 0.0);
}

int IeadTable::find_z_slot(int Z) const
{
  for (int k = 0; k < n_Z; ++k)
    if (Z_grid[k] == Z) return k;
  return -1;
}

void IeadTable::locate_bilinear(double tau, double psi_deg,
                                int &i0, int &i1, double &wt0, double &wt1,
                                int &j0, int &j1, double &wp0, double &wp1) const
{
  // Clamp tau to grid range
  double t = std::max(tau_grid.front(), std::min(tau_grid.back(), tau));
  i0 = 0;
  while (i0 + 1 < n_tau && tau_grid[i0 + 1] <= t) ++i0;
  if (i0 >= n_tau - 1) i0 = n_tau - 2;
  i1 = i0 + 1;
  double dt = tau_grid[i1] - tau_grid[i0];
  wt1 = (dt > 0.0) ? (t - tau_grid[i0]) / dt : 0.0;
  wt0 = 1.0 - wt1;

  double p = std::max(psi_grid_deg.front(), std::min(psi_grid_deg.back(), psi_deg));
  j0 = 0;
  while (j0 + 1 < n_psi && psi_grid_deg[j0 + 1] <= p) ++j0;
  if (j0 >= n_psi - 1) j0 = n_psi - 2;
  j1 = j0 + 1;
  double dp = psi_grid_deg[j1] - psi_grid_deg[j0];
  wp1 = (dp > 0.0) ? (p - psi_grid_deg[j0]) / dp : 0.0;
  wp0 = 1.0 - wp1;
}

double IeadTable::convolve_yield(double tau, double psi_deg, int Z,
                                 double Te_local,
                                 const std::function<double(double, double)> &Y) const
{
  const int kZ = find_z_slot(Z);
  if (kZ < 0) return 0.0;
  if (Te_local <= 0.0) return 0.0;

  int i0, i1, j0, j1;
  double wt0, wt1, wp0, wp1;
  locate_bilinear(tau, psi_deg, i0, i1, wt0, wt1, j0, j1, wp0, wp1);

  // Strides into flat f: index = (((i*n_psi + j)*n_Z + k)*n_E + e)*n_theta + a
  const size_t stride_tau   = size_t(n_psi) * n_Z * n_E * n_theta;
  const size_t stride_psi   = size_t(n_Z)   * n_E * n_theta;
  const size_t stride_z     = size_t(n_E)   * n_theta;
  const size_t stride_e     = size_t(n_theta);

  const size_t base00 = i0 * stride_tau + j0 * stride_psi + kZ * stride_z;
  const size_t base01 = i0 * stride_tau + j1 * stride_psi + kZ * stride_z;
  const size_t base10 = i1 * stride_tau + j0 * stride_psi + kZ * stride_z;
  const size_t base11 = i1 * stride_tau + j1 * stride_psi + kZ * stride_z;

  const double w00 = wt0 * wp0;
  const double w01 = wt0 * wp1;
  const double w10 = wt1 * wp0;
  const double w11 = wt1 * wp1;

  double sumY = 0.0;
  const double Z_Te = static_cast<double>(Z) * Te_local;

  for (int e = 0; e < n_E; ++e) {
    const double Et_c = 0.5 * (Etilde_edges[e] + Etilde_edges[e + 1]);
    const double E_eV = Et_c * Z_Te;
    for (int a = 0; a < n_theta; ++a) {
      const double th_c = 0.5 * (theta_edges_deg[a] + theta_edges_deg[a + 1]);
      const size_t off = e * stride_e + a;
      const double fE = w00 * f[base00 + off] + w01 * f[base01 + off]
                      + w10 * f[base10 + off] + w11 * f[base11 + off];
      if (fE <= 0.0) continue;
      sumY += fE * Y(E_eV, th_c);
    }
  }
  return sumY;
}

}
