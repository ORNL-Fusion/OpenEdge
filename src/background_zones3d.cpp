/* ----------------------------------------------------------------------
   OpenEdge native 3-D curvilinear plasma-background provider
------------------------------------------------------------------------- */

#include "background_zones3d.h"

#include "fix_background.h"

#include <H5Cpp.h>
#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

using namespace SPARTA_NS;

namespace {

constexpr double QE = 1.602176634e-19;
constexpr double AMU = 1.66053906660e-27;
constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double TWO_PI = 2.0 * PI;

bool has_link(hid_t parent, const std::string &name)
{
  return H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0;
}

bool has_attr(hid_t parent, const std::string &name)
{
  return H5Aexists(parent, name.c_str()) > 0;
}

std::string read_string_attr(const H5::H5Object &obj,
                             const std::string &name)
{
  if (!has_attr(obj.getId(), name))
    throw std::runtime_error("background zones3d: missing attribute '" + name + "'");
  H5::Attribute attr = obj.openAttribute(name);
  H5::StrType type = attr.getStrType();
  std::string value;
  attr.read(type, value);
  return value;
}

long long read_int_attr(const H5::H5Object &obj, const std::string &name)
{
  if (!has_attr(obj.getId(), name))
    throw std::runtime_error("background zones3d: missing attribute '" + name + "'");
  long long value = 0;
  obj.openAttribute(name).read(H5::PredType::NATIVE_LLONG, &value);
  return value;
}

double read_double_attr(const H5::H5Object &obj, const std::string &name,
                        double fallback)
{
  if (!has_attr(obj.getId(), name)) return fallback;
  double value = fallback;
  obj.openAttribute(name).read(H5::PredType::NATIVE_DOUBLE, &value);
  return value;
}

std::vector<hsize_t> dataset_shape(const H5::DataSet &ds)
{
  H5::DataSpace space = ds.getSpace();
  const int rank = space.getSimpleExtentNdims();
  if (rank < 0) throw std::runtime_error("background zones3d: invalid dataset rank");
  std::vector<hsize_t> dims(rank);
  if (rank) space.getSimpleExtentDims(dims.data());
  return dims;
}

size_t shape_size(const std::vector<hsize_t> &dims)
{
  size_t n = 1;
  for (hsize_t d : dims) n *= static_cast<size_t>(d);
  return n;
}

void require_shape(const std::string &name, const std::vector<hsize_t> &got,
                   const std::vector<hsize_t> &expected)
{
  if (got == expected) return;
  throw std::runtime_error("background zones3d: dataset '" + name +
                           "' has an unexpected shape");
}

void read_float_dataset(const H5::Group &group, const std::string &name,
                        const std::vector<hsize_t> &shape,
                        std::vector<float> &out)
{
  if (!has_link(group.getId(), name))
    throw std::runtime_error("background zones3d: missing dataset '" + name + "'");
  H5::DataSet ds = group.openDataSet(name);
  require_shape(name, dataset_shape(ds), shape);
  out.resize(shape_size(shape));
  ds.read(out.data(), H5::PredType::NATIVE_FLOAT);
}

void read_double_dataset(const H5::Group &group, const std::string &name,
                         const std::vector<hsize_t> &shape,
                         std::vector<double> &out)
{
  if (!has_link(group.getId(), name))
    throw std::runtime_error("background zones3d: missing dataset '" + name + "'");
  H5::DataSet ds = group.openDataSet(name);
  require_shape(name, dataset_shape(ds), shape);
  out.resize(shape_size(shape));
  ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
}

void read_u8_dataset(const H5::Group &group, const std::string &name,
                     const std::vector<hsize_t> &shape,
                     std::vector<uint8_t> &out)
{
  if (!has_link(group.getId(), name))
    throw std::runtime_error("background zones3d: missing dataset '" + name + "'");
  H5::DataSet ds = group.openDataSet(name);
  require_shape(name, dataset_shape(ds), shape);
  out.resize(shape_size(shape));
  ds.read(out.data(), H5::PredType::NATIVE_UINT8);
}

void bcast_string(std::string &value, MPI_Comm world, int me)
{
  int n = me == 0 ? static_cast<int>(value.size()) : 0;
  MPI_Bcast(&n, 1, MPI_INT, 0, world);
  if (me != 0) value.assign(n, '\0');
  if (n) MPI_Bcast(&value[0], n, MPI_CHAR, 0, world);
}

template <class T>
void bcast_vector(std::vector<T> &value, MPI_Datatype dtype,
                  MPI_Comm world, int me)
{
  unsigned long long n = me == 0
    ? static_cast<unsigned long long>(value.size()) : 0ULL;
  MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
  if (me != 0) value.resize(static_cast<size_t>(n));
  size_t offset = 0;
  while (offset < value.size()) {
    const size_t remaining = value.size() - offset;
    const int count = static_cast<int>(std::min<size_t>(remaining, 100000000));
    MPI_Bcast(value.data() + offset, count, dtype, 0, world);
    offset += count;
  }
}

int clamp_bin(int i, int n)
{
  return std::max(0, std::min(i, n - 1));
}

}  // namespace

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::clear()
{
  zones_.clear();
  sectors_.clear();
  total_cells_ = 0;
  ion_mass_amu_ = 2.014;
  counters_ = Counters{};
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::configure(int flow_sign, int b_sign,
                                  bool cs_te_plus_ti)
{
  if (flow_sign != -1 && flow_sign != 1)
    throw std::runtime_error(
      "background zones3d: flow_sign must be +1 or -1; signed Mach cannot "
      "be converted to a vector without an explicit convention");
  if (b_sign < -1 || b_sign > 1)
    throw std::runtime_error("background zones3d: b_sign must be -1, 0, or +1");
  if (!cs_te_plus_ti)
    throw std::runtime_error(
      "background zones3d: cs_model must be te_plus_ti for Mach conversion");
  flow_sign_ = flow_sign;
  b_sign_ = b_sign;
  cs_te_plus_ti_ = cs_te_plus_ti;
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::load(const std::string &path, MPI_Comm world, int me)
{
  clear();
  std::string load_error;
  if (me == 0) {
    try {
      read_rank0(path);
    } catch (const H5::Exception &e) {
      load_error = "background zones3d: HDF5 error while reading '" + path +
                   "': " + e.getDetailMsg();
    } catch (const std::exception &e) {
      load_error = e.what();
    } catch (...) {
      load_error = "background zones3d: unknown error while reading '" + path + "'";
    }
  }
  bcast_string(load_error, world, me);
  if (!load_error.empty()) throw std::runtime_error(load_error);
  broadcast(world, me);
  build_indices();
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::read_rank0(const std::string &path)
{
  H5::H5File file(path, H5F_ACC_RDONLY);
  const std::string format = read_string_attr(file, "format");
  if (format != "openedge_background3d")
    throw std::runtime_error(
      "background zones3d: root attribute format must be "
      "'openedge_background3d'");
  if (read_int_attr(file, "version") != 1)
    throw std::runtime_error("background zones3d: only schema version 1 is supported");
  if (read_string_attr(file, "length_units") != "m" ||
      read_string_attr(file, "phi_units") != "rad")
    throw std::runtime_error("background zones3d: expected length=m and phi=rad");

  ion_mass_amu_ = read_double_attr(file, "ion_mass_amu", 2.014);
  if (!(ion_mass_amu_ > 0.0))
    throw std::runtime_error("background zones3d: ion_mass_amu must be positive");

  H5::Group zones_group = file.openGroup("zones");
  const hsize_t nobj = zones_group.getNumObjs();
  std::vector<std::string> names;
  for (hsize_t i = 0; i < nobj; ++i) {
    const std::string name = zones_group.getObjnameByIdx(i);
    if (name.rfind("zone_", 0) == 0) names.push_back(name);
  }
  std::sort(names.begin(), names.end());
  if (names.empty())
    throw std::runtime_error("background zones3d: /zones contains no zone groups");

  zones_.reserve(names.size());
  for (const std::string &name : names) {
    H5::Group group = zones_group.openGroup(name);
    Zone zone;
    zone.zid = static_cast<int>(read_int_attr(group, "zone"));
    zone.sector = static_cast<int>(read_int_attr(group, "sector"));
    zone.topology = read_string_attr(group, "topology");
    if (zone.topology == "SOL") zone.topology_priority = 0;
    else if (zone.topology == "PFR") zone.topology_priority = 1;
    else if (zone.topology == "CORE") zone.topology_priority = 2;
    else throw std::runtime_error("background zones3d: unknown topology '" +
                                  zone.topology + "'");

    H5::DataSet rds = group.openDataSet("vertex_r");
    const std::vector<hsize_t> vshape = dataset_shape(rds);
    if (vshape.size() != 3 || vshape[0] < 2 || vshape[1] < 2 || vshape[2] < 2)
      throw std::runtime_error("background zones3d: vertex arrays must be 3-D and at least 2x2x2");
    zone.nr = static_cast<int>(vshape[0]);
    zone.np = static_cast<int>(vshape[1]);
    zone.nt = static_cast<int>(vshape[2]);
    zone.ncr = zone.nr - 1;
    zone.ncp = zone.np - 1;
    zone.nct = zone.nt - 1;
    zone.vertex_r.resize(shape_size(vshape));
    rds.read(zone.vertex_r.data(), H5::PredType::NATIVE_FLOAT);
    read_float_dataset(group, "vertex_z", vshape, zone.vertex_z);
    read_double_dataset(group, "phi", {vshape[2]}, zone.phi);
    if (!std::is_sorted(zone.phi.begin(), zone.phi.end()) ||
        std::adjacent_find(zone.phi.begin(), zone.phi.end(),
                           std::greater_equal<double>()) != zone.phi.end())
      throw std::runtime_error("background zones3d: phi planes must be strictly increasing");
    zone.phi_start = zone.phi.front();
    zone.phi_end = zone.phi.back();

    const std::vector<hsize_t> cshape = {
      static_cast<hsize_t>(zone.ncr), static_cast<hsize_t>(zone.ncp),
      static_cast<hsize_t>(zone.nct)};
    read_u8_dataset(group, "cell_valid", cshape, zone.valid);
    H5::Group cell = group.openGroup("cell");
    read_float_dataset(cell, "te_eV", cshape, zone.te);
    read_float_dataset(cell, "ti_eV", cshape, zone.ti);
    read_float_dataset(cell, "ne_m3", cshape, zone.ne);
    read_float_dataset(cell, "ni_m3", cshape, zone.ni);
    read_float_dataset(cell, "mach", cshape, zone.mach);
    read_float_dataset(cell, "b_mag_T", cshape, zone.bmag);
    read_float_dataset(cell, "trace_r", cshape, zone.trace_r);
    read_float_dataset(cell, "trace_phi", cshape, zone.trace_phi);
    read_float_dataset(cell, "trace_z", cshape, zone.trace_z);
    zones_.push_back(std::move(zone));
  }
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::broadcast(MPI_Comm world, int me)
{
  MPI_Bcast(&ion_mass_amu_, 1, MPI_DOUBLE, 0, world);
  int nzones = me == 0 ? static_cast<int>(zones_.size()) : 0;
  MPI_Bcast(&nzones, 1, MPI_INT, 0, world);
  if (me != 0) zones_.resize(nzones);

  for (Zone &zone : zones_) {
    int meta[8] = {zone.zid, zone.sector, zone.topology_priority,
                   zone.nr, zone.np, zone.nt, zone.ncr, zone.ncp};
    MPI_Bcast(meta, 8, MPI_INT, 0, world);
    if (me != 0) {
      zone.zid = meta[0];
      zone.sector = meta[1];
      zone.topology_priority = meta[2];
      zone.nr = meta[3]; zone.np = meta[4]; zone.nt = meta[5];
      zone.ncr = meta[6]; zone.ncp = meta[7];
      zone.nct = zone.nt - 1;
    }
    bcast_string(zone.topology, world, me);
    bcast_vector(zone.phi, MPI_DOUBLE, world, me);
    bcast_vector(zone.vertex_r, MPI_FLOAT, world, me);
    bcast_vector(zone.vertex_z, MPI_FLOAT, world, me);
    bcast_vector(zone.valid, MPI_UNSIGNED_CHAR, world, me);
    bcast_vector(zone.te, MPI_FLOAT, world, me);
    bcast_vector(zone.ti, MPI_FLOAT, world, me);
    bcast_vector(zone.ne, MPI_FLOAT, world, me);
    bcast_vector(zone.ni, MPI_FLOAT, world, me);
    bcast_vector(zone.mach, MPI_FLOAT, world, me);
    bcast_vector(zone.bmag, MPI_FLOAT, world, me);
    bcast_vector(zone.trace_r, MPI_FLOAT, world, me);
    bcast_vector(zone.trace_phi, MPI_FLOAT, world, me);
    bcast_vector(zone.trace_z, MPI_FLOAT, world, me);
    zone.phi_start = zone.phi.front();
    zone.phi_end = zone.phi.back();
  }
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::build_indices()
{
  std::sort(zones_.begin(), zones_.end(),
            [](const Zone &a, const Zone &b) { return a.zid < b.zid; });
  total_cells_ = 0;
  for (Zone &zone : zones_) {
    zone.global_cell_offset = total_cells_;
    total_cells_ += static_cast<int64_t>(zone.ncr) * zone.ncp * zone.nct;
    zone.hashes.resize(zone.nct);
    for (int it = 0; it < zone.nct; ++it) build_hash(zone, it);
  }

  std::map<int, Sector> by_sector;
  for (int iz = 0; iz < static_cast<int>(zones_.size()); ++iz) {
    const Zone &zone = zones_[iz];
    Sector &sector = by_sector[zone.sector];
    if (sector.zones.empty()) {
      sector.sid = zone.sector;
      sector.phi_start = zone.phi_start;
      sector.phi_end = zone.phi_end;
    } else if (std::abs(sector.phi_start - zone.phi_start) > 1.0e-10 ||
               std::abs(sector.phi_end - zone.phi_end) > 1.0e-10) {
      throw std::runtime_error(
        "background zones3d: zones in one sector have inconsistent phi extents");
    }
    sector.zones.push_back(iz);
  }
  sectors_.clear();
  for (auto &item : by_sector) sectors_.push_back(std::move(item.second));
  std::sort(sectors_.begin(), sectors_.end(),
            [](const Sector &a, const Sector &b) {
              return a.phi_start < b.phi_start;
            });
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::build_hash(Zone &zone, int it)
{
  HashPlane &hash = zone.hashes[it];
  double rmin = std::numeric_limits<double>::infinity();
  double rmax = -rmin;
  double zmin = rmin;
  double zmax = -rmin;
  for (int ir = 0; ir < zone.nr; ++ir) {
    for (int ip = 0; ip < zone.np; ++ip) {
      for (int jt = it; jt <= it + 1; ++jt) {
        const size_t v = zone.vertex_index(ir, ip, jt);
        rmin = std::min(rmin, static_cast<double>(zone.vertex_r[v]));
        rmax = std::max(rmax, static_cast<double>(zone.vertex_r[v]));
        zmin = std::min(zmin, static_cast<double>(zone.vertex_z[v]));
        zmax = std::max(zmax, static_cast<double>(zone.vertex_z[v]));
      }
    }
  }
  const double rspan = std::max(rmax - rmin, 1.0e-6);
  const double zspan = std::max(zmax - zmin, 1.0e-6);
  const double rpad = 1.0e-7 + 1.0e-6 * rspan;
  const double zpad = 1.0e-7 + 1.0e-6 * zspan;
  rmin -= rpad; rmax += rpad;
  zmin -= zpad; zmax += zpad;

  const int ncell2d = zone.ncr * zone.ncp;
  const int target_bins = std::max(16, std::min(4096, ncell2d / 6));
  const double aspect = std::max(0.1, std::min(10.0, rspan / zspan));
  hash.nr = std::max(4, std::min(96,
    static_cast<int>(std::round(std::sqrt(target_bins * aspect)))));
  hash.nz = std::max(4, std::min(96,
    static_cast<int>(std::ceil(static_cast<double>(target_bins) / hash.nr))));
  hash.rmin = static_cast<float>(rmin);
  hash.zmin = static_cast<float>(zmin);
  hash.dr = static_cast<float>((rmax - rmin) / hash.nr);
  hash.dz = static_cast<float>((zmax - zmin) / hash.nz);

  const int nbins = hash.nr * hash.nz;
  std::vector<int> counts(nbins, 0);
  auto cell_bins = [&](int ir, int ip, int &br0, int &br1,
                       int &bz0, int &bz1) {
    double crmin = std::numeric_limits<double>::infinity();
    double crmax = -crmin;
    double czmin = crmin;
    double czmax = -crmin;
    for (int dr = 0; dr <= 1; ++dr) {
      for (int dp = 0; dp <= 1; ++dp) {
        for (int dt = 0; dt <= 1; ++dt) {
          const size_t v = zone.vertex_index(ir + dr, ip + dp, it + dt);
          crmin = std::min(crmin, static_cast<double>(zone.vertex_r[v]));
          crmax = std::max(crmax, static_cast<double>(zone.vertex_r[v]));
          czmin = std::min(czmin, static_cast<double>(zone.vertex_z[v]));
          czmax = std::max(czmax, static_cast<double>(zone.vertex_z[v]));
        }
      }
    }
    br0 = clamp_bin(static_cast<int>((crmin - hash.rmin) / hash.dr), hash.nr);
    br1 = clamp_bin(static_cast<int>((crmax - hash.rmin) / hash.dr), hash.nr);
    bz0 = clamp_bin(static_cast<int>((czmin - hash.zmin) / hash.dz), hash.nz);
    bz1 = clamp_bin(static_cast<int>((czmax - hash.zmin) / hash.dz), hash.nz);
  };

  for (int ir = 0; ir < zone.ncr; ++ir) {
    for (int ip = 0; ip < zone.ncp; ++ip) {
      int br0, br1, bz0, bz1;
      cell_bins(ir, ip, br0, br1, bz0, bz1);
      for (int bz = bz0; bz <= bz1; ++bz)
        for (int br = br0; br <= br1; ++br)
          ++counts[bz * hash.nr + br];
    }
  }
  hash.offsets.resize(nbins + 1, 0);
  std::partial_sum(counts.begin(), counts.end(), hash.offsets.begin() + 1);
  hash.cells.resize(hash.offsets.back());
  std::vector<int> cursor = hash.offsets;
  for (int ir = 0; ir < zone.ncr; ++ir) {
    for (int ip = 0; ip < zone.ncp; ++ip) {
      int br0, br1, bz0, bz1;
      cell_bins(ir, ip, br0, br1, bz0, bz1);
      const int cell2d = ir * zone.ncp + ip;
      for (int bz = bz0; bz <= bz1; ++bz) {
        for (int br = br0; br <= br1; ++br) {
          const int bin = bz * hash.nr + br;
          hash.cells[cursor[bin]++] = cell2d;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

double BackgroundZones3D::wrap_phi(double phi)
{
  if (phi >= -PI && phi < PI) return phi;
  phi = std::fmod(phi + PI, TWO_PI);
  if (phi < 0.0) phi += TWO_PI;
  return phi - PI;
}

/* ---------------------------------------------------------------------- */

int BackgroundZones3D::sector_for(double phi) const
{
  if (sectors_.empty()) return -1;
  auto it = std::upper_bound(
    sectors_.begin(), sectors_.end(), phi,
    [](double p, const Sector &s) { return p < s.phi_start; });
  if (it == sectors_.begin()) return -1;
  --it;
  if (phi < it->phi_start || phi >= it->phi_end) return -1;
  return static_cast<int>(it - sectors_.begin());
}

/* ---------------------------------------------------------------------- */

bool BackgroundZones3D::point_in_quad(double x, double y,
                                      const double qx[4], const double qy[4])
{
  const double scale = 1.0 + std::abs(x) + std::abs(y);
  const double tol = 2.0e-11 * scale;
  for (int e = 0; e < 4; ++e) {
    const int f = (e + 1) & 3;
    const double dx = qx[f] - qx[e];
    const double dy = qy[f] - qy[e];
    const double cross = (x - qx[e]) * dy - (y - qy[e]) * dx;
    const double dot = (x - qx[e]) * dx + (y - qy[e]) * dy;
    const double len2 = dx * dx + dy * dy;
    if (std::abs(cross) <= tol * std::sqrt(len2) &&
        dot >= -tol && dot <= len2 + tol) return true;
  }

  bool inside = false;
  for (int e = 0; e < 4; ++e) {
    const int f = (e + 1) & 3;
    if ((qy[e] > y) != (qy[f] > y)) {
      const double xcross = (qx[f] - qx[e]) * (y - qy[e]) /
                            (qy[f] - qy[e]) + qx[e];
      if (x < xcross) inside = !inside;
    }
  }
  return inside;
}

/* ---------------------------------------------------------------------- */

bool BackgroundZones3D::locate_in_zone(const Zone &zone, double R, double Z,
                                       double phi, int &ir, int &ip,
                                       int &it) const
{
  auto plane = std::upper_bound(zone.phi.begin(), zone.phi.end(), phi);
  if (plane == zone.phi.begin() || plane == zone.phi.end()) return false;
  it = static_cast<int>(plane - zone.phi.begin()) - 1;
  const double dphi = zone.phi[it + 1] - zone.phi[it];
  const double t = std::max(0.0, std::min(1.0,
                         (phi - zone.phi[it]) / dphi));
  const HashPlane &hash = zone.hashes[it];
  const int br = static_cast<int>((R - hash.rmin) / hash.dr);
  const int bz = static_cast<int>((Z - hash.zmin) / hash.dz);
  if (br < 0 || br >= hash.nr || bz < 0 || bz >= hash.nz) return false;
  const int bin = bz * hash.nr + br;

  for (int k = hash.offsets[bin]; k < hash.offsets[bin + 1]; ++k) {
    const int cell2d = hash.cells[k];
    const int cir = cell2d / zone.ncp;
    const int cip = cell2d % zone.ncp;
    double qR[4], qZ[4];
    const int vr[4] = {cir, cir + 1, cir + 1, cir};
    const int vp[4] = {cip, cip, cip + 1, cip + 1};
    for (int v = 0; v < 4; ++v) {
      const size_t i0 = zone.vertex_index(vr[v], vp[v], it);
      const size_t i1 = zone.vertex_index(vr[v], vp[v], it + 1);
      qR[v] = (1.0 - t) * zone.vertex_r[i0] + t * zone.vertex_r[i1];
      qZ[v] = (1.0 - t) * zone.vertex_z[i0] + t * zone.vertex_z[i1];
    }
    if (point_in_quad(R, Z, qR, qZ)) {
      ir = cir;
      ip = cip;
      return true;
    }
  }
  return false;
}

/* ---------------------------------------------------------------------- */

BackgroundZones3D::Location
BackgroundZones3D::locate(double R, double Z, double phi) const
{
  Location loc;
  const int isector = sector_for(phi);
  if (isector < 0) return loc;
  int best_priority = 100;
  int best_zid = std::numeric_limits<int>::max();
  for (int izone : sectors_[isector].zones) {
    const Zone &zone = zones_[izone];
    int ir = -1, ip = -1, it = -1;
    if (!locate_in_zone(zone, R, Z, phi, ir, ip, it)) continue;
    ++loc.nclaims;
    if (zone.topology_priority < best_priority ||
        (zone.topology_priority == best_priority && zone.zid < best_zid)) {
      best_priority = zone.topology_priority;
      best_zid = zone.zid;
      loc.zone_index = izone;
      loc.ir = ir; loc.ip = ip; loc.it = it;
    }
  }
  if (loc.zone_index < 0) return loc;
  const Zone &zone = zones_[loc.zone_index];
  const size_t c = zone.cell_index(loc.ir, loc.ip, loc.it);
  if (!zone.valid[c]) loc.status = PLASMA_SAMPLE_INVALID;
  else if (loc.nclaims > 1) loc.status = PLASMA_SAMPLE_AMBIGUOUS;
  else loc.status = PLASMA_SAMPLE_OK;
  return loc;
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::cell_center_xyz(const Zone &zone, int ir, int ip,
                                        int it, double xyz[3]) const
{
  double r = 0.0, z = 0.0;
  for (int dr = 0; dr <= 1; ++dr) {
    for (int dp = 0; dp <= 1; ++dp) {
      for (int dt = 0; dt <= 1; ++dt) {
        const size_t v = zone.vertex_index(ir + dr, ip + dp, it + dt);
        r += zone.vertex_r[v];
        z += zone.vertex_z[v];
      }
    }
  }
  r *= 0.125;
  z *= 0.125;
  const double phi = 0.5 * (zone.phi[it] + zone.phi[it + 1]);
  xyz[0] = r * std::cos(phi);
  xyz[1] = r * std::sin(phi);
  xyz[2] = z;
}

/* ---------------------------------------------------------------------- */

void BackgroundZones3D::trace_cartesian(const Zone &zone, size_t cell,
                                         double phi, double trace[3]) const
{
  const double er = zone.trace_r[cell];
  const double ep = zone.trace_phi[cell];
  const double ez = zone.trace_z[cell];
  const double cp = std::cos(phi), sp = std::sin(phi);
  trace[0] = er * cp - ep * sp;
  trace[1] = er * sp + ep * cp;
  trace[2] = ez;
  const double norm = std::sqrt(trace[0] * trace[0] +
                                trace[1] * trace[1] +
                                trace[2] * trace[2]);
  if (norm > 1.0e-30) {
    trace[0] /= norm;
    trace[1] /= norm;
    trace[2] /= norm;
  }
}

/* ----------------------------------------------------------------------
   Field-aligned temperature gradient at one native EMC3 cell.

   The carrier is field-line aligned: adjacent toroidal cells at fixed
   (ir,ip) lie along the stored trace.  A centred difference is used in the
   sector interior and a one-sided difference at sector boundaries.  The
   returned Cartesian vector is grad_parallel(T) = e_trace dT/ds; it is
   independent of the arbitrary absolute magnetic-field sign.
------------------------------------------------------------------------- */

void BackgroundZones3D::parallel_gradient(const Zone &zone, int ir, int ip,
                                          int it,
                                          const std::vector<float> &field,
                                          double phi,
                                          double gradient[3]) const
{
  gradient[0] = gradient[1] = gradient[2] = 0.0;
  if (field.empty() || zone.nct < 2) return;

  const size_t c = zone.cell_index(ir, ip, it);
  double trace[3];
  trace_cartesian(zone, c, phi, trace);

  const bool have_left = it > 0 &&
    zone.valid[zone.cell_index(ir, ip, it - 1)];
  const bool have_right = it + 1 < zone.nct &&
    zone.valid[zone.cell_index(ir, ip, it + 1)];
  const int it0 = have_left ? it - 1 : it;
  const int it1 = have_right ? it + 1 : it;
  if (it0 == it1) return;

  double x0[3], x1[3];
  cell_center_xyz(zone, ir, ip, it0, x0);
  cell_center_xyz(zone, ir, ip, it1, x1);
  const double ds = (x1[0] - x0[0]) * trace[0] +
                    (x1[1] - x0[1]) * trace[1] +
                    (x1[2] - x0[2]) * trace[2];
  if (std::abs(ds) < 1.0e-12) return;

  const double f0 = field[zone.cell_index(ir, ip, it0)];
  const double f1 = field[zone.cell_index(ir, ip, it1)];
  const double dfield_ds = (f1 - f0) / ds;
  gradient[0] = dfield_ds * trace[0];
  gradient[1] = dfield_ds * trace[1];
  gradient[2] = dfield_ds * trace[2];
}

/* ---------------------------------------------------------------------- */

bool BackgroundZones3D::sample(const double xyz[3], PlasmaPointSample &out,
                               unsigned request) const
{
  ++counters_.queries;
  const double R = std::hypot(xyz[0], xyz[1]);
  const double phi = wrap_phi(std::atan2(xyz[1], xyz[0]));
  const double Z = xyz[2];
  const Location loc = locate(R, Z, phi);
  out.status = loc.status;
  if (loc.status == PLASMA_SAMPLE_OUTSIDE) {
    ++counters_.outside;
    return false;
  }
  const Zone &zone = zones_[loc.zone_index];
  const size_t c = zone.cell_index(loc.ir, loc.ip, loc.it);
  out.provider_cell = static_cast<int>(zone.global_cell_offset + c);
  out.provider_zone = zone.zid;
  out.provider_ir = loc.ir;
  out.provider_ip = loc.ip;
  out.provider_it = loc.it;
  out.provider_claims = loc.nclaims;
  if (loc.status == PLASMA_SAMPLE_INVALID) {
    ++counters_.invalid;
    return false;
  }
  if (loc.status == PLASMA_SAMPLE_AMBIGUOUS) ++counters_.ambiguous;
  else ++counters_.ok;

  if (request & PLASMA_NEED_THERMO) {
    out.te = zone.te[c]; out.ti = zone.ti[c];
    out.ne = zone.ne[c]; out.ni = zone.ni[c];
  }
  if (request & PLASMA_NEED_FLOW_B) {
    out.mach = zone.mach[c];
    const double er = zone.trace_r[c];
    const double ep = zone.trace_phi[c];
    const double ez = zone.trace_z[c];
    const double cp = std::cos(phi), sp = std::sin(phi);
    const double ex = er * cp - ep * sp;
    const double ey = er * sp + ep * cp;
    out.bmag = zone.bmag[c];
    out.has_b = b_sign_ != 0;
    if (out.has_b) {
      out.b[0] = b_sign_ * out.bmag * ex;
      out.b[1] = b_sign_ * out.bmag * ey;
      out.b[2] = b_sign_ * out.bmag * ez;
    }
    const double cs = std::sqrt(QE * std::max(0.0,
                                static_cast<double>(zone.te[c]) +
                                static_cast<double>(zone.ti[c])) /
                                (ion_mass_amu_ * AMU));
    // flow vector is physical (along the traced direction e); the scalar
    // is the projection on B so that flow = upar * bhat, as in the 2-D
    // provider and as fix coulomb/background assumes.
    const double upar = flow_sign_ * out.mach * cs;
    out.upar = (b_sign_ != 0) ? b_sign_ * upar : upar;
    out.flow[0] = upar * ex;
    out.flow[1] = upar * ey;
    out.flow[2] = upar * ez;
  }
  if (request & PLASMA_NEED_GRAD_TE)
    parallel_gradient(zone, loc.ir, loc.ip, loc.it, zone.te, phi,
                      out.grad_te);
  if (request & PLASMA_NEED_GRAD_TI)
    parallel_gradient(zone, loc.ir, loc.ip, loc.it, zone.ti, phi,
                      out.grad_ti);
  return true;
}
