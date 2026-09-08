/* ----------------------------------------------------------------------
   OpenEdge native 3-D curvilinear plasma-background provider.

   The provider reads the openedge_background3d HDF5 schema produced by
   examples/wip/.../scripts/build_openedge_background.py.  Geometry and
   fields remain on the native EMC3-style ruled hexahedra; no Cartesian
   raster or axisymmetric projection is introduced.
------------------------------------------------------------------------- */

#ifndef SPARTA_BACKGROUND_ZONES3D_H
#define SPARTA_BACKGROUND_ZONES3D_H

#include "spatype.h"

#include <mpi.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SPARTA_NS {

struct PlasmaPointSample;

class BackgroundZones3D {
 public:
  BackgroundZones3D() = default;

  void clear();
  void load(const std::string &path, MPI_Comm world, int me);

  // Reconstruct signed flow and (optionally) the absolute B vector from
  // the field-line trace stored in the carrier file.
  void configure(int flow_sign, int b_sign, bool cs_te_plus_ti);

  bool sample(const double xyz[3], PlasmaPointSample &out,
              unsigned request) const;

  bool loaded() const { return !zones_.empty(); }
  bool has_absolute_b() const { return b_sign_ != 0; }
  int zone_count() const { return static_cast<int>(zones_.size()); }
  int64_t cell_count() const { return total_cells_; }
  double ion_mass_amu() const { return ion_mass_amu_; }

  struct Counters {
    int64_t queries = 0;
    int64_t ok = 0;
    int64_t outside = 0;
    int64_t invalid = 0;
    int64_t ambiguous = 0;
  };
  Counters counters() const { return counters_; }
  void reset_counters() const { counters_ = Counters{}; }

 private:
  struct HashPlane {
    int nr = 0, nz = 0;
    float rmin = 0.0f, zmin = 0.0f;
    float dr = 1.0f, dz = 1.0f;
    std::vector<int> offsets;
    std::vector<int> cells;
  };

  struct Zone {
    int zid = -1;
    int sector = -1;
    int topology_priority = 99;  // SOL=0, PFR=1, CORE=2
    std::string topology;
    int nr = 0, np = 0, nt = 0;
    int ncr = 0, ncp = 0, nct = 0;
    int64_t global_cell_offset = 0;
    double phi_start = 0.0, phi_end = 0.0;

    std::vector<double> phi;
    std::vector<float> vertex_r, vertex_z;
    std::vector<uint8_t> valid;
    std::vector<float> te, ti, ne, ni;
    std::vector<float> mach, bmag;
    std::vector<float> trace_r, trace_phi, trace_z;
    std::vector<HashPlane> hashes;

    size_t vertex_index(int ir, int ip, int it) const
    {
      return (static_cast<size_t>(ir) * np + ip) * nt + it;
    }
    size_t cell_index(int ir, int ip, int it) const
    {
      return (static_cast<size_t>(ir) * ncp + ip) * nct + it;
    }
  };

  struct Sector {
    int sid = -1;
    double phi_start = 0.0, phi_end = 0.0;
    std::vector<int> zones;  // indices into zones_
  };

  struct Location {
    int status = 1;  // PLASMA_SAMPLE_OUTSIDE without including fix_background.h
    int zone_index = -1;
    int ir = -1, ip = -1, it = -1;
    int nclaims = 0;
  };

  std::vector<Zone> zones_;
  std::vector<Sector> sectors_;
  int64_t total_cells_ = 0;
  double ion_mass_amu_ = 2.014;
  int flow_sign_ = 0;
  int b_sign_ = 0;
  bool cs_te_plus_ti_ = false;
  mutable Counters counters_;

  void read_rank0(const std::string &path);
  void broadcast(MPI_Comm world, int me);
  void build_indices();
  void build_hash(Zone &zone, int it);

  int sector_for(double phi) const;
  bool locate_in_zone(const Zone &zone, double R, double Z, double phi,
                      int &ir, int &ip, int &it) const;
  Location locate(double R, double Z, double phi) const;

  void cell_center_xyz(const Zone &zone, int ir, int ip, int it,
                       double xyz[3]) const;
  void trace_cartesian(const Zone &zone, size_t cell, double phi,
                       double trace[3]) const;
  void parallel_gradient(const Zone &zone, int ir, int ip, int it,
                         const std::vector<float> &field, double phi,
                         double gradient[3]) const;

  static double wrap_phi(double phi);
  static bool point_in_quad(double x, double y,
                            const double qx[4], const double qy[4]);
};

}  // namespace SPARTA_NS

#endif
