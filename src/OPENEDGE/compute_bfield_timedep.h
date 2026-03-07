/* ----------------------------------------------------------------------
    OpenEdge: time-dependent magnetic field compute
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Loads a sequence of B-field HDF5 snapshots at prescribed times and
    linearly interpolates between bracketing snapshots as the simulation
    advances.  Outputs (br, bt, bz) per grid cell, consumed by
    fix bfield/grid via c_ID[1], c_ID[2], c_ID[3].

    Input script syntax:
      compute ID bfield/timedep group file_list manifest.txt \
          nevery N interp linear

    Where manifest.txt contains lines:
      # time_seconds  filename
      0.0   bfield_t0.h5
      0.5   bfield_t1.h5
      ...
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(bfield/timedep,ComputeBfieldTimedep)

#else

#ifndef SPARTA_COMPUTE_BFIELD_TIMEDEP_H
#define SPARTA_COMPUTE_BFIELD_TIMEDEP_H

#include "compute.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

// Reuse the same struct layout as compute_plasma_fields.h
struct BfieldSnapshot {
  std::vector<double> r;                        // 1D R coordinates
  std::vector<double> z;                        // 1D Z coordinates
  std::vector<std::vector<double>> br, bt, bz;  // 2D field arrays [nz][nr]
};

// Bilinear interpolation stencil (mirrors ComputePlasmaFields::BilinearStencil)
struct BilinStencil {
  int ir1 = 0, ir2 = 0, iz1 = 0, iz2 = 0;
  double t = 0.0, u = 0.0;
  double w11 = 0.0, w21 = 0.0, w12 = 0.0, w22 = 0.0;
  int valid = 0;
};

class ComputeBfieldTimedep : public Compute {
 public:
  ComputeBfieldTimedep(class SPARTA *, int, char **);
  ~ComputeBfieldTimedep();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 private:
  int groupbit;
  int nglocal;

  // Snapshot manifest
  std::vector<std::string> snap_files;   // ordered HDF5 file paths
  std::vector<double> snap_times;        // simulation time for each snapshot
  int nsnaps;                            // number of snapshots
  std::string manifest_dir;              // directory containing manifest file

  // Two bracketing snapshots (loaded on demand)
  BfieldSnapshot snap_lo, snap_hi;
  int idx_lo, idx_hi;                    // indices into snap_files/snap_times

  // Bilinear stencils: precomputed for snap_lo and snap_hi grids
  std::vector<BilinStencil> stencil_lo, stencil_hi;

  // Update cadence and interpolation mode
  int nevery;
  int interp_mode;   // 0 = step (nearest), 1 = linear

  // Internal methods
  BfieldSnapshot readBfieldH5(const std::string &path);
  void broadcastSnapshot(BfieldSnapshot &snap);
  void parseManifest(const std::string &path);
  void precomputeStencils(const BfieldSnapshot &snap,
                          std::vector<BilinStencil> &stencils);
  BilinStencil makeStencilAtPoint(const double xyz[3],
                                  const std::vector<double> &r_vals,
                                  const std::vector<double> &z_vals) const;
  double interpField2D(const std::vector<std::vector<double>> &field,
                       const BilinStencil &s) const;
  void loadSnapshot(int idx, BfieldSnapshot &snap,
                    std::vector<BilinStencil> &stencils);
};

}

#endif
#endif
