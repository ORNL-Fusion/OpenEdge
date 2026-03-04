/* ----------------------------------------------------------------------
    OpenEdge: time-dependent plasma field compute
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    Loads a sequence of plasma HDF5 snapshots at prescribed times and
    linearly interpolates between bracketing snapshots as the simulation
    advances.  Outputs user-selected columns per grid cell (dens_e,
    temp_e, dens_i, temp_i, parrflow, parrflow_r, parrflow_t, parrflow_z,
    grad_te_r, grad_te_t, grad_te_z, grad_ti_r, grad_ti_t, grad_ti_z).

    Input script syntax:
      compute ID plasma/timedep group file_list manifest.txt \
          nevery N interp linear values dens_e temp_e dens_i temp_i parrflow

    Where manifest.txt contains lines:
      # time_seconds  filename
      0.0   plasma_t0.h5
      0.5   plasma_t1.h5
      ...

    HDF5 files must have the same layout as soledge2openedge.py output:
      /r, /z, /dens_e, /temp_e, /dens_i, /temp_i, /parr_flow,
      /parr_flow_r, /parr_flow_t, /parr_flow_z,
      /grad_te_r, /grad_te_t, /grad_te_z,
      /grad_ti_r, /grad_ti_t, /grad_ti_z
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(plasma/timedep,ComputePlasmaTimedep)

#else

#ifndef SPARTA_COMPUTE_PLASMA_TIMEDEP_H
#define SPARTA_COMPUTE_PLASMA_TIMEDEP_H

#include "compute.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

// Snapshot of plasma fields on a regular (R,Z) grid
struct PlasmaSnapshot {
  std::vector<double> r;  // 1D R coordinates [nr]
  std::vector<double> z;  // 1D Z coordinates [nz]
  // Core scalar fields [nz][nr]
  std::vector<std::vector<double>> dens_e, temp_e;
  std::vector<std::vector<double>> dens_i, temp_i;
  std::vector<std::vector<double>> parr_flow;
  std::vector<std::vector<double>> parr_flow_r, parr_flow_t, parr_flow_z;
  std::vector<std::vector<double>> grad_te_r, grad_te_t, grad_te_z;
  std::vector<std::vector<double>> grad_ti_r, grad_ti_t, grad_ti_z;
};

// Bilinear interpolation stencil (same layout as compute_bfield_timedep)
struct PlasmaStencil {
  int ir1 = 0, ir2 = 0, iz1 = 0, iz2 = 0;
  double t = 0.0, u = 0.0;
  double w11 = 0.0, w21 = 0.0, w12 = 0.0, w22 = 0.0;
  int valid = 0;
};

class ComputePlasmaTimedep : public Compute {
 public:
  ComputePlasmaTimedep(class SPARTA *, int, char **);
  ~ComputePlasmaTimedep();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

 private:
  int groupbit;
  int nglocal;

  // Output value selection
  enum {
    DENS_E, TEMP_E, DENS_I, TEMP_I,
    PARRFLOW, PARRFLOW_R, PARRFLOW_T, PARRFLOW_Z,
    GRAD_TE_R, GRAD_TE_T, GRAD_TE_Z,
    GRAD_TI_R, GRAD_TI_T, GRAD_TI_Z
  };
  int nvalues;
  std::vector<int> value_ids;

  // Snapshot manifest
  std::vector<std::string> snap_files;
  std::vector<double> snap_times;
  int nsnaps;
  std::string manifest_dir;

  // Two bracketing snapshots (loaded on demand)
  PlasmaSnapshot snap_lo, snap_hi;
  int idx_lo, idx_hi;

  // Bilinear stencils per grid cell
  std::vector<PlasmaStencil> stencil_lo, stencil_hi;

  // Update cadence and interpolation mode
  int nevery;
  int interp_mode;  // 0 = step, 1 = linear

  // Internal methods
  PlasmaSnapshot readPlasmaH5(const std::string &path);
  void broadcastSnapshot(PlasmaSnapshot &snap);
  void parseManifest(const std::string &path);
  void precomputeStencils(const PlasmaSnapshot &snap,
                          std::vector<PlasmaStencil> &stencils);
  PlasmaStencil makeStencilAtPoint(const double xyz[3],
                                    const std::vector<double> &r_vals,
                                    const std::vector<double> &z_vals) const;
  double interpField2D(const std::vector<std::vector<double>> &field,
                       const PlasmaStencil &s) const;
  void loadSnapshot(int idx, PlasmaSnapshot &snap,
                    std::vector<PlasmaStencil> &stencils);

  // Map a value enum to the corresponding 2D field in a snapshot
  const std::vector<std::vector<double>> &
  getField(const PlasmaSnapshot &snap, int val_id) const;
};

}

#endif
#endif
