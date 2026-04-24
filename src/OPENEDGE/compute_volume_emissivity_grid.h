/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute volume/emissivity/grid — per-grid volumetric line emissivity
      emissivity = ne * nz * PEC(Te, ne)   [photons/m^3/s/sr]

    where nz is the weighted number density of tracked impurity particles
    (using per-particle pweight), ne and Te come from a plasma/fields
    compute, and PEC is loaded from an HDF5 table.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(volume/emissivity/grid,ComputeVolumeEmissivityGrid)

#else

#ifndef SPARTA_COMPUTE_VOLUME_EMISSIVITY_GRID_H
#define SPARTA_COMPUTE_VOLUME_EMISSIVITY_GRID_H

#include "compute.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class ComputeVolumeEmissivityGrid : public Compute {
 public:
  ComputeVolumeEmissivityGrid(class SPARTA *, int, char **);
  ~ComputeVolumeEmissivityGrid();
  void init();
  void compute_per_grid();
  int query_tally_grid(int, double **&, int *&);
  void post_process_grid(int, int, double **, int *, double *, int);
  void reallocate();
  bigint memory_usage();

 protected:
  int groupbit,imix,ngroup;
  int npergroup;             // always 1 (WCOUNT only)
  int ntotal;                // ngroup * npergroup
  int nglocal;

  int *nmap;
  int **map;
  double **tally;

  int pweight_index;
  int pweight_ewhich;

  // plasma compute reference
  char *plasma_compute_id;
  class ComputePlasmaFields *cp_plasma;

  // PEC table (log10 space) loaded from processes.h5 /volume/pec/*
  std::vector<double> pec_log_te;   // log10(Te [eV])
  std::vector<double> pec_log_ne;   // log10(ne [m^-3])
  std::vector<double> pec_log_val;  // log10(PEC [m^3/s]), flat [nte * nne]
  int pec_nte,pec_nne;
  double pec_unit_conv;             // conversion factor to m^3/s

  double interpolatePEC(double log_te, double log_ne) const;
};

}

#endif
#endif
