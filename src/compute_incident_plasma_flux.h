/* ----------------------------------------------------------------------
   OpenEdge compute incident/flux
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(incident/plasma/flux,ComputeIncidentPlasmaFlux)

#else

#ifndef SPARTA_COMPUTE_INCIDENT_PLASMA_FLUX_H
#define SPARTA_COMPUTE_INCIDENT_PLASMA_FLUX_H

#include "compute.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class ComputeIncidentPlasmaFlux : public Compute {
 public:
  ComputeIncidentPlasmaFlux(class SPARTA *, int, char **);
  ~ComputeIncidentPlasmaFlux();
  void init();
  void compute_per_surf();
  bigint memory_usage();

 protected:
  enum {NFLUX_INCIDENT, NFLUX_NORMAL, NFLUX_SPECIES};

  int groupbit,nvalue;
  int dimension,distributed;
  int firstflag;
  int nsown;
  int *which;
  int *which_species;   // 1-based species slot for species-specific outputs
  std::string plasma_path;

  int nr,nz,nspec;
  std::vector<double> rvals,zvals;
  std::vector<double> dens_i,parr_flow,parr_flow_r,parr_flow_t,parr_flow_z;
  std::vector<double> br,bt,bz;
  std::vector<double> ions_dens,ions_parr_flow,ions_parr_flow_r,ions_parr_flow_t,ions_parr_flow_z;
  std::vector<int> ion_spec_index;
  int has_multi_ion;
  int has_bfield;

  double interp2D(const std::vector<double> &f, double r, double z) const;
  double interp3D(const std::vector<double> &f, int ispec, double r, double z) const;
  void load_plasma();
  int peek_nspec_from_plasma() const;
};

}

#endif
#endif
