/* ----------------------------------------------------------------------
   OpenEdge compute pmi/surf/data
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(pmi/surf/data,ComputePMISurfData)

#else

#ifndef SPARTA_COMPUTE_PMI_SURF_DATA_H
#define SPARTA_COMPUTE_PMI_SURF_DATA_H

#include "compute.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class ComputePMISurfData : public Compute {
 public:
  ComputePMISurfData(class SPARTA *, int, char **);
  ~ComputePMISurfData();
  void init();
  void compute_per_surf();
  bigint memory_usage();

 protected:
  enum {
    NFLUX_SPECIES,
    INCIDENT_ANGLE_SPECIES,
    INCIDENT_ENERGY_SPECIES,
    SPUTTER_YIELD_SPECIES,
    SPUTTER_FLUX_SPECIES,
    SPUTTER_FLUX_TOTAL
  };

  int groupbit,nvalue;
  int dimension,distributed;
  int firstflag;
  int debug_interp;
  int nsown;
  int *which;
  int *which_species;   // 1-based species slot for species-specific outputs
  std::string plasma_path;
  std::string surface_path;
  std::string bfield_path;

  // projectile slots for sputtering (inclusive, 1-based)
  int proj_slot_lo, proj_slot_hi;
  // legacy option kept for input compatibility
  double mass_amu;

  // plasma data
  int nr,nz,nspec;
  std::vector<double> rvals,zvals;
  std::vector<double> dens_i,temp_e,temp_i,parr_flow,parr_flow_r,parr_flow_t,parr_flow_z;
  std::vector<double> br,bt,bz;
  std::vector<double> ions_dens,ions_temp,ions_parr_flow,ions_parr_flow_r,ions_parr_flow_t,ions_parr_flow_z;
  std::vector<int> ion_charge_state_z;
  int has_multi_ion;
  int has_bfield;
  int has_temp;

  // surface BCA data
  int nE,nA;
  std::vector<double> E_axis,A_axis,spyld;
  std::vector<double> debug_E;
  std::vector<double> debug_A;

  double interp2D(const std::vector<double> &f, double r, double z) const;
  double interp3D(const std::vector<double> &f, int ispec, double r, double z) const;
  double interp_yield(double e_eV, double a_deg) const;
  void load_plasma();
  void load_surface_data();
  int peek_nspec_from_plasma() const;
  int in_projectile_slots(int slot1) const;
};

}

#endif
#endif
