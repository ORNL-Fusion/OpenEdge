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

class FixPlasmaData;

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
  int plasma_source_mode;
  int static_cache;
  int cache_valid;
  int *which;
  int *which_species;   // 1-based species slot for species-specific outputs
  std::string plasma_path;
  std::string plasma_data_fix_id;
  std::string surface_path;
  std::string bfield_path;
  std::string equ_path;

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

  // SOLPS boundary polygon mask (R,Z)
  std::string boundary_path;
  std::vector<double> boundary_r, boundary_z;

  // SOLPS mesh triangulation for direct cell-based interpolation
  int has_mesh;
  int mesh_nvtx, mesh_ntri, mesh_ncell;
  std::vector<double> mesh_vtx_r, mesh_vtx_z;
  std::vector<int> mesh_tri;        // (ntri*3) vertex indices
  std::vector<int> mesh_cell_idx;   // (ntri) cell index per triangle
  std::vector<double> mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar;
  int mesh_nion;
  std::vector<double> mesh_ions_dens, mesh_ions_temp, mesh_ions_upar;
  // bounding boxes for triangle search acceleration
  std::vector<double> mesh_tri_rmin, mesh_tri_rmax, mesh_tri_zmin, mesh_tri_zmax;

  double interp2D(const std::vector<double> &f, double r, double z) const;
  double interp3D(const std::vector<double> &f, int ispec, double r, double z) const;
  double interp_yield(double e_eV, double a_deg) const;
  void load_plasma();
  void load_plasma_from_fix(const FixPlasmaData *pd);
  void rebuild_mesh_cache();
  void load_surface_data();
  void load_boundary();
  void load_mesh();
  void load_bfield_from_equ();

  // Precomputed mapped-triangle centroids for nearest-neighbor fallback
  std::vector<double> mapped_cr, mapped_cz;  // centroids of mapped triangles
  std::vector<int> mapped_idx;               // original triangle indices
  int find_nearest_mapped_triangle(double r, double z, double max_dist) const;
  int peek_nspec_from_plasma() const;
  int in_projectile_slots(int slot1) const;
  int point_in_boundary(double r, double z) const;
  int find_mesh_triangle(double r, double z) const;
};

}

#endif
#endif
