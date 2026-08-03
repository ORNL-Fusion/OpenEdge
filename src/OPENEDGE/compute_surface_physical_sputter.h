/* ----------------------------------------------------------------------
   OpenEdge compute surface/physical/sputter
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surface/physical/sputter,ComputeSurfacePhysicalSputter)

#else

#ifndef SPARTA_COMPUTE_SURFACE_PHYSICAL_SPUTTER_H
#define SPARTA_COMPUTE_SURFACE_PHYSICAL_SPUTTER_H

#include "compute.h"
#include "process_library.h"
#include <memory>
#include <string>
#include <vector>

namespace SPARTA_NS {

class FixBackground;
struct IeadTable;

class ComputeSurfacePhysicalSputter : public Compute {
 public:
  ComputeSurfacePhysicalSputter(class SPARTA *, int, char **);
  ~ComputeSurfacePhysicalSputter();
  void init();
  void compute_per_surf();
  bigint memory_usage();

  // True when the per-surf output is frozen (static mode + already built).
  // Consumers can use this to skip re-deriving downstream quantities.
  bool is_static_cached() const { return static_cache && cache_valid; }

 protected:
  enum {
    NFLUX_SPECIES,
    INCIDENT_ANGLE_SPECIES,
    INCIDENT_ENERGY_SPECIES,
    SPUTTER_YIELD_SPECIES,
    SPUTTER_FLUX_SPECIES,
    SPUTTER_FLUX_TOTAL,
    SPUTTER_RATE_TOTAL   // flux x axisymmetric ring area [particles/s]
  };

  int groupbit,nvalue;
  int dimension,distributed;
  int firstflag;
  int debug_interp;
  int nsown;
  int static_cache;
  int cache_valid;
  int *which;
  int *which_species;   // 1-based species slot for species-specific outputs
  std::string plasma_path;
  std::string surface_path;
  std::string bfield_path;
  std::string equ_path;
  std::string background_fix_id;  // ID of fix background (if used)

  // Eckstein analytic sputter mode (no surface HDF5 table).
  // When eckstein_mode != 0, sputter yields are computed from the
  // EIRENE Bohdansky-1993 + Yamamura formulas with the coefficient
  // table in eckstein_sputter_data.h.
  int eckstein_mode;
  std::string eckstein_name;        // e.g. "O_on_W"
  double eck_Z1, eck_M1, eck_Z2, eck_M2;
  double eck_Es, eck_Eth, eck_Q, eck_ETF;

  // projectile slots for sputtering (inclusive, 1-based)
  int proj_slot_lo, proj_slot_hi;
  // legacy option kept for input compatibility
  double mass_amu;

  // Element-aware multi-projectile API (post-2026-04-21):
  //   target W projectiles O,B
  // Resolves one Eckstein entry (or surface HDF5 table) per projectile
  // element, matches plasma.h5 ion slots to tables via /ion_species/elements.
  int api_new;                                 // 1 when target/projectiles used
  std::string target_element;                  // e.g. "W"
  std::vector<std::string> projectile_elements;// e.g. {"D","O"}
  // Override Es (target surface binding energy) at runtime; Eth is rescaled
  // proportionally per pair using the table Eth/Es ratio. <=0 means use
  // Eckstein table values verbatim. Allain 2003 measured U_s = 1.0-2.5 eV
  // on liquid Li depending on chemistry/temperature.
  double target_us = 0.0;
  std::vector<int> slot_to_table;              // per plasma-ion slot (0-based),
                                               // index into per_proj_eck_*
                                               // or -1 if that slot is not a
                                               // projectile element.
  // One Eckstein parameter set per projectile element (parallel arrays; kept
  // as flat doubles to avoid dragging Eckstein::SputterParams into the header).
  std::vector<double> per_proj_Z1, per_proj_M1, per_proj_Z2, per_proj_M2;
  std::vector<double> per_proj_Es, per_proj_Eth, per_proj_Q, per_proj_ETF;
  // Per-projectile angle-resolved TRIM/BCA yield tables from
  // database/processes.h5 /surface/sputter/<proj>_on_<target>; preferred
  // over the analytic Eckstein fit when present.
  std::vector<ProcessLibrary::TrimSputterTable> per_proj_sput_tbl;

  void resolve_projectile_tables(const FixBackground *pd);

  // Compound-target (mixed-material) mode:
  //   compound <mix> conc <attr> <species>
  // Tables become /surface/sputter/<proj>_on_<mix>_<target> with a
  // composition axis C (3D tables mandatory -- no analytic fallback).
  // The expensive pass (plasma sampling + IEAD convolution) runs once
  // per table composition grid point and is cached per surf element;
  // every invocation then just interpolates the cached yield slices at
  // the live per-surf concentration <attr>[<species>] (the mixing-zone
  // array maintained by surf_react surface/pwi). This keeps `static
  // yes` semantics: static refers to the plasma, not the composition.
  std::string compound_mix;        // table infix, e.g. "bw"; empty = pure mode
  std::string conc_attr;           // per-surf custom array name (e.g. sigma_conc)
  std::string conc_species_name;   // species whose column is the c coordinate
  int conc_index = -1;             // surf custom index, resolved lazily
  int conc_isp = -1;               // species index, resolved lazily
  int slices_valid = 0;
  int slice_nc = 0;                // max NC across projectile tables
  int cs_ns = 0;                   // species-slot count at slice build time
  std::vector<double> cs_gamma, cs_angle, cs_energy;  // [nsown*cs_ns]
  std::vector<double> cs_ybar;     // [nsown*cs_ns*slice_nc] per-c yields
  std::vector<double> cs_area;     // [nsown] axisymmetric ring area
  std::vector<char> cs_active;     // [nsown] in group + in plasma domain
  void assemble_compound_outputs();

  // IEAD lookup — when active, replaces Y(<E>, <theta>) with the
  // distribution-weighted < Y > = sum_{Et,theta} Y(Et*Z*Te, theta) f(...).
  // `iead_arg` is the user token from the input deck:
  //   "auto"       -> resolve via database_paths::resolve_iead_file()
  //   "<path.h5>"  -> explicit file (single light table)
  //   ""           -> off (mean-impact path, default)
  // Two tables can be loaded simultaneously when `iead auto` and the
  // projectiles list mixes W with lighter elements: the light table
  // handles Z=1..10 elements up to Ne via D-scaling, and the W-specific
  // table (m_proj=184 amu, Z=1..20) handles tungsten. iead_per_proj maps
  // each projectile element index to the appropriate table (or null when
  // no IEAD lookup applies to that element).
  std::string iead_arg;
  std::unique_ptr<IeadTable> iead_light;
  std::unique_ptr<IeadTable> iead_W;
  std::vector<IeadTable *> iead_per_proj;   // ptrs into iead_light/iead_W
  void load_iead_if_requested();
  IeadTable *iead_for_slot(int s) const;

  // Virtual background impurity (not in plasma.h5)
  // Usage: impurity mass_amu frac Zmax f1 f2 ... fZmax
  //   Synthesizes sputtering from an impurity at fixed fraction of n_e.
  //   Charge state fractions f1..fZmax give the distribution over Z=1..Zmax.
  int has_impurity;
  double imp_mass_amu;     // impurity mass [amu]
  double imp_frac;         // fraction of n_e (e.g. 0.03 for 3%)
  int imp_Zmax;            // max charge state
  std::vector<double> imp_charge_fracs;  // charge state fractions (Z=1..Zmax)

  // plasma data
  int nr,nz,nspec;
  // Column-axis offset for 3D Cartesian sparta_to_RZ. Mirrored from
  // FixBackground in load_plasma_from_fix; default (0, 0) for file mode.
  double column_x0 = 0.0, column_y0 = 0.0;
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
  // mesh/wall_surf_cell[isurf-1] = SOLPS cell index adjacent to wall.surf
  // line `isurf`. When loaded by the converter (and present in plasma.h5),
  // the per-segment plasma sampling uses these directly instead of the
  // nearest-triangle fallback. Avoids the few-mm extrapolation gap that
  // turned out to dominate the OE-vs-SOLPS sputter difference at near
  // grazing incidence.
  std::vector<int> mesh_wall_surf_cell;
  int has_mesh_wall_surf_cell = 0;
  std::vector<double> mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar;
  int mesh_nion;
  std::vector<double> mesh_ions_dens, mesh_ions_temp, mesh_ions_upar;
  // Wall-flux scatter copied from fix background. Empty if absent.
  // Layout (wall_flux_nspec, wall_flux_n) row-major for gamma_i.
  int wall_flux_n = 0;
  int wall_flux_nspec = 0;
  std::vector<double> wall_flux_r, wall_flux_z;
  std::vector<double> wall_flux_te, wall_flux_ti;
  std::vector<double> wall_flux_gamma_i;
  std::vector<double> wall_flux_b_r, wall_flux_b_z, wall_flux_b_t;
  // Cutoff distance [m] for IDW point-query of wall_flux. SPARTA
  // segments farther than this from any source point fall back to the
  // Bohm reconstruction. Default 5 cm.
  double wall_flux_cutoff = 0.05;
  // Per-triangle B-field from fix background (mesh-only plasma.h5 path)
  std::vector<double> mesh_tri_br, mesh_tri_bz, mesh_tri_bt;
  // bounding boxes for triangle search acceleration
  std::vector<double> mesh_tri_rmin, mesh_tri_rmax, mesh_tri_zmin, mesh_tri_zmax;
  // spatial hash for O(1) triangle lookup
  int hash_nr, hash_nz;
  double hash_rmin, hash_zmin, hash_dr, hash_dz;
  std::vector<std::vector<int>> hash_grid;

  double interp2D(const std::vector<double> &f, double r, double z) const;
  double interp3D(const std::vector<double> &f, int ispec, double r, double z) const;
  double interp_yield(double e_eV, double a_deg) const;
  void load_plasma();
  void load_plasma_from_fix(const FixBackground *pd);
  void rebuild_mesh_cache();
  void load_surface_data();
  void load_boundary();
  void load_bfield_from_equ();
  void load_mesh();

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
