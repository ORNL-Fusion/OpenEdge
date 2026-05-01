/* ----------------------------------------------------------------------
   OpenEdge fix background
   Centralised plasma + equilibrium data store.
   Read once, shared by all fixes/computes that need background plasma.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(background,FixBackground)

#else

#ifndef SPARTA_FIX_BACKGROUND_H
#define SPARTA_FIX_BACKGROUND_H

#include "fix.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class FixBackground : public Fix {
 public:
  FixBackground(class SPARTA *, int, char **);
  ~FixBackground();
  int  setmask();
  void init();

  // Invalidate cell_mesh_cell when SPARTA re-migrates cells (adapt/balance).
  // Matches the SPARTA pattern used by FixEmit (gridmigrate=1 + full rebuild
  // in grid_changed, no per-cell pack/unpack).
  void grid_changed() override;

  // ---- Reload interface (for future coupling) ----
  void reload();                          // re-read from current file paths
  void reload_plasma(const std::string &path);
  int  generation;                        // bumped on every reload

  // ---- Point-query API ----
  // interp2D: when icell >= 0 and the cell-indexed mesh cache is built
  // (cell_mesh_cell.size() == grid->nlocal, via build_cell_mesh_index),
  // looks up the per-cell mesh value in O(1). Falls back to the R,Z
  // stencil/mesh path for legacy callers (icell = -1) and for fields that
  // live only on the regular (rvals,zvals) grid.
  double interp2D(const std::vector<double> &field, double R, double Z,
                  int icell = -1) const;
  void   bfield_at(double R, double Z, double &Br, double &Bz, double &Bt,
                   int icell = -1) const;
  double psi_norm_at(double R, double Z) const;

  // ---- Plasma grid ----
  int nr, nz;
  std::vector<double> rvals, zvals;

  // ---- 2D plasma fields on (nz x nr) grid ----
  //      Indexing: field[iz * nr + ir]
  std::vector<double> dens_e, temp_e;
  std::vector<double> dens_i, temp_i;
  std::vector<double> parr_flow, parr_flow_r, parr_flow_t, parr_flow_z;
  std::vector<double> grad_te_r, grad_te_t, grad_te_z;
  std::vector<double> grad_ti_r, grad_ti_t, grad_ti_z;
  std::vector<double> epar;
  // Heat-flux vector (plasma -> wall) on the regular (R,Z) grid. Written
  // by the converter from the SOLPS / SOLEDGE3X surface heat-flux fields
  // (parallel + perpendicular). Consumers (fix evaporation, …) read via
  // interp2D(q_par, R, Z) / interp2D(q_perp, R, Z). has_qheatflux=1 when
  // either component is present in plasma.h5; if neither, consumers fall
  // back to default_q_par / default_q_perp (below) and a one-time warning
  // prints at init.
  int has_qheatflux;
  std::vector<double> q_par, q_perp;
  double default_q_par;   // W/m^2 fallback when has_qheatflux=0 (50e6)
  double default_q_perp;  // W/m^2 fallback when has_qheatflux=0 (0)

  // ---- Magnetic field on plasma grid ----
  int has_bfield;
  std::vector<double> br, bz, bt;

  // ---- Multi-ion species data ----
  int nion;
  std::vector<int> ion_charge_z;
  std::vector<double> ion_mass_amu;
  std::vector<std::string> ion_names;      // e.g. "D+", "O2+", "W5+"
  std::vector<std::string> ion_elements;   // e.g. "D",  "O",   "W"
  // Flat layout: [ispec * nz * nr + iz * nr + ir]
  std::vector<double> ions_dens, ions_temp, ions_upar;

  // ---- Equilibrium (psi) data ----
  int has_equ;
  int equ_jm, equ_km;
  double btf, rtf, psib, psi_axis;
  std::vector<double> equ_r, equ_z;
  std::vector<double> psirz;       // [km * jm], row-major [z][r]

  // ---- Mesh triangulation (from plasma.h5/mesh group) ----
  int has_mesh;
  int mesh_nvtx, mesh_ntri, mesh_ncell, mesh_nion;
  std::vector<double> mesh_vtx_r, mesh_vtx_z;
  std::vector<int> mesh_tri;       // (ntri*3) vertex indices
  std::vector<int> mesh_cell_idx;  // (ntri) cell index per triangle
  std::vector<double> mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar;
  // Precomputed gradients on the B2 mesh (converter writes these).
  // Consumers query via mesh_cell_at(R, Z) + mesh_grad_*_{r,z}[cell].
  std::vector<double> mesh_grad_te_r, mesh_grad_te_z;
  std::vector<double> mesh_grad_ti_r, mesh_grad_ti_z;
  // Per-cell heat-flux components on the EIRENE mesh. Same semantics as
  // q_par/q_perp above. Consumers query via mesh_cell_at(R,Z) then
  // mesh_q_par[cell] / mesh_q_perp[cell]. Empty => regular-grid fallback
  // (q_par / q_perp), then default_q_{par,perp}.
  std::vector<double> mesh_q_par, mesh_q_perp;
  // Electric field E = -grad(phi) precomputed on the B2 mesh. Converter
  // reads the plasma code's native potential (SOLPS /balance.nc po,
  // SOLEDGE3X phi, OEDGE osmns_efpara) and writes E components. No
  // runtime -grad(pe)/(ne*e) approximation.
  std::vector<double> mesh_e_r, mesh_e_z, mesh_e_t;
  // Per-vertex B-field (mesh/vtx_b{r,z,t} from the new converters) and
  // the simple vertex-average per triangle. Consumers pick either form.
  std::vector<double> mesh_vtx_br, mesh_vtx_bz, mesh_vtx_bt;
  std::vector<double> mesh_tri_br, mesh_tri_bz, mesh_tri_bt;
  std::vector<double> mesh_ions_dens, mesh_ions_temp, mesh_ions_upar;
  int has_mesh_wall_face_area;                // 1 if the converter wrote
                                              // mesh/wall_face_area
  std::vector<double> mesh_wall_face_area;    // per-cell wall face area [m^2],
                                              // toroidally integrated. Zero
                                              // for non-boundary cells.

  // ---- SPARTA wall-segment -> B2 cell map ----
  // mesh/wall_surf_cell[isurf] = flat cell index of the B2 cell whose
  // outer face is wall_b2.surf segment isurf. Enables direct (not
  // geographic) lookup in fix surface/emit/recycle. Empty if the converter
  // did not write it, or if a non-matching wall.surf is in use.
  int has_mesh_wall_surf_cell;
  std::vector<int>    mesh_wall_surf_cell;
  std::vector<double> mesh_wall_surf_area;   // captured B2 face area per
                                             // wall segment [m^2]
  std::vector<double> mesh_tri_rmin, mesh_tri_rmax, mesh_tri_zmin, mesh_tri_zmax;
  std::vector<double> mapped_cr, mapped_cz;
  std::vector<int> mapped_idx;
  int hash_nr, hash_nz;
  double hash_rmin, hash_zmin, hash_dr, hash_dz;
  std::vector<std::vector<int>> hash_grid;
  int mesh_cell_at(double R, double Z, double max_dist=0.05) const;

  // ---- Cell-indexed mesh-cell cache ----
  // cell_mesh_cell[icell] = unstructured-mesh cell index at the centroid of
  // SPARTA local cell icell (or -1 if outside mesh). Built lazily from
  // build_cell_mesh_index(). Cleared by grid_changed() (SPARTA's migration
  // hook) and by reload(). Stamp fields (cell_mesh_stamp_n / _id) provide
  // a belt-and-suspenders check in cache_plasma_particles.
  std::vector<int> cell_mesh_cell;
  void build_cell_mesh_index();
  bigint cell_mesh_stamp_id;   // grid->cells[0].id when cache was built
  int    cell_mesh_stamp_n;    // grid->nlocal when cache was built

  // ---- Valid mask ----
  std::vector<int> valid_mask;     // (nz * nr) or empty

  // ---- Configuration ----
  int is_static;                   // 1 = never reload
  int source_mode;                 // 0=file, 1=constant
  std::string plasma_path;
  std::string equ_path;

  // ---- Column-axis offset (3D Cartesian only) ----
  // Position of the axisymmetric plasma column axis in SPARTA (x, y).
  // The 3D Cartesian R = sqrt((x - column_x0)^2 + (y - column_y0)^2).
  // Default (0, 0) preserves SOLPS / SOLEDGE3X axisymmetric behavior
  // and 2D / 2D-axisymmetric paths (where this offset is ignored).
  // Set via 'column_axis x0 y0' keyword for linear-device cases (MPEX,
  // proto-lite) whose box is not centered at the origin.
  double column_x0, column_y0;

 private:
  void clear_loaded_data();
  void load_plasma_h5();
  void load_constant_profile();
  void load_equilibrium();
  void derive_bfield_from_equ();
  void build_mesh_index();
  int find_mesh_triangle(double R, double Z) const;
  int find_nearest_mapped_triangle(double R, double Z, double max_dist) const;
  const std::vector<double> *mesh_field_for(const std::vector<double> &field) const;

  // ---- Constant-mode configuration ----
  int const_has_r_bounds, const_has_z_bounds;
  double const_rmin, const_rmax, const_zmin, const_zmax;
  double const_dens_e, const_temp_e, const_dens_i, const_temp_i;
  int const_has_dens_i, const_has_temp_i;
  double const_parr_flow, const_parr_flow_r, const_parr_flow_t, const_parr_flow_z;
  double const_grad_te_r, const_grad_te_t, const_grad_te_z;
  double const_grad_ti_r, const_grad_ti_t, const_grad_ti_z;
  double const_epar;
  double const_br, const_bz, const_bt;
  int const_has_bfield;
};

}

#endif
#endif
