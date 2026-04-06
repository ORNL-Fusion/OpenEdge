/* ----------------------------------------------------------------------
   OpenEdge fix plasma/data
   Centralised plasma + equilibrium data store.
   Read once, shared by all fixes/computes that need background plasma.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(plasma/data,FixPlasmaData)

#else

#ifndef SPARTA_FIX_PLASMA_DATA_H
#define SPARTA_FIX_PLASMA_DATA_H

#include "fix.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class FixPlasmaData : public Fix {
 public:
  FixPlasmaData(class SPARTA *, int, char **);
  ~FixPlasmaData();
  int  setmask();
  void init();

  // ---- Reload interface (for future coupling) ----
  void reload();                          // re-read from current file paths
  void reload_plasma(const std::string &path);
  int  generation;                        // bumped on every reload

  // ---- Point-query API ----
  double interp2D(const std::vector<double> &field, double R, double Z) const;
  void   bfield_at(double R, double Z, double &Br, double &Bz, double &Bt) const;
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

  // ---- Magnetic field on plasma grid ----
  int has_bfield;
  std::vector<double> br, bz, bt;

  // ---- Multi-ion species data ----
  int nion;
  std::vector<int> ion_charge_z;
  std::vector<double> ion_mass_amu;
  std::vector<std::string> ion_names;
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
  int mesh_nvtx, mesh_ntri, mesh_ncell;
  std::vector<double> mesh_vtx_r, mesh_vtx_z;
  std::vector<int> mesh_tri;       // (ntri*3) vertex indices
  std::vector<int> mesh_cell_idx;  // (ntri) cell index per triangle
  std::vector<double> mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar;

  // ---- Valid mask ----
  std::vector<int> valid_mask;     // (nz * nr) or empty

  // ---- Configuration ----
  int is_static;                   // 1 = never reload
  std::string plasma_path;
  std::string equ_path;

 private:
  void load_plasma_h5();
  void load_equilibrium();
  void derive_bfield_from_equ();
};

}

#endif
#endif
