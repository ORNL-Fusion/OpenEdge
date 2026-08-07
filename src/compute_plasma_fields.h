/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(plasma/fields,ComputePlasmaFields)

#else

#ifndef SPARTA_COMPUTE_PLASMA_FIELDS_H
#define SPARTA_COMPUTE_PLASMA_FIELDS_H

#include "compute.h"
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

namespace SPARTA_NS {

    // Structs for plasma data and parameters
struct PlasmaFileData{
  // Column-axis offset (3D Cartesian only). Mirrored from FixBackground at
  // init / reload so all sparta_to_RZ call sites can apply the offset
  // without holding a FixBackground pointer. Default (0, 0).
  double column_x0 = 0.0, column_y0 = 0.0;
  std::vector<double> r;
  std::vector<double> z;
  std::vector<std::vector<double>> dens_e, temp_e;
  std::vector<std::vector<double>> dens_i, temp_i;
  std::vector<std::vector<double>> parr_flow_r, parr_flow_t, parr_flow_z, parr_flow;
  std::vector<std::vector<double>> grad_temp_e_r, grad_temp_e_t, grad_temp_e_z;
  std::vector<std::vector<double>> grad_temp_i_r, grad_temp_i_t, grad_temp_i_z;
  // Optional heat flux stored inside plasma.h5 (q_mag dataset).
  bool has_qmag = false;
  std::vector<std::vector<double>> q_mag;
  // Optional B-field stored inside plasma.h5 (br, bt, bz datasets).
  // When present these override the separate bfield.h5 for mag_arr.
  bool has_bfield = false;
  std::vector<std::vector<double>> br, bt, bz;
  // Optional mesh triangulation from plasma.h5 (SOLPS cell data)
  bool has_mesh = false;
  int mesh_nvtx = 0, mesh_ntri = 0, mesh_ncell = 0;
  std::vector<double> mesh_vtx_r, mesh_vtx_z;
  std::vector<int> mesh_tri;       // (ntri*3) vertex indices
  std::vector<int> mesh_cell_idx;  // (ntri) cell index per triangle
  std::vector<double> mesh_ne, mesh_te, mesh_ti, mesh_ni, mesh_upar;
  // Electric field on the mesh (from plasma code's native potential,
  // not an internal -grad(pe)/(ne*e) approximation). Empty vector means
  // "no E-field stored"; consumers return zero.
  std::vector<double> mesh_e_r, mesh_e_z, mesh_e_t;
  // Magnetic field on the mesh: per-triangle vertex average of
  // mesh/vtx_b{r,z,t} from the converter. Non-empty => use mesh lookup
  // in query_bfield_at_point; empty => fall back to legacy (br,bz,bt)
  // regular-grid arrays or the equilibrium path.
  std::vector<double> mesh_tri_br, mesh_tri_bz, mesh_tri_bt;
  // Bounding boxes for triangle search
  std::vector<double> mesh_tri_rmin, mesh_tri_rmax, mesh_tri_zmin, mesh_tri_zmax;
  // Precomputed centroids for nearest-neighbor fallback
  std::vector<double> mapped_cr, mapped_cz;
  std::vector<int> mapped_idx;
  // Spatial hash for O(1) triangle lookup
  int hash_nr = 0, hash_nz = 0;
  double hash_rmin = 0, hash_zmin = 0, hash_dr = 0, hash_dz = 0;
  std::vector<std::vector<int>> hash_grid;  // hash_grid[iz*hash_nr+ir] = list of tri indices
  void buildSpatialHash(int nr_bins = 100, int nz_bins = 100);

  // Optional multi-ion extension from plasma.h5
  std::vector<int> ion_spec_index;
  std::vector<int> ion_charge_state_z;
  std::vector<double> ion_mass_amu;
  std::vector<std::string> ion_names;
  int ions_nspec = 0;
  int ions_nz = 0;
  int ions_nr = 0;
  // Flat storage layout: ((ispec * nz) + iz) * nr + ir
  std::vector<double> ions_dens;
  std::vector<double> ions_temp;
  std::vector<double> ions_parr_flow;
  std::vector<double> ions_parr_flow_r;
  std::vector<double> ions_parr_flow_t;
  std::vector<double> ions_parr_flow_z;
};

struct PlasmaFileParams {
  double dens_e;
  double temp_e;
  double dens_i;
  double temp_i;
  double grad_dens_e_r;
  double grad_dens_e_t;
  double grad_dens_e_z;
  double parr_flow;
  double parr_flow_r;
  double parr_flow_t;
  double parr_flow_z;
  double grad_temp_e_r;
  double grad_temp_e_t;
  double grad_temp_e_z;
  double grad_temp_i_r;
  double grad_temp_i_t;
  double grad_temp_i_z;
  double q_mag;   // surface heat flux [W/m^2], 0 if not in plasma.h5
  double epar;    // parallel ambipolar E-field [V/m], 0 if not computed
};

// Equilibrium (ψ) data from .equ file for exact magnetic geometry
struct EquilibriumData {
  int jm = 0, km = 0;                          // grid dimensions (R, Z)
  double btf = 0.0, rtf = 0.0, psib = 0.0;     // toroidal field params
  std::vector<double> r, z;                      // 1D coordinate arrays [m]
  std::vector<std::vector<double>> psi;          // ψ(Z,R) on grid [km x jm]
};

// Per-cell precomputed magnetic geometry from equilibrium
struct MagneticGeometry {
  double kappa[3];    // curvature vector κ = (b·∇)b in cylindrical (R, φ, Z)
  double curl_b[3];   // curl(b̂) in cylindrical (R, φ, Z)
  double gradBmag[3]; // grad(|B|) in cylindrical (R, φ, Z)
  double Bmag;        // |B|
};

// Structs for magnetic field data and parameters
struct MagneticFieldFileData {
  std::vector<double> r;   
  std::vector<double> z;  
  std::vector<std::vector<double>> br;
  std::vector<std::vector<double>> bt;
  std::vector<std::vector<double>> bz;
};

struct MagneticFieldFileDataParams {
  double br;
  double bt;
  double bz;
  double r;
  double z;
  // Gradients of B components in cylindrical (R,Z)
  double dBr_dr, dBr_dz;
  double dBt_dr, dBt_dz;
  double dBz_dr, dBz_dz;
  // |B| and its gradient
  double Bmag;
  double dBmag_dr, dBmag_dz;
};

class ComputePlasmaFields : public Compute {
 public:
  ComputePlasmaFields(class SPARTA *, int, char **);
  ~ComputePlasmaFields();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

  // Bilinear interpolation stencil (public so callers can inspect validity)
  struct BilinearStencil {
    int ir1 = 0, ir2 = 0, iz1 = 0, iz2 = 0;
    double t = 0.0, u = 0.0;
    double inv_dR = 0.0, inv_dZ = 0.0;
    double w11 = 0.0, w21 = 0.0, w12 = 0.0, w22 = 0.0;
    int valid = 0;
  };

  // Point-query API: interpolate background data at arbitrary (x,y,z)
  PlasmaFileParams query_plasma_at_point(const double xyz[3]) const;
  // Existing callers use mesh/grid-first data. GCA requests the smooth
  // equilibrium field and its derivatives when one is available.
  MagneticFieldFileDataParams query_bfield_at_point(
      const double xyz[3], bool prefer_equilibrium = false) const;

  // Reload plasma background from file (for coupling: re-reads HDF5 and re-interpolates)
  void reload_plasma();
  void reload_plasma(const std::string &new_plasma_path);

PlasmaFileParams *plasma_arr;   // size = grid->nlocal
PlasmaFileData plasma_data;
void broadcastPlasmaData(PlasmaFileData& data);
PlasmaFileData readPlasmaFileData(const std::string& path);
int nion_species = 0;
std::vector<double> ion_mass_amu;
std::vector<int> ion_charge_state_z;
std::vector<int> ion_spec_index;
std::vector<std::string> ion_names;
// Pre-interpolated per-grid multi-ion fields: flat [nlocal * nion]
// Layout: icell*nion + ispec
std::vector<double> ion_dens_grid;
std::vector<double> ion_temp_grid;
std::vector<double> ion_parr_flow_grid;
std::vector<double> ion_parr_flow_r_grid;
std::vector<double> ion_parr_flow_t_grid;
std::vector<double> ion_parr_flow_z_grid;

MagneticFieldFileDataParams *mag_arr;
MagneticFieldFileData magnetic_data;
  void broadcastMagneticData(MagneticFieldFileData& data);
MagneticFieldFileData readMagneticFieldFileData(const std::string& filePath);
MagneticFieldFileDataParams bilinearInterpolationMagneticField(
    int icell, const MagneticFieldFileData &data);
PlasmaFileParams bilinearInterpolationPlasma(
    int icell, const PlasmaFileData &data);
int findMeshTriangle(const PlasmaFileData &data, double r, double z) const;
int findNearestMappedTriangle(const PlasmaFileData &data, double r, double z, double max_dist) const;
bool meshLookupPlasma(int icell, const PlasmaFileData &data, PlasmaFileParams &out) const;
bool meshLookupPlasmaAtPoint(const PlasmaFileData &data, double r, double z,
                             PlasmaFileParams &out) const;

// Equilibrium-based magnetic geometry (exact ψ derivatives)
EquilibriumData equ_data;
MagneticGeometry *geom_arr;   // size = nlocal, precomputed per cell
int has_equilibrium;           // 1 if equilibrium file was loaded
std::string equilibriumPath;

EquilibriumData readEquilibriumFile(const std::string &path);
bool readEquilibriumFromPlasmaH5(const std::string &plasma_h5_path,
                                  EquilibriumData &data);
void broadcastEquilibriumData(EquilibriumData &data);
void computeMagneticGeometry(int icell, const EquilibriumData &equ,
                             MagneticGeometry &geom);

protected:
  enum InputMode { MODE_FILE=0, MODE_CONSTANT, MODE_ANALYTIC, MODE_BACKGROUND };
  InputMode input_mode = MODE_FILE;

int nglocal,groupbit;
std::string plasmaStatePath;
std::string magneticFieldsPath;
std::string background_fix_id;  // fix ID for background mode
double bconst[3];
double econst[3];
double teconst;
double ticonst;
double niconst;
double neconst;
double parrflowconst;
double analytic_r0;
double analytic_ne0;
double analytic_te0;
double analytic_te1;
double analytic_ti0;
double analytic_eps;
double analytic_x0;
double analytic_y0;
int analytic_use_x0;
int analytic_use_y0;

int nvalue;        // number of requested outputs (columns)
int *value;        // which outputs (enum)
std::vector<BilinearStencil> plasma_stencil;
std::vector<BilinearStencil> magnetic_stencil;
  BilinearStencil makeStencilAtPoint(
      const double xyz[3],
      const std::vector<double> &r_vals,
      const std::vector<double> &z_vals) const;
  void precomputeStencils(const std::vector<double> &r_vals,
                          const std::vector<double> &z_vals,
                          std::vector<BilinearStencil> &stencil);
  double interpField2D(const std::vector<std::vector<double>> &field,
                       const BilinearStencil &s) const;
  void gradField2D(const std::vector<std::vector<double>> &field,
                   const BilinearStencil &s,
                   double &grad_r,
                   double &grad_z) const;
  double interpField3DFlat(const std::vector<double> &field,
                           int ispec, int nz, int nr,
                           const BilinearStencil &s) const;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

*/
