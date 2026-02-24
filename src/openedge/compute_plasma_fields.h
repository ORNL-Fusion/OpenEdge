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
  std::vector<double> r;   
  std::vector<double> z;  
  std::vector<std::vector<double>> dens_e, temp_e;
  std::vector<std::vector<double>> dens_i, temp_i;
  std::vector<std::vector<double>> parr_flow_r, parr_flow_t, parr_flow_z, parr_flow;
  std::vector<std::vector<double>> grad_temp_e_r, grad_temp_e_t, grad_temp_e_z;
  std::vector<std::vector<double>> grad_temp_i_r, grad_temp_i_t, grad_temp_i_z;
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
};

class ComputePlasmaFields : public Compute {
 public:
  ComputePlasmaFields(class SPARTA *, int, char **);
  ~ComputePlasmaFields();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();

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

protected:
  struct BilinearStencil {
    int ir1 = 0, ir2 = 0, iz1 = 0, iz2 = 0;
    double t = 0.0, u = 0.0;
    double inv_dR = 0.0, inv_dZ = 0.0;
    double w11 = 0.0, w21 = 0.0, w12 = 0.0, w22 = 0.0;
    int valid = 0;
  };

  enum InputMode { MODE_FILE=0, MODE_CONSTANT, MODE_ANALYTIC };
  InputMode input_mode = MODE_FILE;

int nglocal,groupbit;
std::string plasmaStatePath;
std::string magneticFieldsPath;
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
