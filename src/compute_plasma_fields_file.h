/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(plasma/fields/file,ComputePlasmaFieldsFile)

#else

#ifndef SPARTA_COMPUTE_PLASMA_FIELDS_FILE_H
#define SPARTA_COMPUTE_PLASMA_FIELDS_FILE_H

#include "compute.h"
#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

  struct SheathParams {
  double phi;     // V  (negative near wall)
  double E_mag;   // V/m (magnitude along -n)
  double ne;      // m^-3
};

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
};

struct PlasmaFileParams {
  double dens_e;
  double temp_e;
  double dens_i;
  double temp_i;
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

class ComputePlasmaFieldsFile : public Compute {
 public:
  ComputePlasmaFieldsFile(class SPARTA *, int, char **);
  ~ComputePlasmaFieldsFile();
  void init();
  void compute_per_grid();
  void reallocate();
  bigint memory_usage();
  double sheath_phi_over_Te(double alpha_deg, double s_norm);
  double sheath_E_over_Te_per_lambdaD(double alpha_deg, double s_norm);

void compute_dist_grid3(
    int icell, double& mindistance, double*& norm, int& surf_id,
    int& type_group, int& mask_group, double surf_center[3],
    const double sdir_in[3], int sgroupbit_in, bool by_normal_distance, double cos_tilt_max);

void compute_dist_grid(int icell, double& mindistance, double*& norm, int& surf_id, int& type_group, int& mask_group, double surf_center[3]) ;

// set plasma data
std::unordered_map<int, PlasmaFileParams> plasma_map;
std::unordered_map<int, PlasmaFileParams> plasmaDataCache;
PlasmaFileData plasma_data;
void broadcastPlasmaData(PlasmaFileData& data);
PlasmaFileParams bilinearInterpolationPlasma(int icell, const PlasmaFileData& data) ;
PlasmaFileData readPlasmaFileData(const std::string& path);

// set magnetic field data
std::unordered_map<int, MagneticFieldFileDataParams> magnetic_map;
std::unordered_map<int, MagneticFieldFileDataParams> magneticFieldDataCache;
MagneticFieldFileData magnetic_data;
void initializeMagneticData();
void broadcastMagneticData(MagneticFieldFileData& data);
MagneticFieldFileDataParams bilinearInterpolationMagneticField(int icell, const MagneticFieldFileData& data);
MagneticFieldFileData readMagneticFieldFileData(const std::string& filePath);

protected:

int nglocal,groupbit,sgroupbit;
double sdir[3];
std::string plasmaStatePath;
std::string magneticFieldsPath;

int nvalue;        // number of requested outputs (columns)
int *value;        // which outputs (enum)
int *nmap;      // # of inputs per output col (always 1 here)
int **map;      // map[index][0] = which source slot to use
double **vals;        // [nglocal x nvalue] raw results per cell per keyword
double bconst[3];
double econst[3];
double teconst;
double ticonst;
double niconst;
double neconst;
double parrflowconst;
double exconst, eyconst, ezconst;
double nesheathconst;
int  have_bconst;     // 0/1: whether magnetic_field_constant was parsed
int  have_econst;     // 0/1: whether electric_field_constant was parsed
int  have_parrflowconst;
int have_nesheathconst;
int have_niconst;
int have_teconst;
int have_ticonst;
int have_neconst;
int  have_sheath;
double alpha_const_deg, dwall_const;
int query_tally_grid(int index, double **&array, int *&cols);
void post_process_grid(int index, int nsample,
                double **etally, int *emap, double *vec, int nstride);
void sheathEfieldChoduraBrooks(
    const double *B,
    double Te_eV,
    double Ti_eV,
    double ne0_m3,
    double alpha_deg,
    double d_wall_m,
    double V_DS,
    double V_MPS,
    double &E_mag_SI,
    double &ne_m3);

    // angle polynomial fd(α in deg) from GITRm
inline double fd_poly_deg(double a_deg) {
  const double a = a_deg;
  double fd =
      0.98992
    + 5.1220e-3  * a
    - 7.0040e-4  * a*a
    + 3.3591e-5  * a*a*a
    - 8.2917e-7  * std::pow(a,4)
    + 9.5856e-9  * std::pow(a,5)
    - 4.2682e-11 * std::pow(a,6);
  return std::min(1.0, std::max(0.0, fd));
}

SheathParams eval_ds_mps(double d_m,        // distance from wall (m), d >= 0
                       double Te_eV,      // electron temperature (eV)
                       double Ti_eV,      // ion temperature (eV)
                       double ne0_m3,     // upstream electron density (m^-3)
                       double B_T,        // magnetic field magnitude (T)
                       double alpha_deg,  // grazing angle in degrees
                       double mi_kg); // ion mass (kg)
void field_vector_along_minus_n(const SheathParams& o,
                                       const double n[3], // surface normal
                                       double& Ex, double& Ey, double& Ez);
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
