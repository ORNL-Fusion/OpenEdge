/* ----------------------------------------------------------------------
   OpenEdge - Plasma-edge particle transport code
   https://github.com/ORNL-Fusion/OpenEdge

   fix liquid_metal: MHD liquid metal film model for divertor surfaces.
   Solves Smolentsev shallow-water MHD + heat transfer equations on a
   1D strip along the divertor, computes surface temperature, Li
   evaporation flux (Antoine+HK), ad-atom flux, and film thickness
   as per-surf custom attributes.

   Per-surf output columns (via f_ID[i][1..4]):
     1: Tsurf [C]
     2: evap_flux [atoms/m²/s]
     3: adatom_flux [atoms/m²/s]
     4: h_film [m]

   Contributing author: Abdou Diaw (ORNL)
   Based on Fortran code by Sergey Smolentsev (UCLA)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(liquid_metal,FixLiquidMetal)

#else

#ifndef SPARTA_FIX_LIQUID_METAL_H
#define SPARTA_FIX_LIQUID_METAL_H

#include "fix.h"
#include "surf.h"
#include "liquid_metal_strip.h"

namespace SPARTA_NS {

class ComputePlasmaFields;

class FixLiquidMetal : public Fix {
 public:
  FixLiquidMetal(class SPARTA *, int, char **);
  virtual ~FixLiquidMetal();
  int setmask();
  virtual void init();
  virtual void end_of_step();

 private:
  int groupbit;
  int nevery;
  int firstflag;

  // heat flux source (compute or fix)
  int hf_source;        // COMPUTE or FIX or CONSTANT or PLASMA or TARGET
  char *id_hf;
  class Compute *chf;
  class Fix *fhf;
  int hf_index;         // column index for array source
  double hf_constant;   // constant heat flux [W/m^2] if source=CONSTANT

  // plasma/fields compute for point-query mode (PLASMA)
  ComputePlasmaFields *cp_plasma;
  char *id_plasma;
  double hf_scale;      // heat flux multiplier (default 1.0)

  // target HDF5 file mode (TARGET)
  char *target_file;    // path to target_heatflux.h5
  char *target_leg;     // "outer" or "inner"
  std::vector<double> tgt_s;       // arc length [m]
  std::vector<double> tgt_q;       // heat flux [W/m²]
  std::vector<double> tgt_gamma;   // D+ flux [m⁻²s⁻¹]
  void load_target_heatflux();

  // D+ flux source for ad-atom calculation
  int dp_source;        // COMPUTE, FIX, CONSTANT, or PLASMA
  char *id_dp;
  class Compute *cdp;
  class Fix *fdp;
  int dp_index;
  double dp_constant;   // constant D+ flux [ions/m²/s]

  // ad-atom model parameters
  double Yad_D_Li;      // ad-atom yield D on Li (default 1e-3)
  double Yad_Yps;       // Yad/Yps ratio (default 1.0)
  double f_ad_neutral;  // neutral fraction (default 1.0)
  double A_arrhenius;   // Arrhenius pre-factor (default 1e-7)
  double E_eff_eV;      // effective binding energy [eV] (default 0.9)

  // custom per-surf attribute indices
  int tindex;           // Tsurf [C]
  int evap_index;       // evaporation flux [atoms/m^2-s]
  int adatom_index;     // ad-atom flux [atoms/m^2-s]
  int hindex;           // film thickness [m]

  char *id_custom_t;
  char *id_custom_evap;
  char *id_custom_adatom;
  char *id_custom_h;

  // the strip solver
  LiquidMetal::Strip strip;

  // geometry mapping: surf element -> strip x-station
  std::vector<int> surf_to_strip;
  std::vector<double> surf_arc_len;
  void build_geometry_map();

  // helpers
  void gather_heat_flux();
  void gather_dp_flux(std::vector<double> &dp_per_surf);
};

}

#endif
#endif
