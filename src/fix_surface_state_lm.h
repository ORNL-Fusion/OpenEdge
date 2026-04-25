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

FixStyle(surface/state/lm,FixSurfaceStateLm)

#else

#ifndef SPARTA_FIX_SURFACE_STATE_LM_H
#define SPARTA_FIX_SURFACE_STATE_LM_H

#include "fix.h"
#include "surf.h"
#include "liquid_metal_strip.h"

namespace SPARTA_NS {

class ComputePlasmaFields;

class FixSurfaceStateLm : public Fix {
 public:
  FixSurfaceStateLm(class SPARTA *, int, char **);
  virtual ~FixSurfaceStateLm();
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

  // Per-surf custom attribute indices for the fix's outputs.
  // Tsurf and h_film are the model's state. Evaporation and adatom
  // fluxes are now produced by `compute surface/chemical/evaporation`
  // and `compute surface/chemical/adatom` from these state values.
  int tindex;           // Tsurf [C]
  int hindex;           // film thickness [m]

  char *id_custom_t;
  char *id_custom_h;

  // the strip solver
  LiquidMetal::Strip strip;

  // geometry mapping: surf element -> strip x-station
  std::vector<int> surf_to_strip;
  std::vector<double> surf_arc_len;
  void build_geometry_map();

  // helpers
  void gather_heat_flux();
};

}

#endif
#endif
