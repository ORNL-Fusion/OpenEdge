/* ----------------------------------------------------------------------
   OpenEdge - Plasma-edge particle transport code
   https://github.com/ORNL-Fusion/OpenEdge

   fix liquid_metal: MHD liquid metal film model for divertor surfaces.
   Solves Smolentsev shallow-water MHD + heat transfer equations on a
   1D strip along the divertor, computes surface temperature, Li
   evaporation flux, and film thickness as per-surf custom attributes.

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
  int hf_source;        // COMPUTE or FIX or CONSTANT
  char *id_hf;
  class Compute *chf;
  class Fix *fhf;
  int hf_index;         // column index for array source
  double hf_constant;   // constant heat flux [W/m^2] if source=CONSTANT

  // custom per-surf attribute indices
  int tindex;           // Tsurf [K or C]
  int evap_index;       // evaporation flux [atoms/m^2-s]
  int hindex;           // film thickness [m]

  char *id_custom_t;
  char *id_custom_evap;
  char *id_custom_h;

  // the strip solver
  LiquidMetal::Strip strip;

  // geometry mapping: surf element -> strip x-station
  std::vector<int> surf_to_strip;
  std::vector<double> surf_arc_len;
  void build_geometry_map();

  // helper
  void gather_heat_flux();
};

}

#endif
#endif
