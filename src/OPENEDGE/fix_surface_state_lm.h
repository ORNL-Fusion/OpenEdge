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
#include <string>
#include <vector>

namespace SPARTA_NS {

class ComputePlasmaFields;
class FixBackground;

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

  // fix background pointer for BACKGROUND mode (reads mesh/q_par,q_perp
  // from plasma.h5 via fix background's public interp2D).
  FixBackground *fbg;
  char *id_background;

  // target HDF5 file mode (TARGET)
  char *target_file;    // path to target_heatflux.h5
  char *target_leg;     // "outer" or "inner"
  std::vector<double> tgt_s;       // arc length [m] (shifted so [0]=0)
  std::vector<double> tgt_q;       // heat flux [W/m²]
  std::vector<double> tgt_gamma;   // D+ flux [m⁻²s⁻¹]
  std::vector<double> tgt_R;       // R [m] at each tgt_s (Rclfc)
  std::vector<double> tgt_Z;       // Z [m] at each tgt_s (Zclfc)
  void load_target_heatflux();

  // SOLPS b2pl postprocessor text-file mode (SOLPS_B2PL).
  // Currently consumes ld_tg_*.dat (target loading along arc length); a
  // future extension can autodetect wlld.dat. Fills tgt_s/tgt_q/tgt_gamma
  // (same internal storage as TARGET) so the heat-flux interpolation reuses
  // the existing path.
  char *solps_b2pl_path;            // path to ld_tg_*.dat
  double solps_m_ion_amu;           // ion mass for Bohm flux (default 2.014)
  void load_solps_b2pl();

  // static-mode: run the strip once during init() and freeze. Useful when
  // the heat-flux source does not evolve (e.g. solps_b2pl, target HDF5).
  // end_of_step() returns early once frozen_=1.
  int static_mode;
  int frozen_;
  void run_strip_and_write();       // factored from end_of_step()

  // Optional CSV dump of the strip's smooth (Smolentsev) profile after
  // each solve. Header row + one row per strip station n=1..Nx with
  // s [m], R [m], Z [m], Tsurf [degC], h [m], Wtot [W/m^2], Gamma_D [m^-2 s^-1],
  // Gamma_evap [m^-2 s^-1].  Empty path => no CSV.
  std::string csv_path;
  void write_strip_csv();

  // Per-surf custom attribute indices for the fix's outputs.
  // Tsurf and h_film are the model's state. Evaporation and adatom
  // fluxes are now produced by `compute surface/chemical/evaporation`
  // and `compute surface/chemical/adatom` from these state values.
  // Gamma_D_lm carries the D+ Bohm flux when the heat-flux source
  // provides it (SOLPS_B2PL); zero otherwise.
  int tindex;           // Tsurf [C]
  int hindex;           // film thickness [m]
  int gindex;           // Gamma_D [m^-2 s^-1]

  char *id_custom_t;
  char *id_custom_h;
  char *id_custom_g;

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
