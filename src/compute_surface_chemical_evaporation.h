/* ----------------------------------------------------------------------
   OpenEdge:
   Per-surface bulk evaporation flux from local surface temperature.
   Antoine + Hertz-Knudsen for liquid metals (Li by default; coefficients
   live in liquid_metal_strip.h). Reads Tsurf from a per-surf custom
   attribute (typically populated by `fix surface/state/liquid_metal`).
   Output is a per-surf vector consumable by `fix surface/emit/source`.

   Syntax:
     compute ID surface/chemical/evaporation surfgroup \
             tsurf <surf-custom-name> [sticking VAL]
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surface/chemical/evaporation,ComputeSurfaceChemicalEvaporation)

#else

#ifndef SPARTA_COMPUTE_SURFACE_CHEMICAL_EVAPORATION_H
#define SPARTA_COMPUTE_SURFACE_CHEMICAL_EVAPORATION_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeSurfaceChemicalEvaporation : public Compute {
 public:
  ComputeSurfaceChemicalEvaporation(class SPARTA *, int, char **);
  ~ComputeSurfaceChemicalEvaporation();
  void init();
  void compute_per_surf();
  void reallocate();

 protected:
  int groupbit;
  int tsurf_index;     // surf-custom index for Tsurf
  char *tsurf_name;    // surf-custom name for Tsurf (e.g. "Tsurf_lm")
  double sticking;     // alpha in Hertz-Knudsen (default 1.0)

  int nslocal;         // # owned surfs on this rank
};

}

#endif
#endif
