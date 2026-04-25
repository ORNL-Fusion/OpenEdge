/* ----------------------------------------------------------------------
   OpenEdge:
   Per-surface adatom desorption flux. Arrhenius release of D-on-Li
   surface-adsorbed atoms (Yad model from liquid_metal_strip.h).
   Reads Tsurf from a per-surf custom attribute and the incident D+
   flux from a per-surf compute or fix output. Output is a per-surf
   vector consumable by `fix surface/emit/source`.

   Syntax:
     compute ID surface/chemical/adatom surfgroup \
             tsurf <surf-custom-name> \
             dp_flux <c_id|f_id|c_id[col]|f_id[col]> \
             [Yad VAL] [Yad_Yps VAL] [f_ad VAL] \
             [A_arr VAL] [E_eff VAL]

   Default coefficients match `fix liquid_metal` legacy values:
     Yad=1e-3, Yad_Yps=1, f_ad=1, A_arr=1e-7, E_eff=0.9 eV.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surface/chemical/adatom,ComputeSurfaceChemicalAdatom)

#else

#ifndef SPARTA_COMPUTE_SURFACE_CHEMICAL_ADATOM_H
#define SPARTA_COMPUTE_SURFACE_CHEMICAL_ADATOM_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeSurfaceChemicalAdatom : public Compute {
 public:
  ComputeSurfaceChemicalAdatom(class SPARTA *, int, char **);
  ~ComputeSurfaceChemicalAdatom();
  void init();
  void compute_per_surf();
  void reallocate();

 protected:
  int groupbit;
  int tsurf_index;          // surf-custom index for Tsurf
  char *tsurf_name;

  // D+ flux source: compute or fix per-surf vector/array
  enum { SRC_NONE, SRC_COMPUTE, SRC_FIX };
  int dp_kind;
  char *dp_id;
  int dp_col;               // 1-based column for array sources, 0 for vector
  int dp_index;             // resolved compute/fix index

  // adatom physics coefficients
  double Yad_D_Li;
  double Yad_Yps;
  double f_ad;
  double A_arr;
  double E_eff;

  int nslocal;
};

}

#endif
#endif
