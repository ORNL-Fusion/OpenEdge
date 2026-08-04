/* ----------------------------------------------------------------------
   OpenEdge: grain material registry — dust/droplet/powder properties.

   One place for the solid/liquid material data consumed by the grain
   physics fixes (droplet/evaporate, droplet/drag, droplet/charge, ...).
   Built-in materials: Li, B. Decks can add or override with the
   `material` command:

       material Li rho 534 cp 4200
       material B  emissivity 0.75
       material MyStuff mass_amu 9.0 rho 1850 ...   (new material)

   Fixes select a material with their `material NAME` keyword and default
   to "Li" for backward compatibility with the original hardcoded model.
------------------------------------------------------------------------- */

#ifdef COMMAND_CLASS

CommandStyle(material,MaterialCmd)

#else

#ifndef OPENEDGE_GRAIN_MATERIAL_H
#define OPENEDGE_GRAIN_MATERIAL_H

#include "pointers.h"

namespace SPARTA_NS {

struct GrainMaterial {
  char name[16];
  double rho;              // solid/liquid mass density [kg/m^3]
  double cp;               // specific heat [J/kg/K] (liquid, or single-phase)
  double cp_solid;         // specific heat below tmelt_K [J/kg/K]; <= 0
                           //  falls back to cp (legacy single-cp behavior)
  double mass_amu;         // atomic mass [amu]
  double hvap_J_mol;       // latent heat of evaporation/sublimation [J/mol]
  double antoine_a;        // log10 p_sat[atm] = a + b/T  (b < 0; the
  double antoine_b;        //  evaporate fix multiplies by 760 -> mmHg)
  double emissivity;       // total hemispherical emissivity [-]
  double work_function_eV; // thermionic work function [eV]
  double richardson_A;     // Richardson constant [A m^-2 K^-2]
  double tmelt_K;          // melting temperature [K]
  double hmelt_J_mol;      // latent heat of fusion [J/mol]
  double tensile_Pa;       // tensile strength [Pa]; 0 = no breakup
};

// Find by name (case-sensitive). Returns nullptr if unknown.
const GrainMaterial *grain_material_find(const char *name);

// Phase-aware specific heat: cp_solid below the melt when provided,
// cp otherwise. Materials without cp_solid keep the single-cp behavior.
inline double grain_material_cp(const GrainMaterial *m, double T_K)
{
  if (m->cp_solid > 0.0 && m->tmelt_K > 0.0 && T_K < m->tmelt_K)
    return m->cp_solid;
  return m->cp;
}

// Find-or-create a mutable entry (used by the material command).
GrainMaterial *grain_material_define(const char *name);

class MaterialCmd : protected Pointers {
 public:
  MaterialCmd(class SPARTA *sparta) : Pointers(sparta) {}
  void command(int, char **);
};

}

#endif
#endif
