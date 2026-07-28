/* ----------------------------------------------------------------------
   OpenEdge - grain (dust/droplet/powder) material property registry.

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
  double cp;               // specific heat [J/kg/K]
  double mass_amu;         // atomic mass [amu]
  double hvap_J_mol;       // latent heat of evaporation/sublimation [J/mol]
  double antoine_a;        // log10 p_sat[atm] = a + b/T  (b < 0; the
  double antoine_b;        //  evaporate fix multiplies by 760 -> mmHg)
  double emissivity;       // total hemispherical emissivity [-]
  double work_function_eV; // thermionic work function [eV]
  double richardson_A;     // Richardson constant [A m^-2 K^-2]
  double tmelt_K;          // melting temperature [K]
  double hmelt_J_mol;      // latent heat of fusion [J/mol]
};

// Find by name (case-sensitive). Returns nullptr if unknown.
const GrainMaterial *grain_material_find(const char *name);

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
