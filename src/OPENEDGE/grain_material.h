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
#include <cstddef>
#include <cstdint>

namespace SPARTA_NS {

// Bits record values supplied explicitly by an input deck.  Built-in Li/B
// entries deliberately have a zero mask: they preserve DUSTT compatibility,
// but cannot silently satisfy the provenance contract of model dis2021.
enum GrainMaterialProperty : std::uint64_t {
  GRAIN_MAT_RHO          = 1ULL << 0,
  GRAIN_MAT_CP           = 1ULL << 1,
  GRAIN_MAT_CP_SOLID     = 1ULL << 2,
  GRAIN_MAT_MASS_AMU     = 1ULL << 3,
  GRAIN_MAT_HVAP         = 1ULL << 4,
  GRAIN_MAT_ANTOINE_A    = 1ULL << 5,
  GRAIN_MAT_ANTOINE_B    = 1ULL << 6,
  GRAIN_MAT_EMISSIVITY   = 1ULL << 7,
  GRAIN_MAT_WORK_FUNCTION = 1ULL << 8,
  GRAIN_MAT_RICHARDSON   = 1ULL << 9,
  GRAIN_MAT_TMELT        = 1ULL << 10,
  GRAIN_MAT_HMELT        = 1ULL << 11,
  GRAIN_MAT_TENSILE      = 1ULL << 12,
  GRAIN_MAT_SEE_DELTA_M  = 1ULL << 13,
  GRAIN_MAT_SEE_E_M      = 1ULL << 14
};

struct GrainMaterial {
  char name[16];
  char provenance_id[96]; // source-set or documented-assumption identifier
  std::uint64_t explicit_mask;
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
  double see_delta_m;      // secondary-emission max yield [-]; 0 = no SEE data
  double see_E_m_eV;       // incident energy at max yield [eV]
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

// Return true and write a comma-separated list when required properties were
// not supplied explicitly.  A DIS fix uses this to fail closed before any
// trajectories are advanced.
bool grain_material_missing_properties(const GrainMaterial *,
                                       std::uint64_t required,
                                       char *buffer, std::size_t nbuffer);

class MaterialCmd : protected Pointers {
 public:
  MaterialCmd(class SPARTA *sparta) : Pointers(sparta) {}
  void command(int, char **);
};

}

#endif
#endif
