/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/background
   Coulomb collisions of kinetic charged particles against a Maxwellian
   fluid background plasma. Each charged simulation particle pairs with
   a virtual partner sampled from the prescribed (Ti, Ni, Vpar, B-field)
   background. Nanbu (1997) algorithm. Thin subclass of FixCoulombBase.

   Syntax:
     fix ID coulomb/background Nevery \
         {plasma TeSrc NeSrc | background FIXID} \
         A_bg Z_bg [TiSrc NiSrc VparSrc BxSrc BySrc BzSrc]

   The bg-source list is required when using `plasma`; with `background`
   the same fix supplies all bg fields and the source list is omitted.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coulomb/background,FixCoulombBackground)

#else

#ifndef SPARTA_FIX_COULOMB_BACKGROUND_H
#define SPARTA_FIX_COULOMB_BACKGROUND_H

#include "fix_coulomb_base.h"

namespace SPARTA_NS {

class FixCoulombBackground : public FixCoulombBase {
 public:
  FixCoulombBackground(class SPARTA *, int, char **);
};

}  // namespace SPARTA_NS

#endif
#endif
