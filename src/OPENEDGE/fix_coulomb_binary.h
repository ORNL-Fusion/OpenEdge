/* ----------------------------------------------------------------------
   OpenEdge: fix coulomb/binary
   Binary kinetic-kinetic Coulomb collisions among charged simulation
   particles in the same cell. Nanbu (1997) / Takizuka-Abe (1977)
   algorithm. Thin subclass of FixCoulombBase.

   Syntax:
     fix ID coulomb/binary Nevery {plasma TeSrc NeSrc | plasma_data FIXID}
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(coulomb/binary,FixCoulombBinary)

#else

#ifndef SPARTA_FIX_COULOMB_BINARY_H
#define SPARTA_FIX_COULOMB_BINARY_H

#include "fix_coulomb_base.h"

namespace SPARTA_NS {

class FixCoulombBinary : public FixCoulombBase {
 public:
  FixCoulombBinary(class SPARTA *, int, char **);
};

}  // namespace SPARTA_NS

#endif
#endif
