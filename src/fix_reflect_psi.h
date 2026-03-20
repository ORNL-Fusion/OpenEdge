/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix reflect/psi: Reflect particles that cross a psi_norm boundary.

    At END_OF_STEP, evaluates normalized poloidal flux psi_n(R,Z) at
    each particle position using a G-EQDSK equilibrium file.  If
    psi_n < psi_threshold (particle has entered the core), the particle
    is reflected back to its previous position and its velocity is
    reversed along the radial direction.

    psi_n = (psi - simag) / (sibry - simag)
      0 at magnetic axis, 1 at separatrix, >1 in SOL

    Syntax:
      fix ID reflect/psi Nevery geqdsk PATH psi_norm VALUE [action reflect|delete]

    Example:
      fix fcore reflect/psi 1 geqdsk input/g174310.03500_153.X4.equ psi_norm 0.05
      fix fcore reflect/psi 1 geqdsk input/g174310.03500_153.X4.equ psi_norm 0.95 action delete
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(reflect/psi,FixReflectPsi)

#else

#ifndef SPARTA_FIX_REFLECT_PSI_H
#define SPARTA_FIX_REFLECT_PSI_H

#include "fix.h"
#include "geqdsk_reader.h"

namespace SPARTA_NS {

class FixReflectPsi : public Fix {
 public:
  FixReflectPsi(class SPARTA *, int, char **);
  ~FixReflectPsi();
  int  setmask();
  void init();
  void start_of_step();
  void end_of_step();

 protected:
  int nevery_;
  double psi_threshold_;     // normalized psi boundary (default 0.05)
  int action_;               // 0 = reflect, 1 = delete
  GeqdskReader geqdsk_;

  // previous position storage
  int nmax_prev_;
  double **x_prev_;
};

}  // namespace SPARTA_NS

#endif
#endif
