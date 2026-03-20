/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix reflect/psi: Reflect particles that cross a psi_norm boundary.

    At END_OF_STEP, evaluates normalized poloidal flux psi_n(R,Z) at
    each particle position using an equilibrium file.  If
    psi_n < psi_threshold (particle has entered the core), the particle
    is reflected back to its previous position and its velocity is
    reversed along the radial direction.

    psi_n = (psi - psi_axis) / (psib - psi_axis)
      0 at magnetic axis, 1 at separatrix, >1 in SOL

    Supports both G-EQDSK and SOLPS .equ equilibrium formats.

    Syntax:
      fix ID reflect/psi Nevery equ PATH psi_norm VALUE [action reflect|delete]

    Example:
      fix fcore reflect/psi 1 equ input/g174310.03500_153.X4.equ psi_norm 0.926
      fix fcore reflect/psi 1 equ input/g174310.03500_153.X4.equ psi_norm 0.926 action delete
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(reflect/psi,FixReflectPsi)

#else

#ifndef SPARTA_FIX_REFLECT_PSI_H
#define SPARTA_FIX_REFLECT_PSI_H

#include "fix.h"

#include <string>
#include <vector>

namespace SPARTA_NS {

class FixReflectPsi : public Fix {
 public:
  FixReflectPsi(class SPARTA *, int, char **);
  ~FixReflectPsi();
  int  setmask();
  void init();
  void start_of_step();

 protected:
  int nevery_;
  double psi_threshold_;     // normalized psi boundary

  // Equilibrium data
  int nw_, nh_;
  double psi_axis_, psib_;
  std::vector<double> r_grid_, z_grid_;
  std::vector<double> psirz_;   // [nh * nw], row-major [z][r]

  // Interpolation
  double psi_norm_at_point(double R, double Z) const;

  // File readers
  void read_equ_file(const std::string &path);
};

}  // namespace SPARTA_NS

#endif
#endif
