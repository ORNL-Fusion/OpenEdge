/* ----------------------------------------------------------------------
   OpenEdge — Neutral and charge-state transport with plasma-wall interactions
   Built on SPARTA (sparta.github.io; Plimpton et al., Sandia National Labs).

   Author: Abdourahmane Diaw <diawa@ornl.gov>
           Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge

   Distributed under the GNU General Public License, same as SPARTA.
   See the LICENSE file at the top of the repository.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   fix gravity: uniform gravitational acceleration on particles.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(force/gravity,FixForceGravity)
#else
#ifndef SPARTA_FIX_FORCE_GRAVITY_H
#define SPARTA_FIX_FORCE_GRAVITY_H

#include "fix.h"

namespace SPARTA_NS {

class FixForceGravity : public Fix {
public:
  FixForceGravity(class SPARTA*, int, char**);
  ~FixForceGravity() override = default;

  int    setmask() override;
  void   init() override;
  void   start_of_step() override;
  void   end_of_step() override;
  double memory_usage() override;

private:
  // User-specified physical cylindrical components (g_R,g_Z,g_phi).
  double g_[3] = {0.0, 0.0, 0.0};

  void half_kick(double dt_half);
};

} // namespace SPARTA_NS
#endif
#endif
