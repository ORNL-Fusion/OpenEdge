#ifdef FIX_CLASS
FixStyle(gravity,FixGravity)
#else
#ifndef SPARTA_FIX_GRAVITY_H
#define SPARTA_FIX_GRAVITY_H

#include "fix.h"

namespace SPARTA_NS {

class FixGravity : public Fix {
public:
  FixGravity(class SPARTA*, int, char**);
  ~FixGravity() override = default;

  int    setmask() override;
  void   init() override;
  void   start_of_step() override;
  void   end_of_step() override;
  double memory_usage() override;

private:
  // User-specified components:
  //  - Axisymmetric: cylindrical (g_r, g_z, g_phi)
  //  - 2D/3D: Cartesian (g_x, g_y, g_z)
  double g_[3] = {0.0, 0.0, 0.0};

  void half_kick(double dt_half);
};

} // namespace SPARTA_NS
#endif
#endif
