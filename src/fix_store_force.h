/* ----------------------------------------------------------------------
   OpenEdge: store one named per-particle transport-force contribution.

   This is the OpenEdge analogue of LAMMPS fix store/force, with an
   explicit mechanism selector because OpenEdge applies transport operators
   at several points in the timestep rather than through one force array.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(store/force,FixStoreForce)

#else

#ifndef SPARTA_FIX_STORE_FORCE_H
#define SPARTA_FIX_STORE_FORCE_H

#include "fix.h"

namespace SPARTA_NS {

class FixStoreForce : public Fix {
 public:
  enum Channel {
    ELECTRIC_PLASMA = 0,
    ELECTRIC_SHEATH,
    MAGNETIC,
    THERMAL_ION,
    THERMAL_ELECTRON,
    COULOMB_BACKGROUND,
    COULOMB_BINARY,
    NCHANNEL
  };

  FixStoreForce(class SPARTA *, int, char **);
  ~FixStoreForce();

  int setmask();
  void init();
  void start_of_step();
  void end_of_step();
  double memory_usage();

  Channel channel() const { return channel_; }
  bool enabled(int) const;
  void add_impulse(int, const double [3], double);

  static const char *channel_name(Channel);

 private:
  Channel channel_;
  bigint id_max_;
  int force_custom_;
  int step_custom_;

  void refresh_output();
  void touch(int);
};

FixStoreForce *find_store_force(class Modify *, FixStoreForce::Channel);

}  // namespace SPARTA_NS

#endif
#endif
