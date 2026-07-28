/* ----------------------------------------------------------------------
   OpenEdge: fix droplet/charge — OML charging of droplets against a
   plasma background carried by fix background.

   Syntax:
     fix ID droplet/charge Nevery background PD \
         [ion_mass_amu M] [thermionic yes|no] \
         [richardson_A V] [work_function_eV V] \
         [radius R] [mass M] [temp T]

   The fix solves the OML potential equation at each droplet position and
   writes the resulting per-particle charge (in e-units) into the custom
   DOUBLE vector "droplet_charge". The pusher reads it in update.cpp and
   uses it in place of species[is].charge whenever it is non-zero, so each
   droplet sees the local-plasma charge rather than a species mean.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(droplet/charge,FixDropletCharge)
FixStyle(grain/charge,FixDropletCharge)
FixStyle(droplet_charge,FixDropletCharge)

#else

#ifndef SPARTA_FIX_DROPLET_CHARGE_H
#define SPARTA_FIX_DROPLET_CHARGE_H

#include "fix.h"
#include <vector>
#include <string>

namespace SPARTA_NS {

class FixBackground;

class FixDropletCharge : public Fix {
 public:
  FixDropletCharge(class SPARTA *, int, char **);
  ~FixDropletCharge() override;

  int  setmask() override;
  void init() override;
  void start_of_step() override;
  void end_of_step() override;
  double memory_usage() override;

 protected:
  std::string plasma_fix_id_;
  FixBackground *pd_ = nullptr;

  double seed_radius      = -1.0;
  double seed_mass        = -1.0;
  double seed_temp        = -1.0;
  double ion_mass_amu     =  2.0;
  int    thermionic_on    =  0;
  double richardson_A     =  1.2e6;
  double work_function_eV =  2.9;
  char   mat_name_[16]    = "";   // optional grain material (init() applies)
  int    wf_set_          = 0;    // explicit keyword overrides material
  int    ra_set_          = 0;
  const struct GrainMaterial *mat_ = nullptr;
  class RanKnuth *random_ = nullptr;
  double breakup_beta_    = 0.1;  // DUSTT electrostatic-disruption beta
  bigint nbreak_          = 0;
  std::vector<int> split_list_;

  // Optional mixture filter. -1 = no filter. >= 0 = only species in this
  // mixture's groups are charged.
  int imix = -1;

  // Per-particle custom DOUBLE vector "droplet_charge" (e units).
  int qcustom = -1;

  void apply_charge_update();
  bool solve_phi_oml(double Te_eV, double Ti_eV, double ne_m3, double ni_m3,
                     double Td_K, double rd_m, double &phi_V) const;
};

}

#endif
#endif
