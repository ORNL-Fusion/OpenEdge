/* ----------------------------------------------------------------------
   OpenEdge: fix droplet/charge — OML charging of droplets against a
   plasma background carried by fix plasma/data.

   Syntax:
     fix ID droplet/charge Nevery plasma_data PD \
         [ion_mass_amu M] [thermionic yes|no] \
         [richardson_A V] [work_function_eV V] \
         [radius R] [mass M] [temp T]

   The fix solves the OML potential equation at each droplet position and
   stamps the resulting charge (in e-units) onto particle->species[is].charge
   as a per-species mean (all particles of a given species share a charge).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(droplet/charge,FixDropletCharge)
FixStyle(droplet_charge,FixDropletCharge)

#else

#ifndef SPARTA_FIX_DROPLET_CHARGE_H
#define SPARTA_FIX_DROPLET_CHARGE_H

#include "fix.h"
#include <string>

namespace SPARTA_NS {

class FixPlasmaData;

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
  FixPlasmaData *pd_ = nullptr;

  double seed_radius      = -1.0;
  double seed_mass        = -1.0;
  double seed_temp        = -1.0;
  double ion_mass_amu     =  2.0;
  int    thermionic_on    =  0;
  double richardson_A     =  1.2e6;
  double work_function_eV =  2.9;

  void apply_charge_update();
  bool solve_phi_oml(double Te_eV, double Ti_eV, double ne_m3, double ni_m3,
                     double Td_K, double rd_m, double &phi_V) const;
};

}

#endif
#endif
