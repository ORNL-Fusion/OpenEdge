#ifdef FIX_CLASS

FixStyle(droplet/charge,FixDropletCharge)
FixStyle(droplet_charge,FixDropletCharge)

#else

#ifndef SPARTA_FIX_DROPLET_CHARGE_H
#define SPARTA_FIX_DROPLET_CHARGE_H

#include "fix.h"
#include "grid_src.h"

namespace SPARTA_NS {

class ComputePlasmaFields;

class FixDropletCharge : public Fix {
 public:
  FixDropletCharge(class SPARTA*, int, char**);
  ~FixDropletCharge() override;

  int setmask() override;
  void init() override;
  void start_of_step() override;
  void end_of_step() override;
  double memory_usage() override;

 protected:
  int use_grid_plasma = 0;
  int use_point_plasma = 0;
  CollGridSrc srcTe, srcTi, srcNe, srcNi;
  ComputePlasmaFields *point_plasma_compute = nullptr;

  double **plasma_grid = nullptr;  // [nlocal][4] : Te,Ti,Ne,Ni
  int maxgrid_plasma = 0;

  double seed_radius = -1.0;       // optional fallback droplet radius [m]
  double seed_mass = -1.0;         // optional fallback droplet mass [kg]
  double ion_mass_amu = 2.0;       // background ion mass for OML [amu]
  int thermionic_on = 0;           // include Richardson thermionic term
  double richardson_A = 1.2e6;     // A/(m^2 K^2)
  double work_function_eV = 2.9;   // eV
  double seed_temp = -1.0;         // optional fallback droplet temperature [K]
  int diag_flag = 0;
  int diag_every = 100;

  void apply_charge_update();
  bool solve_phi_oml(double Te_eV, double Ti_eV, double ne_m3, double ni_m3,
                     double Td_K, double rd_m, double &phi_V) const;
  void compute_plasma_grid();
  inline void refresh_compute_src(CollGridSrc &S);
  bool point_plasma_ready() const;

  inline double read_cell_src(const CollGridSrc& S, int icell, int col_plasma) const;
};

}  // namespace SPARTA_NS

#endif
#endif
