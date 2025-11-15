#ifdef FIX_CLASS
FixStyle(viscous,FixViscous)
#else
#ifndef SPARTA_FIX_VISCOUS_H
#define SPARTA_FIX_VISCOUS_H

#include "fix.h"
#include "fix_coll_background.h"
#include "particle.h"
#include <cstdio>

namespace SPARTA_NS {

class RanKnuth;

class FixViscous : public Fix {
public:
  FixViscous(class SPARTA*, int, char**);
  ~FixViscous() override;

  int  setmask() override;
  void init() override;
  void start_of_step() override;      // NEW: pre-Boris half-kick
  void end_of_step() override;        // post-Boris half-kick
  double memory_usage() override;
bool   use_gravity = false;
double g_input_[3] = {0.0, 0.0, 0.0};  // 


protected:
  RanKnuth* rng = nullptr;

  // half-kick and parameter builder
  void   kick_half(double dt_half);    
  inline void epstein_params(int icell, const Particle::OnePart &p,
                             double &nuE, double upar[3]);

  // source selectors
  int use_grid_plasma = 0, use_grid_bfield = 0;
  CollGridSrc srcTe, srcTi, srcNi, srcVpar;
  CollGridSrc srcBr, srcBt, srcBz;

  // cached arrays
  double **plasma_grid = nullptr;   // [nlocal][4] : Te,Ti,Ni,Vpar
  double **b_grid      = nullptr;   // [nlocal][3] : Br,Bt,Bz
  int maxgrid_plasma = 0, maxgrid_b = 0;

  // Epstein parameters
  double A_background = 2.0;   // amu (e.g., D+)
  double Z_background = 1.0;   // kept for future
  int    model_epstein = 1;
  double rho_d  = 534.0;       // kg/m^3
  double alpha_E = 1.26;       // accommodation

  // legacy worker (no longer used for a full dt kick)
  void   end_of_step_no_average();

  // grid helpers
  void   compute_plasma_grid();
  void   compute_bfield_grid();
  double fetch_compute_cell_value(const CollGridSrc& S, int icell);
  inline void refresh_compute_src(CollGridSrc &S);

  // legacy per-particle update (kept for reference)
  void   backgroundCollisions(Particle::OnePart *ip);

  // Epstein frequency (SI)
  double epstein_nu(double Ni_m3, double Ti_eV, double rd_m) const;
};

} // namespace SPARTA_NS
#endif
#endif
