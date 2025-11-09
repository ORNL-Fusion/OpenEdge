#ifdef FIX_CLASS

FixStyle(viscous,FixViscous)

#else

#ifndef SPARTA_FIX_VISCOUS_H
#define SPARTA_FIX_VISCOUS_H

#include "fix.h"
#include "fix_coll_background.h"   // CollSrcKind, CollGridSrc (canonical)
#include "particle.h"              // for Particle::OnePart
#include <cstdio>

namespace SPARTA_NS {

class RanKnuth;

class FixViscous : public Fix {
public:
  FixViscous(class SPARTA*, int, char**);
  ~FixViscous() override;

  int  setmask() override;
  void init() override;
  void end_of_step() override;
  double memory_usage() override;

protected:
  // RNG handle
  RanKnuth* rng = nullptr;

  // grid/plasma source selectors
  int use_grid_plasma = 0;
  int use_grid_bfield = 0;

  // plasma sources (Te, Ti, Ni, v_parallel) and B-field
  CollGridSrc srcTe, srcTi, srcNi, srcVpar;
  CollGridSrc srcBr, srcBt, srcBz;

  // cached grid arrays
  double **plasma_grid = nullptr;   // [nlocal][4] : Te,Ti,Ni,Vpar
  double **b_grid      = nullptr;   // [nlocal][3] : Br,Bt,Bz
  int maxgrid_plasma = 0;
  int maxgrid_b      = 0;

  // background ion identity for Epstein (mass)
  double A_background = 2.0;  // amu, e.g. D+
  double Z_background = 1.0;  // (not used by Epstein; kept for future)

  // Epstein parameters
  int    model_epstein = 1;   // 1 → Epstein drag active
  double rho_d = 534.0;       // kg/m^3 (Li)
  double alpha_E = 1.26;      // accommodation factor (~1.0–1.4)

  // main worker
  void end_of_step_no_average();

  // helpers
  void   compute_plasma_grid();
  void   compute_bfield_grid();
  double fetch_compute_cell_value(const CollGridSrc& S, int icell);
  inline void refresh_compute_src(CollGridSrc &S);

  // per-particle drag update
  void   backgroundCollisions(Particle::OnePart *ip);

  // Epstein frequency (SI)
  double epstein_nu(double Ni_m3, double Ti_eV, double rd_m) const;
};

} // namespace SPARTA_NS

#endif // SPARTA_FIX_VISCOUS_H
#endif // FIX_CLASS
