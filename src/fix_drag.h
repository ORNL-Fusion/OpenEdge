/* ----------------------------------------------------------------------
   OpenEdge: fix drag — Epstein / Coulomb drag on droplets.
   Plasma background (Te, Ti, Ni, Vpar, Br, Bt, Bz) is pulled at the
   particle position from fix plasma/data via interp2D / bfield_at.

   Syntax:
     fix ID drag Nevery A_bg Z_bg plasma_data PD \
         [gravity gx gy gz] \
         [model epstein|coulomb] \
         [coulomb/chi V] [coulomb/delta V] [coulomb/lnlambda V] \
         [mass M] [radius R] [temp T]

   Integration: leapfrog-split half-kicks bracketing Boris
   (START_OF_STEP + END_OF_STEP).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(drag,FixDrag)
#else
#ifndef SPARTA_FIX_DRAG_H
#define SPARTA_FIX_DRAG_H

#include "fix.h"
#include "particle.h"
#include <string>

namespace SPARTA_NS {

class FixPlasmaData;

class FixDrag : public Fix {
 public:
  FixDrag(class SPARTA *, int, char **);
  ~FixDrag() override;

  int  setmask() override;
  void init() override;
  void start_of_step() override;
  void end_of_step() override;

  bool   use_gravity = false;
  double g_input_[3] = {0.0, 0.0, 0.0};

 protected:
  enum DragModel { DRAG_EPSTEIN = 0, DRAG_COULOMB = 1 };
  int drag_model = DRAG_EPSTEIN;

  std::string plasma_fix_id_;
  FixPlasmaData *pd_ = nullptr;

  double A_background      = 2.0;
  double Z_background      = 1.0;
  double rho_d             = 534.0;
  double alpha_E           = 1.26;

  double chi_coulomb       = 0.0;
  double delta_ite         = 1.0;
  double ln_lambda_coulomb = 10.0;

  double seed_mass   = -1.0;
  double seed_radius = -1.0;
  double seed_temp   = -1.0;

  void kick_half(double dt_half);
  double epstein_nu(double Ni, double Ti_eV, double rd_m) const;
  double coulomb_multiplier(double u) const;
};

}
#endif
#endif
