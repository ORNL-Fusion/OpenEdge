/* ----------------------------------------------------------------------
   OpenEdge: fix drag — Epstein / Coulomb drag on droplets.
   Plasma background (Te, Ti, Ni, Vpar, Br, Bt, Bz) is pulled at the
   particle position from fix background via interp2D / bfield_at.

   Syntax:
     fix ID drag Nevery A_bg Z_bg background PD \
         [gravity gx gy gz] \
         [model epstein|coulomb] \
         [coulomb/chi V] [coulomb/delta V] [coulomb/lnlambda V] \
         [mass M] [radius R] [temp T]

   Integration: leapfrog-split half-kicks bracketing Boris
   (START_OF_STEP + END_OF_STEP).
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(particulate/drag,FixDropletDrag)
#else
#ifndef SPARTA_FIX_DROPLET_DRAG_H
#define SPARTA_FIX_DROPLET_DRAG_H

#include "fix.h"
#include "particle.h"
#include <string>

namespace SPARTA_NS {

class FixBackground;

class FixDropletDrag : public Fix {
 public:
  FixDropletDrag(class SPARTA *, int, char **);
  ~FixDropletDrag() override;

  int  setmask() override;
  void init() override;
  void start_of_step() override;
  void end_of_step() override;

  bool   use_gravity = false;
  double g_input_[3] = {0.0, 0.0, 0.0};

 protected:
  std::string plasma_fix_id_;
  FixBackground *pd_ = nullptr;

  double A_background      = 2.0;
  double Z_background      = 1.0;
  double rho_d             = 534.0;   // overridden by `material NAME`
  char   mat_name_[16]     = "";      // optional grain material
  int    self_consistent_  = 1;       // coulomb/self: per-particle chi/delta/lnL
  int    efield_on_        = 1;       // DUSTT F_E = Z_d e E (efield yes|no)
  int    neutrals_on_      = 1;       // DUSTT F_fric,n (neutral mass =
                                      //  A_bg amu; auto-off without data)
  int    dq_custom_        = -1;      // particulate_charge custom index

  double chi_coulomb       = 0.0;
  double delta_ite         = 1.0;
  double ln_lambda_coulomb = 10.0;

  double seed_mass   = -1.0;
  double seed_radius = -1.0;
  double seed_temp   = -1.0;

  // Optional mixture filter. -1 = no filter (act on every particle).
  // When >= 0, only species in this mixture's groups are processed.
  int imix = -1;

  void kick_half(double dt_half);
  double ion_drag_nu(double Ni, double Ti_eV, double rd_m) const;
  double coulomb_multiplier(double u) const;
  double coulomb_multiplier(double u, double chi, double delta,
                            double lnlam) const;
};

}
#endif
#endif
