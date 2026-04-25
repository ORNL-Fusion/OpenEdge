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
   fix viscous — DEPRECATED, retained for backward compatibility only.
   Use `fix drag` (FixDropletDrag) instead; the droplet/drag/force and
   droplet_drag_force aliases have moved there.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
FixStyle(droplet/viscous,FixDropletViscous)
#else
#ifndef SPARTA_FIX_DROPLET_VISCOUS_H
#define SPARTA_FIX_DROPLET_VISCOUS_H

#include "fix.h"
#include "grid_src.h"
#include "particle.h"
#include <cstdio>

namespace SPARTA_NS {

class FixDropletViscous : public Fix {
public:
  FixDropletViscous(class SPARTA*, int, char**);
  ~FixDropletViscous() override;

  int  setmask() override;
  void init() override;
  void start_of_step() override;      // NEW: pre-Boris half-kick
  void end_of_step() override;        // post-Boris half-kick
  double memory_usage() override;
bool   use_gravity = false;
double g_input_[3] = {0.0, 0.0, 0.0};  // 


protected:
  enum DragModel { DRAG_EPSTEIN = 0, DRAG_COULOMB = 1 };

  // half-kick and parameter builder
  void   kick_half(double dt_half, int diag_phase);
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
  int    drag_model = DRAG_EPSTEIN;
  double rho_d  = 534.0;       // kg/m^3
  double alpha_E = 1.26;       // accommodation
  // Coulomb drag closure parameters (dimensionless inputs).
  double chi_coulomb = 0.0;    // Coulomb parameter chi
  double delta_ite = 1.0;      // ion-thermal energy ratio delta_ite
  double ln_lambda_coulomb = 10.0; // Coulomb logarithm
  double seed_mass = -1.0;     // optional one-time particle mass seed (kg)
  double seed_radius = -1.0;   // optional one-time particle radius seed (m)
  double seed_temp = -1.0;     // optional one-time particle temperature seed (K)
  int    diag_flag = 0;        // print viscous diagnostics
  int    diag_every = 100;     // print interval in timesteps

  // legacy worker (no longer used for a full dt kick)
  void   end_of_step_no_average();

  // grid helpers
  void   compute_plasma_grid();
  void   compute_bfield_grid();
  double fetch_compute_cell_value(const CollGridSrc& S, int icell);
  inline void refresh_compute_src(CollGridSrc &S);

  // Epstein frequency (SI)
  double epstein_nu(double Ni_m3, double Ti_eV, double rd_m) const;
  double coulomb_drag_multiplier(double u) const;
};

} // namespace SPARTA_NS
#endif
#endif
