/* ----------------------------------------------------------------------
   OpenEdge: Impurity Transport in Modeling of SOL and Edge Physics.
   Built on SPARTA (parallel DSMC code).
   Abdourahmane Diaw, diawa@ornl.gov (2023)
   Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

// FixDrag: Epstein (+ optional Coulomb correction) drag force on droplet particles.
//
// Replaces the retired fix viscous.  Registered under three style names so
// existing input decks continue to work:
//
//   fix ID drag           ...    (canonical new name)
//   fix ID droplet/drag/force ...  (legacy alias)
//   fix ID droplet_drag_force ...  (legacy alias)
//
// Syntax:
//   fix ID drag <nevery> <A_bg> <Z_bg>
//        plasma <TeSrc> <TiSrc> <NiSrc> <VparSrc>
//        bfield <BrSrc> <BtSrc> <BzSrc>
//        [gravity gx gy gz]
//        [model epstein|coulomb]
//        [coulomb/chi val] [coulomb/delta val] [coulomb/lnlambda val]
//        [mass val] [radius val] [temp val]
//        [diag yes|no] [diag/every N]
//
// Sources for plasma and B-field quantities accept either:
//   c_ID[col]   -- column col (1-based) of a per-grid compute
//   VARNAME     -- a per-grid SPARTA variable
//
// Integration: Leapfrog-split half-kicks bracketing Boris (START_OF_STEP +
// END_OF_STEP). Sources are resolved once per timestep in start_of_step and
// the cached array pointers are reused by end_of_step.
//
// Particle loop: flat over particle->nlocal using p.icell (no sort required).
// This structure maps directly to Kokkos::parallel_for for GPU porting.

#ifdef FIX_CLASS
FixStyle(drag,FixDrag)
FixStyle(droplet/drag/force,FixDrag)
FixStyle(droplet_drag_force,FixDrag)
#else
#ifndef SPARTA_FIX_DRAG_H
#define SPARTA_FIX_DRAG_H

#include "fix.h"
#include "grid_src.h"   // CollGridSrc, CollSrcKind
#include "particle.h"
#include <cstdio>

namespace SPARTA_NS {

class FixDrag : public Fix {
 public:
  FixDrag(class SPARTA *, int, char **);
  ~FixDrag() override;

  int    setmask() override;
  void   init() override;
  void   start_of_step() override;  // resolve sources once + pre-Boris half-kick
  void   end_of_step() override;    // post-Boris half-kick (reuses cached sources)
  double memory_usage() override;

  // Gravity vector (Cartesian input; rotated to local basis internally)
  bool   use_gravity = false;
  double g_input_[3] = {0.0, 0.0, 0.0};

  // Cylindrical geometric terms: centrifugal (v_φ²/R) + angular-momentum (-v_φ*v_R/R).
  // Enabled by keyword 'cylindrical yes' for 2D non-axisymmetric runs (x=R, y=Z).
  // This makes the fix DUSTT-style: 2D RZ grid + full 3D cylindrical velocity dynamics.
  bool use_cylindrical = false;

 protected:
  // Drag model selector
  enum DragModel { DRAG_EPSTEIN = 0, DRAG_COULOMB = 1 };
  int drag_model = DRAG_EPSTEIN;

  // Source descriptors — plasma (Te, Ti, Ni, Vpar) and B-field (Br, Bt, Bz).
  // arr_cache and src_index are filled once per timestep in refresh_sources().
  int         use_grid_plasma = 0;
  int         use_grid_bfield = 0;
  CollGridSrc srcTe, srcTi, srcNi, srcVpar;
  CollGridSrc srcBr, srcBt, srcBz;

  // Per-cell scratch buffers for the grid-variable (VAR) source path.
  double **plasma_grid = nullptr;   // [maxgrid_plasma][4]: Te, Ti, Ni, Vpar
  double **b_grid      = nullptr;   // [maxgrid_b][3]:      Br, Bt, Bz
  int maxgrid_plasma   = 0;
  int maxgrid_b        = 0;

  // Physical parameters
  double A_background      = 2.0;    // background ion atomic mass (amu)
  double Z_background      = 1.0;    // background ion charge state (kept for future)
  double rho_d             = 534.0;  // droplet material density (kg/m^3)
  double alpha_E           = 1.26;   // Epstein accommodation coefficient

  // Coulomb closure parameters (dimensionless)
  double chi_coulomb       = 0.0;    // Coulomb chi
  double delta_ite         = 1.0;    // ion-to-electron thermal energy ratio
  double ln_lambda_coulomb = 10.0;   // Coulomb logarithm

  // Optional one-time particle state seeding (used when fix evaporation is absent)
  double seed_mass   = -1.0;
  double seed_radius = -1.0;
  double seed_temp   = -1.0;

  // Diagnostics
  int diag_flag  = 0;
  int diag_every = 100;

  // ---- helpers -------------------------------------------------------

  // Resolve compute/var source pointers once per timestep (called from start_of_step).
  void refresh_sources();

  // Single-source refresh: fills S.arr_cache / S.src_index without a
  // virtual call per particle.  Safe to call twice in one step (cache_ts guard).
  inline void refresh_src(CollGridSrc &S);

  // Inline cell-value reader — zero virtual calls in the hot particle loop.
  // var_buf and var_col are the VAR-path fallback (plasma_grid or b_grid).
  //
  // KOKKOS_NOTE: mark KOKKOS_INLINE_FUNCTION and replace double** with
  // Kokkos::View<double**> when porting.
  inline double read_cell(const CollGridSrc &S, int icell,
                          double **var_buf, int var_col) const;

  // Fill VAR scratch buffers from SPARTA grid variables.
  void compute_plasma_grid();
  void compute_bfield_grid();

  // Half-kick of all local particles: dv/dt = -nu*(v - upar) + g over dt_half.
  // do_diag=true only for the pre-kick (start_of_step) when diagnostics are on.
  //
  // KOKKOS_NOTE: the flat for-loop inside is the target for Kokkos::parallel_for.
  void kick_half(double dt_half, bool do_diag);

  // Physics kernels — pure functions, no hidden state.
  // KOKKOS_NOTE: mark KOKKOS_INLINE_FUNCTION when porting.
  double epstein_nu(double Ni, double Ti_eV, double rd_m) const;
  double coulomb_multiplier(double u) const;

  // Argument-parsing helpers (called from constructor)
  void parse_src   (const char *tok, CollGridSrc &dst, const char *label);

  // Source-binding helpers (called from init)
  void bind_var    (CollGridSrc &S, const char *label);
  void bind_compute(CollGridSrc &S, const char *label);
};

} // namespace SPARTA_NS
#endif
#endif
