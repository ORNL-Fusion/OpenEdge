/* ----------------------------------------------------------------------
   OpenEdge: Impurity Transport in Modeling of SOL and Edge Physics.
   Built on SPARTA (parallel DSMC code).
   Abdourahmane Diaw, diawa@ornl.gov (2023)
   Oak Ridge National Laboratory
   https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

// Performance notes (vs. retired fix_viscous):
//
//  Bug 1 fixed: arr_cache resolved ONCE per timestep in refresh_sources();
//               the hot particle loop reads S.arr_cache[icell][S.src_index]
//               directly — zero virtual/function calls per particle.
//
//  Bug 2 fixed: particle->sort() removed; the flat loop over nlocal uses
//               p.icell, so sorted order is never required.
//
//  Bug 3 fixed: sources are refreshed only in start_of_step().  end_of_step()
//               reuses the same arr_cache pointers (plasma fields don't change
//               within a single timestep).
//
//  MPI: each rank owns its local particle subset; MPI_Allreduce only for
//       optional diagnostics (correct usage with separate send/recv buffers).
//
//  Kokkos path: marked with KOKKOS_NOTE comments.  The flat particle loop in
//               kick_half() maps directly to Kokkos::parallel_for.

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

#include "fix_drag.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
#include "input.h"
#include "modify.h"
#include "compute.h"
#include "variable.h"
#include "math_const.h"

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixDrag::FixDrag(SPARTA *sparta, int narg, char **arg)
    : Fix(sparta, narg, arg)
{
  // fix ID drag <nevery> <A_bg> <Z_bg>
  //      plasma <Te> <Ti> <Ni> <Vpar>
  //      bfield <Br> <Bt> <Bz>
  //      [gravity gx gy gz]
  //      [model epstein|coulomb]
  //      [coulomb/chi v] [coulomb/delta v] [coulomb/lnlambda v]
  //      [mass v] [radius v] [temp v]
  //      [diag yes|no] [diag/every N]

  if (narg < 14)
    error->all(FLERR,
      "fix drag: need nevery A_bg Z_bg plasma Te Ti Ni Vpar bfield Br Bt Bz");

  int iarg = 2;
  nevery       = input->inumeric(FLERR, arg[iarg++]);
  A_background = input->numeric (FLERR, arg[iarg++]);  // amu
  Z_background = input->inumeric(FLERR, arg[iarg++]);  // charge state

  if (strcmp(arg[iarg++], "plasma") != 0)
    error->all(FLERR, "fix drag: expected keyword 'plasma'");

  parse_src(arg[iarg++], srcTe,   "Te");
  parse_src(arg[iarg++], srcTi,   "Ti");
  parse_src(arg[iarg++], srcNi,   "Ni");
  parse_src(arg[iarg++], srcVpar, "Vpar");

  if (strcmp(arg[iarg++], "bfield") != 0)
    error->all(FLERR, "fix drag: expected keyword 'bfield'");

  parse_src(arg[iarg++], srcBr, "Br");
  parse_src(arg[iarg++], srcBt, "Bt");
  parse_src(arg[iarg++], srcBz, "Bz");

  use_grid_plasma = (srcTe.kind == COLL_SRC_VAR || srcTi.kind == COLL_SRC_VAR ||
                     srcNi.kind == COLL_SRC_VAR || srcVpar.kind == COLL_SRC_VAR);
  use_grid_bfield = (srcBr.kind == COLL_SRC_VAR || srcBt.kind == COLL_SRC_VAR ||
                     srcBz.kind == COLL_SRC_VAR);

  // Optional keyword arguments
  while (iarg < narg) {
    if (strcmp(arg[iarg], "gravity") == 0) {
      if (iarg + 3 >= narg)
        error->all(FLERR, "fix drag: 'gravity' requires 3 components gx gy gz");
      g_input_[0] = input->numeric(FLERR, arg[iarg+1]);
      g_input_[1] = input->numeric(FLERR, arg[iarg+2]);
      g_input_[2] = input->numeric(FLERR, arg[iarg+3]);
      use_gravity = true;
      iarg += 4;

    } else if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for 'model'");
      if      (strcmp(arg[iarg+1], "epstein") == 0) drag_model = DRAG_EPSTEIN;
      else if (strcmp(arg[iarg+1], "coulomb") == 0) drag_model = DRAG_COULOMB;
      else error->all(FLERR, "fix drag: model must be 'epstein' or 'coulomb'");
      iarg += 2;

    } else if (strcmp(arg[iarg], "coulomb/chi") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for coulomb/chi");
      chi_coulomb = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "coulomb/delta") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for coulomb/delta");
      delta_ite = input->numeric(FLERR, arg[iarg+1]);
      if (delta_ite <= 0.0) error->all(FLERR, "fix drag: coulomb/delta must be > 0");
      iarg += 2;

    } else if (strcmp(arg[iarg], "coulomb/lnlambda") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for coulomb/lnlambda");
      ln_lambda_coulomb = input->numeric(FLERR, arg[iarg+1]);
      if (ln_lambda_coulomb < 0.0)
        error->all(FLERR, "fix drag: coulomb/lnlambda must be >= 0");
      iarg += 2;

    } else if (strcmp(arg[iarg], "mass") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for mass");
      seed_mass = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "radius") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for radius");
      seed_radius = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for temp");
      seed_temp = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "cylindrical") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix drag: missing yes|no for 'cylindrical'");
      if      (strcmp(arg[iarg+1], "yes") == 0) use_cylindrical = true;
      else if (strcmp(arg[iarg+1], "no")  == 0) use_cylindrical = false;
      else error->all(FLERR, "fix drag: cylindrical must be 'yes' or 'no'");
      iarg += 2;

    } else if (strcmp(arg[iarg], "diag") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix drag: missing yes|no for 'diag'");
      if      (strcmp(arg[iarg+1], "yes") == 0) diag_flag = 1;
      else if (strcmp(arg[iarg+1], "no")  == 0) diag_flag = 0;
      else error->all(FLERR, "fix drag: diag must be 'yes' or 'no'");
      iarg += 2;

    } else if (strcmp(arg[iarg], "diag/every") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix drag: missing value for diag/every");
      diag_every = input->inumeric(FLERR, arg[iarg+1]);
      if (diag_every <= 0) error->all(FLERR, "fix drag: diag/every must be > 0");
      iarg += 2;

    } else {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix drag: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }
}

/* ---------------------------------------------------------------------- */

FixDrag::~FixDrag()
{
  if (copymode) return;
  memory->destroy(plasma_grid);
  memory->destroy(b_grid);
  if (srcTe.vname)   delete [] srcTe.vname;
  if (srcTi.vname)   delete [] srcTi.vname;
  if (srcNi.vname)   delete [] srcNi.vname;
  if (srcVpar.vname) delete [] srcVpar.vname;
  if (srcBr.vname)   delete [] srcBr.vname;
  if (srcBt.vname)   delete [] srcBt.vname;
  if (srcBz.vname)   delete [] srcBz.vname;
  if (srcTe.cid)     delete [] srcTe.cid;
  if (srcTi.cid)     delete [] srcTi.cid;
  if (srcNi.cid)     delete [] srcNi.cid;
  if (srcVpar.cid)   delete [] srcVpar.cid;
  if (srcBr.cid)     delete [] srcBr.cid;
  if (srcBt.cid)     delete [] srcBt.cid;
  if (srcBz.cid)     delete [] srcBz.cid;
}

/* ---------------------------------------------------------------------- */

int FixDrag::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;   // pre-Boris half-kick  (also refreshes sources)
  mask |= END_OF_STEP;     // post-Boris half-kick (reuses cached sources)
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDrag::init()
{
  if (use_gravity) {
    for (int k = 0; k < modify->nfix; ++k)
      if (strcmp(modify->fix[k]->style, "gravity") == 0)
        error->all(FLERR,
          "fix drag: do not combine 'fix gravity' with 'gravity' inside fix drag");
  }

  // Bind grid-variable sources to their variable IDs
  bind_var(srcTe,   "Te");
  bind_var(srcTi,   "Ti");
  bind_var(srcNi,   "Ni");
  bind_var(srcVpar, "Vpar");
  bind_var(srcBr,   "Br");
  bind_var(srcBt,   "Bt");
  bind_var(srcBz,   "Bz");

  // Allocate VAR scratch buffers
  if (use_grid_plasma && grid->maxlocal > 0) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "drag:plasma_grid");
  }
  if (use_grid_bfield && grid->maxlocal > 0) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "drag:b_grid");
  }

  // Bind per-grid compute sources and validate column ranges
  bind_compute(srcTe,   "Te");
  bind_compute(srcTi,   "Ti");
  bind_compute(srcNi,   "Ni");
  bind_compute(srcVpar, "Vpar");
  bind_compute(srcBr,   "Br");
  bind_compute(srcBt,   "Bt");
  bind_compute(srcBz,   "Bz");
}

/* ---------------------------------------------------------------------- */

void FixDrag::start_of_step()
{
  if (update->ntimestep % nevery) return;

  // 1. Fill VAR scratch buffers (grid-variable path only)
  compute_plasma_grid();
  compute_bfield_grid();

  // 2. Resolve compute source pointers ONCE for this timestep.
  //    Both half-kicks share these pointers — no re-resolution in end_of_step.
  refresh_sources();

  // 3. Pre-Boris half-kick (with optional diagnostics)
  const bool do_diag = (diag_flag && (update->ntimestep % diag_every) == 0);
  kick_half(0.5 * update->dt, do_diag);
}

/* ---------------------------------------------------------------------- */

void FixDrag::end_of_step()
{
  if (update->ntimestep % nevery) return;

  // Post-Boris half-kick.  Sources were cached by start_of_step this step;
  // arr_cache pointers remain valid (plasma fields unchanged within one step).
  kick_half(0.5 * update->dt, false);
}

/* ---------------------------------------------------------------------- */

double FixDrag::memory_usage()
{
  double bytes = 0.0;
  if (plasma_grid) bytes += (double)maxgrid_plasma * 4 * sizeof(double);
  if (b_grid)      bytes += (double)maxgrid_b       * 3 * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */
/*  Source resolution (once per timestep)                                  */
/* ---------------------------------------------------------------------- */

void FixDrag::refresh_sources()
{
  refresh_src(srcTe);   refresh_src(srcTi);
  refresh_src(srcNi);   refresh_src(srcVpar);
  refresh_src(srcBr);   refresh_src(srcBt);
  refresh_src(srcBz);
}

inline void FixDrag::refresh_src(CollGridSrc &S)
{
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;   // already resolved this step

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep)
    c->compute_per_grid();

  // Try tally-style mapped access first
  double **arr = nullptr;
  int    *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts  = update->ntimestep;
    return;
  }

  // Fallback: standard per-grid array (e.g. compute plasma/fields)
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  // Source is empty for this step
  S.arr_cache = nullptr;
  S.src_index = -1;
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */
/*  Inline cell-value reader — zero virtual calls in the hot loop          */
/* ---------------------------------------------------------------------- */
//
// KOKKOS_NOTE: mark KOKKOS_INLINE_FUNCTION.  Replace double** with
//   Kokkos::View<const double**> and pass views by value to the lambda.

inline double FixDrag::read_cell(const CollGridSrc &S, int icell,
                                  double **var_buf, int var_col) const
{
  if (S.kind == COLL_SRC_COMP)
    return (S.arr_cache && S.src_index >= 0) ? S.arr_cache[icell][S.src_index] : 0.0;
  if (S.kind == COLL_SRC_VAR)
    return var_buf[icell][var_col];
  return 0.0;
}

/* ---------------------------------------------------------------------- */
/*  VAR buffer fill                                                        */
/* ---------------------------------------------------------------------- */

void FixDrag::compute_plasma_grid()
{
  if (!use_grid_plasma || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "drag:plasma_grid");
  }
  memset(&plasma_grid[0][0], 0, sizeof(double) * grid->nlocal * 4);

  const int stride = 4;
  if (srcTe.kind   == COLL_SRC_VAR)
    input->variable->compute_grid(srcTe.varid,   &plasma_grid[0][0], stride, 0);
  if (srcTi.kind   == COLL_SRC_VAR)
    input->variable->compute_grid(srcTi.varid,   &plasma_grid[0][1], stride, 0);
  if (srcNi.kind   == COLL_SRC_VAR)
    input->variable->compute_grid(srcNi.varid,   &plasma_grid[0][2], stride, 0);
  if (srcVpar.kind == COLL_SRC_VAR)
    input->variable->compute_grid(srcVpar.varid, &plasma_grid[0][3], stride, 0);
}

void FixDrag::compute_bfield_grid()
{
  if (!use_grid_bfield || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_b) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "drag:b_grid");
  }
  memset(&b_grid[0][0], 0, sizeof(double) * grid->nlocal * 3);

  const int stride = 3;
  if (srcBr.kind == COLL_SRC_VAR)
    input->variable->compute_grid(srcBr.varid, &b_grid[0][0], stride, 0);
  if (srcBt.kind == COLL_SRC_VAR)
    input->variable->compute_grid(srcBt.varid, &b_grid[0][1], stride, 0);
  if (srcBz.kind == COLL_SRC_VAR)
    input->variable->compute_grid(srcBz.varid, &b_grid[0][2], stride, 0);
}

/* ---------------------------------------------------------------------- */
/*  Half-kick  (the hot loop)                                              */
/* ---------------------------------------------------------------------- */
//
// KOKKOS_NOTE: replace the for-loop with
//   Kokkos::parallel_for("fix_drag", particle->nlocal, KOKKOS_LAMBDA(int ip){ ... });
// Capture plasma_grid and b_grid as device Views, arr_cache pointers as
// Kokkos::View<double**> device arrays obtained from the compute objects.

void FixDrag::kick_half(double dt_half, bool do_diag)
{
  const int nlocal = particle->nlocal;

  Particle::OnePart * const parts = particle->particles;

  // Diagnostic accumulators — initialised regardless of nlocal so MPI
  // collectives in the do_diag block are always called by every rank,
  // even ranks that own zero particles.  (Early-returning when nlocal==0
  // would skip the MPI_Allreduce and deadlock all other ranks.)
  long long n_part   = 0, n_nu   = 0;
  long long n_ti_pos = 0, n_ni_pos = 0;
  double    nu_sum   = 0.0;
  double    nu_min   = std::numeric_limits<double>::infinity();
  double    nu_max   = 0.0;
  double    rd_min   = std::numeric_limits<double>::infinity();
  double    rd_max   = 0.0;
  double    ti_min   = std::numeric_limits<double>::infinity();
  double    ti_max   = -std::numeric_limits<double>::infinity();
  double    ni_min   = std::numeric_limits<double>::infinity();
  double    ni_max   = -std::numeric_limits<double>::infinity();

  // Pre-compute invariants used in every particle
  const double mi       = A_background * update->proton_mass;
  const double axisym   = domain->axisymmetric;
  const bool   coulomb  = (drag_model == DRAG_COULOMB);
  const double chi_over_delta = chi_coulomb / std::max(delta_ite, 1.0e-12);

  // KOKKOS_NOTE: begin Kokkos::parallel_for here (with TeamPolicy + reducers
  // for the diagnostic accumulators).
  for (int ip = 0; ip < nlocal; ++ip) {
    Particle::OnePart &p = parts[ip];
    const int icell = p.icell;

    // One-time seeding when fix evaporation is absent
    if (seed_mass   > 0.0 && p.mass   <= 0.0) p.mass   = seed_mass;
    if (seed_radius > 0.0 && p.radius <= 0.0) p.radius = seed_radius;
    if (seed_temp   > 0.0 && p.temp   <= 0.0) p.temp   = seed_temp;

    // --- Read plasma/bfield for this cell: direct array dereference, no
    //     virtual call (arr_cache was resolved once in refresh_sources()).
    const double Ti_eV = std::max(read_cell(srcTi,   icell, plasma_grid, 1), 0.0);
    const double Ni    = std::max(read_cell(srcNi,   icell, plasma_grid, 2), 0.0);
    const double Vpar  =          read_cell(srcVpar, icell, plasma_grid, 3);
    const double Br    =          read_cell(srcBr,   icell, b_grid, 0);
    const double Bt    =          read_cell(srcBt,   icell, b_grid, 1);
    const double Bz    =          read_cell(srcBz,   icell, b_grid, 2);
    const double rd    = p.radius;

    // --- Epstein drag frequency (zero if any input is non-physical)
    double nuE = 0.0;
    double upar[3] = {0.0, 0.0, 0.0};

    if (rd > 0.0 && Ni > 0.0 && Ti_eV > 0.0) {
      nuE = epstein_nu(Ni, Ti_eV, rd);

      // Build u_parallel vector from Vpar and B-hat direction.
      // B stored in cylindrical (Br, Bt, Bz); velocity axes are:
      //   Cartesian (2D/3D): v = (vx, vy, vz)
      //   Axisymmetric:      v = (vr, vz, vphi)
      const double Bn = std::sqrt(Br*Br + Bt*Bt + Bz*Bz);
      if (Bn > 1.0e-12) {
        const double br = Br / Bn, bt = Bt / Bn, bz = Bz / Bn;
        if (!axisym) {
          // 2D RZ SPARTA plane: x=R, y=Z, z=toroidal.
          // Br(radial)→vx, Bz(cylindrical-Z)→vy, Bt(toroidal)→vz.
          upar[0] = Vpar * br;   // SPARTA x = R       ↔ Br
          upar[1] = Vpar * bz;   // SPARTA y = Z       ↔ Bz (cylindrical vertical)
          upar[2] = Vpar * bt;   // SPARTA z = toroidal ↔ Bt
        } else {
          upar[0] = Vpar * br;   // vr
          upar[1] = Vpar * bz;   // vz
          upar[2] = Vpar * bt;   // vphi
        }
      }

      // --- Optional Coulomb correction: multiply nuE by xi(u)
      if (coulomb && nuE > 0.0) {
        const double vth_i = std::sqrt(8.0 * (Ti_eV * update->echarge) / (MY_PI * mi));
        if (vth_i > 0.0) {
          const double dv0 = p.v[0] - upar[0];
          const double dv1 = p.v[1] - upar[1];
          const double dv2 = p.v[2] - upar[2];
          const double u   = std::sqrt(dv0*dv0 + dv1*dv1 + dv2*dv2) / vth_i;
          nuE *= coulomb_multiplier(u);
        } else {
          nuE = 0.0;
        }
      }
    }

    // --- Gravity in local velocity basis
    double g0 = 0.0, g1 = 0.0, g2 = 0.0;
    if (use_gravity) {
      if (!axisym) {
        g0 = g_input_[0];
        g1 = g_input_[1];
        g2 = g_input_[2];
      } else {
        // Cartesian (gx, gy, gz) -> cylindrical (gr, gz, gphi) at particle phi
        const double cphi = std::cos(p.x[2]);
        const double sphi = std::sin(p.x[2]);
        g0 =  g_input_[0]*cphi + g_input_[1]*sphi;   // gr
        g1 =  g_input_[2];                            // gz
        g2 = -g_input_[0]*sphi + g_input_[1]*cphi;   // gphi
      }
    }

    // --- Exact half-step integrator for dv/dt = -nuE*(v - upar) + g
    //
    // For nuE > 0:  v(t+dt/2) = upar + g/nuE + (v - upar - g/nuE) * exp(-nuE*dt/2)
    // Taylor expansion for small s = nuE*dt/2 avoids catastrophic cancellation.
    //
    // For nuE == 0:  v(t+dt/2) = v + g * dt/2  (pure gravity)
    if (nuE > 0.0 && std::isfinite(nuE)) {
      const double s   = nuE * dt_half;
      const double ex  = (std::fabs(s) < 1.0e-8)
                         ? (1.0 - s + 0.5*s*s)
                         : std::exp(-s);
      const double inv = 1.0 / nuE;

      p.v[0] = upar[0] + (p.v[0] - upar[0] - g0*inv)*ex + g0*inv;
      p.v[1] = upar[1] + (p.v[1] - upar[1] - g1*inv)*ex + g1*inv;
      p.v[2] = upar[2] + (p.v[2] - upar[2] - g2*inv)*ex + g2*inv;
    } else if (use_gravity) {
      p.v[0] += g0 * dt_half;
      p.v[1] += g1 * dt_half;
      p.v[2] += g2 * dt_half;
    }

    // --- Cylindrical geometric terms (DUSTT-style axisymmetric)
    //
    // For a 2D RZ grid (x=R, y=Z) with toroidal velocity v_φ = p.v[2],
    // the cylindrical equations of motion require two extra terms:
    //
    //   dv_R/dt += v_φ² / R      (centrifugal: pushes droplet outward)
    //   dv_φ/dt += -v_φ * v_R / R  (angular-momentum conservation)
    //
    // Activated by keyword 'cylindrical yes'.  Applied as an explicit half-step
    // after the drag half-kick, using beginning-of-step velocities for both
    // components (explicit Euler for the geometric coupling).
    //
    // Note: do NOT enable together with SPARTA 'global axisym yes', which already
    // includes geometric terms in its own particle mover.
    if (use_cylindrical && !axisym) {
      const double R = p.x[0];
      if (R > 1.0e-10) {
        const double vR  = p.v[0];   // radial   velocity (post-drag, pre-geom)
        const double vph = p.v[2];   // toroidal velocity (post-drag, pre-geom)
        p.v[0] += vph * vph / R * dt_half;   // centrifugal
        p.v[2] -= vph * vR  / R * dt_half;   // angular-momentum
      }
    }

    // --- Diagnostic accumulation (branch-predicted out when do_diag==false)
    if (do_diag) {
      if (rd > 0.0 && std::isfinite(rd)) {
        ++n_part;
        rd_min = std::min(rd_min, rd);
        rd_max = std::max(rd_max, rd);
      }
      if (Ti_eV > 0.0 && std::isfinite(Ti_eV)) {
        ++n_ti_pos;
        ti_min = std::min(ti_min, Ti_eV);
        ti_max = std::max(ti_max, Ti_eV);
      }
      if (Ni > 0.0 && std::isfinite(Ni)) {
        ++n_ni_pos;
        ni_min = std::min(ni_min, Ni);
        ni_max = std::max(ni_max, Ni);
      }
      if (nuE > 0.0 && std::isfinite(nuE)) {
        ++n_nu;
        nu_sum += nuE;
        nu_min  = std::min(nu_min, nuE);
        nu_max  = std::max(nu_max, nuE);
      }
    }
  } // end particle loop — KOKKOS_NOTE: end Kokkos::parallel_for

}

/* ---------------------------------------------------------------------- */
/*  Physics kernels — pure functions (no hidden state read in hot loop)    */
/* ---------------------------------------------------------------------- */
//
// KOKKOS_NOTE: add KOKKOS_INLINE_FUNCTION prefix and pass all constants
// as arguments (not via `this`) to make them callable from device lambdas.

double FixDrag::epstein_nu(double Ni, double Ti_eV, double rd_m) const
{
  if (Ni <= 0.0 || Ti_eV <= 0.0 || rd_m <= 0.0 || rho_d <= 0.0) return 0.0;
  const double mi    = A_background * update->proton_mass;
  const double vth   = std::sqrt(8.0 * (Ti_eV * update->echarge) / (MY_PI * mi));
  const double rho_g = Ni * mi;
  return alpha_E * (rho_g * vth) / (rho_d * rd_m);
}

double FixDrag::coulomb_multiplier(double u) const
{
  // Coulomb-corrected Epstein multiplier xi(u) = xi_coll(u) + xi_orb(u)
  // where u = |Δv| / v_th,i.  See Diaw et al. for derivation.
  const double sqrt_pi       = std::sqrt(MY_PI);
  const double ueff          = std::max(u, 1.0e-8);
  const double chi_over_d    = chi_coulomb / std::max(delta_ite, 1.0e-12);
  const double e2            = std::exp(-ueff * ueff);
  const double erf_u         = std::erf(ueff);

  // Collisional term
  const double cp = 1.0 / (2.0 * ueff * ueff * ueff * sqrt_pi);
  const double ca = ueff * (2.0*ueff*ueff + 1.0 + 2.0*chi_over_d) * e2;
  const double cb = 0.5 * sqrt_pi *
                    (4.0 * std::pow(ueff, 4) - 1.0
                     - 2.0 * (1.0 - 2.0*ueff*ueff) * chi_over_d) * erf_u;
  const double xi_coll = cp * (ca + cb);

  // Orbital term
  const double Y      = erf_u - (2.0 * ueff / sqrt_pi) * e2;
  const double xi_orb = 2.0 * chi_over_d * chi_over_d * ln_lambda_coulomb * (Y / ueff);

  const double xi = xi_coll + xi_orb;
  if (!std::isfinite(xi)) return 0.0;
  return std::max(0.0, xi);
}

/* ---------------------------------------------------------------------- */
/*  Argument-parsing helpers                                               */
/* ---------------------------------------------------------------------- */

void FixDrag::parse_src(const char *tok, CollGridSrc &dst, const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg), "fix drag: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) == 0) {
    dst.kind = COLL_SRC_COMP;
    const char *name = tok + 2;
    const char *lb   = strchr(name, '[');
    const char *rb   = lb ? strrchr(name, ']') : nullptr;
    if (!lb || !rb || rb <= lb + 1) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix drag: %s source '%s' must use c_ID[col] syntax", label, tok);
      error->all(FLERR, msg);
    }
    const int idlen = (int)(lb - name);
    dst.cid = new char[idlen + 1];
    strncpy(dst.cid, name, idlen);
    dst.cid[idlen] = '\0';
    dst.col = atoi(lb + 1);   // 1-based
    if (dst.col <= 0) {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix drag: column index for %s must be >= 1", label);
      error->all(FLERR, msg);
    }
  } else {
    dst.kind  = COLL_SRC_VAR;
    const int len = (int)strlen(tok);
    dst.vname = new char[len + 1];
    strcpy(dst.vname, tok);
  }
}

/* ---------------------------------------------------------------------- */

void FixDrag::bind_var(CollGridSrc &S, const char *label)
{
  if (S.kind != COLL_SRC_VAR) return;
  S.varid = input->variable->find(S.vname);
  if (S.varid < 0 || !input->variable->grid_style(S.varid)) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "fix drag: %s source '%s' is not a grid-style variable", label, S.vname);
    error->all(FLERR, msg);
  }
}

/* ---------------------------------------------------------------------- */

void FixDrag::bind_compute(CollGridSrc &S, const char *label)
{
  if (S.kind != COLL_SRC_COMP) return;
  S.icompute = modify->find_compute(S.cid);
  if (S.icompute < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "fix drag: compute '%s' for %s not found", S.cid, label);
    error->all(FLERR, msg);
  }
  Compute *c = modify->compute[S.icompute];
  if (!c->per_grid_flag)
    error->all(FLERR, "fix drag: compute source must produce per-grid data");
  if (c->size_per_grid_cols == 0)
    error->all(FLERR, "fix drag: compute source has no per-grid columns");
  if (S.col < 1 || S.col > c->size_per_grid_cols) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "fix drag: column %d for compute '%s' (%s) out of range [1..%d]",
             S.col, S.cid, label, c->size_per_grid_cols);
    error->all(FLERR, msg);
  }
}
