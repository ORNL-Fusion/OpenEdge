/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

#include "fix_viscous.h"
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

FixViscous::FixViscous(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{

 // Syntax:
  // fix ID viscous <nevery> <A_bg> <Z_bg>
  //      plasma <TeSrc> <TiSrc> <NiSrc> <VparSrc>
  //      bfield <BrSrc> <BtSrc> <BzSrc>

  if (narg < 14)
    error->all(FLERR,"Illegal fix viscous (need: nevery A_bg Z_bg plasma Te Ti Ni Vpar bfield Br Bt Bz)");

  int iarg = 2;
  nevery        = input->inumeric(FLERR,arg[iarg++]);
  A_background  = input->numeric(FLERR,arg[iarg++]);  // atomic mass (amu)
  Z_background  = input->inumeric(FLERR,arg[iarg++]); // charge state (integer)

  // expect "plasma"
  if (strcmp(arg[iarg++],"plasma") != 0)
    error->all(FLERR,"fix viscous: missing 'plasma' keyword");

  auto dupstr = [](const char* s)->char* { auto *p=new char[strlen(s)+1]; strcpy(p,s); return p; };

  // tiny parser for a token: either VAR name or c_ID[idx]
  auto parse_src = [&](const char *tok, CollGridSrc &dst, const char *label){
    if (!tok || !*tok) {
      char msg[128]; snprintf(msg,sizeof(msg),"fix viscous: empty token for %s",label);
      error->all(FLERR,msg);
    }
    if (strncmp(tok,"c_",2)==0) {
      dst.kind = COLL_SRC_COMP;
      const char *name = tok + 2;
      const char *lb   = strchr(name,'[');
      const char *rb   = (lb ? strrchr(name,']') : nullptr);
      if (!lb || !rb || rb <= lb+1)
        error->all(FLERR,"fix viscous: use c_ID[idx] for compute sources");
      const int idlen = lb - name;
      dst.cid = new char[idlen+1];
      strncpy(dst.cid, name, idlen);
      dst.cid[idlen] = '\0';
      dst.col = atoi(lb+1);         // 1-based
      if (dst.col <= 0)
        error->all(FLERR,"fix viscous: compute column must be >=1");
    } else {
      dst.kind  = COLL_SRC_VAR;
      dst.vname = dupstr(tok);
    }
  };

  // plasma sources
  parse_src(arg[iarg++], srcTe,   "Te");
  parse_src(arg[iarg++], srcTi,   "Ti");
  parse_src(arg[iarg++], srcNi,   "Ni");
  parse_src(arg[iarg++], srcVpar, "Vpar");

  // expect "bfield"
  if (strcmp(arg[iarg++],"bfield") != 0)
    error->all(FLERR,"fix viscous: missing 'bfield' keyword");

  // B sources
  parse_src(arg[iarg++], srcBr, "Br");
  parse_src(arg[iarg++], srcBt, "Bt");
  parse_src(arg[iarg++], srcBz, "Bz");

  use_grid_plasma = (srcTe.kind==COLL_SRC_VAR || srcTi.kind==COLL_SRC_VAR ||
                     srcNi.kind==COLL_SRC_VAR || srcVpar.kind==COLL_SRC_VAR);
  use_grid_bfield = (srcBr.kind==COLL_SRC_VAR || srcBt.kind==COLL_SRC_VAR || srcBz.kind==COLL_SRC_VAR);

  if (iarg < narg && strcmp(arg[iarg], "gravity") == 0) {
  ++iarg;
  if (iarg + 2 >= narg)
    error->all(FLERR, "fix viscous: 'gravity' needs 3 components");

  // numeric parse (same helper style you use elsewhere)
  g_input_[0] = input->numeric(FLERR, arg[iarg++]);
  g_input_[1] = input->numeric(FLERR, arg[iarg++]);
  g_input_[2] = input->numeric(FLERR, arg[iarg++]);
  use_gravity = true;
}

  // Optional one-time state seeding for runs without fix evaporation.
  while (iarg < narg) {
    if (strcmp(arg[iarg], "model") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for model");
      if (strcmp(arg[iarg+1], "epstein") == 0) drag_model = DRAG_EPSTEIN;
      else if (strcmp(arg[iarg+1], "coulomb") == 0) drag_model = DRAG_COULOMB;
      else error->all(FLERR, "fix viscous: model must be epstein or coulomb");
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/chi") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for coulomb/chi");
      chi_coulomb = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/delta") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for coulomb/delta");
      delta_ite = input->numeric(FLERR, arg[iarg+1]);
      if (delta_ite <= 0.0) error->all(FLERR, "fix viscous: coulomb/delta must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "coulomb/lnlambda") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for coulomb/lnlambda");
      ln_lambda_coulomb = input->numeric(FLERR, arg[iarg+1]);
      if (ln_lambda_coulomb < 0.0) error->all(FLERR, "fix viscous: coulomb/lnlambda must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "mass") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for mass");
      seed_mass = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "radius") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for radius");
      seed_radius = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for temp");
      seed_temp = input->numeric(FLERR, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "diag") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for diag yes|no");
      if (strcmp(arg[iarg+1], "yes") == 0) diag_flag = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) diag_flag = 0;
      else error->all(FLERR, "fix viscous: diag must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "diag/every") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix viscous: missing value for diag/every");
      diag_every = input->inumeric(FLERR, arg[iarg+1]);
      if (diag_every <= 0) error->all(FLERR, "fix viscous: diag/every must be > 0");
      iarg += 2;
    } else {
      char msg[200];
      snprintf(msg,sizeof(msg),"fix viscous: unknown optional keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  // defaults
  maxgrid_plasma = maxgrid_b = 0;
  plasma_grid = b_grid = nullptr;
}
    
/* ---------------------------------------------------------------------- */

FixViscous::~FixViscous()
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

int FixViscous::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;   // pre-Boris half-kick
  mask |= END_OF_STEP;     // post-Boris half-kick
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixViscous::init()
{
  
  if (use_gravity) {
  for (int k = 0; k < modify->nfix; ++k) {
    if (strcmp(modify->fix[k]->style, "gravity") == 0) {
      error->all(FLERR,
        "Do not use 'fix gravity' together with 'gravity' inside fix viscous");
    }
  }
}

    // --- resolve VAR sources (grid variables) ---
  auto bind_var = [&](CollGridSrc &S, const char *label){
    if (S.kind != COLL_SRC_VAR) return;
    if (!S.vname || !*S.vname) {
      char msg[160];
      snprintf(msg,sizeof(msg),
               "fix viscous: empty name for %s grid variable", label);
      error->all(FLERR,msg);
    }
    S.varid = input->variable->find(S.vname);
    if (S.varid < 0 || !input->variable->grid_style(S.varid)) {
      char msg[200];
      snprintf(msg,sizeof(msg),
        "fix viscous: %s ('%s') must be a grid-style variable", label, S.vname);
      error->all(FLERR,msg);
    }
  };
  bind_var(srcTe,"Te");
  bind_var(srcTi,"Ti");
  bind_var(srcNi,"Ni");
  bind_var(srcVpar,"Vpar");
  bind_var(srcBr,"Br");
  bind_var(srcBt,"Bt");
  bind_var(srcBz,"Bz");

  // --- allocate VAR buffers if needed ---
  if (use_grid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "viscous");
    if (grid->nlocal)
      memset(&plasma_grid[0][0], 0, sizeof(double)*grid->nlocal*4);
  }
  if (use_grid_bfield) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "viscous:b_grid");
    if (grid->nlocal)
      memset(&b_grid[0][0], 0, sizeof(double)*grid->nlocal*3);
  }

  // --- resolve COMPUTE sources (per-grid arrays) ---
  auto bind_compute = [&](CollGridSrc &S, const char *label){
    if (S.kind != COLL_SRC_COMP) return;
    if (!S.cid || !*S.cid) {
      char msg[160];
      snprintf(msg,sizeof(msg),
               "fix viscous: empty compute ID for %s", label);
      error->all(FLERR,msg);
    }
    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg,sizeof(msg),
               "fix viscous: compute '%s' for %s not found", S.cid, label);
      error->all(FLERR,msg);
    }
    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR,"fix viscous: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR,"fix viscous: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[200];
      snprintf(msg,sizeof(msg),
               "fix viscous: column %d for compute '%s' (%s) out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR,msg);
    }
  };
  bind_compute(srcTe,"Te");
  bind_compute(srcTi,"Ti");
  bind_compute(srcNi,"Ni");
  bind_compute(srcVpar,"Vpar");
  bind_compute(srcBr,"Br");
  bind_compute(srcBt,"Bt");
  bind_compute(srcBz,"Bz");
}



/* ---------------------------------------------------------------------- */

void FixViscous::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  const double dt_half = 0.5 * update->dt;
  if (!particle->sorted) particle->sort();
  kick_half(dt_half, 0);
}


/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixViscous::memory_usage()
{
  double bytes = 0.0;
  if (plasma_grid) bytes += (size_t)maxgrid_plasma * 4 * sizeof(double);
  if (b_grid)      bytes += (size_t)maxgrid_b       * 3 * sizeof(double);
  return bytes;
}

// Compute grid variables into our local buffers (only those requested)
void FixViscous::compute_plasma_grid() {
  if (!use_grid_plasma || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "viscous");
  }
  // zero only the rows we will actually write
  memset(&plasma_grid[0][0], 0, sizeof(double)*grid->nlocal*4);

  const int stride = 4;
  if (srcTe.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcTe.varid,   &plasma_grid[0][0], stride, 0);
  if (srcTi.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcTi.varid,   &plasma_grid[0][1], stride, 0);
  if (srcNi.kind   == COLL_SRC_VAR) input->variable->compute_grid(srcNi.varid,   &plasma_grid[0][2], stride, 0);
  if (srcVpar.kind == COLL_SRC_VAR) input->variable->compute_grid(srcVpar.varid, &plasma_grid[0][3], stride, 0);
}

void FixViscous::compute_bfield_grid() {
  if (!use_grid_bfield || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_b) {
    maxgrid_b = grid->maxlocal;
    memory->destroy(b_grid);
    memory->create(b_grid, maxgrid_b, 3, "viscous:b_grid");
  }
  memset(&b_grid[0][0], 0, sizeof(double)*grid->nlocal*3);

  const int stride = 3;
  if (srcBr.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBr.varid, &b_grid[0][0], stride, 0);
  if (srcBt.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBt.varid, &b_grid[0][1], stride, 0);
  if (srcBz.kind == COLL_SRC_VAR) input->variable->compute_grid(srcBz.varid, &b_grid[0][2], stride, 0);
}

// Ensure compute_per_grid ran and read 1-based column c_ID[col]
double FixViscous::fetch_compute_cell_value(const CollGridSrc& S, int icell)
{
  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  // First try tally-style mapped access (works for tally-backed computes).
  double **arr = nullptr;
  int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    const int src = cols ? cols[0] : (S.col - 1);
    return arr[icell][src];
  }

  // Fallback for regular per-grid computes (e.g. compute plasma/fields).
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    return c->array_grid[icell][S.col - 1];
  }
  if (c->size_per_grid_cols == 0 && c->vector_grid && S.col == 1) {
    return c->vector_grid[icell];
  }

  return 0.0;
}

inline void FixViscous::refresh_compute_src(CollGridSrc &S) {
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr; int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts  = update->ntimestep;
    return;
  }

  // Fallback cache for standard per-grid arrays.
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  S.arr_cache  = nullptr;
  S.src_index  = -1;
  S.cache_ts   = update->ntimestep;
}


double FixViscous::epstein_nu(double Ni, double Ti_eV, double rd_m) const {
  if (Ni<=0.0 || Ti_eV<=0.0 || rd_m<=0.0 || rho_d<=0.0) return 0.0;

  const double mi   = A_background * update->proton_mass;
  const double vth  = std::sqrt(8.0 * (Ti_eV*update->echarge) / (MY_PI*mi));
  const double rho_g= Ni * mi;
  return alpha_E * (rho_g * vth) / (rho_d * rd_m);
}

double FixViscous::coulomb_drag_multiplier(double u) const
{
  // Implements a Coulomb-corrected Epstein scaling using
  // collisional + orbital terms, normalized by baseline Epstein drag.
  const double sqrt_pi = std::sqrt(MY_PI);
  const double ueff = std::max(u, 1.0e-8);
  const double chi_over_delta = chi_coulomb / std::max(delta_ite, 1.0e-12);
  const double e = std::exp(-ueff * ueff);
  const double erf_u = std::erf(ueff);

  const double coll_pref = 1.0 / (2.0 * ueff * ueff * ueff * sqrt_pi);
  const double coll_a = ueff * (2.0 * ueff * ueff + 1.0 + 2.0 * chi_over_delta) * e;
  const double coll_b =
      0.5 * sqrt_pi *
      (4.0 * std::pow(ueff, 4) - 1.0 - 2.0 * (1.0 - 2.0 * ueff * ueff) * chi_over_delta) *
      erf_u;
  const double xi_coll = coll_pref * (coll_a + coll_b);

  const double Y =
      erf_u - (2.0 * ueff / sqrt_pi) * e;
  const double xi_orb =
      2.0 * std::pow(chi_over_delta, 2.0) * ln_lambda_coulomb * (Y / ueff);

  const double xi = xi_coll + xi_orb;
  if (!std::isfinite(xi)) return 0.0;
  return std::max(0.0, xi);
}

void FixViscous::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // refresh grid VAR buffers so pre-kick uses fresh fields
  if (use_grid_plasma) compute_plasma_grid();
  if (use_grid_bfield) compute_bfield_grid();

  // refresh COMP sources once per step
  refresh_compute_src(srcTe);   refresh_compute_src(srcTi);
  refresh_compute_src(srcNi);   refresh_compute_src(srcVpar);
  refresh_compute_src(srcBr);   refresh_compute_src(srcBt);
  refresh_compute_src(srcBz);

  const double dt_half = 0.5 * update->dt;
  if (!particle->sorted) particle->sort();
  kick_half(dt_half, 1);
}


void FixViscous::kick_half(double dt_half, int diag_phase)
{
  if (grid->nlocal == 0) return;
  auto * const parts = particle->particles;
  int * const next   = particle->next;
  auto * const cinfo = grid->cinfo;

  const int nglocal = grid->nlocal;
  const bool do_diag = (diag_flag && diag_phase == 1 && (update->ntimestep % diag_every) == 0);
  long long n_local = 0, n_nu_local = 0;
  double nu_sum_local = 0.0;
  double nu_min_local = std::numeric_limits<double>::infinity();
  double nu_max_local = 0.0;
  double rd_min_local = std::numeric_limits<double>::infinity();
  double rd_max_local = 0.0;
  long long n_ti_pos_local = 0, n_ni_pos_local = 0;
  double ti_min_local = std::numeric_limits<double>::infinity();
  double ti_max_local = -std::numeric_limits<double>::infinity();
  double ni_min_local = std::numeric_limits<double>::infinity();
  double ni_max_local = -std::numeric_limits<double>::infinity();

  for (int icell = 0; icell < nglocal; ++icell) {
    if (cinfo[icell].count == 0) continue;

    int ip = cinfo[icell].first;
    while (ip >= 0) {
      Particle::OnePart &p = parts[ip];

      // Optional one-time initialization when evaporation fix is not used.
      if (seed_mass > 0.0 && p.mass <= 0.0) p.mass = seed_mass;
      if (seed_radius > 0.0 && p.radius <= 0.0) p.radius = seed_radius;
      if (seed_temp > 0.0 && p.temp <= 0.0) p.temp = seed_temp;

      double nuE; double upar[3] = {0,0,0};
      epstein_params(icell, p, nuE, upar);
      double *v = p.v;
      if (do_diag) {
        const double Ti_eV = std::max((srcTi.kind == COLL_SRC_COMP)
                                      ? fetch_compute_cell_value(srcTi, icell)
                                      : plasma_grid[icell][1], 0.0);
        const double Ni    = std::max((srcNi.kind == COLL_SRC_COMP)
                                      ? fetch_compute_cell_value(srcNi, icell)
                                      : plasma_grid[icell][2], 0.0);

        const double rd = p.radius;
        if (rd > 0.0 && std::isfinite(rd)) {
          n_local++;
          rd_min_local = std::min(rd_min_local, rd);
          rd_max_local = std::max(rd_max_local, rd);
        }
        if (Ti_eV > 0.0 && std::isfinite(Ti_eV)) {
          n_ti_pos_local++;
          ti_min_local = std::min(ti_min_local, Ti_eV);
          ti_max_local = std::max(ti_max_local, Ti_eV);
        }
        if (Ni > 0.0 && std::isfinite(Ni)) {
          n_ni_pos_local++;
          ni_min_local = std::min(ni_min_local, Ni);
          ni_max_local = std::max(ni_max_local, Ni);
        }
        if (nuE > 0.0 && std::isfinite(nuE)) {
          n_nu_local++;
          nu_sum_local += nuE;
          nu_min_local = std::min(nu_min_local, nuE);
          nu_max_local = std::max(nu_max_local, nuE);
        }
      }

      // Gravity is interpreted in SPARTA slot order in every mode.
      // (See fix_gravity.cpp for the convention rationale.) The
      // pre-existing axisymmetric branch read p.x[2] as the particle's
      // azimuth, but in SPARTA's native axi mode particles are remapped
      // to the symmetry plane (x[2]=0), so that math was incorrect.
      double g0 = 0.0, g1 = 0.0, g2 = 0.0;
      if (use_gravity) {
        g0 = g_input_[0];
        g1 = g_input_[1];
        g2 = g_input_[2];
      }

      if (nuE > 0.0 && std::isfinite(nuE)) {
        // Exact linear drag+forcing half-step:
        // dv/dt = -nuE (v - upar) + g_local
        const double s   = nuE * dt_half;
        const double ex  = (std::fabs(s) < 1e-8) ? (1.0 - s + 0.5*s*s) : std::exp(-s);
        const double inv = 1.0 / nuE;

        v[0] = upar[0] + (v[0] - upar[0] - g0*inv)*ex + g0*inv;
        v[1] = upar[1] + (v[1] - upar[1] - g1*inv)*ex + g1*inv;
        v[2] = upar[2] + (v[2] - upar[2] - g2*inv)*ex + g2*inv;
      } else if (use_gravity) {
        // Pure gravity when drag is inactive
        v[0] += g0 * dt_half;
        v[1] += g1 * dt_half;
        v[2] += g2 * dt_half;
      }

      ip = next[ip];
    }
  }

}
inline void FixViscous::epstein_params(int icell, const Particle::OnePart &p,
                                       double &nuE, double upar_cyl[3])
{
  auto read_src = [&](const CollGridSrc& S, int col_plasma, int col_b)->double {
    if (S.kind == COLL_SRC_COMP)
      return fetch_compute_cell_value(S, icell);
    if (S.kind == COLL_SRC_VAR)
      return (col_plasma >= 0) ? plasma_grid[icell][col_plasma] : b_grid[icell][col_b];
    return 0.0;
  };

  const double Ti_eV = std::max(read_src(srcTi, 1, -1), 0.0);
  const double Ni    = std::max(read_src(srcNi, 2, -1), 0.0);
  const double Vpar  =          read_src(srcVpar, 3, -1);

  // B components in SPARTA coordinate order (bx, by, bz from compute):
  //   2D: B[0]=B_R, B[1]=B_Z, B[2]=B_toroidal  (matching v[0..2])
  //   3D: B[0]=B_x, B[1]=B_y, B[2]=B_z          (Cartesian)
  const double B0 = read_src(srcBr, -1, 0);
  const double B1 = read_src(srcBt, -1, 1);
  const double B2 = read_src(srcBz, -1, 2);
  const double Bn = std::sqrt(B0*B0 + B1*B1 + B2*B2);

  const double rd = p.radius;  // in meters
  if (rd <= 0.0 || Ni <= 0.0 || Ti_eV <= 0.0) {
    nuE = 0.0;
    upar_cyl[0] = upar_cyl[1] = upar_cyl[2] = 0.0;
    return;
  }

  // Build u_parallel in SPARTA velocity basis (matches v[0..2] slots directly)
  if (Bn > 1e-12) {
    const double invBn = 1.0 / Bn;
    upar_cyl[0] = Vpar * B0 * invBn;
    upar_cyl[1] = Vpar * B1 * invBn;
    upar_cyl[2] = Vpar * B2 * invBn;
  } else {
    // No reliable B direction: relax toward zero flow
    upar_cyl[0] = 0.0;
    upar_cyl[1] = 0.0;
    upar_cyl[2] = 0.0;
  }

  // Baseline Epstein frequency (SI)
  nuE = epstein_nu(Ni, Ti_eV, rd);

  // Optional Coulomb-corrected model.
  if (drag_model == DRAG_COULOMB && nuE > 0.0) {
    const double mi = A_background * update->proton_mass;
    const double vth_i = std::sqrt(8.0 * (Ti_eV * update->echarge) / (MY_PI * mi));
    if (vth_i > 0.0) {
      const double dv0 = p.v[0] - upar_cyl[0];
      const double dv1 = p.v[1] - upar_cyl[1];
      const double dv2 = p.v[2] - upar_cyl[2];
      const double u = std::sqrt(dv0*dv0 + dv1*dv1 + dv2*dv2) / vth_i;
      nuE *= coulomb_drag_multiplier(u);
    } else {
      nuE = 0.0;
    }
  }

  // print upar_cyl
}
