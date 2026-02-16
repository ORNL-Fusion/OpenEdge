/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_viscous.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "domain.h"
#include "math.h"
#include "react_bird.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "random_knuth.h"
#include "math_const.h"
#include <filesystem>
#include "math_extra.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include "compute.h"
#include "variable.h"
#include "update.h"

namespace fs = std::filesystem;
using namespace SPARTA_NS;
#define INVOKED_PER_GRID 16

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
    if (strcmp(arg[iarg], "mass") == 0) {
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
  delete rng;
  rng = nullptr;

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
  const double vth  = std::sqrt(8.0 * (Ti_eV*update->echarge) / (M_PI*mi));
  const double rho_g= Ni * mi;
  return alpha_E * (rho_g * vth) / (rho_d * rd_m);
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

      // Gravity in local velocity basis:
      // - Cartesian (2D/3D): (gx,gy,gz) -> (v0,v1,v2)
      // - Axisymmetric:      (gx,gy,gz) -> (gr,gz,gphi)
      double g0 = 0.0, g1 = 0.0, g2 = 0.0;
      if (use_gravity) {
        if (!domain->axisymmetric) {
          g0 = g_input_[0];
          g1 = g_input_[1];
          g2 = g_input_[2];
        } else {
          const double c = std::cos(p.x[2]);
          const double s = std::sin(p.x[2]);
          g0 =  g_input_[0]*c + g_input_[1]*s;   // gr
          g1 =  g_input_[2];                     // gz
          g2 = -g_input_[0]*s + g_input_[1]*c;   // gphi
        }
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

  if (do_diag) {
    long long n_global = 0, n_nu_global = 0;
    long long n_ti_pos_global = 0, n_ni_pos_global = 0;
    double nu_sum_global = 0.0;
    double nu_min_global = nu_min_local;
    double nu_max_global = nu_max_local;
    double rd_min_global = rd_min_local;
    double rd_max_global = rd_max_local;
    double ti_min_global = ti_min_local;
    double ti_max_global = ti_max_local;
    double ni_min_global = ni_min_local;
    double ni_max_global = ni_max_local;

    MPI_Allreduce(&n_local, &n_global, 1, MPI_LONG_LONG, MPI_SUM, world);
    MPI_Allreduce(&n_nu_local, &n_nu_global, 1, MPI_LONG_LONG, MPI_SUM, world);
    MPI_Allreduce(&n_ti_pos_local, &n_ti_pos_global, 1, MPI_LONG_LONG, MPI_SUM, world);
    MPI_Allreduce(&n_ni_pos_local, &n_ni_pos_global, 1, MPI_LONG_LONG, MPI_SUM, world);
    MPI_Allreduce(&nu_sum_local, &nu_sum_global, 1, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&nu_min_global, &nu_min_global, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&nu_max_global, &nu_max_global, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&rd_min_global, &rd_min_global, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&rd_max_global, &rd_max_global, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&ti_min_global, &ti_min_global, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&ti_max_global, &ti_max_global, 1, MPI_DOUBLE, MPI_MAX, world);
    MPI_Allreduce(&ni_min_global, &ni_min_global, 1, MPI_DOUBLE, MPI_MIN, world);
    MPI_Allreduce(&ni_max_global, &ni_max_global, 1, MPI_DOUBLE, MPI_MAX, world);

    if (comm->me == 0) {
      if (n_nu_global == 0) {
        printf("[fix viscous] step=%lld N(radius>0)=%lld N(Ti>0)=%lld N(Ni>0)=%lld N(nuE>0)=0 rd[min,max]=[%g,%g] Ti[min,max]=[%g,%g] Ni[min,max]=[%g,%g]\n",
               (long long) update->ntimestep, n_global,
               n_ti_pos_global, n_ni_pos_global,
               std::isfinite(rd_min_global) ? rd_min_global : 0.0, rd_max_global,
               std::isfinite(ti_min_global) ? ti_min_global : 0.0,
               std::isfinite(ti_max_global) ? ti_max_global : 0.0,
               std::isfinite(ni_min_global) ? ni_min_global : 0.0,
               std::isfinite(ni_max_global) ? ni_max_global : 0.0);
      } else {
        const double nu_avg = nu_sum_global / static_cast<double>(n_nu_global);
        printf("[fix viscous] step=%lld N(radius>0)=%lld N(Ti>0)=%lld N(Ni>0)=%lld N(nuE>0)=%lld nuE[min,avg,max]=[%g,%g,%g] rd[min,max]=[%g,%g] Ti[min,max]=[%g,%g] Ni[min,max]=[%g,%g]\n",
               (long long) update->ntimestep, n_global, n_nu_global,
               n_ti_pos_global, n_ni_pos_global,
               std::isfinite(nu_min_global) ? nu_min_global : 0.0, nu_avg, nu_max_global,
               std::isfinite(rd_min_global) ? rd_min_global : 0.0, rd_max_global,
               std::isfinite(ti_min_global) ? ti_min_global : 0.0,
               std::isfinite(ti_max_global) ? ti_max_global : 0.0,
               std::isfinite(ni_min_global) ? ni_min_global : 0.0,
               std::isfinite(ni_max_global) ? ni_max_global : 0.0);
      }
      fflush(stdout);
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

  // B components are in cylindrical basis: (Br, Bt, Bz) ~ (e_r, e_phi, e_z)
  const double Br = read_src(srcBr, -1, 0);
  const double Bt = read_src(srcBt, -1, 1);
  const double Bz = read_src(srcBz, -1, 2);
  const double Bn = std::sqrt(Br*Br + Bt*Bt + Bz*Bz);

  // const double rd = p.radius;
  const double rd = p.radius;  // in meters
  if (rd <= 0.0 || Ni <= 0.0 || Ti_eV <= 0.0) {
    nuE = 0.0;
    upar_cyl[0] = upar_cyl[1] = upar_cyl[2] = 0.0;
    return;
  }

  // Epstein frequency (SI)
  nuE = epstein_nu(Ni, Ti_eV, rd);

  // Build u_parallel in local velocity basis:
  // - Cartesian (2D/3D): [vx,vy,vz]
  // - Axisymmetric:      [vr,vz,vphi]
  if (Bn > 1e-12) {
    const double br = Br / Bn;
    const double bt = Bt / Bn;
    const double bz = Bz / Bn;
    if (!domain->axisymmetric) {
      upar_cyl[0] = Vpar * br;
      upar_cyl[1] = Vpar * bt;
      upar_cyl[2] = Vpar * bz;
    } else {
      upar_cyl[0] = Vpar * br;
      upar_cyl[1] = Vpar * bz;
      upar_cyl[2] = Vpar * bt;
    }
  } else {
    // No reliable B direction: relax toward zero flow (or choose a default axis if desired)
    upar_cyl[0] = 0.0;
    upar_cyl[1] = 0.0;
    upar_cyl[2] = 0.0;
  }

  // print upar_cyl
}
