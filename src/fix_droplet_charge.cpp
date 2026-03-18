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
#include "fix_droplet_charge.h"

#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "input.h"
#include "modify.h"
#include "compute.h"
#include "variable.h"
#include "OPENEDGE/compute_plasma_fields.h"

#include "math_const.h"

#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>

using namespace SPARTA_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

FixDropletCharge::FixDropletCharge(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // Syntax:
  // fix ID droplet/charge nevery plasma <TeSrc> <TiSrc> <NeSrc> <NiSrc>
  // fix ID droplet/charge nevery plasma temp_e <TeSrc> temp_i <TiSrc> dens_e <NeSrc> dens_i <NiSrc>
  //      [radius <m>] [mass <kg>] [ion_mass_amu <amu>]
  //      [thermionic yes|no] [richardson_A <A/m^2/K^2>] [work_function_eV <eV>]
  //      [temp <K>] [diag yes|no] [diag/every N]

  if (narg < 8)
    error->all(FLERR, "Illegal fix droplet/charge (need: nevery plasma Te Ti Ne Ni)");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg], "plasma") != 0 && strcmp(arg[iarg], "plasma/fields") != 0)
    error->all(FLERR, "fix droplet/charge: missing 'plasma' or 'plasma/fields' keyword");
  iarg++;

  auto dupstr = [](const char *s) -> char * {
    auto *p = new char[strlen(s) + 1];
    strcpy(p, s);
    return p;
  };

  auto parse_src = [&](const char *tok, CollGridSrc &dst, const char *label) {
    if (!tok || !*tok) {
      char msg[128];
      snprintf(msg, sizeof(msg), "fix droplet/charge: empty token for %s", label);
      error->all(FLERR, msg);
    }

    if (strncmp(tok, "c_", 2) == 0) {
      dst.kind = COLL_SRC_COMP;
      const char *name = tok + 2;
      const char *lb = strchr(name, '[');
      const char *rb = (lb ? strrchr(name, ']') : nullptr);
      if (!lb || !rb || rb <= lb + 1)
        error->all(FLERR, "fix droplet/charge: use c_ID[idx] for compute sources");

      const int idlen = lb - name;
      dst.cid = new char[idlen + 1];
      strncpy(dst.cid, name, idlen);
      dst.cid[idlen] = '\0';
      dst.col = atoi(lb + 1);
      if (dst.col <= 0)
        error->all(FLERR, "fix droplet/charge: compute column must be >= 1");
    } else {
      dst.kind = COLL_SRC_VAR;
      dst.vname = dupstr(tok);
    }
  };

  // Two accepted plasma syntaxes:
  // 1) positional: plasma <TeSrc> <TiSrc> <NeSrc> <NiSrc>
  // 2) explicit:   plasma temp_e <TeSrc> temp_i <TiSrc> dens_e <NeSrc> dens_i <NiSrc>
  if (iarg + 7 < narg &&
      (strcmp(arg[iarg], "temp_e") == 0 || strcmp(arg[iarg], "temp_i") == 0 ||
       strcmp(arg[iarg], "dens_e") == 0 || strcmp(arg[iarg], "dens_i") == 0)) {
    int got_te = 0, got_ti = 0, got_ne = 0, got_ni = 0;
    for (int nset = 0; nset < 4; ++nset) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix droplet/charge: incomplete plasma key/value pair");
      if (strcmp(arg[iarg], "temp_e") == 0) {
        parse_src(arg[iarg + 1], srcTe, "Te");
        got_te = 1;
      } else if (strcmp(arg[iarg], "temp_i") == 0) {
        parse_src(arg[iarg + 1], srcTi, "Ti");
        got_ti = 1;
      } else if (strcmp(arg[iarg], "dens_e") == 0) {
        parse_src(arg[iarg + 1], srcNe, "Ne");
        got_ne = 1;
      } else if (strcmp(arg[iarg], "dens_i") == 0) {
        parse_src(arg[iarg + 1], srcNi, "Ni");
        got_ni = 1;
      } else {
        break;
      }
      iarg += 2;
    }
    if (!(got_te && got_ti && got_ne && got_ni))
      error->all(FLERR, "fix droplet/charge: plasma explicit form requires temp_e,temp_i,dens_e,dens_i");
  } else {
    if (iarg + 3 >= narg)
      error->all(FLERR, "Illegal fix droplet/charge (need plasma Te Ti Ne Ni)");
    parse_src(arg[iarg++], srcTe, "Te");
    parse_src(arg[iarg++], srcTi, "Ti");
    parse_src(arg[iarg++], srcNe, "Ne");
    parse_src(arg[iarg++], srcNi, "Ni");
  }

  use_grid_plasma = (srcTe.kind == COLL_SRC_VAR || srcTi.kind == COLL_SRC_VAR ||
                     srcNe.kind == COLL_SRC_VAR || srcNi.kind == COLL_SRC_VAR);
  use_point_plasma = 0;
  point_plasma_compute = nullptr;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "radius") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for radius");
      seed_radius = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "mass") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for mass");
      seed_mass = input->numeric(FLERR, arg[iarg + 1]);
      if (seed_mass <= 0.0) error->all(FLERR, "fix droplet/charge: mass must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "ion_mass_amu") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for ion_mass_amu");
      ion_mass_amu = input->numeric(FLERR, arg[iarg + 1]);
      if (ion_mass_amu <= 0.0) error->all(FLERR, "fix droplet/charge: ion_mass_amu must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "thermionic") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for thermionic yes|no");
      if (strcmp(arg[iarg + 1], "yes") == 0) thermionic_on = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) thermionic_on = 0;
      else error->all(FLERR, "fix droplet/charge: thermionic must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "richardson_A") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for richardson_A");
      richardson_A = input->numeric(FLERR, arg[iarg + 1]);
      if (richardson_A < 0.0) error->all(FLERR, "fix droplet/charge: richardson_A must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "work_function_eV") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for work_function_eV");
      work_function_eV = input->numeric(FLERR, arg[iarg + 1]);
      if (work_function_eV <= 0.0) error->all(FLERR, "fix droplet/charge: work_function_eV must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for temp");
      seed_temp = input->numeric(FLERR, arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "diag") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for diag yes|no");
      if (strcmp(arg[iarg + 1], "yes") == 0) diag_flag = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) diag_flag = 0;
      else error->all(FLERR, "fix droplet/charge: diag must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "diag/every") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix droplet/charge: missing value for diag/every");
      diag_every = input->inumeric(FLERR, arg[iarg + 1]);
      if (diag_every <= 0) error->all(FLERR, "fix droplet/charge: diag/every must be > 0");
      iarg += 2;
    } else {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix droplet/charge: unknown optional keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  maxgrid_plasma = 0;
  plasma_grid = nullptr;
}

/* ---------------------------------------------------------------------- */

FixDropletCharge::~FixDropletCharge()
{
  if (copymode) return;
  if (srcTe.vname) delete [] srcTe.vname;
  if (srcTi.vname) delete [] srcTi.vname;
  if (srcNe.vname) delete [] srcNe.vname;
  if (srcNi.vname) delete [] srcNi.vname;
  if (srcTe.cid) delete [] srcTe.cid;
  if (srcTi.cid) delete [] srcTi.cid;
  if (srcNe.cid) delete [] srcNe.cid;
  if (srcNi.cid) delete [] srcNi.cid;
  memory->destroy(plasma_grid);
}

/* ---------------------------------------------------------------------- */

int FixDropletCharge::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::init()
{
  auto bind_var = [&](CollGridSrc &S, const char *label) {
    if (S.kind != COLL_SRC_VAR) return;
    if (!S.vname || !*S.vname) {
      char msg[160];
      snprintf(msg, sizeof(msg), "fix droplet/charge: empty name for %s grid variable", label);
      error->all(FLERR, msg);
    }
    S.varid = input->variable->find(S.vname);
    if (S.varid < 0 || !input->variable->grid_style(S.varid)) {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix droplet/charge: %s ('%s') must be a grid-style variable", label, S.vname);
      error->all(FLERR, msg);
    }
  };

  bind_var(srcTe, "Te");
  bind_var(srcTi, "Ti");
  bind_var(srcNe, "Ne");
  bind_var(srcNi, "Ni");

  if (use_grid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "droplet/charge:plasma");
    if (grid->nlocal) memset(&plasma_grid[0][0], 0, sizeof(double) * grid->nlocal * 4);
  }

  auto bind_compute = [&](CollGridSrc &S, const char *label) {
    if (S.kind != COLL_SRC_COMP) return;
    if (!S.cid || !*S.cid) {
      char msg[160];
      snprintf(msg, sizeof(msg), "fix droplet/charge: empty compute ID for %s", label);
      error->all(FLERR, msg);
    }

    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg), "fix droplet/charge: compute '%s' for %s not found", S.cid, label);
      error->all(FLERR, msg);
    }

    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR, "fix droplet/charge: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR, "fix droplet/charge: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[220];
      snprintf(msg, sizeof(msg),
               "fix droplet/charge: column %d for compute '%s' (%s) out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR, msg);
    }
  };

  bind_compute(srcTe, "Te");
  bind_compute(srcTi, "Ti");
  bind_compute(srcNe, "Ne");
  bind_compute(srcNi, "Ni");

  auto same_compute = [&](const CollGridSrc &A, const CollGridSrc &B) {
    return A.kind == COLL_SRC_COMP && B.kind == COLL_SRC_COMP &&
           A.icompute >= 0 && A.icompute == B.icompute;
  };

  if (!use_grid_plasma &&
      same_compute(srcTe, srcTi) &&
      same_compute(srcTe, srcNe) &&
      same_compute(srcTe, srcNi)) {
    Compute *c = modify->compute[srcTe.icompute];
    point_plasma_compute = dynamic_cast<ComputePlasmaFields *>(c);
    use_point_plasma = (point_plasma_compute != nullptr);
  }
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  if (use_grid_plasma) compute_plasma_grid();
  refresh_compute_src(srcTe);
  refresh_compute_src(srcTi);
  refresh_compute_src(srcNe);
  refresh_compute_src(srcNi);

  apply_charge_update();
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  if (use_grid_plasma) compute_plasma_grid();
  refresh_compute_src(srcTe);
  refresh_compute_src(srcTi);
  refresh_compute_src(srcNe);
  refresh_compute_src(srcNi);

  apply_charge_update();
}

/* ---------------------------------------------------------------------- */

double FixDropletCharge::memory_usage()
{
  double bytes = 0.0;
  if (plasma_grid) bytes += static_cast<size_t>(maxgrid_plasma) * 4 * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::compute_plasma_grid()
{
  if (!use_grid_plasma || !grid->nlocal) return;

  if (grid->maxlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_grid);
    memory->create(plasma_grid, maxgrid_plasma, 4, "droplet/charge:plasma");
  }

  memset(&plasma_grid[0][0], 0, sizeof(double) * grid->nlocal * 4);

  const int stride = 4;
  if (srcTe.kind == COLL_SRC_VAR) input->variable->compute_grid(srcTe.varid, &plasma_grid[0][0], stride, 0);
  if (srcTi.kind == COLL_SRC_VAR) input->variable->compute_grid(srcTi.varid, &plasma_grid[0][1], stride, 0);
  if (srcNe.kind == COLL_SRC_VAR) input->variable->compute_grid(srcNe.varid, &plasma_grid[0][2], stride, 0);
  if (srcNi.kind == COLL_SRC_VAR) input->variable->compute_grid(srcNi.varid, &plasma_grid[0][3], stride, 0);
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr;
  int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts = update->ntimestep;
    return;
  }

  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts = update->ntimestep;
    return;
  }

  S.arr_cache = nullptr;
  S.src_index = -1;
  S.cache_ts = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

inline double FixDropletCharge::read_cell_src(const CollGridSrc& S, int icell, int col_plasma) const
{
  if (S.kind == COLL_SRC_COMP) {
    if (S.arr_cache && S.src_index >= 0) return S.arr_cache[icell][S.src_index];
    return 0.0;
  }
  if (S.kind == COLL_SRC_VAR) return plasma_grid[icell][col_plasma];
  return 0.0;
}

/* ---------------------------------------------------------------------- */

bool FixDropletCharge::point_plasma_ready() const
{
  return use_point_plasma && point_plasma_compute;
}

/* ---------------------------------------------------------------------- */

bool FixDropletCharge::solve_phi_oml(double Te_eV, double Ti_eV, double ne_m3, double ni_m3,
                                     double Td_K, double rd_m, double &phi_V) const
{
  if (!(Te_eV > 0.0) || !(Ti_eV > 0.0) || !(ne_m3 > 0.0) || !(ni_m3 > 0.0) || !(rd_m > 0.0))
    return false;

  const double qe = update->echarge;           // C
  const double me = update->electron_mass;     // kg
  const double mi = ion_mass_amu * update->proton_mass; // kg
  const double kB = update->boltz;             // J/K

  const double area_coll = MY_PI * rd_m * rd_m;
  const double area_emit = 4.0 * MY_PI * rd_m * rd_m;

  const double Ce = qe * area_coll * ne_m3 *
                    std::sqrt(8.0 * (Te_eV * qe) / (MY_PI * me));
  const double Ci = qe * area_coll * ni_m3 *
                    std::sqrt(8.0 * (Ti_eV * qe) / (MY_PI * mi));

  auto thermionic_current = [&]() -> double {
    if (!thermionic_on) return 0.0;
    if (!(Td_K > 0.0) || richardson_A <= 0.0 || work_function_eV <= 0.0) return 0.0;

    // Size-corrected work function: W(rd)=W_inf - e^2/(16*pi*eps0*rd)
    const double eps0 = 8.8541878128e-12;
    const double W_inf_J = work_function_eV * qe;
    const double dW = (qe * qe) / (16.0 * MY_PI * eps0 * rd_m);
    const double W_rd = std::max(0.0, W_inf_J - dW);
    return area_emit * richardson_A * Td_K * Td_K * std::exp(-W_rd / (kB * Td_K));
  };

  const double Ith = thermionic_current();

  auto f = [&](double phi) -> double {
    const double Ie = -Ce * std::exp(phi / Te_eV);
    const double Ii =  Ci * (1.0 - phi / Ti_eV);
    return Ie + Ii + Ith;
  };

  // Monotonic f(phi) for this OML closure, so bisection is robust.
  double lo = -80.0 * std::max(Te_eV, Ti_eV);
  double hi =  20.0 * std::max(Te_eV, Ti_eV);
  double flo = f(lo);
  double fhi = f(hi);

  int expand = 0;
  while (std::isfinite(flo) && std::isfinite(fhi) && flo * fhi > 0.0 && expand < 12) {
    lo *= 2.0;
    hi *= 2.0;
    flo = f(lo);
    fhi = f(hi);
    expand++;
  }

  if (!std::isfinite(flo) || !std::isfinite(fhi) || flo * fhi > 0.0)
    return false;

  double mid = -3.0 * Te_eV;
  if (mid < lo || mid > hi) mid = 0.5 * (lo + hi);
  for (int it = 0; it < 100; ++it) {
    mid = 0.5 * (lo + hi);
    const double fm = f(mid);
    if (!std::isfinite(fm)) return false;
    if (std::fabs(fm) <= 1.0e-10 * std::max(1.0, std::fabs(Ci + Ith))) break;

    if (flo * fm <= 0.0) {
      hi = mid;
      fhi = fm;
    } else {
      lo = mid;
      flo = fm;
    }
    if (std::fabs(hi - lo) <= 1.0e-10 * std::max(1.0, std::fabs(mid))) break;
  }

  phi_V = mid;
  return std::isfinite(phi_V);
}

/* ---------------------------------------------------------------------- */

void FixDropletCharge::apply_charge_update()
{
  // Do NOT early-return when grid->nlocal == 0: every rank must reach the
  // MPI_Allreduce below, otherwise ranks with zero cells deadlock the rest.

  if (!particle->sorted) particle->sort();

  auto *parts = particle->particles;
  int *next = particle->next;
  auto *cinfo = grid->cinfo;
  const int nglocal = grid->nlocal;

  // Constants for converting surface potential to equivalent particle charge.
  const double eps0 = 8.8541878128e-12;   // F/m
  const double qe = update->echarge;      // C

  // Species-accumulated charge numbers to avoid last-particle-wins behavior.
  std::vector<double> zsum(static_cast<size_t>(particle->nspecies), 0.0);
  std::vector<long long> zcount(static_cast<size_t>(particle->nspecies), 0);

  long long n_local = 0;
  long long n_valid_local = 0;
  double zd_min_local = std::numeric_limits<double>::infinity();
  double zd_max_local = -std::numeric_limits<double>::infinity();
  double zd_sum_local = 0.0;

  for (int icell = 0; icell < nglocal; ++icell) {
    if (cinfo[icell].count == 0) continue;

    int ip = cinfo[icell].first;
    while (ip >= 0) {
      Particle::OnePart &p = parts[ip];
      n_local++;

      double Te_eV, Ti_eV, Ne_m3, Ni_m3;
      if (point_plasma_ready()) {
        const PlasmaFileParams pp = point_plasma_compute->query_plasma_at_point(p.x);
        Te_eV = std::max(pp.temp_e, 0.0);
        Ti_eV = std::max(pp.temp_i, 0.0);
        Ne_m3 = std::max(pp.dens_e, 0.0);
        Ni_m3 = std::max(pp.dens_i, 0.0);
      } else {
        Te_eV = std::max(read_cell_src(srcTe, icell, 0), 0.0);
        Ti_eV = std::max(read_cell_src(srcTi, icell, 1), 0.0);
        Ne_m3 = std::max(read_cell_src(srcNe, icell, 2), 0.0);
        Ni_m3 = std::max(read_cell_src(srcNi, icell, 3), 0.0);
      }

      double rd = (p.radius > 0.0) ? p.radius : seed_radius;
      const double Td_K = (p.temp > 0.0) ? p.temp : seed_temp;
      if (p.mass <= 0.0 && seed_mass > 0.0) p.mass = seed_mass;
      if (p.radius <= 0.0 && seed_radius > 0.0) p.radius = seed_radius;
      if (p.temp <= 0.0 && seed_temp > 0.0) p.temp = seed_temp;
      if (!(rd > 0.0) || !(Te_eV > 0.0) || !(Ti_eV > 0.0) || !(Ne_m3 > 0.0) || !(Ni_m3 > 0.0)) {
        ip = next[ip];
        continue;
      }

      double phi_s = 0.0;
      if (!solve_phi_oml(Te_eV, Ti_eV, Ne_m3, Ni_m3, Td_K, rd, phi_s)) {
        ip = next[ip];
        continue;
      }

      const double qd_coulomb = 4.0 * MY_PI * eps0 * rd * phi_s;
//      const double zd = std::fabs(qd_coulomb / qe);
        const double zd =qd_coulomb / qe;

      if (std::isfinite(zd)) {
        n_valid_local++;
        zd_sum_local += zd;
        zd_min_local = std::min(zd_min_local, zd);
        zd_max_local = std::max(zd_max_local, zd);

        const int is = p.ispecies;
        if (is >= 0 && is < particle->nspecies) {
          zsum[static_cast<size_t>(is)] += zd;
          zcount[static_cast<size_t>(is)]++;
        }
      }

      ip = next[ip];
    }
  }

  // Aggregate species averages across MPI ranks.
  std::vector<double> zsum_global(zsum.size(), 0.0);
  std::vector<long long> zcount_global(zcount.size(), 0);
  if (!zsum.empty()) {
    MPI_Allreduce(zsum.data(), zsum_global.data(), static_cast<int>(zsum.size()), MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(zcount.data(), zcount_global.data(), static_cast<int>(zcount.size()), MPI_LONG_LONG, MPI_SUM, world);
  }

  for (int is = 0; is < particle->nspecies; ++is) {
    const long long n = zcount_global[static_cast<size_t>(is)];
    if (n > 0) {
      particle->species[is].charge = zsum_global[static_cast<size_t>(is)] / static_cast<double>(n);
    }
  }

}
