/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix cross_diffusion: anomalous perpendicular diffusion and
    convective pinch for impurity ions.

    Position displacement applied at END_OF_STEP (after Boris push).

    Diffusion:
      2D: single perpendicular direction in poloidal plane
          e_perp = (-bhat_Z, bhat_R) / |B_pol|  (in SPARTA slots)
          dx = sqrt(2 * D_perp * dt) * xi * e_perp

      3D: two perpendicular directions via Gram-Schmidt
          dx = sqrt(2 * D_perp * dt) * (xi1 * e1 + xi2 * e2)

    Pinch: constant velocity in cylindrical (R, Z) coordinates.
      2D: dx[0] += Vr * dt, dx[1] += Vz * dt
      3D: dx[0] += Vr*cos(phi)*dt, dx[1] += Vr*sin(phi)*dt, dx[2] += Vz*dt

    After displacement, particles that leave their cell are relocated
    via grid->id_find_child().
------------------------------------------------------------------------- */

#include "fix_cross_diffusion.h"

#include <cmath>
#include <cstring>

#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "memory.h"
#include "modify.h"
#include "particle.h"
#include "random_knuth.h"
#include "random_mars.h"
#include "update.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16

enum { DIFF_NONE=0, DIFF_CONST, DIFF_BOHM };

/* ---------------------------------------------------------------------- */

FixCrossDiffusion::FixCrossDiffusion(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  rng_(nullptr),
  diff_model_(DIFF_NONE),
  D_perp_(0.0),
  bohm_scale_(1.0),
  have_pinch_(0),
  v_pinch_R_(0.0),
  v_pinch_Z_(0.0),
  have_grad_pinch_(0),
  C_p_(0.0)
{
  // fix ID cross_diffusion Nevery bfield BxSRC BySRC BzSRC [keywords...]

  if (narg < 7)
    error->all(FLERR,
      "Illegal fix cross_diffusion command "
      "(need: Nevery bfield BxSRC BySRC BzSRC)");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg++], "bfield") != 0)
    error->all(FLERR, "fix cross_diffusion: missing 'bfield' keyword");

  parse_compute_src(arg[iarg++], srcBx_, "Bx");
  parse_compute_src(arg[iarg++], srcBy_, "By");
  parse_compute_src(arg[iarg++], srcBz_, "Bz");

  // optional keywords
  while (iarg < narg) {

    if (strcmp(arg[iarg], "D_perp") == 0) {
      iarg++;
      if (iarg >= narg)
        error->all(FLERR, "fix cross_diffusion D_perp: need value");
      D_perp_ = input->numeric(FLERR, arg[iarg++]);
      if (D_perp_ < 0.0)
        error->all(FLERR, "fix cross_diffusion: D_perp must be >= 0");
      diff_model_ = DIFF_CONST;

    } else if (strcmp(arg[iarg], "bohm") == 0) {
      iarg++;
      if (iarg >= narg)
        error->all(FLERR, "fix cross_diffusion bohm: need TeSRC");
      parse_compute_src(arg[iarg++], srcTe_, "Te");
      diff_model_ = DIFF_BOHM;

      // optional scale keyword
      if (iarg + 1 < narg && strcmp(arg[iarg], "scale") == 0) {
        iarg++;
        bohm_scale_ = input->numeric(FLERR, arg[iarg++]);
      }

    } else if (strcmp(arg[iarg], "pinch") == 0) {
      iarg++;
      if (iarg + 2 > narg)
        error->all(FLERR, "fix cross_diffusion pinch: need Vr Vz");
      v_pinch_R_ = input->numeric(FLERR, arg[iarg++]);
      v_pinch_Z_ = input->numeric(FLERR, arg[iarg++]);
      have_pinch_ = 1;

    } else if (strcmp(arg[iarg], "gradient_pinch") == 0) {
      iarg++;
      if (iarg + 4 > narg)
        error->all(FLERR,
          "fix cross_diffusion gradient_pinch: need Cp neSRC gradNeR_SRC gradNeZ_SRC");
      C_p_ = input->numeric(FLERR, arg[iarg++]);
      parse_compute_src(arg[iarg++], srcNe_, "Ne");
      parse_compute_src(arg[iarg++], srcGradNeR_, "gradNeR");
      parse_compute_src(arg[iarg++], srcGradNeZ_, "gradNeZ");
      have_grad_pinch_ = 1;

    } else {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix cross_diffusion: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (diff_model_ == DIFF_NONE && !have_pinch_ && !have_grad_pinch_)
    error->all(FLERR,
      "fix cross_diffusion: at least one of D_perp, bohm, pinch, or "
      "gradient_pinch must be specified");
}

/* ---------------------------------------------------------------------- */

FixCrossDiffusion::~FixCrossDiffusion()
{
  if (copymode) return;
  delete rng_;
  delete[] srcBx_.cid;
  delete[] srcBy_.cid;
  delete[] srcBz_.cid;
  if (diff_model_ == DIFF_BOHM)
    delete[] srcTe_.cid;
  if (have_grad_pinch_) {
    delete[] srcNe_.cid;
    delete[] srcGradNeR_.cid;
    delete[] srcGradNeZ_.cid;
  }
}

/* ---------------------------------------------------------------------- */

int FixCrossDiffusion::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::init()
{
  // seed RNG (same pattern as fix_coll_nanbu)
  if (!rng_) {
    rng_ = new RanKnuth(update->ranmaster->uniform());
    double seed = update->ranmaster->uniform();
    rng_->reset(seed, comm->me, 100);
  }

  // resolve compute sources
  auto bind = [&](CollGridSrc &S, const char *label) {
    if (S.kind != COLL_SRC_COMP) return;
    S.icompute = modify->find_compute(S.cid);
    if (S.icompute < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix cross_diffusion: compute '%s' for %s not found",
               S.cid, label);
      error->all(FLERR, msg);
    }
    Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0)
      error->all(FLERR, "fix cross_diffusion: compute must be per-grid");
    if (c->size_per_grid_cols == 0)
      error->all(FLERR, "fix cross_diffusion: compute has no per-grid array");
    if (S.col < 1 || S.col > c->size_per_grid_cols) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix cross_diffusion: column %d for compute '%s' (%s) "
               "out of range [1..%d]",
               S.col, S.cid, label, c->size_per_grid_cols);
      error->all(FLERR, msg);
    }
  };

  bind(srcBx_, "Bx");
  bind(srcBy_, "By");
  bind(srcBz_, "Bz");
  if (diff_model_ == DIFF_BOHM)
    bind(srcTe_, "Te");
  if (have_grad_pinch_) {
    bind(srcNe_, "Ne");
    bind(srcGradNeR_, "gradNeR");
    bind(srcGradNeZ_, "gradNeZ");
  }
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // refresh caches
  refresh_compute_src(srcBx_);
  refresh_compute_src(srcBy_);
  refresh_compute_src(srcBz_);
  if (diff_model_ == DIFF_BOHM)
    refresh_compute_src(srcTe_);
  if (have_grad_pinch_) {
    refresh_compute_src(srcNe_);
    refresh_compute_src(srcGradNeR_);
    refresh_compute_src(srcGradNeZ_);
  }

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  const int nlocal = particle->nlocal;
  const int dim    = domain->dimension;
  const double dt  = update->dt * nevery;
  const double QE  = update->echarge;

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];
    const int isp = p.ispecies;
    if (species[isp].charge == 0.0) continue;  // skip neutrals

    const int icell = p.icell;

    // save position for revert on failed relocation
    const double x_old[3] = {p.x[0], p.x[1], p.x[2]};

    // read B in SPARTA coordinate order
    const double B0 = read_cell_src(srcBx_, icell);
    const double B1 = read_cell_src(srcBy_, icell);
    const double B2 = read_cell_src(srcBz_, icell);
    const double Bmag = std::sqrt(B0*B0 + B1*B1 + B2*B2);
    if (Bmag < 1.0e-20) continue;

    // compute local D_perp
    double D_local = 0.0;
    if (diff_model_ == DIFF_CONST) {
      D_local = D_perp_;
    } else if (diff_model_ == DIFF_BOHM) {
      const double Te_eV = std::max(read_cell_src(srcTe_, icell), 0.0);
      // D_Bohm = Te / (16 * e * B)  [m^2/s]  (Te in eV -> Te*e in J)
      D_local = bohm_scale_ * Te_eV / (16.0 * Bmag);
    }

    // stochastic perpendicular displacement
    if (D_local > 0.0) {
      const double sigma = std::sqrt(2.0 * D_local * dt);

      if (dim == 2) {
        // 2D: B[0]=B_R, B[1]=B_Z in SPARTA slots
        // poloidal B magnitude
        const double Bpol = std::sqrt(B0*B0 + B1*B1);
        if (Bpol > 1.0e-20) {
          const double inv_Bpol = 1.0 / Bpol;
          // perpendicular direction in poloidal plane: (-bhat_Z, bhat_R)
          const double eperp0 = -B1 * inv_Bpol;  // -bhat_Z (SPARTA slot 0)
          const double eperp1 =  B0 * inv_Bpol;  //  bhat_R (SPARTA slot 1)

          const double xi = rng_->gaussian();
          p.x[0] += sigma * xi * eperp0;
          p.x[1] += sigma * xi * eperp1;
        }
      } else {
        // 3D: two perpendicular directions via Gram-Schmidt
        const double inv_Bmag = 1.0 / Bmag;
        const double bhat0 = B0 * inv_Bmag;
        const double bhat1 = B1 * inv_Bmag;
        const double bhat2 = B2 * inv_Bmag;

        // build e1 perpendicular to bhat
        double ax0 = 1.0, ax1 = 0.0, ax2 = 0.0;
        if (std::fabs(bhat0) > 0.9) { ax0 = 0.0; ax1 = 1.0; }
        double dot = ax0*bhat0 + ax1*bhat1 + ax2*bhat2;
        double e1_0 = ax0 - dot*bhat0;
        double e1_1 = ax1 - dot*bhat1;
        double e1_2 = ax2 - dot*bhat2;
        double e1mag = std::sqrt(e1_0*e1_0 + e1_1*e1_1 + e1_2*e1_2);
        e1_0 /= e1mag; e1_1 /= e1mag; e1_2 /= e1mag;

        // e2 = bhat x e1
        double e2_0 = bhat1*e1_2 - bhat2*e1_1;
        double e2_1 = bhat2*e1_0 - bhat0*e1_2;
        double e2_2 = bhat0*e1_1 - bhat1*e1_0;

        const double xi1 = rng_->gaussian();
        const double xi2 = rng_->gaussian();
        p.x[0] += sigma * (xi1*e1_0 + xi2*e2_0);
        p.x[1] += sigma * (xi1*e1_1 + xi2*e2_1);
        p.x[2] += sigma * (xi1*e1_2 + xi2*e2_2);
      }
    }

    // deterministic pinch displacement (cylindrical R, Z)
    if (have_pinch_) {
      if (dim == 2) {
        // 2D: slot 0 = R, slot 1 = Z
        p.x[0] += v_pinch_R_ * dt;
        p.x[1] += v_pinch_Z_ * dt;
      } else {
        // 3D: convert cylindrical pinch to Cartesian
        const double rx = p.x[0];
        const double ry = p.x[1];
        const double R = std::sqrt(rx*rx + ry*ry);
        if (R > 1.0e-20) {
          const double cphi = rx / R;
          const double sphi = ry / R;
          p.x[0] += v_pinch_R_ * cphi * dt;
          p.x[1] += v_pinch_R_ * sphi * dt;
        }
        p.x[2] += v_pinch_Z_ * dt;
      }
    }

    // gradient-driven pinch: V = C_p * D_local * grad_perp(ne) / ne
    // Direction: perpendicular to B in the poloidal plane
    if (have_grad_pinch_ && D_local > 0.0) {
      const double ne_loc = std::max(read_cell_src(srcNe_, icell), 1.0e10);
      const double gNeR = read_cell_src(srcGradNeR_, icell);  // 1/m^4... actually m^-3/m = m^-4
      const double gNeZ = read_cell_src(srcGradNeZ_, icell);

      if (dim == 2) {
        // 2D: perpendicular component of grad(ne) in poloidal plane
        // e_perp = (-bhat_Z, bhat_R) in SPARTA slots (B[0]=B_R, B[1]=B_Z)
        const double Bpol = std::sqrt(B0*B0 + B1*B1);
        if (Bpol > 1.0e-20) {
          const double inv_Bpol = 1.0 / Bpol;
          const double eperp0 = -B1 * inv_Bpol;
          const double eperp1 =  B0 * inv_Bpol;
          // grad_ne projected onto e_perp (cylindrical grad dotted with poloidal e_perp)
          const double grad_ne_perp = gNeR * eperp0 + gNeZ * eperp1;
          const double V_gp = C_p_ * D_local * grad_ne_perp / ne_loc;
          p.x[0] += V_gp * eperp0 * dt;
          p.x[1] += V_gp * eperp1 * dt;
        }
      } else {
        // 3D: project cylindrical gradient to Cartesian perpendicular direction
        const double inv_Bmag = 1.0 / Bmag;
        const double bhat0 = B0 * inv_Bmag;
        const double bhat1 = B1 * inv_Bmag;
        const double bhat2 = B2 * inv_Bmag;
        // convert cylindrical grad_ne to Cartesian
        const double rx = p.x[0], ry = p.x[1];
        const double R = std::sqrt(rx*rx + ry*ry);
        double gNe_x, gNe_y, gNe_z;
        if (R > 1.0e-20) {
          const double cphi = rx / R, sphi = ry / R;
          gNe_x = gNeR * cphi;
          gNe_y = gNeR * sphi;
        } else {
          gNe_x = gNeR; gNe_y = 0.0;
        }
        gNe_z = gNeZ;
        // remove parallel component: grad_perp = grad - (grad . bhat) bhat
        const double gdotb = gNe_x*bhat0 + gNe_y*bhat1 + gNe_z*bhat2;
        const double gp0 = gNe_x - gdotb*bhat0;
        const double gp1 = gNe_y - gdotb*bhat1;
        const double gp2 = gNe_z - gdotb*bhat2;
        const double V_scale = C_p_ * D_local / ne_loc;
        p.x[0] += V_scale * gp0 * dt;
        p.x[1] += V_scale * gp1 * dt;
        p.x[2] += V_scale * gp2 * dt;
      }
    }

    // relocate particle if it left its cell
    relocate_particle(ip, x_old);
  }
}

/* ----------------------------------------------------------------------
   Relocate particle to correct cell after position displacement.
   If particle left the domain or cannot find a valid local cell,
   revert to the saved pre-displacement position.
------------------------------------------------------------------------- */

void FixCrossDiffusion::relocate_particle(int ip, const double x_save[3])
{
  Particle::OnePart &p = particle->particles[ip];
  Grid::ChildCell *cells = grid->cells;
  const int dim = domain->dimension;

  const double *lo = cells[p.icell].lo;
  const double *hi = cells[p.icell].hi;

  // check if still inside current cell
  bool inside = true;
  if (p.x[0] < lo[0] || p.x[0] > hi[0]) inside = false;
  if (p.x[1] < lo[1] || p.x[1] > hi[1]) inside = false;
  if (dim == 3) {
    if (p.x[2] < lo[2] || p.x[2] > hi[2]) inside = false;
  }
  if (inside) return;

  // check if still inside domain
  double *boxlo = domain->boxlo;
  double *boxhi = domain->boxhi;
  if (p.x[0] <= boxlo[0] || p.x[0] >= boxhi[0] ||
      p.x[1] <= boxlo[1] || p.x[1] >= boxhi[1]) {
    p.x[0] = x_save[0]; p.x[1] = x_save[1]; p.x[2] = x_save[2];
    return;
  }
  if (dim == 3) {
    if (p.x[2] <= boxlo[2] || p.x[2] >= boxhi[2]) {
      p.x[0] = x_save[0]; p.x[1] = x_save[1]; p.x[2] = x_save[2];
      return;
    }
  }

  // find new cell
  int newcell = grid->id_find_child(0, 0, boxlo, boxhi, p.x);
  if (newcell < 0) {
    p.x[0] = x_save[0]; p.x[1] = x_save[1]; p.x[2] = x_save[2];
    return;
  }

  // handle split cells (surface-cut cells)
  if (dim == 2) {
    if (cells[newcell].nsplit > 1 && cells[newcell].nsurf >= 0)
      newcell = update->split2d(newcell, p.x);
  } else {
    if (cells[newcell].nsplit > 1 && cells[newcell].nsurf >= 0)
      newcell = update->split3d(newcell, p.x);
  }

  if (newcell < 0) {
    p.x[0] = x_save[0]; p.x[1] = x_save[1]; p.x[2] = x_save[2];
    return;
  }

  // if new cell is on a different processor, revert
  if (cells[newcell].proc != comm->me) {
    p.x[0] = x_save[0]; p.x[1] = x_save[1]; p.x[2] = x_save[2];
    return;
  }

  p.icell = newcell;
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::parse_compute_src(const char *tok, CollGridSrc &dst,
                                           const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "fix cross_diffusion: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) != 0)
    error->all(FLERR,
      "fix cross_diffusion: source must be a compute (c_ID[col])");

  dst.kind = COLL_SRC_COMP;
  const char *name = tok + 2;
  const char *lb   = strchr(name, '[');
  const char *rb   = lb ? strrchr(name, ']') : nullptr;

  if (!lb || !rb || rb <= lb + 1)
    error->all(FLERR,
      "fix cross_diffusion: use c_ID[col] syntax for compute sources");

  int idlen = static_cast<int>(lb - name);
  dst.cid = new char[idlen + 1];
  strncpy(dst.cid, name, idlen);
  dst.cid[idlen] = '\0';
  dst.col = atoi(lb + 1);   // 1-based

  if (dst.col <= 0)
    error->all(FLERR,
      "fix cross_diffusion: compute column must be >= 1");
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind != COLL_SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  // tally-style mapped access (e.g. compute grid)
  double **arr = nullptr;
  int    *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);
  if (nmap > 0 && arr) {
    S.arr_cache = arr;
    S.src_index = cols ? cols[0] : (S.col - 1);
    S.cache_ts  = update->ntimestep;
    return;
  }

  // standard per-grid array (e.g. compute plasma/fields)
  if (c->size_per_grid_cols > 0 && c->array_grid) {
    S.arr_cache = c->array_grid;
    S.src_index = S.col - 1;
    S.cache_ts  = update->ntimestep;
    return;
  }

  S.arr_cache = nullptr;
  S.src_index = -1;
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double FixCrossDiffusion::read_cell_src(const CollGridSrc &S, int icell) const
{
  if (S.kind != COLL_SRC_COMP) return 0.0;
  if (!S.arr_cache || S.src_index < 0) return 0.0;
  return S.arr_cache[icell][S.src_index];
}
