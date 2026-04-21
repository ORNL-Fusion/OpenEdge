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
#include "fix_plasma_data.h"
#include "openedge_geom.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16
enum { INT, DOUBLE };

enum { DIFF_NONE=0, DIFF_CONST, DIFF_BOHM };

/* ---------------------------------------------------------------------- */

FixCrossDiffusion::FixCrossDiffusion(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  rng_(nullptr),
  use_plasma_data_(0),
  plasma_fix_id_(),
  pd_(nullptr),
  diff_model_(DIFF_NONE),
  D_perp_(0.0),
  bohm_scale_(1.0),
  have_pinch_(0),
  v_pinch_R_(0.0),
  v_pinch_Z_(0.0),
  have_grad_pinch_(0),
  C_p_(0.0)
{
  // fix ID cross_diffusion Nevery
  //     {bfield BxSRC BySRC BzSRC | plasma_data FIXID} [keywords...]

  if (narg < 5)
    error->all(FLERR,
      "Illegal fix cross_diffusion command "
      "(need: Nevery {bfield BxSRC BySRC BzSRC | plasma_data FIXID})");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg], "plasma_data") == 0) {
    iarg++;
    if (iarg >= narg)
      error->all(FLERR, "fix cross_diffusion: plasma_data needs a fix ID");
    use_plasma_data_ = 1;
    plasma_fix_id_ = arg[iarg++];
  } else {
    if (strcmp(arg[iarg++], "bfield") != 0)
      error->all(FLERR, "fix cross_diffusion: missing 'bfield' keyword");
    if (iarg + 3 > narg)
      error->all(FLERR,
        "Illegal fix cross_diffusion command "
        "(need: Nevery bfield BxSRC BySRC BzSRC)");
    parse_compute_src(arg[iarg++], srcBx_, "Bx");
    parse_compute_src(arg[iarg++], srcBy_, "By");
    parse_compute_src(arg[iarg++], srcBz_, "Bz");
  }

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
      if (!use_plasma_data_) {
        if (iarg >= narg)
          error->all(FLERR, "fix cross_diffusion bohm: need TeSRC");
        parse_compute_src(arg[iarg++], srcTe_, "Te");
      } else if (iarg < narg && strcmp(arg[iarg], "scale") != 0 &&
                 strcmp(arg[iarg], "D_perp") != 0 &&
                 strcmp(arg[iarg], "bohm") != 0 &&
                 strcmp(arg[iarg], "pinch") != 0 &&
                 strcmp(arg[iarg], "gradient_pinch") != 0) {
        error->all(FLERR,
          "fix cross_diffusion bohm: in plasma_data mode only 'scale VAL' is allowed");
      }
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
      if (iarg >= narg)
        error->all(FLERR, "fix cross_diffusion gradient_pinch: need Cp");
      C_p_ = input->numeric(FLERR, arg[iarg++]);
      if (!use_plasma_data_) {
        if (iarg + 3 > narg)
          error->all(FLERR,
            "fix cross_diffusion gradient_pinch: need Cp neSRC gradNeR_SRC gradNeZ_SRC");
        parse_compute_src(arg[iarg++], srcNe_, "Ne");
        parse_compute_src(arg[iarg++], srcGradNeR_, "gradNeR");
        parse_compute_src(arg[iarg++], srcGradNeZ_, "gradNeZ");
      } else if (iarg < narg &&
                 strcmp(arg[iarg], "D_perp") != 0 &&
                 strcmp(arg[iarg], "bohm") != 0 &&
                 strcmp(arg[iarg], "pinch") != 0 &&
                 strcmp(arg[iarg], "gradient_pinch") != 0) {
        error->all(FLERR,
          "fix cross_diffusion gradient_pinch: in plasma_data mode only Cp is required");
      }
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
  delete[] srcBx_.pname;
  delete[] srcBy_.cid;
  delete[] srcBy_.pname;
  delete[] srcBz_.cid;
  delete[] srcBz_.pname;
  if (diff_model_ == DIFF_BOHM) {
    delete[] srcTe_.cid;
    delete[] srcTe_.pname;
  }
  if (have_grad_pinch_) {
    delete[] srcNe_.cid;
    delete[] srcNe_.pname;
    delete[] srcGradNeR_.cid;
    delete[] srcGradNeR_.pname;
    delete[] srcGradNeZ_.cid;
    delete[] srcGradNeZ_.pname;
  }
}

/* ---------------------------------------------------------------------- */

int FixCrossDiffusion::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
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
    if (S.kind == COLL_SRC_COMP) {
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
      return;
    }
    if (S.kind == COLL_SRC_PCUSTOM) {
      S.ipcustom = particle->find_custom(S.pname);
      if (S.ipcustom < 0) {
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "fix cross_diffusion: particle custom '%s' for %s not found",
                 S.pname, label);
        error->all(FLERR, msg);
      }
      if (particle->etype[S.ipcustom] != DOUBLE)
        error->all(FLERR,
          "fix cross_diffusion: particle custom source must be floating point");
      if (particle->esize[S.ipcustom] != 0)
        error->all(FLERR,
          "fix cross_diffusion: particle custom source must be a vector");
      S.ipwhich = particle->ewhich[S.ipcustom];
    }
  };

  if (use_plasma_data_) {
    const int ifix = modify->find_fix(plasma_fix_id_.c_str());
    if (ifix < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix cross_diffusion: plasma_data fix '%s' not found",
               plasma_fix_id_.c_str());
      error->all(FLERR, msg);
    }
    pd_ = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
    if (!pd_)
      error->all(FLERR,
        "fix cross_diffusion: plasma_data fix must be style plasma/data");
    pd_->init();
  } else {
    bind(srcBx_, "Bx");
    bind(srcBy_, "By");
    bind(srcBz_, "Bz");
  }
  if (diff_model_ == DIFF_BOHM && !use_plasma_data_)
    bind(srcTe_, "Te");
  if (have_grad_pinch_ && !use_plasma_data_) {
    bind(srcNe_, "Ne");
    bind(srcGradNeR_, "gradNeR");
    bind(srcGradNeZ_, "gradNeZ");
  }
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) {
    update->cd_flag = 0;
    return;
  }

  // refresh caches
  if (!use_plasma_data_) {
    refresh_compute_src(srcBx_);
    refresh_compute_src(srcBy_);
    refresh_compute_src(srcBz_);
  }
  if (diff_model_ == DIFF_BOHM && !use_plasma_data_)
    refresh_compute_src(srcTe_);
  if (have_grad_pinch_ && !use_plasma_data_) {
    refresh_compute_src(srcNe_);
    refresh_compute_src(srcGradNeR_);
    refresh_compute_src(srcGradNeZ_);
  }

  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  const int nlocal = particle->nlocal;
  const int dim    = domain->dimension;
  const double dt  = update->dt * nevery;

  // No particles to displace on this rank this step. Leave cross-diffusion
  // inactive so Update::move() will not touch a stale/null dx_cd buffer if
  // particles are inserted later in the step.
  if (nlocal == 0) {
    update->cd_flag = 0;
    return;
  }

  // (re)allocate displacement buffer if needed
  if (nlocal > update->cd_nmax) {
    memory->destroy(update->dx_cd);
    update->cd_nmax = nlocal + nlocal/10 + 1;  // headroom, +1 for nlocal=0
    memory->create(update->dx_cd, update->cd_nmax, 3, "update:dx_cd");
  }
  update->cd_flag = 1;

  double **dx = update->dx_cd;

  // zero the buffer
  if (nlocal > 0)
    memset(&dx[0][0], 0, sizeof(double) * nlocal * 3);

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];
    const int isp = p.ispecies;
    if (species[isp].charge == 0.0) continue;  // skip neutrals

    const int icell = p.icell;
    double B0, B1, B2;
    if (use_plasma_data_) pd_bfield_sparta(p, B0, B1, B2);
    else {
      B0 = read_src(srcBx_, ip, icell);
      B1 = read_src(srcBy_, ip, icell);
      B2 = read_src(srcBz_, ip, icell);
    }
    const double Bmag = std::sqrt(B0*B0 + B1*B1 + B2*B2);
    if (Bmag < 1.0e-20) continue;

    // compute local D_perp
    double D_local = 0.0;
    if (diff_model_ == DIFF_CONST) {
      D_local = D_perp_;
    } else if (diff_model_ == DIFF_BOHM) {
      const double Te_eV = use_plasma_data_ ? std::max(pd_interp(pd_->temp_e, p), 0.0)
                                            : std::max(read_src(srcTe_, ip, icell), 0.0);
      D_local = bohm_scale_ * Te_eV / (16.0 * Bmag);
    }

    // stochastic perpendicular displacement
    if (D_local > 0.0) {
      const double sigma = std::sqrt(2.0 * D_local * dt);

      if (dim == 2) {
        const double Bpol = std::sqrt(B0*B0 + B1*B1);
        if (Bpol > 1.0e-20) {
          const double inv_Bpol = 1.0 / Bpol;
          const double eperp0 = -B1 * inv_Bpol;
          const double eperp1 =  B0 * inv_Bpol;

          const double xi = rng_->gaussian();
          dx[ip][0] += sigma * xi * eperp0;
          dx[ip][1] += sigma * xi * eperp1;
        }
      } else {
        const double inv_Bmag = 1.0 / Bmag;
        const double bhat0 = B0 * inv_Bmag;
        const double bhat1 = B1 * inv_Bmag;
        const double bhat2 = B2 * inv_Bmag;

        double ax0 = 1.0, ax1 = 0.0, ax2 = 0.0;
        if (std::fabs(bhat0) > 0.9) { ax0 = 0.0; ax1 = 1.0; }
        double dot = ax0*bhat0 + ax1*bhat1 + ax2*bhat2;
        double e1_0 = ax0 - dot*bhat0;
        double e1_1 = ax1 - dot*bhat1;
        double e1_2 = ax2 - dot*bhat2;
        double e1mag = std::sqrt(e1_0*e1_0 + e1_1*e1_1 + e1_2*e1_2);
        e1_0 /= e1mag; e1_1 /= e1mag; e1_2 /= e1mag;

        double e2_0 = bhat1*e1_2 - bhat2*e1_1;
        double e2_1 = bhat2*e1_0 - bhat0*e1_2;
        double e2_2 = bhat0*e1_1 - bhat1*e1_0;

        const double xi1 = rng_->gaussian();
        const double xi2 = rng_->gaussian();
        dx[ip][0] += sigma * (xi1*e1_0 + xi2*e2_0);
        dx[ip][1] += sigma * (xi1*e1_1 + xi2*e2_1);
        dx[ip][2] += sigma * (xi1*e1_2 + xi2*e2_2);
      }
    }

    // deterministic pinch displacement (cylindrical R, Z)
    if (have_pinch_) {
      if (dim == 2) {
        double dxs0, dxs1, dxs2_unused;
        OpenEdge::RZphi_force_to_sparta(v_pinch_R_, v_pinch_Z_, 0.0,
                                         dim, domain->axisymmetric, 0.0,
                                         dxs0, dxs1, dxs2_unused);
        dx[ip][0] += dxs0 * dt;
        dx[ip][1] += dxs1 * dt;
      } else {
        const double rx = p.x[0];
        const double ry = p.x[1];
        const double R = std::sqrt(rx*rx + ry*ry);
        if (R > 1.0e-20) {
          const double cphi = rx / R;
          const double sphi = ry / R;
          dx[ip][0] += v_pinch_R_ * cphi * dt;
          dx[ip][1] += v_pinch_R_ * sphi * dt;
        }
        dx[ip][2] += v_pinch_Z_ * dt;
      }
    }

    // gradient-driven pinch
    if (have_grad_pinch_ && D_local > 0.0) {
      const double ne_loc = use_plasma_data_ ? std::max(pd_interp(pd_->dens_e, p), 1.0e10)
                                             : std::max(read_src(srcNe_, ip, icell), 1.0e10);
      double gNeR, gNeZ;
      if (use_plasma_data_) {
        double R, Z;
        OpenEdge::sparta_to_RZ(p.x, dim, domain->axisymmetric, R, Z);
        // Prefer precomputed mesh gradients (converter writes them in
        // /mesh/grad_ne_{r,z}). Fall back to per-particle FD on the
        // regular (R, Z) grid only if the mesh path is unavailable.
        bool used_mesh_grad = false;
        if (!pd_->mesh_grad_ne_r.empty() && pd_->has_mesh) {
          const int cell = pd_->mesh_cell_at(R, Z);
          if (cell >= 0 &&
              cell < static_cast<int>(pd_->mesh_grad_ne_r.size())) {
            gNeR = pd_->mesh_grad_ne_r[cell];
            gNeZ = pd_->mesh_grad_ne_z[cell];
            used_mesh_grad = true;
          }
        }
        if (!used_mesh_grad) {
          if (pd_->rvals.size() < 2 || pd_->zvals.size() < 2) {
            gNeR = 0.0; gNeZ = 0.0;
          } else {
            const double dR = std::max(1.0e-9, 0.5 * std::fabs(pd_->rvals[1] - pd_->rvals[0]));
            const double dZ = std::max(1.0e-9, 0.5 * std::fabs(pd_->zvals[1] - pd_->zvals[0]));
            gNeR = (pd_->interp2D(pd_->dens_e, R + dR, Z) -
                    pd_->interp2D(pd_->dens_e, R - dR, Z)) / (2.0 * dR);
            gNeZ = (pd_->interp2D(pd_->dens_e, R, Z + dZ) -
                    pd_->interp2D(pd_->dens_e, R, Z - dZ)) / (2.0 * dZ);
          }
        }
      } else {
        gNeR = read_src(srcGradNeR_, ip, icell);
        gNeZ = read_src(srcGradNeZ_, ip, icell);
      }

      if (dim == 2) {
        // Work in physical (R,Z) so the math doesn't depend on the SPARTA
        // slot layout (Cart vs axi). Convert B from SPARTA slots to (BR,BZ),
        // build perpendicular unit vector in (R,Z), dot with grad_ne, then
        // convert displacement back to SPARTA slots.
        const double Bvec[3] = {B0, B1, B2};
        double BR_phys, BZ_phys, Bphi_unused;
        OpenEdge::sparta_v_to_RZphi(Bvec, dim, domain->axisymmetric, 0.0,
                                     BR_phys, BZ_phys, Bphi_unused);
        const double Bpol = std::sqrt(BR_phys*BR_phys + BZ_phys*BZ_phys);
        if (Bpol > 1.0e-20) {
          const double inv_Bpol = 1.0 / Bpol;
          const double eperp_R = -BZ_phys * inv_Bpol;
          const double eperp_Z =  BR_phys * inv_Bpol;
          const double grad_ne_perp = gNeR * eperp_R + gNeZ * eperp_Z;
          const double V_gp = C_p_ * D_local * grad_ne_perp / ne_loc;
          double dxs0, dxs1, dxs2_unused;
          OpenEdge::RZphi_force_to_sparta(V_gp * eperp_R, V_gp * eperp_Z, 0.0,
                                           dim, domain->axisymmetric, 0.0,
                                           dxs0, dxs1, dxs2_unused);
          dx[ip][0] += dxs0 * dt;
          dx[ip][1] += dxs1 * dt;
        }
      } else {
        const double inv_Bmag = 1.0 / Bmag;
        const double bhat0 = B0 * inv_Bmag;
        const double bhat1 = B1 * inv_Bmag;
        const double bhat2 = B2 * inv_Bmag;
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
        const double gdotb = gNe_x*bhat0 + gNe_y*bhat1 + gNe_z*bhat2;
        const double gp0 = gNe_x - gdotb*bhat0;
        const double gp1 = gNe_y - gdotb*bhat1;
        const double gp2 = gNe_z - gdotb*bhat2;
        const double V_scale = C_p_ * D_local / ne_loc;
        dx[ip][0] += V_scale * gp0 * dt;
        dx[ip][1] += V_scale * gp1 * dt;
        dx[ip][2] += V_scale * gp2 * dt;
      }
    }
  }
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

  if (strncmp(tok, "c_", 2) == 0) {
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
    return;
  }

  if (strncmp(tok, "p_", 2) == 0) {
    dst.kind = COLL_SRC_PCUSTOM;
    const char *name = tok + 2;
    dst.pname = new char[strlen(name) + 1];
    strcpy(dst.pname, name);
    return;
  }

  error->all(FLERR,
    "fix cross_diffusion: source must be c_ID[col] or p_name");
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind == COLL_SRC_PCUSTOM) {
    if (S.cache_ts == update->ntimestep) return;
    S.pvec_cache = nullptr;
    if (S.ipcustom >= 0) {
      S.ipwhich = particle->ewhich[S.ipcustom];
      if (S.ipwhich >= 0) S.pvec_cache = particle->edvec[S.ipwhich];
    }
    S.cache_ts = update->ntimestep;
    return;
  }
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
  S.pvec_cache = nullptr;
  S.src_index = -1;
  S.cache_ts  = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

double FixCrossDiffusion::read_src(const CollGridSrc &S, int ip, int icell) const
{
  if (S.kind == COLL_SRC_PCUSTOM) {
    if (!S.pvec_cache) return 0.0;
    return S.pvec_cache[ip];
  }
  if (S.kind != COLL_SRC_COMP) return 0.0;
  if (!S.arr_cache || S.src_index < 0) return 0.0;
  return S.arr_cache[icell][S.src_index];
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::particle_rz(const Particle::OnePart &p,
                                    double &R, double &Z) const
{
  OpenEdge::sparta_to_RZ(p.x, domain->dimension, domain->axisymmetric, R, Z);
}

/* ---------------------------------------------------------------------- */

double FixCrossDiffusion::pd_interp(const std::vector<double> &field,
                                    const Particle::OnePart &p) const
{
  if (!pd_) return 0.0;
  double R, Z;
  particle_rz(p, R, Z);
  return pd_->interp2D(field, R, Z);
}

/* ---------------------------------------------------------------------- */

void FixCrossDiffusion::pd_bfield_sparta(const Particle::OnePart &p,
                                         double &B0, double &B1, double &B2) const
{
  B0 = B1 = B2 = 0.0;
  if (!pd_ || !pd_->has_bfield) return;

  double R, Z;
  particle_rz(p, R, Z);

  double Br = 0.0, Bz = 0.0, Bt = 0.0;
  pd_->bfield_at(R, Z, Br, Bz, Bt);

  // Decompose physical (Br, Bz, Bt) onto SPARTA's (B0, B1, B2) slot layout.
  double phi = 0.0;
  if (domain->dimension == 3) phi = std::atan2(p.x[1], p.x[0]);
  OpenEdge::RZphi_force_to_sparta(Br, Bz, Bt, domain->dimension,
                                   domain->axisymmetric, phi, B0, B1, B2);
}
