/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix thermal_force: Braginskii thermal forces on impurity ions.

    Per-particle parallel acceleration:
      a_par = (beta_i * Z^2 * e * grad_par(Ti)
             + alpha_e * Z^2 * e * grad_par(Te)) / m_Z

    Applied as leapfrog half-kicks at START_OF_STEP and END_OF_STEP,
    consistent with the Boris integrator.

    B-field is read in SPARTA coordinate order (bx, by, bz):
      2D: bx=B_R, by=B_Z, bz=B_toroidal
      3D: bx=B_x, by=B_y, bz=B_z (Cartesian)
    Temperature gradients are always cylindrical (grad_T_R, grad_T_Z).
------------------------------------------------------------------------- */

#include "fix_thermal_force.h"

#include <cmath>
#include <cstring>

#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "modify.h"
#include "particle.h"
#include "update.h"
#include "fix_plasma_data.h"
#include "openedge_geom.h"

using namespace SPARTA_NS;

#define INVOKED_PER_GRID 16
enum { INT, DOUBLE };

/* ---------------------------------------------------------------------- */

FixThermalForce::FixThermalForce(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  use_plasma_data_(0),
  plasma_fix_id_(),
  pd_(nullptr),
  have_ion_thermal_(0),
  have_elec_thermal_(0),
  beta_i_(2.6),
  alpha_e_(0.71)
{
  // fix ID thermal_force Nevery {bfield BxSRC BySRC BzSRC | plasma_data FIXID}
  //     [keywords...]

  if (narg < 5)
    error->all(FLERR,
      "Illegal fix thermal_force command "
      "(need: Nevery {bfield BxSRC BySRC BzSRC | plasma_data FIXID})");

  int iarg = 2;
  nevery = input->inumeric(FLERR, arg[iarg++]);

  if (strcmp(arg[iarg], "plasma_data") == 0) {
    iarg++;
    if (iarg >= narg)
      error->all(FLERR, "fix thermal_force: plasma_data needs a fix ID");
    use_plasma_data_ = 1;
    plasma_fix_id_ = arg[iarg++];
  } else {
    if (strcmp(arg[iarg++], "bfield") != 0)
      error->all(FLERR, "fix thermal_force: missing 'bfield' keyword");
    if (iarg + 3 > narg)
      error->all(FLERR,
        "Illegal fix thermal_force command "
        "(need: Nevery bfield BxSRC BySRC BzSRC)");
    parse_compute_src(arg[iarg++], srcBx_, "Bx");
    parse_compute_src(arg[iarg++], srcBy_, "By");
    parse_compute_src(arg[iarg++], srcBz_, "Bz");
  }

  // optional keywords
  while (iarg < narg) {

    if (strcmp(arg[iarg], "ion_thermal") == 0) {
      iarg++;
      int enabled = 1;
      if (iarg < narg && (strcmp(arg[iarg], "yes") == 0 || strcmp(arg[iarg], "no") == 0)) {
        enabled = (strcmp(arg[iarg], "yes") == 0);
        iarg++;
      }
      have_ion_thermal_ = enabled;
      if (!enabled) continue;

      if (!use_plasma_data_) {
        if (iarg + 2 > narg)
          error->all(FLERR,
            "fix thermal_force ion_thermal: need gradTiR_SRC gradTiZ_SRC");
        parse_compute_src(arg[iarg++], srcGradTiR_, "gradTiR");
        parse_compute_src(arg[iarg++], srcGradTiZ_, "gradTiZ");
      } else if (iarg < narg &&
                 strcmp(arg[iarg], "ion_thermal") != 0 &&
                 strcmp(arg[iarg], "elec_thermal") != 0) {
        error->all(FLERR,
          "fix thermal_force ion_thermal: in plasma_data mode use only yes/no");
      }

    } else if (strcmp(arg[iarg], "elec_thermal") == 0) {
      iarg++;
      int enabled = 1;
      if (iarg < narg && (strcmp(arg[iarg], "yes") == 0 || strcmp(arg[iarg], "no") == 0)) {
        enabled = (strcmp(arg[iarg], "yes") == 0);
        iarg++;
      }
      have_elec_thermal_ = enabled;
      if (!enabled) continue;

      if (!use_plasma_data_) {
        if (iarg + 2 > narg)
          error->all(FLERR,
            "fix thermal_force elec_thermal: need gradTeR_SRC gradTeZ_SRC");
        parse_compute_src(arg[iarg++], srcGradTeR_, "gradTeR");
        parse_compute_src(arg[iarg++], srcGradTeZ_, "gradTeZ");
      } else if (iarg < narg &&
                 strcmp(arg[iarg], "ion_thermal") != 0 &&
                 strcmp(arg[iarg], "elec_thermal") != 0) {
        error->all(FLERR,
          "fix thermal_force elec_thermal: in plasma_data mode use only yes/no");
      }

    } else {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix thermal_force: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (!have_ion_thermal_ && !have_elec_thermal_)
    error->all(FLERR,
      "fix thermal_force: at least one of ion_thermal or "
      "elec_thermal must be specified");
}

/* ---------------------------------------------------------------------- */

FixThermalForce::~FixThermalForce()
{
  if (copymode) return;
  delete[] srcBx_.cid;
  delete[] srcBx_.pname;
  delete[] srcBy_.cid;
  delete[] srcBy_.pname;
  delete[] srcBz_.cid;
  delete[] srcBz_.pname;
  if (have_ion_thermal_) {
    delete[] srcGradTiR_.cid;
    delete[] srcGradTiR_.pname;
    delete[] srcGradTiZ_.cid;
    delete[] srcGradTiZ_.pname;
  }
  if (have_elec_thermal_) {
    delete[] srcGradTeR_.cid;
    delete[] srcGradTeR_.pname;
    delete[] srcGradTeZ_.cid;
    delete[] srcGradTeZ_.pname;
  }
}

/* ---------------------------------------------------------------------- */

int FixThermalForce::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::init()
{
  // resolve compute sources
  auto bind = [&](CollGridSrc &S, const char *label) {
    if (S.kind == COLL_SRC_COMP) {
      S.icompute = modify->find_compute(S.cid);
      if (S.icompute < 0) {
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "fix thermal_force: compute '%s' for %s not found",
                 S.cid, label);
        error->all(FLERR, msg);
      }
      Compute *c = modify->compute[S.icompute];
      if (c->per_grid_flag == 0)
        error->all(FLERR, "fix thermal_force: compute must be per-grid");
      if (c->size_per_grid_cols == 0)
        error->all(FLERR, "fix thermal_force: compute has no per-grid array");
      if (S.col < 1 || S.col > c->size_per_grid_cols) {
        char msg[200];
        snprintf(msg, sizeof(msg),
                 "fix thermal_force: column %d for compute '%s' (%s) "
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
                 "fix thermal_force: particle custom '%s' for %s not found",
                 S.pname, label);
        error->all(FLERR, msg);
      }
      if (particle->etype[S.ipcustom] != DOUBLE)
        error->all(FLERR,
          "fix thermal_force: particle custom source must be floating point");
      if (particle->esize[S.ipcustom] != 0)
        error->all(FLERR,
          "fix thermal_force: particle custom source must be a vector");
      S.ipwhich = particle->ewhich[S.ipcustom];
    }
  };

  if (use_plasma_data_) {
    const int ifix = modify->find_fix(plasma_fix_id_.c_str());
    if (ifix < 0) {
      char msg[200];
      snprintf(msg, sizeof(msg),
               "fix thermal_force: plasma_data fix '%s' not found",
               plasma_fix_id_.c_str());
      error->all(FLERR, msg);
    }
    pd_ = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
    if (!pd_)
      error->all(FLERR,
        "fix thermal_force: plasma_data fix must be style plasma/data");
    pd_->init();
  } else {
    bind(srcBx_, "Bx");
    bind(srcBy_, "By");
    bind(srcBz_, "Bz");
  }

  if (have_ion_thermal_ && !use_plasma_data_) {
    bind(srcGradTiR_, "gradTiR");
    bind(srcGradTiZ_, "gradTiZ");
  }
  if (have_elec_thermal_ && !use_plasma_data_) {
    bind(srcGradTeR_, "gradTeR");
    bind(srcGradTeZ_, "gradTeZ");
  }
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // refresh caches once per timestep
  if (!use_plasma_data_) {
    refresh_compute_src(srcBx_);
    refresh_compute_src(srcBy_);
    refresh_compute_src(srcBz_);
  }

  if (have_ion_thermal_ && !use_plasma_data_) {
    refresh_compute_src(srcGradTiR_);
    refresh_compute_src(srcGradTiZ_);
  }
  if (have_elec_thermal_ && !use_plasma_data_) {
    refresh_compute_src(srcGradTeR_);
    refresh_compute_src(srcGradTeZ_);
  }

  kick_half(0.5 * update->dt);
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // Per-particle custom vectors (S.pvec_cache) can be reallocated when
  // particles are created during Update::move() between start_of_step and
  // end_of_step (e.g. by fix emit/surf/pmi). Re-fetch the pointers before
  // the second kick or read_src() will deref freed memory.
  if (!use_plasma_data_) {
    refresh_compute_src(srcBx_);
    refresh_compute_src(srcBy_);
    refresh_compute_src(srcBz_);
  }
  if (have_ion_thermal_ && !use_plasma_data_) {
    refresh_compute_src(srcGradTiR_);
    refresh_compute_src(srcGradTiZ_);
  }
  if (have_elec_thermal_ && !use_plasma_data_) {
    refresh_compute_src(srcGradTeR_);
    refresh_compute_src(srcGradTeZ_);
  }

  kick_half(0.5 * update->dt);
}

/* ----------------------------------------------------------------------
   Apply half-kick from thermal forces.

   Per-particle parallel acceleration:
     a_par = (beta_i * Z^2 * QE * grad_par_Ti
            + alpha_e * Z^2 * QE * grad_par_Te) / m_Z

   B-field is in SPARTA coordinate order:
     2D: B[0]=B_R, B[1]=B_Z, B[2]=B_toroidal (= bx, by, bz output)
     3D: B[0]=B_x, B[1]=B_y, B[2]=B_z (Cartesian)

   Temperature gradients are always cylindrical (grad_T_R, grad_T_Z).

   For the parallel gradient, we need bhat in cylindrical frame:
     2D: bhat_R = B[0]/|B|, bhat_Z = B[1]/|B|
     3D: bhat_R_cyl = (bhat_x*cos_phi + bhat_y*sin_phi)
         bhat_Z_cyl = bhat_z

   Velocity kick is always in SPARTA coordinates (matches bhat order).
------------------------------------------------------------------------- */

void FixThermalForce::kick_half(double dt_half)
{
  Particle::OnePart *particles = particle->particles;
  Particle::Species *species   = particle->species;
  const int nlocal = particle->nlocal;
  const int dim    = domain->dimension;
  const double QE  = update->echarge;   // elementary charge [C]

  for (int ip = 0; ip < nlocal; ip++) {
    Particle::OnePart &p = particles[ip];
    const int isp = p.ispecies;
    const double Z = species[isp].charge;
    if (Z == 0.0) continue;  // skip neutrals

    const double m_Z = species[isp].mass;  // kg
    if (m_Z <= 0.0) continue;

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

    const double inv_Bmag = 1.0 / Bmag;

    // bhat in SPARTA coordinates (for velocity kick)
    const double bhat0 = B0 * inv_Bmag;
    const double bhat1 = B1 * inv_Bmag;
    const double bhat2 = B2 * inv_Bmag;

    // bhat in cylindrical frame (for gradient dot product). Gradients
    // are always (grad_T_R, grad_T_Z); no toroidal component.
    const double bhat_sparta[3] = {bhat0, bhat1, bhat2};
    const double phi_p = (dim == 3)
                          ? std::atan2(p.x[1], p.x[0]) : 0.0;
    double bhat_R_cyl, bhat_Z_cyl, bhat_phi_unused;
    OpenEdge::sparta_v_to_RZphi(bhat_sparta, dim, domain->axisymmetric,
                                 phi_p, bhat_R_cyl, bhat_Z_cyl,
                                 bhat_phi_unused);

    // accumulate parallel acceleration
    double a_par = 0.0;
    const double Z2 = Z * Z;

    if (have_ion_thermal_) {
      const double gTiR = use_plasma_data_
        ? pd_grad(pd_->mesh_grad_ti_r, pd_->grad_ti_r, p)
        : read_src(srcGradTiR_, ip, icell);
      const double gTiZ = use_plasma_data_
        ? pd_grad(pd_->mesh_grad_ti_z, pd_->grad_ti_z, p)
        : read_src(srcGradTiZ_, ip, icell);
      const double grad_par_Ti = gTiR * bhat_R_cyl + gTiZ * bhat_Z_cyl;
      a_par += beta_i_ * Z2 * QE * grad_par_Ti / m_Z;
    }

    if (have_elec_thermal_) {
      const double gTeR = use_plasma_data_
        ? pd_grad(pd_->mesh_grad_te_r, pd_->grad_te_r, p)
        : read_src(srcGradTeR_, ip, icell);
      const double gTeZ = use_plasma_data_
        ? pd_grad(pd_->mesh_grad_te_z, pd_->grad_te_z, p)
        : read_src(srcGradTeZ_, ip, icell);
      const double grad_par_Te = gTeR * bhat_R_cyl + gTeZ * bhat_Z_cyl;
      a_par += alpha_e_ * Z2 * QE * grad_par_Te / m_Z;
    }

    if (a_par == 0.0) continue;

    // apply half-kick along bhat in SPARTA coordinates
    p.v[0] += a_par * bhat0 * dt_half;
    p.v[1] += a_par * bhat1 * dt_half;
    p.v[2] += a_par * bhat2 * dt_half;
  }
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::parse_compute_src(const char *tok, CollGridSrc &dst,
                                        const char *label)
{
  if (!tok || !*tok) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "fix thermal_force: empty token for %s", label);
    error->all(FLERR, msg);
  }

  if (strncmp(tok, "c_", 2) == 0) {
    dst.kind = COLL_SRC_COMP;
    const char *name = tok + 2;
    const char *lb   = strchr(name, '[');
    const char *rb   = lb ? strrchr(name, ']') : nullptr;

    if (!lb || !rb || rb <= lb + 1)
      error->all(FLERR,
        "fix thermal_force: use c_ID[col] syntax for compute sources");

    int idlen = static_cast<int>(lb - name);
    dst.cid = new char[idlen + 1];
    strncpy(dst.cid, name, idlen);
    dst.cid[idlen] = '\0';
    dst.col = atoi(lb + 1);   // 1-based

    if (dst.col <= 0)
      error->all(FLERR,
        "fix thermal_force: compute column must be >= 1");
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
    "fix thermal_force: source must be c_ID[col] or p_name");
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::refresh_compute_src(CollGridSrc &S)
{
  if (S.kind == COLL_SRC_PCUSTOM) {
    // Always re-fetch: particle->edvec[] is reallocated by particle->grow()
    // when new particles are created mid-step, which can happen between two
    // calls to refresh in the same timestep.
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

double FixThermalForce::read_src(const CollGridSrc &S, int ip, int icell) const
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

void FixThermalForce::particle_rz(const Particle::OnePart &p,
                                  double &R, double &Z) const
{
  OpenEdge::sparta_to_RZ(p.x, domain->dimension, domain->axisymmetric, R, Z);
}

/* ---------------------------------------------------------------------- */

double FixThermalForce::pd_interp(const std::vector<double> &field,
                                  const Particle::OnePart &p) const
{
  if (!pd_) return 0.0;
  double R, Z;
  particle_rz(p, R, Z);
  return pd_->interp2D(field, R, Z, p.icell);
}

/* ---------------------------------------------------------------------- */

double FixThermalForce::pd_grad(const std::vector<double> &mesh_grad,
                                const std::vector<double> &regular_grad,
                                const Particle::OnePart &p) const
{
  if (!pd_) return 0.0;
  double R, Z;
  particle_rz(p, R, Z);
  // Prefer mesh-based gradient (precomputed by the converter on the B2
  // (ix, iy) grid with Jacobian projection to physical (R, Z)). Falls
  // back to bilinear interp on the regular (R, Z) grid when mesh
  // gradients are absent (old plasma.h5 without mesh/grad_*_r/z).
  if (!mesh_grad.empty() && pd_->has_mesh) {
    // O(1) cell-indexed lookup when the cache is built; otherwise the
    // hash-grid triangle search.
    int cell = -1;
    if (p.icell >= 0 && p.icell < static_cast<int>(pd_->cell_mesh_cell.size()))
      cell = pd_->cell_mesh_cell[p.icell];
    else
      cell = pd_->mesh_cell_at(R, Z);
    if (cell >= 0 && cell < static_cast<int>(mesh_grad.size()))
      return mesh_grad[cell];
  }
  if (!regular_grad.empty())
    return pd_->interp2D(regular_grad, R, Z, p.icell);
  return 0.0;
}

/* ---------------------------------------------------------------------- */

void FixThermalForce::pd_bfield_sparta(const Particle::OnePart &p,
                                       double &B0, double &B1, double &B2) const
{
  B0 = B1 = B2 = 0.0;
  if (!pd_ || !pd_->has_bfield) return;

  double R, Z;
  particle_rz(p, R, Z);

  double Br = 0.0, Bz = 0.0, Bt = 0.0;
  pd_->bfield_at(R, Z, Br, Bz, Bt, p.icell);

  // Decompose physical (Br, Bz, Bt) onto SPARTA's (B0, B1, B2) slot layout
  // using the same convention as the helper:
  //   2D Cart  : x=R, y=Z, z=phi  -> B0=Br, B1=Bz, B2=Bt
  //   2D axi   : x=Z, y=R, z=phi  -> B0=Bz, B1=Br, B2=Bt
  //   3D       : (x,y) Cartesian rotation by phi
  double phi = 0.0;
  if (domain->dimension == 3) {
    phi = std::atan2(p.x[1], p.x[0]);
  }
  OpenEdge::RZphi_force_to_sparta(Br, Bz, Bt, domain->dimension,
                                   domain->axisymmetric, phi, B0, B1, B2);
}
