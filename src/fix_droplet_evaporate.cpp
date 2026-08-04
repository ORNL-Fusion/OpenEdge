/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    Built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "fix_droplet_evaporate.h"
#include "fix_background.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "input.h"
#include "modify.h"
#include "fix.h"
#include "math_const.h"
#include "domain.h"
#include "openedge_geom.h"
#include "mixture.h"
#include "grain_material.h"
#include "random_knuth.h"
#include "random_mars.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

using namespace SPARTA_NS;
using namespace MathConst;

namespace {
  // Sample a Poisson-distributed integer with mean lam.
  // Knuth direct method for small lam; Gaussian approx for large lam.
  inline int sample_poisson(double lam, SPARTA_NS::RanKnuth *rng) {
    if (lam <= 0.0) return 0;
    if (lam < 30.0) {
      const double L = std::exp(-lam);
      int k = 0;
      double p = 1.0;
      while (true) {
        ++k;
        p *= rng->uniform();
        if (p <= L) return k - 1;
      }
    }
    // Gaussian approximation: Box-Muller.
    const double u1 = std::max(rng->uniform(), 1.0e-300);
    const double u2 = rng->uniform();
    const double z  = std::sqrt(-2.0 * std::log(u1)) *
                      std::cos(2.0 * M_PI * u2);
    long long n = static_cast<long long>(std::floor(lam + std::sqrt(lam) * z + 0.5));
    if (n < 0) n = 0;
    return static_cast<int>(n);
  }
}

/* ---------------------------------------------------------------------- */

FixDropletEvaporate::FixDropletEvaporate(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg),
  heatflux_scale(1.0),
  rocket_eta(0.0),
  mat_(nullptr),
  twall_K_(300.0),
  heating_mode_(0),
  ion_mass_amu_(2.0),
  dq_custom_(-1),
  nw_custom_(-1),
  evap_atoms_local_(0.0),
  pd_(nullptr),
  emit_imix(-1),
  random(nullptr)
{
  strcpy(mat_name_, "Li");
  scalar_flag = 1;
  global_freq = 1;

  // fix ID evaporation Nevery MIXTURE background PD [keywords...]
  if (narg < 6)
    error->all(FLERR,
      "Illegal fix evaporation command "
      "(need: Nevery MIXTURE background PD)");

  nevery = atoi(arg[2]);
  imix   = particle->find_mixture(arg[3]);
  if (imix < 0) error->all(FLERR,"Fix evaporation: unknown mixture ID");

  if (strcmp(arg[4], "background") != 0)
    error->all(FLERR,
      "Fix evaporation: argument 5 must be 'background'");
  plasma_fix_id_ = std::string(arg[5]);

  int i = 6;
  while (i < narg) {
    if (strcmp(arg[i], "heatflux/scale") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heatflux/scale'");
      heatflux_scale = atof(arg[i+1]);
      if (!std::isfinite(heatflux_scale) || heatflux_scale < 0.0)
        error->all(FLERR,"Fix evaporation: heatflux/scale must be finite and >= 0");
      i += 2;
    } else if (strcmp(arg[i], "rocket_eta") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'rocket_eta'");
      rocket_eta = atof(arg[i+1]);
      if (rocket_eta < 0.0 || rocket_eta > 1.0)
        error->all(FLERR,"Fix evaporation: rocket_eta must be in [0,1]");
      i += 2;
    } else if (strcmp(arg[i], "emit_into") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'emit_into'");
      emit_imix = particle->find_mixture(arg[i+1]);
      if (emit_imix < 0)
        error->all(FLERR,"Fix evaporation: unknown emit_into mixture ID");
      i += 2;
    } else if (strcmp(arg[i], "material") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'material'");
      if (strlen(arg[i+1]) >= sizeof(mat_name_))
        error->all(FLERR,"Fix evaporation: material name too long");
      strcpy(mat_name_, arg[i+1]);
      i += 2;
    } else if (strcmp(arg[i], "twall_K") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'twall_K'");
      twall_K_ = atof(arg[i+1]);
      if (!std::isfinite(twall_K_) || twall_K_ < 0.0)
        error->all(FLERR,"Fix evaporation: twall_K must be finite and >= 0");
      i += 2;
    } else if (strcmp(arg[i], "heating") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heating'");
      if      (strcmp(arg[i+1], "flux") == 0) heating_mode_ = 0;
      else if (strcmp(arg[i+1], "oml") == 0)  heating_mode_ = 1;
      else error->all(FLERR,"Fix evaporation: heating must be 'flux' or 'oml'");
      i += 2;
    } else if (strcmp(arg[i], "ion_mass_amu") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'ion_mass_amu'");
      ion_mass_amu_ = atof(arg[i+1]);
      if (!std::isfinite(ion_mass_amu_) || ion_mass_amu_ <= 0.0)
        error->all(FLERR,"Fix evaporation: ion_mass_amu must be > 0");
      i += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Fix evaporation: unknown keyword '%s'", arg[i]);
      error->all(FLERR, msg);
    }
  }

  if (emit_imix >= 0) {
    random = new RanKnuth(update->ranmaster->uniform());
    double seed = comm->me + 1;
    random->reset(seed, comm->me, 100);
  }
}

/* ---------------------------------------------------------------------- */

FixDropletEvaporate::~FixDropletEvaporate()
{
  delete random;
}

/* ---------------------------------------------------------------------- */

int FixDropletEvaporate::setmask()
{
  int mask = 0;
  mask |= START_OF_STEP;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::init()
{
  if (domain->dimension != 2)
    error->all(FLERR,"Fix evaporation: only 2D geometry supported");

  const int ifix = modify->find_fix(plasma_fix_id_.c_str());
  if (ifix < 0) {
    char msg[200];
    snprintf(msg, sizeof(msg),
             "Fix evaporation: background fix '%s' not found",
             plasma_fix_id_.c_str());
    error->all(FLERR, msg);
  }
  pd_ = dynamic_cast<FixBackground *>(modify->fix[ifix]);
  if (!pd_)
    error->all(FLERR,
      "Fix evaporation: background fix must be style background");
  pd_->init();

  // Resolve the grain material (registry + `material` command overrides).
  mat_ = grain_material_find(mat_name_);
  if (!mat_) {
    char msg[128];
    snprintf(msg, sizeof(msg),
             "Fix evaporation: unknown material '%s'", mat_name_);
    error->all(FLERR, msg);
  }
  if (mat_->rho <= 0.0 || mat_->cp <= 0.0 || mat_->mass_amu <= 0.0 ||
      mat_->hvap_J_mol <= 0.0 || mat_->antoine_b >= 0.0)
    error->all(FLERR,
      "Fix evaporation: material is missing rho/cp/mass_amu/hvap_J_mol/"
      "antoine coefficients (define them with the material command)");

  // OML heating uses the grain's OML charge when fix droplet/charge is
  // active (custom vector registered by that fix); else chi defaults.
  dq_custom_ = particle->find_custom((char *) "droplet_charge");
  nw_custom_ = particle->find_custom((char *) "grain_nweight");
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::start_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

void FixDropletEvaporate::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;
  evap_half(0.5 * update->dt);
}

double FixDropletEvaporate::memory_usage() { return 0.0; }

double FixDropletEvaporate::compute_scalar()
{
  double global = 0.0;
  MPI_Allreduce(&evap_atoms_local_, &global, 1, MPI_DOUBLE, MPI_SUM, world);
  return global;
}

/* ---------------------------------------------------------------------- */

void FixDropletEvaporate::evap_half(double dt_half)
{
  if ((update->ntimestep % nevery) != 0) return;

  // Snapshot nlocal before the loop so we don't process atoms spawned by
  // emit_into during this same call.
  const int nlocal = particle->nlocal;
  int *s2g = particle->mixture[imix]->species2group;
  int ndeleted = 0;

  for (int ip = 0; ip < nlocal; ip++) {
    // Refetch pointer each iter — emit_into may have realloc'd particles.
    Particle::OnePart *parts = particle->particles;
    const int is = parts[ip].ispecies;
    const int ig = s2g[is];
    if (ig < 0) continue;

    droplet_evaporation_model(ip, dt_half);

    parts = particle->particles;     // refresh in case spawn_evap reallocated
    const double m_cut = 0.1 * particle->species[is].mass;
    if (parts[ip].mass > 0.0 && parts[ip].mass <= m_cut) {
      parts[ip].mass   = 0.0;
      parts[ip].radius = 0.0;
      parts[ip].temp   = 0.0;
      parts[ip].icell  = -1;
      ndeleted++;
    }
  }

  if (ndeleted > 0) particle->compress_rebalance();
}

/* ----------------------------------------------------------------------
   Sergey's droplet evaporation model:
     dR/dt = -(m_atom / rho) * Gevap
     dT/dt = (3 / (rho * Cp * R)) * (Qs - Gevap * DHm / N_A)
   where Qs = sqrt(q_par^2 + q_perp^2) at the droplet position, pulled
   from fix background. Rocket force uses -grad(Te) as the recoil axis.
------------------------------------------------------------------------- */
void FixDropletEvaporate::droplet_evaporation_model(int idrop,
                                        const double dt_half)
{
  // Material properties from the grain-material registry (default Li,
  // selectable per fix with `material NAME`, tunable with the material
  // command). Values resolved/validated in init().
  const double AM   = mat_->mass_amu * 1.66053906660e-27;  // atom mass [kg]
  const double Rho  = mat_->rho;          // kg/m^3
  const double DHm  = mat_->hvap_J_mol;   // heat of vaporization [J/mol]
  const double Eps  = mat_->emissivity;   // total emissivity [-]
  const double AN   = 6.022e+23;          // 1/mol
  const double SB   = 5.670374419e-8;     // Stefan-Boltzmann [W/m^2/K^4]
  const double Tw4  = twall_K_*twall_K_*twall_K_*twall_K_;
  const double DT   = dt_half;

  // Snapshot droplet state. After spawn_evap_atoms() the Particle::particles
  // array may realloc — never read through ip after that. We refresh the
  // pointer at the end and write final state via index.
  Particle::OnePart *ip = &particle->particles[idrop];
  const double mass   = ip->mass;
  const double radius = ip->radius;
  const double TK     = ip->temp;
  const double xs[3]  = {ip->x[0], ip->x[1], ip->x[2]};
  const int icell_ip  = ip->icell;

  // R/Z at particle position (handles 2D Cart, 2D axi, 3D Cart via helper).
  double R = 0.0, Z = 0.0;
  OpenEdge::sparta_to_RZ(xs, domain->dimension, domain->axisymmetric, R, Z,
                         pd_->column_x0, pd_->column_y0);

  double Qs = 0.0;
  if (heating_mode_ == 0) {
    // Legacy mode: prescribed heat-flux field from plasma.h5. When the
    // file carries q_par/q_perp, interp2D routes to the mesh-native
    // counterpart; otherwise the fix-background defaults apply.
    double q_par  = pd_->default_q_par;
    double q_perp = pd_->default_q_perp;
    if (pd_->has_qheatflux) {
      if (!pd_->mesh_q_par.empty() || !pd_->q_par.empty())
        q_par  = pd_->interp2D(pd_->q_par,  R, Z, ip->icell);
      if (!pd_->mesh_q_perp.empty() || !pd_->q_perp.empty())
        q_perp = pd_->interp2D(pd_->q_perp, R, Z, ip->icell);
    }
    // A sphere in a directional flux intercepts q*piR^2 = (1/4) q * 4piR^2;
    // the model applies Qs over the full surface, so bake the 1/4 in here.
    Qs = 0.25 * std::sqrt(q_par*q_par + q_perp*q_perp);
  } else {
    // OML mode: electron/ion collection heating from LOCAL ne/Te/Ti
    // (DUSTT Eqs. 26-27 style, simplified). This is the physically
    // correct volumetric heating for a grain immersed in plasma — q_par
    // is a field-line target flux, near zero across most of the SOL
    // volume where grains actually fly.
    const double QE = 1.602176634e-19;
    const double ME = 9.1093837015e-31;
    const double MI = ion_mass_amu_ * 1.66053906660e-27;
    const double Te = std::max(pd_->interp2D(pd_->temp_e, R, Z, ip->icell), 0.0);
    const double Ti = std::max(pd_->interp2D(pd_->temp_i, R, Z, ip->icell), 0.0);
    const double ne = std::max(pd_->interp2D(pd_->dens_e, R, Z, ip->icell), 0.0);
    const double ni = pd_->dens_i.empty()
      ? ne : std::max(pd_->interp2D(pd_->dens_i, R, Z, ip->icell), 0.0);
    if (ne > 0.0 && Te > 0.0) {
      // Normalized floating potential chi = e*phi/Te (<= 0). Use the OML
      // charge from fix droplet/charge when present; else the canonical
      // hydrogenic floating value chi ~ -2.5.
      double chi = -2.5;
      if (dq_custom_ >= 0) {
        const int ew = particle->ewhich[dq_custom_];
        if (ew >= 0) {
          const double Zd = particle->edvec[ew][idrop];
          const double r_c = (radius > 1.0e-9) ? radius : 1.0e-9;
          const double phi_V = Zd * QE / (4.0 * MY_PI * 8.8541878128e-12 * r_c);
          if (std::isfinite(phi_V) && phi_V < 0.0 && Te > 0.0)
            chi = std::max(phi_V / Te, -20.0);
        }
      }
      // One-sided thermal fluxes (per unit grain surface).
      const double Ge = 0.25 * ne * std::sqrt(8.0*QE*Te/(MY_PI*ME))
                        * std::exp(chi);                       // repelled e-
      const double Gi = (Ti > 0.0)
        ? 0.25 * ni * std::sqrt(8.0*QE*Ti/(MY_PI*MI)) * (1.0 - chi*Te/std::max(Ti,1e-30))
        : 0.0;                                                 // OML-attracted ions
      // Sheath-transmitted energies (zeta_e = zeta_i = 2.5, DUSTT) plus
      // ion surface-recombination energy (hydrogenic 13.6 eV).
      const double Ee = 2.5 * Te;
      const double Ei = 2.5 * std::max(Ti, 0.0) + (-chi) * Te + 13.6;
      Qs = QE * (Ge * Ee + Gi * Ei);
    }
  }
  if (!std::isfinite(Qs) || Qs < 0.0) Qs = 0.0;
  Qs *= heatflux_scale;

  // grad_Te at particle position (for rocket-force direction).
  double gTeR = 0.0, gTeZ = 0.0;
  if (rocket_eta > 0.0) {
    if (!pd_->mesh_grad_te_r.empty() || !pd_->grad_te_r.empty())
      gTeR = pd_->interp2D(pd_->grad_te_r, R, Z, ip->icell);
    if (!pd_->mesh_grad_te_z.empty() || !pd_->grad_te_z.empty())
      gTeZ = pd_->interp2D(pd_->grad_te_z, R, Z, ip->icell);
  }

  // NOTE: no early-return on Qs <= 0 — a grain in vacuum (real q_par is 0
  // outside the SOLPS mesh) still evaporates and cools radiatively.

  const double a1 = mat_->antoine_a, b1 = mat_->antoine_b;
  const double xm1 = mat_->mass_amu;
  const double max_dT = 25.0;       // K, local explicit substep limiter
  const double max_dR_frac = 0.02;  // limit radius loss in one substep
  const int max_substeps = 10000;

  double R_new = radius;
  double T_new = TK;
  double evap_atoms_step = 0.0;
  double t_left = DT;
  int nsub = 0;

  while (t_left > 0.0 && R_new > 0.0) {
    if (T_new <= 0.0 || !std::isfinite(T_new))
      error->one(FLERR,"Fix evaporation: invalid particle temperature");

    // Antoine + Hertz-Knudsen flux.
    const double vpres1 = 760.0 * std::pow(10.0, a1 + b1/T_new);      // mmHg
    const double Gevap_atoms =
      1.0e4 * 3.513e22 * vpres1 / std::sqrt(xm1 * T_new);
    if (!std::isfinite(Gevap_atoms) || Gevap_atoms < 0.0)
      error->one(FLERR,"Fix evaporation: invalid evaporation flux");

    const double dRdt = -AM * Gevap_atoms / Rho;
    // Energy balance per unit surface: plasma heating - radiative cooling
    // (emissivity * sigma * (T^4 - Twall^4)) - evaporative (latent) cooling.
    // Radiation is negligible for liquid Li (~kW/m^2 at 1000 K, eps~0.1)
    // but is a LEADING term for boron (~MW/m^2 at 2300 K, eps~0.8).
    const double q_rad = Eps * SB * (T_new*T_new*T_new*T_new - Tw4);
    const double HF   = Qs - q_rad - Gevap_atoms * (DHm / AN);
    const double r_safe = (R_new > 1.0e-20) ? R_new : 1.0e-20;
    // Melting as apparent heat capacity: spread the latent heat of fusion
    // over a +-dTm band around Tmelt so the grain pauses there while
    // absorbing/releasing h_melt. No per-particle melt-fraction state
    // needed; adequate for micron grains whose h_melt << h_vap.
    double Cp_eff = grain_material_cp(mat_, T_new);   // solid vs liquid cp
    if (mat_->tmelt_K > 0.0 && mat_->hmelt_J_mol > 0.0) {
      const double dTm = 20.0;   // K, half-width of the melting band
      if (std::fabs(T_new - mat_->tmelt_K) < dTm) {
        const double hmelt_J_kg = mat_->hmelt_J_mol
                                  / (mat_->mass_amu * 1.0e-3);
        Cp_eff += hmelt_J_kg / (2.0 * dTm);
      }
    }
    const double dTdt = (3.0 / (Rho * Cp_eff * r_safe)) * HF;
    if (!std::isfinite(dRdt) || !std::isfinite(dTdt))
      error->one(FLERR,"Fix evaporation: invalid evaporation rate");

    double dt_sub = t_left;
    if (dTdt != 0.0 && std::isfinite(dTdt)) {
      const double dT_limit = (dTdt < 0.0)
                            ? std::min(max_dT, 0.5 * T_new)
                            : max_dT;
      const double dt_T = dT_limit / std::fabs(dTdt);
      if (dt_T > 0.0 && dt_T < dt_sub) dt_sub = dt_T;
    }
    if (dRdt < 0.0 && std::isfinite(dRdt)) {
      const double dt_R = max_dR_frac * R_new / (-dRdt);
      if (dt_R > 0.0 && dt_R < dt_sub) dt_sub = dt_R;
    }
    if (dt_sub <= 0.0 || !std::isfinite(dt_sub))
      error->one(FLERR,"Fix evaporation: invalid adaptive timestep");

    evap_atoms_step += 4.0 * MY_PI * R_new * R_new * Gevap_atoms * dt_sub;
    R_new = std::max(0.0, R_new + dRdt * dt_sub);
    T_new += dTdt * dt_sub;
    t_left -= dt_sub;

    if (++nsub > max_substeps)
      error->one(FLERR,"Fix evaporation: adaptive timestep exceeded limit");
  }

  const double m_new   = (R_new > 0.0)
                         ? (Rho * (4.0/3.0) * MY_PI * R_new*R_new*R_new)
                         : 0.0;
  if (!std::isfinite(T_new) || T_new < 0.0)
    error->one(FLERR,"Fix evaporation: particle temperature dropped below 0 K");

  // grain_nweight: one macro-grain represents nw real grains (set by
  // grain/emit nweight). Scales the atom tally and the vapor source;
  // per-grain trajectory/heating physics is unscaled.
  double nw = 1.0;
  if (nw_custom_ >= 0) {
    const int ew = particle->ewhich[nw_custom_];
    if (ew >= 0) {
      const double v = particle->edvec[ew][idrop];
      if (v > 0.0) nw = v;
    }
  }
  evap_atoms_local_ += evap_atoms_step * nw;

  // Rocket-force kick (in-place velocity update via fresh pointer).
  if (rocket_eta > 0.0 && m_new > 0.0 && evap_atoms_step > 0.0) {
    const double grad_mag = std::sqrt(gTeR*gTeR + gTeZ*gTeZ);
    if (std::isfinite(grad_mag) && grad_mag > 0.0) {
      const double kB = 1.380649e-23;
      const double T_rocket = std::max(1.0, 0.5 * (TK + T_new));
      const double v_thermal = std::sqrt(8.0 * kB * T_rocket / (MY_PI * AM));
      const double dmdt = evap_atoms_step * AM / DT;
      const double a_mag = rocket_eta * dmdt * v_thermal / m_new;
      const double nr = -gTeR / grad_mag;
      const double nz = -gTeZ / grad_mag;

      double phi = 0.0;
      if (domain->dimension == 3) phi = std::atan2(xs[1], xs[0]);
      double dvx, dvy, dvz;
      OpenEdge::RZphi_force_to_sparta(a_mag * nr, a_mag * nz, 0.0,
                                       domain->dimension, domain->axisymmetric,
                                       phi, dvx, dvy, dvz);
      Particle::OnePart *ip_w = &particle->particles[idrop];
      ip_w->v[0] += dvx * DT;
      ip_w->v[1] += dvy * DT;
      ip_w->v[2] += dvz * DT;
    }
  }

  // Volumetric Li source: spawn evaporated atoms in the droplet's cell.
  // Must precede the final ip-> write (spawn may realloc the particle array).
  if (emit_imix >= 0 && evap_atoms_step > 0.0 && radius > 0.0) {
    const double area = 4.0 * MY_PI * radius * radius;
    const double Gevap_emit = nw * evap_atoms_step / (area * DT);
    const double T_emit = std::max(1.0, 0.5 * (TK + T_new));
    spawn_evap_atoms(idrop, area, Gevap_emit, T_emit, dt_half);
  }

  // Final state write through a fresh pointer (any spawn above may have
  // invalidated earlier pointers via Particle::particles realloc).
  Particle::OnePart *ip_w = &particle->particles[idrop];
  ip_w->radius = R_new;
  ip_w->temp   = T_new;
  ip_w->mass   = m_new;
}

/* ----------------------------------------------------------------------
   Spawn evaporated atoms in the droplet's cell. Called only when
   emit_imix >= 0. Lambda = area * Gevap_atoms * dt / fnum.
------------------------------------------------------------------------- */
void FixDropletEvaporate::spawn_evap_atoms(int idrop, double area,
                                            double Gevap_atoms, double TK,
                                            double dt_half)
{
  const double fnum = update->fnum;
  if (fnum <= 0.0) return;
  const double dN_phys = area * Gevap_atoms * dt_half;       // atoms / call
  const double lam     = dN_phys / fnum;                     // sim particles
  if (lam <= 0.0 || !std::isfinite(lam)) return;

  const int n_to_emit = sample_poisson(lam, random);
  if (n_to_emit == 0) return;

  // Pick species inside emit mixture by fraction; cumulative CDF.
  Mixture *mix = particle->mixture[emit_imix];
  const int    nsp_mix = mix->nspecies;
  const double *frac   = mix->fraction;
  const int    *spec   = mix->species;

  // Snapshot droplet position / cell — pointer may invalidate after first add.
  Particle::OnePart *ip_snap = &particle->particles[idrop];
  const double xs[3] = {ip_snap->x[0], ip_snap->x[1], ip_snap->x[2]};
  const int icell_ip = ip_snap->icell;
  if (icell_ip < 0) return;

  const double kB = 1.380649e-23;

  for (int k = 0; k < n_to_emit; ++k) {
    // Pick species via cumulative fraction.
    int isp = spec[0];
    if (nsp_mix > 1) {
      const double u = random->uniform();
      double cum = 0.0;
      for (int s = 0; s < nsp_mix; ++s) {
        cum += frac[s];
        if (u <= cum) { isp = spec[s]; break; }
      }
    }
    const double m_atom = particle->species[isp].mass;
    if (m_atom <= 0.0) continue;

    // 3D Maxwellian velocity at droplet T (sigma = sqrt(kT/m) per component).
    const double sigma = std::sqrt(kB * TK / m_atom);
    double u1 = std::max(random->uniform(), 1.0e-300);
    double u2 = random->uniform();
    double u3 = std::max(random->uniform(), 1.0e-300);
    double u4 = random->uniform();
    const double r12 = std::sqrt(-2.0 * std::log(u1));
    const double r34 = std::sqrt(-2.0 * std::log(u3));
    double v[3];
    v[0] = sigma * r12 * std::cos(2.0 * MY_PI * u2);
    v[1] = sigma * r12 * std::sin(2.0 * MY_PI * u2);
    v[2] = sigma * r34 * std::cos(2.0 * MY_PI * u4);

    double x[3] = {xs[0], xs[1], xs[2]};
    int newid = MAXSMALLINT * random->uniform();
    particle->add_particle(newid, isp, icell_ip, x, v, 0.0, 0.0);

    // Notify update_custom subscribers so per-particle attributes (pweight
    // for fix particle/weight, etc.) are initialized on the new vapor
    // particle — without this, grid/weighted tallies see zero weight for
    // all evaporated (and subsequently ionised) atoms.
    if (modify->n_update_custom) {
      double zero_v[3] = {0.0, 0.0, 0.0};
      modify->update_custom(particle->nlocal - 1, 0.0, 0.0, 0.0, zero_v);
    }
  }
}
