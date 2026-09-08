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
#include "fix_droplet_charge.h"
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
  ion_charge_state_(1.0),
  physics_model_(ParticulateModel::DUSTT2005),
  termination_mode_(TERMINATION_DUSTT),
  termination_explicit_(0),
  dis_mass_tol_(1.0e-12),
  dq_custom_(-1),
  nw_custom_(-1),
  heating_q_custom_(-1),
  debye_ratio_custom_(-1),
  oml_weight_custom_(-1),
  dis_q_ion_custom_(-1),
  dis_q_electron_custom_(-1),
  dis_q_neutralization_custom_(-1),
  dis_q_sheath_custom_(-1),
  dis_q_thermionic_custom_(-1),
  dis_q_secondary_custom_(-1),
  evap_atoms_local_(0.0),
  terminal_remainder_atoms_local_(0.0),
  pd_(nullptr),
  charge_fix_(nullptr),
  emit_imix(-1),
  random(nullptr)
{
  strcpy(mat_name_, "Li");
  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 2;
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
    if (strcmp(arg[i], "model") == 0) {
      if (i+1 >= narg)
        error->all(FLERR,"Fix particulate/thermal: model needs a value");
      if (strcmp(arg[i+1], "dustt2005") == 0 ||
          strcmp(arg[i+1], "dustt") == 0)
        physics_model_ = ParticulateModel::DUSTT2005;
      else if (strcmp(arg[i+1], "dis2021") == 0 ||
               strcmp(arg[i+1], "dis") == 0)
        physics_model_ = ParticulateModel::DIS2021;
      else
        error->all(FLERR,
          "Fix particulate/thermal: model must be dustt2005 or dis2021");
      i += 2;
    } else if (strcmp(arg[i], "heatflux/scale") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'heatflux/scale'");
      heatflux_scale = atof(arg[i+1]);
      if (!std::isfinite(heatflux_scale) || heatflux_scale < 0.0)
        error->all(FLERR,"Fix evaporation: heatflux/scale must be finite and >= 0");
      i += 2;
    } else if (strcmp(arg[i], "alpha_e") == 0) {
      // evaporation/accommodation coefficient on the Hertz-Knudsen flux
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'alpha_e'");
      alpha_e_ = atof(arg[i+1]);
      if (!std::isfinite(alpha_e_) || alpha_e_ <= 0.0 || alpha_e_ > 1.0)
        error->all(FLERR,"Fix evaporation: alpha_e must be in (0,1]");
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
      else if (strcmp(arg[i+1], "auto") == 0) heating_mode_ = 2;
      else error->all(FLERR,"Fix evaporation: heating must be 'flux', 'oml', or 'auto'");
      i += 2;
    } else if (strcmp(arg[i], "charge") == 0) {
      if (i+1 >= narg)
        error->all(FLERR,
          "Fix particulate/thermal: charge requires a fix ID");
      charge_fix_id_ = std::string(arg[i+1]);
      i += 2;
    } else if (strcmp(arg[i], "ion_mass_amu") == 0) {
      if (i+1 >= narg) error->all(FLERR,"Fix evaporation: missing value for 'ion_mass_amu'");
      ion_mass_amu_ = atof(arg[i+1]);
      if (!std::isfinite(ion_mass_amu_) || ion_mass_amu_ <= 0.0)
        error->all(FLERR,"Fix evaporation: ion_mass_amu must be > 0");
      i += 2;
    } else if (strcmp(arg[i], "ion_charge_state") == 0) {
      if (i+1 >= narg)
        error->all(FLERR,"Fix evaporation: missing ion_charge_state");
      ion_charge_state_ = atof(arg[i+1]);
      if (!std::isfinite(ion_charge_state_) || ion_charge_state_ <= 0.0)
        error->all(FLERR,"Fix evaporation: ion_charge_state must be > 0");
      i += 2;
    } else if (strcmp(arg[i], "termination") == 0) {
      if (i+1 >= narg)
        error->all(FLERR,"Fix evaporation: missing value for 'termination'");
      if (strcmp(arg[i+1], "dis") == 0)
        termination_mode_ = TERMINATION_DIS;
      else if (strcmp(arg[i+1], "dustt") == 0)
        termination_mode_ = TERMINATION_DUSTT;
      else
        error->all(FLERR,
          "Fix evaporation: termination must be 'dis' or 'dustt'");
      termination_explicit_ = 1;
      i += 2;
    } else if (strcmp(arg[i], "dis_mass_tol") == 0) {
      if (i+1 >= narg)
        error->all(FLERR,"Fix evaporation: missing value for 'dis_mass_tol'");
      dis_mass_tol_ = atof(arg[i+1]);
      if (!std::isfinite(dis_mass_tol_) || dis_mass_tol_ <= 0.0 ||
          dis_mass_tol_ >= 1.0)
        error->all(FLERR,
          "Fix evaporation: dis_mass_tol must be in (0,1)");
      i += 2;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "Fix evaporation: unknown keyword '%s'", arg[i]);
      error->all(FLERR, msg);
    }
  }

  // The canonical model owns its published termination rule.  Keeping an
  // explicit override is useful for verification, but ordinary decks need
  // only one coherent `model` selection.
  if (!termination_explicit_)
    termination_mode_ = physics_model_ == ParticulateModel::DIS2021
      ? TERMINATION_DIS : TERMINATION_DUSTT;

  if (emit_imix >= 0) {
    random = new RanKnuth(update->ranmaster->uniform());
    double seed = comm->me + 1;
    random->reset(seed, comm->me, 100);
  }

  heating_q_custom_ = particle->find_custom((char *) "droplet_heating_q");
  if (heating_q_custom_ < 0)
    heating_q_custom_ = particle->add_custom((char *) "droplet_heating_q", 1, 0);
  debye_ratio_custom_ = particle->find_custom((char *) "droplet_a_over_lambdaD");
  if (debye_ratio_custom_ < 0)
    debye_ratio_custom_ = particle->add_custom((char *) "droplet_a_over_lambdaD", 1, 0);
  oml_weight_custom_ = particle->find_custom((char *) "droplet_oml_weight");
  if (oml_weight_custom_ < 0)
    oml_weight_custom_ = particle->add_custom((char *) "droplet_oml_weight", 1, 0);
  if (physics_model_ == ParticulateModel::DIS2021) {
    auto add_double_custom = [&](const char *name) -> int {
      int index = particle->find_custom((char *) name);
      if (index < 0) index = particle->add_custom((char *) name, 1, 0);
      return index;
    };
    dis_q_ion_custom_ = add_double_custom("particulate_dis_q_ion");
    dis_q_electron_custom_ = add_double_custom("particulate_dis_q_electron");
    dis_q_neutralization_custom_ =
      add_double_custom("particulate_dis_q_neutralization");
    dis_q_sheath_custom_ = add_double_custom("particulate_dis_q_sheath");
    dis_q_thermionic_custom_ =
      add_double_custom("particulate_dis_q_thermionic");
    dis_q_secondary_custom_ =
      add_double_custom("particulate_dis_q_secondary");
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
  if (heating_mode_ != 1 && !pd_->has_qheatflux)
    error->all(FLERR,
      "Fix particulate/thermal: heating flux/auto requires physical "
      "q_par/q_perp datasets in the plasma background; no heat-flux "
      "fallback is available (use 'heating oml' only when intended)");

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

  if (physics_model_ == ParticulateModel::DIS2021) {
    if (!mat_->provenance_id[0])
      error->all(FLERR,
        "fix particulate/thermal model dis2021 requires a material command "
        "with an explicit provenance ID");
    const std::uint64_t required =
      GRAIN_MAT_RHO | GRAIN_MAT_CP | GRAIN_MAT_CP_SOLID |
      GRAIN_MAT_MASS_AMU | GRAIN_MAT_HVAP | GRAIN_MAT_ANTOINE_A |
      GRAIN_MAT_ANTOINE_B | GRAIN_MAT_EMISSIVITY | GRAIN_MAT_TMELT |
      GRAIN_MAT_HMELT;
    char missing[256];
    if (grain_material_missing_properties(mat_, required,
                                           missing, sizeof(missing))) {
      char msg[512];
      snprintf(msg, sizeof(msg),
        "fix particulate/thermal model dis2021: material '%s' provenance "
        "'%s' did not explicitly supply active properties: %s",
        mat_->name, mat_->provenance_id, missing);
      error->all(FLERR, msg);
    }
    if (!std::isfinite(mat_->cp_solid) || mat_->cp_solid < 0.0 ||
        !std::isfinite(mat_->antoine_a) ||
        !std::isfinite(mat_->emissivity) || mat_->emissivity < 0.0 ||
        mat_->emissivity > 1.0 || !(mat_->tmelt_K > 0.0) ||
        !std::isfinite(mat_->hmelt_J_mol) || mat_->hmelt_J_mol < 0.0)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: invalid cp_solid, "
        "Antoine, emissivity, melting temperature, or fusion enthalpy");

    // A grain species stores its physical initial mass and radius. Catch a
    // diameter/radius or density mismatch before advancing trajectories.
    Mixture *mix = particle->mixture[imix];
    for (int i = 0; i < mix->nspecies; ++i) {
      const int ispecies = mix->species[i];
      const Particle::Species &s = particle->species[ispecies];
      if (!(s.radius > 0.0) || !(s.mass > 0.0))
        error->all(FLERR,
          "fix particulate/thermal model dis2021: every grain species "
          "must define positive initial radius and mass");
      const double expected = (4.0/3.0) * MY_PI * mat_->rho *
                              s.radius * s.radius * s.radius;
      const double rel = std::fabs(s.mass - expected) / expected;
      if (!std::isfinite(rel) || rel > 1.0e-6) {
        char msg[384];
        snprintf(msg, sizeof(msg),
          "fix particulate/thermal model dis2021: species '%s' mass/radius "
          "is inconsistent with material '%s' density (mass=%g kg, "
          "expected=%g kg, relative error=%g)",
          s.id, mat_->name, s.mass, expected, rel);
        error->all(FLERR, msg);
      }
    }
  }

  // OML heating uses the grain's OML charge when fix droplet/charge is
  // active (custom vector registered by that fix); else chi defaults.
  dq_custom_ = particle->find_custom((char *) "particulate_charge");
  if (physics_model_ == ParticulateModel::DIS2021 &&
      heating_mode_ != 0 && dq_custom_ < 0)
    error->all(FLERR,
      "fix particulate/thermal model dis2021 with OML heating requires "
      "fix particulate/charge; DIS does not use a hardcoded floating "
      "potential");
  if (physics_model_ == ParticulateModel::DIS2021 && heating_mode_ != 0) {
    if (charge_fix_id_.empty())
      error->all(FLERR,
        "fix particulate/thermal model dis2021 with OML heating requires "
        "'charge FIXID' so emission currents and cooling use one closure");
    const int iqfix = modify->find_fix(charge_fix_id_.c_str());
    if (iqfix < 0)
      error->all(FLERR,
        "fix particulate/thermal: referenced charge fix was not found");
    charge_fix_ = dynamic_cast<FixDropletCharge *>(modify->fix[iqfix]);
    if (!charge_fix_ ||
        charge_fix_->physics_model() != ParticulateModel::DIS2021)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: charge FIXID must name "
        "a particulate/charge fix using model dis2021");
    const int ithfix = modify->find_fix(id);
    if (ithfix < 0 || iqfix >= ithfix)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: the referenced charge "
        "fix must be defined before the thermal fix");
    if (charge_fix_->background_fix_id() != plasma_fix_id_)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: charge and thermal fixes "
        "must reference the same background fix");
    if (charge_fix_->grain_material() != mat_)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: charge and thermal fixes "
        "must use the same material definition");
    const double mass_rel = std::fabs(charge_fix_->ion_mass_amu_value()
                                      - ion_mass_amu_) / ion_mass_amu_;
    const double charge_rel = std::fabs(charge_fix_->ion_charge_state_value()
                                        - ion_charge_state_) / ion_charge_state_;
    if (mass_rel > 1.0e-12 || charge_rel > 1.0e-12)
      error->all(FLERR,
        "fix particulate/thermal model dis2021: ion_mass_amu and "
        "ion_charge_state must match the referenced charge fix");
  }
  if (physics_model_ == ParticulateModel::DIS2021 && comm->me == 0 && screen)
    fprintf(screen,
      "particulate/thermal dis2021 material=%s provenance=%s "
      "explicit_mask=0x%llx\n", mat_->name, mat_->provenance_id,
      static_cast<unsigned long long>(mat_->explicit_mask));
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

/* ----------------------------------------------------------------------
   Vector diagnostics:
     [1] cumulative continuously evaporated material atoms
     [2] cumulative DUSTT terminal remainder, in atom equivalents
------------------------------------------------------------------------- */

double FixDropletEvaporate::compute_vector(int index)
{
  double local = 0.0;
  if (index == 0) local = evap_atoms_local_;
  else if (index == 1) local = terminal_remainder_atoms_local_;
  double global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, world);
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
    if (parts[ip].mass <= 0.0 || parts[ip].radius <= 0.0) {
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
  const int ispecies  = ip->ispecies;

  const double radius0 = particle->species[ispecies].radius;
  if (!std::isfinite(radius0) || radius0 <= 0.0)
    error->one(FLERR,
      "Fix evaporation: grain species must define a positive initial radius");
  const double terminal_radius = termination_mode_ == TERMINATION_DUSTT
    ? 0.1 * radius0
    : std::cbrt(dis_mass_tol_) * radius0;

  unsigned request = PLASMA_NEED_THERMO;
  if (heating_mode_ != 1) request |= PLASMA_NEED_HEAT;
  if (rocket_eta > 0.0) request |= PLASMA_NEED_GRAD_TE;
  if (physics_model_ == ParticulateModel::DIS2021 && heating_mode_ != 0)
    request |= PLASMA_NEED_FLOW_B;
  PlasmaPointSample plasma;
  pd_->sample_point(xs, plasma, ip->icell, idrop, request);

  const double QE = 1.602176634e-19;
  const double ME = 9.1093837015e-31;
  const double MI = ion_mass_amu_ * 1.66053906660e-27;
  const double EPS0 = 8.8541878128e-12;
  const double Te = std::max(plasma.te, 0.0);
  const double Ti = std::max(plasma.ti, 0.0);
  const double ne = std::max(plasma.ne, 0.0);
  const double ni = std::max(plasma.ni, 0.0);

  const double Qflux = (heating_mode_ != 1)
    ? 0.25 * std::sqrt(plasma.q_par*plasma.q_par +
                       plasma.q_perp*plasma.q_perp)
    : 0.0;

  double Qoml = 0.0;
  double dis_relative_mach = 0.0;
  if (heating_mode_ != 0 && ne > 0.0 && Te > 0.0) {
    if (physics_model_ == ParticulateModel::DIS2021) {
      if (!(Ti > 0.0) || !(ni > 0.0))
        error->one(FLERR,
          "fix particulate/thermal model dis2021: OML heating requires "
          "positive Ti and ni");
      const int ew = dq_custom_ >= 0 ? particle->ewhich[dq_custom_] : -1;
      if (ew < 0)
        error->one(FLERR,
          "fix particulate/thermal model dis2021: particulate_charge "
          "custom vector is unavailable");
      const double Zd = particle->edvec[ew][idrop];
      const double r_c = std::max(radius, 1.0e-12);
      const double phi_V = Zd * QE / (4.0 * MY_PI * EPS0 * r_c);
      if (!std::isfinite(phi_V))
        error->one(FLERR,
          "fix particulate/thermal model dis2021: non-finite grain potential");
      const double chi = -phi_V / Te;
      const double tau_i = Ti / Te;
      const double X = ion_charge_state_ * chi / tau_i;
      const double vti = std::sqrt(2.0 * QE * Ti / MI);
      const double dv0 = ip->v[0] - plasma.flow[0];
      const double dv1 = ip->v[1] - plasma.flow[1];
      const double dv2 = ip->v[2] - plasma.flow[2];
      const double U = std::sqrt(dv0*dv0 + dv1*dv1 + dv2*dv2) / vti;
      dis_relative_mach = U;

      ParticulateModel::DISCurrentState current;
      ParticulateModel::DISHeatFluxState heat;
      if (!charge_fix_ ||
          !charge_fix_->evaluate_dis_state(Te, Ti, ne, ni, TK, r_c, U,
                                           phi_V, current, &heat))
        error->one(FLERR,
          "fix particulate/thermal model dis2021: shared DIS current/heat "
          "evaluation failed");
      Qoml = heat.net_W_m2;

      auto store_dis_component = [&](int custom, double value) {
        if (custom < 0) return;
        const int index = particle->ewhich[custom];
        if (index >= 0) particle->edvec[index][idrop] = value;
      };
      store_dis_component(dis_q_ion_custom_, heat.ion_kinetic_W_m2);
      store_dis_component(dis_q_electron_custom_,
                          heat.electron_kinetic_W_m2);
      store_dis_component(dis_q_neutralization_custom_,
                          heat.ion_neutralization_W_m2);
      store_dis_component(dis_q_sheath_custom_, heat.sheath_W_m2);
      store_dis_component(dis_q_thermionic_custom_,
                          heat.thermionic_cooling_W_m2);
      store_dis_component(dis_q_secondary_custom_,
                          heat.secondary_cooling_W_m2);
    } else {
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
      Qoml = QE * (Ge * Ee + Gi * Ei);
    }
  }

  double a_over_lambdaD = 0.0;
  if (ne > 0.0 && Te > 0.0) {
    const double lambdaD = std::sqrt(EPS0 * Te / (ne * QE));
    if (lambdaD > 0.0 && std::isfinite(lambdaD))
      a_over_lambdaD = radius / lambdaD;
  }

  double oml_weight = 0.0;
  if (heating_mode_ == 1) {
    oml_weight = 1.0;
  } else if (heating_mode_ == 2) {
    if (a_over_lambdaD > 1.0)
      error->one(FLERR,
        "Fix evaporation heating auto: a/lambda_D > 1 requires a "
        "finite-sheath collection model; choose an explicit heating model");
    if (a_over_lambdaD > 0.0) oml_weight = 1.0;
  }
  double Qs = (1.0 - oml_weight) * Qflux + oml_weight * Qoml;
  if (!std::isfinite(Qs))
    error->one(FLERR, "Fix evaporation: non-finite surface heat flux");
  // Preserve the historical non-negative DUSTT input.  DIS emission can
  // physically cool the grain, so its signed net flux must reach the energy
  // equation instead of being silently clamped away.
  if (physics_model_ == ParticulateModel::DUSTT2005 && Qs < 0.0) Qs = 0.0;
  Qs *= heatflux_scale;

  auto store_custom = [&](int custom, double value) {
    if (custom < 0) return;
    const int ew = particle->ewhich[custom];
    if (ew >= 0) particle->edvec[ew][idrop] = value;
  };
  store_custom(heating_q_custom_, Qs);
  store_custom(debye_ratio_custom_, a_over_lambdaD);
  store_custom(oml_weight_custom_, oml_weight);

  // NOTE: no early-return on Qs <= 0 — a grain in vacuum (real q_par is 0
  // outside the SOLPS mesh) still evaporates and cools radiatively.

  const double a1 = mat_->antoine_a, b1 = mat_->antoine_b;
  const double xm1 = mat_->mass_amu;
  const double max_dT = 25.0;       // K, local explicit substep limiter
  const double max_dR_frac = 0.02;  // limit radius loss in one substep
  const int max_substeps = 10000;

  double R_new = radius;
  double T_new = TK;
  double last_Qs = Qs;
  double evap_atoms_step = 0.0;
  double terminal_remainder_atoms_step = 0.0;
  double t_left = DT;
  int nsub = 0;

  while (t_left > 0.0 && R_new > terminal_radius) {
    if (T_new <= 0.0 || !std::isfinite(T_new))
      error->one(FLERR,"Fix evaporation: invalid particle temperature");

    // Thermionic emission can change rapidly with grain temperature.  A DIS
    // adaptive thermal substep therefore re-solves the floating current and
    // its matching heat balance at the current (R,T), using the same charge-
    // fix configuration.  SEE-only boron cases pay little extra because the
    // Kollath flux average is analytic rather than a nested quadrature.
    double Qs_step = Qs;
    if (physics_model_ == ParticulateModel::DIS2021 && oml_weight > 0.0 &&
        Te > 0.0 && Ti > 0.0 && ne > 0.0 && ni > 0.0) {
      ParticulateModel::DISCurrentState sub_current;
      ParticulateModel::DISHeatFluxState sub_heat;
      if (!charge_fix_ ||
          !charge_fix_->solve_dis_state(Te, Ti, ne, ni, T_new, R_new,
                                        dis_relative_mach,
                                        sub_current, &sub_heat))
        error->one(FLERR,
          "fix particulate/thermal model dis2021: substep current/heat "
          "balance failed");
      Qs_step = ((1.0 - oml_weight) * Qflux
                 + oml_weight * sub_heat.net_W_m2) * heatflux_scale;
      if (!std::isfinite(Qs_step))
        error->one(FLERR,
          "fix particulate/thermal model dis2021: non-finite substep heat flux");
      // Keep component diagnostics synchronized with droplet_heating_q,
      // which reports the final adaptive substep rather than the initial
      // state at entry to this half-step.
      store_custom(dis_q_ion_custom_, sub_heat.ion_kinetic_W_m2);
      store_custom(dis_q_electron_custom_, sub_heat.electron_kinetic_W_m2);
      store_custom(dis_q_neutralization_custom_,
                   sub_heat.ion_neutralization_W_m2);
      store_custom(dis_q_sheath_custom_, sub_heat.sheath_W_m2);
      store_custom(dis_q_thermionic_custom_,
                   sub_heat.thermionic_cooling_W_m2);
      store_custom(dis_q_secondary_custom_,
                   sub_heat.secondary_cooling_W_m2);
    }
    last_Qs = Qs_step;

    // Antoine + Hertz-Knudsen flux.
    const double vpres1 = 760.0 * std::pow(10.0, a1 + b1/T_new);      // mmHg
    const double Gevap_atoms =
      alpha_e_ * 1.0e4 * 3.513e22 * vpres1 / std::sqrt(xm1 * T_new);
    if (!std::isfinite(Gevap_atoms) || Gevap_atoms < 0.0)
      error->one(FLERR,"Fix evaporation: invalid evaporation flux");

    const double dRdt = -AM * Gevap_atoms / Rho;
    // Energy balance per unit surface: plasma heating - radiative cooling
    // (emissivity * sigma * (T^4 - Twall^4)) - evaporative (latent) cooling.
    // Radiation is negligible for liquid Li (~kW/m^2 at 1000 K, eps~0.1)
    // but is a LEADING term for boron (~MW/m^2 at 2300 K, eps~0.8).
    const double q_rad = Eps * SB * (T_new*T_new*T_new*T_new - Tw4);
    const double HF   = Qs_step - q_rad - Gevap_atoms * (DHm / AN);
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
      // Land exactly on the selected terminal radius instead of stepping
      // past it.  For DIS this radius corresponds only to the configurable
      // numerical mass tolerance; for DUSTT it is the published R0/10
      // breakup criterion.
      const double dt_terminal = (R_new - terminal_radius) / (-dRdt);
      if (dt_terminal >= 0.0 && dt_terminal < dt_sub) dt_sub = dt_terminal;
    }
    if (dt_sub <= 0.0 || !std::isfinite(dt_sub))
      error->one(FLERR,"Fix evaporation: invalid adaptive timestep");

    const double R_old = R_new;
    R_new = std::max(terminal_radius, R_new + dRdt * dt_sub);
    // Derive the atom tally from the actual spherical mass change.  This
    // makes radius, mass, and emitted-atom accounting identical even at a
    // finite adaptive substep, unlike area_old*Gamma*dt quadrature.
    const double volume_factor = Rho * (4.0/3.0) * MY_PI;
    const double dm = volume_factor *
      (R_old*R_old*R_old - R_new*R_new*R_new);
    if (dm > 0.0) evap_atoms_step += dm / AM;
    T_new += dTdt * dt_sub;
    t_left = std::max(0.0, t_left - dt_sub);

    if (++nsub > max_substeps)
      error->one(FLERR,"Fix evaporation: adaptive timestep exceeded limit");
  }

  const double m_new   = (R_new > 0.0)
                         ? (Rho * (4.0/3.0) * MY_PI * R_new*R_new*R_new)
                         : 0.0;
  store_custom(heating_q_custom_, last_Qs);
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
  double final_mass = m_new;
  if (R_new <= terminal_radius) {
    if (termination_mode_ == TERMINATION_DIS) {
      // DIS represents complete evaporation.  Convert the deliberately tiny
      // numerical remainder into vapor so mass and atom accounting close.
      evap_atoms_step += final_mass / AM;
    } else {
      // DUSTT stops following the grain at R/R0=0.1.  Preserve that 0.1%
      // mass as a separately reported breakup remainder, not as vapor.
      terminal_remainder_atoms_step = final_mass / AM;
    }
    final_mass = 0.0;
    R_new = 0.0;
  }

  evap_atoms_local_ += evap_atoms_step * nw;
  terminal_remainder_atoms_local_ += terminal_remainder_atoms_step * nw;

  // Rocket-force kick (in-place velocity update via fresh pointer).
  if (rocket_eta > 0.0 && final_mass > 0.0 && evap_atoms_step > 0.0) {
    const double grad_mag = std::sqrt(
      plasma.grad_te[0]*plasma.grad_te[0] +
      plasma.grad_te[1]*plasma.grad_te[1] +
      plasma.grad_te[2]*plasma.grad_te[2]);
    if (std::isfinite(grad_mag) && grad_mag > 0.0) {
      const double kB = 1.380649e-23;
      const double T_rocket = std::max(1.0, 0.5 * (TK + T_new));
      const double v_thermal = std::sqrt(8.0 * kB * T_rocket / (MY_PI * AM));
      const double dmdt = evap_atoms_step * AM / DT;
      const double a_mag = rocket_eta * dmdt * v_thermal / final_mass;
      Particle::OnePart *ip_w = &particle->particles[idrop];
      ip_w->v[0] -= a_mag * plasma.grad_te[0] / grad_mag * DT;
      ip_w->v[1] -= a_mag * plasma.grad_te[1] / grad_mag * DT;
      ip_w->v[2] -= a_mag * plasma.grad_te[2] / grad_mag * DT;
    }
  }

  // Volumetric material source: spawn evaporated atoms in the grain's cell.
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
  ip_w->temp   = final_mass > 0.0 ? T_new : 0.0;
  ip_w->mass   = final_mass;
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

  // The appended atoms are not in the per-cell lists. Without this, a later
  // fix balance trusts sorted==1, skips its re-sort, and leaves them behind
  // holding pre-balance cell indices.
  particle->sorted = 0;
}
