/* ----------------------------------------------------------------------
    OpenEdge: BCA-coupled PMI surface reactions
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - Austin Nichols (ORNL, nicholsa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge

    See surf_react_pmi_bca.h for documentation.
------------------------------------------------------------------------- */

#include "math.h"
#include "surf_react_pmi_bca.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "particle.h"
#include "surf.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <cmath>
#include <cstring>
#include <algorithm>

using namespace SPARTA_NS;

/* ---------------------------------------------------------------------- */

SurfReactPMIBCA::SurfReactPMIBCA(SPARTA *sparta, int narg, char **arg) :
  SurfReactPMI(sparta, narg, arg)
{
  // Parse additional keywords after the base pmi arguments
  // Base pmi constructor parsed: ID pmi/bca file reaction_file.surf ...
  // Additional args start after the base parsing

  max_nlayers = 20;
  init_layer_thickness = 100.0;  // 100 Angstroms default
  substrate_density = 6.33e22;   // W density
  substrate_name = "W";
  substrate_species_idx = -1;
  use_surrogate = 0;
  surrogate_threshold = 0.1;

  n_bca_calls = 0;
  n_surrogate_calls = 0;
  n_implanted = 0;
  n_reflected_bca = 0;
  n_sputtered_bca = 0;

  // Scan for bca-specific keywords
  // The base constructor already consumed its arguments, so we
  // re-scan narg/arg for our keywords
  for (int iarg = 3; iarg < narg; iarg++) {
    if (strcmp(arg[iarg], "bca") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing library path after 'bca'");
      bca_lib_path = std::string(arg[iarg + 1]);
      iarg++;
    } else if (strcmp(arg[iarg], "nlayers") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing value after 'nlayers'");
      max_nlayers = input->inumeric(FLERR, arg[iarg + 1]);
      iarg++;
    } else if (strcmp(arg[iarg], "layer_thickness") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing value after 'layer_thickness'");
      init_layer_thickness = input->numeric(FLERR, arg[iarg + 1]);
      iarg++;
    } else if (strcmp(arg[iarg], "substrate") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing species after 'substrate'");
      substrate_name = std::string(arg[iarg + 1]);
      iarg++;
    } else if (strcmp(arg[iarg], "density") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing value after 'density'");
      substrate_density = input->numeric(FLERR, arg[iarg + 1]);
      iarg++;
    } else if (strcmp(arg[iarg], "surrogate") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing path after 'surrogate'");
      surrogate_path = std::string(arg[iarg + 1]);
      use_surrogate = 1;
      iarg++;
    } else if (strcmp(arg[iarg], "surrogate_threshold") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "surf_react pmi/bca: missing value after 'surrogate_threshold'");
      surrogate_threshold = input->numeric(FLERR, arg[iarg + 1]);
      iarg++;
    }
    // Skip keywords already parsed by base class
  }

  if (bca_lib_path.empty())
    error->all(FLERR, "surf_react pmi/bca: 'bca' keyword with library path is required");
}

/* ---------------------------------------------------------------------- */

SurfReactPMIBCA::~SurfReactPMIBCA()
{
  if (comm->me == 0) {
    printf("  surf_react pmi/bca stats:\n");
    printf("    BCA calls: %lld\n", (long long)n_bca_calls);
    printf("    Surrogate calls: %lld\n", (long long)n_surrogate_calls);
    printf("    Implanted: %lld\n", (long long)n_implanted);
    printf("    Reflected (BCA): %lld\n", (long long)n_reflected_bca);
    printf("    Sputtered (BCA): %lld\n", (long long)n_sputtered_bca);
  }
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIBCA::init()
{
  SurfReactPMI::init();

  // Load RustBCA library
  int err = bca.load_library(bca_lib_path);
  if (err) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "surf_react pmi/bca: failed to load BCA library: %s",
             bca.last_error().c_str());
    error->all(FLERR, msg);
  }

  // Setup species info (Z, M, binding energies)
  setup_species_info();

  // Find substrate species in SPARTA species list
  substrate_species_idx = particle->find_species(
      const_cast<char*>(substrate_name.c_str()));

  // Allocate multilayer surface state
  int nsurf = surf->nlocal + surf->nghost;
  if (nsurf <= 0) nsurf = 1;  // safety

  int nspec = static_cast<int>(species_info.size());
  surf_state = std::make_unique<SurfStateMultilayer>(nspec, max_nlayers);
  surf_state->allocate(nsurf);

  // Initialize substrate
  int sub_idx = -1;
  for (int i = 0; i < nspec; i++) {
    if (species_info[i].Z == lookup_species(substrate_name).Z) {
      sub_idx = i;
      break;
    }
  }
  surf_state->init_substrate(init_layer_thickness, substrate_density, sub_idx);

  if (comm->me == 0) {
    printf("  surf_react pmi/bca: loaded %s\n", bca_lib_path.c_str());
    printf("    substrate: %s (Z=%d, density=%.3e atoms/cm^3)\n",
           substrate_name.c_str(),
           sub_idx >= 0 ? species_info[sub_idx].Z : 0,
           substrate_density);
    printf("    max_layers=%d, init_thickness=%.1f A\n",
           max_nlayers, init_layer_thickness);
    printf("    nsurf=%d, nspecies=%d\n", nsurf, nspec);
    if (use_surrogate)
      printf("    surrogate: %s (threshold=%.3f)\n",
             surrogate_path.c_str(), surrogate_threshold);
  }
}

/* ----------------------------------------------------------------------
   BCA-based surface reaction
------------------------------------------------------------------------- */

int SurfReactPMIBCA::react(Particle::OnePart *&ip, int isurf,
                            double *norm,
                            Particle::OnePart *&jp,
                            int &velreset)
{
  jp = NULL;
  last_outcome = 0;

  int ispecies = ip->ispecies;
  SpeciesReactions &sr = species_reactions[ispecies];

  if (sr.ireflect < 0 && sr.isputter < 0 && sr.iabsorb < 0) return 0;

  // Compute incident energy and direction
  double *v = ip->v;
  double vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  double vlen = sqrt(vsq);

  double mass = particle->species[ispecies].mass;
  double E_eV = 0.5 * update->mvv2e * mass * vsq;
  if (strcmp(update->unit_style, "si") == 0) E_eV *= update->joule2ev;

  if (E_eV < 0.01 || vlen < 1.0e-20) {
    // Very low energy: just absorb
    if (sr.iabsorb >= 0) {
      ip = NULL;
      nsingle++;
      tally_single[sr.iabsorb]++;
      last_outcome = 3;
      return sr.iabsorb + 1;
    }
    return 0;
  }

  // Direction cosines relative to surface normal
  // RustBCA convention: uz = inward normal component
  double nlen = MathExtra::len3(norm);
  double ux_lab = v[0] / vlen;
  double uy_lab = v[1] / vlen;
  double uz_lab = v[2] / vlen;

  // Build target layers from multilayer state
  int surf_idx = std::min(isurf, surf_state->size() - 1);
  if (surf_idx < 0) surf_idx = 0;

  std::vector<BCALayer> target = build_target_layers(surf_idx);

  // Get projectile species info
  int proj_Z = 1;   // default hydrogen
  double proj_M = 2.014;
  if (ispecies < static_cast<int>(species_info.size())) {
    proj_Z = species_info[ispecies].Z;
    proj_M = species_info[ispecies].M;
  }

  // Call RustBCA
  BCAResult result = bca.run_bca(proj_Z, proj_M, E_eV,
                                  ux_lab, uy_lab, uz_lab,
                                  target);
  n_bca_calls++;

  velreset = 1;

  // Process result
  if (!result.reflected.empty() && sr.ireflect >= 0) {
    // REFLECT: use BCA-computed reflected particle
    int irxn = sr.ireflect;
    OneReaction *r = &rlist[irxn];

    const auto &ref = result.reflected[0];
    ip->ispecies = r->products[0];

    double mass_out = particle->species[ip->ispecies].mass;
    double E_out_J = ref.E / update->joule2ev / update->mvv2e;
    double vmag_out = sqrt(2.0 * E_out_J / mass_out);

    // Use BCA direction, reflected back from surface
    double dir[3] = {ref.ux, ref.uy, ref.uz};
    double dlen = sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (dlen > 0.0) {
      ip->v[0] = vmag_out * dir[0] / dlen;
      ip->v[1] = vmag_out * dir[1] / dlen;
      ip->v[2] = vmag_out * dir[2] / dlen;
    } else {
      sample_cosine_direction(norm, ip->v, vmag_out);
    }

    ip->erot = 0.0;
    ip->evib = 0.0;

    // Update surface state: projectile reflected, sputtered atoms removed
    apply_bca_result(surf_idx, result, ispecies);

    nsingle++;
    tally_single[irxn]++;
    last_outcome = 1;
    n_reflected_bca++;

    // Handle sputtered particles from BCA (create jp if any)
    if (!result.sputtered.empty() && sr.isputter >= 0) {
      int irxn_sp = sr.isputter;
      OneReaction *r_sp = &rlist[irxn_sp];
      int sput_species = r_sp->products[0];
      double mass_sput = particle->species[sput_species].mass;

      const auto &sp = result.sputtered[0];
      double E_sput_J = sp.E / update->joule2ev / update->mvv2e;
      double vmag_sput = sqrt(2.0 * E_sput_J / mass_sput);

      double x[3];
      memcpy(x, ip->x, 3 * sizeof(double));
      int icell = ip->icell;

      double v_sput[3];
      sample_cosine_direction(norm, v_sput, vmag_sput);

      int id = MAXSMALLINT * random->uniform();
      particle->add_particle(id, sput_species, icell, x, v_sput, 0.0, 0.0);
      jp = &particle->particles[particle->nlocal - 1];

      n_sputtered_bca++;
      tally_single[irxn_sp]++;
    }

    return irxn + 1;

  } else if (result.was_implanted) {
    // ABSORB: particle implanted into surface
    apply_bca_result(surf_idx, result, ispecies);

    // Handle sputtered particles
    if (!result.sputtered.empty() && sr.isputter >= 0) {
      int irxn = sr.isputter;
      OneReaction *r = &rlist[irxn];
      int sput_species = r->products[0];
      double mass_sput = particle->species[sput_species].mass;

      const auto &sp = result.sputtered[0];
      double E_sput_J = sp.E / update->joule2ev / update->mvv2e;
      double vmag_sput = sqrt(2.0 * E_sput_J / mass_sput);

      double x[3];
      memcpy(x, ip->x, 3 * sizeof(double));
      int icell = ip->icell;

      ip = NULL;  // absorb incident

      double v_sput[3];
      sample_cosine_direction(norm, v_sput, vmag_sput);

      int id = MAXSMALLINT * random->uniform();
      particle->add_particle(id, sput_species, icell, x, v_sput, 0.0, 0.0);
      jp = &particle->particles[particle->nlocal - 1];

      nsingle++;
      tally_single[irxn]++;
      last_outcome = 2;
      n_sputtered_bca++;
      n_implanted++;
      return irxn + 1;

    } else if (sr.iabsorb >= 0) {
      int irxn = sr.iabsorb;
      ip = NULL;

      nsingle++;
      tally_single[irxn]++;
      last_outcome = 3;
      n_implanted++;
      return irxn + 1;
    }
  }

  // Fallback: no reaction from BCA
  // Use tabulated data as fallback (base class behavior)
  return SurfReactPMI::react(ip, isurf, norm, jp, velreset);
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIBCA::setup_species_info()
{
  int nspecies = particle->nspecies;
  species_info.resize(nspecies);

  for (int i = 0; i < nspecies; i++) {
    std::string name(particle->species[i].id);
    species_info[i] = lookup_species(name);
    // Override mass with SPARTA particle data
    species_info[i].M = particle->species[i].mass;
    // Convert SPARTA internal mass units if needed
    if (strcmp(update->unit_style, "si") == 0) {
      // SPARTA SI: mass in kg, convert to amu
      species_info[i].M /= 1.66053906660e-27;
    }
  }
}

/* ---------------------------------------------------------------------- */

int SurfReactPMIBCA::find_sparta_species(int Z, double M) const
{
  // Find closest SPARTA species matching Z and M
  int best = -1;
  double best_dm = 1.0e30;

  for (int i = 0; i < static_cast<int>(species_info.size()); i++) {
    if (species_info[i].Z == Z) {
      double dm = fabs(species_info[i].M - M);
      if (dm < best_dm) {
        best_dm = dm;
        best = i;
      }
    }
  }

  return best;
}

/* ---------------------------------------------------------------------- */

std::vector<BCALayer> SurfReactPMIBCA::build_target_layers(int isurf) const
{
  std::vector<BCALayer> target;
  const SurfaceElementState &state = (*surf_state)[isurf];

  for (const auto &lyr : state.layers) {
    BCALayer bl;
    bl.thickness = lyr.thickness;
    bl.density = lyr.density;

    for (int s = 0; s < state.nspecies; s++) {
      if (lyr.composition[s] > 0.0 && s < static_cast<int>(species_info.size())) {
        bl.Z.push_back(species_info[s].Z);
        bl.M.push_back(species_info[s].M);
        bl.fraction.push_back(lyr.composition[s]);
        bl.Eb.push_back(species_info[s].Eb);
        bl.Es.push_back(species_info[s].Es);
        bl.Ec.push_back(species_info[s].Ec);
      }
    }

    // Ensure at least one species
    if (bl.Z.empty()) {
      SpeciesInfo sub = lookup_species(substrate_name);
      bl.Z.push_back(sub.Z);
      bl.M.push_back(sub.M);
      bl.fraction.push_back(1.0);
      bl.Eb.push_back(sub.Eb);
      bl.Es.push_back(sub.Es);
      bl.Ec.push_back(sub.Ec);
    }

    target.push_back(bl);
  }

  return target;
}

/* ---------------------------------------------------------------------- */

void SurfReactPMIBCA::apply_bca_result(int isurf, const BCAResult &result,
                                        int proj_species)
{
  SurfaceElementState &state = (*surf_state)[isurf];

  // Implant projectile if it stopped in the target
  if (result.was_implanted && result.implant_depth > 0.0) {
    int bca_species_idx = proj_species;
    if (bca_species_idx >= 0 && bca_species_idx < state.nspecies) {
      // amount = 1 atom over effective area (normalized later)
      state.add_implanted(bca_species_idx, result.implant_depth, 1.0e14);
    }
  }

  // Account for sputtered atoms: erode surface
  double total_sput_thickness = 0.0;
  for (const auto &sp : result.sputtered) {
    // Rough estimate: each sputtered atom removes ~2 Angstroms of surface
    total_sput_thickness += 2.0;
  }
  if (total_sput_thickness > 0.0) {
    state.erode_surface(total_sput_thickness);
  }

  // Compact layers periodically
  state.total_fluence += 1.0;
  if (static_cast<int>(state.total_fluence) % 100 == 0) {
    state.compact_layers();
  }
}

/* ----------------------------------------------------------------------
   Species database for common fusion-relevant elements
------------------------------------------------------------------------- */

SurfReactPMIBCA::SpeciesInfo
SurfReactPMIBCA::lookup_species(const std::string &name)
{
  SpeciesInfo info;
  info.Z = 1;
  info.M = 1.008;
  info.Eb = 3.0;   // bulk binding (eV)
  info.Es = 2.0;   // surface binding (eV)
  info.Ec = 1.0;   // cutoff energy (eV)

  // Hydrogen isotopes
  if (name == "H" || name == "H1") {
    info.Z = 1; info.M = 1.008; info.Es = 1.1; info.Eb = 1.1; info.Ec = 0.5;
  } else if (name == "D" || name == "D2" || name == "D+" || name == "d") {
    info.Z = 1; info.M = 2.014; info.Es = 1.1; info.Eb = 1.1; info.Ec = 0.5;
  } else if (name == "T" || name == "T3") {
    info.Z = 1; info.M = 3.016; info.Es = 1.1; info.Eb = 1.1; info.Ec = 0.5;
  }
  // Helium
  else if (name == "He" || name == "He4") {
    info.Z = 2; info.M = 4.003; info.Es = 0.0; info.Eb = 0.0; info.Ec = 1.0;
  }
  // Carbon
  else if (name == "C" || name == "C12") {
    info.Z = 6; info.M = 12.011; info.Es = 7.41; info.Eb = 3.0; info.Ec = 1.0;
  }
  // Nitrogen
  else if (name == "N" || name == "N14") {
    info.Z = 7; info.M = 14.007; info.Es = 4.92; info.Eb = 2.0; info.Ec = 1.0;
  }
  // Oxygen
  else if (name == "O" || name == "O16") {
    info.Z = 8; info.M = 15.999; info.Es = 2.58; info.Eb = 2.0; info.Ec = 1.0;
  }
  // Neon
  else if (name == "Ne") {
    info.Z = 10; info.M = 20.180; info.Es = 0.0; info.Eb = 0.0; info.Ec = 1.0;
  }
  // Argon
  else if (name == "Ar") {
    info.Z = 18; info.M = 39.948; info.Es = 0.0; info.Eb = 0.0; info.Ec = 1.0;
  }
  // Iron
  else if (name == "Fe") {
    info.Z = 26; info.M = 55.845; info.Es = 4.28; info.Eb = 3.0; info.Ec = 1.0;
  }
  // Tungsten
  else if (name == "W" || name == "W184") {
    info.Z = 74; info.M = 183.84; info.Es = 8.68; info.Eb = 8.68; info.Ec = 1.0;
  }
  // Beryllium
  else if (name == "Be") {
    info.Z = 4; info.M = 9.012; info.Es = 3.32; info.Eb = 3.32; info.Ec = 1.0;
  }

  return info;
}
