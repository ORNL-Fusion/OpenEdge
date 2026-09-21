/* ----------------------------------------------------------------------
   OpenEdge: grain material registry + `material` input command.
   See grain_material.h for usage.
------------------------------------------------------------------------- */

#include "grain_material.h"
#include "error.h"
#include "input.h"
#include "comm.h"

#include <cstring>
#include <cstdio>
#include <vector>

using namespace SPARTA_NS;

/* ----------------------------------------------------------------------
   Built-in materials.

   Li: values carried over verbatim from the original hardcoded droplet
   model ("Sergey's model", fix_droplet_evaporate) so legacy decks are
   bit-compatible: rho 534, cp 4200, Antoine log10 p[mmHg] = 5.055
   - 8023/T, hvap 1.47e5 J/mol. Emissivity for liquid Li is low (~0.1).

   B: solid boron powder (Afonin NF 2023 context). Sublimation-dominated:
   Antoine fit to vapor-pressure data over 2350-3700 K
   (log10 p[atm] = 8.64 - 32030/T; p_sat ~ 1 Pa only at ~2350 K).
   hvap = enthalpy of sublimation ~5.65e5 J/mol; cp is the high-T solid
   value (T-dependence not yet modeled); emissivity ~0.8 for hot boron.

   Antoine convention (both entries): log10 p_sat[atm] = a + b/T; the
   evaporate fix multiplies by 760 to get mmHg for the Langmuir flux.
------------------------------------------------------------------------- */

// built via a function: gcc (<=9 at least) rejects initializer_list
// aggregate init with a char-array member that clang accepts
static GrainMaterial make_entry(const char *nm, double rho, double cp,
    double cps, double mass, double hvap, double aa, double ab, double em,
    double wf, double ra, double tm, double hm, double tp,
    double sdm = 0.0, double sem = 0.0)
{
  GrainMaterial m;
  strncpy(m.name, nm, sizeof(m.name) - 1);
  m.name[sizeof(m.name) - 1] = '\0';
  m.provenance_id[0] = '\0';
  m.explicit_mask = 0;
  m.rho = rho; m.cp = cp; m.cp_solid = cps;
  m.mass_amu = mass; m.hvap_J_mol = hvap;
  m.antoine_a = aa; m.antoine_b = ab; m.emissivity = em;
  m.work_function_eV = wf; m.richardson_A = ra; m.tmelt_K = tm;
  m.hmelt_J_mol = hm; m.tensile_Pa = tp;
  m.see_delta_m = sdm; m.see_E_m_eV = sem;
  return m;
}

static std::vector<GrainMaterial> make_registry()
{
  std::vector<GrainMaterial> v;
  // Li cp 4200 = liquid; cp_solid 3580 = solid near room T (24.86 J/mol/K)
  v.push_back(make_entry("Li", 534.0, 4200.0, 3580.0, 6.94, 1.47e5, 5.055,
                         -8023.0, 0.10, 2.9, 1.2e6, 453.7, 3.0e3, 0.0,
                         0.5, 85.0));
  // B cp 2000 is already the high-T solid value; cp_solid 0 -> single cp
  v.push_back(make_entry("B", 2340.0, 2000.0, 0.0, 10.81, 5.65e5, 8.64,
                         -32030.0, 0.80, 4.45, 1.2e6, 2349.0, 5.02e4, 1.0e9,
                         1.2, 150.0));
  return v;
}

static std::vector<GrainMaterial> registry = make_registry();

const GrainMaterial *SPARTA_NS::grain_material_find(const char *name)
{
  for (auto &m : registry)
    if (strcmp(m.name, name) == 0) return &m;
  return nullptr;
}

GrainMaterial *SPARTA_NS::grain_material_define(const char *name)
{
  for (auto &m : registry)
    if (strcmp(m.name, name) == 0) return &m;
  GrainMaterial m = {};
  strncpy(m.name, name, sizeof(m.name) - 1);
  // sentinel defaults so a partially-specified new material errors loudly
  m.rho = m.cp = m.mass_amu = m.hvap_J_mol = -1.0;
  m.cp_solid = 0.0;   // 0 = use cp for both phases
  m.antoine_a = 0.0; m.antoine_b = 0.0;
  m.emissivity = 0.0;
  m.work_function_eV = -1.0; m.richardson_A = -1.0;
  m.tmelt_K = -1.0; m.hmelt_J_mol = 0.0; m.tensile_Pa = 0.0;
  m.see_delta_m = 0.0; m.see_E_m_eV = 0.0;
  m.provenance_id[0] = '\0';
  m.explicit_mask = 0;
  registry.push_back(m);
  return &registry.back();
}

bool SPARTA_NS::grain_material_missing_properties(
  const GrainMaterial *m, const std::uint64_t required,
  char *buffer, const std::size_t nbuffer)
{
  if (!buffer || nbuffer == 0) return true;
  buffer[0] = '\0';
  const std::uint64_t missing = m ? required & ~m->explicit_mask : required;
  if (!missing) return false;

  struct Item { std::uint64_t bit; const char *name; };
  static const Item items[] = {
    {GRAIN_MAT_RHO, "rho"}, {GRAIN_MAT_CP, "cp"},
    {GRAIN_MAT_CP_SOLID, "cp_solid"},
    {GRAIN_MAT_MASS_AMU, "mass_amu"},
    {GRAIN_MAT_HVAP, "hvap_J_mol"},
    {GRAIN_MAT_ANTOINE_A, "antoine_a"},
    {GRAIN_MAT_ANTOINE_B, "antoine_b"},
    {GRAIN_MAT_EMISSIVITY, "emissivity"},
    {GRAIN_MAT_WORK_FUNCTION, "work_function_eV"},
    {GRAIN_MAT_RICHARDSON, "richardson_A"},
    {GRAIN_MAT_TMELT, "tmelt_K"},
    {GRAIN_MAT_HMELT, "hmelt_J_mol"},
    {GRAIN_MAT_TENSILE, "tensile_Pa"},
    {GRAIN_MAT_SEE_DELTA_M, "see_delta_m"},
    {GRAIN_MAT_SEE_E_M, "see_E_m_eV"}
  };
  bool first = true;
  for (const Item &item : items) {
    if (!(missing & item.bit)) continue;
    const std::size_t used = strlen(buffer);
    if (used >= nbuffer - 1) break;
    snprintf(buffer + used, nbuffer - used, "%s%s",
             first ? "" : ",", item.name);
    first = false;
  }
  return true;
}

/* ----------------------------------------------------------------------
   material NAME keyword value [keyword value ...]
------------------------------------------------------------------------- */

void MaterialCmd::command(int narg, char **arg)
{
  if (narg < 3 || (narg - 1) % 2 != 0)
    error->all(FLERR, "Illegal material command: material NAME key value ...");

  if (strlen(arg[0]) >= sizeof(GrainMaterial::name))
    error->all(FLERR, "material: name too long (max 15 chars)");

  GrainMaterial *m = grain_material_define(arg[0]);

  for (int i = 1; i < narg; i += 2) {
    const char *key = arg[i];
    if (strcmp(key, "provenance") == 0) {
      if (strlen(arg[i + 1]) >= sizeof(m->provenance_id))
        error->all(FLERR, "material: provenance ID too long (max 95 chars)");
      strcpy(m->provenance_id, arg[i + 1]);
      continue;
    }
    const double v = input->numeric(FLERR, arg[i + 1]);
    if (strcmp(key, "rho") == 0) {
      m->rho = v; m->explicit_mask |= GRAIN_MAT_RHO;
    } else if (strcmp(key, "cp") == 0) {
      m->cp = v; m->explicit_mask |= GRAIN_MAT_CP;
    } else if (strcmp(key, "cp_solid") == 0) {
      m->cp_solid = v; m->explicit_mask |= GRAIN_MAT_CP_SOLID;
    } else if (strcmp(key, "mass_amu") == 0) {
      m->mass_amu = v; m->explicit_mask |= GRAIN_MAT_MASS_AMU;
    } else if (strcmp(key, "hvap_J_mol") == 0) {
      m->hvap_J_mol = v; m->explicit_mask |= GRAIN_MAT_HVAP;
    } else if (strcmp(key, "antoine_a") == 0) {
      m->antoine_a = v; m->explicit_mask |= GRAIN_MAT_ANTOINE_A;
    } else if (strcmp(key, "antoine_b") == 0) {
      m->antoine_b = v; m->explicit_mask |= GRAIN_MAT_ANTOINE_B;
    } else if (strcmp(key, "emissivity") == 0) {
      m->emissivity = v; m->explicit_mask |= GRAIN_MAT_EMISSIVITY;
    } else if (strcmp(key, "work_function_eV") == 0) {
      m->work_function_eV = v;
      m->explicit_mask |= GRAIN_MAT_WORK_FUNCTION;
    } else if (strcmp(key, "richardson_A") == 0) {
      m->richardson_A = v; m->explicit_mask |= GRAIN_MAT_RICHARDSON;
    } else if (strcmp(key, "tmelt_K") == 0) {
      m->tmelt_K = v; m->explicit_mask |= GRAIN_MAT_TMELT;
    } else if (strcmp(key, "hmelt_J_mol") == 0) {
      m->hmelt_J_mol = v; m->explicit_mask |= GRAIN_MAT_HMELT;
    } else if (strcmp(key, "tensile_Pa") == 0) {
      m->tensile_Pa = v; m->explicit_mask |= GRAIN_MAT_TENSILE;
    } else if (strcmp(key, "see_delta_m") == 0) {
      m->see_delta_m = v; m->explicit_mask |= GRAIN_MAT_SEE_DELTA_M;
    } else if (strcmp(key, "see_E_m_eV") == 0) {
      m->see_E_m_eV = v; m->explicit_mask |= GRAIN_MAT_SEE_E_M;
    }
    else {
      char msg[128];
      snprintf(msg, sizeof(msg), "material: unknown keyword '%s'", key);
      error->all(FLERR, msg);
    }
  }

  if (comm->me == 0 && screen)
    fprintf(screen, "material %s: provenance=%s explicit_mask=0x%llx "
            "rho=%g cp=%g m=%g amu hvap=%g J/mol antoine=(%g,%g) "
            "eps=%g W=%g eV Tmelt=%g K SEE=(%g,%g eV)\n",
            m->name, m->provenance_id[0] ? m->provenance_id : "UNSET",
            static_cast<unsigned long long>(m->explicit_mask),
            m->rho, m->cp, m->mass_amu, m->hvap_J_mol,
            m->antoine_a, m->antoine_b, m->emissivity,
            m->work_function_eV, m->tmelt_K,
            m->see_delta_m, m->see_E_m_eV);
}
