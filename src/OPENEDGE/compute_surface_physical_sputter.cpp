/* ----------------------------------------------------------------------
   OpenEdge compute surface/physical/sputter
   Uses incident plasma species fluxes + BCA surface data.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_surface_physical_sputter.h"
#include "surf.h"
#include "domain.h"
#include "comm.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "fix_background.h"
#include "openedge_geom.h"
#include "eckstein_sputter_data.h"
#include "eckstein_sputter.h"
#include "iead_table.h"
#include "database_paths.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPARTA_NS;

ComputeSurfacePhysicalSputter::ComputeSurfacePhysicalSputter(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 6) error->all(FLERR,"Illegal compute surface/physical/sputter command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute surface/physical/sputter surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  eckstein_mode = 0;
  eck_Z1 = eck_M1 = eck_Z2 = eck_M2 = 0.0;
  eck_Es = eck_Eth = eck_Q = eck_ETF = 0.0;

  // Helper: if the 6th token is "eckstein <entry_name>", load
  // analytic Eckstein sputter parameters and return the new iarg.
  auto try_parse_eckstein = [&](int iarg_surf) -> int {
    if (iarg_surf >= narg) return iarg_surf;
    if (strcmp(arg[iarg_surf],"eckstein") != 0) return iarg_surf;
    if (iarg_surf+1 >= narg)
      error->all(FLERR,"compute surface/physical/sputter: eckstein needs entry name (e.g. O_on_W)");
    eckstein_name = std::string(arg[iarg_surf+1]);
    Eckstein::SputterParams p;
    if (!Eckstein::lookup_sputter(eckstein_name.c_str(), p)) {
      std::string msg = "compute surface/physical/sputter: unknown eckstein entry '" +
                        eckstein_name + "' (check eckstein_sputter_data.h)";
      error->all(FLERR, msg.c_str());
    }
    eckstein_mode = 1;
    eck_Z1 = p.Z1; eck_M1 = p.M1; eck_Z2 = p.Z2; eck_M2 = p.M2;
    eck_Es = p.Es; eck_Eth = p.Eth; eck_Q = p.Q; eck_ETF = p.ETF;
    return iarg_surf + 2;
  };

  // Tokens at position 5 (after `file <plasma.h5>` or
  // `background <fix_id>`) can be one of:
  //   "eckstein <name>"   -> analytic single-pair Eckstein (legacy)
  //   "<surface.h5>"      -> tabulated yield HDF5 (legacy file path)
  //   any recognized keyword like `target`, `projectiles`, `static`,
  //     `projectile_slots`, `mass_amu`, ... -> new element-aware API,
  //     no surface file required at this position.
  auto is_keyword_token = [](const char *t) -> bool {
    // Listed keywords start the option loop below. Anything not in this
    // set is treated as a positional surface.h5 path (legacy).
    static const char *kws[] = {
        "target", "projectiles",
        "projectile_slots", "static", "mass_amu",
        "nflux_species", "incident_angle_species", "incident_energy_species",
        "sputter_yield_species", "sputter_flux_species",
        "sputter_flux_total", "sputter_rate_total",
        "erosion_flux", "erosion_rate",
        "bfield", "equilibrium", "background", "boundary", "impurity",
        "iead",
        "debug_interp",
        nullptr};
    for (int i = 0; kws[i]; i++) if (strcmp(t, kws[i]) == 0) return true;
    return false;
  };

  int iarg_start;
  if (strcmp(arg[3],"file") == 0) {
    if (narg < 6) error->all(FLERR,"compute surface/physical/sputter: file needs plasma.h5");
    plasma_path = std::string(arg[4]);
    int after = try_parse_eckstein(5);
    if (after == 5) {
      // No eckstein keyword. Accept a surface.h5 path here, but only if
      // the next token is NOT a known keyword (the new target/projectiles
      // API needs no positional surface file).
      if (narg >= 6 && !is_keyword_token(arg[5])) {
        surface_path = std::string(arg[5]);
        iarg_start = 6;
      } else {
        iarg_start = 5;
      }
    } else {
      iarg_start = after;
    }
  } else if (strcmp(arg[3],"background") == 0) {
    if (narg < 5) error->all(FLERR,"compute surface/physical/sputter: background needs fixID");
    background_fix_id = std::string(arg[4]);
    int after = try_parse_eckstein(5);
    if (after == 5) {
      if (narg >= 6 && !is_keyword_token(arg[5])) {
        surface_path = std::string(arg[5]);
        iarg_start = 6;
      } else {
        iarg_start = 5;
      }
    } else {
      iarg_start = after;
    }
  } else {
    error->all(FLERR,"compute surface/physical/sputter: first keyword must be 'file' or 'background'");
    iarg_start = 6;  // unreachable, suppress warning
  }

  proj_slot_lo = 2;
  proj_slot_hi = std::numeric_limits<int>::max();
  mass_amu = 2.0;  // default: deuterium (used for Bohm sound speed)
  static_cache = 0;
  cache_valid = 0;
  has_impurity = 0;
  imp_mass_amu = 0.0;
  imp_frac = 0.0;
  imp_Zmax = 0;

  // New element-aware API state: inert until the user supplies
  // `target ... projectiles ...` on the compute line.
  api_new = 0;

  std::vector<int> which_tmp;
  std::vector<int> which_species_tmp;
  int iarg = iarg_start;
  auto add_kind = [&](int kind, int slot) {
    which_tmp.push_back(kind);
    which_species_tmp.push_back(slot);
  };
  auto parse_slot_token = [&](const char *tok) -> int {
    std::string s(tok);
    if (s.size() > 2 && s.front() == '[' && s.back() == ']')
      s = s.substr(1,s.size()-2);
    int slot = 0;
    try {
      slot = std::stoi(s);
    } catch (...) {
      error->all(FLERR,"Expected species slot integer or all");
    }
    return slot;
  };
  auto add_slot_list = [&](int kind, const char *slotarg, int sputter_all_range) {
    std::string tok(slotarg);
    if (tok == "all" || tok == "[all]") {
      const int nall = peek_nspec_from_plasma();
      int lo = 1, hi = nall;
      if (sputter_all_range) {
        lo = std::max(1,proj_slot_lo);
        hi = std::min(nall,proj_slot_hi);
      }
      for (int s = lo; s <= hi; s++) add_kind(kind,s);
    } else {
      int slot = parse_slot_token(slotarg);
      if (slot <= 0) error->all(FLERR,"Species slot must be >= 1");
      add_kind(kind,slot);
    }
  };

  while (iarg < narg) {
    if (strcmp(arg[iarg],"projectile_slots") == 0) {
      if (iarg+2 >= narg) error->all(FLERR,"projectile_slots needs lo hi");
      proj_slot_lo = atoi(arg[iarg+1]);
      proj_slot_hi = atoi(arg[iarg+2]);
      if (proj_slot_lo <= 0 || proj_slot_hi < proj_slot_lo)
        error->all(FLERR,"Invalid projectile_slots bounds");
      iarg += 3;
    } else if (strcmp(arg[iarg],"static") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"static needs yes/no");
      if (strcmp(arg[iarg+1],"yes") == 0) static_cache = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) static_cache = 0;
      else error->all(FLERR,"static must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg],"mass_amu") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"mass_amu needs value");
      mass_amu = atof(arg[iarg+1]);
      if (mass_amu < 0.0) error->all(FLERR,"mass_amu must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg],"wall_flux_cutoff") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"wall_flux_cutoff needs value [m]");
      wall_flux_cutoff = atof(arg[iarg+1]);
      if (wall_flux_cutoff <= 0.0)
        error->all(FLERR,"wall_flux_cutoff must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg],"nflux_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"nflux_species needs slot or all");
      add_slot_list(NFLUX_SPECIES,arg[iarg+1],0);
      iarg += 2;
    } else if (strcmp(arg[iarg],"incident_angle_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"incident_angle_species needs slot or all");
      add_slot_list(INCIDENT_ANGLE_SPECIES,arg[iarg+1],0);
      iarg += 2;
    } else if (strcmp(arg[iarg],"incident_energy_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"incident_energy_species needs slot or all");
      add_slot_list(INCIDENT_ENERGY_SPECIES,arg[iarg+1],0);
      iarg += 2;
    } else if (strcmp(arg[iarg],"sputter_yield_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"sputter_yield_species needs slot or all");
      add_slot_list(SPUTTER_YIELD_SPECIES,arg[iarg+1],1);
      iarg += 2;
    } else if (strcmp(arg[iarg],"sputter_flux_species") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"sputter_flux_species needs slot or all");
      add_slot_list(SPUTTER_FLUX_SPECIES,arg[iarg+1],1);
      iarg += 2;
    } else if (strcmp(arg[iarg],"sputter_flux_total") == 0) {
      add_kind(SPUTTER_FLUX_TOTAL,0);
      iarg++;
    } else if (strcmp(arg[iarg],"sputter_rate_total") == 0) {
      // total sputter rate [particles/s] = flux [part/m^2/s] x ring area [m^2]
      // where ring area = 2*pi*R_mid*segment_length (2D) or tri area (3D).
      // Lets the user do `compute reduce sum c_ID` directly with no
      // property/surf + variable gymnastics.
      add_kind(SPUTTER_RATE_TOTAL,0);
      iarg++;
    } else if (strcmp(arg[iarg],"erosion_flux") == 0) {
      // Alias for sputter_flux_total (docs/fixes/pmi_surf_data.md).
      add_kind(SPUTTER_FLUX_TOTAL,0);
      iarg++;
    } else if (strcmp(arg[iarg],"erosion_rate") == 0) {
      // Alias for sputter_rate_total.
      add_kind(SPUTTER_RATE_TOTAL,0);
      iarg++;
    } else if (strcmp(arg[iarg],"target") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"target needs wall element symbol");
      target_element = std::string(arg[iarg+1]);
      api_new = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"projectiles") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"projectiles needs comma-separated element list or 'all'");
      projectile_elements.clear();
      std::string csv = std::string(arg[iarg+1]);
      size_t start = 0;
      while (start < csv.size()) {
        size_t comma = csv.find(',', start);
        std::string tok = csv.substr(start,
                                     comma == std::string::npos ? std::string::npos
                                                                 : comma - start);
        if (!tok.empty()) projectile_elements.push_back(tok);
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
      if (projectile_elements.empty())
        error->all(FLERR,"projectiles list is empty");
      api_new = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"bfield") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"bfield needs HDF5 file path");
      bfield_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"equilibrium") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"equilibrium needs .equ file path");
      equ_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"background") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"background needs fix ID");
      background_fix_id = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"boundary needs polygon file path");
      boundary_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"impurity") == 0) {
      // impurity mass_amu frac Zmax f1 f2 ... fZmax
      // Synthesizes a virtual projectile species from background n_e.
      // Example: impurity 10.81 0.03 5 0.05 0.15 0.30 0.30 0.20
      //   → 3% boron (10.81 amu) with charge fractions for B+ through B5+
      if (iarg+3 >= narg) error->all(FLERR,"impurity needs: mass_amu frac Zmax f1 f2 ... fZmax");
      imp_mass_amu = atof(arg[iarg+1]);
      imp_frac = atof(arg[iarg+2]);
      imp_Zmax = atoi(arg[iarg+3]);
      if (imp_mass_amu <= 0.0) error->all(FLERR,"impurity mass_amu must be > 0");
      if (imp_frac <= 0.0 || imp_frac >= 1.0) error->all(FLERR,"impurity frac must be in (0,1)");
      if (imp_Zmax <= 0) error->all(FLERR,"impurity Zmax must be > 0");
      if (iarg+3+imp_Zmax >= narg)
        error->all(FLERR,"impurity needs Zmax charge state fractions after Zmax");
      imp_charge_fracs.resize(imp_Zmax);
      double fsum = 0.0;
      for (int iz = 0; iz < imp_Zmax; iz++) {
        imp_charge_fracs[iz] = atof(arg[iarg+4+iz]);
        if (imp_charge_fracs[iz] < 0.0)
          error->all(FLERR,"impurity charge fraction must be >= 0");
        fsum += imp_charge_fracs[iz];
      }
      if (std::fabs(fsum - 1.0) > 0.01)
        error->all(FLERR,"impurity charge fractions should sum to ~1.0");
      has_impurity = 1;
      iarg += 4 + imp_Zmax;
    } else if (strcmp(arg[iarg],"iead") == 0) {
      if (iarg+1 >= narg)
        error->all(FLERR,"iead needs auto|<path.h5>|none");
      iead_arg = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"debug_interp") == 0) {
      if (iarg+2 >= narg) error->all(FLERR,"debug_interp needs E(eV) angle(deg)");
      debug_interp = 1;
      debug_E.push_back(atof(arg[iarg+1]));
      debug_A.push_back(atof(arg[iarg+2]));
      iarg += 3;
    } else {
      error->all(FLERR,"Invalid compute surface/physical/sputter value");
    }
  }

  nvalue = static_cast<int>(which_tmp.size());
  if (nvalue == 0) error->all(FLERR,"No outputs requested for compute surface/physical/sputter");
  which = new int[nvalue];
  which_species = new int[nvalue];
  for (int i = 0; i < nvalue; i++) {
    which[i] = which_tmp[i];
    which_species[i] = which_species_tmp[i];
  }

  per_surf_flag = 1;
  if (nvalue == 1) size_per_surf_cols = 0;
  else size_per_surf_cols = nvalue;
  surf_tally_flag = 0;
  timeflag = 0;

  dimension = domain->dimension;
  firstflag = 1;
  debug_interp = 0;
  distributed = 0;
  nsown = 0;
  vector_surf = nullptr;
  array_surf = nullptr;

  nr = nz = nspec = 0;
  nE = nA = 0;
  has_multi_ion = 0;
  has_bfield = 0;
  has_temp = 0;
}

ComputeSurfacePhysicalSputter::~ComputeSurfacePhysicalSputter()
{
  if (copymode) return;
  delete [] which;
  delete [] which_species;
  memory->destroy(vector_surf);
  memory->destroy(array_surf);
}

int ComputeSurfacePhysicalSputter::peek_nspec_from_plasma() const
{
  if (!background_fix_id.empty()) {
    int ifix = modify->find_fix(background_fix_id.c_str());
    if (ifix >= 0) {
      auto *pd = dynamic_cast<FixBackground *>(modify->fix[ifix]);
      if (pd) return (pd->nion > 0) ? pd->nion : 1;
    }
    return 1;
  }

  try {
    H5::H5File file(plasma_path, H5F_ACC_RDONLY);
    if (H5Lexists(file.getId(), "ions/dens", H5P_DEFAULT) > 0) {
      H5::DataSet ds = file.openDataSet("ions/dens");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[3] = {0,0,0};
      sp.getSimpleExtentDims(dims);
      if (dims[0] > 0) return static_cast<int>(dims[0]);
    }
  } catch (...) {
  }
  return 1;
}

int ComputeSurfacePhysicalSputter::in_projectile_slots(int slot1) const
{
  return (slot1 >= proj_slot_lo && slot1 <= proj_slot_hi);
}

void ComputeSurfacePhysicalSputter::rebuild_mesh_cache()
{
  if (!has_mesh) return;

  mesh_tri_rmin.resize(mesh_ntri);
  mesh_tri_rmax.resize(mesh_ntri);
  mesh_tri_zmin.resize(mesh_ntri);
  mesh_tri_zmax.resize(mesh_ntri);
  for (int t = 0; t < mesh_ntri; t++) {
    const int v0 = mesh_tri[t*3+0], v1 = mesh_tri[t*3+1], v2 = mesh_tri[t*3+2];
    const double r0 = mesh_vtx_r[v0], r1 = mesh_vtx_r[v1], r2 = mesh_vtx_r[v2];
    const double z0 = mesh_vtx_z[v0], z1 = mesh_vtx_z[v1], z2 = mesh_vtx_z[v2];
    mesh_tri_rmin[t] = std::min({r0,r1,r2});
    mesh_tri_rmax[t] = std::max({r0,r1,r2});
    mesh_tri_zmin[t] = std::min({z0,z1,z2});
    mesh_tri_zmax[t] = std::max({z0,z1,z2});
  }

  if (!mesh_tri_rmin.empty()) {
    hash_rmin = *std::min_element(mesh_tri_rmin.begin(), mesh_tri_rmin.end());
    const double rmax = *std::max_element(mesh_tri_rmax.begin(), mesh_tri_rmax.end());
    hash_zmin = *std::min_element(mesh_tri_zmin.begin(), mesh_tri_zmin.end());
    const double zmax = *std::max_element(mesh_tri_zmax.begin(), mesh_tri_zmax.end());
    hash_nr = 100;
    hash_nz = 100;
    hash_dr = (rmax - hash_rmin) / hash_nr + 1e-12;
    hash_dz = (zmax - hash_zmin) / hash_nz + 1e-12;
    hash_grid.assign(hash_nr * hash_nz, std::vector<int>());
    for (int t = 0; t < mesh_ntri; t++) {
      const int ir0 = std::max(0, static_cast<int>((mesh_tri_rmin[t] - hash_rmin) / hash_dr));
      const int ir1 = std::min(hash_nr - 1, static_cast<int>((mesh_tri_rmax[t] - hash_rmin) / hash_dr));
      const int iz0 = std::max(0, static_cast<int>((mesh_tri_zmin[t] - hash_zmin) / hash_dz));
      const int iz1 = std::min(hash_nz - 1, static_cast<int>((mesh_tri_zmax[t] - hash_zmin) / hash_dz));
      for (int iz = iz0; iz <= iz1; iz++)
        for (int ir = ir0; ir <= ir1; ir++)
          hash_grid[iz * hash_nr + ir].push_back(t);
    }
  } else {
    hash_nr = hash_nz = 0;
    hash_rmin = hash_zmin = hash_dr = hash_dz = 0.0;
    hash_grid.clear();
  }

  mapped_cr.clear();
  mapped_cz.clear();
  mapped_idx.clear();
  for (int t = 0; t < mesh_ntri; t++) {
    if (mesh_cell_idx[t] < 0) continue;
    const int v0 = mesh_tri[t*3+0], v1 = mesh_tri[t*3+1], v2 = mesh_tri[t*3+2];
    mapped_cr.push_back((mesh_vtx_r[v0] + mesh_vtx_r[v1] + mesh_vtx_r[v2]) / 3.0);
    mapped_cz.push_back((mesh_vtx_z[v0] + mesh_vtx_z[v1] + mesh_vtx_z[v2]) / 3.0);
    mapped_idx.push_back(t);
  }
}

void ComputeSurfacePhysicalSputter::load_plasma_from_fix(const FixBackground *pd)
{
  if (!pd)
    error->all(FLERR, "compute surface/physical/sputter: null background fix");

  if (pd->generation == 0)
    const_cast<FixBackground *>(pd)->init();

  nr = pd->nr;
  nz = pd->nz;
  column_x0 = pd->column_x0;
  column_y0 = pd->column_y0;
  rvals = pd->rvals;
  zvals = pd->zvals;
  dens_i = pd->dens_i;
  temp_e = pd->temp_e;
  temp_i = pd->temp_i;
  parr_flow = pd->parr_flow;
  parr_flow_r = pd->parr_flow_r;
  parr_flow_t = pd->parr_flow_t;
  parr_flow_z = pd->parr_flow_z;
  br = pd->br;
  bz = pd->bz;
  bt = pd->bt;
  has_bfield = pd->has_bfield;
  has_temp = (!temp_e.empty() && !temp_i.empty()) ? 1 : 0;

  if (pd->nion > 0) {
    has_multi_ion = 1;
    nspec = pd->nion;
    ion_charge_state_z = pd->ion_charge_z;
    ions_dens = pd->ions_dens;
    ions_temp = pd->ions_temp;
    ions_parr_flow = pd->ions_upar;
  } else {
    has_multi_ion = 0;
  }

  has_mesh = pd->has_mesh;
  mesh_nvtx = pd->mesh_nvtx;
  mesh_ntri = pd->mesh_ntri;
  mesh_ncell = pd->mesh_ncell;
  mesh_vtx_r = pd->mesh_vtx_r;
  mesh_vtx_z = pd->mesh_vtx_z;
  mesh_tri = pd->mesh_tri;
  mesh_cell_idx = pd->mesh_cell_idx;
  mesh_wall_surf_cell = pd->mesh_wall_surf_cell;
  has_mesh_wall_surf_cell = pd->has_mesh_wall_surf_cell;
  mesh_ne = pd->mesh_ne;
  mesh_te = pd->mesh_te;
  mesh_ti = pd->mesh_ti;
  mesh_ni = pd->mesh_ni;
  mesh_upar = pd->mesh_upar;
  mesh_nion = pd->mesh_nion;
  mesh_ions_dens = pd->mesh_ions_dens;
  mesh_ions_temp = pd->mesh_ions_temp;
  mesh_ions_upar = pd->mesh_ions_upar;
  // Copy /wall_flux/ scatter (empty if plasma.h5 didn't have it).
  wall_flux_n     = pd->wall_flux.n;
  wall_flux_nspec = pd->wall_flux.nspec;
  wall_flux_r     = pd->wall_flux.r;
  wall_flux_z     = pd->wall_flux.z;
  wall_flux_te    = pd->wall_flux.te;
  wall_flux_ti    = pd->wall_flux.ti;
  wall_flux_gamma_i = pd->wall_flux.gamma_i;
  wall_flux_b_r   = pd->wall_flux.b_r;
  wall_flux_b_z   = pd->wall_flux.b_z;
  wall_flux_b_t   = pd->wall_flux.b_t;
  // Per-triangle B-field from mesh-native plasma.h5 (mesh/vtx_b*
  // averaged to triangle centers by fix background). When present,
  // this takes precedence over the legacy top-level br/bz/bt arrays
  // (which are empty on mesh-only plasma.h5 from the new converters).
  mesh_tri_br = pd->mesh_tri_br;
  mesh_tri_bz = pd->mesh_tri_bz;
  mesh_tri_bt = pd->mesh_tri_bt;
  if (has_mesh && !mesh_tri_br.empty() && !mesh_tri_bz.empty())
    has_bfield = 1;
  rebuild_mesh_cache();

  // New target/projectiles API: build per-projectile Eckstein table set and
  // map plasma ion slots to the correct table via /ion_species/elements.
  if (api_new) resolve_projectile_tables(pd);
}

/* ----------------------------------------------------------------------
   Resolve projectile-element yield tables against the plasma ion species.

   Called from load_plasma_from_fix when the new target/projectiles
   keywords are used. Pulls `ion_species/elements` from fix background,
   builds one Eckstein parameter set per projectile element (name
   "<proj>_on_<target>" looked up in eckstein_sputter_data.h), and fills
   slot_to_table[s] so the per-surf loop can dispatch per-slot without
   re-doing any string lookups.

   `projectiles all` expands to the unique element set of
   pd->ion_elements (main ion included — docs example projectiles all
   covers D + O + B). Missing Eckstein entries are a fatal error so a
   silent zero-yield doesn't masquerade as a working case.
------------------------------------------------------------------------- */
void ComputeSurfacePhysicalSputter::resolve_projectile_tables(const FixBackground *pd)
{
  if (target_element.empty())
    error->all(FLERR,
        "compute surface/physical/sputter: target <wall_elem> required with projectiles");
  if (projectile_elements.empty())
    error->all(FLERR,
        "compute surface/physical/sputter: projectiles <list|all> required with target");

  // "all" -> unique elements from /ion_species/elements (preserve ion order
  // so D appears before O etc., consistent with deck-authoring intuition).
  if (projectile_elements.size() == 1 && projectile_elements[0] == "all") {
    projectile_elements.clear();
    std::set<std::string> seen;
    for (const auto &e : pd->ion_elements) {
      if (seen.insert(e).second) projectile_elements.push_back(e);
    }
    if (projectile_elements.empty())
      error->all(FLERR,
          "compute surface/physical/sputter: projectiles all but plasma.h5 has no "
          "/ion_species/elements");
  }

  // Load Eckstein coefficients for each projectile element, name is
  // "<proj>_on_<target>" (matches the eckstein_sputter_data.h convention).
  const size_t N = projectile_elements.size();
  per_proj_Z1.assign(N, 0.0);  per_proj_M1.assign(N, 0.0);
  per_proj_Z2.assign(N, 0.0);  per_proj_M2.assign(N, 0.0);
  per_proj_Es.assign(N, 0.0);  per_proj_Eth.assign(N, 0.0);
  per_proj_Q.assign(N, 0.0);   per_proj_ETF.assign(N, 0.0);
  std::string missing;
  for (size_t i = 0; i < N; i++) {
    const std::string name = projectile_elements[i] + "_on_" + target_element;
    Eckstein::SputterParams p;
    if (!Eckstein::lookup_sputter(name.c_str(), p)) {
      if (!missing.empty()) missing += ", ";
      missing += name;
      continue;
    }
    per_proj_Z1[i] = p.Z1;   per_proj_M1[i] = p.M1;
    per_proj_Z2[i] = p.Z2;   per_proj_M2[i] = p.M2;
    per_proj_Es[i] = p.Es;   per_proj_Eth[i] = p.Eth;
    per_proj_Q[i]  = p.Q;    per_proj_ETF[i] = p.ETF;
  }
  if (!missing.empty()) {
    const std::string msg =
        "compute surface/physical/sputter: missing Eckstein entries: " + missing +
        " (add to src/eckstein_sputter_data.h or drop from projectiles list)";
    error->all(FLERR, msg.c_str());
  }

  // Match plasma ion slots to projectile tables via element symbol.
  slot_to_table.assign(pd->nion, -1);
  int lo = std::numeric_limits<int>::max();
  int hi = 0;
  for (int s = 0; s < pd->nion; s++) {
    if (s >= static_cast<int>(pd->ion_elements.size())) break;
    const std::string &e = pd->ion_elements[s];
    for (size_t i = 0; i < N; i++) {
      if (e == projectile_elements[i]) {
        slot_to_table[s] = static_cast<int>(i);
        const int slot1 = s + 1;
        if (slot1 < lo) lo = slot1;
        if (slot1 > hi) hi = slot1;
        break;
      }
    }
  }
  if (lo == std::numeric_limits<int>::max())
    error->all(FLERR,
        "compute surface/physical/sputter: no plasma ion slots match the projectiles list "
        "(check /ion_species/elements in plasma.h5)");

  // Auto-set the inclusive projectile slot range so in_projectile_slots()
  // returns true exactly for slots the user asked about.
  proj_slot_lo = lo;
  proj_slot_hi = hi;
  // We are now in Eckstein mode with per-slot dispatch (eckstein_mode=1
  // keeps the main loop on the Eckstein branch; slot_to_table steers the
  // coefficients).
  eckstein_mode = 1;

  if (comm->me == 0 && screen) {
    fprintf(screen,
        "compute surface/physical/sputter: target=%s projectiles=",
        target_element.c_str());
    for (size_t i = 0; i < N; i++)
      fprintf(screen, "%s%s",
              projectile_elements[i].c_str(),
              (i + 1 < N) ? "," : "");
    fprintf(screen, "  (slots %d..%d mapped)\n", lo, hi);
  }
}

// Helper: load a single IeadTable from disk and announce it on rank 0.
// Returns the loaded table or null on resolve-miss; throws on malformed.
static std::unique_ptr<IeadTable>
load_iead_table_announce(const std::string &path, const char *tag,
                         FILE *screen, FILE *logfile)
{
  if (path.empty()) return nullptr;
  auto tbl = std::unique_ptr<IeadTable>(new IeadTable);
  tbl->load(path);
  char buf[512];
  snprintf(buf, sizeof(buf),
           "IEAD lookup loaded (%s): %s  axes (n_tau,n_psi,n_Z,n_E,n_theta) = "
           "(%d,%d,%d,%d,%d)  Te_sweep=%.1f eV\n",
           tag, path.c_str(), tbl->n_tau, tbl->n_psi,
           tbl->n_Z, tbl->n_E, tbl->n_theta, tbl->Te_eV_sweep);
  if (screen)  fputs(buf, screen);
  if (logfile) fputs(buf, logfile);
  return tbl;
}

void ComputeSurfacePhysicalSputter::load_iead_if_requested()
{
  iead_light.reset();
  iead_W.reset();
  iead_per_proj.clear();

  if (iead_arg.empty() || iead_arg == "none") return;

  // Decide which tables we need based on projectile_elements (filled by
  // resolve_projectile_tables for the api_new path) or fall back to a
  // single light table for the legacy single-projectile API.
  const int npj = static_cast<int>(projectile_elements.size());
  bool need_light = (npj == 0);   // legacy path: always need the light table
  bool need_W     = false;
  for (int ti = 0; ti < npj; ++ti) {
    if (projectile_elements[ti] == "W") need_W = true;
    else                                 need_light = true;
  }

  // Resolve paths.
  // - "auto" -> database_paths::resolve_iead_file with light/W tag.
  // - explicit path -> treated as the light table (W must be auto-resolved).
  std::string light_path, W_path;
  if (iead_arg == "auto") {
    if (need_light) light_path = resolve_iead_file("light");
    if (need_W)     W_path     = resolve_iead_file("W");
  } else {
    light_path = iead_arg;
    // No way to specify a W table explicitly via this single-arg form;
    // users mixing W projectiles with a custom light path should switch
    // to iead auto + standard install paths.
  }

  if (need_light && light_path.empty()) {
    if (comm->me == 0)
      error->warning(FLERR,
          "compute surface/physical/sputter: iead requested but light "
          "iead_database.h5 not found; lighter projectiles will fall "
          "back to mean-impact yield Y(<E>,<theta>)");
  }
  if (need_W && W_path.empty()) {
    if (comm->me == 0)
      error->warning(FLERR,
          "compute surface/physical/sputter: iead auto requested with W "
          "projectile but iead_database_W.h5 not found; W will use the "
          "light D-scaled table (angular distribution biased; energy OK)");
  }

  try {
    if (!light_path.empty())
      iead_light = load_iead_table_announce(light_path, "light",
                                            screen, logfile);
    if (!W_path.empty())
      iead_W = load_iead_table_announce(W_path, "W", screen, logfile);
  } catch (const std::exception &e) {
    std::string msg =
        std::string("compute surface/physical/sputter: failed loading IEAD "
                    "table: ") + e.what();
    error->all(FLERR, msg.c_str());
  }

  // Build per-projectile dispatch table.
  iead_per_proj.assign(static_cast<size_t>(std::max(1, npj)), nullptr);
  if (npj == 0) {
    iead_per_proj[0] = iead_light.get();   // legacy single-table path
  } else {
    for (int ti = 0; ti < npj; ++ti) {
      if (projectile_elements[ti] == "W")
        iead_per_proj[ti] = iead_W ? iead_W.get() : iead_light.get();
      else
        iead_per_proj[ti] = iead_light.get();
    }
  }
}

IeadTable *ComputeSurfacePhysicalSputter::iead_for_slot(int s) const
{
  // api_new path: route by slot_to_table[s] -> projectile element index
  if (api_new && s >= 0 && s < static_cast<int>(slot_to_table.size())) {
    const int ti = slot_to_table[s];
    if (ti >= 0 && ti < static_cast<int>(iead_per_proj.size()))
      return iead_per_proj[ti];
    return nullptr;
  }
  // Legacy single-table path
  if (!iead_per_proj.empty()) return iead_per_proj[0];
  return nullptr;
}

void ComputeSurfacePhysicalSputter::load_plasma()
{
  if (!background_fix_id.empty()) {
    int ifix = modify->find_fix(background_fix_id.c_str());
    if (ifix < 0)
      error->all(FLERR, "compute surface/physical/sputter: background fix ID not found");
    auto *pd = dynamic_cast<FixBackground *>(modify->fix[ifix]);
    if (!pd)
      error->all(FLERR, "compute surface/physical/sputter: background fix must be style background");
    load_plasma_from_fix(pd);
    return;
  }

  H5::H5File file(plasma_path, H5F_ACC_RDONLY);

  auto hasDataset = [&](const std::string& name) -> bool {
    H5E_auto2_t fn; void *dt;
    H5Eget_auto2(H5E_DEFAULT, &fn, &dt);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    htri_t rc = H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT);
    H5Eset_auto2(H5E_DEFAULT, fn, dt);
    return rc > 0;
  };
  auto read1D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t d0;
    sp.getSimpleExtentDims(&d0);
    out.resize(static_cast<size_t>(d0));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };
  auto read2D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != nz || static_cast<int>(dims[1]) != nr)
      throw std::runtime_error("compute surface/physical/sputter: plasma 2D dataset shape mismatch");
    out.resize(static_cast<size_t>(dims[0]*dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };
  auto read3D = [&](const std::string& name, std::vector<double>& out, int& ns_out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[1]) != nz || static_cast<int>(dims[2]) != nr)
      throw std::runtime_error("compute surface/physical/sputter: plasma 3D dataset shape mismatch");
    ns_out = static_cast<int>(dims[0]);
    out.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  read1D("r", rvals);
  read1D("z", zvals);
  nr = static_cast<int>(rvals.size());
  nz = static_cast<int>(zvals.size());
  if (nr < 2 || nz < 2)
    throw std::runtime_error("compute surface/physical/sputter: plasma r/z must have size >= 2");

  read2D("dens_i", dens_i);
  if (hasDataset("temp_e")) {
    read2D("temp_e", temp_e);
  } else {
    throw std::runtime_error("compute surface/physical/sputter: missing required temp_e for incident_energy_species");
  }
  read2D("parr_flow", parr_flow);
  if (hasDataset("temp_i")) {
    read2D("temp_i", temp_i);
    has_temp = 1;
  }
  if (hasDataset("parr_flow_r")) read2D("parr_flow_r", parr_flow_r);
  if (hasDataset("parr_flow_t")) read2D("parr_flow_t", parr_flow_t);
  if (hasDataset("parr_flow_z")) read2D("parr_flow_z", parr_flow_z);
  // Regular-grid /br, /bz, /bt are no longer emitted by the converters.
  // Mesh-B is picked up separately via pd->mesh_tri_b* (see reloadFromFix
  // path above). Legacy standalone plasma.h5 runs without a pd fix lose
  // B here and fall through to the scalar sheath approximation.
  has_bfield = 0;

  has_multi_ion = 0;
  if (hasDataset("ions/dens") && hasDataset("ions/parr_flow")) {
    int ns1 = 0, ns2 = 0;
    read3D("ions/dens", ions_dens, ns1);
    read3D("ions/parr_flow", ions_parr_flow, ns2);
    int ns3 = 0, ns4 = 0, ns5 = 0, ns6 = 0;
    if (hasDataset("ions/parr_flow_r") && hasDataset("ions/parr_flow_z")) {
      read3D("ions/parr_flow_r", ions_parr_flow_r, ns3);
      read3D("ions/parr_flow_z", ions_parr_flow_z, ns4);
      if (hasDataset("ions/parr_flow_t"))
        read3D("ions/parr_flow_t", ions_parr_flow_t, ns5);
    }
    if (hasDataset("ions/temp")) {
      read3D("ions/temp", ions_temp, ns6);
      has_temp = 1;
    }
    if (ns1 != ns2) throw std::runtime_error("compute surface/physical/sputter: ions/dens vs ions/parr_flow nspec mismatch");
    if (!ions_parr_flow_r.empty() && ns1 != ns3)
      throw std::runtime_error("compute surface/physical/sputter: ions/parr_flow_r nspec mismatch");
    if (!ions_parr_flow_z.empty() && ns1 != ns4)
      throw std::runtime_error("compute surface/physical/sputter: ions/parr_flow_z nspec mismatch");
    if (!ions_parr_flow_t.empty() && ns1 != ns5)
      throw std::runtime_error("compute surface/physical/sputter: ions/parr_flow_t nspec mismatch");
    if (!ions_temp.empty() && ns1 != ns6)
      throw std::runtime_error("compute surface/physical/sputter: ions/temp nspec mismatch");
    nspec = ns1;
    has_multi_ion = (nspec > 0);
    if (hasDataset("ion_species/charge_state_z")) {
      H5::DataSet ds = file.openDataSet("ion_species/charge_state_z");
      H5::DataSpace sp = ds.getSpace();
      hsize_t d0 = 0;
      sp.getSimpleExtentDims(&d0);
      ion_charge_state_z.assign(static_cast<size_t>(nspec),1);
      const int nread = std::min(static_cast<int>(d0),nspec);
      if (nread > 0) {
        std::vector<int> tmp(static_cast<size_t>(d0),1);
        ds.read(tmp.data(), H5::PredType::NATIVE_INT);
        for (int i = 0; i < nread; i++) ion_charge_state_z[i] = tmp[i];
      }
    } else {
      // Fallback for current WEST ordering: slot1=D+, slot2..=O+..O8+
      ion_charge_state_z.assign(static_cast<size_t>(nspec),1);
      for (int i = 1; i < nspec; i++) ion_charge_state_z[i] = i;
    }
  } else {
    nspec = 1;
    ion_charge_state_z.assign(1,1);
  }
}

void ComputeSurfacePhysicalSputter::load_surface_data()
{
  H5::H5File file(surface_path, H5F_ACC_RDONLY);

  auto read1D = [&](const std::string& name, std::vector<double>& out, int &n) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t d0;
    sp.getSimpleExtentDims(&d0);
    out.resize(static_cast<size_t>(d0));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    n = static_cast<int>(d0);
  };
  auto read2D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != nE || static_cast<int>(dims[1]) != nA)
      throw std::runtime_error("compute surface/physical/sputter: spyld shape mismatch");
    out.resize(static_cast<size_t>(dims[0]*dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  read1D("E", E_axis, nE);
  read1D("A", A_axis, nA);
  if (nE < 2 || nA < 2)
    throw std::runtime_error("compute surface/physical/sputter: E/A axes must have size >= 2");
  read2D("spyld", spyld);
}

void ComputeSurfacePhysicalSputter::load_boundary()
{
  boundary_r.clear();
  boundary_z.clear();
  if (boundary_path.empty()) return;

  FILE *fp = fopen(boundary_path.c_str(), "r");
  if (!fp) error->all(FLERR, "compute surface/physical/sputter: cannot open boundary file");
  char line[256];
  while (fgets(line, sizeof(line), fp)) {
    if (line[0] == '#' || line[0] == '\n') continue;
    double r, z;
    if (sscanf(line, "%lf %lf", &r, &z) == 2) {
      boundary_r.push_back(r);
      boundary_z.push_back(z);
    }
  }
  fclose(fp);
  if (boundary_r.size() < 3)
    error->all(FLERR, "compute surface/physical/sputter: boundary polygon needs >= 3 points");
  if (comm->me == 0) {
    if (screen) fprintf(screen, "PMI boundary polygon: %d points\n",
                        static_cast<int>(boundary_r.size()));
    if (logfile) fprintf(logfile, "PMI boundary polygon: %d points\n",
                         static_cast<int>(boundary_r.size()));
  }
}

int ComputeSurfacePhysicalSputter::point_in_boundary(double r, double z) const
{
  // Ray-casting point-in-polygon test
  if (boundary_r.empty()) return 1;  // no boundary = all inside
  const int n = static_cast<int>(boundary_r.size());
  int inside = 0;
  for (int i = 0, j = n - 1; i < n; j = i++) {
    const double ri = boundary_r[i], zi = boundary_z[i];
    const double rj = boundary_r[j], zj = boundary_z[j];
    if (((zi > z) != (zj > z)) &&
        (r < (rj - ri) * (z - zi) / (zj - zi) + ri))
      inside = !inside;
  }
  return inside;
}

void ComputeSurfacePhysicalSputter::load_mesh()
{
  has_mesh = 0;
  mesh_nion = 0;
  try {
    H5::H5File file(plasma_path, H5F_ACC_RDONLY);
    // Silence HDF5 error stack for the existence check — H5Lexists
    // prints verbose diagnostics on some HDF5 versions when the group
    // does not exist.
    H5E_auto2_t old_func; void *old_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
    htri_t has = H5Lexists(file.getId(), "mesh/triangles", H5P_DEFAULT);
    H5Eset_auto2(H5E_DEFAULT, old_func, old_data);
    if (has <= 0) return;

    auto read1D_d = [&](const std::string &name, std::vector<double> &out) {
      H5::DataSet ds = file.openDataSet(name);
      hsize_t d0; ds.getSpace().getSimpleExtentDims(&d0);
      out.resize(static_cast<size_t>(d0));
      ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };
    auto read1D_i = [&](const std::string &name, std::vector<int> &out) {
      H5::DataSet ds = file.openDataSet(name);
      hsize_t d0; ds.getSpace().getSimpleExtentDims(&d0);
      out.resize(static_cast<size_t>(d0));
      ds.read(out.data(), H5::PredType::NATIVE_INT);
    };

    read1D_d("mesh/vtx_r", mesh_vtx_r);
    read1D_d("mesh/vtx_z", mesh_vtx_z);
    mesh_nvtx = static_cast<int>(mesh_vtx_r.size());

    // triangles: (ntri, 3) stored flat
    {
      H5::DataSet ds = file.openDataSet("mesh/triangles");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2]; sp.getSimpleExtentDims(dims);
      mesh_ntri = static_cast<int>(dims[0]);
      mesh_tri.resize(static_cast<size_t>(mesh_ntri * 3));
      ds.read(mesh_tri.data(), H5::PredType::NATIVE_INT);
    }
    read1D_i("mesh/cell_index", mesh_cell_idx);
    read1D_d("mesh/dens_e", mesh_ne);
    read1D_d("mesh/temp_e", mesh_te);
    read1D_d("mesh/dens_i", mesh_ni);
    read1D_d("mesh/temp_i", mesh_ti);
    if (H5Lexists(file.getId(), "mesh/parr_flow", H5P_DEFAULT) > 0)
      read1D_d("mesh/parr_flow", mesh_upar);
    mesh_ncell = static_cast<int>(mesh_ne.size());

    // Multi-ion mesh data (optional)
    if (H5Lexists(file.getId(), "mesh/ions/dens", H5P_DEFAULT) > 0) {
      H5::DataSet ds = file.openDataSet("mesh/ions/dens");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2]; sp.getSimpleExtentDims(dims);
      mesh_nion = static_cast<int>(dims[0]);
      mesh_ions_dens.resize(static_cast<size_t>(mesh_nion * mesh_ncell));
      ds.read(mesh_ions_dens.data(), H5::PredType::NATIVE_DOUBLE);
      if (H5Lexists(file.getId(), "mesh/ions/temp", H5P_DEFAULT) > 0) {
        mesh_ions_temp.resize(static_cast<size_t>(mesh_nion * mesh_ncell));
        file.openDataSet("mesh/ions/temp").read(mesh_ions_temp.data(), H5::PredType::NATIVE_DOUBLE);
      }
      if (H5Lexists(file.getId(), "mesh/ions/parr_flow", H5P_DEFAULT) > 0) {
        mesh_ions_upar.resize(static_cast<size_t>(mesh_nion * mesh_ncell));
        file.openDataSet("mesh/ions/parr_flow").read(mesh_ions_upar.data(), H5::PredType::NATIVE_DOUBLE);
      }
    }

    has_mesh = 1;
    rebuild_mesh_cache();

  } catch (...) {
    has_mesh = 0;
  }

  if (has_mesh && comm->me == 0) {
    if (screen) fprintf(screen, "PMI mesh interpolation: %d triangles, %d vertices, %d cells"
                        " (%d mapped)\n", mesh_ntri, mesh_nvtx, mesh_ncell,
                        static_cast<int>(mapped_idx.size()));
    if (logfile) fprintf(logfile, "PMI mesh interpolation: %d triangles, %d vertices, %d cells"
                         " (%d mapped)\n", mesh_ntri, mesh_nvtx, mesh_ncell,
                         static_cast<int>(mapped_idx.size()));
  }
}

int ComputeSurfacePhysicalSputter::find_mesh_triangle(double r, double z) const
{
  // Use spatial hash if available
  if (hash_nr > 0 && !hash_grid.empty()) {
    int ir = (int)((r - hash_rmin) / hash_dr);
    int iz = (int)((z - hash_zmin) / hash_dz);
    if (ir < 0 || ir >= hash_nr || iz < 0 || iz >= hash_nz) return -1;
    for (int t : hash_grid[iz * hash_nr + ir]) {
      const int v0 = mesh_tri[t*3+0], v1 = mesh_tri[t*3+1], v2 = mesh_tri[t*3+2];
      const double r0 = mesh_vtx_r[v0], z0 = mesh_vtx_z[v0];
      const double r1 = mesh_vtx_r[v1], z1 = mesh_vtx_z[v1];
      const double r2 = mesh_vtx_r[v2], z2 = mesh_vtx_z[v2];
      const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
      if (std::fabs(d) < 1e-30) continue;
      const double u = ((r-r0)*(z2-z0) - (r2-r0)*(z-z0)) / d;
      const double v = ((r1-r0)*(z-z0) - (r-r0)*(z1-z0)) / d;
      if (u >= -1e-10 && v >= -1e-10 && (u+v) <= 1.0+1e-10) return t;
    }
    return -1;
  }

  // Fallback: brute force
  for (int t = 0; t < mesh_ntri; t++) {
    if (r < mesh_tri_rmin[t] || r > mesh_tri_rmax[t] ||
        z < mesh_tri_zmin[t] || z > mesh_tri_zmax[t]) continue;
    const int v0 = mesh_tri[t*3+0], v1 = mesh_tri[t*3+1], v2 = mesh_tri[t*3+2];
    const double r0 = mesh_vtx_r[v0], z0 = mesh_vtx_z[v0];
    const double r1 = mesh_vtx_r[v1], z1 = mesh_vtx_z[v1];
    const double r2 = mesh_vtx_r[v2], z2 = mesh_vtx_z[v2];
    const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
    if (std::fabs(d) < 1e-30) continue;
    const double u = ((r-r0)*(z2-z0) - (r2-r0)*(z-z0)) / d;
    const double v = ((r1-r0)*(z-z0) - (r-r0)*(z1-z0)) / d;
    if (u >= -1e-10 && v >= -1e-10 && (u+v) <= 1.0+1e-10) return t;
  }
  return -1;
}

int ComputeSurfacePhysicalSputter::find_nearest_mapped_triangle(double r, double z,
                                                      double max_dist) const
{
  // Nearest-centroid fallback for wall points that fall in the vacuum gap
  // between the B2.5 plasma domain and the physical wall.  This is the
  // normal situation in SOLPS: Eirene fills the gap with unmapped triangles,
  // so strict point-in-triangle on mapped cells misses wall locations.
  const int nm = static_cast<int>(mapped_idx.size());
  double best_d2 = max_dist * max_dist;
  int best = -1;
  for (int i = 0; i < nm; i++) {
    const double dr = mapped_cr[i] - r;
    const double dz = mapped_cz[i] - z;
    const double d2 = dr*dr + dz*dz;
    if (d2 < best_d2) { best_d2 = d2; best = i; }
  }
  return (best >= 0) ? mapped_idx[best] : -1;
}

void ComputeSurfacePhysicalSputter::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute surface/physical/sputter when surfs do not exist");
  if (surf->implicit)
    error->all(FLERR,"Cannot use compute surface/physical/sputter with implicit surfs");
  if (api_new && background_fix_id.empty())
    error->all(FLERR,
        "compute surface/physical/sputter: target/projectiles API requires "
        "background <fix_id> (HDF5 file mode not supported)");

  cache_valid = 0;

  try {
    load_plasma();
  } catch (const std::exception& e) {
    error->all(FLERR, e.what());
  } catch (...) {
    error->all(FLERR, "compute surface/physical/sputter failed reading plasma file");
  }

  // Load B-field from separate file if not already available.
  // In background mode we stay file-free unless the user explicitly
  // supplies a bfield path override.
  int bfield_user_specified = !bfield_path.empty();
  if (background_fix_id.empty() && !has_bfield && bfield_path.empty() && !plasma_path.empty()) {
    std::string dir = plasma_path;
    size_t slash = dir.find_last_of('/');
    if (slash != std::string::npos)
      bfield_path = dir.substr(0, slash + 1) + "bfield.h5";
    else
      bfield_path = "bfield.h5";
  }
  if (!has_bfield && !bfield_path.empty()) {
    try {
      H5::H5File bf(bfield_path, H5F_ACC_RDONLY);
      auto read1D_bf = [&](const std::string& name, std::vector<double>& out) {
        H5::DataSet ds = bf.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t d0;
        sp.getSimpleExtentDims(&d0);
        out.resize(static_cast<size_t>(d0));
        ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
      };
      auto read2D_bf = [&](const std::string& name, std::vector<double>& out,
                           int nr_bf, int nz_bf) {
        H5::DataSet ds = bf.openDataSet(name);
        H5::DataSpace sp = ds.getSpace();
        hsize_t dims[2];
        sp.getSimpleExtentDims(dims);
        if (static_cast<int>(dims[0]) != nz_bf || static_cast<int>(dims[1]) != nr_bf)
          throw std::runtime_error("bfield HDF5 shape mismatch with plasma grid");
        out.resize(static_cast<size_t>(dims[0]*dims[1]));
        ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
      };
      std::vector<double> r_bf, z_bf;
      read1D_bf("r", r_bf);
      read1D_bf("z", z_bf);
      if (static_cast<int>(r_bf.size()) != nr || static_cast<int>(z_bf.size()) != nz)
        throw std::runtime_error("bfield r/z grid size does not match plasma grid");
      read2D_bf("br", br, nr, nz);
      if (H5Lexists(bf.getId(), "bt", H5P_DEFAULT) > 0)
        read2D_bf("bt", bt, nr, nz);
      read2D_bf("bz", bz, nr, nz);
      has_bfield = (!br.empty() && !bz.empty());
    } catch (...) {
      // If user explicitly specified bfield path, this is a fatal error
      if (bfield_user_specified)
        error->all(FLERR, "compute surface/physical/sputter: failed reading bfield file");
      // Otherwise auto-detect failed silently — bfield not available
    }
  }

  if (!eckstein_mode) {
    try {
      load_surface_data();
    } catch (const std::exception& e) {
      error->all(FLERR, e.what());
    } catch (...) {
      error->all(FLERR, "compute surface/physical/sputter failed reading surface file");
    }
  }

  load_boundary();
  if (background_fix_id.empty()) load_mesh();
  load_iead_if_requested();

  if (debug_interp && !eckstein_mode && comm->me == 0) {
    auto interp_yield_raw = [&](double e_eV, double a_deg)->double {
      const double ec = std::min(std::max(e_eV, E_axis.front()), E_axis.back());
      const double ac = std::min(std::max(a_deg, A_axis.front()), A_axis.back());
      auto itE = std::lower_bound(E_axis.begin(), E_axis.end(), ec);
      auto itA = std::lower_bound(A_axis.begin(), A_axis.end(), ac);
      int iE2 = static_cast<int>(itE - E_axis.begin());
      int iA2 = static_cast<int>(itA - A_axis.begin());
      if (iE2 <= 0) iE2 = 1;
      if (iA2 <= 0) iA2 = 1;
      if (iE2 >= nE) iE2 = nE-1;
      if (iA2 >= nA) iA2 = nA-1;
      const int iE1 = iE2-1;
      const int iA1 = iA2-1;
      const double E1 = E_axis[iE1], E2 = E_axis[iE2];
      const double A1 = A_axis[iA1], A2 = A_axis[iA2];
      const double tE = (E2 != E1) ? (ec-E1)/(E2-E1) : 0.0;
      const double tA = (A2 != A1) ? (ac-A1)/(A2-A1) : 0.0;
      const double w11 = (1.0-tE)*(1.0-tA);
      const double w21 = tE*(1.0-tA);
      const double w12 = (1.0-tE)*tA;
      const double w22 = tE*tA;
      auto at = [&](int ie, int ia)->double { return spyld[static_cast<size_t>(ie)*nA + ia]; };
      return w11*at(iE1,iA1) + w21*at(iE2,iA1) + w12*at(iE1,iA2) + w22*at(iE2,iA2);
    };
    for (size_t k = 0; k < debug_E.size(); k++) {
      const double e = debug_E[k];
      const double a = debug_A[k];
      const double yraw = interp_yield_raw(e,a);
      const double yclamp = interp_yield(e,a);
      if (screen) fprintf(screen,"PMI debug_interp: E=%g eV angle=%g deg -> spyld_raw=%g spyld_clamped=%g\n",e,a,yraw,yclamp);
      if (logfile) fprintf(logfile,"PMI debug_interp: E=%g eV angle=%g deg -> spyld_raw=%g spyld_clamped=%g\n",e,a,yraw,yclamp);
    }
  }

  // Derive B-field from equilibrium file if still missing
  if (!has_bfield && !equ_path.empty()) {
    try {
      load_bfield_from_equ();
    } catch (const std::exception& e) {
      error->all(FLERR, e.what());
    }
  }

  if (!has_bfield)
    error->all(FLERR,
      "compute surface/physical/sputter: need B-field via plasma HDF5 (br/bz), "
      "bfield HDF5, or equilibrium keyword");

  distributed = surf->distributed;
  if (!firstflag) return;
  firstflag = 0;

  nsown = surf->nown;
  if (nvalue == 1)
    memory->create(vector_surf,nsown,"surface/physical/sputter:vector_surf");
  else
    memory->create(array_surf,nsown,nvalue,"surface/physical/sputter:array_surf");
}

double ComputeSurfacePhysicalSputter::interp2D(const std::vector<double> &f, double r, double z) const
{
  if (f.empty()) return 0.0;
  // Return zero for points outside the plasma grid — do not clamp.
  if (r < rvals.front() || r > rvals.back() ||
      z < zvals.front() || z > zvals.back()) return 0.0;
  const double rc = r;
  const double zc = z;
  auto itR = std::lower_bound(rvals.begin(), rvals.end(), rc);
  auto itZ = std::lower_bound(zvals.begin(), zvals.end(), zc);
  int ir2 = static_cast<int>(itR - rvals.begin());
  int iz2 = static_cast<int>(itZ - zvals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (iz2 <= 0) iz2 = 1;
  if (ir2 >= nr) ir2 = nr-1;
  if (iz2 >= nz) iz2 = nz-1;
  const int ir1 = ir2-1;
  const int iz1 = iz2-1;

  const double r1 = rvals[ir1], r2 = rvals[ir2];
  const double z1 = zvals[iz1], z2 = zvals[iz2];
  const double tr = (r2 != r1) ? (rc-r1)/(r2-r1) : 0.0;
  const double tz = (z2 != z1) ? (zc-z1)/(z2-z1) : 0.0;
  const double w11 = (1.0-tr)*(1.0-tz);
  const double w21 = tr*(1.0-tz);
  const double w12 = (1.0-tr)*tz;
  const double w22 = tr*tz;

  auto at2 = [&](int iz, int ir)->double { return f[static_cast<size_t>(iz)*nr + ir]; };
  return w11*at2(iz1,ir1) + w21*at2(iz1,ir2) + w12*at2(iz2,ir1) + w22*at2(iz2,ir2);
}

double ComputeSurfacePhysicalSputter::interp3D(const std::vector<double> &f, int ispec, double r, double z) const
{
  if (f.empty() || ispec < 0 || ispec >= nspec) return 0.0;
  if (r < rvals.front() || r > rvals.back() ||
      z < zvals.front() || z > zvals.back()) return 0.0;
  const double rc = r;
  const double zc = z;
  auto itR = std::lower_bound(rvals.begin(), rvals.end(), rc);
  auto itZ = std::lower_bound(zvals.begin(), zvals.end(), zc);
  int ir2 = static_cast<int>(itR - rvals.begin());
  int iz2 = static_cast<int>(itZ - zvals.begin());
  if (ir2 <= 0) ir2 = 1;
  if (iz2 <= 0) iz2 = 1;
  if (ir2 >= nr) ir2 = nr-1;
  if (iz2 >= nz) iz2 = nz-1;
  const int ir1 = ir2-1;
  const int iz1 = iz2-1;

  const double r1 = rvals[ir1], r2 = rvals[ir2];
  const double z1 = zvals[iz1], z2 = zvals[iz2];
  const double tr = (r2 != r1) ? (rc-r1)/(r2-r1) : 0.0;
  const double tz = (z2 != z1) ? (zc-z1)/(z2-z1) : 0.0;
  const double w11 = (1.0-tr)*(1.0-tz);
  const double w21 = tr*(1.0-tz);
  const double w12 = (1.0-tr)*tz;
  const double w22 = tr*tz;

  const size_t off = static_cast<size_t>(ispec) * nz * nr;
  auto at3 = [&](int iz, int ir)->double { return f[off + static_cast<size_t>(iz)*nr + ir]; };
  return w11*at3(iz1,ir1) + w21*at3(iz1,ir2) + w12*at3(iz2,ir1) + w22*at3(iz2,ir2);
}

double ComputeSurfacePhysicalSputter::interp_yield(double e_eV, double a_deg) const
{
  if (spyld.empty()) return 0.0;
  const double ec = std::min(std::max(e_eV, E_axis.front()), E_axis.back());
  const double ac = std::min(std::max(a_deg, A_axis.front()), A_axis.back());
  auto itE = std::lower_bound(E_axis.begin(), E_axis.end(), ec);
  auto itA = std::lower_bound(A_axis.begin(), A_axis.end(), ac);
  int iE2 = static_cast<int>(itE - E_axis.begin());
  int iA2 = static_cast<int>(itA - A_axis.begin());
  if (iE2 <= 0) iE2 = 1;
  if (iA2 <= 0) iA2 = 1;
  if (iE2 >= nE) iE2 = nE-1;
  if (iA2 >= nA) iA2 = nA-1;
  const int iE1 = iE2-1;
  const int iA1 = iA2-1;
  const double E1 = E_axis[iE1], E2 = E_axis[iE2];
  const double A1 = A_axis[iA1], A2 = A_axis[iA2];
  const double tE = (E2 != E1) ? (ec-E1)/(E2-E1) : 0.0;
  const double tA = (A2 != A1) ? (ac-A1)/(A2-A1) : 0.0;
  const double w11 = (1.0-tE)*(1.0-tA);
  const double w21 = tE*(1.0-tA);
  const double w12 = (1.0-tE)*tA;
  const double w22 = tE*tA;
  auto at = [&](int ie, int ia)->double {
    return spyld[static_cast<size_t>(ie)*nA + ia];
  };
  double y = w11*at(iE1,iA1) + w21*at(iE2,iA1) + w12*at(iE1,iA2) + w22*at(iE2,iA2);
  if (!std::isfinite(y) || y < 0.0) y = 0.0;
  return y;
}

void ComputeSurfacePhysicalSputter::compute_per_surf()
{
  // Physical constants
  const double QE  = 1.602176634e-19;   // elementary charge [C]
  const double AMU = 1.66053906660e-27;  // atomic mass unit [kg]
  const double ME  = 9.1093837015e-31;   // electron mass [kg]

  invoked_per_surf = update->ntimestep;
  if (static_cache && cache_valid) return;
  if (nvalue == 1) {
    for (int i = 0; i < nsown; i++) vector_surf[i] = 0.0;
  } else {
    for (int i = 0; i < nsown; i++)
      for (int j = 0; j < nvalue; j++) array_surf[i][j] = 0.0;
  }

  Surf::Line *lines = distributed ? surf->mylines : surf->lines;
  Surf::Tri *tris = distributed ? surf->mytris : surf->tris;
  const int me = comm->me;
  const int nprocs = comm->nprocs;

  const int ns = has_multi_ion ? nspec : 1;
  std::vector<double> gamma_n(ns,0.0), angle_deg(ns,0.0), energy_eV(ns,0.0), yld(ns,0.0), sput_flux(ns,0.0);

  // Yield dispatch: analytic Eckstein or HDF5 table.
  Eckstein::SputterParams eck_p;
  if (eckstein_mode) {
    eck_p.Z1 = eck_Z1; eck_p.M1 = eck_M1; eck_p.Z2 = eck_Z2; eck_p.M2 = eck_M2;
    eck_p.Es = eck_Es; eck_p.Eth = eck_Eth; eck_p.Q = eck_Q; eck_p.ETF = eck_ETF;
  }
  auto yield_lookup = [&](double E_eV, double a_deg) -> double {
    if (eckstein_mode)
      return Eckstein::sputter_yield(E_eV, a_deg, eck_p);
    return interp_yield(E_eV, a_deg);
  };

  auto clean = [](double x) {
    return (std::fabs(x) < 1.0e-200 || !std::isfinite(x)) ? 0.0 : x;
  };

  for (int i = 0; i < nsown; i++) {
    int m = distributed ? i : me + i*nprocs;
    const int in_group = (dimension == 2) ? (lines[m].mask & groupbit) : (tris[m].mask & groupbit);
    if (!in_group) continue;

    // Surface element centroid in cylindrical (R,Z). Helper picks the
    // SPARTA slot mapping (Cart 2D x=R, axi 2D x=Z, 3D Cartesian).
    double r = 0.0, z = 0.0;
    if (dimension == 2) {
      const double mid[3] = {0.5*(lines[m].p1[0] + lines[m].p2[0]),
                             0.5*(lines[m].p1[1] + lines[m].p2[1]), 0.0};
      OpenEdge::sparta_to_RZ(mid, dimension, domain->axisymmetric, r, z,
                             column_x0, column_y0);
    } else {
      const double mid[3] = {(tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0,
                             (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0,
                             (tris[m].p1[2] + tris[m].p2[2] + tris[m].p3[2]) / 3.0};
      OpenEdge::sparta_to_RZ(mid, dimension, domain->axisymmetric, r, z,
                             column_x0, column_y0);
    }

    // Skip surface elements outside SOLPS domain.
    // When mesh triangulation is loaded, the point-in-triangle test
    // inherently limits to the SOLPS domain.  The boundary polygon is
    // a fallback for rectangular-grid mode.
    if (!has_mesh && !point_in_boundary(r, z)) continue;

    // Surface normal in cylindrical (R, phi, Z). Use helper to invert
    // the SPARTA slot layout: in axi mode SPARTA's normal[0] is the
    // axial component, normal[1] the radial.
    double nr_surf = 0.0, nt_surf = 0.0, nz_surf = 0.0;
    if (dimension == 2) {
      OpenEdge::sparta_v_to_RZphi(lines[m].norm, dimension,
                                   domain->axisymmetric, 0.0,
                                   nr_surf, nz_surf, nt_surf);
    } else {
      const double xc = (tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0;
      const double yc = (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0;
      const double rr = std::sqrt(xc*xc + yc*yc);
      const double cr = (rr > 0.0) ? xc/rr : 1.0;
      const double sr = (rr > 0.0) ? yc/rr : 0.0;
      nr_surf =  tris[m].norm[0]*cr + tris[m].norm[1]*sr;
      nt_surf = -tris[m].norm[0]*sr + tris[m].norm[1]*cr;
      nz_surf =  tris[m].norm[2];
    }

    // Resolve plasma data: mesh-based (exact SOLPS cell) or grid-based (bilinear).
    // Computed BEFORE the B-lookup so mesh-native plasma.h5 (no top-level
    // br/bz; B stored per-triangle in mesh_tri_b*) can use tri_idx for both
    // per-triangle B and per-cell plasma.
    int mesh_cell = -1;
    int tri_idx = -1;
    if (has_mesh) {
      // Always locate the triangle containing (or nearest to) the wall
      // centroid; we need it for per-triangle B-field interpolation
      // regardless of which path supplies the plasma cell.
      tri_idx = find_mesh_triangle(r, z);
      if (tri_idx < 0 || mesh_cell_idx[tri_idx] < 0)
        tri_idx = find_nearest_mapped_triangle(r, z, 0.05);

      // Plasma cell: prefer the converter-supplied wall_surf_cell map,
      // which identifies the SOLPS cell *adjacent* to each wall segment
      // (matching what SOLPS calls the target volume cell). Falls back
      // to the triangle's own cell_index when wall_surf_cell isn't
      // available, or for non-wall surfaces.
      if (has_mesh_wall_surf_cell) {
        const int sid = (dimension == 2)
          ? static_cast<int>(lines[m].id)
          : static_cast<int>(tris[m].id);
        const int wid = sid - 1;        // wall_surf_cell is 0-based by line idx
        if (wid >= 0 && wid < static_cast<int>(mesh_wall_surf_cell.size())) {
          const int wcell = mesh_wall_surf_cell[wid];
          if (wcell >= 0) mesh_cell = wcell;
        }
      }
      if (mesh_cell < 0) {
        if (tri_idx >= 0) mesh_cell = mesh_cell_idx[tri_idx];
        else continue;  // genuinely outside SOLPS domain — zero flux
      }
    }

    // B-field at surface centroid (cylindrical components). Prefer the
    // per-triangle B from fix background (mesh-native plasma.h5); fall
    // back to interp2D on the legacy regular-grid br/bz/bt arrays.
    double br_loc = 0.0, bt_loc = 0.0, bz_loc = 0.0;
    if (!mesh_tri_br.empty() && tri_idx >= 0 && tri_idx < mesh_ntri) {
      br_loc = mesh_tri_br[tri_idx];
      bz_loc = mesh_tri_bz[tri_idx];
      bt_loc = mesh_tri_bt.empty() ? 0.0 : mesh_tri_bt[tri_idx];
    } else {
      br_loc = interp2D(br, r, z);
      bt_loc = bt.empty() ? 0.0 : interp2D(bt, r, z);
      bz_loc = interp2D(bz, r, z);
    }
    // Prefer the smooth /wall_flux/ B-field (IDW K=3 over the same
    // source samples used for gamma and Te/Ti) when available. Otherwise
    // use the per-triangle B (vertex-averaged in fix background) which
    // can have cell-to-cell jitter that propagates into Y(E,theta) near
    // the sputter threshold.
    if (wall_flux_n > 0 && !wall_flux_b_r.empty() && !wall_flux_b_z.empty()) {
      constexpr int KNN = 3;
      double dbest2[KNN]; int ibest[KNN];
      for (int k = 0; k < KNN; k++) { dbest2[k] = 1e300; ibest[k] = -1; }
      const double dmax2 = wall_flux_cutoff * wall_flux_cutoff;
      for (int i = 0; i < wall_flux_n; i++) {
        const double dr = wall_flux_r[i] - r;
        const double dz = wall_flux_z[i] - z;
        const double d2 = dr*dr + dz*dz;
        if (d2 > dmax2) continue;
        if (d2 >= dbest2[KNN-1]) continue;
        int p = KNN - 1;
        while (p > 0 && d2 < dbest2[p-1]) {
          dbest2[p] = dbest2[p-1];
          ibest[p]  = ibest[p-1];
          p--;
        }
        dbest2[p] = d2;
        ibest[p]  = i;
      }
      double wsum = 0.0, br_sum = 0.0, bz_sum = 0.0, bt_sum = 0.0;
      const bool have_bt = !wall_flux_b_t.empty();
      for (int k = 0; k < KNN; k++) {
        if (ibest[k] < 0) break;
        const double d = std::sqrt(dbest2[k]);
        const double w = (d < 1.0e-12) ? 1.0e12 : 1.0 / d;
        wsum += w;
        br_sum += w * wall_flux_b_r[ibest[k]];
        bz_sum += w * wall_flux_b_z[ibest[k]];
        if (have_bt) bt_sum += w * wall_flux_b_t[ibest[k]];
      }
      if (wsum > 0.0) {
        br_loc = br_sum / wsum;
        bz_loc = bz_sum / wsum;
        if (have_bt) bt_loc = bt_sum / wsum;
      }
    }
    const double bmag = std::sqrt(br_loc*br_loc + bt_loc*bt_loc + bz_loc*bz_loc);

    // B-field incidence angle on wall surface.
    // Use only poloidal field (Br, Bz) dotted with the poloidal surface
    // normal (nr, nz).  The toroidal field Bt is tangent to flux surfaces
    // and does not drive particles into the wall.  Including Bt would
    // create a spurious dependence on the toroidal triangulation angle.
    const double bp_dot_n = br_loc * nr_surf + bz_loc * nz_surf;
    double sin_alpha = 0.0;
    if (bmag > 0.0)
      sin_alpha = std::fabs(bp_dot_n) / bmag;

    // Per-species density/temperature lookup
    auto species_plasma = [&](int s, double &ni, double &ti) {
      ni = 0.0; ti = 0.0;
      if (mesh_cell >= 0) {
        // Direct SOLPS cell data — no interpolation artifacts
        if (mesh_nion > 0 && s < mesh_nion) {
          const size_t off = static_cast<size_t>(s) * mesh_ncell + mesh_cell;
          ni = mesh_ions_dens[off];
          if (!mesh_ions_temp.empty()) ti = mesh_ions_temp[off];
        } else {
          ni = mesh_ni[mesh_cell];
          ti = mesh_ti[mesh_cell];
        }
      } else if (has_multi_ion) {
        ni = interp3D(ions_dens, s, r, z);
        if (!ions_temp.empty()) ti = interp3D(ions_temp, s, r, z);
      } else {
        ni = interp2D(dens_i, r, z);
        if (!temp_i.empty()) ti = interp2D(temp_i, r, z);
      }
    };

    std::fill(gamma_n.begin(), gamma_n.end(), 0.0);
    std::fill(angle_deg.begin(), angle_deg.end(), 0.0);
    std::fill(energy_eV.begin(), energy_eV.end(), 0.0);
    std::fill(yld.begin(), yld.end(), 0.0);
    std::fill(sput_flux.begin(), sput_flux.end(), 0.0);

    double sput_total = 0.0;

    // Te, Ti at the wall: prefer the smooth /wall_flux/ scatter (single
    // nearest neighbor within wall_flux_cutoff) when present. Otherwise
    // fall back to the wall-adjacent volume cell. Smooth Te/Ti are
    // critical for the yield path because Y(E,theta) is steep near the
    // sputter threshold; jaggedness in mesh_te[mesh_cell] (cell-to-cell
    // jumps) amplifies into Gamma_erode through Eckstein.
    double te_loc = 0.0, ti_bulk = 0.0;
    bool wf_te_ti_ok = false;
    if (wall_flux_n > 0 && !wall_flux_te.empty() && !wall_flux_ti.empty()) {
      // K=3 IDW (same kernel used for gamma_i above). Single-NN gives
      // stair-step plateaus when SPARTA wall is finer than the
      // wall_flux source — that pumps right back into Y(E,theta)
      // jaggedness near the sputter threshold.
      constexpr int KNN = 3;
      double dbest2[KNN]; int ibest[KNN];
      for (int k = 0; k < KNN; k++) { dbest2[k] = 1e300; ibest[k] = -1; }
      const double dmax2 = wall_flux_cutoff * wall_flux_cutoff;
      for (int i = 0; i < wall_flux_n; i++) {
        const double dr = wall_flux_r[i] - r;
        const double dz = wall_flux_z[i] - z;
        const double d2 = dr*dr + dz*dz;
        if (d2 > dmax2) continue;
        if (d2 >= dbest2[KNN-1]) continue;
        int p = KNN - 1;
        while (p > 0 && d2 < dbest2[p-1]) {
          dbest2[p] = dbest2[p-1];
          ibest[p]  = ibest[p-1];
          p--;
        }
        dbest2[p] = d2;
        ibest[p]  = i;
      }
      double wsum = 0.0, te_sum = 0.0, ti_sum = 0.0;
      for (int k = 0; k < KNN; k++) {
        if (ibest[k] < 0) break;
        const double d = std::sqrt(dbest2[k]);
        const double w = (d < 1.0e-12) ? 1.0e12 : 1.0 / d;
        wsum += w;
        te_sum += w * wall_flux_te[ibest[k]];
        ti_sum += w * wall_flux_ti[ibest[k]];
      }
      if (wsum > 0.0) {
        te_loc  = te_sum / wsum;
        ti_bulk = ti_sum / wsum;
        wf_te_ti_ok = true;
      }
    }
    if (!wf_te_ti_ok) {
      te_loc = (mesh_cell >= 0) ? mesh_te[mesh_cell]
                                : interp2D(temp_e, r, z);
      // ti_bulk remains 0.0 here; per-species ti from species_plasma()
      // will be used in the loop below in the legacy path.
    }

    for (int s = 0; s < ns; s++) {
      double ni, ti;
      species_plasma(s, ni, ti);
      if (ni <= 0.0) continue;

      const int z_inc = (s >= 0 && s < static_cast<int>(ion_charge_state_z.size())) ?
        std::max(1,ion_charge_state_z[s]) : 1;
      const double te_eV = (std::isfinite(te_loc) && te_loc > 0.0) ? te_loc : 0.0;
      // Prefer the bulk Ti from /wall_flux/ over the per-species ti
      // from the volume cell when wall_flux Te/Ti are loaded. Bulk Ti
      // is smoother (one nearest-source value per segment vs. cell-to-
      // cell jumps from mesh_ions_temp). Per-charge-state Ti differences
      // at the wall are small compared to the threshold-region steepness
      // of Y(E,theta), so the smoothness gain dominates the accuracy hit.
      const double ti_for_E = wf_te_ti_ok ? ti_bulk : ti;
      const double ti_eV = (std::isfinite(ti_for_E) && ti_for_E > 0.0) ? ti_for_E : 0.0;

      // Per-species wall-face flux. When /wall_flux/ is present in
      // plasma.h5, query the scatter at the segment centroid (R, Z)
      // via inverse-distance weighting over the K=3 nearest source
      // points within wall_flux_cutoff. Otherwise fall back to the
      // Bohm reconstruction Gamma = n_i * c_s * sin(alpha_B) at the
      // wall-adjacent volume cell.
      double g = 0.0;
      if (wall_flux_n > 0 && s < wall_flux_nspec) {
        // K=3 IDW; sources are scattered along the source-side wall.
        constexpr int KNN = 3;
        double dbest2[KNN]; int ibest[KNN];
        for (int k = 0; k < KNN; k++) { dbest2[k] = 1e300; ibest[k] = -1; }
        const double dmax2 = wall_flux_cutoff * wall_flux_cutoff;
        for (int i = 0; i < wall_flux_n; i++) {
          const double dr = wall_flux_r[i] - r;
          const double dz = wall_flux_z[i] - z;
          const double d2 = dr*dr + dz*dz;
          if (d2 > dmax2) continue;
          if (d2 >= dbest2[KNN-1]) continue;
          int p = KNN - 1;
          while (p > 0 && d2 < dbest2[p-1]) {
            dbest2[p] = dbest2[p-1];
            ibest[p]  = ibest[p-1];
            p--;
          }
          dbest2[p] = d2;
          ibest[p]  = i;
        }
        double wsum = 0.0, gsum = 0.0;
        for (int k = 0; k < KNN; k++) {
          if (ibest[k] < 0) break;
          const double d = std::sqrt(dbest2[k]);
          const double w = (d < 1.0e-12) ? 1.0e12 : 1.0 / d;
          wsum += w;
          gsum += w * wall_flux_gamma_i[
              static_cast<size_t>(s) * wall_flux_n + ibest[k]];
        }
        if (wsum > 0.0) {
          const double gw = gsum / wsum;
          g = (gw > 0.0) ? gw : 0.0;     // outflow only feeds sputter
        } else {
          // No source within cutoff: fall back to Bohm if we have it.
          const double cs_arg = (te_eV + ti_eV) * QE / (mass_amu * AMU);
          const double cs = (cs_arg > 0.0) ? std::sqrt(cs_arg) : 0.0;
          g = ni * cs * sin_alpha;
        }
      } else {
        // Legacy Bohm: cs = sqrt((Te + Ti) / mi); Stangeby ch. 2.
        const double cs_arg = (te_eV + ti_eV) * QE / (mass_amu * AMU);
        const double cs = (cs_arg > 0.0) ? std::sqrt(cs_arg) : 0.0;
        g = ni * cs * sin_alpha;
      }
      gamma_n[s] = g;

      // alpha_B = grazing angle from surface plane (diagnostic output).
      // sin_alpha = |B.n|/|B| = cos(theta_from_normal) = sin(alpha_B).
      const double a_deg = std::asin(std::min(1.0, sin_alpha)) * 180.0 / M_PI;
      angle_deg[s] = a_deg;
      // Yamamura (EIRENE COSIN convention) expects theta measured from the
      // surface normal: cos(theta_deg) must go to 1 at normal incidence.
      // theta_from_normal = 90 - alpha_B.
      const double theta_deg = 90.0 - a_deg;

      // Chankin 2014 eq. (4): mean ion impact energy on divertor target.
      //   <E> = 2*Ti + Z * |e*dpsi|, with
      //   |e*dpsi|/Te = 0.5 * ln( (mi_bg / (2*pi*me)) / (1 + Ti/Te) )
      // mi_bg is the *background* ion mass (sets the sheath); uses mass_amu
      // (default D = 2.014 amu). Do NOT use the impurity mass here.
      // For D+, Ti=Te this reduces to |e*dpsi| = 2.84*Te.
      const double mi_over_me_bg = (mass_amu * AMU) / ME;
      const double ti_ratio = ti_eV / std::max(te_eV, 1.0e-30);
      const double psi_over_Te = 0.5 *
        std::log(mi_over_me_bg / (2.0 * M_PI * (1.0 + ti_ratio)));
      double E = 2.0 * ti_eV +
                 static_cast<double>(z_inc) * psi_over_Te * te_eV;
      energy_eV[s] = E;

      if (in_projectile_slots(s+1)) {
        double ys = 0.0;
        // Decide whether the IEAD distribution-weighted yield is
        // applicable for this projectile slot (falls through to the
        // mean-impact path otherwise — Z outside grid, table absent,
        // or no table for this projectile element).
        IeadTable *tbl = iead_for_slot(s);
        const bool use_iead = tbl &&
            tbl->find_z_slot(z_inc) >= 0 && te_eV > 0.0;
        const double tau_local = ti_eV / std::max(te_eV, 1.0e-30);
        const double psi_local = theta_deg;   // angle from wall normal

        if (api_new && s < static_cast<int>(slot_to_table.size()) &&
            slot_to_table[s] >= 0) {
          // Per-slot Eckstein: pick coefficients by the projectile element
          // that populates plasma slot s (set at init by
          // resolve_projectile_tables()).
          const int ti = slot_to_table[s];
          Eckstein::SputterParams p;
          p.Z1 = per_proj_Z1[ti];  p.M1 = per_proj_M1[ti];
          p.Z2 = per_proj_Z2[ti];  p.M2 = per_proj_M2[ti];
          p.Es = per_proj_Es[ti];  p.Eth = per_proj_Eth[ti];
          p.Q  = per_proj_Q[ti];   p.ETF = per_proj_ETF[ti];
          if (use_iead) {
            ys = tbl->convolve_yield(
                tau_local, psi_local, z_inc, te_eV,
                [&](double e_eV, double th_deg) {
                  return Eckstein::sputter_yield(e_eV, th_deg, p);
                });
          } else {
            ys = Eckstein::sputter_yield(E, theta_deg, p);
          }
        } else {
          if (use_iead) {
            ys = tbl->convolve_yield(
                tau_local, psi_local, z_inc, te_eV,
                [&](double e_eV, double th_deg) {
                  return yield_lookup(e_eV, th_deg);
                });
          } else {
            ys = yield_lookup(E, theta_deg);
          }
        }
        yld[s] = ys;
        sput_flux[s] = g * ys;
        sput_total += sput_flux[s];
      }
    }

    // Virtual background impurity sputtering (not in plasma.h5)
    // Uses n_e from the plasma to synthesize impurity density at each wall element.
    if (has_impurity) {
      // Get n_e and Te, Ti at this wall element
      double ne_loc = 0.0;
      if (mesh_cell >= 0) {
        ne_loc = mesh_ne.empty() ? 0.0 : mesh_ne[mesh_cell];
      } else {
        ne_loc = interp2D(dens_i, r, z);  // n_e ≈ n_i (quasi-neutrality)
      }
      if (ne_loc > 0.0) {
        for (int iz = 0; iz < imp_Zmax; iz++) {
          const int Z = iz + 1;
          const double fz = imp_charge_fracs[iz];
          if (fz <= 0.0) continue;

          // Impurity density for this charge state
          const double n_imp = imp_frac * fz * ne_loc;

          // Use Ti if available; fall back to Te
          double ti_imp = te_loc;
          if (mesh_cell >= 0 && !mesh_ti.empty()) ti_imp = mesh_ti[mesh_cell];
          else if (!temp_i.empty()) ti_imp = interp2D(temp_i, r, z);

          // Chankin 2014 eq. (4): see main loop above for derivation.
          // mi_bg = mass_amu (background D+), NOT imp_mass_amu.
          const double mi_over_me_bg = (mass_amu * AMU) / ME;
          const double ti_ratio = ti_imp / std::max(te_loc, 1.0e-30);
          const double psi_over_Te = 0.5 *
            std::log(mi_over_me_bg / (2.0 * M_PI * (1.0 + ti_ratio)));
          const double E_eck = 2.0 * ti_imp +
                               static_cast<double>(Z) * psi_over_Te * te_loc;

          // Isothermal ion sound speed with impurity mass.
          const double cs_arg = (te_loc + ti_imp) * QE / (imp_mass_amu * AMU);
          const double cs = (cs_arg > 0.0) ? std::sqrt(cs_arg) : 0.0;

          // Bohm flux
          const double g_imp = n_imp * cs * sin_alpha;

          // Sputter yield: analytic Eckstein or HDF5 table.
          // Yamamura theta measured from the surface normal (see main loop).
          const double cos_theta = std::min(1.0, std::max(0.0, sin_alpha));
          const double theta_imp_deg = std::acos(cos_theta) * 180.0 / M_PI;
          const double ys = yield_lookup(E_eck, theta_imp_deg);
          sput_total += g_imp * ys;
        }
      }
    }

    // Axisymmetric ring area of this surface element [m^2].
    // 2D line: 2*pi*R_mid * segment_length  (truncated cone lateral area)
    // 3D tri : physical triangle area (0.5 |e1 x e2|)
    double ring_area = 0.0;
    if (dimension == 2) {
      double R1, Z1, R2, Z2;
      OpenEdge::sparta_to_RZ(lines[m].p1, dimension, domain->axisymmetric,
                              R1, Z1, column_x0, column_y0);
      OpenEdge::sparta_to_RZ(lines[m].p2, dimension, domain->axisymmetric,
                              R2, Z2, column_x0, column_y0);
      const double seg = std::sqrt((R2-R1)*(R2-R1) + (Z2-Z1)*(Z2-Z1));
      const double R_mid = 0.5 * (R1 + R2);
      ring_area = 2.0 * M_PI * R_mid * seg;
    } else {
      const double *p1 = tris[m].p1;
      const double *p2 = tris[m].p2;
      const double *p3 = tris[m].p3;
      const double e1x = p2[0]-p1[0], e1y = p2[1]-p1[1], e1z = p2[2]-p1[2];
      const double e2x = p3[0]-p1[0], e2y = p3[1]-p1[1], e2z = p3[2]-p1[2];
      const double cx = e1y*e2z - e1z*e2y;
      const double cy = e1z*e2x - e1x*e2z;
      const double cz = e1x*e2y - e1y*e2x;
      ring_area = 0.5 * std::sqrt(cx*cx + cy*cy + cz*cz);
    }
    const double sput_rate = sput_total * ring_area;

    if (nvalue == 1) {
      double out = 0.0;
      if (which[0] == SPUTTER_FLUX_TOTAL) out = sput_total;
      else if (which[0] == SPUTTER_RATE_TOTAL) out = sput_rate;
      else {
        const int slot = which_species[0] - 1;
        if (slot >= 0 && slot < ns) {
          if (which[0] == NFLUX_SPECIES) out = gamma_n[slot];
          else if (which[0] == INCIDENT_ANGLE_SPECIES) out = angle_deg[slot];
          else if (which[0] == INCIDENT_ENERGY_SPECIES) out = energy_eV[slot];
          else if (which[0] == SPUTTER_YIELD_SPECIES) out = yld[slot];
          else if (which[0] == SPUTTER_FLUX_SPECIES) out = sput_flux[slot];
        }
      }
      vector_surf[i] = clean(out);
    } else {
      for (int j = 0; j < nvalue; j++) {
        double out = 0.0;
        if (which[j] == SPUTTER_FLUX_TOTAL) out = sput_total;
        else if (which[j] == SPUTTER_RATE_TOTAL) out = sput_rate;
        else {
          const int slot = which_species[j] - 1;
          if (slot >= 0 && slot < ns) {
            if (which[j] == NFLUX_SPECIES) out = gamma_n[slot];
            else if (which[j] == INCIDENT_ANGLE_SPECIES) out = angle_deg[slot];
            else if (which[j] == INCIDENT_ENERGY_SPECIES) out = energy_eV[slot];
            else if (which[j] == SPUTTER_YIELD_SPECIES) out = yld[slot];
            else if (which[j] == SPUTTER_FLUX_SPECIES) out = sput_flux[slot];
          }
        }
        array_surf[i][j] = clean(out);
      }
    }
  }

  if (static_cache) cache_valid = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeSurfacePhysicalSputter::load_bfield_from_equ()
{
  // Read SOLPS/EQDSK-style equilibrium file and derive B-field on the
  // plasma (R,Z) grid.
  //
  //   Br = -(1/R) dpsi/dZ
  //   Bz =  (1/R) dpsi/dR
  //   Bt =  btf * rtf / R

  std::ifstream ifs(equ_path);
  if (!ifs.good())
    throw std::runtime_error(
        std::string("compute surface/physical/sputter: cannot open equilibrium '")
        + equ_path + "'");
  std::string text((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
  ifs.close();

  auto parse_int = [&](const char *name) -> int {
    size_t pos = text.find(std::string(name));
    while (pos != std::string::npos) {
      size_t eq = text.find('=', pos);
      if (eq != std::string::npos && eq - pos < 20) {
        int val = atoi(text.c_str() + eq + 1);
        if (val > 0) return val;
      }
      pos = text.find(std::string(name), pos + 1);
    }
    return 0;
  };
  auto parse_double = [&](const char *name) -> double {
    size_t pos = text.find(std::string(name));
    if (pos != std::string::npos) {
      size_t eq = text.find('=', pos);
      if (eq != std::string::npos) return atof(text.c_str() + eq + 1);
    }
    return 0.0;
  };
  auto read_floats_after = [&](const std::string &marker, int n,
                               std::vector<double> &out) {
    size_t pos = text.find(marker);
    if (pos == std::string::npos)
      throw std::runtime_error("cannot find '" + marker + "' in " + equ_path);
    pos += marker.size();
    out.clear();
    out.reserve(n);
    const char *c = text.c_str() + pos;
    const char *end = text.c_str() + text.size();
    while ((int)out.size() < n && c < end) {
      while (c < end && !std::isdigit(*c) && *c!='+' && *c!='-' && *c!='.') c++;
      if (c >= end) break;
      char *endp;
      double val = strtod(c, &endp);
      if (endp > c) { out.push_back(val); c = endp; }
      else c++;
    }
    if ((int)out.size() < n)
      throw std::runtime_error("not enough values after '" + marker + "'");
  };

  int jm = parse_int("jm");
  int km = parse_int("km");
  if (jm < 2 || km < 2)
    throw std::runtime_error("cannot parse jm/km from equilibrium");

  double btf = parse_double("btf");
  double rtf = parse_double("rtf");
  double psib = parse_double("psib");

  std::vector<double> r_eq, z_eq, psi_raw;
  read_floats_after("r(1:jm);", jm, r_eq);
  read_floats_after("z(1:km);", km, z_eq);
  read_floats_after("((psi(j,k)-psib,j=1,jm),k=1,km)", jm * km, psi_raw);

  std::vector<double> psi(jm * km);
  for (int i = 0; i < jm * km; i++) psi[i] = psi_raw[i] + psib;

  double dr_eq = r_eq[1] - r_eq[0];
  double dz_eq = z_eq[1] - z_eq[0];

  if (comm->me == 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "surface/physical/sputter: deriving B-field from equilibrium %s "
             "(jm=%d km=%d btf=%.3f rtf=%.3f)",
             equ_path.c_str(), jm, km, btf, rtf);
    if (screen) fprintf(screen, "%s\n", msg);
    if (logfile) fprintf(logfile, "%s\n", msg);
  }

  // Bilinear interpolation on equilibrium grid
  auto interp_psi = [&](double R, double Z) -> double {
    double fi = (R - r_eq.front()) / dr_eq;
    double fj = (Z - z_eq.front()) / dz_eq;
    int i0 = std::max(0, std::min((int)fi, jm - 2));
    int j0 = std::max(0, std::min((int)fj, km - 2));
    double s = fi - i0, t = fj - j0;
    s = std::max(0.0, std::min(1.0, s));
    t = std::max(0.0, std::min(1.0, t));
    return (1-s)*(1-t)*psi[j0*jm+i0] + s*(1-t)*psi[j0*jm+i0+1]
         + (1-s)*t*psi[(j0+1)*jm+i0] + s*t*psi[(j0+1)*jm+i0+1];
  };

  // Compute Br, Bz, Bt on the plasma (nr x nz) grid
  br.resize(static_cast<size_t>(nz) * nr);
  bz.resize(static_cast<size_t>(nz) * nr);
  bt.resize(static_cast<size_t>(nz) * nr);

  double eps = 1e-4;  // finite-difference step (m)
  for (int iz = 0; iz < nz; iz++) {
    for (int ir = 0; ir < nr; ir++) {
      double R = rvals[ir];
      double Z = zvals[iz];
      size_t idx = static_cast<size_t>(iz) * nr + ir;

      if (R < 0.01) { br[idx] = bz[idx] = bt[idx] = 0.0; continue; }

      // Br = -(1/R) dpsi/dZ
      double dpsi_dz = (interp_psi(R, Z + eps) - interp_psi(R, Z - eps)) / (2.0 * eps);
      br[idx] = -dpsi_dz / R;

      // Bz = (1/R) dpsi/dR
      double dpsi_dr = (interp_psi(R + eps, Z) - interp_psi(R - eps, Z)) / (2.0 * eps);
      bz[idx] = dpsi_dr / R;

      // Bt = btf * rtf / R
      bt[idx] = btf * rtf / R;
    }
  }

  has_bfield = 1;
}

/* ---------------------------------------------------------------------- */

bigint ComputeSurfacePhysicalSputter::memory_usage()
{
  return static_cast<bigint>(nsown) * nvalue * sizeof(double);
}
