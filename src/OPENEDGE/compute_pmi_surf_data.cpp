/* ----------------------------------------------------------------------
   OpenEdge compute pmi/surf/data
   Uses incident plasma species fluxes + BCA surface data.
------------------------------------------------------------------------- */

#include "string.h"
#include "compute_pmi_surf_data.h"
#include "surf.h"
#include "domain.h"
#include "comm.h"
#include "modify.h"
#include "update.h"
#include "memory.h"
#include "error.h"
#include "fix_plasma_data.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPARTA_NS;

namespace {
enum { PMI_PLASMA_SOURCE_FILE = 0, PMI_PLASMA_SOURCE_FIX = 1 };
}

ComputePMISurfData::ComputePMISurfData(SPARTA *sparta, int narg, char **arg) :
  Compute(sparta, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal compute pmi/surf/data command");

  int igroup = surf->find_group(arg[2]);
  if (igroup < 0) error->all(FLERR,"Compute pmi/surf/data surf group ID does not exist");
  groupbit = surf->bitmask[igroup];

  if (strcmp(arg[3],"file") == 0) {
    plasma_source_mode = PMI_PLASMA_SOURCE_FILE;
    plasma_path = std::string(arg[4]);
  } else if (strcmp(arg[3],"plasma_data") == 0) {
    plasma_source_mode = PMI_PLASMA_SOURCE_FIX;
    plasma_data_fix_id = std::string(arg[4]);
  } else {
    error->all(FLERR,
      "compute pmi/surf/data syntax: compute ID pmi/surf/data surfgroup "
      "file plasma.h5 surface.h5 values ... or plasma_data fixID surface.h5 values ...");
  }
  surface_path = std::string(arg[5]);

  proj_slot_lo = 2;
  proj_slot_hi = std::numeric_limits<int>::max();
  mass_amu = 2.0;  // default: deuterium (used for Bohm sound speed)
  static_cache = 0;
  cache_valid = 0;

  std::vector<int> which_tmp;
  std::vector<int> which_species_tmp;
  int iarg = 6;
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
    } else if (strcmp(arg[iarg],"bfield") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"bfield needs HDF5 file path");
      bfield_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"equilibrium") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"equilibrium needs .equ file path");
      equ_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"boundary") == 0) {
      if (iarg+1 >= narg) error->all(FLERR,"boundary needs polygon file path");
      boundary_path = std::string(arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"debug_interp") == 0) {
      if (iarg+2 >= narg) error->all(FLERR,"debug_interp needs E(eV) angle(deg)");
      debug_interp = 1;
      debug_E.push_back(atof(arg[iarg+1]));
      debug_A.push_back(atof(arg[iarg+2]));
      iarg += 3;
    } else {
      error->all(FLERR,"Invalid compute pmi/surf/data value");
    }
  }

  nvalue = static_cast<int>(which_tmp.size());
  if (nvalue == 0) error->all(FLERR,"No outputs requested for compute pmi/surf/data");
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

ComputePMISurfData::~ComputePMISurfData()
{
  if (copymode) return;
  delete [] which;
  delete [] which_species;
  memory->destroy(vector_surf);
  memory->destroy(array_surf);
}

int ComputePMISurfData::peek_nspec_from_plasma() const
{
  if (plasma_source_mode == PMI_PLASMA_SOURCE_FIX) {
    int ifix = modify->find_fix(plasma_data_fix_id.c_str());
    if (ifix >= 0) {
      auto *pd = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
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

int ComputePMISurfData::in_projectile_slots(int slot1) const
{
  return (slot1 >= proj_slot_lo && slot1 <= proj_slot_hi);
}

void ComputePMISurfData::rebuild_mesh_cache()
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

void ComputePMISurfData::load_plasma_from_fix(const FixPlasmaData *pd)
{
  if (!pd)
    error->all(FLERR, "compute pmi/surf/data: null plasma/data fix");

  if (pd->generation == 0)
    const_cast<FixPlasmaData *>(pd)->init();

  nr = pd->nr;
  nz = pd->nz;
  rvals = pd->rvals;
  zvals = pd->zvals;
  dens_i = pd->dens_i;
  temp_e = pd->temp_e;
  temp_i = pd->temp_i;
  parr_flow = pd->parr_flow;
  parr_flow_r = pd->parr_flow_r;
  parr_flow_t = pd->parr_flow_t;
  parr_flow_z = pd->parr_flow_z;

  has_bfield = pd->has_bfield;
  br = pd->br;
  bt = pd->bt;
  bz = pd->bz;

  ion_charge_state_z = pd->ion_charge_z;
  nspec = (pd->nion > 0) ? pd->nion : 1;
  has_multi_ion = (pd->nion > 0 && !pd->ions_dens.empty());
  ions_dens = pd->ions_dens;
  ions_temp = pd->ions_temp;
  ions_parr_flow = pd->ions_upar;
  ions_parr_flow_r.clear();
  ions_parr_flow_t.clear();
  ions_parr_flow_z.clear();
  has_temp = !temp_i.empty();

  has_mesh = pd->has_mesh;
  mesh_nvtx = pd->mesh_nvtx;
  mesh_ntri = pd->mesh_ntri;
  mesh_ncell = pd->mesh_ncell;
  mesh_vtx_r = pd->mesh_vtx_r;
  mesh_vtx_z = pd->mesh_vtx_z;
  mesh_tri = pd->mesh_tri;
  mesh_cell_idx = pd->mesh_cell_idx;
  mesh_ne = pd->mesh_ne;
  mesh_te = pd->mesh_te;
  mesh_ti = pd->mesh_ti;
  mesh_ni = pd->mesh_ni;
  mesh_upar = pd->mesh_upar;
  mesh_nion = pd->mesh_nion;
  mesh_ions_dens = pd->mesh_ions_dens;
  mesh_ions_temp = pd->mesh_ions_temp;
  mesh_ions_upar = pd->mesh_ions_upar;
  rebuild_mesh_cache();
}

void ComputePMISurfData::load_plasma()
{
  if (plasma_source_mode == PMI_PLASMA_SOURCE_FIX) {
    int ifix = modify->find_fix(plasma_data_fix_id.c_str());
    if (ifix < 0)
      error->all(FLERR, "compute pmi/surf/data: plasma_data fix ID not found");
    auto *pd = dynamic_cast<FixPlasmaData *>(modify->fix[ifix]);
    if (!pd)
      error->all(FLERR, "compute pmi/surf/data: plasma_data fix must be style plasma/data");
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
      throw std::runtime_error("compute pmi/surf/data: plasma 2D dataset shape mismatch");
    out.resize(static_cast<size_t>(dims[0]*dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };
  auto read3D = [&](const std::string& name, std::vector<double>& out, int& ns_out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[1]) != nz || static_cast<int>(dims[2]) != nr)
      throw std::runtime_error("compute pmi/surf/data: plasma 3D dataset shape mismatch");
    ns_out = static_cast<int>(dims[0]);
    out.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  read1D("r", rvals);
  read1D("z", zvals);
  nr = static_cast<int>(rvals.size());
  nz = static_cast<int>(zvals.size());
  if (nr < 2 || nz < 2)
    throw std::runtime_error("compute pmi/surf/data: plasma r/z must have size >= 2");

  read2D("dens_i", dens_i);
  if (hasDataset("temp_e")) {
    read2D("temp_e", temp_e);
  } else {
    throw std::runtime_error("compute pmi/surf/data: missing required temp_e for incident_energy_species");
  }
  read2D("parr_flow", parr_flow);
  if (hasDataset("temp_i")) {
    read2D("temp_i", temp_i);
    has_temp = 1;
  }
  if (hasDataset("parr_flow_r")) read2D("parr_flow_r", parr_flow_r);
  if (hasDataset("parr_flow_t")) read2D("parr_flow_t", parr_flow_t);
  if (hasDataset("parr_flow_z")) read2D("parr_flow_z", parr_flow_z);
  if (hasDataset("br")) read2D("br", br);
  if (hasDataset("bt")) read2D("bt", bt);
  if (hasDataset("bz")) read2D("bz", bz);
  has_bfield = (!br.empty() && !bz.empty());

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
    if (ns1 != ns2) throw std::runtime_error("compute pmi/surf/data: ions/dens vs ions/parr_flow nspec mismatch");
    if (!ions_parr_flow_r.empty() && ns1 != ns3)
      throw std::runtime_error("compute pmi/surf/data: ions/parr_flow_r nspec mismatch");
    if (!ions_parr_flow_z.empty() && ns1 != ns4)
      throw std::runtime_error("compute pmi/surf/data: ions/parr_flow_z nspec mismatch");
    if (!ions_parr_flow_t.empty() && ns1 != ns5)
      throw std::runtime_error("compute pmi/surf/data: ions/parr_flow_t nspec mismatch");
    if (!ions_temp.empty() && ns1 != ns6)
      throw std::runtime_error("compute pmi/surf/data: ions/temp nspec mismatch");
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

void ComputePMISurfData::load_surface_data()
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
      throw std::runtime_error("compute pmi/surf/data: spyld shape mismatch");
    out.resize(static_cast<size_t>(dims[0]*dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  read1D("E", E_axis, nE);
  read1D("A", A_axis, nA);
  if (nE < 2 || nA < 2)
    throw std::runtime_error("compute pmi/surf/data: E/A axes must have size >= 2");
  read2D("spyld", spyld);
}

void ComputePMISurfData::load_boundary()
{
  boundary_r.clear();
  boundary_z.clear();
  if (boundary_path.empty()) return;

  FILE *fp = fopen(boundary_path.c_str(), "r");
  if (!fp) error->all(FLERR, "compute pmi/surf/data: cannot open boundary file");
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
    error->all(FLERR, "compute pmi/surf/data: boundary polygon needs >= 3 points");
  if (comm->me == 0) {
    if (screen) fprintf(screen, "PMI boundary polygon: %d points\n",
                        static_cast<int>(boundary_r.size()));
    if (logfile) fprintf(logfile, "PMI boundary polygon: %d points\n",
                         static_cast<int>(boundary_r.size()));
  }
}

int ComputePMISurfData::point_in_boundary(double r, double z) const
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

void ComputePMISurfData::load_mesh()
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

int ComputePMISurfData::find_mesh_triangle(double r, double z) const
{
  // Brute-force with bounding-box prefilter.
  // Returns triangle index or -1 if not found.
  for (int t = 0; t < mesh_ntri; t++) {
    if (r < mesh_tri_rmin[t] || r > mesh_tri_rmax[t] ||
        z < mesh_tri_zmin[t] || z > mesh_tri_zmax[t]) continue;
    const int v0 = mesh_tri[t*3+0], v1 = mesh_tri[t*3+1], v2 = mesh_tri[t*3+2];
    const double r0 = mesh_vtx_r[v0], z0 = mesh_vtx_z[v0];
    const double r1 = mesh_vtx_r[v1], z1 = mesh_vtx_z[v1];
    const double r2 = mesh_vtx_r[v2], z2 = mesh_vtx_z[v2];
    // Barycentric coordinate test
    const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
    if (std::fabs(d) < 1e-30) continue;
    const double u = ((r-r0)*(z2-z0) - (r2-r0)*(z-z0)) / d;
    const double v = ((r1-r0)*(z-z0) - (r-r0)*(z1-z0)) / d;
    if (u >= -1e-10 && v >= -1e-10 && (u+v) <= 1.0+1e-10) return t;
  }
  return -1;
}

int ComputePMISurfData::find_nearest_mapped_triangle(double r, double z,
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

void ComputePMISurfData::init()
{
  if (!surf->exist)
    error->all(FLERR,"Cannot use compute pmi/surf/data when surfs do not exist");
  if (surf->implicit)
    error->all(FLERR,"Cannot use compute pmi/surf/data with implicit surfs");

  cache_valid = 0;

  try {
    load_plasma();
  } catch (const std::exception& e) {
    error->all(FLERR, e.what());
  } catch (...) {
    error->all(FLERR, "compute pmi/surf/data failed reading plasma file");
  }

  // Load B-field from separate file if not already available.
  // In plasma_data mode we stay file-free unless the user explicitly
  // supplies a bfield path override.
  int bfield_user_specified = !bfield_path.empty();
  if (plasma_source_mode == PMI_PLASMA_SOURCE_FILE &&
      !has_bfield && bfield_path.empty()) {
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
        error->all(FLERR, "compute pmi/surf/data: failed reading bfield file");
      // Otherwise auto-detect failed silently — bfield not available
    }
  }

  try {
    load_surface_data();
  } catch (const std::exception& e) {
    error->all(FLERR, e.what());
  } catch (...) {
    error->all(FLERR, "compute pmi/surf/data failed reading surface file");
  }

  load_boundary();
  if (plasma_source_mode == PMI_PLASMA_SOURCE_FILE) load_mesh();

  if (debug_interp && comm->me == 0) {
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
      "compute pmi/surf/data: need B-field via plasma/data fix, plasma HDF5 "
      "(br/bz), bfield HDF5, or equilibrium keyword");

  distributed = surf->distributed;
  if (!firstflag) return;
  firstflag = 0;

  nsown = surf->nown;
  if (nvalue == 1)
    memory->create(vector_surf,nsown,"pmi/surf/data:vector_surf");
  else
    memory->create(array_surf,nsown,nvalue,"pmi/surf/data:array_surf");
}

double ComputePMISurfData::interp2D(const std::vector<double> &f, double r, double z) const
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

double ComputePMISurfData::interp3D(const std::vector<double> &f, int ispec, double r, double z) const
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

double ComputePMISurfData::interp_yield(double e_eV, double a_deg) const
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

void ComputePMISurfData::compute_per_surf()
{
  // Physical constants
  const double QE  = 1.602176634e-19;   // elementary charge [C]
  const double AMU = 1.66053906660e-27;  // atomic mass unit [kg]

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

  auto clean = [](double x) {
    return (std::fabs(x) < 1.0e-200 || !std::isfinite(x)) ? 0.0 : x;
  };

  for (int i = 0; i < nsown; i++) {
    int m = distributed ? i : me + i*nprocs;
    const int in_group = (dimension == 2) ? (lines[m].mask & groupbit) : (tris[m].mask & groupbit);
    if (!in_group) continue;

    // Surface element centroid in cylindrical (R,Z)
    double r = 0.0, z = 0.0;
    if (dimension == 2) {
      r = 0.5*(lines[m].p1[0] + lines[m].p2[0]);
      z = 0.5*(lines[m].p1[1] + lines[m].p2[1]);
    } else {
      const double xc = (tris[m].p1[0] + tris[m].p2[0] + tris[m].p3[0]) / 3.0;
      const double yc = (tris[m].p1[1] + tris[m].p2[1] + tris[m].p3[1]) / 3.0;
      const double zc = (tris[m].p1[2] + tris[m].p2[2] + tris[m].p3[2]) / 3.0;
      r = std::sqrt(xc*xc + yc*yc);
      z = zc;
    }

    // Skip surface elements outside SOLPS domain.
    // When mesh triangulation is loaded, the point-in-triangle test
    // inherently limits to the SOLPS domain.  The boundary polygon is
    // a fallback for rectangular-grid mode.
    if (!has_mesh && !point_in_boundary(r, z)) continue;

    // Surface normal in cylindrical (R, phi, Z)
    double nr_surf = 0.0, nt_surf = 0.0, nz_surf = 0.0;
    if (dimension == 2) {
      nr_surf = lines[m].norm[0];
      nz_surf = lines[m].norm[1];
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

    // B-field at surface centroid (cylindrical components)
    const double br_loc = interp2D(br, r, z);
    const double bt_loc = bt.empty() ? 0.0 : interp2D(bt, r, z);
    const double bz_loc = interp2D(bz, r, z);
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

    // Resolve plasma data: mesh-based (exact SOLPS cell) or grid-based (bilinear)
    int mesh_cell = -1;
    if (has_mesh) {
      int tri_idx = find_mesh_triangle(r, z);
      // Wall centroids typically fall in the vacuum gap between the B2.5
      // domain and the physical wall.  Fall back to nearest mapped triangle
      // within 5 cm (covers typical SOLPS Eirene vacuum layer thickness).
      if (tri_idx < 0 || mesh_cell_idx[tri_idx] < 0)
        tri_idx = find_nearest_mapped_triangle(r, z, 0.05);
      if (tri_idx >= 0) mesh_cell = mesh_cell_idx[tri_idx];
      else continue;  // genuinely outside SOLPS domain — zero flux
    }

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
    const double te_loc = (mesh_cell >= 0) ? mesh_te[mesh_cell]
                                           : interp2D(temp_e, r, z);

    for (int s = 0; s < ns; s++) {
      double ni, ti;
      species_plasma(s, ni, ti);
      if (ni <= 0.0) continue;

      const int z_inc = (s >= 0 && s < static_cast<int>(ion_charge_state_z.size())) ?
        std::max(1,ion_charge_state_z[s]) : 1;
      const double te_eV = (std::isfinite(te_loc) && te_loc > 0.0) ? te_loc : 0.0;
      const double ti_eV = (std::isfinite(ti) && ti > 0.0) ? ti : 0.0;

      // Bohm sound speed: cs = sqrt((Te + Ti) * e / (2 * m_ion * AMU))
      const double cs_arg = (te_eV + ti_eV) * QE / (2.0 * mass_amu * AMU);
      const double cs = (cs_arg > 0.0) ? std::sqrt(cs_arg) : 0.0;

      // Bohm flux: Gamma = n_i * c_s * sin(alpha_B)
      const double g = ni * cs * sin_alpha;
      gamma_n[s] = g;

      // Incidence angle for sputter yield = alpha_B
      const double a_deg = std::asin(std::min(1.0, sin_alpha)) * 180.0 / M_PI;
      angle_deg[s] = a_deg;

      // Eckstein incident energy: E = 3*Te + 2*Z*Ti
      double E = 3.0 * te_eV + 2.0 * static_cast<double>(z_inc) * ti_eV;
      energy_eV[s] = E;

      if (in_projectile_slots(s+1)) {
        const double ys = interp_yield(E, a_deg);
        yld[s] = ys;
        sput_flux[s] = g * ys;
        sput_total += sput_flux[s];
      }
    }

    if (nvalue == 1) {
      double out = 0.0;
      if (which[0] == SPUTTER_FLUX_TOTAL) out = sput_total;
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

void ComputePMISurfData::load_bfield_from_equ()
{
  // Read SOLPS/EQDSK-style equilibrium file and derive B-field on the
  // plasma (R,Z) grid.  Uses the same parser as fix reflect/psi.
  //
  //   Br = -(1/R) dpsi/dZ
  //   Bz =  (1/R) dpsi/dR
  //   Bt =  btf * rtf / R

  std::ifstream ifs(equ_path);
  if (!ifs.good())
    throw std::runtime_error(
        std::string("compute pmi/surf/data: cannot open equilibrium '")
        + equ_path + "'");
  std::string text((std::istreambuf_iterator<char>(ifs)),
                    std::istreambuf_iterator<char>());
  ifs.close();

  // --- helpers ---
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

  // psi = psi_raw + psib  (stored as psi-psib in file)
  std::vector<double> psi(jm * km);
  for (int i = 0; i < jm * km; i++) psi[i] = psi_raw[i] + psib;

  double dr = r_eq[1] - r_eq[0];
  double dz = z_eq[1] - z_eq[0];

  if (comm->me == 0) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "pmi/surf/data: deriving B-field from equilibrium %s "
             "(jm=%d km=%d btf=%.3f rtf=%.3f)",
             equ_path.c_str(), jm, km, btf, rtf);
    if (screen) fprintf(screen, "%s\n", msg);
    if (logfile) fprintf(logfile, "%s\n", msg);
  }

  // Bilinear interpolation helper on the equilibrium grid
  auto interp_psi = [&](double R, double Z) -> double {
    double fi = (R - r_eq.front()) / dr;
    double fj = (Z - z_eq.front()) / dz;
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

      // Br = -(1/R) dpsi/dZ   (central difference)
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

bigint ComputePMISurfData::memory_usage()
{
  return static_cast<bigint>(nsown) * nvalue * sizeof(double);
}
