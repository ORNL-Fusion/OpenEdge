/* ----------------------------------------------------------------------
   OpenEdge fix background
   Centralised plasma + equilibrium data store.
   Read once, shared by all fixes/computes that need background plasma.

   Usage:
     fix ID background file plasma.h5 [equilibrium file.equ] [static yes]
     fix ID background constant [r_bounds rmin rmax] [z_bounds zmin zmax]
                         [ne val] [te val] [ni val] [ti val]
                         [parr_flow val] [parr_flow_r val] [parr_flow_t val]
                         [parr_flow_z val] [grad_te_r val] [grad_te_t val]
                         [grad_te_z val] [grad_ti_r val] [grad_ti_t val]
                         [grad_ti_z val] [epar val] [br val] [bz val] [bt val]
                         [equilibrium file.equ] [static yes]

   Other fixes/computes access this via:
     int ifix = modify->find_fix("ID");
     auto *pd = dynamic_cast<FixBackground*>(modify->fix[ifix]);
     double te = pd->interp2D(pd->temp_e, R, Z);
------------------------------------------------------------------------- */

#include "string.h"
#include "fix_background.h"
#include "comm.h"
#include "compute_plasma_fields.h"
#include "domain.h"
#include "error.h"
#include "grid.h"
#include "input.h"
#include "modify.h"
#include "openedge_geom.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace SPARTA_NS;

namespace {
enum { PLASMA_SOURCE_FILE = 0, PLASMA_SOURCE_CONSTANT = 1 };
}

/* ---------------------------------------------------------------------- */

FixBackground::FixBackground(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
  // fix ID background file PATH [equilibrium PATH] [static yes/no]
  // fix ID background constant ...
  if (narg < 3)
    error->all(FLERR, "Illegal fix background command");

  // gridmigrate=1: SPARTA drives grid_changed() on adapt/balance (same
  // pattern as FixEmit). cell_mesh_cell is rebuilt from current grid
  // state after migration — no per-cell pack/unpack needed.
  gridmigrate = 1;
  cell_mesh_stamp_id = -1;
  cell_mesh_stamp_n  = -1;

  nr = nz = 0;
  nion = 0;
  has_bfield = 0;
  has_equ = 0;
  has_mesh = 0;
  has_mesh_wall_face_area = 0;
  has_mesh_wall_surf_cell = 0;
  has_qheatflux = 0;
  default_q_par  = 50.0e6;
  default_q_perp = 0.0;
  is_static = 0;
  source_mode = -1;
  generation = 0;
  equ_jm = equ_km = 0;
  btf = rtf = psib = psi_axis = 0.0;
  mesh_nvtx = mesh_ntri = mesh_ncell = mesh_nion = 0;
  hash_nr = hash_nz = 0;
  hash_rmin = hash_zmin = hash_dr = hash_dz = 0.0;
  const_has_r_bounds = const_has_z_bounds = 0;
  const_rmin = 0.0; const_rmax = 1.0;
  const_zmin = 0.0; const_zmax = 1.0;
  const_dens_e = const_temp_e = 0.0;
  const_dens_i = const_temp_i = 0.0;
  const_has_dens_i = const_has_temp_i = 0;
  const_parr_flow = const_parr_flow_r = const_parr_flow_t = const_parr_flow_z = 0.0;
  const_grad_te_r = const_grad_te_t = const_grad_te_z = 0.0;
  const_grad_ti_r = const_grad_ti_t = const_grad_ti_z = 0.0;
  const_epar = 0.0;
  const_br = const_bz = const_bt = 0.0;
  const_has_bfield = 0;
  column_x0 = column_y0 = 0.0;

  int iarg = 2;
  auto parse_scalar = [&](double &dst, const char *label) {
    if (iarg + 1 >= narg) {
      char msg[256];
      snprintf(msg, sizeof(msg), "fix background: %s needs a numeric value", label);
      error->all(FLERR, msg);
    }
    dst = input->numeric(FLERR, arg[iarg + 1]);
    iarg += 2;
  };
  while (iarg < narg) {
    if (strcmp(arg[iarg], "file") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix background: file needs path");
      if (source_mode == PLASMA_SOURCE_CONSTANT)
        error->all(FLERR, "fix background: choose either file or constant mode");
      source_mode = PLASMA_SOURCE_FILE;
      plasma_path = std::string(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "constant") == 0) {
      if (source_mode == PLASMA_SOURCE_FILE)
        error->all(FLERR, "fix background: choose either file or constant mode");
      source_mode = PLASMA_SOURCE_CONSTANT;
      iarg += 1;
    } else if (strcmp(arg[iarg], "equilibrium") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix background: equilibrium needs path");
      equ_path = std::string(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "static") == 0) {
      if (iarg + 1 >= narg) error->all(FLERR, "fix background: static needs yes/no");
      if (strcmp(arg[iarg + 1], "yes") == 0) is_static = 1;
      else if (strcmp(arg[iarg + 1], "no") == 0) is_static = 0;
      else error->all(FLERR, "fix background: static must be yes or no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "r_bounds") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "fix background: r_bounds needs rmin rmax");
      const_rmin = input->numeric(FLERR, arg[iarg + 1]);
      const_rmax = input->numeric(FLERR, arg[iarg + 2]);
      const_has_r_bounds = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "z_bounds") == 0) {
      if (iarg + 2 >= narg) error->all(FLERR, "fix background: z_bounds needs zmin zmax");
      const_zmin = input->numeric(FLERR, arg[iarg + 1]);
      const_zmax = input->numeric(FLERR, arg[iarg + 2]);
      const_has_z_bounds = 1;
      iarg += 3;
    } else if (strcmp(arg[iarg], "ne") == 0 || strcmp(arg[iarg], "dens_e") == 0) {
      parse_scalar(const_dens_e, arg[iarg]);
    } else if (strcmp(arg[iarg], "te") == 0 || strcmp(arg[iarg], "temp_e") == 0) {
      parse_scalar(const_temp_e, arg[iarg]);
    } else if (strcmp(arg[iarg], "ni") == 0 || strcmp(arg[iarg], "dens_i") == 0) {
      parse_scalar(const_dens_i, arg[iarg]);
      const_has_dens_i = 1;
    } else if (strcmp(arg[iarg], "ti") == 0 || strcmp(arg[iarg], "temp_i") == 0) {
      parse_scalar(const_temp_i, arg[iarg]);
      const_has_temp_i = 1;
    } else if (strcmp(arg[iarg], "parr_flow") == 0 || strcmp(arg[iarg], "upar") == 0) {
      parse_scalar(const_parr_flow, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_r") == 0 || strcmp(arg[iarg], "upar_r") == 0) {
      parse_scalar(const_parr_flow_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_t") == 0 || strcmp(arg[iarg], "upar_t") == 0) {
      parse_scalar(const_parr_flow_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "parr_flow_z") == 0 || strcmp(arg[iarg], "upar_z") == 0) {
      parse_scalar(const_parr_flow_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_r") == 0) {
      parse_scalar(const_grad_te_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_t") == 0) {
      parse_scalar(const_grad_te_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_te_z") == 0) {
      parse_scalar(const_grad_te_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_r") == 0) {
      parse_scalar(const_grad_ti_r, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_t") == 0) {
      parse_scalar(const_grad_ti_t, arg[iarg]);
    } else if (strcmp(arg[iarg], "grad_ti_z") == 0) {
      parse_scalar(const_grad_ti_z, arg[iarg]);
    } else if (strcmp(arg[iarg], "epar") == 0) {
      parse_scalar(const_epar, arg[iarg]);
    } else if (strcmp(arg[iarg], "br") == 0) {
      parse_scalar(const_br, arg[iarg]);
      const_has_bfield = 1;
    } else if (strcmp(arg[iarg], "bz") == 0) {
      parse_scalar(const_bz, arg[iarg]);
      const_has_bfield = 1;
    } else if (strcmp(arg[iarg], "bt") == 0) {
      parse_scalar(const_bt, arg[iarg]);
      const_has_bfield = 1;
    } else if (strcmp(arg[iarg], "column_axis") == 0) {
      if (iarg + 2 >= narg)
        error->all(FLERR, "fix background: column_axis needs x0 y0");
      column_x0 = input->numeric(FLERR, arg[iarg + 1]);
      column_y0 = input->numeric(FLERR, arg[iarg + 2]);
      iarg += 3;
    } else {
      char msg[256];
      snprintf(msg, sizeof(msg),
               "fix background: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  if (source_mode < 0)
    error->all(FLERR, "fix background: must specify either 'file PATH' or 'constant'");
  if (source_mode == PLASMA_SOURCE_FILE && plasma_path.empty())
    error->all(FLERR, "fix background: must specify 'file PATH'");
  if (const_has_r_bounds && const_rmax <= const_rmin)
    error->all(FLERR, "fix background: r_bounds requires rmax > rmin");
  if (const_has_z_bounds && const_zmax <= const_zmin)
    error->all(FLERR, "fix background: z_bounds requires zmax > zmin");
}

/* ---------------------------------------------------------------------- */

FixBackground::~FixBackground() {}

/* ---------------------------------------------------------------------- */

int FixBackground::setmask()
{
  return 0;
}

/* ---------------------------------------------------------------------- */

void FixBackground::init()
{
  // Only load on first init, or if not static
  if (generation > 0 && is_static) return;

  reload();
}

/* ---------------------------------------------------------------------- */

void FixBackground::reload()
{
  clear_loaded_data();

  try {
    if (comm->me == 0) {
      if (source_mode == PLASMA_SOURCE_FILE) load_plasma_h5();
      else load_constant_profile();
    }
  } catch (const std::exception &e) {
    error->one(FLERR, e.what());
  }

  // Broadcast plasma grid dimensions
  MPI_Bcast(&nr, 1, MPI_INT, 0, world);
  MPI_Bcast(&nz, 1, MPI_INT, 0, world);
  MPI_Bcast(&has_bfield, 1, MPI_INT, 0, world);
  MPI_Bcast(&has_qheatflux, 1, MPI_INT, 0, world);
  MPI_Bcast(&nion, 1, MPI_INT, 0, world);
  MPI_Bcast(&has_mesh, 1, MPI_INT, 0, world);
  MPI_Bcast(&mesh_nvtx, 1, MPI_INT, 0, world);
  MPI_Bcast(&mesh_ntri, 1, MPI_INT, 0, world);
  MPI_Bcast(&mesh_ncell, 1, MPI_INT, 0, world);
  MPI_Bcast(&mesh_nion, 1, MPI_INT, 0, world);
  // Psi map broadcast flag (0 = not loaded from plasma.h5)
  int has_psi_map = (has_equ && !psirz.empty()) ? 1 : 0;
  MPI_Bcast(&has_psi_map, 1, MPI_INT, 0, world);

  // Equilibrium grid dims (independent of the regular-grid nr/nz — for
  // mesh-only plasma.h5 the regular grid is absent but the equilibrium
  // still carries a 257x257 psi map). Broadcast so non-root sizes equ_r,
  // equ_z, psirz to the right shape.
  MPI_Bcast(&equ_jm, 1, MPI_INT, 0, world);
  MPI_Bcast(&equ_km, 1, MPI_INT, 0, world);

  size_t grid_n = static_cast<size_t>(nz) * nr;

  // Resize on non-root
  if (comm->me != 0) {
    rvals.resize(nr);
    zvals.resize(nz);
    dens_e.resize(grid_n);
    temp_e.resize(grid_n);
    dens_i.resize(grid_n);
    temp_i.resize(grid_n);
    parr_flow.resize(grid_n);
    parr_flow_r.resize(grid_n);
    parr_flow_t.resize(grid_n);
    parr_flow_z.resize(grid_n);
    grad_te_r.resize(grid_n);
    grad_te_t.resize(grid_n);
    grad_te_z.resize(grid_n);
    grad_ti_r.resize(grid_n);
    grad_ti_t.resize(grid_n);
    grad_ti_z.resize(grid_n);
    epar.resize(grid_n);
    if (has_bfield) {
      br.resize(grid_n);
      bz.resize(grid_n);
      bt.resize(grid_n);
    }
    if (nion > 0) {
      ion_charge_z.resize(nion);
      ion_mass_amu.resize(nion);
      size_t ion_n = static_cast<size_t>(nion) * grid_n;
      ions_dens.resize(ion_n);
      ions_temp.resize(ion_n);
      ions_upar.resize(ion_n);
    }
    if (has_mesh) {
      mesh_vtx_r.resize(mesh_nvtx);
      mesh_vtx_z.resize(mesh_nvtx);
      mesh_tri.resize(static_cast<size_t>(mesh_ntri) * 3);
      mesh_cell_idx.resize(mesh_ntri);
      mesh_ne.resize(mesh_ncell);
      mesh_te.resize(mesh_ncell);
      mesh_ti.resize(mesh_ncell);
      mesh_ni.resize(mesh_ncell);
      mesh_upar.resize(mesh_ncell);
      if (mesh_nion > 0) {
        const size_t mesh_ion_n = static_cast<size_t>(mesh_nion) * mesh_ncell;
        mesh_ions_dens.resize(mesh_ion_n);
        mesh_ions_temp.resize(mesh_ion_n);
        mesh_ions_upar.resize(mesh_ion_n);
      }
    }
    if (has_psi_map) {
      // equ_jm / equ_km already broadcast from rank 0 above — they may
      // differ from nr/nz when plasma.h5 is mesh-only (nr=nz=0 but the
      // equilibrium is still a full 257x257).
      equ_r.resize(equ_jm);
      equ_z.resize(equ_km);
      psirz.resize(static_cast<size_t>(equ_jm) * equ_km);
    }
  }

  // Broadcast all arrays
  MPI_Bcast(rvals.data(), nr, MPI_DOUBLE, 0, world);
  MPI_Bcast(zvals.data(), nz, MPI_DOUBLE, 0, world);
  MPI_Bcast(dens_e.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(temp_e.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(dens_i.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(temp_i.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(parr_flow_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_te_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_r.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_t.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(grad_ti_z.data(), grid_n, MPI_DOUBLE, 0, world);
  MPI_Bcast(epar.data(), grid_n, MPI_DOUBLE, 0, world);

  // q_par / q_perp may be absent on either the regular grid or the mesh
  // (or both). Broadcast a per-component flag so non-root ranks know
  // whether to expect the payload.
  {
    int have_qpar_reg  = (comm->me == 0 && static_cast<int>(q_par.size())  == grid_n) ? 1 : 0;
    int have_qperp_reg = (comm->me == 0 && static_cast<int>(q_perp.size()) == grid_n) ? 1 : 0;
    MPI_Bcast(&have_qpar_reg,  1, MPI_INT, 0, world);
    MPI_Bcast(&have_qperp_reg, 1, MPI_INT, 0, world);
    if (have_qpar_reg) {
      if (static_cast<int>(q_par.size()) != grid_n) q_par.assign(grid_n, 0.0);
      MPI_Bcast(q_par.data(), grid_n, MPI_DOUBLE, 0, world);
    } else q_par.clear();
    if (have_qperp_reg) {
      if (static_cast<int>(q_perp.size()) != grid_n) q_perp.assign(grid_n, 0.0);
      MPI_Bcast(q_perp.data(), grid_n, MPI_DOUBLE, 0, world);
    } else q_perp.clear();
  }

  if (has_bfield) {
    MPI_Bcast(br.data(), grid_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(bz.data(), grid_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(bt.data(), grid_n, MPI_DOUBLE, 0, world);
  }

  if (nion > 0) {
    MPI_Bcast(ion_charge_z.data(), nion, MPI_INT, 0, world);
    MPI_Bcast(ion_mass_amu.data(), nion, MPI_DOUBLE, 0, world);
    size_t ion_n = static_cast<size_t>(nion) * grid_n;
    MPI_Bcast(ions_dens.data(), ion_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(ions_temp.data(), ion_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(ions_upar.data(), ion_n, MPI_DOUBLE, 0, world);
    // String vectors: ion_names and ion_elements (added 2026-04-21).
    // Read on rank 0 only; without this broadcast, non-root ranks see
    // empty vectors, which breaks downstream consumers that rely on
    // element-based slot routing (e.g. compute surface/physical/sputter with
    // `target W projectiles O`).
    auto bcast_strings = [&](std::vector<std::string> &vec) {
      int n = static_cast<int>(vec.size());
      MPI_Bcast(&n, 1, MPI_INT, 0, world);
      if (comm->me != 0) vec.assign(n, std::string());
      for (int i = 0; i < n; i++) {
        int len = (comm->me == 0) ? static_cast<int>(vec[i].size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, world);
        if (comm->me != 0) vec[i].assign(len, '\0');
        if (len > 0) MPI_Bcast(&vec[i][0], len, MPI_CHAR, 0, world);
      }
    };
    bcast_strings(ion_names);
    bcast_strings(ion_elements);
  }

  if (has_mesh) {
    MPI_Bcast(mesh_vtx_r.data(), mesh_nvtx, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_vtx_z.data(), mesh_nvtx, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_tri.data(), mesh_ntri * 3, MPI_INT, 0, world);
    MPI_Bcast(mesh_cell_idx.data(), mesh_ntri, MPI_INT, 0, world);
    MPI_Bcast(mesh_ne.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_te.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_ti.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_ni.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    MPI_Bcast(mesh_upar.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    // Precomputed gradients — broadcast presence flags then payload.
    // A vector of size 0 means "not written in plasma.h5"; size
    // mesh_ncell means available.
    auto bcast_grad = [&](std::vector<double> &v) {
      int has = (static_cast<int>(v.size()) == mesh_ncell) ? 1 : 0;
      MPI_Bcast(&has, 1, MPI_INT, 0, world);
      if (!has) { v.clear(); return; }
      if (static_cast<int>(v.size()) != mesh_ncell) v.assign(mesh_ncell, 0.0);
      MPI_Bcast(v.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    };
    bcast_grad(mesh_grad_te_r);
    bcast_grad(mesh_grad_te_z);
    bcast_grad(mesh_grad_ti_r);
    bcast_grad(mesh_grad_ti_z);
    bcast_grad(mesh_q_par);
    bcast_grad(mesh_q_perp);
    bcast_grad(mesh_e_r);
    bcast_grad(mesh_e_z);
    bcast_grad(mesh_e_t);
    // Per-vertex B (nvtx-long). Broadcast as nvtx, not ncell.
    auto bcast_vtx = [&](std::vector<double> &v) {
      int has = (static_cast<int>(v.size()) == mesh_nvtx) ? 1 : 0;
      MPI_Bcast(&has, 1, MPI_INT, 0, world);
      if (!has) { v.clear(); return; }
      if (static_cast<int>(v.size()) != mesh_nvtx) v.assign(mesh_nvtx, 0.0);
      MPI_Bcast(v.data(), mesh_nvtx, MPI_DOUBLE, 0, world);
    };
    bcast_vtx(mesh_vtx_br);
    bcast_vtx(mesh_vtx_bz);
    bcast_vtx(mesh_vtx_bt);
    // Derive per-triangle B on all ranks (simple mean of 3 vertex values).
    const bool have_vtx_b = !mesh_vtx_br.empty()
                         && !mesh_vtx_bz.empty()
                         && !mesh_vtx_bt.empty();
    if (have_vtx_b) {
      mesh_tri_br.assign(mesh_ntri, 0.0);
      mesh_tri_bz.assign(mesh_ntri, 0.0);
      mesh_tri_bt.assign(mesh_ntri, 0.0);
      for (int t = 0; t < mesh_ntri; t++) {
        const int v0 = mesh_tri[3*t+0];
        const int v1 = mesh_tri[3*t+1];
        const int v2 = mesh_tri[3*t+2];
        mesh_tri_br[t] = (mesh_vtx_br[v0] + mesh_vtx_br[v1] + mesh_vtx_br[v2]) / 3.0;
        mesh_tri_bz[t] = (mesh_vtx_bz[v0] + mesh_vtx_bz[v1] + mesh_vtx_bz[v2]) / 3.0;
        mesh_tri_bt[t] = (mesh_vtx_bt[v0] + mesh_vtx_bt[v1] + mesh_vtx_bt[v2]) / 3.0;
      }
      // Mesh-native B counts toward has_bfield so consumers that gate
      // on it (pusher_boris_2d fallback, cache_plasma_particles, ...)
      // actually call bfield_at(), where the mesh dispatch lives.
      has_bfield = 1;
    } else {
      mesh_tri_br.clear();
      mesh_tri_bz.clear();
      mesh_tri_bt.clear();
    }
    MPI_Bcast(&has_mesh_wall_face_area, 1, MPI_INT, 0, world);
    if (has_mesh_wall_face_area) {
      if (static_cast<int>(mesh_wall_face_area.size()) != mesh_ncell)
        mesh_wall_face_area.resize(mesh_ncell, 0.0);
      MPI_Bcast(mesh_wall_face_area.data(), mesh_ncell, MPI_DOUBLE, 0, world);
    }
    MPI_Bcast(&has_mesh_wall_surf_cell, 1, MPI_INT, 0, world);
    int n_wall_surf = has_mesh_wall_surf_cell
                    ? static_cast<int>(mesh_wall_surf_cell.size()) : 0;
    MPI_Bcast(&n_wall_surf, 1, MPI_INT, 0, world);
    if (has_mesh_wall_surf_cell) {
      if (static_cast<int>(mesh_wall_surf_cell.size()) != n_wall_surf)
        mesh_wall_surf_cell.resize(n_wall_surf, -1);
      MPI_Bcast(mesh_wall_surf_cell.data(), n_wall_surf, MPI_INT, 0, world);
      if (static_cast<int>(mesh_wall_surf_area.size()) != n_wall_surf)
        mesh_wall_surf_area.resize(n_wall_surf, 0.0);
      MPI_Bcast(mesh_wall_surf_area.data(), n_wall_surf,
                MPI_DOUBLE, 0, world);
    }
    if (mesh_nion > 0) {
      const size_t mesh_ion_n = static_cast<size_t>(mesh_nion) * mesh_ncell;
      MPI_Bcast(mesh_ions_dens.data(), mesh_ion_n, MPI_DOUBLE, 0, world);
      MPI_Bcast(mesh_ions_temp.data(), mesh_ion_n, MPI_DOUBLE, 0, world);
      MPI_Bcast(mesh_ions_upar.data(), mesh_ion_n, MPI_DOUBLE, 0, world);
    }
  }

  // Psi map (from plasma.h5) broadcast to all ranks and light up has_equ
  // so downstream consumers (fix reflect/psi, psi_norm_at) can query psi.
  if (has_psi_map) {
    const size_t equ_n = static_cast<size_t>(equ_jm) * equ_km;
    MPI_Bcast(equ_r.data(), equ_jm, MPI_DOUBLE, 0, world);
    MPI_Bcast(equ_z.data(), equ_km, MPI_DOUBLE, 0, world);
    MPI_Bcast(psirz.data(), equ_n, MPI_DOUBLE, 0, world);
    MPI_Bcast(&psi_axis, 1, MPI_DOUBLE, 0, world);
    MPI_Bcast(&psib, 1, MPI_DOUBLE, 0, world);
    has_equ = 1;
  }

  // Load equilibrium (all ranks, text file is cheap)
  if (!equ_path.empty()) {
    try {
      load_equilibrium();
    } catch (const std::exception &e) {
      error->all(FLERR, e.what());
    }
    // Derive B-field if not already in plasma.h5
    if (!has_bfield) {
      derive_bfield_from_equ();
    }
  }

  if (has_mesh) build_mesh_index();

  // Mesh triangles may have changed — drop any cell-indexed cache so the
  // next cache_plasma_particles call rebuilds from the new hash grid.
  cell_mesh_cell.clear();

  generation++;

  if (comm->me == 0) {
    if (screen) {
      // Separate reports for regular-grid vs mesh carriers, so a
      // mesh-only plasma.h5 (nr=nz=0 is expected) doesn't read as
      // "0 x 0 grid".
      if (nr > 0 && nz > 0) {
        fprintf(screen,
          "[background] Loaded: %d x %d regular grid, ", nr, nz);
      } else {
        fprintf(screen, "[background] Loaded: ");
      }
      if (has_mesh) {
        fprintf(screen, "mesh %d tri / %d cell / %d vtx, ",
                mesh_ntri, mesh_ncell, mesh_nvtx);
      }
      // bfield source: "grid" = legacy regular-grid br/bz datasets;
      // "mesh" = per-vertex mesh/vtx_b* from the new converters; "no" = neither.
      const char *bf_src =
          has_bfield ? "grid"
          : (!mesh_tri_br.empty() ? "mesh" : "no");
      fprintf(screen,
        "%d ion species, bfield=%s, equ=%s, gen=%d\n",
        nion, bf_src,
        has_equ ? "yes" : "no", generation);
    }
    // Only warn about missing q_par/q_perp if a consumer that actually
    // reads them (currently: fix evaporation) is present. Decks without
    // liquid-metal physics don't care that the file lacks heat flux.
    if (!has_qheatflux && generation == 1) {
      bool has_heatflux_consumer = false;
      for (int i = 0; i < modify->nfix; i++) {
        if (strcmp(modify->fix[i]->style, "evaporation") == 0) {
          has_heatflux_consumer = true;
          break;
        }
      }
      if (has_heatflux_consumer) {
        char msg[200];
        snprintf(msg, sizeof(msg),
          "[background] no q_par/q_perp in %s — defaulting to "
          "q_par=%.2e W/m^2, q_perp=%.2e W/m^2 for evaporation queries",
          plasma_path.c_str(), default_q_par, default_q_perp);
        error->warning(FLERR, msg);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBackground::clear_loaded_data()
{
  nr = nz = 0;
  nion = 0;
  has_bfield = 0;
  has_equ = 0;
  has_mesh = 0;
  equ_jm = equ_km = 0;
  btf = rtf = psib = psi_axis = 0.0;
  mesh_nvtx = mesh_ntri = mesh_ncell = mesh_nion = 0;
  hash_nr = hash_nz = 0;
  hash_rmin = hash_zmin = hash_dr = hash_dz = 0.0;

  rvals.clear();
  zvals.clear();
  dens_e.clear();
  temp_e.clear();
  dens_i.clear();
  temp_i.clear();
  parr_flow.clear();
  parr_flow_r.clear();
  parr_flow_t.clear();
  parr_flow_z.clear();
  grad_te_r.clear();
  grad_te_t.clear();
  grad_te_z.clear();
  grad_ti_r.clear();
  grad_ti_t.clear();
  grad_ti_z.clear();
  epar.clear();
  q_par.clear();
  q_perp.clear();
  has_qheatflux = 0;
  br.clear();
  bz.clear();
  bt.clear();
  ion_charge_z.clear();
  ion_mass_amu.clear();
  ion_names.clear();
  ion_elements.clear();
  ions_dens.clear();
  ions_temp.clear();
  ions_upar.clear();
  equ_r.clear();
  equ_z.clear();
  psirz.clear();
  mesh_vtx_r.clear();
  mesh_vtx_z.clear();
  mesh_tri.clear();
  mesh_cell_idx.clear();
  mesh_ne.clear();
  mesh_wall_face_area.clear();
  has_mesh_wall_face_area = 0;
  mesh_wall_surf_cell.clear();
  has_mesh_wall_surf_cell = 0;
  mesh_te.clear();
  mesh_ti.clear();
  mesh_ni.clear();
  mesh_upar.clear();
  mesh_ions_dens.clear();
  mesh_ions_temp.clear();
  mesh_ions_upar.clear();
  mesh_q_par.clear();
  mesh_q_perp.clear();
  mesh_tri_rmin.clear();
  mesh_tri_rmax.clear();
  mesh_tri_zmin.clear();
  mesh_tri_zmax.clear();
  mapped_cr.clear();
  mapped_cz.clear();
  mapped_idx.clear();
  hash_grid.clear();
  valid_mask.clear();
}

/* ---------------------------------------------------------------------- */

void FixBackground::reload_plasma(const std::string &path)
{
  plasma_path = path;
  source_mode = PLASMA_SOURCE_FILE;
  reload();
}

/* ---------------------------------------------------------------------- */

void FixBackground::load_constant_profile()
{
  double rmin = const_rmin, rmax = const_rmax;
  double zmin = const_zmin, zmax = const_zmax;

  if (!const_has_r_bounds || !const_has_z_bounds) {
    if (domain->dimension == 2) {
      // Axi: SPARTA x = Z, y = R. Cart 2D: SPARTA x = R, y = Z.
      const int iR = domain->axisymmetric ? 1 : 0;
      const int iZ = domain->axisymmetric ? 0 : 1;
      if (!const_has_r_bounds) { rmin = domain->boxlo[iR]; rmax = domain->boxhi[iR]; }
      if (!const_has_z_bounds) { zmin = domain->boxlo[iZ]; zmax = domain->boxhi[iZ]; }
    } else {
      const double r00 = std::hypot(domain->boxlo[0], domain->boxlo[1]);
      const double r01 = std::hypot(domain->boxlo[0], domain->boxhi[1]);
      const double r10 = std::hypot(domain->boxhi[0], domain->boxlo[1]);
      const double r11 = std::hypot(domain->boxhi[0], domain->boxhi[1]);
      if (!const_has_r_bounds) {
        rmin = 0.0;
        rmax = std::max(std::max(r00, r01), std::max(r10, r11));
      }
      if (!const_has_z_bounds) { zmin = domain->boxlo[2]; zmax = domain->boxhi[2]; }
    }
  }

  if (rmax <= rmin) rmax = rmin + 1.0;
  if (zmax <= zmin) zmax = zmin + 1.0;

  nr = 2;
  nz = 2;
  rvals = {rmin, rmax};
  zvals = {zmin, zmax};

  const size_t n = 4;
  const double ni_uniform = const_has_dens_i ? const_dens_i : const_dens_e;
  const double ti_uniform = const_has_temp_i ? const_temp_i : const_temp_e;

  dens_e.assign(n, const_dens_e);
  temp_e.assign(n, const_temp_e);
  dens_i.assign(n, ni_uniform);
  temp_i.assign(n, ti_uniform);
  parr_flow.assign(n, const_parr_flow);
  parr_flow_r.assign(n, const_parr_flow_r);
  parr_flow_t.assign(n, const_parr_flow_t);
  parr_flow_z.assign(n, const_parr_flow_z);
  grad_te_r.assign(n, const_grad_te_r);
  grad_te_t.assign(n, const_grad_te_t);
  grad_te_z.assign(n, const_grad_te_z);
  grad_ti_r.assign(n, const_grad_ti_r);
  grad_ti_t.assign(n, const_grad_ti_t);
  grad_ti_z.assign(n, const_grad_ti_z);
  epar.assign(n, const_epar);

  has_bfield = const_has_bfield;
  if (has_bfield) {
    br.assign(n, const_br);
    bz.assign(n, const_bz);
    bt.assign(n, const_bt);
  }

  if (screen) {
    fprintf(screen,
            "[background] Using inline constant fields on synthetic %d x %d grid"
            " (R:[%.6g,%.6g] Z:[%.6g,%.6g])\n",
            nr, nz, rmin, rmax, zmin, zmax);
  }
}

/* ---------------------------------------------------------------------- */

void FixBackground::load_plasma_h5()
{
  if (screen)
    fprintf(screen, "[background] Reading %s\n", plasma_path.c_str());

  H5::Exception::dontPrint();
  H5::H5File file(plasma_path, H5F_ACC_RDONLY);

  auto read1D = [&](const std::string &name, std::vector<double> &out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dim;
    sp.getSimpleExtentDims(&dim);
    out.resize(dim);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  auto hasDataset = [&](const std::string &name) -> bool {
    htri_t exists = 0;
    H5E_BEGIN_TRY { exists = H5Oexists_by_name(file.getId(), name.c_str(), H5P_DEFAULT); }
    H5E_END_TRY;
    return exists > 0;
  };

  // Optional regular (R, Z) grid. Current SOLPS/SOLEDGE3X/OEDGE
  // converters write plasma fields ONLY on the EIRENE triangulation
  // (see /mesh/*). The regular-grid path below is kept alive only to
  // read older plasma.h5 files that still carry top-level r/z +
  // dens_e/temp_e etc. New files have mesh-only data and skip this
  // entire block.
  nr = nz = 0;
  size_t n = 0;
  const bool has_regular_grid = hasDataset("r") && hasDataset("z");
  if (has_regular_grid) {
    read1D("r", rvals);
    read1D("z", zvals);
    nr = static_cast<int>(rvals.size());
    nz = static_cast<int>(zvals.size());
    n = static_cast<size_t>(nz) * nr;

    // Read 2D field into flat vector [iz * nr + ir]
    auto read2D = [&](const std::string &name, std::vector<double> &out) {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2];
      sp.getSimpleExtentDims(dims);
      if (static_cast<int>(dims[0]) != nz || static_cast<int>(dims[1]) != nr)
        throw std::runtime_error("Shape mismatch in " + name);
      out.resize(n);
      ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };
    auto read2D_optional = [&](const std::string &name, std::vector<double> &out) {
      if (hasDataset(name)) read2D(name, out);
      else { out.assign(n, 0.0); }
    };

    if (hasDataset("dens_e")) read2D("dens_e", dens_e);
    if (hasDataset("temp_e")) read2D("temp_e", temp_e);
    if (hasDataset("dens_i")) read2D("dens_i", dens_i);
    if (hasDataset("temp_i")) read2D("temp_i", temp_i);
    read2D_optional("parr_flow", parr_flow);
    read2D_optional("parr_flow_r", parr_flow_r);
    read2D_optional("parr_flow_t", parr_flow_t);
    read2D_optional("parr_flow_z", parr_flow_z);
    read2D_optional("grad_te_r", grad_te_r);
    read2D_optional("grad_te_t", grad_te_t);
    read2D_optional("grad_te_z", grad_te_z);
    read2D_optional("grad_ti_r", grad_ti_r);
    read2D_optional("grad_ti_t", grad_ti_t);
    read2D_optional("grad_ti_z", grad_ti_z);
    read2D_optional("epar", epar);
    if (hasDataset("q_par"))  { read2D("q_par",  q_par);  has_qheatflux = 1; }
    if (hasDataset("q_perp")) { read2D("q_perp", q_perp); has_qheatflux = 1; }

    // Regular-grid B (legacy /br, /bz, /bt) is no longer emitted by the
    // converters; B lives on mesh/vtx_b* (loaded below) and is
    // vertex-averaged per triangle after the MPI broadcast.
  }

  // Unified /equilibrium group written by convert_solps_plasma.py,
  // convert_oedge_plasma.py, and convert_s3x_plasma.py. Populates the
  // equ_* buffers directly from plasma.h5 so no external .equ file is
  // needed. Takes precedence over the SOLEDGE-legacy /psi layout
  // handled below.
  //
  // When this group is present, fix background behaves as if the
  // caller had passed `equilibrium <file>`; psi_norm_at() and
  // fix reflect/psi just work.
  if (hasDataset("equilibrium/psi") && hasDataset("equilibrium/r") &&
      hasDataset("equilibrium/z")) {
    auto read1D = [&](const std::string &name) -> std::vector<double> {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim = 0;
      sp.getSimpleExtentDims(&dim);
      std::vector<double> vec(dim);
      ds.read(vec.data(), H5::PredType::NATIVE_DOUBLE);
      return vec;
    };
    auto read_scalar = [&](const std::string &name, double &out) -> bool {
      if (!hasDataset(name)) return false;
      H5::DataSet ds = file.openDataSet(name);
      double tmp = 0.0;
      ds.read(&tmp, H5::PredType::NATIVE_DOUBLE);
      out = tmp;
      return true;
    };
    equ_r = read1D("equilibrium/r");
    equ_z = read1D("equilibrium/z");
    equ_jm = static_cast<int>(equ_r.size());
    equ_km = static_cast<int>(equ_z.size());

    H5::DataSet dspsi = file.openDataSet("equilibrium/psi");
    H5::DataSpace sp = dspsi.getSpace();
    hsize_t dims[2] = {0, 0};
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) == equ_km &&
        static_cast<int>(dims[1]) == equ_jm) {
      psirz.assign(static_cast<size_t>(equ_jm) * equ_km, 0.0);
      dspsi.read(psirz.data(), H5::PredType::NATIVE_DOUBLE);

      double btf_val = 0.0, rtf_val = 0.0, psib_val = 0.0;
      read_scalar("equilibrium/btf",  btf_val);
      read_scalar("equilibrium/rtf",  rtf_val);
      read_scalar("equilibrium/psib", psib_val);
      btf = btf_val;
      rtf = rtf_val;
      psib = psib_val;

      // psi_axis: prefer an explicit equilibrium/psi_axis dataset when
      // available (SOLEDGE3X writes psicore there, since mesh.h5 exposes
      // psicore/psisep but no jm/km-style toroidal axis). Fall back to
      // min(psirz), the SOLPS .equ convention for on-axis flux.
      double psi_axis_explicit = 0.0;
      if (read_scalar("equilibrium/psi_axis", psi_axis_explicit)) {
        psi_axis = psi_axis_explicit;
      } else {
        double psi_min = psirz[0];
        for (size_t k = 0; k < psirz.size(); ++k)
          if (psirz[k] < psi_min) psi_min = psirz[k];
        psi_axis = psi_min;
      }
      has_equ = 1;
      if (screen)
        fprintf(screen,
                "[background] loaded embedded equilibrium "
                "(jm=%d km=%d psi_axis=%.6e psib=%.6e)\n",
                equ_jm, equ_km, psi_axis, psib);
    }
  }

  // SOLEDGE-legacy /psi, /psicore, /psisep layout — only meaningful
  // when the regular (R, Z) grid exists (psi is 2D on r/z). Skipped
  // for mesh-only plasma.h5 files.
  else if (has_regular_grid && hasDataset("psi")) {
    H5::DataSet ds = file.openDataSet("psi");
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) == nz && static_cast<int>(dims[1]) == nr) {
      psirz.resize(static_cast<size_t>(nz) * nr);
      ds.read(psirz.data(), H5::PredType::NATIVE_DOUBLE);
      equ_jm = nr;
      equ_km = nz;
      equ_r  = rvals;
      equ_z  = zvals;
      auto read_scalar = [&](const std::string &name, double &out) -> bool {
        if (!hasDataset(name)) return false;
        H5::DataSet ds2 = file.openDataSet(name);
        double tmp = 0.0;
        ds2.read(&tmp, H5::PredType::NATIVE_DOUBLE);
        out = tmp;
        return true;
      };
      double psicore_val = 0.0, psisep_val = 0.0;
      bool have_core = read_scalar("psicore", psicore_val);
      bool have_sep  = read_scalar("psisep",  psisep_val);
      if (have_core && have_sep && psicore_val != psisep_val) {
        psi_axis = psicore_val;
        psib     = psisep_val;
        has_equ  = 1;
        if (screen)
          fprintf(screen,
                  "[background] loaded psi map (psicore=%.6e psisep=%.6e)\n",
                  psicore_val, psisep_val);
      }
    }
  }

  // Multi-ion species
  nion = 0;
  if (hasDataset("ion_species/charge_state_z")) {
    H5::DataSet ds = file.openDataSet("ion_species/charge_state_z");
    H5::DataSpace sp = ds.getSpace();
    hsize_t dim;
    sp.getSimpleExtentDims(&dim);
    nion = static_cast<int>(dim);
    ion_charge_z.resize(nion);
    ds.read(ion_charge_z.data(), H5::PredType::NATIVE_INT);
  }
  if (hasDataset("ion_species/mass_amu")) {
    H5::DataSet ds = file.openDataSet("ion_species/mass_amu");
    ion_mass_amu.resize(nion);
    ds.read(ion_mass_amu.data(), H5::PredType::NATIVE_DOUBLE);
  }
  // Variable-length string datasets for /ion_species/names and /elements.
  // Falls back to element parsing from the name if /elements is absent
  // (converters older than 2026-04-21 do not write it).
  auto read_strings = [&](const std::string &dset,
                          std::vector<std::string> &out) {
    if (!hasDataset(dset)) return false;
    H5::DataSet ds = file.openDataSet(dset);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dim = 0;
    sp.getSimpleExtentDims(&dim);
    H5::StrType stype = ds.getStrType();
    const bool variable = H5Tis_variable_str(stype.getId()) > 0;
    out.clear();
    out.reserve(dim);
    if (variable) {
      std::vector<char *> ptrs(dim, nullptr);
      ds.read(ptrs.data(), stype);
      for (hsize_t i = 0; i < dim; i++) {
        out.push_back(ptrs[i] ? std::string(ptrs[i]) : std::string());
      }
      H5Dvlen_reclaim(stype.getId(), sp.getId(), H5P_DEFAULT, ptrs.data());
    } else {
      const size_t slen = stype.getSize();
      std::vector<char> buf(dim * slen, 0);
      ds.read(buf.data(), stype);
      for (hsize_t i = 0; i < dim; i++) {
        const char *base = buf.data() + i * slen;
        size_t n_used = 0;
        while (n_used < slen && base[n_used] != '\0') ++n_used;
        out.emplace_back(base, n_used);
      }
    }
    return true;
  };
  read_strings("ion_species/names", ion_names);
  if (!read_strings("ion_species/elements", ion_elements)) {
    // Strip trailing charge-state suffix (e.g. "O2+" -> "O")
    ion_elements.clear();
    ion_elements.reserve(ion_names.size());
    for (const auto &name : ion_names) {
      size_t k = name.size();
      while (k > 0) {
        char c = name[k-1];
        if (c == '+' || c == '-' || (c >= '0' && c <= '9')) --k;
        else break;
      }
      ion_elements.push_back(name.substr(0, k));
    }
  }

  // 3D ion fields: (nion, nz, nr) -> flat
  auto read3D = [&](const std::string &name, std::vector<double> &out) {
    if (!hasDataset(name)) { out.assign(static_cast<size_t>(nion) * n, 0.0); return; }
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    out.resize(dims[0] * dims[1] * dims[2]);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  if (nion > 0) {
    read3D("ions/dens", ions_dens);
    read3D("ions/temp", ions_temp);
    read3D("ions/parr_flow", ions_upar);
  }

  // Mesh triangulation (optional)
  has_mesh = 0;
  if (hasDataset("mesh/vtx_r")) {
    has_mesh = 1;
    mesh_nion = 0;
    // Read mesh data
    auto read1Dint = [&](const std::string &name, std::vector<int> &out) {
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      out.resize(dim);
      ds.read(out.data(), H5::PredType::NATIVE_INT);
    };
    read1D("mesh/vtx_r", mesh_vtx_r);
    read1D("mesh/vtx_z", mesh_vtx_z);
    mesh_nvtx = static_cast<int>(mesh_vtx_r.size());

    if (hasDataset("mesh/triangles")) {
      H5::DataSet ds = file.openDataSet("mesh/triangles");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2];
      sp.getSimpleExtentDims(dims);
      mesh_ntri = static_cast<int>(dims[0]);
      mesh_tri.resize(mesh_ntri * 3);
      ds.read(mesh_tri.data(), H5::PredType::NATIVE_INT);
    }
    if (hasDataset("mesh/cell_index")) {
      read1Dint("mesh/cell_index", mesh_cell_idx);
    }

    auto read1D_mesh = [&](const std::string &name, std::vector<double> &out) {
      if (!hasDataset(name)) return;
      H5::DataSet ds = file.openDataSet(name);
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      out.resize(dim);
      ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };
    read1D_mesh("mesh/dens_e", mesh_ne);
    read1D_mesh("mesh/temp_e", mesh_te);
    read1D_mesh("mesh/temp_i", mesh_ti);
    read1D_mesh("mesh/dens_i", mesh_ni);
    read1D_mesh("mesh/parr_flow", mesh_upar);
    // Optional precomputed gradients. Missing → leave vectors empty;
    // consumers treat empty as "no mesh gradient available" and fall
    // back (to per-particle FD or zero).
    auto read1D_mesh_opt = [&](const std::string &name,
                                std::vector<double> &out) {
      if (hasDataset(name)) read1D_mesh(name, out);
      else out.clear();
    };
    read1D_mesh_opt("mesh/grad_te_r", mesh_grad_te_r);
    read1D_mesh_opt("mesh/grad_te_z", mesh_grad_te_z);
    read1D_mesh_opt("mesh/grad_ti_r", mesh_grad_ti_r);
    read1D_mesh_opt("mesh/grad_ti_z", mesh_grad_ti_z);
    if (hasDataset("mesh/q_par"))  { read1D_mesh("mesh/q_par",  mesh_q_par);  has_qheatflux = 1; }
    if (hasDataset("mesh/q_perp")) { read1D_mesh("mesh/q_perp", mesh_q_perp); has_qheatflux = 1; }
    read1D_mesh_opt("mesh/e_r", mesh_e_r);
    read1D_mesh_opt("mesh/e_z", mesh_e_z);
    read1D_mesh_opt("mesh/e_t", mesh_e_t);
    // Per-vertex B-field (optional; written by convert_{solps,s3x,oedge})
    read1D_mesh_opt("mesh/vtx_br", mesh_vtx_br);
    read1D_mesh_opt("mesh/vtx_bz", mesh_vtx_bz);
    read1D_mesh_opt("mesh/vtx_bt", mesh_vtx_bt);
    if (hasDataset("mesh/wall_face_area")) {
      read1D_mesh("mesh/wall_face_area", mesh_wall_face_area);
      has_mesh_wall_face_area = 1;
    } else {
      has_mesh_wall_face_area = 0;
      mesh_wall_face_area.clear();
    }
    if (hasDataset("mesh/wall_surf_cell")) {
      H5::DataSet ds = file.openDataSet("mesh/wall_surf_cell");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      mesh_wall_surf_cell.resize(dim);
      ds.read(mesh_wall_surf_cell.data(), H5::PredType::NATIVE_INT);
      has_mesh_wall_surf_cell = 1;
      if (hasDataset("mesh/wall_surf_area")) {
        mesh_wall_surf_area.resize(dim);
        file.openDataSet("mesh/wall_surf_area").read(
            mesh_wall_surf_area.data(), H5::PredType::NATIVE_DOUBLE);
      }
    } else {
      has_mesh_wall_surf_cell = 0;
      mesh_wall_surf_cell.clear();
      mesh_wall_surf_area.clear();
    }
    mesh_ncell = mesh_ne.empty() ? 0 : static_cast<int>(mesh_ne.size());

    if (hasDataset("mesh/ions/dens")) {
      H5::DataSet ds = file.openDataSet("mesh/ions/dens");
      H5::DataSpace sp = ds.getSpace();
      hsize_t dims[2];
      sp.getSimpleExtentDims(dims);
      mesh_nion = static_cast<int>(dims[0]);
      mesh_ions_dens.resize(static_cast<size_t>(mesh_nion) * mesh_ncell);
      ds.read(mesh_ions_dens.data(), H5::PredType::NATIVE_DOUBLE);
      if (hasDataset("mesh/ions/temp")) {
        mesh_ions_temp.resize(static_cast<size_t>(mesh_nion) * mesh_ncell);
        file.openDataSet("mesh/ions/temp").read(mesh_ions_temp.data(),
                                                H5::PredType::NATIVE_DOUBLE);
      }
      if (hasDataset("mesh/ions/parr_flow")) {
        mesh_ions_upar.resize(static_cast<size_t>(mesh_nion) * mesh_ncell);
        file.openDataSet("mesh/ions/parr_flow").read(mesh_ions_upar.data(),
                                                     H5::PredType::NATIVE_DOUBLE);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBackground::load_equilibrium()
{
  if (comm->me == 0 && screen)
    fprintf(screen, "[background] Reading equilibrium %s\n", equ_path.c_str());

  std::ifstream fin(equ_path);
  if (!fin.is_open())
    throw std::runtime_error("Cannot open equilibrium file: " + equ_path);

  auto readDoubles = [&](int count) -> std::vector<double> {
    std::vector<double> vals;
    vals.reserve(count);
    std::string line;
    while (static_cast<int>(vals.size()) < count && std::getline(fin, line)) {
      std::istringstream iss(line);
      double v;
      while (iss >> v) {
        vals.push_back(v);
        if (static_cast<int>(vals.size()) >= count) break;
      }
    }
    return vals;
  };

  equ_jm = equ_km = 0;
  btf = rtf = psib = 0.0;

  std::string line;
  while (std::getline(fin, line)) {
    if (line.find("jm") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> equ_jm;
    }
    if (line.find("km") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      if (line.find("km") < line.find("=")) {
        std::istringstream iss(line.substr(line.find("=") + 1));
        iss >> equ_km;
      }
    }
    if (line.find("psib") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> psib;
    }
    if (line.find("btf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> btf;
    }
    if (line.find("rtf") != std::string::npos && line.find("=") != std::string::npos
        && line.find(":=") == std::string::npos) {
      std::istringstream iss(line.substr(line.find("=") + 1));
      iss >> rtf;
    }
    if (line.find("r(1:jm)") != std::string::npos) break;
  }

  if (equ_jm < 2 || equ_km < 2)
    throw std::runtime_error("Equilibrium: failed to parse jm/km");

  equ_r = readDoubles(equ_jm);
  if (static_cast<int>(equ_r.size()) != equ_jm)
    throw std::runtime_error("Equilibrium: incomplete r array");

  while (std::getline(fin, line)) {
    if (line.find("z(1:km)") != std::string::npos) break;
  }

  equ_z = readDoubles(equ_km);
  if (static_cast<int>(equ_z.size()) != equ_km)
    throw std::runtime_error("Equilibrium: incomplete z array");

  while (std::getline(fin, line)) {
    if (line.find("psi(j,k)") != std::string::npos) break;
  }

  int total = equ_jm * equ_km;
  std::vector<double> psi_flat = readDoubles(total);
  if (static_cast<int>(psi_flat.size()) != total)
    throw std::runtime_error("Equilibrium: incomplete psi array");

  psirz.resize(total);
  for (int i = 0; i < total; i++)
    psirz[i] = psi_flat[i] + psib;

  // Find psi_axis (minimum psi in central region)
  psi_axis = 1e30;
  int j0 = equ_km / 4, j1 = 3 * equ_km / 4;
  int i0 = equ_jm / 4, i1 = 3 * equ_jm / 4;
  for (int j = j0; j < j1; j++)
    for (int i = i0; i < i1; i++) {
      double p = psirz[j * equ_jm + i];
      if (p < psi_axis) psi_axis = p;
    }

  has_equ = 1;
  fin.close();

  if (comm->me == 0 && screen)
    fprintf(screen,
      "  Equilibrium: jm=%d km=%d btf=%.3f rtf=%.3f psib=%.6e psi_axis=%.6e\n",
      equ_jm, equ_km, btf, rtf, psib, psi_axis);
}

/* ---------------------------------------------------------------------- */

void FixBackground::derive_bfield_from_equ()
{
  if (!has_equ || nr == 0 || nz == 0) return;

  size_t n = static_cast<size_t>(nz) * nr;
  br.resize(n);
  bz.resize(n);
  bt.resize(n);

  double dr_eq = equ_r[1] - equ_r[0];
  double dz_eq = equ_z[1] - equ_z[0];

  auto interp_psi = [&](double R, double Z) -> double {
    double fi = (R - equ_r.front()) / dr_eq;
    double fj = (Z - equ_z.front()) / dz_eq;
    int i0 = std::max(0, std::min((int)fi, equ_jm - 2));
    int j0 = std::max(0, std::min((int)fj, equ_km - 2));
    double s = std::max(0.0, std::min(1.0, fi - i0));
    double t = std::max(0.0, std::min(1.0, fj - j0));
    return (1-s)*(1-t)*psirz[j0*equ_jm+i0] + s*(1-t)*psirz[j0*equ_jm+i0+1]
         + (1-s)*t*psirz[(j0+1)*equ_jm+i0] + s*t*psirz[(j0+1)*equ_jm+i0+1];
  };

  double eps = 1e-4;
  for (int iz = 0; iz < nz; iz++) {
    for (int ir = 0; ir < nr; ir++) {
      double R = rvals[ir];
      double Z = zvals[iz];
      size_t idx = static_cast<size_t>(iz) * nr + ir;

      if (R < 0.01) { br[idx] = bz[idx] = bt[idx] = 0.0; continue; }

      double dpsi_dz = (interp_psi(R, Z + eps) - interp_psi(R, Z - eps)) / (2.0 * eps);
      br[idx] = -dpsi_dz / R;

      double dpsi_dr = (interp_psi(R + eps, Z) - interp_psi(R - eps, Z)) / (2.0 * eps);
      bz[idx] = dpsi_dr / R;

      bt[idx] = btf * rtf / R;
    }
  }

  has_bfield = 1;

  if (comm->me == 0 && screen)
    fprintf(screen, "  B-field derived from equilibrium on %d x %d grid\n", nr, nz);
}

/* ---------------------------------------------------------------------- */

void FixBackground::build_mesh_index()
{
  mesh_tri_rmin.clear();
  mesh_tri_rmax.clear();
  mesh_tri_zmin.clear();
  mesh_tri_zmax.clear();
  mapped_cr.clear();
  mapped_cz.clear();
  mapped_idx.clear();
  hash_grid.clear();
  hash_nr = hash_nz = 0;
  hash_rmin = hash_zmin = hash_dr = hash_dz = 0.0;

  if (!has_mesh || mesh_ntri <= 0 || mesh_tri.size() != static_cast<size_t>(mesh_ntri) * 3 ||
      mesh_vtx_r.empty() || mesh_vtx_z.empty()) {
    return;
  }

  mesh_tri_rmin.resize(mesh_ntri);
  mesh_tri_rmax.resize(mesh_ntri);
  mesh_tri_zmin.resize(mesh_ntri);
  mesh_tri_zmax.resize(mesh_ntri);

  for (int t = 0; t < mesh_ntri; t++) {
    const int v0 = mesh_tri[t*3+0];
    const int v1 = mesh_tri[t*3+1];
    const int v2 = mesh_tri[t*3+2];
    const double r0 = mesh_vtx_r[v0], r1 = mesh_vtx_r[v1], r2 = mesh_vtx_r[v2];
    const double z0 = mesh_vtx_z[v0], z1 = mesh_vtx_z[v1], z2 = mesh_vtx_z[v2];
    mesh_tri_rmin[t] = std::min({r0,r1,r2});
    mesh_tri_rmax[t] = std::max({r0,r1,r2});
    mesh_tri_zmin[t] = std::min({z0,z1,z2});
    mesh_tri_zmax[t] = std::max({z0,z1,z2});
    if (t < static_cast<int>(mesh_cell_idx.size()) && mesh_cell_idx[t] >= 0) {
      mapped_cr.push_back((r0 + r1 + r2) / 3.0);
      mapped_cz.push_back((z0 + z1 + z2) / 3.0);
      mapped_idx.push_back(t);
    }
  }

  if (mesh_tri_rmin.empty()) return;

  hash_rmin = *std::min_element(mesh_tri_rmin.begin(), mesh_tri_rmin.end());
  const double rmax = *std::max_element(mesh_tri_rmax.begin(), mesh_tri_rmax.end());
  hash_zmin = *std::min_element(mesh_tri_zmin.begin(), mesh_tri_zmin.end());
  const double zmax = *std::max_element(mesh_tri_zmax.begin(), mesh_tri_zmax.end());

  hash_nr = 100;
  hash_nz = 100;
  hash_dr = (rmax - hash_rmin) / hash_nr + 1.0e-12;
  hash_dz = (zmax - hash_zmin) / hash_nz + 1.0e-12;
  hash_grid.assign(static_cast<size_t>(hash_nr) * hash_nz, std::vector<int>());

  for (int t = 0; t < mesh_ntri; t++) {
    const int ir0 = std::max(0, static_cast<int>((mesh_tri_rmin[t] - hash_rmin) / hash_dr));
    const int ir1 = std::min(hash_nr - 1, static_cast<int>((mesh_tri_rmax[t] - hash_rmin) / hash_dr));
    const int iz0 = std::max(0, static_cast<int>((mesh_tri_zmin[t] - hash_zmin) / hash_dz));
    const int iz1 = std::min(hash_nz - 1, static_cast<int>((mesh_tri_zmax[t] - hash_zmin) / hash_dz));
    for (int iz = iz0; iz <= iz1; iz++) {
      for (int ir = ir0; ir <= ir1; ir++) {
        hash_grid[iz * hash_nr + ir].push_back(t);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int FixBackground::find_mesh_triangle(double R, double Z) const
{
  if (!has_mesh || mesh_ntri <= 0 || mesh_tri.empty()) return -1;
  if (hash_nr <= 0 || hash_nz <= 0 || hash_grid.empty()) return -1;

  // Query outside the hash-grid bbox: no triangle can match there anyway
  // (all triangle bboxes are inside the hash grid by construction), so
  // return -1 directly instead of falling through to an O(N_triangles)
  // linear scan that always returns -1.
  const int ir = static_cast<int>((R - hash_rmin) / hash_dr);
  const int iz = static_cast<int>((Z - hash_zmin) / hash_dz);
  if (ir < 0 || ir >= hash_nr || iz < 0 || iz >= hash_nz) return -1;

  const auto &candidates = hash_grid[iz * hash_nr + ir];
  for (int t : candidates) {
    const int v0 = mesh_tri[t*3+0];
    const int v1 = mesh_tri[t*3+1];
    const int v2 = mesh_tri[t*3+2];
    const double r0 = mesh_vtx_r[v0], z0 = mesh_vtx_z[v0];
    const double r1 = mesh_vtx_r[v1], z1 = mesh_vtx_z[v1];
    const double r2 = mesh_vtx_r[v2], z2 = mesh_vtx_z[v2];
    const double d = (r1-r0)*(z2-z0) - (r2-r0)*(z1-z0);
    if (std::fabs(d) < 1.0e-30) continue;
    const double a = ((R-r0)*(z2-z0) - (r2-r0)*(Z-z0)) / d;
    const double b = ((r1-r0)*(Z-z0) - (R-r0)*(z1-z0)) / d;
    if (a >= -1.0e-10 && b >= -1.0e-10 && (a+b) <= 1.0+1.0e-10) return t;
  }
  return -1;
}

/* ---------------------------------------------------------------------- */

int FixBackground::find_nearest_mapped_triangle(double R, double Z, double max_dist) const
{
  const double max_d2 = max_dist * max_dist;
  double best_d2 = max_d2;
  int best = -1;
  for (int i = 0; i < static_cast<int>(mapped_idx.size()); i++) {
    const double dr = mapped_cr[i] - R;
    const double dz = mapped_cz[i] - Z;
    const double d2 = dr*dr + dz*dz;
    if (d2 < best_d2) {
      best_d2 = d2;
      best = mapped_idx[i];
    }
  }
  return best;
}

/* ---------------------------------------------------------------------- */

int FixBackground::mesh_cell_at(double R, double Z, double max_dist) const
{
  if (!has_mesh || mesh_ncell <= 0 || mesh_cell_idx.empty()) return -1;
  int tri = find_mesh_triangle(R, Z);
  if (tri < 0 || tri >= static_cast<int>(mesh_cell_idx.size()) || mesh_cell_idx[tri] < 0)
    tri = find_nearest_mapped_triangle(R, Z, max_dist);
  if (tri < 0 || tri >= static_cast<int>(mesh_cell_idx.size())) return -1;
  const int cell = mesh_cell_idx[tri];
  if (cell < 0 || cell >= mesh_ncell) return -1;
  return cell;
}

/* ----------------------------------------------------------------------
   Build per-SPARTA-cell mesh-cell index at cell centroids. Populates
   cell_mesh_cell[icell] for icell in [0, grid->nlocal). For split
   sub-cells, uses the parent cell's bbox since sub-cells share a parent
   bbox and the plasma data only varies on mesh-cell scale anyway.

   Coordinate mapping matches update.cpp's per-particle xyz_to_rz(), so
   the cached lookup is byte-exact with the per-particle fallback path.
------------------------------------------------------------------------- */

void FixBackground::build_cell_mesh_index()
{
  const int nglocal = grid->nlocal;
  cell_mesh_cell.assign(nglocal, -1);
  cell_mesh_stamp_n = nglocal;
  cell_mesh_stamp_id = (nglocal > 0 && grid->cells) ? grid->cells[0].id : -1;
  if (!has_mesh || nglocal == 0 || mesh_cell_idx.empty()) return;

  Grid::ChildCell *cells = grid->cells;
  Grid::SplitInfo *sinfo = grid->sinfo;
  if (!cells) return;
  const int dim = domain->dimension;
  const int axi = domain->axisymmetric;

  const int ncell_idx = static_cast<int>(mesh_cell_idx.size());

  for (int icell = 0; icell < nglocal; icell++) {
    int pcell = icell;
    // Sub-cells share the parent cell's bbox. sinfo may be NULL on ranks
    // that have no split cells; guard before dereferencing.
    if (sinfo && cells[icell].nsplit <= 0 && cells[icell].isplit >= 0)
      pcell = sinfo[cells[icell].isplit].icell;
    const double *lo = cells[pcell].lo;
    const double *hi = cells[pcell].hi;
    double xc[3] = {0.5 * (lo[0] + hi[0]),
                    0.5 * (lo[1] + hi[1]),
                    0.5 * (lo[2] + hi[2])};

    // Physical (R, Z) — matches OpenEdge::sparta_to_RZ convention used by
    // all other pd consumers (fix thermal_force, cross_diffusion, emit/surf/
    // recycle). In axi: SPARTA x = Z_axis, y = R_radial.
    // 3D Cartesian honors the column_axis offset (default 0,0).
    double R, Z;
    if (axi)          { Z = xc[0]; R = xc[1]; }
    else if (dim == 2) { R = xc[0]; Z = xc[1]; }
    else {
      const double dx = xc[0] - column_x0;
      const double dy = xc[1] - column_y0;
      R = std::sqrt(dx*dx + dy*dy);
      Z = xc[2];
    }

    // Fast hash-grid lookup only — do NOT run find_nearest_mapped_triangle.
    // Cells outside the plasma mesh (wall shadow, vacuum) correctly get -1;
    // the pcache loop's fallback path then returns 0 for those particles,
    // which is what we want physically (no plasma → no reactions). The
    // O(N_triangles) linear-scan fallback would dominate the rebuild cost
    // when adapt/balance fires frequently.
    const int tri = find_mesh_triangle(R, Z);
    if (tri >= 0 && tri < ncell_idx) {
      const int cell = mesh_cell_idx[tri];
      if (cell >= 0 && cell < mesh_ncell) cell_mesh_cell[icell] = cell;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixBackground::grid_changed()
{
  cell_mesh_cell.clear();
  cell_mesh_stamp_n  = -1;
  cell_mesh_stamp_id = -1;
}

/* ---------------------------------------------------------------------- */

const std::vector<double> *FixBackground::mesh_field_for(const std::vector<double> &field) const
{
  if (!has_mesh) return nullptr;
  if (&field == &dens_e) return &mesh_ne;
  if (&field == &temp_e) return &mesh_te;
  if (&field == &dens_i) return &mesh_ni;
  if (&field == &temp_i) return &mesh_ti;
  if (&field == &parr_flow) return &mesh_upar;
  return nullptr;
}

/* ---------------------------------------------------------------------- */

double FixBackground::interp2D(const std::vector<double> &field,
                                double R, double Z, int icell) const
{
  if (const std::vector<double> *mesh_field = mesh_field_for(field)) {
    // Fast path: O(1) lookup when the caller knows the SPARTA cell and
    // the cell-indexed cache is built. Skips the hash-grid triangle
    // search entirely. Used by fix thermal_force / cross_diffusion /
    // coll_nanbu per-particle loops.
    if (icell >= 0 && icell < static_cast<int>(cell_mesh_cell.size())) {
      const int mc = cell_mesh_cell[icell];
      if (mc >= 0 && mc < static_cast<int>(mesh_field->size()))
        return (*mesh_field)[mc];
      // icell maps to no-mesh-cell (-1): fall through to stencil. Same
      // behaviour as the R,Z path when mesh_cell_at returns -1.
    } else {
      const int cell = mesh_cell_at(R, Z);
      if (cell >= 0 && cell < static_cast<int>(mesh_field->size()))
        return (*mesh_field)[cell];
    }
  }

  if (field.empty() || nr < 2 || nz < 2) return 0.0;

  double Rc = std::min(std::max(R, rvals.front()), rvals.back());
  double Zc = std::min(std::max(Z, zvals.front()), zvals.back());

  double dr = rvals[1] - rvals[0];
  double dz = zvals[1] - zvals[0];
  double fi = (Rc - rvals.front()) / dr;
  double fj = (Zc - zvals.front()) / dz;

  int ir0 = std::max(0, std::min((int)fi, nr - 2));
  int iz0 = std::max(0, std::min((int)fj, nz - 2));
  double s = std::max(0.0, std::min(1.0, fi - ir0));
  double t = std::max(0.0, std::min(1.0, fj - iz0));

  return (1-s)*(1-t)*field[iz0*nr+ir0] + s*(1-t)*field[iz0*nr+ir0+1]
       + (1-s)*t*field[(iz0+1)*nr+ir0] + s*t*field[(iz0+1)*nr+ir0+1];
}

/* ---------------------------------------------------------------------- */

void FixBackground::bfield_at(double R, double Z,
                               double &Br_out, double &Bz_out,
                               double &Bt_out, int icell) const
{
  // Mesh-native B from plasma.h5 mesh/vtx_b* (vertex-averaged per
  // triangle) takes precedence when loaded. New converters emit B only
  // on the unstructured mesh; the regular-grid br/bz/bt arrays are
  // populated only by derive_bfield_from_equ() and the constant-profile
  // fallback.
  if (!mesh_tri_br.empty()) {
    const int tri = find_mesh_triangle(R, Z);
    if (tri >= 0 && tri < static_cast<int>(mesh_tri_br.size())) {
      Br_out = mesh_tri_br[tri];
      Bz_out = mesh_tri_bz[tri];
      Bt_out = mesh_tri_bt[tri];
      return;
    }
    // outside the mesh footprint: fall through to regular-grid / zero.
  }

  Br_out = interp2D(br, R, Z, icell);
  Bz_out = interp2D(bz, R, Z, icell);
  Bt_out = interp2D(bt, R, Z, icell);
}

/* ----------------------------------------------------------------------
   Cylindrical-derivative B query (mirrors
   ComputePlasmaFields::query_bfield_at_point). Branch order:
     1. mesh-native B (per-triangle vertex average) — no derivatives
     2. regular-grid br/bz/bt with finite-difference derivatives
     3. equilibrium ψ analytic derivatives
   For GCA smoothness, prefer the equilibrium branch when has_equ is
   set: derive_bfield_from_equ() also writes br/bz/bt on the regular
   grid, which would otherwise short-circuit branch 2 with finite-
   difference noise.
------------------------------------------------------------------------- */

MagneticFieldFileDataParams
FixBackground::query_bfield_at_point(const double xyz[3]) const
{
  MagneticFieldFileDataParams B{};

  const int dim = domain->dimension;
  const bool axi = domain->axisymmetric;
  double R, Z;
  OpenEdge::sparta_to_RZ(xyz, dim, axi, R, Z, column_x0, column_y0);
  B.r = R;
  B.z = Z;

  // 1. Mesh-native B (per-triangle vertex average) — no derivatives.
  if (!mesh_tri_br.empty()) {
    const int tri = find_mesh_triangle(R, Z);
    if (tri >= 0 && tri < static_cast<int>(mesh_tri_br.size())) {
      B.br = mesh_tri_br[tri];
      B.bz = mesh_tri_bz[tri];
      B.bt = mesh_tri_bt[tri];
      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
      return B;
    }
    // outside mesh footprint: fall through.
  }

  // 3. Equilibrium analytic derivatives — preferred over branch 2 when
  // available because derive_bfield_from_equ() populates br/bz/bt from
  // the same ψ map, so branch 2 would just be a noisier finite-difference
  // version of the same physics.
  if (has_equ && equ_jm >= 3 && equ_km >= 3 && !psirz.empty() && R > 1.0e-10) {
    const int jm = equ_jm;
    const int km = equ_km;
    const double dr = equ_r[1] - equ_r[0];
    const double dz = equ_z[1] - equ_z[0];
    if (dr > 0.0 && dz > 0.0) {
      double fj = (R - equ_r[0]) / dr;
      double fk = (Z - equ_z[0]) / dz;
      int jc = static_cast<int>(std::round(fj));
      int kc = static_cast<int>(std::round(fk));
      jc = std::max(1, std::min(jc, jm - 2));
      kc = std::max(1, std::min(kc, km - 2));

      auto P = [&](int k, int j) -> double {
        return psirz[static_cast<size_t>(k) * jm + j];
      };

      const double dR = equ_r[jc+1] - equ_r[jc-1];
      const double dZ = equ_z[kc+1] - equ_z[kc-1];
      const double dpsi_dR = (P(kc, jc+1) - P(kc, jc-1)) / dR;
      const double dpsi_dZ = (P(kc+1, jc) - P(kc-1, jc)) / dZ;

      const double dR1 = equ_r[jc+1] - equ_r[jc];
      const double dR0 = equ_r[jc] - equ_r[jc-1];
      const double dZ1 = equ_z[kc+1] - equ_z[kc];
      const double dZ0 = equ_z[kc] - equ_z[kc-1];

      const double d2psi_dR2 = 2.0 * (P(kc, jc+1) / (dR1*(dR1+dR0))
                                      - P(kc, jc) / (dR1*dR0)
                                      + P(kc, jc-1) / (dR0*(dR1+dR0)));
      const double d2psi_dZ2 = 2.0 * (P(kc+1, jc) / (dZ1*(dZ1+dZ0))
                                      - P(kc, jc) / (dZ1*dZ0)
                                      + P(kc-1, jc) / (dZ0*(dZ1+dZ0)));
      const double d2psi_dRdZ = (P(kc+1, jc+1) - P(kc+1, jc-1)
                                 - P(kc-1, jc+1) + P(kc-1, jc-1)) / (dR * dZ);

      const double invR = 1.0 / R;
      const double invR2 = invR * invR;

      B.br = -dpsi_dZ * invR;
      B.bz = dpsi_dR * invR;
      B.bt = btf * rtf * invR;

      B.dBr_dr = dpsi_dZ * invR2 - d2psi_dRdZ * invR;
      B.dBr_dz = -d2psi_dZ2 * invR;
      B.dBz_dr = -dpsi_dR * invR2 + d2psi_dR2 * invR;
      B.dBz_dz = d2psi_dRdZ * invR;
      B.dBt_dr = -btf * rtf * invR2;
      B.dBt_dz = 0.0;

      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
      if (B.Bmag > 0.0) {
        B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr + B.bz*B.dBz_dr) / B.Bmag;
        B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz + B.bz*B.dBz_dz) / B.Bmag;
      }
      return B;
    }
  }

  // 2. Regular-grid bilinear with finite-difference derivatives.
  if (has_bfield && !br.empty() && nr >= 2 && nz >= 2) {
    const double dr = rvals[1] - rvals[0];
    const double dz = zvals[1] - zvals[0];
    if (dr > 0.0 && dz > 0.0) {
      const double Rc = std::min(std::max(R, rvals.front()), rvals.back());
      const double Zc = std::min(std::max(Z, zvals.front()), zvals.back());
      const double fi = (Rc - rvals.front()) / dr;
      const double fj = (Zc - zvals.front()) / dz;
      const int ir0 = std::max(0, std::min((int)fi, nr - 2));
      const int iz0 = std::max(0, std::min((int)fj, nz - 2));
      const double s = std::max(0.0, std::min(1.0, fi - ir0));
      const double t = std::max(0.0, std::min(1.0, fj - iz0));
      const size_t k00 = static_cast<size_t>(iz0) * nr + ir0;
      const size_t k10 = k00 + 1;
      const size_t k01 = k00 + nr;
      const size_t k11 = k01 + 1;

      auto interp = [&](const std::vector<double> &f) {
        return (1-s)*(1-t)*f[k00] + s*(1-t)*f[k10]
             + (1-s)*t*f[k01] + s*t*f[k11];
      };
      auto grad_r = [&](const std::vector<double> &f) {
        return ((1-t)*(f[k10]-f[k00]) + t*(f[k11]-f[k01])) / dr;
      };
      auto grad_z = [&](const std::vector<double> &f) {
        return ((1-s)*(f[k01]-f[k00]) + s*(f[k11]-f[k10])) / dz;
      };

      B.br = interp(br); B.bz = interp(bz); B.bt = interp(bt);
      B.dBr_dr = grad_r(br); B.dBr_dz = grad_z(br);
      B.dBt_dr = grad_r(bt); B.dBt_dz = grad_z(bt);
      B.dBz_dr = grad_r(bz); B.dBz_dz = grad_z(bz);
      B.Bmag = std::sqrt(B.br*B.br + B.bt*B.bt + B.bz*B.bz);
      if (B.Bmag > 0.0) {
        B.dBmag_dr = (B.br*B.dBr_dr + B.bt*B.dBt_dr + B.bz*B.dBz_dr) / B.Bmag;
        B.dBmag_dz = (B.br*B.dBr_dz + B.bt*B.dBt_dz + B.bz*B.dBz_dz) / B.Bmag;
      }
      return B;
    }
  }

  return B;
}

/* ---------------------------------------------------------------------- */

double FixBackground::psi_norm_at(double R, double Z) const
{
  if (!has_equ || psirz.empty()) return 1.0;

  double dr_eq = equ_r[1] - equ_r[0];
  double dz_eq = equ_z[1] - equ_z[0];

  double Rc = std::min(std::max(R, equ_r.front()), equ_r.back());
  double Zc = std::min(std::max(Z, equ_z.front()), equ_z.back());

  double fi = (Rc - equ_r.front()) / dr_eq;
  double fj = (Zc - equ_z.front()) / dz_eq;
  int i0 = std::max(0, std::min((int)fi, equ_jm - 2));
  int j0 = std::max(0, std::min((int)fj, equ_km - 2));
  double s = std::max(0.0, std::min(1.0, fi - i0));
  double t = std::max(0.0, std::min(1.0, fj - j0));

  double psi = (1-s)*(1-t)*psirz[j0*equ_jm+i0] + s*(1-t)*psirz[j0*equ_jm+i0+1]
             + (1-s)*t*psirz[(j0+1)*equ_jm+i0] + s*t*psirz[(j0+1)*equ_jm+i0+1];

  double denom = psib - psi_axis;
  if (std::abs(denom) < 1e-30) return 1.0;
  return (psi - psi_axis) / denom;
}
