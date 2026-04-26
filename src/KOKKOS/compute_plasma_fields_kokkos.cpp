/* ----------------------------------------------------------------------
   OpenEdge: plasma fields compute — Kokkos wrapper.
------------------------------------------------------------------------- */

#include "compute_plasma_fields_kokkos.h"
#include "grid.h"
#include "memory_kokkos.h"

using namespace SPARTA_NS;

ComputePlasmaFieldsKokkos::ComputePlasmaFieldsKokkos(
    SPARTA *sparta, int narg, char **arg) :
  ComputePlasmaFields(sparta, narg, arg)
{
  kokkos_flag = 1;
  maxgrid_kk = 0;
}

ComputePlasmaFieldsKokkos::~ComputePlasmaFieldsKokkos()
{
  if (copymode) return;
}

void ComputePlasmaFieldsKokkos::compute_per_grid()
{
  ComputePlasmaFields::compute_per_grid();
  sync_to_device();
  sync_equilibrium_to_device();   // idempotent; only first call allocates
  sync_mesh_to_device();          // idempotent; only first call allocates
}

/* ----------------------------------------------------------------------
   Mirror ComputePlasmaFields::equ_data onto device-resident Views so
   the GPU pusher can run bilinear B interpolation at particle position.
   Idempotent: re-running with the same dimensions is a no-op besides
   the deep_copy. has_equilibrium flag is mirrored to d_has_equilibrium
   so callers can skip the kernel branch when no equilibrium is loaded.
------------------------------------------------------------------------- */
void ComputePlasmaFieldsKokkos::sync_equilibrium_to_device()
{
  if (!has_equilibrium || equ_data.jm < 3 || equ_data.km < 3) {
    d_has_equilibrium = 0;
    return;
  }

  const int jm = equ_data.jm;
  const int km = equ_data.km;

  if (d_equ_jm != jm || d_equ_km != km) {
    d_equ_r   = Kokkos::View<double*,  DeviceType>("equ_r",  jm);
    d_equ_z   = Kokkos::View<double*,  DeviceType>("equ_z",  km);
    d_equ_psi = Kokkos::View<double**, DeviceType>("equ_psi", km, jm);
    d_equ_jm = jm;
    d_equ_km = km;
  }

  auto h_r   = Kokkos::create_mirror_view(d_equ_r);
  auto h_z   = Kokkos::create_mirror_view(d_equ_z);
  auto h_psi = Kokkos::create_mirror_view(d_equ_psi);
  for (int j = 0; j < jm; j++) h_r(j) = equ_data.r[j];
  for (int k = 0; k < km; k++) h_z(k) = equ_data.z[k];
  for (int k = 0; k < km; k++)
    for (int j = 0; j < jm; j++)
      h_psi(k, j) = equ_data.psi[k][j];
  Kokkos::deep_copy(d_equ_r,   h_r);
  Kokkos::deep_copy(d_equ_z,   h_z);
  Kokkos::deep_copy(d_equ_psi, h_psi);

  d_equ_btf = equ_data.btf;
  d_equ_rtf = equ_data.rtf;
  d_has_equilibrium = 1;
}

/* ----------------------------------------------------------------------
   Mirror ComputePlasmaFields::plasma_data mesh + spatial hash onto device
   Views, so the GPU pusher can do the same triangulated-B lookup the CPU
   does for SOLPS / SOLEDGE3X plasmas. The host-side hash_grid is a
   variable-length-per-bin std::vector<std::vector<int>>; we flatten it
   into a CSR pair (offset, entries) for device-friendly access.

   Idempotent — first call allocates, subsequent calls are a no-op when
   triangle counts haven't changed. Called from compute_per_grid().
------------------------------------------------------------------------- */
void ComputePlasmaFieldsKokkos::sync_mesh_to_device()
{
  const auto &pd = plasma_data;
  if (!pd.has_mesh || pd.mesh_tri_br.empty()) {
    d_has_mesh_b = 0;
    return;
  }

  const int nvtx = pd.mesh_nvtx;
  const int ntri = pd.mesh_ntri;
  if (nvtx <= 0 || ntri <= 0) { d_has_mesh_b = 0; return; }

  // Allocate device views once (re-allocate if mesh changed)
  if (d_mesh_ntri != ntri) {
    d_mesh_vtx_r    = Kokkos::View<double*, DeviceType>("mesh_vtx_r",    nvtx);
    d_mesh_vtx_z    = Kokkos::View<double*, DeviceType>("mesh_vtx_z",    nvtx);
    d_mesh_tri      = Kokkos::View<int*,    DeviceType>("mesh_tri",      ntri*3);
    d_mesh_tri_br   = Kokkos::View<double*, DeviceType>("mesh_tri_br",   ntri);
    d_mesh_tri_bz   = Kokkos::View<double*, DeviceType>("mesh_tri_bz",   ntri);
    d_mesh_tri_bt   = Kokkos::View<double*, DeviceType>("mesh_tri_bt",   ntri);
    d_mesh_tri_rmin = Kokkos::View<double*, DeviceType>("mesh_tri_rmin", ntri);
    d_mesh_tri_rmax = Kokkos::View<double*, DeviceType>("mesh_tri_rmax", ntri);
    d_mesh_tri_zmin = Kokkos::View<double*, DeviceType>("mesh_tri_zmin", ntri);
    d_mesh_tri_zmax = Kokkos::View<double*, DeviceType>("mesh_tri_zmax", ntri);
    d_mesh_ntri = ntri;
  }

  // Copy primary arrays
  auto h_vtx_r    = Kokkos::create_mirror_view(d_mesh_vtx_r);
  auto h_vtx_z    = Kokkos::create_mirror_view(d_mesh_vtx_z);
  auto h_tri      = Kokkos::create_mirror_view(d_mesh_tri);
  auto h_tri_br   = Kokkos::create_mirror_view(d_mesh_tri_br);
  auto h_tri_bz   = Kokkos::create_mirror_view(d_mesh_tri_bz);
  auto h_tri_bt   = Kokkos::create_mirror_view(d_mesh_tri_bt);
  auto h_tri_rmin = Kokkos::create_mirror_view(d_mesh_tri_rmin);
  auto h_tri_rmax = Kokkos::create_mirror_view(d_mesh_tri_rmax);
  auto h_tri_zmin = Kokkos::create_mirror_view(d_mesh_tri_zmin);
  auto h_tri_zmax = Kokkos::create_mirror_view(d_mesh_tri_zmax);

  for (int i = 0; i < nvtx; i++) {
    h_vtx_r(i) = pd.mesh_vtx_r[i];
    h_vtx_z(i) = pd.mesh_vtx_z[i];
  }
  for (int t = 0; t < ntri; t++) {
    h_tri(3*t+0)  = pd.mesh_tri[3*t+0];
    h_tri(3*t+1)  = pd.mesh_tri[3*t+1];
    h_tri(3*t+2)  = pd.mesh_tri[3*t+2];
    h_tri_br(t)   = pd.mesh_tri_br[t];
    h_tri_bz(t)   = pd.mesh_tri_bz[t];
    h_tri_bt(t)   = pd.mesh_tri_bt[t];
    h_tri_rmin(t) = pd.mesh_tri_rmin[t];
    h_tri_rmax(t) = pd.mesh_tri_rmax[t];
    h_tri_zmin(t) = pd.mesh_tri_zmin[t];
    h_tri_zmax(t) = pd.mesh_tri_zmax[t];
  }

  Kokkos::deep_copy(d_mesh_vtx_r,    h_vtx_r);
  Kokkos::deep_copy(d_mesh_vtx_z,    h_vtx_z);
  Kokkos::deep_copy(d_mesh_tri,      h_tri);
  Kokkos::deep_copy(d_mesh_tri_br,   h_tri_br);
  Kokkos::deep_copy(d_mesh_tri_bz,   h_tri_bz);
  Kokkos::deep_copy(d_mesh_tri_bt,   h_tri_bt);
  Kokkos::deep_copy(d_mesh_tri_rmin, h_tri_rmin);
  Kokkos::deep_copy(d_mesh_tri_rmax, h_tri_rmax);
  Kokkos::deep_copy(d_mesh_tri_zmin, h_tri_zmin);
  Kokkos::deep_copy(d_mesh_tri_zmax, h_tri_zmax);

  // Flatten host CSR spatial hash
  const int nr = pd.hash_nr, nz = pd.hash_nz;
  if (nr > 0 && nz > 0 && !pd.hash_grid.empty()) {
    const int nbins = nr * nz;
    std::vector<int> offset(nbins + 1, 0);
    for (int b = 0; b < nbins; b++)
      offset[b + 1] = offset[b] + static_cast<int>(pd.hash_grid[b].size());
    const int ntotal = offset[nbins];

    if (static_cast<int>(d_hash_offset.extent(0))  != nbins + 1)
      d_hash_offset  = Kokkos::View<int*, DeviceType>("hash_offset", nbins + 1);
    if (static_cast<int>(d_hash_entries.extent(0)) != ntotal)
      d_hash_entries = Kokkos::View<int*, DeviceType>("hash_entries", ntotal);

    auto h_offset  = Kokkos::create_mirror_view(d_hash_offset);
    auto h_entries = Kokkos::create_mirror_view(d_hash_entries);
    for (int i = 0; i <= nbins; i++) h_offset(i) = offset[i];
    int k = 0;
    for (int b = 0; b < nbins; b++)
      for (int t : pd.hash_grid[b]) h_entries(k++) = t;
    Kokkos::deep_copy(d_hash_offset,  h_offset);
    Kokkos::deep_copy(d_hash_entries, h_entries);

    d_mesh_hash_rmin = pd.hash_rmin;
    d_mesh_hash_zmin = pd.hash_zmin;
    d_mesh_hash_dr   = pd.hash_dr;
    d_mesh_hash_dz   = pd.hash_dz;
    d_mesh_hash_nr   = nr;
    d_mesh_hash_nz   = nz;
  } else {
    d_mesh_hash_nr = d_mesh_hash_nz = 0;  // brute-force fallback on device
  }

  d_has_mesh_b = 1;
}

void ComputePlasmaFieldsKokkos::sync_to_device()
{
  int ng = grid->nlocal;
  if (ng <= 0) return;

  int nc = size_per_grid_cols;
  if (nc == 0) nc = 1;

  // Guard: host arrays may not be allocated yet
  if (size_per_grid_cols == 0 && !vector_grid) return;
  if (size_per_grid_cols > 0 && !array_grid) return;

  if (ng > maxgrid_kk || maxgrid_kk == 0) {
    maxgrid_kk = grid->maxlocal;
    k_array_grid = DAT::tdual_float_2d_lr("plasma_fields:array", maxgrid_kk, nc);
    d_array_grid = k_array_grid.d_view;
  }

  auto h_arr = k_array_grid.h_view;

  if (size_per_grid_cols == 0) {
    for (int i = 0; i < ng; i++)
      h_arr(i, 0) = vector_grid[i];
  } else {
    for (int i = 0; i < ng; i++)
      for (int j = 0; j < size_per_grid_cols; j++)
        h_arr(i, j) = array_grid[i][j];
  }

  k_array_grid.modify_host();
  k_array_grid.sync_device();
  d_array_grid = k_array_grid.d_view;
}
