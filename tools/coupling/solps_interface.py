"""
SOLPS-ITER file interface for OpenEdge direct coupling.

Provides functions to:
  - Read SOLPS geometry and plasma state from .mat files
  - Write source2d.* files (Li source from OpenEdge evaporation)
  - Write b2.sources.profile namelist chain
  - Convert SOLPS plasma state to OpenEdge HDF5 format
  - Build SOLPS mesh triangulation for cell mapping

Usage:
  from solps_interface import SolpsInterface
  si = SolpsInterface('/path/to/solps/run')
  si.load_geometry()
  si.load_plasma_state()
  si.write_source2d(source_array, species_idx, filename)
  si.write_plasma_h5(output_path)
"""

import numpy as np
import os
import sys
import h5py

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class SolpsInterface:
    """Interface to read/write SOLPS-ITER files for coupling with OpenEdge."""

    def __init__(self, run_dir, baserun_dir=None):
        self.run_dir = run_dir
        self.baserun_dir = baserun_dir or os.path.join(os.path.dirname(run_dir), 'baserun')
        self.geo = None    # geometry data
        self.state = None  # plasma state data
        self.nx = 0
        self.ny = 0
        self.ns = 0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def load_geometry(self, path=None):
        """Load SOLPS geometry from b2fgmtry.mat."""
        if not HAS_SCIPY:
            raise ImportError("scipy required for .mat file reading")
        path = path or os.path.join(self.run_dir, 'b2fgmtry.mat')
        m = sio.loadmat(path)
        self.geo = m['Geo'][0, 0]
        self.nx = int(self.geo['nx'].flat[0])
        self.ny = int(self.geo['ny'].flat[0])
        return self.geo

    def get_cell_centers(self):
        """Return (R, Z) cell center arrays, shape (nx+2, ny+2)."""
        if self.geo is None:
            self.load_geometry()
        # crx/cry are corner coords: shape (nx+2, ny+2, 4)
        # cell center = average of 4 corners
        crx = self.geo['crx']  # (nx+2, ny+2, 4)
        cry = self.geo['cry']
        R_cen = np.mean(crx, axis=2)  # (nx+2, ny+2)
        Z_cen = np.mean(cry, axis=2)
        return R_cen, Z_cen

    def get_cell_volumes(self):
        """Return cell volumes, shape (nx+2, ny+2)."""
        if self.geo is None:
            self.load_geometry()
        return np.array(self.geo['vol'])

    def build_triangulation(self):
        """Build triangle mesh from SOLPS cell corners for point-in-cell lookup.

        Each SOLPS cell (quadrilateral) is split into 2 triangles.
        Returns:
            vtx_r, vtx_z: vertex coordinates
            triangles: (ntri, 3) vertex indices
            cell_idx: (ntri,) SOLPS flat cell index for each triangle
        """
        if self.geo is None:
            self.load_geometry()

        crx = self.geo['crx']  # (nx+2, ny+2, 4)
        cry = self.geo['cry']

        nxp2, nyp2, _ = crx.shape
        vertices_r = []
        vertices_z = []
        triangles = []
        cell_indices = []
        vertex_map = {}

        def add_vertex(r, z):
            key = (round(r, 10), round(z, 10))
            if key not in vertex_map:
                vertex_map[key] = len(vertices_r)
                vertices_r.append(r)
                vertices_z.append(z)
            return vertex_map[key]

        for ix in range(nxp2):
            for iy in range(nyp2):
                corners_r = crx[ix, iy, :]
                corners_z = cry[ix, iy, :]
                # Skip cells with zero-area (guard cells)
                area = 0.5 * abs(
                    (corners_r[1] - corners_r[0]) * (corners_z[2] - corners_z[0]) -
                    (corners_r[2] - corners_r[0]) * (corners_z[1] - corners_z[0])
                )
                if area < 1e-20:
                    continue

                v = [add_vertex(corners_r[k], corners_z[k]) for k in range(4)]
                flat_idx = ix * nyp2 + iy

                # Split quad into 2 triangles: (0,1,2) and (0,2,3)
                triangles.append([v[0], v[1], v[2]])
                cell_indices.append(flat_idx)
                triangles.append([v[0], v[2], v[3]])
                cell_indices.append(flat_idx)

        return (np.array(vertices_r), np.array(vertices_z),
                np.array(triangles, dtype=np.int32),
                np.array(cell_indices, dtype=np.int32))

    # ------------------------------------------------------------------
    # Plasma state
    # ------------------------------------------------------------------

    def load_plasma_state(self, path=None):
        """Load SOLPS plasma state from b2fstate (text) or b2fstate.mat.

        Prefers text format b2fstate (written by b2run) over .mat format,
        since .mat files require separate post-processing to generate.
        """
        # Try text-format b2fstate first (written directly by b2run)
        txt_path = path or os.path.join(self.run_dir, 'b2fstate')
        if os.path.exists(txt_path) and os.path.getsize(txt_path) > 10000:
            return self._load_plasma_state_text(txt_path)

        # Fall back to .mat format
        if not HAS_SCIPY:
            raise ImportError("scipy required for .mat file reading")
        mat_path = path or os.path.join(self.run_dir, 'b2fstate.mat')
        m = sio.loadmat(mat_path)
        self.state = m['State'][0, 0]
        self.ns = int(self.state['ns'].flat[0])
        if self.nx == 0:
            self.nx = int(self.state['nx'].flat[0])
            self.ny = int(self.state['ny'].flat[0])
        return self.state

    def _load_plasma_state_text(self, path):
        """Parse text-format b2fstate/b2fstati written by b2run.

        Format: *cf: header lines followed by data values.
          *cf:    type    count    name
        """
        fields = {}
        with open(path, 'r') as f:
            lines = f.readlines()

        i = 0
        nx = ny = ns = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('*cf:'):
                parts = line.split()
                # *cf:  type  count  name
                dtype = parts[1]
                count = int(parts[2])
                name = parts[3] if len(parts) > 3 else ''

                i += 1
                # Read data values
                vals = []
                while i < len(lines) and not lines[i].strip().startswith('*cf:') \
                        and not lines[i].strip().startswith('VERSION'):
                    vals.extend(lines[i].split())
                    i += 1
                    if len(vals) >= count:
                        break

                if dtype == 'int':
                    arr = np.array([int(v) for v in vals[:count]])
                    if name == 'nx,ny,ns' and len(arr) == 3:
                        nx, ny, ns = int(arr[0]), int(arr[1]), int(arr[2])
                        fields['nx'] = np.array([[nx]])
                        fields['ny'] = np.array([[ny]])
                        fields['ns'] = np.array([[ns]])
                elif dtype == 'real':
                    arr = np.array([float(v) for v in vals[:count]])
                    fields[name] = arr
                elif dtype == 'char':
                    fields[name] = ' '.join(vals[:count])
            else:
                i += 1

        if nx == 0:
            raise ValueError(f"Could not parse nx,ny,ns from {path}")

        nxp2, nyp2 = nx + 2, ny + 2

        # Reshape key fields to expected dimensions
        shape_map = {
            'ne': (nxp2, nyp2),
            'te': (nxp2, nyp2),
            'ti': (nxp2, nyp2),
            'po': (nxp2, nyp2),
            'fch_p': (nxp2, nyp2),
            'na': (nxp2, nyp2, ns),
            'ua': (nxp2, nyp2, ns),
            'kinrgy': (nxp2, nyp2, ns),
            'zamin': (ns,),
            'zamax': (ns,),
            'zn': (ns,),
            'am': (ns,),
            'fna': (nxp2, nyp2, 2, ns),
            'fhe': (nxp2, nyp2, 2),
            'fhi': (nxp2, nyp2, 2),
            'fch': (nxp2, nyp2, 2),
            'uadia': (nxp2, nyp2, 2, ns),
        }

        state = {}
        for name, arr in fields.items():
            if isinstance(arr, np.ndarray) and name in shape_map:
                expected = shape_map[name]
                expected_size = int(np.prod(expected))
                if len(arr) == expected_size:
                    state[name] = arr.reshape(expected)
                else:
                    state[name] = arr
            else:
                state[name] = arr

        # Store as dict-like object compatible with get_plasma_fields
        self.state = state
        self.ns = ns
        if self.nx == 0:
            self.nx = nx
            self.ny = ny
        return self.state

    def get_plasma_fields(self):
        """Return dict of plasma fields: ne, te, ti, each (nx+2, ny+2)."""
        if self.state is None:
            self.load_plasma_state()
        return {
            'ne': np.array(self.state['ne']),
            'te': np.array(self.state['te']),
            'ti': np.array(self.state['ti']),
            'na': np.array(self.state['na']),  # (nx+2, ny+2, ns)
            'ua': np.array(self.state['ua']),  # (nx+2, ny+2, ns)
            'po': np.array(self.state['po']),
        }

    # ------------------------------------------------------------------
    # Write source2d files (OpenEdge -> SOLPS)
    # ------------------------------------------------------------------

    def write_source2d(self, source_array, filename):
        """Write a source2d file in SOLPS format.

        Args:
            source_array: shape (nx+2, ny+2, ns) particle source [atoms/s]
                          or (nx+2, ny+2) for single-species
            filename: output file name (e.g., 'source2d.00001')

        File format: ns*(ny+2) rows, each row has nx+2 values.
        Loop order: species j=0..ns-1, poloidal i=-1..ny (i.e., i=0..ny+1 in 0-based).
        Each row: radial values profile3d(-1:nx, i, j).
        """
        nxp2 = self.nx + 2
        nyp2 = self.ny + 2

        if source_array.ndim == 2:
            # Single species: expand to (nx+2, ny+2, ns) with zeros
            full = np.zeros((nxp2, nyp2, self.ns))
            # Find Li species index (zn=3, am=7)
            li_idx = self._find_li_species_idx()
            if li_idx >= 0:
                full[:, :, li_idx] = source_array
            source_array = full

        filepath = os.path.join(self.run_dir, filename)
        with open(filepath, 'w') as f:
            for j in range(self.ns):
                for i in range(nyp2):
                    row = source_array[:, i, j]
                    line = ' '.join(f'{v:.8e}' for v in row)
                    f.write(line + '\n')

    def write_sources_profile_chain(self, n_windows, dt_windows, t_start=0.0,
                                    source_filenames=None):
        """Write b2.sources.profile and its chain files.

        Args:
            n_windows: number of time windows (source2d files)
            dt_windows: list of dt for each window [s]
            t_start: simulation start time [s]
            source_filenames: optional list of explicit source2d file names
                              to reference, one per time window
        """
        if source_filenames is not None:
            if len(source_filenames) != n_windows:
                raise ValueError(
                    "source_filenames must have one entry per time window")
        else:
            source_filenames = [
                f'source2d.{k:05d}' for k in range(1, n_windows + 1)
            ]

        t = t_start
        for k in range(1, n_windows + 1):
            fname = f'b2.sources.profile' if k == 1 else f'b2.sources.profile.{k}'
            filepath = os.path.join(self.run_dir, fname)

            switch_time = t + dt_windows[k - 1]
            next_profile = f'b2.sources.profile.{k + 1}' if k < n_windows else ''

            with open(filepath, 'w') as f:
                f.write('&profile\n')
                f.write('read_sna0_2d=.true.\n')
                f.write(f'sna0_2d_filename="{source_filenames[k - 1]}"\n')
                if next_profile:
                    f.write(f'sources_time_switch={switch_time:.6e}\n')
                    f.write(f'sources_filename="{next_profile}"\n')
                else:
                    # No next window: set switch time far in the future so
                    # SOLPS uses this source for the entire run without error
                    f.write(f'sources_time_switch={1.0e+20:.6e}\n')
                f.write('/\n')

            t = switch_time

    def _find_li_species_idx(self):
        """Find Li+ species index (zn=3, zamin=1)."""
        if self.state is None:
            return -1
        zn = self.state['zn'].flatten()
        zamin = self.state['zamin'].flatten()
        for i in range(len(zn)):
            if int(zn[i]) == 3 and int(zamin[i]) == 1:
                return i
        return -1

    # ------------------------------------------------------------------
    # Convert SOLPS plasma to OpenEdge HDF5 (SOLPS -> OpenEdge)
    # ------------------------------------------------------------------

    def write_plasma_h5(self, output_path, r_grid=None, z_grid=None,
                        nr=200, nz=400):
        """Convert SOLPS plasma state to OpenEdge plasma.h5 format.

        OpenEdge expects a regular (R, Z) grid with fields:
            /r [nr], /z [nz], /dens_e [nz, nr], /temp_e [nz, nr], etc.

        This method interpolates from the SOLPS structured grid to a
        regular grid using nearest-cell mapping.

        Args:
            output_path: path for output HDF5 file
            r_grid: 1D array of R values, or None to auto-generate
            z_grid: 1D array of Z values, or None to auto-generate
            nr, nz: grid resolution if auto-generating
        """
        if self.geo is None:
            self.load_geometry()
        if self.state is None:
            self.load_plasma_state()

        R_cen, Z_cen = self.get_cell_centers()
        fields = self.get_plasma_fields()

        # Build regular grid
        if r_grid is None:
            r_min, r_max = np.nanmin(R_cen[R_cen > 0]), np.nanmax(R_cen[R_cen > 0])
            r_grid = np.linspace(r_min - 0.01, r_max + 0.01, nr)
        if z_grid is None:
            z_min, z_max = np.nanmin(Z_cen), np.nanmax(Z_cen)
            z_grid = np.linspace(z_min - 0.01, z_max + 0.01, nz)

        # Build KD-tree from valid SOLPS cell centers for nearest-neighbor mapping
        valid = (R_cen > 0) & np.isfinite(R_cen) & np.isfinite(Z_cen)
        pts = np.column_stack([R_cen[valid].ravel(), Z_cen[valid].ravel()])

        from scipy.spatial import cKDTree
        tree = cKDTree(pts)

        # Query all regular grid points
        RR, ZZ = np.meshgrid(r_grid, z_grid)  # (nz, nr)
        query_pts = np.column_stack([RR.ravel(), ZZ.ravel()])
        _, idx = tree.query(query_pts)

        # Map fields
        def interp_field(field_2d):
            flat = field_2d[valid].ravel()
            result = flat[idx].reshape(nz, nr)
            return result

        ne_reg = interp_field(fields['ne'])  # m^-3 (same units)

        # SOLPS stores Te, Ti in Joules; OpenEdge expects eV
        eV = 1.602176634e-19
        te_reg = interp_field(fields['te']) / eV  # J -> eV
        ti_reg = interp_field(fields['ti']) / eV  # J -> eV

        # Parallel flow: use main ion species (index 1 typically = D+)
        ua_main = fields['ua'][:, :, 1]  # D+ parallel velocity
        upar_reg = interp_field(ua_main)

        # Compute gradients via finite differences on regular grid
        # Te, Ti are already in eV so gradients are in eV/m
        dR = r_grid[1] - r_grid[0]
        dZ = z_grid[1] - z_grid[0]
        grad_te_r = np.gradient(te_reg, dR, axis=1)  # eV/m
        grad_te_z = np.gradient(te_reg, dZ, axis=0)  # eV/m
        grad_ti_r = np.gradient(ti_reg, dR, axis=1)  # eV/m
        grad_ti_z = np.gradient(ti_reg, dZ, axis=0)  # eV/m

        # Ion density: use D+ (species 1)
        na_main = fields['na'][:, :, 1]
        ni_reg = interp_field(na_main)

        # Compute heat flux from SOLPS face fluxes if available
        qmag_reg = None
        try:
            from pathlib import Path
            sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                            '..', 'converters'))
            from convert_solps_heatflux import compute_heatflux_cell_center
            from convert_solps_plasma import _interp_field
            rc_q, zc_q, qmag_q = compute_heatflux_cell_center(
                Path(self.run_dir), method="jeremy_total")
            mask_q = np.isfinite(qmag_q.ravel())
            if mask_q.sum() > 10:
                src_rz = np.column_stack([rc_q.ravel()[mask_q],
                                          zc_q.ravel()[mask_q]])
                RR, ZZ = np.meshgrid(r_grid, z_grid)
                tgt_rz = np.column_stack([RR.ravel(), ZZ.ravel()])
                qmag_reg = _interp_field(src_rz, qmag_q.ravel()[mask_q],
                                         tgt_rz, len(z_grid), len(r_grid))
                qmag_reg = np.nan_to_num(qmag_reg, nan=0.0)
                print(f'  q_mag: [{qmag_reg.min():.2e}, {qmag_reg.max():.2e}] W/m^2')
        except Exception as e:
            print(f'  WARNING: could not compute q_mag: {e}')

        # Write HDF5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('r', data=r_grid)
            f.create_dataset('z', data=z_grid)
            f.create_dataset('dens_e', data=ne_reg)
            f.create_dataset('temp_e', data=te_reg)
            f.create_dataset('dens_i', data=ni_reg)
            f.create_dataset('temp_i', data=ti_reg)
            f.create_dataset('parr_flow', data=upar_reg)
            # Zero toroidal for 2D
            f.create_dataset('parr_flow_r', data=np.zeros_like(upar_reg))
            f.create_dataset('parr_flow_t', data=np.zeros_like(upar_reg))
            f.create_dataset('parr_flow_z', data=upar_reg)
            f.create_dataset('grad_te_r', data=grad_te_r)
            f.create_dataset('grad_te_t', data=np.zeros_like(grad_te_r))
            f.create_dataset('grad_te_z', data=grad_te_z)
            f.create_dataset('grad_ti_r', data=grad_ti_r)
            f.create_dataset('grad_ti_t', data=np.zeros_like(grad_ti_r))
            f.create_dataset('grad_ti_z', data=grad_ti_z)
            if qmag_reg is not None:
                f.create_dataset('q_mag', data=qmag_reg)

        print(f'Wrote {output_path}: r[{nr}] x z[{nz}], '
              f'ne=[{ne_reg.min():.2e}, {ne_reg.max():.2e}], '
              f'te=[{te_reg.min():.2e}, {te_reg.max():.2e}]')
        return output_path

    # ------------------------------------------------------------------
    # Build SOLPS mesh HDF5 for OpenEdge (triangulation + per-cell fields)
    # ------------------------------------------------------------------

    def write_mesh_h5(self, output_path):
        """Write SOLPS mesh triangulation + per-cell plasma data for OpenEdge.

        This is the format used by compute_pmi_surf_data.cpp for
        direct cell lookup (more accurate than regular grid interpolation).
        """
        vtx_r, vtx_z, triangles, cell_idx = self.build_triangulation()
        fields = self.get_plasma_fields()

        # Flatten fields to match flat cell indices
        ne_flat = fields['ne'].ravel()
        te_flat = fields['te'].ravel()
        ti_flat = fields['ti'].ravel()

        ncell = fields['ne'].size

        with h5py.File(output_path, 'w') as f:
            grp = f.create_group('mesh')
            grp.create_dataset('vtx_r', data=vtx_r)
            grp.create_dataset('vtx_z', data=vtx_z)
            grp.create_dataset('triangles', data=triangles)
            grp.create_dataset('cell_index', data=cell_idx)
            grp.create_dataset('dens_e', data=ne_flat[:ncell])
            grp.create_dataset('temp_e', data=te_flat[:ncell])
            grp.create_dataset('temp_i', data=ti_flat[:ncell])

            # Ion density and parallel flow (D+ = species 1)
            grp.create_dataset('dens_i', data=fields['na'][:, :, 1].ravel()[:ncell])
            grp.create_dataset('parr_flow', data=fields['ua'][:, :, 1].ravel()[:ncell])

        print(f'Wrote {output_path}: {len(vtx_r)} vertices, '
              f'{len(triangles)} triangles, {ncell} cells')
        return output_path


# ------------------------------------------------------------------
# Quick test
# ------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    run_dir = sys.argv[1] if len(sys.argv) > 1 else '/home/cloud/li/v0_1'

    si = SolpsInterface(run_dir)
    si.load_geometry()
    print(f'Grid: nx={si.nx}, ny={si.ny}')

    R, Z = si.get_cell_centers()
    print(f'Cell centers: R[{R.shape}], Z[{Z.shape}]')

    si.load_plasma_state()
    print(f'Species: ns={si.ns}')
    fields = si.get_plasma_fields()
    print(f'ne range: [{fields["ne"].min():.2e}, {fields["ne"].max():.2e}]')
    print(f'te range: [{fields["te"].min():.2e}, {fields["te"].max():.2e}]')

    # Test write_plasma_h5
    si.write_plasma_h5('/tmp/test_plasma_from_solps.h5')

    # Test write_source2d (all zeros)
    source = np.zeros((si.nx + 2, si.ny + 2))
    si.write_source2d(source, 'source2d.test')
    print(f'Wrote {run_dir}/source2d.test')

    # Test triangulation
    vtx_r, vtx_z, tri, cidx = si.build_triangulation()
    print(f'Triangulation: {len(vtx_r)} vertices, {len(tri)} triangles')
