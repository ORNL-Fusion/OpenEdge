#!/usr/bin/env python3
"""
OpenEdge-SOLPS coupling driver (v1: subprocess + file exchange).

Orchestrates alternating execution of OpenEdge (droplet transport) and
SOLPS-ITER (plasma transport) with data exchange between coupling steps.

Coupling loop:
  1. Run OpenEdge for N_oe steps -> dump mass_loss grid
  2. Read mass_loss, map to SOLPS cells, write source2d
  3. Run SOLPS for N_solps steps
  4. Read updated SOLPS plasma state
  5. Write updated plasma.h5 for OpenEdge
  6. Repeat

Usage:
  python3 openedge_solps_driver.py config.yaml

Config file specifies paths, grid parameters, coupling cadence, etc.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

# Add tools/coupling to path for solps_interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solps_interface import SolpsInterface


# ======================================================================
# OpenEdge mass_loss reader
# ======================================================================

def read_openedge_mass_loss(filepath):
    """Read OpenEdge grid dump (mass_loss.txt format).

    Returns:
        xc, yc: cell center coordinates (R, Z in 2D cylindrical)
        dm_kg: mass lost per cell [kg]
        dn_atoms: atoms lost per cell
    """
    xc, yc, dm, dn = [], [], [], []
    in_data = False
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('ITEM: CELLS'):
                in_data = True
                continue
            if line.startswith('ITEM:'):
                in_data = False
                continue
            if in_data:
                parts = line.split()
                if len(parts) >= 5:
                    xc.append(float(parts[1]))
                    yc.append(float(parts[2]))
                    dm.append(float(parts[3]))
                    dn.append(float(parts[4]))
    return np.array(xc), np.array(yc), np.array(dm), np.array(dn)


def read_last_mass_loss_frame(filepath):
    """Read the LAST timestep frame from a multi-frame mass_loss file."""
    frames = []
    current_lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('ITEM: TIMESTEP') and current_lines:
                frames.append(current_lines)
                current_lines = []
            current_lines.append(line)
    if current_lines:
        frames.append(current_lines)

    if not frames:
        return None, None, None, None

    # Parse last frame
    xc, yc, dm, dn = [], [], [], []
    in_data = False
    for line in frames[-1]:
        line = line.strip()
        if line.startswith('ITEM: CELLS'):
            in_data = True
            continue
        if line.startswith('ITEM:'):
            in_data = False
            continue
        if in_data:
            parts = line.split()
            if len(parts) >= 5:
                xc.append(float(parts[1]))
                yc.append(float(parts[2]))
                dm.append(float(parts[3]))
                dn.append(float(parts[4]))
    return np.array(xc), np.array(yc), np.array(dm), np.array(dn)


# ======================================================================
# Map OpenEdge grid -> SOLPS cells
# ======================================================================

def map_source_to_solps(xc, yc, dn_atoms, solps, dt_oe):
    """Map OpenEdge per-cell atom source to SOLPS grid.

    Args:
        xc, yc: OpenEdge cell center coords (R, Z)
        dn_atoms: atoms lost per cell over the coupling window
        solps: SolpsInterface with loaded geometry
        dt_oe: OpenEdge time window [s]

    Returns:
        source2d: (nx+2, ny+2) array of particle source rate [atoms/s]
    """
    from scipy.spatial import cKDTree

    nxp2 = solps.nx + 2
    nyp2 = solps.ny + 2

    R_cen, Z_cen = solps.get_cell_centers()

    # Build KD-tree from valid SOLPS cell centers
    valid = (R_cen > 0) & np.isfinite(R_cen) & np.isfinite(Z_cen)
    pts_solps = np.column_stack([R_cen[valid].ravel(), Z_cen[valid].ravel()])
    # Map flat index back to (ix, iy)
    valid_flat = np.where(valid.ravel())[0]

    tree = cKDTree(pts_solps)

    # For each OpenEdge cell with nonzero source, find nearest SOLPS cell
    source_flat = np.zeros(nxp2 * nyp2)
    mask = np.abs(dn_atoms) > 0
    if mask.sum() == 0:
        return source_flat.reshape(nxp2, nyp2)

    oe_pts = np.column_stack([xc[mask], yc[mask]])
    dist, idx = tree.query(oe_pts)

    # Accumulate source into SOLPS cells
    for i, (d, j) in enumerate(zip(dist, idx)):
        if d < 0.5:  # max distance threshold [m]
            solps_flat_idx = valid_flat[j]
            source_flat[solps_flat_idx] += dn_atoms[mask][i]

    # Convert from total atoms to rate [atoms/s]
    if dt_oe > 0:
        source_flat /= dt_oe

    return source_flat.reshape(nxp2, nyp2)


# ======================================================================
# Generate OpenEdge input script for a coupling chunk
# ======================================================================

def write_openedge_continue_script(output_script, template_script, n_steps,
                                   plasma_h5_path):
    """Write an OpenEdge input script that continues from a restart file.

    Reads the template to extract fix/compute/dump definitions that need
    to be re-registered after read_restart (which only restores grid,
    particles, surfaces, species, mixtures — not fixes/computes/dumps).
    """
    # Parse template for lines that define fixes, computes, dumps, globals
    redefine_lines = []
    with open(template_script, 'r') as f:
        lines = f.readlines()

    # Collect multi-line commands (lines ending with &)
    full_lines = []
    buf = ''
    for line in lines:
        stripped = line.rstrip()
        if stripped.endswith('&'):
            buf += stripped[:-1] + ' '
        else:
            buf += stripped
            full_lines.append(buf)
            buf = ''

    for line in full_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        # Keep: compute, fix, dump, global, variable, surf_collide, surf_react, surf_modify, stats
        first_word = stripped.split()[0] if stripped.split() else ''
        # Skip gridcut/comm/sort — hardcoded before balance_grid above
        if first_word == 'global' and any(kw in stripped for kw in
                ['gridcut', 'comm/sort']):
            continue
        if first_word in ('timestep', 'compute', 'fix', 'dump',
                          'global', 'variable', 'surf_collide', 'surf_react',
                          'surf_modify', 'stats', 'mixture', 'group'):
            redefine_lines.append(stripped)

    # Sort: seed first, variables, then global setup (gridcut/fnum/temp),
    # then surface/compute/fix, then global that references fixes (efield/boris),
    # then dump/stats
    def sort_key(line):
        word = line.split()[0]
        words = line.split()
        if word == 'variable': return (0, 0)
        if word == 'timestep': return (2, 0)
        # global commands that set up grid/physics (must come before compute/fix)
        if word == 'global' and len(words) > 1 and words[1] in (
                'gridcut', 'fnum', 'nrho', 'temp', 'comm/sort'):
            return (3, 0)
        if word == 'mixture': return (4, 0)
        if word == 'group': return (4, 0)
        if word == 'surf_collide': return (5, 0)
        if word == 'surf_react': return (5, 1)
        if word == 'surf_modify': return (5, 2)
        if word == 'compute': return (6, 0)
        if word == 'fix': return (7, 0)
        # global commands that reference fixes (efield, boris_subcycles)
        if word == 'global': return (8, 0)
        if word == 'dump': return (9, 0)
        if word == 'stats': return (10, 0)
        return (11, 0)
    redefine_lines.sort(key=sort_key)

    with open(output_script, 'w') as f:
        f.write('# Auto-generated continue script (from restart)\n')
        f.write('read_restart restart.dat\n')
        f.write('seed 12345\n')
        f.write('global gridcut 0.0 comm/sort no\n')
        f.write('balance_grid rcb cell\n\n')
        f.write('# Re-register fixes, computes, dumps (not saved in restart)\n')
        for line in redefine_lines:
            f.write(line + '\n')
        f.write(f'\nrun {n_steps}\n')
        f.write('write_restart restart.dat\n')


def write_openedge_chunk_script(template_script, output_script, n_steps,
                                plasma_h5, mass_loss_file, append=False):
    """Write an OpenEdge input script for one coupling chunk.

    Reads the template, replaces run command and plasma file path.
    """
    with open(template_script, 'r') as f:
        lines = f.readlines()

    with open(output_script, 'w') as f:
        for line in lines:
            stripped = line.strip()

            # Skip original run commands
            if stripped.startswith('run ') and not stripped.startswith('run 0'):
                continue

            # Update plasma file path
            if 'plasma/fields' in stripped and 'file' in stripped:
                # Replace the plasma.h5 path
                line = line.replace('../test_evaporation/input/plasma.h5',
                                   plasma_h5)

            # Update mass_loss dump path
            if 'mass_loss' in stripped and 'dump' in stripped:
                line = line.replace('output/mass_loss.txt', mass_loss_file)

            f.write(line)

        # Add the run command at the end
        f.write(f'\nrun {n_steps}\n')


# ======================================================================
# Main coupling driver
# ======================================================================

def run_coupling(config):
    """Main coupling loop."""

    # Unpack config
    solps_run_dir    = config['solps_run_dir']
    solps_base_dir   = config.get('solps_base_dir', os.path.join(
        os.path.dirname(solps_run_dir), 'baserun'))
    oe_run_dir       = config['openedge_run_dir']
    oe_template      = config['openedge_template_script']
    oe_binary        = config['openedge_binary']
    n_coupling_steps = config.get('n_coupling_steps', 5)
    n_oe_steps       = config.get('n_oe_steps', 100)
    n_solps_steps    = config.get('n_solps_steps', 2)
    dt_oe            = config.get('dt_oe', 1e-5)
    alpha            = config.get('relaxation_alpha', 0.5)
    mpi_np_oe        = config.get('mpi_np_openedge', 1)
    mpi_np_solps     = config.get('mpi_np_solps', 1)
    plasma_h5_path   = config.get('plasma_h5', os.path.join(oe_run_dir, 'plasma.h5'))
    bfield_h5_path   = config.get('bfield_h5', '')
    solps_run_script = config.get('solps_run_script', '')
    coupled_dir      = config.get('coupled_dir', '')

    # Initialize SOLPS interface
    solps = SolpsInterface(solps_run_dir, solps_base_dir)
    solps.load_geometry()
    solps.load_plasma_state()
    print(f'SOLPS grid: nx={solps.nx}, ny={solps.ny}, ns={solps.ns}')

    # Write initial plasma.h5 from SOLPS state
    solps.write_plasma_h5(plasma_h5_path)

    nxp2, nyp2 = solps.nx + 2, solps.ny + 2
    source_prev = np.zeros((nxp2, nyp2))
    dt_window = n_oe_steps * dt_oe

    print(f'\n{"="*60}')
    print(f'Starting coupling: {n_coupling_steps} steps')
    print(f'  OpenEdge: {n_oe_steps} steps/chunk, dt={dt_oe:.1e}s')
    print(f'  SOLPS: {n_solps_steps} steps/chunk')
    print(f'  Relaxation alpha={alpha}')
    print(f'{"="*60}\n')

    for k in range(n_coupling_steps):
        t0 = time.time()
        print(f'--- Coupling step {k+1}/{n_coupling_steps} ---')

        # ---- 1. Run OpenEdge ----
        mass_loss_file = os.path.join(oe_run_dir, 'output', 'mass_loss.txt')
        os.makedirs(os.path.join(oe_run_dir, 'output'), exist_ok=True)
        # Remove old mass_loss so we only read this chunk's output
        if os.path.exists(mass_loss_file):
            os.remove(mass_loss_file)

        restart_file = os.path.join(oe_run_dir, 'restart.dat')
        chunk_script = os.path.join(oe_run_dir, f'in.chunk_{k:04d}')

        if k == 0:
            # First chunk: use full template (sets up geometry, species, etc.)
            with open(oe_template, 'r') as fin:
                template_content = fin.read()
            with open(chunk_script, 'w') as fout:
                fout.write(template_content)
                fout.write(f'\nrun {n_oe_steps}\n')
                fout.write(f'write_restart restart.dat\n')
        else:
            # Subsequent chunks: read restart, redefine fixes/computes/dumps
            write_openedge_continue_script(
                chunk_script, oe_template, n_oe_steps, plasma_h5_path)

        print(f'  Running OpenEdge ({n_oe_steps} steps, '
              f'{"init" if k == 0 else "restart"})...')
        oe_cmd = f'mpirun -np {mpi_np_oe} {oe_binary} -in {os.path.basename(chunk_script)}'
        result = subprocess.run(oe_cmd, shell=True, cwd=oe_run_dir,
                                capture_output=True, text=True, env={
                                    **os.environ,
                                    'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu/hdf5/serial'
                                })
        if result.returncode != 0:
            print(f'  ERROR: OpenEdge failed:\n{result.stderr[-500:]}')
            # Save stderr for debugging
            with open(os.path.join(oe_run_dir, f'error_{k:04d}.log'), 'w') as ef:
                ef.write(result.stdout + '\n' + result.stderr)
            break
        print(f'  OpenEdge done.')

        # ---- 2. Read mass loss and map to SOLPS ----
        if not os.path.exists(mass_loss_file):
            print(f'  WARNING: No mass_loss file, skipping source update')
            source_new = np.zeros((nxp2, nyp2))
        else:
            xc, yc, dm, dn = read_last_mass_loss_frame(mass_loss_file)
            if xc is not None and len(xc) > 0:
                source_new = map_source_to_solps(xc, yc, dn, solps, dt_window)
                total_source = np.sum(source_new)
                print(f'  Li source: {total_source:.3e} atoms/s '
                      f'({np.count_nonzero(source_new)} active cells)')
            else:
                source_new = np.zeros((nxp2, nyp2))

        # ---- 3. Apply under-relaxation ----
        if k == 0:
            source_used = source_new
        else:
            source_used = alpha * source_new + (1 - alpha) * source_prev
        source_prev = source_used.copy()

        # ---- 4. Write source2d and run SOLPS ----
        source_file = f'source2d.{k+1:05d}'
        solps.write_source2d(source_used, source_file)

        # Write single-window sources profile
        solps.write_sources_profile_chain(
            n_windows=1,
            dt_windows=[dt_window],
            t_start=k * dt_window)

        # Update b2mn.dat for N_solps steps
        update_b2mn_ntim(solps_run_dir, n_solps_steps)

        # Clean SOLPS output so b2run doesn't skip ("up to date")
        for f in ['b2mn.prt', 'b2fstate', 'b2fplasma', 'b2fparam']:
            p = os.path.join(solps_run_dir, f)
            if os.path.exists(p):
                os.remove(p)

        print(f'  Running SOLPS ({n_solps_steps} steps)...')
        if solps_run_script:
            solps_cmd = f'{solps_run_script} {mpi_np_solps}'
            cwd = coupled_dir or os.path.dirname(solps_run_dir)
        else:
            solps_cmd = f'b2run b2mn'
            if mpi_np_solps > 1:
                solps_cmd = f'b2run -m "mpirun -np {mpi_np_solps}" b2mn'
            cwd = solps_run_dir
        result = subprocess.run(solps_cmd, shell=True, cwd=cwd,
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f'  WARNING: SOLPS may have issues:\n{result.stderr[-300:]}')
        print(f'  SOLPS done.')

        # ---- 5. Read updated plasma and refresh OpenEdge ----
        # Reload state from .mat (b2fstate.mat is updated after each run)
        try:
            solps.load_plasma_state()
            solps.write_plasma_h5(plasma_h5_path)
            print(f'  Updated {plasma_h5_path}')
        except Exception as e:
            print(f'  WARNING: Could not update plasma: {e}')

        elapsed = time.time() - t0
        print(f'  Step {k+1} completed in {elapsed:.1f}s\n')

    print(f'{"="*60}')
    print(f'Coupling complete: {n_coupling_steps} steps')
    print(f'{"="*60}')


def update_b2mn_ntim(run_dir, ntim):
    """Update b2mndr_ntim in b2mn.dat."""
    dat_path = os.path.join(run_dir, 'b2mn.dat')
    if not os.path.exists(dat_path):
        return

    with open(dat_path, 'r') as f:
        lines = f.readlines()

    with open(dat_path, 'w') as f:
        for line in lines:
            if "'b2mndr_ntim'" in line:
                # Preserve comment
                parts = line.split('#')
                comment = ' # ' + parts[1].strip() if len(parts) > 1 else ''
                f.write(f"'b2mndr_ntim'                      '{ntim}'{comment}\n")
            else:
                f.write(line)


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='OpenEdge-SOLPS coupling driver')
    parser.add_argument('config', help='JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    run_coupling(config)


if __name__ == '__main__':
    main()
