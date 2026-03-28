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
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Add tools/coupling to path for solps_interface
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solps_interface import SolpsInterface

console = Console()
eV = 1.602176634e-19


# ======================================================================
# Error handling
# ======================================================================

class CouplingError(Exception):
    """Base exception for coupling failures."""
    pass

class SOLPSError(CouplingError):
    """SOLPS run failed (XERRAB, missing b2fstate, nonzero returncode)."""
    pass

class OpenEdgeError(CouplingError):
    """OpenEdge run failed (nonzero returncode, missing output)."""
    pass


def check_solps_success(solps_run_dir, label="SOLPS"):
    """Check that SOLPS produced a valid b2fstate. Raise SOLPSError if not."""
    # Check run.log for XERRAB
    solps_log = os.path.join(solps_run_dir, 'run.log')
    if os.path.exists(solps_log):
        with open(solps_log, 'r') as lf:
            log_text = lf.read()
        if 'XERRAB' in log_text or 'xerrab--error' in log_text.lower():
            # Extract error context
            err_lines = [l.strip() for l in log_text.splitlines()
                         if 'XERRAB' in l or 'DUE TO' in l]
            detail = '; '.join(err_lines[:3]) if err_lines else 'XERRAB detected'
            raise SOLPSError(
                f"{label} failed: {detail}\n"
                f"  Log: {solps_log}")

    # Check b2fstate exists and is large enough
    b2fstate = os.path.join(solps_run_dir, 'b2fstate')
    if not os.path.exists(b2fstate):
        raise SOLPSError(
            f"{label} failed: no b2fstate produced\n"
            f"  Run dir: {solps_run_dir}")
    if os.path.getsize(b2fstate) < 10000:
        raise SOLPSError(
            f"{label} failed: b2fstate too small "
            f"({os.path.getsize(b2fstate)} bytes)\n"
            f"  Run dir: {solps_run_dir}")


# ======================================================================
# Per-step output archiving
# ======================================================================

# Essential SOLPS files to preserve per step (skip huge geometry/database)
SOLPS_SAVE_FILES = [
    'b2fstate', 'b2fstati', 'b2fplasma', 'b2fparam', 'b2mn.prt',
    'run.log', 'b2mn.dat',
]

def save_warmup_output(solps_run_dir, coupled_dir, plasma_h5_path):
    """Archive warmup output to warmup/ directory."""
    warmup_dir = os.path.join(coupled_dir, 'warmup')
    os.makedirs(warmup_dir, exist_ok=True)

    # Save key SOLPS files
    for fname in SOLPS_SAVE_FILES:
        src = os.path.join(solps_run_dir, fname)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, os.path.join(warmup_dir, fname))

    # Save source2d used during warmup
    for fname in os.listdir(solps_run_dir):
        if fname.startswith('source2d.'):
            shutil.copy2(os.path.join(solps_run_dir, fname),
                         os.path.join(warmup_dir, fname))

    # Save plasma.h5
    if os.path.exists(plasma_h5_path):
        shutil.copy2(plasma_h5_path, os.path.join(warmup_dir, 'plasma.h5'))

    console.print(f"  [dim]Warmup output saved to {warmup_dir}[/dim]")
    return warmup_dir


def restore_warmup(warmup_dir, solps_run_dir, plasma_h5_path):
    """Restore warmup state from a previous run. Raises SOLPSError if
    the archived warmup state is missing or invalid."""
    warmup_state = os.path.join(warmup_dir, 'b2fstate')
    if not os.path.exists(warmup_state) \
            or os.path.getsize(warmup_state) < 10000:
        raise SOLPSError(
            f"Cannot restore warmup: b2fstate missing or invalid in "
            f"{warmup_dir} ({os.path.getsize(warmup_state) if os.path.exists(warmup_state) else 0} bytes).\n"
            f"  Remove the warmup/ folder and re-run, or set skip_warmup: false.")

    shutil.copy2(warmup_state, os.path.join(solps_run_dir, 'b2fstati'))

    warmup_plasma = os.path.join(warmup_dir, 'plasma.h5')
    if os.path.exists(warmup_plasma):
        shutil.copy2(warmup_plasma, plasma_h5_path)

    console.print(f"  [green]Restored warmup state from {warmup_dir}[/green]")


def save_step_output(step_k, coupled_dir, solps_run_dir, oe_run_dir,
                     plasma_h5_path, source_used, solps):
    """Archive coupling step output to step_NNN/ directory."""
    step_dir = os.path.join(coupled_dir, f'step_{step_k+1:03d}')
    solps_out = os.path.join(step_dir, 'solps')
    oe_out = os.path.join(step_dir, 'openedge')
    os.makedirs(solps_out, exist_ok=True)
    os.makedirs(oe_out, exist_ok=True)

    # Save key SOLPS files
    for fname in SOLPS_SAVE_FILES:
        src = os.path.join(solps_run_dir, fname)
        if os.path.exists(src) and os.path.getsize(src) > 0:
            shutil.copy2(src, os.path.join(solps_out, fname))

    # Save source2d used this step
    for fname in os.listdir(solps_run_dir):
        if fname.startswith('source2d.'):
            shutil.copy2(os.path.join(solps_run_dir, fname),
                         os.path.join(solps_out, fname))

    # Save OpenEdge files
    for fname in ['restart.dat', f'in.chunk_{step_k:04d}',
                  'log.openedge', f'error_{step_k:04d}.log']:
        src = os.path.join(oe_run_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(oe_out, fname))

    # Save mass_loss
    ml_src = os.path.join(oe_run_dir, 'output', 'mass_loss.txt')
    if os.path.exists(ml_src):
        shutil.copy2(ml_src, os.path.join(oe_out, 'mass_loss.txt'))

    # Save plasma.h5 for this step
    if os.path.exists(plasma_h5_path):
        shutil.copy2(plasma_h5_path, os.path.join(step_dir, 'plasma.h5'))

    # Save source array as numpy for easy reloading
    np.save(os.path.join(step_dir, 'source_used.npy'), source_used)

    console.print(f"  [dim]Step {step_k+1} output saved to {step_dir}[/dim]")


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


def read_particle_mass_loss(filepath, rho=534.0):
    """Compute per-particle mass loss from a SPARTA particle dump.

    Reads the first and last frames, computes mass from radius
    (mass = rho * 4/3 * pi * r^3), then dm = mass_first - mass_last.
    Uses the particle's average position across frames for spatial mapping.

    Returns (R, Z, dm_kg, dn_atoms) arrays, or (None,)*4 if empty.
    """
    AM = 1.53e-26  # Li atom mass [kg]

    # Parse all frames
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

    if len(frames) < 2:
        return None, None, None, None

    def parse_frame(frame_lines):
        """Parse a single frame into {id: (x, y, radius)} dict."""
        col_map = None
        particles = {}
        in_data = False
        for line in frame_lines:
            stripped = line.strip()
            if stripped.startswith('ITEM: ATOMS') \
                    or stripped.startswith('ITEM: PARTICLES'):
                cols = stripped.split()[2:]
                col_map = {c: j for j, c in enumerate(cols)}
                in_data = True
                continue
            if stripped.startswith('ITEM:'):
                in_data = False
                continue
            if in_data and col_map:
                parts = stripped.split()
                if len(parts) >= len(col_map):
                    pid = int(parts[col_map['id']])
                    x = float(parts[col_map['x']])
                    y = float(parts[col_map['y']])
                    r = float(parts[col_map['radius']])
                    particles[pid] = (x, y, r)
        return particles

    def radius_to_mass(r):
        return rho * (4.0 / 3.0) * np.pi * r**3 if r > 0 else 0.0

    # Find first frame where particles have nonzero radius (step 0 may
    # have radius=0 because one-time seeding hasn't run yet).
    first = None
    for frame in frames:
        parsed = parse_frame(frame)
        if parsed and any(r > 0 for _, _, r in parsed.values()):
            first = parsed
            break
    last = parse_frame(frames[-1])

    if not first:
        return None, None, None, None

    R_list, Z_list, dm_list = [], [], []
    for pid, (x0, y0, r0) in first.items():
        m0 = radius_to_mass(r0)
        if m0 <= 0:
            continue
        if pid in last:
            x1, y1, r1 = last[pid]
            m1 = radius_to_mass(r1)
            dm = m0 - m1
            R_list.append(0.5 * (x0 + x1))
            Z_list.append(0.5 * (y0 + y1))
        else:
            dm = m0
            R_list.append(x0)
            Z_list.append(y0)
        if dm > 0:
            dm_list.append(dm)
        else:
            R_list.pop()
            Z_list.pop()

    if not dm_list:
        return None, None, None, None

    dm_arr = np.array(dm_list)
    dn_arr = dm_arr / AM
    return np.array(R_list), np.array(Z_list), dm_arr, dn_arr


# ======================================================================
# Parse OpenEdge particle count from stdout
# ======================================================================

def parse_openedge_npart(stdout):
    """Parse the total particle count from OpenEdge (SPARTA) stats output.

    Stats format: 'Step CPU Np' header followed by data lines.
    Returns the Np value from the last stats line, or None.
    """
    npart = None
    in_stats = False
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith('Step') and 'Np' in stripped:
            in_stats = True
            continue
        if in_stats and stripped and stripped[0].isdigit():
            parts = stripped.split()
            if len(parts) >= 3:
                try:
                    npart = int(parts[2])
                except ValueError:
                    pass
        elif in_stats and stripped and not stripped[0].isdigit():
            in_stats = False
    return npart


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
        if first_word in ('timestep', 'compute', 'fix', 'dump', 'dump_modify',
                          'global', 'variable', 'surf_collide', 'surf_react',
                          'surf_modify', 'stats', 'mixture', 'group'):
            # Substitute Ndump to match n_steps (strip trailing comment)
            if first_word == 'variable' and 'Ndump' in stripped:
                ndump = max(1, n_steps // 2)
                stripped = re.sub(r'#.*$', '', stripped).strip()
                stripped = re.sub(
                    r'(variable\s+Ndump\s+equal\s+)\S+',
                    rf'\1{ndump}', stripped)
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
        if word == 'dump_modify': return (9, 1)
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
# Rich display helpers
# ======================================================================

def make_header_panel(config, solps):
    """Create the header panel with coupling configuration."""
    n_coupling = config.get('n_coupling_steps', 5)
    n_oe = config.get('n_oe_steps', 100)
    n_solps = config.get('n_solps_steps', 2)
    n_warmup = config.get('n_warmup_solps_steps', 0)
    warmup_ntim = config.get('warmup_ntim', 500)
    dt = config.get('dt_oe', 1e-5)
    alpha = config.get('relaxation_alpha', 0.5)
    np_oe = config.get('mpi_np_openedge', 1)
    np_solps = config.get('mpi_np_solps', 1)

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    table.add_column(style="bold cyan")
    table.add_column()
    table.add_row("SOLPS grid", f"nx={solps.nx}  ny={solps.ny}  ns={solps.ns}")
    if n_warmup > 0:
        table.add_row("Warmup", f"{n_warmup} rounds x {warmup_ntim} SOLPS steps (no Li)")
    table.add_row("Coupling steps", str(n_coupling))
    table.add_row("OpenEdge", f"{n_oe} steps/chunk, dt={dt:.1e}s, {np_oe} MPI ranks")
    table.add_row("SOLPS", f"{n_solps} steps/chunk, {np_solps} MPI ranks")
    table.add_row("Relaxation", f"alpha={alpha}")

    return Panel(table, title="[bold]OpenEdge-SOLPS Coupling[/bold]",
                 border_style="blue", expand=False)


def make_history_table(history):
    """Create the coupling history table from accumulated step data."""
    table = Table(title="Coupling History", box=box.ROUNDED,
                  title_style="bold", border_style="dim")
    table.add_column("Step", style="bold", justify="center", width=6)
    table.add_column("Particles", justify="right", width=11)
    table.add_column("Li Source [atoms/s]", justify="right", width=16)
    table.add_column("ne range [m^-3]", justify="right", width=26)
    table.add_column("Te range [eV]", justify="right", width=26)
    table.add_column("max|dne/ne|", justify="right", width=12)
    table.add_column("max|dTe/Te|", justify="right", width=12)
    table.add_column("Time [s]", justify="right", width=9)

    for h in history:
        # Color the delta columns based on magnitude
        dne_str = f"{h['dne_max']:.2e}" if h['dne_max'] is not None else "--"
        dte_str = f"{h['dte_max']:.2e}" if h['dte_max'] is not None else "--"

        if h['dne_max'] is not None and h['dne_max'] > 0.01:
            dne_str = f"[green]{dne_str}[/green]"
        elif h['dne_max'] is not None and h['dne_max'] > 0.001:
            dne_str = f"[yellow]{dne_str}[/yellow]"

        if h['dte_max'] is not None and h['dte_max'] > 0.01:
            dte_str = f"[green]{dte_str}[/green]"
        elif h['dte_max'] is not None and h['dte_max'] > 0.001:
            dte_str = f"[yellow]{dte_str}[/yellow]"

        # Color particles: red if 0
        npart_str = str(h['npart']) if h['npart'] is not None else "--"
        if h['npart'] == 0:
            npart_str = f"[red]{npart_str}[/red]"

        ne_str = f"[{h['ne_min']:.2e}, {h['ne_max']:.2e}]"
        te_str = f"[{h['te_min']:.2e}, {h['te_max']:.2e}]"
        source_str = f"{h['li_source']:.3e}" if h['li_source'] > 0 else "[dim]0[/dim]"

        table.add_row(
            h['step_label'],
            npart_str,
            source_str,
            ne_str,
            te_str,
            dne_str,
            dte_str,
            f"{h['elapsed']:.1f}",
        )
    return table


def make_status_line(step, total, phase, detail=""):
    """Create a status line for the current operation."""
    dots = "." * ((int(time.time() * 2) % 3) + 1)
    step_str = f"[bold cyan]Step {step}/{total}[/bold cyan]"
    if phase == "openedge":
        return f"  {step_str}  [yellow]Running OpenEdge{dots}[/yellow]  {detail}"
    elif phase == "solps":
        return f"  {step_str}  [magenta]Running SOLPS{dots}[/magenta]  {detail}"
    elif phase == "mapping":
        return f"  {step_str}  [blue]Mapping sources{dots}[/blue]  {detail}"
    elif phase == "done":
        return f"  {step_str}  [green]Done[/green]  {detail}"
    return f"  {step_str}  {phase}  {detail}"


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
    n_warmup_steps   = config.get('n_warmup_solps_steps', 0)
    warmup_ntim      = config.get('warmup_ntim', 500)
    dt_oe            = config.get('dt_oe', 1e-5)
    alpha            = config.get('relaxation_alpha', 0.5)
    mpi_np_oe        = config.get('mpi_np_openedge', 1)
    mpi_np_solps     = config.get('mpi_np_solps', 1)
    mpi_np_warmup    = config.get('mpi_np_warmup', mpi_np_solps)
    plasma_h5_path   = config.get('plasma_h5', os.path.join(oe_run_dir, 'plasma.h5'))
    bfield_h5_path   = config.get('bfield_h5', '')
    solps_run_script = config.get('solps_run_script', '')
    coupled_dir      = config.get('coupled_dir', '')

    # Initialize SOLPS interface
    solps = SolpsInterface(solps_run_dir, solps_base_dir)
    solps.load_geometry()
    solps.load_plasma_state()

    # Write initial plasma.h5 from SOLPS state
    solps.write_plasma_h5(plasma_h5_path)

    nxp2, nyp2 = solps.nx + 2, solps.ny + 2
    source_prev = np.zeros((nxp2, nyp2))
    dt_window = n_oe_steps * dt_oe

    # Store initial plasma state for change tracking
    fields0 = solps.get_plasma_fields()
    ne_prev = np.array(fields0['ne'])
    te_prev = np.array(fields0['te']) / eV  # J -> eV

    # Print header
    console.print()
    console.print(make_header_panel(config, solps))
    console.print()

    # ---- Warmup: converge SOLPS without Li source ----
    skip_warmup = config.get('skip_warmup', False)
    warmup_dir = os.path.join(coupled_dir, 'warmup') if coupled_dir else \
        os.path.join(os.path.dirname(solps_run_dir), 'warmup')

    if n_warmup_steps > 0 and skip_warmup and os.path.exists(warmup_dir):
        # Restore from previous warmup
        console.print(Panel(
            f"[bold]Skipping warmup[/bold] — restoring from {warmup_dir}",
            border_style="green", expand=False))
        console.print()
        restore_warmup(warmup_dir, solps_run_dir, plasma_h5_path)

        # Reload plasma state for change tracking
        solps.load_plasma_state()
        fields0 = solps.get_plasma_fields()
        ne_prev = np.array(fields0['ne'])
        te_prev = np.array(fields0['te']) / eV
        console.print()

    elif n_warmup_steps > 0:
        # Clean stale warmup folder from previous (possibly failed) run
        if os.path.exists(warmup_dir):
            shutil.rmtree(warmup_dir)
            console.print(f"  [dim]Removed stale warmup/ folder[/dim]")

        console.print(Panel(
            f"[bold]SOLPS warmup:[/bold] {n_warmup_steps} rounds x "
            f"{warmup_ntim} timesteps (no Li source)",
            border_style="yellow", expand=False))
        console.print()

        # Ensure valid b2fstati
        b2fstati_path = os.path.join(solps_run_dir, 'b2fstati')
        baserun_stati = os.path.join(solps_base_dir, 'b2fstati')
        if not os.path.exists(b2fstati_path) \
                or os.path.getsize(b2fstati_path) < 10000:
            if os.path.exists(baserun_stati):
                shutil.copy2(baserun_stati, b2fstati_path)

        # Clean old source and profile files from previous runs
        import glob as _glob
        for old_src in _glob.glob(os.path.join(solps_run_dir, 'source2d.*')):
            os.remove(old_src)
        for old_sp in _glob.glob(os.path.join(solps_run_dir, 'b2.sources.profile*')):
            os.remove(old_sp)
        old_tw = os.path.join(solps_run_dir, 'sources_time_windows.txt')
        if os.path.exists(old_tw):
            os.remove(old_tw)

        # Disable external source for warmup (no Li)
        warmup_profile = os.path.join(solps_run_dir, 'b2.sources.profile')
        with open(warmup_profile, 'w') as f:
            f.write('&profile\n')
            f.write('read_sna0_2d=.false.\n')
            f.write('/\n')

        for w in range(n_warmup_steps):
            t0w = time.time()
            update_b2mn_dat(solps_run_dir, {
                "b2mndr_ntim": str(warmup_ntim),
                "b2sral_inputfile": "0",  # no external source during warmup
            })

            # Copy b2fstate -> b2fstati for restart continuity
            b2fstate_path = os.path.join(solps_run_dir, 'b2fstate')
            if w > 0 and os.path.exists(b2fstate_path) \
                    and os.path.getsize(b2fstate_path) > 10000:
                shutil.copy2(b2fstate_path, b2fstati_path)

            # Clean outputs
            for f_clean in ['b2mn.prt', 'b2fstate', 'b2fplasma', 'b2fparam']:
                p = os.path.join(solps_run_dir, f_clean)
                if os.path.exists(p):
                    os.remove(p)

            with console.status(
                    f"[yellow]Warmup {w+1}/{n_warmup_steps}[/yellow]  "
                    f"SOLPS ({warmup_ntim} steps, {mpi_np_warmup} ranks, no Li)...",
                    spinner="dots"):
                if solps_run_script:
                    solps_cmd = f'{solps_run_script} {mpi_np_warmup}'
                    cwd = coupled_dir or os.path.dirname(solps_run_dir)
                else:
                    solps_cmd = f'b2run b2mn'
                    if mpi_np_warmup > 1:
                        solps_cmd = f'b2run -m "mpirun -np {mpi_np_warmup}" b2mn'
                    cwd = solps_run_dir
                result = subprocess.run(solps_cmd, shell=True, cwd=cwd,
                                        capture_output=True, text=True)

            # Fail fast on SOLPS errors
            check_solps_success(solps_run_dir,
                                label=f"Warmup {w+1}/{n_warmup_steps}")

            # Track convergence
            try:
                solps.load_plasma_state()
                fields_w = solps.get_plasma_fields()
                ne_w = np.array(fields_w['ne'])
                te_w = np.array(fields_w['te']) / eV

                ne_mask = ne_prev > 1e10
                te_mask = te_prev > 0.01
                dne_rel = np.zeros_like(ne_w)
                dte_rel = np.zeros_like(te_w)
                dne_rel[ne_mask] = (ne_w[ne_mask] - ne_prev[ne_mask]) / ne_prev[ne_mask]
                dte_rel[te_mask] = (te_w[te_mask] - te_prev[te_mask]) / te_prev[te_mask]
                dne_max = float(np.max(np.abs(dne_rel)))
                dte_max = float(np.max(np.abs(dte_rel)))

                elapsed_w = time.time() - t0w
                console.print(
                    f"  [green]Warmup {w+1}/{n_warmup_steps}[/green]  "
                    f"ne=[{ne_w.min():.2e}, {ne_w.max():.2e}]  "
                    f"Te=[{te_w.min():.2e}, {te_w.max():.2e}] eV  "
                    f"|dne|={dne_max:.2e}  |dTe|={dte_max:.2e}  "
                    f"({elapsed_w:.1f}s)")

                ne_prev = ne_w
                te_prev = te_w
            except Exception as e:
                console.print(f"  [red]Warmup {w+1}: Could not read state: {e}[/red]")

        # Write converged plasma.h5 for OpenEdge
        solps.write_plasma_h5(plasma_h5_path)

        # Archive warmup output
        save_warmup_output(solps_run_dir, coupled_dir or
                           os.path.dirname(solps_run_dir), plasma_h5_path)

        console.print()
        console.print("[bold green]Warmup complete — starting coupled iterations[/bold green]")
        console.print()

    # Coupling history for the summary table
    history = []

    for k in range(n_coupling_steps):
        t0 = time.time()
        step_label = f"{k+1}/{n_coupling_steps}"

        # ---- 1. Run OpenEdge ----
        particle_dump = os.path.join(oe_run_dir, 'output', 'particles.txt')
        os.makedirs(os.path.join(oe_run_dir, 'output'), exist_ok=True)
        # Remove old particle dump so we only read this chunk's output
        if os.path.exists(particle_dump):
            os.remove(particle_dump)

        restart_file = os.path.join(oe_run_dir, 'restart.dat')
        chunk_script = os.path.join(oe_run_dir, f'in.chunk_{k:04d}')

        if k == 0:
            with open(oe_template, 'r') as fin:
                template_content = fin.read()
            # Substitute Ndump: dump at start, middle, and end of chunk.
            # Step 0 may have radius=0 (before seeding), so we need >= 3 frames.
            ndump = max(1, n_oe_steps // 2)
            template_content = re.sub(
                r'(variable\s+Ndump\s+equal\s+)\S+',
                rf'\g<1>{ndump}',
                template_content)
            with open(chunk_script, 'w') as fout:
                fout.write(template_content)
                fout.write(f'\nrun {n_oe_steps}\n')
                fout.write(f'write_restart restart.dat\n')
        else:
            write_openedge_continue_script(
                chunk_script, oe_template, n_oe_steps, plasma_h5_path)

        mode = "init" if k == 0 else "restart"
        with console.status(
                f"[yellow]Step {step_label}[/yellow]  "
                f"Running OpenEdge ({n_oe_steps} steps, {mode})...",
                spinner="dots"):
            oe_cmd = (f'mpirun -np {mpi_np_oe} {oe_binary} '
                      f'-in {os.path.basename(chunk_script)}')
            result = subprocess.run(oe_cmd, shell=True, cwd=oe_run_dir,
                                    capture_output=True, text=True, env={
                                        **os.environ,
                                        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu/hdf5/serial'
                                    })

        npart = 0
        li_source = 0.0
        n_active = 0
        if result.returncode != 0:
            err_log = os.path.join(oe_run_dir, f'error_{k:04d}.log')
            with open(err_log, 'w') as ef:
                ef.write(result.stdout + '\n' + result.stderr)
            raise OpenEdgeError(
                f"Step {step_label}: OpenEdge failed (rc={result.returncode})\n"
                f"  Error log: {err_log}\n"
                f"  Run dir: {oe_run_dir}")
        else:
            npart = parse_openedge_npart(result.stdout)
            npart_color = "green" if npart and npart > 0 else "red"
            console.print(
                f"  [green]OpenEdge done[/green]  "
                f"Particles: [{npart_color}]{npart}[/{npart_color}]")

            # ---- 2. Read particle mass loss and map to SOLPS ----
            if not os.path.exists(particle_dump):
                console.print("  [yellow]No particle dump, zero source[/yellow]")
                source_new = np.zeros((nxp2, nyp2))
            else:
                xc, yc, dm, dn = read_particle_mass_loss(particle_dump)
                if xc is not None and len(xc) > 0:
                    source_new = map_source_to_solps(xc, yc, dn, solps, dt_window)
                    li_source = np.sum(source_new)
                    n_active = np.count_nonzero(source_new)
                    console.print(
                        f"  Li source: [cyan]{li_source:.3e}[/cyan] atoms/s  "
                        f"({n_active} active SOLPS cells)")
                else:
                    source_new = np.zeros((nxp2, nyp2))

        # ---- 3. Apply under-relaxation ----
        if k == 0:
            source_used = source_new
        else:
            source_used = alpha * source_new + (1 - alpha) * source_prev
        source_prev = source_used.copy()

        # ---- Stop if no particles and no source ----
        if npart == 0 and np.sum(np.abs(source_used)) == 0:
            console.print(
                f"  [yellow]All droplets evaporated and Li source is zero — "
                f"stopping coupling loop[/yellow]")
            # Still record history for this step
            history.append({
                'step_label': step_label,
                'npart': 0,
                'li_source': 0.0,
                'ne_min': ne_prev.min(), 'ne_max': ne_prev.max(),
                'te_min': te_prev.min(), 'te_max': te_prev.max(),
                'dne_max': None, 'dte_max': None,
                'elapsed': time.time() - t0,
            })
            break

        # ---- 4. Write source2d and run SOLPS ----
        source_file = f'source2d.{k+1:05d}'
        solps.write_source2d(source_used, source_file)

        solps.write_sources_profile_chain(
            n_windows=1,
            dt_windows=[dt_window],
            t_start=k * dt_window)

        update_b2mn_dat(solps_run_dir, {
            "b2mndr_ntim": str(n_solps_steps),
            "b2sral_inputfile": "2",  # read external source from b2.sources.profile
        })

        # Ensure b2fstati exists for SOLPS restart
        b2fstate_path = os.path.join(solps_run_dir, 'b2fstate')
        b2fstati_path = os.path.join(solps_run_dir, 'b2fstati')
        baserun_stati = os.path.join(solps_base_dir, 'b2fstati')
        if k > 0 and os.path.exists(b2fstate_path) \
                and os.path.getsize(b2fstate_path) > 10000:
            # Use latest SOLPS output as restart
            shutil.copy2(b2fstate_path, b2fstati_path)
        elif not os.path.exists(b2fstati_path) \
                or os.path.getsize(b2fstati_path) < 10000:
            # Restore from baserun if b2fstati is missing or empty
            if os.path.exists(baserun_stati):
                shutil.copy2(baserun_stati, b2fstati_path)
                console.print(f"  [dim]Restored b2fstati from baserun[/dim]")

        # Clean SOLPS output so b2run doesn't skip ("up to date")
        for f_clean in ['b2mn.prt', 'b2fstate', 'b2fplasma', 'b2fparam']:
            p = os.path.join(solps_run_dir, f_clean)
            if os.path.exists(p):
                os.remove(p)

        with console.status(
                f"[yellow]Step {step_label}[/yellow]  "
                f"Running SOLPS ({n_solps_steps} steps)...",
                spinner="dots"):
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

        # Fail fast on SOLPS errors
        if result.returncode != 0:
            raise SOLPSError(
                f"Step {step_label}: SOLPS process failed "
                f"(rc={result.returncode})\n"
                f"  Run dir: {solps_run_dir}")
        check_solps_success(solps_run_dir,
                            label=f"Step {step_label}")

        # ---- 5. Read updated plasma and refresh OpenEdge ----
        dne_max = None
        dte_max = None
        ne_min = ne_prev.min()
        ne_max = ne_prev.max()
        te_min = te_prev.min()
        te_max = te_prev.max()

        solps.load_plasma_state()
        solps.write_plasma_h5(plasma_h5_path)

        fields_new = solps.get_plasma_fields()
        ne_new = np.array(fields_new['ne'])
        te_new = np.array(fields_new['te']) / eV

        ne_min, ne_max = ne_new.min(), ne_new.max()
        te_min, te_max = te_new.min(), te_new.max()

        # Compute relative changes
        ne_mask = ne_prev > 1e10
        te_mask = te_prev > 0.01
        dne_rel = np.zeros_like(ne_new)
        dte_rel = np.zeros_like(te_new)
        dne_rel[ne_mask] = (ne_new[ne_mask] - ne_prev[ne_mask]) / ne_prev[ne_mask]
        dte_rel[te_mask] = (te_new[te_mask] - te_prev[te_mask]) / te_prev[te_mask]
        dne_max = float(np.max(np.abs(dne_rel)))
        dte_max = float(np.max(np.abs(dte_rel)))

        ne_prev = ne_new
        te_prev = te_new

        console.print(f"  [green]SOLPS done[/green]  "
                      f"|dne/ne|={dne_max:.2e}  |dTe/Te|={dte_max:.2e}")

        elapsed = time.time() - t0

        # Record history
        history.append({
            'step_label': step_label,
            'npart': npart,
            'li_source': li_source,
            'ne_min': ne_min, 'ne_max': ne_max,
            'te_min': te_min, 'te_max': te_max,
            'dne_max': dne_max, 'dte_max': dte_max,
            'elapsed': elapsed,
        })

        # Archive step output
        save_step_output(k, coupled_dir or os.path.dirname(solps_run_dir),
                         solps_run_dir, oe_run_dir, plasma_h5_path,
                         source_used, solps)

        # Print running summary table
        console.print()
        console.print(make_history_table(history))
        console.print()

    # Final summary
    total_time = sum(h['elapsed'] for h in history)
    console.print(Panel(
        f"[bold green]Coupling complete[/bold green]  "
        f"{n_coupling_steps} steps in {total_time:.1f}s",
        border_style="green", expand=False))


def update_b2mn_dat(run_dir, updates):
    """Update key=value pairs in b2mn.dat.

    Args:
        run_dir: SOLPS run directory containing b2mn.dat
        updates: dict of {key: value} to set, e.g.
                 {"b2mndr_ntim": "100", "b2sral_inputfile": "0"}
    """
    dat_path = os.path.join(run_dir, 'b2mn.dat')
    if not os.path.exists(dat_path):
        return

    with open(dat_path, 'r') as f:
        lines = f.readlines()

    with open(dat_path, 'w') as f:
        for line in lines:
            matched = False
            for key, val in updates.items():
                if f"'{key}'" in line:
                    parts = line.split('#')
                    comment = ' # ' + parts[1].strip() if len(parts) > 1 else ''
                    # Pad key to match SOLPS formatting
                    f.write(f"'{key}'{' ' * max(1, 30 - len(key))}'{val}'{comment}\n")
                    matched = True
                    break
            if not matched:
                f.write(line)


def update_b2mn_ntim(run_dir, ntim):
    """Update b2mndr_ntim in b2mn.dat."""
    update_b2mn_dat(run_dir, {"b2mndr_ntim": str(ntim)})


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='OpenEdge-SOLPS coupling driver')
    parser.add_argument('config', help='JSON config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    try:
        run_coupling(config)
    except SOLPSError as e:
        console.print(Panel(
            f"[bold red]SOLPS FAILED[/bold red]\n\n{e}",
            border_style="red", expand=False))
        sys.exit(1)
    except OpenEdgeError as e:
        console.print(Panel(
            f"[bold red]OpenEdge FAILED[/bold red]\n\n{e}",
            border_style="red", expand=False))
        sys.exit(1)
    except CouplingError as e:
        console.print(Panel(
            f"[bold red]Coupling FAILED[/bold red]\n\n{e}",
            border_style="red", expand=False))
        sys.exit(1)


if __name__ == '__main__':
    main()
