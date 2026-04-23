# OpenEdge Package for SPARTA

OpenEdge is a plasma-edge simulation package built on top of SPARTA. It follows
the LAMMPS/SPARTA package convention with install/uninstall support.

## Package Contents

The package has two categories of files:

**Override files** (24 files: goal is to reduce this number and stick with native sparta files) — modified versions of base SPARTA files:

    update.cpp/h  input.cpp/h  particle.cpp/h  variable.cpp/h
    compute_grid.cpp/h  dump_particle.cpp/h  sparta.cpp/h
    fix_emit_face_file.cpp/h  fix_emit_surf.cpp/h  fix_field_grid.cpp/h
    fix_field_particle.cpp/h  surf_collide_diffuse.cpp/h

**New files** (54 files) — entirely new OpenEdge additions:

    boris_grid.h  sheath_models.cpp/h  nanbu_scatter_table.h
    compute_incident_plasma_flux.cpp/h
    compute_plasma_fields.cpp/h  compute_pmi_surf_data.cpp/h
    compute_sheath_geometry_grid.cpp/h
    compute_surf_ead.cpp/h  compute_thermal_sheath_grid.cpp/h
    fix_bfield_grid.cpp/h  fix_bfield_particle.cpp/h
    fix_chem_adas.cpp/h  fix_coll_background.cpp/h  fix_coll_nanbu.cpp/h
    fix_drag.cpp/h  fix_efield_grid.cpp/h  fix_efield_particle.cpp/h
    fix_emit_droplet.cpp/h  fix_emit_surf_file.cpp/h  fix_surface_emit_sputter.cpp/h
    fix_evaporation.cpp/h  fix_gravity.cpp/h
    fix_thermal_force.cpp/h  fix_thermal_force_e.cpp/h  fix_thermal_force_i.cpp/h
    fix_cross_diffusion.cpp/h  fix_reflect_psi.cpp/h  fix_viscous.cpp/h
    fix_particle_weight.cpp/h
    geqdsk_reader.cpp/h  grid_src.h
    surf_react_mpex.cpp/h  surf_react_pmi.cpp/h

## Prerequisites

- **cmake** >= 3.12
- **MPI** — any MPI implementation (Intel MPI, OpenMPI, MPICH)
- **HDF5** with C++ bindings (`libhdf5-dev` or equivalent)

For Kokkos GPU acceleration or other optional libraries, see the
[SPARTA manual](https://sparta.github.io/doc/Section_start.html).

## Building on ORNL cloud (mora)

### Quick build

```bash
cd ~/buildOpenEdge
./build_on_cloud.sh

# Then build:
source /opt/intel/oneapi/setvars.sh --force
LD_LIBRARY_PATH= make -j16
```

### Manual build

```bash
mkdir -p ~/buildOpenEdge && cd ~/buildOpenEdge

source /opt/intel/oneapi/setvars.sh --force

LD_LIBRARY_PATH= HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial cmake \
    -C ../OpenEdge/cmake/presets/mpi.cmake ../OpenEdge/cmake/ \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DHDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/serial \
    -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=TRUE \
    -DPKG_OPENEDGE=ON

LD_LIBRARY_PATH= make -j16
```

The binary is produced at `~/buildOpenEdge/src/spa_mpi`.

### ORNL cloud notes

- `LD_LIBRARY_PATH=` is cleared to avoid linuxbrew glibc conflicts on mora.
- `-DPKG_OPENEDGE=ON` enables the OpenEdge package and adds the
  `-DSPARTA_OPENEDGE` compile flag.
- The override files are already committed to `src/`, so the cmake build picks
  them up automatically. You do **not** need to run `make yes-openedge` before
  a cmake build.

### Rebuilding after code changes

If you edit files in `src/OPENEDGE/`, re-run cmake configure and build:

```bash
cd ~/buildOpenEdge
source /opt/intel/oneapi/setvars.sh --force
LD_LIBRARY_PATH= cmake ../OpenEdge/cmake/ -DPKG_OPENEDGE=ON
LD_LIBRARY_PATH= make -j16
```

If only source files changed (no new files added/removed), just rebuild:

```bash
cd ~/buildOpenEdge
source /opt/intel/oneapi/setvars.sh --force
LD_LIBRARY_PATH= make -j16
```

## Makefile package management

If you are using the traditional Makefile build (not cmake), the package can be
installed and uninstalled like any SPARTA package:

```bash
cd ~/OpenEdge/src

# Install the OpenEdge package into src/
make yes-openedge

# Check installation status
make package-status

# Uninstall (restores original SPARTA files)
make no-openedge
```

### How install/uninstall works

- `make yes-openedge` copies all files from `src/OPENEDGE/` into `src/`.
  Override files are backed up to `*.sparta_orig` before being replaced.
- `make no-openedge` restores the original SPARTA files from `*.sparta_orig`
  backups and removes new OpenEdge files from `src/`.
- `make package-status` reports whether the package is currently installed.

## Sheath Models

OpenEdge provides two approaches for applying the sheath electric field to
particles approaching PFC surfaces:

### Spatially-resolved sheath E-field (`kick no`, default)

The sheath potential profile is modeled as a function of distance to the
nearest surface element.  At each Boris subcycle, the local E-field is
evaluated and applied to the particle.  Three models are available:

- **borodkina** (default) — Polynomial blending between Debye sheath (DS) and
  Chodura/magnetic pre-sheath (CS) based on the Borodkina & Komm (2015)
  parameterization.
- **coulette_manfredi** — Two-exponential fit to kinetic PIC data from
  Coulette & Manfredi, PPCF 58 025008 (2016).  Captures the full CS→DS
  transition for α ∈ [2°,90°] with coefficients fit to Vlasov simulation
  data at ρ_i = 20λ_D, scaled to arbitrary ρ_i/λ_D ratios.

An overshoot guard prevents energy loss when Boris subcycling pushes a
particle past the wall surface: the initial signed distance is recorded
before the subcycle loop and sheath E-field is only applied while the
particle remains on its original side of the wall.

### Sheath velocity kick (`kick yes`, recommended for IEADs)

Instead of resolving the sheath spatially, the full sheath potential drop
is applied as a velocity boost when the particle hits the wall surface.
This is the approach used by EIRENE, ERO2.0, and WallDYN.

At each surface collision:

1. Local Te, Ti are queried at the particle position from the plasma compute
2. The floating potential is computed:
   φ_float = 0.5·ln[m_D/(2π·m_e)·1/(1 + Ti/Te)] · Te
3. The particle gains kinetic energy ΔE = Z·e·φ along the wall normal:
   v_n,new = √(v_n² + 2·Z·e·φ/m)

This guarantees correct total sheath energy regardless of Boris timestep,
eliminates gyro-orbit resonance at grazing angles, and avoids per-subcycle
sheath E-field evaluations (faster).  Validated against an independent
Fortran sheath tracker for Ta²⁺/Ta³⁺/Ta⁴⁺ at α = 0°, 45°, 85° with
<1% energy error.

### Input syntax

```
compute   cgeom sheath/geometry/grid all all dist nx ny nz surfidx
global    sheath geom_compute cgeom plasma_compute cplasma &
          [model borodkina/coulette_manfredi] &
          [mD_amu 2.0] [pot_mult 0] [dmax 0.02] &
          [kick yes/no]
```

- **pot_mult** — If > 0, use pot_mult·Te as the wall potential (eV).
  If 0, use the self-consistent floating potential formula.
- **mD_amu** — Background ion mass (amu) for sound speed and Debye length.
- **kick yes** — Apply sheath as velocity kick at wall (recommended).
  When active, the `model` keyword is ignored (no spatial E-field).

## Core Boundary (fix reflect/psi)

Reflects or deletes particles that cross a normalized poloidal flux
threshold, providing a core boundary without requiring a watertight
inner surface mesh.

Reads a SOLPS `.equ` equilibrium file, computes psi_norm = (psi - psi_axis) /
(psib - psi_axis) at each particle position, and reflects (or deletes)
particles with psi_norm below the threshold.

### Input syntax

```
fix ID reflect/psi Nevery equ PATH psi_norm VALUE [action reflect|delete]
```

- **Nevery** — check every N timesteps (1 = every step)
- **equ PATH** — path to SOLPS `.equ` equilibrium file (also accepts `geqdsk` keyword)
- **psi_norm VALUE** — normalized psi threshold (0 = axis, 1 = separatrix)
- **action reflect** (default) — restore previous position and reverse radial velocity
- **action delete** — remove particle from simulation

### Example

```
# Reflect particles that enter the core (psi_norm < 0.926)
fix fcore reflect/psi 1 equ input/g174310.03500_153.X4.equ psi_norm 0.926

# Delete particles instead of reflecting
fix fcore reflect/psi 1 equ input/g174310.03500_153.X4.equ psi_norm 0.926 action delete
```

### Determining the psi_norm threshold

Use `check_psi_norm.py` to evaluate psi_norm at the points of a flux surface:

```bash
python3 input/check_psi_norm.py input/equilibrium.equ input/flux_surface.surf
```

Use `plot_psi_surface.py` to visualize psi contours with the flux surface overlaid:

```bash
python3 input/plot_psi_surface.py input/equilibrium.equ input/flux_surface.surf
```
