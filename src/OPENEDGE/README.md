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

**New files** (56 files) — entirely new OpenEdge additions:

    boris_grid.h  sheath_models.cpp/h  nanbu_scatter_table.h
    compute_iead.cpp/h  compute_incident_plasma_flux.cpp/h
    compute_plasma_fields.cpp/h  compute_pmi_surf_data.cpp/h
    compute_sheath_fields_grid.cpp/h  compute_sheath_geometry_grid.cpp/h
    compute_surf_ead.cpp/h  compute_thermal_sheath_grid.cpp/h
    fix_bfield_grid.cpp/h  fix_bfield_particle.cpp/h
    fix_chem_adas.cpp/h  fix_coll_background.cpp/h  fix_coll_nanbu.cpp/h
    fix_drag.cpp/h  fix_efield_grid.cpp/h  fix_efield_particle.cpp/h
    fix_emit_droplet.cpp/h  fix_emit_surf_file.cpp/h  fix_emit_surf_pmi.cpp/h
    fix_evaporation.cpp/h  fix_gravity.cpp/h
    fix_thermal_force_e.cpp/h  fix_thermal_force_i.cpp/h  fix_viscous.cpp/h
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
