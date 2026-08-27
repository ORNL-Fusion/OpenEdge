# Compiling and installing OpenEdge

OpenEdge is built with the CMake build system it inherits from SPARTA.
The executable is named `spa_<SPARTA_MACHINE>` (e.g. `spa_kokkos_omp`,
`spa_kokkos_cuda_perlmutter`). Note: the CMake option names keep their
upstream `SPARTA_*` / `PKG_*` / `BUILD_*` prefixes — those are the real
knobs; only the project content differs.

```bash
cd /path/to/openedge

# Out-of-source build (recommended; use -S/-B so the source tree stays
# clean and the build can live on a fast/scratch filesystem)
cmake -S /path/to/openedge/cmake -B /path/to/build \
      -DCMAKE_INSTALL_PREFIX=/path/to/install \
      -DPKG_OPENEDGE=ON

# List project-specific options
cmake -L /path/to/openedge/cmake

# List project-specific options and help strings
cmake -LH /path/to/openedge/cmake

# List all options and help strings
cmake -LAH

# Show all generated targets
cmake --build /path/to/build --target help

# Build the project
cmake --build /path/to/build -j <N>

# Install the binaries and libraries
cmake --install /path/to/build
```

`PKG_OPENEDGE=ON` is required for OpenEdge physics (plasma-edge styles,
PWI surface chemistry, background plasma fixes, pushers); it is OFF by
default so a plain-SPARTA build is still possible from the same tree.
OpenEdge builds also need an HDF5 installation (plasma files and the
reaction database are HDF5); on Cray systems `module load cray-hdf5`
before configuring.

# OpenEdge preset files

Commonly used configuration settings ship as cmake cache files under
`/path/to/openedge/cmake/presets`. This lets developers set
configuration knobs before building.

## Using a preset file
```bash
cmake -C /path/to/openedge/cmake/presets/<NAME>.cmake /path/to/openedge/cmake
```

Presets most relevant to OpenEdge work:

* `kokkos_omp.cmake` — Kokkos with the OpenMP host backend
  (`spa_kokkos_omp`). Exercises the same Kokkos code paths as CUDA on
  CPU hardware; the standard first target when validating device ports.
* `kokkos_cuda.cmake` — generic Kokkos/CUDA build.
* `kokkos_cuda_perlmutter.cmake` — Kokkos/CUDA pinned for NERSC
  Perlmutter A100s (`sm_80`, seeds the arch flag at
  compiler-identification time so CMake's compiler probe survives newer
  CUDA toolkits). Produces `spa_kokkos_cuda_perlmutter`.
* `mpi.cmake`, `serial.cmake`, `mac*.cmake`, … — upstream SPARTA
  presets, still functional for non-Kokkos builds.

## NERSC Perlmutter example (validated stack)

```bash
module reset
module load cpe/25.09
module load PrgEnv-gnu
module load cudatoolkit/12.9      # GPU builds only
module unload craype-accel-nvidia80
unset CRAY_ACCEL_TARGET
module load cray-hdf5

cmake -S ~/OpenEdge/cmake -B $PSCRATCH/openedge-build-gpu \
      -C ~/OpenEdge/cmake/presets/kokkos_cuda_perlmutter.cmake \
      -DPKG_OPENEDGE=ON
cmake --build $PSCRATCH/openedge-build-gpu -j 16
```

Two Perlmutter-specific cautions learned the hard way:
1. `craype-accel-nvidia80` at build or run time drags in the CUDA-13
   GTL and breaks the pinned cudatoolkit-12 stack — keep it unloaded.
2. Adding or removing source files re-triggers the CMake glob
   (`CONFIGURE_DEPENDS`) reconfigure, which is sensitive to Lustre
   health; if a reconfigure hangs in `cl_sync_io_wait`, abandon that
   build directory and configure a fresh one rather than retrying.

# Keyword listing

## Package options
* PKG_OPENEDGE
  * Whether to enable the OpenEdge plasma-edge package (plasma-wall
    interaction, background plasma, impurity chemistry, pushers).
    Default OFF; required for OpenEdge builds.
* PKG_KOKKOS
  * Whether to enable the KOKKOS package (GPU/threaded backends,
    including the OpenEdge Kokkos ports under `src/KOKKOS`).
* PKG_FFT
  * Whether to enable the FFT package.
* PKG_MPI_STUBS
  * Whether to enable the MPI_STUBS package (serial builds).
* PKG_VTK
  * Whether to enable the VTK dump package (`grid/vtk`, `surf/vtk`,
    `particle/vtk` styles). Requires an external VTK >= 7.1.

## Third Party Library (TPL) options
* BUILD_KOKKOS
  * Whether to build the bundled Kokkos TPL (`lib/kokkos`).
* BUILD_MPI
  * Whether to enable the MPI TPL.
* BUILD_JPEG
  * Whether to enable the JPEG TPL.
* BUILD_PNG
  * Whether to enable the PNG TPL.
* FFT
  * Which FFT TPL to enable: FFTW3, MKL, or KISS.
* FFT_KOKKOS
  * Which Kokkos FFT TPL to enable: CUFFT, HIPFFT, FFTW3, MKL, or KISS.

Note: To point to a TPL installation, export `<TPL>_ROOT=/path/to/tpl/install`
before running cmake.

## Other options
* SPARTA_MACHINE
  * String to form the `spa_$SPARTA_MACHINE` binary file name.
* SPARTA_CXX_COMPILE_FLAGS
  * Selected compiler flags used when building object files for `spa_$SPARTA_MACHINE`.
* SPARTA_DEFAULT_CXX_COMPILE_FLAGS
  * Default compiler flags used when building object files for `spa_$SPARTA_MACHINE`.
* SPARTA_LIST_PKGS
  * Print the packages and exit.
* SPARTA_LIST_TPLS
  * Print the TPLs and exit.
* SPARTA_ENABLE_TESTING
  * Add tests in examples to be run via ctest.
* SPARTA_ENABLE_PARAVIEW_TESTING
  * Enable ParaView tests. Default is OFF.
  * When ON, must specify SPARTA_PARAVIEW_BIN_DIR and SPARTA_PARAVIEW_MPIEXEC.
* SPARTA_PARAVIEW_BIN_DIR
  * Path to ParaView bin directory containing pvbatch and pvpython.
* SPARTA_PARAVIEW_MPIEXEC
  * Path to program used to start ParaView mpi jobs, typically mpiexec in SPARTA_PARAVIEW_BIN_DIR.
* SPARTA_DSMC_TESTING_PATH
  * Add tests in SPARTA_DSMC_TESTING_PATH/examples to be run via ctest.
  * Run all tests via SPARTA_DSMC_TESTING_PATH/regression.py.
* SPARTA_SPA_ARGS
  * Additional arguments for the binary. Only applied if SPARTA_ENABLE_TESTING or
  SPARTA_DSMC_TESTING_PATH are enabled.
* SPARTA_DSMC_TESTING_DRIVER_ARGS
  * Additional arguments for SPARTA_DSMC_TESTING_PATH/regression.py.
* SPARTA_CTEST_CONFIGS
  * Additional ctest configurations, separated by `;`, that allow `SPARTA_SPA_ARGS_<CONFIG_NAME>` or `SPARTA_DSMC_TESTING_DRIVER_ARGS_<CONFIG_NAME>` to be specified.
* SPARTA_MULTIBUILD_CONFIGS
  * Additional build configurations, separated by `;`, build with the cache file from `SPARTA_MULTIBUILD_PRESET_DIR/<CONFIG_NAME>.cmake`.
* SPARTA_MULTIBUILD_PRESET_DIR
  * The path to custom preset files when using `SPARTA_MULTIBUILD_CONFIGS`. Only applied if `SPARTA_MULTIBUILD_CONFIGS` is enabled.

## Examples
### Selecting packages via the command line
```bash
cmake -DPKG_<NAME>=[ON|OFF] /path/to/openedge/cmake
```

### Selecting TPLs via the command line
```bash
cmake -DBUILD_<NAME>_TPL=[ON|OFF] /path/to/openedge/cmake
```

### Specifying build flags via the command line
```bash
cmake -DSPARTA_DEFAULT_CXX_COMPILE_FLAGS=<FLAGS> /path/to/openedge/cmake
```

### Specifying multiple ctest configurations via the command line
```bash
cmake -DSPARTA_CTEST_CONFIGS="PARALLEL;SERIAL" \
      -DSPARTA_SPA_ARGS_SERIAL=spa_args \
      -DSPARTA_DSMC_TESTING_DRIVER_ARGS_PARALLEL=driver_args \
      /path/to/openedge/cmake

make -j

ctest -C SERIAL
ctest -C PARALLEL
```

### Specifying multiple build configurations via the command line
```bash
# Assumes that /path/to/openedge/cmake/presets/{test_mac_mpi,test_mac}.cmake exist
cmake -DSPARTA_MULTIBUILD_CONFIGS="test_mac;test_mac_mpi" \
      -DSPARTA_MULTIBUILD_PRESET_DIR=/path/to/openedge/cmake/presets/ \
      /path/to/openedge/cmake

make -j

ctest -VV
```

# Build system design
## Targets and dependency resolution
This build system consists of five targets:

1. `spa_$CONFIG_STRING`: The final OpenEdge executable
2. `pkg_fft`: The optional FFT package
3. `pkg_mpi_stubs`: The optional MPI STUBS package
4. `pkg_kokkos`: The optional kokkos wrapper package (contains the
   OpenEdge device ports when PKG_OPENEDGE is also enabled)
5. `pkg_openedge`: The OpenEdge physics package

Every target is responsible for resolving its own dependencies. Every target A that
relies on another target B will pull in the dependencies that target B resolved.

Targets 2-5 are optional packages built as static libraries; target 1
links against those that are enabled.

Source discovery uses CMake `GLOB ... CONFIGURE_DEPENDS`, so newly
added source files are picked up by the next `cmake --build` without a
manual reconfigure. Style headers (`style_*.h`) are regenerated at that
reconfigure; a fix/compute/pusher added under `src/OPENEDGE` or
`src/KOKKOS` with the usual `FixStyle(...)`-macro header registers
automatically.

## The structure of the `openedge/cmake` directory
This directory contains two directories: `common` and `presets`.
### presets
Contains preset options that can be selected via:
`cmake -C /path/to/presets/<NAME>.cmake`
### common
Contains three directories: `set`, `process`, and `print`.  Each of these
directories contains cmake files that are included by the top-level
`CMakeLists.txt`. These `common` cmake files set build options,
process those build options, and finally print the settings that were selected.

# Build system triaging
## Quick start
```bash
cmake --log-level=VERBOSE [-C /path/to/openedge/cmake/presets/<NAME>.cmake] /path/to/openedge/cmake
make VERBOSE=1
```

One override-file caution specific to this fork: several sources exist
both flat in `src/` and under `src/OPENEDGE/`; the build compiles the
flat copy and the linker picks one implementation. When editing such a
file, keep the flat and package copies byte-identical or the binary can
silently carry the stale twin.
