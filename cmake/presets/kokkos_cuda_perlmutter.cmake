# Kokkos/CUDA build for NERSC Perlmutter GPU nodes (NVIDIA A100, sm_80).

include(${CMAKE_CURRENT_LIST_DIR}/kokkos_cuda.cmake)

# nvcc_wrapper is CMake's C++ compiler, so CMake probes it before Kokkos has
# had a chance to translate Kokkos_ARCH_AMPERE80 into an nvcc flag.  Kokkos
# 4.5's wrapper otherwise defaults that probe to sm_70, which newer Perlmutter
# CUDA toolkits no longer accept.  Seed sm_80 at compiler-identification time;
# the wrapper de-duplicates the identical flag later supplied by Kokkos.
set(CMAKE_CXX_FLAGS
    "-g -O3 -arch=sm_80"
    CACHE STRING "C++/CUDA flags for Perlmutter A100" FORCE)

set(SPARTA_MACHINE
    kokkos_cuda_perlmutter
    CACHE STRING
          "OpenEdge Kokkos/CUDA build for NERSC Perlmutter A100 GPUs"
          FORCE)

set(Kokkos_ARCH_AMPERE80
    ON
    CACHE BOOL "Target NVIDIA A100 (sm_80)" FORCE)
