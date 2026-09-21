/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#include "mpi.h"
#include "sparta.h"
#include "input.h"
#include "spaexception.h"

#include "stdlib.h"

#ifdef SPARTA_KOKKOS
#include <Kokkos_Core.hpp>
#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif
#endif

using namespace SPARTA_NS;

/* ----------------------------------------------------------------------
   main program to drive SPARTA
------------------------------------------------------------------------- */

int main(int argc, char **argv)
{
#if defined(SPARTA_KOKKOS) && defined(KOKKOS_ENABLE_CUDA)
  // OpenEdge (2026-09-14): select this rank's GPU BEFORE MPI_Init. Cray MPICH's GTL
  // records the current device at init; when Kokkos later selects local_rank % ngpus
  // (package kokkos "g N" with all GPUs visible) the GTL reports "not using the same
  // device it did during gtl_init" and drops CUDA IPC for that rank. Same mapping as
  // KokkosSPARTA (SLURM_LOCALID % device count); no-op with one visible device.
  // OE_NO_PRESET_DEVICE=1 disables it.
  if (!getenv("OE_NO_PRESET_DEVICE")) {
    const char *lid = getenv("SLURM_LOCALID");
    int ndev = 0;
    if (lid && cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 1) {
      cudaSetDevice(atoi(lid) % ndev);
      cudaFree(0);   // create the context on that device now
    }
  }
#endif
  MPI_Init(&argc,&argv);

  try {
    SPARTA *sparta = new SPARTA(argc,argv,MPI_COMM_WORLD);
    sparta->input->file();
    delete sparta;
#ifdef SPARTA_KOKKOS
    // The executable owns the process lifetime, so finalize Kokkos after all
    // SPARTA objects (and their Views) have been destroyed but before C++
    // static destruction begins.  Deferring this solely to atexit is unsafe
    // for CUDA: lazily-created Kokkos lock Views can be destroyed before the
    // earlier-registered atexit handler, causing a double decrement there.
    // Library users do not enter this main and retain the reusable lifetime
    // policy implemented by KokkosSPARTA.
    if (Kokkos::is_initialized() && !Kokkos::is_finalized())
      Kokkos::finalize();
#endif
  } catch (SpartaAbortException &e) {
    MPI_Abort(e.get_universe(),1);
  } catch (SpartaException &) {
    // error message was already printed by the Error class
    MPI_Finalize();
    exit(1);
  }

  MPI_Finalize();
}
