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

#ifndef KOKKOS_SPARTA_H
#define KOKKOS_SPARTA_H

#include "pointers.h"
#include "kokkos_type.h"
#include <string>
#include <vector>

namespace SPARTA_NS {

class KokkosSPARTA : protected Pointers {
 public:
  int kokkos_exists;
  int comm_serial;
  int atomic_reduction;
  int prewrap;
  int auto_sync;
  int nthreads,ngpus;
  int need_atomics;
  int gpu_aware_flag;
  int react_retry_flag;
  double react_extra;
  int fallback_strict;       // OpenEdge: 1 = error on any host fallback (package kokkos fallback error / OE_KK_STRICT)
  int checksync;             // OpenEdge: 0 off, 1 count DualView both-modified conflicts, 2 error on the first (package kokkos checksync / OE_KK_CHECKSYNC)

  KokkosSPARTA(class SPARTA *, int, char **);
  ~KokkosSPARTA();
  void accelerator(int, char **);

  // OpenEdge host-fallback ledger: every Kokkos class that runs a hook on
  // the host instead of the device calls note_fallback() on each such
  // call; fallback_report() (end of every run) prints the per-class totals
  // over all ranks, or a single 'none' line, then clears the ledger.
  void note_fallback(const char *who, const char *why);
  void fallback_report(FILE *screen, FILE *logfile);

  // OpenEdge DualView conflict ledger: called by Particle/Grid/SurfKokkos::modify
  // when the other memory space already holds newer data (that data is lost)
  void note_sync_conflict(const char *what, const char *space);

  template<class DeviceType>
  int need_dup()
  {
    int value = 0;

    if (need_atomics)
      value = std::is_same<typename NeedDup<1,DeviceType>::value,Kokkos::Experimental::ScatterDuplicated>::value;

    return value;
  }

 private:
  struct FallbackEntry { std::string who, why; long count; long first, last; };
  std::vector<FallbackEntry> fallbacks;
  std::vector<FallbackEntry> conflicts;
  void ledger_report(std::vector<FallbackEntry> &ledger, const char *title, const char *none, FILE *screen, FILE *logfile);
  static void my_signal_handler(int);
};

}

#endif

/* ERROR/WARNING messages:

E: Invalid Kokkos command-line args

Self-explanatory.  See Section ? of the manual for details.

E: Could not determine local MPI rank for multiple GPUs with Kokkos CUDA because MPI library not recognized

The local MPI rank was not found in one of five supported environment variables.

E: GPUs are requested but Kokkos has not been compiled for CUDA

Recompile Kokkos with CUDA support to use GPUs.

E: Kokkos has been compiled for CUDA but no GPUs are requested

One or more GPUs must be used when Kokkos is compiled for CUDA.
*/
