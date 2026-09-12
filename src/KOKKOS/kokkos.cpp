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

#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "ctype.h"
#include "signal.h"
#include "kokkos.h"
#include "sparta.h"
#include "error.h"
#include "memory_kokkos.h"
#include "update.h"
#include "comm.h"
#include <map>

using namespace SPARTA_NS;

// Kokkos may be initialized at most once per process and, once finalized, can
// never be initialized again. When SPARTA is embedded as a library (a host
// application reusing one process across many open/close cycles) Kokkos must be
// initialized on first use and finalized once at process exit -- not in the
// KokkosSPARTA destructor. These file-scope helpers track that lifetime.

static int kokkos_initialized_nthreads = 0;

static void sparta_kokkos_atexit()
{
  if (Kokkos::is_initialized() && !Kokkos::is_finalized()) Kokkos::finalize();
}

/* ---------------------------------------------------------------------- */

KokkosSPARTA::KokkosSPARTA(SPARTA *sparta, int narg, char **arg) : Pointers(sparta)
{
  kokkos_exists = 1;
  sparta->kokkos = this;

  delete memory;
  memory = new MemoryKokkos(sparta);
  memoryKK = (MemoryKokkos*) memory;

  int me = 0;
  MPI_Comm_rank(world,&me);
  if (me == 0) error->message(FLERR,"KOKKOS mode is enabled");

  // process any command-line args that invoke Kokkos settings

  ngpus = 0;
  int device = 0;
  nthreads = 1;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"d") == 0 || strcmp(arg[iarg],"device") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Invalid Kokkos command-line args");
      device = atoi(arg[iarg+1]);
      iarg += 2;

    } else if (strcmp(arg[iarg],"g") == 0 ||
               strcmp(arg[iarg],"gpus") == 0) {
#ifndef SPARTA_KOKKOS_GPU
      error->all(FLERR,"GPUs are requested but Kokkos has not been compiled with a GPU-enabled backend");
#endif
      if (iarg+2 > narg) error->all(FLERR,"Invalid Kokkos command-line args");
      ngpus = atoi(arg[iarg+1]);

      int skip_gpu = 9999;
      if (iarg+2 < narg && isdigit(arg[iarg+2][0])) {
        skip_gpu = atoi(arg[iarg+2]);
        iarg++;
      }
      iarg += 2;

      int set_flag = 0;
      char *str;
      if ((str = getenv("SLURM_LOCALID"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("FLUX_TASK_LOCAL_ID"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("MPT_LRANK"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("MV2_COMM_WORLD_LOCAL_RANK"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }
      if ((str = getenv("PMI_LOCAL_RANK"))) {
        int local_rank = atoi(str);
        device = local_rank % ngpus;
        if (device >= skip_gpu) device++;
        set_flag = 1;
      }

      if (ngpus > 1 && !set_flag)
        error->all(FLERR,"Could not determine local MPI rank for multiple "
                           "GPUs with because MPI library not recognized");

    } else if (strcmp(arg[iarg],"t") == 0 ||
               strcmp(arg[iarg],"threads") == 0) {
      nthreads = atoi(arg[iarg+1]);
      iarg += 2;

    } else error->all(FLERR,"Invalid Kokkos command-line args");
  }

  // initialize Kokkos

  if (me == 0) {
    if (screen) fprintf(screen,"  requested %d GPU(s) per node\n",ngpus);
    if (logfile) fprintf(logfile,"  requested %d GPU(s) per node\n",ngpus);

    if (screen) fprintf(screen,"  requested %d thread(s) per MPI task\n",nthreads);
    if (logfile) fprintf(logfile,"  requested %d thread(s) per MPI task\n",nthreads);
  }

#ifdef SPARTA_KOKKOS_GPU
  if (ngpus <= 0)
    error->all(FLERR,"Kokkos has been compiled with a GPU-enabled backend but no GPUs are requested");
#endif

#ifndef KOKKOS_ENABLE_SERIAL
  if (nthreads == 1 && me == 0)
    error->warning(FLERR,"When using a single thread, the Kokkos Serial backend "
                         "(i.e. Makefile.kokkos_mpi_only) gives better performance "
                         "than the OpenMP backend");
#endif

  Kokkos::InitializationSettings args;
  args.set_num_threads(nthreads);
  args.set_device_id(device);

  // Initialize Kokkos only once per process (it can be initialized at most
  // once), and register a one-time handler to finalize it at process exit.
  // On any later re-open the requested thread count cannot be changed, so keep
  // the count Kokkos was actually initialized with -- otherwise the atomics
  // decision below would be made for the wrong number of threads.
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize(args);
    kokkos_initialized_nthreads = nthreads;
    atexit(sparta_kokkos_atexit);
  } else {
    if (nthreads != kokkos_initialized_nthreads && me == 0)
      error->warning(FLERR,"Kokkos is already initialized in this process; "
                     "ignoring the new thread count. Restart to change the "
                     "number of threads.");
    nthreads = kokkos_initialized_nthreads;
  }

  // default settings for package kokkos command

  prewrap = 1;
  auto_sync = 1;
  gpu_aware_flag = 1;

  if (ngpus > 0) {
    comm_serial = 0;
#ifdef KOKKOS_ARCH_AMD_GFX942
    atomic_reduction = 0;
#else
    atomic_reduction = 1;
#endif
  } else {
    comm_serial = 1;
    atomic_reduction = 0;
  }

  need_atomics = 1;
  if (nthreads == 1 && ngpus == 0)
    need_atomics = 0;

  react_retry_flag = 0;
  react_extra = 1.1;
  fallback_strict = getenv("OE_KK_STRICT") ? 1 : 0;

  // finalize Kokkos on abort

  signal(SIGABRT, my_signal_handler);
}

/* ---------------------------------------------------------------------- */

KokkosSPARTA::~KokkosSPARTA()
{
  // Kokkos is finalized once at process exit (see sparta_kokkos_atexit,
  // registered in the constructor), not here, so a library embedder can
  // destroy and re-create SPARTA in the same process without tripping over
  // Kokkos's initialize-at-most-once restriction.
}

/* ----------------------------------------------------------------------
   invoked by package kokkos command
------------------------------------------------------------------------- */

void KokkosSPARTA::accelerator(int narg, char **arg)
{
  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"comm") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package kokkos command");
      if (strcmp(arg[iarg+1],"serial") == 0) {
        comm_serial = 1;
      } else if (strcmp(arg[iarg+1],"classic") == 0) { // deprecated
        comm_serial = 1;
      } else if (strcmp(arg[iarg+1],"threaded") == 0) {
        comm_serial = 0;
      } else error->all(FLERR,"Illegal package kokkos command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"react/retry") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package kokkos command");
      if (strcmp(arg[iarg+1],"yes") == 0) {
        react_retry_flag = 1;
      } else if (strcmp(arg[iarg+1],"no") == 0) {
        react_retry_flag = 0;
      } else error->all(FLERR,"Illegal package kokkos command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"fallback") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package kokkos command");
      if (strcmp(arg[iarg+1],"warn") == 0) fallback_strict = 0;
      else if (strcmp(arg[iarg+1],"error") == 0) fallback_strict = 1;
      else error->all(FLERR,"Illegal package kokkos command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"react/extra") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal package kokkos command");
      react_extra = atof(arg[iarg+1]);
      iarg += 2;
    } else if ((strcmp(arg[iarg],"gpu/aware") == 0)
               || (strcmp(arg[iarg],"gpu/direct") == 0)) { // gpu/direct is deprecated
      if (iarg+2 > narg) error->all(FLERR,"Illegal package kokkos command");
      if (strcmp(arg[iarg+1],"yes") == 0) {
        gpu_aware_flag = 1;
      } else if (strcmp(arg[iarg+1],"no") == 0) {
        gpu_aware_flag = 0;
      } else error->all(FLERR,"Illegal package kokkos command");
      iarg += 2;
    } else error->all(FLERR,"Illegal package kokkos command");
  }
}

/* ----------------------------------------------------------------------
   OpenEdge host-fallback ledger
------------------------------------------------------------------------- */

void KokkosSPARTA::note_fallback(const char *who, const char *why)
{
  if (!why) why = "(no reason recorded)";
  const long step = (long) update->ntimestep;
  for (auto &e : fallbacks) {
    if (e.who == who) {
      e.count++; e.last = step;
      if (fallback_strict) {
        char msg[512];
        snprintf(msg,sizeof(msg),"%s: host fallback (%s) with package kokkos fallback error",who,why);
        error->one(FLERR,msg);
      }
      return;
    }
  }
  FallbackEntry e; e.who = who; e.why = why; e.count = 1; e.first = e.last = step;
  fallbacks.push_back(e);
  if (fallback_strict) {
    char msg[512];
    snprintf(msg,sizeof(msg),"%s: host fallback (%s) with package kokkos fallback error",who,why);
    error->one(FLERR,msg);
  }
}

/* ----------------------------------------------------------------------
   gather every rank's ledger to rank 0 and print one line per class:
   total host calls over ranks, ranks affected, step range, first reason
------------------------------------------------------------------------- */

void KokkosSPARTA::fallback_report(FILE *screen, FILE *logfile)
{
  int me,nprocs;
  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  std::string mine;
  for (auto &e : fallbacks)
    mine += e.who + "\t" + e.why + "\t" + std::to_string(e.count) + "\t" +
            std::to_string(e.first) + "\t" + std::to_string(e.last) + "\n";
  fallbacks.clear();

  int n = (int) mine.size();
  std::vector<int> counts(nprocs),displs(nprocs);
  MPI_Gather(&n,1,MPI_INT,counts.data(),1,MPI_INT,0,world);
  int total = 0;
  if (me == 0) for (int i = 0; i < nprocs; i++) { displs[i] = total; total += counts[i]; }
  std::vector<char> all(me == 0 ? total+1 : 1);
  MPI_Gatherv(mine.data(),n,MPI_CHAR,all.data(),counts.data(),displs.data(),MPI_CHAR,0,world);
  if (me != 0) return;

  struct Agg { std::string why; long calls = 0; int ranks = 0; long first = 0, last = 0; };
  std::map<std::string,Agg> agg;
  std::vector<std::string> order;
  for (int r = 0; r < nprocs; r++) {
    std::string blob(all.data()+displs[r],counts[r]);
    size_t pos = 0;
    while (pos < blob.size()) {
      size_t nl = blob.find('\n',pos); if (nl == std::string::npos) nl = blob.size();
      std::string line = blob.substr(pos,nl-pos); pos = nl+1;
      std::vector<std::string> f; size_t p = 0;
      while (true) { size_t t = line.find('\t',p); f.push_back(line.substr(p,t==std::string::npos?std::string::npos:t-p)); if (t == std::string::npos) break; p = t+1; }
      if (f.size() < 5) continue;
      Agg &a = agg[f[0]];
      if (a.ranks == 0) { a.why = f[1]; a.first = atol(f[3].c_str()); a.last = atol(f[4].c_str()); order.push_back(f[0]); }
      a.calls += atol(f[2].c_str()); a.ranks++;
      a.first = std::min(a.first,atol(f[3].c_str())); a.last = std::max(a.last,atol(f[4].c_str()));
    }
  }

  FILE *outs[2] = {screen,logfile};
  for (FILE *out : outs) {
    if (!out) continue;
    if (order.empty()) { fprintf(out,"Kokkos host fallbacks this run: none\n"); continue; }
    fprintf(out,"Kokkos host fallbacks this run (host calls summed over ranks):\n");
    for (auto &who : order) {
      Agg &a = agg[who];
      fprintf(out,"  %-34s calls %-9ld ranks %d/%d  steps %ld-%ld  %s\n",
              who.c_str(),a.calls,a.ranks,nprocs,a.first,a.last,a.why.c_str());
    }
  }
}

void KokkosSPARTA::my_signal_handler(int sig)
{
  if (sig == SIGABRT) Kokkos::finalize();
}
