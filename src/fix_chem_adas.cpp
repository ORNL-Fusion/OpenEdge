/* ----------------------------------------------------------------------
    OpenEdge: ADAS ionization/recombination chemistry fix
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
      - 42d
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "stdlib.h"
#include "string.h"
#include "fix_chem_adas.h"
#include "update.h"
#include "grid.h"
#include "particle.h"
#include "memory.h"
#include "error.h"
#include "comm.h"
#include "math.h"
#include "react_bird.h"
#include "input.h"
#include "collide.h"
#include "modify.h"
#include "fix.h"
#include "random_knuth.h"
#include "math_const.h"
#include <filesystem>
#include <vector>
#include "math_extra.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "variable.h"
#include "compute.h"
#include "compute_plasma_fields.h"

namespace fs = std::filesystem;
using namespace SPARTA_NS;
enum{IONIZATION,RECOMBINATION,EXCHANGE,DISSOCIATION};  // file-local reaction types
enum{IONIZATIONRATE, RECOMBINATIONRATE};               // other files
enum{ADAS,JANEV};                                      // rate styles


#define MAXREACTANT 2
#define MAXPRODUCT 3
#define MAXCOEFF 10              // Janev polynomials use up to 9 coeffs (b0..b8)
#define INVOKED_PER_GRID 16
#define MAXLINE 1024
#define DELTALIST 16
/* ---------------------------------------------------------------------- */

FixChemAdas::FixChemAdas(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
    // fix ID chem/adas <nevery> <Z> <reactions_file> [adas_dir <path>] [plasma <TeVar> <NeVar>]

  if (narg < 5)     error->all(FLERR,"Illegal fix chem/adas command (need: nevery Z reactions_file [adas_dir path] [plasma TeVar NeVar])");
    nevery = atoi(arg[2]);
    atomic_number = atoi(arg[3]);

    // per-cell array for aveflag = 1 case

    nlist = maxlist = 0;
    rlist = NULL;
    readfile(arg[4]);
    check_duplicate();

    // --- Optional adas_dir keyword (default: "adas") ---
    // Scan remaining args for adas_dir before consuming plasma keyword.

    std::string adas_base_dir = "adas";
    int iarg = 5;
    if (iarg < narg && strcmp(arg[iarg], "adas_dir") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"fix chem/adas: adas_dir requires a path argument");
      adas_base_dir = arg[iarg + 1];
      iarg += 2;
    }

    // read ADAS rate data

    {
      fs::path fullPath = fs::path(adas_base_dir) /
          ("ADAS_Rates_" + std::to_string(atomic_number) + ".h5");
      if (comm->me == 0)
        printf("Reading ADAS data for Z=%d from %s\n",
               atomic_number, fullPath.string().c_str());
      readRateDataParallel(fullPath.string(), materials_rate_data[atomic_number]);
    }

    // //
    maxgrid = 0;
    reactions = NULL;
    list_ij = NULL;

    tally_reactions = new bigint[nlist];
    tally_reactions_all = new bigint[nlist];
    tally_flag = 0;
    nreact_one = nreact_running = 0;
    rng_adas = nullptr;
    cp_plasma_cached_ = nullptr;

    //
   // --- Optional plasma grid args: plasma <TeSrc> <NeSrc> ---
   // If omitted, Te/ne are read from the per-particle plasma cache
   // populated by update.cpp (requires sheath or GCA plasma compute).
  if (iarg < narg && strcmp(arg[iarg], "plasma") == 0) {
    if (narg < iarg + 3)
      error->all(FLERR,"fix chem/adas plasma requires: plasma <TeSrc> <NeSrc>");

    use_grid_plasma = 1;

    auto parse_src = [&](const char *tok, GridSrc &dst, const char *label) {
      if (strncmp(tok,"c_",2)==0) {
        dst.kind = SRC_COMP;
        const char *name = tok + 2;
        const char *lb   = strchr(name,'[');
        if (!lb || tok[strlen(tok)-1] != ']') {
          char msg[160];
          snprintf(msg, sizeof(msg),
                  "fix chem/adas: bad %s token (use c_id[idx])", label);
          error->all(FLERR, msg);
        }
        const int idlen = lb - name;
        dst.cid = new char[idlen+1];
        strncpy(dst.cid, name, idlen);
        dst.cid[idlen] = '\0';
        dst.col = atoi(lb+1);
        if (dst.col <= 0) {
          char msg[160];
          snprintf(msg, sizeof(msg),
                  "fix chem/adas: %s column must be >=1", label);
          error->all(FLERR, msg);
        }
      } else {
        dst.kind = SRC_VAR;
        int n = strlen(tok) + 1;
        char *copy = new char[n];
        strcpy(copy, tok);
        if (&dst == &srcTe) tstr = copy; else nstr = copy;
      }
    };

    parse_src(arg[iarg+1], srcTe, "Te");
    parse_src(arg[iarg+2], srcNe, "ne");
    use_grid_plasma = (srcTe.kind == SRC_VAR) || (srcNe.kind == SRC_VAR);
    iarg += 3;
  }

  // --- Optional Mode A (EIRENE-semantics) keywords ---
  //   mode neutral                         -> delete particle on ionization
  //   source_species <sp1> [sp2] ...       -> species to count for exhaustion
  //   stop_on_exhaust yes|no               -> halt run when source species = 0
  while (iarg < narg) {
    if (strcmp(arg[iarg], "mode") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix chem/adas: mode requires a value (kinetic|neutral)");
      if (strcmp(arg[iarg+1], "neutral") == 0) eirene_mode = 1;
      else if (strcmp(arg[iarg+1], "kinetic") == 0) eirene_mode = 0;
      else error->all(FLERR, "fix chem/adas: unknown mode (use kinetic|neutral)");
      iarg += 2;
    } else if (strcmp(arg[iarg], "source_species") == 0) {
      // consume species names until the next keyword or end of args
      int j = iarg + 1;
      while (j < narg &&
             strcmp(arg[j], "mode") != 0 &&
             strcmp(arg[j], "stop_on_exhaust") != 0 &&
             strcmp(arg[j], "source_species") != 0 &&
             strcmp(arg[j], "units") != 0 &&
             strcmp(arg[j], "exhaust_threshold") != 0) j++;
      nsrc_species = j - (iarg + 1);
      if (nsrc_species <= 0)
        error->all(FLERR, "fix chem/adas: source_species needs >=1 species name");
      src_species_names = new char*[nsrc_species];
      for (int k = 0; k < nsrc_species; k++) {
        int n = strlen(arg[iarg + 1 + k]) + 1;
        src_species_names[k] = new char[n];
        strcpy(src_species_names[k], arg[iarg + 1 + k]);
      }
      iarg = j;
    } else if (strcmp(arg[iarg], "stop_on_exhaust") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix chem/adas: stop_on_exhaust requires yes|no");
      if (strcmp(arg[iarg+1], "yes") == 0) stop_on_exhaust = 1;
      else if (strcmp(arg[iarg+1], "no") == 0) stop_on_exhaust = 0;
      else error->all(FLERR, "fix chem/adas: stop_on_exhaust must be yes|no");
      iarg += 2;
    } else if (strcmp(arg[iarg], "exhaust_threshold") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix chem/adas: exhaust_threshold requires <N>");
      exhaust_threshold = ATOBIGINT(arg[iarg+1]);
      if (exhaust_threshold < 0)
        error->all(FLERR, "fix chem/adas: exhaust_threshold must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "units") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "fix chem/adas: units requires counts|rate|eirene");
      if (strcmp(arg[iarg+1], "counts") == 0) {
        tally_units = TALLY_COUNTS;
        iarg += 2;
      } else if (strcmp(arg[iarg+1], "rate") == 0) {
        tally_units = TALLY_RATE;
        iarg += 2;
      } else if (strcmp(arg[iarg+1], "batch") == 0) {
        // Batch (EIRENE-style MC): `units batch <N_trajectories> <R_puff>`
        if (iarg + 3 >= narg)
          error->all(FLERR, "fix chem/adas: units batch requires N R_puff");
        batch_N      = atoi(arg[iarg+2]);
        batch_R_puff = atof(arg[iarg+3]);
        if (batch_N <= 0 || batch_R_puff <= 0.0)
          error->all(FLERR, "fix chem/adas: units batch needs N>0 and R_puff>0");
        tally_units = TALLY_BATCH;
        iarg += 4;
      } else if (strcmp(arg[iarg+1], "batch_fix") == 0) {
        // EIRENE-batch auto-scaled from a paired emit fix:
        //   `units batch_fix <emit_fix_id> <R_puff>`
        if (iarg + 3 >= narg)
          error->all(FLERR, "fix chem/adas: units batch_fix requires <fix_id> R_puff");
        int n = strlen(arg[iarg+2]) + 1;
        batch_fix_id = new char[n];
        strcpy(batch_fix_id, arg[iarg+2]);
        batch_R_puff = atof(arg[iarg+3]);
        if (batch_R_puff <= 0.0)
          error->all(FLERR, "fix chem/adas: units batch_fix needs R_puff > 0");
        tally_units = TALLY_BATCH_FIX;
        iarg += 4;
      } else {
        error->all(FLERR, "fix chem/adas: units must be counts|rate|batch|batch_fix");
      }
    } else {
      char msg[160];
      snprintf(msg, sizeof(msg), "fix chem/adas: unknown keyword '%s'", arg[iarg]);
      error->all(FLERR, msg);
    }
  }

  tvar = nvar = -1;
  maxgrid_plasma = 0;
  plasma_cache_2d = NULL;

  // Expose 20-column per-grid source tally as Fix output (Gkeyll handoff).
  // Layout is quantity-major, so columns 1-4 = ionization/recomb/CX/dissoc
  // event COUNTS (same semantics as the original 4-column version).
  //
  //   cols  1 -  4 : count per cell per reaction type
  //   cols  5 -  8 : sum of m*vx at each reaction event [kg m/s]
  //   cols  9 - 12 : sum of m*vy
  //   cols 13 - 16 : sum of m*vz
  //   cols 17 - 20 : sum of 0.5*m*(vx^2+vy^2+vz^2)  [J]
  //
  // Dividing by a coupling time window t gives {particle, momentum, energy}
  // source *rates* ready for a Gkeyll collision-operator coefficient.
  // Stored in the inherited base-class array_grid.
  per_grid_flag      = 1;
  size_per_grid_cols = 20;
  per_grid_freq      = 1;     // tally is updated every step; dump_grid
                              //   does  (nevery % per_grid_freq)  so this
                              //   MUST be non-zero (SIGFPE otherwise).
  array_grid         = NULL;
  maxgrid_src        = 0;

}

/* ---------------------------------------------------------------------- */

FixChemAdas::~FixChemAdas()
{
  if (copymode) return;


  delete [] tally_reactions;
  delete [] tally_reactions_all;

  if (rlist) {
    for (int i = 0; i < maxlist; i++) {
      for (int j = 0; j < rlist[i].nreactant; j++)
        delete [] rlist[i].id_reactants[j];
      for (int j = 0; j < rlist[i].nproduct; j++)
        delete [] rlist[i].id_products[j];
      delete [] rlist[i].id_reactants;
      delete [] rlist[i].id_products;
      delete [] rlist[i].reactants;
      delete [] rlist[i].products;
      delete [] rlist[i].coeff;
      delete [] rlist[i].id;
    }
  }
  memory->destroy(rlist);

  memory->destroy(reactions);
  memory->destroy(list_ij);

delete [] tstr; delete [] nstr;
memory->destroy(plasma_cache_2d);
memory->destroy(array_grid);
delete rng_adas;

if (src_species_names) {
  for (int k = 0; k < nsrc_species; k++) delete [] src_species_names[k];
  delete [] src_species_names;
}

delete [] batch_fix_id;



}

/* ---------------------------------------------------------------------- */

void FixChemAdas::reset_tally()
{
  if (array_grid && maxgrid_src > 0)
    memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * 20);
}

/* ---------------------------------------------------------------------- */

int FixChemAdas::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixChemAdas::init()
{

  tally_flag = 0;
  nreact_one = nreact_running = 0;
  for (int i = 0; i < nlist; i++) tally_reactions[i] = 0;

  // convert species IDs to species indices
  // flag reactions as active/inactive depending on whether all species exist
  // mark recombination reactions inactive if recombflag_user = 0

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    r->active = 1;

    for (int i = 0; i < r->nreactant; i++) {
      r->reactants[i] = particle->find_species(r->id_reactants[i]);
      if (r->reactants[i] < 0) {
        r->active = 0;
        break;
      }
    }

    for (int i = 0; i < r->nproduct; i++) {
      r->products[i] = particle->find_species(r->id_products[i]);
      if (r->products[i] < 0) {
        r->active = 0;
        break;
      }
    }
  }

  // count possible active reactions for each species pair
  // include J,I reactions in I,J list and vice versa
  // this allows collision pair I,J to be in either order in Collide

  memory->destroy(reactions);
  int nspecies = particle->nspecies;
  reactions = memory->create(reactions,nspecies,
                             "react/bird:reactions");

  for (int i = 0; i < nspecies; i++)
      reactions[i].n = 0;

  int n = 0;
  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].n++;
    n++;
  }

  // allocate list_IJ = contiguous list of reactions for each IJ pair

  memory->destroy(list_ij);
  memory->create(list_ij,n,"react/bird:list_ij");

  // reactions[i][j].list = pointer into full list_ij vector

  int offset = 0;
  for (int i = 0; i < nspecies; i++){
      reactions[i].list = &list_ij[offset];
      offset += reactions[i].n;
  }
    

  // reactions[i][j].list = indices of reactions for each species pair
  // include J,I reactions in I,J list and vice versa

  for (int i = 0; i < nspecies; i++)
      reactions[i].n = 0;

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].list[reactions[i].n++] = m;
  }

  // modify Arrhenius coefficients for TCE model
  // C1,C2 Bird 94, p 127
  // initflag logic insures only done once per reaction

  Particle::Species *species = particle->species;

  for (int m = 0; m < nlist; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    if (r->initflag) continue;
    r->initflag = 1;

    int isp = r->reactants[0];
  }

// NOTE: if srcTe/srcNe are SRC_NONE, the per-particle plasma cache
// (set in update->init) will be checked at runtime in end_of_step().
// We cannot validate here because init() runs before update->init().

  // --- resolve VARIABLE sources (old path) ---
if (srcTe.kind == SRC_VAR) {
  tvar = input->variable->find(tstr);
  if (tvar < 0 || !input->variable->grid_style(tvar))
    error->all(FLERR,"Temperature variable for chem/adas must be grid-style");
}
if (srcNe.kind == SRC_VAR) {
  nvar = input->variable->find(nstr);
  if (nvar < 0 || !input->variable->grid_style(nvar))
    error->all(FLERR,"Density variable for chem/adas must be grid-style");
}
if ((srcTe.kind == SRC_VAR) || (srcNe.kind == SRC_VAR)) {
  if (grid->nlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_cache_2d);
    memory->create(plasma_cache_2d, maxgrid_plasma, 2, "chem/adas:plasma_cache_2d");
  }
  if (grid->nlocal)
    memset(&plasma_cache_2d[0][0], 0, sizeof(double)*grid->nlocal*2);
}

// Allocate per-cell source-tally output (array_grid, 20 cols) up front so
// dumps that fire at step 0 (dump_modify ... first yes) see a valid buffer.
if (grid->maxlocal > maxgrid_src) {
  maxgrid_src = grid->maxlocal;
  memory->grow(array_grid, maxgrid_src, 20, "chem/adas:array_grid(src)");
}
if (maxgrid_src > 0) {
  memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * 20);
}

// --- resolve COMPUTE sources (new path) ---
auto bind_compute = [&](GridSrc &S, const char *label){
  if (S.kind != SRC_COMP) return;

  S.icompute = modify->find_compute(S.cid);
  if (S.icompute < 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
            "fix chem/adas: compute ID for %s not found", label);
    error->all(FLERR, msg);
  }

  Compute *c = modify->compute[S.icompute];
    if (c->per_grid_flag == 0) {
      char msg[160];
      snprintf(msg, sizeof(msg),
              "fix chem/adas: compute for %s is not per-grid", label);
      error->all(FLERR, msg);
    };
if (c->size_per_grid_cols == 0) {
  if (S.col != 1) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "fix chem/adas: compute column for %s must be 1 for vector source", label);
    error->all(FLERR, msg);
  }
} else if (S.col < 1 || S.col > c->size_per_grid_cols) {
  char msg[160];
  snprintf(msg, sizeof(msg),
           "fix chem/adas: compute column for %s out of range", label);
  error->all(FLERR, msg);
}
};

if (srcTe.kind == SRC_COMP || srcNe.kind == SRC_COMP) {
  bind_compute(srcTe, "Te");
  bind_compute(srcNe, "ne");
}

// initialize SPARTA RNG (same pattern as fix_coll_nanbu)
if (!rng_adas) {
  rng_adas = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  rng_adas->reset(seed, comm->me, 100);
}

// TALLY_BATCH_FIX: resolve the emit-fix ID to a modify->fix index so we can
// query its cumulative ntotal at tally time. We only need the index; the fix
// pointer is looked up per-step via modify->fix[idx] (the index is stable
// for the lifetime of the run once both fixes are defined).
if (tally_units == TALLY_BATCH_FIX) {
  batch_fix_idx = modify->find_fix(batch_fix_id);
  if (batch_fix_idx < 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "fix chem/adas: units batch_fix source fix '%s' not found",
             batch_fix_id);
    error->all(FLERR, msg);
  }
}

// Mode A: resolve source_species names to species indices.
// Runs here (not ctor) because `species` commands are parsed before `fix`
// in the usual input deck but find_species must be callable either way.
source_species.clear();
for (int k = 0; k < nsrc_species; k++) {
  int sp = particle->find_species(src_species_names[k]);
  if (sp < 0) {
    char msg[160];
    snprintf(msg, sizeof(msg),
             "fix chem/adas: source_species '%s' not found",
             src_species_names[k]);
    error->all(FLERR, msg);
  }
  source_species.push_back(sp);
}
if (stop_on_exhaust && source_species.empty() && eirene_mode && comm->me == 0) {
  error->warning(FLERR,
    "fix chem/adas: stop_on_exhaust requested without source_species; "
    "run will rely on SPARTA's built-in nglobal==0 termination");
}

// cache dynamic_cast for per-particle plasma interpolation
cp_plasma_cached_ = nullptr;
if (srcTe.kind == SRC_COMP && srcTe.icompute >= 0) {
  Compute *c = modify->compute[srcTe.icompute];
  cp_plasma_cached_ = dynamic_cast<ComputePlasmaFields *>(c);
}
if (!cp_plasma_cached_ && srcNe.kind == SRC_COMP && srcNe.icompute >= 0) {
  Compute *c = modify->compute[srcNe.icompute];
  cp_plasma_cached_ = dynamic_cast<ComputePlasmaFields *>(c);
}

}

/* ---------------------------------------------------------------------- */

void FixChemAdas::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  // One-time check: if no explicit Te/ne source, require plasma cache
  if (srcTe.kind == SRC_NONE && srcNe.kind == SRC_NONE &&
      !update->plasma_cache_flag) {
    error->all(FLERR,
      "fix chem/adas: no plasma source — either pass 'plasma <Te> <Ne>' "
      "or configure sheath/GCA so the per-particle plasma cache is active");
  }

  nreact_one = 0;
  if (!particle->sorted) particle->sort();
  end_of_step_no_average();
  nreact_running += nreact_one;
}

/* ----------------------------------------------------------------------
   end-of-run summary: cumulative per-type reaction tally, mirrors the
   surf_react tally block printed at the end of each `run`.
---------------------------------------------------------------------- */

void FixChemAdas::post_run()
{
  std::vector<bigint> local_tally(nlist, 0), global_tally(nlist, 0);
  for (int i = 0; i < nlist; i++) local_tally[i] = tally_reactions[i];
  MPI_Allreduce(local_tally.data(), global_tally.data(), nlist,
                MPI_SPARTA_BIGINT, MPI_SUM, world);

  if (comm->me != 0) return;

  bigint nI = 0, nR = 0, nE = 0, nD = 0, nall = 0;
  for (int i = 0; i < nlist; i++) {
    nall += global_tally[i];
    if (rlist[i].type == IONIZATION) nI += global_tally[i];
    else if (rlist[i].type == RECOMBINATION) nR += global_tally[i];
    else if (rlist[i].type == EXCHANGE) nE += global_tally[i];
    else if (rlist[i].type == DISSOCIATION) nD += global_tally[i];
  }

  auto print_block = [&](FILE *fp) {
    if (!fp) return;
    fprintf(fp,
      "Chem/ADAS reaction tallies:\n"
      "  id %s Z=%d #-of-reactions %d\n"
      "    reaction all: " BIGINT_FORMAT "\n"
      "    ioniz: " BIGINT_FORMAT "\n"
      "    recomb: " BIGINT_FORMAT "\n"
      "    CX: " BIGINT_FORMAT "\n"
      "    dissoc: " BIGINT_FORMAT "\n",
      id, atomic_number, nlist, nall, nI, nR, nE, nD);
  };
  print_block(screen);
  print_block(logfile);
}


/* ----------------------------------------------------------------------
   current thermal temperature is calculated on a per-cell basis
---------------------------------------------------------------------- */

void FixChemAdas::end_of_step_no_average()
{
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  deferred_particles.clear();

  // TALLY_BATCH_FIX: pull current cumulative N from the paired emit fix.
  // One allreduce per chem nevery step (compute_vector does the reduce
  // internally). Cached N_batch stays constant within this call so every
  // event tallied in attempt() uses the same weight.
  if (tally_units == TALLY_BATCH_FIX && batch_fix_idx >= 0) {
    Fix *fe = modify->fix[batch_fix_idx];
    double n_global = fe->compute_vector(1);   // FixEmit ntotal (global sum)
    batch_N_cached = static_cast<bigint>(n_global);
  }

  // Ensure per-cell source-tally array (20 columns) tracks current grid size.
  // In TALLY_COUNTS mode this grows but never zeros across steps (cumulative).
  // In TALLY_RATE mode we additionally zero the whole buffer each call so the
  // subsequent accumulate + normalize yields an instantaneous rate over the
  // current `nevery` window.
  if (grid->maxlocal > maxgrid_src) {
    const int oldmax = maxgrid_src;
    maxgrid_src = grid->maxlocal;
    memory->grow(array_grid, maxgrid_src, 20, "chem/adas:array_grid(src)");
    // zero the freshly-added rows (count mode relies on this for fresh cells)
    if (maxgrid_src > oldmax) {
      memset(&array_grid[oldmax][0], 0,
             sizeof(double) * (maxgrid_src - oldmax) * 20);
    }
  }
  if (tally_units == TALLY_RATE && maxgrid_src > 0) {
    memset(&array_grid[0][0], 0, sizeof(double) * maxgrid_src * 20);
  }

  // Fast path: read Te/ne/Ti/vpar/B from per-particle plasma cache
  if (update->plasma_cache_flag &&
      update->pc_te_custom >= 0 && update->pc_ne_custom >= 0) {
    double *te_vec = particle->edvec[particle->ewhich[update->pc_te_custom]];
    double *ne_vec = particle->edvec[particle->ewhich[update->pc_ne_custom]];
    double *ti_vec = (update->pc_ti_custom >= 0 && particle->ewhich[update->pc_ti_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_ti_custom]] : nullptr;
    double *vpar_vec = (update->pc_vpar_custom >= 0 && particle->ewhich[update->pc_vpar_custom] >= 0)
                       ? particle->edvec[particle->ewhich[update->pc_vpar_custom]] : nullptr;
    double *bx_vec = (update->pc_bx_custom >= 0 && particle->ewhich[update->pc_bx_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_bx_custom]] : nullptr;
    double *by_vec = (update->pc_by_custom >= 0 && particle->ewhich[update->pc_by_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_by_custom]] : nullptr;
    double *bz_vec = (update->pc_bz_custom >= 0 && particle->ewhich[update->pc_bz_custom] >= 0)
                     ? particle->edvec[particle->ewhich[update->pc_bz_custom]] : nullptr;

    for (int icell = 0; icell < nglocal; icell++) {
      if (cinfo[icell].count == 0) continue;
      int ip = cinfo[icell].first;
      while (ip >= 0) {
        const double Te_eV = std::max(te_vec[ip], 1e-6);
        const double ne_m3 = std::max(ne_vec[ip], 0.0);
        const double Ti_eV = ti_vec ? std::max(ti_vec[ip], 0.0) : 0.0;
        const double vp = vpar_vec ? vpar_vec[ip] : 0.0;
        const double Bx = bx_vec ? bx_vec[ip] : 0.0;
        const double By = by_vec ? by_vec[ip] : 0.0;
        const double Bz = bz_vec ? bz_vec[ip] : 0.0;
        attempt(&particles[ip], ip, Te_eV, ne_m3, Ti_eV, vp, Bx, By, Bz);
        ip = next[ip];
      }
    }
  } else {
    // Fallback: per-cell values from grid variable or compute
    if (use_grid_plasma) compute_plasma_grid();
    refresh_compute_src(srcTe);
    refresh_compute_src(srcNe);

    for (int icell = 0; icell < nglocal; icell++) {
      if (cinfo[icell].count == 0) continue;
      const double Te_eV = std::max(read_cell(srcTe, icell, 0), 1e-6);
      const double ne_m3 = std::max(read_cell(srcNe, icell, 1), 0.0);
      int ip = cinfo[icell].first;
      while (ip >= 0) {
        attempt(&particles[ip], ip, Te_eV, ne_m3);
        ip = next[ip];
      }
    }
  }

  // Rate-mode normalization. Tally was zeroed at the start of this call and
  // just accumulated raw {count, m*v, 0.5 m v^2} over `nevery` steps. Convert
  // each column to a volumetric rate in SI units:
  //    S_n [m^-3 s^-1]    = count * fnum / (vol * window)
  //    S_m [kg m^-2 s^-2] = sum(m v) * fnum / (vol * window)
  //    S_E [W m^-3]       = sum(0.5 m v^2) * fnum / (vol * window)
  // Cells with zero flow volume (outside the domain / fully covered by solid)
  // are set to 0 to avoid divide-by-zero rather than NaN so downstream
  // `fix ave/grid` can aggregate without masking.
  if (tally_units == TALLY_RATE && maxgrid_src > 0) {
    const double window_s = nevery * update->dt;
    const double fnum     = update->fnum;
    if (window_s > 0.0 && fnum > 0.0) {
      const double num = fnum / window_s;
      for (int icell = 0; icell < nglocal; icell++) {
        const double vol = cinfo[icell].volume;
        if (vol <= 0.0) {
          double *row = array_grid[icell];
          for (int c = 0; c < 20; c++) row[c] = 0.0;
          continue;
        }
        const double scale = num / vol;
        double *row = array_grid[icell];
        for (int c = 0; c < 20; c++) row[c] *= scale;
      }
    }
  }

  // Mode A: remove particles flagged for deletion on ionization.
  // Do this BEFORE spawning deferred (dissociation) products so the new
  // particles get stable indices at the end of the array.
  if (!dellist.empty()) {
    particle->compress_reactions(static_cast<int>(dellist.size()), dellist.data());
    particle->sorted = 0;   // linked list is now stale; next use will re-sort
    dellist.clear();
  }

  // Create deferred particles from dissociation reactions
  for (size_t i = 0; i < deferred_particles.size(); i++) {
    DeferredParticle &dp = deferred_particles[i];
    int id = MAXSMALLINT * rng_adas->uniform();
    particle->add_particle(id, dp.species, dp.icell,
                           dp.x, dp.v, 0.0, 0.0);
  }

  // Mode A: optional stop-when-exhausted across the global source-species pool.
  // Only required when non-source-species particles remain (e.g. impurities);
  // the pure-neutral case is already handled by Update::run's nglobal==0 break.
  if (eirene_mode && stop_on_exhaust && !source_species.empty()) {
    bigint alive_local = 0;
    Particle::OnePart *pp = particle->particles;
    const int np = particle->nlocal;
    for (int i = 0; i < np; i++) {
      const int sp = pp[i].ispecies;
      for (size_t k = 0; k < source_species.size(); k++) {
        if (sp == source_species[k]) { alive_local++; break; }
      }
    }
    bigint alive_global = 0;
    MPI_Allreduce(&alive_local, &alive_global, 1, MPI_SPARTA_BIGINT,
                  MPI_SUM, world);
    // Arm the check only after the batch has ramped above the threshold --
    // otherwise we'd trigger early-exit at step 1 while the emit fix is
    // still filling the initial population.
    if (!exhaust_armed && alive_global > exhaust_threshold) exhaust_armed = 1;

    if (exhaust_armed && alive_global <= exhaust_threshold) {
      // Flag picked up by Update::run on the next iteration; triggers a
      // clean break after final output is written. See update.cpp.
      // Default threshold 0 = wait until population is fully drained.
      // Nonzero threshold = skip the slow fat tail for large speedups with
      // <1% impact on rate moments.
      update->early_exit_requested = 1;
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double FixChemAdas::memory_usage()
{
  double bytes = 0.0;
  bytes += maxgrid*3 * sizeof(double);    // vcom
  return bytes;
}

/* ----------------------------------------------------------------------
   attempt a reaction for a single particle
------------------------------------------------------------------------- */

int FixChemAdas::attempt(Particle::OnePart *ip, int ip_index,
                         double Te_eV, double ne_m3,
                         double Ti_eV, double vpar, double bx, double by, double bz)
{
  Particle::Species *species = particle->species;

  const int isp0 = ip->ispecies;
  if (reactions[isp0].n == 0) return 0;

  const int icell = ip->icell;
  if (icell < 0 || icell >= grid->nlocal) return 0;

  if (Te_eV <= 0.0 || ne_m3 <= 0.0) return 0;

  const double logTe    = std::log10(Te_eV);
  const double logne_cm = std::log10(std::max(ne_m3 * 1e-6, 1e-99));

  int isp = ip->ispecies;
  const int n = reactions[isp].n;
  if (n == 0) return 0;

  // Poisson competing-channel selection:
  //   λ_i = k_i * ne * dt_chem   for each channel i
  //   λ_total = Σ λ_i
  //   P(any event) = 1 - exp(-λ_total)
  //   channel selected proportional to λ_i / λ_total

  const double dt_chem = nevery * update->dt;

  double lambda[16];   // per-channel Poisson rates (max 16 channels)
  int    ridx_map[16]; // reaction index for each channel
  int    nchan = 0;
  double lambda_total = 0.0;

  for (int i = 0; i < n && nchan < 16; ++i) {
    const int ridx = reactions[isp].list[i];
    OneReaction *r = &rlist[ridx];

    const size_t q = static_cast<size_t>(std::max(0.0, species[isp].charge));

    double rate_log10_cm3s = -INFINITY;

    if (r->type == IONIZATION) {
      if (q >= static_cast<size_t>(atomic_number)) continue;
      interpolateRateData(atomic_number, q,   icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::Ionization);
    } else if (r->type == RECOMBINATION) {
      if (q == 0) continue;
      interpolateRateData(atomic_number, q-1, icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::Recombination);
    } else if (r->type == EXCHANGE) {
      if (q == 0) continue;
      interpolateRateData(atomic_number, q-1, icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::ChargeExchange);
    } else if (r->type == DISSOCIATION && r->style == JANEV) {
      // Janev polynomial: ln<sv> = sum_n b_n (ln Te)^n, Te in eV
      const double lnT = std::log(Te_eV);
      double lnsv = r->coeff[0];
      double lnTn = 1.0;
      for (int k = 1; k < r->ncoeff; k++) {
        lnTn *= lnT;
        lnsv += r->coeff[k] * lnTn;
      }
      // Convert from ln(cm3/s) to log10(cm3/s)
      rate_log10_cm3s = lnsv / 2.302585092994046;
    } else {
      continue;
    }

    if (!std::isfinite(rate_log10_cm3s)) continue;

    const double lam = computeReactionLambda(rate_log10_cm3s, dt_chem, ne_m3);
    if (lam <= 0.0) continue;

    lambda[nchan] = lam;
    ridx_map[nchan] = ridx;
    lambda_total += lam;
    nchan++;
  }

  if (nchan == 0 || lambda_total <= 0.0) return 0;

  // P(at least one event in dt_chem) = 1 - exp(-λ_total)
  const double P_any = -std::expm1(-lambda_total);
  const double u = rng_adas->uniform();
  if (u > P_any) return 0;

  // select channel proportional to λ_i / λ_total
  int chosen = 0;
  if (nchan > 1) {
    const double v = rng_adas->uniform() * lambda_total;
    double cumsum = 0.0;
    for (int i = 0; i < nchan; i++) {
      cumsum += lambda[i];
      if (v <= cumsum) { chosen = i; break; }
    }
  }

  const int best_idx = ridx_map[chosen];
  OneReaction *rchosen = &rlist[best_idx];
  tally_reactions[best_idx]++;
  nreact_one++;

  // Per-cell source-term tally for Gkeyll / external coupling.
  // Quantity-major layout:
  //   col  0.. 3 = count  (ioniz, recomb, CX, dissoc)
  //   col  4.. 7 = sum(m*vx)
  //   col  8..11 = sum(m*vy)
  //   col 12..15 = sum(m*vz)
  //   col 16..19 = sum(0.5*m*|v|^2)   [energy in Joules]
  // Use REACTANT species mass and pre-reaction velocity -- this represents the
  // sink of the neutral phase-space density at the event. Gkeyll applies its
  // own <sigma v> times n_e to the neutral moments, so this output is most
  // useful as a cross-check / direct coupling fallback.
  if (array_grid && icell >= 0 && icell < maxgrid_src) {
    int rtype_off = -1;
    switch (rchosen->type) {
      case IONIZATION:     rtype_off = 0; break;
      case RECOMBINATION:  rtype_off = 1; break;
      case EXCHANGE:       rtype_off = 2; break;
      case DISSOCIATION:   rtype_off = 3; break;
    }
    if (rtype_off >= 0) {
      const double m = particle->species[isp0].mass;       // kg
      const double vx0 = ip->v[0];
      const double vy0 = ip->v[1];
      const double vz0 = ip->v[2];
      const double ke  = 0.5 * m * (vx0*vx0 + vy0*vy0 + vz0*vz0);
      double *row = array_grid[icell];

      // Per-event weight. COUNTS and RATE modes accumulate raw totals; RATE
      // divides by (vol * window / fnum) at the end of end_of_step_no_average.
      // BATCH mode folds the trajectory weight and cell volume in here so
      // each event contributes its steady-state source-rate increment directly.
      double scale = 1.0;
      if (tally_units == TALLY_BATCH) {
        const double vol = grid->cinfo[icell].volume;
        const double w   = batch_R_puff / static_cast<double>(batch_N);
        scale = (vol > 0.0) ? (w / vol) : 0.0;
      } else if (tally_units == TALLY_BATCH_FIX && batch_N_cached > 0) {
        const double vol = grid->cinfo[icell].volume;
        const double w   = batch_R_puff / static_cast<double>(batch_N_cached);
        scale = (vol > 0.0) ? (w / vol) : 0.0;
      }

      row[rtype_off]         += scale;
      row[4  + rtype_off]    += m * vx0 * scale;
      row[8  + rtype_off]    += m * vy0 * scale;
      row[12 + rtype_off]    += m * vz0 * scale;
      row[16 + rtype_off]    += ke        * scale;
    }
  }

  // Mode A (EIRENE semantics): neutral is consumed on ionization.
  // Defer deletion -- we are iterating the cell's linked list via Particle::next,
  // so actually removing ip now would invalidate that traversal. The caller
  // compresses the particle array after the cell loop completes.
  // NB: only IONIZATION terminates; CX/DISSOCIATION still run their species
  // swap + velocity re-sampling below so neutral-neutral chains keep working.
  if (eirene_mode && rchosen->type == IONIZATION) {
    dellist.push_back(ip_index);
    return 1;
  }

  // Assign first product
  ip->ispecies = rchosen->products[0];

  // Velocity re-sampling for CX and dissociation products:
  // Sample from shifted Maxwellian at local Ti and bulk flow (EIRENE-like)
  const bool resample = (rchosen->type == EXCHANGE || rchosen->type == DISSOCIATION)
                        && Ti_eV > 0.0;
  if (resample) {
    const double kB = 1.380649e-23;
    const double eV_to_J = 1.602176634e-19;
    const double Ti_K = Ti_eV * eV_to_J / kB;
    const double m_prod = particle->species[rchosen->products[0]].mass;

    // thermal speed: v_th = sqrt(kB * Ti / m)
    const double v_th = (m_prod > 0.0) ? std::sqrt(kB * Ti_K / m_prod) : 0.0;

    // flow velocity vector: v_flow = vpar * b_hat
    const double Bmag = std::sqrt(bx*bx + by*by + bz*bz);
    double vfx = 0.0, vfy = 0.0, vfz = 0.0;
    if (Bmag > 1e-30) {
      const double invB = 1.0 / Bmag;
      vfx = vpar * bx * invB;
      vfy = vpar * by * invB;
      vfz = vpar * bz * invB;
    }

    // Box-Muller Gaussian samples
    const double u1 = std::max(rng_adas->uniform(), 1e-30);
    const double u2 = rng_adas->uniform();
    const double u3 = rng_adas->uniform();
    const double u4 = std::max(rng_adas->uniform(), 1e-30);
    const double g1 = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307 * u2);
    const double g2 = std::sqrt(-2.0 * std::log(u1)) * std::sin(6.283185307 * u2);
    const double g3 = std::sqrt(-2.0 * std::log(u4)) * std::cos(6.283185307 * u3);

    ip->v[0] = vfx + v_th * g1;
    ip->v[1] = vfy + v_th * g2;
    ip->v[2] = vfz + v_th * g3;
  }

  // For dissociation with 2 products: defer creation of second particle
  if (rchosen->nproduct == 2) {
    DeferredParticle dp;
    dp.x[0] = ip->x[0]; dp.x[1] = ip->x[1]; dp.x[2] = ip->x[2];

    if (resample) {
      // Second product also gets independent Maxwellian sample
      const double kB = 1.380649e-23;
      const double eV_to_J = 1.602176634e-19;
      const double Ti_K = Ti_eV * eV_to_J / kB;
      const double m_prod2 = particle->species[rchosen->products[1]].mass;
      const double v_th2 = (m_prod2 > 0.0) ? std::sqrt(kB * Ti_K / m_prod2) : 0.0;
      const double Bmag = std::sqrt(bx*bx + by*by + bz*bz);
      double vfx = 0.0, vfy = 0.0, vfz = 0.0;
      if (Bmag > 1e-30) {
        const double invB = 1.0 / Bmag;
        vfx = vpar * bx * invB;
        vfy = vpar * by * invB;
        vfz = vpar * bz * invB;
      }
      const double u1 = std::max(rng_adas->uniform(), 1e-30);
      const double u2 = rng_adas->uniform();
      const double u3 = rng_adas->uniform();
      const double u4 = std::max(rng_adas->uniform(), 1e-30);
      dp.v[0] = vfx + v_th2 * std::sqrt(-2.0*std::log(u1)) * std::cos(6.283185307*u2);
      dp.v[1] = vfy + v_th2 * std::sqrt(-2.0*std::log(u1)) * std::sin(6.283185307*u2);
      dp.v[2] = vfz + v_th2 * std::sqrt(-2.0*std::log(u4)) * std::cos(6.283185307*u3);
    } else {
      dp.v[0] = ip->v[0]; dp.v[1] = ip->v[1]; dp.v[2] = ip->v[2];
    }

    dp.species = rchosen->products[1];
    dp.icell = ip->icell;
    deferred_particles.push_back(dp);
  }

  return 1;
}

inline void FixChemAdas::compute_plasma_grid() {
  if (!use_grid_plasma) return;
  if (!grid->nlocal)    return;

  const bool need_Te = (srcTe.kind == SRC_VAR);
  const bool need_ne = (srcNe.kind == SRC_VAR);
  if (!need_Te && !need_ne) return;   // both are compute-sourced

  if (grid->maxlocal > maxgrid_plasma) {
    maxgrid_plasma = grid->maxlocal;
    memory->destroy(plasma_cache_2d);
    memory->create(plasma_cache_2d, maxgrid_plasma, 2, "chem/adas:plasma_cache_2d");
  }

  // plasma_cache_2d[icell][0]=Te ; [icell][1]=ne
  if (grid->nlocal) memset(&plasma_cache_2d[0][0], 0, sizeof(double)*grid->nlocal*2);

  const int stride = 2;
  if (need_Te) input->variable->compute_grid(tvar, &plasma_cache_2d[0][0], stride, 0);
  if (need_ne) input->variable->compute_grid(nvar, &plasma_cache_2d[0][1], stride, 0);
}

/* ---------------------------------------------------------------------- */

void FixChemAdas::readfile(char *fname)
{
  int n,n1,n2,eof;
  char line1[MAXLINE],line2[MAXLINE];
  char copy1[MAXLINE],copy2[MAXLINE];
  char *word;
  OneReaction *r;

  // proc 0 opens file

  if (comm->me == 0) {
    fp = fopen(fname,"r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open reaction file %s",fname);
      error->one(FLERR,str);
    }
  }

  // read reactions one at a time and store their info in rlist

  while (1) {
    if (comm->me == 0) eof = readone(line1,line2,n1,n2);
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;

    MPI_Bcast(&n1,1,MPI_INT,0,world);
    MPI_Bcast(&n2,1,MPI_INT,0,world);
    MPI_Bcast(line1,n1,MPI_CHAR,0,world);
    MPI_Bcast(line2,n2,MPI_CHAR,0,world);

    if (nlist == maxlist) {
      maxlist += DELTALIST;
      rlist = (OneReaction *)
        memory->srealloc(rlist,maxlist*sizeof(OneReaction),"react/adas:rlist");
      for (int i = nlist; i < maxlist; i++) {
        r = &rlist[i];
        r->nreactant = r->nproduct = 0;
        r->id_reactants = new char*[MAXREACTANT];
        r->id_products = new char*[MAXPRODUCT];
        r->reactants = new int[MAXREACTANT];
        r->products = new int[MAXPRODUCT];
        r->coeff = new double[MAXCOEFF];
        r->id = NULL;
      }
    }

    strcpy(copy1,line1);
    strcpy(copy2,line2);

    r = &rlist[nlist];
    r->initflag = 0;

    int side = 0;
    int species = 1;

    n = strlen(line1) - 1;
    r->id = new char[n+1];
    strncpy(r->id,line1,n);
    r->id[n] = '\0';

    word = strtok(line1," \t\n\r");

    while (1) {
      if (!word) {
        if (side == 0) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        if (species) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        break;
      }
      if (species) {
        species = 0;
        if (side == 0) {
          if (r->nreactant == MAXREACTANT) {
            print_reaction(copy1,copy2);
            error->all(FLERR,"Too many reactants in a reaction formula");
          }
          n = strlen(word) + 1;
          r->id_reactants[r->nreactant] = new char[n];
          strcpy(r->id_reactants[r->nreactant],word);
          r->nreactant++;
        } else {
          if (r->nreactant == MAXPRODUCT) {
            print_reaction(copy1,copy2);
            error->all(FLERR,"Too many products in a reaction formula");
          }
          n = strlen(word) + 1;
          r->id_products[r->nproduct] = new char[n];
          strcpy(r->id_products[r->nproduct],word);
          r->nproduct++;
        }
      } else {
        species = 1;
        if (strcmp(word,"+") == 0) {
          word = strtok(NULL," \t\n\r");
          continue;
        }
        if (strcmp(word,"-->") != 0) {
          print_reaction(copy1,copy2);
          error->all(FLERR,"Invalid reaction formula in file");
        }
        side = 1;
      }
      word = strtok(NULL," \t\n\r");
    }

    word = strtok(line2," \t\n\r");
    if (!word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction type in file");
    }
    if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    else if (word[0] == 'I' || word[0] == 'i') r->type = IONIZATION;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
    else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    else {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction type in file");
    }

    // check that reactant/product counts are consistent with type

   if (r->type == IONIZATION) {
      if (r->nreactant != 1 || (r->nproduct != 1 && r->nproduct != 1)) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid ionization reaction");
      }
    } else if (r->type == RECOMBINATION) {
      if (r->nreactant != 1 || r->nproduct != 1) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid recombination reaction");
      }
    } else if (r->type == EXCHANGE) {
      if (r->nreactant != 1 || r->nproduct != 1) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid charge exchange reaction");
      }
    } else if (r->type == DISSOCIATION) {
      if (r->nreactant != 1 || (r->nproduct != 1 && r->nproduct != 2)) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid dissociation reaction");
      }
    }

    word = strtok(NULL," \t\n\r");
    if (!word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (word[0] == 'A' || word[0] == 'a') r->style = ADAS;
    else if (word[0] == 'J' || word[0] == 'j') r->style = JANEV;
    else {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (r->style == ADAS) r->ncoeff = 5;
    else if (r->style == JANEV) r->ncoeff = 9;  // b0..b8

    for (int i = 0; i < r->ncoeff; i++) {
      word = strtok(NULL," \t\n\r");
      if (!word) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid reaction coefficients in file");
      }
      r->coeff[i] = input->numeric(FLERR,word);
    }

    word = strtok(NULL," \t\n\r");
    if (word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Too many coefficients in a reaction formula");
    }

    nlist++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   print reaction as read from file
   only proc 0 performs output
------------------------------------------------------------------------- */

void FixChemAdas::print_reaction(char *line1, char *line2)
{
  if (comm->me) return;
  printf("Bad reaction format:\n");
  printf("%s\n%s\n",line1,line2);
};

/* ----------------------------------------------------------------------
   print reaction as stored in rlist
   only proc 0 performs output
------------------------------------------------------------------------- */

void FixChemAdas::print_reaction(OneReaction *r)
{
  if (comm->me) return;
  printf("Bad reaction:\n");
  char type;
  if (r->type == IONIZATION) type = 'I';
  else if (r->type == RECOMBINATION) type = 'R';
  else if (r->type == EXCHANGE) type = 'E';
  else if (r->type == DISSOCIATION) type = 'D';
  else type = '?';

  char style;
  if (r->style == ADAS) style = 'A';
  else if (r->style == JANEV) style = 'J';
  else style = '?';

  if (r->nproduct == 1)
    printf("  %c %c: %s + %s --> %s\n",type,style,
           r->id_reactants[0],r->id_reactants[1],
           r->id_products[0]);
};

/* ----------------------------------------------------------------------
   read one reaction from file
   reaction = 2 lines
   return 1 if end-of-file, else return 0
------------------------------------------------------------------------- */

int FixChemAdas::readone(char *line1, char *line2, int &n1, int &n2)
{
  char *eof;
  while ((eof = fgets(line1,MAXLINE,fp))) {
    size_t pre = strspn(line1," \t\n\r");
    if (pre == strlen(line1) || line1[pre] == '#') continue;
    eof = fgets(line2,MAXLINE,fp);
    if (!eof) break;
    n1 = strlen(line1) + 1;
    n2 = strlen(line2) + 1;
    return 0;
  }

  return 1;
}


/* ----------------------------------------------------------------------
   check for duplicates in list of reactions read from file
   error if any exist
------------------------------------------------------------------------- */

void FixChemAdas::check_duplicate()
{
  OneReaction *r,*s;

  for (int i = 0; i < nlist; i++) {
    r = &rlist[i];

    for (int j = i+1; j < nlist; j++) {
      s = &rlist[j];

      if (r->type != s->type) continue;
      if (r->style != s->style) continue;
      if (r->nreactant != s->nreactant) continue;
      if (r->nproduct != s->nproduct) continue;

      int reactant_match = 0;
      if (strcmp(r->id_reactants[0],s->id_reactants[0]) == 0)
        reactant_match = 1;
      if (!reactant_match) continue;

      int product_match = 0;
      if (r->nproduct == 1) {
        if (strcmp(r->id_products[0],s->id_products[0]) == 0)
          product_match = 1;
      } 
      if (!product_match) continue;

      if (comm->me == 0) {
        printf("MATCH %d %d %d: %d\n",i,j,nlist,product_match);
        printf("MATCH %d %d\n",
               r->products[0],s->products[0]);
      }
      print_reaction(r);
      print_reaction(s);
      error->all(FLERR,"Duplicate reactions in reaction file");
    }
  }
}

/* ----------------------------------------------------------------------
   Compute Poisson rate λ = k * ne * dt for a single reaction channel.
   Returns λ (dimensionless).  Caller sums λ across channels and draws
   from Poisson: P(≥1 event) = 1 - exp(-λ_total).
------------------------------------------------------------------------- */
double FixChemAdas::computeReactionLambda(double rate_log10_cm3s, // log10(k [cm^3/s])
                                          double dt,              // [s]
                                          double ne_m3)           // [m^-3]
{
  if (!(dt > 0.0) || !(ne_m3 > 0.0) || !std::isfinite(rate_log10_cm3s))
    return 0.0;

  // k in cm^3/s -> m^3/s  (exp is ~5x faster than pow(10,x))
  const double k_cm3s = std::exp(rate_log10_cm3s * 2.302585092994046);
  const double k_m3s  = std::max(0.0, k_cm3s) * 1e-6;

  // λ = k [m^3/s] * n_e [m^-3] * dt [s]
  double lambda = k_m3s * ne_m3 * dt;

  if (!std::isfinite(lambda)) return 50.0;
  return std::min(lambda, 50.0);
}



/*----------------------------------------------------------------------
   Read ADAS data from HDF5 file
-------------------------------------------------------------------------*/
void FixChemAdas::readRateData(const std::string& filePath, RateData& rd) {
  try {
      H5::H5File file(filePath, H5F_ACC_RDONLY);

      // Read 1D dataset
      auto read1D = [&file](const std::string& name) {
          H5::DataSet ds = file.openDataSet(name);
          H5::DataSpace space = ds.getSpace();
          hsize_t dims[1];
          space.getSimpleExtentDims(dims, nullptr);
          std::vector<double> data(dims[0]);
          ds.read(data.data(), H5::PredType::NATIVE_DOUBLE);
          return data;
      };

      // Read 3D dataset as flat contiguous array
      auto readFlat3D = [&file](const std::string& name,
                                std::vector<double>& out,
                                int &d0, int &d1, int &d2) {
          H5::DataSet ds = file.openDataSet(name);
          H5::DataSpace space = ds.getSpace();
          hsize_t dims[3];
          space.getSimpleExtentDims(dims, nullptr);
          d0 = static_cast<int>(dims[0]);
          d1 = static_cast<int>(dims[1]);
          d2 = static_cast<int>(dims[2]);
          out.resize(d0 * d1 * d2);
          ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
      };

      readFlat3D("IonizationRateCoeff", rd.ion_coeff,
                 rd.ion_nQ, rd.ion_nT, rd.ion_nD);
      readFlat3D("RecombinationRateCoeff", rd.rec_coeff,
                 rd.rec_nQ, rd.rec_nT, rd.rec_nD);

      rd.Atomic_Number = read1D("Atomic_Number");
      rd.gridD_ion     = read1D("gridDensity_Ionization");
      rd.gridD_rec     = read1D("gridDensity_Recombination");
      rd.gridT_ion     = read1D("gridTemperature_Ionization");
      rd.gridT_rec     = read1D("gridTemperature_Recombination");

      // CX data is optional (backward compat with old HDF5 files)
      rd.cx_nQ = rd.cx_nT = rd.cx_nD = 0;
      if (H5Lexists(file.getId(), "ChargeExchangeRateCoeff", H5P_DEFAULT) > 0) {
        readFlat3D("ChargeExchangeRateCoeff", rd.cx_coeff,
                   rd.cx_nQ, rd.cx_nT, rd.cx_nD);
        rd.gridD_cx = read1D("gridDensity_ChargeExchange");
        rd.gridT_cx = read1D("gridTemperature_ChargeExchange");
      }

  } catch (const H5::Exception& e) {
      throw std::runtime_error("Error reading ADAS file " + filePath + ": " + std::string(e.getCDetailMsg()));
  }
}



void FixChemAdas::interpolateRateData(int atomic_number, double charge, int /*icell*/, double te, double ne, double& rate_final, ReactionType reactionType) {

  size_t charge_idx = static_cast<size_t>(charge);

  double x0, x1, y0, y1;
  double f00, f01, f10, f11;
  bool success = setupInterpolation(reactionType, atomic_number, charge_idx, te, ne, x0, x1, y0, y1, f00, f01, f10, f11);
  if (!success) {
      rate_final = 0.0;
      return;
  }

  MathExtra::bilinearInterpolate(x0, x1, y0, y1, f00, f01, f10, f11, te, ne, rate_final);
}


/*----------------------------------------------------------------------
   find indices for bilinear interpolation
------------------------------------------------------------------------- */

bool FixChemAdas::setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx, double te, double ne, double& x0, double& x1, double& y0, double& y1, double& f00, double& f01, double& f10, double& f11) {
  auto& rd = materials_rate_data[atomic_number];

  auto bracket_index = [](const std::vector<double>& grid, double x,
                          size_t &ilo, size_t &ihi) -> bool {
    const size_t n = grid.size();
    if (n < 2) return false;
    if (x <= grid[0])   { ilo = 0; ihi = 1; return true; }
    if (x >= grid[n-1]) { ihi = n-1; ilo = ihi-1; return true; }
    const size_t hi = static_cast<size_t>(
        std::lower_bound(grid.begin(), grid.end(), x) - grid.begin());
    if (hi == 0 || hi >= n) return false;
    ihi = hi;
    ilo = hi - 1;
    return true;
  };

  size_t tlo, thi, nlo, nhi;

  if (reactionType == ReactionType::Recombination) {
      if (static_cast<int>(charge_idx) >= rd.rec_nQ) return false;
      if (!bracket_index(rd.gridT_rec, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_rec, ne, nlo, nhi)) return false;

      x0 = rd.gridT_rec[tlo];  x1 = rd.gridT_rec[thi];
      y0 = rd.gridD_rec[nlo];  y1 = rd.gridD_rec[nhi];

      // flat layout: [charge][temperature][density]
      f00 = rd.rec_at(charge_idx, tlo, nlo);
      f01 = rd.rec_at(charge_idx, tlo, nhi);
      f10 = rd.rec_at(charge_idx, thi, nlo);
      f11 = rd.rec_at(charge_idx, thi, nhi);

  } else if (reactionType == ReactionType::ChargeExchange) {
      if (static_cast<int>(charge_idx) >= rd.cx_nQ) return false;
      if (!bracket_index(rd.gridT_cx, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_cx, ne, nlo, nhi)) return false;

      x0 = rd.gridT_cx[tlo];  x1 = rd.gridT_cx[thi];
      y0 = rd.gridD_cx[nlo];  y1 = rd.gridD_cx[nhi];

      f00 = rd.cx_at(charge_idx, tlo, nlo);
      f01 = rd.cx_at(charge_idx, tlo, nhi);
      f10 = rd.cx_at(charge_idx, thi, nlo);
      f11 = rd.cx_at(charge_idx, thi, nhi);

  } else {
      if (static_cast<int>(charge_idx) >= rd.ion_nQ) return false;
      if (!bracket_index(rd.gridT_ion, te, tlo, thi)) return false;
      if (!bracket_index(rd.gridD_ion, ne, nlo, nhi)) return false;

      x0 = rd.gridT_ion[tlo];  x1 = rd.gridT_ion[thi];
      y0 = rd.gridD_ion[nlo];  y1 = rd.gridD_ion[nhi];

      f00 = rd.ion_at(charge_idx, tlo, nlo);
      f01 = rd.ion_at(charge_idx, tlo, nhi);
      f10 = rd.ion_at(charge_idx, thi, nlo);
      f11 = rd.ion_at(charge_idx, thi, nhi);
  }

  return true;
}


void FixChemAdas::readRateDataParallel(const std::string& filePath, RateData& rateData) {
  int me = comm->me;

  std::vector<char> filePathBuffer;

  if (me == 0) {
      filePathBuffer.assign(filePath.begin(), filePath.end());
  }

  // First broadcast the file path length and string
  size_t pathLength = filePathBuffer.size();
  MPI_Bcast(&pathLength, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
  if (me != 0) filePathBuffer.resize(pathLength);
  MPI_Bcast(filePathBuffer.data(), pathLength, MPI_CHAR, 0, world);

  // Convert back to string
  std::string broadcastedPath(filePathBuffer.begin(), filePathBuffer.end());

  // Only rank 0 reads HDF5 file
  if (me == 0) {
      readRateData(broadcastedPath, rateData);
  }

  // Now broadcast all datasets
  broadcastRateData(rateData);
}

void FixChemAdas::broadcastRateData(RateData& rd) {

  // Helper: broadcast a flat vector (single MPI_Bcast for the whole buffer)
  auto bcast1D = [this](std::vector<double>& vec) {
      size_t n = vec.size();
      MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      if (comm->me != 0) vec.resize(n);
      if (n > 0) MPI_Bcast(vec.data(), n, MPI_DOUBLE, 0, world);
  };

  // Broadcast dimensions then flat data (one bcast per table, not per row)
  MPI_Bcast(&rd.ion_nQ, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.ion_nT, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.ion_nD, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.rec_nQ, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.rec_nT, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.rec_nD, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.cx_nQ, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.cx_nT, 1, MPI_INT, 0, world);
  MPI_Bcast(&rd.cx_nD, 1, MPI_INT, 0, world);

  bcast1D(rd.ion_coeff);
  bcast1D(rd.rec_coeff);
  bcast1D(rd.cx_coeff);
  bcast1D(rd.Atomic_Number);
  bcast1D(rd.gridD_ion);
  bcast1D(rd.gridD_rec);
  bcast1D(rd.gridD_cx);
  bcast1D(rd.gridT_ion);
  bcast1D(rd.gridT_rec);
  bcast1D(rd.gridT_cx);
}


double FixChemAdas::read_cell(const GridSrc &S, int icell, int var_col)
{
  if (S.kind == SRC_COMP) {
    if (S.src_index < 0) {
      return S.vec_cache ? S.vec_cache[icell] : 0.0;
    }
    if (!S.arr_cache) return 0.0;
    return S.arr_cache[icell][S.src_index];
  }
  // VAR path
  return plasma_cache_2d ? plasma_cache_2d[icell][var_col] : 0.0;
}

void FixChemAdas::refresh_compute_src(GridSrc &S) {
  if (S.kind != SRC_COMP) return;
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (!(c->invoked_flag & INVOKED_PER_GRID)) {
    c->compute_per_grid();
    c->invoked_flag |= INVOKED_PER_GRID;
  }

  S.arr_cache = c->array_grid;
  S.vec_cache = c->vector_grid;
  S.src_index = (c->size_per_grid_cols == 0) ? -1 : (S.col - 1);
  S.cache_ts  = update->ntimestep;
}
