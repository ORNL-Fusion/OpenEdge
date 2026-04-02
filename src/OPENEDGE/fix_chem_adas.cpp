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
  }

  tvar = nvar = -1;
  maxgrid_plasma = 0;
  array_grid = NULL;

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
memory->destroy(array_grid);
delete rng_adas;



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
    memory->destroy(array_grid);
    memory->create(array_grid, maxgrid_plasma, 2, "chem/adas:array_grid");
  }
  if (grid->nlocal)
    memset(&array_grid[0][0], 0, sizeof(double)*grid->nlocal*2);
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

  // periodic per-type reaction tally (every 10000 steps)
  if (comm->me == 0 && update->ntimestep % 10000 == 0 && nreact_running > 0) {
    bigint nI = 0, nR = 0, nE = 0, nD = 0;
    for (int i = 0; i < nlist; i++) {
      if (rlist[i].type == IONIZATION) nI += tally_reactions[i];
      else if (rlist[i].type == RECOMBINATION) nR += tally_reactions[i];
      else if (rlist[i].type == EXCHANGE) nE += tally_reactions[i];
      else if (rlist[i].type == DISSOCIATION) nD += tally_reactions[i];
    }
    if (screen) fprintf(screen,
      "  chem/adas step " BIGINT_FORMAT ": ioniz=" BIGINT_FORMAT
      " recomb=" BIGINT_FORMAT " CX=" BIGINT_FORMAT
      " dissoc=" BIGINT_FORMAT "\n",
      update->ntimestep, nI, nR, nE, nD);
  }
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

  // Fast path: read Te/ne from per-particle plasma cache (populated by update)
  if (update->plasma_cache_flag &&
      update->pc_te_custom >= 0 && update->pc_ne_custom >= 0) {
    double *te_vec = particle->edvec[particle->ewhich[update->pc_te_custom]];
    double *ne_vec = particle->edvec[particle->ewhich[update->pc_ne_custom]];

    for (int icell = 0; icell < nglocal; icell++) {
      if (cinfo[icell].count == 0) continue;
      int ip = cinfo[icell].first;
      while (ip >= 0) {
        const double Te_eV = std::max(te_vec[ip], 1e-6);
        const double ne_m3 = std::max(ne_vec[ip], 0.0);
        attempt(&particles[ip], Te_eV, ne_m3);
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
        attempt(&particles[ip], Te_eV, ne_m3);
        ip = next[ip];
      }
    }
  }

  // Create deferred particles from dissociation reactions
  for (size_t i = 0; i < deferred_particles.size(); i++) {
    DeferredParticle &dp = deferred_particles[i];
    int id = MAXSMALLINT * rng_adas->uniform();
    particle->add_particle(id, dp.species, dp.icell,
                           dp.x, dp.v, 0.0, 0.0);
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

int FixChemAdas::attempt(Particle::OnePart *ip, double Te_eV, double ne_m3)
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

  // Assign first product
  ip->ispecies = rchosen->products[0];

  // For dissociation with 2 products: defer creation of second particle
  if (rchosen->nproduct == 2) {
    DeferredParticle dp;
    dp.x[0] = ip->x[0]; dp.x[1] = ip->x[1]; dp.x[2] = ip->x[2];
    dp.v[0] = ip->v[0]; dp.v[1] = ip->v[1]; dp.v[2] = ip->v[2];
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
    memory->destroy(array_grid);
    memory->create(array_grid, maxgrid_plasma, 2, "chem/adas:array_grid");
  }

  // array_grid[icell][0]=Te ; [icell][1]=ne
  if (grid->nlocal) memset(&array_grid[0][0], 0, sizeof(double)*grid->nlocal*2);

  const int stride = 2;
  if (need_Te) input->variable->compute_grid(tvar, &array_grid[0][0], stride, 0);
  if (need_ne) input->variable->compute_grid(nvar, &array_grid[0][1], stride, 0);
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
  return array_grid ? array_grid[icell][var_col] : 0.0;
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
