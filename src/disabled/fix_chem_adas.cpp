/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
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
#include <random>
#include <cstdlib>  // at top of file for random(), RAND_MAX
#include "compute.h"

namespace fs = std::filesystem;
using namespace SPARTA_NS;
enum{IONIZATION,RECOMBINATION};   // other files
enum{IONIZATIONRATE, RECOMBINATIONRATE};   // other files
enum{ADAS};                               // other react files


#define MAXREACTANT 2
#define MAXPRODUCT 3
#define MAXCOEFF 7               // 5 in file, extra for pre-computation
#define INVOKED_PER_GRID 16
#define MAXLINE 1024
#define DELTALIST 16
/* ---------------------------------------------------------------------- */

FixChemAdas::FixChemAdas(SPARTA *sparta, int narg, char **arg) :
  Fix(sparta, narg, arg)
{
    // fix 22 chem/adas  <nevery> <Z>  <reactions_file>  plasma <TeVar> <NeVar>

  if (narg < 8)     error->all(FLERR,"Illegal fix chem/adas command (need: nevery Z reactions_file plasma TeVar NeVar)");
    nevery = atoi(arg[2]);
    atomic_number = atoi(arg[3]);

    // per-cell array for aveflag = 1 case

    nlist = maxlist = 0;
    rlist = NULL;
    readfile(arg[4]);
    check_duplicate();

    // read ADAS rate data

   if (comm->me == 0) {
        fs::path baseDir = "adas";
        std::string fileNameStr = "ADAS_Rates_" + std::to_string(atomic_number) + ".h5";
        fs::path fileName = fileNameStr;
        fs::path fullPath = baseDir / fileName;
        printf("Reading ADAS data for %d from %s\n", atomic_number, fullPath.string().c_str());
    }
    
    std::string fullPathStr = "adas/ADAS_Rates_" + std::to_string(atomic_number) + ".h5";
    readRateDataParallel(fullPathStr, materials_rate_data[atomic_number]);

    // //
    maxgrid = 0;
    reactions = NULL;
    list_ij = NULL;

    tally_reactions = new bigint[nlist];
    tally_reactions_all = new bigint[nlist];
    tally_flag = 0;

    // 
   // --- REQUIRED plasma grid args ---
  int iarg = 5;
  if (strcmp(arg[iarg], "plasma") != 0 || narg < iarg + 3)
    error->all(FLERR,"fix chem/adas requires: plasma <TeSrc> <NeSrc>");

use_grid_plasma = 1;

// tiny parser for one token (either grid-var name, or c_ID[idx])
auto parse_src = [&](const char *tok, GridSrc &dst, const char *label) {
  if (strncmp(tok,"c_",2)==0) {
    dst.kind = SRC_COMP;

    const char *name = tok + 2;              // after 'c_'
    const char *lb   = strchr(name,'[');
    if (!lb || tok[strlen(tok)-1] != ']')
          {
            char msg[160];
            snprintf(msg, sizeof(msg),
                    "fix chem/adas: bad %s token (use c_id[idx])", label);
            error->all(FLERR, msg);
          }


    const int idlen = lb - name;
    dst.cid = new char[idlen+1];
    strncpy(dst.cid, name, idlen);
    dst.cid[idlen] = '\0';

    dst.col = atoi(lb+1);                    // 1-based
    if (dst.col <= 0) {
      char msg[160];
      snprintf(msg, sizeof(msg),
              "fix chem/adas: %s column must be >=1", label);
      error->all(FLERR, msg);
} 
  } else {
    // treat as grid-variable name (legacy path)
    dst.kind = SRC_VAR;
    int n = strlen(tok) + 1;
    char *copy = new char[n];
    strcpy(copy, tok);
    if (&dst == &srcTe) tstr = copy; else nstr = copy;
  }
};

// parse Te, ne
parse_src(arg[iarg+1], srcTe, "Te");
parse_src(arg[iarg+2], srcNe, "ne");

// keep your existing defaults
tvar = nvar = -1;
maxgrid_plasma = 0;
array_grid = NULL;


    use_grid_plasma = (srcTe.kind == SRC_VAR) || (srcNe.kind == SRC_VAR);

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
  char msg[160];
  snprintf(msg, sizeof(msg),
           "fix chem/adas: compute for %s has no per-grid array", label);
  error->all(FLERR, msg);
}
if (S.col < 1 || S.col > c->size_per_grid_cols) {
  char msg[160];
  snprintf(msg, sizeof(msg),
           "fix chem/adas: compute column for %s out of range", label);
  error->all(FLERR, msg);
}
};

bind_compute(srcTe, "Te");
bind_compute(srcNe, "ne");



}

/* ---------------------------------------------------------------------- */

void FixChemAdas::end_of_step()
{
  if ((update->ntimestep % nevery) != 0) return;

  if (!particle->sorted) particle->sort();
  end_of_step_no_average();
}


/* ----------------------------------------------------------------------
   current thermal temperature is calculated on a per-cell basis
---------------------------------------------------------------------- */

void FixChemAdas::end_of_step_no_average()
{

  if (use_grid_plasma) compute_plasma_grid();
  refresh_compute_src(srcTe);
  refresh_compute_src(srcNe);


  Particle::OnePart *particles = particle->particles;
  Particle::Species *species = particle->species;
  int *next = particle->next;
  Grid::ChildInfo *cinfo = grid->cinfo;
  int nglocal = grid->nlocal;

  int ip;
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
  if (icell < 0 || icell >= grid->nlocal) return 0;  // safety: particle not mapped to local cell


   if (Te_eV <= 0.0 || ne_m3 <= 0.0) return 0;

// printf("FixChemAdas::attempt: icell=%d isp0=%d Te_eV=%g ne_m3=%g\n", icell, isp0, Te_eV, ne_m3);
  const double logTe    = std::log10(Te_eV);
  const double logne_cm = std::log10(std::max(ne_m3 * 1e-6, 1e-99));

  int isp = ip->ispecies;
  const int n = reactions[isp].n;
  if (n == 0) return 0;

  // ---- compute per-channel probability and pick the highest ----
  double best_p   = 0.0;
  int    best_idx = -1;

  for (int i = 0; i < n; ++i) {
    const int ridx = reactions[isp].list[i];
    OneReaction *r = &rlist[ridx];

    // species charge (ensure non-negative integer)
    const size_t q = static_cast<size_t>(std::max(0.0, species[isp].charge));

    double rate_log10_cm3s = -INFINITY;  // ADAS returns log10(k[cm^3/s])

    if (r->type == IONIZATION) {
      if (q >= static_cast<size_t>(atomic_number)) continue; // already fully stripped
      interpolateRateData(atomic_number, q,   icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::Ionization);
    } else if (r->type == RECOMBINATION) {
      if (q == 0) continue; // neutral cannot recombine
      interpolateRateData(atomic_number, q-1, icell, logTe, logne_cm,
                          rate_log10_cm3s, ReactionType::Recombination);
    } else {
      continue;
    }

    if (!std::isfinite(rate_log10_cm3s)) continue;

    const double p = computeReactionProbability(rate_log10_cm3s, update->dt, ne_m3);

    if (p > best_p) { best_p = p; best_idx = ridx; }

  }

  if (best_idx < 0 || best_p <= 0.0) return 0;

  // one reaction max per step
  const double u = static_cast<double>(::random()) / (static_cast<double>(RAND_MAX) + 1.0);

  if (u > best_p) return 0;

  OneReaction *rchosen = &rlist[best_idx];
  tally_reactions[best_idx]++;
  ip->ispecies = rchosen->products[0];
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
    // if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    // else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    if (word[0] == 'I' || word[0] == 'i') r->type = IONIZATION;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
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
      if (r->nreactant != 1 || (r->nproduct != 1 && r->nproduct != 1)) {
        print_reaction(copy1,copy2);
        error->all(FLERR,"Invalid recombination reaction");
      }
    }

    word = strtok(NULL," \t\n\r");
    if (!word) {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (word[0] == 'A' || word[0] == 'a') r->style = ADAS;
    else {
      print_reaction(copy1,copy2);
      error->all(FLERR,"Invalid reaction style in file");
    }
    if (r->style == ADAS) r->ncoeff = 5;

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

  char style;
  if (r->style == ADAS) style = 'A';

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
   compute reaction probability P from rate coefficient k, dt, ne
   k [cm^3/s], dt [s], ne [m^-3]
   P = 1 - exp(-λ) with λ = k [m^3/s] * ne [m^-3] * dt [s]
   use expm1 for good small-λ accuracy
------------------------------------------------------------------------- */
double FixChemAdas::computeReactionProbability(double rate_log10_cm3s, // log10(k [cm^3/s])
                                               double dt,             // [s]
                                               double ne_m3)          // [m^-3]
{
  if (!(dt > 0.0) || !(ne_m3 > 0.0) || !std::isfinite(rate_log10_cm3s))
    return 0.0;

  // k in cm^3/s -> m^3/s
  const double k_cm3s = std::pow(10.0, rate_log10_cm3s);
  const double k_m3s  = std::max(0.0, k_cm3s) * 1e-6;

  // λ = k [m^3/s] * n_e [m^-3] * dt [s]
  double lambda = k_m3s * ne_m3 * dt;

  // numerical safety
  if (!std::isfinite(lambda)) return 1.0;
  lambda = std::min(lambda, 50.0); // exp(-50) ~ 1.9e-22

  // P = 1 - exp(-λ) with good small-λ accuracy
  double P = -std::expm1(-lambda);
  if (P < 0.0) P = 0.0;
  if (P > 1.0) P = 1.0;
  return P;
}



/*----------------------------------------------------------------------
   Read ADAS data from HDF5 file
-------------------------------------------------------------------------*/
void FixChemAdas::readRateData(const std::string& filePath, RateData& rateData) {
  try {
      H5::H5File file(filePath, H5F_ACC_RDONLY);

      // Helper: Convert flat vector to 2D
      auto to2D = [](const std::vector<double>& flat, const std::vector<hsize_t>& dims) {
          std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
          for (hsize_t i = 0; i < dims[0]; ++i)
              for (hsize_t j = 0; j < dims[1]; ++j)
                  data[i][j] = flat[i * dims[1] + j];
          return data;
      };

      // Helper: Convert flat vector to 3D
      auto to3D = [](const std::vector<double>& flat, const std::vector<hsize_t>& dims) {
          std::vector<std::vector<std::vector<double>>> data(dims[0],
              std::vector<std::vector<double>>(dims[1], std::vector<double>(dims[2])));
          for (hsize_t i = 0; i < dims[0]; ++i)
              for (hsize_t j = 0; j < dims[1]; ++j)
                  for (hsize_t k = 0; k < dims[2]; ++k)
                      data[i][j][k] = flat[i * dims[1] * dims[2] + j * dims[2] + k];
          return data;
      };

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

      // Read 2D dataset
      auto read2D = [&file, &to2D](const std::string& name) {
          H5::DataSet ds = file.openDataSet(name);
          H5::DataSpace space = ds.getSpace();
          std::vector<hsize_t> dims(2);
          space.getSimpleExtentDims(dims.data(), nullptr);
          std::vector<double> flat(dims[0] * dims[1]);
          ds.read(flat.data(), H5::PredType::NATIVE_DOUBLE);
          return to2D(flat, dims);
      };

      // Read 3D dataset
      auto read3D = [&file, &to3D](const std::string& name) {
          H5::DataSet ds = file.openDataSet(name);
          H5::DataSpace space = ds.getSpace();
          std::vector<hsize_t> dims(3);
          space.getSimpleExtentDims(dims.data(), nullptr);
          std::vector<double> flat(dims[0] * dims[1] * dims[2]);
          ds.read(flat.data(), H5::PredType::NATIVE_DOUBLE);
          return to3D(flat, dims);
      };

      // Read all datasets into rateData
      rateData.IonizationRateCoeff         = read3D("IonizationRateCoeff");
      rateData.RecombinationRateCoeff      = read3D("RecombinationRateCoeff");
      rateData.gridChargeState_Ionization  = read2D("gridChargeState_Ionization");
      rateData.gridChargeState_Recombination = read2D("gridChargeState_Recombination");
      rateData.Atomic_Number               = read1D("Atomic_Number");
      rateData.gridDensity_Ionization      = read1D("gridDensity_Ionization");
      rateData.gridDensity_Recombination   = read1D("gridDensity_Recombination");
      rateData.gridTemperature_Ionization  = read1D("gridTemperature_Ionization");
      rateData.gridTemperature_Recombination = read1D("gridTemperature_Recombination");

  } catch (const H5::Exception& e) {
      throw std::runtime_error("Error reading ADAS file " + filePath + ": " + std::string(e.getCDetailMsg()));
  }
}



void FixChemAdas::interpolateRateData(int atomic_number, double charge, int icell, double te, double ne, double& rate_final, ReactionType reactionType) {
    
  size_t charge_idx = static_cast<size_t>(charge);

  // if ((reactionType == ReactionType::Recombination && charge_idx == 0) ||
  //     (reactionType == ReactionType::Ionization && charge_idx >= atomic_number)) {
  //     rate_final = 0.0;
  //     return;
  // }

  // Create a cache key
  InterpolationKey key {icell, static_cast<int>(charge), atomic_number, reactionType};
  // Check if we already computed this value
  auto it = rateCache.find(key);
  if (it != rateCache.end()) {
      // Use the cached value
      rate_final = it->second;
      return;
  }



  double x0, x1, y0, y1;
  double f00, f01, f10, f11;
  bool success = setupInterpolation(reactionType, atomic_number, charge_idx, te, ne, x0, x1, y0, y1, f00, f01, f10, f11);
  if (!success) {
      rate_final = 0.0;
      return;
  }

  MathExtra::bilinearInterpolate(x0, x1, y0, y1, f00, f01, f10, f11, te, ne, rate_final);

  // Store the result in the cache
  // rateCache[key] = rate_final;
}


/*----------------------------------------------------------------------
   find indices for bilinear interpolation
------------------------------------------------------------------------- */

bool FixChemAdas::setupInterpolation(ReactionType reactionType, int atomic_number, size_t charge_idx, double te, double ne, double& x0, double& x1, double& y0, double& y1, double& f00, double& f01, double& f10, double& f11) {
  size_t indT, indN;
  auto& material_data = materials_rate_data[atomic_number];

  if (reactionType == ReactionType::Recombination) {
      auto& tempGrid = material_data.gridTemperature_Recombination;
      auto& densGrid = material_data.gridDensity_Recombination;

      indT = std::lower_bound(tempGrid.begin(), tempGrid.end(), te) - tempGrid.begin();
      indN = std::lower_bound(densGrid.begin(), densGrid.end(), ne) - densGrid.begin();

      if (indT >= tempGrid.size()) indT = tempGrid.size() - 1;
      if (indN >= densGrid.size()) indN = densGrid.size() - 1;

      if (indT == 0) indT = 1;
      if (indN == 0) indN = 1;

      x0 = tempGrid[indT - 1];
      x1 = tempGrid[indT];
      y0 = densGrid[indN - 1];
      y1 = densGrid[indN];

      f00 = material_data.RecombinationRateCoeff[charge_idx][indN - 1][indT - 1];
      f01 = material_data.RecombinationRateCoeff[charge_idx][indN][indT - 1];
      f10 = material_data.RecombinationRateCoeff[charge_idx][indN - 1][indT];
      f11 = material_data.RecombinationRateCoeff[charge_idx][indN][indT];

  } else if (reactionType == ReactionType::Ionization) {
      auto& tempGrid = material_data.gridTemperature_Ionization;
      auto& densGrid = material_data.gridDensity_Ionization;

      indT = std::lower_bound(tempGrid.begin(), tempGrid.end(), te) - tempGrid.begin();
      indN = std::lower_bound(densGrid.begin(), densGrid.end(), ne) - densGrid.begin();

      if (indT >= tempGrid.size()) indT = tempGrid.size() - 1;
      if (indN >= densGrid.size()) indN = densGrid.size() - 1;

      if (indT == 0) indT = 1;
      if (indN == 0) indN = 1;

      x0 = tempGrid[indT - 1];
      x1 = tempGrid[indT];
      y0 = densGrid[indN - 1];
      y1 = densGrid[indN];

      f00 = material_data.IonizationRateCoeff[charge_idx][indN - 1][indT - 1];
      f01 = material_data.IonizationRateCoeff[charge_idx][indN][indT - 1];
      f10 = material_data.IonizationRateCoeff[charge_idx][indN - 1][indT];
      f11 = material_data.IonizationRateCoeff[charge_idx][indN][indT];
  } else {
      error->all(FLERR, "Illegal ReactionType");
      return false;
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

void FixChemAdas::broadcastRateData(RateData& rateData) {
  int me = comm->me;

  // Helper lambda: broadcast 1D vector
  auto bcast1D = [this](std::vector<double>& vec) {
      size_t size = vec.size();
      MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      if (comm->me != 0) vec.resize(size);
      MPI_Bcast(vec.data(), size, MPI_DOUBLE, 0, world);
  };

  // Helper lambda: broadcast 2D vector
  auto bcast2D = [this](std::vector<std::vector<double>>& vec) {
      size_t dim1 = vec.size();
      size_t dim2 = dim1 ? vec[0].size() : 0;
      MPI_Bcast(&dim1, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      MPI_Bcast(&dim2, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      if (comm->me != 0) vec.resize(dim1, std::vector<double>(dim2));
      for (size_t i = 0; i < dim1; ++i)
          MPI_Bcast(vec[i].data(), dim2, MPI_DOUBLE, 0, world);
  };

  // Helper lambda: broadcast 3D vector
  auto bcast3D = [this](std::vector<std::vector<std::vector<double>>>& vec) {
      size_t dim1 = vec.size();
      size_t dim2 = dim1 ? vec[0].size() : 0;
      size_t dim3 = (dim2 && dim1) ? vec[0][0].size() : 0;
      MPI_Bcast(&dim1, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      MPI_Bcast(&dim2, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      MPI_Bcast(&dim3, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
      if (comm->me != 0)
          vec.resize(dim1, std::vector<std::vector<double>>(dim2, std::vector<double>(dim3)));
      for (size_t i = 0; i < dim1; ++i)
          for (size_t j = 0; j < dim2; ++j)
              MPI_Bcast(vec[i][j].data(), dim3, MPI_DOUBLE, 0, world);
  };

  // Broadcast all RateData members
  bcast3D(rateData.IonizationRateCoeff);
  bcast3D(rateData.RecombinationRateCoeff);
  bcast2D(rateData.gridChargeState_Ionization);
  bcast2D(rateData.gridChargeState_Recombination);
  bcast1D(rateData.Atomic_Number);
  bcast1D(rateData.gridDensity_Ionization);
  bcast1D(rateData.gridDensity_Recombination);
  bcast1D(rateData.gridTemperature_Ionization);
  bcast1D(rateData.gridTemperature_Recombination);
}


inline double FixChemAdas::fetch_compute_cell_value(const GridSrc& S, int icell)
{
  // S.kind guaranteed == SRC_COMP
  Compute *c = modify->compute[S.icompute];

  // Ensure it’s computed on this step
  if (c->invoked_per_grid != update->ntimestep) {
    c->compute_per_grid();
  }

  // Ask the compute where the data live and which column to read
  double **arr = nullptr; 
  int *cols = nullptr;
  // S.col is 1-based “c_ID[S.col]”
  int nmap = c->query_tally_grid(S.col, arr, cols);   // returns >=1 if OK
  // For simple 1:1 columns, cols[0] == (S.col-1)
  const int src = cols ? cols[0] : (S.col - 1);

  return arr[icell][src];
}

    inline void FixChemAdas::refresh_compute_src(GridSrc &S) {
  if (S.kind != SRC_COMP) return;                     // now visible
  if (S.cache_ts == update->ntimestep) return;

  Compute *c = modify->compute[S.icompute];
  if (c->invoked_per_grid != update->ntimestep) c->compute_per_grid();

  double **arr = nullptr; 
  int *cols = nullptr;
  const int nmap = c->query_tally_grid(S.col, arr, cols);

  S.arr_cache = (nmap > 0) ? arr : nullptr;
  S.src_index = (nmap > 0) ? (cols ? cols[0] : (S.col - 1)) : -1;
  S.cache_ts  = update->ntimestep;
}
