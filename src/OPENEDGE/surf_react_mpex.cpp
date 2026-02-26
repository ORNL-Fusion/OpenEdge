/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "math.h"
#include "surf_react_mpex.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include <random>
#include "math_extra.h"
#include "memory.h"
#include "error.h"
#include <map>
#include <vector>
#include <fstream>
#include <string>  
#include <algorithm>
#include <iostream>
#include "math_const.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION};
enum{SIMPLE};
// const char *reaction_type_names[] = {"DISSOCIATION", "EXCHANGE", "RECOMBINATION"};

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXCOEFF 2

#define MAXLINE 1024
#define DELTALIST 16

/* ---------------------------------------------------------------------- */

SurfReactMpex::SurfReactMpex(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal surf_react prob command");

  // initialize reaction data structs

  nlist_prob = maxlist_prob = 0;
  rlist = NULL;
  reactions = NULL;
  indices = NULL;

  // read reaction file

  readfile(arg[2]);

  // setup the reaction tallies

  nsingle = ntotal = 0;

  nlist = nlist_prob;
  tally_single = new int[nlist];
  tally_total = new int[nlist];
  tally_single_all = new int[nlist];
  tally_total_all = new int[nlist];

  size_vector = 2 + 2*nlist;

  // initialize RNG

  random = new RanKnuth(update->ranmaster->uniform());
  double seed = update->ranmaster->uniform();
  random->reset(seed,comm->me,100);
}

/* ---------------------------------------------------------------------- */

SurfReactMpex::~SurfReactMpex()
{
  if (copy) return;

  delete random;

  if (rlist) {
    for (int i = 0; i < maxlist_prob; i++) {
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
    memory->destroy(rlist);
  }

  memory->destroy(reactions);
  memory->destroy(indices);
}

/* ---------------------------------------------------------------------- */

void SurfReactMpex::init()
{
  SurfReact::init();
  init_reactions();
}

/* ----------------------------------------------------------------------
   select surface reaction to perform for particle with ptr IP on surface
   return which reaction 1 to N, 0 = no reaction
   if dissociation, add particle and return ptr JP
------------------------------------------------------------------------- */

int SurfReactMpex::react(Particle::OnePart *&ip,
                         int isurf,
                         double *norm,
                         Particle::OnePart *&jp,
                         int &velreset)
{
  Particle::Species *species = particle->species;

  // no secondaries in this model
  jp = nullptr;
  // let SurfCollideDiffuse::diffuse() modify the direction if we reflect
  velreset = 0;

  // If no reaction templates exist for this species, do nothing
  const int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;
  int *list = reactions[ip->ispecies].list;
  OneReaction *r = &rlist[list[0]];  // used only for tally index

  // --- Incidence angle θ relative to surface normal (0° = normal, 90° = grazing) ---
  double theta_deg = 0.0;
  double *v = ip->v;
  double vlen = MathExtra::len3(v);
  double nlen = MathExtra::len3(norm);

  if (vlen > 0.0 && nlen > 0.0) {
    // incoming direction is -v (into surface), normal is 'norm' (out of surface)
    double dot = MathExtra::dot3(v, norm);
    double cos_theta = -dot / (vlen * nlen);   // minus sign: use -v
    cos_theta = clamp(cos_theta, -1.0, 1.0);
    double theta_rad = acos(cos_theta);
    theta_deg = theta_rad * 180.0 / MathConst::MY_PI;
  }
  // --- Reflection coefficients RN(θ), RE(θ) from MPEX brief ---
  // θ is in degrees; formulas use θπ/180 inside the tanh.
  double argN = 2.09 * theta_deg * MathConst::MY_PI / 180.0 - 2.63;
  double argE = 2.7  * theta_deg * MathConst::MY_PI / 180.0 - 4.12;

  double RN = 0.71 + 0.7  * tanh(argN);  // particle reflection probability
  double RE = 0.93 + 0.93 * tanh(argE);  // energy reflection fraction

  RN = clamp(RN, 0.0, 1.0);
  RE = clamp(RE, 0.0, 1.0);

  // --- Single event selection: reflect vs stick ---
  double u = random->uniform();

  if (u < RN) {
    // REFLECTION: keep ip alive; SurfCollideDiffuse::diffuse will
    // randomize direction, but we can enforce the energy loss via RE.
    if (RE > 0.0) {
      double scale = sqrt(RE);      // v' = sqrt(RE) * v  => E' = RE * E
      ip->v[0] *= scale;
      ip->v[1] *= scale;
      ip->v[2] *= scale;
    }
 ip->ispecies = r->products[0];
    nsingle++;
    tally_single[list[0]]++;        // bookkeeping

    return list[0] + 1;
  }

  velreset = 0;
  ip = nullptr;
  nsingle++;
  tally_single[list[0]]++;
  return list[0] + 1;
}

/* ---------------------------------------------------------------------- */

char *SurfReactMpex::reactionID(int m)
{
  return rlist[m].id;
}

/* ---------------------------------------------------------------------- */

double SurfReactMpex::reaction_coeff(int m)
{
  return rlist[m].coeff[1];
}

/* ---------------------------------------------------------------------- */

int SurfReactMpex::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species,rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int SurfReactMpex::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species,rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactMpex::init_reactions()
{
  // convert species IDs to species indices
  // flag reactions as active/inactive depending on whether all species exist

  for (int m = 0; m < nlist_prob; m++) {
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

  // count possible reactions for each species

  memory->destroy(reactions);
  int nspecies = particle->nspecies;
  reactions = memory->create(reactions,nspecies,
                             "surf_react:reactions");

  for (int i = 0; i < nspecies; i++) reactions[i].n = 0;

  int n = 0;
  for (int m = 0; m < nlist_prob; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].n++;
    n++;
  }

  // allocate indices = entire list of reactions for all I species

  memory->destroy(indices);
  memory->create(indices,n,"surf_react:indices");

  // reactions[i].list = offset into full indices vector

  int offset = 0;
  for (int i = 0; i < nspecies; i++) {
    reactions[i].list = &indices[offset];
    offset += reactions[i].n;
  }

  // reactions[i].list = indices of possible reactions for each species

  for (int i = 0; i < nspecies; i++) reactions[i].n = 0;

  for (int m = 0; m < nlist_prob; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].list[reactions[i].n++] = m;
  }

  // check that summed reaction probabilities for each species <= 1.0

  double sum;
  for (int i = 0; i < nspecies; i++) {
    sum = 0.0;
    for (int j = 0; j < reactions[i].n; j++)
      sum += rlist[reactions[i].list[j]].coeff[0];
    if (sum > 1.0)
      error->all(FLERR,"Surface reaction probability for a species > 1.0");
  }
}

/* ---------------------------------------------------------------------- */

void SurfReactMpex::readfile(char *fname)
{
  int n,n1,n2,eof;
  char line1[MAXLINE],line2[MAXLINE];
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

    if (nlist_prob == maxlist_prob) {
      maxlist_prob += DELTALIST;
      rlist = (OneReaction *)
        memory->srealloc(rlist,maxlist_prob*sizeof(OneReaction),
                         "surf_react:rlist");
      for (int i = nlist_prob; i < maxlist_prob; i++) {
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

    r = &rlist[nlist_prob];

    int side = 0;
    int species = 1;

    n = strlen(line1) - 1;
    r->id = new char[n+1];
    strncpy(r->id,line1,n);
    r->id[n] = '\0';

    word = strtok(line1," \t\n");

    while (1) {
      if (!word) {
        if (side == 0) error->all(FLERR,"Invalid reaction formula in file");
        if (species) error->all(FLERR,"Invalid reaction formula in file");
        break;
      }
      if (species) {
        species = 0;
        if (side == 0) {
          if (r->nreactant == MAXREACTANT)
            error->all(FLERR,"Too many reactants in a reaction formula");
          n = strlen(word) + 1;
          r->id_reactants[r->nreactant] = new char[n];
          strcpy(r->id_reactants[r->nreactant],word);
          r->nreactant++;
        } else {
          if (r->nreactant == MAXPRODUCT)
            error->all(FLERR,"Too many products in a reaction formula");
          n = strlen(word) + 1;
          r->id_products[r->nproduct] = new char[n];
          strcpy(r->id_products[r->nproduct],word);
          r->nproduct++;
        }
      } else {
        species = 1;
        if (strcmp(word,"+") == 0) {
          word = strtok(NULL," \t\n");
          continue;
        }
        if (strcmp(word,"-->") != 0)
          error->all(FLERR,"Invalid reaction formula in file");
        side = 1;
      }
      word = strtok(NULL," \t\n");
    }

    // replace single NULL product with no products

    if (r->nproduct == 1 && strcmp(r->id_products[0],"NULL") == 0) {
      delete [] r->id_products[0];
      r->id_products[0] = NULL;
      r->nproduct = 0;
    }

    word = strtok(line2," \t\n");
    if (!word) error->all(FLERR,"Invalid reaction type in file");
    if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
    else error->all(FLERR,"Invalid reaction type in file");

    // check that reactant/product counts are consistent with type

    if (r->type == DISSOCIATION) {
      // if (r->nreactant != 1 || r->nproduct != 1)
      if (r->nreactant != 1 || r->nproduct != 2)
        error->all(FLERR,"Invalid dissociation reaction");
    } else if (r->type == EXCHANGE) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR,"Invalid exchange reaction");
    } else if (r->type == RECOMBINATION) {
      if (r->nreactant != 1 || r->nproduct != 0)
        error->all(FLERR,"Invalid recombination reaction");
    }

    word = strtok(NULL," \t\n");
    if (!word) error->all(FLERR,"Invalid reaction style in file");
    if (word[0] == 'S' || word[0] == 's') r->style = SIMPLE;
    else error->all(FLERR,"Invalid reaction style in file");

    if (r->style == SIMPLE) r->ncoeff = 2;

    for (int i = 0; i < r->ncoeff; i++) {
      word = strtok(NULL," \t\n");

      // second coeff is optional

      if (!word) {
        if (i == 0) error->all(FLERR,"Invalid reaction coefficients in file");
        else r->coeff[i] = 0.0;
      } else {
        r->coeff[i] = input->numeric(FLERR,word);
      }
    }

    word = strtok(NULL," \t\n");
    if (word) error->all(FLERR,"Too many coefficients in a reaction formula");

    nlist_prob++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   read one reaction from file
   reaction = 2 lines
   return 1 if end-of-file, else return 0
------------------------------------------------------------------------- */

int SurfReactMpex::readone(char *line1, char *line2, int &n1, int &n2)
{
  char *eof;
  while ((eof = fgets(line1,MAXLINE,fp))) {
    size_t pre = strspn(line1," \t\n");
    if (pre == strlen(line1) || line1[pre] == '#') continue;
    eof = fgets(line2,MAXLINE,fp);
    if (!eof) break;
    n1 = strlen(line1) + 1;
    n2 = strlen(line2) + 1;
    return 0;
  }

  return 1;
}

// PMI
/* ----------------------------------------------------------------------
   create thompson scattering data
------------------------------------------------------------------------- */

double SurfReactMpex::random_energy_thompson(double ub, double te){
    /*
    Generate a random number 'e' from a Thompson distribution function F(e, ub, emax).
    The distribution is given by F(e, ub, emax) = const * e / (e + ub)**3, for 0 < e < emax.
    The constant is calculated as const = ub / (0.5 * 1./(emax/ub+1.)**2 - 1./(emax/ub+1.) + 0.5).
    
    :param ub: The parameter UB in the distribution.
    :param emax: The maximum value of e (EMAX).
    :return: A random number 'e' from the Thompson distribution.
    */
    double emax = 3.0 * te ;  
    double emu = 1.0 / (emax / ub + 1.0);
    double betad2 = 1.0 / (emu * emu - emu - emu + 1.0);

    // generate random number
    double r = random->uniform();
    double arg = r / betad2;

    double energy_sample = ub / ( 1.0 - std::sqrt(arg)) - ub;

    return energy_sample;

}
