/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "math.h"
#include "surf_react_bca.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"
#include <map>
#include <vector>
#include <fstream>
#include <string>  
#include <algorithm>
#include <random>
#include <iostream>

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION};
enum{SIMPLE};

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXCOEFF 2

#define MAXLINE 1024
#define DELTALIST 16

/* ---------------------------------------------------------------------- */

SurfReactBca::SurfReactBca(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal surf_react bca command");

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

SurfReactBca::~SurfReactBca()
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

void SurfReactBca::init()
{
  SurfReact::init();
  init_reactions();
}

/* ----------------------------------------------------------------------
   select surface reaction to perform for particle with ptr IP on surface
   return which reaction 1 to N, 0 = no reaction
   if dissociation, add particle and return ptr JP
------------------------------------------------------------------------- */

int SurfReactBca::react(Particle::OnePart *&ip, int, double *,
                         Particle::OnePart *&jp, int &)
{
  Particle::Species *species = particle->species;
  int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;

  int *list = reactions[ip->ispecies].list;

  // probablity to compare to reaction probability

  double react_prob = 0.0;
  double random_prob = random->uniform();
   const int icell = ip->icell;

  // loop over possible reactions for this species
  // if dissociation performs a realloc:
  //   make copy of x,v with new species
  //   rot/vib energies will be reset by SurfCollide
  //   repoint ip to new particles data struct if reallocated

  OneReaction *r;

  for (int i = 0; i < n; i++) {
    r = &rlist[list[i]];

  // Calculate kinetic energy in eV
  const double normVel = std::sqrt(std::inner_product(ip->v, ip->v + 3, ip->v, 0.0));
  const double incident_part_charge = species[ip->ispecies].charge;
  const double R   = std::max(ip->x[0], 1e-8);
  const double Z   = ip->x[1];
  double dens  = update->interpolatePlasma_RZ_clamped(R, Z, update->plasma_data).dens_i;
  double ti  = update->interpolatePlasma_RZ_clamped(R, Z, update->plasma_data).temp_i;
  double te  = update->interpolatePlasma_RZ_clamped(R, Z, update->plasma_data).temp_e;
  double energy = Z * ti + 3 * te;
  printf("R: %f, Z: %f, ne: %e, ti: %f, te: %f, energy: %f\n", R, Z, dens, ti, te, energy);
  const double kinetic_energy_eV = incident_part_charge * ti + 3 * te;

   // Tungsten is the target
   double target_Z = 74;
   double target_m = 183.84;
    double reflectionCoeff = thomas_reflection(Z, species[ip->ispecies].molwt, target_Z, target_m, kinetic_energy_eV);
    double sputteringCoeff = yamamura_yield_normal(Z,species[ip->ispecies].molwt, target_Z, target_m, kinetic_energy_eV);

    double totalCoeff = reflectionCoeff + sputteringCoeff;

        // Check for division by zero
    double reflectionProb = 0.0;
    double sputteringProb = 0.0;

     if (totalCoeff <= 0.0) {
    return 0;
  }

    if (totalCoeff > 0.0) {
        // Normal case: calculate reflection and sputtering probabilities
        reflectionProb = reflectionCoeff / totalCoeff;
        sputteringProb = sputteringCoeff / totalCoeff;
      
    } else {
        // Handle the case where totalCoeff is zero (full reflection)
        reflectionProb = 1.0;  // 100% reflection
        sputteringProb = 0.0;  // 0% sputtering
    }
    if (r->type == 0) // DISSOCIATION
    {
      react_prob += sputteringProb;
    }
    else if (r->type == 1) // EXCHANGE
    {
      react_prob += reflectionProb;
    }
    else {
      error->all(FLERR, "Invalid reaction type");
    }

   if (react_prob > random_prob) {
      nsingle++;
      tally_single[list[i]]++;
      switch (r->type) {
      case DISSOCIATION: // sputtering coefficient
        {
          double x[3],v[3];
          ip->ispecies = r->products[0];
          int id = MAXSMALLINT*random->uniform();
          memcpy(x,ip->x,3*sizeof(double));
          memcpy(v,ip->v,3*sizeof(double));
          Particle::OnePart *particles = particle->particles;
          int reallocflag =
            particle->add_particle(id,r->products[1],ip->icell,x,v,0.0,0.0);
          if (reallocflag) ip = particle->particles + (ip - particles);
          jp = &particle->particles[particle->nlocal-1];
          int nd = list[i] + 1;
          // remove ip from tally
          ip = NULL;
          return (list[i] + 1);
        }
      case EXCHANGE: // reflection coefficient 
        {
          ip->ispecies = r->products[0];
          return (list[i] + 1);
        }
      }
    }
  }
  // no reaction

  return 0;
}

/* ---------------------------------------------------------------------- */

char *SurfReactBca::reactionID(int m)
{
  return rlist[m].id;
}

/* ---------------------------------------------------------------------- */

double SurfReactBca::reaction_coeff(int m)
{
  return rlist[m].coeff[1];
}

/* ---------------------------------------------------------------------- */

int SurfReactBca::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species,rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int SurfReactBca::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species,rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactBca::init_reactions()
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

void SurfReactBca::readfile(char *fname)
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

int SurfReactBca::readone(char *line1, char *line2, int &n1, int &n2)
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


/// PMI
double SurfReactBca::thomas_reflection(double Z1, double M1, double Z2, double M2,
                                double energy_eV)
{

    if (energy_eV <= 0.0) return 0.0;

    // mass ratio mu; fit is defined for mu >= 1
    double mu = M2 / M1;
    mu = std::max(1.0, mu);    // clamp to valid domain lower bound
    mu = std::min(mu, 20.0);   // tables provided up to ~20

    // Thomas–Fermi reduced energy, input E in keV
    const double E_keV = energy_eV * 1e-3;
    const double tf_denom = (M1 + M2) * Z1 * Z2 * (std::pow(Z1, 0.23) + std::pow(Z2, 0.23));
    const double eps_TF = (tf_denom > 0.0) ? 32.55 * E_keV * M2 / tf_denom : 0.0;

    // coefficient tables vs mu (M2/M1)
    static const double MU[7] = { 1, 3, 6, 7, 12, 15, 20 };
    static const double A1[7] = { 0.02129, 0.3680, 0.5173, 0.5173, 0.6192, 0.6192, 0.8250 };
    static const double A2[7] = { 16.39,   2.985,  2.549,  2.549,  20.01,  20.01,  21.41  };
    static const double A3[7] = { 26.39,   7.122,  5.325,  5.325,  8.922,  8.922,  8.606  };
    static const double A4[7] = { 0.9131,  0.5802, 0.5719, 0.5719, 0.6669, 0.6669, 0.6425 };
    static const double A5[7] = { 6.249,   4.211,  1.094,  1.094,  1.864,  1.864,  1.907  };
    static const double A6[7] = { 2.550,   1.597,  1.933,  1.933,  1.899,  1.899,  1.927  };

    const double a1 = interp1d_clamped(MU, A1, 7, mu);
    const double a2 = interp1d_clamped(MU, A2, 7, mu);
    const double a3 = interp1d_clamped(MU, A3, 7, mu);
    const double a4 = interp1d_clamped(MU, A4, 7, mu);
    const double a5 = interp1d_clamped(MU, A5, 7, mu);
    const double a6 = interp1d_clamped(MU, A6, 7, mu);

    const double eul = std::exp(1.0);
    const double num = a1 * std::log(a2 * eps_TF + eul);
    const double den = 1.0 + a3 * std::pow(eps_TF, a4) + a5 * std::pow(eps_TF, a6);
    const double R   = (den > 0.0) ? (num / den) : 0.0;

    return clamp(R, 0.0, 1.0);
}

// ============================================================================
// Yamamura SPUTTERING YIELD (normal incidence)
// Returns Y [atoms/ion]. If E <= Eth, returns 0.
// ============================================================================
 double SurfReactBca::yamamura_yield_normal(double z1, double m1, double z2, double m2,
                                    double energy_eV)
{

    const double Us = 8.9;                   // eV (surface binding)
    const double Q  =0.72;                    // dimensionless Yamamura coefficient

    if (energy_eV <= 0.0 || Us <= 0.0) return 0.0;

    const double msum = m1 + m2;
    const double r1 = m1 / msum;
    const double r2 = m2 / msum;

    // Lindhard reduced energy (Yamamura), with eV input
    const double denom_z = z1 * z2 * std::sqrt(std::pow(z1, 2.0/3.0) + std::pow(z2, 2.0/3.0));
    const double eps = (denom_z > 0.0) ? (0.03255 * r2 * energy_eV / denom_z) : 0.0;

    // Yamamura constants
    const double K = 8.478 * z1 * z2 * r1 / std::sqrt(std::pow(z1, 2.0/3.0) + std::pow(z2, 2.0/3.0));
    const double a_star = 0.08 + 0.164 * std::pow(m2/m1, 0.4) + 0.0145 * std::pow(m2/m1, 1.29);

    // Sputtering threshold energy
    const double Eth = (1.9 + 3.8 * (m1/m2) + 0.134 * std::pow(m2/m1, 1.24)) * Us;
    if (energy_eV <= Eth) return 0.0;

    const double sqrt_eps = std::sqrt(std::max(eps, 0.0));
    const double eul      = std::exp(1.0);

    // LSS nuclear stopping cross-section (Yamamura fit)
    const double num_sn = 3.441 * sqrt_eps * std::log(eps + eul);
    const double den_sn = 1.0 + 6.355 * sqrt_eps + eps * (-1.708 + 6.882 * sqrt_eps);
    const double sn = (den_sn > 0.0) ? (num_sn / den_sn) : 0.0;

    // LSS electronic stopping (simple sqrt-eps scaling)
    const double kfac_num = 0.079 * std::pow(m1 + m2, 1.5) * std::pow(z1, 2.0/3.0) * std::sqrt((double)z2);
    const double kfac_den = std::pow(m1, 1.5) * std::sqrt(m2) * std::pow(std::pow(z1, 2.0/3.0) + std::pow(z2, 2.0/3.0), 0.75);
    const double kfac = (kfac_den > 0.0) ? (kfac_num / kfac_den) : 0.0;
    const double se = kfac * sqrt_eps;

    const double shape = std::pow(std::max(0.0, 1.0 - std::sqrt(Eth / energy_eV)), 2.8);

    // Final Yamamura yield (normal incidence)
    const double Y = 0.42 * a_star * Q * K * sn / Us / (1.0 + 0.35 * Us * se) * shape;

    return std::max(0.0, Y);
}