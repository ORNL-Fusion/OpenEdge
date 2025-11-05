/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "math.h"
#include "surf_react_prob.h"
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

// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <numeric>
// #include <iostream>

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION};
enum{SIMPLE};
const char *reaction_type_names[] = {"DISSOCIATION", "EXCHANGE", "RECOMBINATION"};

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXCOEFF 2

#define MAXLINE 1024
#define DELTALIST 16

/* ---------------------------------------------------------------------- */

SurfReactProb::SurfReactProb(SPARTA *sparta, int narg, char **arg) :
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

SurfReactProb::~SurfReactProb()
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

void SurfReactProb::init()
{
  SurfReact::init();
  init_reactions();
}

/* ----------------------------------------------------------------------
   select surface reaction to perform for particle with ptr IP on surface
   return which reaction 1 to N, 0 = no reaction
   if dissociation, add particle and return ptr JP
------------------------------------------------------------------------- */

int SurfReactProb::react(Particle::OnePart *&ip, int, double *, Particle::OnePart *&jp, int &)
{
  Particle::Species *species = particle->species;

  // --- Global, fixed target material (OK per your model) ---
  const double target_Z = update->target_material_charge;
  const double target_m = update->target_material_mass;

  // --- If no reaction templates exist for this species, exit early ---
  const int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;
  int *list = reactions[ip->ispecies].list;

  // We'll use the FIRST reaction entry only as a template for sputtered species.
  OneReaction *r = &rlist[list[0]];

  // --- Local state at impact point (coordinates only used for plasma interp) ---
  const double R_pos = std::max(ip->x[0], 1e-8);
  const double Z_pos = ip->x[1];

  // --- Projectile properties ---
  const double Zi   = species[ip->ispecies].charge;   // charge state
  const double mi   = species[ip->ispecies].molwt;    // mass (units expected by your coeff models)

  // --- Plasma-based incident energy model (your requested form) ---
  const auto Pd   = update->interpolatePlasma_RZ_clamped(R_pos, Z_pos, update->plasma_data);
  const double Ti = Pd.temp_i;
  const double Te = Pd.temp_e;
  const double E_inc_eV = Zi*Ti + 3.0*Te;             // your sheath-based incident "energy"

  // --- Incidence angle θ (radians). TODO: compute from surface normal & velocity ---
  // If/when you have n_hat at the hit point:
  // double n_hat[3]; get_surface_normal(ip->x, n_hat);
  // double vmag = sqrt(ip->v[0]*ip->v[0] + ip->v[1]*ip->v[1] + ip->v[2]*ip->v[2]);
  // double cos_theta = (vmag>0.0) ? - (ip->v[0]*n_hat[0] + ip->v[1]*n_hat[1] + ip->v[2]*n_hat[2]) / vmag : 1.0;
  // double theta = acos(std::clamp(cos_theta, -1.0, 1.0));
  const double theta = 0.0; // TEMP: normal incidence

  // --- Surface models (clamp to physical ranges) ---
  double R = thomas_reflection(/*Zi=*/Zi, /*Mi=*/mi, target_Z, target_m, E_inc_eV /*, theta if you have an angular overload*/);
  R = std::max(0.0, std::min(1.0, R));

  double Y = yamamura_yield_normal(/*Zi=*/Zi, /*Mi=*/mi, target_Z, target_m, E_inc_eV /* normal-incidence for now */);
  if (Y < 0.0) Y = 0.0;

  // --- Single event selection for this wall hit ---
  const double u = random->uniform();

  // 1) Reflection branch (mutually exclusive with sputtering, per your request)
  if (u < R) {
    // Reflect parent. If you have a normal, do specular/diffuse; for now, keep velocity as-is or add a TODO.
    // TODO: apply reflection law v' = v - 2 (v·n) n with energy-loss model if desired.
    nsingle++;
    tally_single[list[0]]++;           // count a "reaction" for bookkeeping
    // No species change for reflection unless you model charge-exchange; keep as-is:
    // ip->ispecies = ip->ispecies;
    jp = nullptr;                      // no secondary created on reflection in this model
    return (list[0] + 1);
  }

  // 2) Not reflected: Sputtering attempt via yield Y
  // Sample N ~ Poisson(Y) using floor+fractional Bernoulli (avoids needing a Poisson RNG)
  int N = static_cast<int>(std::floor(Y));
  const double frac = Y - static_cast<double>(N);
  if (random->uniform() < frac) ++N;

  if (N > 0) {
    // Spawn N sputtered target particles; parent is removed
    // Choose sputtered species template:
    // Use r->products[0] if your table maps to target-neutral (common choice)
    const int sput_species = r->products[0];

    double x[3], v[3];
    memcpy(x, ip->x, 3*sizeof(double));

    for (int k = 0; k < N; ++k) {
      // TODO: sample sputtered energy & angle (e.g., Thompson spectrum + cosine angular)
      // TEMP: give them the parent’s current velocity direction with small magnitude
      // to avoid NaNs until proper sampler is added.
      v[0] = ip->v[0]; v[1] = ip->v[1]; v[2] = ip->v[2];

      const int id = static_cast<int>(MAXSMALLINT * random->uniform());
      Particle::OnePart *particles = particle->particles;
      const int reallocflag = particle->add_particle(id, sput_species, ip->icell, x, v, 0.0, 0.0);
      if (reallocflag) ip = particle->particles + (ip - particles);
    }

    // Remove parent (no implant model)
    ip = nullptr;
    nsingle++;
    tally_single[list[0]]++;
    // Return code referencing the reaction template we used
    return (list[0] + 1);
  }

  // 3) Neither reflected nor sputtered: remove parent (absorb/stick), no secondaries
  ip = nullptr;
  nsingle++;
  tally_single[list[0]]++;
  return (list[0] + 1);
}

/* ---------------------------------------------------------------------- */

char *SurfReactProb::reactionID(int m)
{
  return rlist[m].id;
}

/* ---------------------------------------------------------------------- */

double SurfReactProb::reaction_coeff(int m)
{
  return rlist[m].coeff[1];
}

/* ---------------------------------------------------------------------- */

int SurfReactProb::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species,rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int SurfReactProb::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species,rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactProb::init_reactions()
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

void SurfReactProb::readfile(char *fname)
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

int SurfReactProb::readone(char *line1, char *line2, int &n1, int &n2)
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

double SurfReactProb::random_energy_thompson(double ub, double te){
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


/* ----------------------------------------------------------------------
   Thomas reflection coefficient
------------------------------------------------------------------------- */


double SurfReactProb::thomas_reflection(double projectile_Z, double projectile_mass, double target_Z, double target_mass, double energy_eV, double Us) {
      /*'''
     Adapted from:  https://github.com/RustBCA/blob/main/scripts/materials.py
    Wierzbicki-Biersack empirical reflection coefficient (1994); not as widely
        applicable as Thomas et al.

    https://doi.org/10.1080/10420159408221042

    Args:
        ion (dict): a dictionary with the fields Z (atomic number), m (mass)
        target (dict): a dictionary with the fields Z (atomic number), m (mass)
        energy_eV (float): energy in electron-volts

    Returns:
        R (float): reflection coefficient of ion on target with energy_eV
    '''
    */

        int Z1 = static_cast<int>(projectile_Z);
        int Z2 = static_cast<int>(target_Z);
        double M1 = projectile_mass;
        double M2 = target_mass;
        double energy_keV = energy_eV / 1000.0;


       // Thomas-Fermi reduced energy
        double reduced_energy = 32.55 * energy_keV * M2 / ((M1 + M2) * Z1 * Z2 * (pow(Z1, 0.23) + pow(Z2, 0.23)));
        double mu = M2 / M1;
        if (mu < 1) {
            return 0.0;
        }
      // Interpolation parameters
      std::vector<int> mu_ranges = {1, 3, 6, 7, 12, 15, 20};
      std::vector<double> A1 = {0.02129, 0.3680, 0.5173, 0.5173, 0.6192, 0.6192, 0.8250};
      std::vector<double> A2 = {16.39, 2.985, 2.549, 2.549, 20.01, 20.01, 21.41};
      std::vector<double> A3 = {26.39, 7.122, 5.325, 5.325, 8.922, 8.922, 8.606};
      std::vector<double> A4 = {0.9131, 0.5802, 0.5719, 0.5719, 0.6669, 0.6669, 0.6425};
      std::vector<double> A5 = {6.249, 4.211, 1.094, 1.094, 1.864, 1.864, 1.907};
      std::vector<double> A6 = {2.550, 1.597, 1.933, 1.933, 1.899, 1.899, 1.927};

    // Linear interpolation
        double a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
        for (size_t i = 0; i < mu_ranges.size() - 1; ++i) {
            if (mu >= mu_ranges[i] && mu <= mu_ranges[i + 1]) {
                double t = (mu - mu_ranges[i]) / (mu_ranges[i + 1] - mu_ranges[i]);
                a1 = A1[i] * (1 - t) + A1[i + 1] * t;
                a2 = A2[i] * (1 - t) + A2[i + 1] * t;
                a3 = A3[i] * (1 - t) + A3[i + 1] * t;
                a4 = A4[i] * (1 - t) + A4[i + 1] * t;
                a5 = A5[i] * (1 - t) + A5[i + 1] * t;
                a6 = A6[i] * (1 - t) + A6[i + 1] * t;
                break;
            }
        }

// Compute Thomas reflection coefficient
    double reflection_coefficient = a1 * log(a2 * reduced_energy + 2.718) / (1. + a3 * pow(reduced_energy, a4) + a5 * pow(reduced_energy, a6));

    // Check if the reflection coefficient is NaN or negative
    if (std::isnan(reflection_coefficient) || reflection_coefficient < 0) {
        return 0.0;
    }

    // Return the computed reflection coefficient
    return reflection_coefficient;
}

/* ----------------------------------------------------------------------
   find species iD from species name
------------------------------------------------------------------------- */
int SurfReactProb::findSpeciesID( double mass, double charge) {
    Particle::Species *species = particle->species;
    int ntypes = particle->nspecies;

    for (int i = 0; i < ntypes; i++) {
        if (species[i].molwt == mass && species[i].charge == charge) {
            return i; // Return the ID of the species with same mass and new charge
        }
    }
    return -1; // Return -1 if no matching species is found
}


/// PMI
double SurfReactProb::thomas_reflection(double Z1, double M1, double Z2, double M2,
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
 double SurfReactProb::yamamura_yield_normal(double z1, double m1, double z2, double m2,
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