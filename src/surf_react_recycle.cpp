/* ----------------------------------------------------------------------
    OpenEdge: Surface recycling reaction model for neutral transport.
    Incoming ions are neutralized and re-emitted with cosine angular
    distribution at a specified energy (Franck-Condon or thermal).

    Reaction file format:
      D+ --> D
      E 1.0 3.0
      D+ --> D + D
      D 1.0 3.0 0.025

    Line 1: reactant --> product(s)
    Line 2: type prob energy1 [energy2]
      type: E=exchange (1->1), D=dissociation (1->2), R=recombination (absorb)
      prob: recycling probability (0 to 1)
      energy1: return energy for first product [eV]
      energy2: return energy for second product [eV] (dissociation only)

    Velocity is sampled from cosine angular distribution relative to
    surface normal, with magnitude set by the specified energy.

    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#include "math.h"
#include "string.h"
#include "surf_react_recycle.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "particle.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION};

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXLINE 1024
#define DELTALIST 16
#define MY_2PI 6.283185307179586

/* ---------------------------------------------------------------------- */

SurfReactRecycle::SurfReactRecycle(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal surf_react recycle command");

  nlist_recycle = maxlist_recycle = 0;
  rlist = NULL;
  reactions = NULL;
  indices = NULL;

  readfile(arg[2]);

  nsingle = ntotal = 0;
  nlist = nlist_recycle;
  tally_single = new int[nlist];
  tally_total = new int[nlist];
  tally_single_all = new int[nlist];
  tally_total_all = new int[nlist];

  random = new RanKnuth(update->ranmaster->uniform());
}

/* ---------------------------------------------------------------------- */

SurfReactRecycle::~SurfReactRecycle()
{
  if (rlist) {
    for (int i = 0; i < maxlist_recycle; i++) {
      OneReaction *r = &rlist[i];
      for (int j = 0; j < r->nreactant; j++) delete [] r->id_reactants[j];
      for (int j = 0; j < r->nproduct; j++) delete [] r->id_products[j];
      delete [] r->id_reactants;
      delete [] r->id_products;
      delete [] r->reactants;
      delete [] r->products;
      delete [] r->energy;
      delete [] r->id;
    }
    memory->sfree(rlist);
  }
  memory->destroy(reactions);
  memory->destroy(indices);
  delete random;
}

/* ---------------------------------------------------------------------- */

void SurfReactRecycle::init()
{
  SurfReact::init();
  init_reactions();
}

/* ---------------------------------------------------------------------- */

int SurfReactRecycle::react(Particle::OnePart *&ip, int, double *norm,
                            Particle::OnePart *&jp, int &)
{
  int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;

  int *list = reactions[ip->ispecies].list;

  double react_prob = 0.0;
  double random_prob = random->uniform();

  OneReaction *r;

  for (int i = 0; i < n; i++) {
    r = &rlist[list[i]];
    react_prob += r->prob;

    if (react_prob > random_prob) {
      nsingle++;
      tally_single[list[i]]++;

      switch (r->type) {
      case DISSOCIATION:
        {
          ip->ispecies = r->products[0];
          double mass0 = particle->species[r->products[0]].mass;
          sample_cosine_velocity(ip->v, norm, r->energy[0], mass0);

          int id = MAXSMALLINT * random->uniform();
          double x[3], v[3];
          memcpy(x, ip->x, 3*sizeof(double));
          double mass1 = particle->species[r->products[1]].mass;
          sample_cosine_velocity(v, norm, r->energy[1], mass1);
          Particle::OnePart *particles = particle->particles;
          int reallocflag =
            particle->add_particle(id, r->products[1], ip->icell, x, v, 0.0, 0.0);
          if (reallocflag) ip = particle->particles + (ip - particles);
          jp = &particle->particles[particle->nlocal-1];
          return (list[i] + 1);
        }
      case EXCHANGE:
        {
          ip->ispecies = r->products[0];
          double mass0 = particle->species[r->products[0]].mass;
          sample_cosine_velocity(ip->v, norm, r->energy[0], mass0);
          return (list[i] + 1);
        }
      case RECOMBINATION:
        {
          ip = NULL;
          return (list[i] + 1);
        }
      }
    }
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactRecycle::sample_cosine_velocity(double *v, double *norm,
                                               double energy_eV, double mass)
{
  // speed from kinetic energy: E = 0.5 * m * v^2
  const double eV_to_J = 1.602176634e-19;
  double speed = 0.0;
  if (mass > 0.0 && energy_eV > 0.0)
    speed = sqrt(2.0 * energy_eV * eV_to_J / mass);

  // cosine angular distribution: cos(theta) = sqrt(xi1), phi = 2*pi*xi2
  double xi1 = random->uniform();
  double xi2 = random->uniform();
  double cosTheta = sqrt(xi1);   // cosine distribution: P(theta) ~ cos(theta)
  double sinTheta = sqrt(std::max(0.0, 1.0 - cosTheta*cosTheta));
  double phi = MY_2PI * xi2;
  double cosPhi = cos(phi);
  double sinPhi = sin(phi);

  // build local basis: norm, tangent1, tangent2
  double tangent1[3], tangent2[3];

  // pick a vector not parallel to norm
  double tmp[3] = {0.0, 0.0, 1.0};
  if (fabs(norm[2]) > 0.9) { tmp[0] = 1.0; tmp[2] = 0.0; }
  MathExtra::cross3(norm, tmp, tangent1);
  MathExtra::norm3(tangent1);
  MathExtra::cross3(norm, tangent1, tangent2);

  // outgoing direction
  v[0] = speed * (sinTheta*cosPhi*tangent1[0] + sinTheta*sinPhi*tangent2[0] + cosTheta*norm[0]);
  v[1] = speed * (sinTheta*cosPhi*tangent1[1] + sinTheta*sinPhi*tangent2[1] + cosTheta*norm[1]);
  v[2] = speed * (sinTheta*cosPhi*tangent1[2] + sinTheta*sinPhi*tangent2[2] + cosTheta*norm[2]);
}

/* ---------------------------------------------------------------------- */

char *SurfReactRecycle::reactionID(int m)
{
  return rlist[m].id;
}

double SurfReactRecycle::reaction_coeff(int m)
{
  return rlist[m].prob;
}

int SurfReactRecycle::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species, rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

int SurfReactRecycle::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species, rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactRecycle::init_reactions()
{
  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    r->active = 1;
    for (int i = 0; i < r->nreactant; i++) {
      r->reactants[i] = particle->find_species(r->id_reactants[i]);
      if (r->reactants[i] < 0) { r->active = 0; break; }
    }
    for (int i = 0; i < r->nproduct; i++) {
      r->products[i] = particle->find_species(r->id_products[i]);
      if (r->products[i] < 0) { r->active = 0; break; }
    }
  }

  memory->destroy(reactions);
  int nspecies = particle->nspecies;
  reactions = memory->create(reactions, nspecies, "surf_react_recycle:reactions");

  for (int i = 0; i < nspecies; i++) reactions[i].n = 0;

  int ntot = 0;
  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    reactions[r->reactants[0]].n++;
    ntot++;
  }

  memory->destroy(indices);
  memory->create(indices, ntot > 0 ? ntot : 1, "surf_react_recycle:indices");

  int offset = 0;
  for (int i = 0; i < nspecies; i++) {
    reactions[i].list = &indices[offset];
    offset += reactions[i].n;
  }
  for (int i = 0; i < nspecies; i++) reactions[i].n = 0;

  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int i = r->reactants[0];
    reactions[i].list[reactions[i].n++] = m;
  }
}

/* ---------------------------------------------------------------------- */

void SurfReactRecycle::readfile(char *fname)
{
  int n, n1, n2, eof;
  char line1[MAXLINE], line2[MAXLINE];
  char *word;
  OneReaction *r;

  if (comm->me == 0) {
    fp = fopen(fname, "r");
    if (fp == NULL) {
      char str[128];
      sprintf(str, "Cannot open reaction file %s", fname);
      error->one(FLERR, str);
    }
  }

  while (1) {
    if (comm->me == 0) eof = readone(line1, line2, n1, n2);
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;

    MPI_Bcast(&n1, 1, MPI_INT, 0, world);
    MPI_Bcast(&n2, 1, MPI_INT, 0, world);
    MPI_Bcast(line1, n1, MPI_CHAR, 0, world);
    MPI_Bcast(line2, n2, MPI_CHAR, 0, world);

    if (nlist_recycle == maxlist_recycle) {
      maxlist_recycle += DELTALIST;
      rlist = (OneReaction *)
        memory->srealloc(rlist, maxlist_recycle*sizeof(OneReaction),
                         "surf_react_recycle:rlist");
      for (int i = nlist_recycle; i < maxlist_recycle; i++) {
        r = &rlist[i];
        r->nreactant = r->nproduct = 0;
        r->id_reactants = new char*[MAXREACTANT];
        r->id_products = new char*[MAXPRODUCT];
        r->reactants = new int[MAXREACTANT];
        r->products = new int[MAXPRODUCT];
        r->energy = new double[MAXPRODUCT];
        r->energy[0] = r->energy[1] = 0.0;
        r->id = NULL;
      }
    }

    r = &rlist[nlist_recycle];

    // parse line1: reactant --> product(s)
    int side = 0;
    int species = 1;

    n = strlen(line1) - 1;
    r->id = new char[n+1];
    strncpy(r->id, line1, n);
    r->id[n] = '\0';

    word = strtok(line1, " \t\n");
    while (1) {
      if (!word) {
        if (side == 0) error->all(FLERR, "Invalid reaction formula in recycle file");
        if (species) error->all(FLERR, "Invalid reaction formula in recycle file");
        break;
      }
      if (species) {
        species = 0;
        if (side == 0) {
          if (r->nreactant == MAXREACTANT)
            error->all(FLERR, "Too many reactants in recycle reaction");
          n = strlen(word) + 1;
          r->id_reactants[r->nreactant] = new char[n];
          strcpy(r->id_reactants[r->nreactant], word);
          r->nreactant++;
        } else {
          if (r->nproduct == MAXPRODUCT)
            error->all(FLERR, "Too many products in recycle reaction");
          n = strlen(word) + 1;
          r->id_products[r->nproduct] = new char[n];
          strcpy(r->id_products[r->nproduct], word);
          r->nproduct++;
        }
      } else {
        species = 1;
        if (strcmp(word, "+") == 0) {
          word = strtok(NULL, " \t\n");
          continue;
        }
        if (strcmp(word, "-->") != 0)
          error->all(FLERR, "Invalid reaction formula in recycle file");
        side = 1;
      }
      word = strtok(NULL, " \t\n");
    }

    // handle NULL product (absorption)
    if (r->nproduct == 1 && strcmp(r->id_products[0], "NULL") == 0) {
      delete [] r->id_products[0];
      r->id_products[0] = NULL;
      r->nproduct = 0;
    }

    // parse line2: type prob energy1 [energy2]
    word = strtok(line2, " \t\n");
    if (!word) error->all(FLERR, "Invalid reaction type in recycle file");
    if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
    else error->all(FLERR, "Invalid reaction type in recycle file");

    // validate reactant/product counts
    if (r->type == DISSOCIATION) {
      if (r->nreactant != 1 || r->nproduct != 2)
        error->all(FLERR, "Dissociation recycle reaction needs 1 reactant, 2 products");
    } else if (r->type == EXCHANGE) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR, "Exchange recycle reaction needs 1 reactant, 1 product");
    } else if (r->type == RECOMBINATION) {
      if (r->nreactant != 1 || r->nproduct != 0)
        error->all(FLERR, "Recombination recycle reaction needs 1 reactant, 0 products");
    }

    // probability
    word = strtok(NULL, " \t\n");
    if (!word) error->all(FLERR, "Missing probability in recycle reaction");
    r->prob = input->numeric(FLERR, word);

    // energy for product(s)
    for (int i = 0; i < r->nproduct; i++) {
      word = strtok(NULL, " \t\n");
      if (!word) {
        if (i == 0 && r->nproduct > 0)
          error->all(FLERR, "Missing energy in recycle reaction");
        r->energy[i] = 0.0;
      } else {
        r->energy[i] = input->numeric(FLERR, word);
      }
    }

    nlist_recycle++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int SurfReactRecycle::readone(char *line1, char *line2, int &n1, int &n2)
{
  char *ptr;
  while ((ptr = fgets(line1, MAXLINE, fp))) {
    int pre = strspn(line1, " \t\n\r");
    if (pre == (int)strlen(line1)) continue;
    if (line1[pre] == '#') continue;
    break;
  }
  if (ptr == NULL) return 1;

  while ((ptr = fgets(line2, MAXLINE, fp))) {
    int pre = strspn(line2, " \t\n\r");
    if (pre == (int)strlen(line2)) continue;
    if (line2[pre] == '#') continue;
    break;
  }
  if (ptr == NULL) return 1;

  n1 = strlen(line1) + 1;
  n2 = strlen(line2) + 1;
  return 0;
}
