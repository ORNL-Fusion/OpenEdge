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
#include "surf_react_surface_pwi.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "particle.h"
#include "modify.h"
#include "surf.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "database_paths.h"
#include "process_library.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"

#include <stdexcept>
#include "H5Cpp.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION,TRIM_REFLECT,ABSORB_REEMIT};
enum{INT,DOUBLE};                        // match surf.cpp custom-attribute type codes

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXLINE 1024
#define DELTALIST 16
#define MY_2PI 6.283185307179586

/* ---------------------------------------------------------------------- */

SurfReactSurfacePWI::SurfReactSurfacePWI(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal surf_react surface/pwi command");

  nlist_recycle = maxlist_recycle = 0;
  rlist = NULL;
  reactions = NULL;
  indices = NULL;
  twall = -1.0;
  twall_attr = NULL;
  tindex_custom = -1;
  R_attr = NULL;
  rindex_custom = -1;
  trim_dir.clear();

  // parse keyword options BEFORE reading reactions file, so that the
  // reaction parser can resolve `T <table_name>` entries against
  // trim_dir immediately.
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"twall") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      twall = input->numeric(FLERR,arg[iarg+1]);
      if (twall <= 0.0) error->all(FLERR,"surf_react recycle twall must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg],"twall_surf") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      int n = strlen(arg[iarg+1]) + 1;
      twall_attr = new char[n];
      strcpy(twall_attr, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"trim_dir") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      trim_dir = arg[iarg+1];
      iarg += 2;
    } else if (strcmp(arg[iarg],"R_surf") == 0) {
      // per-surface recycling coefficient. When set, the R value of any
      // A-type reaction is overridden by the per-surf attribute lookup.
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      int n = strlen(arg[iarg+1]) + 1;
      R_attr = new char[n];
      strcpy(R_attr, arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal surf_react surface/pwi command");
  }

  if (twall > 0.0 && twall_attr)
    error->all(FLERR,"surf_react surface/pwi: cannot set both twall and twall_surf");

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

SurfReactSurfacePWI::~SurfReactSurfacePWI()
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
  delete [] twall_attr;
  delete [] R_attr;
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWI::init()
{
  SurfReact::init();
  init_reactions();

  // bind per-surf twall attribute if requested. Deferred to init() so the
  // attribute can be created by read_surf (via custom columns) or by
  // fix surf/temp before this runs.
  if (twall_attr) {
    tindex_custom = surf->find_custom(twall_attr);
    if (tindex_custom < 0) {
      char str[256];
      sprintf(str, "surf_react surface/pwi twall_surf: custom attribute '%s' not found", twall_attr);
      error->all(FLERR, str);
    }
    if (surf->etype[tindex_custom] != DOUBLE)
      error->all(FLERR,"surf_react surface/pwi twall_surf attribute must be DOUBLE");
    if (surf->esize[tindex_custom] != 0)
      error->all(FLERR,"surf_react surface/pwi twall_surf attribute must be scalar per-surf");
    // ensure local+ghost copy is in sync; safe in serial, needed when distributed
    surf->spread_custom(tindex_custom);
  }

  // bind per-surf recycling-coefficient attribute if requested.
  if (R_attr) {
    rindex_custom = surf->find_custom(R_attr);
    if (rindex_custom < 0) {
      char str[256];
      sprintf(str, "surf_react surface/pwi R_surf: custom attribute '%s' not found", R_attr);
      error->all(FLERR, str);
    }
    if (surf->etype[rindex_custom] != DOUBLE)
      error->all(FLERR,"surf_react surface/pwi R_surf attribute must be DOUBLE");
    if (surf->esize[rindex_custom] != 0)
      error->all(FLERR,"surf_react surface/pwi R_surf attribute must be scalar per-surf");
    surf->spread_custom(rindex_custom);
  }
}

/* ---------------------------------------------------------------------- */

int SurfReactSurfacePWI::react(Particle::OnePart *&ip, int isurf, double *norm,
                            Particle::OnePart *&jp, int &velreset)
{
  int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;

  int *list = reactions[ip->ispecies].list;

  double react_prob = 0.0;
  double random_prob = random->uniform();

  // resolve effective wall temperature for this surf:
  // per-surf custom attribute if bound, else scalar twall, else <=0
  // (pass-through / zero-reset).
  double twall_eff = twall;
  if (tindex_custom >= 0)
    twall_eff = surf->edvec_local[tindex_custom][isurf];

  // precompute incident (E, theta) for TRIM channels (needed at most once
  // per event; skip work if no TRIM reactions exist for this species).
  double E_in_eV = 0.0, theta_in_deg = 0.0;
  bool trim_precomputed = false;

  OneReaction *r;

  for (int i = 0; i < n; i++) {
    r = &rlist[list[i]];

    double p_this;
    if (r->type == TRIM_REFLECT) {
      if (!trim_precomputed) {
        double mass_in = particle->species[ip->ispecies].mass;
        double v2 = ip->v[0]*ip->v[0] + ip->v[1]*ip->v[1] + ip->v[2]*ip->v[2];
        if (mass_in > 0.0) {
          double Ein_J = 0.5 * mass_in * v2;
          E_in_eV = Ein_J * update->joule2ev * update->mvv2e;
        }
        double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
        double vlen = sqrt(v2);
        if (nlen > 0.0 && vlen > 0.0) {
          double cos_th = -(ip->v[0]*norm[0] + ip->v[1]*norm[1] + ip->v[2]*norm[2])
                         / (nlen * vlen);
          if (cos_th < 0.0) cos_th = 0.0;
          if (cos_th > 1.0) cos_th = 1.0;
          theta_in_deg = acos(cos_th) * 180.0 / Reflection::PI_CONST;
        }
        trim_precomputed = true;
      }
      Reflection::View tv = trim_tables[r->trim_table].view();
      p_this = Reflection::R_N_interp(tv, E_in_eV, theta_in_deg);
    } else {
      p_this = r->prob;
    }

    react_prob += p_this;

    if (react_prob > random_prob) {
      nsingle++;
      tally_single[list[i]]++;
      velreset = 1;   // react() sets the outgoing velocity; don't let collide override it

      switch (r->type) {
      case DISSOCIATION:
        {
          int sp0 = r->products[0];
          int sp1 = r->products[1];
          // dissociation fragments: KE absorbs the bond energy,
          // internal rovib of products is reset (or sampled at twall).
          double erot0 = (twall_eff > 0.0) ? particle->erot(sp0, twall_eff, random) : 0.0;
          double evib0 = (twall_eff > 0.0) ? particle->evib(sp0, twall_eff, random) : 0.0;
          double erot1 = (twall_eff > 0.0) ? particle->erot(sp1, twall_eff, random) : 0.0;
          double evib1 = (twall_eff > 0.0) ? particle->evib(sp1, twall_eff, random) : 0.0;

          ip->ispecies = sp0;
          ip->erot = erot0;
          ip->evib = evib0;
          double mass0 = particle->species[sp0].mass;
          sample_cosine_velocity(ip->v, norm, r->energy[0], mass0);

          int id = MAXSMALLINT * random->uniform();
          double x[3], v[3];
          memcpy(x, ip->x, 3*sizeof(double));
          double mass1 = particle->species[sp1].mass;
          sample_cosine_velocity(v, norm, r->energy[1], mass1);
          Particle::OnePart *particles = particle->particles;
          int reallocflag =
            particle->add_particle(id, sp1, ip->icell, x, v, erot1, evib1);
          if (reallocflag) ip = particle->particles + (ip - particles);
          jp = &particle->particles[particle->nlocal-1];
          // Notify update_custom subscribers (e.g. fix particle/weight
          // setting pweight = fnum) for the spawned dissociation product.
          if (modify->n_update_custom) {
            double zero_v[3] = {0.0, 0.0, 0.0};
            modify->update_custom(particle->nlocal - 1, 0.0, 0.0, 0.0, zero_v);
          }
          return (list[i] + 1);
        }
      case EXCHANGE:
        {
          int sp0 = r->products[0];
          // EXCHANGE: if twall_eff set, accommodate at the wall.
          // Else, preserve internal energy iff species is unchanged
          // (reflection); otherwise reset (species change implies a
          // chemical event that destroys the incoming internal state).
          if (twall_eff > 0.0) {
            ip->erot = particle->erot(sp0, twall_eff, random);
            ip->evib = particle->evib(sp0, twall_eff, random);
          } else if (sp0 != ip->ispecies) {
            ip->erot = 0.0;
            ip->evib = 0.0;
          }
          ip->ispecies = sp0;
          double mass0 = particle->species[sp0].mass;
          sample_cosine_velocity(ip->v, norm, r->energy[0], mass0);
          return (list[i] + 1);
        }
      case RECOMBINATION:
        {
          ip = NULL;
          return (list[i] + 1);
        }
      case ABSORB_REEMIT:
        {
          // EIRENE-style stochastic recycling. Reactant has already been
          // absorbed; we now roll for: atomic re-emission, molecular
          // re-emission, or retained (pumped).
          //
          // Rules:
          //   If reactant species == product species (simple return):
          //     P(re-emit same species) = R
          //     P(pump)                 = 1 - R
          //   Else (atomic reactant -> molecular product, e.g. D -> D2):
          //     P(re-emit atomic, same species)    = R * (1 - f_mol)
          //     P(re-emit molecular, product spec) = R * f_mol / 2
          //     P(pump)                            = 1 - R + R*f_mol/2
          //   where R_eff = per-surf R (if R_surf bound) else r->energy[0],
          //         f_mol = r->energy[1].
          //
          // Outgoing velocity: half-Maxwellian flux at twall_eff.
          // Outgoing rovib: thermal at twall_eff (nonzero only for
          // polyatomic products via species DOFs).
          double R_rec;
          if (rindex_custom >= 0) {
            R_rec = surf->edvec_local[rindex_custom][isurf];
            if (R_rec < 0.0) R_rec = 0.0;
            if (R_rec > 1.0) R_rec = 1.0;
          } else {
            R_rec = r->energy[0];
          }
          double f_mol = r->energy[1];
          int sp_atom = ip->ispecies;    // same as reactant
          int sp_mol  = r->products[0];  // the named product (may == reactant)
          bool has_mol_channel = (sp_mol != sp_atom);

          double u = random->uniform();
          double p_atom, p_mol;
          if (has_mol_channel) {
            p_atom = R_rec * (1.0 - f_mol);
            p_mol  = R_rec * f_mol * 0.5;  // factor of 2: 2 atoms -> 1 molecule
          } else {
            p_atom = R_rec;
            p_mol  = 0.0;
          }

          // For the re-emission branches, use twall_eff (fall back to
          // 300 K if not set so we never emit with KE = 0).
          double T_out = (twall_eff > 0.0) ? twall_eff : 300.0;

          if (u < p_atom) {
            // atomic re-emission at T_wall
            double mass_out = particle->species[sp_atom].mass;
            sample_thermal_flux_velocity(ip->v, norm, T_out, mass_out);
            ip->ispecies = sp_atom;
            ip->erot = particle->erot(sp_atom, T_out, random);
            ip->evib = particle->evib(sp_atom, T_out, random);
            return (list[i] + 1);
          } else if (u < p_atom + p_mol) {
            // molecular re-emission at T_wall
            double mass_out = particle->species[sp_mol].mass;
            sample_thermal_flux_velocity(ip->v, norm, T_out, mass_out);
            ip->ispecies = sp_mol;
            ip->erot = particle->erot(sp_mol, T_out, random);
            ip->evib = particle->evib(sp_mol, T_out, random);
            return (list[i] + 1);
          } else {
            // pump / retain: delete incoming particle
            ip = NULL;
            return (list[i] + 1);
          }
        }
      case TRIM_REFLECT:
        {
          int sp0 = r->products[0];
          Reflection::View tv = trim_tables[r->trim_table].view();
          double u1 = random->uniform();
          double u2 = random->uniform();
          double u3 = random->uniform();
          double E_out_eV = 0.0, cos_polar = 1.0, cos_azim = 1.0;
          Reflection::sample_reflection(tv, E_in_eV, theta_in_deg,
                                        u1, u2, u3,
                                        &E_out_eV, &cos_polar, &cos_azim);

          double mass_out = particle->species[sp0].mass;
          double v_out[3];
          sample_reflected_velocity(v_out, ip->v, norm, E_out_eV,
                                    cos_polar, cos_azim, mass_out);

          // internal energy: accommodate at twall if set; else reset to 0.
          // Reflected atoms are almost always monatomic (D, H, T, He, ...),
          // so erot/evib are 0 by species DOF regardless; this covers the
          // general case if a polyatomic reflection table is ever provided.
          if (twall_eff > 0.0) {
            ip->erot = particle->erot(sp0, twall_eff, random);
            ip->evib = particle->evib(sp0, twall_eff, random);
          } else {
            ip->erot = 0.0;
            ip->evib = 0.0;
          }
          ip->ispecies = sp0;
          ip->v[0] = v_out[0];
          ip->v[1] = v_out[1];
          ip->v[2] = v_out[2];
          return (list[i] + 1);
        }
      }
    }
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWI::sample_cosine_velocity(double *v, double *norm,
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

char *SurfReactSurfacePWI::reactionID(int m)
{
  return rlist[m].id;
}

double SurfReactSurfacePWI::reaction_coeff(int m)
{
  // SPARTA's compute surf etot expects reaction_coeff() in J/event
  // (chemical energy released/absorbed). For PWI (wall recycling via
  // TRIM tables) there is no chemical energy budget: the kinetic
  // energy change is already captured via (V_post - V_pre).  Returning
  // rlist[m].prob (a unitless probability ~1) caused compute surf to
  // add ~1 J per event, scaled by fnum/dt, giving spurious wall energy
  // ~1e23 W per macroparticle hit.  Return 0 so etot tallies only KE.
  return 0.0;
}

int SurfReactSurfacePWI::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species, rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

int SurfReactSurfacePWI::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species, rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWI::init_reactions()
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
  reactions = memory->create(reactions, nspecies, "surf_react_surface_pwi:reactions");

  for (int i = 0; i < nspecies; i++) reactions[i].n = 0;

  int ntot = 0;
  for (int m = 0; m < nlist_recycle; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    reactions[r->reactants[0]].n++;
    ntot++;
  }

  memory->destroy(indices);
  memory->create(indices, ntot > 0 ? ntot : 1, "surf_react_surface_pwi:indices");

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

void SurfReactSurfacePWI::readfile(char *fname)
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
                         "surf_react_surface_pwi:rlist");
      for (int i = nlist_recycle; i < maxlist_recycle; i++) {
        r = &rlist[i];
        r->nreactant = r->nproduct = 0;
        r->id_reactants = new char*[MAXREACTANT];
        r->id_products = new char*[MAXPRODUCT];
        r->reactants = new int[MAXREACTANT];
        r->products = new int[MAXPRODUCT];
        r->energy = new double[MAXPRODUCT];
        r->energy[0] = r->energy[1] = 0.0;
        r->prob = 0.0;
        r->trim_table = -1;
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
    //           or: T <table_name>                    (TRIM reflection)
    word = strtok(line2, " \t\n");
    if (!word) error->all(FLERR, "Invalid reaction type in recycle file");
    if (word[0] == 'D' || word[0] == 'd') r->type = DISSOCIATION;
    else if (word[0] == 'E' || word[0] == 'e') r->type = EXCHANGE;
    else if (word[0] == 'R' || word[0] == 'r') r->type = RECOMBINATION;
    else if (word[0] == 'T' || word[0] == 't') r->type = TRIM_REFLECT;
    else if (word[0] == 'A' || word[0] == 'a') r->type = ABSORB_REEMIT;
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
    } else if (r->type == TRIM_REFLECT) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR, "TRIM reflect recycle reaction needs 1 reactant, 1 product");
    } else if (r->type == ABSORB_REEMIT) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR, "Absorb/re-emit recycle reaction needs 1 reactant, 1 product "
                          "(product = returning species; same as reactant for simple "
                          "return, or D2 if atomic-to-molecular conversion allowed)");
    }

    if (r->type == TRIM_REFLECT) {
      // TRIM reaction: parse `T <table_name>`.
      // prob is evaluated per-event from R_N(E,theta); the fixed prob slot
      // is set to 0 here so cumulative-prob iteration in react() skips to
      // the dynamic probe.  Table is resolved via ProcessLibrary from
      // database/processes.h5 (/surface/reflection/<pair>/).
      word = strtok(NULL, " \t\n");
      if (!word) error->all(FLERR, "Missing TRIM table name in recycle reaction");
      int it = load_or_get_trim_table(word);
      if (it < 0) {
        char str[256];
        snprintf(str, sizeof(str),
                 "surf_react surface/pwi: failed to load TRIM table '%s' "
                 "from database/processes.h5", word);
        error->all(FLERR, str);
      }
      r->trim_table = it;
      r->prob = 0.0;

    } else if (r->type == ABSORB_REEMIT) {
      // EIRENE-style absorb-and-re-emit: catch-all after TRIM.
      // Syntax: A <R> [<f_mol>]
      //   R     = recycling coefficient (0..1). Fraction R returned, (1-R) pumped.
      //   f_mol = fraction of returned atoms that come back in molecular form
      //           (i.e. as the PRODUCT species). Species-agnostic: D/W uses
      //           D -> D2 with f_mol = f_D2; H/Li uses H -> H2 with f_mol = f_H2;
      //           applies equally to T, He recombination channels, etc.
      //           Only meaningful when reactant != product (atom-to-molecule
      //           conversion). Ignored if reactant == product (simple return).
      // Stored as r->prob=1 (cumulative catch-all) + r->energy[0]=R + energy[1]=f_mol.
      word = strtok(NULL, " \t\n");
      if (!word) error->all(FLERR, "Missing R coefficient in A recycle reaction");
      double R_val = input->numeric(FLERR, word);
      if (R_val < 0.0 || R_val > 1.0)
        error->all(FLERR, "A reaction R coefficient must be in [0,1]");
      r->energy[0] = R_val;

      word = strtok(NULL, " \t\n");
      if (word) {
        double f_val = input->numeric(FLERR, word);
        if (f_val < 0.0 || f_val > 1.0)
          error->all(FLERR, "A reaction f_mol coefficient must be in [0,1]");
        r->energy[1] = f_val;
      } else {
        // f_mol not provided: simple return rule (reactant==product expected).
        r->energy[1] = 0.0;
      }
      r->prob = 1.0;

    } else {
      // non-TRIM: probability
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
    }

    nlist_recycle++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Load a reflection table from trim_dir/<name>.h5, or return cached.
   HDF5 schema matches database/surface/trim/*.h5 (EIRENE rdtrim.f-style,
   but any BCA-generated file on the same (E, theta, q) grid layout works).
   Returns index into trim_tables, or -1 on failure.
------------------------------------------------------------------------- */

int SurfReactSurfacePWI::load_or_get_trim_table(const char *name)
{
  std::string sname(name);
  auto it = trim_index.find(sname);
  if (it != trim_index.end()) return it->second;

  Reflection::Table t;
  t.name = sname;
  t.Z1 = t.M1 = t.Z2 = t.M2 = 0.0;

  // Prefer database/processes.h5 /surface/reflection/<pair>/ via
  // ProcessLibrary.  Pair keys in processes.h5 are lowercase
  // (e.g. "d_on_w"); legacy per-pair files use original casing
  // ("D_on_W.h5").  Try both.
  std::string pair_lower = sname;
  for (auto &c : pair_lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

  bool loaded = false;
  std::string processes_path = resolve_processes_file();
  if (!processes_path.empty()) {
    ProcessLibrary lib;
    lib.open(processes_path, world, error);
    if (lib.is_open()) {
      ProcessLibrary::TrimReflectionTable pt;
      if (lib.load_trim_reflection(pair_lower, pt) &&
          pt.NE == Reflection::NE &&
          pt.NTHETA == Reflection::NTHETA &&
          pt.NQ == Reflection::NQ) {
        t.E_grid      = pt.E;
        t.theta_grid  = pt.theta;
        t.raar        = pt.raar;
        t.R_N         = pt.R_N;
        t.Eout_q      = pt.Eout_q;
        t.Eout_min    = pt.Eout_min;
        t.Eout_max    = pt.Eout_max;
        t.cos_polar_q = pt.cos_polar_q;
        t.cos_azim_q  = pt.cos_azim_q;
        t.Z1 = pt.Z1; t.M1 = pt.M1; t.Z2 = pt.Z2; t.M2 = pt.M2;
        loaded = true;
        if (comm->me == 0)
          fprintf(screen ? screen : logfile,
                  "surf_react surface/pwi: loaded TRIM table '%s' from "
                  "%s (/surface/reflection/%s/)\n",
                  sname.c_str(), processes_path.c_str(),
                  pair_lower.c_str());
      }
    }
  }

  if (!loaded) {
    std::string msg = "surf_react surface/pwi: TRIM table '" + sname +
                      "' not found in " +
                      (processes_path.empty() ? "database/processes.h5 (file missing)"
                                              : processes_path + " at /surface/reflection/" + pair_lower);
    error->all(FLERR, msg.c_str());
    return -1;  // not reached
  }

  int idx = static_cast<int>(trim_tables.size());
  trim_tables.push_back(std::move(t));
  trim_index[sname] = idx;
  return idx;
}

/* ----------------------------------------------------------------------
   Half-Maxwellian flux sampler at wall temperature T_K (Kelvin).

   For a particle of mass m leaving the wall in thermal equilibrium with
   the surface, Bird 1994 (DSMC) gives the outgoing velocity as:

     v_n  = vrm * sqrt(-ln(u1))        (flux-biased normal component)
     v_t1 ~ Gaussian(0, sqrt(kB*T/m))  (tangential, unconstrained)
     v_t2 ~ Gaussian(0, sqrt(kB*T/m))  (tangential, unconstrained)

   where vrm = sqrt(2*kB*T/m) is the most-probable speed of the 3D
   Maxwell distribution. The normal PDF is f(v_n) proportional to
   v_n * exp(-v_n^2/vrm^2), which is the FLUX distribution (i.e. the
   speed distribution of particles crossing an imaginary plane per unit
   area per unit time), not the density distribution.

   Invariants at steady state with R=1, f_mol=0 (pure atomic recycle):
     <v_n> = sqrt(pi * kB * T / (2 m))    (mean outgoing normal speed)
     <E>   = 2 kB * T                     (mean outgoing KE — NOT 3/2 kT)

   Species-agnostic: works for D, H, T, He, Li, D2, H2, any species in
   particle->species. Mass is taken from that table.
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::sample_thermal_flux_velocity(double *v,
                                                     const double *norm,
                                                     double T_K, double mass)
{
  if (mass <= 0.0 || T_K <= 0.0) { v[0] = v[1] = v[2] = 0.0; return; }

  const double kB = 1.380649e-23;
  double vrm = sqrt(2.0 * kB * T_K / mass);
  double vtan_scale = vrm / sqrt(2.0);  // = sqrt(kB*T/m)

  // flux-biased normal component (half-Maxwellian), always positive
  double u1 = random->uniform();
  if (u1 < 1e-300) u1 = 1e-300;
  double vn = vrm * sqrt(-log(u1));

  // two independent Gaussians for tangentials via Box-Muller
  double u2 = random->uniform();
  double u3 = random->uniform();
  if (u2 < 1e-300) u2 = 1e-300;
  double r  = vtan_scale * sqrt(-2.0 * log(u2));
  double ph = MY_2PI * u3;
  double vt1 = r * cos(ph);
  double vt2 = r * sin(ph);

  // orthonormal basis: nh + two tangents
  double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
  double nh[3] = {norm[0]/nlen, norm[1]/nlen, norm[2]/nlen};

  double arb[3] = {1.0, 0.0, 0.0};
  if (fabs(nh[0]) > 0.9) { arb[0] = 0.0; arb[1] = 1.0; arb[2] = 0.0; }
  double dot = arb[0]*nh[0] + arb[1]*nh[1] + arb[2]*nh[2];
  double t1[3] = {arb[0] - dot*nh[0],
                  arb[1] - dot*nh[1],
                  arb[2] - dot*nh[2]};
  double t1len = sqrt(t1[0]*t1[0] + t1[1]*t1[1] + t1[2]*t1[2]);
  t1[0] /= t1len; t1[1] /= t1len; t1[2] /= t1len;
  double t2[3] = {
    nh[1]*t1[2] - nh[2]*t1[1],
    nh[2]*t1[0] - nh[0]*t1[2],
    nh[0]*t1[1] - nh[1]*t1[0]
  };

  v[0] = vn*nh[0] + vt1*t1[0] + vt2*t2[0];
  v[1] = vn*nh[1] + vt1*t1[1] + vt2*t2[1];
  v[2] = vn*nh[2] + vt1*t1[2] + vt2*t2[2];
}

/* ----------------------------------------------------------------------
   Build outgoing velocity from TRIM sample_reflection output.
   v_in is the incident velocity; norm is the outward surface normal.
   cos_polar is w.r.t. the outward normal; cos_azim is w.r.t. the
   plane containing (norm, v_in) -- this is EIRENE's convention.
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::sample_reflected_velocity(double *v_out,
                                                  const double *v_in,
                                                  const double *norm,
                                                  double E_out_eV,
                                                  double cos_polar,
                                                  double cos_azim,
                                                  double mass_out)
{
  // outgoing speed from reflected kinetic energy
  double E_out_J = E_out_eV / update->joule2ev / update->mvv2e;
  double vmag = (mass_out > 0.0 && E_out_J > 0.0)
              ? sqrt(2.0 * E_out_J / mass_out)
              : 0.0;

  // build orthonormal basis (n_hat, t_in_hat, b_hat) where t_in_hat is
  // the tangential component of -v_in (so cos_azim = 1 -> forward-scatter
  // direction along the incident projection). This matches how EIRENE
  // labels its azimuthal quantile axis.
  double nlen = sqrt(norm[0]*norm[0] + norm[1]*norm[1] + norm[2]*norm[2]);
  double nh[3] = {norm[0]/nlen, norm[1]/nlen, norm[2]/nlen};
  double vn = v_in[0]*nh[0] + v_in[1]*nh[1] + v_in[2]*nh[2];
  double tang[3] = {v_in[0] - vn*nh[0],
                    v_in[1] - vn*nh[1],
                    v_in[2] - vn*nh[2]};
  double tlen = sqrt(tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2]);
  if (tlen < 1e-12) {
    // normal incidence: pick arbitrary tangent
    double arb[3] = {1.0, 0.0, 0.0};
    if (fabs(nh[0]) > 0.9) { arb[0]=0.0; arb[1]=1.0; arb[2]=0.0; }
    double dot = arb[0]*nh[0] + arb[1]*nh[1] + arb[2]*nh[2];
    tang[0] = arb[0] - dot*nh[0];
    tang[1] = arb[1] - dot*nh[1];
    tang[2] = arb[2] - dot*nh[2];
    tlen = sqrt(tang[0]*tang[0] + tang[1]*tang[1] + tang[2]*tang[2]);
  }
  // flip sign: forward-scatter direction of reflected particle
  double tin_hat[3] = {-tang[0]/tlen, -tang[1]/tlen, -tang[2]/tlen};
  double b_hat[3] = {
    nh[1]*tin_hat[2] - nh[2]*tin_hat[1],
    nh[2]*tin_hat[0] - nh[0]*tin_hat[2],
    nh[0]*tin_hat[1] - nh[1]*tin_hat[0]
  };

  double sin_polar = sqrt(fmax(0.0, 1.0 - cos_polar*cos_polar));
  double sin_azim  = sqrt(fmax(0.0, 1.0 - cos_azim*cos_azim));
  // TRIM doesn't specify the hemisphere of the azimuth -- randomize sign
  if (random->uniform() < 0.5) sin_azim = -sin_azim;

  for (int d = 0; d < 3; d++) {
    v_out[d] = vmag * (cos_polar * nh[d]
                     + sin_polar * (cos_azim * tin_hat[d]
                                  + sin_azim * b_hat[d]));
  }
}

/* ---------------------------------------------------------------------- */

int SurfReactSurfacePWI::readone(char *line1, char *line2, int &n1, int &n2)
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
