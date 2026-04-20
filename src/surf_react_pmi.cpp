/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge

   surf_react pmi — Tabulated PMI Surface Reactions

   Reads reflection/sputtering coefficients from HDF5 tables (e.g.,
   FTRIDYN output like O(8)_on_W(74).h5), computes incident energy/angle
   at each surface hit, and stochastically decides reflect/sputter/absorb.

   Syntax:
     surf_react ID pmi constant reaction_file.surf RN RE Y Ebind
     surf_react ID pmi file     reaction_file.surf surface_data.h5

   Reaction file types (second line first letter):
     R = Reflect:  incident -> neutral product (ip transforms, jp=NULL)
     S = Sputter:  incident absorbed, wall atom emitted (ip=NULL, jp=product)
     A = Absorb:   incident sticks (ip=NULL, jp=NULL)
------------------------------------------------------------------------- */

#include "math.h"
#include "surf_react_pmi.h"
#include "input.h"
#include "update.h"
#include "comm.h"
#include "domain.h"
#include "surf.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

#include <H5Cpp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SPARTA_NS;

enum{REFLECT,SPUTTER,ABSORB};
enum{SIMPLE,ECKSTEIN,TRIM_STYLE};

#define MAXREACTANT 1
#define MAXPRODUCT 2
#define MAXCOEFF 12  // bumped from 2 to fit Eckstein parameter pack

#define MAXLINE 1024
#define DELTALIST 16

// eV <-> Joule conversion
static const double EV_TO_J = 1.602176634e-19;

/* ---------------------------------------------------------------------- */

SurfReactPMI::SurfReactPMI(SPARTA *sparta, int narg, char **arg) :
  SurfReact(sparta, narg, arg)
{
  if (narg < 4) error->all(FLERR,"Illegal surf_react pmi command");

  // initialize reaction data structs

  nlist_prob = maxlist_prob = 0;
  rlist = NULL;
  species_reactions = NULL;
  last_outcome = 0;
  nE = nA = 0;
  nb_cdf = 0;
  has_cdf_R = has_cdf_Y = 0;
  Ebind = 0.0;
  logflag = 0;
  logfp = NULL;
  logfile_base = NULL;
  me = comm->me;

  // parse mode: constant or file
  // then scan remaining args for log/file keywords

  int iarg_end;

  // Pre-scan for trim_dir so Trim reactions parsed inside readfile()
  // can resolve their table names.  trim_dir may appear anywhere in the
  // argument list after the mode keyword; we pick it up here.
  for (int k = 3; k + 1 < narg; k++) {
    if (strcmp(arg[k],"trim_dir") == 0) {
      trim_dir = std::string(arg[k+1]);
      break;
    }
  }

  if (strcmp(arg[2],"constant") == 0) {
    mode = 0;
    if (narg < 8) error->all(FLERR,
      "surf_react pmi constant requires: reaction_file RN RE Y Ebind");
    readfile(arg[3]);
    const_RN    = input->numeric(FLERR,arg[4]);
    const_RE    = input->numeric(FLERR,arg[5]);
    const_Y     = input->numeric(FLERR,arg[6]);
    const_Ebind = input->numeric(FLERR,arg[7]);
    if (const_RN < 0.0 || const_RN > 1.0)
      error->all(FLERR,"surf_react pmi: RN must be in [0,1]");
    if (const_RE < 0.0 || const_RE > 1.0)
      error->all(FLERR,"surf_react pmi: RE must be in [0,1]");
    if (const_Y < 0.0)
      error->all(FLERR,"surf_react pmi: Y must be >= 0");
    if (const_Ebind < 0.0)
      error->all(FLERR,"surf_react pmi: Ebind must be >= 0");
    Ebind = const_Ebind;
    iarg_end = 8;
  } else if (strcmp(arg[2],"file") == 0) {
    mode = 1;
    if (narg < 5) error->all(FLERR,
      "surf_react pmi file requires: reaction_file surface_data.h5");
    readfile(arg[3]);
    h5_path = std::string(arg[4]);
    iarg_end = 5;
  } else if (strcmp(arg[2],"eckstein") == 0) {
    // analytic mode: no bulk HDF5 table.  Reflect and Sputter reactions
    // must use Eckstein (analytic) or Trim (tabulated EIRENE reflection).
    // Absorb reactions stay Simple.
    mode = 2;
    if (narg < 4) error->all(FLERR,
      "surf_react pmi eckstein requires: reaction_file");
    readfile(arg[3]);
    for (int i = 0; i < nlist_prob; i++) {
      if (rlist[i].type == ABSORB) continue;
      if (rlist[i].style != ECKSTEIN && rlist[i].style != TRIM_STYLE) {
        char str[256];
        snprintf(str,sizeof(str),
          "surf_react pmi eckstein: reflect/sputter reaction '%s' must be "
          "Eckstein or Trim style",
          rlist[i].id ? rlist[i].id : "(unknown)");
        error->all(FLERR,str);
      }
    }
    iarg_end = 4;
  } else {
    error->all(FLERR,"surf_react pmi: mode must be 'constant', 'file', "
                     "or 'eckstein'");
    iarg_end = narg;
  }

  // parse optional trailing keywords: log yes/no, file <path>, trim_dir <path>.
  // trim_dir must be set BEFORE readfile() if reactions reference Trim
  // tables.  The constructor handles this by delaying readfile() until
  // after the keyword sweep - but we already called readfile() above in
  // the eckstein path.  For ease of use we support trim_dir both before
  // the reaction file (via an extended syntax) and after; if a Trim
  // reaction was parsed before trim_dir was known, load_or_get_trim_table
  // will already have errored.  The idiomatic ordering is:
  //   surf_react ID pmi eckstein trim_dir <path> <reactions_file>
  // but we also accept trailing trim_dir for backward compat.

  int iarg = iarg_end;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"log") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"surf_react pmi: log requires yes/no");
      if (strcmp(arg[iarg+1],"yes") == 0) logflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) logflag = 0;
      else error->all(FLERR,"surf_react pmi: log requires yes/no");
      iarg += 2;
    } else if (strcmp(arg[iarg],"file") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"surf_react pmi: file requires a path");
      delete [] logfile_base;
      int n = strlen(arg[iarg+1]) + 1;
      logfile_base = new char[n];
      strcpy(logfile_base, arg[iarg+1]);
      logflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"trim_dir") == 0) {
      if (iarg + 1 >= narg)
        error->all(FLERR,"surf_react pmi: trim_dir requires a path");
      trim_dir = std::string(arg[iarg+1]);
      iarg += 2;
    } else {
      error->all(FLERR,"surf_react pmi: unknown keyword");
    }
  }

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

SurfReactPMI::~SurfReactPMI()
{
  if (copy) return;

  close_logfile();
  delete [] logfile_base;
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

  memory->destroy(species_reactions);
}

/* ---------------------------------------------------------------------- */

void SurfReactPMI::init()
{
  SurfReact::init();
  init_reactions();

  if (logflag) open_logfile();

  if (mode == 1) {
    try {
      load_surface_file();
    } catch (const std::exception& e) {
      error->all(FLERR, e.what());
    } catch (...) {
      error->all(FLERR, "surf_react pmi: failed reading HDF5 surface file");
    }
  }
}

/* ----------------------------------------------------------------------
   PMI surface reaction: reflect / sputter / absorb
   return which reaction 1 to N, 0 = no reaction
   sets velreset = 1 (we own the outgoing velocities)
------------------------------------------------------------------------- */

int SurfReactPMI::react(Particle::OnePart *&ip, int isurf,
                         double *norm,
                         Particle::OnePart *&jp,
                         int &velreset)
{
  jp = NULL;
  last_outcome = 0;

  int ispecies = ip->ispecies;
  SpeciesReactions &sr = species_reactions[ispecies];

  // check if any PMI reactions exist for this species
  if (sr.ireflect < 0 && sr.isputter < 0 && sr.iabsorb < 0) return 0;

  // --- Compute incident kinetic energy (eV) and angle (degrees) ---

  double *v = ip->v;
  double vsq = v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
  double vlen = sqrt(vsq);

  double mass = particle->species[ispecies].mass;
  double E_eV = 0.5 * update->mvv2e * mass * vsq;
  if (strcmp(update->unit_style,"si") == 0) E_eV *= update->joule2ev;

  // angle relative to surface normal: 0 = normal incidence, 90 = grazing
  double theta_deg = 0.0;
  if (vlen > 0.0) {
    double nlen = MathExtra::len3(norm);
    if (nlen > 0.0) {
      double dot = MathExtra::dot3(v, norm);
      double cos_theta = -dot / (vlen * nlen);
      if (cos_theta < -1.0) cos_theta = -1.0;
      if (cos_theta > 1.0) cos_theta = 1.0;
      theta_deg = acos(cos_theta) * 180.0 / MathConst::MY_PI;
    }
  }

  // --- Look up coefficients ---
  //
  // Per-reaction style: each reaction (reflect, sputter, absorb) can
  // independently use SIMPLE (table or constant) or ECKSTEIN (analytic
  // formula with parameters packed into coeff[]).

  double RN = 0.0, RE = 0.0, Y = 0.0, Eb = 0.0;

  // Defaults from SIMPLE-mode path (table or constant).  In mode==2
  // (pure Eckstein) there is no table, so leave these at zero - they
  // won't be read because all reactions are ECKSTEIN style.
  double RN_simple = 0.0, RE_simple = 0.0, Y_simple = 0.0, Eb_simple = 0.0;
  if (mode == 0) {
    RN_simple = const_RN;
    RE_simple = const_RE;
    Y_simple  = const_Y;
    Eb_simple = const_Ebind;
  } else if (mode == 1) {
    RN_simple = interp_table(RN_table, E_eV, theta_deg);
    RE_simple = interp_table(RE_table, E_eV, theta_deg);
    Y_simple  = interp_table(spyld_table, E_eV, theta_deg);
    Eb_simple = Ebind;
  }

  // Reflect reaction -> RN, RE
  if (sr.ireflect >= 0) {
    OneReaction *rr = &rlist[sr.ireflect];
    if (rr->style == ECKSTEIN) {
      Eckstein::ReflectParams rp;
      Eckstein::unpack_reflect(rr->coeff, rp);
      RN = Eckstein::reflection_RN(E_eV, theta_deg, rp);
      RE = rp.re_frac;  // <E_out>/<E_in> fraction
    } else if (rr->style == TRIM_STYLE) {
      int it = static_cast<int>(rr->coeff[0]);
      if (it >= 0 && it < (int)trim_tables.size()) {
        Reflection::View tv = trim_tables[it].view();
        RN = Reflection::R_N_interp(tv, E_eV, theta_deg);
        // RE is not used on the Trim path - outgoing energy is sampled
        // directly below in the reflect branch, not computed as RE*E.
        RE = 0.0;
      } else {
        RN = 0.0; RE = 0.0;
      }
    } else {
      RN = RN_simple;
      RE = RE_simple;
    }
  }

  // Sputter reaction -> Y (and Eb from the same params when Eckstein)
  if (sr.isputter >= 0) {
    OneReaction *rs = &rlist[sr.isputter];
    if (rs->style == ECKSTEIN) {
      Eckstein::SputterParams sp;
      Eckstein::unpack_sputter(rs->coeff, sp);
      Y  = Eckstein::sputter_yield(E_eV, theta_deg, sp);
      Eb = sp.Es;
    } else {
      Y  = Y_simple;
      Eb = Eb_simple;
    }
  } else {
    Eb = Eb_simple;
  }

  // clamp to valid range
  if (RN < 0.0) RN = 0.0;
  if (RN > 1.0) RN = 1.0;
  if (RE < 0.0) RE = 0.0;
  if (RE > 1.0) RE = 1.0;
  if (Y < 0.0) Y = 0.0;

  // renormalize if RN+Y > 1
  double sum = RN + Y;
  if (sum > 1.0) {
    RN /= sum;
    Y  /= sum;
  }

  // --- Stochastic decision ---

  double u = random->uniform();
  velreset = 1;  // we set outgoing velocities directly

  if (u < RN && sr.ireflect >= 0) {
    // REFLECT: ip -> neutral product
    //   SIMPLE/ECKSTEIN path: cosine-law direction + E_out = RE * E_in
    //   TRIM path:            sample E_out, polar, azimuthal from tables

    int irxn = sr.ireflect;
    OneReaction *r = &rlist[irxn];
    ip->ispecies = r->products[0];

    double mass_out = particle->species[ip->ispecies].mass;
    double E_out = 0.0;

    if (r->style == TRIM_STYLE) {
      int it = static_cast<int>(r->coeff[0]);
      Reflection::View tv = trim_tables[it].view();
      double u1 = random->uniform();
      double u2 = random->uniform();
      double u3 = random->uniform();
      double cos_polar = 1.0, cos_azim = 1.0;
      Reflection::sample_reflection(tv, E_eV, theta_deg, u1, u2, u3,
                                    &E_out, &cos_polar, &cos_azim);

      double E_out_J = E_out / update->joule2ev / update->mvv2e;
      double vmag = sqrt(2.0 * E_out_J / mass_out);

      // Build an orthonormal basis (n_hat, t_in_hat, b_hat) where
      //   n_hat is the outward surface normal,
      //   t_in_hat is the tangential component of the incident velocity
      //            (so the reflected particle's azimuthal direction is
      //             measured relative to the incidence plane, matching
      //             how EIRENE labels its azimuthal quantile tables).
      double nlen = MathExtra::len3(norm);
      double nh[3] = {norm[0]/nlen, norm[1]/nlen, norm[2]/nlen};
      double vin[3] = {ip->v[0], ip->v[1], ip->v[2]};
      double vn = MathExtra::dot3(vin, nh);
      double tang[3] = {vin[0] - vn*nh[0],
                        vin[1] - vn*nh[1],
                        vin[2] - vn*nh[2]};
      double tlen = sqrt(tang[0]*tang[0]+tang[1]*tang[1]+tang[2]*tang[2]);
      if (tlen < 1e-12) {
        // normal incidence: pick an arbitrary tangent
        double arb[3] = {1.0, 0.0, 0.0};
        if (fabs(nh[0]) > 0.9) { arb[0]=0.0; arb[1]=1.0; arb[2]=0.0; }
        tang[0] = arb[0] - (arb[0]*nh[0]+arb[1]*nh[1]+arb[2]*nh[2])*nh[0];
        tang[1] = arb[1] - (arb[0]*nh[0]+arb[1]*nh[1]+arb[2]*nh[2])*nh[1];
        tang[2] = arb[2] - (arb[0]*nh[0]+arb[1]*nh[1]+arb[2]*nh[2])*nh[2];
        tlen = sqrt(tang[0]*tang[0]+tang[1]*tang[1]+tang[2]*tang[2]);
      }
      // tin_hat: flip sign so reflected azimuth=0 points in forward
      // scatter direction (away from the wall projection of incident v).
      double tin_hat[3] = {-tang[0]/tlen, -tang[1]/tlen, -tang[2]/tlen};
      double b_hat[3];
      b_hat[0] = nh[1]*tin_hat[2] - nh[2]*tin_hat[1];
      b_hat[1] = nh[2]*tin_hat[0] - nh[0]*tin_hat[2];
      b_hat[2] = nh[0]*tin_hat[1] - nh[1]*tin_hat[0];

      double sin_polar = sqrt(fmax(0.0, 1.0 - cos_polar*cos_polar));
      double sin_azim  = sqrt(fmax(0.0, 1.0 - cos_azim*cos_azim));
      // randomize the sign of the azimuthal out-of-plane component
      if (random->uniform() < 0.5) sin_azim = -sin_azim;

      for (int d = 0; d < 3; d++) {
        ip->v[d] = vmag * (cos_polar * nh[d]
                         + sin_polar * (cos_azim * tin_hat[d]
                                      + sin_azim * b_hat[d]));
      }
    } else {
      E_out = RE * E_eV;
      double E_out_J = E_out / update->joule2ev / update->mvv2e;
      double vmag_out = sqrt(2.0 * E_out_J / mass_out);
      // cosine-law direction sampling into outward hemisphere
      sample_cosine_direction(norm, ip->v, vmag_out);
    }

    ip->erot = 0.0;
    ip->evib = 0.0;

    if (logflag) log_impact(ip, isurf, norm, E_eV, theta_deg, 1, E_out);

    nsingle++;
    tally_single[irxn]++;
    last_outcome = 1;
    return irxn + 1;

  } else if (u < RN + Y && sr.isputter >= 0) {
    // SPUTTER: ip absorbed, create jp as wall atom

    int irxn = sr.isputter;
    OneReaction *r = &rlist[irxn];

    // log before nulling ip
    if (logflag) log_impact(ip, isurf, norm, E_eV, theta_deg, 2, 0.0);

    // save ip position/cell before nulling
    double x[3];
    memcpy(x, ip->x, 3*sizeof(double));
    int icell = ip->icell;

    // absorb incident particle
    ip = NULL;

    // create sputtered particle
    int sput_species = r->products[0];
    double mass_sput = particle->species[sput_species].mass;

    // sputtered energy from Thompson distribution or tabulated CDF
    double E_sput;
    if (has_cdf_Y) {
      // find nearest grid indices for CDF lookup
      int iE = 0, iA = 0;
      if (nE > 0) {
        auto itE = std::lower_bound(E_axis.begin(), E_axis.end(), E_eV);
        iE = static_cast<int>(itE - E_axis.begin());
        if (iE >= nE) iE = nE - 1;
      }
      if (nA > 0) {
        auto itA = std::lower_bound(A_axis.begin(), A_axis.end(), theta_deg);
        iA = static_cast<int>(itA - A_axis.begin());
        if (iA >= nA) iA = nA - 1;
      }
      E_sput = sample_from_cdf(edist_Y, iE, iA);
    } else {
      E_sput = sample_thompson(Eb / 2.0, E_eV);
    }
    if (E_sput < 0.0) E_sput = 0.0;

    double E_sput_J = E_sput / update->joule2ev / update->mvv2e;
    double vmag_sput = sqrt(2.0 * E_sput_J / mass_sput);

    double v_sput[3];
    sample_cosine_direction(norm, v_sput, vmag_sput);

    int id = MAXSMALLINT * random->uniform();
    Particle::OnePart *particles = particle->particles;
    int reallocflag =
      particle->add_particle(id, sput_species, icell, x, v_sput, 0.0, 0.0);
    if (reallocflag) {
      // ip is already NULL, no need to repoint
    }
    jp = &particle->particles[particle->nlocal-1];

    nsingle++;
    tally_single[irxn]++;
    last_outcome = 2;
    return irxn + 1;

  } else if (sr.iabsorb >= 0) {
    // ABSORB: ip sticks to surface

    int irxn = sr.iabsorb;
    if (logflag) log_impact(ip, isurf, norm, E_eV, theta_deg, 3, 0.0);
    ip = NULL;

    nsingle++;
    tally_single[irxn]++;
    last_outcome = 3;
    return irxn + 1;
  }

  // no reaction (shouldn't normally reach here if reactions are defined)
  return 0;
}

/* ---------------------------------------------------------------------- */

char *SurfReactPMI::reactionID(int m)
{
  return rlist[m].id;
}

/* ---------------------------------------------------------------------- */

double SurfReactPMI::reaction_coeff(int m)
{
  return rlist[m].coeff[0];
}

/* ---------------------------------------------------------------------- */

int SurfReactPMI::match_reactant(char *species, int m)
{
  for (int i = 0; i < rlist[m].nreactant; i++)
    if (strcmp(species,rlist[m].id_reactants[i]) == 0) return 1;
  return 0;
}

/* ---------------------------------------------------------------------- */

int SurfReactPMI::match_product(char *species, int m)
{
  for (int i = 0; i < rlist[m].nproduct; i++)
    if (strcmp(species,rlist[m].id_products[i]) == 0) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   map species to their PMI reaction roles (reflect/sputter/absorb)
------------------------------------------------------------------------- */

void SurfReactPMI::init_reactions()
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

  // build per-species reaction role lookup

  memory->destroy(species_reactions);
  int nspecies = particle->nspecies;
  species_reactions = memory->create(species_reactions, nspecies,
                                     "surf_react_pmi:species_reactions");

  for (int i = 0; i < nspecies; i++) {
    species_reactions[i].ireflect = -1;
    species_reactions[i].isputter = -1;
    species_reactions[i].iabsorb = -1;
  }

  for (int m = 0; m < nlist_prob; m++) {
    OneReaction *r = &rlist[m];
    if (!r->active) continue;
    int ispec = r->reactants[0];
    if (r->type == REFLECT)  species_reactions[ispec].ireflect = m;
    if (r->type == SPUTTER)  species_reactions[ispec].isputter = m;
    if (r->type == ABSORB)   species_reactions[ispec].iabsorb  = m;
  }
}

/* ----------------------------------------------------------------------
   load HDF5 surface data file with reflection/sputtering tables
------------------------------------------------------------------------- */

void SurfReactPMI::load_surface_file()
{
  H5::H5File file(h5_path, H5F_ACC_RDONLY);

  // read 1D axis arrays
  auto read1D = [&](const std::string& name, std::vector<double>& out, int &n) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t d0;
    sp.getSimpleExtentDims(&d0);
    out.resize(static_cast<size_t>(d0));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    n = static_cast<int>(d0);
  };

  // read 2D table [nE x nA]
  auto read2D = [&](const std::string& name, std::vector<double>& out) {
    H5::DataSet ds = file.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[2];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != nE || static_cast<int>(dims[1]) != nA)
      throw std::runtime_error(
        "surf_react pmi: " + name + " shape mismatch, expected [" +
        std::to_string(nE) + "," + std::to_string(nA) + "]");
    out.resize(static_cast<size_t>(dims[0] * dims[1]));
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
  };

  // helper to check if dataset exists
  auto has_dataset = [&](const std::string& name) -> bool {
    return H5Lexists(file.getId(), name.c_str(), H5P_DEFAULT) > 0;
  };

  read1D("E", E_axis, nE);
  read1D("A", A_axis, nA);
  if (nE < 2 || nA < 2)
    throw std::runtime_error("surf_react pmi: E/A axes must have size >= 2");

  read2D("RN", RN_table);
  read2D("RE", RE_table);
  read2D("spyld", spyld_table);

  // read scalar binding energy
  if (has_dataset("E_bind")) {
    H5::DataSet ds = file.openDataSet("E_bind");
    ds.read(&Ebind, H5::PredType::NATIVE_DOUBLE);
  }

  // read optional tabulated CDF distributions
  has_cdf_R = has_cdf_Y = 0;

  if (has_dataset("edist_R") && has_dataset("adist_R")) {
    H5::DataSet ds = file.openDataSet("edist_R");
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) == nE && static_cast<int>(dims[1]) == nA) {
      nb_cdf = static_cast<int>(dims[2]);
      edist_R.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
      ds.read(edist_R.data(), H5::PredType::NATIVE_DOUBLE);
      adist_R.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
      H5::DataSet ds2 = file.openDataSet("adist_R");
      ds2.read(adist_R.data(), H5::PredType::NATIVE_DOUBLE);
      has_cdf_R = 1;
    }
  }

  if (has_dataset("edist_Y") && has_dataset("adist_Y")) {
    H5::DataSet ds = file.openDataSet("edist_Y");
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[3];
    sp.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) == nE && static_cast<int>(dims[1]) == nA) {
      int nb = static_cast<int>(dims[2]);
      if (nb_cdf == 0) nb_cdf = nb;
      edist_Y.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
      ds.read(edist_Y.data(), H5::PredType::NATIVE_DOUBLE);
      adist_Y.resize(static_cast<size_t>(dims[0]*dims[1]*dims[2]));
      H5::DataSet ds2 = file.openDataSet("adist_Y");
      ds2.read(adist_Y.data(), H5::PredType::NATIVE_DOUBLE);
      has_cdf_Y = 1;
    }
  }

  if (comm->me == 0) {
    if (screen)
      fprintf(screen,"surf_react pmi: loaded %s (%d x %d grid, Ebind=%g eV,"
              " cdf_R=%d cdf_Y=%d)\n",
              h5_path.c_str(), nE, nA, Ebind, has_cdf_R, has_cdf_Y);
    if (logfile)
      fprintf(logfile,"surf_react pmi: loaded %s (%d x %d grid, Ebind=%g eV,"
              " cdf_R=%d cdf_Y=%d)\n",
              h5_path.c_str(), nE, nA, Ebind, has_cdf_R, has_cdf_Y);
  }
}

/* ----------------------------------------------------------------------
   bilinear interpolation on E x A grid with clamping
------------------------------------------------------------------------- */

double SurfReactPMI::interp_table(const std::vector<double> &table,
                                   double e_eV, double a_deg) const
{
  if (table.empty()) return 0.0;

  const double ec = std::min(std::max(e_eV, E_axis.front()), E_axis.back());
  const double ac = std::min(std::max(a_deg, A_axis.front()), A_axis.back());

  auto itE = std::lower_bound(E_axis.begin(), E_axis.end(), ec);
  auto itA = std::lower_bound(A_axis.begin(), A_axis.end(), ac);

  int iE2 = static_cast<int>(itE - E_axis.begin());
  int iA2 = static_cast<int>(itA - A_axis.begin());

  if (iE2 <= 0) iE2 = 1;
  if (iA2 <= 0) iA2 = 1;
  if (iE2 >= nE) iE2 = nE - 1;
  if (iA2 >= nA) iA2 = nA - 1;

  const int iE1 = iE2 - 1;
  const int iA1 = iA2 - 1;

  const double E1 = E_axis[iE1], E2 = E_axis[iE2];
  const double A1 = A_axis[iA1], A2 = A_axis[iA2];

  const double tE = (E2 != E1) ? (ec - E1) / (E2 - E1) : 0.0;
  const double tA = (A2 != A1) ? (ac - A1) / (A2 - A1) : 0.0;

  const double w11 = (1.0 - tE) * (1.0 - tA);
  const double w21 = tE * (1.0 - tA);
  const double w12 = (1.0 - tE) * tA;
  const double w22 = tE * tA;

  auto at = [&](int ie, int ia) -> double {
    return table[static_cast<size_t>(ie) * nA + ia];
  };

  double val = w11*at(iE1,iA1) + w21*at(iE2,iA1) +
               w12*at(iE1,iA2) + w22*at(iE2,iA2);

  if (!std::isfinite(val)) val = 0.0;
  return val;
}

/* ----------------------------------------------------------------------
   Thompson energy distribution sampling
   F(E) ~ E / (E + Eb)^3 for 0 < E < Emax
------------------------------------------------------------------------- */

double SurfReactPMI::sample_thompson(double Eb, double Emax)
{
  if (Eb <= 0.0 || Emax <= 0.0) return 0.0;

  double emu = 1.0 / (Emax / Eb + 1.0);
  double betad2 = 1.0 / (emu * emu - 2.0 * emu + 1.0);

  double r = random->uniform();
  double arg = r / betad2;
  double energy = Eb / (1.0 - sqrt(arg)) - Eb;

  if (energy < 0.0) energy = 0.0;
  if (energy > Emax) energy = Emax;
  return energy;
}

/* ----------------------------------------------------------------------
   load an EIRENE TRIM reflection table on demand.
   Returns index into trim_tables or -1 on failure.
------------------------------------------------------------------------- */

int SurfReactPMI::load_or_get_trim_table(const char *name)
{
  std::string sname(name);
  auto it = trim_index.find(sname);
  if (it != trim_index.end()) return it->second;

  if (trim_dir.empty()) return -1;

  std::string path = trim_dir + "/" + sname + ".h5";

  try {
    H5::H5File file(path, H5F_ACC_RDONLY);

    auto read_1d = [&](const std::string &ds, std::vector<double> &out) {
      H5::DataSet d = file.openDataSet(ds);
      H5::DataSpace sp = d.getSpace();
      hsize_t dim;
      sp.getSimpleExtentDims(&dim);
      out.resize(static_cast<size_t>(dim));
      d.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };
    auto read_flat = [&](const std::string &ds, std::vector<double> &out,
                         size_t expected) {
      H5::DataSet d = file.openDataSet(ds);
      H5::DataSpace sp = d.getSpace();
      int rank = sp.getSimpleExtentNdims();
      std::vector<hsize_t> dims(rank);
      sp.getSimpleExtentDims(dims.data());
      size_t total = 1;
      for (int r = 0; r < rank; r++) total *= dims[r];
      if (total != expected) {
        throw std::runtime_error(
          "surf_react pmi TRIM: " + ds + " has " + std::to_string(total) +
          " elements, expected " + std::to_string(expected));
      }
      out.resize(total);
      d.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    };

    Reflection::Table t;
    t.name = sname;

    read_1d("E",     t.E_grid);
    read_1d("theta", t.theta_grid);
    read_1d("raar",  t.raar);

    const int nE = Reflection::NE;
    const int nW = Reflection::NTHETA;
    const int nR = Reflection::NQ;

    if ((int)t.E_grid.size() != nE || (int)t.theta_grid.size() != nW ||
        (int)t.raar.size() != nR) {
      throw std::runtime_error(
        "surf_react pmi TRIM: axis shape mismatch in " + path);
    }

    read_flat("R_N",         t.R_N,         (size_t)nE * nW);
    read_flat("Eout_q",      t.Eout_q,      (size_t)nE * nW * nR);
    read_flat("Eout_min",    t.Eout_min,    (size_t)nE * nW);
    read_flat("Eout_max",    t.Eout_max,    (size_t)nE * nW);
    read_flat("cos_polar_q", t.cos_polar_q, (size_t)nE * nW * nR * nR);
    read_flat("cos_azim_q",  t.cos_azim_q,  (size_t)nE * nW * nR * nR * nR);

    // optional header attributes
    t.Z1 = t.M1 = t.Z2 = t.M2 = 0.0;
    auto read_attr = [&](const std::string &aname, double &val) {
      if (file.attrExists(aname)) {
        H5::Attribute a = file.openAttribute(aname);
        a.read(H5::PredType::NATIVE_DOUBLE, &val);
      }
    };
    read_attr("Z1", t.Z1); read_attr("M1", t.M1);
    read_attr("Z2", t.Z2); read_attr("M2", t.M2);

    int idx = static_cast<int>(trim_tables.size());
    trim_tables.push_back(std::move(t));
    trim_index[sname] = idx;

    if (comm->me == 0) {
      fprintf(screen ? screen : logfile,
              "surf_react pmi: loaded TRIM table '%s' from %s\n",
              sname.c_str(), path.c_str());
    }
    return idx;
  } catch (const std::exception &e) {
    if (comm->me == 0)
      fprintf(screen ? screen : logfile,
              "surf_react pmi TRIM load error for '%s' (%s): %s\n",
              sname.c_str(), path.c_str(), e.what());
    return -1;
  } catch (...) {
    return -1;
  }
}

/* ----------------------------------------------------------------------
   cosine-law hemisphere direction sampling
   samples direction in outward hemisphere defined by norm
   sets dir[] to direction vector with magnitude vmag
------------------------------------------------------------------------- */

void SurfReactPMI::sample_cosine_direction(double *norm, double *dir,
                                            double vmag)
{
  // build tangent frame from surface normal
  double tangent1[3], tangent2[3];

  // pick an arbitrary vector not parallel to norm
  double arb[3];
  if (fabs(norm[0]) < 0.9) {
    arb[0] = 1.0; arb[1] = 0.0; arb[2] = 0.0;
  } else {
    arb[0] = 0.0; arb[1] = 1.0; arb[2] = 0.0;
  }

  MathExtra::cross3(norm, arb, tangent1);
  MathExtra::norm3(tangent1);
  MathExtra::cross3(norm, tangent1, tangent2);

  // cosine-law: cos(theta) = sqrt(xi1) for diffuse emission
  double xi1 = random->uniform();
  double xi2 = random->uniform();

  double cosTheta = sqrt(xi1);
  double sinTheta = sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
  double phi = MathConst::MY_2PI * xi2;

  double cosPhi = cos(phi);
  double sinPhi = sin(phi);

  dir[0] = sinTheta * cosPhi * tangent1[0] +
           sinTheta * sinPhi * tangent2[0] +
           cosTheta * norm[0];
  dir[1] = sinTheta * cosPhi * tangent1[1] +
           sinTheta * sinPhi * tangent2[1] +
           cosTheta * norm[1];
  dir[2] = sinTheta * cosPhi * tangent1[2] +
           sinTheta * sinPhi * tangent2[2] +
           cosTheta * norm[2];

  MathExtra::norm3(dir);

  dir[0] *= vmag;
  dir[1] *= vmag;
  dir[2] *= vmag;
}

/* ----------------------------------------------------------------------
   inverse-CDF sampling from a tabulated CDF
   cdf is [nE x nA x nb_cdf], returns sampled bin center value
------------------------------------------------------------------------- */

double SurfReactPMI::sample_from_cdf(const std::vector<double> &cdf,
                                      int iE, int iA)
{
  if (cdf.empty() || nb_cdf <= 0) return 0.0;

  double u = random->uniform();

  // offset into the CDF for this (iE, iA) pair
  size_t base = static_cast<size_t>(iE) * nA * nb_cdf +
                static_cast<size_t>(iA) * nb_cdf;

  // binary search for the bin where CDF >= u
  int lo = 0, hi = nb_cdf - 1;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (cdf[base + mid] < u)
      lo = mid + 1;
    else
      hi = mid;
  }

  // linear interpolation within the bin
  double cdf_lo = (lo > 0) ? cdf[base + lo - 1] : 0.0;
  double cdf_hi = cdf[base + lo];
  double frac = (cdf_hi > cdf_lo) ? (u - cdf_lo) / (cdf_hi - cdf_lo) : 0.5;

  // return normalized bin center (0 to 1 range, caller scales)
  return (lo + frac) / nb_cdf;
}

/* ---------------------------------------------------------------------- */

void SurfReactPMI::readfile(char *fname)
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

    // parse reaction type: R=Reflect, S=Sputter, A=Absorb

    word = strtok(line2," \t\n");
    if (!word) error->all(FLERR,"Invalid reaction type in file");
    if (word[0] == 'R' || word[0] == 'r') r->type = REFLECT;
    else if (word[0] == 'S' || word[0] == 's') r->type = SPUTTER;
    else if (word[0] == 'A' || word[0] == 'a') r->type = ABSORB;
    else error->all(FLERR,
      "Invalid PMI reaction type in file (must be R, S, or A)");

    // validate reactant/product counts for each type

    if (r->type == REFLECT) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR,
          "PMI Reflect reaction must have 1 reactant and 1 product");
    } else if (r->type == SPUTTER) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR,
          "PMI Sputter reaction must have 1 reactant and 1 product");
    } else if (r->type == ABSORB) {
      if (r->nreactant != 1 || r->nproduct != 0)
        error->all(FLERR,
          "PMI Absorb reaction must have 1 reactant and 0 products");
    }

    word = strtok(NULL," \t\n");
    if (!word) error->all(FLERR,"Invalid reaction style in file");
    // Accept "Simple ...", "Eckstein <shorthand>", or "Trim <shorthand>"
    if (strcasecmp(word,"Eckstein") == 0) {
      r->style = ECKSTEIN;
    } else if (strcasecmp(word,"Trim") == 0) {
      r->style = TRIM_STYLE;
    } else if (strcasecmp(word,"Simple") == 0) {
      r->style = SIMPLE;
    } else {
      error->all(FLERR,"Invalid reaction style in file "
                       "(expected 'Simple', 'Eckstein', or 'Trim')");
    }

    if (r->style == SIMPLE) {
      r->ncoeff = 2;
      for (int i = 0; i < r->ncoeff; i++) {
        word = strtok(NULL," \t\n");
        if (!word) {
          if (i == 0) error->all(FLERR,"Invalid reaction coefficients in file");
          else r->coeff[i] = 0.0;
        } else {
          r->coeff[i] = input->numeric(FLERR,word);
        }
      }
      word = strtok(NULL," \t\n");
      if (word) error->all(FLERR,"Too many coefficients in a reaction formula");

    } else if (r->style == ECKSTEIN) {
      // ECKSTEIN: next token is a shorthand name like "W_on_W"
      word = strtok(NULL," \t\n");
      if (!word)
        error->all(FLERR,"Eckstein reaction requires a shorthand name "
                         "(e.g. W_on_W)");
      r->ncoeff = Eckstein::NCOEFF_ECKSTEIN;

      if (r->type == SPUTTER) {
        Eckstein::SputterParams sp;
        if (!Eckstein::lookup_sputter(word,sp)) {
          char str[256];
          snprintf(str,sizeof(str),
                   "Unknown Eckstein sputter shorthand '%s' in reactions file",
                   word);
          error->all(FLERR,str);
        }
        Eckstein::pack_sputter(sp,r->coeff);
      } else if (r->type == REFLECT) {
        Eckstein::ReflectParams rp;
        if (!Eckstein::lookup_reflect(word,rp)) {
          char str[256];
          snprintf(str,sizeof(str),
                   "Unknown Eckstein reflect shorthand '%s' in reactions file",
                   word);
          error->all(FLERR,str);
        }
        Eckstein::pack_reflect(rp,r->coeff);
      } else {
        error->all(FLERR,"Eckstein style only valid for Reflect or Sputter "
                         "reactions");
      }

      word = strtok(NULL," \t\n");
      if (word) error->all(FLERR,"Too many coefficients after Eckstein "
                                 "shorthand");

    } else {
      // TRIM_STYLE: next token is an EIRENE TRIM combination name like
      // "W_on_W"; reflection data is loaded lazily from trim_dir/<name>.h5.
      // Only Reflect reactions may use Trim style.
      if (r->type != REFLECT)
        error->all(FLERR,"Trim style is only valid for Reflect reactions "
                         "(use Eckstein for sputtering)");
      word = strtok(NULL," \t\n");
      if (!word)
        error->all(FLERR,"Trim reaction requires a shorthand name "
                         "(e.g. W_on_W)");
      int idx = load_or_get_trim_table(word);
      if (idx < 0) {
        char str[256];
        snprintf(str,sizeof(str),
                 "Cannot load Trim table for '%s' from trim_dir='%s'. "
                 "Check that the surf_react pmi command specified trim_dir, "
                 "and that <name>.h5 exists there.",
                 word, trim_dir.c_str());
        error->all(FLERR,str);
      }
      r->ncoeff = 1;
      r->coeff[0] = static_cast<double>(idx);
      word = strtok(NULL," \t\n");
      if (word) error->all(FLERR,"Too many arguments after Trim shorthand");
    }

    // prepend type tag to reaction ID for unambiguous tally output
    {
      const char *tag = (r->type == REFLECT) ? "R: " :
                        (r->type == SPUTTER) ? "S: " : "A: ";
      int old_len = strlen(r->id);
      int tag_len = strlen(tag);
      char *new_id = new char[tag_len + old_len + 1];
      strcpy(new_id, tag);
      strcat(new_id, r->id);
      delete [] r->id;
      r->id = new_id;
    }

    nlist_prob++;
  }

  if (comm->me == 0) fclose(fp);
}

/* ----------------------------------------------------------------------
   read one reaction from file
   reaction = 2 lines
   return 1 if end-of-file, else return 0
------------------------------------------------------------------------- */

int SurfReactPMI::readone(char *line1, char *line2, int &n1, int &n2)
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

/* ----------------------------------------------------------------------
   open per-rank CSV log file for impact logging
------------------------------------------------------------------------- */

void SurfReactPMI::open_logfile()
{
  if (!logflag || logfp) return;

  const char *base = logfile_base ? logfile_base : "surf_react_pmi.csv";
  int nbase = strlen(base);
  char *filename = new char[nbase + 32];
  snprintf(filename, nbase + 32, "%s.rank%d", base, me);

  logfp = fopen(filename, "w");
  delete [] filename;
  if (!logfp) return;

  fprintf(logfp,
          "timestep,rank,surf_id,species,E_in_eV,theta_deg,"
          "outcome,E_out_eV,x,y,z,vx,vy,vz,nx,ny,nz\n");
}

/* ---------------------------------------------------------------------- */

void SurfReactPMI::close_logfile()
{
  if (logfp) {
    fclose(logfp);
    logfp = NULL;
  }
}

/* ----------------------------------------------------------------------
   log a single PMI impact event to CSV
   outcome: 1=reflect, 2=sputter, 3=absorb
------------------------------------------------------------------------- */

void SurfReactPMI::log_impact(Particle::OnePart *ip, int isurf, double *norm,
                               double E_eV, double theta_deg, int outcome,
                               double E_out_eV)
{
  if (!logfp && logflag) open_logfile();
  if (!logfp || !ip) return;

  long long surf_id = -1;
  if (isurf >= 0) {
    if (domain->dimension == 2) surf_id = (long long) surf->lines[isurf].id;
    else surf_id = (long long) surf->tris[isurf].id;
  }

  const char *tag = (outcome == 1) ? "R" : (outcome == 2) ? "S" : "A";
  const char *sname = particle->species[ip->ispecies].id;

  fprintf(logfp,
          "%lld,%d,%lld,%s,%.6g,%.4f,%s,%.6g,"
          "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
          (long long) update->ntimestep, me, surf_id, sname,
          E_eV, theta_deg, tag, E_out_eV,
          ip->x[0], ip->x[1], ip->x[2],
          ip->v[0], ip->v[1], ip->v[2],
          norm ? norm[0] : 0.0, norm ? norm[1] : 0.0, norm ? norm[2] : 0.0);
}
