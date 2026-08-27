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
#include "domain.h"
#include "comm.h"
#include "particle.h"
#include "modify.h"
#include "compute.h"
#include "surf.h"
#include "random_mars.h"
#include "random_knuth.h"
#include "database_paths.h"
#include "process_library.h"
#include "math_extra.h"
#include "memory.h"
#include "error.h"
#include "eckstein_sputter.h"
#include "eckstein_sputter_data.h"

#include <stdexcept>
#include "H5Cpp.h"

using namespace SPARTA_NS;

enum{DISSOCIATION,EXCHANGE,RECOMBINATION,TRIM_REFLECT,ABSORB_REEMIT,SPUTTER};
enum{INT,DOUBLE};                        // match surf.cpp custom-attribute type codes

// Thompson sputtered-atom energy with the recoil (max-transferable) cutoff:
//   f(E) proportional to E/(E+Ub)^3 * [1 - sqrt((E+Ub)/(Emax+Ub))],  0<=E<=Emax
// where Emax = gamma*E_in - Ub is the maximum ejection energy (gamma = the
// binary energy-transfer factor 4 M1 M2/(M1+M2)^2).
// Proposal = sharp-truncated Thompson via its exact inverse CDF: f has
// CDF F(E) = (E/(E+Ub))^2, so E = Ub*s/(1-s) with s = sqrt(v),
// v ~ U(0, vmax), vmax = (Emax/(Emax+Ub))^2; then rejection with the
// smooth-cutoff factor. NOTE: the old proposal E = Ub(1/sqrt(u)-1)
// inverted 1-F of f ~ 1/(E+Ub)^3 (missing the leading E) and biased
// emission energies soft (median 0.41*Ub instead of ~2.4*Ub).
static inline double pwi_sample_thompson(double Ub_eV, double Emax_eV,
                                         RanKnuth *rng) {
  double denom = Emax_eV + Ub_eV;
  double rmax = Emax_eV / denom;
  double vmax = rmax * rmax;
  for (int it = 0; it < 64; it++) {
    double s = sqrt(rng->uniform() * vmax);
    double E = Ub_eV * s / (1.0 - s);
    if (rng->uniform() < 1.0 - sqrt((E + Ub_eV) / denom)) return E;
  }
  return 0.5 * Emax_eV;                              // rare fallback
}

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
  sigma_attr = NULL;
  sindex_custom = -1;
  sigma_nevery = 1;
  sigma_ncols = 0;
  sigma_nsurf = 0;
  sigma_delta = NULL;
  sigma_buf = NULL;
  sigma_area = NULL;

  snet_index = sdep_index = sero_index = -1;
  strata_K = 0;
  strata_minthick = 0.5;
  rough_dm = 0.0;
  strata_index = -1;
  strata_ncols = 0;
  nmat = 0;
  sigma_zone = 1.0e19;
  sigma_feedback = 1;
  sconc_index = -1;
  substrate_isp = -1;
  dep_delta = dep_buf = NULL;
  trim_dir.clear();
  ehist_file = NULL;
  ehist_all = ehist_sput = ahist_all = ahist_sput = NULL;
  ehist_z = NULL; ehist_nsp = 0;
  ehist_nbin = 0; ehist_every = 0; ehist_emax = 0.0;

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
    } else if (strcmp(arg[iarg],"ehist") == 0) {
      // impact histograms: ehist <file> <nbins_E> <emax_eV> <every>
      if (iarg+5 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      int n = strlen(arg[iarg+1]) + 1;
      ehist_file = new char[n];
      strcpy(ehist_file,arg[iarg+1]);
      ehist_nbin = input->inumeric(FLERR,arg[iarg+2]);
      ehist_emax = input->numeric(FLERR,arg[iarg+3]);
      ehist_every = input->inumeric(FLERR,arg[iarg+4]);
      if (ehist_nbin < 1 || ehist_emax <= 0.0 || ehist_every < 1)
        error->all(FLERR,"surf_react surface/pwi ehist: bad parameters");
      ehist_all = new double[ehist_nbin]();
      ehist_sput = new double[ehist_nbin]();
      ahist_all = new double[EHIST_NANG]();
      ahist_sput = new double[EHIST_NANG]();
      // per-species impact-E histograms (species must be defined first)
      ehist_nsp = particle->nspecies;
      if (ehist_nsp > 0) {
        ehist_z = new double*[ehist_nsp];
        for (int k = 0; k < ehist_nsp; k++)
          ehist_z[k] = new double[ehist_nbin]();
      }
      iarg += 5;
    } else if (strcmp(arg[iarg],"adens_surf") == 0 ||
               strcmp(arg[iarg],"sigma_surf") == 0) {
      // WallDYN-style areal-density accounting (adens = areal density).
      // Creates (or reuses) a per-surf custom DOUBLE array with one column
      // per species [atoms/m^2]; retained particles add +pweight/area to
      // their species column (influx), sputtered atoms add -pweight/area
      // to the sputtered species column (erosion flux).
      // sigma_* keyword spellings are accepted as legacy aliases.
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      int n = strlen(arg[iarg+1]) + 1;
      sigma_attr = new char[n];
      strcpy(sigma_attr, arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"adens_erosion") == 0 ||
               strcmp(arg[iarg],"sigma_erosion") == 0) {
      // background gross-erosion flux: adens_erosion <computeID> <col>
      // <species> [noconc]. col 0 = compute's vector_surf, col N =
      // array_surf column N. noconc: skip the concentration-limiting
      // factor -- for computes that already fold the composition in
      // (compound-table mode).
      if (iarg+4 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      sigma_ero_id.push_back(arg[iarg+1]);
      sigma_ero_col.push_back(input->inumeric(FLERR,arg[iarg+2]));
      sigma_ero_species.push_back(arg[iarg+3]);
      iarg += 4;
      if (iarg < narg && strcmp(arg[iarg],"noconc") == 0) {
        sigma_ero_noconc.push_back(1);
        iarg++;
      } else sigma_ero_noconc.push_back(0);
    } else if (strcmp(arg[iarg],"adens_init") == 0 ||
               strcmp(arg[iarg],"sigma_init") == 0) {
      // initial areal density for a species (e.g. a boronization layer),
      // applied once when the areal-density attribute is first created
      if (iarg+3 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      sigma_init_names.push_back(arg[iarg+1]);
      sigma_init_vals.push_back(input->numeric(FLERR,arg[iarg+2]));
      iarg += 3;
    } else if (strcmp(arg[iarg],"rzone") == 0 ||
               strcmp(arg[iarg],"sigma_zone") == 0) {
      // reaction-zone total areal density [atoms/m^2] for surface
      // concentrations (WallDYN's RZoneWidth is the same quantity given
      // as a thickness in Angstrom; ours is the areal density directly)
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      sigma_zone = input->numeric(FLERR,arg[iarg+1]);
      if (sigma_zone <= 0.0)
        error->all(FLERR,"surf_react surface/pwi rzone must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg],"conc_feedback") == 0 ||
               strcmp(arg[iarg],"sigma_feedback") == 0) {
      // yes (default): 'mat' channels weighted by reaction-zone
      // concentrations; no: weights forced to 1 (feedback off, A/B runs)
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      if (strcmp(arg[iarg+1],"yes") == 0) sigma_feedback = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) sigma_feedback = 0;
      else error->all(FLERR,"surf_react surface/pwi conc_feedback must be yes/no");
      iarg += 2;
    } else if (strcmp(arg[iarg],"roughness_dm") == 0) {
      // mean surface angle delta_m [deg]: yields evaluated at
      // theta_eff = max(0, theta - delta_m)
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      rough_dm = input->numeric(FLERR,arg[iarg+1]);
      if (rough_dm < 0.0 || rough_dm >= 90.0)
        error->all(FLERR,"surf_react surface/pwi roughness_dm must be in [0,90)");
      iarg += 2;
    } else if (strcmp(arg[iarg],"strata") == 0) {
      // opt-in Lagrangian strata stack: `strata <K>` with K = max
      // strata per element (per MATERIAL composition). Absent = the
      // homogeneous reaction-zone model, completely unchanged.
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      strata_K = input->inumeric(FLERR,arg[iarg+1]);
      if (strata_K < 2 || strata_K > 64)
        error->all(FLERR,"surf_react surface/pwi strata K must be in 2..64");
      iarg += 2;
      // optional: minthick <Angstrom> — lower for compressed-timescale
      // test configs whose artificial layers are sub-Angstrom
      if (iarg+1 < narg && strcmp(arg[iarg],"minthick") == 0) {
        strata_minthick = input->numeric(FLERR,arg[iarg+1]);
        if (strata_minthick <= 0.0)
          error->all(FLERR,"surf_react surface/pwi strata minthick must be > 0");
        iarg += 2;
      }
    } else if (strcmp(arg[iarg],"adens_implant") == 0) {
      // background implantation + retention saturation (see header)
      if (iarg+3 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      imp_species.push_back(arg[iarg+1]);
      iarg += 2;
      if (strcmp(arg[iarg],"compute") == 0) {
        if (iarg+3 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
        imp_mode.push_back(0);
        imp_comp_id.push_back(arg[iarg+1]);
        imp_col.push_back(input->inumeric(FLERR,arg[iarg+2]));
        imp_flux.push_back(0.0);
        iarg += 3;
      } else if (strcmp(arg[iarg],"const") == 0) {
        if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
        imp_mode.push_back(1);
        imp_comp_id.push_back("");
        imp_col.push_back(0);
        imp_flux.push_back(input->numeric(FLERR,arg[iarg+1]));
        iarg += 2;
      } else error->all(FLERR,"adens_implant needs compute <ID> <col> or const <flux>");
      imp_rcoef.push_back(0.0);
      imp_cmax.push_back(-1.0);
      imp_alpha.push_back(100.0);
      imp_depth.push_back(0.0);
      while (iarg+1 < narg) {
        if      (strcmp(arg[iarg],"rcoef") == 0) { imp_rcoef.back() = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
        else if (strcmp(arg[iarg],"cmax")  == 0) { imp_cmax.back()  = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
        else if (strcmp(arg[iarg],"alpha") == 0) { imp_alpha.back() = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
        else if (strcmp(arg[iarg],"depth") == 0) { imp_depth.back() = input->numeric(FLERR,arg[iarg+1]); iarg += 2; }
        else break;
      }
      if (imp_cmax.back() <= 0.0)
        error->all(FLERR,"adens_implant requires cmax <c_sat> (physics "
                   "choice, no default; ~0.1 for D in W at wall temps)");
    } else if (strcmp(arg[iarg],"strata_dens") == 0) {
      // solid number density override for a material:
      // strata_dens <species> <atoms/m^3>  (repeatable; any charge state
      // of the material selects it). Defaults are built in per element.
      if (iarg+3 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      strata_dens_names.push_back(arg[iarg+1]);
      strata_dens_vals.push_back(input->numeric(FLERR,arg[iarg+2]));
      if (strata_dens_vals.back() <= 0.0)
        error->all(FLERR,"surf_react surface/pwi strata_dens must be > 0");
      iarg += 3;
    } else if (strcmp(arg[iarg],"adens_nevery") == 0 ||
               strcmp(arg[iarg],"sigma_nevery") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal surf_react surface/pwi command");
      sigma_nevery = input->inumeric(FLERR,arg[iarg+1]);
      if (sigma_nevery < 1)
        error->all(FLERR,"surf_react surface/pwi adens_nevery must be >= 1");
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
  // Kokkos KKCopy shallow copy: base state is owned by the host original;
  // freeing it here would double-free the reaction tables
  if (copy) return;

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
      delete [] r->mat_id;
      delete [] r->conc_id;
    }
    memory->sfree(rlist);
  }
  memory->destroy(reactions);
  memory->destroy(indices);
  delete random;
  delete [] twall_attr;
  delete [] R_attr;
  delete [] sigma_attr;
  delete [] ehist_file;
  delete [] ehist_all;
  delete [] ehist_sput;
  delete [] ahist_all;
  delete [] ahist_sput;
  if (ehist_z) {
    for (int k = 0; k < ehist_nsp; k++) delete [] ehist_z[k];
    delete [] ehist_z;
  }
  memory->destroy(sigma_delta);
  memory->destroy(sigma_buf);
  memory->destroy(sigma_area);
  memory->destroy(dep_delta);
  memory->destroy(dep_buf);
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWI::init()
{
  SurfReact::init();
  init_reactions();

  // pweight custom (fix particle/weight): sputtered atoms inherit the
  // incident macroparticle's pweight. -1 if fix particle/weight is absent
  // (then sputtered atoms fall back to the fnum default).
  // store the custom INDEX (stable while the attribute exists) and resolve
  // ewhich at use time: other modules (sheath, plasma cache) add particle
  // customs after this init, which can invalidate a cached ewhich value.
  pweight_ewhich = particle->find_custom((char *) "pweight");

  // composition-weighted (mat/conc) channels need the sigma mixing zone
  for (int m = 0; m < nlist_recycle; m++)
    if (rlist[m].active && (rlist[m].mat_id || rlist[m].conc_id) && !sigma_attr)
      error->all(FLERR,"surf_react surface/pwi: 'mat'/'conc' reaction channels "
                 "require adens_surf (reaction-zone concentrations)");

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

  // bind or create the per-surf areal-density (sigma) custom array.
  // One column per species, units atoms/m^2 (2D planar: atoms/m^2 per m
  // of depth; axisymmetric and 3D: true areal density). The attribute is
  // created here (not the constructor) so nspecies is final, and reused
  // if it already exists (e.g. from a restart file, which carries custom
  // attributes) so sigma accumulates across restarts.
  if (sigma_attr) {
    if (surf->implicit)
      error->all(FLERR,"surf_react surface/pwi adens_surf requires explicit surfs");
    if (surf->distributed)
      error->all(FLERR,"surf_react surface/pwi adens_surf does not yet support "
                 "distributed surfs");

    sigma_ncols = particle->nspecies;
    sindex_custom = surf->find_custom(sigma_attr);
    if (sindex_custom < 0) {
      sindex_custom = surf->add_custom(sigma_attr, DOUBLE, sigma_ncols);
      // apply initial layers (only on creation: restarts keep their state)
      if (sigma_init_names.size()) {
        double **sig0 = surf->edarray[surf->ewhich[sindex_custom]];
        for (size_t k = 0; k < sigma_init_names.size(); k++) {
          int isp = particle->find_species((char *) sigma_init_names[k].c_str());
          if (isp < 0)
            error->all(FLERR,"surf_react surface/pwi adens_init: unknown species");
          for (int i = 0; i < surf->nown; i++) sig0[i][isp] = sigma_init_vals[k];
        }
      }
    } else {
      if (surf->etype[sindex_custom] != DOUBLE)
        error->all(FLERR,"surf_react surface/pwi adens_surf attribute must be DOUBLE");
      if (surf->esize[sindex_custom] != sigma_ncols)
        error->all(FLERR,"surf_react surface/pwi adens_surf attribute must have "
                   "one column per species");
    }
    surf->spread_custom(sindex_custom);

    // local delta ledger + allreduce buffer, indexed by (surfID-1)*ncols + isp

    sigma_nsurf = surf->nsurf;
    bigint ntally = sigma_nsurf * (bigint) sigma_ncols;
    if (ntally > MAXSMALLINT)
      error->all(FLERR,"surf_react surface/pwi sigma_surf: too many surfs*species");
    memory->destroy(sigma_delta);
    memory->destroy(sigma_buf);
    memory->create(sigma_delta,(int) ntally,"surf_react_pwi:sigma_delta");
    memory->create(sigma_buf,(int) ntally,"surf_react_pwi:sigma_buf");
    for (int i = 0; i < (int) ntally; i++) sigma_delta[i] = 0.0;

    // derived scalar customs: <attr>_net (row sum incl. debits) and
    // <attr>_dep (gross deposition: positive credits only)
    char dname[130];
    snprintf(dname,sizeof(dname),"%s_net",sigma_attr);
    snet_index = surf->find_custom(dname);
    if (snet_index < 0) snet_index = surf->add_custom(dname, DOUBLE, 0);
    snprintf(dname,sizeof(dname),"%s_dep",sigma_attr);
    sdep_index = surf->find_custom(dname);
    if (sdep_index < 0) sdep_index = surf->add_custom(dname, DOUBLE, 0);
    snprintf(dname,sizeof(dname),"%s_conc",sigma_attr);
    sconc_index = surf->find_custom(dname);
    if (sconc_index < 0) sconc_index = surf->add_custom(dname, DOUBLE, sigma_ncols);
    surf->spread_custom(sconc_index);
    if (substrate_isp < 0) {
      substrate_isp = particle->find_species((char *) "W");
      if (substrate_isp < 0) substrate_isp = 0;
    }

    // group species into materials by ELEMENT (strip charge-state suffix:
    // W, W+, W2+ .. are one tungsten material; B, B+ .. one boron material)
    mat_of.assign(sigma_ncols, 0);
    {
      std::vector<std::string> elems(sigma_ncols);
      for (int j = 0; j < sigma_ncols; j++) {
        std::string nm = particle->species[j].id;
        size_t e = 0;
        while (e < nm.size() && isalpha((unsigned char) nm[e])) e++;
        elems[j] = nm.substr(0, e);
      }
      for (int j = 0; j < sigma_ncols; j++) {
        mat_of[j] = j;
        for (int k = 0; k < j; k++)
          if (elems[k] == elems[j]) { mat_of[j] = mat_of[k]; break; }
      }
    }
    snprintf(dname,sizeof(dname),"%s_ero",sigma_attr);
    sero_index = surf->find_custom(dname);
    if (sero_index < 0) sero_index = surf->add_custom(dname, DOUBLE, 0);
    surf->spread_custom(sero_index);
    surf->spread_custom(snet_index);
    surf->spread_custom(sdep_index);

    // opt-in strata stack: build per-material bookkeeping and initial
    // stacks (substrate + adens_init layers) BEFORE the first conc
    // derivation so consumers see the stratified surface from step 0
    if (strata_K > 0) {
      // material roots (mat_of[j] == j) and species->material index
      std::vector<int> roots;
      for (int j = 0; j < sigma_ncols; j++)
        if (mat_of[j] == j) roots.push_back(j);
      nmat = (int) roots.size();
      mat_root_of.assign(sigma_ncols, 0);
      for (int j = 0; j < sigma_ncols; j++)
        for (int m = 0; m < nmat; m++)
          if (mat_of[j] == roots[m]) { mat_root_of[j] = m; break; }

      // solid number densities by element symbol [atoms/m^3]
      mat_dens.assign(nmat, 6.3e28);
      for (int m = 0; m < nmat; m++) {
        std::string nm = particle->species[roots[m]].id;
        size_t e = 0;
        while (e < nm.size() && isalpha((unsigned char) nm[e])) e++;
        std::string el = nm.substr(0, e);
        if      (el == "W")  mat_dens[m] = 6.306e28;
        else if (el == "B")  mat_dens[m] = 1.32e29;
        else if (el == "O")  mat_dens[m] = 4.29e28;
        else if (el == "C")  mat_dens[m] = 1.13e29;
        else if (el == "Li") mat_dens[m] = 4.63e28;
        else if (el == "Be") mat_dens[m] = 1.23e29;
        else if (comm->me == 0)
          error->warning(FLERR,"surf_react surface/pwi strata: unknown "
                         "element solid density, using 6.3e28 m^-3 "
                         "(override with strata_dens)");
      }
      // deck overrides win over the built-in element defaults
      for (size_t k = 0; k < strata_dens_names.size(); k++) {
        int isp = particle->find_species((char *) strata_dens_names[k].c_str());
        if (isp < 0)
          error->all(FLERR,"surf_react surface/pwi strata_dens: unknown species");
        mat_dens[mat_root_of[isp]] = strata_dens_vals[k];
      }

      strata_ncols = 4 + strata_K * (2 + nmat);
      int sub_m = mat_root_of[substrate_isp];

      strata_state.assign(surf->nown, SurfaceElementState(nmat, strata_K));
      // deck minthick is in Angstrom (documented); internal state is SI
      for (auto &st : strata_state) st.min_thickness = strata_minthick * 1.0e-10;
      char sname[130];
      snprintf(sname,sizeof(sname),"%s_strata",sigma_attr);
      strata_index = surf->find_custom(sname);
      bool fresh = (strata_index < 0);
      if (fresh) strata_index = surf->add_custom(sname, DOUBLE, strata_ncols);
      double **sb = surf->edarray[surf->ewhich[strata_index]];
      if (fresh) {
        for (int i = 0; i < surf->nown; i++) {
          // semi-infinite substrate (1 mm) + initial deposited layers
          strata_state[i].init_substrate(1.0e-3, mat_dens[sub_m], sub_m);
          for (size_t k = 0; k < sigma_init_names.size(); k++) {
            int isp = particle->find_species((char *) sigma_init_names[k].c_str());
            if (isp < 0) continue;
            int m = mat_root_of[isp];
            strata_state[i].deposit(m, sigma_init_vals[k], mat_dens[m]);
          }
          for (int c = 0; c < strata_ncols; c++) sb[i][c] = 0.0;
          // cap the stack at max_layers BEFORE packing: with several
          // adens_init layers the fresh stack can exceed K and pack()
          // would write past the row (heap corruption, F2)
          strata_state[i].compact_layers();
          strata_state[i].pack(sb[i]);
        }
      } else {
        // restart: rebuild stacks from the persisted custom
        for (int i = 0; i < surf->nown; i++)
          strata_state[i].unpack(sb[i]);
      }
      surf->estatus[strata_index] = 0;
      surf->spread_custom(strata_index);
    }

    // seed the concentration array from the current sigma so the first
    // debit/emission read (before the first sync) sees the initial layer
    derive_sigma_conc();
    memory->destroy(dep_delta);
    memory->destroy(dep_buf);
    memory->create(dep_delta,(int) sigma_nsurf,"surf_react_pwi:dep_delta");
    memory->create(dep_buf,(int) sigma_nsurf,"surf_react_pwi:dep_buf");
    for (int i = 0; i < (int) sigma_nsurf; i++) dep_delta[i] = 0.0;

    // per-surf element area, indexed by local surf index (same indexing as
    // react()'s isurf). Mirrors compute_surf::init_normflux: line length in
    // planar 2D, revolved area in axisymmetric, triangle area in 3D.

    int nslocal = surf->nlocal + surf->nghost;
    memory->destroy(sigma_area);
    memory->create(sigma_area,nslocal,"surf_react_pwi:sigma_area");
    int dim = domain->dimension;
    int axisymmetric = domain->axisymmetric;
    double tmp;
    for (int i = 0; i < nslocal; i++) {
      if (dim == 3) sigma_area[i] = surf->tri_size(i,tmp);
      else if (axisymmetric) sigma_area[i] = surf->axi_line_size(i);
      else sigma_area[i] = surf->line_size(i);
    }

    // bind the gross-erosion computes + target species columns
    sigma_ero_compute.clear();
    sigma_ero_isp.clear();
    for (size_t k = 0; k < sigma_ero_id.size(); k++) {
      int ic = modify->find_compute((char *) sigma_ero_id[k].c_str());
      if (ic < 0)
        error->all(FLERR,"surf_react surface/pwi adens_erosion: compute ID not found");
      Compute *ce = modify->compute[ic];
      if (!ce->per_surf_flag)
        error->all(FLERR,"surf_react surface/pwi adens_erosion: compute is not per-surf");
      if (sigma_ero_col[k] == 0 && ce->size_per_surf_cols != 0)
        error->all(FLERR,"surf_react surface/pwi adens_erosion: compute produces an "
                   "array; give a column number");
      if (sigma_ero_col[k] > 0 && sigma_ero_col[k] > ce->size_per_surf_cols)
        error->all(FLERR,"surf_react surface/pwi adens_erosion: column out of range");
      int isp = particle->find_species((char *) sigma_ero_species[k].c_str());
      if (isp < 0)
        error->all(FLERR,"surf_react surface/pwi adens_erosion: unknown species");
      sigma_ero_compute.push_back(ce);
      sigma_ero_isp.push_back(isp);
    }

    // bind the implantation sources (task 5)
    imp_compute.assign(imp_species.size(), (Compute *) NULL);
    imp_isp.assign(imp_species.size(), -1);
    for (size_t k = 0; k < imp_species.size(); k++) {
      int isp = particle->find_species((char *) imp_species[k].c_str());
      if (isp < 0)
        error->all(FLERR,"surf_react surface/pwi adens_implant: species not "
                   "in the species list (add it, e.g. D, even if never spawned)");
      imp_isp[k] = isp;
      if (imp_mode[k] == 0) {
        int ic = modify->find_compute((char *) imp_comp_id[k].c_str());
        if (ic < 0)
          error->all(FLERR,"surf_react surface/pwi adens_implant: compute ID not found");
        imp_compute[k] = modify->compute[ic];
        if (!imp_compute[k]->per_surf_flag)
          error->all(FLERR,"surf_react surface/pwi adens_implant: compute is not per-surf");
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Additive self-sputtering. For each SPUTTER channel of the incident species,
   evaluate the Eckstein yield Y(E,theta), draw N = floor(Y)+Bernoulli(frac Y)
   whole atoms, and emit each as the product species (neutral W) with a
   Thompson energy + cosine angle at the impact point. Each sputtered atom
   inherits the incident pweight, so the emitted real-atom flux = Y x incident
   flux. Independent of the reflect/absorb outcome (may run with ip about to be
   reflected OR absorbed). add_particle may grow the particle list -> re-point
   ip (by reference) and re-index the pweight custom.
------------------------------------------------------------------------- */

int SurfReactSurfacePWI::emit_sputtered(Particle::OnePart *&ip, int isurf,
                                        double *norm, double E_in_eV,
                                        double theta_in_deg)
{
  int n = reactions[ip->ispecies].n;
  if (n == 0) return 0;
  int nemit_total = 0;
  int *list = reactions[ip->ispecies].list;

  // incident pweight (real atoms this macroparticle represents)
  double pw_inc = update->fnum;
  if (pweight_ewhich >= 0)
    pw_inc = particle->edvec[particle->ewhich[pweight_ewhich]][ip - particle->particles];

  for (int i = 0; i < n; i++) {
    OneReaction *r = &rlist[list[i]];
    if (r->type != SPUTTER) continue;

    Eckstein::SputterParams p;
    p.Es = r->sp_Es; p.Eth = r->sp_Eth; p.Q = r->sp_Q; p.ETF = r->sp_ETF;
    // Prefer the angle-resolved TRIM/BCA table (processes.h5
    // /surface/sputter/<pair>) over the analytic Eckstein fit.
    // 3D compound-target tables encode the composition dependence
    // themselves: evaluate at the local mixing-zone concentration of the
    // `conc` species and skip the linear `mat` rescaling. With
    // sigma_feedback off, mat_conc() returns 1.0 = the c=1 endpoint
    // (fresh lead-element surface).
    double Y;
    const double theta_eff =
        (theta_in_deg > rough_dm) ? theta_in_deg - rough_dm : 0.0;
    const bool compound = (r->sp_tbl >= 0 && sput_tables[r->sp_tbl].NC > 0);
    const bool ttable   = (r->sp_tbl >= 0 && sput_tables[r->sp_tbl].NT > 0);
    if (compound) {
      double c = (r->conc_isp >= 0) ? mat_conc(isurf, r->conc_isp) : 1.0;
      Y = sput_tables[r->sp_tbl].yield(E_in_eV, theta_eff, c);
    } else if (ttable) {
      // temperature-axis table (liquid-metal): evaluate at the element's
      // wall temperature (twall_surf custom if bound, else scalar twall).
      double twall_eff = twall;
      if (tindex_custom >= 0)
        twall_eff = surf->edvec_local[surf->ewhich[tindex_custom]][isurf];
      Y = sput_tables[r->sp_tbl].yield_at_T(E_in_eV, theta_eff, twall_eff);
      if (r->mat_isp >= 0) Y *= mat_conc(isurf, r->mat_isp);
    } else {
      Y = (r->sp_tbl >= 0)
          ? sput_tables[r->sp_tbl].yield(E_in_eV, theta_eff)
          : Eckstein::sputter_yield(E_in_eV, theta_eff, p);
      if (r->mat_isp >= 0) Y *= mat_conc(isurf, r->mat_isp);
    }
    if (Y <= 0.0) continue;

    int nemit = (int) Y;                         // floor(Y)
    if (random->uniform() < Y - nemit) nemit++;  // + Bernoulli(frac Y)
    if (nemit == 0) continue;
    if (nemit > 20) {                            // corrupt table / bad E guard
      static int warned = 0;
      if (!warned) {
        warned = 1;
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "surface/pwi: sputter yield %.1f clamped to 20 "
                 "(suspect table entry)", Y);
        error->warning(FLERR, msg);
      }
      nemit = 20;
    }

    int sp = r->products[0];                     // sputtered species (neutral W)
    double mass = particle->species[sp].mass;

    // recoil cutoff: max ejection energy Emax = gamma*E_in - Es, with the
    // binary transfer factor gamma = 4 M1 M2/(M1+M2)^2 (~1 for W-on-W).
    double m_in = particle->species[ip->ispecies].mass;
    double gamma = 4.0 * m_in * mass / ((m_in + mass) * (m_in + mass));
    double Emax = gamma * E_in_eV - p.Es;
    if (Emax <= 0.0) continue;                   // incident too soft to eject

    nemit_total += nemit;
    for (int k = 0; k < nemit; k++) {
      double E_eV = pwi_sample_thompson(p.Es, Emax, random);  // Ub, cutoff
      double x[3], v[3];
      memcpy(x, ip->x, 3*sizeof(double));
      sample_cosine_velocity(v, norm, E_eV, mass);

      int id = MAXSMALLINT * random->uniform();
      Particle::OnePart *particles = particle->particles;
      int reallocflag = particle->add_particle(id, sp, ip->icell, x, v, 0.0, 0.0);
      if (reallocflag) ip = particle->particles + (ip - particles);

      if (pweight_ewhich >= 0)                    // inherit incident pweight
        particle->edvec[particle->ewhich[pweight_ewhich]][particle->nlocal-1] = pw_inc;

      nsingle++;
      tally_single[list[i]]++;
    }

    // sigma ledger: nemit real-atom-weighted macroatoms of species sp left
    // this surf element -> net erosion of the sputtered species
    if (sindex_custom >= 0)
      sigma_accumulate(isurf, sp, -((double) nemit) * pw_inc);
  }
  return nemit_total;
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
    twall_eff = surf->edvec_local[surf->ewhich[tindex_custom]][isurf];

  // Incident impact (E, theta) at the wall. The incident velocity here
  // already includes the sheath boundary energy boost (the mover kicks v
  // inbound before the collision), so E_in_eV is the physical sputtering
  // energy. Computed up front and reused by TRIM and SPUTTER.
  double E_in_eV = 0.0, theta_in_deg = 0.0;
  bool trim_precomputed = false;
  {
    double mass_in = particle->species[ip->ispecies].mass;
    double v2 = ip->v[0]*ip->v[0] + ip->v[1]*ip->v[1] + ip->v[2]*ip->v[2];
    if (mass_in > 0.0)
      E_in_eV = 0.5 * mass_in * v2 * update->joule2ev * update->mvv2e;
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

  // Additive self-sputtering: emit sputtered atoms BEFORE the reflect/absorb
  // lottery (uses the intact incident state). May add particles and thus
  // re-point ip (handled by reference).
  int nsput = emit_sputtered(ip, isurf, norm, E_in_eV, theta_in_deg);
  if (ehist_file) {
    double pw_hist = update->fnum;
    if (pweight_ewhich >= 0)
      pw_hist = particle->edvec[particle->ewhich[pweight_ewhich]][ip - particle->particles];
    ehist_accumulate(E_in_eV, theta_in_deg, pw_hist, nsput > 0, ip->ispecies);
  }

  OneReaction *r;

  for (int i = 0; i < n; i++) {
    r = &rlist[list[i]];
    if (r->type == SPUTTER) continue;   // additive; handled in emit_sputtered

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
      if (r->refl_tbl >= 0) {
        // composition-resolved reflection coefficient R(E,theta,c) at the
        // local concentration; roughness shift applied consistently
        const double th_eff =
            (theta_in_deg > rough_dm) ? theta_in_deg - rough_dm : 0.0;
        double cval = (r->conc_isp >= 0) ? mat_conc(isurf, r->conc_isp) : 1.0;
        p_this = (sput_tables[r->refl_tbl].NT > 0)
            ? sput_tables[r->refl_tbl].yield_at_T(E_in_eV, th_eff, twall_eff)
            : sput_tables[r->refl_tbl].yield(E_in_eV, th_eff, cval);
        if (p_this < 0.0) p_this = 0.0;
        if (p_this > 1.0) p_this = 1.0;
      } else {
        p_this = Reflection::R_N_interp(tv, E_in_eV, theta_in_deg);
        if (r->mat_isp >= 0) p_this *= mat_conc(isurf, r->mat_isp);
      }
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
          // sigma ledger: incident particle retained in the wall
          if (sindex_custom >= 0) {
            double pw_inc = update->fnum;
            if (pweight_ewhich >= 0)
              pw_inc = particle->edvec[particle->ewhich[pweight_ewhich]][ip - particle->particles];
            sigma_accumulate(isurf, ip->ispecies, pw_inc);
          }
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
            R_rec = surf->edvec_local[surf->ewhich[rindex_custom]][isurf];
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
            // sigma ledger: retained atoms deposit on this surf element
            if (sindex_custom >= 0) {
              double pw_inc = update->fnum;
              if (pweight_ewhich >= 0)
                pw_inc = particle->edvec[particle->ewhich[pweight_ewhich]][ip - particle->particles];
              sigma_accumulate(isurf, sp_atom, pw_inc);
            }
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

/* ----------------------------------------------------------------------
   surface concentration of material species isp at local surf isurf,
   from the <attr>_conc custom array (1.0 if weighting inactive)
------------------------------------------------------------------------- */

double SurfReactSurfacePWI::mat_conc(int isurf, int isp)
{
  if (!sigma_feedback || isp < 0 || sconc_index < 0) return 1.0;
  return surf->edarray_local[surf->ewhich[sconc_index]][isurf][isp];
}

/* ----------------------------------------------------------------------
   sigma ledger: record a net change of datoms real atoms of species isp
   on surf element isurf (local index), as an areal density (atoms/m^2).
   Accumulated locally; folded into the owned custom array in sync_sigma().
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::sigma_accumulate(int isurf, int isp, double datoms)
{
  double area = sigma_area[isurf];
  if (area <= 0.0) return;

  surfint gid;
  if (domain->dimension == 2) gid = surf->lines[isurf].id;
  else gid = surf->tris[isurf].id;

  bigint idx = ((bigint) gid - 1) * sigma_ncols + isp;
  sigma_delta[idx] += datoms / area;
  if (datoms > 0.0) dep_delta[gid - 1] += datoms / area;
}

/* ----------------------------------------------------------------------
   called by Update at end of every timestep (via tally_update). Every
   sigma_nevery steps: allreduce the per-rank deltas (each impact happens
   on exactly one rank), fold into the owned custom array, refresh the
   local+ghost copies so dumps and future per-event reads see current sigma.
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::sync_sigma()
{
  int n = (int) (sigma_nsurf * (bigint) sigma_ncols);

  MPI_Allreduce(sigma_delta,sigma_buf,n,MPI_DOUBLE,MPI_SUM,world);
  for (int i = 0; i < n; i++) sigma_delta[i] = 0.0;

  // owned surf i on this rank is local surf m = me + i*nprocs
  // (explicit non-distributed striding, cf. fix_surf_temp)

  int me = comm->me;
  int nprocs = comm->nprocs;
  int dim = domain->dimension;
  int nsown = surf->nown;
  double **sig = surf->edarray[surf->ewhich[sindex_custom]];

  surfint gid;
  for (int i = 0; i < nsown; i++) {
    bigint m = me + (bigint) i*nprocs;
    if (dim == 2) gid = surf->lines[m].id;
    else gid = surf->tris[m].id;
    bigint base = ((bigint) gid - 1) * sigma_ncols;
    for (int j = 0; j < sigma_ncols; j++)
      sig[i][j] += sigma_buf[base + j];

    // strata stack: apply this sync's particle-impact net per MATERIAL
    // (charge states pool). Positive net -> surface deposition (low-E
    // redeposition sticks at the top); negative -> preferential
    // erosion, bounded by what the stack actually exposes.
    if (strata_K > 0) {
      for (int mm = 0; mm < nmat; mm++) {
        double net = 0.0;
        for (int j = 0; j < sigma_ncols; j++)
          if (mat_root_of[j] == mm) net += sigma_buf[base + j];
        if (net > 0.0)
          strata_state[i].deposit(mm, net, mat_dens[mm]);
        else if (net < 0.0)
          strata_state[i].erode_species(mm, -net);
      }
    }
  }

  // gross-erosion debit: sigma[ero_species] -= flux * dt * sigma_nevery,
  // flux [atoms/m^2/s] per owned surf from the bound per-surf compute
  // (e.g. background-plasma sputtering of the wall feeding the W source)
  for (size_t k = 0; k < sigma_ero_compute.size(); k++) {
    Compute *ce = sigma_ero_compute[k];
    int isp = sigma_ero_isp[k];
    if (ce->invoked_per_surf != update->ntimestep) {
      ce->compute_per_surf();
      ce->post_process_surf();
    }
    double debit_dt = update->dt * sigma_nevery;
    // concentration-limited: the background erodes material isp only in
    // proportion to its surface concentration (owned conc array from the
    // previous derived pass; protective deposits shield the substrate).
    // noconc bindings skip the factor: their compute already evaluated
    // the yield at the local composition (compound tables).
    double **conck = surf->edarray[surf->ewhich[sconc_index]];
    const int noconc = sigma_ero_noconc[k];
    if (sigma_ero_col[k] == 0) {
      double *fvec = ce->vector_surf;
      for (int i = 0; i < nsown; i++) {
        double d = fvec[i] * (noconc ? 1.0 : conck[i][isp]) * debit_dt;
        sig[i][isp] -= d;
        if (strata_K > 0 && d > 0.0)
          strata_state[i].erode_species(mat_root_of[isp], d);
      }
    } else {
      double **farr = ce->array_surf;
      int icol = sigma_ero_col[k] - 1;
      for (int i = 0; i < nsown; i++) {
        double d = farr[i][icol] * (noconc ? 1.0 : conck[i][isp]) * debit_dt;
        sig[i][isp] -= d;
        if (strata_K > 0 && d > 0.0)
          strata_state[i].erode_species(mat_root_of[isp], d);
      }
    }
  }

  // derived columns: gross deposition and net (row sum)
  MPI_Allreduce(dep_delta,dep_buf,(int) sigma_nsurf,MPI_DOUBLE,MPI_SUM,world);
  for (int i = 0; i < (int) sigma_nsurf; i++) dep_delta[i] = 0.0;
  double *depvec = surf->edvec[surf->ewhich[sdep_index]];
  double *netvec = surf->edvec[surf->ewhich[snet_index]];
  double *erovec = surf->edvec[surf->ewhich[sero_index]];
  for (int i = 0; i < nsown; i++) {
    bigint m = me + (bigint) i*nprocs;
    if (dim == 2) gid = surf->lines[m].id;
    else gid = surf->tris[m].id;
    depvec[i] += dep_buf[gid - 1];
    double nsum = 0.0;
    for (int j = 0; j < sigma_ncols; j++) nsum += sig[i][j];
    netvec[i] = nsum;
    erovec[i] = depvec[i] - nsum;    // gross removal (positive)
  }
  // background implantation + retention saturation (task 5)
  for (size_t k = 0; k < imp_species.size(); k++) {
    int isp = imp_isp[k];
    double debit_dt = update->dt * sigma_nevery;
    double **conck = surf->edarray[surf->ewhich[sconc_index]];
    double *fvec = NULL; double **farr = NULL; int icol = 0;
    if (imp_mode[k] == 0) {
      Compute *ce = imp_compute[k];
      if (ce->invoked_per_surf != update->ntimestep) {
        ce->compute_per_surf();
        ce->post_process_surf();
      }
      if (imp_col[k] == 0) fvec = ce->vector_surf;
      else { farr = ce->array_surf; icol = imp_col[k] - 1; }
    }
    for (int i = 0; i < nsown; i++) {
      double flux = (imp_mode[k] == 1) ? imp_flux[k]
                    : (fvec ? fvec[i] : farr[i][icol]);
      if (flux <= 0.0) continue;
      double cD = conck[i][isp];
      double turnon = 0.5 * (tanh(imp_alpha[k] * (cD - imp_cmax[k])) + 1.0);
      double dimp = flux * (1.0 - imp_rcoef[k]) * (1.0 - turnon) * debit_dt;
      if (dimp <= 0.0) continue;
      sig[i][isp] += dimp;
      if (strata_K > 0) {
        if (imp_depth[k] > 0.0)
          strata_state[i].add_implanted(mat_root_of[isp], imp_depth[k], dimp);
        else
          strata_state[i].deposit(mat_root_of[isp], dimp, mat_dens[mat_root_of[isp]]);
      }
    }
  }

  derive_sigma_conc();

  surf->estatus[snet_index] = 0;
  surf->estatus[sdep_index] = 0;
  surf->estatus[sero_index] = 0;
  surf->spread_custom(sero_index);
  surf->spread_custom(snet_index);
  surf->spread_custom(sdep_index);

  surf->estatus[sindex_custom] = 0;
  surf->spread_custom(sindex_custom);
}

/* ----------------------------------------------------------------------
   mixing-zone concentrations: c_i = max(sigma_i,0)/zone, clipped to
   sum <= 1; remainder is substrate. Deposits fill the zone from the top.
   concentrations are per MATERIAL (element): all charge states of an
   element pool into one c; every species column of that element carries
   the material concentration, so mat_conc() works with any of them.
   Called from sync_sigma() every sync, and once at init() so the very
   first debit/emission read sees the initial (e.g. boronized) surface
   instead of an all-zero conc array.
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::derive_sigma_conc()
{
  double **sig = surf->edarray[surf->ewhich[sindex_custom]];
  double **conc = surf->edarray[surf->ewhich[sconc_index]];
  int nsown = surf->nown;
  int sub_mat = mat_of[substrate_isp];

  // strata mode: the stack is the source of truth for concentrations.
  // Serve the atom-weighted composition of the top `rzone` worth of
  // material (areal -> depth via the local surface density), compact,
  // and pack into the <attr>_strata custom for dumps/restart. The
  // pooled-sigma derivation below stays as the legacy (2-layer) path.
  if (strata_K > 0) {
    double **sb = surf->edarray[surf->ewhich[strata_index]];
    for (int i = 0; i < nsown; i++) {
      SurfaceElementState &st = strata_state[i];
      st.compact_layers();
      double dens = st.surface_density();
      if (dens <= 0.0) dens = mat_dens[mat_root_of[substrate_isp]];
      double depth = sigma_zone / dens;            // [m], factor-free SI
      std::vector<double> comp = st.get_surface_composition(depth);
      for (int j = 0; j < sigma_ncols; j++)
        conc[i][j] = comp[mat_root_of[j]];
      for (int c = 0; c < strata_ncols; c++) sb[i][c] = 0.0;
      st.pack(sb[i]);
    }
    surf->estatus[strata_index] = 0;
    surf->spread_custom(strata_index);
    surf->estatus[sconc_index] = 0;
    surf->spread_custom(sconc_index);
    return;
  }
  for (int i = 0; i < nsown; i++) {
    for (int j = 0; j < sigma_ncols; j++) conc[i][j] = 0.0;
    // pool each material's columns BEFORE clipping: charge states share
    // one inventory, and redeposited-ion credits (B+ columns) must cancel
    // the neutral column's erosion debit. Clipping per column instead let
    // c stay at 1 while the summed inventory went negative, so the
    // concentration-limited debit never shut off (unbounded negative
    // sigma at high-redeposition elements).
    for (int j = 0; j < sigma_ncols; j++)
      conc[i][mat_of[j]] += sig[i][j] / sigma_zone;
    double csum = 0.0;
    for (int j = 0; j < sigma_ncols; j++)
      if (mat_of[j] == j) {
        if (conc[i][j] < 0.0) conc[i][j] = 0.0;
        csum += conc[i][j];
      }
    if (csum > 1.0) {
      for (int j = 0; j < sigma_ncols; j++) conc[i][j] /= csum;
      csum = 1.0;
    }
    conc[i][sub_mat] += 1.0 - csum;         // substrate fills the rest
    for (int j = 0; j < sigma_ncols; j++)   // broadcast material c to
      if (mat_of[j] != j) conc[i][j] = conc[i][mat_of[j]];  // all its species
  }
  surf->estatus[sconc_index] = 0;
  surf->spread_custom(sconc_index);
}

/* ---------------------------------------------------------------------- */

void SurfReactSurfacePWI::tally_update()
{
  SurfReact::tally_update();
  if (sindex_custom >= 0 && update->ntimestep % sigma_nevery == 0)
    sync_sigma();
  if (ehist_file && ehist_every > 0 &&
      update->ntimestep % ehist_every == 0) ehist_write();
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
    if (r->mat_id) {
      r->mat_isp = particle->find_species(r->mat_id);
      if (r->mat_isp < 0)
        error->all(FLERR,"surf_react surface/pwi: unknown 'mat' species in reaction file");
    }
    if (r->conc_id) {
      r->conc_isp = particle->find_species(r->conc_id);
      if (r->conc_isp < 0)
        error->all(FLERR,"surf_react surface/pwi: unknown 'conc' species in reaction file");
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
        r->mat_id = NULL;
        r->mat_isp = -1;
        r->conc_id = NULL;
        r->conc_isp = -1;
        r->refl_tbl = -1;
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
    else if (word[0] == 'S' || word[0] == 's') r->type = SPUTTER;
    else error->all(FLERR, "Invalid reaction type in recycle file");

    // Tag the printed reaction label with the channel type, so the surf
    // reaction tally distinguishes reflect/absorb/sputter/... (otherwise every
    // "W+ --> W" channel prints identically). r->id was built from line1 above.
    {
      const char *tag = (r->type == TRIM_REFLECT)  ? " [T:reflect]"  :
                        (r->type == ABSORB_REEMIT) ? " [A:absorb]"   :
                        (r->type == SPUTTER)       ? " [S:sputter]"  :
                        (r->type == EXCHANGE)      ? " [E:exchange]" :
                        (r->type == RECOMBINATION) ? " [R:recomb]"   :
                        (r->type == DISSOCIATION)  ? " [D:dissoc]"   : "";
      if (r->id && tag[0]) {
        char *newid = new char[strlen(r->id) + strlen(tag) + 1];
        strcpy(newid, r->id);
        strcat(newid, tag);
        delete [] r->id;
        r->id = newid;
      }
    }

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
    } else if (r->type == SPUTTER) {
      if (r->nreactant != 1 || r->nproduct != 1)
        error->all(FLERR, "Sputter recycle reaction needs 1 reactant, 1 product "
                          "(product = sputtered species, e.g. W)");
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

      // optional keywords:
      //   mat <species>   linear concentration weighting of the pure R
      //   conc <species> rtable <name>   composition-resolved reflection:
      //     probability from the 3D R(E,theta,c) table <name> at the local
      //     concentration of <species> (no linear rescale); the outgoing
      //     energy/angle spectrum is still sampled from the pure TRIM
      //     table (documented approximation: outgoing distributions are
      //     far less composition-sensitive than the coefficient)
      while ((word = strtok(NULL, " \t\n"))) {
        if (strcmp(word,"mat") == 0) {
          word = strtok(NULL, " \t\n");
          if (!word) error->all(FLERR, "Missing species after 'mat' in T reaction");
          int n2 = strlen(word) + 1;
          r->mat_id = new char[n2];
          strcpy(r->mat_id, word);
        } else if (strcmp(word,"conc") == 0) {
          word = strtok(NULL, " \t\n");
          if (!word) error->all(FLERR, "Missing species after 'conc' in T reaction");
          int n2 = strlen(word) + 1;
          r->conc_id = new char[n2];
          strcpy(r->conc_id, word);
        } else if (strcmp(word,"rtable") == 0) {
          word = strtok(NULL, " \t\n");
          if (!word) error->all(FLERR, "Missing table name after 'rtable' in T reaction");
          r->refl_tbl = load_or_get_sputter_table(word);
          if (r->refl_tbl < 0 || sput_tables[r->refl_tbl].NC <= 0)
            error->all(FLERR, "T reaction rtable needs a 3D R(E,theta,c) "
                       "table in processes.h5");
        } else {
          error->all(FLERR, "Unknown keyword in T reaction (expect mat/conc/rtable)");
        }
      }

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

    } else if (r->type == SPUTTER) {
      // Sputter reaction: `S <eckstein_entry>` (e.g. W_on_W). Additive — it
      // does NOT participate in the reflect/absorb lottery (prob = 0); the
      // yield Y(E,theta) is evaluated per event in emit_sputtered().
      word = strtok(NULL, " \t\n");
      if (!word) error->all(FLERR, "Missing Eckstein entry name in S recycle reaction");
      r->sp_tbl = load_or_get_sputter_table(word);
      Eckstein::SputterParams p;
      if (Eckstein::lookup_sputter(word, p)) {
        r->sp_Es = p.Es; r->sp_Eth = p.Eth; r->sp_Q = p.Q; r->sp_ETF = p.ETF;
      } else if (r->sp_tbl >= 0 && sput_tables[r->sp_tbl].Es > 0.0) {
        // no catalog entry, but the table carries the emitted species'
        // surface binding energy (compound tables); analytic fallback
        // params stay zero -- the table always overrides them.
        r->sp_Es = sput_tables[r->sp_tbl].Es;
        r->sp_Eth = r->sp_Q = r->sp_ETF = 0.0;
      } else {
        char str[256];
        snprintf(str, sizeof(str),
                 "surf_react surface/pwi: sputter entry '%s' not in "
                 "eckstein_sputter_data.h and no processes.h5 table with "
                 "an Es_eV attribute", word);
        error->all(FLERR, str);
      }
      r->prob = 0.0;

      // optional: `mat <species>` weights the yield by the surface
      // concentration of that material (composition-weighted sputtering);
      // `conc <species>` names the concentration that is the composition
      // coordinate of a 3D compound-target table (no yield rescaling)
      while ((word = strtok(NULL, " \t\n"))) {
        if (strcmp(word,"mat") == 0) {
          word = strtok(NULL, " \t\n");
          if (!word) error->all(FLERR, "Missing species after 'mat' in S reaction");
          int n2 = strlen(word) + 1;
          r->mat_id = new char[n2];
          strcpy(r->mat_id, word);
        } else if (strcmp(word,"conc") == 0) {
          word = strtok(NULL, " \t\n");
          if (!word) error->all(FLERR, "Missing species after 'conc' in S reaction");
          int n2 = strlen(word) + 1;
          r->conc_id = new char[n2];
          strcpy(r->conc_id, word);
        } else {
          error->all(FLERR, "Unknown keyword in S reaction (expect mat/conc)");
        }
      }

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

int SurfReactSurfacePWI::load_or_get_sputter_table(const char *name)
{
  std::string pair_lower(name);
  for (auto &c : pair_lower)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  auto it = sput_index.find(pair_lower);
  if (it != sput_index.end()) return it->second;

  int idx = -1;
  std::string processes_path = resolve_processes_file();
  if (!processes_path.empty()) {
    ProcessLibrary lib;
    lib.open(processes_path, world, error);
    if (lib.is_open()) {
      ProcessLibrary::TrimSputterTable t;
      if (lib.load_trim_sputter(pair_lower, t)) {
        idx = static_cast<int>(sput_tables.size());
        sput_tables.push_back(std::move(t));
        if (comm->me == 0) {
          char msg[256];
          if (sput_tables[idx].NT > 0)
            snprintf(msg, sizeof(msg),
                     "surf_react surface/pwi: loaded T-dependent sputter "
                     "yield table '%s' (%d T x %d x %d, %g-%g K) from "
                     "database/processes.h5\n",
                     pair_lower.c_str(), sput_tables[idx].NT,
                     sput_tables[idx].NE, sput_tables[idx].NTHETA,
                     sput_tables[idx].Tax.front(), sput_tables[idx].Tax.back());
          else if (sput_tables[idx].NC > 0)
            snprintf(msg, sizeof(msg),
                     "surf_react surface/pwi: loaded compound sputter yield "
                     "table '%s' (%d c x %d x %d, Es=%.2f eV) from "
                     "database/processes.h5\n",
                     pair_lower.c_str(), sput_tables[idx].NC,
                     sput_tables[idx].NE, sput_tables[idx].NTHETA,
                     sput_tables[idx].Es);
          else
            snprintf(msg, sizeof(msg),
                   "surf_react surface/pwi: loaded sputter yield table '%s' "
                   "(%d x %d) from database/processes.h5\n",
                   pair_lower.c_str(), sput_tables[idx].NE,
                   sput_tables[idx].NTHETA);
          if (screen)  fputs(msg, screen);
          if (logfile) fputs(msg, logfile);
        }
      }
    }
  }
  if (idx < 0) {
    char msg[384];
    snprintf(msg, sizeof(msg),
             "no angle/energy sputter table for '%s' in processes.h5 -- "
             "using the ANALYTIC Eckstein/Yamamura model (unreliable at "
             "grazing incidence; add tables with "
             "tools/converters/add_sputter_tables.py)", pair_lower.c_str());
    error->warning(FLERR, msg);
  }
  sput_index[pair_lower] = idx;
  return idx;
}

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

/* ----------------------------------------------------------------------
   ehist: accumulate one wall impact into the (E,theta) histograms
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::ehist_accumulate(double E_eV, double theta_deg,
                                           double pw, int sput, int isp)
{
  int ie = (int) (E_eV / ehist_emax * ehist_nbin);
  if (ie < 0) ie = 0;
  if (ie >= ehist_nbin) ie = ehist_nbin - 1;   // overflow -> last bin
  int ia = (int) (theta_deg / 5.0);
  if (ia < 0) ia = 0;
  if (ia >= EHIST_NANG) ia = EHIST_NANG - 1;
  ehist_all[ie] += pw;
  ahist_all[ia] += pw;
  if (sput) {
    ehist_sput[ie] += pw;
    ahist_sput[ia] += pw;
  }
  if (ehist_z && isp >= 0 && isp < ehist_nsp) ehist_z[isp][ie] += pw;
}

/* ----------------------------------------------------------------------
   ehist: reduce across ranks, append a windowed block, reset local bins
------------------------------------------------------------------------- */

void SurfReactSurfacePWI::ehist_write()
{
  int nz = (ehist_z) ? ehist_nsp*ehist_nbin : 0;
  int ntot = 2*ehist_nbin + 2*EHIST_NANG + nz;
  double *loc = new double[ntot];
  double *glob = new double[ntot];
  for (int i = 0; i < ehist_nbin; i++) {
    loc[i] = ehist_all[i];
    loc[ehist_nbin+i] = ehist_sput[i];
  }
  for (int i = 0; i < EHIST_NANG; i++) {
    loc[2*ehist_nbin+i] = ahist_all[i];
    loc[2*ehist_nbin+EHIST_NANG+i] = ahist_sput[i];
  }
  int off = 2*ehist_nbin + 2*EHIST_NANG;
  for (int k = 0; k < ehist_nsp && ehist_z; k++)
    for (int i = 0; i < ehist_nbin; i++)
      loc[off + k*ehist_nbin + i] = ehist_z[k][i];
  MPI_Reduce(loc,glob,ntot,MPI_DOUBLE,MPI_SUM,0,world);
  if (comm->me == 0) {
    FILE *fp = fopen(ehist_file,"a");
    if (fp) {
      fprintf(fp,"# step %ld nbin %d emax %g nang %d dang 5\n",
              (long) update->ntimestep, ehist_nbin, ehist_emax, EHIST_NANG);
      for (int i = 0; i < ehist_nbin; i++)
        fprintf(fp,"E %g %.6e %.6e\n",
                (i+0.5)*ehist_emax/ehist_nbin, glob[i], glob[ehist_nbin+i]);
      for (int i = 0; i < EHIST_NANG; i++)
        fprintf(fp,"A %g %.6e %.6e\n",
                (i+0.5)*5.0, glob[2*ehist_nbin+i],
                glob[2*ehist_nbin+EHIST_NANG+i]);
      // per-species impact-E rows; skip species with no weight this window
      for (int k = 0; k < ehist_nsp && ehist_z; k++) {
        double tot = 0.0;
        for (int i = 0; i < ehist_nbin; i++) tot += glob[off + k*ehist_nbin + i];
        if (tot <= 0.0) continue;
        for (int i = 0; i < ehist_nbin; i++)
          fprintf(fp,"S %s %g %.6e\n", particle->species[k].id,
                  (i+0.5)*ehist_emax/ehist_nbin, glob[off + k*ehist_nbin + i]);
      }
      fclose(fp);
    }
  }
  for (int i = 0; i < ehist_nbin; i++) ehist_all[i] = ehist_sput[i] = 0.0;
  for (int i = 0; i < EHIST_NANG; i++) ahist_all[i] = ahist_sput[i] = 0.0;
  if (ehist_z)
    for (int k = 0; k < ehist_nsp; k++)
      for (int i = 0; i < ehist_nbin; i++) ehist_z[k][i] = 0.0;
  delete [] loc;
  delete [] glob;
}

