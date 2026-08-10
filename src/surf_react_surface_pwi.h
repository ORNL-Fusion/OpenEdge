/* ----------------------------------------------------------------------
    OpenEdge: Plasma-wall interaction (PWI) surface reaction model.

    Unified handler for EIRENE-style wall recycling: TRIM reflection,
    fixed-fraction re-emission, absorb-and-re-emit with recycling
    coefficient R and molecular fraction f_mol, and the simpler
    Franck-Condon / thermal exchange channels. Species- and material-
    agnostic (works for D/W, H/Li, T/Be, etc.).

    See doc/surf_react_surface_pwi.txt for the reactions file grammar and
    the physics of each channel type.

    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(surface/pwi,SurfReactSurfacePWI)

#else

#ifndef SPARTA_SURF_REACT_SURFACE_PWI_H
#define SPARTA_SURF_REACT_SURFACE_PWI_H

#include "surf_react.h"
#include "process_library.h"
#include "reflection_tables.h"
#include "surf_state_multilayer.h"
#include <map>
#include <string>
#include <vector>

namespace SPARTA_NS {

class SurfReactSurfacePWI : public SurfReact {
 public:
  SurfReactSurfacePWI(class SPARTA *, int, char **);
  ~SurfReactSurfacePWI();
  void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  void tally_update();
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  struct OneReaction {
    int active;
    int type;                      // EXCHANGE, DISSOCIATION, RECOMBINATION, TRIM_REFLECT, ABSORB_REEMIT, SPUTTER
    int nreactant, nproduct;
    char **id_reactants, **id_products;
    int *reactants, *products;
    double prob;                   // fixed recycling probability (0 to 1); ignored for TRIM_REFLECT
    double *energy;                // return energy per product [eV]; ignored for TRIM_REFLECT
    int trim_table;                // index into trim_tables, -1 if not a TRIM reaction
    // SPUTTER channel: additive Eckstein yield Y_WW(E,theta), emits the
    // product species with a Thompson energy. Not part of the reflect/absorb
    // first-to-fire lottery.
    double sp_Es, sp_Eth, sp_Q, sp_ETF;  // Eckstein sputter params (type==SPUTTER)
    int sp_tbl;                          // index into sput_tables, -1 = analytic
    char *mat_id;                        // material species name for composition
    int mat_isp;                         //   weighting via <attr>_conc, -1 = off
    char *conc_id;                       // species whose <attr>_conc is the
    int conc_isp;                        //   composition coordinate of a 3D
                                         //   (compound-target) sputter table
    int refl_tbl;                        // T channel: 3D R(E,theta,c) table
                                         //   overriding the reflection
                                         //   probability (outgoing spectrum
                                         //   still sampled from the pure
                                         //   TRIM table), -1 = off
    char *id;
  };

 protected:
  class RanKnuth *random;
  OneReaction *rlist;
  int nlist_recycle, maxlist_recycle;
  double twall;                    // scalar wall temp for erot/evib resample (<=0 = pass-through)
  char *twall_attr;                // name of per-surf custom attribute, NULL if unused
  int tindex_custom;               // surf->find_custom index, -1 if twall_attr unset

  // per-surface recycling coefficient override. When R_attr is set, the
  // R value in any `A`-type reaction is replaced per-event by the
  // per-surf custom attribute. Allows spatially varying pumping
  // (e.g. R=1.0 on main wall, R=0.1 on a cryopump dome).
  char *R_attr;                    // name of per-surf custom attribute, NULL if unused
  int rindex_custom;               // surf->find_custom index, -1 if R_attr unset

  // per-surf areal densities (WallDYN terminology: adens; deck keyword
  // adens_surf, legacy alias sigma_surf). When sigma_attr is set, every
  // retained incident macroparticle adds +pweight/area to its species
  // column (influx) and every sputtered atom adds -pweight/area to the
  // sputtered species column (erosion flux) of a per-surf custom DOUBLE
  // array (esize = nspecies, units atoms/m^2). Deltas are accumulated
  // locally per rank and folded into the owned custom array every
  // adens_nevery steps in tally_update(). Explicit non-distributed
  // surfs only.
  char *sigma_attr;                // name of per-surf custom attribute, NULL if unused
  int sindex_custom;               // surf->find_custom index, -1 if unset
  int sigma_nevery;                // sync interval in steps (default 1)
  int sigma_ncols;                 // = particle->nspecies at init
  bigint sigma_nsurf;              // total surf count
  double *sigma_delta;             // [nsurf*ncols] local unsynced increments
  double *sigma_buf;               // allreduce receive buffer
  double *sigma_area;              // per-surf element area, local index
  // optional gross-erosion debit: per sync, sigma[species] -= flux * dt,
  // flux read per owned surf from a per-surf compute (e.g.
  // surface/physical/sputter ... erosion_flux)
  // repeatable: one binding per eroded target material; each debit is
  // scaled by that material's surface concentration (protective layers
  // shield the substrate)
  std::vector<std::string> sigma_ero_id;       // compute IDs
  std::vector<int> sigma_ero_col;              // 0 = vector, >0 = array col
  std::vector<std::string> sigma_ero_species;  // debited species names
  std::vector<int> sigma_ero_noconc;           // 1 = flux already c-weighted
  std::vector<class Compute *> sigma_ero_compute;
  std::vector<int> sigma_ero_isp;
  int snet_index, sdep_index, sero_index;  // derived: _net, _dep, _ero
  // WallDYN-style homogeneous reaction zone (deck keyword rzone, legacy
  // alias sigma_zone): concentrations c_i of the top rzone atoms/m^2,
  // stored in per-surf custom array <attr>_conc (esize = nspecies);
  // remainder of the zone is substrate (bulk) material. WallDYN gives the
  // same quantity as a thickness RZoneWidth [A]; ours is areal density.
  // surface-roughness yield correction (Schmid PSI review / Cupak
  // Appl.Surf.Sci. 570 (2021)): roughness flattens the angular
  // dependence of Y; pragmatic whole-device recipe is to evaluate
  // yields at theta_eff = max(0, theta - delta_m), delta_m = mean
  // surface angle (~20 deg for rough W). 0 = smooth (off).
  double rough_dm;                 // [deg from normal]
  double sigma_zone;               // reaction-zone total areal density [atoms/m^2]
  int sigma_feedback;              // 0 = ignore mat weights (Y,R without feedback)
  int sconc_index;                 // custom index of <attr>_conc
  int substrate_isp;               // species index of the substrate (default W)
  std::vector<int> mat_of;         // species -> material group (by element)
  std::vector<std::string> sigma_init_names;   // species with initial sigma
  std::vector<double> sigma_init_vals;         //   (boronization layers etc.)
  double *dep_delta, *dep_buf;     // gross-deposition ledger (positive credits)
  double mat_conc(int isurf, int isp);
  void sigma_accumulate(int isurf, int isp, double datoms);
  void sync_sigma();
  void derive_sigma_conc();   // owned sigma -> owned <attr>_conc + spread

  // Lagrangian strata stack (opt-in `strata <K>`): per-owned-element
  // multilayer state per MATERIAL (charge states pool; W, W+, ... are
  // one tungsten column in the stack). Deposition pushes strata,
  // erosion pops preferentially, concentrations served to all
  // consumers come from the top `rzone` worth of the stack. The adens
  // per-species columns keep accumulating as the balance ledger;
  // stack-vs-ledger agreement is a closure check.
  // DATA-LAYOUT POLICY (2026-08-04): the CANONICAL state is the flat
  // per-surf custom <attr>_strata (plain doubles, width 4+K*(2+nmat))
  // -- that is what MPI/dumps/restart and any future Kokkos port see.
  // The std::vector stack below is rank-local host working memory
  // rebuilt from pack()/unpack(); cold path (once per sync over owned
  // elements), matching this file's existing std::vector usage for
  // table caches. Strict flat-buffer-only rewrite = optional cleanup.
  int strata_K;                        // 0 = off (legacy 2-layer path)
  double strata_minthick;              // min stratum thickness [A]
  int strata_index;                    // custom index of <attr>_strata
  int strata_ncols;                    // 4 + K*(2+nmat)
  int nmat;                            // number of material roots
  std::vector<int> mat_root_of;        // species col -> material idx
  std::vector<double> mat_dens;        // material solid density [atoms/m^3]
  // deck overrides: strata_dens <species> <atoms/m^3> (repeatable);
  // built-in element defaults (W, B, O, C, Li, Be) used otherwise
  std::vector<std::string> strata_dens_names;
  std::vector<double> strata_dens_vals;
  std::vector<SurfaceElementState> strata_state;   // [surf->nown]

  // background implantation + retention saturation (task 5):
  //   adens_implant <species> compute <ID> <col> | const <flux_m2s>
  //                 [rcoef <R0>] [cmax <c_sat>] [alpha <a>] [depth <m>]
  // Per sync: implanted = flux * (1-R0) * (1-turnon) * dt with the
  // WallDYN-style switch turnon = 0.5*(tanh(alpha*(c - c_sat)) + 1),
  // credited to the species' adens column and (strata mode) implanted
  // at `depth` via add_implanted(). rcoef/depth are deck constants in
  // v1; binding the compute's mean-E/angle columns to evaluate the 3D
  // R and depth tables live is the documented follow-up.
  std::vector<std::string> imp_species, imp_comp_id;
  std::vector<int> imp_mode;            // 0 = compute col, 1 = const
  std::vector<int> imp_col, imp_isp;
  std::vector<double> imp_flux, imp_rcoef, imp_cmax, imp_alpha, imp_depth;
  std::vector<class Compute *> imp_compute;

  // reflection-table support (shared format with surf_react pmi; tables
  // from database/surface/trim/*.h5, originally EIRENE TRIM but any
  // BCA-derived HDF5 with the same schema works).
  std::string trim_dir;
  std::vector<Reflection::Table> trim_tables;
  std::map<std::string, int> trim_index;
  int load_or_get_trim_table(const char *name);

  struct ReactionI {
    int *list;
    int n;
  };

  ReactionI *reactions;
  int *indices;

  // additive W-on-W self-sputtering: emit N = floor(Y)+Bernoulli(frac Y)
  // product atoms (Thompson energy, cosine angle) per incident, weighted by
  // the incident pweight. pweight_ewhich = edvec index of the pweight custom
  // (-1 if fix particle/weight is absent).
  int pweight_ewhich;
  std::vector<ProcessLibrary::TrimSputterTable> sput_tables;
  std::map<std::string,int> sput_index;
  int load_or_get_sputter_table(const char *name);

  int emit_sputtered(Particle::OnePart *&ip, int isurf, double *norm,
                     double E_in_eV, double theta_in_deg);

  // ehist: pweight-weighted incident (E,theta) histograms of wall impacts,
  // all impacts vs impacts that sputtered; windowed blocks appended to
  // ehist_file every ehist_every steps
  static const int EHIST_NANG = 18;       // 5-degree angle bins, 0-90
  char *ehist_file;
  int ehist_nbin, ehist_every;
  double ehist_emax;
  double *ehist_all, *ehist_sput, *ahist_all, *ahist_sput;
  void ehist_accumulate(double E_eV, double theta_deg, double pw, int sput);
  void ehist_write();

  void init_reactions();
  void readfile(char *);
  int readone(char *, char *, int &, int &);
  void sample_cosine_velocity(double *v, double *norm, double energy_eV, double mass);
  void sample_reflected_velocity(double *v_out, const double *v_in,
                                 const double *norm, double E_out_eV,
                                 double cos_polar, double cos_azim,
                                 double mass_out);
  // Half-Maxwellian flux sampler: emits a particle leaving the wall at
  // temperature T_K. Normal component is flux-biased (~v*exp(-v^2/vrm^2)),
  // tangentials are Gaussian(0, sqrt(kT/m)). Bird 1994 formulation.
  void sample_thermal_flux_velocity(double *v, const double *norm,
                                    double T_K, double mass);
};

}

#endif
#endif
