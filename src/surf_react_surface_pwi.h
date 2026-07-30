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

  // per-surf areal-density ledger (sigma). When sigma_attr is set, every
  // absorbed incident macroparticle adds +pweight/area to its species
  // column and every sputtered atom adds -pweight/area to the sputtered
  // species column of a per-surf custom DOUBLE array (esize = nspecies,
  // units atoms/m^2). Deltas are accumulated locally per rank and folded
  // into the owned custom array every sigma_nevery steps in tally_update().
  // Explicit non-distributed surfs only.
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
  std::vector<class Compute *> sigma_ero_compute;
  std::vector<int> sigma_ero_isp;
  int snet_index, sdep_index, sero_index;  // derived: _net, _dep, _ero
  // WallDYN-style homogeneous mixing zone: concentrations c_i of the top
  // sigma_zone atoms/m^2, stored in per-surf custom array <attr>_conc
  // (esize = nspecies); remainder of the zone is substrate material.
  double sigma_zone;               // reaction-zone areal density [atoms/m^2]
  int sigma_feedback;              // 0 = ignore mat weights (Y,R without sigma)
  int sconc_index;                 // custom index of <attr>_conc
  int substrate_isp;               // species index of the substrate (default W)
  std::vector<int> mat_of;         // species -> material group (by element)
  std::vector<std::string> sigma_init_names;   // species with initial sigma
  std::vector<double> sigma_init_vals;         //   (boronization layers etc.)
  double *dep_delta, *dep_buf;     // gross-deposition ledger (positive credits)
  double mat_conc(int isurf, int isp);
  void sigma_accumulate(int isurf, int isp, double datoms);
  void sync_sigma();

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

  void emit_sputtered(Particle::OnePart *&ip, int isurf, double *norm,
                      double E_in_eV, double theta_in_deg);

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
