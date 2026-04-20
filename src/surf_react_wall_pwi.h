/* ----------------------------------------------------------------------
    OpenEdge: Plasma-wall interaction (PWI) surface reaction model.

    Unified handler for EIRENE-style wall recycling: TRIM reflection,
    fixed-fraction re-emission, absorb-and-re-emit with recycling
    coefficient R and molecular fraction f_mol, and the simpler
    Franck-Condon / thermal exchange channels. Species- and material-
    agnostic (works for D/W, H/Li, T/Be, etc.).

    See doc/surf_react_wall_pwi.txt for the reactions file grammar and
    the physics of each channel type.

    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(wall_pwi,SurfReactWallPWI)
SurfReactStyle(recycle,SurfReactWallPWI)   // legacy alias; prefer wall_pwi

#else

#ifndef SPARTA_SURF_REACT_WALL_PWI_H
#define SPARTA_SURF_REACT_WALL_PWI_H

#include "surf_react.h"
#include "reflection_tables.h"
#include <map>
#include <string>
#include <vector>

namespace SPARTA_NS {

class SurfReactWallPWI : public SurfReact {
 public:
  SurfReactWallPWI(class SPARTA *, int, char **);
  ~SurfReactWallPWI();
  void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  struct OneReaction {
    int active;
    int type;                      // EXCHANGE, DISSOCIATION, RECOMBINATION, TRIM_REFLECT
    int nreactant, nproduct;
    char **id_reactants, **id_products;
    int *reactants, *products;
    double prob;                   // fixed recycling probability (0 to 1); ignored for TRIM_REFLECT
    double *energy;                // return energy per product [eV]; ignored for TRIM_REFLECT
    int trim_table;                // index into trim_tables, -1 if not a TRIM reaction
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
