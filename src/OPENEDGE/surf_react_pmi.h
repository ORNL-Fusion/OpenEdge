/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(pmi,SurfReactPMI)

#else

#ifndef SPARTA_SURF_REACT_PMI_H
#define SPARTA_SURF_REACT_PMI_H

#include "surf_react.h"
#include <vector>
#include <string>

namespace SPARTA_NS {

class SurfReactPMI : public SurfReact {
 public:
  SurfReactPMI(class SPARTA *, int, char **);
  SurfReactPMI(class SPARTA *sparta) : SurfReact(sparta) {} // needed for Kokkos
  virtual ~SurfReactPMI();
  virtual void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  // reaction info, as read from file

  struct OneReaction {
    int active;                    // 1 if reaction is active
    int type;                      // reaction type = REFLECT, SPUTTER, ABSORB
    int style;                     // reaction style = SIMPLE
    int ncoeff;                    // # of numerical coeffs
    int nreactant,nproduct;        // # of reactants and products
    char **id_reactants,**id_products;  // species IDs of reactants/products
    int *reactants,*products;      // species indices of reactants/products
    double *coeff;                 // numerical coeffs for reaction
    char *id;                      // reaction ID (formula)
  };

  int last_outcome;                // 1=reflect, 2=sputter, 3=absorb

 protected:
  class RanKnuth *random;     // RNG for reaction probabilities

  int mode;                        // 0=constant, 1=file

  // constant-mode parameters
  double const_RN, const_RE, const_Y, const_Ebind;

  // file-mode HDF5 tables
  int nE, nA;
  std::vector<double> E_axis, A_axis;
  std::vector<double> RN_table, RE_table, spyld_table;
  double Ebind;

  // optional tabulated CDF distributions
  int nb_cdf;
  int has_cdf_R, has_cdf_Y;
  std::vector<double> edist_R, adist_R, edist_Y, adist_Y;

  // reaction list
  OneReaction *rlist;
  int nlist_prob;                  // # of reactions read from file
  int maxlist_prob;                // max # of reactions in rlist

  // per-species reaction role lookup (replaces generic reaction list)
  struct SpeciesReactions {
    int ireflect;                  // index into rlist, or -1
    int isputter;                  // index into rlist, or -1
    int iabsorb;                   // index into rlist, or -1
  };
  SpeciesReactions *species_reactions;

  std::string h5_path;

  virtual void init_reactions();
  void load_surface_file();
  double interp_table(const std::vector<double> &table,
                      double e_eV, double a_deg) const;
  double sample_thompson(double Eb, double Emax);
  void sample_cosine_direction(double *norm, double *dir, double vmag);
  double sample_from_cdf(const std::vector<double> &cdf, int iE, int iA);
  void readfile(char *);
  int readone(char *, char *, int &, int &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running SPARTA to see the offending line.

*/
