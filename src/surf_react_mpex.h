/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(mpex,SurfReactMpex)

#else

#ifndef SPARTA_SURF_REACT_MPEX_H
#define SPARTA_SURF_REACT_MPEX_H

#include "surf_react.h"
#include <vector> 
#include <cmath>


namespace SPARTA_NS {


class SurfReactMpex : public SurfReact {
 public:
  SurfReactMpex(class SPARTA *, int, char **);
  SurfReactMpex(class SPARTA *sparta) : SurfReact(sparta) {} // needed for Kokkos
  virtual ~SurfReactMpex();
  virtual void init();
  int react(Particle::OnePart *&ip,
          int isurf,
          double *norm,
          Particle::OnePart *&jp,
          int &velreset);

  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);


  // reaction info, as read from file

  struct OneReaction {
    int active;                    // 1 if reaction is active
    int type;                      // reaction type = DISSOCIATION, etc
    int style;                     // reaction style = ARRHENIUS, etc
    int ncoeff;                    // # of numerical coeffs
    int nreactant,nproduct;        // # of reactants and products
    char **id_reactants,**id_products;  // species IDs of reactants/products
    int *reactants,*products;      // species indices of reactants/products
    double *coeff;                 // numerical coeffs for reaction
    char *id;                      // reaction ID (formula)
  };

   double random_energy_thompson(double , double );


    inline double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }


    // helper for analytic Te(r) [eV]
  static inline double Te_profile(double x, double y)
  {
    const double R0 = 0.02;  // 2 cm
    double r = sqrt(x*x + y*y);
    return 1.0 + 4.0 * exp( -pow(r / R0, 12.0) );
  }


 protected:
  class RanKnuth *random;     // RNG for reaction probabilities

  OneReaction *rlist;              // list of all reactions read from file
  int nlist_prob;                  // # of reactions read from file
  int maxlist_prob;                // max # of reactions in rlist

  // possible reactions a reactant species is part of

  struct ReactionI {
    int *list;           // list of indices into rlist, ptr into indices
    int n;               // # of reactions in list
  };

  ReactionI *reactions;       // reactions for all species
  int *indices;               // master list of indices

  virtual void init_reactions();
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
