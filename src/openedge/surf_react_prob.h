/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Abdourahmane Diaw,  diawa@ornl.gov (2023)
    Oak Ridge National Laboratory
https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(prob,SurfReactProb)

#else

#ifndef SPARTA_SURF_REACT_Prob_H
#define SPARTA_SURF_REACT_Prob_H

#include "surf_react.h"
#include <vector> 

namespace SPARTA_NS {

struct ReactionProbabilities {
    double reflection;
    double sputtering;
};

class SurfReactProb : public SurfReact {
 public:
  SurfReactProb(class SPARTA *, int, char **);
  SurfReactProb(class SPARTA *sparta) : SurfReact(sparta) {} // needed for Kokkos
  virtual ~SurfReactProb();
  virtual void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  double sputtering_yield_sample(const std::vector<double>& yields);

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
   double bohdansky_heavy_sputtering_yield(double , double , double , double , double , double );
   double bohdansky_light_sputtering_yield(double , double , double , double , double , double );
   double thomas_reflection(double , double , double , double , double , double );
   double wierzbicki_biersack(double , double , double , double , double , double );
   int findSpeciesID(double , double );
   void get_probability_ref_sputter(Particle::OnePart *&);
   ReactionProbabilities get_probability_sputter(Particle::OnePart *&);

     inline double clamp(double x, double lo, double hi) { return std::max(lo, std::min(hi, x)); }

// Linear interpolation on a sorted grid, clamped at the ends
inline double interp1d_clamped(const double* xs, const double* ys, int n, double x) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[n-1]) return ys[n-1];
    // find interval: xs[i] <= x < xs[i+1]
    int i = 0, j = n - 1;
    while (j - i > 1) {
        int m = (i + j) / 2;
        if (x >= xs[m]) i = m; else j = m;
    }
    double t = (x - xs[i]) / (xs[i+1] - xs[i]);
    return ys[i] + t * (ys[i+1] - ys[i]);
}

  double thomas_reflection(double Z1, double M1, double Z2, double M2,
                                double energy_eV);
 double yamamura_yield_normal(double Z1, double m1, double Z2, double m2,
                                    double energy_eV);

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
