/* ----------------------------------------------------------------------
    OpenEdge: Surface recycling reaction model for neutral transport.
    Incoming ions are neutralized and re-emitted with cosine angular
    distribution at a specified energy (Franck-Condon or thermal).
    Contributors:
      - Abdourahmane (Abdou) Diaw (ORNL, diawa@ornl.gov, 2025)
    https://github.com/ORNL-Fusion/OpenEdge
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(recycle,SurfReactRecycle)

#else

#ifndef SPARTA_SURF_REACT_RECYCLE_H
#define SPARTA_SURF_REACT_RECYCLE_H

#include "surf_react.h"

namespace SPARTA_NS {

class SurfReactRecycle : public SurfReact {
 public:
  SurfReactRecycle(class SPARTA *, int, char **);
  ~SurfReactRecycle();
  void init();
  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int);
  int match_reactant(char *, int);
  int match_product(char *, int);

  struct OneReaction {
    int active;
    int type;                      // EXCHANGE or DISSOCIATION
    int nreactant, nproduct;
    char **id_reactants, **id_products;
    int *reactants, *products;
    double prob;                   // recycling probability (0 to 1)
    double *energy;                // return energy per product [eV]
    char *id;
  };

 protected:
  class RanKnuth *random;
  OneReaction *rlist;
  int nlist_recycle, maxlist_recycle;

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
};

}

#endif
#endif
