/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix population/control: roulette down-sampling to cap the marker
    population. Every Nevery steps, markers are bucketed per (grid cell,
    species); any bucket over Nmax keeps Nmax randomly-chosen survivors
    whose pweights are rescaled by (bucket weight)/(kept weight), and the
    rest are deleted. Total physical atoms are conserved exactly;
    pweight-linear estimators are unbiased; composition is preserved
    per species.

    Syntax:
      fix ID population/control Nevery Nmax
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(population/control,FixPopulationControl)

#else

#ifndef SPARTA_FIX_POPULATION_CONTROL_H
#define SPARTA_FIX_POPULATION_CONTROL_H

#include "fix.h"

namespace SPARTA_NS {

class RanKnuth;

class FixPopulationControl : public Fix {
 public:
  FixPopulationControl(class SPARTA *, int, char **);
  ~FixPopulationControl();
  int setmask();
  void init();
  void end_of_step();

 private:
  int nmax_;                 // max markers per (cell, species)
  int pweight_index_;        // custom pweight attribute index
  int pweight_ewhich_;
  class RanKnuth *rng_;
  bigint ndeleted_total_;
};

}

#endif
#endif
