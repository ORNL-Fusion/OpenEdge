/* ----------------------------------------------------------------------
   SPARTA - Stochastic PArallel Rarefied-gas Time-accurate Analyzer
   http://sparta.github.io
   Steve Plimpton, sjplimp@gmail.com, Michael Gallis, magalli@sandia.gov
   Sandia National Laboratories

   Copyright (2014) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level SPARTA directory.
------------------------------------------------------------------------- */

#ifdef SURF_REACT_CLASS

SurfReactStyle(vanish_track,SurfReactVanish)

#else

#ifndef SPARTA_SURF_REACT_VANISH_H
#define SPARTA_SURF_REACT_VANISH_H

#include "surf_react.h"

namespace SPARTA_NS {

class SurfReactVanish : public SurfReact {
 public:
  SurfReactVanish(class SPARTA *, int, char **);
  virtual ~SurfReactVanish();

  int react(Particle::OnePart *&, int, double *, Particle::OnePart *&, int &);
  char *reactionID(int);
  double reaction_coeff(int) { return 0.0; }
  int match_reactant(char *, int) { return 1; }
  int match_product(char *, int) { return 1; }

 protected:
  int me;
  int logflag;
  FILE *logfp;
  char *logfile;

  void open_logfile();
  void close_logfile();
  void log_event(Particle::OnePart *, int);
};

}

#endif
#endif
