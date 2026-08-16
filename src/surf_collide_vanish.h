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

#ifdef SURF_COLLIDE_CLASS

SurfCollideStyle(vanish,SurfCollideVanish)

#else

#ifndef SPARTA_SURF_COLLIDE_VANISH_H
#define SPARTA_SURF_COLLIDE_VANISH_H

#include "surf_collide.h"
#include "particle.h"
#include <unordered_map>

namespace SPARTA_NS {

class SurfCollideVanish : public SurfCollide {
 public:
  SurfCollideVanish(class SPARTA *, int, char **);
  SurfCollideVanish(class SPARTA *sparta) : SurfCollide(sparta) {} // needed for Kokkos
  virtual ~SurfCollideVanish();
  void init();
  Particle::OnePart *collide(Particle::OnePart *&, double &,
                             int, double *, int, int &);

  // cumulative real-atom weight absorbed by this collide (local to rank);
  // consumers (fix surface/emit/puff slave mode) take deltas between reads
  double escaped_weight() const { return wsum_cum; }

 protected:
  int me;
  int pweight_index;
  double wsum_cum;
  int logflag;
  int tallyflag;
  FILE *logfp;
  char *logfile;
  std::unordered_map<long long,long long> surf_hit_counts;
  std::unordered_map<long long,double> surf_wsums;

  void open_logfile();
  void close_logfile();
  void log_event(Particle::OnePart *, int, double *, double);
  void write_counts_file();
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
