/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.
    Oak Ridge National Laboratory
    https://github.com/ORNL-Fusion/OpenEdge

    fix population/control: caps the marker population and keeps the
    marker weights (pweight) even. Every Nevery steps the markers are
    bucketed per (grid cell, species).

    Resampling. A bucket with more than Nmax markers, or whose heaviest
    marker exceeds wratio times the bucket mean, is resampled to
    m = min(n, Nmax) markers of EQUAL weight W/m (W = bucket weight):
    systematic resampling on the cumulative weight, so a marker is kept
    with multiplicity proportional to its weight. Multiplicity k > 1 means
    the marker is split into k copies (all custom attributes copied, new
    ids); markers not selected are deleted. The bucket weight is conserved
    exactly, weighted estimators are unbiased for any weights, and the
    weight spread inside the bucket is reset at every pass. (Up to
    2026-11-14 the survivors were drawn uniformly, ignoring weight, and
    rescaled by W/W_kept: unbiased, but the weights compounded pass after
    pass and a few markers ended up carrying most of the inventory.)

    Splitting. After that, a marker heavier than split times the mean
    marker weight of its cell (all species) is split into up to 8 equal
    copies: heavy markers arriving in a region of light ones (W returning
    from the core to the edge) no longer reach the wall as one lump. This
    is the only step that adds markers.

    Syntax:
      fix ID population/control Nevery Nmax [wratio R] [split S]
        R default 4 (0 = resample only buckets over Nmax)
        S default 10 (0 = no splitting)
    Global vector: [1] markers deleted, [2] copies made (both at the last
    pass), [3] effective sample size (sum w)^2 / sum w^2 of all markers
    and [4] of the markers carrying the transit tag, after the last pass.
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
  double compute_vector(int);

 protected:                 // the Kokkos twin reuses these
  int nmax_;                 // max markers per (cell, species)
  int pweight_index_;        // custom pweight attribute index
  int pweight_ewhich_;
  class RanKnuth *rng_;
  bigint ndeleted_total_;
  double wratio_, split_;    // resample trigger (max/mean in a bucket), split trigger (w / cell mean)
  double tally_loc_[6];      // last pass: deleted, copies, sum w, sum w^2, tagged sum w, tagged sum w^2
};

}

#endif
#endif
