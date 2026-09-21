/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute surf/weighted — like compute surf, but weights per-surf flux
    tallies by the per-particle `pweight` custom attribute (from
    fix particle/weight) instead of the global fnum. Stock compute surf is
    unaware of pweight, so with OpenEdge's variable-weight macroparticles it
    undercounts fluxes by the pweight/fnum ratio. This compute fixes the
    deposition-flux tally so it is comparable to the (pweight-aware) grid
    density and the analytic erosion flux.

    Incidence-only: emission (surf/emit/source) is not tallied here — that
    side is compute surface/physical/sputter (erosion_flux). So nflux =
    deposition (redeposition) flux and nflux_incident = gross wall load;
    net erosion = erosion_flux - nflux is formed in post.

    Only the count/flux keywords are supported (num, numwt, nflux,
    nflux_incident, mflux, mflux_incident).

    Requires `fix particle/weight` to provide the `pweight` custom.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(surf/weighted,ComputeSurfWeighted)

#else

#ifndef SPARTA_COMPUTE_SURF_WEIGHTED_H
#define SPARTA_COMPUTE_SURF_WEIGHTED_H

#include "compute_surf.h"

namespace SPARTA_NS {

class ComputeSurfWeighted : public ComputeSurf {
 public:
  ComputeSurfWeighted(class SPARTA *, int, char **);
  void init();
  void surf_tally(double, int, int, int, Particle::OnePart *,
                  Particle::OnePart *, Particle::OnePart *);

 protected:
  int pweight_index;     // index of pweight custom attribute
  int pweight_ewhich;    // index into edvec
};

}

#endif
#endif
