/* ----------------------------------------------------------------------
    OpenEdge:
    Impurity Transport in Modeling of SOL and Edge Physics:
    This code built on top of SPARTA, a parallel DSMC code.

    compute impact/energy — per-collision incident kinetic-energy histogram
    at the wall (Di Genova fig 7c). Surf-tally driven: for every incident
    particle of the chosen mixture that hits a surf in the group, bin its
    impact KE (the sheath-boosted ion energy, since the mover kicks v before
    the collision) into a fixed energy grid, pweight-weighted. Output is a
    global vector of nbins counts (real atoms), for fix ave/time + dump/print.

    compute ID impact/energy group-ID mix-ID nbins Emin_eV Emax_eV [log|lin]

    Bin edges (reconstruct in post): log -> logspace(Emin,Emax,nbins+1);
    lin -> linspace(Emin,Emax,nbins+1). Under/overflow clamp to the edge bins.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(impact/energy,ComputeImpactEnergy)

#else

#ifndef SPARTA_COMPUTE_IMPACT_ENERGY_H
#define SPARTA_COMPUTE_IMPACT_ENERGY_H

#include "compute.h"

namespace SPARTA_NS {

class ComputeImpactEnergy : public Compute {
 public:
  ComputeImpactEnergy(class SPARTA *, int, char **);
  ~ComputeImpactEnergy();
  void init();
  void clear();
  void surf_tally(double, int, int, int, Particle::OnePart *,
                  Particle::OnePart *, Particle::OnePart *);
  void compute_vector();

 protected:
  int groupbit;            // surf group mask
  int imix;                // particle mixture index (species filter)
  int nbins;               // # energy bins
  int logscale;            // 1 = log-spaced bins, 0 = linear
  double emin, emax;       // energy range [eV]
  double invdE;            // 1/binwidth (linear)
  double logemin, invlogdE;// log10(emin), 1/binwidth in log10 (log)
  double *hist;            // local per-bin accumulator
  int dim;                 // domain dimension
};

}

#endif
#endif
