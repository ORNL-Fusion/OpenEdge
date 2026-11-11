/* ----------------------------------------------------------------------
   OpenEdge: two-sided auxiliary-layer exchange between the 1-D core
   model and weighted markers. Both descriptions are closed at the
   interface; a core layer and a marker layer exchange inventory with
   rates k_c = G/V_c (core -> markers) and k_m = G/V_m (markers -> core),
   G from the integrated flux operator at the interface. Over one
   coupling interval the exact two-box probabilities are used, so
   neither side can be overdrawn. Inventory owed to the markers leaves
   the core at once and waits in a per-state pending bank until it makes
   whole markers. No surface collisions, no MPI: the caller reduces the
   layer inventories and creates/deletes markers.
------------------------------------------------------------------------- */

#ifndef OPENEDGE_CORE_TRANSPORT_EXCHANGE_H
#define OPENEDGE_CORE_TRANSPORT_EXCHANGE_H

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace SPARTA_NS {

class CoreExchange {
 public:
  explicit CoreExchange(int nstates) :
    n_(nstates), pcm_(0.0), pmc_(0.0), pending_(nstates,0.0),
    to_core_(nstates,0.0), to_markers_(nstates,0.0), owed_(nstates,0.0) {}

  // exact two-state probabilities over tau
  void set_interval(double kc, double km, double tau) {
    if (!(kc >= 0.0) || !(km >= 0.0) || !(tau > 0.0) || !std::isfinite(kc + km + tau))
      throw std::runtime_error("core exchange: invalid rates or interval");
    const double k = kc + km;
    const double e = (k > 0.0) ? -std::expm1(-k*tau) : 0.0;
    pcm_ = (k > 0.0) ? kc/k*e : 0.0;
    pmc_ = (k > 0.0) ? km/k*e : 0.0;
  }
  double p_core_to_markers() const { return pcm_; }
  double p_markers_to_core() const { return pmc_; }

  // uniform in [0,1) from (marker id, window, stream): the same on any
  // decomposition, so marker decisions do not depend on the rank layout
  static double uniform(uint64_t id, uint64_t window, uint64_t stream) {
    uint64_t x = id*0x9E3779B97F4A7C15ULL + window*0xBF58476D1CE4E5B9ULL + stream*0x94D049BB133111EBULL;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    return (x >> 11)*(1.0/9007199254740992.0);
  }
  // does this layer marker move to the core in this interval?
  bool take(uint64_t id, uint64_t window) const { return uniform(id,window,7) < pmc_; }

  // One interval, per state, with globally reduced inputs:
  //   core_layer[z]  inventory of the core layer before the exchange
  //   removed[z]     summed weight of the layer markers that were taken
  //   wref[z]        weight of the markers to create for state z (> 0)
  // Outputs: core_change[z] to add to the core layer inventory (never
  // below -core_layer[z]) and nemit[z] markers of weight wref[z] to
  // create in the marker layer. What is owed but not yet emitted stays
  // in pending(z), part of the ledger.
  void transfer(const double *core_layer, const double *removed, const double *wref,
                double *core_change, int *nemit) {
    for (int z = 0; z < n_; ++z) {
      if (!(core_layer[z] >= 0.0) || !(removed[z] >= 0.0) || !(wref[z] > 0.0) ||
          !std::isfinite(core_layer[z] + removed[z] + wref[z]))
        throw std::runtime_error("core exchange: invalid inventory or marker weight");
      const double owed = core_layer[z]*pcm_;
      pending_[z] += owed;
      const double cnt = std::floor(pending_[z]/wref[z]);
      if (cnt > 2.0e9) throw std::runtime_error("core exchange: marker weight too small for the owed inventory");
      nemit[z] = (int) cnt;
      pending_[z] -= nemit[z]*wref[z];
      if (pending_[z] < 0.0) pending_[z] = 0.0;           // round-off only
      core_change[z] = removed[z] - owed;
      owed_[z] += owed; to_core_[z] += removed[z]; to_markers_[z] += nemit[z]*wref[z];
    }
  }

  double pending(int z) const { return pending_[z]; }
  double to_core(int z) const { return to_core_[z]; }           // endpoint transfers, cumulative
  double to_markers(int z) const { return to_markers_[z]; }
  double owed(int z) const { return owed_[z]; }
  int nstates() const { return n_; }

 private:
  int n_;
  double pcm_, pmc_;
  std::vector<double> pending_, to_core_, to_markers_, owed_;
};

}

#endif
