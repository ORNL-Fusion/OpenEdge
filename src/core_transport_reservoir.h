// Independent weighted reservoir slots (sampling with replacement).
// Each slot is an exponential race: min[-log(U)/weight]. Slotwise minima
// merge exactly across ranks; uniform choice of a slot needs no second
// weight factor. Hash inputs are event identities, never rank or order.
#ifndef OPENEDGE_CORE_TRANSPORT_RESERVOIR_H
#define OPENEDGE_CORE_TRANSPORT_RESERVOIR_H
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>
namespace SPARTA_NS {
struct CoreEntry {
  double key, weight, surf, x[3];
  CoreEntry() : key(std::numeric_limits<double>::infinity()), weight(0), surf(0), x{0,0,0} {}
};
inline uint64_t core_entry_hash(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);
  x = (x^(x>>27))*UINT64_C(0x94d049bb133111eb);
  return x^(x>>31);
}
inline void core_entry_offer(std::vector<CoreEntry> &slots, uint64_t event,
                             int surf, const double *x, double weight) {
  for (size_t k = 0; k < slots.size(); ++k) {
    uint64_t bits = core_entry_hash(event^core_entry_hash(k));
    double u = ((bits>>12)+0.5)/4503599627370496.0;
    double key = -std::log(u)/weight;
    if (key < slots[k].key) {
      slots[k].key = key; slots[k].weight = weight; slots[k].surf = surf;
      for (int d = 0; d < 3; ++d) slots[k].x[d] = x[d];
    }
  }
}
}
#endif
