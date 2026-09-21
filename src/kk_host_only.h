/* ----------------------------------------------------------------------
   OpenEdge: host-only container member of a class that Kokkos copies by value
   as a functor. Kokkos copies the enclosing object (UpdateKokkos and the KKCopy
   surf collide/react wrappers inside it) for every kernel launch; a plain
   std::vector member is then deep-copied and freed on every launch — for
   surf_react surface/pwi on the WEST 30-degree wedge that was ~40 ms of
   malloc/free per step (2026-09-15 nsys CPU sampling). This wrapper IS-A
   std::vector (every access, method and pass-by-reference is unchanged); only
   the copy constructor differs: the copy is empty. Device code never reads a
   host vector, and the host object itself is never copy-constructed with data
   (KKCopy uses memcpy), so nothing observes the difference.
------------------------------------------------------------------------- */
#ifndef SPARTA_KK_HOST_ONLY_H
#define SPARTA_KK_HOST_ONLY_H
#include <utility>
namespace SPARTA_NS {
template <class V>
struct KKHostOnly : public V {
  using V::V;
  KKHostOnly() = default;
  KKHostOnly(const KKHostOnly &) : V() {}                    // functor copy: empty
  KKHostOnly(KKHostOnly &&o) noexcept : V(std::move(o)) {}
  KKHostOnly &operator=(const KKHostOnly &o) { V::operator=(o); return *this; }
  KKHostOnly &operator=(KKHostOnly &&o) noexcept { V::operator=(std::move(o)); return *this; }
  KKHostOnly &operator=(const V &o) { V::operator=(o); return *this; }
  KKHostOnly &operator=(V &&o) { V::operator=(std::move(o)); return *this; }
};
}
#endif
