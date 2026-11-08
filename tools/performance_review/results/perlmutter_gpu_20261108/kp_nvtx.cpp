// Minimal Kokkos Tools -> NVTX connector: every parallel_for/reduce/scan, fence and
// deep_copy becomes an NVTX range carrying the Kokkos label, so nsys attributes
// CUDA kernels/memcpys to Kokkos kernel names. No extra fences requested.
#include "impl/Kokkos_Profiling_C_Interface.h"
#include <nvtx3/nvToolsExt.h>
#include <cstdio>
#include <cstdint>
#include <string>
extern "C" {
void kokkosp_request_tool_settings(const uint32_t, Kokkos_Tools_ToolSettings *s) { s->requires_global_fencing = false; }
void kokkosp_init_library(const int, const uint64_t, const uint32_t, Kokkos_Profiling_KokkosPDeviceInfo *) {}
void kokkosp_finalize_library() {}
static void push(const char *kind, const char *n) { std::string s(kind); s += ':'; s += n; nvtxRangePushA(s.c_str()); }
void kokkosp_begin_parallel_for(const char *n, const uint32_t, uint64_t *id) { *id = 0; push("for", n); }
void kokkosp_end_parallel_for(uint64_t) { nvtxRangePop(); }
void kokkosp_begin_parallel_reduce(const char *n, const uint32_t, uint64_t *id) { *id = 0; push("red", n); }
void kokkosp_end_parallel_reduce(uint64_t) { nvtxRangePop(); }
void kokkosp_begin_parallel_scan(const char *n, const uint32_t, uint64_t *id) { *id = 0; push("scan", n); }
void kokkosp_end_parallel_scan(uint64_t) { nvtxRangePop(); }
void kokkosp_begin_fence(const char *n, const uint32_t, uint64_t *id) { *id = 0; push("fence", n); }
void kokkosp_end_fence(uint64_t) { nvtxRangePop(); }
void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dst, const char *dn, const void *,
                             Kokkos_Profiling_SpaceHandle src, const char *sn, const void *, uint64_t bytes) {
  char buf[512]; snprintf(buf, sizeof buf, "copy:%s:%s->%s:%s:%llu", src.name, sn, dst.name, dn, (unsigned long long)bytes);
  nvtxRangePushA(buf);
}
void kokkosp_end_deep_copy() { nvtxRangePop(); }
void kokkosp_push_profile_region(const char *n) { nvtxRangePushA(n); }
void kokkosp_pop_profile_region() { nvtxRangePop(); }
}
