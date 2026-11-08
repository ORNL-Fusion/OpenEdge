// Kokkos Tools counter. Does not add fences or time asynchronous launches.
// Build: c++ -shared -fPIC -std=c++17 -I lib/kokkos/core/src ... -o counts.so
// KOKKOS_TOOLS_LIBS=.../counts.so OE_KOKKOS_COUNTS_DIR=/absolute/output/path
#include "impl/Kokkos_Profiling_C_Interface.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unistd.h>

namespace {
struct Count { uint64_t calls=0,bytes=0; };
std::map<std::string,Count> counts;
std::mutex lock;
void add(const std::string &key,uint64_t bytes=0) {
  std::lock_guard<std::mutex> guard(lock);
  auto &c=counts[key]; c.calls++; c.bytes+=bytes;
}
void kernel(const char *kind,const char *name,uint64_t *id) {
  *id=0; add(std::string(kind)+":"+name);
}
}
extern "C" {
void kokkosp_request_tool_settings(const uint32_t,Kokkos_Tools_ToolSettings *s) {
  s->requires_global_fencing=false;
}
void kokkosp_init_library(const int,const uint64_t,const uint32_t,Kokkos_Profiling_KokkosPDeviceInfo *) {}
void kokkosp_begin_parallel_for(const char *n,const uint32_t,uint64_t *id) {kernel("for",n,id);}
void kokkosp_end_parallel_for(uint64_t) {}
void kokkosp_begin_parallel_reduce(const char *n,const uint32_t,uint64_t *id) {kernel("reduce",n,id);}
void kokkosp_end_parallel_reduce(uint64_t) {}
void kokkosp_begin_parallel_scan(const char *n,const uint32_t,uint64_t *id) {kernel("scan",n,id);}
void kokkosp_end_parallel_scan(uint64_t) {}
void kokkosp_begin_fence(const char *n,const uint32_t,uint64_t *id) {kernel("fence",n,id);}
void kokkosp_end_fence(uint64_t) {}
void kokkosp_begin_deep_copy(Kokkos_Profiling_SpaceHandle dst,const char *dn,const void *,
                           Kokkos_Profiling_SpaceHandle src,const char *sn,const void *,uint64_t bytes) {
  add(std::string("copy:")+src.name+":"+sn+"->"+dst.name+":"+dn,bytes);
}
void kokkosp_end_deep_copy() {}
void kokkosp_finalize_library() {
  const char *dir=getenv("OE_KOKKOS_COUNTS_DIR");
  const char *rank=getenv("OMPI_COMM_WORLD_RANK");
  if(!rank) rank=getenv("SLURM_PROCID");
  char path[4096];
  snprintf(path,sizeof(path),"%s/kokkos_counts_rank%s_pid%ld.tsv",dir?dir:".",rank?rank:"unknown",long(getpid()));
  FILE *f=fopen(path,"w"); if(!f) return;
  fprintf(f,"event\tcalls\tbytes\n");
  for(const auto &entry:counts) fprintf(f,"%s\t%llu\t%llu\n",entry.first.c_str(),
    (unsigned long long)entry.second.calls,(unsigned long long)entry.second.bytes);
  fclose(f);
}
}
