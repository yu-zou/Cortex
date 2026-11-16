#include "loader_lease_probe.h"
#include <cerrno>
#include <cstring>

namespace {
LoaderLeaseProbe* probe = nullptr;
int* resource = nullptr;
__attribute__((destructor)) void unloaded() {
  if (probe) {
    ++probe->unloads;
    probe->unload_order = ++probe->sequence;
  }
}
}
extern "C" {
void cortex_test_set_probe(LoaderLeaseProbe* p) { probe = p; }
uint32_t cortex_api_version() { return CORTEX_API_VERSION; }
int cortex_driver_init(const char*) { resource = new int(42); return 0; }
void cortex_driver_shutdown() {
  if (probe) {
    ++probe->shutdowns;
    probe->shutdown_order = ++probe->sequence;
  }
  delete resource; resource = nullptr;
  if (probe) { ++probe->frees; probe->free_order = ++probe->sequence; }
}
int cortex_semantic_write(uint32_t, const void*, uint64_t,
                          const void*, uint64_t, uint32_t) { return 0; }
int cortex_is_cluster_cached(uint32_t) { return 0; }
uint32_t cortex_get_last_cluster_id() { return resource ? *resource : 0; }
cortex_read_result cortex_semantic_read(uint32_t id, const void* b, uint64_t s,
    const float* q, uint32_t d, uint32_t k, uint32_t m, uint32_t mode) {
  if (probe && probe->read) return probe->read(id,b,s,q,d,k,m,mode);
  return {};
}
}
