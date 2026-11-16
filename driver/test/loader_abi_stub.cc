// Transport-only DSO, compiled against the driver's authoritative header.
// This is not a Ceph or hardware integration implementation.
#include "cortex_driver_api.h"
#include <cerrno>
#include <cstdlib>
#include <cstring>

extern "C" {
uint32_t cortex_api_version() {
#ifdef CORTEX_TEST_OLD_VERSION
  return 1;
#else
  return CORTEX_API_VERSION;
#endif
}
int cortex_driver_init(const char* config) {
#if defined(CORTEX_TEST_OLD_VERSION) || defined(CORTEX_TEST_MISSING_SYMBOL)
  std::abort();  // version/symbol gates must run BEFORE init
#endif
  return std::strcmp(config, "abi-contract") == 0 ? 0 : -EINVAL;
}
void cortex_driver_shutdown() {}
int cortex_semantic_write(uint32_t, const void*, uint64_t,
                          const void*, uint64_t, uint32_t) { return 0; }
int cortex_is_cluster_cached(uint32_t id) { return id == 73; }
#ifndef CORTEX_TEST_MISSING_SYMBOL
uint32_t cortex_get_last_cluster_id() { return 73; }
#endif
cortex_read_result cortex_semantic_read(
    uint32_t id, const void* data, uint64_t size, const float* query,
    uint32_t dim, uint32_t k, uint32_t metric, uint32_t mode) {
  cortex_read_result r{};
  r.status = -EINVAL;
  if (id != 73 || !data || size != 3 ||
      std::memcmp(data, "abi", 3) || !query || dim != 2 ||
      query[0] != 1.25f || query[1] != -2.5f || k != CORTEX_TOPK_MAX ||
      metric > CORTEX_METRIC_IP || mode != 0x13579bdfu) return r;
  r.status = metric == CORTEX_METRIC_IP ? -EIO : 0;
  r.count = CORTEX_TOPK_MAX;
  for (uint32_t i = 0; i < r.count; ++i) {
    r.entries[i] = {float(i) + 0.5f, 0xfedcba9876540000ULL + i,
                    4096 + i, 0xabcdef01u};
  }
  return r;
}
}
