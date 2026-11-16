// Compile independently through each public include path (not in one TU: guards
// with identical names would hide the former incompatible Ceph declaration).
#include CORTEX_CONTRACT_HEADER
#include <cstddef>
#include <type_traits>

using Read = cortex_read_result (*)(uint32_t, const void*, uint64_t,
                                   const float*, uint32_t, uint32_t,
                                   uint32_t, uint32_t);
static_assert(CORTEX_API_VERSION == 2, "ambiguous version 1 must be rejected");
static_assert(std::is_same<decltype(&cortex_semantic_read), Read>::value,
              "semantic_read must take all eight arguments");
static_assert(std::is_same<decltype(&cortex_semantic_write),
              int (*)(uint32_t, const void*, uint64_t, const void*,
                      uint64_t, uint32_t)>::value);
static_assert(std::is_same<decltype(&cortex_api_version), uint32_t (*)()>::value);
static_assert(std::is_same<decltype(&cortex_driver_init), int (*)(const char*)>::value);
static_assert(std::is_same<decltype(&cortex_driver_shutdown), void (*)()>::value);
static_assert(std::is_same<decltype(&cortex_is_cluster_cached), int (*)(uint32_t)>::value);
static_assert(std::is_same<decltype(&cortex_get_last_cluster_id), uint32_t (*)()>::value);
static_assert(CORTEX_TOPK_MAX == 500);
static_assert(CORTEX_METRIC_L2 == 0 && CORTEX_METRIC_IP == 1);
static_assert(sizeof(cortex_topk_entry) == 24);
static_assert(alignof(cortex_topk_entry) == 8);
static_assert(offsetof(cortex_topk_entry, distance) == 0);
static_assert(offsetof(cortex_topk_entry, doc_addr) == 8);
static_assert(offsetof(cortex_topk_entry, doc_length) == 16);
static_assert(offsetof(cortex_topk_entry, _pad) == 20);
static_assert(sizeof(cortex_read_result) == 12008);
static_assert(offsetof(cortex_read_result, status) == 0);
static_assert(offsetof(cortex_read_result, count) == 4);
static_assert(offsetof(cortex_read_result, entries) == 8);
static_assert(std::is_same<decltype(cortex_read_result::status), int32_t>::value);
static_assert(std::is_same<decltype(cortex_read_result::count), uint32_t>::value);
static_assert(std::extent<decltype(cortex_read_result::entries)>::value == 500);
