#pragma once
#include "cortex_driver_api.h"
#include "smartssd_device.h"
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace cortex {

class SmartSSDDriver {
public:
    SmartSSDDriver();
    ~SmartSSDDriver();

    int init(const char* config_json);
    void shutdown();

    cortex_read_result semantic_read(
        uint32_t cluster_id, const void* object_data, uint64_t object_size,
        const float* query_vec, uint32_t query_dim, uint32_t top_k,
        uint32_t metric_type, uint32_t search_mode);

    int semantic_write(
        uint32_t cluster_id, const void* codebook, uint64_t codebook_size,
        const void* pq_payload, uint64_t payload_size, uint32_t vector_count);

    bool is_cluster_cached(uint32_t cluster_id) const;
    bool was_codebook_reloaded(uint32_t cluster_id) const;
    uint32_t get_last_cluster_id() const;

private:
    // LRU eviction — must be called with mutex held
    void evict_lru_locked();

    std::unique_ptr<SmartSSDDevice> device_;
    void* hw_handle_ = nullptr;  // dlopen handle for hw mock .so

    mutable std::mutex mutex_;
    // LRU: front = most recent, back = least recent
    std::list<uint32_t> lru_order_;
    std::unordered_map<uint32_t, std::list<uint32_t>::iterator> cache_map_;
    std::unordered_map<uint32_t, bool> reload_flags_;
    uint32_t max_cache_slots_ = 64;
    uint32_t last_cluster_id_ = 0;
    bool initialized_ = false;
};

} // namespace cortex
