#pragma once
#include <cstdint>
#include <vector>

namespace cortex {

enum class MetricType : uint32_t { L2 = 0, InnerProduct = 1 };
enum class SearchMode : uint32_t { IVFPQ = 0, HNSW = 1 };

/* Matches FPGA 128-bit result: dist(32b) + doc_addr(64b) + doc_length(32b) */
struct TopKEntry {
    float distance;
    uint64_t doc_addr;
    uint32_t doc_length;
};

struct SearchResult {
    int32_t status;   // 0 = success
    std::vector<TopKEntry> entries;
};

// Abstract interface — mock implements this, real FPGA device will too
class SmartSSDDevice {
public:
    virtual ~SmartSSDDevice() = default;
    
    // Initialize device with JSON config string
    virtual int init(const char* config_json) = 0;
    virtual void shutdown() = 0;
    
    // Load PQ codebook + IVF data for a cluster
    virtual int load_cluster(
        uint32_t cluster_id,
        const void* codebook, uint64_t codebook_size,
        const void* pq_payload, uint64_t payload_size,
        uint32_t vector_count) = 0;
    
    // Run vector search on a loaded cluster/graph
    virtual SearchResult search(
        uint32_t cluster_id,
        const void* object_data, uint64_t object_size,
        const float* query_vec, uint32_t query_dim,
        uint32_t top_k, MetricType metric, SearchMode mode) = 0;
    
    // Check if cluster data is loaded
    virtual bool is_cluster_loaded(uint32_t cluster_id) const = 0;
    
    // Evict cluster data (for LRU replacement)
    virtual void evict_cluster(uint32_t cluster_id) = 0;
};

} // namespace cortex
