#pragma once
#include "smartssd_device.h"
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace cortex {

class MockSmartSSD : public SmartSSDDevice {
public:
    MockSmartSSD() = default;
    ~MockSmartSSD() override;

    int init(const char* config_json) override;
    void shutdown() override;

    int load_cluster(uint32_t cluster_id,
                     const void* codebook, uint64_t codebook_size,
                     const void* pq_payload, uint64_t payload_size,
                     uint32_t vector_count) override;

    SearchResult search(uint32_t cluster_id,
                        const void* object_data, uint64_t object_size,
                        const float* query_vec, uint32_t query_dim,
                        uint32_t top_k, MetricType metric) override;

    bool is_cluster_loaded(uint32_t cluster_id) const override;
    void evict_cluster(uint32_t cluster_id) override;

private:
    struct ClusterData {
        uint32_t M;
        uint32_t ksub;
        uint32_t D;
        uint32_t dsub;
        uint32_t num_vectors;
        std::vector<float> codebook;
        std::vector<uint8_t> pq_codes;
    };

    mutable std::mutex mutex_;
    std::unordered_map<uint32_t, ClusterData> clusters_;
};

}
