#pragma once
#include "smartssd_device.h"
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace cortex {

// MockSmartSSD — software simulation of the FPGA Top-Level Module Design
// 
// Supports two search modes:
//   Mode 0 (IVFPQ): IVF cluster format
//     Offset 0x00: 64B Header {M, DIM, N, reserved[13]}
//     Legacy: all 13 reserved words zero, implicit KS=16 only.
//     CPU compact v1: LE u32 at +12 = 0x31515043 ("CPQ1"),
//       +16 = version 1, +20 = explicit KS (16 or 256), +24..63 zero.
//       v1 supports raw ADC L2 only; no residual/coarse/IP semantics.
//     CPU compact v2: same tag, +16 = 2, +20 = KS (16 or 256),
//       +24 = flags (exactly 1: residual-trained), +28..63 zero.
//     Offset 0x40: LE IEEE float32 codebook [M][KS][DIM/M].
//     v2 only: codebook followed by DIM LE float32 coarse centroid values.
//     Then PQ entries: N * (M + 16) bytes.
//     v2 caller supplies full raw q: L2 uses q-centroid; IP returns
//       -dot(q,centroid)-sum(dot(q_m,PQcode_m)), lower is better.
//   Mode 1 (HNSW): HNSW graph format
//     Offset 0x00: 16B Header {num_nodes, entry_point, dim, max_degree}
//     Offset 0x10+: Node data, each node: dim floats + num_nbrs uint32 + max_degree uint32 (adjacency)
//
// PQ entry format (IVFPQ): M bytes PQ codes + 8B doc_addr + 8B doc_length
// Exact size: 64 + KS*DIM*4 + (v2 ? DIM*4 : 0) + N*(M+16).
// No padding/trailing bytes; no unversioned residual inference.
// doc_length must fit uint32 (host result). CPU M96 stride is 112B,
// NOT the FPGA 128B stride/beat order. Never infer KS from object size.
// Search result format: dist(32b) + doc_addr(64b) + doc_length(32b) = 128 bit

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
                        uint32_t top_k, MetricType metric, SearchMode mode) override;

    bool is_cluster_loaded(uint32_t cluster_id) const override;
    void evict_cluster(uint32_t cluster_id) override;

private:

    struct PQEntry {
        uint64_t doc_addr;
        uint32_t doc_length;
        // PQ codes stored in flat array, indexed by entry_idx * M
    };

    struct ClusterData {
        uint32_t M;
        uint32_t DIM;
        uint32_t Ds;          // DIM / M, dimension per sub-vector
        uint32_t ksub;        // centroids per sub-quantizer in codebook (FPGA hardcoded=256)
        uint32_t num_vectors;
        uint32_t compact_version; // 0 legacy, 1 raw L2, 2 residual L2/IP
        std::vector<float> coarse_centroid; // DIM floats for v2 only
        std::vector<float>  codebook;     // M * ksub * Ds floats
        std::vector<uint8_t> codes_flat;  // N * M bytes, all PQ codes concatenated
        std::vector<PQEntry> entries;     // N entries, each with doc_addr + doc_length
    };

    // Parse 64B IVF Header
    static int parse_header(const uint8_t* data, uint64_t size,
                            uint32_t& M, uint32_t& DIM, uint32_t& N);

    // HNSW graph structures
    struct HNSWGraphData {
        uint32_t num_nodes;
        uint32_t entry_point;
        uint32_t dim;
        uint32_t max_degree;
        std::vector<float>    vectors;       // num_nodes * dim floats
        std::vector<uint32_t> num_nbrs;      // num_nodes entries
        std::vector<uint32_t> adjacency;     // num_nodes * max_degree entries
        std::vector<uint64_t> doc_addrs;     // num_nodes entries (node_id = doc_addr)
    };

    SearchResult search_ivfpq(const ClusterData& cd,
                              const float* query_vec, uint32_t query_dim,
                              uint32_t top_k, MetricType metric);

    SearchResult search_hnsw(const HNSWGraphData& g,
                             const float* query_vec, uint32_t query_dim,
                             uint32_t top_k, MetricType metric);

    mutable std::mutex mutex_;
    std::unordered_map<uint32_t, ClusterData> clusters_;
    std::unordered_map<uint32_t, HNSWGraphData> hnsw_graphs_;
};

} // namespace cortex
