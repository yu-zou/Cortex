#include "mock_smartssd.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cortex {

// ─── IVF Header (64B) ───
// Offset 0x00: codebook_dim (M, 32b)
// Offset 0x04: pq_dim (DIM, 32b)
// Offset 0x08: pq_vectors (N, 32b)
// Offset 0x0C~0x3F: reserved[13] (52B)
static constexpr size_t kHeaderSize = 64;

static uint32_t read_le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
           (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static bool checked_product(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) return false;
    out = a * b;
    return true;
}

int MockSmartSSD::parse_header(const uint8_t* data, uint64_t size,
                                uint32_t& M, uint32_t& DIM, uint32_t& N) {
    if (size < kHeaderSize) return -EINVAL;

    // Little-endian read
    M   = read_le32(data);
    DIM = read_le32(data + 4);
    N   = read_le32(data + 8);

    if (M == 0 || DIM == 0 || DIM % M != 0) return -EINVAL;
    return 0;
}

MockSmartSSD::~MockSmartSSD() {
    shutdown();
}

int MockSmartSSD::init(const char*) {
    return 0;
}

void MockSmartSSD::shutdown() {
    std::lock_guard<std::mutex> lk(mutex_);
    clusters_.clear();
    hnsw_graphs_.clear();
}

int MockSmartSSD::load_cluster(
    uint32_t cluster_id,
    const void* object_data, uint64_t object_size,
    const void*, uint64_t, uint32_t)
{
    if (!object_data || object_size < kHeaderSize ||
        object_size > std::numeric_limits<size_t>::max()) {
        return -EINVAL;
    }

    const auto* base = static_cast<const uint8_t*>(object_data);

    // Auto-detect format: try IVF first, fallback to HNSW
    uint32_t M, DIM, N;
    int rc = parse_header(base, object_size, M, DIM, N);
    // A tagged CPU object must never fall through to the untagged HNSW parser.
    if (rc != 0 && read_le32(base + 12) == 0x31515043) return -EINVAL;
    if (rc == 0 && DIM % M == 0) {
        // IVF format detected
        uint32_t Ds = DIM / M;
        const bool compact = read_le32(base + 12) == 0x31515043;
        const uint32_t version = compact ? read_le32(base + 16) : 0;
        const uint32_t KS_dynamic = compact ? read_le32(base + 20) : 16;
        if (compact && ((version != 1 && version != 2) ||
                          (KS_dynamic != 16 && KS_dynamic != 256))) return -EINVAL;
        if (version == 2 && read_le32(base + 24) != 1) return -EINVAL;
        for (size_t off = version == 2 ? 28 : (compact ? 24 : 12);
             off < kHeaderSize; ++off)
            if (base[off] != 0) return -EINVAL;

        const uint64_t entry_bytes = uint64_t(M) + 16;
        const uint64_t centroid_bytes = version == 2 ? uint64_t(DIM) * 4 : 0;
        uint64_t pq_bytes, cb_bytes;
        if (!checked_product(N, entry_bytes, pq_bytes) ||
            !checked_product(DIM, uint64_t(KS_dynamic) * 4, cb_bytes) ||
            cb_bytes > object_size - kHeaderSize ||
            centroid_bytes > object_size - kHeaderSize - cb_bytes ||
            pq_bytes != object_size - kHeaderSize - cb_bytes - centroid_bytes ||
            cb_bytes / 4 > std::vector<float>().max_size() ||
            uint64_t(N) * M > std::vector<uint8_t>().max_size() ||
            N > std::vector<PQEntry>().max_size()) return -EINVAL;

        ClusterData data;
        data.M = M;
        data.DIM = DIM;
        data.Ds = Ds;
        data.num_vectors = N;
        data.ksub = KS_dynamic;
        data.compact_version = version;

        data.codebook.resize(static_cast<size_t>(M) * KS_dynamic * Ds);
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
                      "CPU compact requires IEEE float32");
        for (size_t i = 0; i < data.codebook.size(); ++i) {
            uint32_t bits = read_le32(base + kHeaderSize + i * 4);
            std::memcpy(&data.codebook[i], &bits, 4);
            if (!std::isfinite(data.codebook[i])) return -EINVAL;
        }

        data.coarse_centroid.resize(centroid_bytes / 4);
        for (size_t i = 0; i < data.coarse_centroid.size(); ++i) {
            uint32_t bits = read_le32(base + kHeaderSize + cb_bytes + i * 4);
            std::memcpy(&data.coarse_centroid[i], &bits, 4);
            if (!std::isfinite(data.coarse_centroid[i])) return -EINVAL;
        }

        data.codes_flat.resize(static_cast<size_t>(N) * M);
        data.entries.resize(N);

        const uint8_t* pq_base = base + kHeaderSize + cb_bytes + centroid_bytes;
        for (uint32_t i = 0; i < N; ++i) {
            const uint8_t* entry = pq_base + static_cast<uint64_t>(i) * entry_bytes;
            for (uint32_t m = 0; m < M; ++m)
                if (entry[m] >= KS_dynamic) return -EINVAL;
            std::memcpy(data.codes_flat.data() + static_cast<size_t>(i) * M, entry, M);

            uint64_t doc_addr = 0;
            for (int b = 0; b < 8; ++b)
                doc_addr |= static_cast<uint64_t>(entry[M + b]) << (b * 8);
            data.entries[i].doc_addr = doc_addr;

            uint64_t doc_len_raw = 0;
            for (int b = 0; b < 8; ++b)
                doc_len_raw |= static_cast<uint64_t>(entry[M + 8 + b]) << (b * 8);
            if (doc_len_raw > UINT32_MAX) return -EINVAL;
            data.entries[i].doc_length = static_cast<uint32_t>(doc_len_raw);
        }

        std::lock_guard<std::mutex> lk(mutex_);
        clusters_[cluster_id] = std::move(data);
        return 0;
    }

    // Try HNSW format: 16B header {num_nodes, entry_point, dim, max_degree}
    if (object_size < 16) return -EINVAL;
    uint32_t num_nodes  = base[0] | (base[1] << 8) | (base[2] << 16) | (base[3] << 24);
    uint32_t entry_pt   = base[4] | (base[5] << 8) | (base[6] << 16) | (base[7] << 24);
    uint32_t g_dim      = base[8] | (base[9] << 8) | (base[10] << 16) | (base[11] << 24);
    uint32_t max_deg    = base[12] | (base[13] << 8) | (base[14] << 16) | (base[15] << 24);

    if (num_nodes == 0 || g_dim == 0 || max_deg == 0 || entry_pt >= num_nodes)
        return -EINVAL;

    uint64_t expected = 16ULL + num_nodes * (g_dim * 4ULL + 4ULL + max_deg * 4ULL);
    if (object_size < expected) return -EINVAL;

    HNSWGraphData g;
    g.num_nodes   = num_nodes;
    g.entry_point = entry_pt;
    g.dim         = g_dim;
    g.max_degree  = max_deg;
    g.vectors.resize(static_cast<size_t>(num_nodes) * g_dim);
    g.num_nbrs.resize(num_nodes);
    g.adjacency.resize(static_cast<size_t>(num_nodes) * max_deg);
    g.doc_addrs.resize(num_nodes);

    const uint8_t* ptr = base + 16;
    for (uint32_t n = 0; n < num_nodes; ++n) {
        std::memcpy(g.vectors.data() + static_cast<size_t>(n) * g_dim,
                    ptr, g_dim * 4);
        ptr += g_dim * 4;

        g.num_nbrs[n] = ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
        ptr += 4;

        for (uint32_t d = 0; d < max_deg; ++d) {
            g.adjacency[static_cast<size_t>(n) * max_deg + d] =
                ptr[0] | (ptr[1] << 8) | (ptr[2] << 16) | (ptr[3] << 24);
            ptr += 4;
        }

        g.doc_addrs[n] = n;  // node_id = doc_addr
    }

    std::lock_guard<std::mutex> lk(mutex_);
    hnsw_graphs_[cluster_id] = std::move(g);
    return 0;
}

SearchResult MockSmartSSD::search(
    uint32_t cluster_id,
    const void*, uint64_t,
    const float* query_vec, uint32_t query_dim,
    uint32_t top_k, MetricType metric, SearchMode mode)
{
    std::lock_guard<std::mutex> lk(mutex_);

    if (mode == SearchMode::HNSW) {
        auto it = hnsw_graphs_.find(cluster_id);
        if (it == hnsw_graphs_.end()) {
            SearchResult r; r.status = -ENOENT; return r;
        }
        return search_hnsw(it->second, query_vec, query_dim, top_k, metric);
    }

    // Default: IVFPQ
    auto it = clusters_.find(cluster_id);
    if (it == clusters_.end()) {
        SearchResult r; r.status = -ENOENT; return r;
    }
    return search_ivfpq(it->second, query_vec, query_dim, top_k, metric);
}

SearchResult MockSmartSSD::search_ivfpq(
    const ClusterData& cd,
    const float* query_vec, uint32_t query_dim,
    uint32_t top_k, MetricType metric)
{
    const size_t M  = cd.M;
    const uint32_t ksub = cd.ksub;
    const uint32_t Ds = cd.Ds;
    const uint32_t N  = cd.num_vectors;

    if (!query_vec || query_dim != cd.DIM ||
        (metric != MetricType::L2 && metric != MetricType::InnerProduct)) {
        SearchResult r; r.status = -EINVAL; return r;
    }
    if (cd.compact_version == 1 && metric != MetricType::L2) {
        SearchResult r; r.status = -ENOTSUP; return r;
    }
    for (uint32_t j = 0; j < query_dim; ++j) {
        if (!std::isfinite(query_vec[j])) {
            SearchResult r; r.status = -EINVAL; return r;
        }
    }

    float coarse_score = 0.0f;
    std::vector<float> residual_query;
    if (cd.compact_version == 2) {
        if (metric == MetricType::L2) residual_query.resize(query_dim);
        for (uint32_t j = 0; j < query_dim; ++j) {
            if (metric == MetricType::L2) {
                residual_query[j] = query_vec[j] - cd.coarse_centroid[j];
                if (!std::isfinite(residual_query[j])) {
                    SearchResult r; r.status = -ERANGE; return r;
                }
            } else {
                coarse_score -= query_vec[j] * cd.coarse_centroid[j];
                if (!std::isfinite(coarse_score)) {
                    SearchResult r; r.status = -ERANGE; return r;
                }
            }
        }
    }
    const float* l2_query = residual_query.empty() ? query_vec : residual_query.data();

    // Precompute distance lookup table: M × ksub
    std::vector<float> dist_table(static_cast<size_t>(M) * ksub);

    if (metric == MetricType::L2) {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t k = 0; k < ksub; ++k) {
                float d = 0.0f;
                for (uint32_t j = 0; j < Ds; ++j) {
                    float diff = l2_query[m * Ds + j]
                               - cd.codebook[(m * ksub + k) * Ds + j];
                    d += diff * diff;
                }
                if (!std::isfinite(d)) {
                    SearchResult r; r.status = -ERANGE; return r;
                }
                dist_table[m * ksub + k] = d;
            }
        }
    } else {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t k = 0; k < ksub; ++k) {
                float dot = 0.0f;
                for (uint32_t j = 0; j < Ds; ++j) {
                    dot += query_vec[m * Ds + j]
                         * cd.codebook[(m * ksub + k) * Ds + j];
                }
                if (!std::isfinite(dot)) {
                    SearchResult r; r.status = -ERANGE; return r;
                }
                dist_table[m * ksub + k] = -dot;
            }
        }
    }

    // ADC: accumulate distance via table lookup
    std::vector<std::pair<float, uint32_t>> scores(N);
    for (uint32_t i = 0; i < N; ++i) {
        float adc = coarse_score;
        const uint8_t* codes = cd.codes_flat.data() + static_cast<size_t>(i) * M;
        for (uint32_t m = 0; m < M; ++m) {
            adc += dist_table[m * ksub + codes[m]];
        }
        scores[i] = {adc, i};
        if (!std::isfinite(adc)) {
            SearchResult r; r.status = -ERANGE; return r;
        }
    }

    const uint32_t actual_k = std::min(top_k, N);
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    SearchResult r;
    r.status = 0;
    r.entries.resize(actual_k);
    for (uint32_t i = 0; i < actual_k; ++i) {
        uint32_t idx = scores[i].second;
        r.entries[i].distance   = scores[i].first;
        r.entries[i].doc_addr   = cd.entries[idx].doc_addr;
        r.entries[i].doc_length = cd.entries[idx].doc_length;
    }
    return r;
}

SearchResult MockSmartSSD::search_hnsw(
    const HNSWGraphData& g,
    const float* query_vec, uint32_t query_dim,
    uint32_t top_k, MetricType metric)
{
    if (query_dim < g.dim) {
        SearchResult r; r.status = -EINVAL; return r;
    }

    // Compute distances for all nodes
    std::vector<std::pair<float, uint32_t>> scores(g.num_nodes);
    for (uint32_t n = 0; n < g.num_nodes; ++n) {
        float d2 = 0.0f;
        const float* vec = g.vectors.data() + static_cast<size_t>(n) * g.dim;
        if (metric == MetricType::L2) {
            for (uint32_t j = 0; j < g.dim; ++j) {
                float diff = query_vec[j] - vec[j];
                d2 += diff * diff;
            }
        } else {
            float dot = 0.0f;
            for (uint32_t j = 0; j < g.dim; ++j) {
                dot += query_vec[j] * vec[j];
            }
            d2 = -dot;  // lower = better for sorting
        }
        scores[n] = {d2, n};
    }

    const uint32_t actual_k = std::min(top_k, g.num_nodes);
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    SearchResult r;
    r.status = 0;
    r.entries.resize(actual_k);
    for (uint32_t i = 0; i < actual_k; ++i) {
        uint32_t nid = scores[i].second;
        r.entries[i].distance   = scores[i].first;
        r.entries[i].doc_addr   = g.doc_addrs[nid];
        r.entries[i].doc_length = 0;
    }
    return r;
}

bool MockSmartSSD::is_cluster_loaded(uint32_t cluster_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return clusters_.count(cluster_id) > 0 || hnsw_graphs_.count(cluster_id) > 0;
}

void MockSmartSSD::evict_cluster(uint32_t cluster_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    clusters_.erase(cluster_id);
    hnsw_graphs_.erase(cluster_id);
}

} // namespace cortex
