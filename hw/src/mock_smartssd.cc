#include "mock_smartssd.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>

namespace cortex {

namespace {

struct ClusterHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t M;
    uint32_t ksub;
    uint32_t D;
    uint32_t dsub;
    uint32_t num_vectors;
};

static constexpr uint32_t kMagic = 0x43505100;
static constexpr uint32_t kVersion = 1;
static constexpr size_t kHeaderSize = sizeof(ClusterHeader);

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
}

int MockSmartSSD::load_cluster(
    uint32_t cluster_id,
    const void* object_data, uint64_t object_size,
    const void*, uint64_t, uint32_t)
{
    if (!object_data || object_size < kHeaderSize) {
        return -EINVAL;
    }

    ClusterHeader hdr;
    std::memcpy(&hdr, object_data, kHeaderSize);

    if (hdr.magic != kMagic || hdr.version != kVersion) {
        return -EINVAL;
    }

    const uint64_t codebook_bytes = static_cast<uint64_t>(hdr.M) * hdr.ksub * hdr.dsub * sizeof(float);
    const uint64_t codes_bytes = static_cast<uint64_t>(hdr.num_vectors) * hdr.M;
    const uint64_t expected_size = kHeaderSize + codebook_bytes + codes_bytes;

    if (object_size < expected_size) {
        return -EINVAL;
    }

    const auto* base = static_cast<const uint8_t*>(object_data);

    ClusterData data;
    data.M = hdr.M;
    data.ksub = hdr.ksub;
    data.D = hdr.D;
    data.dsub = hdr.dsub;
    data.num_vectors = hdr.num_vectors;

    data.codebook.resize(static_cast<size_t>(hdr.M) * hdr.ksub * hdr.dsub);
    std::memcpy(data.codebook.data(), base + kHeaderSize, codebook_bytes);

    data.pq_codes.resize(static_cast<size_t>(hdr.num_vectors) * hdr.M);
    std::memcpy(data.pq_codes.data(), base + kHeaderSize + codebook_bytes, codes_bytes);

    std::lock_guard<std::mutex> lk(mutex_);
    clusters_[cluster_id] = std::move(data);
    return 0;
}

SearchResult MockSmartSSD::search(
    uint32_t cluster_id,
    const void*, uint64_t,
    const float* query_vec, uint32_t query_dim,
    uint32_t top_k, MetricType metric)
{
    std::lock_guard<std::mutex> lk(mutex_);

    auto it = clusters_.find(cluster_id);
    if (it == clusters_.end()) {
        SearchResult r;
        r.status = -ENOENT;
        return r;
    }

    const ClusterData& cd = it->second;
    const uint32_t M = cd.M;
    const uint32_t ksub = cd.ksub;
    const uint32_t dsub = cd.dsub;
    const uint32_t nv = cd.num_vectors;

    if (query_dim < M * dsub) {
        SearchResult r;
        r.status = -EINVAL;
        return r;
    }

    std::vector<float> dist_table(static_cast<size_t>(M) * ksub);

    if (metric == MetricType::L2) {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t k = 0; k < ksub; ++k) {
                float d = 0.0f;
                for (uint32_t j = 0; j < dsub; ++j) {
                    float diff = query_vec[m * dsub + j]
                               - cd.codebook[(m * ksub + k) * dsub + j];
                    d += diff * diff;
                }
                dist_table[m * ksub + k] = d;
            }
        }
    } else {
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t k = 0; k < ksub; ++k) {
                float dot = 0.0f;
                for (uint32_t j = 0; j < dsub; ++j) {
                    dot += query_vec[m * dsub + j]
                         * cd.codebook[(m * ksub + k) * dsub + j];
                }
                dist_table[m * ksub + k] = -dot;
            }
        }
    }

    std::vector<std::pair<float, uint64_t>> scores(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        float adc = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            adc += dist_table[m * ksub + cd.pq_codes[i * M + m]];
        }
        scores[i] = {adc, static_cast<uint64_t>(i)};
    }

    const uint32_t actual_k = std::min(top_k, nv);
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    SearchResult r;
    r.status = 0;
    r.entries.resize(actual_k);
    for (uint32_t i = 0; i < actual_k; ++i) {
        r.entries[i].vector_id = scores[i].second;
        r.entries[i].distance = scores[i].first;
    }
    return r;
}

bool MockSmartSSD::is_cluster_loaded(uint32_t cluster_id) const {
    std::lock_guard<std::mutex> lk(mutex_);
    return clusters_.count(cluster_id) > 0;
}

void MockSmartSSD::evict_cluster(uint32_t cluster_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    clusters_.erase(cluster_id);
}

}
