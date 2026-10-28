#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>

#include "smartssd_device.h"
#include "mock_smartssd.h"
#include "fixture_loader.h"

#ifndef CORTEX_FIXTURE_DIR
#define CORTEX_FIXTURE_DIR "/cortex/hw/test/fixtures/"
#endif

namespace {

static const std::string kFixtureDir = CORTEX_FIXTURE_DIR;

std::vector<uint8_t> load_cluster_file(int n) {
    return cortex::test::load_uint8_bin(kFixtureDir + "cluster_" + std::to_string(n) + ".bin");
}

std::vector<float> load_queries() {
    return cortex::test::load_float_bin(kFixtureDir + "queries.bin");
}

std::vector<float> load_golden_distances() {
    return cortex::test::load_float_bin(kFixtureDir + "golden_distances.bin");
}

struct GoldenTopKEntry {
    uint64_t doc_addr;
    float distance;
};

std::vector<GoldenTopKEntry> load_golden_topk() {
    std::ifstream f(kFixtureDir + "golden_topk.bin", std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open golden_topk.bin");
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    size_t count = bytes / (sizeof(uint64_t) + sizeof(float));
    std::vector<GoldenTopKEntry> result(count);
    for (size_t i = 0; i < count; ++i) {
        f.read(reinterpret_cast<char*>(&result[i].doc_addr), sizeof(uint64_t));
        f.read(reinterpret_cast<char*>(&result[i].distance), sizeof(float));
    }
    return result;
}

// IVF Header (64B): {M(u32), DIM(u32), N(u32), reserved[13](u32)}
static constexpr size_t kHeaderSize = 64;

// Parse M, DIM, N from 64B Header; derive KS and Ds
struct ParsedHeader {
    uint32_t M;
    uint32_t DIM;
    uint32_t N;
    uint32_t KS;    // derived from data size
    uint32_t Ds;
};

ParsedHeader parse_header(const uint8_t* data, uint64_t object_size) {
    ParsedHeader h{};
    // little-endian read
    h.M   = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    h.DIM = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24);
    h.N   = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24);
    h.Ds = h.DIM / h.M;

    // KS hardcoded to 16 (from metadata.json: ksub=16)
    // File-size derivation was unreliable with regenerated IVF-format fixtures
    h.KS = 16;
    (void)object_size;  // not used
    return h;
}

// extract from codebook at specified position
const float* codebook_ptr(const uint8_t* data, uint64_t) {
    return reinterpret_cast<const float*>(data + kHeaderSize);
}

// PQ codes pointer (M bytes per vector)
const uint8_t* pq_codes_ptr(const uint8_t* data, uint64_t object_size,
                            const ParsedHeader& h) {
    uint64_t cb_bytes = static_cast<uint64_t>(h.M) * h.KS * h.Ds * 4;
    return data + kHeaderSize + cb_bytes;
}

std::vector<float> compute_adc_l2(
    const float* query, uint32_t M, uint32_t KS, uint32_t Ds,
    const float* codebook, const uint8_t* pq_codes, uint32_t N)
{
    std::vector<float> dist_table(M * KS);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < KS; ++k) {
            float d = 0.0f;
            for (uint32_t j = 0; j < Ds; ++j) {
                float diff = query[m * Ds + j] - codebook[(m * KS + k) * Ds + j];
                d += diff * diff;
            }
            dist_table[m * KS + k] = d;
        }
    }
    std::vector<float> dists(N);
    for (uint32_t i = 0; i < N; ++i) {
        float adc = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            adc += dist_table[m * KS + pq_codes[i * M + m]];
        }
        dists[i] = adc;
    }
    return dists;
}

// extract doc_addr from PQ entry (offset M, 8 bytes LE)
uint64_t extract_doc_addr(const uint8_t* entry, uint32_t M) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
        v |= static_cast<uint64_t>(entry[M + b]) << (b * 8);
    return v;
}

}  // namespace

TEST(MockSmartSSD, LoadAndSearchMatchesGoldenDistances) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* cb = codebook_ptr(cluster_bytes.data(), cluster_bytes.size());
    // PQ codes: each entry = M+16 bytes, extract M bytes of PQ codes
    uint32_t entry_bytes = hdr.M + 16;
    const uint8_t* raw_pq = pq_codes_ptr(cluster_bytes.data(), cluster_bytes.size(), hdr);

    // extract pure PQ codes
    std::vector<uint8_t> codes_flat(static_cast<size_t>(hdr.N) * hdr.M);
    for (uint32_t i = 0; i < hdr.N; ++i)
        std::memcpy(codes_flat.data() + i * hdr.M, raw_pq + i * entry_bytes, hdr.M);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.KS, hdr.Ds, cb, codes_flat.data(), hdr.N);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto result = device.search(0, nullptr, 0, query0, hdr.DIM, hdr.N, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(hdr.N, result.entries.size());

    // sort results by doc_addr (=vector index) for comparison
    std::vector<float> returned_dists(hdr.N);
    for (const auto& e : result.entries) {
        ASSERT_LT(e.doc_addr, hdr.N);
        returned_dists[e.doc_addr] = e.distance;
    }

    for (uint32_t i = 0; i < hdr.N; ++i) {
        EXPECT_NEAR(expected[i], returned_dists[i], 1e-4f)
            << "Mismatch at vector " << i;
    }
}

TEST(MockSmartSSD, TopKResultsMatchGolden) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* cb = codebook_ptr(cluster_bytes.data(), cluster_bytes.size());
    uint32_t entry_bytes = hdr.M + 16;
    const uint8_t* raw_pq = pq_codes_ptr(cluster_bytes.data(), cluster_bytes.size(), hdr);

    std::vector<uint8_t> codes_flat(static_cast<size_t>(hdr.N) * hdr.M);
    for (uint32_t i = 0; i < hdr.N; ++i)
        std::memcpy(codes_flat.data() + i * hdr.M, raw_pq + i * entry_bytes, hdr.M);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.KS, hdr.Ds, cb, codes_flat.data(), hdr.N);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    const uint32_t top_k = 10;
    const uint32_t actual_k = std::min(top_k, hdr.N);
    auto result = device.search(0, nullptr, 0, query0, hdr.DIM, top_k, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(actual_k, result.entries.size());

    // expected Top-K: ascending by distance, doc_addr = vector index
    std::vector<std::pair<float, uint64_t>> sorted_expected(hdr.N);
    for (uint32_t i = 0; i < hdr.N; ++i)
        sorted_expected[i] = {expected[i], i};
    std::sort(sorted_expected.begin(), sorted_expected.end());

    for (uint32_t i = 0; i < actual_k; ++i) {
        EXPECT_EQ(sorted_expected[i].second, result.entries[i].doc_addr)
            << "Top-k order mismatch at rank " << i;
        EXPECT_NEAR(sorted_expected[i].first, result.entries[i].distance, 1e-4f)
            << "Distance mismatch at rank " << i;
    }
}

TEST(MockSmartSSD, SearchOnUnloadedClusterReturnsError) {
    cortex::MockSmartSSD device;
    float q[32] = {};
    auto result = device.search(99, nullptr, 0, q, 32, 10, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    EXPECT_NE(0, result.status);
    EXPECT_EQ(-ENOENT, result.status);
}

TEST(MockSmartSSD, EvictClearsCachedData) {
    auto cluster_bytes = load_cluster_file(0);
    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));
    EXPECT_TRUE(device.is_cluster_loaded(0));
    device.evict_cluster(0);
    EXPECT_FALSE(device.is_cluster_loaded(0));
}

TEST(MockSmartSSD, MultipleClustersLoadedSimultaneously) {
    cortex::MockSmartSSD device;
    for (int i = 0; i < 4; ++i) {
        auto bytes = load_cluster_file(i);
        ASSERT_EQ(0, device.load_cluster(static_cast<uint32_t>(i), bytes.data(), bytes.size(), nullptr, 0, 0));
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(device.is_cluster_loaded(static_cast<uint32_t>(i)));
    }
}

TEST(MockSmartSSD, ConcurrentSearchThreadSafety) {
    auto cluster_bytes = load_cluster_file(0);
    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    auto queries = load_queries();
    const float* q0 = queries.data();

    std::vector<std::thread> threads;
    std::vector<int> statuses(10 * 100, -1);

    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&device, q0, &hdr, &statuses, t]() {
            for (int i = 0; i < 100; ++i) {
                auto r = device.search(0, nullptr, 0, q0, hdr.DIM, 5, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
                statuses[t * 100 + i] = r.status;
            }
        });
    }
    for (auto& th : threads) th.join();

    for (int i = 0; i < 10 * 100; ++i) {
        EXPECT_EQ(0, statuses[i]) << "Thread error at index " << i;
    }
}

TEST(MockSmartSSD, L2VsIPMetricProduceDifferentResults) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* q0 = queries.data();

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto l2 = device.search(0, nullptr, 0, q0, hdr.DIM, hdr.N, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    auto ip = device.search(0, nullptr, 0, q0, hdr.DIM, hdr.N, cortex::MetricType::InnerProduct, cortex::SearchMode::IVFPQ);

    ASSERT_EQ(0, l2.status);
    ASSERT_EQ(0, ip.status);
    ASSERT_FALSE(l2.entries.empty());
    ASSERT_FALSE(ip.entries.empty());

    bool any_diff = false;
    for (size_t i = 0; i < l2.entries.size() && i < ip.entries.size(); ++i) {
        if (std::abs(l2.entries[i].distance - ip.entries[i].distance) > 1e-6f) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "L2 and IP metrics produced identical distances";
}
