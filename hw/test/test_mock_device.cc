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
    uint64_t vector_id;
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
        f.read(reinterpret_cast<char*>(&result[i].vector_id), sizeof(uint64_t));
        f.read(reinterpret_cast<char*>(&result[i].distance), sizeof(float));
    }
    return result;
}

struct ClusterHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t M;
    uint32_t ksub;
    uint32_t D;
    uint32_t dsub;
    uint32_t num_vectors;
};

static constexpr size_t kHeaderSize = 7 * sizeof(uint32_t);

std::vector<float> compute_adc_l2(
    const float* query, uint32_t M, uint32_t ksub, uint32_t dsub,
    const float* codebook, const uint8_t* pq_codes, uint32_t nv)
{
    std::vector<float> dist_table(M * ksub);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < ksub; ++k) {
            float d = 0.0f;
            for (uint32_t j = 0; j < dsub; ++j) {
                float diff = query[m * dsub + j] - codebook[(m * ksub + k) * dsub + j];
                d += diff * diff;
            }
            dist_table[m * ksub + k] = d;
        }
    }
    std::vector<float> dists(nv);
    for (uint32_t i = 0; i < nv; ++i) {
        float adc = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            adc += dist_table[m * ksub + pq_codes[i * M + m]];
        }
        dists[i] = adc;
    }
    return dists;
}

}

TEST(MockSmartSSD, LoadAndSearchMatchesGoldenDistances) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    ClusterHeader hdr;
    std::memcpy(&hdr, cluster_bytes.data(), kHeaderSize);
    const float* codebook = reinterpret_cast<const float*>(cluster_bytes.data() + kHeaderSize);
    const uint8_t* codes = cluster_bytes.data() + kHeaderSize
        + static_cast<size_t>(hdr.M) * hdr.ksub * hdr.dsub * sizeof(float);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.ksub, hdr.dsub, codebook, codes, hdr.num_vectors);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto result = device.search(0, nullptr, 0, query0, hdr.D, hdr.num_vectors, cortex::MetricType::L2);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(hdr.num_vectors, result.entries.size());

    std::vector<float> returned_dists(hdr.num_vectors);
    for (const auto& e : result.entries) {
        ASSERT_LT(e.vector_id, hdr.num_vectors);
        returned_dists[e.vector_id] = e.distance;
    }

    for (uint32_t i = 0; i < hdr.num_vectors; ++i) {
        EXPECT_NEAR(expected[i], returned_dists[i], 1e-4f)
            << "Mismatch at vector " << i;
    }
}

TEST(MockSmartSSD, TopKResultsMatchGolden) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    ClusterHeader hdr;
    std::memcpy(&hdr, cluster_bytes.data(), kHeaderSize);
    const float* codebook = reinterpret_cast<const float*>(cluster_bytes.data() + kHeaderSize);
    const uint8_t* codes = cluster_bytes.data() + kHeaderSize
        + static_cast<size_t>(hdr.M) * hdr.ksub * hdr.dsub * sizeof(float);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.ksub, hdr.dsub, codebook, codes, hdr.num_vectors);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    const uint32_t top_k = 10;
    const uint32_t actual_k = std::min(top_k, hdr.num_vectors);
    auto result = device.search(0, nullptr, 0, query0, hdr.D, top_k, cortex::MetricType::L2);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(actual_k, result.entries.size());

    std::vector<std::pair<float, uint64_t>> sorted_expected(hdr.num_vectors);
    for (uint32_t i = 0; i < hdr.num_vectors; ++i) {
        sorted_expected[i] = {expected[i], i};
    }
    std::sort(sorted_expected.begin(), sorted_expected.end());

    for (uint32_t i = 0; i < actual_k; ++i) {
        EXPECT_EQ(sorted_expected[i].second, result.entries[i].vector_id)
            << "Top-k order mismatch at rank " << i;
        EXPECT_NEAR(sorted_expected[i].first, result.entries[i].distance, 1e-4f)
            << "Distance mismatch at rank " << i;
    }
}

TEST(MockSmartSSD, SearchOnUnloadedClusterReturnsError) {
    cortex::MockSmartSSD device;
    float q[32] = {};
    auto result = device.search(99, nullptr, 0, q, 32, 10, cortex::MetricType::L2);
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

    ClusterHeader hdr;
    std::memcpy(&hdr, cluster_bytes.data(), kHeaderSize);

    auto queries = load_queries();
    const float* q0 = queries.data();

    std::vector<std::thread> threads;
    std::vector<int> statuses(10 * 100, -1);

    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&device, q0, &hdr, &statuses, t]() {
            for (int i = 0; i < 100; ++i) {
                auto r = device.search(0, nullptr, 0, q0, hdr.D, 5, cortex::MetricType::L2);
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

    ClusterHeader hdr;
    std::memcpy(&hdr, cluster_bytes.data(), kHeaderSize);
    const float* q0 = queries.data();

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto l2 = device.search(0, nullptr, 0, q0, hdr.D, hdr.num_vectors, cortex::MetricType::L2);
    auto ip = device.search(0, nullptr, 0, q0, hdr.D, hdr.num_vectors, cortex::MetricType::InnerProduct);

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
