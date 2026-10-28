#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <cerrno>
#include <cstring>
#include "smartssd_driver.h"

#ifndef CORTEX_FIXTURE_DIR
#define CORTEX_FIXTURE_DIR "/cortex/hw/test/fixtures"
#endif

#ifndef HW_LIB_PATH
#define HW_LIB_PATH "/cortex/build/lib/libcortex_hw_mock.so"
#endif

static std::string fixture_dir() {
    return CORTEX_FIXTURE_DIR;
}

static std::vector<uint8_t> load_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

static std::vector<float> load_floats(const std::string& path) {
    auto raw = load_file(path);
    std::vector<float> out(raw.size() / sizeof(float));
    memcpy(out.data(), raw.data(), out.size() * sizeof(float));
    return out;
}

static std::string init_json() {
    return std::string("{\"hw_lib_path\":\"") + HW_LIB_PATH + "\"}";
}

static std::string init_json_slots(uint32_t slots) {
    return std::string("{\"hw_lib_path\":\"") + HW_LIB_PATH
        + "\",\"max_cache_slots\":" + std::to_string(slots) + "}";
}

// ============================================================================
// Test 1: EvictOldest — load 5 clusters with cache=3, verify oldest 2 evicted
// ============================================================================
TEST(CacheStress, EvictOldest) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json_slots(3);
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto blob0 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob1 = load_file(fixture_dir() + "/cluster_1.bin");
    auto blob2 = load_file(fixture_dir() + "/cluster_2.bin");
    auto blob3 = load_file(fixture_dir() + "/cluster_3.bin");
    auto blob4 = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Load clusters 0,1,2 → fill the 3-slot cache
    driver.semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(1, blob1.data(), blob1.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(2, blob2.data(), blob2.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    // All three should be cached at this point
    EXPECT_TRUE(driver.is_cluster_cached(0));
    EXPECT_TRUE(driver.is_cluster_cached(1));
    EXPECT_TRUE(driver.is_cluster_cached(2));

    // Load cluster 3 → should evict cluster 0 (oldest, at back of LRU list)
    driver.semantic_read(3, blob3.data(), blob3.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_TRUE(driver.is_cluster_cached(3));
    EXPECT_TRUE(driver.is_cluster_cached(2));
    EXPECT_TRUE(driver.is_cluster_cached(1));
    EXPECT_FALSE(driver.is_cluster_cached(0));

    // Load cluster 4 → should evict cluster 1 (now the oldest)
    driver.semantic_read(4, blob4.data(), blob4.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_TRUE(driver.is_cluster_cached(4));
    EXPECT_TRUE(driver.is_cluster_cached(3));
    EXPECT_TRUE(driver.is_cluster_cached(2));
    EXPECT_FALSE(driver.is_cluster_cached(1));
    EXPECT_FALSE(driver.is_cluster_cached(0));

    driver.shutdown();
}

// ============================================================================
// Test 2: LRUReorder — access a middle entry to promote it, verify correct eviction
// ============================================================================
TEST(CacheStress, LRUReorder) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json_slots(3);
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto blob0 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob1 = load_file(fixture_dir() + "/cluster_1.bin");
    auto blob2 = load_file(fixture_dir() + "/cluster_2.bin");
    auto blob3 = load_file(fixture_dir() + "/cluster_3.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Load 0,1,2 → LRU order: [2(front), 1, 0(back)]
    driver.semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(1, blob1.data(), blob1.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(2, blob2.data(), blob2.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_TRUE(driver.is_cluster_cached(0));
    EXPECT_TRUE(driver.is_cluster_cached(1));
    EXPECT_TRUE(driver.is_cluster_cached(2));

    // Re-access cluster 0 → promotes it to front: LRU now [0(front), 2, 1(back)]
    driver.semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    // Load cluster 3 → should evict cluster 1 (back of LRU, least recently used)
    driver.semantic_read(3, blob3.data(), blob3.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    // Cluster 1 should be evicted; cluster 0 should survive (promoted by re-access)
    EXPECT_FALSE(driver.is_cluster_cached(1));
    EXPECT_TRUE(driver.is_cluster_cached(0));
    EXPECT_TRUE(driver.is_cluster_cached(2));
    EXPECT_TRUE(driver.is_cluster_cached(3));

    driver.shutdown();
}

// ============================================================================
// Test 3: CacheHitRate — load once, then 10 reads all hit the cache
// ============================================================================
TEST(CacheStress, CacheHitRate) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json_slots(3);
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Cold load — populates the cache
    cortex_read_result first = driver.semantic_read(
        0, blob.data(), blob.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    ASSERT_EQ(first.status, 0);
    ASSERT_TRUE(driver.is_cluster_cached(0));

    // 10 subsequent reads — all must be cache hits (cluster already in cache)
    const int kNumSearches = 10;
    uint32_t hit_count = 0;

    for (int i = 0; i < kNumSearches; ++i) {
        // Verify the cluster is cached BEFORE each read (expectation of a hit)
        if (driver.is_cluster_cached(0)) {
            ++hit_count;
        }

        cortex_read_result r = driver.semantic_read(
            0, blob.data(), blob.size(),
            queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
        EXPECT_EQ(r.status, 0);
        EXPECT_TRUE(driver.is_cluster_cached(0));
    }

    // All 10 reads should be cache hits = 100% hit rate
    EXPECT_EQ(hit_count, static_cast<uint32_t>(kNumSearches));

    driver.shutdown();
}

// ============================================================================
// Test 4: FullEviction — fill cache, evict all entries, verify empty
// ============================================================================
TEST(CacheStress, FullEviction) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json_slots(3);
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto blob0 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob1 = load_file(fixture_dir() + "/cluster_1.bin");
    auto blob2 = load_file(fixture_dir() + "/cluster_2.bin");
    auto blob3 = load_file(fixture_dir() + "/cluster_3.bin");
    auto blob4 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob5 = load_file(fixture_dir() + "/cluster_1.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Fill the 3-slot cache with clusters 0,1,2
    driver.semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(1, blob1.data(), blob1.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(2, blob2.data(), blob2.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_TRUE(driver.is_cluster_cached(0));
    EXPECT_TRUE(driver.is_cluster_cached(1));
    EXPECT_TRUE(driver.is_cluster_cached(2));

    // Load 3 more clusters — each evicts the LRU entry
    driver.semantic_read(3, blob3.data(), blob3.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(4, blob4.data(), blob4.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    driver.semantic_read(5, blob5.data(), blob5.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    // All three original clusters should be evicted
    EXPECT_FALSE(driver.is_cluster_cached(0));
    EXPECT_FALSE(driver.is_cluster_cached(1));
    EXPECT_FALSE(driver.is_cluster_cached(2));

    // The new clusters occupy the cache
    EXPECT_TRUE(driver.is_cluster_cached(3));
    EXPECT_TRUE(driver.is_cluster_cached(4));
    EXPECT_TRUE(driver.is_cluster_cached(5));

    driver.shutdown();
}

// ============================================================================
// Test 5: ConcurrentAccess — 4 threads reading the same cluster concurrently
// ============================================================================
TEST(CacheStress, ConcurrentAccess) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // First, populate the cache with a single read
    cortex_read_result first = driver.semantic_read(
        0, cluster_blob.data(), cluster_blob.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    ASSERT_EQ(first.status, 0);

    const int kNumThreads = 4;
    const int kIterationsPerThread = 100;
    std::atomic<int> errors{0};
    std::atomic<int> total_reads{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < kNumThreads; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < kIterationsPerThread; ++i) {
                cortex_read_result r = driver.semantic_read(
                    0, cluster_blob.data(), cluster_blob.size(),
                    queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
                if (r.status != 0) {
                    errors.fetch_add(1);
                }
                total_reads.fetch_add(1);
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    EXPECT_EQ(errors.load(), 0);
    EXPECT_EQ(total_reads.load(), kNumThreads * kIterationsPerThread);

    driver.shutdown();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
