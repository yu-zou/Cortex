#include <gtest/gtest.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include "cortex_driver_api.h"

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

// ---------------------------------------------------------------------------
// Test 1: Load a cluster fixture via dlopen'ed mock HW, run search, verify
//         non-empty results.
// ---------------------------------------------------------------------------
TEST(DriverHWIntegration, LoadClusterAndSearch) {
    std::string cfg = init_json();
    ASSERT_EQ(cortex_driver_init(cfg.c_str()), 0)
        << "cortex_driver_init failed — check HW_LIB_PATH=" << HW_LIB_PATH;

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Write the cluster codebook so the driver can pq-decode during read
    int wrc = cortex_semantic_write(0,
        cluster_blob.data(), cluster_blob.size(),
        nullptr, 0, 0);
    EXPECT_EQ(wrc, 0) << "cortex_semantic_write(0) failed";

    // Search on the loaded cluster
    struct cortex_read_result r = cortex_semantic_read(
        0, cluster_blob.data(), cluster_blob.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(r.status, 0) << "cortex_semantic_read returned error "
        << r.status << " (" << strerror(-r.status) << ")";
    EXPECT_GT(r.count, 0u) << "Expected at least one result";
    EXPECT_LE(r.count, top_k) << "Result count exceeds top_k limit";

    cortex_driver_shutdown();
}

// ---------------------------------------------------------------------------
// Test 2: Perform 3 sequential searches on the same cluster; verify that
//         result counts are consistent across all calls.
// ---------------------------------------------------------------------------
TEST(DriverHWIntegration, SequentialSearches) {
    std::string cfg = init_json();
    ASSERT_EQ(cortex_driver_init(cfg.c_str()), 0);

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    int wrc = cortex_semantic_write(0,
        cluster_blob.data(), cluster_blob.size(),
        nullptr, 0, 0);
    ASSERT_EQ(wrc, 0);

    // Run 3 sequential searches
    struct cortex_read_result results[3];
    for (int i = 0; i < 3; ++i) {
        results[i] = cortex_semantic_read(
            0, cluster_blob.data(), cluster_blob.size(),
            queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

        EXPECT_EQ(results[i].status, 0)
            << "Search " << i << " failed with status " << results[i].status;
        EXPECT_GT(results[i].count, 0u)
            << "Search " << i << " returned zero results";
    }

    // All 3 runs should report the same number of results
    EXPECT_EQ(results[0].count, results[1].count);
    EXPECT_EQ(results[1].count, results[2].count);

    cortex_driver_shutdown();
}

// ---------------------------------------------------------------------------
// Test 3: Evict a cluster by overflowing a small cache, then search it
//         again; verify the driver transparently reloads (cache-miss path).
// ---------------------------------------------------------------------------
TEST(DriverHWIntegration, ClusterEvictAndReload) {
    // Limit cache to 2 slots to force eviction on the 3rd load
    std::string cfg = init_json_slots(2);
    ASSERT_EQ(cortex_driver_init(cfg.c_str()), 0);

    auto blob0 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob1 = load_file(fixture_dir() + "/cluster_1.bin");
    auto blob2 = load_file(fixture_dir() + "/cluster_2.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    // Load cluster 0 and 1 — both should stay cached
    ASSERT_EQ(cortex_semantic_write(0,
        blob0.data(), blob0.size(), nullptr, 0, 0), 0);
    ASSERT_EQ(cortex_semantic_write(1,
        blob1.data(), blob1.size(), nullptr, 0, 0), 0);

    struct cortex_read_result r0 = cortex_semantic_read(
        0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    ASSERT_EQ(r0.status, 0);
    ASSERT_GT(r0.count, 0u);

    EXPECT_NE(cortex_is_cluster_cached(0), 0)
        << "Cluster 0 should be cached after first search";
    EXPECT_NE(cortex_is_cluster_cached(1), 0)
        << "Cluster 1 should be cached after write";

    // Load cluster 2 — with 2-slot cache this evicts the least-recently used
    // (cluster 0 if LRU, cluster 1 otherwise; depends on impl — we just
    //  verify that the old cluster's data gets transparently reloaded).
    ASSERT_EQ(cortex_semantic_write(2,
        blob2.data(), blob2.size(), nullptr, 0, 0), 0);

    struct cortex_read_result r2 = cortex_semantic_read(
        2, blob2.data(), blob2.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    ASSERT_EQ(r2.status, 0);

    // At this point cluster 0 may or may not be cached — but searching it
    // again must succeed (driver reloads on cache miss).
    struct cortex_read_result r_reload = cortex_semantic_read(
        0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(r_reload.status, 0)
        << "Reload of evicted cluster 0 failed — cache miss handler broken";
    EXPECT_GT(r_reload.count, 0u)
        << "Reload of cluster 0 returned zero results";
    // Results should match original
    EXPECT_EQ(r_reload.count, r0.count)
        << "Reloaded cluster 0 returned different result count";

    cortex_driver_shutdown();
}

// ---------------------------------------------------------------------------
// Test 4: Call cortex_semantic_read before cortex_driver_init; verify the
//         driver returns -EINVAL (not a crash or garbage).
// ---------------------------------------------------------------------------
TEST(DriverHWIntegration, SearchBeforeInit) {
    std::vector<float> query(32, 0.5f);
    std::vector<uint8_t> dummy(64, 0);

    struct cortex_read_result r = cortex_semantic_read(
        0, dummy.data(), dummy.size(),
        query.data(), 32, 5, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(r.status, -EINVAL)
        << "Expected -EINVAL when searching before init, got " << r.status;
    EXPECT_EQ(r.count, 0u)
        << "Expected zero count when searching before init";
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
