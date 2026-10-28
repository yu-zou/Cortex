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

TEST(SmartSSDDriverTest, InitSuccess) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json();
    int rc = driver.init(cfg.c_str());
    EXPECT_EQ(rc, 0);
    driver.shutdown();
}

TEST(SmartSSDDriverTest, SemanticReadGolden) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    const float* q = queries.data();
    cortex_read_result result = driver.semantic_read(
        0, cluster_blob.data(), cluster_blob.size(),
        q, query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(result.status, 0);
    EXPECT_GT(result.count, 0u);
    EXPECT_LE(result.count, top_k);

    driver.shutdown();
}

TEST(SmartSSDDriverTest, L2CacheHit) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    cortex_read_result r1 = driver.semantic_read(
        0, cluster_blob.data(), cluster_blob.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    EXPECT_EQ(r1.status, 0);

    EXPECT_TRUE(driver.is_cluster_cached(0));

    cortex_read_result r2 = driver.semantic_read(
        0, cluster_blob.data(), cluster_blob.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    EXPECT_EQ(r2.status, 0);
    EXPECT_EQ(r1.count, r2.count);

    EXPECT_TRUE(driver.is_cluster_cached(0));

    driver.shutdown();
}

TEST(SmartSSDDriverTest, L2CacheEviction) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json_slots(2);
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto blob0 = load_file(fixture_dir() + "/cluster_0.bin");
    auto blob1 = load_file(fixture_dir() + "/cluster_1.bin");
    auto blob2 = load_file(fixture_dir() + "/cluster_2.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    driver.semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    EXPECT_TRUE(driver.is_cluster_cached(0));

    driver.semantic_read(1, blob1.data(), blob1.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    EXPECT_TRUE(driver.is_cluster_cached(0));
    EXPECT_TRUE(driver.is_cluster_cached(1));

    driver.semantic_read(2, blob2.data(), blob2.size(),
        queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_TRUE(driver.is_cluster_cached(2));
    EXPECT_TRUE(driver.is_cluster_cached(1));
    EXPECT_FALSE(driver.is_cluster_cached(0));

    driver.shutdown();
}

TEST(SmartSSDDriverTest, UninitializedDriverError) {
    cortex::SmartSSDDriver driver;

    std::vector<float> query(32, 0.0f);
    std::vector<uint8_t> dummy(64, 0);

    cortex_read_result result = driver.semantic_read(
        0, dummy.data(), dummy.size(),
        query.data(), 32, 5, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(result.status, -EINVAL);
    EXPECT_EQ(result.count, 0u);
}

TEST(SmartSSDDriverTest, ThreadSafety) {
    cortex::SmartSSDDriver driver;
    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    auto cluster_blob = load_file(fixture_dir() + "/cluster_0.bin");
    auto queries = load_floats(fixture_dir() + "/queries.bin");

    const uint32_t query_dim = 32;
    const uint32_t top_k = 5;

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&]() {
            cortex_read_result r = driver.semantic_read(
                0, cluster_blob.data(), cluster_blob.size(),
                queries.data(), query_dim, top_k, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
            if (r.status != 0) errors.fetch_add(1);
        });
    }

    for (auto& th : threads) th.join();

    EXPECT_EQ(errors.load(), 0);
    driver.shutdown();
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
