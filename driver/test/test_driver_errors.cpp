#include <gtest/gtest.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cortex_driver_api.h"
#include "smartssd_driver.h"

// ---------------------------------------------------------------------------
// Error-code contract — maps driver's negated-errno convention to
// semantically-named constants that these tests verify.
// ---------------------------------------------------------------------------
static constexpr int CORTEX_ERROR_BAD_HANDLE = -EINVAL;  // null / invalid param
static constexpr int CORTEX_NOT_INITIALIZED  = -EINVAL;  // op before init
static constexpr int CORTEX_ERR_LIB_MISSING  = -ENOENT;  // dlopen(3) failure

#ifndef HW_LIB_PATH
#define HW_LIB_PATH "libcortex_hw_mock.so"
#endif

// ---------------------------------------------------------------------------
// Helper: build init JSON that uses the (mock) HW library
// ---------------------------------------------------------------------------
static std::string init_json() {
    return std::string("{\"hw_lib_path\":\"") + HW_LIB_PATH + "\"}";
}

// ---------------------------------------------------------------------------
// Test 1 — Pass null object_data (analogous to a null handle).
// The driver forwards to the mock device's load_cluster(), which rejects
// null data with -EINVAL.
// ---------------------------------------------------------------------------
TEST(DriverErrors, SearchNullHandle) {
    cortex::SmartSSDDriver driver;

    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    std::vector<float> query(32, 0.0f);

    // Use a cluster_id guaranteed not to be cached so load_cluster runs.
    cortex_read_result result = driver.semantic_read(
        9999,
        nullptr, 0,             // null "handle" → object_data
        query.data(), 32, 5,
        CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(result.status, CORTEX_ERROR_BAD_HANDLE);
    EXPECT_EQ(result.count, 0u);

    driver.shutdown();
}

// ---------------------------------------------------------------------------
// Test 2 — Call semantic_read without ever calling init.
// SmartSSDDriver::semantic_read() checks the initialized_ flag and
// immediately returns -EINVAL.
// ---------------------------------------------------------------------------
TEST(DriverErrors, SearchWithoutInit) {
    cortex::SmartSSDDriver driver;
    // NOTE: no init() call

    std::vector<float> query(32, 0.0f);
    std::vector<uint8_t> dummy(64, 0);

    cortex_read_result result = driver.semantic_read(
        0, dummy.data(), dummy.size(),
        query.data(), 32, 5,
        CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(result.status, CORTEX_NOT_INITIALIZED);
    EXPECT_EQ(result.count, 0u);
}

// ---------------------------------------------------------------------------
// Test 3 — Object too small for the mandatory 64-byte IVFPQ header.
// The mock device's load_cluster returns -EINVAL when object_size <
// kHeaderSize (64), which exercises the "invalid cluster" code path.
// ---------------------------------------------------------------------------
TEST(DriverErrors, InvalidClusterPath) {
    cortex::SmartSSDDriver driver;

    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    std::vector<float> query(32, 0.0f);
    // 16 bytes is well below the 64-byte header minimum
    std::vector<uint8_t> tiny_data(16, 0xFF);

    cortex_read_result result = driver.semantic_read(
        9998,
        tiny_data.data(), tiny_data.size(),
        query.data(), 32, 5,
        CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(result.status, CORTEX_ERROR_BAD_HANDLE);
    EXPECT_EQ(result.count, 0u);

    driver.shutdown();
}

// ---------------------------------------------------------------------------
// Test 4 — Completely corrupted cluster binary (garbage bytes).
// The header fields will be nonsensical so mock parse_header() returns
// -EINVAL.  The critical check is that the driver *does not crash* and
// surfaces a non-zero status.
// ---------------------------------------------------------------------------
TEST(DriverErrors, CorruptedClusterData) {
    cortex::SmartSSDDriver driver;

    std::string cfg = init_json();
    ASSERT_EQ(driver.init(cfg.c_str()), 0);

    std::vector<float> query(32, 0.0f);

    // 128 bytes of deterministic garbage
    std::vector<uint8_t> garbage(128);
    for (size_t i = 0; i < garbage.size(); ++i)
        garbage[i] = static_cast<uint8_t>(i * 37 + 0xAA);
    // Ensure parse_header fails: M == 0 → -EINVAL
    garbage[0] = 0;
    garbage[1] = 0;
    garbage[2] = 0;
    garbage[3] = 0;

    cortex_read_result result = driver.semantic_read(
        9997,
        garbage.data(), garbage.size(),
        query.data(), 32, 5,
        CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    // Must NOT crash — any non-zero status is acceptable
    EXPECT_NE(result.status, 0);
    EXPECT_EQ(result.count, 0u);

    driver.shutdown();
}

// ---------------------------------------------------------------------------
// Test 5 — init() with a HW-library path that cannot be dlopen(3)-ed.
// The driver returns -ENOENT when the shared object is missing.
// ---------------------------------------------------------------------------
TEST(DriverErrors, MissingHwLibrary) {
    cortex::SmartSSDDriver driver;

    std::string bad_cfg =
        "{\"hw_lib_path\":\"/cortex/does-not-exist/libcortex_hw_fake.so\"}";
    int rc = driver.init(bad_cfg.c_str());

    EXPECT_EQ(rc, CORTEX_ERR_LIB_MISSING);

    // shutdown() is safe even after a failed init (initialized_ stays false)
    driver.shutdown();
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
