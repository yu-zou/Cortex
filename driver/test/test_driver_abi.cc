#include <gtest/gtest.h>
#include <dlfcn.h>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include "cortex_driver_api.h"

#ifndef HW_LIB_PATH
#define HW_LIB_PATH "/cortex/build/lib/libcortex_hw_mock.so"
#endif

#ifndef DRIVER_SO_PATH
#define DRIVER_SO_PATH "/build/lib/libcortex_driver.so"
#endif

#ifndef CORTEX_FIXTURE_DIR
#define CORTEX_FIXTURE_DIR "/cortex/hw/test/fixtures"
#endif

static std::string valid_init_json() {
    return std::string("{\"hw_lib_path\":\"") + HW_LIB_PATH + "\"}";
}

TEST(DriverABITest, ApiVersion) {
    EXPECT_EQ(cortex_api_version(), static_cast<uint32_t>(CORTEX_API_VERSION));
    EXPECT_EQ(cortex_api_version(), 1u);
}

TEST(DriverABITest, InitNullConfig) {
    int rc = cortex_driver_init(nullptr);
    if (rc == 0) {
        cortex_driver_shutdown();
    }
}

TEST(DriverABITest, InitValid) {
    std::string cfg = valid_init_json();
    int rc = cortex_driver_init(cfg.c_str());
    EXPECT_EQ(rc, 0);
    cortex_driver_shutdown();
}

TEST(DriverABITest, ReadBeforeInit) {
    std::vector<float> query(32, 0.5f);
    std::vector<uint8_t> dummy(64, 0);

    struct cortex_read_result r = cortex_semantic_read(
        0, dummy.data(), dummy.size(),
        query.data(), 32, 5, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    EXPECT_EQ(r.status, -EINVAL);
    EXPECT_EQ(r.count, 0u);
}

TEST(DriverABITest, FullLifecycle) {
    std::string cfg = valid_init_json();
    ASSERT_EQ(cortex_driver_init(cfg.c_str()), 0);

    std::string cluster_path = std::string(CORTEX_FIXTURE_DIR) + "/cluster_0.bin";
    std::ifstream f(cluster_path, std::ios::binary);
    ASSERT_TRUE(f.good()) << "Cannot open: " << cluster_path;
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read(reinterpret_cast<char*>(blob.data()), sz);

    std::string query_path = std::string(CORTEX_FIXTURE_DIR) + "/queries.bin";
    std::ifstream qf(query_path, std::ios::binary);
    ASSERT_TRUE(qf.good()) << "Cannot open: " << query_path;
    qf.seekg(0, std::ios::end);
    size_t qsz = qf.tellg();
    qf.seekg(0);
    std::vector<float> queries(qsz / sizeof(float));
    qf.read(reinterpret_cast<char*>(queries.data()), qsz);

    int wrc = cortex_semantic_write(0, blob.data(), blob.size(), nullptr, 0, 0);
    EXPECT_EQ(wrc, 0);

    struct cortex_read_result r = cortex_semantic_read(
        0, blob.data(), blob.size(),
        queries.data(), 32, 5, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    EXPECT_EQ(r.status, 0);
    EXPECT_GT(r.count, 0u);

    cortex_driver_shutdown();

    ASSERT_EQ(cortex_driver_init(cfg.c_str()), 0);
    cortex_driver_shutdown();
}

TEST(DriverABITest, DlopenAllSymbols) {
    void* handle = dlopen(DRIVER_SO_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(handle, nullptr) << "dlopen failed: " << dlerror() << " path=" << DRIVER_SO_PATH;

    const char* symbols[] = {
        "cortex_api_version",
        "cortex_driver_init",
        "cortex_driver_shutdown",
        "cortex_semantic_read",
        "cortex_semantic_write",
        "cortex_is_cluster_cached",
        "cortex_get_last_cluster_id",
    };

    for (const char* sym : symbols) {
        void* fn = dlsym(handle, sym);
        EXPECT_NE(fn, nullptr) << "dlsym failed for: " << sym;
    }

    dlclose(handle);
    std::cout << "DLOPEN_OK" << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
