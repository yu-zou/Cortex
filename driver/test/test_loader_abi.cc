#include "cortex/loader/cortex_driver_loader.h"
#include <gtest/gtest.h>
#include <array>
#include <cerrno>
#include <dlfcn.h>
#include <type_traits>

using Read = cortex_read_result (*)(uint32_t, const void*, uint64_t,
                                   const float*, uint32_t, uint32_t,
                                   uint32_t, uint32_t);
static_assert(std::is_same<decltype(cortex::CortexDriverLoader::API::semantic_read),
                           Read>::value);

TEST(LoaderABI, RejectAmbiguousVersionOneBeforeInit) {
  auto& loader = cortex::CortexDriverLoader::instance();
  loader.unload();
  EXPECT_EQ(loader.load(OLD_ABI_STUB, "abi-contract"), -ENOTSUP);
  EXPECT_FALSE(loader.is_loaded());
  EXPECT_EQ(loader.api.semantic_read, nullptr);
}

static void expect_empty_api(const cortex::CortexDriverLoader::API& api) {
  EXPECT_EQ(api.api_version, nullptr);
  EXPECT_EQ(api.driver_init, nullptr);
  EXPECT_EQ(api.driver_shutdown, nullptr);
  EXPECT_EQ(api.semantic_read, nullptr);
  EXPECT_EQ(api.semantic_write, nullptr);
  EXPECT_EQ(api.is_cluster_cached, nullptr);
  EXPECT_EQ(api.get_last_cluster_id, nullptr);
}

TEST(LoaderABI, MissingLastSymbolNeverPublishesPointers) {
  auto& loader = cortex::CortexDriverLoader::instance();
  loader.unload();
  EXPECT_EQ(loader.load(INCOMPLETE_ABI_STUB, "abi-contract"), -EINVAL);
  EXPECT_FALSE(loader.is_loaded());
  expect_empty_api(loader.api);
  loader.unload();
  expect_empty_api(loader.api);
  ASSERT_EQ(loader.load(ABI_STUB, "abi-contract"), 0);
  loader.unload();
  expect_empty_api(loader.api);
}

TEST(LoaderABI, FailedInitLeavesNoPublishedAPI) {
  auto& loader = cortex::CortexDriverLoader::instance();
  loader.unload();
  EXPECT_EQ(loader.load(ABI_STUB, "invalid-config"), -EINVAL);
  EXPECT_FALSE(loader.is_loaded());
  expect_empty_api(loader.api);
  EXPECT_EQ(loader.load(""), -ENOTSUP);
  expect_empty_api(loader.api);
}

TEST(LoaderABI, LoadedDriverCannotBeOverwrittenByFailedReload) {
  auto& loader = cortex::CortexDriverLoader::instance();
  loader.unload();
  ASSERT_EQ(loader.load(ABI_STUB, "abi-contract"), 0);
  auto read = loader.api.semantic_read;
  EXPECT_EQ(loader.load(INCOMPLETE_ABI_STUB), -EBUSY);
  EXPECT_TRUE(loader.is_loaded());
  EXPECT_EQ(loader.api.semantic_read, read);
  EXPECT_EQ(loader.api.get_last_cluster_id(), 73u);
  loader.unload();
  expect_empty_api(loader.api);
}

TEST(LoaderABI, EightArgumentsAndFullResultTail) {
  auto& loader = cortex::CortexDriverLoader::instance();
  ASSERT_EQ(loader.load(ABI_STUB, "abi-contract"), 0);
  const float query[] = {1.25f, -2.5f};
  struct Guarded {
    std::array<uint64_t, 4> before;
    cortex_read_result result;
    std::array<uint64_t, 4> after;
  } guarded{};
  guarded.before.fill(0x1122334455667788ULL);
  guarded.after.fill(0x8877665544332211ULL);
  for (uint32_t metric : {CORTEX_METRIC_L2, CORTEX_METRIC_IP}) {
    guarded.result = loader.api.semantic_read(
        73, "abi", 3, query, 2, CORTEX_TOPK_MAX, metric, 0x13579bdfu);
    EXPECT_EQ(guarded.result.status, metric == CORTEX_METRIC_IP ? -EIO : 0);
    ASSERT_EQ(guarded.result.count, 500u);
    for (uint32_t i = 0; i < guarded.result.count; ++i) {
      EXPECT_EQ(guarded.result.entries[i].distance, float(i) + 0.5f);
      EXPECT_EQ(guarded.result.entries[i].doc_addr, 0xfedcba9876540000ULL + i);
      EXPECT_EQ(guarded.result.entries[i].doc_length, 4096 + i);
      EXPECT_EQ(guarded.result.entries[i]._pad, 0xabcdef01u);
    }
    for (auto v : guarded.before) EXPECT_EQ(v, 0x1122334455667788ULL);
    for (auto v : guarded.after) EXPECT_EQ(v, 0x8877665544332211ULL);
  }
  EXPECT_EQ(loader.api.get_last_cluster_id(), 73u);
  EXPECT_EQ(loader.api.is_cluster_cached(73), 1);
  loader.unload();
  EXPECT_EQ(loader.api.semantic_read, nullptr);
}

TEST(LoaderABI, ActualDriverExportUninitializedError) {
  // Real driver export, no hardware initialization or fake Ceph environment.
  void* handle = dlopen(DRIVER_SO_PATH, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();
  auto version = reinterpret_cast<decltype(&cortex_api_version)>(
      dlsym(handle, "cortex_api_version"));
  auto read = reinterpret_cast<Read>(dlsym(handle, "cortex_semantic_read"));
  ASSERT_NE(version, nullptr);
  ASSERT_NE(read, nullptr);
  EXPECT_EQ(version(), 2u);
  const auto result = read(0, nullptr, 0, nullptr, 0, 500,
                           CORTEX_METRIC_IP, 0x13579bdfu);
  EXPECT_EQ(result.status, -EINVAL);
  EXPECT_EQ(result.count, 0u);
  dlclose(handle);
}
