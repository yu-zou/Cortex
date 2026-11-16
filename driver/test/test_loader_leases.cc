#include "cortex/loader/cortex_driver_loader.h"
#include <gtest/gtest.h>
#include <cerrno>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <type_traits>

static_assert(std::is_nothrow_destructible<cortex::CortexDriverLoader::Lease>::value);
static_assert(!std::is_constructible<cortex::CortexDriverLoader::Lease,
                                   std::shared_ptr<void>>::value);

TEST(LoaderLeases, StopRejectsNewLeaseAndRetainsGeneration) {
  auto& loader = cortex::CortexDriverLoader::instance();
  ASSERT_EQ(0, loader.load(ABI_STUB, "abi-contract"));
  cortex::CortexDriverLoader::Lease lease;
  ASSERT_EQ(0, loader.try_acquire_lease(lease));
  EXPECT_EQ(-EBUSY, loader.request_unload());
  EXPECT_TRUE(loader.is_loaded());
  cortex::CortexDriverLoader::Lease rejected;
  EXPECT_EQ(-ESHUTDOWN, loader.try_acquire_lease(rejected));
  EXPECT_EQ(-EBUSY, loader.load(ABI_STUB, "abi-contract"));
  loader.unload();
  EXPECT_TRUE(loader.is_loaded());
  EXPECT_EQ(73u, lease.api().get_last_cluster_id());
  lease = {};
  EXPECT_TRUE(loader.is_loaded()); // last release never performs shutdown
  EXPECT_EQ(0, loader.request_unload());
  EXPECT_FALSE(loader.is_loaded());
  ASSERT_EQ(0, loader.load(ABI_STUB, "abi-contract"));
  EXPECT_EQ(0, loader.try_acquire_lease(rejected)); // fresh admission/generation
  rejected = {};
  EXPECT_EQ(0, loader.request_unload());
}

TEST(LoaderLeases, CopiesMovesAndFinalReleaseOnAnotherThreadAreInert) {
  auto& loader = cortex::CortexDriverLoader::instance();
  ASSERT_EQ(0, loader.load(ABI_STUB, "abi-contract"));
  cortex::CortexDriverLoader::Lease original;
  ASSERT_EQ(0, loader.try_acquire_lease(original));
  EXPECT_EQ(-EINVAL, loader.try_acquire_lease(original));
  auto copy = original;
  auto moved = std::move(original);
  EXPECT_FALSE(bool(original));
  copy = copy; // self assignment must not prematurely release
  EXPECT_EQ(-EBUSY, loader.request_unload());
  moved = {};
  std::mutex mutex;
  std::condition_variable cv;
  bool released = false, finish = false;
  std::thread releaser([token = std::move(copy), &mutex, &cv, &released, &finish]() mutable {
    token = {}; // final release; thread remains alive until coordinator checks
    std::unique_lock<std::mutex> lock(mutex);
    released = true; cv.notify_all();
    cv.wait(lock, [&] { return finish; });
  });
  {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return released; });
    EXPECT_TRUE(loader.is_loaded());
    EXPECT_EQ(0, loader.request_unload());
    finish = true; cv.notify_all();
  }
  releaser.join();
  EXPECT_FALSE(loader.is_loaded());
}
