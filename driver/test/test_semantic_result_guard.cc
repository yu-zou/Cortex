#include "cortex/semantic_result_guard.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace {
struct Result {
  int status = 0;
  std::vector<int> entries;
};

TEST(SemanticResultGuard, WireVersionPolicyRejectsVectorIDSchema) {
  EXPECT_EQ(cortex::validate_semantic_wire_version(0), -EPROTO);
  EXPECT_EQ(cortex::validate_semantic_wire_version(1), -EPROTO);
  EXPECT_EQ(cortex::validate_semantic_wire_version(2), 0);
  // Future compatibility is still checked by Ceph DECODE_START itself.
  EXPECT_EQ(cortex::validate_semantic_wire_version(3), 0);
}

TEST(SemanticResultGuard, DriverStatusAndArrayBoundsBeforeAccess) {
  cortex_read_result result{};
  EXPECT_EQ(cortex::validate_driver_result(result), 0);
  result.count = 500;
  EXPECT_EQ(cortex::validate_driver_result(result), 0);
  result.count = 501;
  EXPECT_EQ(cortex::validate_driver_result(result), -EOVERFLOW);
  result.count = UINT32_MAX;
  EXPECT_EQ(cortex::validate_driver_result(result), -EOVERFLOW);
  result.count = 0;
  result.status = -ENOTSUP;
  EXPECT_EQ(cortex::validate_driver_result(result), -ENOTSUP);
  result.status = 7;
  EXPECT_EQ(cortex::validate_driver_result(result), -EIO);
}

TEST(SemanticResultGuard, EverySelectedClusterMustSucceed) {
  cortex::SemanticQueryGuard guard(2);
  int accepted = 0;
  auto decode = [] { return Result{0, {1, 2}}; };
  auto accept = [&](const Result&) { ++accepted; };
  EXPECT_EQ(guard.status(), -EIO);
  guard.consume(0, 0, decode, accept);
  EXPECT_EQ(guard.status(), -EIO);
  guard.consume(0, 0, decode, accept);
  EXPECT_EQ(guard.status(), 0);
  EXPECT_EQ(accepted, 2);
}

TEST(SemanticResultGuard, PartialAndAllFailedAreWholeQueryErrors) {
  for (bool partial : {false, true}) {
    for (int failure : {0, 1, 2, 3, 4, 5}) {
      cortex::SemanticQueryGuard guard(2);
      int accepted = 0;
      auto accept = [&](const Result&) { ++accepted; };
      if (partial) guard.consume(0, 0, [] { return Result{}; }, accept);
      else guard.consume(-EIO, 0, [] { return Result{}; }, accept);
      switch (failure) {
        case 0: guard.fail(-ENOENT); break; // submission failed, no completion
        case 1: guard.consume(-EIO, 0, [] { return Result{}; }, accept); break;
        case 2: guard.consume(0, -EINVAL, [] { return Result{}; }, accept); break;
        case 3: guard.consume(0, 0, [] { return Result{-ENOTSUP, {}}; }, accept); break;
        case 4: guard.consume(0, 0, []() -> Result {
          throw std::runtime_error("incompatible wire");
        }, accept); break;
        case 5: guard.consume(0, 0, [] {
          return Result{0, std::vector<int>(501)};
        }, accept); break;
      }
      EXPECT_LT(guard.status(), 0) << partial << ":" << failure;
      EXPECT_EQ(accepted, partial ? 1 : 0);
    }
  }
}

TEST(SemanticResultGuard, ErrorPathsDoNotDecodeOrAcceptAndRemainFailed) {
  cortex::SemanticQueryGuard guard(2);
  int decoded = 0;
  int accepted = 0;
  auto decode = [&] { ++decoded; return Result{}; };
  auto accept = [&](const Result&) { ++accepted; };
  guard.consume(-EIO, 0, decode, accept);
  guard.consume(0, 0, decode, accept);
  EXPECT_EQ(guard.status(), -EIO);
  EXPECT_EQ(decoded, 1);
  EXPECT_EQ(accepted, 1);
}
} // namespace
