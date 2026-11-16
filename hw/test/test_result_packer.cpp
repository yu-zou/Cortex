// SPDX-License-Identifier: MIT
//
// Unit tests for 128-bit result word pack/unpack round-trip.
//
// Bit layout (matching FIFO_RES / compute_engine output):
//   [31:0]   = dist      (FP32)
//   [95:32]  = doc_addr  (uint64)
//   [127:96] = doc_len   (uint32)
//
// This test uses pure C++ types with __uint128_t for the 128-bit word.
// No HLS / ap_uint types are used in the test code itself.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

// ─── Result entry structure (local definition, no HLS dependency) ───
struct ResultEntry {
    float        dist;
    uint64_t     addr;
    uint32_t     len;
};

// ===========================================================================
// Pure C++ 128-bit result word helpers (no ap_uint dependency)
// ===========================================================================

/// Pack dist (FP32, [31:0]), addr (uint64, [95:32]), len (uint32, [127:96])
/// into a 128-bit result word.
static inline __uint128_t pack_result_128(float dist, uint64_t addr, uint32_t len) {
    __uint128_t word = 0;

    // [31:0]  — dist as raw FP32 bits
    uint32_t dist_bits;
    std::memcpy(&dist_bits, &dist, sizeof(dist_bits));
    word |= static_cast<__uint128_t>(dist_bits);

    // [95:32] — addr
    word |= static_cast<__uint128_t>(addr) << 32;

    // [127:96] — len
    word |= static_cast<__uint128_t>(len) << 96;

    return word;
}

/// Unpack a 128-bit result word into a ResultEntry struct.
static inline ResultEntry unpack_result_128_pure(__uint128_t word) {
    ResultEntry e;

    // [31:0]  → dist
    uint32_t dist_bits = static_cast<uint32_t>(word & UINT64_C(0xFFFFFFFF));
    std::memcpy(&e.dist, &dist_bits, sizeof(e.dist));

    // [95:32] → addr
    e.addr = static_cast<uint64_t>((word >> 32) & UINT64_C(0xFFFFFFFFFFFFFFFF));

    // [127:96] → len
    e.len = static_cast<uint32_t>((word >> 96) & UINT64_C(0xFFFFFFFF));

    return e;
}

// ===========================================================================
// Tests
// ===========================================================================

/// Nominal round-trip: arbitrary values in all three fields.
TEST(ResultPacker, RoundTrip) {
    const float    dist = 1.5f;
    const uint64_t addr = 0xABCDEF0123456789ULL;
    const uint32_t len  = 100000;

    const __uint128_t  packed = pack_result_128(dist, addr, len);
    const ResultEntry  e      = unpack_result_128_pure(packed);

    EXPECT_NEAR(dist, e.dist, 1e-6f);
    EXPECT_EQ(addr, e.addr);
    EXPECT_EQ(len,  e.len);
}

/// Edge case: dist near FP32 maximum (~3.4e38).
TEST(ResultPacker, Fp32Max) {
    const float    dist = 3.4e38f;
    const uint64_t addr = 0x42;
    const uint32_t len  = 1;

    const __uint128_t  packed = pack_result_128(dist, addr, len);
    const ResultEntry  e      = unpack_result_128_pure(packed);

    EXPECT_NEAR(dist, e.dist, 1e-6f);
    EXPECT_EQ(addr, e.addr);
    EXPECT_EQ(len,  e.len);
}

/// Edge case: all fields are zero.
TEST(ResultPacker, ZeroValues) {
    const __uint128_t  packed = pack_result_128(0.0f, 0, 0);
    const ResultEntry  e      = unpack_result_128_pure(packed);

    EXPECT_FLOAT_EQ(0.0f, e.dist);
    EXPECT_EQ(0ULL,       e.addr);
    EXPECT_EQ(0U,         e.len);
}

/// Edge case: maximum 64-bit address value (all ones).
TEST(ResultPacker, MaxAddr) {
    const float    dist = 42.0f;
    const uint64_t addr = 0xFFFFFFFFFFFFFFFFULL;
    const uint32_t len  = 999;

    const __uint128_t  packed = pack_result_128(dist, addr, len);
    const ResultEntry  e      = unpack_result_128_pure(packed);

    EXPECT_NEAR(dist, e.dist, 1e-6f);
    EXPECT_EQ(addr, e.addr);
    EXPECT_EQ(len,  e.len);
}
