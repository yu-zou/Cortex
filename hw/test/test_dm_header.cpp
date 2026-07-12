// SPDX-License-Identifier: MIT
//
// Unit tests for IVFHeader parsing from raw DRAM bytes.
//
// IVFHeader is a 64-byte structure stored little-endian in DRAM:
//   offset 0x00: codebook_dim (uint32_t) — M (sub-quantizers)
//   offset 0x04: pq_dim       (uint32_t) — DIM (original vector dimension)
//   offset 0x08: pq_vectors   (uint32_t) — N (number of PQ-encoded vectors)
//   offset 0x0C: reserved[13] (52 bytes, zero-filled)
//
// This test does NOT depend on Vitis HLS headers (ap_int.h).  It reads raw
// bytes from a simulated DRAM buffer using little-endian parsing, matching
// the data_manager logic in acc_top.

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>

// ---------------------------------------------------------------------------
// Plain C++ struct matching the DRAM layout of IVFHeader.
// No Vitis HLS dependency — uses standard uint32_t.
// ---------------------------------------------------------------------------
struct PlainIVFHeader {
    uint32_t codebook_dim;   // M
    uint32_t pq_dim;         // DIM
    uint32_t pq_vectors;     // N
    uint32_t reserved[13];   // 52 bytes
};
static_assert(sizeof(PlainIVFHeader) == 64,
              "IVFHeader layout must be exactly 64 bytes");

// Size of the header in DRAM.
static constexpr size_t kHeaderSize = 64;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse the three mandatory fields from a 64-byte DRAM buffer (little-endian).
static PlainIVFHeader parse_header(const uint8_t *data) {
    PlainIVFHeader h{};
    h.codebook_dim = static_cast<uint32_t>(data[0])
                   | (static_cast<uint32_t>(data[1]) << 8)
                   | (static_cast<uint32_t>(data[2]) << 16)
                   | (static_cast<uint32_t>(data[3]) << 24);
    h.pq_dim       = static_cast<uint32_t>(data[4])
                   | (static_cast<uint32_t>(data[5]) << 8)
                   | (static_cast<uint32_t>(data[6]) << 16)
                   | (static_cast<uint32_t>(data[7]) << 24);
    h.pq_vectors   = static_cast<uint32_t>(data[8])
                   | (static_cast<uint32_t>(data[9]) << 8)
                   | (static_cast<uint32_t>(data[10]) << 16)
                   | (static_cast<uint32_t>(data[11]) << 24);
    return h;
}

/// Write a uint32_t value into @p buf at @p offset (little-endian).
static void write_u32le(uint8_t *buf, size_t offset, uint32_t val) {
    buf[offset + 0] =  val        & 0xFF;
    buf[offset + 1] = (val >> 8)  & 0xFF;
    buf[offset + 2] = (val >> 16) & 0xFF;
    buf[offset + 3] = (val >> 24) & 0xFF;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Nominal case: M=16, DIM=128, N=10000.
TEST(DMHeader, ValidHeader) {
    uint8_t dram[kHeaderSize] = {};
    write_u32le(dram, 0, 16);      // codebook_dim
    write_u32le(dram, 4, 128);     // pq_dim
    write_u32le(dram, 8, 10000);   // pq_vectors

    const auto h = parse_header(dram);
    EXPECT_EQ(16,    h.codebook_dim);
    EXPECT_EQ(128,   h.pq_dim);
    EXPECT_EQ(10000, h.pq_vectors);
}

/// Boundary case: minimum legal values M=1, DIM=1, N=1.
TEST(DMHeader, MinValues) {
    uint8_t dram[kHeaderSize] = {};
    write_u32le(dram, 0, 1);   // codebook_dim
    write_u32le(dram, 4, 1);   // pq_dim
    write_u32le(dram, 8, 1);   // pq_vectors

    const auto h = parse_header(dram);
    EXPECT_EQ(1, h.codebook_dim);
    EXPECT_EQ(1, h.pq_dim);
    EXPECT_EQ(1, h.pq_vectors);
}

/// Edge case: N=0 (empty cluster).
TEST(DMHeader, ZeroVectors) {
    uint8_t dram[kHeaderSize] = {};
    write_u32le(dram, 0, 16);
    write_u32le(dram, 4, 128);
    write_u32le(dram, 8, 0);   // pq_vectors = 0

    const auto h = parse_header(dram);
    EXPECT_EQ(0, h.pq_vectors);
}

/// The reserved[13] region must not influence the three mandatory fields.
TEST(DMHeader, ReservedIgnored) {
    uint8_t dram[kHeaderSize] = {};
    write_u32le(dram, 0, 16);
    write_u32le(dram, 4, 128);
    write_u32le(dram, 8, 10000);

    // Fill reserved region with a known non-zero pattern.
    for (int i = 0; i < 13; ++i)
        write_u32le(dram, 12 + i * 4, 0xDEADBEEF + static_cast<uint32_t>(i));

    const auto h = parse_header(dram);
    EXPECT_EQ(16,    h.codebook_dim);
    EXPECT_EQ(128,   h.pq_dim);
    EXPECT_EQ(10000, h.pq_vectors);
}

/// Structural check: the header must occupy exactly 64 bytes in DRAM.
TEST(DMHeader, ByteAlignment) {
    EXPECT_EQ(64, sizeof(PlainIVFHeader));
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
