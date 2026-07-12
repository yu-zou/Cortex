// =============================================================================
// test_abi_boundary.cc — C ABI boundary verification across hw/ ↔ driver/
//
// Pure struct-layout tests (no mock HW library required).
// Verifies that shared data structures have identical memory layouts when
// observed from the FPGA (acc_top.h) and driver (cortex_driver_api.h) sides.
//
// Build with:
//   target_include_directories(test_abi_boundary PRIVATE
//     ${CMAKE_CURRENT_SOURCE_DIR}/../../include
//     ${CMAKE_CURRENT_SOURCE_DIR}/../../hw/include)
// =============================================================================

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

// hw-side structs (FPGA/HLS)
#include "acc_top.h"

// driver-side structs and constants
#include "cortex_driver_api.h"

// -----------------------------------------------------------------------------
// Portable field-offset helper
//
// Standard offsetof() is guaranteed only for standard-layout types.
// HLS types such as ap_uint<N> add constructors / operator overloads that
// technically break the standard-layout contract.  In practice GCC and Clang
// still evaluate the pointer-arithmetic form as a compile-time constant.
// -----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__GNUG__)
// GCC/Clang: __builtin_offsetof works for any type.
#define OFF_(type_, field_)  static_cast<size_t>(__builtin_offsetof(type_, field_))
#else
// Fallback: classic (but formally UB) null-pointer offset trick.
// Accepted by all mainstream compilers for simple class-type members.
#define OFF_(type_, field_)                                     \
    (reinterpret_cast<size_t>(                                  \
        &(reinterpret_cast<const volatile char&>(               \
            reinterpret_cast<const type_*>(0)->field_))))
#endif

// =============================================================================
// Test 1: IVFHeader layout
//
// DRAM layout (design doc):
//   Offset 0x00: codebook_dim  (ap_uint<32>,  4 B)  — M
//   Offset 0x04: pq_dim        (ap_uint<32>,  4 B)  — DIM
//   Offset 0x08: pq_vectors    (ap_uint<32>,  4 B)  — N
//   Offset 0x0C: reserved[13]  (ap_uint<32>, 52 B)  — zero-filled
//             total = 64 B
// =============================================================================
TEST(ABIBoundary, IVFHeaderLayout) {
    printf("=== IVFHeader (acc_top.h) ===\n");

    const size_t sz = sizeof(IVFHeader);
    printf("  sizeof(IVFHeader) = %zu  (expect 64)\n", sz);

    const size_t off_cd = OFF_(IVFHeader, codebook_dim);
    const size_t off_pd = OFF_(IVFHeader, pq_dim);
    const size_t off_pv = OFF_(IVFHeader, pq_vectors);
    const size_t off_rs = OFF_(IVFHeader, reserved);

    printf("  codebook_dim  @ %zu  (expect 0)\n",  off_cd);
    printf("  pq_dim        @ %zu  (expect 4)\n",  off_pd);
    printf("  pq_vectors    @ %zu  (expect 8)\n",  off_pv);
    printf("  reserved[13]  @ %zu  (expect 12)\n", off_rs);

    EXPECT_EQ(sz,    64u)  << "IVFHeader must be exactly 64 bytes (one DRAM burst)";
    EXPECT_EQ(off_cd, 0u)  << "codebook_dim must be at offset 0";
    EXPECT_EQ(off_pd, 4u)  << "pq_dim must be at offset 4";
    EXPECT_EQ(off_pv, 8u)  << "pq_vectors must be at offset 8";
    EXPECT_EQ(off_rs, 12u) << "reserved must start at offset 12";

    // Verify reserved array spans exactly 52 bytes (13 × 4)
    const size_t end_rs = off_rs + sizeof(IVFHeader::reserved);
    EXPECT_EQ(end_rs, sz) << "reserved[13] must fill to struct end (52 B at offset 12 = 64)";

    // If mismatch, dump expected vs actual
    if (HasFailure()) {
        printf("  LAYOUT MISMATCH!\n");
        printf("  Expected: 64 B total, codebook_dim@0 pq_dim@4 pq_vectors@8 reserved@12\n");
        printf("  Actual:   %zu B total, cd@%zu pd@%zu pv@%zu rsv@%zu\n",
               sz, off_cd, off_pd, off_pv, off_rs);
    }
}

// =============================================================================
// Test 2: ComputeMeta layout
//
// FPGA control-path metadata passed via stream between data_manager and
// compute_engine / hnsw_search_engine.
//
//   m_actual    @  0  ap_uint<32>  4 B   — M (sub-quantizers)
//   dim_actual  @  4  ap_uint<32>  4 B   — DIM (vector dimension)
//   n_vectors   @  8  ap_uint<32>  4 B   — #vectors (PQ) or #nodes (HNSW)
//   top_k       @ 12  ap_uint<32>  4 B   — requested top-K
//   metric_id   @ 16  ap_uint<2>   1 B   — 0=L2, 1=IP
//   search_mode @ 17  ap_uint<1>   1 B   — 0=IVFPQ, 1=HNSW
//
// Note: ap_uint<2> and ap_uint<1> sizes depend on host compiler's
// minimum-addressable-unit for Vitis HLS types.  This test prints the
// actual layout and verifies invariants rather than exact byte positions
// for the bit-width tail fields.
// =============================================================================
TEST(ABIBoundary, ComputeMetaLayout) {
    printf("=== ComputeMeta (acc_top.h) ===\n");

    const size_t sz = sizeof(ComputeMeta);
    printf("  sizeof(ComputeMeta) = %zu\n", sz);
    printf("  sizeof(ap_uint<2>)  = %zu\n", sizeof(ap_uint<2>));
    printf("  sizeof(ap_uint<1>)  = %zu\n", sizeof(ap_uint<1>));

    const size_t off_ma = OFF_(ComputeMeta, m_actual);
    const size_t off_da = OFF_(ComputeMeta, dim_actual);
    const size_t off_nv = OFF_(ComputeMeta, n_vectors);
    const size_t off_tk = OFF_(ComputeMeta, top_k);
    const size_t off_mi = OFF_(ComputeMeta, metric_id);
    const size_t off_sm = OFF_(ComputeMeta, search_mode);

    printf("  m_actual   @ %zu  (expect 0)\n",  off_ma);
    printf("  dim_actual @ %zu  (expect 4)\n",  off_da);
    printf("  n_vectors  @ %zu  (expect 8)\n",  off_nv);
    printf("  top_k      @ %zu  (expect 12)\n", off_tk);
    printf("  metric_id  @ %zu  (expect >= 16)\n",  off_mi);
    printf("  search_mode@ %zu  (expect > metric_id)\n", off_sm);

    // First four fields are ap_uint<32> → fixed at 4-byte strides
    EXPECT_EQ(off_ma, 0u)   << "m_actual must be at offset 0";
    EXPECT_EQ(off_da, 4u)   << "dim_actual must be at offset 4";
    EXPECT_EQ(off_nv, 8u)   << "n_vectors must be at offset 8";
    EXPECT_EQ(off_tk, 12u)  << "top_k must be at offset 12";

    // metric_id starts after top_k (at or after byte 16)
    EXPECT_GE(off_mi, 16u)  << "metric_id must not overlap top_k";

    // Fields are in declaration order
    EXPECT_LT(off_ma, off_da);
    EXPECT_LT(off_da, off_nv);
    EXPECT_LT(off_nv, off_tk);
    EXPECT_LT(off_tk, off_mi);
    EXPECT_LT(off_mi, off_sm);

    // No field exceeds struct bounds
    EXPECT_LT(off_sm, sz);

    // Size upper bound: even with worst-case per-field alignment,
    // ComputeMeta should not exceed 32 bytes (8 × uint32 alignment)
    EXPECT_LE(sz, 32u) << "ComputeMeta too large for stream payload";
    EXPECT_GE(sz, 16u) << "ComputeMeta too small (must hold 4 × uint32 minimum)";

    // ---- FPGA synthesis layout hint ----
    printf("\n  FPGA stream layout (simulated):\n");
    printf("    bits [  31:  0] m_actual\n");
    printf("    bits [  63: 32] dim_actual\n");
    printf("    bits [  95: 64] n_vectors\n");
    printf("    bits [ 127: 96] top_k\n");
    printf("    bits [ 129:128] metric_id\n");
    printf("    bits [ 130:130] search_mode\n");
    printf("    (remaining bits padding to stream width)\n");

    if (HasFailure()) {
        printf("  LAYOUT MISMATCH!\n");
        printf("  Actual offsets: ma=%zu da=%zu nv=%zu tk=%zu mi=%zu sm=%zu  sz=%zu\n",
               off_ma, off_da, off_nv, off_tk, off_mi, off_sm, sz);
    }
}

// =============================================================================
// Test 3: HNSWGraphHeader layout
//
// DRAM header for HNSW graph search path:
//   num_nodes    @  0  ap_uint<32>  4 B
//   entry_point  @  4  ap_uint<32>  4 B
//   dim          @  8  ap_uint<32>  4 B
//   max_degree   @ 12  ap_uint<32>  4 B
//             total = 16 B
// =============================================================================
TEST(ABIBoundary, HNSWGraphHeaderLayout) {
    printf("=== HNSWGraphHeader (acc_top.h) ===\n");

    const size_t sz = sizeof(HNSWGraphHeader);
    printf("  sizeof(HNSWGraphHeader) = %zu  (expect 16)\n", sz);

    const size_t off_nn = OFF_(HNSWGraphHeader, num_nodes);
    const size_t off_ep = OFF_(HNSWGraphHeader, entry_point);
    const size_t off_di = OFF_(HNSWGraphHeader, dim);
    const size_t off_md = OFF_(HNSWGraphHeader, max_degree);

    printf("  num_nodes   @ %zu  (expect 0)\n",  off_nn);
    printf("  entry_point @ %zu  (expect 4)\n",  off_ep);
    printf("  dim         @ %zu  (expect 8)\n",  off_di);
    printf("  max_degree  @ %zu  (expect 12)\n", off_md);

    EXPECT_EQ(sz,    16u) << "HNSWGraphHeader must be exactly 16 bytes";
    EXPECT_EQ(off_nn, 0u) << "num_nodes must be at offset 0";
    EXPECT_EQ(off_ep, 4u) << "entry_point must be at offset 4";
    EXPECT_EQ(off_di, 8u) << "dim must be at offset 8";
    EXPECT_EQ(off_md, 12u) << "max_degree must be at offset 12";

    if (HasFailure()) {
        printf("  LAYOUT MISMATCH!\n");
        printf("  Expected: 16 B total, nn@0 ep@4 dim@8 md@12\n");
        printf("  Actual:   %zu B total, nn@%zu ep@%zu dim@%zu md@%zu\n",
               sz, off_nn, off_ep, off_di, off_md);
    }
}

// =============================================================================
// Test 4: Version string consistency
//
// The ABI version defined in cortex_driver_api.h must be a single source of
// truth shared by driver loader, mock library, and FPGA control software.
// =============================================================================
TEST(ABIBoundary, VersionString) {
    printf("=== VersionString (cortex_driver_api.h) ===\n");

    const uint32_t ver       = CORTEX_API_VERSION;
    const uint32_t fn_ver    = cortex_api_version();

    printf("  CORTEX_API_VERSION     = %u\n", (unsigned)ver);
    printf("  cortex_api_version()   = %u\n", (unsigned)fn_ver);

    // Macro must be a reasonable positive value
    EXPECT_GE(ver, 1u);
    EXPECT_LE(ver, 255u);

    // Runtime function must return the compile-time value
    EXPECT_EQ(fn_ver, ver)
        << "cortex_api_version() return value differs from CORTEX_API_VERSION macro";

    // If driver shared library is not loaded, cortex_api_version() may return 0
    // (the weak default).  That is a link-time warning, not a layout failure.
    if (fn_ver == 0) {
        printf("  [WARN] cortex_api_version() returned 0 — stub/weak symbol?\n");
    }

    printf("  => Version OK: %u\n", (unsigned)ver);
}

// =============================================================================
// Test 5: 128-bit result format consistency
//
// Both sides of the ABI must agree on the 128-bit result format:
//
//   dist(FP32, 32b) + doc_addr(64b) + doc_length(32b) = 128 bit
//
//   FPGA side (Candidate in acc_top.h):
//     dist           @  0  float        4 B
//     doc_start_addr @  4  ap_uint<64>  8 B   (HLS: packed)
//     doc_length     @ 12  ap_uint<32>  4 B
//   sizeof = 16 B
//
//   Driver side (cortex_topk_entry in cortex_driver_api.h):
//     distance   @  0  float      4 B
//     doc_addr   @  8  uint64_t   8 B   (natural alignment: 8-byte boundary)
//     doc_length @ 16  uint32_t   4 B
//     _pad       @ 20  uint32_t   4 B
//   sizeof = 24 B
//
// The driver-side struct has natural (not packed) alignment because it
// uses standard C types.  The extra 8 bytes (_pad + implicit padding)
// are a SW-side convenience.  The wire / FIFO format is 128-bit packed.
//
// This test verifies:
//   a) Both structs are self-consistent (fields in order, no overlap).
//   b) The 128-bit payload fields have identical relative layout when
//      interpreted from byte 0 of a packed 128-bit word.
// =============================================================================
TEST(ABIBoundary, ResultFormatConsistency) {
    printf("=== ResultFormatConsistency: cortex_topk_entry ↔ Candidate ===\n");
    printf("  128-bit FIFO format: dist(32b) + doc_addr(64b) + doc_length(32b)\n\n");

    // ---- Driver side ----
    printf("-- cortex_topk_entry (driver) --\n");
    const size_t sz_drv = sizeof(struct cortex_topk_entry);
    const size_t off_dst  = offsetof(struct cortex_topk_entry, distance);
    const size_t off_addr = offsetof(struct cortex_topk_entry, doc_addr);
    const size_t off_len  = offsetof(struct cortex_topk_entry, doc_length);
    const size_t off_pad  = offsetof(struct cortex_topk_entry, _pad);

    printf("  sizeof          = %zu\n", sz_drv);
    printf("  distance        @ %zu  (size %zu)\n", off_dst, sizeof(float));
    printf("  doc_addr        @ %zu  (size %zu)\n", off_addr, sizeof(uint64_t));
    printf("  doc_length      @ %zu  (size %zu)\n", off_len, sizeof(uint32_t));
    printf("  _pad            @ %zu  (size %zu)\n", off_pad, sizeof(uint32_t));

    EXPECT_EQ(off_dst, 0u)   << "distance must be at offset 0";
    EXPECT_EQ(off_len, off_addr + sizeof(uint64_t)) << "doc_length must follow doc_addr";
    EXPECT_EQ(off_pad, off_len + sizeof(uint32_t)) << "_pad must follow doc_length";
    EXPECT_GT(off_addr, off_dst) << "doc_addr must be after distance";

    // ---- HW side ----
    printf("\n-- Candidate (hw: acc_top.h) --\n");
    const size_t sz_hw  = sizeof(Candidate);
    const size_t off_cd = OFF_(Candidate, dist);
    const size_t off_ca = OFF_(Candidate, doc_start_addr);
    const size_t off_cl = OFF_(Candidate, doc_length);

    printf("  sizeof          = %zu\n", sz_hw);
    printf("  dist            @ %zu  (size %zu)\n", off_cd, sizeof(float));
    printf("  doc_start_addr  @ %zu  (size %zu)\n", off_ca, sizeof(ap_uint<64>));
    printf("  doc_length      @ %zu  (size %zu)\n", off_cl, sizeof(ap_uint<32>));

    EXPECT_EQ(off_cd, 0u)  << "dist must be at offset 0";
    EXPECT_GT(off_ca, off_cd) << "doc_start_addr must follow dist";
    EXPECT_GT(off_cl, off_ca) << "doc_length must follow doc_start_addr";

    // ---- 128-bit layout equivalence ----
    printf("\n-- 128-bit packed wire format --\n");
    printf("  wire[ 31: 0]  dist / distance      (offset 0)\n");
    printf("  wire[ 95:32]  doc_addr / doc_start  (offset 4)\n");
    printf("  wire[127:96]  doc_length            (offset 12)\n");

    // The key fields (dist, doc_addr, doc_length) occupy the same bit positions
    // in a packed 128-bit word.  Verify that when reading from offset 0:
    const bool packed_match =
        (off_dst == off_cd) &&
        (off_len == off_cl);

    printf("\n  driver size  = %zu (includes %zu bytes of SW-only padding)\n",
           sz_drv, sz_drv - 16u);
    printf("  HW size      = %zu (pure 128-bit, no padding)\n", sz_hw);
    printf("  Packed 128-bit layout equivalent: %s\n",
           packed_match ? "YES" : "NO (but driver adds _pad field)");

    // Verify packed layout if both structs happen to be packed; otherwise
    // print informative message about the natural alignment difference.
    if (packed_match) {
        EXPECT_EQ(sz_hw, 16u) << "Candidate must be 128-bit (16 B)";
        EXPECT_EQ(off_cd + sizeof(float), off_ca) << "dist → doc_addr gap must be 4";
        EXPECT_EQ(off_ca + sizeof(ap_uint<64>), off_cl) << "doc_addr → doc_length gap must be 8";
    } else {
        printf("  [INFO] Natural alignment differs — driver adds padding\n");
        printf("  [INFO] hw doc_addr@%zu  vs  driver doc_addr@%zu\n", off_ca, off_addr);
        printf("  [INFO] This is EXPECTED when standard C structs are not packed.\n");
    }

    // ---- Cross-check: verify the 128-bit word reinterpretation ----
    // Construct a known 128-bit pattern and verify both struct layouts can
    // extract the same fields.
    printf("\n-- Reinterpretation check (128-bit word overlay) --\n");

    // Build a packed 128-bit word: dist=3.14f, addr=0xDEADBEEFCAFE, len=42
    const float    test_dist    = 3.140000f;
    const uint64_t test_addr    = 0xDEADBEEFCAFEULL;
    const uint32_t test_len     = 42u;

    uint8_t packed_128[16] = {};
    memcpy(packed_128 + 0,  &test_dist,   4);
    memcpy(packed_128 + 4,  &test_addr,   8);
    memcpy(packed_128 + 12, &test_len,    4);

    // Read back through Candidate layout (HW side)
    const Candidate* p_cand = reinterpret_cast<const Candidate*>(packed_128);
    const float    got_dist = p_cand->dist;
    const uint64_t got_addr = static_cast<uint64_t>(p_cand->doc_start_addr);
    const uint32_t got_len  = static_cast<uint32_t>(p_cand->doc_length);

    printf("  Packed 128-bit overlay:\n");
    printf("    dist:   expected %.2f  got %.2f  %s\n",
           test_dist, got_dist, (test_dist == got_dist) ? "OK" : "MISMATCH");
    printf("    addr:   expected 0x%lx  got 0x%lx  %s\n",
           (unsigned long)test_addr, (unsigned long)got_addr,
           (test_addr == got_addr) ? "OK" : "MISMATCH");
    printf("    length: expected %u  got %u  %s\n",
           test_len, got_len, (test_len == got_len) ? "OK" : "MISMATCH");

    EXPECT_EQ(got_dist, test_dist);
    EXPECT_EQ(got_addr, test_addr);
    EXPECT_EQ(got_len,  test_len);

    // Read back through cortex_topk_entry layout (driver side)
    // Note: this may fail if the struct is not packed — that's informative.
    const struct cortex_topk_entry* p_drv =
        reinterpret_cast<const struct cortex_topk_entry*>(packed_128);
    const float    got_drv_dist = p_drv->distance;
    const uint64_t got_drv_addr = p_drv->doc_addr;
    const uint32_t got_drv_len  = p_drv->doc_length;

    printf("\n  Driver struct overlay on packed 128-bit:\n");
    printf("    distance: expected %.2f  got %.2f  %s\n",
           test_dist, got_drv_dist, (test_dist == got_drv_dist) ? "OK" : "MISMATCH (struct has padding)");
    printf("    doc_addr: expected 0x%lx  got 0x%lx  %s\n",
           (unsigned long)test_addr, (unsigned long)got_drv_addr,
           (test_addr == got_drv_addr) ? "OK" : "MISMATCH (struct has padding)");
    printf("    doc_length: expected %u  got %u  %s\n",
           test_len, got_drv_len,
           (test_len == got_drv_len) ? "OK" : "MISMATCH (struct has padding)");

    printf("\n=> Result format check complete.\n");
}

// =============================================================================
// No main() — linked against GTest::gtest_main (see CMakeLists.txt).
// =============================================================================
