// =============================================================================
// test_systolic_topk_sw.cpp — Standalone software verification of systolic
// Top-K insertion algorithm.  Mirrors the HLS logic in acc_top.cpp but uses
// plain-C++ types so it can run under a standard C++ toolchain with GTest.
//
// The systolic algorithm is an unrolled insertion-sort chain: a candidate
// propagates through cells 0..K-1; at each cell, if candidate < cell[i],
// the two are swapped and the displaced value continues down the chain.
// The end result is a sorted (ascending) array of the K smallest
// distances seen so far.
//
// Integer-bit-pattern comparison is used on the float distance, matching
// the HLS optimization for non-negative L2 distances.
// =============================================================================

#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <limits>
#include <random>

// ─── Standalone systolic cell ────────────────────────────────────────────────
struct CellSW {
    float    dist;
    uint64_t addr;
    uint32_t len;
};

// ─── Software reference of the HLS systolic_topk_insert() ──────────────────
// Mirror of acc_top.cpp lines 116-147: integer-bit-compare on the float,
// then swap-and-cascade through the full K-cell array.
static void systolic_topk_insert_sw(float cand_dist, uint64_t cand_addr,
                                     uint32_t cand_len, CellSW cells[],
                                     int K) {
    // Reinterpret the float as a uint32 so we can compare bit patterns.
    // This matches the HLS trick: for non-negative L2 distances the
    // IEEE 754 encoding preserves ordering under unsigned comparison.
    float       cd_float = cand_dist;
    uint32_t    cd_int;
    std::memcpy(&cd_int, &cand_dist, sizeof(cd_int));

    uint64_t ca = cand_addr;
    uint32_t cl = cand_len;

    for (int i = 0; i < K; i++) {
        uint32_t cell_int;
        std::memcpy(&cell_int, &cells[i].dist, sizeof(cell_int));
        if (cd_int < cell_int) {
            // Swap candidate into cell i, cascade the displaced value.
            float       td = cells[i].dist;
            uint64_t    ta = cells[i].addr;
            uint32_t    tl = cells[i].len;

            cells[i].dist = cd_float;
            cells[i].addr = ca;
            cells[i].len  = cl;

            cd_float = td;
            std::memcpy(&cd_int, &td, sizeof(cd_int));
            ca = ta;
            cl = tl;
        }
    }
}

// ─── Helper: initialise K cells to +inf (sentinel) ──────────────────────────
static void init_cells(CellSW cells[], int K) {
    const float inf = std::numeric_limits<float>::infinity();
    for (int i = 0; i < K; i++) {
        cells[i].dist = inf;
        cells[i].addr = 0;
        cells[i].len  = 0;
    }
}

// ─── Helper: verify sorted invariant ────────────────────────────────────────
static bool is_sorted(CellSW cells[], int K) {
    for (int i = 0; i + 1 < K; i++) {
        // Infinity sentinels at the tail are fine — stop at first inf.
        if (std::numeric_limits<float>::infinity() == cells[i].dist) break;
        if (cells[i].dist > cells[i + 1].dist) return false;
    }
    return true;
}

// ─── Helper: count how many cells hold a finite value ───────────────────────
static int count_valid(CellSW cells[], int K) {
    int n = 0;
    for (int i = 0; i < K; i++) {
        if (cells[i].dist < std::numeric_limits<float>::infinity()) n++;
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1 — BasicSort: insert K random values into K cells, verify sorted
// ═══════════════════════════════════════════════════════════════════════════
TEST(SystolicTopK, BasicSort) {
    const int K = 10;
    CellSW cells[K];
    init_cells(cells, K);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    // Insert exactly K values — should fill all cells.
    for (int i = 0; i < K; i++) {
        float d = dist(rng);
        systolic_topk_insert_sw(d, uint64_t(i), 1, cells, K);
    }

    EXPECT_EQ(count_valid(cells, K), K);
    EXPECT_TRUE(is_sorted(cells, K));
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2 — Overflow: insert 2×K values; confirm cells hold the K smallest
// ═══════════════════════════════════════════════════════════════════════════
TEST(SystolicTopK, Overflow) {
    const int K = 10;
    CellSW cells[K];
    init_cells(cells, K);

    // Generate 20 values, record the 10 smallest via std::nth_element.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    float all[20];
    for (int i = 0; i < 20; i++) {
        all[i] = dist(rng);
    }

    // Insert all 20 into the systolic array.
    for (int i = 0; i < 20; i++) {
        systolic_topk_insert_sw(all[i], uint64_t(i), 1, cells, K);
    }

    // Reference: smallest 10 via partial sort.
    float ref[20];
    std::copy(all, all + 20, ref);
    std::nth_element(ref, ref + K, ref + 20);
    std::sort(ref, ref + K);

    EXPECT_EQ(count_valid(cells, K), K);
    EXPECT_TRUE(is_sorted(cells, K));

    for (int i = 0; i < K; i++) {
        EXPECT_EQ(cells[i].dist, ref[i]) << " at position " << i;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3 — Identical: insert 5 identical values, verify all stored
// ═══════════════════════════════════════════════════════════════════════════
TEST(SystolicTopK, Identical) {
    const int K = 10;
    CellSW cells[K];
    init_cells(cells, K);

    // 5 identical distances (all same bit pattern).
    const float val = 42.5f;
    for (int i = 0; i < 5; i++) {
        systolic_topk_insert_sw(val, uint64_t(i), 1, cells, K);
    }

    // All 5 should be in the array (the compare is cd_int < cell_int,
    // so identical values do not trigger a swap — they fall through
    // to the end.  Cells[0..4] should still be inf, cells[5..9] should
    // hold the identical values … Actually let's verify by tracing.
    //
    // Because 42.5 < inf triggers a swap at cell 0 on first insert, so
    // cell 0 correctly holds 42.5 after first call.  On second call with
    // same value, cd_int == cell_int (both 42.5), so no swap — candidate
    // falls through to cell 1 where it compares against inf and swaps.
    // Result: entries are spread across the first N cells.

    int n = count_valid(cells, K);
    EXPECT_EQ(n, 5);  // all 5 made it in

    // Every valid cell should hold exactly 42.5
    for (int i = 0; i < n; i++) {
        EXPECT_EQ(cells[i].dist, val) << " at position " << i;
    }
    EXPECT_TRUE(is_sorted(cells, K));
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4 — Decreasing: insert strictly decreasing values, verify sorted
// ═══════════════════════════════════════════════════════════════════════════
TEST(SystolicTopK, Decreasing) {
    const int K = 10;
    CellSW cells[K];
    init_cells(cells, K);

    // Insert values 100, 99, 98, ..., 91 (decreasing order).
    for (int i = 0; i < K; i++) {
        float d = 100.0f - float(i);
        systolic_topk_insert_sw(d, uint64_t(i), 1, cells, K);
    }

    EXPECT_EQ(count_valid(cells, K), K);
    EXPECT_TRUE(is_sorted(cells, K));

    // Expected: 91, 92, 93, ..., 100 (increasing).
    for (int i = 0; i < K; i++) {
        float expected = 91.0f + float(i);
        EXPECT_EQ(cells[i].dist, expected) << " at position " << i;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5 — LargeDataset: 500 random values into K=500, verify sorted
// ═══════════════════════════════════════════════════════════════════════════
TEST(SystolicTopK, LargeDataset) {
    const int K = 500;
    CellSW cells[K];
    init_cells(cells, K);

    std::mt19937 rng(12345);
    // Mix of small and large values to exercise the compare logic.
    std::uniform_real_distribution<float> dist(0.0f, 1e6f);

    for (int i = 0; i < K; i++) {
        float d = dist(rng);
        systolic_topk_insert_sw(d, uint64_t(i), 1, cells, K);
    }

    // After inserting exactly K values, all cells should be valid.
    EXPECT_EQ(count_valid(cells, K), K);
    EXPECT_TRUE(is_sorted(cells, K));

    // Also verify that inserting more values (2×) maintains both
    // capacity and sortedness, and that the stored set is correct.
    for (int i = K; i < 2 * K; i++) {
        float d = dist(rng);
        systolic_topk_insert_sw(d, uint64_t(i), 1, cells, K);
    }

    EXPECT_EQ(count_valid(cells, K), K)
        << "Overflow should not increase cell count";

    EXPECT_TRUE(is_sorted(cells, K));

    // Build oracle: collect all 1000 values, pick smallest 500.
    // We need deterministic generator — re-seed with same seed.
    std::mt19937 rng_oracle(12345);
    std::uniform_real_distribution<float> dist_oracle(0.0f, 1e6f);

    float oracle[1000];
    for (int i = 0; i < 1000; i++) {
        oracle[i] = dist_oracle(rng_oracle);
    }
    std::nth_element(oracle, oracle + K, oracle + 1000);
    std::sort(oracle, oracle + K);

    for (int i = 0; i < K; i++) {
        EXPECT_EQ(cells[i].dist, oracle[i])
            << " at position " << i;
    }
}
