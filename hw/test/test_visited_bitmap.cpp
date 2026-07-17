// ═══════════════════════════════════════════════════════════════
// test_visited_bitmap.cpp — Unit tests for visited bitmap
// operations used in HNSW graph traversal.
//
// The production HNSW engine (acc_top.cpp) uses:
//   ap_uint<1> visited[HNSW_MAX_NODES];
// with init, set, and check via array element assignment.
//
// These tests simulate the same logic using a standard bool array
// so they compile in a pure C++ environment without HLS headers.
// ═══════════════════════════════════════════════════════════════

#include <gtest/gtest.h>

// ─── Constants matching the HLS engine ───
static constexpr int HNSW_MAX_NODES = 200;

// ─── Helper: mark a node as visited ───
static inline void set_visited(bool* bitmap, int idx) {
    bitmap[idx] = true;
}

// ─── Helper: check if a node was visited ───
static inline bool is_visited(const bool* bitmap, int idx) {
    return bitmap[idx];
}

// ─── Helper: clear the entire bitmap ───
static inline void clear_all(bool* bitmap, int size) {
    for (int i = 0; i < size; i++) {
        bitmap[i] = false;
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: SetAndCheck
//   Set index 5, then verify:
//     - index 5  == visited (true)
//     - index 4  == not visited (false)
//     - index 6  == not visited (false)
// ═══════════════════════════════════════════════════════════════
TEST(VisitedBitmap, SetAndCheck) {
    bool visited[HNSW_MAX_NODES] = {false};

    // Set index 5 (equivalent to visited[5] = 1 in HLS)
    set_visited(visited, 5);

    // Check the set bit
    EXPECT_TRUE(is_visited(visited, 5));

    // Neighbors must remain unset
    EXPECT_FALSE(is_visited(visited, 4));
    EXPECT_FALSE(is_visited(visited, 6));
}

// ═══════════════════════════════════════════════════════════════
// Test 2: InitAllZero
//   Initialize a 200-entry bitmap and verify every entry is false.
//   This mirrors the VISIT_INIT loop in acc_top.cpp:
//     for (ap_uint<32> i = 0; i < HNSW_MAX_NODES; i++)
//       visited[i] = 0;
// ═══════════════════════════════════════════════════════════════
TEST(VisitedBitmap, InitAllZero) {
    bool visited[HNSW_MAX_NODES] = {false};

    // Brace-initialization already zeros the array; verify all 200
    for (int i = 0; i < HNSW_MAX_NODES; i++) {
        EXPECT_FALSE(is_visited(visited, i))
            << "Index " << i << " should be 0 after init";
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 3: BoundaryAccess
//   Set the first (0) and last (HNSW_MAX_NODES - 1) entries.
//   Verify both are set.  The HNSW engine visits nodes by index
//   and must correctly handle the extreme ends of the bitmap.
// ═══════════════════════════════════════════════════════════════
TEST(VisitedBitmap, BoundaryAccess) {
    bool visited[HNSW_MAX_NODES] = {false};

    // Set first and last entries
    set_visited(visited, 0);
    set_visited(visited, HNSW_MAX_NODES - 1);  // 199

    // Verify both are set
    EXPECT_TRUE(is_visited(visited, 0));
    EXPECT_TRUE(is_visited(visited, HNSW_MAX_NODES - 1));

    // Verify a middle entry is not affected
    EXPECT_FALSE(is_visited(visited, 100));
}

// ═══════════════════════════════════════════════════════════════
// Test 4: ClearAll
//   Set several entries, then clear the entire bitmap.
//   Verify all entries are false after clear.
//   This simulates re-initializing the bitmap between queries.
// ═══════════════════════════════════════════════════════════════
TEST(VisitedBitmap, ClearAll) {
    bool visited[HNSW_MAX_NODES] = {false};

    // Set several entries across the range
    set_visited(visited, 10);
    set_visited(visited, 50);
    set_visited(visited, 150);
    set_visited(visited, 198);

    // Confirm they are set
    EXPECT_TRUE(is_visited(visited, 10));
    EXPECT_TRUE(is_visited(visited, 50));
    EXPECT_TRUE(is_visited(visited, 150));
    EXPECT_TRUE(is_visited(visited, 198));

    // Clear all (equivalent to the VISIT_INIT loop)
    clear_all(visited, HNSW_MAX_NODES);

    // Verify all entries are now false
    for (int i = 0; i < HNSW_MAX_NODES; i++) {
        EXPECT_FALSE(is_visited(visited, i))
            << "Index " << i << " should be 0 after clear_all";
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 5: LargeBitmap
//   Test with 1000 entries — larger than the default HNSW_MAX_NODES
//   to stress the set/check logic with an alternating pattern.
//   Set even indices, skip odd indices, then verify the pattern.
// ═══════════════════════════════════════════════════════════════
TEST(VisitedBitmap, LargeBitmap) {
    static constexpr int LARGE_SIZE = 1000;
    bool visited[LARGE_SIZE] = {false};

    // Set all even indices
    for (int i = 0; i < LARGE_SIZE; i += 2) {
        set_visited(visited, i);
    }

    // Verify pattern: evens = true, odds = false
    for (int i = 0; i < LARGE_SIZE; i++) {
        if (i % 2 == 0) {
            EXPECT_TRUE(is_visited(visited, i))
                << "Even index " << i << " should be set";
        } else {
            EXPECT_FALSE(is_visited(visited, i))
                << "Odd index " << i << " should NOT be set";
        }
    }

    // Clear and verify all zero again
    clear_all(visited, LARGE_SIZE);
    for (int i = 0; i < LARGE_SIZE; i++) {
        EXPECT_FALSE(is_visited(visited, i))
            << "Index " << i << " should be 0 after clear in large bitmap";
    }
}
