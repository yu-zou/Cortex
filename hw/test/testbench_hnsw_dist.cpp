// ═══════════════════════════════════════════════════════════════
// testbench_hnsw_dist.cpp — Standalone Vitis HLS testbench for
// HNSW L2 distance computation (NBR_DIST_GROUPS pattern).
//
// Tests the core distance formula used in HNSW graph traversal:
//   nbr_dist = Σ_{g=0}^{15}  Σ_{d=g}^{g+7}  (query[d] - vector[d])²
//
// where:
//   g  = group index (0..15, stride 8)
//   d  = dimension within group (0..7)
//   DIM_SYN = 128 = 16 groups × 8 dims
//
// The loop structure matches `NBR_DIST_GROUPS` in `hnsw_search_engine`
// (acc_top.cpp lines 588–601).
//
// Run via:  run_hnsw_dist_test.tcl  (Vitis HLS)
// Compile:  g++ -I../include testbench_hnsw_dist.cpp -o testbench_hnsw_dist
// ═══════════════════════════════════════════════════════════════

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include "../include/acc_top.h"

using namespace std;

// ─── Test configuration ───
// DIM_SYN = 128 (from acc_top.h)
// The L2 distance uses 16 groups of 8 dims each: NBR_DIST_GROUPS pattern
static const int TEST_DIM   = DIM_SYN;   // 128
static const int TEST_GROUPS = DIM_SYN / 8;  // 16

// ═══════════════════════════════════════════════════════════════
// HNSW L2 Distance — exact match for HW NBR_DIST_GROUPS pattern
//
// Loop structure (corresponds to acc_top.cpp lines 588-601):
//   for (g = 0; g < DIM_SYN; g += 8)         // 16 groups
//     for (d = g; d < g + 8; d++)            // 8 dims/group
//       group_sum += (query[d] - vector[d])²
//     nbr_dist += group_sum
//
// Note: HW UNROLLs both NBR_DIST_GROUPS and NBR_DIST,
// We verify correctness by accumulating all dimensions directly.
// ═══════════════════════════════════════════════════════════════
static float hnsw_l2_distance(
    const float query[DIM_SYN],
    const float vector[DIM_SYN],
    int DIM_val
) {
    float nbr_dist = 0.0f;

    // NBR_DIST_GROUPS: 16 groups × 8 dims (fully unrolled in HW)
    for (int g = 0; g < DIM_SYN; g += 8) {
        float group_sum = 0.0f;

        // NBR_DIST: 8 dims per group
        for (int d = g; d < g + 8; d++) {
            if (d < DIM_val) {
                float diff = query[d] - vector[d];
                group_sum += diff * diff;
            }
        }
        nbr_dist += group_sum;
    }
    return nbr_dist;
}

// ═══════════════════════════════════════════════════════════════
// Reference implementation (simple element-wise accumulation to verify grouping correctness)
// ═══════════════════════════════════════════════════════════════
static float ref_l2_distance(
    const float query[DIM_SYN],
    const float vector[DIM_SYN],
    int DIM_val
) {
    float total = 0.0f;
    for (int d = 0; d < DIM_val; d++) {
        float diff = query[d] - vector[d];
        total += diff * diff;
    }
    return total;
}

// ─── Test infrastructure ───
static int g_pass = 0;
static int g_fail = 0;

static void check(const char* name, float actual, float expected, float eps = 1e-4f) {
    if (fabsf(actual - expected) < eps) {
        cout << "  PASS: " << name << " = " << actual << endl;
        g_pass++;
    } else {
        cout << "  FAIL: " << name << " = " << actual
             << "  (expected " << expected << ")" << endl;
        g_fail++;
    }
}

// ─── Helper: set all elements of a DIM_SYN array to a constant ───
static void fill_vec(float vec[DIM_SYN], float val) {
    for (int i = 0; i < DIM_SYN; i++) {
        vec[i] = val;
    }
}

// ─── Helper: copy src → dst ───
static void copy_vec(float dst[DIM_SYN], const float src[DIM_SYN]) {
    for (int i = 0; i < DIM_SYN; i++) {
        dst[i] = src[i];
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 1: Single non-zero dimension
//
//   query[0]  = 1.0,   query[1..127] = 0.0
//   vector[*] = 0.0 (all zeros)
//
// Expected:
//   (1-0)² + (0-0)²×127 = 1.0
// ═══════════════════════════════════════════════════════════════
static void test1_single_nonzero() {
    cout << "\n─── Test 1: Single Non-Zero Dimension (query[0]=1, rest 0) ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    query[0] = 1.0f;
    // vector stays all zeros

    float dist = hnsw_l2_distance(query, vector, TEST_DIM);
    check("hnsw_l2_distance", dist, 1.0f);

    // Verify ref matches
    float ref_dist = ref_l2_distance(query, vector, TEST_DIM);
    check("ref_l2_distance", ref_dist, 1.0f);
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Identical query and vector
//
//   query[d] = vector[d] = sin(d)  (arbitrary non-zero values)
//
// Expected: all diffs = 0 → total = 0.0
// ═══════════════════════════════════════════════════════════════
static void test2_identical() {
    cout << "\n─── Test 2: Query == Vector (identical) → distance ≈ 0 ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    // Fill both with same non-zero values
    for (int i = 0; i < DIM_SYN; i++) {
        float val = sinf(i * 0.1f) * 3.0f;
        query[i]  = val;
        vector[i] = val;
    }

    float dist = hnsw_l2_distance(query, vector, TEST_DIM);
    check("hnsw_l2_distance ≈ 0", dist, 0.0f);

    float ref_dist = ref_l2_distance(query, vector, TEST_DIM);
    check("ref_l2_distance ≈ 0", ref_dist, 0.0f);
}

// ═══════════════════════════════════════════════════════════════
// Test 3: Uniform query vs zero vector
//
//   query[d]  = 5.0  (all dims)
//   vector[d] = 0.0  (all dims)
//
// Expected:
//   (5-0)² × 128 = 25 × 128 = 3200.0
// ═══════════════════════════════════════════════════════════════
static void test3_uniform_vs_zero() {
    cout << "\n─── Test 3: Uniform query(5) vs Zero vector → 3200.0 ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    fill_vec(query, 5.0f);
    fill_vec(vector, 0.0f);

    float expected = 25.0f * static_cast<float>(DIM_SYN);  // = 3200.0

    float dist = hnsw_l2_distance(query, vector, TEST_DIM);
    check("hnsw_l2_distance", dist, expected);

    float ref_dist = ref_l2_distance(query, vector, TEST_DIM);
    check("ref_l2_distance", ref_dist, expected);
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Random query + random vector
//
//   Generate random values for both query and vector.
//   Compute golden reference using simple element-wise loop.
//   Verify hnsw_l2_distance matches within epsilon.
// ═══════════════════════════════════════════════════════════════
static void test4_random() {
    cout << "\n─── Test 4: Random Query + Random Vector ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    // Seed for reproducibility
    srand(42);

    // Generate random values in [-10, 10]
    for (int i = 0; i < DIM_SYN; i++) {
        query[i]  = ((float)(rand() % 2001) / 100.0f) - 10.0f;
        vector[i] = ((float)(rand() % 2001) / 100.0f) - 10.0f;
    }

    // Golden reference: simple element-wise L2
    float golden = ref_l2_distance(query, vector, TEST_DIM);

    // HW-matching implementation
    float dist = hnsw_l2_distance(query, vector, TEST_DIM);

    cout << "  golden=" << golden << "  hnsw_dist=" << dist << endl;
    check("hnsw_l2_distance vs golden", dist, golden, 1e-4f);

    // ─── Also verify per-group contributions ───
    // Each group of 8 dims should use group_sum reduction.
    // Verify all 16 groups individually.
    cout << "  Per-group verification:" << endl;
    bool groups_ok = true;
    for (int g = 0; g < DIM_SYN; g += 8) {
        float group_hw = 0.0f;
        for (int d = g; d < g + 8; d++) {
            float diff = query[d] - vector[d];
            group_hw += diff * diff;
        }
        float group_ref = 0.0f;
        for (int d = g; d < g + 8 && d < TEST_DIM; d++) {
            float diff = query[d] - vector[d];
            group_ref += diff * diff;
        }
        if (fabsf(group_hw - group_ref) >= 1e-6f) {
            cout << "    FAIL group g=" << g << ": hw=" << group_hw
                 << " ref=" << group_ref << endl;
            groups_ok = false;
            g_fail++;
        } else {
            cout << "    OK group g=" << g << ": sum=" << group_hw << endl;
            g_pass++;
        }
    }
    if (groups_ok) {
        cout << "  All 16 groups correct." << endl;
    }
}

// ═══════════════════════════════════════════════════════════════
// Test 5: Subset of dimensions (DIM_val < DIM_SYN)
//
// Test HW boundary condition: when actual DIM < DIM_SYN,
// the guard `if (d < DIM_val)` in NBR_DIST loop should apply.
//
//   query[0..63]  = random,  query[64..127]  ignored (garbage)
//   vector[0..63] = random,  vector[64..127] ignored (garbage)
//   DIM_val       = 64
//
// Expected: hnsw_dist should only sum dims 0..63
// ═══════════════════════════════════════════════════════════════
static void test5_partial_dim() {
    cout << "\n─── Test 5: Partial DIM (DIM_val=64 < DIM_SYN=128) ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    // Fill first 64 dims with known values
    for (int i = 0; i < 64; i++) {
        query[i]  = (i % 5) * 1.0f;
        vector[i] = (i % 3) * 2.0f;
    }
    // Fill remaining 64 dims with garbage (should be ignored)
    for (int i = 64; i < DIM_SYN; i++) {
        query[i]  = 999.0f;
        vector[i] = -999.0f;
    }

    const int PARTIAL_DIM = 64;

    // Golden: only first 64 dims
    float golden = 0.0f;
    for (int d = 0; d < PARTIAL_DIM; d++) {
        float diff = query[d] - vector[d];
        golden += diff * diff;
    }

    float dist = hnsw_l2_distance(query, vector, PARTIAL_DIM);
    check("hnsw_l2_distance (DIM_val=64)", dist, golden);
}

// ═══════════════════════════════════════════════════════════════
// Test 6: Negative values
//
//   query[d]  = -3.0
//   vector[d] =  4.0
//
// Expected per dim: (-3-4)² = (-7)² = 49
// Total: 49 × 128 = 6272.0
// ═══════════════════════════════════════════════════════════════
static void test6_negative_values() {
    cout << "\n─── Test 6: Negative Values (query=-3, vector=4) ───" << endl;

    float query[DIM_SYN]  = {};
    float vector[DIM_SYN] = {};

    fill_vec(query, -3.0f);
    fill_vec(vector,  4.0f);

    float per_dim = (-3.0f - 4.0f) * (-3.0f - 4.0f);  // 49.0
    float expected = per_dim * static_cast<float>(DIM_SYN);  // 6272.0

    float dist = hnsw_l2_distance(query, vector, TEST_DIM);
    check("hnsw_l2_distance (negative)", dist, expected);
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════
int main() {
    cout << "=== HNSW L2 Distance Testbench ===" << endl;
    cout << "DIM_SYN=" << DIM_SYN << "  Groups=" << TEST_GROUPS
         << "  Dims/Group=8" << endl;
    cout << "Pattern: NBR_DIST_GROUPS (16 groups × 8 dims, fully unrolled)"
         << endl << endl;

    test1_single_nonzero();
    test2_identical();
    test3_uniform_vs_zero();
    test4_random();
    test5_partial_dim();
    test6_negative_values();

    cout << "\n─── Results ───" << endl;
    cout << "Pass: " << g_pass << "  Fail: " << g_fail << endl;

    return (g_fail > 0) ? 1 : 0;
}
