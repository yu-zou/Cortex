// ═══════════════════════════════════════════════════════════════
// testbench_adc.cpp — Standalone Vitis HLS testbench for ADC
// (L2 distance) computation.
//
// Tests the core distance formula:
//   total_dist = Σ_m Σ_d  (query[m·Ds + d] - codebook[m][c][d])²
//
// where:
//   m  = sub-quantizer index (0..M_val-1)
//   c  = centroid index selected by PQ code
//   d  = dimension within sub-vector (0..Ds-1)
//   Ds = DIM_val / M_val
//
// Run via:  run_adc_test.tcl  (Vitis HLS)
// Compile:  g++ -I../include testbench_adc.cpp -o testbench_adc
// ═══════════════════════════════════════════════════════════════

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "../include/acc_top.h"

// Forward declaration (defined in src/acc_top.cpp)
float sub_dist_l2(float q, float c);
void systolic_topk_insert(float cand_dist, ap_uint<64> cand_addr,
                          ap_uint<32> cand_len, TopKCell cells[TOPK_MAX]);

using namespace std;

// ─── Test configuration (small M=2 for clarity) ───
// Array dimensions from acc_top.h: M_SYN=16, KS=256, DS_SYN=8
// We only populate the first TEST_M sub-quantizers.
static const int TEST_M   = 2;          // Active sub-quantizers
static const int TEST_DIM = TEST_M * DS_SYN;  // = 16
static const int TEST_DS  = DS_SYN;     // = 8

// ─── Reference: single L2 sub-distance (identical to acc_top.cpp) ───
static float ref_sub_dist_l2(float q, float c) {
    float d = q - c;
    return d * d;
}

// ─── Reference: full ADC L2 distance (matches compute_engine Stage 3) ───
static float ref_adc_l2(
    const float codebook[M_SYN][KS][DS_SYN],
    const float* query,
    const uint8_t* pq_codes,    // centroid index per sub-quantizer
    int M_val,
    int DIM_val
) {
    int Ds = DIM_val / M_val;
    float total = 0.0f;
    for (int m = 0; m < M_val; m++) {
        uint8_t c    = pq_codes[m];
        float   sub  = 0.0f;
        for (int d = 0; d < Ds; d++) {
            float cent  = codebook[m][c][d];
            float q_sub = query[m * Ds + d];
            sub += ref_sub_dist_l2(q_sub, cent);
        }
        total += sub;
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

// ═══════════════════════════════════════════════════════════════
// Test 1: Query exactly matches codebook centroid → distance ≈ 0
//
// Sets:
//   codebook[0][0][d] = d/10.0   for m=0, centroid=0
//   codebook[1][5][d] = (d+5)/10.0  for m=1, centroid=5
//   query[m*DS+d]     = same values (exact match per sub-vector)
//   pq_codes          = {0, 5}
//
// Expected: total_dist = 0
// ═══════════════════════════════════════════════════════════════
static void test1_exact_match() {
    cout << "\n─── Test 1: Query = Codebook Centroid → distance ≈ 0 ───" << endl;

    float    codebook[M_SYN][KS][DS_SYN] = {};
    float    query[DIM_SYN] = {};
    uint8_t  pq_codes[M_SYN] = {0, 5};

    // Populate centroids with known values
    for (int d = 0; d < DS_SYN; d++) {
        codebook[0][0][d] =  d        / 10.0f;   // m=0, centroid 0
        codebook[1][5][d] = (d + 5.0f) / 10.0f;  // m=1, centroid 5
    }

    // Query matches the centroid values exactly
    for (int d = 0; d < DS_SYN; d++) {
        query[0 * DS_SYN + d] = codebook[0][0][d];  // m=0 → centroid 0
        query[1 * DS_SYN + d] = codebook[1][5][d];  // m=1 → centroid 5
    }

    float dist = ref_adc_l2(codebook, query, pq_codes, TEST_M, TEST_DIM);
    check("Total ADC distance", dist, 0.0f);

    // Also verify per-sub-quantizer contribution is zero
    float sub0 = 0.0f;
    for (int d = 0; d < DS_SYN; d++)
        sub0 += ref_sub_dist_l2(query[0 * DS_SYN + d], codebook[0][0][d]);
    check("  sub_dist(m=0)", sub0, 0.0f);

    float sub1 = 0.0f;
    for (int d = 0; d < DS_SYN; d++)
        sub1 += ref_sub_dist_l2(query[1 * DS_SYN + d], codebook[1][5][d]);
    check("  sub_dist(m=1)", sub1, 0.0f);
}

// ═══════════════════════════════════════════════════════════════
// Test 2: Uniform query (1.0) vs zero-initialized codebook
//
//   codebook[m][c][d] = 0.0  (all centroids zero)
//   query[d]          = 1.0  (all dims)
//   pq_codes          = {0, 0}
//
// Expected: total_dist = DIM * (1.0 - 0.0)² = TEST_DIM = 16.0
// ═══════════════════════════════════════════════════════════════
static void test2_uniform_query_zero_cb() {
    cout << "\n─── Test 2: Uniform Query (1.0) vs Zero Codebook ───" << endl;

    float    codebook[M_SYN][KS][DS_SYN] = {};
    float    query[DIM_SYN] = {};
    uint8_t  pq_codes[M_SYN] = {0, 0};

    for (int d = 0; d < TEST_DIM; d++)
        query[d] = 1.0f;

    float dist   = ref_adc_l2(codebook, query, pq_codes, TEST_M, TEST_DIM);
    float expect = static_cast<float>(TEST_DIM);  // 16 * (1.0)² = 16.0

    check("Total ADC distance", dist, expect);
}

// ═══════════════════════════════════════════════════════════════
// Test 3: Zero codebook + non-uniform query → distance = Σ(query²)
//
//   codebook[m][c][d] = 0.0
//   query[0..7]       = 2.0
//   query[8..15]      = 3.0
//   pq_codes          = {0, 0}
//
// Expected:  8*4.0 + 8*9.0 = 32.0 + 72.0 = 104.0
// ═══════════════════════════════════════════════════════════════
static void test3_zero_cb_nonzero_query() {
    cout << "\n─── Test 3: Zero Codebook, Non-Zero Query → Σ(query²) ───" << endl;

    float    codebook[M_SYN][KS][DS_SYN] = {};
    float    query[DIM_SYN] = {};
    uint8_t  pq_codes[M_SYN] = {0, 0};

    // m=0 sub-vector: all 2.0
    for (int d = 0; d < DS_SYN; d++)
        query[0 * DS_SYN + d] = 2.0f;
    // m=1 sub-vector: all 3.0
    for (int d = 0; d < DS_SYN; d++)
        query[1 * DS_SYN + d] = 3.0f;

    float dist   = ref_adc_l2(codebook, query, pq_codes, TEST_M, TEST_DIM);
    float expect = DS_SYN * (4.0f + 9.0f);  // 8*4 + 8*9 = 104

    check("Total ADC distance", dist, expect);
}

// ═══════════════════════════════════════════════════════════════
// Test 4: Verify m=0 sub-dist = 0 (spec requirement)
//
// "Verify for m=0: codebook[0][0][d] = d/10.0,
//  query[0*8+d] = d/10.0 → sub_dist = 0 for m=0"
//
// m=1 uses arbitrary non-matching values to ensure total ≠ 0.
// ═══════════════════════════════════════════════════════════════
static void test4_m0_match() {
    cout << "\n─── Test 4: m=0 centroid match → sub_dist(m=0) = 0 ───" << endl;

    float    codebook[M_SYN][KS][DS_SYN] = {};
    float    query[DIM_SYN] = {};
    uint8_t  pq_codes[M_SYN] = {0, 0};

    // m=0: query matches centroid exactly
    for (int d = 0; d < DS_SYN; d++) {
        codebook[0][0][d] = d / 10.0f;
        query[0 * DS_SYN + d] = d / 10.0f;
    }

    // m=1: arbitrary non-matching values
    for (int d = 0; d < DS_SYN; d++) {
        codebook[1][0][d] = 42.0f;
        query[1 * DS_SYN + d] = 99.0f;
    }

    // sub-dist for m=0 should be exactly 0
    float sub0 = 0.0f;
    for (int d = 0; d < DS_SYN; d++)
        sub0 += ref_sub_dist_l2(query[0 * DS_SYN + d], codebook[0][0][d]);
    check("sub_dist_l2(m=0)", sub0, 0.0f);

    // sub-dist for m=1 should be 8 * (99 - 42)² = 8 * 3249 = 25992
    float sub1_expect = DS_SYN * (99.0f - 42.0f) * (99.0f - 42.0f);
    float sub1 = 0.0f;
    for (int d = 0; d < DS_SYN; d++)
        sub1 += ref_sub_dist_l2(query[1 * DS_SYN + d], codebook[1][0][d]);
    check("sub_dist_l2(m=1)", sub1, sub1_expect);

    // Total = m=0 + m=1
    float dist = ref_adc_l2(codebook, query, pq_codes, TEST_M, TEST_DIM);
    check("Total ADC distance (= m=1 only)", dist, sub1_expect);
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════
int main() {
    cout << "=== ADC (L2 Distance) Testbench ===" << endl;
    cout << "M_SYN=" << M_SYN << "  DIM_SYN=" << DIM_SYN
         << "  DS_SYN=" << DS_SYN << "  KS=" << KS << endl;
    cout << "Test: M=" << TEST_M << "  DIM=" << TEST_DIM
         << "  DS=" << TEST_DS << endl;

    test1_exact_match();
    test2_uniform_query_zero_cb();
    test3_zero_cb_nonzero_query();
    test4_m0_match();

    cout << "\n─── Results ───" << endl;
    cout << "Pass: " << g_pass << "  Fail: " << g_fail << endl;

    return (g_fail > 0) ? 1 : 0;
}
