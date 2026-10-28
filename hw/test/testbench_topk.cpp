// ==========================================================================
// testbench_topk.cpp — Standalone Vitis HLS C Testbench for
//                      systolic_topk_insert()
//
// Tests the sorting invariant of the systolic Top-K array:
//   cells[i].best_dist <= cells[i+1].best_dist  for all i
//
// Compilation (via run_topk_test.tcl):
//   open_project topk_test_proj -reset
//   set_top systolic_topk_insert
//   add_files src/acc_top.cpp
//   add_files -tb test/testbench_topk.cpp
//   csim_design -clean
//
// Notes:
//   - systolic_topk_insert is static inline in acc_top.cpp.
//     The Vitis HLS "set_top + add_files src" flow makes it available
//     during C simulation even though it is declared static.
//   - No DRAM, no compute_engine, no streams — pure function test.
// ==========================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

#include "../include/acc_top.h"

// Forward declaration (defined in src/acc_top.cpp)
void systolic_topk_insert(float cand_dist, ap_uint<64> cand_addr,
                          ap_uint<32> cand_len, TopKCell cells[TOPK_MAX]);

using namespace std;

// ─── Constants ───
static const float FP32_MAX = 3.402823466e+38f;

// ─── Helpers ───

/// Print first N and last N cells of a TopK array.
static void print_cells(const TopKCell cells[TOPK_MAX], int first_n = 5, int last_n = 5) {
    int printed = 0;

    for (int i = 0; i < first_n && i < TOPK_MAX; i++) {
        printf("  [%4d] dist=%-12.4f  addr=0x%016llx  len=%u\n",
               i,
               (double)cells[i].best_dist,
               (unsigned long long)cells[i].best_addr.to_uint64(),
               (unsigned)cells[i].best_len.to_uint());
        printed++;
    }

    if (printed < TOPK_MAX - last_n) {
        printf("  ... (%d cells omitted) ...\n", TOPK_MAX - first_n - last_n);
    }

    int start = TOPK_MAX - last_n;
    if (start < first_n) start = first_n;
    for (int i = start; i < TOPK_MAX; i++) {
        printf("  [%4d] dist=%-12.4f  addr=0x%016llx  len=%u\n",
               i,
               (double)cells[i].best_dist,
               (unsigned long long)cells[i].best_addr.to_uint64(),
               (unsigned)cells[i].best_len.to_uint());
    }
}

/// Verify the sorted invariant: cells[i].best_dist <= cells[i+1].best_dist
/// Also verify all non-inf cells have expected values (if expected != nullptr).
/// Returns number of failures (0 = pass).
static int verify_sorted(const TopKCell cells[TOPK_MAX],
                         const float expected[] = nullptr,
                         int n_expected = 0) {
    int failures = 0;

    // Check sorted ascending
    for (int i = 0; i < TOPK_MAX - 1; i++) {
        float a = cells[i].best_dist;
        float b = cells[i + 1].best_dist;
        if (a > b + 1e-6f) {  // tolerate FP epsilon
            printf("  FAIL: cells[%d].dist=%.4f > cells[%d].dist=%.4f\n",
                   i, (double)a, i + 1, (double)b);
            failures++;
        }
    }

    // Check expected values (if provided)
    for (int i = 0; i < n_expected; i++) {
        float got = cells[i].best_dist;
        float want = expected[i];
        if (fabs(got - want) > 1e-6f) {
            printf("  FAIL: cells[%d].dist=%.4f  expected=%.4f\n",
                   i, (double)got, (double)want);
            failures++;
        }
    }

    // Check remaining cells are still FP32_MAX (no corruption)
    for (int i = n_expected; i < TOPK_MAX; i++) {
        if (cells[i].best_dist < FP32_MAX - 1.0f) {
            printf("  FAIL: cells[%d].dist=%.4f  should be FP32_MAX (unexpected value)\n",
                   i, (double)cells[i].best_dist);
            failures++;
        }
    }

    return failures;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 1: Basic ascending sort
//   Insert [5.0, 3.0, 7.0, 1.0, 9.0] with unique addrs/lens.
//   Expected final sorted order: [1.0, 3.0, 5.0, 7.0, 9.0]
// ─────────────────────────────────────────────────────────────────────────
static int test_basic_insert() {
    cout << "\n=== Test 1: Basic Insert (5 values) ===" << endl;

    TopKCell cells[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; i++) {
        cells[i].best_dist = FP32_MAX;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // Insert in arbitrary order
    float vals[] = {5.0f, 3.0f, 7.0f, 1.0f, 9.0f};
    ap_uint<64> addrs[] = {100, 200, 300, 400, 500};
    ap_uint<32> lens[]  = {10,  20,  30,  40,  50};

    for (int i = 0; i < 5; i++) {
        systolic_topk_insert(vals[i], addrs[i], lens[i], cells);
    }

    print_cells(cells);

    float expected[] = {1.0f, 3.0f, 5.0f, 7.0f, 9.0f};
    int fail = verify_sorted(cells, expected, 5);

    if (fail == 0) {
        cout << "  >>> PASS: Test 1 (sorted correctly) <<<" << endl;
    } else {
        cout << "  >>> FAIL: Test 1 (" << fail << " errors) <<<" << endl;
    }
    return fail;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 2: Decreasing order (stress test)
//   Insert 10 values in strictly decreasing order [10, 9, 8, ..., 1].
//   This is the worst case — every insertion shifts all existing values.
//   Expected final sorted order: [1, 2, 3, ..., 10]
// ─────────────────────────────────────────────────────────────────────────
static int test_decreasing_order() {
    cout << "\n=== Test 2: Decreasing Order (10 values, worst case) ===" << endl;

    TopKCell cells[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; i++) {
        cells[i].best_dist = FP32_MAX;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // Insert 10 .. 1
    for (int i = 10; i >= 1; i--) {
        systolic_topk_insert((float)i, (ap_uint<64>)(i * 100), (ap_uint<32>)(i * 10), cells);
    }

    print_cells(cells);

    float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    int fail = verify_sorted(cells, expected, 10);

    if (fail == 0) {
        cout << "  >>> PASS: Test 2 (decreasing order sorted correctly) <<<" << endl;
    } else {
        cout << "  >>> FAIL: Test 2 (" << fail << " errors) <<<" << endl;
    }
    return fail;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 3: Duplicates
//   Insert [4.0, 4.0, 4.0].
//   Since systolic_topk_insert uses strict-less-than (<), equal values
//   are NOT swapped at the position they first encounter, but cascade
//   to the next FP32_MAX cell. All three should be stored consecutively
//   at cells[0..2] = 4.0.
// ─────────────────────────────────────────────────────────────────────────
static int test_duplicates() {
    cout << "\n=== Test 3: Duplicates (3 x 4.0) ===" << endl;

    TopKCell cells[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; i++) {
        cells[i].best_dist = FP32_MAX;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    systolic_topk_insert(4.0f, (ap_uint<64>)100, (ap_uint<32>)10, cells);
    systolic_topk_insert(4.0f, (ap_uint<64>)200, (ap_uint<32>)20, cells);
    systolic_topk_insert(4.0f, (ap_uint<64>)300, (ap_uint<32>)30, cells);

    print_cells(cells);

    int failures = 0;

    // Check sorted invariant
    for (int i = 0; i < TOPK_MAX - 1; i++) {
        if (cells[i].best_dist > cells[i + 1].best_dist + 1e-6f) {
            printf("  FAIL: cells[%d].dist=%.4f > cells[%d].dist=%.4f\n",
                   i, (double)cells[i].best_dist,
                   i + 1, (double)cells[i + 1].best_dist);
            failures++;
        }
    }

    // Check first three are 4.0
    for (int i = 0; i < 3; i++) {
        if (fabs(cells[i].best_dist - 4.0f) > 1e-6f) {
            printf("  FAIL: cells[%d].dist=%.4f  expected=4.0\n",
                   i, (double)cells[i].best_dist);
            failures++;
        }
    }

    // Check cells[3..] are still FP32_MAX
    for (int i = 3; i < TOPK_MAX; i++) {
        if (cells[i].best_dist < FP32_MAX - 1.0f) {
            printf("  FAIL: cells[%d].dist=%.4f  should be FP32_MAX (unexpected duplicate)\n",
                   i, (double)cells[i].best_dist);
            failures++;
        }
    }

    // Verify each insertion has a unique addr/len
    // Since duplicates are not swapped among themselves, the FIRST
    // insertion goes to cells[0], SECOND to cells[1], THIRD to cells[2].
    ap_uint<64> expected_addrs[] = {100, 200, 300};
    ap_uint<32> expected_lens[]  = {10,  20,  30};
    for (int i = 0; i < 3; i++) {
        if (cells[i].best_addr != expected_addrs[i]) {
            printf("  FAIL: cells[%d].addr=0x%llx  expected=0x%llx\n",
                   i,
                   (unsigned long long)cells[i].best_addr.to_uint64(),
                   (unsigned long long)expected_addrs[i].to_uint64());
            failures++;
        }
        if (cells[i].best_len != expected_lens[i]) {
            printf("  FAIL: cells[%d].len=%u  expected=%u\n",
                   i,
                   (unsigned)cells[i].best_len.to_uint(),
                   (unsigned)expected_lens[i].to_uint());
            failures++;
        }
    }

    if (failures == 0) {
        cout << "  >>> PASS: Test 3 (duplicates stored correctly) <<<" << endl;
    } else {
        cout << "  >>> FAIL: Test 3 (" << failures << " errors) <<<" << endl;
    }
    return failures;
}

// ─────────────────────────────────────────────────────────────────────────
// Test 4: doc_addr / doc_len propagation
//   Insert values with unique (addr, len) pairs and verify they propagate
//   correctly through the systolic chain. Use increasingly large distances
//   so each insertion lands at the next free position — then verify the
//   corresponding addr/len are correct.
// ─────────────────────────────────────────────────────────────────────────
static int test_addr_len_propagation() {
    cout << "\n=== Test 4: Addr/Len Propagation ===" << endl;

    TopKCell cells[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; i++) {
        cells[i].best_dist = FP32_MAX;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // Insert with increasing distance — each goes to the end of the
    // current occupied range. Use distinct addr/len to verify tracking.
    struct {
        float        dist;
        ap_uint<64>  addr;
        ap_uint<32>  len;
    } inserts[] = {
        {10.0f,  ap_uint<64>(0xAAAABBBBCCCCDDDDULL), ap_uint<32>(128)},
        {20.0f,  ap_uint<64>(0x1111222233334444ULL), ap_uint<32>(256)},
        {30.0f,  ap_uint<64>(0xDEADBEEFCAFEBABEULL), ap_uint<32>(512)},
    };

    int n_inserts = sizeof(inserts) / sizeof(inserts[0]);

    for (int i = 0; i < n_inserts; i++) {
        systolic_topk_insert(inserts[i].dist,
                             inserts[i].addr,
                             inserts[i].len,
                             cells);
    }

    print_cells(cells);

    int failures = 0;

    // Verify sorted invariant
    for (int i = 0; i < TOPK_MAX - 1; i++) {
        if (cells[i].best_dist > cells[i + 1].best_dist + 1e-6f) {
            printf("  FAIL: cells[%d].dist=%.4f > cells[%d].dist=%.4f\n",
                   i, (double)cells[i].best_dist,
                   i + 1, (double)cells[i + 1].best_dist);
            failures++;
        }
    }

    // Verify each inserted (dist, addr, len) triplet appears somewhere
    // and that addr/len are correctly associated with their dist value.
    for (int ins = 0; ins < n_inserts; ins++) {
        bool found_dist = false;
        for (int c = 0; c < TOPK_MAX; c++) {
            if (fabs(cells[c].best_dist - inserts[ins].dist) < 1e-6f &&
                cells[c].best_addr == inserts[ins].addr &&
                cells[c].best_len  == inserts[ins].len) {
                found_dist = true;
                break;
            }
        }
        if (!found_dist) {
            printf("  FAIL: (dist=%.1f, addr=0x%016llx, len=%u) not found in cells\n",
                   (double)inserts[ins].dist,
                   (unsigned long long)inserts[ins].addr.to_uint64(),
                   (unsigned)inserts[ins].len.to_uint());
            failures++;
        }
    }

    if (failures == 0) {
        cout << "  >>> PASS: Test 4 (addr/len propagate correctly) <<<" << endl;
    } else {
        cout << "  >>> FAIL: Test 4 (" << failures << " errors) <<<" << endl;
    }
    return failures;
}

// ─────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────
int main() {
    cout << "==============================================================" << endl;
    cout << "  systolic_topk_insert() — Standalone Testbench" << endl;
    cout << "  TOPK_MAX = " << TOPK_MAX << endl;
    cout << "==============================================================" << endl;

    int total_fail = 0;

    total_fail += test_basic_insert();
    total_fail += test_decreasing_order();
    total_fail += test_duplicates();
    total_fail += test_addr_len_propagation();

    cout << "\n==============================================================" << endl;
    if (total_fail == 0) {
        cout << "  ALL TESTS PASSED" << endl;
    } else {
        cout << "  SOME TESTS FAILED (" << total_fail << " total errors)" << endl;
    }
    cout << "==============================================================" << endl;

    return (total_fail == 0) ? 0 : 1;
}
