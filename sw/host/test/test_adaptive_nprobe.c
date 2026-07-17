/*
 * Unit tests for adaptive_nprobe()
 *
 * Compile:
 *   gcc -std=c11 -o /tmp/test_anp test_adaptive_nprobe.c ../adaptive_nprobe.c -lm && /tmp/test_anp
 */
#include <stdio.h>
#include <math.h>
#include "../adaptive_nprobe.c"

static int passed = 0, failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { printf("  FAIL at %s:%d\n", __FILE__, __LINE__); failed++; } \
    else { passed++; } \
} while(0)

int main(void)
{
    /* =================================================================
     * adaptive_nprobe() — individual tests
     * ================================================================= */

    /* ---- Test 1: Clear gap detected early ----------------------------
     * dist[2] (60.0) is 60× larger than dist[0] (1.0).
     * ratio = 1.0/60.0 ≈ 0.0167 < 3.0 → gap found at k=2.
     */
    {
        float dists[] = {1.0f, 50.0f, 60.0f, 70.0f, 80.0f};
        int result = adaptive_nprobe(dists, 5, 3.0f, 2);
        printf("Test 1 (clear gap):          adaptive_nprobe() = %d  (expect 2)\n", result);
        CHECK(result == 2);
    }

    /* ---- Test 2: No gap with small threshold -------------------------
     * With threshold=0.15: ratio 1.0/3.0 ≈ 0.333 >= 0.15 → no gap at k=2.
     * All subsequent checks also pass → return nprobe_max=5.
     * (threshold > 1.0 always triggers on sorted asc data; we use 0.15
     *  to demonstrate the no-gap path.)
     */
    {
        float dists[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        int result = adaptive_nprobe(dists, 5, 0.15f, 2);
        printf("Test 2 (no gap, thr=0.15):  adaptive_nprobe() = %d  (expect 5)\n", result);
        CHECK(result == 5);
    }

    /* ---- Test 3: Exact match (first_dist == 0) -----------------------
     * first_dist ≤ 0.0 → immediate return of nprobe_max = 3.
     */
    {
        float dists[] = {0.0f, 10.0f, 20.0f};
        int result = adaptive_nprobe(dists, 3, 3.0f, 1);
        printf("Test 3 (first_dist=0):       adaptive_nprobe() = %d  (expect 3)\n", result);
        CHECK(result == 3);
    }

    /* ---- Test 4: min_nprobe >= nprobe_max ---------------------------
     * Guard clause: min_nprobe (5) >= nprobe_max (3) → return nprobe_max = 3.
     */
    {
        float dists[] = {1.0f, 2.0f, 3.0f};
        int result = adaptive_nprobe(dists, 3, 3.0f, 5);
        printf("Test 4 (min>=max):           adaptive_nprobe() = %d  (expect 3)\n", result);
        CHECK(result == 3);
    }

    /* ---- Test 5: NULL dists -----------------------------------------
     * Guard clause: !dists → return min_nprobe = 4.
     */
    {
        int result = adaptive_nprobe(NULL, 10, 3.0f, 4);
        printf("Test 5 (NULL dists):         adaptive_nprobe() = %d  (expect 4)\n", result);
        CHECK(result == 4);
    }

    /* ---- Test 6: nprobe_max < 1 -------------------------------------
     * Guard clause: nprobe_max < 1 → return min_nprobe = 2.
     */
    {
        float dists[] = {1.0f};
        int result = adaptive_nprobe(dists, 0, 3.0f, 2);
        printf("Test 6 (nprobe_max < 1):     adaptive_nprobe() = %d  (expect 2)\n", result);
        CHECK(result == 2);
    }

    /* ---- Test 7: min_nprobe < 1 clamped to 1 ------------------------
     * Clamp: min_nprobe=0 → min_nprobe=1. Then always returns min_nprobe=1
     * for threshold=3.0 with positive first_dist.
     */
    {
        float dists[] = {5.0f, 10.0f, 15.0f};
        int result = adaptive_nprobe(dists, 3, 3.0f, 0);
        printf("Test 7 (min_nprobe clamp):   adaptive_nprobe() = %d  (expect 1)\n", result);
        CHECK(result == 1);
    }

    /* =================================================================
     * adaptive_nprobe_default() — convenience wrapper (thr=3.0, min=8)
     * ================================================================= */

    /* ---- Test 8: Default with gap -----------------------------------
     * With thr=3.0 and min=8, the loop always triggers at k=8 because
     * first_dist / dists[8] ≤ 1.0 < 3.0. Returns min_nprobe = 8.
     * (This tests the default-path code, not gap detection logic.)
     */
    {
        float dists[] = {1.0f, 50.0f, 60.0f, 70.0f, 80.0f,
                         80.0f, 80.0f, 80.0f, 80.0f, 80.0f,
                         80.0f, 80.0f, 80.0f, 80.0f, 80.0f};
        int result = adaptive_nprobe_default(dists, 15);
        printf("Test 8 (default w/ gap):     adaptive_nprobe_default() = %d  (expect 8)\n", result);
        CHECK(result == 8);
    }

    /* ---- Test 9: Default, nprobe_max < min_nprobe -------------------
     * nprobe_max=5 < min_nprobe=8 → guard triggers → return nprobe_max=5.
     */
    {
        float dists[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        int result = adaptive_nprobe_default(dists, 5);
        printf("Test 9 (default small max): adaptive_nprobe_default() = %d  (expect 5)\n", result);
        CHECK(result == 5);
    }

    /* =================================================================
     * Summary
     * ================================================================= */
    printf("\nadaptive_nprobe: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
