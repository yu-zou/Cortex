/*
 * test_query_scheduler.c — Unit tests for batch-query scheduler
 *
 * Validates schedule_queries() grouping/counting and
 * schedule_reuse_ratio() percentage computation.
 */

#include "../query_scheduler.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  %s: ", name)
#define CHECK(cond) do { \
    if (!(cond)) { printf("FAIL (%s:%d)\n", __FILE__, __LINE__); tests_failed++; } \
    else { tests_passed++; } \
} while(0)
#define TEST_DONE() printf("OK\n")

int main(void)
{
    /* ================================================================
     *  schedule_queries() tests
     * ================================================================ */

    printf("--- schedule_queries ---\n");

    /* ---- 1. Normal: {2, 1, 2, 1, 2}, n=5 ------------------------- */
    {
        TEST("Normal grouping by cluster_id ASC");
        const uint32_t cluster_ids[] = {2, 1, 2, 1, 2};
        int order[5];
        int n_clusters = schedule_queries(cluster_ids, 5, order);
        /* Expected order: cluster 1 first (indices 1,3), then cluster 2 (indices 0,2,4) */
        CHECK(n_clusters == 2);
        CHECK(order[0] == 1);
        CHECK(order[1] == 3);
        CHECK(order[2] == 0);
        CHECK(order[3] == 2);
        CHECK(order[4] == 4);
        TEST_DONE();
    }

    /* ---- 2. Single element: {5}, n=1 ------------------------------ */
    {
        TEST("Single element");
        const uint32_t cluster_ids[] = {5};
        int order[1];
        int n_clusters = schedule_queries(cluster_ids, 1, order);
        CHECK(n_clusters == 1);
        CHECK(order[0] == 0);
        TEST_DONE();
    }

    /* ---- 3. All same: {3, 3, 3}, n=3 ------------------------------ */
    {
        TEST("All queries in same cluster");
        const uint32_t cluster_ids[] = {3, 3, 3};
        int order[3];
        int n_clusters = schedule_queries(cluster_ids, 3, order);
        CHECK(n_clusters == 1);
        /* Identity permutation since all equal */
        CHECK(order[0] == 0);
        CHECK(order[1] == 1);
        CHECK(order[2] == 2);
        TEST_DONE();
    }

    /* ---- 4a. NULL cluster_ids pointer ----------------------------- */
    {
        TEST("NULL cluster_ids returns 0");
        int order[3];
        CHECK(schedule_queries(NULL, 3, order) == 0);
        TEST_DONE();
    }

    /* ---- 4b. NULL order pointer ---------------------------------- */
    {
        TEST("NULL order returns 0");
        const uint32_t cluster_ids[] = {1, 2, 3};
        CHECK(schedule_queries(cluster_ids, 3, NULL) == 0);
        TEST_DONE();
    }

    /* ---- 4c. n_queries == 0 --------------------------------------- */
    {
        TEST("n_queries == 0 returns 0");
        const uint32_t cluster_ids[] = {1, 2, 3};
        int order[3];
        CHECK(schedule_queries(cluster_ids, 0, order) == 0);
        TEST_DONE();
    }

    /* ---- 4d. n_queries < 0 (negative) ----------------------------- */
    {
        TEST("n_queries < 0 returns 0");
        const uint32_t cluster_ids[] = {1, 2, 3};
        int order[3];
        CHECK(schedule_queries(cluster_ids, -1, order) == 0);
        TEST_DONE();
    }

    /* ================================================================
     *  schedule_reuse_ratio() tests
     * ================================================================ */

    printf("--- schedule_reuse_ratio ---\n");

    /* ---- 1. {2,2,2,5,5,3} → 60% ---------------------------------- */
    {
        TEST("Mixed clusters — 60% reuse");
        const uint32_t cluster_ids[] = {2, 2, 2, 5, 5, 3};
        CHECK(schedule_reuse_ratio(cluster_ids, 6) == 60);
        TEST_DONE();
    }

    /* ---- 2. {1,1,1} → 100% ---------------------------------------- */
    {
        TEST("All same — 100% reuse");
        const uint32_t cluster_ids[] = {1, 1, 1};
        CHECK(schedule_reuse_ratio(cluster_ids, 3) == 100);
        TEST_DONE();
    }

    /* ---- 3. {1,2,3} → 0% ------------------------------------------ */
    {
        TEST("All different — 0% reuse");
        const uint32_t cluster_ids[] = {1, 2, 3};
        CHECK(schedule_reuse_ratio(cluster_ids, 3) == 0);
        TEST_DONE();
    }

    /* ---- 4a. NULL pointer ------------------------------------------ */
    {
        TEST("NULL returns 0");
        CHECK(schedule_reuse_ratio(NULL, 5) == 0);
        TEST_DONE();
    }

    /* ---- 4b. n_queries == 0 ---------------------------------------- */
    {
        TEST("n_queries == 0 returns 0");
        const uint32_t cluster_ids[] = {1, 2, 3};
        CHECK(schedule_reuse_ratio(cluster_ids, 0) == 0);
        TEST_DONE();
    }

    /* ---- 4c. n_queries == 1 (no transitions) ---------------------- */
    {
        TEST("n_queries == 1 returns 0");
        const uint32_t cluster_ids[] = {42};
        CHECK(schedule_reuse_ratio(cluster_ids, 1) == 0);
        TEST_DONE();
    }

    /* ---- Summary --------------------------------------------------- */
    printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
