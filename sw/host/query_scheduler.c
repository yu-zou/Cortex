/*
 * query_scheduler.c — Batch-query scheduler for codebook-reuse optimization
 *
 * Reorders queries by cluster_id so that queries in the same cluster are
 * dispatched consecutively.  On the FPGA this means the codebook is already
 * loaded for the second and subsequent queries in each cluster group,
 * eliminating redundant reload_codebook cycles.
 *
 * Algorithm:
 *   1. Fill order[] with identity permutation {0, 1, …, n_queries-1}
 *   2. Stable-sort by (cluster_id, original_index) using qsort
 *   3. Count unique clusters during a linear pass
 *
 * Complexity: O(n log n) time, O(1) auxiliary space (order array supplied
 * by caller).
 */
#include "query_scheduler.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

/* ---- internal helpers ------------------------------------------------ */

/** Comparator context (avoids qsort_r/qsort_s portability headaches). */
static const uint32_t *cmp_cluster_ids = NULL;

/** Comparator: sort by cluster_id ASC, then by original index ASC. */
static int cmp_order(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    uint32_t ca = cmp_cluster_ids[ia];
    uint32_t cb = cmp_cluster_ids[ib];

    if (ca < cb) return -1;
    if (ca > cb) return +1;
    /* cluster_ids equal — preserve original order */
    if (ia < ib) return -1;
    if (ia > ib) return +1;
    return 0;
}

/* ---- public API ------------------------------------------------------ */

int schedule_queries(const uint32_t *cluster_ids, int n_queries, int *order)
{
    if (!cluster_ids || !order || n_queries <= 0)
        return 0;

    /* 1. Identity permutation */
    for (int i = 0; i < n_queries; i++)
        order[i] = i;

    /* 2. Sort by (cluster_id, original_index) using plain qsort */
    cmp_cluster_ids = cluster_ids;
    qsort(order, (size_t)n_queries, sizeof(int), cmp_order);
    cmp_cluster_ids = NULL;

    /* 3. Count unique clusters */
    int unique_clusters = 1;  /* at least one cluster is present */
    for (int i = 1; i < n_queries; i++) {
        uint32_t prev = cluster_ids[order[i - 1]];
        uint32_t cur  = cluster_ids[order[i]];
        if (cur != prev)
            unique_clusters++;
    }

    return unique_clusters;
}

int schedule_reuse_ratio(const uint32_t *cluster_ids, int n_queries)
{
    if (!cluster_ids || n_queries <= 1)
        return 0;

    int reused = 0;
    for (int i = 1; i < n_queries; i++) {
        if (cluster_ids[i] == cluster_ids[i - 1])
            reused++;
    }

    /* integer percentage (rounds down; caller can also compute float) */
    return (reused * 100) / (n_queries - 1);
}

/* ---- self-test ------------------------------------------------------- */

#ifdef SCHEDULER_TEST_MAIN

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_QUERIES  100
#define N_CLUSTERS  10

int main(void)
{
    uint32_t cluster_ids[N_QUERIES];
    int      order[N_QUERIES];

    srand((unsigned int)time(NULL));

    /* Generate random cluster assignments */
    for (int i = 0; i < N_QUERIES; i++)
        cluster_ids[i] = (uint32_t)(rand() % N_CLUSTERS);

    /* Schedule */
    int n_clusters = schedule_queries(cluster_ids, N_QUERIES, order);

    /* Build reordered cluster array for reuse-ratio computation */
    uint32_t reordered[N_QUERIES];
    for (int i = 0; i < N_QUERIES; i++)
        reordered[i] = cluster_ids[order[i]];

    int reuse = schedule_reuse_ratio(reordered, N_QUERIES);

    /* ---- print statistics ------------------------------------------- */
    printf("=== Query Scheduler Statistics ===\n");
    printf("  Total queries          : %d\n", N_QUERIES);
    printf("  Unique clusters        : %d\n", n_clusters);
    printf("  Codebook-reuse ratio   : %d%%\n", reuse);

    /* Per-cluster query counts */
    printf("\n  Queries per cluster (after reordering):\n");
    int cnt = 1;
    for (int i = 1; i <= N_QUERIES; i++) {
        if (i == N_QUERIES || reordered[i] != reordered[i - 1]) {
            printf("    cluster %2u : %d queries\n",
                   reordered[i - 1], cnt);
            cnt = 1;
        } else {
            cnt++;
        }
    }

    /* Sparse "table view" of first 20 reordered indices */
    printf("\n  First 20 dispatch order (cluster_id):\n   ");
    for (int i = 0; i < 20 && i < N_QUERIES; i++)
        printf(" [%d]%u", order[i], reordered[i]);
    printf("\n");

    /* Re-ordered cluster sequence — visual groups */
    printf("\n  Cluster sequence (groups):\n   ");
    uint32_t prev = reordered[0];
    printf("%u", prev);
    for (int i = 1; i < N_QUERIES; i++) {
        if (reordered[i] != prev) {
            printf(" → %u", reordered[i]);
            prev = reordered[i];
        }
    }
    printf("\n");

    /* Sanity check: all original indices present exactly once */
    int seen[N_QUERIES];
    for (int i = 0; i < N_QUERIES; i++) seen[i] = 0;
    for (int i = 0; i < N_QUERIES; i++) {
        int idx = order[i];
        if (idx < 0 || idx >= N_QUERIES) {
            printf("  ERROR: order[%d] = %d out of bounds\n", i, idx);
            return 1;
        }
        seen[idx]++;
    }
    for (int i = 0; i < N_QUERIES; i++) {
        if (seen[i] != 1) {
            printf("  ERROR: original index %d appears %d times\n",
                   i, seen[i]);
            return 1;
        }
    }

    printf("\n  ✓ Permutation valid (every index appears exactly once)\n");
    printf("  ✓ Reuse ratio: %d%% of queries share cluster with predecessor\n",
           reuse);
    printf("=== End ===\n");
    return 0;
}

#endif /* SCHEDULER_TEST_MAIN */
