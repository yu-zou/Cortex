/*
 * query_scheduler.h — Batch-query scheduler for codebook-reuse optimization
 *
 * Sorts an array of queries by cluster_id so that consecutive queries sharing
 * the same cluster can skip FPGA codebook reload (reload_codebook=0).
 *
 * Reference: Cortex FPGA reload_codebook optimization — when consecutive
 * queries hit the same cluster, the codebook is already loaded on-chip.
 */
#ifndef QUERY_SCHEDULER_H
#define QUERY_SCHEDULER_H

#include <stdint.h>

/**
 * schedule_queries — Group batch queries by cluster_id
 *
 * @cluster_ids[0..n_queries-1]:  Input cluster assignments
 * @n_queries:                     Number of queries in the batch
 * @order[0..n_queries-1]:        Output permutation — query indices sorted
 *                                 by (cluster_id ASC, original_index ASC)
 *
 * Returns: number of unique clusters in the batch.
 *
 * After calling, dispatch queries in the order given by @order.  Consecutive
 * queries with the same cluster_id benefit from codebook reuse.
 */
int schedule_queries(const uint32_t *cluster_ids, int n_queries, int *order);

/**
 * schedule_reuse_ratio — Compute codebook-reuse percentage after reordering
 *
 * @cluster_ids[0..n_queries-1]:  Cluster assignments *in dispatch order*
 *                                (i.e., cluster_ids[order[i]] after scheduling)
 * @n_queries:                     Number of queries
 *
 * Returns: percentage (0–100) of queries whose cluster matches the preceding
 *          query.  100 % means every query benefits from codebook reuse.
 *
 * Example: cluster_ids = {2,2,2,5,5,3}  →  queries 1,2 share with prev (2),
 *          query 4 shares with prev (5)  →  3/5 = 60 %
 */
int schedule_reuse_ratio(const uint32_t *cluster_ids, int n_queries);

#endif /* QUERY_SCHEDULER_H */
