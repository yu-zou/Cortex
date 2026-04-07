#ifndef CORTEX_DRIVER_API_H
#define CORTEX_DRIVER_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version — loader checks this first */
#define CORTEX_API_VERSION ((uint32_t)1)

/* Maximum top-K results — fixed-size array, no heap allocation across ABI */
#define CORTEX_TOPK_MAX 500

/* Metric types */
#define CORTEX_METRIC_L2 0
#define CORTEX_METRIC_IP 1

/* Version check — FIRST function called by loader */
uint32_t cortex_api_version(void);

/* Lifecycle */
int cortex_driver_init(const char* config_json);
void cortex_driver_shutdown(void);

/* Top-K result entry */
struct cortex_topk_entry {
    uint64_t vector_id;
    float distance;
};

/* Semantic read result — fixed-size, no pointers across ABI */
struct cortex_read_result {
    int32_t status;       /* 0 = success, negative errno = error */
    uint32_t count;       /* number of valid entries in entries[] */
    struct cortex_topk_entry entries[CORTEX_TOPK_MAX];
};

/* Core operations — all thread-safe, driver handles synchronization */

/**
 * cortex_semantic_read — Run IVFPQ ADC search on a cluster object
 * @cluster_id:   IVF cluster identifier (chosen by RGW via centroid distance)
 * @object_data:  Raw bytes of the Ceph object (ClusterHeader + codebook + PQ codes)
 * @object_size:  Size of object_data in bytes
 * @query_vec:    Query vector (query_dim floats)
 * @query_dim:    Dimensionality of query vector
 * @top_k:        Number of results to return (max CORTEX_TOPK_MAX)
 * @metric_type:  CORTEX_METRIC_L2 or CORTEX_METRIC_IP
 */
struct cortex_read_result cortex_semantic_read(
    uint32_t cluster_id,
    const void* object_data,
    uint64_t object_size,
    const float* query_vec,
    uint32_t query_dim,
    uint32_t top_k,
    uint32_t metric_type
);

/**
 * cortex_semantic_write — Load codebook and PQ data for a cluster
 * @cluster_id:     IVF cluster identifier
 * @codebook:       PQ codebook bytes (M * ksub * dsub floats)
 * @codebook_size:  Size of codebook in bytes
 * @pq_payload:     PQ-encoded vector bytes
 * @payload_size:   Size of pq_payload in bytes
 * @vector_count:   Number of vectors in pq_payload
 */
int cortex_semantic_write(
    uint32_t cluster_id,
    const void* codebook,
    uint64_t codebook_size,
    const void* pq_payload,
    uint64_t payload_size,
    uint32_t vector_count
);

/* Cache state queries */
int cortex_is_cluster_cached(uint32_t cluster_id);
uint32_t cortex_get_last_cluster_id(void);

#ifdef __cplusplus
}
#endif

#endif /* CORTEX_DRIVER_API_H */
