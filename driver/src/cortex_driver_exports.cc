#include "cortex_driver_api.h"
#include "smartssd_driver.h"
#include <errno.h>

namespace cortex {
    extern SmartSSDDriver& get_driver();
    extern void destroy_driver();
}

extern "C" {

uint32_t cortex_api_version(void) {
    return CORTEX_API_VERSION;
}

int cortex_driver_init(const char* config_json) {
    return cortex::get_driver().init(config_json);
}

void cortex_driver_shutdown(void) {
    cortex::get_driver().shutdown();
    cortex::destroy_driver();
}

struct cortex_read_result cortex_semantic_read(
    uint32_t cluster_id, const void* object_data, uint64_t object_size,
    const float* query_vec, uint32_t query_dim, uint32_t top_k, uint32_t metric_type)
{
    return cortex::get_driver().semantic_read(
        cluster_id, object_data, object_size, query_vec, query_dim, top_k, metric_type);
}

int cortex_semantic_write(
    uint32_t cluster_id, const void* codebook, uint64_t codebook_size,
    const void* pq_payload, uint64_t payload_size, uint32_t vector_count)
{
    return cortex::get_driver().semantic_write(
        cluster_id, codebook, codebook_size, pq_payload, payload_size, vector_count);
}

int cortex_is_cluster_cached(uint32_t cluster_id) {
    return cortex::get_driver().is_cluster_cached(cluster_id) ? 1 : 0;
}

uint32_t cortex_get_last_cluster_id(void) {
    return cortex::get_driver().get_last_cluster_id();
}

}
