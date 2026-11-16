#pragma once
#include "cortex_driver_api.h"
#include "cortex_fpga_protocol.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional ABI2 extension, resolved only by the explicitly selected FPGA OSD
 * backend. Callback is borrowed for this call, must not block or throw.
 * Base ABI2 semantic_read uses the same finite configured deadline.
 */
typedef int (*cortex_fpga_cancel_fn)(void*);
struct cortex_read_result cortex_fpga_read_cancellable(
    uint32_t cluster_id, const void* object_data, uint64_t object_size,
    const float* query_vec, uint32_t query_dim, uint32_t top_k,
    uint32_t metric_type, uint32_t search_mode,
    cortex_fpga_cancel_fn canceled, void* cancel_context);
#ifdef __cplusplus
}
#endif
