#ifndef CORTEX_FPGA_PROTOCOL_H
#define CORTEX_FPGA_PROTOCOL_H
#include <stdint.h>

/* Wire integers and IEEE binary32 are little endian; never send C structs.
 * One connection: HELLO/reply, SEARCH/reply, then close. See driver/FPGA_OSD.md.
 */
#define CORTEX_FPGA_MAGIC UINT32_C(0x31465043) /* CPF1 */
#define CORTEX_FPGA_PROTOCOL_VERSION 1
#define CORTEX_FPGA_HEADER_BYTES 40
#define CORTEX_FPGA_HELLO 1
#define CORTEX_FPGA_SEARCH 2
#define CORTEX_FPGA_REPLY UINT32_C(0x80000000)
#define CORTEX_FPGA_PROFILE_CPQ1V2_RESIDUAL_L2 1
#define CORTEX_FPGA_PROFILE_CPQ1V2_RESIDUAL_IP 2
#define CORTEX_FPGA_EXECUTION_HARDWARE 1
#define CORTEX_FPGA_MAX_OBJECT_BYTES (64 + 786432 + 3072 + 8192 * 112)
#define CORTEX_FPGA_DRIVER_ABI 2

/* Header offsets: magic u32@0, version u32@4, opcode u32@8,
 * profile u32@12, request_id u64@16, payload_bytes u64@24,
 * signed negative-errno status i32@32, reserved zero u32@36.
 * HELLO request: empty. HELLO reply: execution_kind u32, layout_version u32.
 * SEARCH request: cluster_id u32, dim u32, k u32, metric u32,
 * object_bytes u64, timeout_remaining_ms u32, reserved zero u32,
 * query[dim] binary32, exact CPQ1v2 object bytes.
 * SEARCH success: count u32, reserved zero u32, count packed 16-byte records:
 * distance f32, row_token u64, document_length u32. Error: empty payload.
 */
#endif
