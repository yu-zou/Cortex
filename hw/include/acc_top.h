#ifndef ACC_TOP_H
#define ACC_TOP_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_axi_sdata.h>

// ─── Design Parameters ───
// KS=256 hardcoded, KS field no longer stored in Header
// FIFO_RES: 128b width, depth=TOPK_MAX
// Result format: dist(FP32,32b) + doc_addr(64b) + doc_length(32b) = 128 bit
#define M_MAX      96
#define DIM_MAX    768
#define KS         256         // Hardcoded, default PQ Codebook centroid count
#define TOPK_MAX   500
#define TOPK_INF_KEY     0x7F800000u

// ─── Synthesis parameters ───
// M=96 → Ds=8 (recall-optimal config per recall_compare.py)
#define M_SYN      96
#define DIM_SYN    768
#define DS_SYN     (DIM_SYN / M_SYN)  // = 8

// ─── PQ Entry format ───
// entry = M_SYN codes + 16 metadata = 112 bytes > 64 bytes
// → each entry occupies 2 × 512-bit beats (128 bytes, 16 bytes pad)
#define ENTRY_BYTES_SYN  (M_SYN + 16)        // = 112
#define ENTRY_BITS_SYN   (ENTRY_BYTES_SYN * 8)  // = 896
#define ENTRY_BEATS_SYN  2                    // beats per entry (512-bit each)
#define ENTRY_BYTES_PAD  128                  // padded entry size = 2*64

// ─── Result: 128-bit ───
typedef ap_uint<128>  result_word_t;

// ─── Internal stream words ───
typedef ap_uint<512>  cb_pq_word_t;      // codebook beat (16 FP32) / single 512-bit beat
typedef ap_uint<1024> pq_entry_word_t;   // PQ entry: 2 beats merged (codes + metadata)
typedef ap_uint<128>  res_word_t;

// ─── AXI4-Stream result packet ───
typedef ap_axiu<128, 1, 0, 0> axis_result_pkt_t;

// ─── Metadata ───
struct ComputeMeta {
    ap_uint<32>  m_actual;
    ap_uint<32>  dim_actual;
    ap_uint<32>  n_vectors;       // N vectors
    ap_uint<32>  top_k;
    ap_uint<2>   metric_id;
};

// ─── Top-K candidate ───
struct TopKCandidate {
    ap_uint<32>  dist_key;
    ap_uint<64>  doc_addr;
    ap_uint<32>  doc_len;
};

// ─── Top-level ───
void acc_top(
    ap_uint<64>  cluster_start_addr,
    ap_uint<64>  query_ddr_addr,
    ap_uint<64>  result_ddr_addr,
    ap_uint<32>  top_k,
    ap_uint<2>   metric_id,
    ap_uint<1>   reload_codebook,
    volatile bool &done,
    volatile bool  start,
    ap_uint<8>  *dram,
    hls::stream<axis_result_pkt_t> &axis_result
);

// ─── Sub-modules ───
void control_fsm(
    ap_uint<64>  cluster_start_addr,
    ap_uint<64>  query_ddr_addr,
    ap_uint<32>  top_k,
    ap_uint<2>   metric_id,
    ap_uint<1>   reload_codebook,
    volatile bool &done,
    volatile bool  start,
    volatile bool &dm_start,
    volatile bool  dm_done,
    volatile bool &comp_start,
    volatile bool  comp_done
);

void data_manager(
    ap_uint<64>                cluster_start_addr,
    ap_uint<64>                query_ddr_addr,
    ap_uint<64>                result_ddr_addr,
    ap_uint<1>                 reload_codebook,
    ap_uint<32>                top_k,
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<pq_entry_word_t> &pq_fifo,
    hls::stream<float>        &query_fifo,
    hls::stream<res_word_t>   &res_fifo_in,
    hls::stream<axis_result_pkt_t> &axis_result,
    ComputeMeta                &meta_out,
    volatile bool              &dm_done,
    volatile bool               dm_start,
    ap_uint<8>                *dram
);

void compute_engine(
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<pq_entry_word_t> &pq_fifo,
    hls::stream<res_word_t>   &res_fifo_out,
    hls::stream<float>        &query_fifo,
    ComputeMeta                 meta_in,
    ap_uint<32>                 top_k,
    ap_uint<1>                  reload_codebook,
    volatile bool              &comp_done,
    volatile bool               comp_start
);

#endif
