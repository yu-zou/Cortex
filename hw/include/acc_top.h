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
#define M_MAX      48
#define DIM_MAX    768
#define KS         256         // Hardcoded, default PQ Codebook centroid count
#define TOPK_MAX   500
#define TOPK_BANK_SIZE 16    // Cells per systolic bank for pipeline balancing

// ─── Synthesis parameters (smaller for faster iteration) ───
#define M_SYN      16
#define DIM_SYN    128
#define DS_SYN     (DIM_SYN / M_SYN)  // = 8

// ─── PQ Entry format ───
#define PQ_CODE_BYTES    M_SYN
#define PQ_META_BYTES    16
#define PQ_ENTRY_BYTES   (PQ_CODE_BYTES + PQ_META_BYTES)
#define PQ_ENTRY_BITS    (PQ_ENTRY_BYTES * 8)

// ─── Result: 128-bit ───
typedef ap_uint<128>  result_word_t;

// ─── Internal stream words ───
typedef ap_uint<512>  cb_pq_word_t;
typedef ap_uint<128>  res_word_t;

// ─── Metadata ───
struct ComputeMeta {
    ap_uint<32>  m_actual;
    ap_uint<32>  dim_actual;
    ap_uint<32>  n_vectors;       // N vectors
    ap_uint<32>  top_k;
    ap_uint<2>   metric_id;
};

// ─── Systolic cell ───
struct TopKCell {
    float        best_dist;
    ap_uint<64>  best_addr;
    ap_uint<32>  best_len;
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
    ap_uint<8>  *dram
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
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<cb_pq_word_t> &pq_fifo,
    hls::stream<float>        &query_fifo,
    hls::stream<res_word_t>   &res_fifo_in,
    ComputeMeta                &meta_out,
    volatile bool              &dm_done,
    volatile bool               dm_start,
    ap_uint<8>                *dram
);

void compute_engine(
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<cb_pq_word_t> &pq_fifo,
    hls::stream<res_word_t>   &res_fifo_out,
    hls::stream<float>        &query_fifo,
    ComputeMeta                 meta_in,
    ap_uint<1>                  reload_codebook,
    volatile bool              &comp_done,
    volatile bool               comp_start
);

#endif
