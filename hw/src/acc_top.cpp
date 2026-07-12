#include "../include/acc_top.h"

// NU_PQ: per-sub-quantizer bit width array (from nu_bits.bin analysis).
// KS_nu[m] = 2^bits[m], e.g. 256/16/4 for 8/4/2-bit.
static const ap_uint<4> pq_bits[M_SYN] = {
    8, 8, 8, 8,
    4, 4, 4, 4,
    2, 2, 2, 2,
    4, 4, 4, 4
};

static ap_uint<10> pq_ks_from_bits(ap_uint<4> bits) {
#pragma HLS INLINE
    return ((ap_uint<10>)1) << bits;
}

// ═══════════════════════════════════════════════════════════════
// DATA MANAGER — Bidirectional DRAM interface
// INPUT:  DRAM(AXI4-MM read) → cb_fifo, pq_fifo, query_fifo
// OUTPUT: res_fifo_in ← Compute Engine → DRAM(AXI4-MM write) for ARM to read
// DRAM layout: [Header 64B][Codebook M*256*Ds*4B][PQ Codes N*(M+16)B]
// Result layout: [Results TOPK_MAX×128b] at result_ddr_addr
// ═══════════════════════════════════════════════════════════════
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
) {
#pragma HLS INTERFACE m_axi port=dram depth=1048576 offset=direct
#pragma HLS INTERFACE ap_ctrl_none port=return
    dm_done = false;
    if (!dm_start) return;

    ap_uint<64> cs = cluster_start_addr;
    ap_uint<64> qa = query_ddr_addr;

    // Step 1: Read 64B Header
    ap_uint<32>* hdr_ptr = (ap_uint<32>*)(dram + cs.to_uint64());
    ap_uint<32> M_val   = hdr_ptr[0];
    ap_uint<32> DIM_val = hdr_ptr[1];
    ap_uint<32> N_val   = hdr_ptr[2];
    ap_uint<32> Ds_val  = DIM_val / M_val;

    // Step 2: Stream Query from DDR4 → query_fifo
    ap_uint<32>* qptr = (ap_uint<32>*)(dram + qa.to_uint64());
QRY_READ:
    for (ap_uint<32> i = 0; i < DIM_val; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<32> raw = qptr[i];
        query_fifo.write(*((float*)&raw));
    }

    // Step 3: Compute sizes
    ap_uint<10> ks_per_m[M_SYN];
#pragma HLS ARRAY_PARTITION variable=ks_per_m complete dim=1
    ap_uint<32> cb_total = 0;
DM_KS_INIT:
    for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
        ks_per_m[m] = pq_ks_from_bits(pq_bits[m]);
        if (m < M_val) {
            cb_total += ks_per_m[m] * Ds_val;
        }
    }

    ap_uint<32> cb_bytes = cb_total * 4;
    ap_uint<32> cb_beats = (cb_bytes + 63) / 64;
    ap_uint<32> pq_bytes = N_val * (M_val + 16);
    ap_uint<32> pq_beats = (pq_bytes + 63) / 64;

    meta_out.m_actual   = M_val;
    meta_out.dim_actual = DIM_val;
    meta_out.n_vectors  = N_val;
    meta_out.top_k      = TOPK_MAX;
    meta_out.metric_id  = 0;
    meta_out.search_mode = 0;   // default IVFPQ mode

    // Step 4: Stream Codebook → FIFO_CB (burst-optimized: full-beat reads)
    ap_uint<64> cb_addr = cs + 64;
    ap_uint<512>* cb_beats_ptr = (ap_uint<512>*)(dram + cb_addr.to_uint64());
    if (reload_codebook) {
CB_STREAM:
        for (ap_uint<32> b = 0; b < cb_beats; b++) {
#pragma HLS PIPELINE II=1
            cb_pq_word_t beat = cb_beats_ptr[b];
            cb_fifo.write(beat);
        }
    }

    // Step 5: Stream PQ Codes → FIFO_PQ
    // Byte-level reads preserve LE byte order for compute_engine's right-align logic
    ap_uint<64> pq_addr = cs + 64 + cb_bytes;
    ap_uint<32> entry_bytes = M_val + 16;
PQ_STREAM:
    for (ap_uint<32> n = 0; n < N_val; n++) {
#pragma HLS PIPELINE II=1
        cb_pq_word_t beat = 0;
        ap_uint<64> entry_off = n.to_uint64() * entry_bytes;
        for (int by = 0; by < 64 && by < entry_bytes; by++) {
#pragma HLS UNROLL
            ap_uint<64> src_byte = entry_off + by;
            beat.range(511 - by*8, 511 - by*8 - 7) = dram[pq_addr.to_uint64() + src_byte];
        }
        pq_fifo.write(beat);
    }

    // Step 6: Write Results (Compute Engine → res_fifo_in → DRAM)
    // Hardware path only — in CSIM, testbench reads fifo_res directly since
    // data_manager and compute_engine execute sequentially and fifo_res is
    // empty when data_manager runs first.
#ifdef __SYNTHESIS__
    // Results pushed by compute_engine Stage 5, written by DM to DRAM for ARM.
    // Format: TOPK_MAX × 128-bit [dist(32b) | doc_addr(64b) | doc_len(32b)]
    ap_uint<32>* res_ptr = (ap_uint<32>*)(dram + result_ddr_addr.to_uint64());
RESULT_WRITE:
    for (ap_uint<32> i = 0; i < TOPK_MAX; i++) {
#pragma HLS PIPELINE II=1
        res_word_t entry = res_fifo_in.read();
        res_ptr[i * 4 + 0] = entry.range(31, 0);     // dist (FP32)
        res_ptr[i * 4 + 1] = entry.range(63, 32);     // doc_addr low 32b
        res_ptr[i * 4 + 2] = entry.range(95, 64);     // doc_addr high 32b
        res_ptr[i * 4 + 3] = entry.range(127, 96);    // doc_len
    }
#endif

    dm_done = true;
}


// ═══════════════════════════════════════════════════════════════
// COMPUTE ENGINE — Six-stage pipeline
// Stage 0: Codebook load (when reload_codebook=1)
// Stage 1: Top-K initialization (FP32_MAX)
// Stage 2: PQ vector unpack (1 beat/vec, 512bit, data at high bits, low bits zero-padded)
// Stage 3: ADC L2 distance computation (M sub-quantizers fully unrolled in parallel)
// Stage 4: Systolic Top-K streaming sort
// Stage 5: Result push → FIFO_RES (128bit: dist+doc_addr+doc_length)
// ═══════════════════════════════════════════════════════════════

float sub_dist_l2(float q, float c) {
#pragma HLS INLINE
    float d = q - c;
    return d * d;
}

// Systolic Top-K: fully unrolled chain of TOPK_MAX independent cells.
// Uses integer comparison on FP32 bit patterns for non-negative L2 distances.
// This saves ~100 LUTs per cell vs float comparison.
void systolic_topk_insert(
    float        cand_dist,
    ap_uint<64>  cand_addr,
    ap_uint<32>  cand_len,
    TopKCell     cells[TOPK_MAX]
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=cells complete dim=1

    float        cd_float = cand_dist;
    ap_uint<32>  cd_int   = *((ap_uint<32>*)&cand_dist);
    ap_uint<64>  ca = cand_addr;
    ap_uint<32>  cl = cand_len;

SYSTOLIC_CHAIN:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS UNROLL
        ap_uint<32> cell_int = *((ap_uint<32>*)&cells[i].best_dist);
        if (cd_int < cell_int) {
            float        td = cells[i].best_dist;
            ap_uint<64>  ta = cells[i].best_addr;
            ap_uint<32>  tl = cells[i].best_len;
            cells[i].best_dist = cd_float;
            cells[i].best_addr = ca;
            cells[i].best_len  = cl;
            cd_float = td;
            cd_int   = *((ap_uint<32>*)&td);
            ca = ta;
            cl = tl;
        }
    }
}

void compute_engine(
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<cb_pq_word_t> &pq_fifo,
    hls::stream<res_word_t>   &res_fifo_out,
    hls::stream<float>        &query_fifo,
    ComputeMeta                 meta_in,
    ap_uint<1>                  reload_codebook,
    volatile bool              &comp_done,
    volatile bool               comp_start
) {
#pragma HLS INTERFACE ap_ctrl_none port=return
    comp_done = false;
    if (!comp_start) return;

    ap_uint<32> M_val   = meta_in.m_actual;
    ap_uint<32> DIM_val = meta_in.dim_actual;
    ap_uint<32> N_val   = meta_in.n_vectors;
    ap_uint<32> Ds_val  = DIM_val / M_val;

    // NU_PQ: compute active centroid count for each sub-quantizer.
    ap_uint<10> ks_per_m[M_SYN];
#pragma HLS ARRAY_PARTITION variable=ks_per_m complete dim=1
KS_INIT:
    for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
        ks_per_m[m] = pq_ks_from_bits(pq_bits[m]);
    }

    // Codebook BRAM. KS remains the max allocation; only entries
    // [0, ks_per_m[m]) are valid for sub-quantizer m.
    static float codebook[M_SYN][KS][DS_SYN];
#pragma HLS RESOURCE variable=codebook core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=codebook complete dim=1
#pragma HLS ARRAY_PARTITION variable=codebook complete dim=3

    // Query BRAM
    float query[DIM_SYN];
#pragma HLS ARRAY_PARTITION variable=query complete dim=1

    // Load Query
QRY_LOAD:
    for (ap_uint<32> i = 0; i < DIM_val; i++) {
#pragma HLS PIPELINE II=1
        query[i] = query_fifo.read();
    }

    // Stage 0: Load Codebook
    ap_uint<32> cb_total = 0;
CB_TOTAL:
    for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
        if (m < M_val) {
            cb_total += ks_per_m[m] * Ds_val;
        }
    }
    ap_uint<32> cb_beats = (cb_total * 4 + 63) / 64;

CB_LOAD:
    if (reload_codebook) {
    CB_ZERO_M:
        for (int m = 0; m < M_SYN; m++) {
        CB_ZERO_K:
            for (int k = 0; k < KS; k++) {
            CB_ZERO_D:
                for (int d = 0; d < DS_SYN; d++) {
#pragma HLS PIPELINE II=1
                    codebook[m][k][d] = 0.0f;
                }
            }
        }

        for (ap_uint<32> b = 0; b < cb_beats; b++) {
#pragma HLS PIPELINE II=1
            cb_pq_word_t beat = cb_fifo.read();
            for (int f = 0; f < 16; f++) {
#pragma HLS UNROLL
                ap_uint<32> gidx = b * 16 + f;
                if (gidx < cb_total) {
                    ap_uint<32> raw   = beat.range(32*f+31, 32*f);
                    float        val  = *((float*)&raw);
                    ap_uint<32> base = 0;
                CB_MAP_M:
                    for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
                        ap_uint<32> span = ks_per_m[m] * Ds_val;
                        if (m < M_val && gidx >= base && gidx < base + span) {
                            ap_uint<32> rem    = gidx - base;
                            ap_uint<32> ks_idx = rem / Ds_val;
                            ap_uint<32> ds_idx = rem % Ds_val;
                            codebook[m][ks_idx][ds_idx] = val;
                        }
                        if (m < M_val) {
                            base += span;
                        }
                    }
                }
            }
        }
    }



    // Stage 1: Top-K Init
    TopKCell cells[TOPK_MAX];
#pragma HLS ARRAY_PARTITION variable=cells complete dim=1
TK_INIT:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS UNROLL
        cells[i].best_dist = 3.402823466e+38f;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // Stage 2-4: Main Processing
    // 1 beat = 1 vector (padded to 512 bits)
    ap_uint<32> entry_bytes = M_val + 16;

PQ_PROCESS:
    for (ap_uint<32> n = 0; n < N_val; n++) {
#pragma HLS PIPELINE II=1
        cb_pq_word_t beat = pq_fifo.read();
        ap_uint<512> raw  = beat;

        // Right-align: data at HIGH bits, zeros at LOW bits
        ap_uint<512> shifted = raw >> (512 - entry_bytes * 8);

        // Extract fields from right-aligned entry.
        // DM packs bytes from HIGH to LOW bits; reconstruct LE values.
        ap_uint<32> entry_bits = entry_bytes * 8;

        // doc_addr: M_val bytes after PQ codes (bytes M_val..M_val+7)
        ap_uint<64> doc_addr = 0;
        for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
            doc_addr.range(i*8+7, i*8) = shifted.range(
                entry_bits - M_val*8 - 1 - i*8,
                entry_bits - M_val*8 - 8 - i*8);
        }

        // doc_len: 8 bytes after doc_addr, use lower 32 bits (bytes M+8..M+11)
        ap_uint<32> doc_len = 0;
        for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
            doc_len.range(i*8+7, i*8) = shifted.range(
                entry_bits - M_val*8 - 64 - 1 - i*8,
                entry_bits - M_val*8 - 64 - 8 - i*8);
        }

        // PQ codes: bytes 0..M_SYN-1 of entry (constant bound for unrolling)
        ap_uint<8> pq_codes[M_SYN];
#pragma HLS ARRAY_PARTITION variable=pq_codes complete dim=1
        for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
            if (m < M_val) {
                pq_codes[m] = shifted.range(entry_bits - 1 - m*8,
                                            entry_bits - 8 - m*8);
            } else {
                pq_codes[m] = 0;
            }
        }

        // Stage 3: ADC (L2)
        float total_dist = 0.0f;
    ADC_M:
        for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
            if (m < M_val) {
                ap_uint<8> c_raw = pq_codes[m];
                ap_uint<9> mask_wide = (((ap_uint<9>)1) << pq_bits[m]) - 1;
                ap_uint<8> mask = mask_wide.range(7, 0);
                ap_uint<8> c = c_raw & mask;
                float sub = 0.0f;
            ADC_D:
                for (int d = 0; d < DS_SYN; d++) {
#pragma HLS UNROLL
                    if (d < Ds_val) {
                        float cent  = codebook[m][c][d];
                        float q_sub = query[m * Ds_val + d];
                        sub += sub_dist_l2(q_sub, cent);
                    }
                }
                total_dist += sub;
            }
        }

        // Stage 4: Systolic Top-K
        systolic_topk_insert(total_dist, doc_addr, doc_len, cells);
    }

    // Stage 5: Push Results → FIFO_RES
RESULT_PUSH:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS PIPELINE II=1
        res_word_t result = 0;
        ap_uint<32> dist_bits = *((ap_uint<32>*)&cells[i].best_dist);
        result.range(31,  0)  = dist_bits;
        result.range(95,  32) = cells[i].best_addr;
        result.range(127, 96) = cells[i].best_len;
        res_fifo_out.write(result);
    }

    comp_done = true;
}


// ═══════════════════════════════════════════════════════════════
// CONTROL FSM
// ═══════════════════════════════════════════════════════════════
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
) {
#pragma HLS INTERFACE s_axilite port=cluster_start_addr
#pragma HLS INTERFACE s_axilite port=query_ddr_addr
#pragma HLS INTERFACE s_axilite port=top_k
#pragma HLS INTERFACE s_axilite port=metric_id
#pragma HLS INTERFACE s_axilite port=reload_codebook
#pragma HLS INTERFACE s_axilite port=return bundle=control

    dm_start   = false;
    comp_start = false;
    done       = false;
    if (!start) return;

    // Phase 1: Launch DM
    dm_start = true;
#ifndef __SYNTHESIS__
    // C sim: return, caller handles sequencing
    return;
#endif
DM_WAIT:
    while (!dm_done) {}
    dm_start = false;

    // Phase 2: Launch Compute
    comp_start = true;
COMP_WAIT:
    while (!comp_done) {}
    comp_start = false;

    done = true;
}


// ═══════════════════════════════════════════════════════════════
// ACC_TOP — FPGA Top-Level Module
// Three-module decoupled architecture: Control(FSM) + Data Manager(AXI4-MM) + Compute(datapath)
// Inter-module communication: FIFO_CB(512b×256) + FIFO_PQ(512b×64) + FIFO_RES(128b×TOPK_MAX)
// Query path: Host DMA→DDR4→Data Manager AXI4-MM→query_fifo→Compute BRAM
// ═══════════════════════════════════════════════════════════════
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
) {
#pragma HLS INTERFACE s_axilite port=cluster_start_addr
#pragma HLS INTERFACE s_axilite port=query_ddr_addr
#pragma HLS INTERFACE s_axilite port=result_ddr_addr
#pragma HLS INTERFACE s_axilite port=top_k
#pragma HLS INTERFACE s_axilite port=metric_id
#pragma HLS INTERFACE s_axilite port=reload_codebook
#pragma HLS INTERFACE s_axilite port=done
#pragma HLS INTERFACE s_axilite port=start
#pragma HLS INTERFACE m_axi port=dram depth=1048576 offset=direct
#pragma HLS INTERFACE s_axilite port=return bundle=control

    done = false;
    if (!start) return;

    // Internal FIFOs
    static hls::stream<cb_pq_word_t> fifo_cb("fifo_cb");
    static hls::stream<cb_pq_word_t> fifo_pq("fifo_pq");
    static hls::stream<res_word_t>   fifo_res("fifo_res");
    static hls::stream<float>        fifo_qry("fifo_qry");
#pragma HLS STREAM variable=fifo_cb  depth=256
#pragma HLS STREAM variable=fifo_pq  depth=64
#pragma HLS STREAM variable=fifo_res depth=500
#pragma HLS STREAM variable=fifo_qry depth=256

    volatile bool dm_start   = false;
    volatile bool dm_done    = false;
    volatile bool comp_start = false;
    volatile bool comp_done  = false;

    ComputeMeta meta;

#ifndef __SYNTHESIS__
    // ─── C Simulation: sequential execution ───
    // Step 1: Data Manager (INPUT: DRAM→FIFOs, OUTPUT: res_fifo_in→DRAM)
    dm_start = true;
    data_manager(cluster_start_addr, query_ddr_addr, result_ddr_addr, reload_codebook,
                 fifo_cb, fifo_pq, fifo_qry, fifo_res,
                 meta, dm_done, dm_start, dram);

    // Step 2: Route to appropriate engine
    comp_start = true;
    if (meta.search_mode == 1) {
        // HNSW graph search mode: uses DRAM directly + query_fifo
        // (fifo_cb/fifo_pq not used in HNSW mode)
        hnsw_search_engine(fifo_qry, fifo_res, meta,
                          comp_done, comp_start, dram);
    } else {
        // Default: IVFPQ PQ search mode
        compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                       meta, reload_codebook, comp_done, comp_start);
    }

    done = dm_done && comp_done;
#else
    // ─── Synthesis: DATAFLOW ───
// Sequential execution (DATAFLOW pragma removed for synthesis compatibility)
    control_fsm(cluster_start_addr, query_ddr_addr,
                top_k, metric_id, reload_codebook,
                done, start, dm_start, dm_done,
                comp_start, comp_done);

        data_manager(cluster_start_addr, query_ddr_addr, result_ddr_addr,
                     reload_codebook,
                     fifo_cb, fifo_pq, fifo_qry, fifo_res,
                     meta, dm_done, dm_start, dram);

    if (meta.search_mode == 1) {
        hnsw_search_engine(fifo_qry, fifo_res, meta,
                          comp_done, comp_start, dram);
    } else {
        compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                       meta, reload_codebook, comp_done, comp_start);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════
// HNSW GRAPH SEARCH ENGINE (BRAM-based, synthesizable)
// ═══════════════════════════════════════════════════════════════
//
// Implements HNSW graph traversal on FPGA, inspired by SmartANNS
// (Tian et al., USENIX ATC 2024).
//
// Key design: pre-load graph data (vectors + adjacency lists) into
// partitioned BRAM. Traversal reads BRAM only — no dynamic DRAM
// access during search. Neighbors processed sequentially at II=1.
// Reuses systolic Top-K and L2 distance from IVFPQ engine.
//
// DRAM layout (for initial load):
//   [HNSWGraphHeader (16B)] [Node0: DIM floats | num_nbrs(u32) | nbr_ids[MAX_DEG*u32]] ...

void hnsw_search_engine(
    hls::stream<float>        &query_fifo,
    hls::stream<res_word_t>   &res_fifo_out,
    ComputeMeta                 meta_in,
    volatile bool              &comp_done,
    volatile bool               comp_start,
    ap_uint<8>                *dram
) {
#pragma HLS INTERFACE m_axi port=dram depth=1048576 offset=direct
#pragma HLS INTERFACE ap_ctrl_none port=return
    comp_done = false;
    if (!comp_start) return;

    ap_uint<32> num_nodes  = meta_in.n_vectors;
    ap_uint<32> DIM_val    = meta_in.dim_actual;
    ap_uint<32> top_k      = meta_in.top_k;
    const ap_uint<32> MAX_DEG = HNSW_MAX_DEGREE;

    // ─── Read Graph Header ───
    HNSWGraphHeader ghdr;
    ap_uint<32>* hdr_ptr = (ap_uint<32>*)dram;
    ghdr.num_nodes   = hdr_ptr[0];
    ghdr.entry_point = hdr_ptr[1];
    ghdr.dim         = hdr_ptr[2];
    ghdr.max_degree  = hdr_ptr[3];
    ap_uint<32> entry_node = ghdr.entry_point;
    ap_uint<64> graph_data_off = 16;

    // ─── BRAM: Vector Store [MAX_NODES][DIM] ───
    // Partitioned on dim=2 (complete) = DIM independent BRAMs for parallel reads
    float vectors[HNSW_MAX_NODES][DIM_SYN];
#pragma HLS RESOURCE variable=vectors core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=vectors complete dim=2

    // ─── BRAM: Adjacency List [MAX_NODES][MAX_DEG] ───
    ap_uint<32> adjacency[HNSW_MAX_NODES][HNSW_MAX_DEGREE];
#pragma HLS RESOURCE variable=adjacency core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=adjacency cyclic factor=8 dim=1

    // ─── BRAM: Num Neighbors per Node ───
    ap_uint<8> num_nbrs[HNSW_MAX_NODES];
#pragma HLS RESOURCE variable=num_nbrs core=RAM_1P_BRAM

    // ─── Pre-load Graph from DRAM → BRAM ───
    // Sequential per-node load: 1 float/cycle (pipelined), no UNROLL on DRAM reads
    ap_uint<32> node_bytes = DIM_val * 4 + 4 + MAX_DEG * 4;
PRELOAD_NODES:
    for (ap_uint<32> n = 0; n < num_nodes && n < HNSW_MAX_NODES; n++) {
#pragma HLS PIPELINE II=1
        ap_uint<64> node_off = graph_data_off + n.to_uint64() * node_bytes;
        ap_uint<32>* vptr = (ap_uint<32>*)(dram + node_off.to_uint64());

        // Load vector sequentially (1 float per cycle)
        for (ap_uint<32> d = 0; d < DIM_SYN; d++) {
            if (d < DIM_val) {
                vectors[n][d] = *((float*)&vptr[d]);
            }
        }

        // Load num_neighbors
        ap_uint<32> nn = vptr[DIM_val];
        num_nbrs[n] = (nn < MAX_DEG) ? nn : MAX_DEG;

        // Load neighbor IDs sequentially
        for (ap_uint<32> k = 0; k < MAX_DEG; k++) {
            adjacency[n][k] = vptr[DIM_val + 1 + k];
        }
    }

    // ─── Load Query ───
    float query[DIM_SYN];
#pragma HLS ARRAY_PARTITION variable=query complete dim=1
QRY_LOAD_HNSW:
    for (ap_uint<32> i = 0; i < DIM_val; i++) {
#pragma HLS PIPELINE II=1
        query[i] = query_fifo.read();
    }

    // ─── Visited Bitmap ───
    // Changed from RAM_1P_BRAM to complete partition (register array)
    // Eliminates the RAW hazard that forced SEQUENTIAL_TRAVERSE II=64
    ap_uint<1> visited[HNSW_MAX_NODES];
#pragma HLS ARRAY_PARTITION variable=visited complete dim=1
VISIT_INIT:
    for (ap_uint<32> i = 0; i < HNSW_MAX_NODES; i++) {
#pragma HLS PIPELINE II=1
        visited[i] = 0;
    }

    // ─── Systolic Top-K ───
    TopKCell cells[TOPK_MAX];
#pragma HLS ARRAY_PARTITION variable=cells complete dim=1
TK_INIT_HNSW:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS UNROLL
        cells[i].best_dist = 3.402823466e+38f;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // Seed: mark entry as visited (distance computed after loop)
    visited[entry_node] = 1;

    // ═══ Sequential Graph Traversal (pipelined, not unrolled) ═══
    // visited[] is now a register array (complete partition) - zero-cycle access.
    // NBR_LOOP runs one neighbor per cycle (PIPELINE II=1, factor=1 prevents auto-unroll).
    // vectors[] dim=2 is already complete-partitioned = 128 independent BRAM banks.

SEQUENTIAL_TRAVERSE:
    for (ap_uint<32> iter = 0; iter < num_nodes && iter < HNSW_MAX_NODES; iter++) {
#pragma HLS LOOP_TRIPCOUNT min=20 max=200
        ap_uint<8> nn = num_nbrs[iter];

        // ─── Process neighbors one per cycle (pipelined, not unrolled) ───
    NBR_LOOP:
        for (ap_uint<8> ni = 0; ni < MAX_DEG; ni++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=1
            if (ni < nn) {
                ap_uint<32> nbr_id = adjacency[iter][ni];

                if (nbr_id < num_nodes && visited[nbr_id] == 0) {
                    visited[nbr_id] = 1;

                    // ─── L2 Distance (pipelined: 16 groups × 8 dims) ───
                    float nbr_dist = 0.0f;
                NBR_DIST_GROUPS:
                    for (ap_uint<32> g = 0; g < DIM_SYN; g += 8) {
#pragma HLS UNROLL
                        float group_sum = 0.0f;
                    NBR_DIST:
                        for (ap_uint<32> d = g; d < g + 8; d++) {
#pragma HLS UNROLL
                            if (d < DIM_val) {
                                float diff = query[d] - vectors[nbr_id][d];
                                group_sum += diff * diff;
                            }
                        }
                        nbr_dist += group_sum;
                    }

                    // Insert into systolic Top-K
                    systolic_topk_insert(nbr_dist, nbr_id.to_uint64(), 0, cells);
                }
            }
        }
    }

    // Also process the entry node itself
    visited[entry_node] = 1;
    {
        float entry_dist = 0.0f;
        for (ap_uint<32> d = 0; d < DIM_SYN; d++) {
#pragma HLS UNROLL
            if (d < DIM_val) {
                float diff = query[d] - vectors[entry_node][d];
                entry_dist += diff * diff;
            }
        }
        systolic_topk_insert(entry_dist, entry_node.to_uint64(), 0, cells);
    }

    // ─── Push Results ───
RESULT_PUSH_HNSW:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS PIPELINE II=1
        res_word_t result = 0;
        ap_uint<32> dist_bits = *((ap_uint<32>*)&cells[i].best_dist);
        result.range(31,  0)  = dist_bits;
        result.range(95,  32) = cells[i].best_addr;
        result.range(127, 96) = cells[i].best_len;
        res_fifo_out.write(result);
    }

    comp_done = true;
}
