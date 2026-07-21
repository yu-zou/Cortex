#include "../include/acc_top.h"

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
    hls::stream<axis_result_pkt_t> &axis_result,
    ComputeMeta                &meta_out,
    volatile bool              &dm_done,
    volatile bool               dm_start,
    ap_uint<8>                *dram
) {
#pragma HLS INTERFACE m_axi port=dram depth=1048576 offset=direct
#pragma HLS INTERFACE ap_ctrl_hs port=return
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

    // Step 3: Compute sizes (uniform KS=256)
    ap_uint<32> cb_bytes = M_val * KS * Ds_val * 4;
    ap_uint<32> cb_beats = (cb_bytes + 63) / 64;
    ap_uint<32> pq_bytes = N_val * (M_val + 16);
    ap_uint<32> pq_beats = (pq_bytes + 63) / 64;

    meta_out.m_actual   = M_val;
    meta_out.dim_actual = DIM_val;
    meta_out.n_vectors  = N_val;
    meta_out.top_k      = TOPK_MAX;
    meta_out.metric_id  = 0;
    // meta_out.search_mode removed — compute_engine is the only execution path

    // ─── Pre-compute span/base for each sub-quantizer (serial, one-time) ───
    ap_uint<32> span[M_SYN], base_arr[M_SYN];
#pragma HLS ARRAY_PARTITION variable=span complete dim=1
#pragma HLS ARRAY_PARTITION variable=base_arr complete dim=1
    ap_uint<32> accum = 0;
    for (int m = 0; m < M_SYN; m++) {
#pragma HLS PIPELINE II=1
        if (m < M_val) {
            ap_uint<32> ks = KS;
            span[m] = ks * Ds_val;
            base_arr[m] = accum;
            accum += span[m];
        } else {
            span[m] = 0;
            base_arr[m] = 0;
        }
    }

    // Step 4: Stream Codebook → FIFO_CB
    // Compute [m][ks_idx][ds_idx] address in DM (serial 1 divider), pack into FIFO
    // Format per float: m(5b)|ks_idx(8b)|ds_idx(4b)|val(32b) = 49b, 10 entries/beat
    ap_uint<64> cb_addr = cs + 64;
    ap_uint<32>* cb_ptr = (ap_uint<32>*)(dram + cb_addr.to_uint64());
    if (reload_codebook) {
        ap_uint<512> cb_beat = 0;
        ap_uint<5> entry_cnt = 0;
    CB_SERIAL:
        for (ap_uint<32> gidx = 0; gidx < cb_bytes / 4; gidx++) {
#pragma HLS PIPELINE II=1
            ap_uint<32> raw = cb_ptr[gidx];
            // Serial: find m, compute ks_idx = rem/Ds_val, ds_idx = rem%Ds_val
            ap_uint<5> m_out = 0;
            ap_uint<8> k_out = 0;
            ap_uint<4> d_out = 0;
        FIND_M:
            for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
                if (m < M_val && gidx >= base_arr[m] && gidx < base_arr[m] + span[m]) {
                    ap_uint<32> rem_val = gidx - base_arr[m];
                    m_out = m;
                    k_out = rem_val / Ds_val;
                    d_out = rem_val % Ds_val;
                }
            }
            // Pack: m[4:0] | ks_idx[7:0] | ds_idx[3:0] | val[31:0] → 49 bit
            ap_uint<49> entry = ((ap_uint<49>)m_out << 44) |
                                ((ap_uint<49>)k_out << 36) |
                                ((ap_uint<49>)d_out << 32) |
                                raw;
            ap_uint<10> pos = entry_cnt * 49;
            cb_beat.range(pos + 48, pos) = entry;
            entry_cnt++;
            if (entry_cnt == 10) {
                cb_fifo.write(cb_beat);
                cb_beat = 0;
                entry_cnt = 0;
            }
        }
        if (entry_cnt > 0) {
            cb_fifo.write(cb_beat);
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
        // DM pre-extracts doc_addr/doc_len to save CE LUT
        uint64_t da = 0;
        for (int bi = 0; bi < 8; bi++) {
            da |= ((uint64_t)beat.range(511 - (M_val + bi)*8, 511 - (M_val + bi)*8 - 7)) << (bi * 8);
        }
        uint32_t dl = 0;
        for (int bi = 0; bi < 4; bi++) {
            dl |= ((uint32_t)beat.range(511 - (M_val + 8 + bi)*8, 511 - (M_val + 8 + bi)*8 - 7)) << (bi * 8);
        }
        beat.range(95, 0) = ((ap_uint<96>)da << 32) | dl;
        pq_fifo.write(beat);
    }

    // Step 6: Write Results (Compute Engine → res_fifo_in → DRAM)
    // Hardware path only — in CSIM, testbench reads fifo_res directly since
    // data_manager and compute_engine execute sequentially and fifo_res is
    // empty when data_manager runs first.
    // Results pushed by compute_engine Stage 5, written by DM to DRAM for ARM.
    // Format: TOPK_MAX × 128-bit [dist(32b) | doc_addr(64b) | doc_len(32b)]
#ifdef __SYNTHESIS__
    ap_uint<32>* res_ptr = (ap_uint<32>*)(dram + result_ddr_addr.to_uint64());
#endif
RESULT_WRITE:
    for (ap_uint<32> i = 0; i < TOPK_MAX; i++) {
#pragma HLS PIPELINE II=1
        res_word_t entry = res_fifo_in.read();
#ifdef __SYNTHESIS__
        res_ptr[i * 4 + 0] = entry.range(31, 0);     // dist (FP32)
        res_ptr[i * 4 + 1] = entry.range(63, 32);     // doc_addr low 32b
        res_ptr[i * 4 + 2] = entry.range(95, 64);     // doc_addr high 32b
        res_ptr[i * 4 + 3] = entry.range(127, 96);    // doc_len
#endif
        // Write to AXI4-Stream (parallel to DDR4 write)
        axis_result_pkt_t axis_pkt;
        axis_pkt.data = entry;
        axis_pkt.last = (i == TOPK_MAX - 1) ? 1 : 0;
        axis_pkt.keep = -1;
        axis_result.write(axis_pkt);
    }

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
#pragma HLS INTERFACE ap_ctrl_hs port=return
    comp_done = false;
    if (!comp_start) return;

    ap_uint<32> M_val   = meta_in.m_actual;
    ap_uint<32> DIM_val = meta_in.dim_actual;
    ap_uint<32> N_val   = meta_in.n_vectors;
    ap_uint<32> Ds_val  = DIM_val / M_val;

    // Codebook BRAM (KS=256 uniform)
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

    // Stage 0: Load Codebook — pre-computed addresses from DM, direct BRAM write
    ap_uint<32> cb_total = M_val * KS * Ds_val;
    ap_uint<32> cb_beats = (cb_total * 4 + 63) / 64;

CB_LOAD:
    if (reload_codebook) {
        // Zero initialize BRAM
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

        // Direct write: each 512b beat holds 10 entries of (m,k,d,val)
        for (ap_uint<32> b = 0; b < cb_beats; b++) {
#pragma HLS PIPELINE II=1
            cb_pq_word_t cb_beat = cb_fifo.read();
            for (int e = 0; e < 10; e++) {
#pragma HLS UNROLL
                ap_uint<10> pos = e * 49;
                ap_uint<5>  m_out = cb_beat.range(pos + 48, pos + 44);
                ap_uint<8>  k_out = cb_beat.range(pos + 43, pos + 36);
                ap_uint<4>  d_out = cb_beat.range(pos + 35, pos + 32);
                ap_uint<32> val   = cb_beat.range(pos + 31, pos);
                if (m_out < M_val) {
                    codebook[m_out][k_out][d_out] = *((float*)&val);
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

        // DM pre-extracted doc_addr/doc_len → read directly from low 96 bits
        ap_uint<64> doc_addr = beat.range(95, 32);
        ap_uint<32> doc_len  = beat.range(31, 0);

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
                ap_uint<8> c = pq_codes[m];  // 8-bit uniform PQ
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
    ap_uint<8>  *dram,
    hls::stream<axis_result_pkt_t> &axis_result
) {
#pragma HLS INTERFACE axis port=axis_result
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
                 axis_result, meta, dm_done, dm_start, dram);

    // Step 2: compute_engine is the only execution path
    comp_start = true;
    compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                   meta, reload_codebook, comp_done, comp_start);

    done = dm_done && comp_done;
#else
    // ─── Synthesis: DATAFLOW ───
    control_fsm(cluster_start_addr, query_ddr_addr,
                top_k, metric_id, reload_codebook,
                done, start, dm_start, dm_done,
                comp_start, comp_done);

#pragma HLS DATAFLOW disable_start_propagation
    {
        data_manager(cluster_start_addr, query_ddr_addr, result_ddr_addr,
                     reload_codebook,
                     fifo_cb, fifo_pq, fifo_qry, fifo_res,
                     axis_result, meta, dm_done, dm_start, dram);

        compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                       meta, reload_codebook, comp_done, comp_start);
    }
#endif
}


// HNSW search engine removed — compute_engine is the only execution path
