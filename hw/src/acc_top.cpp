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
    ap_uint<32>                top_k,
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
    // 含义2: M/DIM fixed to synth config (M_SYN/DIM_SYN); Header fields are validation-only
    (void)M_val;
    (void)DIM_val;

    // Step 2: Stream Query from DDR4 → query_fifo
    ap_uint<32>* qptr = (ap_uint<32>*)(dram + qa.to_uint64());
QRY_READ:
    for (ap_uint<32> i = 0; i < DIM_SYN; i++) {
#pragma HLS PIPELINE II=1
        ap_uint<32> raw = qptr[i];
        query_fifo.write(*((float*)&raw));
    }

    // Step 3: Compute sizes (uniform KS=256)
    ap_uint<32> cb_bytes = M_SYN * KS * DS_SYN * 4;
    ap_uint<32> cb_beats = (cb_bytes + 63) / 64;
    ap_uint<32> pq_bytes = N_val * ENTRY_BYTES_SYN;
    ap_uint<32> pq_beats = (pq_bytes + 63) / 64;

    meta_out.m_actual   = M_SYN;
    meta_out.dim_actual = DIM_SYN;
    meta_out.n_vectors  = N_val;
    meta_out.top_k      = top_k;
    // meta_out.metric_id set by acc_top before calling data_manager
    // meta_out.search_mode removed — compute_engine is the only execution path

    // Step 4: Stream Codebook → FIFO_CB
    // Use 512-bit aligned burst reads. Each beat = 16 FP32 floats.
    // Codebook layout: [m=0][k=0..255][d=0..Ds-1], [m=1][k=0..255][d=0..Ds-1], ...
    // POWER-OF-2 optimization: per_m = 256*8=2048, Ds_val=8
    //   m   = gidx / per_m  →  gidx >> 11
    //   rem = gidx % per_m  →  gidx & 0x7FF
    //   k   = rem / Ds_val  →  rem >> 3
    //   d   = rem % Ds_val  →  rem & 0x7
    // Step 4: Stream Codebook → FIFO_CB (raw 16 FP32 floats per beat, sequential)
    ap_uint<64> cb_addr = cs + 64;
    ap_uint<512>* cb_beats_ptr = (ap_uint<512>*)(dram + cb_addr.to_uint64());
    if (reload_codebook) {
    CB_STREAM:
        for (ap_uint<32> b = 0; b < cb_beats; b++) {
#pragma HLS PIPELINE II=1
            cb_fifo.write(cb_beats_ptr[b]);
        }
    }

    // Step 5: Stream PQ Codes → FIFO_PQ (512-bit aligned burst reads)
    ap_uint<64> pq_addr = cs + 64 + cb_bytes;
    ap_uint<32> entry_bytes = ENTRY_BYTES_SYN;
    ap_uint<512>* pq_beats_ptr = (ap_uint<512>*)(dram + pq_addr.to_uint64());
PQ_STREAM:
    for (ap_uint<32> n = 0; n < N_val; n++) {
#pragma HLS PIPELINE II=1
        // Each PQ entry occupies exactly 1 beat (M_val+16 ≤ 64 bytes).
        // Entries stored sequentially at 64-byte boundaries in DRAM.
        cb_pq_word_t beat = 0;
        beat = pq_beats_ptr[n];
        
        // Pre-extract doc_addr/doc_len from beat's known byte positions
        // PQ entry layout (LE byte order at HIGH bits of beat):
        //   bytes 0..M_val-1: PQ codes
        //   bytes M_val..M_val+7: doc_addr (8 bytes, LE)
        //   bytes M_val+8..M_val+11: doc_len (4 bytes low, LE)
        // doc_addr: 8 bytes starting at entry byte M_val
        // entry byte M_val+bi → beat bits [511-(M_val+bi)*8 : 511-(M_val+bi)*8-7]
        uint64_t da = 0;
        for (int bi = 0; bi < 8; bi++) {
            da |= ((uint64_t)beat.range(511 - (M_SYN + bi)*8,
                                        511 - (M_SYN + bi)*8 - 7)) << (bi * 8);
        }
        // doc_len: 4 bytes starting at entry byte M_val+8
        // entry byte M_val+8+bi → beat bits [511-(M_val+8+bi)*8 : 511-(M_val+8+bi)*8-7]
        uint32_t dl = 0;
        for (int bi = 0; bi < 4; bi++) {
            dl |= ((uint32_t)beat.range(511 - (M_SYN + 8 + bi)*8,
                                        511 - (M_SYN + 8 + bi)*8 - 7)) << (bi * 8);
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
    ap_uint<128>* res128_ptr = (ap_uint<128>*)(dram + result_ddr_addr.to_uint64());
    (void)res128_ptr;  // Used in RESULT_WRITE loop
#endif
    ap_uint<32> push_count = (top_k > 0 && top_k <= TOPK_MAX) ? top_k : (ap_uint<32>)TOPK_MAX;
RESULT_WRITE:
    for (ap_uint<32> i = 0; i < push_count; i++) {
#pragma HLS PIPELINE II=1
        res_word_t entry = res_fifo_in.read();
#ifdef __SYNTHESIS__
        res128_ptr[i] = entry;
#endif
        // Write to AXI4-Stream (parallel to DDR4 write)
        axis_result_pkt_t axis_pkt;
        axis_pkt.data = entry;
        axis_pkt.last = (i == push_count - 1) ? 1 : 0;
        axis_pkt.keep = -1;
        axis_result.write(axis_pkt);
    }

    dm_done = true;
}


// ═══════════════════════════════════════════════════════════════
// COMPUTE ENGINE — DATAFLOW-decoupled ADC + exact Top-K
// adc_engine: codebook/query load + ADC distance computation
// topk_engine: exact 500-cell integer-key systolic insertion sort
// ═══════════════════════════════════════════════════════════════

float sub_dist_l2(float q, float c) {
#pragma HLS INLINE
    float d = q - c;
    return d * d;
}

typedef ap_uint<128> topk_candidate_word_t;

static topk_candidate_word_t pack_topk_candidate(
    ap_uint<32> dist_key,
    ap_uint<64> doc_addr,
    ap_uint<32> doc_len
) {
#pragma HLS INLINE
    topk_candidate_word_t word = 0;
    word.range(31,  0)  = dist_key;
    word.range(95,  32) = doc_addr;
    word.range(127, 96) = doc_len;
    return word;
}

void adc_engine(
    hls::stream<cb_pq_word_t>    &cb_fifo,
    hls::stream<cb_pq_word_t>    &pq_fifo,
    hls::stream<float>           &query_fifo,
    hls::stream<topk_candidate_word_t> &cand_out,
    ComputeMeta                   meta_in,
    ap_uint<1>                    reload_codebook
) {
#pragma HLS INLINE off
    ap_uint<32> N_val   = meta_in.n_vectors;

    // Codebook BRAM (KS=256 uniform)
    static float codebook[M_SYN][KS][DS_SYN];
#pragma HLS RESOURCE variable=codebook core=RAM_2P_BRAM
#pragma HLS ARRAY_PARTITION variable=codebook complete dim=1
#pragma HLS ARRAY_PARTITION variable=codebook complete dim=3

    static float dist_table[M_SYN][KS];
#pragma HLS ARRAY_PARTITION variable=dist_table complete dim=1

    // Query registers
    float query[DIM_SYN];
#pragma HLS ARRAY_PARTITION variable=query complete dim=1

QRY_LOAD:
    for (ap_uint<32> i = 0; i < DIM_SYN; i++) {
#pragma HLS PIPELINE II=1
        query[i] = query_fifo.read();
    }

    // Load Codebook — sequential float stream, compute address from counter.
    ap_uint<32> cb_total = M_SYN * KS * DS_SYN;
    ap_uint<32> cb_beats = (cb_total * 4 + 63) / 64;

CB_LOAD:
    if (reload_codebook) {
        ap_uint<32> counter = 0;
        for (ap_uint<32> b = 0; b < cb_beats; b++) {
#pragma HLS PIPELINE II=1
            cb_pq_word_t beat = cb_fifo.read();
            for (int f = 0; f < 16; f++) {
#pragma HLS UNROLL
                ap_uint<32> gidx = counter + f;
                if (gidx < cb_total) {
                    ap_uint<32> raw = beat.range(32*f+31, 32*f);
                    float val = *((float*)&raw);
                    // M_SYN=24, DS_SYN=32: per_m = KS*DS_SYN = 8192.
                    ap_uint<6>  m_addr = gidx >> 13;
                    ap_uint<32> rem    = gidx & 0x1FFF;
                    ap_uint<8>  k_addr = rem >> 5;
                    ap_uint<5>  d_addr = rem & 0x1F;
                    if (m_addr < M_SYN) {
                        codebook[m_addr][k_addr][d_addr] = val;
                    }
                }
            }
            counter += 16;
        }
    }

DIST_TABLE_BUILD:
    for (int c = 0; c < KS; c++) {
        for (int mg = 0; mg < M_SYN; mg += 8) {
#pragma HLS PIPELINE II=1
            for (int i = 0; i < 8; i++) {
#pragma HLS UNROLL
                int m = mg + i;
                float acc = 0.0f;
                for (int d = 0; d < DS_SYN; d += 16) {
#pragma HLS UNROLL factor=16
                    acc += sub_dist_l2(query[m * DS_SYN + d], codebook[m][c][d]);
                }
                dist_table[m][c] = acc;
            }
        }
    }

PQ_PROCESS:
    for (ap_uint<32> n = 0; n < N_val; n++) {
#pragma HLS PIPELINE II=1
        cb_pq_word_t beat = pq_fifo.read();
        ap_uint<512> raw  = beat;

        // Right-align fixed synthesis entry width: M_SYN PQ bytes + 16 metadata bytes.
        ap_uint<512> shifted = raw >> (512 - ENTRY_BITS_SYN);

        // DM pre-extracted doc_addr/doc_len → read directly from low 96 bits.
        ap_uint<64> doc_addr = beat.range(95, 32);
        ap_uint<32> doc_len  = beat.range(31, 0);

        // PQ codes: compile-time bit positions avoid runtime variable part-selects.
        ap_uint<8> pq_codes[M_SYN];
#pragma HLS ARRAY_PARTITION variable=pq_codes complete dim=1
        for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
            pq_codes[m] = shifted.range(ENTRY_BITS_SYN - 1 - m*8,
                                        ENTRY_BITS_SYN - 8 - m*8);
        }

        float part[M_SYN];
#pragma HLS ARRAY_PARTITION variable=part complete dim=1
    DIST_LOOKUP:
        for (int m = 0; m < M_SYN; m++) {
#pragma HLS UNROLL
            part[m] = dist_table[m][pq_codes[m]];
        }

        float l1[12];
#pragma HLS ARRAY_PARTITION variable=l1 complete dim=1
    DIST_SUM_L1:
        for (int i = 0; i < 12; i++) {
#pragma HLS UNROLL
            l1[i] = part[2*i] + part[2*i + 1];
        }

        float l2[6];
#pragma HLS ARRAY_PARTITION variable=l2 complete dim=1
    DIST_SUM_L2:
        for (int i = 0; i < 6; i++) {
#pragma HLS UNROLL
            l2[i] = l1[2*i] + l1[2*i + 1];
        }

        float l3[3];
#pragma HLS ARRAY_PARTITION variable=l3 complete dim=1
    DIST_SUM_L3:
        for (int i = 0; i < 3; i++) {
#pragma HLS UNROLL
            l3[i] = l2[2*i] + l2[2*i + 1];
        }

        float l4[2];
#pragma HLS ARRAY_PARTITION variable=l4 complete dim=1
        l4[0] = l3[0] + l3[1];
        l4[1] = l3[2];
        float total_dist = l4[0] + l4[1];

        ap_uint<32> dist_key = *((ap_uint<32>*)&total_dist);
        cand_out.write(pack_topk_candidate(dist_key, doc_addr, doc_len));
    }
}

void topk_engine(
    hls::stream<topk_candidate_word_t> &cand_in,
    hls::stream<res_word_t>             &res_out,
    ap_uint<32>                          n_total,
    ap_uint<32>                          top_k
) {
#pragma HLS INLINE off
    topk_candidate_word_t cells[TOPK_MAX];
#pragma HLS ARRAY_PARTITION variable=cells complete dim=1

TOPK_INIT:
    for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS PIPELINE II=1
        cells[i] = pack_topk_candidate(TOPK_INF_KEY, 0, 0);
    }

TOPK_PROCESS:
    for (ap_uint<32> n = 0; n < n_total; n++) {
#pragma HLS PIPELINE II=1
        topk_candidate_word_t cand = cand_in.read();
    TOPK_CHAIN:
        for (int i = 0; i < TOPK_MAX; i++) {
#pragma HLS UNROLL
            if (cand.range(31, 0) < cells[i].range(31, 0)) {
                topk_candidate_word_t tmp = cells[i];
                cells[i] = cand;
                cand = tmp;
            }
        }
    }

    ap_uint<32> push_count = (top_k > 0 && top_k <= TOPK_MAX) ? top_k : (ap_uint<32>)TOPK_MAX;
RESULT_PUSH:
    for (ap_uint<32> i = 0; i < push_count; i++) {
#pragma HLS PIPELINE II=1
        res_out.write(cells[i]);
    }
}

void compute_engine(
    hls::stream<cb_pq_word_t> &cb_fifo,
    hls::stream<cb_pq_word_t> &pq_fifo,
    hls::stream<res_word_t>   &res_fifo_out,
    hls::stream<float>        &query_fifo,
    ComputeMeta                 meta_in,
    ap_uint<32>                 top_k,
    ap_uint<1>                  reload_codebook,
    volatile bool              &comp_done,
    volatile bool               comp_start
) {
#pragma HLS INTERFACE ap_ctrl_hs port=return
    comp_done = false;
    if (!comp_start) return;

    static hls::stream<topk_candidate_word_t> cand_stream("cand_stream");
#pragma HLS STREAM variable=cand_stream depth=64

#pragma HLS DATAFLOW
    adc_engine(cb_fifo, pq_fifo, query_fifo, cand_stream, meta_in, reload_codebook);
    topk_engine(cand_stream, res_fifo_out, meta_in.n_vectors, top_k);

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
    // ComputeMeta carries runtime parameters from Host (via AXI4-Lite) and
    // DRAM Header to the Compute Engine:
    //   m_actual/dim_actual/n_vectors → read from DDR4 Header (L35-L37)
    //   top_k → from AXI4-Lite register (L468)
    //   metric_id → from AXI4-Lite register (L469)
    meta.metric_id = metric_id;

#ifndef __SYNTHESIS__
    // ─── C Simulation: sequential execution ───
    // Step 1: Data Manager (INPUT: DRAM→FIFOs, OUTPUT: res_fifo_in→DRAM)
    dm_start = true;
    data_manager(cluster_start_addr, query_ddr_addr, result_ddr_addr, reload_codebook, top_k,
                 fifo_cb, fifo_pq, fifo_qry, fifo_res,
                 axis_result, meta, dm_done, dm_start, dram);

    // Step 2: compute_engine is the only execution path
    comp_start = true;
    compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                   meta, top_k, reload_codebook, comp_done, comp_start);

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
                     reload_codebook, top_k,
                     fifo_cb, fifo_pq, fifo_qry, fifo_res,
                     axis_result, meta, dm_done, dm_start, dram);

        compute_engine(fifo_cb, fifo_pq, fifo_res, fifo_qry,
                       meta, top_k, reload_codebook, comp_done, comp_start);
    }
#endif
}


// HNSW search engine removed — compute_engine is the only execution path
