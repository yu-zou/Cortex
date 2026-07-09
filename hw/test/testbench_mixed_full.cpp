#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../include/acc_top.h"

using namespace std;
typedef ap_uint<8> dram_byte_t;

// ─── Small parameters for fast simulation ───
#define TB_M          4
#define TB_DIM        32
#define TB_DS         (TB_DIM / TB_M)   // = 8
#define TB_KS         256
#define TB_TOP_PQ     5
#define TB_N_PQ       10
#define TB_NODES_GRAPH 10
#define TB_DEG        4
#define TB_TOP_GR     5
#define DRAM_SIZE     (16 * 1024 * 1024)

// ─── Separate DRAM regions ───
#define CLUSTER_BASE  0x10000   // IVFHeader + codebook + PQ codes
#define QUERY_BASE    0x00000   // Query vector for IVFPQ
#define GRAPH_BASE    0x80000   // HNSWGraphHeader + graph nodes

// ─── Helper: write float to DRAM byte array ───
static void write_float(dram_byte_t* b, uint32_t o, float v) {
    uint32_t r; memcpy(&r,&v,4);
    b[o+0]= r      &0xFF;
    b[o+1]=(r>>8 ) &0xFF;
    b[o+2]=(r>>16) &0xFF;
    b[o+3]=(r>>24) &0xFF;
}

// ─── Helper: write uint32 to DRAM byte array ───
static void write_u32(dram_byte_t* b, uint32_t o, uint32_t v) {
    b[o+0]= v      &0xFF;
    b[o+1]=(v>>8 ) &0xFF;
    b[o+2]=(v>>16) &0xFF;
    b[o+3]=(v>>24) &0xFF;
}

// ─── Helper: write uint64 to DRAM byte array ───
static void write_u64(dram_byte_t* b, uint32_t o, uint64_t v) {
    for(int i=0;i<8;i++) b[o+i]=(v>>(i*8))&0xFF;
}

/**
 * Mixed-mode full pipeline testbench.
 * Tests BOTH search_mode=0 (IVFPQ) and search_mode=1 (HNSW) in a single run.
 * Both modes share the same systolic_topk_insert via their respective engines.
 */
int main() {
    cout << "=== Mixed-Mode (IVFPQ+HNSW) Full Pipeline Test ===" << endl;
    bool global_pass = true;

    // ═══════════════════════════════════════════════════════
    // TEST MODE 0: IVFPQ
    // ═══════════════════════════════════════════════════════
    cout << "\n--- MODE 0: IVFPQ PQ Search (M=" << TB_M
         << " DIM=" << TB_DIM << " N=" << TB_N_PQ
         << " TopK=" << TB_TOP_PQ << ") ---" << endl;
    {
        dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();

        // ── IVFHeader at CLUSTER_BASE (64B) ──
        write_u32(dram, CLUSTER_BASE + 0, TB_M);           // codebook_dim
        write_u32(dram, CLUSTER_BASE + 4, TB_DIM);         // pq_dim
        write_u32(dram, CLUSTER_BASE + 8, TB_N_PQ);        // pq_vectors
        // reserved[13] at +0x0C..+0x3F — already zeroed by DRAM init

        // ── Codebook: M * KS * DS floats ──
        uint32_t cbf = TB_M * TB_KS * TB_DS;   // total floats
        uint32_t cbb = cbf * 4;                 // byte size
        float* cbr  = new float[cbf];
        srand(42);
        for(uint32_t i=0;i<cbf;i++){
            cbr[i] = ((float)(rand()%1000)/100.0f) - 5.0f;
            write_float(dram, CLUSTER_BASE + 64 + i*4, cbr[i]);
        }

        // ── PQ codes (each entry: M code-bytes + 16 metadata bytes) ──
        uint32_t pqs = TB_M + 16;   // stride per PQ entry (PQ_ENTRY_BYTES)
        for(uint32_t n=0;n<TB_N_PQ;n++){
            uint32_t o = CLUSTER_BASE + 64 + cbb + n * pqs;
            for(uint32_t m=0;m<TB_M;m++) dram[o+m] = rand() % 256;
            write_u64(dram, o+TB_M,    ((uint64_t)rand()<<32)|rand());
            write_u64(dram, o+TB_M+8,  (uint64_t)(rand()%(1<<20)));
        }

        // ── Query vector at QUERY_BASE ──
        srand(999);
        for(uint32_t d=0;d<TB_DIM;d++)
            write_float(dram, QUERY_BASE + d*4,
                        ((float)(rand()%1000)/100.0f)-5.0f);

        // ── Run IVFPQ pipeline: data_manager → compute_engine ──
        hls::stream<cb_pq_word_t> fc, fp;
        hls::stream<float>        fq;
        hls::stream<res_word_t>   fr;
        ComputeMeta meta;
        meta.search_mode = 0;
        meta.top_k       = TB_TOP_PQ;
        volatile bool dm_done=false, comp_done=false;
        volatile bool dm_start=true,  comp_start=true;

        data_manager(CLUSTER_BASE, QUERY_BASE,
                     fc, fp, fr, fq, meta,
                     dm_done, dm_start, dram);
        compute_engine(fc, fp, fr, fq, meta,
                       comp_done, comp_start);

        // ── Read and verify IVFPQ results ──
        int valid = 0;
        float prev_dist = -1e38f;
        for(int i=0;i<TB_TOP_PQ;i++){
            res_word_t r = fr.read();
            ap_uint<32> db = r.range(31, 0);
            float d;
            memcpy(&d, &db, 4);

            if(d < 1e37f) {
                valid++;
                // Verify sorted ascending
                if(d < prev_dist - 0.001f) {
                    cout << "  IVFPQ WARNING: result[" << i
                         << "] dist=" << d << " < prev=" << prev_dist
                         << " (not sorted)" << endl;
                }
                prev_dist = d;
            }
        }
        cout << "  IVFPQ: " << valid << "/" << TB_TOP_PQ
             << " valid results (non-FP32_MAX)" << endl;
        if(valid < TB_TOP_PQ) {
            cout << "  >> IVFPQ FAIL" << endl;
            global_pass = false;
        } else {
            cout << "  >> IVFPQ PASS" << endl;
        }

        delete[] dram;
        delete[] cbr;
    }

    // ═══════════════════════════════════════════════════════
    // TEST MODE 1: HNSW Graph Search
    // ═══════════════════════════════════════════════════════
    cout << "\n--- MODE 1: HNSW Graph Search (N=" << TB_NODES_GRAPH
         << " DIM=" << TB_DIM << " TopK=" << TB_TOP_GR
         << " Degree=" << TB_DEG << ") ---" << endl;
    {
        dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();

        // ── HNSWGraphHeader at GRAPH_BASE (16 bytes) ──
        ap_uint<32>* hdr = (ap_uint<32>*)(dram + GRAPH_BASE);
        hdr[0] = TB_NODES_GRAPH;   // num_nodes
        hdr[1] = 0;                // entry_point (start from node 0)
        hdr[2] = TB_DIM;           // vector dimension
        hdr[3] = TB_DEG;           // max_degree

        // ── Graph nodes: vector(DS*4B) + num_nbrs(1B) + nbr_ids(TB_DEG*4B) ──
        // Stride matches how hnsw_search_engine walks DRAM from header
        srand(12345);
        uint64_t nbytes = TB_DIM * 4 + 4 + TB_DEG * 4;
        for(uint32_t n=0; n<TB_NODES_GRAPH; n++){
            uint64_t o = GRAPH_BASE + 16 + n * nbytes;

            // Vector data (DIM floats)
            for(uint32_t d=0; d<TB_DIM; d++)
                write_float(dram, o + d*4,
                            ((float)(rand()%1000)/100.0f)-5.0f);

            // Number of neighbors (1 byte)
            uint32_t nn = min((uint32_t)TB_DEG, (uint32_t)TB_DEG);
            dram[o + TB_DIM*4] = nn & 0xFF;

            // Neighbor IDs (padded to TB_DEG)
            for(uint32_t k=0; k<TB_DEG; k++){
                uint32_t nid = (k<nn) ? ((n + k + 1) % TB_NODES_GRAPH) : 0;
                write_u32(dram, o + TB_DIM*4 + 4 + k*4, nid);
            }
        }

        // ── Query vector: streamed directly (not via DRAM) ──
        hls::stream<float>      fq;
        hls::stream<res_word_t> fr;
        srand(67890);
        for(uint32_t d=0; d<TB_DIM; d++)
            fq.write(((float)(rand()%1000)/100.0f)-5.0f);

        // ── Invoke HNSW search engine ──
        ComputeMeta meta;
        meta.n_vectors   = TB_NODES_GRAPH;
        meta.dim_actual  = TB_DIM;
        meta.top_k       = TB_TOP_GR;
        meta.search_mode = 1;
        meta.metric_id   = 0;
        volatile bool comp_done=false, comp_start=true;

        // Pass dram+GRAPH_BASE so engine reads header from graph region
        hnsw_search_engine(fq, fr, meta,
                           comp_done, comp_start,
                           dram + GRAPH_BASE);

        // ── Read and verify HNSW results ──
        int valid = 0;
        float prev_dist = -1e38f;
        for(int i=0;i<TB_TOP_GR;i++){
            res_word_t r = fr.read();
            ap_uint<32> db = r.range(31, 0);
            float d;
            memcpy(&d, &db, 4);

            if(d < 1e37f) {
                valid++;
                if(d < prev_dist - 0.001f) {
                    cout << "  HNSW WARNING: result[" << i
                         << "] dist=" << d << " < prev=" << prev_dist
                         << " (not sorted)" << endl;
                }
                prev_dist = d;
            }
        }
        cout << "  HNSW: " << valid << "/" << TB_TOP_GR
             << " valid results (non-FP32_MAX)" << endl;
        if(valid < TB_TOP_GR) {
            cout << "  >> HNSW FAIL" << endl;
            global_pass = false;
        } else {
            cout << "  >> HNSW PASS" << endl;
        }

        delete[] dram;
    }

    // ═══════════════════════════════════════════════════════
    // VERDICT
    // ═══════════════════════════════════════════════════════
    cout << "\n========================================" << endl;
    if(global_pass) {
        cout << "MIXED-MODE FULL PIPELINE: ALL MODES PASS" << endl;
        return 0;
    } else {
        cout << "MIXED-MODE FULL PIPELINE: SOME MODES FAILED" << endl;
        return 1;
    }
}
