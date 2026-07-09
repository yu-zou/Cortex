// ═══════════════════════════════════════════════════════════════
// testbench_hnsw_full.cpp — Full HNSW Pipeline Vitis HLS Testbench
// Tests hnsw_search_engine end-to-end:
//   DRAM graph → BRAM preload → traversal → systolic Top-K → result output
//
// Graph: N=20 nodes, DIM=128, DEG=32, ring topology.
// Uses known seed (srand(42)) for reproducible random vectors/query.
// Verifies: non-FP32_MAX distances, ascending sort, valid node IDs.
// ═══════════════════════════════════════════════════════════════

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../include/acc_top.h"

using namespace std;

typedef ap_uint<8> dram_byte_t;

// ─── Test Parameters ───
#define DIM       128          // Vector dimension (matches DIM_SYN)
#define N         20           // Number of graph nodes
#define DEG       32           // Max degree (matches HNSW_MAX_DEGREE)
#define TOPK      10           // Number of results to verify
#define DRAM_SIZE (16 * 1024 * 1024)  // 16 MB DRAM

// ─── Helper: write float to DRAM byte array (little-endian) ───
static void write_float_le(dram_byte_t* buf, uint64_t offset, float val) {
    uint32_t raw;
    memcpy(&raw, &val, sizeof(raw));
    buf[offset + 0] = (raw >> 0)  & 0xFF;
    buf[offset + 1] = (raw >> 8)  & 0xFF;
    buf[offset + 2] = (raw >> 16) & 0xFF;
    buf[offset + 3] = (raw >> 24) & 0xFF;
}

// ─── Helper: write uint32 to DRAM byte array (little-endian) ───
static void write_u32_le(dram_byte_t* buf, uint64_t offset, uint32_t val) {
    buf[offset + 0] = (val >> 0)  & 0xFF;
    buf[offset + 1] = (val >> 8)  & 0xFF;
    buf[offset + 2] = (val >> 16) & 0xFF;
    buf[offset + 3] = (val >> 24) & 0xFF;
}

int main() {
    // Known seed for reproducibility
    srand(42);

    cout << "╔══════════════════════════════════════════════════════════════╗" << endl;
    cout << "║   HNSW Full Pipeline Test: DRAM→BRAM→Traversal→Top-K→Out  ║" << endl;
    cout << "╚══════════════════════════════════════════════════════════════╝" << endl;
    cout << "Parameters: DIM=" << DIM << " N=" << N
         << " DEG=" << DEG << " TOPK=" << TOPK << endl;

    // ═══════════════════════════════════════════════════════════════
    // Step 1: Build ring graph in host memory
    // Ring topology: node i connects to (i+1)%N and (i-1+N)%N
    // Simple, deterministic, guarantees full connectivity.
    // ═══════════════════════════════════════════════════════════════
    vector<vector<float>>       vecs(N, vector<float>(DIM));
    vector<vector<uint32_t>>    nbrs(N);

    cout << "\nGenerating ring graph (N=" << N << ") with seed=42..." << endl;

    // Generate random vectors
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t d = 0; d < DIM; d++) {
            vecs[i][d] = ((float)(rand() % 10000) / 100.0f) - 50.0f;
        }
    }

    // Build ring connectivity
    for (uint32_t i = 0; i < N; i++) {
        nbrs[i].push_back((i + 1) % N);
        nbrs[i].push_back((i - 1 + N) % N);
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 2: Write graph to DRAM (preload path — graph must come
    // from DRAM, NOT pre-populated BRAM)
    //
    // DRAM layout:
    //   [0..15]    HNSWGraphHeader  (num_nodes, entry_point, dim, max_degree)
    //   [16..]     Node data, each node = HNSW_NODE_SIZE bytes:
    //                [0..DIM*4-1]       vector (DIM floats)
    //                [DIM*4..DIM*4+3]   num_neighbors (uint32)
    //                [DIM*4+4..]        neighbor IDs (DEG × uint32)
    // ═══════════════════════════════════════════════════════════════
    dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();

    // Write HNSWGraphHeader at offset 0
    HNSWGraphHeader hdr;
    hdr.num_nodes   = N;
    hdr.entry_point = 0;       // Start at node 0
    hdr.dim         = DIM;
    hdr.max_degree  = DEG;

    memcpy(dram, &hdr, sizeof(HNSWGraphHeader));

    // Write each node's data
    const uint64_t graph_data_off = sizeof(HNSWGraphHeader);  // = 16

    for (uint32_t n = 0; n < N; n++) {
        uint64_t base = graph_data_off + n * HNSW_NODE_SIZE;

        // Write vector (DIM floats, little-endian)
        for (uint32_t d = 0; d < DIM; d++) {
            write_float_le(dram, base + d * 4, vecs[n][d]);
        }

        // Write num_neighbors (uint32)
        uint32_t nn = (uint32_t)nbrs[n].size();
        write_u32_le(dram, base + DIM * 4, nn);

        // Write neighbor IDs (HNSW_MAX_DEGREE entries × uint32)
        // Unused entries are zero-padded
        for (uint32_t k = 0; k < HNSW_MAX_DEGREE; k++) {
            uint32_t nid = (k < nbrs[n].size()) ? nbrs[n][k] : 0;
            write_u32_le(dram, base + DIM * 4 + 4 + k * 4, nid);
        }
    }

    cout << "Graph written to DRAM: " << N << " nodes, "
         << (graph_data_off + N * HNSW_NODE_SIZE) << " bytes used" << endl;

    // ═══════════════════════════════════════════════════════════════
    // Step 3: Create query vector
    // ═══════════════════════════════════════════════════════════════
    hls::stream<float> query_fifo;
    for (uint32_t d = 0; d < DIM; d++) {
        float q = ((float)(rand() % 10000) / 100.0f) - 50.0f;
        query_fifo.write(q);
    }
    cout << "\nQuery vector (" << DIM << " floats) written to query_fifo" << endl;

    // ═══════════════════════════════════════════════════════════════
    // Step 4: Set up metadata
    // ═══════════════════════════════════════════════════════════════
    ComputeMeta meta;
    meta.m_actual    = M_SYN;
    meta.dim_actual  = DIM;
    meta.n_vectors   = N;
    meta.top_k       = TOPK;
    meta.metric_id   = 0;       // L2 distance
    meta.search_mode = 1;       // HNSW graph search

    // ═══════════════════════════════════════════════════════════════
    // Step 5: Run HNSW Search Engine
    // ═══════════════════════════════════════════════════════════════
    hls::stream<res_word_t> res_fifo;
    volatile bool comp_done = false;
    volatile bool comp_start = true;

    cout << "\n─── Launching hnsw_search_engine (full pipeline) ───" << endl;
    hnsw_search_engine(query_fifo, res_fifo, meta, comp_done, comp_start, dram);
    cout << "Engine returned. comp_done = " << comp_done << endl;

    // Verify comp_done
    if (!comp_done) {
        cerr << "FAIL: hnsw_search_engine did not set comp_done = true" << endl;
        delete[] dram;
        return 1;
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 6: Read results from res_fifo
    // The engine writes TOPK_MAX (500) results. We read all of them
    // but verify only the first TOPK=10 as real results.
    // ═══════════════════════════════════════════════════════════════
    struct Result {
        float      dist;
        uint32_t   node_id;
    };

    vector<Result> results;
    results.reserve(TOPK_MAX);

    for (int i = 0; i < TOPK_MAX; i++) {
        if (res_fifo.empty()) {
            cerr << "FAIL: res_fifo empty at result index " << i
                 << " (expected " << TOPK_MAX << " results)" << endl;
            delete[] dram;
            return 1;
        }
        res_word_t r = res_fifo.read();

        // Parse: dist[31:0] | node_id[95:32] | unused[127:96]
        ap_uint<32> dist_bits = r.range(31, 0);
        float dist;
        memcpy(&dist, &dist_bits, sizeof(dist));
        uint32_t node_id = r.range(95, 32).to_uint();
        results.push_back({dist, node_id});
    }

    // Print first 10 results
    cout << "\n─── First " << TOPK << " Results from Systolic Top-K ───" << endl;
    for (int i = 0; i < TOPK; i++) {
        printf("  [%2d] node=%2u  dist=%.6f\n",
               i, results[i].node_id, results[i].dist);
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 7: Verify results
    // ═══════════════════════════════════════════════════════════════
    const float FP32_MAX = 3.402823466e+38f;
    int nfail = 0;

    // Verification 1: first TOPK results are non-FP32_MAX
    // (they contain real computed distances, not initial sentinel values)
    for (int i = 0; i < TOPK; i++) {
        if (results[i].dist >= FP32_MAX) {
            cerr << "FAIL [" << i << "]: dist=FP32_MAX (uninitialized result, "
                 << "expected a real computed distance)" << endl;
            nfail++;
        }
    }

    // Verification 2: results are sorted in ascending distance order
    for (int i = 1; i < TOPK_MAX; i++) {
        // Once we hit FP32_MAX sentinels, further entries are uninitialized;
        // skip check for those.
        if (results[i].dist >= FP32_MAX) break;
        if (results[i].dist < results[i - 1].dist) {
            cerr << "FAIL [" << i << "]: not sorted (dist[" << (i - 1)
                 << "]=" << results[i - 1].dist
                 << " > dist[" << i << "]=" << results[i].dist << ")" << endl;
            nfail++;
        }
    }

    // Verification 3: node IDs are valid (0 <= id < N)
    for (int i = 0; i < TOPK_MAX; i++) {
        if (results[i].dist >= FP32_MAX) break;  // remaining are sentinels
        if (results[i].node_id >= N) {
            cerr << "FAIL [" << i << "]: node_id=" << results[i].node_id
                 << " out of valid range [0, " << (N - 1) << "]" << endl;
            nfail++;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Step 8: Result summary
    // ═══════════════════════════════════════════════════════════════
    cout << "\n─── Verification Summary ───" << endl;
    if (nfail == 0) {
        cout << "PASS: All checks passed" << endl;
        cout << "  ✓ Non-FP32_MAX: first " << TOPK << " results are real distances" << endl;
        cout << "  ✓ Sorted: results in ascending distance order" << endl;
        cout << "  ✓ Node IDs: all in range [0, " << (N - 1) << "]" << endl;
        cout << "  ✓ comp_done: correctly set to true" << endl;
    } else {
        cout << "FAIL: " << nfail << " check(s) failed" << endl;
    }

    delete[] dram;
    return (nfail == 0) ? 0 : 1;
}
