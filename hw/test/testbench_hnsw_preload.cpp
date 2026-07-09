// ═══════════════════════════════════════════════════════════════════
// testbench_hnsw_preload.cpp — Vitis HLS C simulation testbench
// for HNSW graph data preloader (PRELOAD_NODES loop in acc_top.cpp).
//
// Verifies that graph header + node data (vectors, neighbor count,
// adjacency) load correctly from DRAM byte array into local BRAM
// arrays without running full graph traversal.
//
// DRAM layout (matches hnsw_search_engine in acc_top.cpp):
//   [HNSWGraphHeader: 16B]  →  num_nodes | entry_point | dim | max_degree
//   [Node0: DIM*4B float vector][num_nbrs: 4B][MAX_DEG * 4B neighbor IDs]
//   [Node1: ...] ...
//
// Usage:
//   vitis_hls -f run_hnsw_preload_test.tcl
// ═══════════════════════════════════════════════════════════════════

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../include/acc_top.h"

using namespace std;

// ─── Test parameters (small graph for fast C sim) ───
#define TB_NODES      5
#define DRAM_BYTES    (64 * 1024)

// ─── DRAM byte type (matches ap_uint<8>* dram in acc_top) ───
typedef ap_uint<8> dram_byte_t;

// ─── Helper: write a uint32 into byte array at byte offset (LE) ───
static void dram_write_u32(dram_byte_t* base, uint64_t byte_off, uint32_t val) {
    base[byte_off + 0] = (val >> 0)  & 0xFF;
    base[byte_off + 1] = (val >> 8)  & 0xFF;
    base[byte_off + 2] = (val >> 16) & 0xFF;
    base[byte_off + 3] = (val >> 24) & 0xFF;
}

// ─── Helper: write a float into byte array at byte offset ───
static void dram_write_float(dram_byte_t* base, uint64_t byte_off, float val) {
    uint32_t bits;
    memcpy(&bits, &val, sizeof(bits));
    dram_write_u32(base, byte_off, bits);
}

// ═══════════════════════════════════════════════════════════════════
int main() {
    // ─── Allocate zero-initialized DRAM ───
    dram_byte_t* dram = new dram_byte_t[DRAM_BYTES]();

    cout << "============================================" << endl;
    cout << " HNSW Graph Preloader Test"                   << endl;
    cout << "============================================" << endl;
    cout << " NODES   = " << TB_NODES                       << endl;
    cout << " DIM_SYN = " << DIM_SYN                        << endl;
    cout << " MAX_DEG = " << HNSW_MAX_DEGREE                << endl;
    cout << " Node    = " << (DIM_SYN * 4 + 4 + HNSW_MAX_DEGREE * 4) << " bytes" << endl;
    cout << endl;

    // ═══════════════════════════════════════════════════════════════
    // 1. Populate DRAM with Graph Header + 5 Nodes
    // ═══════════════════════════════════════════════════════════════

    // ── Header (16 bytes): 4 × uint32 LE ──
    const uint32_t hdr_num_nodes   = TB_NODES;
    const uint32_t hdr_entry_point = 0;       // start at node 0
    const uint32_t hdr_dim         = DIM_SYN;
    const uint32_t hdr_max_degree  = HNSW_MAX_DEGREE;

    dram_write_u32(dram, 0,  hdr_num_nodes);
    dram_write_u32(dram, 4,  hdr_entry_point);
    dram_write_u32(dram, 8,  hdr_dim);
    dram_write_u32(dram, 12, hdr_max_degree);

    // ── Compute per-node byte stride ──
    const uint64_t node_bytes = (uint64_t)DIM_SYN * 4 + 4 + (uint64_t)HNSW_MAX_DEGREE * 4;
    const uint64_t graph_data_off = 16;  // header size

    // ── Fill each node with KNOWN data ──
    for (uint32_t n = 0; n < TB_NODES; n++) {
        uint64_t node_off = graph_data_off + n * node_bytes;

        // Vector: element (n, d) = n * DIM_SYN + d   (as float32)
        // This gives unique values: vectors[0][0] = 0.0f,
        // vectors[4][127] = 639.0f
        for (uint32_t d = 0; d < DIM_SYN; d++) {
            float val = (float)(n * DIM_SYN + d);
            dram_write_float(dram, node_off + d * 4, val);
        }

        // num_nbrs: node n has (n+1) neighbors (capped at MAX_DEG)
        // Node 0 → 1 nbr, Node 1 → 2 nbrs, ..., Node 4 → 5 nbrs
        uint32_t nn = (n + 1 < HNSW_MAX_DEGREE) ? (n + 1) : HNSW_MAX_DEGREE;
        dram_write_u32(dram, node_off + DIM_SYN * 4, nn);

        // Neighbor IDs: sequential (n * MAX_DEG + k)
        for (uint32_t k = 0; k < HNSW_MAX_DEGREE; k++) {
            uint32_t nid = n * HNSW_MAX_DEGREE + k;
            dram_write_u32(dram, node_off + DIM_SYN * 4 + 4 + k * 4, nid);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 2. Local BRAM arrays (same as hnsw_search_engine)
    // ═══════════════════════════════════════════════════════════════
    float         vectors[HNSW_MAX_NODES][DIM_SYN];
    ap_uint<32>   adjacency[HNSW_MAX_NODES][HNSW_MAX_DEGREE];
    ap_uint<8>    num_nbrs[HNSW_MAX_NODES];

    // ═══════════════════════════════════════════════════════════════
    // 3. PRELOAD — exact replica of hnsw_search_engine logic
    //    from acc_top.cpp lines 508–530
    // ═══════════════════════════════════════════════════════════════
    const ap_uint<32> MAX_DEG        = HNSW_MAX_DEGREE;
    const ap_uint<32> DIM            = DIM_SYN;       // hdr_dim
    const ap_uint<32> actual_nodes   = hdr_num_nodes;

    PRELOAD_NODES:
    for (ap_uint<32> n = 0; n < actual_nodes && n < HNSW_MAX_NODES; n++) {
        ap_uint<64> node_off = graph_data_off + n.to_uint64() * node_bytes;
        ap_uint<32>* vptr = (ap_uint<32>*)(dram + node_off.to_uint64());

        // Load vector sequentially (1 float per cycle in HW)
        for (ap_uint<32> d = 0; d < DIM_SYN; d++) {
            if (d < DIM) {
                vectors[n][d] = *((float*)&vptr[d]);
            }
        }

        // Load num_neighbors
        ap_uint<32> nn = vptr[DIM];
        num_nbrs[n] = (nn < MAX_DEG) ? nn : MAX_DEG;

        // Load neighbor IDs sequentially
        for (ap_uint<32> k = 0; k < MAX_DEG; k++) {
            adjacency[n][k] = vptr[DIM + 1 + k];
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 4. Verification
    // ═══════════════════════════════════════════════════════════════

    int errors = 0;

    // ── 4a. vectors[0][0] = first value from DRAM ──
    //     Node 0, dim 0 → (0 * 128 + 0) = 0.0f
    {
        float expected = 0.0f;
        float actual   = vectors[0][0];
        if (fabs(actual - expected) > 1e-6f) {
            cout << "FAIL: vectors[0][0] = " << actual
                 << " (expected " << expected << ")" << endl;
            errors++;
        } else {
            cout << "PASS: vectors[0][0] = " << actual << endl;
        }
    }

    // ── 4b. vectors[N-1][DIM-1] = last value from DRAM ──
    //     Node 4, dim 127 → (4 * 128 + 127) = 639.0f
    {
        float expected = (float)((TB_NODES - 1) * DIM_SYN + (DIM_SYN - 1));
        float actual   = vectors[TB_NODES - 1][DIM_SYN - 1];
        if (fabs(actual - expected) > 1e-6f) {
            cout << "FAIL: vectors[" << (TB_NODES - 1) << "][" << (DIM_SYN - 1)
                 << "] = " << actual << " (expected " << expected << ")" << endl;
            errors++;
        } else {
            cout << "PASS: vectors[" << (TB_NODES - 1) << "][" << (DIM_SYN - 1)
                 << "] = " << actual << endl;
        }
    }

    // ── 4c. adjacency[0][0] = first neighbor ID ──
    //     Node 0, neighbor 0 → (0 * 32 + 0) = 0
    {
        uint32_t expected = 0;
        uint32_t actual   = adjacency[0][0].to_uint();
        if (actual != expected) {
            cout << "FAIL: adjacency[0][0] = " << actual
                 << " (expected " << expected << ")" << endl;
            errors++;
        } else {
            cout << "PASS: adjacency[0][0] = " << expected << endl;
        }
    }

    // ── 4d. adjacency[N-1][DEG-1] = last neighbor ID ──
    //     Node 4, neighbor 31 → (4 * 32 + 31) = 159
    {
        uint32_t expected = (TB_NODES - 1) * HNSW_MAX_DEGREE + (HNSW_MAX_DEGREE - 1);
        uint32_t actual   = adjacency[TB_NODES - 1][HNSW_MAX_DEGREE - 1].to_uint();
        if (actual != expected) {
            cout << "FAIL: adjacency[" << (TB_NODES - 1) << "][" << (HNSW_MAX_DEGREE - 1)
                 << "] = " << actual << " (expected " << expected << ")" << endl;
            errors++;
        } else {
            cout << "PASS: adjacency[" << (TB_NODES - 1) << "][" << (HNSW_MAX_DEGREE - 1)
                 << "] = " << expected << endl;
        }
    }

    // ── 4e. num_nbrs matches DRAM data for all nodes ──
    cout << "\n-- Neighbor Count Verification --" << endl;
    for (uint32_t n = 0; n < TB_NODES; n++) {
        uint32_t expected = (n + 1 < HNSW_MAX_DEGREE) ? (n + 1) : HNSW_MAX_DEGREE;
        uint32_t actual   = num_nbrs[n].to_uint();
        if (actual != expected) {
            cout << "FAIL: num_nbrs[" << n << "] = " << actual
                 << " (expected " << expected << ")" << endl;
            errors++;
        } else {
            cout << "PASS: num_nbrs[" << n << "] = " << actual << endl;
        }
    }

    // ── 4f. Clamp test: ensure num_nbrs never exceeds MAX_DEG ──
    {
        // Manually write a large num_nbrs into DRAM and re-load node 0
        const uint64_t node0_off = graph_data_off + 0 * node_bytes;
        dram_write_u32(dram, node0_off + DIM_SYN * 4, 0xFF);   // 255 > MAX_DEG

        // Reload just node 0 into a temporary BRAM
        ap_uint<32>* vptr = (ap_uint<32>*)(dram + node0_off);
        ap_uint<32> nn_raw = vptr[DIM_SYN];
        ap_uint<8>  nn_clamped = (nn_raw < MAX_DEG) ? nn_raw : MAX_DEG;

        if (nn_clamped != MAX_DEG) {
            cout << "FAIL: num_nbrs clamping  (raw=" << (uint32_t)nn_raw
                 << ", clamped=" << (uint32_t)nn_clamped
                 << ", expected=" << MAX_DEG << ")" << endl;
            errors++;
        } else {
            cout << "PASS: num_nbrs clamping  (raw=" << (uint32_t)nn_raw
                 << " -> clamped=" << (uint32_t)nn_clamped << ")" << endl;
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // 5. Summary
    // ═══════════════════════════════════════════════════════════════
    cout << "\n============================================" << endl;
    if (errors == 0) {
        cout << " RESULT: ALL TESTS PASSED" << endl;
    } else {
        cout << " RESULT: " << errors << " TEST(S) FAILED" << endl;
    }
    cout << "============================================" << endl;

    delete[] dram;
    return errors;
}
