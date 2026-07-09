#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../include/acc_top.h"

using namespace std;

// ─────────────────────────────────────────────────────────────
// Replica of systolic_topk_insert from acc_top.cpp
// (original is file-static, not exposed via header)
// ─────────────────────────────────────────────────────────────
static void systolic_topk_insert(
    float        cand_dist,
    ap_uint<64>  cand_addr,
    ap_uint<32>  cand_len,
    TopKCell     cells[TOPK_MAX]
) {
    float        cd_float = cand_dist;
    ap_uint<32>  cd_int   = *((ap_uint<32>*)&cand_dist);
    ap_uint<64>  ca = cand_addr;
    ap_uint<32>  cl = cand_len;

    for (int i = 0; i < TOPK_MAX; i++) {
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

// ─────────────────────────────────────────────────────────────
// Reference L2-squared distance
// ─────────────────────────────────────────────────────────────
static float l2_sq(const float* query, const float* vec, int dim) {
    float dist = 0.0f;
    for (int d = 0; d < dim; d++) {
        float diff = query[d] - vec[d];
        dist += diff * diff;
    }
    return dist;
}

// ─────────────────────────────────────────────────────────────
// Main: BRAM-direct HNSW sequential traversal test
// ─────────────────────────────────────────────────────────────
int main() {
    const int N       = 8;            // graph nodes
    const int DIM     = DIM_SYN;      // 128 (from acc_top.h)
    const int DEG     = 8;            // max degree (≤ HNSW_MAX_DEGREE)
    const int ENTRY   = 0;            // entry point

    cout << "=== HNSW Sequential Traverse Test ===" << endl;
    cout << "N=" << N << " DIM=" << DIM << " DEG=" << DEG
         << " TOPK_MAX=" << TOPK_MAX << endl;

    // ─── BRAM arrays (directly populated, no DRAM) ───
    float       vectors[N][DIM];
    ap_uint<32> adjacency[N][DEG];
    ap_uint<8>  num_nbrs[N];
    ap_uint<1>  visited[N];
    TopKCell    cells[TOPK_MAX];
    float       query[DIM];

    // ─── Init vectors with deterministic known values ───
    // vectors[n][d] = n*100 + d*0.1  →  strictly ordered by n
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < DIM; d++) {
            vectors[n][d] = n * 100.0f + d * 0.1f;
        }
    }

    // ─── Init adjacency: ring connections ───
    // Node i → (i+1)%N, (i-1+N)%N  (guarantees full graph connectivity)
    for (int i = 0; i < N; i++) {
        adjacency[i][0] = (i + 1) % N;
        adjacency[i][1] = (i + N - 1) % N;
        for (int k = 2; k < DEG; k++) {
            adjacency[i][k] = 0;   // unused, filtered by num_nbrs
        }
        num_nbrs[i] = 2;           // each node has exactly 2 ring neighbors
    }

    // ─── Init query at origin ───
    for (int d = 0; d < DIM; d++) {
        query[d] = 0.0f;
    }

    // ─── Init visited bitmap = 0 ───
    for (int i = 0; i < N; i++) {
        visited[i] = 0;
    }

    // ─── Init systolic Top-K = FP32_MAX ───
    for (int i = 0; i < TOPK_MAX; i++) {
        cells[i].best_dist = 3.402823466e+38f;
        cells[i].best_addr = 0;
        cells[i].best_len  = 0;
    }

    // ═══════════════════════════════════════════════════════
    // SEQUENTIAL TRAVERSE  (matches SEQUENTIAL_TRAVERSE in
    // acc_top.cpp hnsw_search_engine, lines 572-608)
    // ═══════════════════════════════════════════════════════

    // Step 1: process entry node
    visited[ENTRY] = 1;
    {
        float entry_dist = l2_sq(query, vectors[ENTRY], DIM);
        systolic_topk_insert(entry_dist, ENTRY, 0, cells);
    }

    // Step 2: for each node i, visit all unvisited neighbors
    for (int iter = 0; iter < N; iter++) {
        ap_uint<8> nn = num_nbrs[iter];
        for (ap_uint<8> ni = 0; ni < DEG; ni++) {
            if (ni < nn) {
                ap_uint<32> nbr_id = adjacency[iter][ni];
                if (nbr_id < N && visited[nbr_id] == 0) {
                    visited[nbr_id] = 1;
                    float nbr_dist = l2_sq(query, vectors[nbr_id], DIM);
                    systolic_topk_insert(nbr_dist, nbr_id, 0, cells);
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════
    // VERIFICATION
    // ═══════════════════════════════════════════════════════
    int errors = 0;

    // 1. At least N/2 nodes visited
    int visited_cnt = 0;
    for (int i = 0; i < N; i++) {
        if (visited[i] != 0) visited_cnt++;
    }
    cout << "\nVisited: " << visited_cnt << "/" << N << " nodes" << endl;
    if (visited_cnt < N / 2) {
        cout << "FAIL: only " << visited_cnt << " nodes visited"
             << " (need ≥ " << N / 2 << ")" << endl;
        errors++;
    }

    // 2. Collect valid (non-FP32_MAX) entries from systolic Top-K
    vector<pair<float, ap_uint<64>>> valid;
    for (int i = 0; i < TOPK_MAX; i++) {
        if (cells[i].best_dist < 3.402823466e+38f) {
            valid.push_back({cells[i].best_dist, cells[i].best_addr});
        }
    }
    cout << "Valid Top-K entries: " << valid.size() << endl;
    if (valid.empty()) {
        cout << "FAIL: no valid entries in systolic Top-K" << endl;
        errors++;
    }

    // 3. Distances must be sorted in ascending order
    for (size_t i = 1; i < valid.size(); i++) {
        if (valid[i].first < valid[i - 1].first - 1e-5f) {
            cout << "FAIL: Top-K not sorted at index " << i
                 << " (" << valid[i - 1].first << " > "
                 << valid[i].first << ")" << endl;
            errors++;
        }
    }

    // 4. Print results (top 10)
    cout << "\nTop-K (" << min((size_t)10, valid.size())
         << "/" << valid.size() << "):" << endl;
    for (size_t i = 0; i < min((size_t)10, valid.size()); i++) {
        printf("  [%zu] node=%llu dist=%.6f\n", i,
               (unsigned long long)valid[i].second.to_uint64(),
               valid[i].first);
    }

    cout << "\nErrors: " << errors << endl;
    return (errors == 0) ? 0 : 1;
}
