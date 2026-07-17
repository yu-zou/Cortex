#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../include/acc_top.h"

using namespace std;

typedef ap_uint<8> dram_byte_t;

#define TB_DIM   128
#define TB_NODES 50
#define TB_DEG   32
#define TB_TOPK  10
#define DRAM_SIZE (16 * 1024 * 1024)

static void write_float(dram_byte_t* b, uint32_t o, float v) {
    uint32_t r; memcpy(&r,&v,4);
    b[o+0]=r&0xFF; b[o+1]=(r>>8)&0xFF; b[o+2]=(r>>16)&0xFF; b[o+3]=(r>>24)&0xFF;
}

// Generate a random connected graph for testing
struct GraphNode { vector<float> vec; vector<uint32_t> neighbors; };
static void gen_random_graph(vector<GraphNode>& graph, uint32_t N, uint32_t D, uint32_t max_deg) {
    graph.resize(N);
    srand(12345);
    for (uint32_t i = 0; i < N; i++) {
        graph[i].vec.resize(D);
        for (uint32_t d = 0; d < D; d++)
            graph[i].vec[d] = ((float)(rand()%1000)/100.0f) - 5.0f;
    }
    // Connect as a ring (guaranteed connectivity) + random edges
    for (uint32_t i = 0; i < N; i++) {
        graph[i].neighbors.push_back((i+1) % N);
        if (i > 0) graph[i].neighbors.push_back(i-1);
        // Add random neighbors
        for (uint32_t j = 0; j < min(max_deg-2, (uint32_t)(N/2)); j++) {
            uint32_t nbr = rand() % N;
            if (nbr != i && find(graph[i].neighbors.begin(), graph[i].neighbors.end(), nbr) == graph[i].neighbors.end()) {
                graph[i].neighbors.push_back(nbr);
            }
        }
    }
}

// Reference: Sequential traversal with neighbor expansion (matches HW)
static void ref_sequential_search(vector<GraphNode>& graph, vector<float>& query,
    uint32_t entry, uint32_t num_nodes, uint32_t K, vector<pair<float,uint32_t>>& results)
{
    vector<bool> visited(graph.size(), false);
    vector<pair<float,uint32_t>> all_dists;

    // Process entry point
    visited[entry] = true;
    {
        float ep_dist = 0;
        for (uint32_t d = 0; d < graph[entry].vec.size(); d++) {
            float diff = query[d] - graph[entry].vec[d];
            ep_dist += diff * diff;
        }
        all_dists.push_back({ep_dist, entry});
    }

    // Sequential: visit node 0..N-1, for each node, process ALL unvisited neighbors
    for (uint32_t i = 0; i < num_nodes; i++) {
        for (uint32_t nbr : graph[i].neighbors) {
            if (!visited[nbr] && nbr < num_nodes) {
                visited[nbr] = true;
                float dist = 0;
                for (uint32_t d = 0; d < graph[nbr].vec.size(); d++) {
                    float diff = query[d] - graph[nbr].vec[d];
                    dist += diff * diff;
                }
                all_dists.push_back({dist, nbr});
            }
        }
    }

    sort(all_dists.begin(), all_dists.end());
    results.clear();
    for (uint32_t k = 0; k < K && k < all_dists.size(); k++)
        results.push_back(all_dists[k]);
}

int main() {
    cout << "=== HNSW Graph Search Test ===" << endl;
    cout << "DIM=" << TB_DIM << " NODES=" << TB_NODES
         << " DEG=" << TB_DEG << " K=" << TB_TOPK << endl;

    // Generate random graph
    vector<GraphNode> graph;
    gen_random_graph(graph, TB_NODES, TB_DIM, TB_DEG);

    // ─── Write to DRAM ───
    dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();

    // Graph header: num_nodes, entry_point, dim, max_degree
    ap_uint<32>* hdr = (ap_uint<32>*)dram;
    hdr[0] = TB_NODES;
    hdr[1] = 0;  // entry_point = node 0
    hdr[2] = TB_DIM;
    hdr[3] = TB_DEG;

    // Node data (offset 16 = 4 * uint32)
    uint64_t node_bytes = TB_DIM * 4 + 4 + TB_DEG * 4;
    for (uint32_t n = 0; n < TB_NODES; n++) {
        uint64_t off = 16 + n * node_bytes;
        // Write vector
        for (uint32_t d = 0; d < TB_DIM; d++)
            write_float(dram, off + d*4, graph[n].vec[d]);
        // Write num_neighbors
        uint32_t nn = min((uint32_t)graph[n].neighbors.size(), (uint32_t)TB_DEG);
        dram[off + TB_DIM*4 + 0] = nn & 0xFF;
        dram[off + TB_DIM*4 + 1] = (nn>>8) & 0xFF;
        dram[off + TB_DIM*4 + 2] = (nn>>16) & 0xFF;
        dram[off + TB_DIM*4 + 3] = (nn>>24) & 0xFF;
        // Write neighbor IDs
        for (uint32_t k = 0; k < TB_DEG; k++) {
            uint32_t nid = (k < graph[n].neighbors.size()) ? graph[n].neighbors[k] : 0;
            uint32_t bo = off + TB_DIM*4 + 4 + k*4;
            dram[bo+0] = nid & 0xFF;
            dram[bo+1] = (nid>>8) & 0xFF;
            dram[bo+2] = (nid>>16) & 0xFF;
            dram[bo+3] = (nid>>24) & 0xFF;
        }
    }

    // ─── Query ───
    vector<float> query(TB_DIM);
    for (uint32_t d = 0; d < TB_DIM; d++)
        query[d] = ((float)(rand()%1000)/100.0f) - 5.0f;

    // ─── Reference search ───
    vector<pair<float,uint32_t>> ref_results;
    ref_sequential_search(graph, query, 0, TB_NODES, TB_TOPK, ref_results);
    cout << "\nReference Top-5:" << endl;
    for (int i = 0; i < 5 && i < (int)ref_results.size(); i++)
        printf("  [%d] node=%u dist=%.4f\n", i, ref_results[i].second, ref_results[i].first);

    // ─── Feed query into stream ───
    hls::stream<float> fq;
    hls::stream<res_word_t> fr;
    for (uint32_t d = 0; d < TB_DIM; d++) fq.write(query[d]);

    ComputeMeta meta;
    meta.n_vectors  = TB_NODES;
    meta.dim_actual = TB_DIM;
    meta.top_k      = TB_TOPK;
    meta.search_mode = 1;

    volatile bool cd = false, cs = true;

    // ─── Run HNSW Engine ───
    cout << "\n=== Running HNSW Search Engine ===" << endl;
    hnsw_search_engine(fq, fr, meta, cd, cs, dram);
    cout << "Done. comp_done=" << cd << endl;

    // ─── Read Results ───
    cout << "\nHardware Top-10 (from systolic Top-K):" << endl;
    int pass = 0, fail = 0;
    for (int i = 0; i < TB_TOPK; i++) {
        res_word_t r = fr.read();
        ap_uint<32> db = r.range(31,0);
        float hd; memcpy(&hd, &db, 4);
        uint32_t hn = r.range(95,32).to_uint();

        if (i < 10) printf("  [%d] node=%u dist=%.4f\n", i, hn, hd);

        // Find in reference
        bool found = false;
        for (auto& ref : ref_results) {
            if (ref.second == hn && fabs(hd - ref.first) < 1.0f) {
                found = true; pass++; break;
            }
        }
        if (!found && hd < 1e37f) {
            printf("  FAIL [%d] node=%u dist=%.4f not in reference\n", i, hn, hd);
            fail++;
        }
    }
    cout << "\nPass: " << pass << "/" << TB_TOPK << " Fail: " << fail << endl;

    delete[] dram;
    return (fail == 0) ? 0 : 1;
}
