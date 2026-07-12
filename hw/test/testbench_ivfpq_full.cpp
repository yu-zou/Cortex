#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// ─── Override synthesis parameters for small test ───
// Must come before #include "acc_top.h" to override defaults (M_SYN=16, DIM_SYN=128)
#define M_SYN   4
#define DIM_SYN 32
#define DS_SYN  (DIM_SYN / M_SYN)
#define PQ_CODE_BYTES   M_SYN
#define PQ_ENTRY_BYTES  (PQ_CODE_BYTES + 16)
#define PQ_ENTRY_BITS   (PQ_ENTRY_BYTES * 8)
#include "../include/acc_top.h"

using namespace std;

typedef ap_uint<8> dram_byte_t;

// ─── Test parameters ───
#define TEST_DIM   32
#define TEST_M     4
#define TEST_DS    (TEST_DIM / TEST_M)   // = 8
#define TEST_KS    256
#define TEST_N     20
#define TEST_TOPK  10
#define DRAM_SIZE  (16 * 1024 * 1024)

// ─── Reference result entry ───
struct RefEntry {
    float        dist;
    uint64_t     addr;
    uint32_t     len;
};

// ═══════════════════════════════════════════════════════════════
// DRAM helpers (little-endian byte packing)
// ═══════════════════════════════════════════════════════════════
static void write_float(dram_byte_t* b, uint32_t o, float v) {
    uint32_t r; memcpy(&r, &v, 4);
    b[o+0] =  r        & 0xFF;
    b[o+1] = (r >> 8)  & 0xFF;
    b[o+2] = (r >> 16) & 0xFF;
    b[o+3] = (r >> 24) & 0xFF;
}

static void write_u32(dram_byte_t* b, uint32_t o, uint32_t v) {
    b[o+0] =  v        & 0xFF;
    b[o+1] = (v >> 8)  & 0xFF;
    b[o+2] = (v >> 16) & 0xFF;
    b[o+3] = (v >> 24) & 0xFF;
}

static void write_u64(dram_byte_t* b, uint32_t o, uint64_t v) {
    for (int i = 0; i < 8; i++)
        b[o+i] = (v >> (i * 8)) & 0xFF;
}

static float read_float(dram_byte_t* b, uint32_t o) {
    uint32_t r = b[o] | (b[o+1] << 8) | (b[o+2] << 16) | (b[o+3] << 24);
    float v; memcpy(&v, &r, 4); return v;
}

static uint64_t read_u64(dram_byte_t* b, uint32_t o) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= ((uint64_t)b[o+i]) << (i * 8);
    return v;
}

// ═══════════════════════════════════════════════════════════════
// Software reference: brute-force L2 distance + sort
// ═══════════════════════════════════════════════════════════════
static void ref_topk(
    float*      q,       // [DIM]
    float*      cb,      // [M * KS * Ds]
    dram_byte_t*pq,      // [N * (M+16)]
    uint32_t    Mv,
    uint32_t    Dv,
    uint32_t    Nv,
    uint32_t    K,
    RefEntry*   ref)
{
    uint32_t Ds = Dv / Mv;
    vector<RefEntry> a;
    a.reserve(Nv);

    for (uint32_t n = 0; n < Nv; n++) {
        uint32_t base = n * (Mv + 16);

        // doc_addr from bytes M..M+7
        uint64_t da = read_u64(pq, base + Mv);
        // doc_len from bytes M+8..M+11 (lower 32 bits of 8-byte field)
        uint32_t dl = (uint32_t)(read_u64(pq, base + Mv + 8) & 0xFFFFFFFFULL);

        // L2 distance: sum of sub-distances per sub-quantizer
        float d = 0.0f;
        for (uint32_t m = 0; m < Mv; m++) {
            uint8_t c = pq[base + m];   // PQ code for this sub-vector
            for (uint32_t dd = 0; dd < Ds; dd++) {
                float ct = cb[(m * TEST_KS + c) * Ds + dd];
                float qs = q[m * Ds + dd];
                float df = qs - ct;
                d += df * df;
            }
        }
        a.push_back({d, da, dl});
    }

    // Sort ascending by distance
    sort(a.begin(), a.end(),
         [](const RefEntry& x, const RefEntry& y) {
             return x.dist < y.dist;
         });

    // Fill reference results (pad with FP32_MAX sentinel if needed)
    for (uint32_t k = 0; k < K && k < a.size(); k++)
        ref[k] = a[k];
    for (uint32_t k = a.size(); k < K; k++)
        ref[k] = {3.402823e+38f, 0, 0};
}

// ═══════════════════════════════════════════════════════════════
// MAIN — Full IVFPQ Pipeline Test
// ═══════════════════════════════════════════════════════════════
int main() {
    cout << "=== IVF-PQ Full Pipeline Test ===" << endl;
    cout << "DIM=" << TEST_DIM
         << "  M=" << TEST_M
         << "  KS=" << TEST_KS
         << "  N=" << TEST_N
         << "  TopK=" << TEST_TOPK << endl;
    cout << "DS_SYN=" << DS_SYN
         << "  PQ_CODE_BYTES=" << PQ_CODE_BYTES
         << "  PQ_ENTRY_BYTES=" << PQ_ENTRY_BYTES
         << "  TOPK_MAX=" << TOPK_MAX << endl;

    // ─── Allocate DRAM ───
    dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();
    const uint64_t CLUSTER_BASE = 0x10000;  // cluster_start_addr
    const uint64_t QUERY_ADDR   = 0x80000;  // query_ddr_addr
    const uint64_t RESULT_ADDR  = 0x90000;  // result_ddr_addr (hardware output path)

    // ═════════════════════════════════════════════════════
    // Step 1: IVFHeader at CLUSTER_BASE (64 bytes)
    //   offset 0x00: codebook_dim  (M)
    //   offset 0x04: pq_dim        (DIM)
    //   offset 0x08: pq_vectors    (N)
    //   offset 0x0C~0x3F: reserved (zeroed by alloc)
    // ═════════════════════════════════════════════════════
    write_u32(dram, CLUSTER_BASE + 0, TEST_M);
    write_u32(dram, CLUSTER_BASE + 4, TEST_DIM);
    write_u32(dram, CLUSTER_BASE + 8, TEST_N);

    // ═════════════════════════════════════════════════════
    // Step 2: Codebook at CLUSTER_BASE + 64
    //   Layout: [M][KS][Ds] floats, row-major
    //   Total: M * KS * Ds = 4 * 256 * 8 = 8192 floats
    // ═════════════════════════════════════════════════════
    uint32_t cb_floats = TEST_M * TEST_KS * TEST_DS;
    uint32_t cb_bytes  = cb_floats * 4;
    uint64_t cb_offset = CLUSTER_BASE + 64;

    float* cb_host = new float[cb_floats];
    srand(42);  // Known seed for reproducibility
    for (uint32_t i = 0; i < cb_floats; i++) {
        cb_host[i] = ((float)(rand() % 1000) / 100.0f) - 5.0f;
        write_float(dram, cb_offset + i * 4, cb_host[i]);
    }

    // ═════════════════════════════════════════════════════
    // Step 3: PQ entries at cb_offset + cb_bytes
    //   Each entry: M bytes PQ code + 8 bytes doc_addr + 8 bytes doc_len
    //   Total: N * (M + 16) bytes
    // ═════════════════════════════════════════════════════
    uint32_t entry_bytes = TEST_M + 16;   // 20 per entry
    uint32_t pq_bytes   = TEST_N * entry_bytes;
    uint64_t pq_offset  = cb_offset + cb_bytes;

    // Host-side copy for reference computation (matching DRAM layout)
    dram_byte_t* pq_host = new dram_byte_t[pq_bytes];

    for (uint32_t n = 0; n < TEST_N; n++) {
        uint32_t off = n * entry_bytes;

        // PQ codes: M bytes
        for (uint32_t m = 0; m < TEST_M; m++) {
            uint8_t c = (uint8_t)(rand() % 256);
            dram[pq_offset + off + m] = c;
            pq_host[off + m] = c;
        }

        // doc_addr: 8 bytes at offset M
        uint64_t da = ((uint64_t)rand() << 32) | (uint64_t)rand();
        write_u64(dram,   pq_offset + off + TEST_M, da);
        write_u64(pq_host,                       off + TEST_M, da);

        // doc_len: 4 bytes at offset M+8 (written as 8-byte field,
        // only lower 32 bits used by compute_engine)
        uint32_t dl = (uint32_t)(rand() % (1024 * 1024));
        write_u32(dram,   pq_offset + off + TEST_M + 8, dl);
        // Write as 64-bit to pq_host (lower 32 = dl, upper 32 = 0)
        write_u64(pq_host, off + TEST_M + 8, (uint64_t)dl);
    }

    // ═════════════════════════════════════════════════════
    // Step 4: Query vector at QUERY_ADDR
    // ═════════════════════════════════════════════════════
    float* query = new float[TEST_DIM];
    for (uint32_t d = 0; d < TEST_DIM; d++) {
        query[d] = ((float)(rand() % 1000) / 100.0f) - 5.0f;
        write_float(dram, QUERY_ADDR + d * 4, query[d]);
    }

    // ═════════════════════════════════════════════════════
    // Step 5: Compute software reference
    // ═════════════════════════════════════════════════════
    RefEntry refr[TEST_TOPK];
    ref_topk(query, cb_host, pq_host,
             TEST_M, TEST_DIM, TEST_N, TEST_TOPK, refr);

    cout << "\nReference Top-" << TEST_TOPK << " (sorted ascending):" << endl;
    for (uint32_t k = 0; k < TEST_TOPK; k++) {
        printf("  [%u] dist=%.4f  addr=0x%016llx  len=%u\n",
               k, refr[k].dist,
               (unsigned long long)refr[k].addr,
               refr[k].len);
    }

    // ═════════════════════════════════════════════════════
    // Step 6: Declare FIFOs (matching acc_top.cpp)
    // ═════════════════════════════════════════════════════
    hls::stream<cb_pq_word_t> fifo_cb("fifo_cb");
    hls::stream<cb_pq_word_t> fifo_pq("fifo_pq");
    hls::stream<res_word_t>   fifo_res("fifo_res");
    hls::stream<float>        fifo_qry("fifo_qry");

    volatile bool dm_start   = false;
    volatile bool dm_done    = false;
    volatile bool comp_start = false;
    volatile bool comp_done  = false;

    ComputeMeta meta;

    // ═════════════════════════════════════════════════════
    // FULL PIPELINE: Data Manager → Compute Engine
    // Matches acc_top.cpp C simulation path exactly:
    //
    //   dm_start = true;
    //   data_manager(...);
    //   comp_start = true;
    //   compute_engine(...);
    //
    // ═════════════════════════════════════════════════════

    // ─── Phase 1: Data Manager ───
    cout << "\n>>> Calling data_manager() ..." << endl;
    ap_uint<1> reload_codebook = 1;
    dm_start = true;
    data_manager(
        CLUSTER_BASE,           // cluster_start_addr
        QUERY_ADDR,             // query_ddr_addr
        RESULT_ADDR,            // result_ddr_addr (hardware output path)
        reload_codebook,        // reload_codebook (1 = load codebook from DRAM)
        fifo_cb,                // cb_fifo (output: codebook stream)
        fifo_pq,                // pq_fifo (output: PQ code stream)
        fifo_qry,               // query_fifo (output: query vector)
        fifo_res,               // res_fifo_in (input: results from compute_engine)
        meta,                   // meta_out (populated from IVFHeader)
        dm_done,                // done flag
        dm_start,               // start flag
        (ap_uint<8>*)dram       // DRAM pointer
    );

    if (!dm_done) {
        cerr << "FAIL: data_manager() did not set dm_done" << endl;
        delete[] dram; delete[] cb_host; delete[] pq_host; delete[] query;
        return 1;
    }
    cout << "  Data Manager completed." << endl;
    cout << "  meta: M=" << meta.m_actual.to_uint()
         << "  DIM=" << meta.dim_actual.to_uint()
         << "  N=" << meta.n_vectors.to_uint()
         << "  top_k=" << meta.top_k.to_uint() << endl;

    // ─── Phase 2: Compute Engine ───
    cout << "\n>>> Calling compute_engine() ..." << endl;
    comp_start = true;
    compute_engine(
        fifo_cb,                // cb_fifo (input: codebook from DM)
        fifo_pq,                // pq_fifo (input: PQ codes from DM)
        fifo_res,               // res_fifo_out (output: Top-K results)
        fifo_qry,               // query_fifo (input: query from DM)
        meta,                   // meta_in (from DM output)
        comp_done,              // done flag
        comp_start              // start flag
    );

    if (!comp_done) {
        cerr << "FAIL: compute_engine() did not set comp_done" << endl;
        delete[] dram; delete[] cb_host; delete[] pq_host; delete[] query;
        return 1;
    }
    cout << "  Compute Engine completed." << endl;

    // ═════════════════════════════════════════════════════
    // VERIFICATION
    //
    // compute_engine always pushes TOPK_MAX entries
    // (systolic array size). Read the first TEST_TOPK
    // and verify correctness.
    // ═════════════════════════════════════════════════════

    cout << "\n>>> HW Results (first " << TEST_TOPK << " of " << TOPK_MAX << "):" << endl;

    // Read all results (TOPK_MAX = 500) from fifo_res
    res_word_t hw_results[TOPK_MAX];
    for (int i = 0; i < TOPK_MAX; i++) {
        hw_results[i] = fifo_res.read();
    }

    int pass = 0;
    int fail = 0;

    for (uint32_t k = 0; k < TEST_TOPK; k++) {
        res_word_t r = hw_results[k];
        ap_uint<32> db    = r.range(31, 0);       // dist (FP32)
        float       hd; memcpy(&hd, &db, 4);      // HW distance
        uint64_t    ha    = r.range(95, 32).to_uint64();  // doc_addr
        uint32_t    hl    = r.range(127, 96).to_uint();   // doc_len

        printf("  [%u] HW: dist=%.4f  addr=0x%016llx  len=%u\n",
               k, hd, (unsigned long long)ha, hl);

        // ─── Check 1: Non-FP32_MAX (valid result) ───
        if (hd > 3.402823e+37f) {
            printf("  FAIL [%u]: sentinel FP32_MAX (expected real result)\n", k);
            fail++;
            continue;
        }

        // ─── Check 2: Sorted ascending ───
        if (k > 0) {
            res_word_t pr     = hw_results[k - 1];
            ap_uint<32> pdb   = pr.range(31, 0);
            float pd; memcpy(&pd, &pdb, 4);
            if (hd < pd - 1e-4f) {
                printf("  FAIL [%u]: NOT ascending (prev=%.4f, curr=%.4f)\n",
                       k, pd, hd);
                fail++;
                continue;
            }
        }

        // ─── Check 3: First result is the smallest ───
        if (k == 0) {
            // Verify first result is the best (smallest distance) among all tested
            // This is inherent if sorted ascending, but check explicitly
            for (uint32_t j = 1; j < TEST_TOPK; j++) {
                res_word_t jr     = hw_results[j];
                ap_uint<32> jdb   = jr.range(31, 0);
                float jd; memcpy(&jd, &jdb, 4);
                if (jd < hd - 1e-4f) {
                    printf("  FAIL [first not smallest]: result[%u]=%.4f < result[0]=%.4f\n",
                           j, jd, hd);
                    fail++;
                    break;
                }
            }
        }

        // ─── Check 4: addr and len are non-zero ───
        if (ha == 0) {
            printf("  FAIL [%u]: zero doc_addr\n", k);
            fail++;
            continue;
        }
        if (hl == 0) {
            printf("  FAIL [%u]: zero doc_len\n", k);
            fail++;
            continue;
        }

        // ─── Check 5: Match reference within tolerance ───
        bool matched_ref = false;
        for (uint32_t j = 0; j < TEST_TOPK; j++) {
            if (refr[j].addr == ha &&
                refr[j].len  == hl &&
                fabs(hd - refr[j].dist) < 0.5f) {
                matched_ref = true;
                break;
            }
        }
        if (!matched_ref) {
            printf("  FAIL [%u]: no matching reference (addr=0x%016llx len=%u dist=%.4f)\n",
                   k, (unsigned long long)ha, hl, hd);
            fail++;
        } else {
            pass++;
        }
    }

    // ─── Summary ───
    cout << "\n========================================" << endl;
    cout << "  Passed: " << pass << " / " << TEST_TOPK << endl;
    cout << "  Failed: " << fail << " / " << TEST_TOPK << endl;
    cout << "========================================" << endl;

    // ─── Cleanup ───
    delete[] dram;
    delete[] cb_host;
    delete[] pq_host;
    delete[] query;

    return (fail == 0) ? 0 : 1;
}
