// ═══════════════════════════════════════════════════════════════════
// testbench_cb_load.cpp — Vitis HLS testbench for codebook loading
//
// Tests that codebook data loaded from DRAM via Data Manager stream
// arrives correctly in BRAM. Uses a simplified path:
//   1. Populate DRAM with deterministic float values
//   2. Stream DRAM → hls::stream<cb_pq_word_t> (512-bit beats)
//   3. Read from FIFO → local array using CB_LOAD style indexing
//   4. Verify first, last, and spot-check random indices
//
// Parameters (small for fast C simulation):
//   M  = 16  (M_SYN)
//   KS =  8  (reduced from 256)
//   Ds =  8  (DS_SYN = DIM_SYN / M_SYN)
// ═══════════════════════════════════════════════════════════════════

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../include/acc_top.h"

using namespace std;
typedef ap_uint<8> dram_byte_t;

// ─── Test parameters (small for fast simulation) ───
#define TB_M      M_SYN       // 16
#define TB_KS     8           // reduced from 256 for fast sim
#define TB_Ds     DS_SYN      // 8
#define TB_FLOATS (TB_M * TB_KS * TB_Ds)  // 1024
#define TB_BEATS  ((TB_FLOATS * 4 + 63) / 64)  // 64

// ═══════════════════════════════════════════════════════════════════
// Helpers: write a float to DRAM byte array (little-endian)
// ═══════════════════════════════════════════════════════════════════
static void write_float_le(dram_byte_t* buf, uint32_t byte_off, float val) {
    uint32_t raw;
    memcpy(&raw, &val, 4);
    buf[byte_off + 0] = (raw >> 0)  & 0xFF;
    buf[byte_off + 1] = (raw >> 8)  & 0xFF;
    buf[byte_off + 2] = (raw >> 16) & 0xFF;
    buf[byte_off + 3] = (raw >> 24) & 0xFF;
}

// ═══════════════════════════════════════════════════════════════════
// Verification helper
// ═══════════════════════════════════════════════════════════════════
static int check_float(float got, float expected, const char* desc) {
    if (fabsf(got - expected) > 1e-6f) {
        printf("  FAIL %s: got %.1f  expected %.1f\n", desc, got, expected);
        return 1;
    }
    printf("  PASS %s = %.1f\n", desc, got);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════
// Reference value for codebook[m][ks][ds]
// ═══════════════════════════════════════════════════════════════════
static float ref_val(uint32_t m, uint32_t ks, uint32_t ds) {
    return (float)(m * 10000 + ks * 100 + ds);
}

// ═══════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════
int main() {
    printf("=== Codebook Load Test ===\n");
    printf("M=%d  KS=%d  Ds=%d  total_floats=%d  beats=%d\n",
           TB_M, TB_KS, TB_Ds, TB_FLOATS, TB_BEATS);

    int errors = 0;
    const uint64_t DRAM_BASE = 0x10000;
    uint32_t dram_bytes = TB_FLOATS * 4;

    // ─── 1. Allocate DRAM and populate with known data ──────────
    dram_byte_t* dram = new dram_byte_t[DRAM_BASE + dram_bytes]();

    // Layout: flat float array in row-major [M][KS][Ds] order
    //   value[m][ks][ds] = m*10000 + ks*100 + ds  (deterministic)
    for (uint32_t m = 0; m < TB_M; m++) {
        for (uint32_t ks = 0; ks < TB_KS; ks++) {
            for (uint32_t ds = 0; ds < TB_Ds; ds++) {
                uint32_t gidx = (m * TB_KS + ks) * TB_Ds + ds;
                write_float_le(dram, DRAM_BASE + gidx * 4, ref_val(m, ks, ds));
            }
        }
    }

    // ─── 2. Stream DRAM → hls::stream (simulates CB_STREAM) ────
    hls::stream<cb_pq_word_t> cb_fifo("cb_fifo");

    // Read floats as 32-bit words from byte-addressable DRAM,
    // pack 16 floats (512 bits = 1 cb_pq_word_t) per beat.
    ap_uint<32>* cbptr = (ap_uint<32>*)(dram + DRAM_BASE);
    for (uint32_t b = 0; b < TB_BEATS; b++) {
        cb_pq_word_t beat = 0;
        for (int f = 0; f < 16; f++) {
            uint32_t gidx = b * 16 + f;
            if (gidx < TB_FLOATS) {
                beat.range(32 * f + 31, 32 * f) = cbptr[gidx];
            }
        }
        cb_fifo.write(beat);
    }

    // ─── 3. Read from FIFO into codebook array (simulates CB_LOAD) ──
    //      Indexing: gidx → (m_idx, ks_idx, ds_idx)
    //        m_idx  = gidx / (KS * Ds)
    //        rem    = gidx % (KS * Ds)
    //        ks_idx = rem / Ds
    //        ds_idx = rem % Ds
    float codebook[TB_M][TB_KS][TB_Ds];
    uint32_t cb_total = TB_M * TB_KS * TB_Ds;
    uint32_t cb_beats = (cb_total * 4 + 63) / 64;

    for (uint32_t b = 0; b < cb_beats; b++) {
        cb_pq_word_t beat = cb_fifo.read();
        for (int f = 0; f < 16; f++) {
            uint32_t gidx   = b * 16 + f;
            if (gidx < cb_total) {
                uint32_t raw   = beat.range(32 * f + 31, 32 * f);
                float    val   = *((float*)&raw);
                uint32_t m_idx = gidx / (TB_KS * TB_Ds);
                uint32_t rem   = gidx % (TB_KS * TB_Ds);
                uint32_t ks_idx = rem / TB_Ds;
                uint32_t ds_idx = rem % TB_Ds;
                codebook[m_idx][ks_idx][ds_idx] = val;
            }
        }
    }

    // ─── 4. Verification ────────────────────────────────────────
    printf("\n--- Verification ---\n");

    // 4a. First element: codebook[0][0][0] = ref_val(0,0,0) = 0.0
    errors += check_float(codebook[0][0][0], ref_val(0, 0, 0),
                          "[0][0][0]");

    // 4b. Last element: codebook[M-1][KS-1][Ds-1]
    {
        uint32_t lm = TB_M - 1, lk = TB_KS - 1, ld = TB_Ds - 1;
        char desc[64];
        snprintf(desc, sizeof(desc), "[%u][%u][%u]", lm, lk, ld);
        errors += check_float(codebook[lm][lk][ld], ref_val(lm, lk, ld), desc);
    }

    // 4c. Spot-check: 5 random-looking indices
    struct { uint32_t m, ks, ds; } spots[5] = {
        {2,  3, 5},
        {7,  1, 2},
        {10, 6, 7},
        {4,  0, 3},
        {13, 5, 1}
    };
    for (int s = 0; s < 5; s++) {
        uint32_t m   = spots[s].m;
        uint32_t ks  = spots[s].ks;
        uint32_t ds  = spots[s].ds;
        char desc[64];
        snprintf(desc, sizeof(desc), "spot[%d] [%u][%u][%u]", s, m, ks, ds);
        errors += check_float(codebook[m][ks][ds], ref_val(m, ks, ds), desc);
    }

    // ─── 5. Summary ─────────────────────────────────────────────
    printf("\n=== %s (%d errors) ===\n",
           errors == 0 ? "ALL PASS" : "FAILED", errors);

    delete[] dram;
    return errors;
}
