#ifndef TEST_UTILS_H
#define TEST_UTILS_H

// ─── Shared Test Utilities for C++ CTest and Vitis HLS Testbenches ───
// Header-only, inline, compatible with ap_int.h types.
// No std::vector, no Faiss headers, no variadic functions.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include "../include/acc_top.h"

// ─── FP32 comparison with tolerance ───
// Returns true if |a - b| < epsilon.
static inline bool float_approx_eq(float a, float b, float epsilon = 1e-4f) {
    return std::fabs(a - b) < epsilon;
}

// ─── Fill float array with random values in [-5, 5) ───
// Uses rand() — caller should srand() before use.
static inline void generate_random_query(float* buf, int dim) {
    for (int i = 0; i < dim; i++) {
        buf[i] = ((float)(rand() % 1000) / 100.0f) - 5.0f;
    }
}

// ─── Result unpack structure ───
// Matches the 128-bit result word format from compute_engine/systolic Top-K.
struct ResultEntry {
    float        dist;
    uint64_t     addr;
    uint32_t     len;
};

// ─── Unpack 128-bit result word into structured fields ───
// Bit layout (matching FIFO_RES / compute_engine output):
//   [31:0]   = dist      (FP32)
//   [95:32]  = doc_addr  (uint64)
//   [127:96] = doc_len   (uint32)
static inline ResultEntry unpack_result_128(ap_uint<128> result) {
    ResultEntry e;
    ap_uint<32> db = result.range(31, 0);
    std::memcpy(&e.dist, &db, 4);
    e.addr = result.range(95, 32).to_uint64();
    e.len  = result.range(127, 96).to_uint();
    return e;
}

// ─── Pack PQ entry into 512-bit data bus beat ───
// Matches Data Manager packing format.
//
// Bit layout (MSB → LSB):
//   [511 : 512-M*8]       = PQ codes            (M bytes, codes[0] at MSB)
//   [512-M*8-1 : 512-M*8-64]  = doc_addr        (8 bytes)
//   [512-M*8-65 : 512-M*8-128] = doc_len         (8 bytes, zero-extended from 32b)
//   [512-M*8-129 : 0]        = zero padding
//
// @tparam M  Number of PQ code bytes (compile-time constant for HLS).
// @param codes     Pointer to M code bytes.
// @param doc_addr  Document start address (64-bit).
// @param doc_len   Document length (32-bit, zero-extended to 64 bits in the word).
// @return ap_uint<512> packed beat.
template <int M>
static inline ap_uint<512> pack_pq_entry(const uint8_t* codes, ap_uint<64> doc_addr, ap_uint<32> doc_len) {
    ap_uint<512> word = 0;

    // codes[0] at MSB byte [511:504], codes[1] at [503:496], ...
    for (int i = 0; i < M; i++) {
        int hi = 511 - i * 8;
        word.range(hi, hi - 7) = codes[i];
    }

    // doc_addr (64 bits) right after codes
    const int addr_lsb = 512 - M * 8 - 64;
    word.range(addr_lsb + 63, addr_lsb) = doc_addr;

    // doc_len (64-bit field, zero-extended from 32-bit input)
    const int len_lsb = addr_lsb - 64;
    word.range(len_lsb + 63, len_lsb) = (ap_uint<64>)doc_len;

    return word;
}

#endif // TEST_UTILS_H
