#include "../include/cemu_def.h"
#include <stdint.h>
#include <string.h>

#define M_SYN 96
#define DIM_SYN 768
#define DS_SYN 8
#define KS 256
#define TOPK_MAX 500
#define ENTRY_BYTES_SYN 112

typedef struct {
    float dist;
    uint64_t doc_addr;
    uint32_t doc_len;
} __attribute__((packed)) ResultEntry;

typedef struct {
    uint64_t doc_addr;
    uint32_t doc_len;
} Payload;

long long ivfpq(struct cemu_args *args) {
    int numr = args->numr;
    void **mr_addr = args->mr_addr;
    long long cparam1 = args->cparam1;  // top_k
    long long cparam2 = args->cparam2;  // reload_codebook
    
    uint8_t *cluster = (uint8_t *)mr_addr[0];
    float *query = (float *)mr_addr[1];
    ResultEntry *result = (ResultEntry *)mr_addr[2];
    
    // Parse header
    uint32_t M = *(uint32_t *)(cluster + 0);
    uint32_t DIM = *(uint32_t *)(cluster + 4);
    uint32_t N = *(uint32_t *)(cluster + 8);
    
    uint32_t top_k = (cparam1 > 0) ? (uint32_t)cparam1 : TOPK_MAX;
    int reload_codebook = (cparam2 == 1);
    
    // Codebook
    static float codebook[M_SYN][KS][DS_SYN];
    if (reload_codebook) {
        float *cb = (float *)(cluster + 64);
        for (uint32_t gidx = 0; gidx < M_SYN * KS * DS_SYN; gidx++) {
            uint32_t m = gidx >> 11;  // /2048 (M=96, Ds=8)
            uint32_t rem = gidx & 0x7FF;
            uint32_t k = rem >> 3;    // /8
            uint32_t d = rem & 0x7;   // %8
            codebook[m][k][d] = cb[gidx];
        }
    }
    
    // Dist table
    static float dist_table[M_SYN][KS];
    for (int c = 0; c < KS; c++) {
        for (int m = 0; m < M_SYN; m++) {
            float acc = 0.0f;
            for (int d = 0; d < DS_SYN; d++) {
                float diff = query[m * DS_SYN + d] - codebook[m][c][d];
                acc += diff * diff;
            }
            dist_table[m][c] = acc;
        }
    }
    
    // PQ scan
    uint8_t *pq = cluster + 64 + M_SYN * KS * DS_SYN * 4;
    Payload payload[8192];
    uint32_t keys[TOPK_MAX];
    uint16_t idx[TOPK_MAX];
    
    for (int i = 0; i < TOPK_MAX; i++) {
        keys[i] = 0x7F800000;  // +INF
        idx[i] = 0;
    }
    
    for (uint32_t n = 0; n < N; n++) {
        uint8_t *entry = pq + n * ENTRY_BYTES_SYN;
        uint8_t codes[M_SYN];
        memcpy(codes, entry, M_SYN);
        memcpy(&payload[n].doc_addr, entry + M_SYN, 8);
        memcpy(&payload[n].doc_len, entry + M_SYN + 8, 4);
        
        float part[M_SYN];
        for (int m = 0; m < M_SYN; m++) {
            part[m] = dist_table[m][codes[m]];
        }
        
        // Tree reduction
        float l1[48];
        for (int i = 0; i < 48; i++) l1[i] = part[2*i] + part[2*i+1];
        float l2[24];
        for (int i = 0; i < 24; i++) l2[i] = l1[2*i] + l1[2*i+1];
        float l3[12];
        for (int i = 0; i < 12; i++) l3[i] = l2[2*i] + l2[2*i+1];
        float l4[6];
        for (int i = 0; i < 6; i++) l4[i] = l3[2*i] + l3[2*i+1];
        float l5[3];
        for (int i = 0; i < 3; i++) l5[i] = l4[2*i] + l4[2*i+1];
        float total = (l5[0] + l5[1]) + l5[2];
        
        uint32_t dist_key;
        memcpy(&dist_key, &total, 4);
        
        // Systolic top-k
        uint32_t cd = dist_key;
        uint16_t ci = (uint16_t)n;
        for (int i = 0; i < TOPK_MAX; i++) {
            if (cd < keys[i]) {
                uint32_t tk = keys[i];
                keys[i] = cd;
                cd = tk;
                uint16_t ti = idx[i];
                idx[i] = ci;
                ci = ti;
            }
        }
    }
    
    // Output
    uint32_t k = (top_k < TOPK_MAX) ? top_k : TOPK_MAX;
    for (uint32_t i = 0; i < k; i++) {
        memcpy(&result[i].dist, &keys[i], 4);
        result[i].doc_addr = payload[idx[i]].doc_addr;
        result[i].doc_len = payload[idx[i]].doc_len;
    }
    
    return k;
}
