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
#define TB_M     16
#define TB_Ds    (TB_DIM / TB_M)
#define TB_KS    256
#define TB_TOP_PQ 10
#define TB_N_PQ   50
#define TB_NODES_GRAPH 50
#define TB_DEG   32
#define TB_TOP_GR 10
#define DRAM_SIZE (16 * 1024 * 1024)

static void write_float(dram_byte_t* b, uint32_t o, float v) {
    uint32_t r; memcpy(&r,&v,4);
    b[o+0]=r&0xFF; b[o+1]=(r>>8)&0xFF; b[o+2]=(r>>16)&0xFF; b[o+3]=(r>>24)&0xFF;
}
static void write_u32(dram_byte_t* b, uint32_t o, uint32_t v) {
    b[o+0]=v&0xFF; b[o+1]=(v>>8)&0xFF; b[o+2]=(v>>16)&0xFF; b[o+3]=(v>>24)&0xFF;
}
static void write_u64(dram_byte_t* b, uint32_t o, uint64_t v) {
    for(int i=0;i<8;i++) b[o+i]=(v>>(i*8))&0xFF;
}

int main() {
    cout << "=== Unified IVFPQ/HNSW Accelerator Test ===" << endl;
    bool global_pass = true;

    // ═══════════════════════════════════════════
    // TEST MODE 0: IVFPQ
    // ═══════════════════════════════════════════
    cout << "\n--- MODE 0: IVFPQ PQ Search ---" << endl;
    {
        dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();
        const uint64_t CB = 0x10000, QB = 0x80000;

        write_u32(dram, CB+0, TB_M); write_u32(dram, CB+4, TB_DIM);
        write_u32(dram, CB+8, TB_N_PQ);

        uint32_t cbf = TB_M * TB_KS * TB_Ds, cbb = cbf*4;
        float* cbr = new float[cbf];
        srand(42);
        for(uint32_t i=0;i<cbf;i++){cbr[i]=((float)(rand()%1000)/100.0f)-5.0f; write_float(dram,CB+64+i*4,cbr[i]);}

        uint32_t pqs = TB_M+16;
        for(uint32_t n=0;n<TB_N_PQ;n++){uint32_t o=CB+64+cbb+n*pqs;
            for(uint32_t m=0;m<TB_M;m++) dram[o+m]=rand()%256;
            write_u64(dram,o+TB_M,((uint64_t)rand()<<32)|rand());
            write_u64(dram,o+TB_M+8,(uint64_t)(rand()%(1<<20)));
        }
        for(uint32_t d=0;d<TB_DIM;d++) write_float(dram,QB+d*4,((float)(rand()%1000)/100.0f)-5.0f);

        hls::stream<cb_pq_word_t> fc,fp; hls::stream<float> fq; hls::stream<res_word_t> fr;
        ComputeMeta meta; volatile bool dd=false,cd=false,ds=true,cs=true;

        data_manager(CB,QB,fc,fp,fr,fq,meta,dd,ds,dram);
        compute_engine(fc,fp,fr,fq,meta,cd,cs);

        int valid=0;
        for(int i=0;i<TB_TOP_PQ;i++){
            res_word_t r=fr.read(); ap_uint<32> db=r.range(31,0);
            float d; memcpy(&d,&db,4);
            if(d<1e37f) valid++;
        }
        cout << "  IVFPQ: " << valid << "/" << TB_TOP_PQ << " valid results (non-FP32_MAX)" << endl;
        if(valid < TB_TOP_PQ) global_pass = false;

        delete[] dram; delete[] cbr;
    }

    // ═══════════════════════════════════════════
    // TEST MODE 1: HNSW Graph Search
    // ═══════════════════════════════════════════
    cout << "\n--- MODE 1: HNSW Graph Search ---" << endl;
    {
        dram_byte_t* dram = new dram_byte_t[DRAM_SIZE]();

        // Graph header
        ap_uint<32>* hdr = (ap_uint<32>*)dram;
        hdr[0] = TB_NODES_GRAPH; hdr[1] = 0; hdr[2] = TB_DIM; hdr[3] = TB_DEG;

        // Graph nodes
        srand(12345);
        uint64_t nbytes = TB_DIM*4 + 4 + TB_DEG*4;
        for(uint32_t n=0;n<TB_NODES_GRAPH;n++){
            uint64_t o=16+n*nbytes;
            for(uint32_t d=0;d<TB_DIM;d++) write_float(dram,o+d*4,((float)(rand()%1000)/100.0f)-5.0f);
            uint32_t nn=min(32u,(uint32_t)TB_DEG); dram[o+TB_DIM*4]=nn&0xFF;
            for(uint32_t k=0;k<TB_DEG;k++){
                uint32_t nid=(k<nn)?((n+k+1)%TB_NODES_GRAPH):0;
                write_u32(dram,o+TB_DIM*4+4+k*4,nid);
            }
        }

        hls::stream<float> fq; hls::stream<res_word_t> fr;
        srand(67890);
        for(uint32_t d=0;d<TB_DIM;d++) fq.write(((float)(rand()%1000)/100.0f)-5.0f);
        ComputeMeta meta; meta.n_vectors=TB_NODES_GRAPH; meta.dim_actual=TB_DIM;
        meta.top_k=TB_TOP_GR; meta.search_mode=1;
        volatile bool cd=false, cs=true;

        hnsw_search_engine(fq,fr,meta,cd,cs,dram);

        int valid=0;
        for(int i=0;i<TB_TOP_GR;i++){
            res_word_t r=fr.read(); ap_uint<32> db=r.range(31,0);
            float d; memcpy(&d,&db,4);
            if(d<1e37f) valid++;
        }
        cout << "  HNSW: " << valid << "/" << TB_TOP_GR << " valid results (non-FP32_MAX)" << endl;
        if(valid < TB_TOP_GR) global_pass = false;

        delete[] dram;
    }

    cout << "\n========================================" << endl;
    if(global_pass) cout << "UNIFIED ACCELERATOR: ALL MODES PASS" << endl;
    else cout << "SOME MODES FAILED" << endl;
    return global_pass ? 0 : 1;
}
