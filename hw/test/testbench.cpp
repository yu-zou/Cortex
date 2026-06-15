#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "../include/acc_top.h"

using namespace std;

typedef ap_uint<8> dram_byte_t;

#define TB_DIM  128
#define TB_M    16
#define TB_Ds   (TB_DIM / TB_M)
#define TB_KS   256
#define TB_N    50
#define TB_TOPK 50
#define DRAM_SIZE (16 * 1024 * 1024)

struct RefEntry { float dist; uint64_t addr; uint32_t len; };

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
static uint64_t read_u64(dram_byte_t* b, uint32_t o) {
    uint64_t v=0; for(int i=0;i<8;i++) v|=((uint64_t)b[o+i].to_uint())<<(i*8);
    return v;
}

static void ref_topk(float* q, float* cb, dram_byte_t* pq,
    uint32_t Mv, uint32_t Dv, uint32_t Nv, uint32_t K, RefEntry* ref)
{
    uint32_t Ds=Dv/Mv; vector<RefEntry> a; a.reserve(Nv);
    for(uint32_t n=0;n<Nv;n++){
        uint32_t base=n*(Mv+16);
        uint64_t da=read_u64(pq,base+Mv);
        uint32_t dl=(uint32_t)(read_u64(pq,base+Mv+8)&0xFFFFFFFFULL);
        float d=0;
        for(uint32_t m=0;m<Mv;m++){uint8_t c=pq[base+m].to_uint();
            for(uint32_t dd=0;dd<Ds;dd++){
                float ct=cb[(m*TB_KS+c)*Ds+dd],qs=q[m*Ds+dd],df=qs-ct; d+=df*df;
            }}
        a.push_back({d,da,dl});
    }
    sort(a.begin(),a.end(),[](const RefEntry&x,const RefEntry&y){return x.dist<y.dist;});
    for(uint32_t k=0;k<K&&k<a.size();k++) ref[k]=a[k];
    for(uint32_t k=a.size();k<K;k++) ref[k]={3.402823e+38f,0,0};
}

int main() {
    cout << "=== IVF-PQ Compute Engine Test ===" << endl;
    cout << "DIM="<<TB_DIM<<" M="<<TB_M<<" KS="<<TB_KS<<" N="<<TB_N<<" K="<<TB_TOPK<<endl;

    dram_byte_t* dram=new dram_byte_t[DRAM_SIZE]();
    const uint64_t CBASE=0x10000, QBASE=0x80000;

    // ─── Prepare data (same as DM would) ───
    write_u32(dram,CBASE+0,TB_M); write_u32(dram,CBASE+4,TB_DIM); write_u32(dram,CBASE+8,TB_N);

    uint32_t cbf=TB_M*TB_KS*TB_Ds, cbb=cbf*4; uint64_t cbo=CBASE+64;
    float* cbr=new float[cbf]; srand(42);
    for(uint32_t i=0;i<cbf;i++){cbr[i]=((float)(rand()%1000)/100.0f)-5.0f; write_float(dram,cbo+i*4,cbr[i]);}

    uint32_t pqs=TB_M+16, pqb=TB_N*pqs; uint64_t pqo=CBASE+64+cbb;
    dram_byte_t* pqr=new dram_byte_t[pqb];
    for(uint32_t n=0;n<TB_N;n++){uint32_t o=n*pqs;
        for(uint32_t m=0;m<TB_M;m++){uint8_t c=rand()%256; dram[pqo+o+m]=c; pqr[o+m]=c;}
        uint64_t da=((uint64_t)rand()<<32)|rand(); write_u64(dram,pqo+o+TB_M,da); write_u64(pqr,o+TB_M,da);
        uint32_t dl=rand()%(1024*1024); write_u64(dram,pqo+o+TB_M+8,(uint64_t)dl); write_u64(pqr,o+TB_M+8,(uint64_t)dl);
    }

    float* query=new float[TB_DIM];
    for(uint32_t d=0;d<TB_DIM;d++){query[d]=((float)(rand()%1000)/100.0f)-5.0f; write_float(dram,QBASE+d*4,query[d]);}

    RefEntry refr[TB_TOPK]; ref_topk(query,cbr,pqr,TB_M,TB_DIM,TB_N,TB_TOPK,refr);

    // ─── Stream data into compute_engine (bypassing DM) ───
    hls::stream<cb_pq_word_t> fc,fp; hls::stream<float> fq; hls::stream<res_word_t> fr;

    // Stream query
    ap_uint<32>* qptr=(ap_uint<32>*)(dram+QBASE);
    for(uint32_t i=0;i<TB_DIM;i++){fq.write(*((float*)&qptr[i]));}

    // Stream codebook
    ap_uint<32>* cbptr=(ap_uint<32>*)(dram+cbo);
    uint32_t cb_total=TB_M*TB_KS*TB_Ds, cb_beats=(cb_total*4+63)/64;
    for(uint32_t b=0;b<cb_beats;b++){
        cb_pq_word_t beat=0;
        for(int f=0;f<16;f++){uint32_t gi=b*16+f; if(gi<cb_total) beat.range(32*f+31,32*f)=cbptr[gi];}
        fc.write(beat);
    }

    // Stream PQ codes (1 beat per entry, padded)
    for(uint32_t n=0;n<TB_N;n++){
        cb_pq_word_t beat=0;
        uint32_t eb=TB_M+16, ebits=eb*8;
        for(int by=0;by<64&&by<eb;by++)
            beat.range(511-by*8,511-by*8-7)=dram[pqo+n*eb+by];
        fp.write(beat);
    }

    ComputeMeta meta; meta.m_actual=TB_M; meta.dim_actual=TB_DIM;
    meta.n_vectors=TB_N; meta.top_k=TB_TOPK; meta.metric_id=0;

    // ─── Run compute_engine ───
    cout << "\n=== Running Compute Engine ===" << endl;
    volatile bool cd=false, cs=true;
    compute_engine(fc,fp,fr,fq,meta,cd,cs);

    if(!cd){cerr<<"Compute failed\n";return 1;}

    // ─── Verify ───
    cout << "Top-10 results:" << endl;
    int pass=0,fail=0;
    for(int i=0;i<TB_TOPK;i++){
        res_word_t r=fr.read(); ap_uint<32> db=r.range(31,0);
        float hd; memcpy(&hd,&db,4);
        uint64_t ha=r.range(95,32).to_uint64(); uint32_t hl=r.range(127,96).to_uint();
        if(i<10) printf("  [%d] dist=%.4f addr=0x%016llx len=%u\n",i,hd,(unsigned long long)ha,hl);
        bool found=false;
        for(int j=0;j<TB_TOPK;j++){
            if(refr[j].addr==ha&&refr[j].len==hl&&fabs(hd-refr[j].dist)<0.5f){found=true;pass++;break;}
        }
        if(!found){printf("  FAIL [%d]: HW addr=0x%016llx\n",i,(unsigned long long)ha);fail++;}
    }
    cout << "Pass: "<<pass<<"/"<<TB_TOPK<<" Fail: "<<fail<<endl;

    delete[] dram; delete[] cbr; delete[] pqr; delete[] query;
    return (fail==0)?0:1;
}
