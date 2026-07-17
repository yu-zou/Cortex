/*
 * ARM Scheduler — Minimum Prototype for Cortex MPSoC
 * 
 * Lifecycle: NVMe read cluster → DMA to DRAM → trigger FPGA → poll done → read Top-K
 * Compile: arm-linux-gnueabihf-gcc -static -o scheduler scheduler.c
 * Target:  MPSoC PS (Cortex-A53), Linux with NVMe driver, shared DRAM with FPGA PL
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

/* ─── Hardware Constants ─── */
#define FPGA_BASE_ADDR      0xA0000000ULL  /* PL AXI slave base (example) */
#define FPGA_CTRL_OFFSET    0x00           /* Start/Stop/Done register */
#define FPGA_META_OFFSET    0x10           /* ComputeMeta struct (8 × 32b) */
#define FPGA_RESULT_OFFSET  0x100          /* Top-K results (500 × 128b) */
#define DRAM_SHARED_BASE    0x10000000ULL  /* Shared DRAM window for DMA */
#define CLUSTER_DRAM_OFF    0x00000000ULL  /* Cluster data in shared DRAM */
#define QUERY_DRAM_OFF      0x01000000ULL  /* Query vector in shared DRAM */
#define NVME_DEVICE         "/dev/nvme0n1"
#define CLUSTER_MAX_BYTES   (4 * 1024 * 1024)  /* 4MB max cluster */
#define TOPK_MAX            500
#define TOPK_BYTES          (TOPK_MAX * 16)    /* 500 × 128-bit results */

/* ─── ComputeMeta (matches acc_top.h) ─── */
typedef struct {
    uint32_t m_actual;
    uint32_t dim_actual;
    uint32_t n_vectors;
    uint32_t top_k;
    uint32_t metric_id;     /* ap_uint<2> metric_id + ap_uint<1> search_mode packed */
} ComputeMeta;

/* ─── TopK Result Entry ─── */
typedef struct {
    float    dist;
    uint64_t doc_addr;
    uint32_t doc_len;
} TopKResult;

/* ─── NVMe read into shared DRAM ─── */
static int nvme_read_cluster(const char *device, off_t lba_offset,
                              void *dram_buf, size_t bytes)
{
    int fd = open(device, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("nvme open");
        return -1;
    }

    /* Align buffer and size for O_DIRECT (512-byte sectors) */
    size_t aligned_bytes = (bytes + 511) & ~511ULL;
    void *aligned_buf = NULL;
    if (posix_memalign(&aligned_buf, 512, aligned_bytes) != 0) {
        perror("posix_memalign");
        close(fd);
        return -1;
    }

    ssize_t rd = pread(fd, aligned_buf, aligned_bytes, lba_offset * 512);
    if (rd < 0) {
        perror("nvme pread");
        free(aligned_buf);
        close(fd);
        return -1;
    }

    /* Copy to DRAM destination (DMA would do this in HW; memcpy is placeholder) */
    memcpy(dram_buf, aligned_buf, bytes);
    free(aligned_buf);
    close(fd);
    return 0;
}

/* ─── FPGA register access via mmap ─── */
static volatile uint32_t *fpga_map = NULL;

static int fpga_init(void)
{
    /* Placeholder: On real MPSoC, open /dev/mem and mmap FPGA registers */
    /* For QEMU/prototype, allocate a dummy buffer */
    fpga_map = (volatile uint32_t *)calloc(4096, 1);
    if (!fpga_map) {
        fprintf(stderr, "FPGA mmap failed (QEMU prototype mode)\n");
        return -1;
    }
    printf("FPGA interface mapped (prototype)\n");
    return 0;
}

static void fpga_write_meta(const ComputeMeta *meta)
{
    uint32_t *reg = (uint32_t *)(fpga_map + FPGA_META_OFFSET / 4);
    reg[0] = meta->m_actual;
    reg[1] = meta->dim_actual;
    reg[2] = meta->n_vectors;
    reg[3] = meta->top_k;
    reg[4] = meta->metric_id;
}

static void fpga_trigger(void)
{
    fpga_map[FPGA_CTRL_OFFSET / 4] = 1;  /* start = 1 */
}

static int fpga_poll_done(int timeout_ms)
{
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        if (fpga_map[FPGA_CTRL_OFFSET / 4] & 2) {  /* done flag */
            fpga_map[FPGA_CTRL_OFFSET / 4] = 0;     /* clear start */
            return 0;
        }
        usleep(100);
        elapsed++;
    }
    fprintf(stderr, "FPGA timeout after %d ms\n", timeout_ms);
    return -1;
}

static void fpga_read_results(TopKResult *results, int count)
{
    uint32_t *res_ptr = (uint32_t *)(fpga_map + FPGA_RESULT_OFFSET / 4);
    for (int i = 0; i < count && i < TOPK_MAX; i++) {
        uint32_t *entry = res_ptr + i * 4;  /* 128-bit = 4 × 32-bit */
        results[i].dist     = *(float *)(&entry[0]);
        results[i].doc_addr = ((uint64_t)entry[2] << 32) | entry[1];
        results[i].doc_len  = entry[3];
    }
}

/* ─── DMA placeholder ─── */
static void dma_copy_cluster_to_fpga(void *dram_src, size_t bytes,
                                      uint64_t fpga_dram_offset)
{
    /* On real MPSoC: program AXI CDMA to move data.
       For prototype: data is already in shared DRAM; just notify FPGA of address. */
    printf("DMA: %zu bytes @ 0x%lx → FPGA DRAM offset 0x%lx\n",
           bytes, (uint64_t)dram_src, fpga_dram_offset);
}

/* ─── Prefetch thread state ─── */
typedef struct {
    int      cluster_id;
    void    *dram_buf;
    size_t   bytes;
    int      ready;         /* 0=pending, 1=ready, -1=error */
    pthread_mutex_t lock;
    pthread_cond_t  cond;
} PrefetchSlot;

#define PREFETCH_SLOTS 2
static PrefetchSlot prefetch_slots[PREFETCH_SLOTS];
static int prefetch_hit_count   = 0;
static int prefetch_miss_count  = 0;
static int prefetch_running     = 0;

/* ─── Prefetch worker thread ─── */
static void *prefetch_thread(void *arg)
{
    (void)arg;
    while (prefetch_running) {
        for (int i = 0; i < PREFETCH_SLOTS; i++) {
            pthread_mutex_lock(&prefetch_slots[i].lock);
            if (!prefetch_slots[i].ready) {
                /* Placeholder: issue NVMe read for adjacent cluster */
                int adjacent_id = prefetch_slots[i].cluster_id;
                printf("Prefetch: cluster %d (adjacent) → DRAM\n", adjacent_id);
                prefetch_slots[i].ready = 1;
                prefetch_hit_count++;
                pthread_cond_signal(&prefetch_slots[i].cond);
            }
            pthread_mutex_unlock(&prefetch_slots[i].lock);
        }
        usleep(10000);  /* 10ms polling interval */
    }
    return NULL;
}

static void prefetch_init(void)
{
    for (int i = 0; i < PREFETCH_SLOTS; i++) {
        pthread_mutex_init(&prefetch_slots[i].lock, NULL);
        pthread_cond_init(&prefetch_slots[i].cond, NULL);
        prefetch_slots[i].ready = 0;
        prefetch_slots[i].dram_buf = malloc(CLUSTER_MAX_BYTES);
    }
    prefetch_running = 1;
    pthread_t tid;
    pthread_create(&tid, NULL, prefetch_thread, NULL);
}

static float prefetch_hit_rate(void)
{
    int total = prefetch_hit_count + prefetch_miss_count;
    return total > 0 ? (float)prefetch_hit_count / total : 0.0f;
}

static void prefetch_stop(void)
{
    prefetch_running = 0;
}

/* ═══════════════════════════════════════════════════════════
 * MAIN SCHEDULER LOOP
 * ═══════════════════════════════════════════════════════════ */
#ifndef UNIT_TEST
int main(int argc, char **argv)
{
    int cluster_id = 0;
    int num_queries = 100;

    if (argc > 1) cluster_id = atoi(argv[1]);

    printf("Cortex ARM Scheduler — Minimal Prototype\n");
    printf("Target cluster: %d, Simulated queries: %d\n", cluster_id, num_queries);

    /* 1. Initialize FPGA interface */
    if (fpga_init() != 0) {
        fprintf(stderr, "FPGA init failed — running in log-only mode\n");
        return 1;
    }

    /* 2. Initialize prefetch engine */
    prefetch_init();

    /* 3. Scheduler main loop */
    uint8_t *cluster_buf = (uint8_t *)malloc(CLUSTER_MAX_BYTES);
    float    query[128];
    TopKResult results[TOPK_MAX];
    struct timespec t_start, t_end;
    static uint32_t last_cluster_id = UINT32_MAX;

    for (int q = 0; q < num_queries; q++) {
        clock_gettime(CLOCK_MONOTONIC, &t_start);

        /* 3a. NVMe read cluster data into shared DRAM (skip if same cluster) */
        size_t cluster_bytes = CLUSTER_MAX_BYTES;  /* actual size from header */
        if (cluster_id == last_cluster_id) {
            printf("[batch] cluster %d codebook reused\n", cluster_id);
        } else {
            if (nvme_read_cluster(NVME_DEVICE, cluster_id * 8192,
                                   cluster_buf, cluster_bytes) != 0) {
                fprintf(stderr, "Query %d: NVMe read failed for cluster %d\n",
                        q, cluster_id);
                continue;
            }
            printf("[cold] full load cluster %d\n", cluster_id);
            last_cluster_id = cluster_id;
        }

        /* 3b. DMA cluster data to FPGA-visible DRAM region */
        dma_copy_cluster_to_fpga(cluster_buf, cluster_bytes,
                                  DRAM_SHARED_BASE + CLUSTER_DRAM_OFF);

        /* 3c. Write query vector to DRAM (placeholder: all-zeros query) */
        memset(query, 0, sizeof(query));
        /* On real HW: DMA or memcpy query to QUERY_DRAM_OFF */

        /* 3d. Write ComputeMeta and trigger FPGA */
        ComputeMeta meta = {
            .m_actual   = 16,
            .dim_actual = 128,
            .n_vectors  = 100,
            .top_k      = 10,
            .metric_id  = 0,   /* L2 distance, IVFPQ mode */
        };
        fpga_write_meta(&meta);
        fpga_trigger();

        /* 3e. Poll FPGA done (timeout 500ms) */
        if (fpga_poll_done(500) != 0) {
            fprintf(stderr, "Query %d: FPGA timeout\n", q);
            continue;
        }

        /* 3f. Read Top-K results */
        fpga_read_results(results, 10);

        clock_gettime(CLOCK_MONOTONIC, &t_end);
        long elapsed_us = (t_end.tv_sec - t_start.tv_sec) * 1000000L
                        + (t_end.tv_nsec - t_start.tv_nsec) / 1000L;

        /* 3g. Submit adjacent cluster IDs for prefetch */
        /* Placeholder: IVF adjacent clusters = cluster_id ± 1 */
        for (int i = 0; i < PREFETCH_SLOTS && i < 2; i++) {
            pthread_mutex_lock(&prefetch_slots[i].lock);
            prefetch_slots[i].cluster_id = cluster_id + (i == 0 ? -1 : 1);
            prefetch_slots[i].ready = 0;
            pthread_mutex_unlock(&prefetch_slots[i].lock);
        }

        printf("Query %3d: Top-1 dist=%.4f addr=0x%lx len=%u | %ld us\n",
               q, results[0].dist, results[0].doc_addr, results[0].doc_len,
               elapsed_us);
    }

    prefetch_stop();
    printf("\nPrefetch hit rate: %.1f%% (%d/%d)\n",
           100.0f * prefetch_hit_rate(), prefetch_hit_count,
           prefetch_hit_count + prefetch_miss_count);
    printf("Scheduler done.\n");

    free(cluster_buf);
    free((void *)fpga_map);
    return 0;
}
#endif /* UNIT_TEST */
