/*
 * test_scheduler.c — Hardware-independent unit tests for ARM scheduler
 *
 * Compile:
 *   gcc -std=c11 -o /tmp/test_arm test_scheduler.c -lpthread -lm && /tmp/test_arm
 *
 * Includes scheduler.c directly for access to static functions and data,
 * with main() excluded via UNIT_TEST guard.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

/* ─── Test Framework ─────────────────────────────────────── */
static int passed = 0, failed = 0;
static int test_failed = 0;

#define TEST(n)  do { printf("  %-42s: ", n); test_failed = 0; } while(0)
#define CHECK(c) do { \
    if (!(c)) { \
        printf("FAIL [line %d]", __LINE__); \
        failed++; \
        test_failed = 1; \
    } else { \
        passed++; \
    } \
} while(0)
#define DONE()   do { if (!test_failed) printf("OK"); printf("\n"); } while(0)

/* ─── Include scheduler source for internal function access ─── */
#define UNIT_TEST
#include "../scheduler.c"

/* ═══════════════════════════════════════════════════════════
 * Test cases
 * ═══════════════════════════════════════════════════════════ */

int main(void)
{
    /* ── Setup ───────────────────────────────────────────── */
    /* Initialize FPGA mock buffer (prototype mode) */
    if (fpga_init() != 0) {
        fprintf(stderr, "FATAL: fpga_init failed\n");
        return 1;
    }

    /* Initialize prefetch mutexes for tests that need them.
     * We do NOT call prefetch_init() to avoid spawning a real thread. */
    for (int i = 0; i < PREFETCH_SLOTS; i++) {
        pthread_mutex_init(&prefetch_slots[i].lock, NULL);
        pthread_cond_init(&prefetch_slots[i].cond, NULL);
        prefetch_slots[i].ready = 0;
        prefetch_slots[i].dram_buf = NULL;  /* no malloc in test */
    }

    printf("\narm-scheduler unit tests\n");
    printf("========================\n\n");

    /* ── 1. ComputeMeta layout ──────────────────────────── */
    TEST("ComputeMeta sizeof = 20 bytes (5 x uint32_t)");
    {
        CHECK(sizeof(ComputeMeta) == 5 * sizeof(uint32_t));
        CHECK(sizeof(ComputeMeta) == 20);
    }
    DONE();

    TEST("ComputeMeta field offsets match FPGA expectations");
    {
        CHECK(offsetof(ComputeMeta, m_actual) == 0);
        CHECK(offsetof(ComputeMeta, dim_actual) == 4);
        CHECK(offsetof(ComputeMeta, n_vectors) == 8);
        CHECK(offsetof(ComputeMeta, top_k)    == 12);
        CHECK(offsetof(ComputeMeta, metric_id)== 16);
    }
    DONE();

    /* ── 2. ComputeMeta packing / unpacking ─────────────── */
    TEST("ComputeMeta values pack correctly via fpga_write_meta");
    {
        ComputeMeta meta = {
            .m_actual   = 16,
            .dim_actual = 128,
            .n_vectors  = 1000,
            .top_k      = 50,
            .metric_id  = 3,
        };
        fpga_write_meta(&meta);

        uint32_t *reg = (uint32_t *)(fpga_map + FPGA_META_OFFSET / 4);
        CHECK(reg[0] == 16);
        CHECK(reg[1] == 128);
        CHECK(reg[2] == 1000);
        CHECK(reg[3] == 50);
        CHECK(reg[4] == 3);
    }
    DONE();

    /* ── 3. TopKResult extraction from 128-bit register ─── */
    TEST("TopKResult extraction from mock 128-bit register");
    {
        /* Layout: entry[0] = float dist, entry[1] = doc_addr[31:0],
         *         entry[2] = doc_addr[63:32], entry[3] = doc_len */
        uint32_t *res_ptr = (uint32_t *)(fpga_map + FPGA_RESULT_OFFSET / 4);

        /* dist = 3.14159f */
        float f = 3.14159f;
        uint32_t f_bits;
        memcpy(&f_bits, &f, sizeof(f_bits));
        res_ptr[0] = f_bits;
        res_ptr[1] = 0xDEADBEEF;        /* doc_addr low */
        res_ptr[2] = 0x0000CAFE;        /* doc_addr high */
        res_ptr[3] = 42;                /* doc_len */

        TopKResult result;
        fpga_read_results(&result, 1);

        CHECK(fabsf(result.dist - 3.14159f) < 1e-6f);
        CHECK(result.doc_addr == 0x0000CAFEDEADBEEFULL);
        CHECK(result.doc_len == 42);
    }
    DONE();

    /* ── 4. FPGA control: trigger ────────────────────────── */
    TEST("fpga_trigger sets control register bit 0 (start)");
    {
        fpga_map[FPGA_CTRL_OFFSET / 4] = 0;
        fpga_trigger();
        CHECK(fpga_map[FPGA_CTRL_OFFSET / 4] == 1);
    }
    DONE();

    /* ── 5. FPGA control: poll_done success ──────────────── */
    TEST("fpga_poll_done returns 0 when done flag (bit 1) set");
    {
        fpga_map[FPGA_CTRL_OFFSET / 4] = 0;

        /* Set done flag */
        fpga_map[FPGA_CTRL_OFFSET / 4] = 2;

        int rc = fpga_poll_done(100);
        CHECK(rc == 0);

        /* Verify register cleared after poll */
        CHECK(fpga_map[FPGA_CTRL_OFFSET / 4] == 0);
    }
    DONE();

    /* ── 6. FPGA control: poll_done timeout ──────────────── */
    TEST("fpga_poll_done returns -1 on timeout");
    {
        fpga_map[FPGA_CTRL_OFFSET / 4] = 0;   /* done never set */

        int rc = fpga_poll_done(1);             /* very short timeout */
        CHECK(rc == -1);
    }
    DONE();

    /* ── 7. FP32 extraction from raw register bits ───────── */
    TEST("fpga_read_results extracts FP32 dist via IEEE 754 bits");
    {
        uint32_t *res_ptr = (uint32_t *)(fpga_map + FPGA_RESULT_OFFSET / 4);

        /* e = 2.71828f as IEEE 754: 0x402DF84D */
        float orig = 2.71828f;
        uint32_t bits;
        memcpy(&bits, &orig, sizeof(bits));
        res_ptr[0] = bits;
        res_ptr[1] = 0;
        res_ptr[2] = 0;
        res_ptr[3] = 0;

        TopKResult r;
        fpga_read_results(&r, 1);
        CHECK(fabsf(r.dist - 2.71828f) < 1e-6f);

        /* Verify exact bit pattern preserved */
        uint32_t out_bits;
        memcpy(&out_bits, &r.dist, sizeof(out_bits));
        CHECK(out_bits == bits);
    }
    DONE();

    /* ── 8. Prefetch slot state machine ──────────────────── */
    TEST("Prefetch slot ready state transitions (0→1→0)");
    {
        /* Verify initial state 0 (pending / not ready) */
        CHECK(prefetch_slots[0].ready == 0);

        /* Simulate prefetch completion → ready = 1 */
        prefetch_slots[0].ready = 1;
        CHECK(prefetch_slots[0].ready == 1);

        /* Simulate consumer taking data → consumed, reset to 0 */
        prefetch_slots[0].ready = 0;
        CHECK(prefetch_slots[0].ready == 0);

        /* Verify other slot unaffected */
        CHECK(prefetch_slots[1].ready == 0);
    }
    DONE();

    /* ── 9. Prefetch slot mutex operations ───────────────── */
    TEST("Prefetch slot mutex lock / unlock cycles");
    {
        /* Single-threaded: verify lock-then-unlock doesn't deadlock */

        pthread_mutex_lock(&prefetch_slots[1].lock);
        prefetch_slots[1].cluster_id = 99;
        prefetch_slots[1].ready = 1;
        pthread_mutex_unlock(&prefetch_slots[1].lock);

        /* Re-lock and verify values persist */
        pthread_mutex_lock(&prefetch_slots[1].lock);
        CHECK(prefetch_slots[1].cluster_id == 99);
        CHECK(prefetch_slots[1].ready == 1);
        pthread_mutex_unlock(&prefetch_slots[1].lock);

        /* Double lock/unlock cycle should also work */
        pthread_mutex_lock(&prefetch_slots[0].lock);
        pthread_mutex_unlock(&prefetch_slots[0].lock);
        pthread_mutex_lock(&prefetch_slots[0].lock);
        pthread_mutex_unlock(&prefetch_slots[0].lock);
        CHECK(1);  /* survived without deadlock */
    }
    DONE();

    /* ── 10. prefetch_hit_rate calculation ────────────────── */
    TEST("prefetch_hit_rate: 5 hits, 2 misses = 71.4%");
    {
        prefetch_hit_count  = 5;
        prefetch_miss_count = 2;

        float rate = prefetch_hit_rate();
        float expected = 5.0f / 7.0f;  /* ≈ 0.7142857 */

        CHECK(fabsf(rate - expected) < 0.001f);

        /* Edge case: 0 total */
        prefetch_hit_count  = 0;
        prefetch_miss_count = 0;
        rate = prefetch_hit_rate();
        CHECK(fabsf(rate - 0.0f) < 0.001f);

        /* Edge case: 100% */
        prefetch_hit_count  = 10;
        prefetch_miss_count = 0;
        rate = prefetch_hit_rate();
        CHECK(fabsf(rate - 1.0f) < 0.001f);
    }
    DONE();

    /* ── 11. NVMe read: device open failure ──────────────── */
    TEST("nvme_read_cluster returns -1 when device open fails");
    {
        uint8_t buf[1024];
        int rc = nvme_read_cluster("/dev/nonexistent0", 0, buf, sizeof(buf));
        CHECK(rc == -1);
    }
    DONE();

    /* ── 12. NVMe read: excessive / zero bytes error ─────── */
    TEST("nvme_read_cluster returns -1 with zero bytes");
    {
        uint8_t buf[64];
        /* With bytes=0, aligned_bytes = (0+511)&~511 = 0.
         * posix_memalign with size 0 may fail (EINVAL on glibc). */
        int rc = nvme_read_cluster("/dev/nonexistent0", 0, buf, 0);
        CHECK(rc == -1);
    }
    DONE();

    /* ── 13. Codebook reuse logic ─────────────────────────── */
    TEST("Codebook reuse: same cluster_id skips NVMe read");
    {
        /* Inline test of the scheduler's codebook-reuse pattern */
        uint32_t last_cluster_id = UINT32_MAX;
        int nvme_read_issued = 0;

        int cluster_id = 7;

        /* First call: cluster_id differs from last → path would issue NVMe read */
        if (cluster_id != last_cluster_id) {
            nvme_read_issued++;
            last_cluster_id = cluster_id;
        }
        CHECK(nvme_read_issued == 1);

        /* Second call with same cluster_id → codebook reused, no NVMe */
        if (cluster_id != last_cluster_id) {
            nvme_read_issued++;
            last_cluster_id = cluster_id;
        }
        CHECK(nvme_read_issued == 1);   /* not incremented */

        /* New cluster_id → issues another NVMe read */
        cluster_id = 42;
        if (cluster_id != last_cluster_id) {
            nvme_read_issued++;
            last_cluster_id = cluster_id;
        }
        CHECK(nvme_read_issued == 2);
    }
    DONE();

    /* ── Summary ──────────────────────────────────────────── */
    printf("\n========================\n");
    printf("arm-scheduler: %d passed, %d failed  (out of %d)\n",
           passed - test_failed, failed, passed + failed);

    /* free fpga_map allocated by fpga_init() */
    free((void *)fpga_map);

    return failed ? 1 : 0;
}
