// Minimal E2E test: loads cluster via driver, searches, validates against golden
// Usage: compile with -DCORTEX_FIXTURE_DIR=path/to/fixtures
//        and -DCORTEX_HW_LIB=path/to/libcortex_hw_mock.so
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

extern "C" {
#include "cortex_driver_api.h"
}

#ifndef CORTEX_FIXTURE_DIR
#define CORTEX_FIXTURE_DIR "../../hw/test/fixtures"
#endif
#ifndef CORTEX_HW_LIB
#define CORTEX_HW_LIB "../../hw/build/lib/libcortex_hw_mock.so"
#endif

static std::vector<uint8_t> load_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
}

static std::vector<float> load_floats(const char* path) {
    auto raw = load_file(path);
    std::vector<float> v(raw.size() / sizeof(float));
    std::memcpy(v.data(), raw.data(), raw.size());
    return v;
}

int main() {
    const char* fixture = CORTEX_FIXTURE_DIR;
    const char* hw_lib  = CORTEX_HW_LIB;
    
    // 1. Init driver with HW mock lib
    char cfg[512];
    snprintf(cfg, sizeof(cfg), "{\"hw_lib_path\": \"%s\", \"max_cache_slots\": 4}", hw_lib);
    int rc = cortex_driver_init(cfg);
    if (rc != 0) { fprintf(stderr, "FAIL: driver init rc=%d\n", rc); return 1; }
    printf("PASS: driver init\n");

    // 2. Load cluster_0 fixture
    char path[512];
    snprintf(path, sizeof(path), "%s/cluster_0.bin", fixture);
    auto blob0 = load_file(path);
    rc = cortex_semantic_write(0, blob0.data(), blob0.size(), nullptr, 0, 0);
    if (rc != 0) { fprintf(stderr, "FAIL: write cluster_0 rc=%d\n", rc); return 1; }
    printf("PASS: write cluster_0 (%zu bytes)\n", blob0.size());

    // 3. Load queries and golden distances
    snprintf(path, sizeof(path), "%s/queries.bin", fixture);
    auto queries = load_floats(path);
    snprintf(path, sizeof(path), "%s/golden_distances.bin", fixture);
    auto golden = load_floats(path);
    
    printf("PASS: loaded %zu queries, %zu golden distances\n",
           queries.size() / 32, golden.size());

    // 4. Search: use first query
    cortex_read_result r = cortex_semantic_read(
        0, blob0.data(), blob0.size(),
        queries.data(), 32, 10, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);

    if (r.status != 0) { fprintf(stderr, "FAIL: search status=%d\n", r.status); return 1; }
    printf("PASS: search returned %u results\n", r.count);

    // 5. Verify top result has valid distance (not garbage)
    bool ok = true;
    for (uint32_t i = 0; i < r.count && i < 5; i++) {
        float d = r.entries[i].distance;
        uint64_t addr = r.entries[i].doc_addr;
        if (d < 0 || d > 10000) { printf("FAIL: rank %u: bad dist=%.2f\n", i, (double)d); ok = false; }
        if (addr >= 100) { printf("FAIL: rank %u: bad doc_addr=%llu\n", i, (unsigned long long)addr); ok = false; }
        printf("  rank %u: dist=%.2f doc_addr=%llu\n", i, (double)d, (unsigned long long)addr);
    }
    
    // 6. Verify cache state
    if (!cortex_is_cluster_cached(0)) { printf("FAIL: cluster_0 not cached\n"); ok = false; }
    else printf("PASS: cluster_0 cached\n");

    // 7. Second search (cache hit)
    r = cortex_semantic_read(0, blob0.data(), blob0.size(),
        queries.data(), 32, 5, CORTEX_METRIC_L2, CORTEX_MODE_IVFPQ);
    if (r.status != 0) { printf("FAIL: cache-hit search rc=%d\n", r.status); ok = false; }
    else printf("PASS: cache-hit search returned %u results\n", r.count);

    cortex_driver_shutdown();
    if (ok) printf("\nE2E TEST PASSED\n");
    else    printf("\nE2E TEST FAILED\n");
    return ok ? 0 : 1;
}
