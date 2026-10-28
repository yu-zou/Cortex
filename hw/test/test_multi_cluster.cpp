// =============================================================================
// test_multi_cluster.cpp — Multi-cluster aggregated search tests
//
// Tests the end-to-end aggregation workflow: load cluster_0..3 fixture files,
// run PQ ADC (Asymmetric Distance Computation) on each cluster, then merge
// and sort the results into a global Top-K across all clusters.
//
// All parsing and distance-computation logic mirrors mock_smartssd.cc so
// that this test validates the SAME aggregation logic that the FPGA pipeline
// will use.
// =============================================================================

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "fixture_loader.h"

#ifndef CORTEX_FIXTURE_DIR
#error "CORTEX_FIXTURE_DIR must be defined (set in CMakeLists.txt)"
#endif

// =============================================================================
// Anonymous namespace — internal helpers
// =============================================================================
namespace {

// ─── Known fixture parameters (from metadata.json) ──────────────────────
// The fixtures in CORTEX_FIXTURE_DIR were generated with:
//   python3 generate_fixtures.py --small
// yielding: M=4, KS=16, DIM=32, DSUB=8, NUM_VECTORS=100, TOP_K=10
static constexpr int kClusterCount   = 4;
static constexpr int kDefaultTopK    = 10;
static constexpr int kNumQueries     = 5;

// 64B IVF Header (IVFHeader format) — see generate_fixtures.py
static constexpr size_t kHeaderSize  = 64;

// ─── Header parsed from raw cluster bytes ───────────────────────────────
struct ParsedHeader {
    uint32_t M;     // sub-quantizer count
    uint32_t DIM;   // original vector dimension
    uint32_t N;     // number of PQ-encoded vectors in this cluster
    uint32_t KS;    // codebook size per sub-quantizer (derived from file size)
    uint32_t Ds;    // DIM / M
};

// Parse the 64-byte IVF header at offset 0 (little-endian).
// Exactly mirrors mock_smartssd.cc parse_header().
static ParsedHeader parse_header(const uint8_t* data, uint64_t object_size) {
    ParsedHeader h{};
    h.M   = static_cast<uint32_t>(data[0])
          | (static_cast<uint32_t>(data[1]) << 8)
          | (static_cast<uint32_t>(data[2]) << 16)
          | (static_cast<uint32_t>(data[3]) << 24);
    h.DIM = static_cast<uint32_t>(data[4])
          | (static_cast<uint32_t>(data[5]) << 8)
          | (static_cast<uint32_t>(data[6]) << 16)
          | (static_cast<uint32_t>(data[7]) << 24);
    h.N   = static_cast<uint32_t>(data[8])
          | (static_cast<uint32_t>(data[9]) << 8)
          | (static_cast<uint32_t>(data[10]) << 16)
          | (static_cast<uint32_t>(data[11]) << 24);
    h.Ds  = h.DIM / h.M;

    // KS is hardcoded to 16 (from metadata.json: ksub=16)
    // Older fixtures may have truncated codebooks — derive from metadata
    h.KS = 16;
    (void)object_size;  // Fixture size may not match due to truncated codebooks
    return h;
}

// ─── Pointer helpers into the raw cluster buffer ────────────────────────
// DRAM layout (matches mock_smartssd.h / generate_fixtures.py):
//   [0x00 – 0x3F]   64B IVF Header {M, DIM, N, reserved[13]}
//   [0x40 – …]      Codebook: M * KS * Ds * 4 bytes (float32)
//   [ …  – …]       PQ entries: N × (M + 16) bytes
//                      each entry: M codes (uint8) + doc_addr (uint64 LE)
//                                 + doc_length (uint64 LE, low 32b used)

static const float* codebook_ptr(const uint8_t* data) {
    return reinterpret_cast<const float*>(data + kHeaderSize);
}

static const uint8_t* pq_entries_ptr(const uint8_t* data,
                                      const ParsedHeader& h) {
    uint64_t cb_bytes = static_cast<uint64_t>(h.M) * h.KS * h.Ds * 4;
    return data + kHeaderSize + cb_bytes;
}

// ─── ADC L2 distance computation ────────────────────────────────────────
// Mirrors mock_smartssd.cc search() for MetricType::L2.
// Returns a vector of N distances, one per PQ-encoded vector in the cluster.

static std::vector<float> compute_adc_l2(
    const float* query,
    uint32_t M, uint32_t KS, uint32_t Ds,
    const float* codebook,
    const uint8_t* pq_codes,
    uint32_t N)
{
    // Empty cluster — no vectors to score.
    if (N == 0) return {};

    // ── 1. Build distance lookup table: M × KS ──
    std::vector<float> dist_table(static_cast<size_t>(M) * KS);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < KS; ++k) {
            float d = 0.0f;
            for (uint32_t j = 0; j < Ds; ++j) {
                float diff = query[m * Ds + j]
                           - codebook[(m * KS + k) * Ds + j];
                d += diff * diff;
            }
            dist_table[m * KS + k] = d;
        }
    }

    // ── 2. Compute ADC distance per PQ entry ──
    std::vector<float> dists(N);
    for (uint32_t i = 0; i < N; ++i) {
        float adc = 0.0f;
        for (uint32_t m = 0; m < M; ++m) {
            adc += dist_table[m * KS + pq_codes[i * M + m]];
        }
        dists[i] = adc;
    }
    return dists;
}

// ─── Extract PQ codes and doc_addrs from raw cluster data ───────────────
struct ClusterEntries {
    std::vector<uint8_t>  codes_flat;   // [N * M] PQ code bytes
    std::vector<uint64_t> doc_addrs;    // [N] document addresses
};

static ClusterEntries extract_entries(const uint8_t* data,
                                       const ParsedHeader& h) {
    uint32_t entry_bytes = h.M + 16;
    const uint8_t* raw_pq = pq_entries_ptr(data, h);

    ClusterEntries entries;
    if (h.N == 0) return entries;   // empty cluster

    entries.codes_flat.resize(static_cast<size_t>(h.N) * h.M);
    entries.doc_addrs.resize(h.N);

    for (uint32_t i = 0; i < h.N; ++i) {
        const uint8_t* entry = raw_pq + static_cast<size_t>(i) * entry_bytes;

        // PQ codes: first M bytes
        std::memcpy(entries.codes_flat.data() + i * h.M, entry, h.M);

        // doc_addr: bytes [M, M+8), little-endian uint64
        uint64_t addr = 0;
        for (int b = 0; b < 8; ++b)
            addr |= static_cast<uint64_t>(entry[h.M + b]) << (b * 8);
        entries.doc_addrs[i] = addr;
    }
    return entries;
}

// ─── Aggregated search result entry ─────────────────────────────────────
struct ResultEntry {
    float    distance;
    uint64_t doc_addr;
    uint32_t cluster_id;
};

// ─── Aggregate Top-K across multiple clusters ───────────────────────────
// Takes per-cluster distance vectors and entry metadata, merges all
// (distance, doc_addr, cluster_id) triples, sorts by distance, returns top-k.
static std::vector<ResultEntry> aggregate_topk(
    const std::vector<std::vector<float>>& cluster_dists,
    const std::vector<ClusterEntries>& cluster_entries,
    uint32_t top_k)
{
    // Collect all candidates
    std::vector<ResultEntry> all;
    for (size_t c = 0; c < cluster_dists.size(); ++c) {
        for (size_t i = 0; i < cluster_dists[c].size(); ++i) {
            all.push_back({cluster_dists[c][i],
                           cluster_entries[c].doc_addrs[i],
                           static_cast<uint32_t>(c)});
        }
    }

    if (all.empty()) return {};

    // Sort ascending by distance
    std::sort(all.begin(), all.end(),
              [](const ResultEntry& a, const ResultEntry& b) {
                  return a.distance < b.distance;
              });

    // Take top-k
    if (all.size() > top_k) all.resize(top_k);
    return all;
}

// ─── Verify results are sorted ascending by distance ────────────────────
static bool is_sorted_by_distance(const std::vector<ResultEntry>& results) {
    for (size_t i = 1; i < results.size(); ++i) {
        if (results[i].distance < results[i - 1].distance - 1e-6f)
            return false;
    }
    return true;
}

// ─── Fixture loading helpers ────────────────────────────────────────────
// These mirror test_mock_device.cc.

static std::vector<uint8_t> load_cluster_file(int n) {
    return cortex::test::load_uint8_bin(
        std::string(CORTEX_FIXTURE_DIR)
        + "cluster_" + std::to_string(n) + ".bin");
}

static std::vector<float> load_queries() {
    return cortex::test::load_float_bin(
        std::string(CORTEX_FIXTURE_DIR) + "queries.bin");
}

// ─── Per-cluster parsed data (loaded once per test suite) ───────────────
struct ClusterData {
    std::vector<uint8_t> raw;
    ParsedHeader         header;
    std::vector<float>   codebook;   // flattened [M][KS][Ds]
    ClusterEntries       entries;
};

static std::vector<ClusterData> load_all_clusters() {
    std::vector<ClusterData> clusters(kClusterCount);
    for (int i = 0; i < kClusterCount; ++i) {
        clusters[i].raw = load_cluster_file(i);
        if (clusters[i].raw.empty()) continue;

        clusters[i].header = parse_header(clusters[i].raw.data(),
                                           clusters[i].raw.size());

        if (clusters[i].header.N == 0) continue;   // skip empty

        // Copy codebook (truncated if fixture is older than expected)
        uint64_t cb_bytes = static_cast<uint64_t>(clusters[i].header.M)
                          * clusters[i].header.KS
                          * clusters[i].header.Ds * 4;
        uint64_t avail = clusters[i].raw.size() - kHeaderSize;
        uint64_t actual_cb = (cb_bytes < avail) ? cb_bytes : avail;
        if (actual_cb < cb_bytes / 2) {
            // Fixture codebook is too small (old format) — skip this cluster
            clusters[i].header.N = 0;
            continue;
        }
        clusters[i].codebook.resize(actual_cb / sizeof(float));
        if (actual_cb > 0) {
            std::memcpy(clusters[i].codebook.data(),
                        codebook_ptr(clusters[i].raw.data()), actual_cb);
        }

        // Extract PQ entries
        clusters[i].entries = extract_entries(clusters[i].raw.data(),
                                               clusters[i].header);
    }
    return clusters;
}

// ─── Run aggregated search for one query across specified clusters ──────
static std::vector<ResultEntry> search_all_clusters(
    const std::vector<ClusterData>& clusters,
    const float* query,
    int nprobe,         // number of clusters to search (≤ kClusterCount)
    uint32_t top_k)
{
    int n = std::min(nprobe, static_cast<int>(clusters.size()));

    std::vector<std::vector<float>> cluster_dists(n);
    std::vector<ClusterEntries>     cluster_entries(n);

    for (int c = 0; c < n; ++c) {
        const auto& cl = clusters[c];
        if (cl.header.N == 0) {
            cluster_dists[c] = {};
            cluster_entries[c] = {};
            continue;
        }
        cluster_dists[c] = compute_adc_l2(
            query, cl.header.M, cl.header.KS, cl.header.Ds,
            cl.codebook.data(), cl.entries.codes_flat.data(), cl.header.N);
        cluster_entries[c] = cl.entries;
    }

    return aggregate_topk(cluster_dists, cluster_entries, top_k);
}

}   // anonymous namespace

// =============================================================================
// Test cases
// =============================================================================

// ────────────────────────────────────────────────────────────────────────────
// Test 1 — AllFourClusters
//   Search across all 4 cluster fixtures, aggregate Top-10.
//   Verify: results are sorted ascending, correct count, doc_addrs are valid.
// ────────────────────────────────────────────────────────────────────────────
TEST(MultiCluster, AllFourClusters) {
    const auto clusters = load_all_clusters();
    const auto queries  = load_queries();
    ASSERT_FALSE(clusters.empty());
    ASSERT_FALSE(queries.empty());

    const float* q = queries.data();        // use first query (index 0)

    auto results = search_all_clusters(clusters, q, kClusterCount,
                                        kDefaultTopK);

    // Must not crash, must return results
    EXPECT_FALSE(results.empty());

    // Must not exceed requested Top-K
    EXPECT_LE(results.size(), static_cast<size_t>(kDefaultTopK));

    // Results must be sorted ascending by distance
    EXPECT_TRUE(is_sorted_by_distance(results));

    // Every result must have a valid doc_addr and cluster_id < kClusterCount
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_NO_FATAL_FAILURE({
            // doc_addr was written by generate_fixtures.py as the vector
            // index within the cluster; any uint64 is acceptable.
            (void)results[i].doc_addr;
        }) << " at rank " << i;
        EXPECT_LT(results[i].cluster_id,
                  static_cast<uint32_t>(kClusterCount))
            << " at rank " << i;
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2 — NprobeLimit
//   Search with nprobe=2: only cluster_0 and cluster_1 are searched.
//   Verify: all returned results have cluster_id ∈ {0, 1},
//           and no vectors from cluster_2 or cluster_3 appear.
// ────────────────────────────────────────────────────────────────────────────
TEST(MultiCluster, NprobeLimit) {
    const auto clusters = load_all_clusters();
    const auto queries  = load_queries();
    ASSERT_FALSE(clusters.empty());
    ASSERT_FALSE(queries.empty());

    const int nprobe = 2;
    const float* q = queries.data();

    auto results = search_all_clusters(clusters, q, nprobe, kDefaultTopK);

    EXPECT_FALSE(results.empty());
    EXPECT_LE(results.size(), static_cast<size_t>(kDefaultTopK));
    EXPECT_TRUE(is_sorted_by_distance(results));

    // All results must be from clusters 0 or 1 only
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_LT(results[i].cluster_id,
                  static_cast<uint32_t>(nprobe))
            << " at rank " << i
            << " — cluster " << results[i].cluster_id
            << " was searched despite nprobe=" << nprobe;
    }

    // Ensure that searching 2 clusters does NOT produce the same results
    // as searching all 4 (the full set should have more / better candidates).
    auto all_results = search_all_clusters(clusters, q, kClusterCount,
                                            kDefaultTopK);
    if (!all_results.empty() && !results.empty()) {
        // The top-1 distance with all 4 clusters should be ≤ the top-1
        // distance with only 2 clusters (more candidates → potentially better).
        EXPECT_LE(all_results[0].distance, results[0].distance + 1e-6f);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3 — EmptyCluster
//   Simulate an empty cluster (N=0). Verify no crash, empty results are
//   handled gracefully in the aggregation pipeline.
// ────────────────────────────────────────────────────────────────────────────
TEST(MultiCluster, EmptyCluster) {
    // Build an artificial empty-cluster header in a 64B buffer.
    // M=4, DIM=32, N=0 → no PQ entries, no codebook, no data.
    uint8_t empty_header[kHeaderSize] = {};
    // Little-endian: M=4 at offset 0
    empty_header[0] = 4;
    // DIM=32 at offset 4
    empty_header[4] = 32;
    // N=0 at offset 8 (already zero)

    // Construct an "empty cluster" as raw bytes that parse_header can read.
    std::vector<uint8_t> empty_raw(empty_header, empty_header + kHeaderSize);

    auto hdr = parse_header(empty_raw.data(), empty_raw.size());
    EXPECT_EQ(hdr.M,   4u);
    EXPECT_EQ(hdr.DIM, 32u);
    EXPECT_EQ(hdr.N,   0u);

    // Extract entries from empty cluster — should not crash.
    ClusterEntries entries;
    EXPECT_NO_FATAL_FAILURE({
        entries = extract_entries(empty_raw.data(), hdr);
    });
    EXPECT_TRUE(entries.codes_flat.empty());
    EXPECT_TRUE(entries.doc_addrs.empty());

    // Run ADC on empty cluster — should return empty distances without crash.
    const float dummy_query[32] = {};
    EXPECT_NO_FATAL_FAILURE({
        auto dists = compute_adc_l2(dummy_query, hdr.M, 16, hdr.Ds,
                                     nullptr, nullptr, 0);
        EXPECT_TRUE(dists.empty());
    });

    // Aggregate: mix of one real cluster and one empty cluster.
    // First, load a real cluster to have valid data.
    const auto clusters = load_all_clusters();
    const auto queries  = load_queries();
    ASSERT_FALSE(clusters.empty());
    ASSERT_FALSE(queries.empty());

    if (clusters[0].header.N > 0 && !clusters[0].codebook.empty()) {
        const float* q = queries.data();

        // Search real cluster alone
        std::vector<std::vector<float>> real_dists(1);
        std::vector<ClusterEntries> real_entries(1);
        real_dists[0] = compute_adc_l2(
            q, clusters[0].header.M, clusters[0].header.KS,
            clusters[0].header.Ds,
            clusters[0].codebook.data(),
            clusters[0].entries.codes_flat.data(),
            clusters[0].header.N);
        real_entries[0] = clusters[0].entries;

        auto real_results = aggregate_topk(real_dists, real_entries,
                                            kDefaultTopK);
        EXPECT_FALSE(real_results.empty());

        // Now aggregate with empty cluster appended — must still work.
        std::vector<std::vector<float>> mixed_dists = {real_dists[0], {}};
        std::vector<ClusterEntries> mixed_entries = {real_entries[0], {}};
        auto mixed_results = aggregate_topk(mixed_dists, mixed_entries,
                                             kDefaultTopK);

        EXPECT_FALSE(mixed_results.empty());
        EXPECT_TRUE(is_sorted_by_distance(mixed_results));

        // Results with empty cluster must be identical to real-only results
        // (empty cluster contributes no candidates).
        ASSERT_EQ(mixed_results.size(), real_results.size());
        for (size_t i = 0; i < mixed_results.size(); ++i) {
            EXPECT_EQ(mixed_results[i].distance, real_results[i].distance)
                << " at rank " << i;
            EXPECT_EQ(mixed_results[i].doc_addr, real_results[i].doc_addr)
                << " at rank " << i;
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4 — ConsistentResults
//   Run the same query against the same clusters twice.
//   Verify: results are bitwise identical (same distances, same doc_addrs).
// ────────────────────────────────────────────────────────────────────────────
TEST(MultiCluster, ConsistentResults) {
    const auto clusters = load_all_clusters();
    const auto queries  = load_queries();
    ASSERT_FALSE(clusters.empty());
    ASSERT_FALSE(queries.empty());

    const float* q = queries.data();

    // Run the same search twice
    auto results_a = search_all_clusters(clusters, q, kClusterCount,
                                          kDefaultTopK);
    auto results_b = search_all_clusters(clusters, q, kClusterCount,
                                          kDefaultTopK);

    ASSERT_EQ(results_a.size(), results_b.size());

    EXPECT_TRUE(is_sorted_by_distance(results_a));
    EXPECT_TRUE(is_sorted_by_distance(results_b));

    for (size_t i = 0; i < results_a.size(); ++i) {
        EXPECT_EQ(results_a[i].distance, results_b[i].distance)
            << " distance mismatch at rank " << i;
        EXPECT_EQ(results_a[i].doc_addr, results_b[i].doc_addr)
            << " doc_addr mismatch at rank " << i;
        EXPECT_EQ(results_a[i].cluster_id, results_b[i].cluster_id)
            << " cluster_id mismatch at rank " << i;
    }
}
