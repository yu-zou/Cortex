#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

#include "smartssd_device.h"
#include "mock_smartssd.h"
#include "fixture_loader.h"

#ifndef CORTEX_FIXTURE_DIR
#define CORTEX_FIXTURE_DIR "/cortex/hw/test/fixtures/"
#endif

namespace {

static const std::string kFixtureDir = CORTEX_FIXTURE_DIR;

std::vector<uint8_t> load_cluster_file(int n) {
    return cortex::test::load_uint8_bin(kFixtureDir + "cluster_" + std::to_string(n) + ".bin");
}

std::vector<float> load_queries() {
    return cortex::test::load_float_bin(kFixtureDir + "queries.bin");
}

std::vector<float> load_golden_distances() {
    return cortex::test::load_float_bin(kFixtureDir + "golden_distances.bin");
}

struct GoldenTopKEntry {
    uint64_t doc_addr;
    float distance;
};

std::vector<GoldenTopKEntry> load_golden_topk() {
    std::ifstream f(kFixtureDir + "golden_topk.bin", std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open golden_topk.bin");
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0);
    size_t count = bytes / (sizeof(uint64_t) + sizeof(float));
    std::vector<GoldenTopKEntry> result(count);
    for (size_t i = 0; i < count; ++i) {
        f.read(reinterpret_cast<char*>(&result[i].doc_addr), sizeof(uint64_t));
        f.read(reinterpret_cast<char*>(&result[i].distance), sizeof(float));
    }
    return result;
}

// IVF Header (64B): {M(u32), DIM(u32), N(u32), reserved[13](u32)}
static constexpr size_t kHeaderSize = 64;

// Parse M, DIM, N from 64B Header; derive KS and Ds
struct ParsedHeader {
    uint32_t M;
    uint32_t DIM;
    uint32_t N;
    uint32_t KS;    // derived from data size
    uint32_t Ds;
};

ParsedHeader parse_header(const uint8_t* data, uint64_t object_size) {
    ParsedHeader h{};
    // little-endian read
    h.M   = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24);
    h.DIM = data[4] | (data[5] << 8) | (data[6] << 16) | (data[7] << 24);
    h.N   = data[8] | (data[9] << 8) | (data[10] << 16) | (data[11] << 24);
    h.Ds = h.DIM / h.M;

    // KS hardcoded to 16 (from metadata.json: ksub=16)
    // File-size derivation was unreliable with regenerated IVF-format fixtures
    h.KS = 16;
    (void)object_size;  // not used
    return h;
}

// extract from codebook at specified position
const float* codebook_ptr(const uint8_t* data, uint64_t) {
    return reinterpret_cast<const float*>(data + kHeaderSize);
}

// PQ codes pointer (M bytes per vector)
const uint8_t* pq_codes_ptr(const uint8_t* data, uint64_t object_size,
                            const ParsedHeader& h) {
    uint64_t cb_bytes = static_cast<uint64_t>(h.M) * h.KS * h.Ds * 4;
    return data + kHeaderSize + cb_bytes;
}

std::vector<float> compute_adc_l2(
    const float* query, uint32_t M, uint32_t KS, uint32_t Ds,
    const float* codebook, const uint8_t* pq_codes, uint32_t N)
{
    std::vector<float> dist_table(M * KS);
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t k = 0; k < KS; ++k) {
            float d = 0.0f;
            for (uint32_t j = 0; j < Ds; ++j) {
                float diff = query[m * Ds + j] - codebook[(m * KS + k) * Ds + j];
                d += diff * diff;
            }
            dist_table[m * KS + k] = d;
        }
    }
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

// extract doc_addr from PQ entry (offset M, 8 bytes LE)
uint64_t extract_doc_addr(const uint8_t* entry, uint32_t M) {
    uint64_t v = 0;
    for (int b = 0; b < 8; ++b)
        v |= static_cast<uint64_t>(entry[M + b]) << (b * 8);
    return v;
}

}  // namespace

TEST(MockSmartSSD, LoadAndSearchMatchesGoldenDistances) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* cb = codebook_ptr(cluster_bytes.data(), cluster_bytes.size());
    // PQ codes: each entry = M+16 bytes, extract M bytes of PQ codes
    uint32_t entry_bytes = hdr.M + 16;
    const uint8_t* raw_pq = pq_codes_ptr(cluster_bytes.data(), cluster_bytes.size(), hdr);

    // extract pure PQ codes
    std::vector<uint8_t> codes_flat(static_cast<size_t>(hdr.N) * hdr.M);
    for (uint32_t i = 0; i < hdr.N; ++i)
        std::memcpy(codes_flat.data() + i * hdr.M, raw_pq + i * entry_bytes, hdr.M);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.KS, hdr.Ds, cb, codes_flat.data(), hdr.N);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto result = device.search(0, nullptr, 0, query0, hdr.DIM, hdr.N, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(hdr.N, result.entries.size());

    // sort results by doc_addr (=vector index) for comparison
    std::vector<float> returned_dists(hdr.N);
    for (const auto& e : result.entries) {
        ASSERT_LT(e.doc_addr, hdr.N);
        returned_dists[e.doc_addr] = e.distance;
    }

    for (uint32_t i = 0; i < hdr.N; ++i) {
        EXPECT_NEAR(expected[i], returned_dists[i], 1e-4f)
            << "Mismatch at vector " << i;
    }
}

TEST(MockSmartSSD, TopKResultsMatchGolden) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries_flat = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* cb = codebook_ptr(cluster_bytes.data(), cluster_bytes.size());
    uint32_t entry_bytes = hdr.M + 16;
    const uint8_t* raw_pq = pq_codes_ptr(cluster_bytes.data(), cluster_bytes.size(), hdr);

    std::vector<uint8_t> codes_flat(static_cast<size_t>(hdr.N) * hdr.M);
    for (uint32_t i = 0; i < hdr.N; ++i)
        std::memcpy(codes_flat.data() + i * hdr.M, raw_pq + i * entry_bytes, hdr.M);

    const float* query0 = queries_flat.data();
    auto expected = compute_adc_l2(query0, hdr.M, hdr.KS, hdr.Ds, cb, codes_flat.data(), hdr.N);

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    const uint32_t top_k = 10;
    const uint32_t actual_k = std::min(top_k, hdr.N);
    auto result = device.search(0, nullptr, 0, query0, hdr.DIM, top_k, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, result.status);
    ASSERT_EQ(actual_k, result.entries.size());

    // expected Top-K: ascending by distance, doc_addr = vector index
    std::vector<std::pair<float, uint64_t>> sorted_expected(hdr.N);
    for (uint32_t i = 0; i < hdr.N; ++i)
        sorted_expected[i] = {expected[i], i};
    std::sort(sorted_expected.begin(), sorted_expected.end());

    for (uint32_t i = 0; i < actual_k; ++i) {
        EXPECT_EQ(sorted_expected[i].second, result.entries[i].doc_addr)
            << "Top-k order mismatch at rank " << i;
        EXPECT_NEAR(sorted_expected[i].first, result.entries[i].distance, 1e-4f)
            << "Distance mismatch at rank " << i;
    }
}

TEST(MockSmartSSD, SearchOnUnloadedClusterReturnsError) {
    cortex::MockSmartSSD device;
    float q[32] = {};
    auto result = device.search(99, nullptr, 0, q, 32, 10, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    EXPECT_NE(0, result.status);
    EXPECT_EQ(-ENOENT, result.status);
}

TEST(MockSmartSSD, EvictClearsCachedData) {
    auto cluster_bytes = load_cluster_file(0);
    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));
    EXPECT_TRUE(device.is_cluster_loaded(0));
    device.evict_cluster(0);
    EXPECT_FALSE(device.is_cluster_loaded(0));
}

TEST(MockSmartSSD, MultipleClustersLoadedSimultaneously) {
    cortex::MockSmartSSD device;
    for (int i = 0; i < 4; ++i) {
        auto bytes = load_cluster_file(i);
        ASSERT_EQ(0, device.load_cluster(static_cast<uint32_t>(i), bytes.data(), bytes.size(), nullptr, 0, 0));
    }
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(device.is_cluster_loaded(static_cast<uint32_t>(i)));
    }
}

TEST(MockSmartSSD, ConcurrentSearchThreadSafety) {
    auto cluster_bytes = load_cluster_file(0);
    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    auto queries = load_queries();
    const float* q0 = queries.data();

    std::vector<std::thread> threads;
    std::vector<int> statuses(10 * 100, -1);

    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&device, q0, &hdr, &statuses, t]() {
            for (int i = 0; i < 100; ++i) {
                auto r = device.search(0, nullptr, 0, q0, hdr.DIM, 5, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
                statuses[t * 100 + i] = r.status;
            }
        });
    }
    for (auto& th : threads) th.join();

    for (int i = 0; i < 10 * 100; ++i) {
        EXPECT_EQ(0, statuses[i]) << "Thread error at index " << i;
    }
}

TEST(MockSmartSSD, L2VsIPMetricProduceDifferentResults) {
    auto cluster_bytes = load_cluster_file(0);
    auto queries = load_queries();

    auto hdr = parse_header(cluster_bytes.data(), cluster_bytes.size());
    const float* q0 = queries.data();

    cortex::MockSmartSSD device;
    ASSERT_EQ(0, device.load_cluster(0, cluster_bytes.data(), cluster_bytes.size(), nullptr, 0, 0));

    auto l2 = device.search(0, nullptr, 0, q0, hdr.DIM, hdr.N, cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    auto ip = device.search(0, nullptr, 0, q0, hdr.DIM, hdr.N, cortex::MetricType::InnerProduct, cortex::SearchMode::IVFPQ);

    ASSERT_EQ(0, l2.status);
    ASSERT_EQ(0, ip.status);
    ASSERT_FALSE(l2.entries.empty());
    ASSERT_FALSE(ip.entries.empty());

    bool any_diff = false;
    for (size_t i = 0; i < l2.entries.size() && i < ip.entries.size(); ++i) {
        if (std::abs(l2.entries[i].distance - ip.entries[i].distance) > 1e-6f) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff) << "L2 and IP metrics produced identical distances";
}

namespace {
// Independent byte writer: compact CPU v1, not the FPGA beat/128B layout.
void put_le(std::vector<uint8_t>& b, size_t off, uint64_t v, size_t n = 4) {
    for (size_t j = 0; j < n; ++j) b[off + j] = static_cast<uint8_t>(v >> (8 * j));
}
float centroid(uint32_t m, uint32_t k, uint32_t j) {
    return float(int(k) - 128) / 64 + float(m + j) / 128;
}
std::vector<uint8_t> compact_object(uint32_t ks = 256, bool versioned = true) {
    constexpr uint32_t M = 96, D = 768, N = 3;
    std::vector<uint8_t> b(64 + ks * D * 4 + N * (M + 16), 0);
    put_le(b, 0, M); put_le(b, 4, D); put_le(b, 8, N);
    if (versioned) {
        put_le(b, 12, 0x31515043); // bytes "CPQ1"
        put_le(b, 16, 1); put_le(b, 20, ks);
    }
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t k = 0; k < ks; ++k)
            for (uint32_t j = 0; j < 8; ++j) {
                float f = centroid(m, k, j);
                uint32_t bits; std::memcpy(&bits, &f, 4);
                put_le(b, 64 + ((m * ks + k) * 8 + j) * 4, bits);
            }
    for (uint32_t i = 0; i < N; ++i) {
        size_t off = 64 + ks * D * 4 + i * 112;
        for (uint32_t m = 0; m < M; ++m) b[off + m] = i == 2 ? ks - 1 : i;
        put_le(b, off + M, 0x1234567800000000ULL + i, 8);
        put_le(b, off + M + 8, 100 + i, 8);
    }
    return b;
}
int load_bytes(cortex::MockSmartSSD& d, const std::vector<uint8_t>& b) {
    return d.load_cluster(77, b.data(), b.size(), nullptr, 0, 0);
}
}

TEST(CompactPQ, Explicit768M96KS256RawL2AndUnalignedBytes) {
    auto b = compact_object();
    ASSERT_EQ(64u + 256u * 768u * 4u + 3u * 112u, b.size());
    ASSERT_EQ(786496u, b.size() - 3 * 112);
    EXPECT_EQ(255, b[786496 + 2 * 112 + 95]);
    cortex::MockSmartSSD d;
    std::vector<uint8_t> unaligned(b.size() + 1);
    std::copy(b.begin(), b.end(), unaligned.begin() + 1);
    ASSERT_EQ(0, d.load_cluster(77, unaligned.data() + 1, b.size(), nullptr, 0, 0));
    std::vector<float> q(768, 0.25f);
    auto r = d.search(77, nullptr, 0, q.data(), 768, 3,
                     cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, r.status); ASSERT_EQ(3u, r.entries.size());
    for (const auto& e : r.entries) {
        uint64_t i = e.doc_addr - 0x1234567800000000ULL;
        ASSERT_LT(i, 3u); EXPECT_EQ(100 + i, e.doc_length);
        float expected = 0;
        // Direct reconstruction per sub-vector, no mock LUT helper.
        for (uint32_t m = 0; m < 96; ++m) {
            float sub = 0;
            for (uint32_t j = 0; j < 8; ++j) {
                float diff = q[m * 8 + j] - centroid(m, i == 2 ? 255 : i, j);
                sub += diff * diff;
            }
            expected += sub;
        }
        EXPECT_FLOAT_EQ(expected, e.distance);
    }
}

TEST(CompactPQ, RejectMalformedHeadersAndSizesBeforeAllocation) {
    auto valid = compact_object();
    cortex::MockSmartSSD d;
    for (size_t n : {size_t(0), size_t(12), size_t(63), size_t(64),
                     size_t(786495), valid.size() - 1}) {
        auto b = valid; b.resize(n); EXPECT_NE(0, load_bytes(d, b)) << n;
    }
    for (auto change : std::vector<std::pair<size_t, uint32_t>>{
             {0, 0}, {0, UINT32_MAX}, {4, 767}, {4, 0}, {8, UINT32_MAX},
             {12, 0}, {12, 123}, {16, 0}, {16, 2}, {20, 0}, {20, 17},
             {20, 257}, {24, 1}, {60, 1}, {0, 0xfffffff0}, {4, 0xfffffff0}}) {
        auto b = valid; put_le(b, change.first, change.second);
        EXPECT_NE(0, load_bytes(d, b)) << change.first << ":" << change.second;
    }
    auto padded = valid; padded.resize(valid.size() + 3 * 16);
    EXPECT_NE(0, load_bytes(d, padded)); // hardware 128B stride is not compact 112B
    auto huge = valid;
    put_le(huge, 0, UINT32_MAX); put_le(huge, 4, UINT32_MAX);
    put_le(huge, 8, UINT32_MAX);
    EXPECT_NE(0, load_bytes(d, huge)); // N*(M+16) exceeds uint64
    valid.push_back(0); EXPECT_NE(0, load_bytes(d, valid));
    EXPECT_FALSE(d.is_cluster_loaded(77));
}

TEST(CompactPQ, LegacyKS16ExplicitKS16AndNoSizeInference) {
    cortex::MockSmartSSD d;
    EXPECT_EQ(0, load_bytes(d, compact_object(16, false)));
    EXPECT_EQ(0, load_bytes(d, compact_object(16, true)));
    EXPECT_NE(0, load_bytes(d, compact_object(256, false)));
    auto bad = compact_object(16, false); bad[64 + 16 * 768 * 4] = 16;
    EXPECT_NE(0, load_bytes(d, bad));
}

TEST(CompactPQ, RejectNonfiniteMetadataOverflowAndInvalidQueries) {
    cortex::MockSmartSSD d;
    auto valid = compact_object();
    ASSERT_EQ(0, load_bytes(d, valid));
    for (uint32_t bits : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
        auto bad = valid; put_le(bad, 64, bits);
        EXPECT_NE(0, load_bytes(d, bad));
    }
    auto bad = valid; put_le(bad, 786496 + 96 + 8, 1ULL << 32, 8);
    EXPECT_NE(0, load_bytes(d, bad));
    std::vector<float> q(769, 0);
    auto search = [&](const float* p, uint32_t dim, cortex::MetricType metric) {
        return d.search(77, nullptr, 0, p, dim, 3, metric, cortex::SearchMode::IVFPQ);
    };
    EXPECT_NE(0, search(nullptr, 768, cortex::MetricType::L2).status);
    EXPECT_NE(0, search(q.data(), 767, cortex::MetricType::L2).status);
    EXPECT_NE(0, search(q.data(), 769, cortex::MetricType::L2).status);
    EXPECT_NE(0, search(q.data(), 768, cortex::MetricType::InnerProduct).status);
    EXPECT_NE(0, search(q.data(), 768, static_cast<cortex::MetricType>(99)).status);
    for (float f : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::max()}) {
        q[0] = f; EXPECT_NE(0, search(q.data(), 768, cortex::MetricType::L2).status);
    }
    q[0] = 0; EXPECT_EQ(0, search(q.data(), 768, cortex::MetricType::L2).status);
}

TEST(CompactPQ, EmptyListAndZeroTopK) {
    cortex::MockSmartSSD d;
    auto b = compact_object();
    ASSERT_EQ(0, load_bytes(d, b));
    std::vector<float> q(768, 0);
    auto r = d.search(77, nullptr, 0, q.data(), 768, 0,
                     cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    EXPECT_EQ(0, r.status); EXPECT_TRUE(r.entries.empty());
    put_le(b, 8, 0); b.resize(786496);
    ASSERT_EQ(0, load_bytes(d, b));
    r = d.search(77, nullptr, 0, q.data(), 768, 3,
                 cortex::MetricType::L2, cortex::SearchMode::IVFPQ);
    EXPECT_EQ(0, r.status); EXPECT_TRUE(r.entries.empty());
}

namespace {
void put_float(std::vector<uint8_t>& b, size_t off, float f) {
    uint32_t bits; std::memcpy(&bits, &f, 4); put_le(b, off, bits);
}
// D2/M2/KS16, residuals (1,2) and (3,4); independent LE byte fixture.
std::vector<uint8_t> residual_object(float c0, float c1, uint64_t id) {
    std::vector<uint8_t> b(64 + 16 * 2 * 4 + 2 * 4 + 2 * 18, 0);
    put_le(b, 0, 2); put_le(b, 4, 2); put_le(b, 8, 2);
    put_le(b, 12, 0x31515043); put_le(b, 16, 2); put_le(b, 20, 16);
    put_le(b, 24, 1); // residual flag, all other bits unsupported
    for (size_t k = 0; k < 16; ++k) {
        put_float(b, 64 + k * 4, 1 + 2 * k);
        put_float(b, 64 + (16 + k) * 4, 2 + 2 * k);
    }
    put_float(b, 192, c0); put_float(b, 196, c1);
    for (size_t i = 0; i < 2; ++i) {
        size_t off = 200 + i * 18;
        b[off] = b[off + 1] = i;
        put_le(b, off + 2, id + i, 8); put_le(b, off + 10, 7 + i, 8);
    }
    return b;
}
cortex::SearchResult residual_search(cortex::MockSmartSSD& d, uint32_t id,
                                     const float* q, cortex::MetricType metric) {
    return d.search(id, nullptr, 0, q, 2, 2, metric, cortex::SearchMode::IVFPQ);
}
}

TEST(ResidualPQ, HandCalculatedCrossListIPAndL2WithRawQuery) {
    cortex::MockSmartSSD d;
    auto a = residual_object(10, 1, 100), b = residual_object(-2, 3, 200);
    // List B has larger residuals, so omitting centers reverses the winner.
    b[200] = b[201] = 2; b[218] = b[219] = 3;
    ASSERT_EQ(0, d.load_cluster(1, a.data(), a.size(), nullptr, 0, 0));
    ASSERT_EQ(0, d.load_cluster(2, b.data(), b.size(), nullptr, 0, 0));
    const float q[] = {2, 1};
    auto ra = residual_search(d, 1, q, cortex::MetricType::InnerProduct);
    auto rb = residual_search(d, 2, q, cortex::MetricType::InnerProduct);
    ASSERT_EQ(0, ra.status); ASSERT_EQ(0, rb.status);
    ASSERT_EQ(2u, ra.entries.size()); ASSERT_EQ(2u, rb.entries.size());
    EXPECT_EQ(101u, ra.entries[0].doc_addr); EXPECT_FLOAT_EQ(-31, ra.entries[0].distance);
    EXPECT_EQ(201u, rb.entries[0].doc_addr); EXPECT_FLOAT_EQ(-21, rb.entries[0].distance);
    EXPECT_LT(ra.entries[0].distance, rb.entries[0].distance);
    auto l2 = residual_search(d, 1, q, cortex::MetricType::L2);
    ASSERT_EQ(0, l2.status); ASSERT_EQ(2u, l2.entries.size());
    EXPECT_EQ(100u, l2.entries[0].doc_addr); EXPECT_FLOAT_EQ(85, l2.entries[0].distance);
    EXPECT_FLOAT_EQ(137, l2.entries[1].distance);
}

TEST(ResidualPQ, DecodedReconstructionOracleRandomTinyQueries) {
    cortex::MockSmartSSD d;
    std::mt19937 rng(1827);
    std::uniform_int_distribution<int> values(-32, 32);
    for (int trial = 0; trial < 40; ++trial) {
        auto b = residual_object(values(rng) / 8.0f, values(rng) / 8.0f, 100);
        for (size_t off = 64; off < 200; off += 4)
            put_float(b, off, values(rng) / 8.0f);
        b[200] = rng() % 16; b[201] = rng() % 16;
        b[218] = rng() % 16; b[219] = rng() % 16;
        auto decode = [&](size_t off) {
            uint32_t bits = 0;
            for (int j = 0; j < 4; ++j) bits |= uint32_t(b[off + j]) << (8 * j);
            float f; std::memcpy(&f, &bits, 4); return double(f);
        };
        ASSERT_EQ(0, load_bytes(d, b));
        const float q[] = {values(rng) / 8.0f, values(rng) / 8.0f};
        for (auto metric : {cortex::MetricType::InnerProduct, cortex::MetricType::L2}) {
            auto r = residual_search(d, 77, q, metric);
            ASSERT_EQ(0, r.status); ASSERT_EQ(2u, r.entries.size());
            for (const auto& e : r.entries) {
                ASSERT_GE(e.doc_addr, 100u); ASSERT_LT(e.doc_addr, 102u);
                size_t rec = 200 + (e.doc_addr - 100) * 18;
                double expected = 0;
                for (size_t j = 0; j < 2; ++j) {
                    double x = decode(192 + 4 * j) + decode(64 + (j * 16 + b[rec + j]) * 4);
                    expected += metric == cortex::MetricType::L2 ?
                        (q[j] - x) * (q[j] - x) : -q[j] * x;
                }
                EXPECT_DOUBLE_EQ(expected, e.distance);
            }
            EXPECT_LE(r.entries[0].distance, r.entries[1].distance);
        }
    }
}

TEST(ResidualPQ, RejectFlagsTruncatedCentroidNonfiniteAndOverflow) {
    cortex::MockSmartSSD d;
    auto good = residual_object(10, 1, 100);
    for (uint32_t flags : {0u, 2u, 3u, UINT32_MAX}) {
        auto b = good; put_le(b, 24, flags); EXPECT_EQ(-EINVAL, load_bytes(d, b));
    }
    for (size_t n : {size_t(192), size_t(196), size_t(199), good.size() - 1}) {
        auto b = good; b.resize(n); EXPECT_EQ(-EINVAL, load_bytes(d, b));
    }
    auto empty = good; put_le(empty, 8, 0); empty.resize(199);
    EXPECT_EQ(-EINVAL, load_bytes(d, empty)); // incomplete centroid even with no records
    for (size_t off : {size_t(64), size_t(192), size_t(196)})
        for (uint32_t bits : {0x7fc00000u, 0x7f800000u, 0xff800000u}) {
            auto b = good; put_le(b, off, bits); EXPECT_EQ(-EINVAL, load_bytes(d, b));
        }
    auto reserved = good; put_le(reserved, 28, 1); EXPECT_EQ(-EINVAL, load_bytes(d, reserved));
    ASSERT_EQ(0, load_bytes(d, good));
    for (auto metric : {cortex::MetricType::L2, cortex::MetricType::InnerProduct}) {
        float q[] = {std::numeric_limits<float>::quiet_NaN(), 1};
        EXPECT_EQ(-EINVAL, residual_search(d, 77, q, metric).status);
        q[0] = std::numeric_limits<float>::max();
        EXPECT_EQ(-ERANGE, residual_search(d, 77, q, metric).status);
    }
    auto huge = good; put_float(huge, 192, std::numeric_limits<float>::max());
    ASSERT_EQ(0, load_bytes(d, huge));
    const float q[] = {2, 1};
    EXPECT_EQ(-ERANGE, residual_search(d, 77, q, cortex::MetricType::InnerProduct).status);
}

TEST(ResidualPQ, Full768M96KS256ExactLayout) {
    auto b = compact_object();
    put_le(b, 16, 2); put_le(b, 24, 1);
    b.insert(b.begin() + 786496, 768 * 4, 0);
    put_float(b, 786496, 2);
    ASSERT_EQ(64u + 786432u + 3072u + 3u * 112u, b.size());
    cortex::MockSmartSSD d;
    ASSERT_EQ(0, load_bytes(d, b));
    std::vector<float> q(768, 0); q[0] = 1;
    auto r = d.search(77, nullptr, 0, q.data(), 768, 3,
                     cortex::MetricType::InnerProduct, cortex::SearchMode::IVFPQ);
    ASSERT_EQ(0, r.status); ASSERT_EQ(3u, r.entries.size());
    EXPECT_EQ(0x1234567800000002ULL, r.entries[0].doc_addr);
    EXPECT_FLOAT_EQ(-2 - centroid(0, 255, 0), r.entries[0].distance);
    b.resize(b.size() + 3 * 16); EXPECT_EQ(-EINVAL, load_bytes(d, b));
}

TEST(ResidualPQ, LegacyAndV1RemainRawL2) {
    cortex::MockSmartSSD d;
    auto b = residual_object(10, 1, 100);
    b.erase(b.begin() + 192, b.begin() + 200);
    put_le(b, 16, 1); put_le(b, 24, 0);
    const float q[] = {2, 1};
    for (bool legacy : {false, true}) {
        if (legacy) std::fill(b.begin() + 12, b.begin() + 64, 0);
        ASSERT_EQ(0, load_bytes(d, b));
        auto r = residual_search(d, 77, q, cortex::MetricType::L2);
        ASSERT_EQ(0, r.status); ASSERT_EQ(2u, r.entries.size());
        EXPECT_EQ(100u, r.entries[0].doc_addr); EXPECT_FLOAT_EQ(2, r.entries[0].distance);
        EXPECT_FLOAT_EQ(10, r.entries[1].distance);
        if (!legacy) EXPECT_EQ(-ENOTSUP,
            residual_search(d, 77, q, cortex::MetricType::InnerProduct).status);
    }
}
