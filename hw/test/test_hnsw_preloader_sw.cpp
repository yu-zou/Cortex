// ============================================================================
// test_hnsw_preloader_sw.cpp — Software-level HNSW graph preloader
// verification.  Loads HNSW binary fixtures and validates vector and
// adjacency data without hardware dependency.
//
// Reads from: CORTEX_FIXTURE_DIR/hnsw/
//   graph_header.bin  — 4 × uint32 LE header
//   vectors.bin       — N × DIM × float32 LE
//   adjacency.bin     — N × MAX_DEG × uint32 LE (padded with 0xFFFFFFFF)
//   metadata.json     — plain JSON configuration
// ============================================================================

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ============================================================================
// Anonymous namespace — internal helpers
// ============================================================================
namespace {

// --------------------------------------------------------------------------
// Simple JSON integer extractor — handles the flat metadata.json keys only.
// --------------------------------------------------------------------------
static int jsonGetInt(const std::string& json, const std::string& key) {
  auto pos = json.find("\"" + key + "\"");
  if (pos == std::string::npos) return 0;
  pos = json.find(':', pos);
  if (pos == std::string::npos) return 0;
  while (pos < json.size() &&
         (json[pos] == ':' || json[pos] == ' ' || json[pos] == '\t'))
    ++pos;
  int val = 0;
  bool neg = false;
  if (pos < json.size() && json[pos] == '-') {
    neg = true;
    ++pos;
  }
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    val = val * 10 + (json[pos] - '0');
    ++pos;
  }
  return neg ? -val : val;
}

// --------------------------------------------------------------------------
// Fixture paths
// --------------------------------------------------------------------------
struct HnswFixturePaths {
  std::string dir;

  explicit HnswFixturePaths(const std::string& d) : dir(d) {}

  std::string metadata()       const { return dir + "metadata.json"; }
  std::string header()         const { return dir + "graph_header.bin"; }
  std::string vectors()        const { return dir + "vectors.bin"; }
  std::string adjacency()      const { return dir + "adjacency.bin"; }
};

// --------------------------------------------------------------------------
// HNSW header layout (matches generate_hnsw_fixtures.py)
// --------------------------------------------------------------------------
#pragma pack(push, 1)
struct HnswGraphHeader {
  uint32_t num_nodes;
  uint32_t entry_point;
  uint32_t dim;
  uint32_t max_degree;
};
#pragma pack(pop)

// --------------------------------------------------------------------------
// Load a raw binary file into a vector of uint8_t (C-style fopen/fread)
// --------------------------------------------------------------------------
static std::vector<uint8_t> loadBinaryFile(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    ADD_FAILURE() << "Cannot open: " << path;
    return {};
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<uint8_t> buf(static_cast<size_t>(size));
  if (size > 0) {
    size_t nread = std::fread(buf.data(), 1, static_cast<size_t>(size), f);
    EXPECT_EQ(nread, static_cast<size_t>(size))
        << "Short read on " << path;
  }
  std::fclose(f);
  return buf;
}

}  // anonymous namespace

// ============================================================================
// Test fixture
// ============================================================================
struct HNSWPreloaderTest : public ::testing::Test {
  HnswFixturePaths paths;
  int N         = 0;
  int DIM       = 0;
  int MAX_DEG   = 0;
  int entry_pt  = 0;

  std::vector<uint8_t> header_bytes;
  std::vector<uint8_t> vectors_bytes;
  std::vector<uint8_t> adjacency_bytes;

  HNSWPreloaderTest()
      : paths(std::string(CORTEX_FIXTURE_DIR) + "/hnsw/") {}

  void SetUp() override {
    // ---- Load and parse metadata.json ----
    std::ifstream f(paths.metadata());
    ASSERT_TRUE(f.good()) << "Cannot open " << paths.metadata();
    std::string json((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());

    N        = jsonGetInt(json, "N");
    DIM      = jsonGetInt(json, "DIM");
    MAX_DEG  = jsonGetInt(json, "MAX_DEG");
    entry_pt = jsonGetInt(json, "entry_point");

    ASSERT_GT(N, 0)       << "N must be > 0 from metadata.json";
    ASSERT_GT(DIM, 0)     << "DIM must be > 0 from metadata.json";
    ASSERT_GT(MAX_DEG, 0) << "MAX_DEG must be > 0 from metadata.json";

    // ---- Load binary fixtures ----
    header_bytes    = loadBinaryFile(paths.header());
    vectors_bytes   = loadBinaryFile(paths.vectors());
    adjacency_bytes = loadBinaryFile(paths.adjacency());
  }
};

// ============================================================================
// Test 1 — HeaderValid
//   Verify graph_header.bin matches metadata.json parameters and the
//   packed struct layout is correct.
// ============================================================================
TEST_F(HNSWPreloaderTest, HeaderValid) {
  ASSERT_GE(header_bytes.size(), sizeof(HnswGraphHeader))
      << "graph_header.bin too small for header struct";

  HnswGraphHeader hdr;
  std::memcpy(&hdr, header_bytes.data(), sizeof(hdr));

  EXPECT_EQ(static_cast<int>(hdr.num_nodes),  N)
      << "num_nodes mismatch";
  EXPECT_EQ(static_cast<int>(hdr.entry_point), entry_pt)
      << "entry_point mismatch";
  EXPECT_EQ(static_cast<int>(hdr.dim),         DIM)
      << "dim mismatch";
  EXPECT_EQ(static_cast<int>(hdr.max_degree),  MAX_DEG)
      << "max_degree mismatch";
}

// ============================================================================
// Test 2 — VectorsSize
//   vectors.bin must be exactly N * DIM * sizeof(float) bytes.
// ============================================================================
TEST_F(HNSWPreloaderTest, VectorsSize) {
  const size_t expected = static_cast<size_t>(N) *
                          static_cast<size_t>(DIM) *
                          sizeof(float);
  EXPECT_EQ(vectors_bytes.size(), expected);
}

// ============================================================================
// Test 3 — AdjacencySize
//   adjacency.bin must be exactly N * MAX_DEG * sizeof(uint32_t) bytes.
// ============================================================================
TEST_F(HNSWPreloaderTest, AdjacencySize) {
  const size_t expected = static_cast<size_t>(N) *
                          static_cast<size_t>(MAX_DEG) *
                          sizeof(uint32_t);
  EXPECT_EQ(adjacency_bytes.size(), expected);
}

// ============================================================================
// Test 4 — RingConnectivity
//   The graph generator builds a ring backbone: node i connects to (i+1)%N.
//   Verify node 0 has neighbor 1, and node N-1 has neighbor 0.
// ============================================================================
TEST_F(HNSWPreloaderTest, RingConnectivity) {
  const size_t row_bytes = static_cast<size_t>(MAX_DEG) * sizeof(uint32_t);

  ASSERT_GE(adjacency_bytes.size(), row_bytes * static_cast<size_t>(N))
      << "adjacency.bin too small";

  // ---- Node 0: must have neighbor 1 ----
  {
    const uint32_t* row0 = reinterpret_cast<const uint32_t*>(
        adjacency_bytes.data());
    bool found_n1 = false;
    for (int i = 0; i < MAX_DEG; ++i) {
      if (row0[i] == 1) {
        found_n1 = true;
        break;
      }
    }
    EXPECT_TRUE(found_n1)
        << "Node 0 does not have neighbor 1 (ring backbone broken)";
  }

  // ---- Node N-1: must have neighbor 0 ----
  {
    const uint32_t* row_last = reinterpret_cast<const uint32_t*>(
        adjacency_bytes.data() + static_cast<size_t>(N - 1) * row_bytes);
    bool found_n0 = false;
    for (int i = 0; i < MAX_DEG; ++i) {
      if (row_last[i] == 0) {
        found_n0 = true;
        break;
      }
    }
    EXPECT_TRUE(found_n0)
        << "Node N-1 does not have neighbor 0 (ring backbone broken)";
  }
}

// ============================================================================
// Test 5 — VectorRange
//   The generator creates vectors uniformly in [-5, 5).  Sample 10 vectors
//   (indices 0, 10, 20, ..., 90) and verify every element is in [-5, 5).
// ============================================================================
TEST_F(HNSWPreloaderTest, VectorRange) {
  const size_t vec_bytes = static_cast<size_t>(DIM) * sizeof(float);
  ASSERT_GE(vectors_bytes.size(), vec_bytes * static_cast<size_t>(N))
      << "vectors.bin too small";

  // Sample 10 evenly-spaced vectors
  const int kSampleCount = 10;
  const int step = (N > kSampleCount) ? N / kSampleCount : 1;

  for (int si = 0; si < kSampleCount && si * step < N; ++si) {
    int idx = si * step;
    const float* vec = reinterpret_cast<const float*>(
        vectors_bytes.data() + static_cast<size_t>(idx) * vec_bytes);

    for (int d = 0; d < DIM; ++d) {
      float val = vec[d];
      EXPECT_GE(val, -5.0f) << "vec[" << idx << "][" << d << "] = " << val
                            << " is below -5";
      EXPECT_LT(val, 5.0f)  << "vec[" << idx << "][" << d << "] = " << val
                            << " is >= 5";
    }
  }
}
