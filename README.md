# Cortex (灵枢)

Cortex (灵枢) is a Near-Data Processing (NDP) system designed for serverless RAG (Retrieval-Augmented Generation) vector search. It is built on top of Ceph distributed object storage and leverages SmartSSD FPGA acceleration to offload compute-intensive vector search operations directly to the storage layer.

By offloading IVFPQ ADC and HNSW graph search to SmartSSD devices, Cortex significantly reduces data movement between storage and host CPU, minimizing latency and maximizing throughput for large-scale vector retrieval.

The system consists of three independently-built components linked via a stable C ABI and runtime `dlopen()` mechanism:
1. **hw/**: SmartSSD FPGA device layer (currently providing a high-performance mock implementation).
2. **driver/**: Host driver bridging the hardware and the storage system.
3. **sys/**: A specialized fork of Ceph (integrated as a git submodule) providing semantic search endpoints and OSD-side processing logic.

Cortex is designed for graceful degradation; if the hardware driver is absent or version-mismatched, Ceph continues to operate independently, returning `-ENOTSUP` for semantic operations without impacting core storage stability.

## Architecture

The following diagram illustrates the request flow from the client through the RGW (RADOS Gateway) to the OSDs (Object Storage Daemons) and finally to the SmartSSD hardware.

```
Client Request
     │
     ▼
┌─────────────────────┐
│   RGW (sys/)        │  ← Eigen3 centroid distance → pick cluster_id
│  Semantic Search    │  ← async aio_operate scatter to OSDs
└──────────┬──────────┘
           │ SEMANTIC_READ op (RADOS)
           ▼
┌─────────────────────┐
│   OSD (sys/)        │  ← SemanticOpWQ (P0/P1/P2 locality-aware)
│  PrimaryLogPG       │  ← reads cluster object from BlueStore
└──────────┬──────────┘
           │ cortex_semantic_read()
           ▼
┌─────────────────────┐
│  Driver (driver/)   │  ← dlopen() loads libcortex_hw_mock.so
│  SmartSSDDriver     │  ← LRU cluster cache
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HW Mock (hw/)      │  ← Manual IVFPQ ADC search
│  MockSmartSSD       │  ← Codebook + PQ code decoding
└─────────────────────┘
```

### hw/ — SmartSSD Device Layer
The hardware layer defines an abstract `SmartSSDDevice` interface (in `smartssd_device.h`) with methods for lifecycle management and search operations: `init()`, `shutdown()`, `load_cluster()`, `search()`, `is_cluster_loaded()`, and `evict_cluster()`.

The current implementation, `MockSmartSSD`, performs manual IVFPQ ADC search. It processes IVF cluster binary data: a 64-byte header (M, DIM, N as little-endian uint32) followed by a codebook (M × ksub × dsub floats) and PQ-encoded vector entries (M code bytes + 8B doc_addr + 8B doc_length per vector). This layer builds as `libcortex_hw_mock.so`.

### driver/ — Host Driver
The driver layer implements the `SmartSSDDriver` class, which handles the runtime loading of the hardware library via `dlopen()`. It includes an LRU cluster cache to avoid redundant hardware loads of the same vector clusters.

The `DriverManager` serves as a thread-safe singleton (guarded by `std::mutex g_driver_mutex`) and exports the core C ABI functions: `cortex_driver_init()`, `cortex_driver_shutdown()`, `cortex_semantic_read()`, and `cortex_semantic_write()`. It builds as `libcortex_driver.so`.

### sys/ — Ceph Fork
The system layer is a Ceph fork containing the following Cortex-specific enhancements:
- **CortexDriverLoader**: Dynamically loads `libcortex_driver.so` at OSD startup with strict ABI version validation.
- **RGWOp_Semantic_Search**: An HTTP endpoint that uses Eigen3 for high-performance centroid distance calculations and `aio_operate()` for asynchronous scatter-gather requests to OSDs.
- **CentroidLRUCache**: Caches decoded centroid data using a `shared_mutex` for concurrent read access.
- **SemanticOpWQ**: A priority-aware work queue that classifies operations into P0 (hot/same-cluster), P1 (DRAM-cached), and P2 (cold miss) levels.
- **PrimaryLogPG**: Integrated handlers for the new `SEMANTIC_READ` and `SEMANTIC_WRITE` RADOS opcodes.

### C ABI Contract
The interface between the storage system and the driver is defined by a fixed-size C ABI to ensure compatibility and eliminate heap allocation overhead across the boundary.

```c
#define CORTEX_API_VERSION  1
#define CORTEX_TOPK_MAX     500
#define CORTEX_METRIC_L2    0
#define CORTEX_METRIC_IP    1

/* Top-K result entry — matches FPGA 128-bit output:
 *   dist(FP32,32b) + doc_addr(64b) + doc_length(32b) + pad(32b) */
struct cortex_topk_entry {
    float distance;
    uint64_t doc_addr;     /* document start address in object storage */
    uint32_t doc_length;   /* document byte length */
    uint32_t _pad;         /* padding to 128-bit */
};

struct cortex_read_result {
    int32_t  status;   /* 0=success, negative errno=error */
    uint32_t count;
    struct cortex_topk_entry entries[CORTEX_TOPK_MAX];
};

uint32_t cortex_api_version(void);
int      cortex_driver_init(const char* config_json);
void     cortex_driver_shutdown(void);

struct cortex_read_result cortex_semantic_read(
    uint32_t cluster_id, const void* object_data, uint64_t object_size,
    const float* query_vec, uint32_t query_dim, uint32_t top_k,
    uint32_t metric_type, uint32_t search_mode);

int cortex_semantic_write(
    uint32_t cluster_id, const void* codebook, uint64_t codebook_size,
    const void* pq_payload, uint64_t payload_size, uint32_t vector_count);

int cortex_batch_search(
    uint32_t cluster_id, const void* object_data, uint64_t object_size,
    const float* queries, uint32_t batch_size, uint32_t query_dim,
    uint32_t top_k, uint32_t metric_type, uint32_t search_mode,
    struct cortex_read_result* results);

int      cortex_is_cluster_cached(uint32_t cluster_id);
uint32_t cortex_get_last_cluster_id(void);
```

## Directory Structure

```
Cortex/
├── include/                         # Shared C ABI header
│   └── cortex_driver_api.h          # Used by both driver/ and sys/
├── hw/                              # SmartSSD FPGA device layer
│   ├── Dockerfile                   # Build container (Ubuntu 22.04 + cmake)
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── smartssd_device.h        # Abstract SmartSSDDevice interface
│   ├── src/
│   │   ├── mock_smartssd.h/.cc      # MockSmartSSD: manual IVFPQ ADC search
│   │   └── cortex_hw_mock_exports.cc
│   └── test/                        # Device unit tests + Python fixture generator
├── driver/                          # Host driver (bridges hw/ and sys/)
│   ├── Dockerfile                   # Build container (Ubuntu 22.04 + cmake, no Faiss)
│   ├── CMakeLists.txt
│   ├── src/
│   │   ├── smartssd_driver.h/.cc    # SmartSSDDriver: dlopen + LRU cache
│   │   ├── driver_manager.cc        # Thread-safe singleton DriverManager
│   │   └── cortex_driver_exports.cc # C ABI exports
│   └── test/                        # Driver unit tests (5 tests)
├── sys/                             # Ceph fork (git submodule → yu-zou/ceph)
│   └── src/cortex/
│       ├── rgw/                     # RGW semantic search (Eigen3 + async scatter)
│       ├── osd/                     # OSD ops, PerfCounters, SemanticOpWQ
│       └── loader/                  # CortexDriverLoader (dlopen)
├── sw/                              # Software modules (ARM + Host)
│   ├── arm/
│   │   ├── scheduler.c              # ARM scheduler: NVMe→DMA→FPGA→poll→read
│   │   └── Makefile                 # ARM cross-compile (arm-linux-gnueabihf-gcc)
│   └── host/
│       └── adaptive_nprobe.c        # Adaptive IVF nprobe decision
└── scripts/
    └── build.sh                     # Docker-based build orchestration
```

## Building

Each component builds independently in its own Docker container. There are no cross-component compile-time dependencies — only the runtime `dlopen()` chain links them at execution.

### Prerequisites
- Docker (all components)
- `ceph-build:main.ubuntu22.04` Docker image (for `sys/` component)

### hw/ — Mock Device
```bash
# Build Docker image
docker build -t cortex-hw-build -f hw/Dockerfile .

# Build libcortex_hw_mock.so
mkdir -p build/lib
docker run --rm \
  -v "$(pwd):/cortex" \
  -v "$(pwd)/build/lib:/output" \
  cortex-hw-build bash -c \
    "mkdir -p /build && cd /build && \
     cmake /cortex/hw -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/output && \
     make -j$(nproc) && make install"
# Output: build/lib/libcortex_hw_mock.so
```

### driver/ — Host Driver
```bash
# Build Docker image
docker build -t cortex-driver-build -f driver/Dockerfile .

# Build libcortex_driver.so
docker run --rm \
  -v "$(pwd):/cortex" \
  -v "$(pwd)/build/lib:/output" \
  cortex-driver-build bash -c \
    "mkdir -p /build && cd /build && \
     cmake /cortex/driver -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/output \
       -DCORTEX_HW_LIB_DIR=/output/lib && \
     make -j$(nproc) && make install"
# Output: build/lib/libcortex_driver.so
```

### sys/ — Ceph Fork
Building the Ceph component requires `libeigen3-dev` to be available in the build environment.

```bash
docker run --rm \
  -v "$(pwd)/sys:/ceph" \
  ceph-build:main.ubuntu22.04 bash -c \
    "apt-get install -y libeigen3-dev && \
     cd /ceph/build.u2204 && \
     ninja cortex_rgw radosgw ceph-osd"
```

### Orchestrated build
The provided helper script automates the build of `hw/` and `driver/` components:
```bash
bash scripts/build.sh
# Builds hw/ and driver/ sequentially, stages artifacts to build/lib/
```

## Testing

### hw/ Unit Tests

9 CTest targets (100% pass, no Vitis HLS required):

| Test | What It Verifies |
|---|---|
| `test_mock_device` (7 subtests) | MockSmartSSD ADC search vs golden, TopK order, cache, concurrency |
| `test_device_interface` | SmartSSDDevice polymorphism compliance |
| `test_dm_header` | IVFHeader 64B DRAM byte parsing |
| `test_result_packer` | 128-bit result word pack/unpack round-trip |
| `test_systolic_topk_sw` | Systolic Top-K vs std::partial_sort |
| `test_visited_bitmap` | HNSW visited bitmap operations |
| `test_hnsw_preloader_sw` | HNSW graph binary fixture loading |
| `test_multi_cluster` | Multi-cluster PQ ADC + global Top-K aggregation |
| `test_adc_distance_sw` | ADC L2 distance vs golden fixtures |

```bash
cd hw/build && cmake .. && make -j$(nproc) && ctest --output-on-failure
# 9/9 tests pass
```

### driver/ Unit Tests

5 CTest targets (100% pass, requires `libcortex_hw_mock.so` built first):

| Test | What It Verifies |
|---|---|
| `test_smartssd_driver` | Driver lifecycle: init → write → search → shutdown |
| `test_driver_abi` | C ABI contract: `cortex_api_version()`, dlopen loading |
| `test_driver_hw_integration` | Full search round-trip + cache hit + eviction |
| `test_driver_cache_stress` | LRU cache concurrent access stress |
| `test_driver_errors` | Error paths: null handle, uninitialized, corrupt data |

```bash
cd driver/build
cmake .. -DCORTEX_HW_LIB_DIR=$(pwd)/../../hw/build/lib && make -j$(nproc) && ctest --output-on-failure
# 5/5 tests pass
```

### sys/ Integration Tests (requires Ceph build environment)

8 tests in `sys_ceph/src/test/cortex/` covering Ceph↔Cortex integration:
```bash
docker run --rm -v "$(pwd)/sys:/ceph" ceph-build:main.ubuntu22.04 bash -c \
  "apt-get install -y libeigen3-dev && \
   cd /ceph/build.u2204 && ctest -R cortex --output-on-failure"
# 8/8 tests pass
```

| Test | Covers |
|------|--------|
| `unittest_cortex_driver_loader` | dlopen loading, ABI version check, `-ENOTSUP` fallback |
| `unittest_cortex_semantic_op` | `SemanticOp` encode/decode, Ceph ENCODE macros |
| `unittest_cortex_centroid_cache` | `CentroidLRUCache` LRU eviction, `shared_mutex` |
| `unittest_cortex_perf_counters` | PerfCounter registration and increment |
| `unittest_cortex_extent_map` | BlueStore `get_extent_map()` physical extent lookup |
| `unittest_cortex_semantic_opwq` | `SemanticOpWQ` P0/P1/P2 priority ordering |
| `unittest_cortex_rgw_semantic` | RGW HTTP handler, Eigen3 centroid math, async scatter |
| `unittest_cortex_osd_semantic_ops` | OSD `SEMANTIC_READ`/`SEMANTIC_WRITE` dispatch |

### Integration Test
The binary `ceph_test_cortex_integration` requires a live Ceph cluster with a configured `pool_vector_compute` pool. Following standard Ceph conventions, this test is not built by default and must be invoked manually using a cluster runner. See `sys/src/test/cortex/test_integration.sh` for details.

## Key Design Decisions

### Eigen3 for centroid distance
Centroid selection in RGW utilizes `Eigen::Map<VectorXf>` to leverage SIMD-optimized routines for L2 (`squaredNorm()`) and inner-product (`dot()`) distance calculations. This replaces manual loops with production-grade linear algebra performance.

### Async scatter-gather via aio_operate
The RGW semantic search endpoint fans out requests to all relevant OSDs using the RADOS `aio_operate()` interface with `AioCompletion` callbacks. This ensures consistency with Ceph's native asynchronous I/O patterns and avoids blocking RGW worker threads on sequential network operations.

### Faiss isolation to hw/ only
To maintain a clean ABI and simplify deployment, the Faiss library is treated as a dependency exclusively for the mock hardware layer (`libcortex_hw_mock.so`). The host driver and Ceph OSD have zero linkage to Faiss. The actual ADC logic in the mock is implemented manually to match the future FPGA hardware behavior.

### Fixed-size C ABI — no heap across boundary
The `cortex_read_result` structure uses a fixed array `entries[500]`. This design choice eliminates complex heap management across the ABI boundary, preventing memory leaks and ownership conflicts between the driver and the storage daemon.

### dlopen for runtime loading
The `CortexDriverLoader` uses `dlopen()` at OSD startup. This allows Ceph to maintain zero compile-time dependencies on the Cortex driver. If the driver is missing or if `cortex_api_version()` returns a mismatching value, the OSD logs a warning and gracefully falls back to returning `-ENOTSUP` for semantic operations.

### Three-level locality-aware OpWQ
The `SemanticOpWQ` optimizes for hardware cache locality by classifying operations into three levels:
- **P0**: The request targets the same cluster as the previous operation, ensuring L1/L2 cache hits.
- **P1**: The request targets a different cluster that is already resident in the driver's DRAM cache.
- **P2**: Cold miss; the cluster must be loaded from BlueStore.

## Contributing

### Component independence
`hw/`, `driver/`, and `sys/` are developed as decoupled modules. Developers should ensure that changes in one layer do not introduce compile-time dependencies on others.

### Test requirements
All `unittest_*` suites must pass before any merge. New features should be accompanied by corresponding unit tests in the appropriate layer.

### Guardrails
- **G1**: No Faiss outside `hw/`.
- **G2**: No C++ types (e.g., `std::vector`, `std::string`) across the C ABI defined in `cortex_driver_api.h`.
- **G3**: Modifications to Ceph files must be restricted to the declared integration touchpoints (`PrimaryLogPG.cc`, `OSD.cc`, and related build files).
- **G4**: The `build-with-container.py` orchestration script must not be modified.

### Submodule Management
The `sys/` directory is a git submodule pointing to `yu-zou/ceph.git`. Changes to the system layer must be committed to the submodule repository first, followed by a separate pointer update in the parent Cortex repository.

### Commit conventions
- `feat(component):` for new features
- `fix(component):` for bug fixes
- `chore: update sys submodule to` for submodule bumps
- `test(scope):` for test additions
- `docs:` for documentation updates

## Roadmap

- **Real FPGA SmartSSD device**: Implement `hw/src/fpga_smartssd.cc` targeting the Samsung SmartSSD P4 hardware interface; replace the mock library with `libcortex_hw_fpga.so`.
- ~~**Integration test automation**: Automate `ceph_test_cortex_integration` in CI using an ephemeral `vstart.sh` cluster.~~ **Done** — `bash test/integration/run.sh` boots an ephemeral single-node Ceph cluster in a container, runs the 8-step bash test and all 4 GTests, and exits 0.
- **OSD-side async optimization**: Replace the current synchronous `store->read()` call in `SEMANTIC_READ` with an asynchronous BlueStore callback.
- **Performance benchmarking suite**: Develop a standardized latency and throughput benchmark comparing NDP-based search against host-CPU ADC search.
- **Production deployment guide**: Provide Kubernetes/Rook-Ceph deployment manifests and a SmartSSD device plugin.
