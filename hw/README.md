# Cortex hw/ — Unified IVFPQ/HNSW FPGA Accelerator

## Project Structure

```
hw/
├── .gitignore                       # Excludes HLS build artifacts (*_proj/)
├── run.sh                           # One-click build script (sim/syn/clean modes)
├── run_hls.tcl                      # Full flow: C sim + C synth + RTL export
├── run_sim.tcl                      # C simulation only
├── run_syn.tcl                      # C synthesis + RTL export only
├── run_hnsw.tcl                     # HNSW mode: C sim + C synth
├── run_unified.tcl                  # Unified testbench: C sim for both modes
├── run_topk_test.tcl                # Systolic Top-K unit test (csim + csynth)
├── run_adc_test.tcl                 # ADC distance unit test (csim + csynth)
├── run_cb_load_test.tcl             # Codebook loader unit test
├── run_ivfpq_full_test.tcl          # Full IVFPQ pipeline test
├── run_hnsw_full.tcl                # Full HNSW pipeline test (csim + csynth)
├── run_hnsw_preload_test.tcl        # HNSW DRAM preload test
├── run_hnsw_traverse_test.tcl       # HNSW graph traversal test
├── run_hnsw_dist_test.tcl           # HNSW L2 distance test
├── run_mixed_full.tcl               # Mixed-mode IVFPQ+HNSW test
├── CMakeLists.txt                   # Root CMake (builds hw mock library)
├── Dockerfile                       # Docker-based Vitis HLS build environment
│
├── include/
│   ├── acc_top.h                    # Unified accelerator header (IVFPQ + HNSW)
│   └── smartssd_device.h           # Abstract SmartSSD device interface
│
├── src/
│   ├── acc_top.cpp                  # Unified accelerator implementation
│   ├── mock_smartssd.h              # Mock SmartSSD implementation header
│   ├── mock_smartssd.cc             # Mock SmartSSD (ADC + HNSW software mock)
│   └── cortex_hw_mock_exports.cc    # dlopen factory exports (create/destroy)
│
├── test/
│   ├── CMakeLists.txt               # CTest targets (9 tests registered)
│   ├── test_utils.h                 # Shared test utilities (pack/unpack comparators)
│   ├── fixture_loader.h             # Binary fixture file loader
│   ├── generate_fixtures.py         # IVF fixture generator (requires faiss)
│   ├── generate_hnsw_fixtures.py    # HNSW fixture generator (numpy-based)
│   │
│   ├── test_mock_device.cc          # MockSmartSSD ADC search vs golden (7 subtests)
│   ├── test_device_interface.cc     # SmartSSDDevice polymorphism compliance
│   ├── test_dm_header.cpp           # IVFHeader 64B DRAM parsing
│   ├── test_result_packer.cpp       # 128-bit result word pack/unpack round-trip
│   ├── test_systolic_topk_sw.cpp    # Systolic Top-K insertion sort (software ref)
│   ├── test_visited_bitmap.cpp      # HNSW visited bitmap set/check/clear
│   ├── test_hnsw_preloader_sw.cpp   # HNSW graph preload from binary fixtures
│   ├── test_multi_cluster.cpp       # Multi-cluster aggregated Top-K search
│   ├── test_adc_distance_sw.cpp     # ADC L2 distance vs golden fixtures (3 subtests)
│   │
│   ├── testbench.cpp                # IVFPQ testbench (HLS csim)
│   ├── testbench_hnsw.cpp           # HNSW mode testbench (HLS csim)
│   ├── testbench_unified.cpp        # Dual-mode testbench (HLS csim)
│   ├── testbench_topk.cpp           # Systolic Top-K testbench (HLS csim + csynth)
│   ├── testbench_adc.cpp            # ADC distance testbench (HLS csim + csynth)
│   ├── testbench_cb_load.cpp        # Codebook load testbench (HLS csim)
│   ├── testbench_ivfpq_full.cpp     # Full IVFPQ pipeline testbench (HLS csim)
│   ├── testbench_hnsw_full.cpp      # Full HNSW pipeline testbench (HLS csim + csynth)
│   ├── testbench_hnsw_preload.cpp   # HNSW preload testbench (HLS csim)
│   ├── testbench_hnsw_traverse.cpp  # HNSW traversal testbench (HLS csim)
│   ├── testbench_hnsw_dist.cpp      # HNSW L2 distance testbench (HLS csim)
│   ├── testbench_mixed_full.cpp     # Mixed-mode testbench (HLS csim)
│   │
│   └── fixtures/                    # Test data (16 .bin files + metadata)
│       ├── cluster_0.bin  (4112B, N=100)     ├── codebook.bin      (2048B)
│       ├── cluster_1.bin  (2112B, N=0)       ├── queries.bin       (640B)
│       ├── cluster_2.bin  (2112B, N=0)       ├── vectors.bin       (400B)
│       ├── cluster_3.bin  (2112B, N=0)       ├── centroids.bin     (512B)
│       ├── golden_distances.bin (2000B)       ├── golden_topk.bin  (600B)
│       └── hnsw/  (7 files: graph header, vectors, adjacency, queries, etc.)
│
└── *_proj/                          # Vitis HLS build output (gitignored)
    ├── adc_test_proj/               # compute_engine synthesis
    ├── topk_test_proj/              # systolic_topk_insert synthesis
    ├── hnsw_proj/                   # hnsw_search_engine synthesis
    ├── hnsw_full_proj/              # Full HNSW pipeline synthesis
    ├── ivfpq_full_proj/             # Full IVFPQ pipeline synthesis
    ├── mixed_full_proj/             # Mixed-mode synthesis
    └── unified_proj/                # Dual-mode synthesis
```

## What This Does

A **unified FPGA accelerator** supporting two vector search modes, deployed as two separate bitstreams:

| Mode | Algorithm | Storage | Fmax | LUT | Key Loop II |
|---|---|---|---|---|---|
| Mode 0 (IVFPQ) | IVF + Product Quantization | PQ compressed codes (M bytes/vector) | 278 MHz | 83% | PQ_PROCESS II=1 |
| Mode 1 (HNSW) | HNSW Graph Traversal | Full vectors in BRAM | 245 MHz | 51% | SEQUENTIAL_TRAVERSE II=1 |

**Shared hardware**: Systolic Top-K sorting array (500 cells, Fmax=1246MHz), L2 distance calculator

**Mode switching**: Via `ComputeMeta.search_mode` field (0=IVFPQ, 1=HNSW). IVFPQ and HNSW are synthesized as **two independent bitstreams** — they cannot co-exist in one FPGA due to Vitis HLS DATAFLOW limitations. The FPGA supports partial reconfiguration (~100ms switch time).

## Key Design Features

1. **Systolic Top-K**: 500 independent pipeline cells, integer comparison on FP32 bit patterns for non-negative L2 distances. II=1 throughput.

2. **IVFPQ Mode**: M×Ds parallel ADC distance computation. Codebook BRAM partitioned for 128-way parallel reads. PQ codes packed in 512-bit beats with padding.

3. **HNSW Mode**: Graph data pre-loaded into partitioned BRAM (vectors + adjacency). Sequential traversal with neighbor expansion. Avoids BFS queue for synthesis simplicity.

4. **128-bit Result Format**: dist(FP32,32b) + doc_addr(64b) + doc_len(32b). 1 beat per result through FIFO_RES.

## Build & Run

```bash
# Full flow (IVFPQ mode):
cd hw/
./run.sh

# HNSW mode only:
source /tools/Xilinx/Vitis_HLS/2021.2/settings64.sh
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
vitis_hls -f run_hnsw.tcl

# Unified both-modes test:
vitis_hls -f run_unified.tcl
```

## Synthesis Results

| Metric | IVFPQ (TOPK=500) | HNSW (200 nodes) |
|---|---|---|
| Fmax | 278.32 MHz | 245.52 MHz |
| LUT | 498,746 (83%) | 308,405 (51%) |
| FF | 330,180 (27%) | 334,764 (27%) |
| BRAM | 128 (6%) | 150 (7%) |
| DSP | 932 (26%) | 464 (13%) |
| PQ_PROCESS / TRAVERSE II | 1 (depth 158) | 1 (depth 144) |

Target device: Xilinx VU5P (xcvu5p-flva2104-1-e)
Clock: 200 MHz (5 ns period)
Tool: Vitis HLS 2021.2

## Running Tests

### Test Suite Overview

| Layer | Count | Environment | What It Tests |
|---|---|---|---|
| **HLS C Simulation** | 5 testbenches | Vitis HLS `csim_design` | Functional correctness of FPGA logic |
| **HLS C Synthesis** | 3 modules | Vitis HLS `csynth_design` | RTL generation, timing closure |
| **HW CTest (unit)** | 9 targets | cmake + GTest + g++ | Software-level algorithm verification |
| **Driver CTest (integration)** | 5 targets | cmake + GTest + g++ | Driver ↔ Mock HW boundary |
| **E2E** | 1 test | g++ + driver .so | Full pipeline: init → write → search → verify |

### Prerequisites

| Test Suite | Required Tools |
|---|---|
| HLS C Sim / C Synth | Vitis HLS 2021.2 |
| HW CTest | cmake ≥ 3.16, gcc ≥ 9, make |
| Driver CTest | cmake ≥ 3.16, gcc ≥ 9, `libcortex_hw_mock.so` built |
| E2E | Driver .so built, fixtures generated |

### HLS C Simulation Tests (5/5 PASS)

Vitis HLS `csim_design` functionally verifies the FPGA logic. TCL scripts and testbenches are in `hw/` and `hw/test/` respectively.

| TCL Script | Testbench | What It Tests |
|---|---|---|
| `run_topk_test.tcl` | `testbench_topk.cpp` | Systolic Top-K: 5 distances inserted, verify ascending sort |
| `run_adc_test.tcl` | `testbench_adc.cpp` | ADC L2 distance: 8 sub-tests (centroid match, zero codebook, etc.) |
| `run_ivfpq_full_test.tcl` | `testbench_ivfpq_full.cpp` | Full IVFPQ pipeline: DM reads IVF cluster + compute engine ADC + systolic Top-K |
| `run_hnsw_full.tcl` | `testbench_hnsw_full.cpp` | Full HNSW pipeline: preload + traversal + systolic Top-K |
| `run_unified.tcl` | `testbench_unified.cpp` | Dual-mode: both IVF cluster and HNSW graph in DRAM, switch modes |

```bash
source /tools/Xilinx/Vitis_HLS/2021.2/settings64.sh
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
vitis_hls -f run_topk_test.tcl   # Individual test
bash ../scripts/build.sh test-hls  # All HLS tests
```

Additional testbenches available: `testbench_cb_load.cpp` (codebook load), `testbench_hnsw_dist.cpp` (HNSW L2 distance), `testbench_hnsw_preload.cpp` (graph preload), `testbench_hnsw_traverse.cpp` (graph traversal), `testbench_mixed_full.cpp` (mixed-mode full pipeline).

### HLS C Synthesis Tests (3/3 PASS)

`csynth_design` generates Vivado RTL and verifies timing.

| TCL Script | Top Function | Fmax | LUT | Critical Loop II | Result |
|---|---|---|---|---|---|
| `run_adc_test.tcl` (csynth) | compute_engine | 278 MHz | 83% | PQ_PROCESS II=1 | PASS |
| `run_hnsw_full.tcl` (csynth) | hnsw_search_engine | 245 MHz | 51% | TRAVERSE II=1 | PASS |
| `run_topk_test.tcl` (csynth) | systolic_topk_insert | 1246 MHz | — | II=1 | PASS |

### HW CTest — Software Unit Tests (9/9 PASS)

Pure C++ (no Vitis HLS required). Validates algorithm correctness against golden references.

```bash
cd hw/build && cmake .. && make -j$(nproc) && ctest --output-on-failure
```

| Test | Source | Sub-tests | What It Verifies |
|---|---|---|---|
| `test_mock_device` | `test_mock_device.cc` | 7 | MockSmartSSD ADC vs golden, TopK order, error paths, cache, concurrency |
| `test_device_interface` | `test_device_interface.cc` | 1 | SmartSSDDevice polymorphism compliance |
| `test_dm_header` | `test_dm_header.cpp` | 5 | IVFHeader 64B DRAM byte parsing (LE fields) |
| `test_result_packer` | `test_result_packer.cpp` | 4 | 128-bit result word pack/unpack round-trip |
| `test_systolic_topk_sw` | `test_systolic_topk_sw.cpp` | 5 | Systolic Top-K vs std::partial_sort reference |
| `test_visited_bitmap` | `test_visited_bitmap.cpp` | 5 | HNSW visited bitmap set/check/clear operations |
| `test_hnsw_preloader_sw` | `test_hnsw_preloader_sw.cpp` | 5 | HNSW graph binary fixture loading |
| `test_multi_cluster` | `test_multi_cluster.cpp` | 4 | Multi-cluster PQ ADC + global Top-K aggregation |
| `test_adc_distance_sw` | `test_adc_distance_sw.cpp` | 3 | ADC L2 distance vs golden distances (ε=5e-4) |

Expected output:
```
100% tests passed, 0 tests failed out of 9
```

**Fixtures** (all 16 .bin files in `hw/test/fixtures/`):
- IVF: cluster_0.bin (4112B, N=100), cluster_1~3.bin (2112B, N=0), codebook.bin (2048B), queries.bin (640B), vectors.bin (400B), golden_distances.bin (2000B), golden_topk.bin (600B), centroids.bin (512B), metadata.json
- HNSW: graph_header.bin (16B), vectors.bin (10240B), adjacency.bin (2560B), num_nbrs.bin (20B), queries.bin (2560B), golden_topk.bin (400B), metadata.json

### Driver CTest — Integration Tests (5/5 PASS)

Tests the boundary between the Cortex driver and the HW mock library via `dlopen`.

```bash
cd driver/build
cmake .. -DCORTEX_HW_LIB_DIR=$(pwd)/../../hw/build/lib && make -j$(nproc) && ctest --output-on-failure
```

| Test | Source | What It Verifies |
|---|---|---|
| `test_smartssd_driver` | `test_smartssd_driver.cc` | Driver lifecycle: init → write → search → shutdown |
| `test_driver_abi` | `test_driver_abi.cc` | C ABI contract: `cortex_api_version()`, `cortex_semantic_read()` |
| `test_driver_hw_integration` | `test_driver_hw_integration.cc` | Full search round-trip + cache hit + eviction |
| `test_driver_cache_stress` | `test_driver_cache_stress.cc` | LRU cache concurrent access stress |
| `test_driver_errors` | `test_driver_errors.cpp` | Error paths: null handle, uninitialized, corrupt data |

Note: `test_abi_boundary` is disabled — it requires Vitis HLS headers (`ap_int.h`) for struct layout verification.

Expected output:
```
100% tests passed, 0 tests failed out of 5
```

### End-to-End Test

A minimal E2E test exercises the complete query pipeline:

```bash
cd driver/build
g++ -std=c++17 -I../../include -I../../hw/include \
    ../../test/e2e/test_e2e_simple.cpp \
    -L./lib -lcortex_driver -Wl,-rpath,./lib -o /tmp/test_e2e
LD_LIBRARY_PATH=./lib /tmp/test_e2e
```

Also available: `test/e2e/test_e2e.sh` (1146-line bash script with 8-step build+test pipeline, requires Docker).

### Ceph Integration Test (requires Docker)

For full system verification including Ceph object storage backend. See `test/integration/run.sh`.
Requires: Docker, `ceph-build:main.ubuntu22.04` image (~50GB disk, 32GB+ RAM).
