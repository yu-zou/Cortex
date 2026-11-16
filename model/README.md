# model/ — Cortex performance model

Analytical + discrete-event model of the Cortex retrieval path: client →
RGW (primary-replica routing) → OSD queue → 6-stage OSD service (mapping /
SSD load / codebook / FPGA scan / raw rerank / return).

The model is pure Python (3.8+, no dependency on the C/C++ sources and no
board required), but it is **not self-contained**: the DES needs three
corpus-derived workload files that are too large to live in this repository
(see *Inputs*). Without them only the analytical formulas and the unit tests
can run.

## Layout

```
model/
├── params.py                # ★ single source of truth for every constant
├── strata_sim/              # the model package
│   ├── contracts/schema.py  # event schema, TILE_MAX_DOCS=8192
│   ├── des/                 # deterministic DES engine + FPGA service model
│   ├── sched/               # OSD 6-stage timing, query queue/score, RGW, CRUSH
│   ├── model/strata_model.py# analytical formulas (T_load/T_scan/T_raw/...)
│   ├── functional/ivfpq.py  # IVFPQ index fixture (recall checks)
│   ├── metrics/requests.py  # arrival trace + percentile helpers
│   └── safety.py            # path containment guards
├── scripts/
│   ├── rep_v2.py            # main runner: workload → DES → metrics (env-knob driven)
│   ├── exp_osd_sched.py     # DES kernel: run_simulation, ceph_placement
│   ├── gen_1b_mt.py         # 1B-scale multi-tenant workload generator
│   └── gen_2p5b_v2.py       # 2.39B-scale workload generator
└── tests/
    └── test_time_model.py   # numeric checks against RTL calibration points
```

## Inputs (not in this repository)

All are derived from the indexed corpus and are excluded by `.gitignore`
(`*.npz`, `model/data/`):

| Path (relative to `model/`) | Content | Producer |
|---|---|---|
| `data/derived/workload_v2/spec_window.npz` | cluster sizes, per-tenant cluster membership | data preparation step |
| `data/derived/trace/events.jsonl` | client arrival trace, one JSON object per event | trace export |
| `data/derived/msmarco_9m/assign.npy` | cluster id per indexed vector (for the tenant split) | index build |

`scripts/gen_1b_mt.py` reads all three and emits the 0.96B multi-tenant spec;
`scripts/gen_2p5b_v2.py` reads that spec plus `assign.npy` and emits the
2.39B one (`data/derived/workload_1b_mt/*.npz`, `data/derived/workload_2p5b/*.npz`).

## What the model assumes

- **Raw vectors live on the board SSD**, not in DDR. Exact rerank fetches `m`
  candidates per cluster (`RERANK`, default 64) on demand and pays the SSD
  path (`L_cmd + C/B_ssd + C/B_ddr`); there is no raw-residency assumption.
  The 2 GiB raw region is only an LRU budget for those fetches.
- **Rerank runs on the workstation host CPU**, `cores_per_host / boards_per_host`
  = 2 cores per OSD, after the candidates cross PCIe. The board's ARM core is
  not modelled.
- A subquery is complete when the OSD returns its top-k; the client waits for
  all 32 sub-queries (barrier), so query latency is the max, not the sum.

## Run

`rep_v2.py` is env-knob driven (no CLI flags) and must be run from `model/`:

```bash
cd model
# 4,000-query multi-tenant run, FIFO, CRUSH placement:
V2_PLACEMENT=crush python3 scripts/rep_v2.py
# knobs (env): V2_SCALE, V2_K, NUM_OSDS, V2_PLACEMENT=crush|roundrobin,
#   B_SSD_GBPS, B_DDR_GBPS, BRAM_SLOTS, V2_WINDOW, V2_SPEC,
#   OSD_PIPELINE=0 (serial, default) | 1 (overlap), OSD_TABLE_REUSE=1 ...
```

Sanity test — this one needs no inputs, only the standard library:

```bash
python3 tests/test_time_model.py        # 7 numeric checks
```

Regenerate the shared C header from `params.py` (do NOT hand-edit the header):

```bash
python3 model/params.py --emit-h > include/cortex_model_params.h   # from repo root
```

## What is measured vs assumed

`params.py` marks every constant with its provenance. The hard numbers:

| Value | Source |
|---|---|
| c_proc = 2 cycles/doc | RTL co-sim, AXI-512 redesign asymptote |
| TOPK_INIT = 500 cycles | HLS loop trip count, II=1 |
| codebook stream = 12,299 cycles | RTL cold-vs-warm co-sim delta (n=17) |
| pipeline fixed = 2,211 cycles | fitted residual (cold base − codebook) |
| T_base = 2,711 cycles = 15.06 µs | sum of the two above (warm call) |
| codebook 786,432 B = 143 µs @ 5.5 GB/s | board-measured DDR bandwidth |
| B_ddr = 5.5 GB/s | board N-sweep fit, error ≤ 6.1% |
| PQ entry = 128 B | board: 2 × 512-bit beats |

Assumed (not yet measured): B_ssd target 6.4 GB/s, B_xdma 20 GB/s,
L_cmd 20 µs, L_pcie 2 µs, L_cpu 100 ns, SLO 5/7/11 ms.
Current-implementation measurement, used only in explicit runs: B_ssd 0.735 GB/s.
