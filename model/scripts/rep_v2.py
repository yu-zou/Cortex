#!/usr/bin/env python3
"""Workload-v2 system runner: per-tenant private corpora + shared public corpus.

Env knobs:
  V2_NPROBE_PRIV=28  V2_NPROBE_PUB=4  V2_K=8  V2_SCALE=5000
  V2_NSTAR=404       V2_HYBRID=1      V2_ONLY=fifo|cb4
  V2_SEED_BASE=22101 V2_TAG=v2
"""
import json
import os
import pathlib
import sys
import zlib

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from exp_osd_sched import (  # noqa: E402
    build_fpga_model,
    ceph_placement,
    run_simulation,
    SyntheticQuery,
)
from strata_sim.sched.osd_timing import OSDHardwareProfile as HP  # noqa: E402
from strata_sim.sched.query_queue import ScoreWeights  # noqa: E402
from strata_sim.sched.rgw_scheduler import RGWScoreWeights  # noqa: E402

ROOT = pathlib.Path(".")
NP = int(os.environ.get("V2_NPROBE_PRIV", "28"))
NQ = int(os.environ.get("V2_NPROBE_PUB", "4"))
K = int(os.environ.get("V2_K", "8"))
SCALE = float(os.environ.get("V2_SCALE", "5000"))
NUM_OSDS = int(os.environ.get("NUM_OSDS", "20"))
RERANK = int(os.environ.get("RERANK", "64"))
NSTAR = float(os.environ.get("V2_NSTAR", "404"))
TENANCY = os.environ.get("V2_TENANCY", "mix")   # mix | d3
NPRIV_TENANTS = int(os.environ.get("V2_PRIVATE_N", "43"))
NPROBE32 = int(os.environ.get("V2_NPROBE", "32"))
HYBRID = os.environ.get("V2_HYBRID", "1") == "1"
TAG = os.environ.get("V2_TAG", "v2")
BASE_BUILD = float(os.environ.get("BASE_BUILD", "1615"))
DIRECT_CYC = 6.0

SPEC_PATH = os.environ.get("V2_SPEC", "data/derived/workload_v2/spec_window.npz")
spec = np.load(SPEC_PATH, allow_pickle=True)
tenant_names = [str(x) for x in spec["tenant_names"]]
offsets = spec["offsets"]
priv = spec["private_sizes"]
pub = spec["pub_sizes"]
PUB_OFF, PRIV_OFF = 0, len(pub)
idx = {t: i for i, t in enumerate(tenant_names)}

events = []
with open("data/derived/trace/events.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 500:
            break
        events.append(json.loads(line))
t0 = events[0]["time_ns"]
span = events[-1]["time_ns"] - t0

fpga_base = build_fpga_model(ROOT)


class CountingFpga:
    """Wrap the table model and count service-path cycles (predict_cycles)."""

    def __init__(self, base):
        self.base = base
        self.svc_cycles = 0.0
        self.svc_calls = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def scan_cycles_per_doc(self, n_docs=None):
        return self.base.scan_cycles_per_doc(n_docs)

    def predict_cycles(self, n_docs, top_k, *, init_runs=1, codebook_resident=False):
        c = self.base.predict_cycles(n_docs, top_k, init_runs=init_runs,
                                     codebook_resident=codebook_resident)
        self.svc_cycles += c
        self.svc_calls += 1
        return c

    def predict_ns(self, n_docs, top_k, *, init_runs=1, codebook_resident=False):
        return self.base.predict_ns(n_docs, top_k, init_runs=init_runs,
                                    codebook_resident=codebook_resident)


class HybridFpga(CountingFpga):
    """Unified engine cost model: direct mode (no build, 6 cyc/doc) when n < N*."""

    def __init__(self, base, n_star, build=BASE_BUILD, direct=DIRECT_CYC):
        super().__init__(base)
        self.n_star = n_star
        self.build = build
        self.direct = direct

    def _direct_cycles(self, n_docs, top_k, codebook_resident):
        f0 = self.base.predict_cycles(0, top_k, codebook_resident=codebook_resident)
        return f0 - self.build + self.direct * n_docs

    def predict_cycles(self, n_docs, top_k, *, init_runs=1, codebook_resident=False):
        if n_docs and n_docs < self.n_star:
            c = self._direct_cycles(n_docs, top_k, codebook_resident)
        else:
            c = self.base.predict_cycles(n_docs, top_k, init_runs=init_runs,
                                         codebook_resident=codebook_resident)
        self.svc_cycles += c
        self.svc_calls += 1
        return c

    def predict_ns(self, n_docs, top_k, *, init_runs=1, codebook_resident=False):
        if n_docs and n_docs < self.n_star:
            return self._direct_cycles(n_docs, top_k, codebook_resident) * self.base.clock_ns
        return self.base.predict_ns(n_docs, top_k, init_runs=init_runs,
                                    codebook_resident=codebook_resident)


queries = []
csizes = {}
_CLS = spec["cls"] if "cls" in spec.files else None
_priv_tenants = set()
if TENANCY == "d3":
    _S = spec["S"]
    _order = np.argsort(_S)[::-1]
    _priv_tenants = set(int(i) for i in _order[:NPRIV_TENANTS])
    print("D3 tenancy: %d private tenants (largest S), %d shared"
          % (len(_priv_tenants), len(tenant_names) - len(_priv_tenants)))
for r in range(K):
    rng = np.random.RandomState(int(os.environ.get("V2_SEED_BASE", "22101")) + r)
    base_t = r * span
    for e in events:
        t = e["tenant"]
        i = idx[t]
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        if TENANCY == "cloud":
            if _CLS is not None and int(_CLS[i]) == 1:
                # large tenant: shared public index, shared codebook
                k2 = min(NPROBE32, len(pub))
                ids = [PUB_OFF + int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
                cb_id = "cb-pub"
            else:
                # small tenant: own small index, own codebook
                k1 = min(NPROBE32, hi - lo)
                ids = [PRIV_OFF + int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
                cb_id = "cb-priv-" + t
        elif TENANCY == "d3" and i not in _priv_tenants:
            # shared tenant: probe ONLY the public corpus
            k2 = min(NPROBE32, len(pub))
            ids = [PUB_OFF + int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
            cb_id = "cb-pub"
        elif TENANCY == "d3":
            # private tenant: probe ONLY its own corpus
            k1 = min(NPROBE32, hi - lo)
            ids = [PRIV_OFF + int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
            cb_id = "cb-priv-" + t
        else:
            k1 = min(NP, hi - lo)
            ids = [PRIV_OFF + int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
            k2 = min(NQ, len(pub))
            ids += [PUB_OFF + int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
            cb_id = "cb-priv-" + t
        for cid in ids:
            if cid not in csizes:
                csizes[cid] = int(pub[cid]) if cid < PRIV_OFF else int(priv[cid - PRIV_OFF])
        z = zlib.crc32(t.encode()) % 100
        slo = 5_000_000 if z < 50 else (7_000_000 if z < 80 else 11_000_000)
        a = int(round((base_t + e["time_ns"] - t0) / SCALE))
        queries.append(SyntheticQuery(query_id="q%05d" % (r * len(events) + e["event_index"]),
                                      tenant=t, arrival_ns=a, deadline_ns=a + slo,
                                      cluster_ids=tuple(sorted(ids)), top_k=10,
                                      codebook_id=cb_id))
queries.sort(key=lambda q: (q.arrival_ns, q.query_id))
allc = set()
for q in queries:
    allc.update(q.cluster_ids)
if os.environ.get("V2_PLACEMENT", "roundrobin") == "crush":
    from strata_sim.sched.crush import crush_placement
    pl = crush_placement(NUM_OSDS, sorted(allc), replication=3)
    print("PLACEMENT=crush (straw2, 5 hosts x 4 OSDs)")
else:
    pl = ceph_placement(NUM_OSDS, sorted(allc), replication=3)
rate = len(queries) / (queries[-1].arrival_ns / 1e9)
print("TAG=%s tenancy=%s probes=%d queries=%d distinct_clusters=%d span=%.2fs rate=%.0f q/s"
      % (TAG, TENANCY, (NPROBE32 if TENANCY == "d3" else NP + NQ),
         len(queries), len(allc), queries[-1].arrival_ns / 1e9, rate))

VPC = int(np.median(priv))
RGW_W = RGWScoreWeights(cache=float(os.environ.get("RGW_CACHE", "3.0")),
                        load=float(os.environ.get("RGW_LOAD", "1.5")),
                        fpga=float(os.environ.get("RGW_FPGA", "1.0")),
                        primary=float(os.environ.get("RGW_PRIMARY", "0.5")),
                        queue_depth_cap=int(os.environ.get("RGW_CAP", "32")))
CFG = [
    ("fifo", "fifo", None),
    ("cb4", "multifactor",
     ScoreWeights(aging=4.0, codebook=1.0, ddr=3.0, slow_first=0, deadline=0.5, cost=0.25)),
    ("cbf", "multifactor",
     ScoreWeights(aging=2.0, codebook=3.0, ddr=1.0, slow_first=0, deadline=0.5, cost=0.25)),
    ("ddf", "multifactor",
     ScoreWeights(aging=2.0, codebook=0.0, ddr=4.0, slow_first=0, deadline=0.5, cost=0.25)),
    ("mixf", "multifactor",
     ScoreWeights(aging=2.0, codebook=2.0, ddr=2.0, slow_first=0, deadline=0.5, cost=0.25)),
]
CFG.append(("mf", "multifactor", ScoreWeights(
    aging=float(os.environ.get("W_AGING", "4.0")),
    codebook=float(os.environ.get("W_CODEBOOK", "1.0")),
    ddr=float(os.environ.get("W_DDR", "3.0")),
    slow_first=float(os.environ.get("W_SLOW_FIRST", "0.0")),
    slo_window_ns=int(float(os.environ.get("SLO_WINDOW_NS", "100e6"))),
    starvation_cap_ns=int(float(os.environ.get("STARVE_CAP_NS", "200e6"))),
    tenant_penalty=float(os.environ.get("TENANT_PENALTY", "0.5")),
    fairness_factor=float(os.environ.get("FAIRNESS_FACTOR", "1.5")),
    deadline_feasible=os.environ.get("DEADLINE_FEASIBLE", "0") == "1",
    deadline=float(os.environ.get("W_DEADLINE", "0.5")),
    cost=float(os.environ.get("W_COST", "0.25")),
    byte=float(os.environ.get("W_BYTE", "0.0")),
    cgb=float(os.environ.get("W_CGB", "0.0")))))
_only = os.environ.get("V2_ONLY", "").strip().lower()
if _only:
    CFG = [c for c in CFG if c[0] == _only]


def run(strat, wt, fpga):
    import dataclasses as _dc
    _prof = os.environ.get("OSD_MODEL_PROFILE")
    _cache_kw = dict(
        vectors_per_cluster=VPC,
        pq_cache_bytes=int(float(os.environ.get("V2_PQ_CACHE_GIB", "3.0")) * 1024 ** 3),
        raw_cache_bytes=int(2.0 * 1024 ** 3), cache_raw_vectors=False,
        bram_capacity_slots=int(os.environ.get("BRAM_SLOTS", "1")))
    hw = (_dc.replace(HP.from_model(_prof), **_cache_kw) if _prof
          else HP(**_cache_kw))
    _bddr = os.environ.get("B_DDR_GBPS")
    if _bddr:
        hw = _dc.replace(hw, board_ddr_read_bytes_per_ns=float(_bddr),
                         ddr_bytes_per_ns=float(_bddr))
    _bssd = os.environ.get("B_SSD_GBPS")
    if _bssd:
        # explicit SSD feed override ("current implementation" = 0.735)
        hw = _dc.replace(hw, ssd_to_ddr_bytes_per_ns=float(_bssd))
    return run_simulation(queries, num_osds=NUM_OSDS, strategy=strat,
                          window_width=int(os.environ.get("V2_WINDOW", "64")),
                          fpga_model=fpga, placement=pl, hardware=hw, osd_weights=wt,
                          rgw_weights=RGW_W, cluster_sizes=csizes, adaptive=False,
                          rerank_per_cluster=RERANK)


print("\n%-12s %7s %6s %8s %8s %8s %8s %11s" %
      ("config", "SLO%", "viol", "P50ms", "P90ms", "P99ms", "mean", "FPGA Mcyc"))
res = {}
for label, strat, wt in CFG:
    modes = [("table", CountingFpga(fpga_base))]
    if HYBRID:
        modes.append(("hybrid", HybridFpga(fpga_base, NSTAR)))
    for mtag, fm in modes:
        c, _info = run(strat, wt, fm)
        if _info:
            _pt = _info.get("per_tenant", {})
            _w = [v["wait_p50_ns"] for v in _pt.values() if v.get("wait_p50_ns") is not None]
            def _j(v):
                v=[float(x) for x in v]
                return (sum(v)**2)/(len(v)*sum(x*x for x in v)) if v and sum(x*x for x in v) else 1.0
            print("JAIN %s:%s jain=%.4f demand=%.4f waitp50=%.4f" % (
                label, mtag, _info.get("jain_index", -1), _info.get("jain_demand_normalized") or -1, _j(_w)))
        _dump = os.environ.get("V2_DUMP_LAT")
        if _dump:
            np.savez("%s_%s_%s.npz" % (_dump, label, mtag),
                     qid=np.array([x.query.query_id for x in c]),
                     tenant=np.array([x.query.tenant for x in c]),
                     lat=np.array([x.latency_ns for x in c]),
                     nsub=np.array([len(x.timings) for x in c]),
                     fpga=np.array([fm.svc_cycles]))
        _ev = sum(t.ddr_evictions for x in c for t in x.timings)
        print("EVICT %s:%s total=%d" % (label, mtag, _ev))
        viol = sum(1 for x in c if x.finish_ns > x.query.deadline_ns)
        lat = np.array([x.latency_ns for x in c])
        res["%s:%s" % (label, mtag)] = dict(
            slo=viol / len(c) * 100, viol=viol,
            p50=np.percentile(lat, 50) / 1e6, p90=np.percentile(lat, 90) / 1e6,
            p99=np.percentile(lat, 99) / 1e6, mean=lat.mean() / 1e6,
            fpga=fm.svc_cycles / 1e6)
        print("%-12s %7.2f %6d %8.3f %8.3f %8.3f %8.3f %11.1f"
              % (label + ":" + mtag, viol / len(c) * 100, viol,
                 np.percentile(lat, 50) / 1e6, np.percentile(lat, 90) / 1e6,
                 np.percentile(lat, 99) / 1e6, lat.mean() / 1e6, fm.svc_cycles / 1e6))

if HYBRID:
    print("\nhybrid vs table-only (same workload):")
    for label, _s, _w in CFG:
        a = res[label + ":table"]
        b = res[label + ":hybrid"]
        print("  %-6s FPGA cycles %+.2f%% | P99 %+.3fms | P50 %+.3fms | SLO %+.2fpp"
              % (label, (b["fpga"] - a["fpga"]) / a["fpga"] * 100,
                 b["p99"] - a["p99"], b["p50"] - a["p50"], b["slo"] - a["slo"]))

json.dump(res, open("/tmp/%s.json" % TAG, "w"), indent=2)
print("saved -> /tmp/%s.json" % TAG)
