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
from dataclasses import replace

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
from load_measurement import uniform_arrivals, window_metrics

ROOT = pathlib.Path(".")
NP = int(os.environ.get("V2_NPROBE_PRIV", "28"))
NQ = int(os.environ.get("V2_NPROBE_PUB", "4"))
K = int(os.environ.get("V2_K", "1"))
TOP_K = int(os.environ.get("V2_TOP_K", "10"))
SCALE = float(os.environ.get("V2_SCALE", "5000"))
NUM_OSDS = int(os.environ.get("NUM_OSDS", "20"))
RERANK = int(os.environ.get("RERANK", "64"))
NSTAR = float(os.environ.get("V2_NSTAR", "404"))
TENANCY = os.environ.get("V2_TENANCY", "cloud")
NPRIV_TENANTS = int(os.environ.get("V2_PRIVATE_N", "43"))
NPROBE32 = int(os.environ.get("V2_NPROBE", "32"))
HYBRID = os.environ.get("V2_HYBRID", "1") == "1"
TAG = os.environ.get("V2_TAG", "v2")
BASE_BUILD = float(os.environ.get("BASE_BUILD", "1615"))
DIRECT_CYC = 6.0

SPEC_PATH = os.environ.get("V2_SPEC", "data/derived/workload_v2/spec_window.npz")
spec = np.load(SPEC_PATH, allow_pickle=True)
tenant_names = [str(x) for x in spec["tenant_names"]]
user_ids = spec["user_ids"]          # tenant -> user index
S_per_user = spec["S"]               # user -> docs
nlist_per_user = spec["nlist"]       # user -> nlist
offsets = spec["offsets"]            # user -> cluster offset
all_cluster_sizes = spec["all_cluster_sizes"]  # flat cluster sizes
idx = {t: i for i, t in enumerate(tenant_names)}
# tenant -> user -> cluster range (lo, hi)
tenant_to_user = {}
for t, ui in zip(tenant_names, user_ids):
    tenant_to_user[t] = int(ui)
# Build csizes from all_cluster_sizes
# Populate only touched clusters below; constructing 32M unused dict entries
# costs memory and startup time without changing any simulated request.
csizes = {}
# Build tenant -> cluster range
def tenant_cluster_range(tenant_name):
    ui = tenant_to_user[tenant_name]
    lo = int(offsets[ui])
    hi = int(offsets[ui + 1])
    return lo, hi

events = []
event_offset = int(os.environ.get('V2_EVENTS_OFFSET', '0'))
if event_offset < 0:
    raise ValueError('negative event offset')
with open("data/derived/trace/events.jsonl") as f:
    for i, line in enumerate(f):
        if i < event_offset:
            continue
        if len(events) >= int(os.environ.get("V2_EVENTS", "500")):
            break
        events.append(json.loads(line))
if len(events) != int(os.environ.get('V2_EVENTS', '500')):
    raise ValueError('requested event segment is incomplete')
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
        if t not in idx:
            raise ValueError("trace tenant missing from spec: " + t)
        lo, hi = tenant_cluster_range(t)
        if "user_ids" in spec.files:
            k1 = min(NPROBE32, hi - lo)
            ids = [int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
            cb_id = "cb-user-%d" % tenant_to_user[t]
        elif TENANCY == "cloud":
            if _CLS is not None and int(_CLS[i]) == 1:
                # large tenant: shared public index, shared codebook
                k2 = min(NPROBE32, len(pub))
                ids = [int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
                cb_id = "cb-pub"
            else:
                # small tenant: own small index, own codebook
                k1 = min(NPROBE32, hi - lo)
                ids = [int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
                cb_id = "cb-priv-" + t
        elif TENANCY == "d3" and i not in _priv_tenants:
            # shared tenant: probe ONLY the public corpus
            k2 = min(NPROBE32, len(pub))
            ids = [int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
            cb_id = "cb-pub"
        elif TENANCY == "d3":
            # private tenant: probe ONLY its own corpus
            k1 = min(NPROBE32, hi - lo)
            ids = [int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
            cb_id = "cb-priv-" + t
        else:
            k1 = min(NP, hi - lo)
            ids = [int(p) for p in (lo + rng.choice(hi - lo, size=k1, replace=False))]
            k2 = min(NQ, len(pub))
            ids += [int(p) for p in rng.choice(len(pub), size=k2, replace=False)]
            cb_id = "cb-priv-" + t
        for cid in ids:
            if cid not in csizes:
                csizes[cid] = int(all_cluster_sizes[cid]) 
        z = zlib.crc32(t.encode()) % 100
        slo = 5_000_000 if z < 50 else (7_000_000 if z < 80 else 11_000_000)
        a = int(round((base_t + e["time_ns"] - t0) / SCALE))
        queries.append(SyntheticQuery(query_id="q%05d" % (r * len(events) + e["event_index"]),
                                      tenant=t, arrival_ns=a, deadline_ns=a + slo,
                                      cluster_ids=tuple(sorted(ids)), top_k=TOP_K,
                                      codebook_id=cb_id))
queries.sort(key=lambda q: (q.arrival_ns, q.query_id))
target_qps = os.environ.get("V2_TARGET_QPS")
arrival_mode = os.environ.get("V2_ARRIVAL_MODE", "open")
if arrival_mode not in ("open", "batch", "uniform"):
    raise ValueError("V2_ARRIVAL_MODE must be open, batch or uniform")
if arrival_mode == 'uniform':
    queries = uniform_arrivals(queries, float(target_qps))
    target_qps = None
if arrival_mode == "batch":
    queries = [replace(q, arrival_ns=0, deadline_ns=q.deadline_ns-q.arrival_ns)
               for q in queries]
    target_qps = None
if target_qps:
    target_qps = float(target_qps)
    if target_qps <= 0 or len(queries) < 2:
        raise ValueError("positive V2_TARGET_QPS and at least two queries required")
    first = queries[0].arrival_ns
    original_span = queries[-1].arrival_ns - first
    if original_span <= 0:
        raise ValueError("trace arrival span must be positive")
    factor = ((len(queries) - 1) * 1e9 / target_qps) / original_span
    queries = [replace(q, arrival_ns=int(round((q.arrival_ns - first) * factor)),
                       deadline_ns=int(round((q.arrival_ns - first) * factor))
                       + q.deadline_ns - q.arrival_ns) for q in queries]
allc = set()
for q in queries:
    allc.update(q.cluster_ids)
if os.environ.get("V2_PLACEMENT", "roundrobin") == "crush":
    from strata_sim.sched.crush import crush_placement
    pl = crush_placement(NUM_OSDS, sorted(allc), replication=3)
    print("PLACEMENT=crush (straw2, 5 hosts x 4 OSDs)")
else:
    pl = ceph_placement(NUM_OSDS, sorted(allc), replication=3)
rate = (len(queries)-1) / (queries[-1].arrival_ns / 1e9) if queries[-1].arrival_ns else 0
print("TAG=%s tenancy=%s probes=%d queries=%d distinct_clusters=%d span=%.2fs rate=%.0f q/s"
      % (TAG, TENANCY, (NPROBE32 if TENANCY == "d3" else NP + NQ),
         len(queries), len(allc), queries[-1].arrival_ns / 1e9, rate))

VPC = int(np.median(all_cluster_sizes))
RGW_W = RGWScoreWeights(cache=float(os.environ.get("RGW_CACHE", "3.0")),
                        load=float(os.environ.get("RGW_LOAD", "1.5")),
                        fpga=float(os.environ.get("RGW_FPGA", "1.0")),
                        primary=float(os.environ.get("RGW_PRIMARY", "0.5")),
                        queue_depth_cap=int(os.environ.get("RGW_CAP", "32")))
CFG = [
    ("fifo", "fifo", None),
    ("lstf", "lstf", None),
    ("spt", "spt", None),
    ("batch", "batch", None),
    ("cache", "cache", None),
    ("edf", "edf", None),
    ("drr", "drr", None),
    ("bwc", "bwc", None),
    ("cbw", "cbw", None),
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
    if not CFG:
        raise ValueError("unknown V2_ONLY: " + _only)


def run(strat, wt, fpga):
    import dataclasses as _dc
    _prof = os.environ.get("OSD_MODEL_PROFILE", "ideal")
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
    print("EFFECTIVE_PROFILE", json.dumps(_dc.asdict(hw), sort_keys=True), flush=True)
    print("CODEBOOK_LOWER_BOUND_NS", hw.codebook_duration_ns(), flush=True)
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
        if len(c) != len(queries):
            raise RuntimeError("incomplete run: %d/%d clients" % (len(c), len(queries)))
        if _info:
            _pt = _info.get("per_tenant", {})
            _w = [v["wait_p50_ns"] for v in _pt.values() if v.get("wait_p50_ns") is not None]
            def _j(v):
                v=[float(x) for x in v]
                return (sum(v)**2)/(len(v)*sum(x*x for x in v)) if v and sum(x*x for x in v) else 1.0
            print("JAIN %s:%s jain=%.4f demand=%.4f waitp50=%.4f" % (
                 label, mtag, _info.get("jain_index", -1), _info.get("jain_demand_normalized") or -1, _j(_w)))
        _ev = sum(t.ddr_evictions for x in c for t in x.timings)
        _bev = sum(t.bram_evictions for x in c for t in x.timings)
        _dump = os.environ.get("V2_DUMP_LAT")
        if _dump:
         np.savez("%s_%s_%s.npz" % (_dump, label, mtag),
                  qid=np.array([x.query.query_id for x in c]),
                  tenant=np.array([x.query.tenant for x in c]),
                  lat=np.array([x.latency_ns for x in c]),
                  nsub=np.array([len(x.timings) for x in c]),
                     fpga=np.array([fm.svc_cycles]))
         print("EVICT %s:%s ddr=%d bram=%d" % (label, mtag, _ev, _bev))
        viol = sum(1 for x in c if x.finish_ns > x.query.deadline_ns)
        lat = np.array([x.latency_ns for x in c])
        res["%s:%s" % (label, mtag)] = dict(
            slo=viol / len(c) * 100, viol=viol,
            p50=np.percentile(lat, 50) / 1e6, p90=np.percentile(lat, 90) / 1e6,
            p99=np.percentile(lat, 99) / 1e6, mean=lat.mean() / 1e6,
            fpga=fm.svc_cycles / 1e6)
        duration_s = (max(x.finish_ns for x in c) - min(q.arrival_ns for q in queries)) / 1e9
        arrival_end = max(q.arrival_ns for q in queries)
        arrivals_span_s = (arrival_end - min(q.arrival_ns for q in queries)) / 1e9
        metrics = res["%s:%s" % (label, mtag)]
        if arrival_mode == 'uniform':
            qps = float(os.environ['V2_TARGET_QPS'])
            horizon = round(len(queries)*1e9/qps)
            start = round(float(os.environ.get('V2_WARMUP_FRACTION', '.2'))*horizon)
            if not 0 <= start < horizon:
                raise ValueError('warmup must lie within supply interval')
            metrics['measurement'] = window_metrics(c, start, horizon)
            metrics['event_offset'] = event_offset
        metrics.update(completed_clients=len(c), elapsed_including_drain_s=duration_s,
                       completed_qps_including_drain=len(c) / duration_s,
                       goodput_qps_including_drain=(len(c) - viol) / duration_s,
                       offered_qps=((len(queries) - 1) / arrivals_span_s
                                    if arrivals_span_s else None),
                       arrival_mode=arrival_mode,
                       unfinished_at_last_arrival=sum(x.finish_ns > arrival_end for x in c),
                       ddr_evictions=_ev, bram_evictions=_bev)
        timings = [t for client in c for t in client.timings]
        metrics["codebook_loads"] = sum(t.codebook_loaded for t in timings)
        metrics["pq_loads"] = sum(t.ddr_transfers for t in timings)
        metrics["subqueries"] = len(timings)
        metrics["per_osd"] = {}
        for osd in range(NUM_OSDS):
            ts = [t for t in timings if t.osd_id == osd]
            metrics["per_osd"][str(osd)] = dict(
                 subqueries=len(ts),
                 busy_ns=sum(t.end_ns - t.start_ns for t in ts),
                 codebook_loads=sum(t.codebook_loaded for t in ts),
                 pq_loads=sum(t.ddr_transfers for t in ts),
                 ddr_evictions=sum(t.ddr_evictions for t in ts),
                 bram_evictions=sum(t.bram_evictions for t in ts))
        print("COMPLETION_METRICS", json.dumps(metrics, sort_keys=True), flush=True)
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

output = os.environ.get("V2_RESULT_PATH", "/tmp/%s.json" % TAG)
with open(output, "w") as handle:
    json.dump(res, handle, indent=2)
print("saved ->", output, flush=True)
