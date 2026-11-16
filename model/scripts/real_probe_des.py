#!/usr/bin/env python3
"""Real, contract-validated probes into the existing DES; ideal PQ-only costs.

Saved probes were computed from real embeddings upstream. This adapter neither
loads vectors nor searches an index, returns document IDs, or replays arrivals.
Arrivals are a declared uniform synthetic load over each supplied query once.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.probe_contract import IndexMetadata, load_probes, strict_json
from scripts.exp_osd_sched import STRATEGIES, SyntheticQuery, compute_metrics, run_simulation
from strata_sim.sched.osd_timing import OSDHardwareProfile
from strata_sim.model.strata_model import profile


CANDIDATE_SCOPE = "PQ-only modeled scheduling not reranked-quality cost"


def _key(parts):
    """Injective structured string, independent of Python hash randomization."""
    return json.dumps(parts, ensure_ascii=True, separators=(",", ":"))


def _positive_int(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_input(probes_path, metadata_path, sizes_path, encoder_path, *, expected_queries):
    """Validate a complete input before constructing any simulation state."""
    _positive_int("expected_queries", expected_queries)
    metadata = IndexMetadata(**strict_json(Path(metadata_path).read_text()))
    encoder = strict_json(Path(encoder_path).read_text())
    for name, expected in (("model", metadata.encoder_model),
                           ("revision", metadata.encoder_revision),
                           ("dimension", metadata.dimension),
                           ("normalize_embeddings", metadata.normalized)):
        if type(encoder.get(name)) is not type(expected) or encoder[name] != expected:
            raise ValueError("encoder mismatch: " + name)
    rows = load_probes(probes_path, metadata, strict_json(Path(sizes_path).read_text()))
    if len(rows) != expected_queries:
        raise ValueError("unexpected query count")
    return rows


@dataclass(frozen=True)
class AdaptedTrace:
    queries: tuple
    cluster_sizes: dict
    cluster_mapping: list
    references: dict
    metadata: list
    rate_qps: str
    slo_ns: int


def adapt_probes(probes, *, rate_qps, slo_ns, top_k):
    """Dense sorted deployment IDs; probe order and exact list sizes retained.

    IDs are deterministic for this supplied deployment set, not permanent IDs
    across additions/removals of indexes. Query identities include namespace
    and version; uniform arrivals follow supplied row order, once per query.
    """
    _positive_int("slo_ns", slo_ns)
    _positive_int("top_k", top_k)
    if isinstance(rate_qps, bool):
        raise ValueError("rate_qps must be finite and positive")
    try:
        rate = Fraction(str(rate_qps))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("rate_qps must be finite and positive") from exc
    if rate <= 0 or rate > 1_000_000_000:
        raise ValueError("rate_qps must be > 0 and <= 1e9")
    probes = tuple(probes)
    if not probes:
        raise ValueError("empty probes")
    sizes, indexes, seen = {}, {}, set()
    for p in probes:
        index = (p.metadata.index_namespace, p.metadata.index_version)
        if index in indexes and indexes[index] != p.metadata:
            raise ValueError("conflicting metadata for namespace/version")
        indexes[index] = p.metadata
        identity = (*index, p.query_id)
        if identity in seen:
            raise ValueError("duplicate query identity")
        seen.add(identity)
        if (len(p.centroid_ids) != p.metadata.nprobe
                or len(p.cluster_sizes) != p.metadata.nprobe
                or len(set(p.centroid_ids)) != p.metadata.nprobe
                or type(p.candidate_count) is not int
                or p.candidate_count != sum(p.cluster_sizes)):
            raise ValueError("invalid probe cardinality or candidate_count")
        for cid, size in zip(p.centroid_ids, p.cluster_sizes):
            if (type(cid) is not int or not 0 <= cid < p.metadata.total_centroids
                    or type(size) is not int or size < 0):
                raise ValueError("invalid cluster ID or size")
            key = (*index, cid)
            if key in sizes and sizes[key] != size:
                raise ValueError("conflicting cluster sizes")
            sizes[key] = size
    mapping = {key: i for i, key in enumerate(sorted(sizes))}
    if len(mapping) > 2**32:
        raise ValueError("global cluster ID overflow")
    queries, references = [], {}
    for i, p in enumerate(probes):
        index = (p.metadata.index_namespace, p.metadata.index_version)
        qid = _key((*index, p.query_id))
        # floor of the absolute rational schedule avoids accumulated rounding.
        arrival = (i * 1_000_000_000 * rate.denominator) // rate.numerator
        queries.append(SyntheticQuery(qid, _key(index), arrival, arrival + slo_ns,
                       tuple(mapping[c.key] for c in p.cluster_identities), top_k, _key(index)))
        references[qid] = dict(index_namespace=index[0], index_version=index[1],
                               query_id=p.query_id, query_row=p.query_row,
                               centroid_ids=list(p.centroid_ids),
                               cluster_sizes=list(p.cluster_sizes), candidate_count=p.candidate_count,
                               source="saved coarse-probe oracle; not retrieval document IDs")
    return AdaptedTrace(tuple(queries), {mapping[k]: v for k, v in sizes.items()},
                        [dict(global_id=i, index_namespace=k[0], index_version=k[1],
                              cluster_local_id=k[2], size=sizes[k]) for k, i in mapping.items()],
                        references, [asdict(indexes[k]) for k in sorted(indexes)], str(rate), slo_ns)


@dataclass(frozen=True)
class IdealScanModel:
    """Explicit analytical assumptions, no extracted/calibrated hardware claim."""
    clock_ns: float = 1000 / 180
    fixed_cycles: float = 2160.0  # declared 12 us ideal fixed cost
    process_cycles_per_doc: float = 2.0

    def scan_cycles_per_doc(self, n_docs):
        return self.process_cycles_per_doc

    def predict_cycles(self, n_docs, top_k, *, codebook_resident=True):
        return self.fixed_cycles + self.process_cycles_per_doc * n_docs

    def predict_ns(self, n_docs, top_k, *, codebook_resident=True):
        return self.predict_cycles(n_docs, top_k) * self.clock_ns


def ideal_hardware():
    # board_calibrated selects the existing separate SSD/DDR bandwidth path;
    # all numbers here are ideal assumptions, never board measurements.
    return replace(OSDHardwareProfile.from_model("ideal"), cpu_ns_per_candidate=0,
                   board_pipeline_scan=False, cache_raw_vectors=False, raw_cache_bytes=0)


def run_policies(trace, *, policies=("fifo", "lstf", "multifactor"), num_osds=3,
                 window_width=128):
    _positive_int("num_osds", num_osds)
    _positive_int("window_width", window_width)
    if num_osds < 3:
        raise ValueError("existing default placement requires at least 3 OSDs")
    if not policies or len(set(policies)) != len(policies) or any(p not in STRATEGIES for p in policies):
        raise ValueError("invalid policies")
    # Reject implicit experimental environment overrides rather than silently
    # changing pipeline, prefetch, cache, stats or scheduler defaults.
    import os
    overrides = sorted(k for k in os.environ if k.startswith("OSD_") or k == "SCHED_STATS_PATH")
    if overrides:
        raise ValueError("unset DES environment overrides: " + ", ".join(overrides))
    results = {}
    hw, model = ideal_hardware(), IdealScanModel()
    for policy in policies:
        start = time.perf_counter()
        completed, info = run_simulation(trace.queries, num_osds=num_osds, strategy=policy,
            window_width=window_width, fpga_model=model, hardware=hw,
            cluster_sizes=trace.cluster_sizes, rerank_per_cluster=0)
        elapsed = time.perf_counter() - start
        if len(completed) != len(trace.queries) or {c.query.query_id for c in completed} != set(trace.references):
            raise ValueError("incomplete or duplicate request barrier")
        requests = []
        for c in sorted(completed, key=lambda c: c.query.query_id):
            expected = {f"{c.query.query_id}:c{cid}" for cid in c.query.cluster_ids}
            if (len(c.timings) != len(expected) or {t.query_id for t in c.timings} != expected
                    or c.finish_ns != max(t.end_ns for t in c.timings)):
                raise ValueError("incomplete cluster completion barrier")
            requests.append(dict(query_id=c.query.query_id, oracle_reference=trace.references[c.query.query_id],
                arrival_ns=c.arrival_ns, deadline_ns=c.query.deadline_ns, start_ns=c.start_ns,
                finish_ns=c.finish_ns, latency_ns=c.latency_ns, wait_ns=c.wait_ns,
                subqueries=[dict(query_id=t.query_id, osd_id=t.osd_id, start_ns=t.start_ns,
                                 end_ns=t.end_ns) for t in c.timings]))
        timings = [t for c in completed for t in c.timings]
        counters = dict(offered=len(trace.queries), completed=len(completed), failed=0, pending=0,
            subqueries=len(timings), expected_subqueries=sum(len(q.cluster_ids) for q in trace.queries),
            raw_transfers=sum(t.raw_transfers for t in timings),
            cpu_reservations=sum(r.resource.endswith("-cpu") for t in timings for r in t.reservations))
        if counters["raw_transfers"] or counters["cpu_reservations"]:
            raise ValueError("PQ-only cost scope violated")
        results[policy] = dict(metrics=compute_metrics(completed, strategy=policy,
            window_width=window_width, num_osds=num_osds, samples=len(trace.queries),
            claims={"ideal_only": True, "hardware_execution": False}, queries=trace.queries),
            counters=counters, requests=requests, simulation_info=info, execution_wall_seconds=elapsed)
    return dict(schema="cortex.real_probe_des.v1", backend_kind="offline_des_ideal_only",
        candidate_scope=CANDIDATE_SCOPE, result_source="probe references only; DES computes no true document IDs",
        hardware_execution=False, real_components=["saved query-specific coarse probes", "cluster sizes"],
        modeled_components=["uniform arrivals", "routing", "queues", "SSD", "DDR", "PQ scan", "network"],
        excluded_costs=["embedding", "coarse search", "query-level top-R selection/merge", "raw reads", "rerank"],
        workload_cost_limits=[
            "FiQA source constructs Faiss IndexIVFPQ(768,64,96,8,IP); residual flag is not in probe metadata",
            "Residual-IP scoring requires coarse centroid contribution plus residual PQ ADC; DES computes neither",
            "Fixed-plus-linear scan assumptions do not validate residual table construction, score correction, or top-k cost",
            "128-byte modeled entries are a device layout assumption, not Faiss 96-byte code storage equivalence",
            "No quality configuration is selected from test queries; query-level R64 rerank cost remains excluded",
        ],
        model=dict(kind="ideal-only", calibrated=False, scan=asdict(model), hardware=asdict(hw),
                   board_calibrated_flag="executor path selector only; all parameters ideal", analytical=asdict(profile("ideal"))),
        metadata=trace.metadata, cluster_mapping=trace.cluster_mapping,
        arrivals=dict(kind="uniform synthetic, no replay", rate_qps=trace.rate_qps, slo_ns=trace.slo_ns),
        quality_provenance=dict(reference="rerank_v1 CPU evidence, separate", candidate_scope="query-level R64",
                                cost_equivalent=False, quality_computed_by_DES=False), policies=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("probes", "metadata", "cluster-sizes", "encoder-config", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--rate-qps", required=True)
    parser.add_argument("--slo-ns", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-osds", type=int, default=3)
    parser.add_argument("--policies", nargs="+", default=["fifo", "lstf", "multifactor"])
    args = parser.parse_args()
    try:
        rows = load_input(args.probes, args.metadata, args.cluster_sizes, args.encoder_config,
                          expected_queries=args.expected_queries)
        trace = adapt_probes(rows, rate_qps=args.rate_qps, slo_ns=args.slo_ns, top_k=args.top_k)
        report = run_policies(trace, policies=tuple(args.policies), num_osds=args.num_osds)
        report["input_files"] = {name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in
            (("probes", args.probes), ("metadata", args.metadata), ("cluster_sizes", args.cluster_sizes),
             ("encoder_config", args.encoder_config))}
        report["implementation"] = {str(p): digest(p) for p in (Path(__file__),
            Path(__file__).with_name("exp_osd_sched.py"), Path(__file__).with_name("probe_contract.py"))}
        report["python"] = sys.version
        report["loaded_modules"] = {name: dict(path=module.__file__, sha256=digest(module.__file__))
            for name, module in sorted(sys.modules.items())
            if name.startswith(("scripts.", "strata_sim."))
            and getattr(module, "__file__", None) and module.__file__.endswith(".py")}
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({p: r["counters"] for p, r in report["policies"].items()}))
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(1, "real probe DES: {}\n".format(exc))


if __name__ == "__main__":
    main()
