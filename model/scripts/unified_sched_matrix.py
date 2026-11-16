#!/usr/bin/env python3
"""Run a reproducible, strategy-complete OSD scheduling experiment matrix.

This runner deliberately keeps trace construction outside the strategy loop:
every strategy in an arm sees the identical ordered query trace, placement,
hardware profile, cache budget, and model.  It is an orchestration layer over
``exp_osd_sched`` and does not change the legacy single-run CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp_osd_sched import (  # noqa: E402
    SCHEMA as HARNESS_SCHEMA,
    STRATEGIES,
    SyntheticQuery,
    build_fpga_model,
    ceph_placement,
    compute_metrics,
    generate_trace,
    run_simulation,
    OSDHardwareProfile,
)


SCHEMA = "cortex.unified_sched_matrix.v1"
DEFAULT_WORKLOADS = ("uniform", "tenant_zipf_cluster_zipf")
DEFAULT_SEEDS = (7, 19, 43)
DEFAULT_OSDS = (1, 4)
DEFAULT_CACHE_MIB = (1, 4, 64)
_RESULT_AFFECTING_ENV_PREFIXES = ("OSD_", "SEL")
_RESULT_AFFECTING_ENV_NAMES = ("ADAPT_CB", "SCHED_STATS_PATH")


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("comma-separated value must not be empty")
    return values


def _ints(value: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError(f"{name} must contain integers") from error
    if not values or any(value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _trace_digest(queries: Sequence[SyntheticQuery]) -> str:
    rows = [
        {
            "query_id": query.query_id,
            "tenant": query.tenant,
            "arrival_ns": query.arrival_ns,
            "deadline_ns": query.deadline_ns,
            "cluster_ids": list(query.cluster_ids),
            "top_k": query.top_k,
            "codebook_id": query.codebook_id,
        }
        for query in queries
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _workload_options(name: str) -> tuple[str, str]:
    if name == "uniform":
        return "uniform", "uniform"
    if name == "tenant_zipf_cluster_zipf":
        return "zipf", "zipf"
    raise ValueError(
        f"unknown workload {name!r}; expected uniform or tenant_zipf_cluster_zipf"
    )


def _validate_unique(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and unique")
    return result


def _reject_result_affecting_environment() -> None:
    """Prevent ambient scheduler overrides from changing a matrix run."""
    overrides = sorted(
        name for name in os.environ
        if name.startswith(_RESULT_AFFECTING_ENV_PREFIXES)
        or name in _RESULT_AFFECTING_ENV_NAMES
    )
    if overrides:
        raise ValueError(
            "result-affecting environment overrides are not allowed: "
            + ", ".join(overrides)
        )


def run_matrix(
    *,
    project_root: Path,
    workloads: Sequence[str] = DEFAULT_WORKLOADS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_osds_values: Sequence[int] = DEFAULT_OSDS,
    cache_mib_values: Sequence[int] = DEFAULT_CACHE_MIB,
    strategies: Sequence[str] = STRATEGIES,
    num_queries: int = 256,
    num_tenants: int = 8,
    num_clusters: int = 64,
    span_ns: int = 10_000_000,
    deadline_ns: int = 200_000_000,
    window_width: int = 64,
    top_k: int = 10,
    nprobe: int = 4,
    vectors_per_cluster: int = 64,
    cache_raw_vectors: bool = True,
    fpga_model: Any | None = None,
) -> dict[str, Any]:
    """Execute all combinations and return a self-describing JSON payload."""
    _reject_result_affecting_environment()
    selected_strategies = _validate_unique(strategies, "strategies")
    if any(strategy not in STRATEGIES for strategy in selected_strategies):
        raise ValueError(f"invalid strategy; expected one of {STRATEGIES}")
    selected_workloads = _validate_unique(workloads, "workloads")
    for workload in selected_workloads:
        _workload_options(workload)
    seeds = tuple(seeds)
    osd_values = tuple(num_osds_values)
    cache_values = tuple(cache_mib_values)
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain unique non-negative integers")
    if not osd_values or any(osds < 1 for osds in osd_values) or len(set(osd_values)) != len(osd_values):
        raise ValueError("num_osds_values must contain unique positive integers")
    if not cache_values or any(cache < 1 for cache in cache_values) or len(set(cache_values)) != len(cache_values):
        raise ValueError("cache_mib_values must contain unique positive integers")
    for name, value in (("num_queries", num_queries), ("num_tenants", num_tenants),
                        ("num_clusters", num_clusters), ("span_ns", span_ns),
                        ("deadline_ns", deadline_ns), ("window_width", window_width),
                        ("top_k", top_k), ("nprobe", nprobe),
                        ("vectors_per_cluster", vectors_per_cluster)):
        if value < 1:
            raise ValueError(f"{name} must be positive")

    model = fpga_model if fpga_model is not None else build_fpga_model(project_root)
    model_evidence = "injected_fixture" if fpga_model is not None else "formal_calibrated"
    model_ref = {
        "evidence": model_evidence,
        "profile_id": getattr(getattr(model, "ref", None), "profile_id", None),
        "evidence_id": getattr(getattr(model, "ref", None), "evidence_id", None),
    }
    base_hardware = OSDHardwareProfile(
        vectors_per_cluster=vectors_per_cluster,
        cache_raw_vectors=cache_raw_vectors,
    )
    arms: list[dict[str, Any]] = []
    trace_records: dict[str, dict[str, Any]] = {}
    for workload in selected_workloads:
        tenant_skew, cluster_skew = _workload_options(workload)
        for seed in seeds:
            queries = generate_trace(
                seed=seed, num_queries=num_queries, num_tenants=num_tenants,
                num_clusters=num_clusters, span_ns=span_ns,
                deadline_ns=deadline_ns, skew=tenant_skew, top_k=top_k,
                nprobe=nprobe, codebook_per_tenant=True,
                cluster_skew=cluster_skew,
            )
            digest = _trace_digest(queries)
            trace_records[f"{workload}:{seed}"] = {
                "workload": workload, "seed": seed, "samples": len(queries),
                "digest_sha256": digest,
            }
            all_clusters = sorted({cid for query in queries for cid in query.cluster_ids})
            for num_osds in osd_values:
                placement = ceph_placement(
                    num_osds, all_clusters, replication=min(3, num_osds)
                )
                for cache_mib in cache_values:
                    hardware = replace(base_hardware,
                                       pq_cache_bytes=cache_mib * 1024 * 1024)
                    for strategy in selected_strategies:
                        completed, info = run_simulation(
                            queries, num_osds=num_osds, strategy=strategy,
                            window_width=window_width, fpga_model=model,
                            placement=placement, hardware=hardware,
                        )
                        if len(completed) != len(queries):
                            raise ValueError(
                                f"incomplete arm {workload}/{seed}/{num_osds}/"
                                f"{cache_mib}/{strategy}: {len(completed)} of {len(queries)}"
                            )
                        metrics = compute_metrics(
                            completed, strategy=strategy, window_width=window_width,
                            num_osds=num_osds, samples=len(queries), queries=queries,
                            claims={
                                "matrix_schema": SCHEMA,
                                "harness_schema": HARNESS_SCHEMA,
                                "same_trace_across_strategies": True,
                                "synthetic_arrivals": True,
                                "real_users": False,
                                "hardware_execution": False,
                                "model_evidence": (
                                    model_evidence
                                ),
                                "model_ref": model_ref,
                            },
                        )
                        arms.append({
                            "workload": workload, "seed": seed,
                            "trace_digest_sha256": digest, "num_osds": num_osds,
                            "placement": {
                                str(cluster_id): list(replica_set.osd_ids)
                                for cluster_id, replica_set in placement.items()
                            },
                            "cache_budget_bytes": hardware.pq_cache_bytes,
                            "cache_raw_vectors": cache_raw_vectors,
                            "strategy": strategy, "metrics": metrics,
                            "simulation_info": info,
                        })
    return {
        "schema": SCHEMA,
        "project_root": str(project_root),
        "configuration": {
            "workloads": list(selected_workloads), "seeds": list(seeds),
            "num_osds": list(osd_values), "cache_mib": list(cache_values),
            "strategies": list(selected_strategies), "num_queries": num_queries,
            "num_tenants": num_tenants, "num_clusters": num_clusters,
            "span_ns": span_ns, "deadline_ns": deadline_ns,
            "window_width": window_width, "top_k": top_k, "nprobe": nprobe,
            "vectors_per_cluster": vectors_per_cluster,
            "cache_raw_vectors": cache_raw_vectors,
            "model_ref": model_ref,
        },
        "trace_records": list(trace_records.values()),
        "arms": arms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", default=",".join(DEFAULT_WORKLOADS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--num-osds", default=",".join(map(str, DEFAULT_OSDS)))
    parser.add_argument("--cache-mib", default=",".join(map(str, DEFAULT_CACHE_MIB)))
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--tenants", type=int, default=8)
    parser.add_argument("--clusters", type=int, default=64)
    parser.add_argument("--window-width", type=int, default=64)
    parser.add_argument("--cache-raw", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = run_matrix(
            project_root=args.project_root,
            workloads=_csv(args.workloads), seeds=_ints(args.seeds, "seeds"),
            num_osds_values=_ints(args.num_osds, "num_osds"),
            cache_mib_values=_ints(args.cache_mib, "cache_mib"),
            strategies=_csv(args.strategies), num_queries=args.queries,
            num_tenants=args.tenants, num_clusters=args.clusters,
            window_width=args.window_width, cache_raw_vectors=args.cache_raw,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "arms": len(payload["arms"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
