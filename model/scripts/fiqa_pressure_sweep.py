#!/usr/bin/env python3
"""Run a deterministic FiQA strategy sweep with an explicit PQ-cache budget.

The saved probes and their order are kept unchanged.  The runner varies only
the OSD queue strategy and ``pq_cache_bytes`` so a low-budget arm can prove
that the modeled DDR/PQ cache is actually under pressure.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.exp_osd_sched import (  # noqa: E402
    STRATEGIES,
    ceph_placement,
    compute_metrics,
    run_simulation,
)
from scripts.real_probe_des import (  # noqa: E402
    IdealScanModel,
    adapt_probes,
    digest,
    load_input,
    ideal_hardware,
)


SCHEMA = "cortex.fiqa_pressure_sweep.v1"
DEFAULT_STRATEGIES = ("fifo", "cache", "bwc", "cbw", "multifactor")
DEFAULT_PRESSURE_BUDGETS_MIB = (1, 2, 4)
DEFAULT_CONTROL_BUDGET_GIB = 6


def _positive_int(name: str, value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unique_positive(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(type(value) is not int or value < 1 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def run_sweep(
    *,
    probes: Path,
    metadata: Path,
    cluster_sizes: Path,
    encoder_config: Path,
    expected_queries: int,
    rate_qps: str,
    slo_ns: int,
    top_k: int,
    num_osds: int,
    window_width: int,
    strategies: Sequence[str],
    pressure_budgets_mib: Sequence[int],
    control_budget_gib: int,
    require_pressure: bool = False,
) -> dict[str, Any]:
    _positive_int("expected_queries", expected_queries)
    _positive_int("slo_ns", slo_ns)
    _positive_int("top_k", top_k)
    _positive_int("num_osds", num_osds)
    _positive_int("window_width", window_width)
    pressure_budgets = _unique_positive(pressure_budgets_mib, "pressure_budgets_mib")
    _positive_int("control_budget_gib", control_budget_gib)
    selected = tuple(strategies)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("strategies must be non-empty and unique")
    if any(strategy not in STRATEGIES for strategy in selected):
        raise ValueError(f"invalid strategy; expected one of {STRATEGIES}")

    rows = load_input(
        probes,
        metadata,
        cluster_sizes,
        encoder_config,
        expected_queries=expected_queries,
    )
    trace = adapt_probes(rows, rate_qps=rate_qps, slo_ns=slo_ns, top_k=top_k)
    all_clusters = sorted({cid for query in trace.queries for cid in query.cluster_ids})
    if num_osds == 1:
        placement = ceph_placement(1, all_clusters, replication=1)
    else:
        placement = ceph_placement(num_osds, all_clusters, replication=num_osds)

    base_hardware = replace(
        ideal_hardware(),
        bram_capacity_slots=1,
        pq_cache_bytes=1,
    )
    budgets = [(budget_mib * 1024 * 1024, "pressure") for budget_mib in pressure_budgets]
    budgets.append((control_budget_gib * 1024 * 1024 * 1024, "control"))
    arms: list[dict[str, Any]] = []
    for cache_budget_bytes, budget_kind in budgets:
        hardware = replace(base_hardware, pq_cache_bytes=cache_budget_bytes)
        for strategy in selected:
            completed, simulation_info = run_simulation(
                trace.queries,
                num_osds=num_osds,
                strategy=strategy,
                window_width=window_width,
                fpga_model=IdealScanModel(),
                placement=placement,
                hardware=hardware,
                cluster_sizes=trace.cluster_sizes,
                rerank_per_cluster=0,
            )
            if len(completed) != len(trace.queries):
                raise ValueError(
                    f"incomplete arm {strategy}/{cache_budget_bytes}: "
                    f"{len(completed)} of {len(trace.queries)} completed"
                )
            metrics = compute_metrics(
                completed,
                strategy=strategy,
                window_width=window_width,
                num_osds=num_osds,
                samples=len(trace.queries),
                claims={
                    "real_fiqa_probes": True,
                    "uniform_synthetic_arrivals": True,
                    "fixed_route": True,
                    "hardware_execution": False,
                    "pq_cache_budget_is_pressure_knob": True,
                },
                queries=trace.queries,
            )
            arm = {
                "strategy": strategy,
                "cache_budget_bytes": cache_budget_bytes,
                "cache_budget_kind": budget_kind,
                "metrics": metrics,
                "simulation_info": simulation_info,
            }
            arms.append(arm)

    by_kind = {
        kind: [arm for arm in arms if arm["cache_budget_kind"] == kind]
        for kind in ("pressure", "control")
    }
    pressure_evictions = [arm["metrics"]["ddr_evictions"] for arm in by_kind["pressure"]]
    control_evictions = [arm["metrics"]["ddr_evictions"] for arm in by_kind["control"]]
    pressure_summary = {
        "pressure_arms": len(by_kind["pressure"]),
        "control_arms": len(by_kind["control"]),
        "pressure_min_ddr_evictions": min(pressure_evictions),
        "pressure_max_ddr_evictions": max(pressure_evictions),
        "control_max_ddr_evictions": max(control_evictions),
        "pressure_active": min(pressure_evictions) > 0,
        "control_is_cold_start_only": max(control_evictions) == 0,
    }
    if require_pressure and not pressure_summary["pressure_active"]:
        raise ValueError("pressure arms produced no DDR evictions")
    if require_pressure and not pressure_summary["control_is_cold_start_only"]:
        raise ValueError("control arm unexpectedly produced DDR evictions")

    files = {
        "probes": probes,
        "metadata": metadata,
        "cluster_sizes": cluster_sizes,
        "encoder_config": encoder_config,
    }
    return {
        "schema": SCHEMA,
        "workload": "FiQA saved coarse probes; PQ-only modeled DES",
        "input_files": {
            name: {"path": str(path.resolve()), "sha256": digest(path)}
            for name, path in files.items()
        },
        "trace": {
            "queries": len(trace.queries),
            "rate_qps": trace.rate_qps,
            "slo_ns": trace.slo_ns,
            "top_k": top_k,
            "cluster_count": len(trace.cluster_sizes),
            "probe_order_preserved": True,
        },
        "placement": {
            "num_osds": num_osds,
            "replication": 1 if num_osds == 1 else num_osds,
            "cluster_ids_sha256": hashlib.sha256(
                json.dumps(all_clusters, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        },
        "hardware": {
            "model": "ideal",
            "hardware_execution": False,
            "bram_capacity_slots": 1,
            "cache_raw_vectors": False,
            "pressure_control": "pq_cache_bytes",
            "control_budget_gib": control_budget_gib,
        },
        "strategies": list(selected),
        "cache_budgets_mib": list(pressure_budgets),
        "pressure_summary": pressure_summary,
        "arms": arms,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cluster-sizes", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--expected-queries", type=int, default=648)
    parser.add_argument("--rate-qps", default="1000")
    parser.add_argument("--slo-ns", type=int, default=200_000_000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-osds", type=int, default=1)
    parser.add_argument("--window-width", type=int, default=128)
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument(
        "--pressure-budget-mib",
        type=int,
        nargs="+",
        default=list(DEFAULT_PRESSURE_BUDGETS_MIB),
    )
    parser.add_argument("--control-budget-gib", type=int, default=DEFAULT_CONTROL_BUDGET_GIB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pressure", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_sweep(
            probes=args.probes,
            metadata=args.metadata,
            cluster_sizes=args.cluster_sizes,
            encoder_config=args.encoder_config,
            expected_queries=args.expected_queries,
            rate_qps=args.rate_qps,
            slo_ns=args.slo_ns,
            top_k=args.top_k,
            num_osds=args.num_osds,
            window_width=args.window_width,
            strategies=args.strategies,
            pressure_budgets_mib=args.pressure_budget_mib,
            control_budget_gib=args.control_budget_gib,
            require_pressure=args.require_pressure,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"fiqa pressure sweep: {exc}\n")
    print(json.dumps(report["pressure_summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
