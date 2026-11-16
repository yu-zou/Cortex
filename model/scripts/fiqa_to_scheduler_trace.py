#!/usr/bin/env python3
"""Strict adapter from frozen FiQA probes to C++ scheduler trace input.

Converts actual FiQA query/probe metadata into the scheduler_trace_runner CSV
format. No synthetic substitution; missing inputs cause explicit failure.
Arrival trace is separate from query content; only uniform replay supported.
Service costs are declared scan proxy (not complete device plan, not calibrated).
"""
import argparse
import csv
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Tuple

# Import existing contract validators
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.probe_contract import IndexMetadata, load_probes, strict_json


UINT64_MAX = 2**64 - 1


def digest(path: Path) -> str:
    """SHA256 digest of file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs(probes_path: Path, metadata_path: Path, sizes_path: Path,
                    encoder_path: Path, expected_queries: int) -> Tuple:
    """Validate all inputs exist and match contract before processing."""
    if not probes_path.exists():
        raise FileNotFoundError(f"Probes file not found: {probes_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not sizes_path.exists():
        raise FileNotFoundError(f"Cluster sizes file not found: {sizes_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder config file not found: {encoder_path}")
    
    if expected_queries <= 0:
        raise ValueError("expected_queries must be positive")
    
    # Load and validate metadata
    metadata = IndexMetadata(**strict_json(metadata_path.read_text()))
    
    # Validate encoder matches metadata
    encoder = strict_json(encoder_path.read_text())
    for name, expected in (("model", metadata.encoder_model),
                          ("revision", metadata.encoder_revision),
                          ("dimension", metadata.dimension),
                          ("normalize_embeddings", metadata.normalized)):
        if type(encoder.get(name)) is not type(expected) or encoder[name] != expected:
            raise ValueError(f"encoder config mismatch: {name}")
    
    # Load cluster sizes
    cluster_sizes = strict_json(sizes_path.read_text())
    if not isinstance(cluster_sizes, list):
        raise ValueError("cluster_sizes must be a JSON array")
    if len(cluster_sizes) != metadata.total_centroids:
        raise ValueError(f"cluster_sizes length {len(cluster_sizes)} != total_centroids {metadata.total_centroids}")
    
    # Load and validate probes
    probes = load_probes(probes_path, metadata, cluster_sizes)
    
    if len(probes) != expected_queries:
        raise ValueError(f"Found {len(probes)} probes but expected {expected_queries}")
    
    # Check for duplicate query IDs
    query_ids = [p.query_id for p in probes]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Duplicate query IDs found in probes")
    
    return probes, metadata, cluster_sizes


def query_identity(metadata: IndexMetadata, probe) -> Tuple[str, str, str]:
    """Structured identity: (namespace, index version, original query ID)."""
    return (metadata.index_namespace, metadata.index_version, probe.query_id)


def build_query_id_mapping(probes, metadata: IndexMetadata) -> Tuple[Dict, List]:
    """Assign deterministic sequential positive query IDs by first occurrence.

    Identity is a structured tuple, never a delimiter join or randomized hash.
    Duplicate full identities are rejected before any output is written.
    """
    mapping = {}
    order = []
    for probe in probes:
        identity = query_identity(metadata, probe)
        if identity in mapping:
            raise ValueError(f"duplicate query identity: {identity!r}")
        next_id = len(order) + 1
        if next_id > UINT64_MAX:
            raise ValueError("query ID mapping exceeds UINT64_MAX")
        mapping[identity] = next_id
        order.append(identity)
    return mapping, order


def compute_cluster_service_cost(cluster_size: int, cost_model: str, model_params: Dict) -> int:
    """Compute service time in nanoseconds for a single cluster scan.
    
    Args:
        cluster_size: Number of candidates in this cluster
        cost_model: One of "declared_scan_proxy"
        model_params: Model-specific parameters
    
    Returns:
        Service time in nanoseconds (integer, ceil of exact rational)
    """
    if cost_model == "declared_scan_proxy":
        # Explicit scan proxy: fixed + linear per candidate
        # Not a complete device service plan; not calibrated hardware
        clock_ns = model_params.get("clock_ns")
        fixed_cycles = model_params.get("fixed_cycles")
        cycles_per_doc = model_params.get("cycles_per_doc")
        
        # Validate parameters
        if clock_ns is None or fixed_cycles is None or cycles_per_doc is None:
            raise ValueError("declared_scan_proxy requires clock_ns, fixed_cycles, cycles_per_doc")
        
        # Reject bool-as-number, NaN, Inf, negative
        for name, value in [("clock_ns", clock_ns), ("fixed_cycles", fixed_cycles), 
                           ("cycles_per_doc", cycles_per_doc)]:
            if isinstance(value, bool):
                raise ValueError(f"{name} must not be boolean")
            try:
                val = Fraction(str(value))
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"{name} must be finite rational")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        
        # clock_ns and fixed_cycles must be positive (not zero)
        if Fraction(str(clock_ns)) <= 0:
            raise ValueError("clock_ns must be positive")
        if Fraction(str(fixed_cycles)) <= 0:
            raise ValueError("fixed_cycles must be positive")
        
        if type(cluster_size) is not int or cluster_size < 0:
            raise ValueError("cluster_size must be non-negative integer")
        
        # Exact rational arithmetic: cycles = fixed + per_doc * size
        clock = Fraction(str(clock_ns))
        fixed = Fraction(str(fixed_cycles))
        per_doc = Fraction(str(cycles_per_doc))
        
        total_cycles = fixed + per_doc * cluster_size
        service_ns_exact = total_cycles * clock
        
        # Ceil to integer nanoseconds
        import math
        service_ns = math.ceil(service_ns_exact)
        
        if service_ns < 0:
            raise ValueError("computed service_ns is negative")
        if service_ns > 2**64 - 1:
            raise ValueError("computed service_ns exceeds UINT64_MAX")
        
        return service_ns
    
    else:
        raise ValueError(f"Unknown cost_model: {cost_model}")


def generate_trace(probes, metadata, arrival_spec: Dict, cost_model: str,
                   model_params: Dict) -> List[Dict]:
    """Generate scheduler trace from validated probes.
    
    Args:
        probes: Validated probe tuple
        metadata: IndexMetadata
        arrival_spec: {"kind": "uniform", "rate_qps": float, "start_ns": int}
        cost_model: Service cost model name
        model_params: Cost model parameters
    
    Returns:
        List of task dictionaries for CSV output
    """
    if arrival_spec["kind"] != "uniform":
        raise ValueError("Only uniform arrival replay supported in first adapter")
    
    # Validate and parse rate_qps
    rate_qps_raw = arrival_spec["rate_qps"]
    if isinstance(rate_qps_raw, bool):
        raise ValueError("rate_qps must not be boolean")
    try:
        rate_qps = Fraction(str(rate_qps_raw))
    except (ValueError, ZeroDivisionError):
        raise ValueError("rate_qps must be finite rational")
    if rate_qps <= 0:
        raise ValueError("rate_qps must be positive")
    
    start_ns = arrival_spec.get("start_ns", 0)
    if isinstance(start_ns, bool):
        raise ValueError("start_ns must not be boolean")
    if type(start_ns) is not int or start_ns < 0:
        raise ValueError("start_ns must be non-negative integer")
    
    query_mapping, _ = build_query_id_mapping(probes, metadata)
    tasks = []
    task_id = 1
    
    for i, probe in enumerate(probes):
        # Uniform arrival: floor of rational schedule
        try:
            arrival_offset = (i * 1_000_000_000 * rate_qps.denominator) // rate_qps.numerator
            if arrival_offset > 2**64 - 1 - start_ns:
                raise ValueError(f"arrival_ns exceeds UINT64_MAX for query {i}")
            arrival_ns = start_ns + arrival_offset
        except OverflowError:
            raise ValueError(f"arrival_ns computation overflow for query {i}")
        
        # Each probe generates nprobe subtasks (one per cluster)
        query_id = query_mapping[query_identity(metadata, probe)]
        
        for j, (cid, size) in enumerate(zip(probe.centroid_ids, probe.cluster_sizes)):
            # Compute service cost independently for this cluster size
            service_ns = compute_cluster_service_cost(size, cost_model, model_params)
            
            if arrival_ns + service_ns > 2**64 - 1:
                raise ValueError(f"arrival_ns + service_ns exceeds UINT64_MAX for task {task_id}")
            
            task_class = f"cluster_{cid}"
            query_class = f"nprobe{metadata.nprobe}_cands{probe.candidate_count}"
            
            # All subtasks arrive simultaneously (fanout model)
            tasks.append({
                "task_id": task_id,
                "query_id": query_id,
                "probe_query_id": probe.query_id,
                "cluster_id": cid,
                "cluster_size": size,
                "arrival_ns": arrival_ns,
                "service_ns": service_ns,
                "estimate_ns": service_ns,  # Perfect estimates
                "task_class": task_class,
                "query_class": query_class,
            })
            task_id += 1
    
    return tasks


def _csv_text(value, field: str) -> str:
    """Reject multiline text; the C++ loader is line-oriented."""
    if isinstance(value, str) and ("\n" in value or "\r" in value):
        raise ValueError(f"{field} contains unsupported newline")
    return value


def write_scheduler_csv(tasks: List[Dict], output_path: Path):
    """Write tasks to scheduler_trace_runner CSV format.
    
    CSV schema from scheduler_trace_runner.cc:66-68:
    trace,seed,task_id,query_id,arrival_ns,service_ns,estimate_ns,estimate_label,task_class,query_class
    query_id is a uint64-compatible integer supplied by the adapter mapping.
    """
    rows = []
    for task in tasks:
        query_id = task["query_id"]
        if type(query_id) is not int or not 1 <= query_id <= UINT64_MAX:
            raise ValueError("query_id must be a positive uint64 integer")
        rows.append([
            "fiqa_real", 0, task["task_id"], query_id,
            task["arrival_ns"], task["service_ns"], task["estimate_ns"],
            "trace_supplied",
            _csv_text(task["task_class"], "task_class"),
            _csv_text(task["query_class"], "query_class"),
        ])

    with output_path.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow([
            "trace", "seed", "task_id", "query_id", "arrival_ns",
            "service_ns", "estimate_ns", "estimate_label", "task_class",
            "query_class",
        ])
        writer.writerows(rows)


def write_manifest(probes, metadata, cluster_sizes, input_files: Dict,
                   arrival_spec: Dict, cost_model: str, model_params: Dict,
                   output_path: Path):
    """Write identity manifest documenting inputs and model boundary."""
    query_mapping, identity_order = build_query_id_mapping(probes, metadata)
    manifest = {
        "schema": "cortex.fiqa_scheduler_adapter.v1",
        "description": "Strict adapter from frozen FiQA probes to scheduler trace CSV",
        "csv_estimate_label": "trace_supplied",
        "query_id_mapping": [
            {
                "query_id": query_mapping[identity],
                "index_namespace": identity[0],
                "index_version": identity[1],
                "original_query_id": identity[2],
                "probe_index": index,
                "query_row": probe.query_row,
            }
            for index, (identity, probe) in enumerate(zip(identity_order, probes))
        ],
        "input_files": {
            name: {"path": str(path), "sha256": digest(path)}
            for name, path in input_files.items()
        },
        "metadata": {
            "index_namespace": metadata.index_namespace,
            "index_version": metadata.index_version,
            "dataset": metadata.dataset,
            "encoder_model": metadata.encoder_model,
            "encoder_revision": metadata.encoder_revision,
            "dimension": metadata.dimension,
            "metric": metadata.metric,
            "normalized": metadata.normalized,
            "total_centroids": metadata.total_centroids,
            "nprobe": metadata.nprobe,
        },
        "query_count": len(probes),
        "subtask_count": len(probes) * metadata.nprobe,
        "arrival_model": arrival_spec,
        "service_cost_model": {
            "name": cost_model,
            "parameters": model_params,
            "source": "declared scan proxy, not complete device service plan, not calibrated",
            "scope": "PQ-only scan; no coarse search, no rerank, no CPU ADC",
            "excluded": [
                "embedding",
                "coarse search",
                "query-level top-R selection/merge",
                "raw reads",
                "rerank",
                "R64 quality cost",
            ],
        },
        "model_boundary": {
            "real_components": [
                "saved query-specific coarse probes",
                "cluster sizes from actual index",
                "query order preserved",
            ],
            "modeled_components": [
                "uniform arrivals (not replay)",
                "declared scan proxy (per-cluster)",
            ],
            "not_modeled": [
                "network",
                "queuing",
                "placement/routing",
                "actual hardware execution",
            ],
        },
        "validation": {
            "no_duplicate_queries": True,
            "all_required_subtasks": True,
            "dimension_hash_match": True,
            "cluster_counts_match": True,
        },
    }
    
    with output_path.open('w') as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True,
                       help="Frozen FiQA probes JSONL file")
    parser.add_argument("--metadata", type=Path, required=True,
                       help="Index metadata JSON file")
    parser.add_argument("--cluster-sizes", type=Path, required=True,
                       help="Cluster sizes JSON array file")
    parser.add_argument("--encoder-config", type=Path, required=True,
                       help="Encoder config JSON file")
    parser.add_argument("--expected-queries", type=int, required=True,
                       help="Expected number of queries (validation)")
    parser.add_argument("--output-csv", type=Path, required=True,
                       help="Output scheduler trace CSV file")
    parser.add_argument("--output-manifest", type=Path, required=True,
                       help="Output identity manifest JSON file")
    parser.add_argument("--rate-qps", type=str, required=True,
                       help="Uniform arrival rate (queries per second, rational string preserved)")
    parser.add_argument("--start-ns", type=int, default=0,
                       help="Start time offset in nanoseconds")
    parser.add_argument("--cost-model", default="declared_scan_proxy",
                       choices=["declared_scan_proxy"],
                       help="Service cost model")
    parser.add_argument("--clock-ns", type=str, default="1000/180",
                       help="Clock period in ns (default: 180 MHz, rational string preserved)")
    parser.add_argument("--fixed-cycles", type=str, default="2160",
                       help="Fixed overhead cycles (default: 12 us @ 180 MHz)")
    parser.add_argument("--cycles-per-doc", type=str, default="2",
                       help="Cycles per document (default: 2)")
    
    args = parser.parse_args()
    
    try:
        # Validate all inputs
        probes, metadata, cluster_sizes = validate_inputs(
            args.probes, args.metadata, args.cluster_sizes,
            args.encoder_config, args.expected_queries
        )
        
        # Build arrival specification
        arrival_spec = {
            "kind": "uniform",
            "rate_qps": args.rate_qps,
            "start_ns": args.start_ns,
        }
        
        # Build cost model parameters
        model_params = {
            "clock_ns": args.clock_ns,
            "fixed_cycles": args.fixed_cycles,
            "cycles_per_doc": args.cycles_per_doc,
        }
        
        # Generate trace
        tasks = generate_trace(probes, metadata, arrival_spec,
                             args.cost_model, model_params)
        
        # Write outputs
        write_scheduler_csv(tasks, args.output_csv)
        
        input_files = {
            "probes": args.probes,
            "metadata": args.metadata,
            "cluster_sizes": args.cluster_sizes,
            "encoder_config": args.encoder_config,
        }
        write_manifest(probes, metadata, cluster_sizes, input_files,
                      arrival_spec, args.cost_model, model_params,
                      args.output_manifest)
        
        print(json.dumps({
            "query_count": len(probes),
            "subtask_count": len(tasks),
            "output_csv": str(args.output_csv),
            "output_manifest": str(args.output_manifest),
        }))
        
    except (ValueError, TypeError, FileNotFoundError, OSError) as exc:
        parser.exit(1, f"fiqa_to_scheduler_trace: {exc}\n")


if __name__ == "__main__":
    main()
