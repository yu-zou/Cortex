#!/usr/bin/env python3
"""Offline, scoped storage arithmetic; never downloads, deletes or contacts hosts.

Default scope is pilot artifacts, NOT a total deployment budget. Its historical
PQ proxy includes float32 vectors, per-record PQ codes, IDs, fixed headers,
2**bits * dim float32 codebook, extra encoded-index copies, queries and corpus.
SciFact's actual IVFFlat layout differs; this proxy is not an exact inventory.
Unknown corpus bytes are omitted, making arithmetic a lower bound, never an
approval. No per-document text-size allowance is treated as conservative.

Deployment scope replicates dataset artifacts plus optional packed records
(128 B/record when requested) across Ceph copies, then adds model and build
temp. It requires explicit additional overhead (IVF/quantizer/list metadata,
Ceph/DB/filesystem overhead, logs, download/extraction staging and other
deployment artifacts) and a positive free reserve. These are operator inputs,
not measured guarantees. Free space must represent the same physical capacity
scope as replication; aggregate arithmetic does not establish per-OSD fit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

# Encoder: sentence-transformers/all-mpnet-base-v2, revision
# e8c3b32edf5434bc2275fc9bab85f82640a19130; 768D float32, normalized.
DIM = 768
PQ_M = 96
PQ_NBITS = 8
ID_BYTES = 8
HEADER_BYTES = 4096

# Official BEIR dataset statistics (documents / queries).  These are fixed
# facts from the BEIR benchmark release, not values measured by downloading.
# corpus_bytes is documented from the verified pilot manifest where available
# and unknown otherwise; document counts cannot bound raw text storage.
DATASET_SIZES: Dict[str, Dict[str, Any]] = {
    "scifact": {
        "docs": 5_183,
        "queries": 300,
        "corpus_bytes": 8_106_566,
        "corpus_source": "pilot artifact_manifest.json (verified)",
        "source": "BEIR official statistics",
    },
    "fiqa": {
        "docs": 57_638,
        "queries": 648,
        "corpus_bytes": 47_949_497,
        "corpus_source": "pilot artifact_manifest.json (verified)",
        "source": "BEIR official statistics",
    },
    "hotpotqa": {
        "docs": 5_233_329,
        "queries": 7_405,
        "corpus_bytes": None,
        "corpus_source": "unknown (not measured)",
        "source": "BEIR official statistics",
    },
    "nq": {
        "docs": 2_681_468,
        "queries": 3_452,
        "corpus_bytes": None,
        "corpus_source": "unknown (not measured)",
        "source": "BEIR official statistics",
    },
}

# One all-mpnet-base-v2 revision: model.safetensors 437,971,872 B plus
# tokenizer/config files.  The shared ~/.cache/huggingface directory may hold
# several revisions; pass --model-cache-bytes to budget the real cache.
MODEL_CACHE_BYTES = 438_700_000
BUILD_TEMP_FRACTION = 0.08
BUILD_TEMP_MIN_BYTES = 64 * 1024 * 1024
EXTRA_COPIES = 1
LARGE_DATASETS = ("hotpotqa", "nq")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _integer(name: str, value: int, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %d" % (name, minimum))
    return value


def codebook_bytes(dim: int = DIM, pq_nbits: int = PQ_NBITS) -> int:
    _integer("dim", dim, 1)
    _integer("pq_nbits", pq_nbits, 1)
    if pq_nbits > 16:
        raise ValueError("pq_nbits must be <= 16")
    return (2 ** pq_nbits) * dim * 4


def estimate_dataset(
    name: str,
    *,
    docs: Optional[int] = None,
    queries: Optional[int] = None,
    corpus_bytes: Optional[int] = None,
    dim: int = DIM,
    pq_m: int = PQ_M,
    pq_nbits: int = PQ_NBITS,
    id_bytes: int = ID_BYTES,
    header_bytes: int = HEADER_BYTES,
    extra_copies: int = EXTRA_COPIES,
) -> Dict[str, Any]:
    """Pure per-dataset estimate; no I/O, no downloads, no deletion."""
    if name not in DATASET_SIZES:
        raise ValueError(
            "unknown dataset %r; choose from %s"
            % (name, ", ".join(sorted(DATASET_SIZES)))
        )
    spec = DATASET_SIZES[name]
    n_docs = _integer("docs", spec["docs"] if docs is None else docs)
    n_queries = _integer("queries", spec["queries"] if queries is None else queries)
    # A changed document population invalidates the pilot corpus measurement.
    known_corpus = corpus_bytes if corpus_bytes is not None else (
        spec["corpus_bytes"] if n_docs == spec["docs"] else None
    )
    n_corpus = 0 if known_corpus is None else _integer("corpus_bytes", known_corpus)
    for key, value, minimum in (("dim", dim, 1), ("pq_m", pq_m, 1),
                                ("id_bytes", id_bytes, 1), ("header_bytes", header_bytes, 0),
                                ("extra_copies", extra_copies, 0)):
        _integer(key, value, minimum)
    book = codebook_bytes(dim, pq_nbits)
    if dim % pq_m:
        raise ValueError("dim must be divisible by pq_m")

    vectors = n_docs * dim * 4
    pq_codes = n_docs * _ceil_div(pq_m * pq_nbits, 8)
    ids = n_docs * id_bytes
    headers = 2 * header_bytes + book
    encoded = vectors + pq_codes + ids + headers
    copies = encoded * extra_copies
    query_vectors = n_queries * dim * 4
    dataset_total = encoded + copies + query_vectors + n_corpus

    return {
        "dataset": name,
        "source": spec["source"],
        "docs": n_docs,
        "queries": n_queries,
        "estimate_status": "lower_bound" if known_corpus is None else "scoped_estimate",
        "assumptions": {
            "dim": dim,
            "pq_m": pq_m,
            "pq_nbits": pq_nbits,
            "id_bytes": id_bytes,
            "header_bytes": headers,
            "codebook_bytes": book,
            "extra_copies": extra_copies,
            "corpus_bytes": known_corpus,
            "corpus_source": ("explicit override" if corpus_bytes is not None else
                              spec["corpus_source"] if known_corpus is not None else
                              "unknown (not measured)"),
        },
        "components": {
            "vectors_bytes": vectors,
            "pq_codes_bytes": pq_codes,
            "ids_bytes": ids,
            "headers_bytes": headers,
            "encoded_bytes": encoded,
            "extra_copies_bytes": copies,
            "query_vectors_bytes": query_vectors,
            "corpus_bytes": n_corpus,
        },
        "total_bytes": dataset_total,
    }


def estimate_plan(
    datasets: List[str],
    *,
    free_bytes: int,
    budget_bytes: Optional[int] = None,
    extra_copies: int = EXTRA_COPIES,
    model_cache_bytes: int = MODEL_CACHE_BYTES,
    build_temp_fraction: float = BUILD_TEMP_FRACTION,
    build_temp_min_bytes: int = BUILD_TEMP_MIN_BYTES,
    allow_both_large: bool = False,
    corpus_overrides: Optional[Dict[str, int]] = None,
    scope: str = "offline_artifacts",
    free_reserve_bytes: int = 0,
    ceph_replicas: int = 3,
    packed_record_bytes: int = 0,
    deployment_overhead_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Pure plan estimate plus refusal reasons.

    Missing corpus/approval or deployment allowances fail closed. Reserve is
    required in addition to the estimated footprint; budget caps the footprint.
    An offline-artifact fit never grants deployment download readiness.
    """
    if not datasets:
        raise ValueError("at least one dataset is required")
    for key, value, minimum in (
        ("free_bytes", free_bytes, 0), ("model_cache_bytes", model_cache_bytes, 0),
        ("build_temp_min_bytes", build_temp_min_bytes, 0),
        ("free_reserve_bytes", free_reserve_bytes, 0), ("extra_copies", extra_copies, 0),
        ("ceph_replicas", ceph_replicas, 1), ("packed_record_bytes", packed_record_bytes, 0),
    ):
        _integer(key, value, minimum)
    if budget_bytes is not None:
        _integer("budget_bytes", budget_bytes)
    if deployment_overhead_bytes is not None:
        _integer("deployment_overhead_bytes", deployment_overhead_bytes)
    if (isinstance(build_temp_fraction, bool)
            or not isinstance(build_temp_fraction, (int, float))
            or not math.isfinite(build_temp_fraction) or build_temp_fraction < 0):
        raise ValueError("build_temp_fraction must be finite and non-negative")
    if scope not in ("offline_artifacts", "deployment"):
        raise ValueError("scope must be offline_artifacts or deployment")
    if not isinstance(allow_both_large, bool):
        raise ValueError("allow_both_large must be boolean")
    corpus_overrides = {} if corpus_overrides is None else corpus_overrides
    for name, value in corpus_overrides.items():
        if name not in datasets:
            raise ValueError("corpus override must name a requested dataset")
        _integer("corpus override", value)

    per_dataset: Dict[str, Dict[str, Any]] = {}
    datasets_total = 0
    for name in datasets:
        if name in per_dataset:
            raise ValueError("duplicate dataset %r" % name)
        item = estimate_dataset(name, extra_copies=extra_copies,
                                corpus_bytes=corpus_overrides.get(name))
        per_dataset[name] = item
        datasets_total += item["total_bytes"]

    packed = sum(item["docs"] for item in per_dataset.values()) * packed_record_bytes
    replicated = (datasets_total + packed) * (ceph_replicas if scope == "deployment" else 1)
    subtotal = replicated + (deployment_overhead_bytes or 0)
    if not math.isfinite(subtotal * build_temp_fraction):
        raise ValueError("build temp calculation overflow")
    build_temp = max(
        int(build_temp_min_bytes),
        int(math.ceil(subtotal * build_temp_fraction)),
    )
    estimated = subtotal + model_cache_bytes + build_temp
    required_free = estimated + free_reserve_bytes
    refusals: List[str] = []
    unknown = [name for name, item in per_dataset.items()
               if item["estimate_status"] == "lower_bound"]
    if unknown:
        refusals.append("REFUSED: unknown corpus bytes for %s; arithmetic is a lower bound" %
                        ", ".join(unknown))
    if budget_bytes is None:
        refusals.append("REFUSED: no explicit approved budget")
    if scope == "deployment":
        if deployment_overhead_bytes is None:
            refusals.append("REFUSED: deployment overhead is unknown; supply an explicit allowance")
        if free_reserve_bytes == 0:
            refusals.append("REFUSED: deployment requires a positive free reserve")

    if budget_bytes is not None and int(budget_bytes) == 0:
        refusals.append(
            "REFUSED: storage budget is zero (no approved budget); "
            "estimated %d bytes requires explicit approval" % estimated
        )
    if required_free > free_bytes:
        refusals.append(
            "REFUSED: estimated footprint plus reserve %d bytes (%.2f GiB) exceeds available free "
            "space %d bytes (%.2f GiB); nothing was deleted"
            % (required_free, required_free / 1024 ** 3, free_bytes, free_bytes / 1024 ** 3)
        )
    if budget_bytes is not None and int(budget_bytes) > 0 and estimated > int(budget_bytes):
        refusals.append(
            "REFUSED: estimated %d bytes exceeds approved budget %d bytes"
            % (estimated, int(budget_bytes))
        )
    requested_large = sorted(set(datasets).intersection(LARGE_DATASETS))
    if len(requested_large) == len(LARGE_DATASETS) and not allow_both_large:
        refusals.append(
            "REFUSED: both large corpora requested (%s); encoding HotpotQA and "
            "NQ together requires explicit approval"
            % ", ".join(requested_large)
        )

    headroom = free_bytes - required_free
    return {
        "scope": scope,
        "status": "refused" if refusals else "scoped_budget_fit",
        "estimate_status": "lower_bound" if unknown or (
            scope == "deployment" and deployment_overhead_bytes is None) else "scoped_estimate",
        "download_ready": scope == "deployment" and not refusals,
        "readiness_scope": "conditional on supplied allowances and capacity scope; not per-OSD fit",
        "free_reserve_bytes": free_reserve_bytes,
        "required_free_bytes": required_free,
        "ceph_replicas_applied": ceph_replicas if scope == "deployment" else 1,
        "packed_record_bytes": packed_record_bytes,
        "packed_records_bytes": packed,
        "replicated_dataset_bytes": replicated,
        "deployment_overhead_bytes": deployment_overhead_bytes,
        "datasets_requested": list(datasets),
        "datasets": per_dataset,
        "datasets_total_bytes": datasets_total,
        "model_cache_bytes": int(model_cache_bytes),
        "build_temp_bytes": build_temp,
        "estimated_total_bytes": estimated,
        "free_bytes": int(free_bytes),
        "budget_bytes": None if budget_bytes is None else int(budget_bytes),
        "headroom_bytes": headroom,
        "arithmetic_fits_free": required_free <= free_bytes,
        "arithmetic_fits_budget": budget_bytes is not None and budget_bytes > 0 and estimated <= budget_bytes,
        "fits_free": None if unknown or (scope == "deployment" and deployment_overhead_bytes is None)
                     else required_free <= free_bytes,
        "fits_budget": (
            None if unknown or (scope == "deployment" and deployment_overhead_bytes is None)
            else budget_bytes is not None and budget_bytes > 0 and estimated <= budget_bytes
        ),
        "large_corpora": requested_large,
        "allow_both_large": bool(allow_both_large),
        "refusals": refusals,
    }


def available_bytes(path: str) -> int:
    """Free bytes reported by statvfs for an existing path (no network)."""
    usage = os.statvfs(path)
    return int(usage.f_bavail) * int(usage.f_frsize)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline storage budget estimator (never downloads or deletes data)"
    )
    parser.add_argument(
        "--datasets",
        default="scifact,fiqa",
        help="comma-separated datasets: " + ",".join(sorted(DATASET_SIZES)),
    )
    parser.add_argument(
        "--free-bytes",
        type=int,
        default=None,
        help="current free bytes (from df -B1); default: statvfs --free-path",
    )
    parser.add_argument(
        "--free-path",
        default=".",
        help="path used for statvfs when --free-bytes is omitted",
    )
    parser.add_argument(
        "--budget-bytes",
        type=int,
        default=None,
        help="approved storage budget in bytes; 0 means no approval",
    )
    parser.add_argument("--extra-copies", type=int, default=EXTRA_COPIES)
    parser.add_argument("--scope", choices=("offline_artifacts", "deployment"), default="offline_artifacts")
    parser.add_argument("--corpus-bytes", action="append", default=[], metavar="DATASET=BYTES",
                        help="explicit corpus allowance; repeat per dataset")
    parser.add_argument("--free-reserve-bytes", type=int, default=0)
    parser.add_argument("--ceph-replicas", type=int, default=3)
    parser.add_argument("--packed-record-bytes", type=int, default=0,
                        help="additional packed record per document (e.g. 128); 0 excludes it")
    parser.add_argument("--deployment-overhead-bytes", type=int, default=None,
                        help="explicit additional physical allowance, beyond replicated artifacts")
    parser.add_argument("--model-cache-bytes", type=int, default=MODEL_CACHE_BYTES)
    parser.add_argument(
        "--build-temp-fraction", type=float, default=BUILD_TEMP_FRACTION
    )
    parser.add_argument(
        "--build-temp-min-bytes", type=int, default=BUILD_TEMP_MIN_BYTES
    )
    parser.add_argument(
        "--allow-both-large",
        action="store_true",
        help="explicitly approve encoding HotpotQA and NQ together",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    datasets = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    try:
        overrides = {}
        for entry in args.corpus_bytes:
            name, value = entry.split("=", 1)
            if name in overrides:
                raise ValueError("duplicate corpus override")
            overrides[name] = int(value)
        free_bytes = args.free_bytes
        if free_bytes is None:
            free_bytes = available_bytes(args.free_path)
        plan = estimate_plan(
            datasets,
            free_bytes=free_bytes,
            budget_bytes=args.budget_bytes,
            extra_copies=args.extra_copies,
            model_cache_bytes=args.model_cache_bytes,
            build_temp_fraction=args.build_temp_fraction,
            build_temp_min_bytes=args.build_temp_min_bytes,
            allow_both_large=args.allow_both_large,
            corpus_overrides=overrides,
            scope=args.scope,
            free_reserve_bytes=args.free_reserve_bytes,
            ceph_replicas=args.ceph_replicas,
            packed_record_bytes=args.packed_record_bytes,
            deployment_overhead_bytes=args.deployment_overhead_bytes,
        )
    except (ValueError, OSError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print(json.dumps(plan, indent=2, sort_keys=True))
    if plan["refusals"]:
        for reason in plan["refusals"]:
            print(reason, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
