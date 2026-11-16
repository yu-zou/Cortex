#!/usr/bin/env python3
"""Fail-closed per-query compatibility preflight, not a cost simulator.

Artifact hashes verify bytes; identity fields remain declarations until bound
to the actual index/query/profile by a future executor. A compatible reference
never authorizes integration: no global candidate barrier executor exists here.
No quality configuration selection or qrels processing is performed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


IDENTITY = ("index_namespace", "index_version", "index_sha256", "query_id",
            "query_sha256", "profile_sha256", "nprobe", "top_r", "top_k",
            "metric", "by_residual", "encoder_revision")
EXECUTION_INPUTS = (
    "per-list candidate IDs/scores and deterministic local-top-R tie semantics",
    "global dedup/top-R merge placement, resource, service profile and complete input barrier",
    "versioned document-ID to raw-object/offset/owner mapping and raw-cache identity",
    "raw fetch route, batching/read amplification, cache policy and calibrated service profile",
    "rerank CPU placement/resource sharing and calibrated exact-IP service profile",
    "local candidate and final result network payloads/routes and calibrated service profile",
    "residual-IP scan/local-top-R service profile for the pinned index/layout",
    "global candidate barrier executor not implemented",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value, size=64):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{%d}" % size, value),
             "missing or invalid identity hash")


def _identity(value):
    _require(type(value) is dict and set(value) == set(IDENTITY), "incomplete identity")
    for name in ("index_namespace", "index_version", "query_id"):
        v = value[name]
        _require(type(v) is str and v.strip() == v and bool(v), "invalid identity text")
    for name in ("index_sha256", "query_sha256", "profile_sha256"):
        _hash(value[name])
    _hash(value["encoder_revision"], 40)
    for name in ("nprobe", "top_r", "top_k"):
        _require(type(value[name]) is int and value[name] > 0, "invalid search count")
    _require(value["top_r"] == 64 and value["top_k"] <= 64,
             "this contract requires query-global R64")
    _require(value["metric"] == "IP" and type(value["by_residual"]) is bool,
             "explicit IP/residual identity required")


def _json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            _require(k not in result, "duplicate JSON key")
            result[k] = v
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def _artifact(ref):
    _require(type(ref) is dict and set(ref) == {"path", "sha256"}, "artifact reference missing")
    _hash(ref["sha256"])
    data = Path(ref["path"]).read_bytes()
    _require(hashlib.sha256(data).hexdigest() == ref["sha256"], "artifact hash mismatch")
    return _json(data)


def _ids(values, count):
    _require(type(values) is list and len(values) == count, "candidate/result count mismatch")
    _require(all(type(v) is int and v >= 0 for v in values), "invalid document row IDs")
    _require(len(set(values)) == count, "duplicate document row IDs")


def _scores(values, count):
    _require(type(values) is list and len(values) == count, "score count mismatch")
    _require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "invalid score")


def assess_alignment(expected, quality, cost):
    """Validate ONE query reference. Always blocked for integrated DES costs.

    Rows artifact: JSON array containing exactly one explicit query row. Quality
    artifact: explicit positive cohort count and measured ndcg@10. These checks
    bind saved references, not their computational correctness or chronology.
    Missing information is an error, never an inferred zero observation.
    """
    blockers = []
    try:
        _identity(expected)
        _require(quality is not None, "quality evidence missing")
        _require(type(quality) is dict, "quality evidence must be an object")
        _require(type(cost) is dict, "cost evidence missing")
        for label, document in (("quality", quality), ("cost", cost)):
            _identity(document.get("identity"))
            _require(document["identity"] == expected, label + " index/profile/query identity mismatch")
            _require(document.get("candidate_scope") == "query_global_unique",
                     label + " candidate scope must be query-global unique R64")
        _require(type(cost.get("rerank_per_cluster")) is int and cost["rerank_per_cluster"] == 0,
                 "per-cluster rerank is not query-global R64")
        _require(type(quality.get("query_count")) is int and quality["query_count"] > 0,
                 "quality query_count missing or invalid")
        metrics = _artifact(quality.get("quality"))
        _require(type(metrics) is dict and type(metrics.get("query_count")) is int
                 and metrics["query_count"] == quality["query_count"], "quality count mismatch")
        ndcg = metrics.get("ndcg@10")
        _require(type(ndcg) in (int, float) and math.isfinite(ndcg) and 0 <= ndcg <= 1,
                 "quality data missing or invalid")
        rows = _artifact(quality.get("rows"))
        _require(type(rows) is list and len(rows) == 1, "exactly one query reference required")
        row = rows[0]
        _require(type(row) is dict and row.get("query_id") == expected["query_id"]
                 and row.get("query_sha256") == expected["query_sha256"], "query row identity mismatch")
        for field, count in (("candidate", 64), ("result", expected["top_k"])):
            _ids(row.get(field + "_ids"), count)
            _scores(row.get(field + "_scores"), count)
        _require(set(row["result_ids"]) <= set(row["candidate_ids"]), "result outside candidates")
    except (ValueError, TypeError, OSError, KeyError) as exc:
        blockers.append(str(exc))
    return dict(schema="cortex.query_cost_alignment.v1", status="blocked",
        reference_compatible=not blockers, integration_ready=False,
        candidate_scope="query_global_unique_R64", cost_or_quality_computed=False,
        blockers=blockers + list(EXECUTION_INPUTS),
        evidence_boundary="saved-reference compatibility only; no selection replay or cost authorization")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", required=True, type=Path)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--cost", type=Path)
    args = parser.parse_args()
    try:
        read = lambda p: _json(p.read_text()) if p else None
        result = assess_alignment(read(args.expected), read(args.quality), read(args.cost))
    except (ValueError, TypeError, OSError) as exc:
        result = dict(status="blocked", integration_ready=False, blockers=[str(exc)])
    print(json.dumps(result, indent=2, allow_nan=False))
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
