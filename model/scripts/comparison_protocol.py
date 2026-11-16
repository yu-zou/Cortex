"""Offline frozen comparison gate. Hash references are declarations, not attestation.

validate_run raises ValueError without echoing config values. compare_runs never
selects parameters: validation-selected and immutable test reports enter together.
No I/O, packages, cloud access, timing, or performance inference is performed.
"""
import hashlib
import json
import math
import re
from decimal import Decimal


def digest(value):
    """Canonical JSON SHA256, also used to bind selected search parameters."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fields(obj, names):
    _require(type(obj) is dict and set(obj) == set(names.split()), "invalid fields")


def _text(value):
    _require(type(value) is str and bool(value.strip()) and value == value.strip(),
             "nonempty unpadded text required")


def _hash(value, size=64):
    _require(type(value) is str and re.fullmatch("[0-9a-f]{%d}" % size, value),
             "invalid pinned hash")


def _number(value, minimum=0):
    _require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
             "finite nonnegative number required")


def _integer(value, minimum=1):
    _require(type(value) is int and value >= minimum, "invalid integer")


def _clean(value):
    if type(value) is dict:
        for key, item in value.items():
            _text(key)
            _require(not re.search(r"secret|password|token|credential|api.?key|access.?key|private.?key",
                                   key, re.I), "secret-bearing config key")
            _clean(item)
    elif type(value) is list:
        for item in value:
            _clean(item)
    elif type(value) is str:
        _require(not re.search(r"://[^/\s]*@|Bearer\s+|BEGIN .*PRIVATE KEY|"
                               r"(?:password|token|secret|api[_-]?key)=|\b(?:AKIA|ASIA)[A-Z0-9]{16}",
                               value, re.I), "credential-bearing config value")
    else:
        _require(value is None or type(value) in (bool, int, float), "JSON values required")
        if type(value) in (int, float):
            _require(math.isfinite(value), "finite config value required")


def _resources(r):
    _fields(r, "instances client storage")
    _require(type(r["instances"]) is list and r["instances"], "instance topology required")
    ids = []
    for host in r["instances"]:
        _fields(host, "instance instance_type cpu_vcpu memory_bytes network_bytes_per_s devices processes")
        _text(host["instance"])
        _text(host["instance_type"])
        ids.append(host["instance"])
        for key in ("cpu_vcpu", "memory_bytes", "network_bytes_per_s"):
            _integer(host[key])
        _require(type(host["devices"]) is list, "explicit devices required")
        for device in host["devices"]:
            _fields(device, "kind count evidence_sha256")
            _text(device["kind"])
            _integer(device["count"])
            _hash(device["evidence_sha256"])
        _require(type(host["processes"]) is list and host["processes"], "process quotas required")
        for process in host["processes"]:
            _fields(process, "role cpu_vcpu memory_bytes")
            _text(process["role"])
            _number(process["cpu_vcpu"], 0.001)
            _integer(process["memory_bytes"])
        for key in ("cpu_vcpu", "memory_bytes"):
            _require(sum(p[key] for p in host["processes"]) <= host[key], "process quota exceeds host")
    _require(len(ids) == len(set(ids)), "duplicate instance identity")
    _fields(r["client"], "instance_type cpu_vcpu memory_bytes network_bytes_per_s")
    _text(r["client"]["instance_type"])
    for key in ("cpu_vcpu", "memory_bytes", "network_bytes_per_s"):
        _integer(r["client"][key])
    s = r["storage"]
    _fields(s, "storage_class disks capacity_bytes iops bandwidth_bytes_per_s replicas min_size failure_domain io_path physical_pool_sha256")
    if s["physical_pool_sha256"] is not None:
        _hash(s["physical_pool_sha256"])
    for key in ("storage_class", "failure_domain", "io_path"):
        _text(s[key])
    for key in ("disks", "capacity_bytes", "iops", "bandwidth_bytes_per_s", "replicas", "min_size"):
        _integer(s[key])
    _require(s["min_size"] <= s["replicas"], "invalid replicas/min_size")


def validate_run(run):
    """Validate a closed JSON manifest; return nothing and never modify it."""
    _clean(run)
    _fields(run, "schema_version category protocol index selection backend resources cost")
    _require(type(run["schema_version"]) is int and run["schema_version"] == 1, "invalid version")
    _require(run["category"] in ("same_resource", "same_cost", "architecture_specific"), "invalid category")
    p = run["protocol"]
    _fields(p, "dataset corpus_sha256 queryset_sha256 qrels_sha256 train_queryset_sha256 validation_queryset_sha256 split tuning_split embedding metric direction top_k rerank filter failure_policy cache_mode index_namespace index_version quality")
    for key in ("dataset", "filter", "failure_policy", "cache_mode", "index_namespace", "index_version"):
        _text(p[key])
    for key in ("corpus_sha256", "queryset_sha256", "qrels_sha256", "train_queryset_sha256", "validation_queryset_sha256"):
        _hash(p[key])
    _require(len({p[k] for k in ("queryset_sha256", "train_queryset_sha256", "validation_queryset_sha256")}) == 3,
             "split querysets must differ; verify ID disjointness externally")
    _require(p["split"] == "test" and p["tuning_split"] == "validation", "test reporting / validation tuning required")
    e = p["embedding"]
    _fields(e, "model revision dimension normalized")
    _text(e["model"])
    _hash(e["revision"], 40)
    _integer(e["dimension"])
    _require(type(e["normalized"]) is bool, "normalized must be boolean")
    _require(p["metric"] in ("IP", "L2"), "invalid metric")
    _require(p["direction"] == {"IP": "higher", "L2": "lower"}[p["metric"]], "invalid metric direction")
    _integer(p["top_k"])
    rr = p["rerank"]
    _fields(rr, "method candidates scope")
    _require(rr["method"] in ("none", "exact_IP", "exact_L2"), "invalid rerank method")
    _require(rr["method"] in ("none", "exact_" + p["metric"]), "rerank metric mismatch")
    _integer(rr["candidates"], p["top_k"])
    _require(rr["scope"] in ("per_query_union", "per_cluster"), "invalid candidate scope")
    q = p["quality"]
    _fields(q, "metric target k threshold tolerance registration_sha256 declared_at")
    _require((q["metric"], q["target"]) in (("ndcg", "qrels"), ("recall", "qrels"),
                                                         ("overlap", "ann_exact")), "invalid quality metric/target")
    _require(type(q["k"]) is int and q["k"] == p["top_k"], "quality k mismatch")
    for key in ("threshold", "tolerance"):
        _number(q[key])
        _require(q[key] <= 1, "quality range must be [0,1]")
    _require(q["tolerance"] <= q["threshold"], "invalid quality tolerance")
    _hash(q["registration_sha256"])
    _require(q["declared_at"] == "before_validation", "quality gate must be predeclared")
    _fields(run["index"], "sha256 namespace version")
    _hash(run["index"]["sha256"])
    _require((run["index"]["namespace"], run["index"]["version"]) ==
             (p["index_namespace"], p["index_version"]), "index namespace/version mismatch")
    sel = run["selection"]
    _fields(sel, "split config config_sha256 evidence_sha256 validation_score final_test_score test_used_for_selection")
    _require(sel["split"] == "validation", "selection requires validation split")
    _require(sel["test_used_for_selection"] is False, "test tuning prohibited")
    _require(type(sel["config"]) is dict and sel["config"], "selected config required")
    _hash(sel["config_sha256"])
    _hash(sel["evidence_sha256"])
    _require(digest(sel["config"]) == sel["config_sha256"], "selected config hash mismatch")
    for key in ("validation_score", "final_test_score"):
        _number(sel[key])
        _require(sel[key] <= 1, "quality score range must be [0,1]")
    b = run["backend"]
    _fields(b, "kind real_components modeled_components result_source evidence_sha256 profile_sha256 device_sha256 bit_sha256")
    _require(b["kind"] in ("cpu", "emulated", "physical"), "offline projection excluded")
    for key in ("real_components", "modeled_components"):
        _require(type(b[key]) is list, "component list required")
        for component in b[key]:
            _text(component)
        _require(len(set(b[key])) == len(b[key]), "duplicate component")
    _require(b["real_components"] and not set(b["real_components"]) & set(b["modeled_components"]), "component overlap")
    _hash(b["evidence_sha256"])
    if b["kind"] == "emulated":
        _require(b["modeled_components"] and b["result_source"] == "response_oracle", "emulation labels inconsistent")
        _hash(b["profile_sha256"])
    else:
        _require(not b["modeled_components"] and b["result_source"] == "online_compute", "online backend labels inconsistent")
        _require(b["profile_sha256"] is None, "online backend cannot carry emulation profile")
    if b["kind"] == "physical":
        _require(b["device_sha256"] is not None and b["bit_sha256"] is not None, "physical evidence required")
        _hash(b["device_sha256"])
        _hash(b["bit_sha256"])
    else:
        _require(b["device_sha256"] is None and b["bit_sha256"] is None, "nonphysical device labels prohibited")
    _resources(run["resources"])
    c = run["cost"]
    _fields(c, "currency unit build run storage client duration_s evidence_sha256 basis")
    _require(type(c["currency"]) is str and re.fullmatch("[A-Z]{3}", c["currency"]), "invalid currency")
    _require(c["unit"] == "currency_total", "explicit total cost unit required")
    for key in ("build", "run", "storage", "client"):
        _number(c[key])
    _number(c["duration_s"], 0.001)
    _hash(c["evidence_sha256"])
    _require(c["basis"] in ("unit_fixture", "quote", "invoice"), "invalid cost basis")


def _provisioned(r):
    """Canonical reused-pool hardware, excluding software process allocations."""
    hosts = []
    for host in r["instances"]:
        hardware = {k: v for k, v in host.items() if k not in ("processes", "devices")}
        hardware["devices"] = sorted(host["devices"], key=digest)
        hosts.append(hardware)
    return dict(instances=sorted(hosts, key=lambda h: h["instance"]),
                client=r["client"], storage={k: v for k, v in r["storage"].items()
                                           if k != "io_path"})


def compare_runs(left, right):
    """Check declared comparability, not truth of referenced artifacts or speedup."""
    for run in (left, right):
        validate_run(run)
    category = left["category"]
    _require(category == right["category"], "category mismatch")
    _require(left["protocol"] == right["protocol"], "quality protocol mismatch")
    if category == "same_resource":
        pools = [r["resources"]["storage"]["physical_pool_sha256"] for r in (left, right)]
        _require(pools[0] is not None and pools[0] == pools[1],
                 "storage physical equivalence unestablished; use architecture_specific")
        _require(_provisioned(left["resources"]) == _provisioned(right["resources"]),
                 "exact provisioned resource topology/totals and reused instance IDs required")
    if category == "same_cost":
        def cost_key(c):
            return (c["currency"], c["unit"], c["duration_s"], c["basis"],
                    sum(Decimal(str(c[k])) for k in ("build", "run", "storage", "client")))
        _require(cost_key(left["cost"]) == cost_key(right["cost"]), "exact cost budget/units/window required")
    q = left["protocol"]["quality"]
    scores = [run["selection"][k] for run in (left, right)
              for k in ("validation_score", "final_test_score")]
    threshold, tolerance = Decimal(str(q["threshold"])), Decimal(str(q["tolerance"]))
    matched = (all(Decimal(str(scores[i])) >= threshold for i in (0, 2))
               and all(Decimal(str(scores[i])) >= threshold - tolerance for i in (1, 3))
               and abs(Decimal(str(scores[1])) - Decimal(str(scores[3]))) <= tolerance)
    kinds = {left["backend"]["kind"], right["backend"]["kind"]}
    paths = [r["resources"]["storage"]["io_path"] for r in (left, right)]
    claim = ("conditional_emulation_comparison" if "emulated" in kinds else
             "conditional_architecture_comparison" if paths[0] != paths[1] else
             "physical_backend_comparison" if "physical" in kinds else "software_comparison")
    return dict(category=category, quality_matched=matched,
                architecture_difference=paths[0] != paths[1], io_paths=paths,
                equal_resource_claim=category == "same_resource" and matched,
                claim=claim if matched else "unmatched_quality",
                hardware_speedup_claim=False,
                evidence_scope="unit_fixture" if any(r["cost"]["basis"] == "unit_fixture"
                                                     for r in (left, right)) else "declared_artifact_references")
