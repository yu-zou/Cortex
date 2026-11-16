"""Task 15: workload arrival sources and the per-request DAG metrics runner.

Open-loop contract
------------------
Arrival times are always computed ahead of execution and never consulted from
completion times.  Two sources are provided: a real RAGPulse trace replay
(``TraceArrivalSource``) and a seeded Poisson process (``PoissonArrivalSource``).

Per-query DAG
-------------
Every query executes the same DAG: nprobe cluster selection -> per-tile compute
tasks -> rerank -> merge.  The runner books each stage on the engine's resource
calendars, so queueing is explicit and no wall clock is used.

Metrics contract
----------------
- latency is measured from the **intended arrival** time and reported together
  with the observed admission-based latency (no coordinated omission);
- timeout and failed queries stay in the denominator, their served recall is 0
  while algorithm recall is averaged over successful queries only;
- P50/P95/P99 are nearest-rank statistics of the raw samples;
- any query that never finishes makes the whole run ``incomplete`` and forbids
  a success benchmark label.

Run artifacts are sealed under ``results/<run_id>/``: an immutable config JSON,
provenance, the per-request JSONL log and the summary JSON.  A colliding
``run_id`` and any target outside the project are refused.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from strata_sim.contracts.schema import ContractError
from strata_sim.des.engine import DesEngine, DesError
from strata_sim.des.resources import ResourceError, ResourceKind
from strata_sim.safety import DEFAULT_PROJECT_ROOT, assert_within_project

PERCENTILES = (50, 95, 99)


class MetricsError(ContractError):
    """A workload or metrics invariant was violated."""


class WorkloadError(MetricsError):
    """The workload itself is invalid (NaN, empty, duplicate, ...)."""


class RunCollisionError(MetricsError):
    """The run directory or one of its artifacts already exists."""


class IncompleteRunError(MetricsError):
    """The run has unfinished queries and therefore no success label."""


def _finite_nonneg_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        if isinstance(value, float) and not math.isfinite(value):
            raise WorkloadError(f"{name} must be finite, got {value!r}")
        raise WorkloadError(f"{name} must be a non-negative int, got {value!r}")
    if value < 0:
        raise WorkloadError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _positive_int(name: str, value: object) -> int:
    result = _finite_nonneg_int(name, value)
    if result < 1:
        raise WorkloadError(f"{name} must be >= 1, got {value!r}")
    return result


def _nonempty_str(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise WorkloadError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _id_tuple(name: str, values: object) -> tuple[int, ...]:
    if values is None:
        return ()
    if not isinstance(values, (tuple, list)):
        raise WorkloadError(f"{name} must be a tuple of ints, got {values!r}")
    checked = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkloadError(f"{name} entries must be ints, got {value!r}")
        if value < 0:
            raise WorkloadError(f"{name} entries must be non-negative, got {value!r}")
        checked.append(int(value))
    return tuple(checked)


@dataclass(frozen=True)
class Arrival:
    """One open-loop offered query (never derived from a completion)."""

    query_id: str
    time_ns: int
    deadline_ns: int
    tenant: str = "derived"
    session_id: str | None = None
    attempt_id: int = 0

    def __post_init__(self) -> None:
        _nonempty_str("query_id", self.query_id)
        _finite_nonneg_int("time_ns", self.time_ns)
        _finite_nonneg_int("deadline_ns", self.deadline_ns)
        _nonempty_str("tenant", self.tenant)
        _finite_nonneg_int("attempt_id", self.attempt_id)
        if self.session_id is not None:
            _nonempty_str("session_id", self.session_id)


@dataclass(frozen=True)
class TileWork:
    """One tile of one probed cluster."""

    tile_index: int
    doc_count: int
    compute_ns: int

    def __post_init__(self) -> None:
        _finite_nonneg_int("tile_index", self.tile_index)
        _finite_nonneg_int("doc_count", self.doc_count)
        _finite_nonneg_int("compute_ns", self.compute_ns)


@dataclass(frozen=True)
class ClusterWork:
    """The tiles a query probes in one coarse cluster."""

    cluster_id: int
    tiles: tuple[TileWork, ...] = ()

    def __post_init__(self) -> None:
        _finite_nonneg_int("cluster_id", self.cluster_id)
        if not isinstance(self.tiles, (tuple, list)):
            raise WorkloadError(f"tiles must be a tuple, got {self.tiles!r}")
        checked = []
        seen = set()
        for tile in self.tiles:
            if not isinstance(tile, TileWork):
                raise WorkloadError(f"tiles entries must be TileWork, got {tile!r}")
            if tile.tile_index in seen:
                raise WorkloadError(f"duplicate tile_index {tile.tile_index}")
            seen.add(tile.tile_index)
            checked.append(tile)
        object.__setattr__(self, "tiles", tuple(checked))


@dataclass(frozen=True)
class QueryPlan:
    """Functional plan of one query DAG (nprobe -> tiles -> rerank -> merge)."""

    query_id: str
    probe_ns: int
    clusters: tuple[ClusterWork, ...] = ()
    rerank_ns: int = 0
    merge_ns: int = 0
    topk_ids: tuple[int, ...] = ()
    gt_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _nonempty_str("query_id", self.query_id)
        _finite_nonneg_int("probe_ns", self.probe_ns)
        _finite_nonneg_int("rerank_ns", self.rerank_ns)
        _finite_nonneg_int("merge_ns", self.merge_ns)
        if not isinstance(self.clusters, (tuple, list)):
            raise WorkloadError(f"clusters must be a tuple, got {self.clusters!r}")
        checked = []
        seen = set()
        for cluster in self.clusters:
            if not isinstance(cluster, ClusterWork):
                raise WorkloadError(f"clusters entries must be ClusterWork, got {cluster!r}")
            if cluster.cluster_id in seen:
                raise WorkloadError(f"duplicate cluster_id {cluster.cluster_id}")
            seen.add(cluster.cluster_id)
            checked.append(cluster)
        object.__setattr__(self, "clusters", tuple(checked))
        object.__setattr__(self, "topk_ids", _id_tuple("topk_ids", self.topk_ids))
        object.__setattr__(self, "gt_ids", _id_tuple("gt_ids", self.gt_ids))

    @property
    def tile_count(self) -> int:
        return sum(len(cluster.tiles) for cluster in self.clusters)


@dataclass(frozen=True)
class RunResources:
    """Engine resource names and capacities used by the DAG runner."""

    compute: str = "compute"
    cpu: str = "cpu"
    compute_capacity: int = 1
    cpu_capacity: int = 1

    def install(self, engine: DesEngine) -> "RunResources":
        _nonempty_str("compute", self.compute)
        _nonempty_str("cpu", self.cpu)
        _positive_int("compute_capacity", self.compute_capacity)
        _positive_int("cpu_capacity", self.cpu_capacity)
        existing = set(engine.resource_names)
        if self.compute not in existing:
            engine.add_resource(
                self.compute, ResourceKind.FPGA_COMPUTE, capacity=self.compute_capacity
            )
        elif engine.resource(self.compute).capacity != self.compute_capacity:
            raise WorkloadError(
                f"resource {self.compute!r} exists with capacity "
                f"{engine.resource(self.compute).capacity}, requested {self.compute_capacity}"
            )
        if self.cpu not in existing:
            engine.add_resource(self.cpu, ResourceKind.CPU, capacity=self.cpu_capacity)
        elif engine.resource(self.cpu).capacity != self.cpu_capacity:
            raise WorkloadError(
                f"resource {self.cpu!r} exists with capacity "
                f"{engine.resource(self.cpu).capacity}, requested {self.cpu_capacity}"
            )
        return self


@dataclass(frozen=True)
class RequestRecord:
    """The auditable per-request log line."""

    query_id: str
    attempt_id: int
    tenant: str
    session_id: str | None
    intended_arrival_ns: int
    admit_ns: int
    start_ns: int | None
    finish_ns: int | None
    deadline_ns: int
    status: str
    latency_intended_ns: int | None
    latency_observed_ns: int | None
    served_recall: float | None
    algorithm_recall: float | None
    topk_ids: tuple[int, ...]
    gt_ids: tuple[int, ...]
    tile_count: int

    @property
    def lags_deadline(self) -> bool:
        return self.finish_ns is not None and self.finish_ns > self.deadline_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "attempt_id": self.attempt_id,
            "tenant": self.tenant,
            "session_id": self.session_id,
            "arrival_ns": self.intended_arrival_ns,
            "intended_arrival_ns": self.intended_arrival_ns,
            "admit_ns": self.admit_ns,
            "start_ns": self.start_ns,
            "finish_ns": self.finish_ns,
            "deadline_ns": self.deadline_ns,
            "status": self.status,
            "latency_intended_ns": self.latency_intended_ns,
            "latency_observed_ns": self.latency_observed_ns,
            "served_recall": self.served_recall,
            "algorithm_recall": self.algorithm_recall,
            "topk_ids": list(self.topk_ids),
            "gt_ids": list(self.gt_ids),
            "tile_count": self.tile_count,
        }


def recall_at_k(topk_ids: Sequence[int], gt_ids: Sequence[int], k: int) -> float:
    """Recall@k of the top-k set against the ground-truth top-k set."""
    k = _positive_int("k", k)
    if not gt_ids:
        raise WorkloadError("ground truth is empty; recall is undefined")
    if k > len(gt_ids):
        raise WorkloadError(f"k={k} exceeds ground-truth size {len(gt_ids)}")
    top = list(topk_ids)[:k]
    expected = set(gt_ids[:k])
    hits = len(set(top) & expected)
    return hits / k


def percentile(samples: Iterable[int], p: float) -> int | None:
    """Nearest-rank percentile (ceil(p/100 * n)); None for no samples."""
    if not 0 < p <= 100:
        raise WorkloadError(f"percentile must be in (0, 100], got {p!r}")
    values = sorted(samples)
    if not values:
        return None
    rank = math.ceil(p / 100 * len(values))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]


def percentiles(samples: Iterable[int], ps: Sequence[float] = PERCENTILES) -> dict[str, int | None]:
    values = list(samples)
    return {f"p{int(p)}": percentile(values, p) for p in ps}


@dataclass(frozen=True)
class WorkloadResult:
    """All per-request records plus the derived run status."""

    records: tuple[RequestRecord, ...]
    status: str
    stop_ns: int | None = None

    @property
    def incomplete_query_ids(self) -> tuple[str, ...]:
        return tuple(
            record.query_id
            for record in self.records
            if record.status in ("incomplete", "failed")
        )

    def summary(self) -> dict[str, Any]:
        records = list(self.records)
        offered = len(records)
        succeeded = sum(1 for record in records if record.status == "ok")
        timeout = sum(1 for record in records if record.status == "timeout")
        failed = sum(1 for record in records if record.status == "failed")
        incomplete = sum(1 for record in records if record.status == "incomplete")
        completed = succeeded + timeout
        failures = timeout + failed + incomplete

        gt_records = [record for record in records if record.served_recall is not None]
        served_recall = (
            sum(float(record.served_recall) for record in gt_records) / len(gt_records)
            if gt_records
            else None
        )
        algorithm_samples = [
            float(record.algorithm_recall)
            for record in records
            if record.algorithm_recall is not None
        ]
        algorithm_recall = (
            sum(algorithm_samples) / len(algorithm_samples) if algorithm_samples else None
        )

        intended = [
            record.latency_intended_ns
            for record in records
            if record.latency_intended_ns is not None
        ]
        observed = [
            record.latency_observed_ns
            for record in records
            if record.latency_observed_ns is not None
        ]
        finishes = [record.finish_ns for record in records if record.finish_ns is not None]
        window_start = min(record.intended_arrival_ns for record in records)
        if self.stop_ns is not None:
            window_end = max(self.stop_ns, window_start)
        elif finishes:
            window_end = max(finishes)
        else:
            window_end = window_start
        span_ns = window_end - window_start
        offered_goodput_qps = offered * 1e9 / span_ns if span_ns > 0 else None
        achieved_goodput_qps = succeeded * 1e9 / span_ns if span_ns > 0 else None

        return {
            "status": self.status,
            "benchmark_label": (
                "success" if self.status == "complete" else "forbidden-incomplete"
            ),
            "offered": offered,
            "completed": completed,
            "succeeded": succeeded,
            "timeout": timeout,
            "failed": failed,
            "incomplete": incomplete,
            "success_rate": succeeded / offered if offered else None,
            "failure_rate": failures / offered if offered else None,
            "served_recall": served_recall,
            "algorithm_recall": algorithm_recall,
            "latency_samples": {"intended": len(intended), "observed": len(observed)},
            "latency_ns": {
                "intended": percentiles(intended),
                "observed": percentiles(observed),
            },
            "offered_goodput_qps": offered_goodput_qps,
            "achieved_goodput_qps": achieved_goodput_qps,
            "window_start_ns": window_start,
            "window_end_ns": window_end,
            "span_ns": span_ns,
            "incomplete_query_ids": list(self.incomplete_query_ids),
        }


def _admission_time(
    in_flight: list[tuple[int, int]], arrival_ns: int, limit: int
) -> int:
    admitted = arrival_ns
    active = sorted(end for _start, end in in_flight if end > admitted)
    while len(active) >= limit:
        admitted = active.pop(0)
        active = [end for end in active if end > admitted]
    return admitted


def _book_query(
    engine: DesEngine, names: RunResources, plan: QueryPlan, start_ns: int
) -> int:
    cursor = start_ns
    if plan.probe_ns > 0:
        cursor = engine.acquire(
            names.cpu, f"{plan.query_id}:probe", plan.probe_ns, start_ns=start_ns
        ).end_ns
    tile_ready = cursor
    tile_ends = [tile_ready]
    for cluster in plan.clusters:
        for tile in cluster.tiles:
            if tile.compute_ns <= 0:
                continue
            task_id = f"{plan.query_id}:tile:{cluster.cluster_id}:{tile.tile_index}"
            end_ns = engine.acquire(
                names.compute, task_id, tile.compute_ns, start_ns=tile_ready
            ).end_ns
            tile_ends.append(end_ns)
    cursor = max(tile_ends)
    if plan.rerank_ns > 0:
        cursor = engine.acquire(
            names.cpu, f"{plan.query_id}:rerank", plan.rerank_ns, start_ns=cursor
        ).end_ns
    if plan.merge_ns > 0:
        cursor = engine.acquire(
            names.cpu, f"{plan.query_id}:merge", plan.merge_ns, start_ns=cursor
        ).end_ns
    return cursor


def _request_record(
    arrival: Arrival, plan: QueryPlan, admit_ns: int, finish_ns: int | None,
    start_ns: int | None, *, stop_ns: int | None,
) -> RequestRecord:
    if finish_ns is None:
        status = "failed"
        latency_intended = None
        latency_observed = None
        recall = None
    else:
        recall = (
            recall_at_k(plan.topk_ids, plan.gt_ids, len(plan.gt_ids))
            if plan.gt_ids
            else None
        )
        latency_intended = finish_ns - arrival.time_ns
        latency_observed = finish_ns - admit_ns
        if stop_ns is not None and finish_ns > stop_ns:
            status = "incomplete"
        elif finish_ns > arrival.deadline_ns:
            status = "timeout"
        else:
            status = "ok"
    served = recall if status == "ok" or recall is None else 0.0
    return RequestRecord(
        query_id=arrival.query_id,
        attempt_id=arrival.attempt_id,
        tenant=arrival.tenant,
        session_id=arrival.session_id,
        intended_arrival_ns=arrival.time_ns,
        admit_ns=admit_ns,
        start_ns=start_ns,
        finish_ns=finish_ns,
        deadline_ns=arrival.deadline_ns,
        status=status,
        latency_intended_ns=latency_intended,
        latency_observed_ns=latency_observed,
        served_recall=served,
        algorithm_recall=recall if status == "ok" else None,
        topk_ids=plan.topk_ids,
        gt_ids=plan.gt_ids,
        tile_count=plan.tile_count,
    )


def run_workload(
    engine: DesEngine,
    workload: Iterable[tuple[Arrival, QueryPlan]],
    *,
    resources: RunResources | None = None,
    stop_ns: int | None = None,
    admission_limit: int | None = None,
) -> WorkloadResult:
    """Execute an open-loop workload and return per-request records."""
    if not isinstance(engine, DesEngine):
        raise WorkloadError(f"engine must be a DesEngine, got {engine!r}")
    items = list(workload)
    if not items:
        raise WorkloadError("workload is empty; refusing to summarize nothing")
    if stop_ns is not None:
        _finite_nonneg_int("stop_ns", stop_ns)
    if admission_limit is not None:
        _positive_int("admission_limit", admission_limit)

    names = (resources if resources is not None else RunResources()).install(engine)
    validated: list[tuple[Arrival, QueryPlan]] = []
    seen: set[tuple[str, int]] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise WorkloadError(f"workload entries must be (Arrival, QueryPlan), got {item!r}")
        arrival, plan = item
        if not isinstance(arrival, Arrival) or not isinstance(plan, QueryPlan):
            raise WorkloadError(f"workload entries must be (Arrival, QueryPlan), got {item!r}")
        if arrival.query_id != plan.query_id:
            raise WorkloadError(
                f"arrival {arrival.query_id!r} does not match plan {plan.query_id!r}"
            )
        key = (arrival.query_id, arrival.attempt_id)
        if key in seen:
            raise WorkloadError(f"duplicate query attempt {key!r}")
        seen.add(key)
        validated.append((arrival, plan))
    validated.sort(key=lambda pair: (pair[0].time_ns, pair[0].query_id, pair[0].attempt_id))

    records: list[RequestRecord] = []
    in_flight: list[tuple[int, int]] = []
    for arrival, plan in validated:
        engine.advance_to(arrival.time_ns)
        admit_ns = arrival.time_ns
        if admission_limit is not None:
            admit_ns = _admission_time(in_flight, arrival.time_ns, admission_limit)
        try:
            finish_ns = _book_query(engine, names, plan, admit_ns)
        except (ResourceError, DesError, ContractError):
            records.append(
                _request_record(arrival, plan, admit_ns, None, None, stop_ns=stop_ns)
            )
            continue
        in_flight.append((admit_ns, finish_ns))
        records.append(
            _request_record(arrival, plan, admit_ns, finish_ns, admit_ns, stop_ns=stop_ns)
        )

    if stop_ns is None or stop_ns >= engine.now_ns:
        engine.run(until_ns=stop_ns)
    run_status = (
        "incomplete"
        if any(record.status in ("incomplete", "failed") for record in records)
        else "complete"
    )
    return WorkloadResult(records=tuple(records), status=run_status, stop_ns=stop_ns)


class PoissonArrivalSource:
    """Seeded open-loop Poisson arrivals (computed before any execution)."""

    def __init__(
        self,
        *,
        seed: int,
        qps: float,
        deadline_ns: int,
        count: int | None = None,
        until_ns: int | None = None,
        start_ns: int = 0,
        tenant: str = "poisson",
        query_prefix: str = "poisson",
    ) -> None:
        _finite_nonneg_int("seed", seed)
        if not isinstance(qps, (int, float)) or isinstance(qps, bool):
            raise WorkloadError(f"qps must be a positive number, got {qps!r}")
        if not math.isfinite(float(qps)) or qps <= 0:
            raise WorkloadError(f"qps must be finite and positive, got {qps!r}")
        _positive_int("deadline_ns", deadline_ns)
        _finite_nonneg_int("start_ns", start_ns)
        _nonempty_str("tenant", tenant)
        _nonempty_str("query_prefix", query_prefix)
        if (count is None) == (until_ns is None):
            raise WorkloadError("exactly one of count or until_ns must be given")
        if count is not None:
            _positive_int("count", count)
        if until_ns is not None:
            _positive_int("until_ns", until_ns)
            if until_ns <= start_ns:
                raise WorkloadError("until_ns must be after start_ns")
        self.seed = int(seed)
        self.qps = float(qps)
        self.deadline_ns = int(deadline_ns)
        self.count = None if count is None else int(count)
        self.until_ns = None if until_ns is None else int(until_ns)
        self.start_ns = int(start_ns)
        self.tenant = tenant
        self.query_prefix = query_prefix

    def arrivals(self) -> list[Arrival]:
        rng = random.Random(self.seed)
        rate_per_ns = self.qps / 1e9
        arrivals: list[Arrival] = []
        cursor = float(self.start_ns)
        index = 0
        while True:
            if self.count is not None and index >= self.count:
                break
            cursor += rng.expovariate(rate_per_ns)
            time_ns = int(round(cursor))
            if self.until_ns is not None and time_ns >= self.until_ns:
                break
            arrivals.append(
                Arrival(
                    query_id=f"{self.query_prefix}-{index:06d}",
                    time_ns=time_ns,
                    deadline_ns=time_ns + self.deadline_ns,
                    tenant=self.tenant,
                )
            )
            index += 1
        return arrivals


class TraceArrivalSource:
    """Real RAGPulse ``events.jsonl`` replay with split, scaling and window."""

    def __init__(
        self,
        path: str | Path,
        *,
        deadline_ns: int,
        split: str | None = None,
        scale_factor: float = 1.0,
        window_start_ns: int = 0,
        window_end_ns: int | None = None,
        limit: int | None = None,
        tenant_field: str = "tenant",
        session_field: str = "session_id",
        time_field: str = "time_ns",
        event_field: str = "event_index",
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise WorkloadError(f"trace events file {self.path} does not exist")
        _positive_int("deadline_ns", deadline_ns)
        if split is not None:
            _nonempty_str("split", split)
        if not isinstance(scale_factor, (int, float)) or isinstance(scale_factor, bool):
            raise WorkloadError(f"scale_factor must be positive, got {scale_factor!r}")
        if not math.isfinite(float(scale_factor)) or scale_factor <= 0:
            raise WorkloadError(f"scale_factor must be finite and positive, got {scale_factor!r}")
        _finite_nonneg_int("window_start_ns", window_start_ns)
        if window_end_ns is not None:
            _finite_nonneg_int("window_end_ns", window_end_ns)
            if window_end_ns <= window_start_ns:
                raise WorkloadError("window_end_ns must be after window_start_ns")
        if limit is not None:
            _positive_int("limit", limit)
        self.deadline_ns = int(deadline_ns)
        self.split = split
        self.scale_factor = float(scale_factor)
        self.window_start_ns = int(window_start_ns)
        self.window_end_ns = None if window_end_ns is None else int(window_end_ns)
        self.limit = None if limit is None else int(limit)
        self.tenant_field = tenant_field
        self.session_field = session_field
        self.time_field = time_field
        self.event_field = event_field

    def _load_events(self) -> list[dict[str, Any]]:
        events = []
        with open(self.path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise WorkloadError(
                        f"{self.path}:{line_number} is not valid JSON: {error}"
                    ) from None
                if not isinstance(event, dict):
                    raise WorkloadError(f"{self.path}:{line_number} is not a JSON object")
                if self.split is not None and event.get("split") != self.split:
                    continue
                if self.time_field not in event:
                    raise WorkloadError(
                        f"{self.path}:{line_number} is missing {self.time_field!r}"
                    )
                _finite_nonneg_int(
                    f"{self.path}:{line_number} {self.time_field}", event[self.time_field]
                )
                events.append(event)
        return events

    def arrivals(self) -> list[Arrival]:
        events = self._load_events()
        if not events:
            raise WorkloadError(
                f"trace {self.path} has no events for split {self.split!r}"
            )
        t0_ns = min(int(event[self.time_field]) for event in events)
        arrivals: list[Arrival] = []
        for index, event in enumerate(events):
            replay_ns = int(round((int(event[self.time_field]) - t0_ns) / self.scale_factor))
            if replay_ns < self.window_start_ns:
                continue
            if self.window_end_ns is not None and replay_ns >= self.window_end_ns:
                continue
            event_id = event.get(self.event_field, index)
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                event_id = index
            arrivals.append(
                Arrival(
                    query_id=f"trace:{event_id}",
                    time_ns=replay_ns,
                    deadline_ns=replay_ns + self.deadline_ns,
                    tenant=str(event.get(self.tenant_field, "derived")),
                    session_id=(
                        str(event[self.session_field])
                        if event.get(self.session_field) is not None
                        else None
                    ),
                )
            )
        arrivals.sort(key=lambda arrival: (arrival.time_ns, arrival.query_id))
        if self.limit is not None:
            arrivals = arrivals[: self.limit]
        return arrivals


class RunWorkspace:
    """A sealed ``results/<run_id>/`` directory (collision and escape safe)."""

    def __init__(
        self,
        run_id: str,
        *,
        results_root: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise WorkloadError(f"run_id must be a non-empty string, got {run_id!r}")
        if (
            Path(run_id).is_absolute()
            or "/" in run_id
            or "\\" in run_id
            or run_id in (".", "..")
        ):
            raise WorkloadError(f"run_id must be a single path component, got {run_id!r}")
        self.run_id = run_id
        self.project_root = (
            Path(project_root) if project_root is not None else DEFAULT_PROJECT_ROOT
        )
        root = (
            Path(results_root)
            if results_root is not None
            else self.project_root / "results"
        )
        self.path = assert_within_project(root / run_id, self.project_root)
        if self.path.exists():
            raise RunCollisionError(f"run directory {self.path} already exists")

    def create(self) -> Path:
        if self.path.exists():
            raise RunCollisionError(f"run directory {self.path} already exists")
        try:
            self.path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            raise RunCollisionError(f"run directory {self.path} already exists") from None
        return self.path

    def _target(self, name: str) -> Path:
        return assert_within_project(self.path / name, self.project_root)

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        target = self._target(name)
        if target.exists():
            raise RunCollisionError(f"refusing to overwrite immutable artifact {target}")
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        with open(target, "x", encoding="utf-8") as handle:
            handle.write(text)
        return target

    def write_jsonl(self, name: str, rows: Iterable[Any]) -> Path:
        target = self._target(name)
        if target.exists():
            raise RunCollisionError(f"refusing to overwrite immutable artifact {target}")
        lines = []
        for row in rows:
            payload = row.to_dict() if hasattr(row, "to_dict") else row
            lines.append(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        with open(target, "x", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line + "\n")
        return target

    def write_config(self, config: Mapping[str, Any]) -> Path:
        return self.write_json("config.json", config)

    def write_provenance(self, provenance: Mapping[str, Any]) -> Path:
        return self.write_json("provenance.json", provenance)

    def write_requests(self, records: Iterable[RequestRecord]) -> Path:
        return self.write_jsonl("requests.jsonl", records)

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        return self.write_json("summary.json", summary)


def require_complete(outcome: WorkloadResult | Mapping[str, Any]) -> None:
    """Raise when an outcome cannot carry a success benchmark label."""
    summary = outcome.summary() if isinstance(outcome, WorkloadResult) else outcome
    if summary.get("status") != "complete":
        raise IncompleteRunError(
            f"run status is {summary.get('status')!r}; success labels are forbidden"
        )


__all__ = [
    "Arrival",
    "ClusterWork",
    "IncompleteRunError",
    "MetricsError",
    "PERCENTILES",
    "PoissonArrivalSource",
    "QueryPlan",
    "RequestRecord",
    "RunCollisionError",
    "RunResources",
    "RunWorkspace",
    "TileWork",
    "TraceArrivalSource",
    "WorkloadError",
    "WorkloadResult",
    "percentile",
    "percentiles",
    "recall_at_k",
    "require_complete",
    "run_workload",
]
