#!/usr/bin/env python3
"""OSD scheduling integration harness.

Wires the four scheduler pieces (RGW multi-factor routing, OSD state feedback,
per-OSD query ordering, per-OSD timing) into one closed-loop deterministic
event-driven simulation.  Runs over a synthetic query trace and emits the
standard experiment metrics (latency / wait percentiles, SLO violations,
throughput, Jain fairness, codebook reloads, DDR transfers, per-tenant
breakdown).

RGW coarseness vs OSD precision (§3.3)
--------------------------------------
The design doc §3.3 specifies a split: each OSD schedules its own queue
*precisely* from local state, but the RGW only sees each OSD's state via a
**100 ms heartbeat** (TTL 300 ms).  The RGW's view of the world is therefore
deliberately *coarse* — a snapshot that lags behind the live OSD state.

In the harness, ``heartbeat_ns > 0`` turns on **periodic** heartbeats: every
OSD emits one every ``heartbeat_ns`` nanoseconds *regardless of whether it
served a query*, so an idle OSD's state does not silently age into
``is_stale()``.  Between beats the RGW keeps seeing the previous snapshot, and
the OSD's own scheduling is unaffected (it always uses its exact local state).
When ``heartbeat_ns == 0`` (the default) the harness instead refreshes the
state manager after every serviced query, reproducing the legacy behaviour and
keeping all existing tests passing.  ``state_ttl_ns`` (default 300 ms per
§3.3) configures the RGW's ``is_stale`` check — when the last heartbeat is
older than ``state_ttl_ns`` the RGW treats the OSD as worst-case (DDR miss,
FPGA busy).

Strategy knob
-------------
``--strategy`` selects the per-OSD ordering policy:

- ``multifactor`` — :class:`SchedulingWindow` (aging + codebook + DDR + deadline
  + cost + soft fairness);
- ``fifo`` / ``spt`` / ``edf`` / ``drr`` — :class:`QueryQueue` ablation
  baselines.

The loop is fully deterministic: fixed seed, sorted event order, ties broken
by query_id, no wall clock, no unseeded randomness.  The ``formal`` mode calls
``require_model_gate`` and refuses with ``blocked`` when the gate has not
passed; ``fixture`` / ``dry-run`` use synthetic traces only.
"""

from __future__ import annotations

import os
import argparse
import json
import random
import sys

import numpy as np
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_DEFAULT))

from scripts.validate_model import (  # noqa: E402
    DEFAULT_GATE_PATH,
    ModelGateError,
    ModelGateNotPassedError,
    require_model_gate,
)
from strata_sim.contracts.schema import ClockKind, ContractError, ProfileRef  # noqa: E402
from strata_sim.des import fpga  # noqa: E402
from strata_sim.des.engine import DesEngine  # noqa: E402
from strata_sim.functional.ivfpq import IVFPQIndex  # noqa: E402
from strata_sim.metrics.requests import TraceArrivalSource, WorkloadError, percentile  # noqa: E402
from strata_sim.safety import UnsafePathError, assert_within_project  # noqa: E402
from strata_sim.sched.osd_state_manager import OSDStateManager  # noqa: E402
from strata_sim.sched.osd_timing import (  # noqa: E402
    OSDHardwareProfile,
    OSDTimingModel,
    QueryTiming,
    add_osd_resources,
)
from strata_sim.sched.query_queue import (  # noqa: E402
    OsdView,
    QueuedQuery,
    QueryQueue,
    QueueStrategy,
    ScoreWeights,
    SchedulingWindow,
)
from strata_sim.sched.rgw_scheduler import (  # noqa: E402
    ClusterAssignment,
    QueryRequest,
    RGWScheduler,
    RGWScoreWeights,
    ReplicaSet,
    SchedulingDecision,
)


# ---------------------------------------------------------------------------
# Constants / schema
# ---------------------------------------------------------------------------


SCHEMA = "strata.osd_sched_experiment.v1"
TASK = "task-osd-sched"
STATUS_BLOCKED = "blocked"
STATUS_COMPLETE = "complete"
STATUS_FIXTURE = "fixture"

STRATEGIES = ("fifo", "spt", "edf", "drr", "multifactor", "batch", "lstf")
EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_BLOCKED = 4


class OsdSchedError(ContractError):
    """The OSD scheduling harness contract was violated."""


# ---------------------------------------------------------------------------
# Trace types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticQuery:
    """One deterministic synthetic query arrival."""

    query_id: str
    tenant: str
    arrival_ns: int
    deadline_ns: int
    cluster_ids: tuple[int, ...]
    top_k: int
    codebook_id: str

    def __post_init__(self) -> None:
        if not self.query_id:
            raise OsdSchedError("query_id must be non-empty")
        if not self.tenant:
            raise OsdSchedError("tenant must be non-empty")
        if self.arrival_ns < 0:
            raise OsdSchedError(f"arrival_ns must be >= 0, got {self.arrival_ns}")
        if self.deadline_ns < self.arrival_ns:
            raise OsdSchedError("deadline_ns must be >= arrival_ns")
        if not self.cluster_ids:
            raise OsdSchedError("cluster_ids must be non-empty")
        if self.top_k < 1:
            raise OsdSchedError(f"top_k must be >= 1, got {self.top_k}")


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------


def ceph_placement(
    num_osds: int,
    cluster_ids: Sequence[int],
    replication: int = 3,
) -> dict[int, ReplicaSet]:
    """Default Ceph-like 3-replica placement: cluster c lives on
    OSDs ``c%N, (c+1)%N, ..., (c+replication-1)%N`` with primary ``c%N``.
    Exposed so tests can override.  ``cluster_ids`` enumerates every cluster
    that may be referenced by the trace; entries are pre-built so the dict
    is a plain mapping (the scheduler copies placement with ``dict()``).
    """
    if num_osds < 1:
        raise OsdSchedError(f"num_osds must be >= 1, got {num_osds}")
    if replication < 1 or replication > num_osds:
        raise OsdSchedError(
            f"replication must be in [1, num_osds={num_osds}], got {replication}")
    placement: dict[int, ReplicaSet] = {}
    for cluster_id in sorted(set(cluster_ids)):
        raw = [(cluster_id + k) % num_osds for k in range(replication)]
        osd_ids = tuple(sorted(set(raw)))
        primary = cluster_id % num_osds
        placement[cluster_id] = ReplicaSet(
            cluster_id=cluster_id, osd_ids=osd_ids, primary_osd_id=primary)
    return placement


# ---------------------------------------------------------------------------
# FPGA model builder (shared across all OSDs)
# ---------------------------------------------------------------------------


def build_fpga_model(project_root: Path) -> fpga.FpgaServiceModel:
    """Build one calibrated FpgaServiceModel from the real artifacts.

    Constructs the model with ``codebook_load_cycles=fpga.CODEBOOK_LOAD_CYCLES``
    so that ``codebook_resident=True`` subtracts the codebook stream and a
    BRAM-resident codebook is never charged twice.  Calibrates on the canonical
    train/holdout split from ``results/validation/cosim-holdout.json``.
    """
    profile_path = project_root / "results/validation/profiles/table-planb-axi512.json"
    cosim_path = project_root / "results/validation/cosim-holdout.json"
    if not profile_path.is_file():
        raise OsdSchedError(f"HLS profile not found: {profile_path}")
    if not cosim_path.is_file():
        raise OsdSchedError(f"co-sim holdout not found: {cosim_path}")
    from scripts.extract_hls import read_profile
    profile = read_profile(profile_path)
    ref = ProfileRef(
        profile_id=profile.profile_id,
        clock_kind=ClockKind.DES_LOGICAL,
        source=str(profile_path),
        evidence_id="osd-sched-harness",
    )
    base = fpga.FpgaServiceModel.from_profile(profile, ref)
    model = fpga.FpgaServiceModel(
        profile=base.profile,
        ref=base.ref,
        clock_ns=base.clock_ns,
        topk_init_cycles=base.topk_init_cycles,
        process_cycles_per_doc=base.process_cycles_per_doc,
        push_cycles_per_result=base.push_cycles_per_result,
        codebook_load_cycles=fpga.CODEBOOK_LOAD_CYCLES,
    )
    cosim_doc = json.loads(cosim_path.read_text(encoding="utf-8"))
    raw_points = cosim_doc.get("points") or []
    train = []
    holdout = []
    for raw in raw_points:
        point = fpga.CalibrationPoint(
            n_docs=int(raw["n_docs"]),
            top_k=int(raw["top_k"]),
            total_cycles=float(raw["total_cycles"]),
            origin=str(raw.get("origin", "cosim")),
        )
        if point.n_docs in (1, 17, 255):
            train.append(point)
        elif point.n_docs in (1023, 4097, 8192):
            holdout.append(point)
    report = model.calibrate(train, holdout)
    if not report.passed:
        raise OsdSchedError(
            f"FPGA model calibration did not pass the gate: status={report.status}")
    return model


# ---------------------------------------------------------------------------
# Synthetic trace
# ---------------------------------------------------------------------------


def generate_trace(
    *,
    seed: int,
    num_queries: int,
    num_tenants: int,
    num_clusters: int,
    span_ns: int,
    deadline_ns: int,
    skew: str,
    top_k: int,
    nprobe: int,
    codebook_per_tenant: bool,
    cluster_skew: str = "uniform",
) -> list[SyntheticQuery]:
    """Deterministic synthetic arrivals with tenant skew (uniform or zipf).

    ``cluster_skew`` controls the distribution over cluster ids for the
    ``nprobe`` draws per query:

    - ``uniform``: each cluster id is drawn uniformly from
      ``[0, num_clusters)`` (original behaviour).
    - ``zipf``: cluster ids are drawn from a Zipf(0.9) distribution over
      ``num_clusters`` so a few "hot" clusters dominate.  This makes
      DDR/BRAM reuse have signal — a scan of cache capacities can
      actually distinguish hits from misses because the working set is
      concentrated on the hot clusters.

    The two skews are independent: ``skew`` controls *tenant* frequency
    and ``cluster_skew`` controls *cluster* frequency.  Fully deterministic
    for the same seed.
    """
    if seed < 0:
        raise OsdSchedError(f"seed must be >= 0, got {seed}")
    if num_queries < 1:
        raise OsdSchedError(f"num_queries must be >= 1, got {num_queries}")
    if num_tenants < 1:
        raise OsdSchedError(f"num_tenants must be >= 1, got {num_tenants}")
    if num_clusters < 1:
        raise OsdSchedError(f"num_clusters must be >= 1, got {num_clusters}")
    if span_ns < 1:
        raise OsdSchedError(f"span_ns must be >= 1, got {span_ns}")
    if deadline_ns < 1:
        raise OsdSchedError(f"deadline_ns must be >= 1, got {deadline_ns}")
    if top_k < 1:
        raise OsdSchedError(f"top_k must be >= 1, got {top_k}")
    if nprobe < 1:
        raise OsdSchedError(f"nprobe must be >= 1, got {nprobe}")
    if skew not in ("uniform", "zipf"):
        raise OsdSchedError(f"unknown skew {skew!r}; expected uniform or zipf")
    if cluster_skew not in ("uniform", "zipf"):
        raise OsdSchedError(
            f"unknown cluster_skew {cluster_skew!r}; expected uniform or zipf")

    rng = random.Random(seed)
    tenants = [f"tenant-{i:02d}" for i in range(num_tenants)]
    if skew == "uniform":
        weights = [1.0] * num_tenants
    else:
        weights = [1.0 / ((i + 1) ** 0.9) for i in range(num_tenants)]
    total = sum(weights)
    probs = [w / total for w in weights]

    # Pre-compute Zipf(0.9) cluster probabilities once, independent of the
    # per-query RNG draws, so the cluster skew is deterministic given
    # num_clusters and the seed.
    if cluster_skew == "zipf":
        cluster_weights = [1.0 / ((i + 1) ** 0.9) for i in range(num_clusters)]
        cluster_total = sum(cluster_weights)
        cluster_probs = [w / cluster_total for w in cluster_weights]
        cluster_cumulative: list[float] = []
        running = 0.0
        for cp in cluster_probs:
            running += cp
            cluster_cumulative.append(running)
    else:
        cluster_cumulative = None

    queries: list[SyntheticQuery] = []
    for q_index in range(num_queries):
        arrival_ns = rng.randrange(span_ns)
        r = rng.random()
        cumulative = 0.0
        tenant_idx = num_tenants - 1
        for t_i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                tenant_idx = t_i
                break
        tenant = tenants[tenant_idx]
        # Draw nprobe cluster ids.  With Zipf cluster_skew the draws are
        # skewed toward hot clusters (low ids), so DDR/BRAM reuse has
        # signal; with uniform they are spread evenly.
        if cluster_cumulative is not None:
            drawn: set[int] = set()
            for _ in range(nprobe):
                cr = rng.random()
                cidx = num_clusters - 1
                for c_i, cc in enumerate(cluster_cumulative):
                    if cr <= cc:
                        cidx = c_i
                        break
                drawn.add(cidx)
            cluster_ids = tuple(sorted(drawn))
        else:
            cluster_ids = tuple(sorted({
                rng.randrange(num_clusters) for _ in range(nprobe)}))
        if not cluster_ids:
            cluster_ids = (rng.randrange(num_clusters),)
        codebook_id = f"cb-{tenant}" if codebook_per_tenant else "cb-shared"
        queries.append(SyntheticQuery(
            query_id=f"q{q_index:05d}",
            tenant=tenant,
            arrival_ns=arrival_ns,
            deadline_ns=arrival_ns + _slo_ns(arrival.tenant, deadline_ns, deadline_mode),
            cluster_ids=cluster_ids,
            top_k=top_k,
            codebook_id=codebook_id,
        ))
    queries.sort(key=lambda q: (q.arrival_ns, q.query_id))
    return queries


def generate_trace_from_real(
    trace_path: str | Path,
    *,
    deadline_ns: int,
    index_manifest: str | Path,
    project_root: Path,
    nprobe: int,
    top_k: int,
    split: str | None = None,
    limit: int | None = None,
    scale_factor: float = 1.0,
    codebook_groups: int = 1,
    codebook_mode: str = "groups",
    hybrid_head_min: int = 5,
    deadline_mode: str = "uniform",
) -> list[SyntheticQuery]:
    """Build a :class:`SyntheticQuery` list from a real RAGPulse trace + IVF index.

    **Synthetic pairing disclaimer.**  The RAGPulse trace contains arrival times,
    tenants and session ids but *no* query embeddings.  Each arrival is therefore
    paired with an index query vector by round-robin (arrival *i* uses
    ``index.queries[i % index.queries.shape[0]]``).  The cluster ids come from a
    genuine ``IVFPQIndex.coarse_probes`` call on that synthetic query, but the
    pairing itself is *not* the user's real query — it is a deterministic
    placeholder that lets the harness exercise the full IVF + scheduling path
    with real arrival patterns and a real coarse index.

    Parameters
    ----------
    trace_path
        Path to the ``events.jsonl`` RAGPulse trace file.
    deadline_ns
        Per-query deadline budget in nanoseconds (added to each arrival time).
    index_manifest
        Path to the IVFPQ index manifest JSON.
    project_root
        Project root used to resolve relative manifest paths.
    nprobe
        Number of coarse clusters to probe per query.
    top_k
        Client-requested top-k for the query.
    split
        Optional trace split filter (e.g. ``"test"``).
    limit
        Optional cap on the number of arrivals to consume.
    scale_factor
        Time-scaling factor for the trace replay (> 1 = slower replay).
    codebook_groups
        Number of codebook groups for tenant→codebook mapping.  When 1 (default),
        all tenants share ``"cb-shared"``.  When > 1, the tenant is hashed to a
        group id via ``zlib.crc32(tenant.encode()) % codebook_groups`` producing
        ``"cb-<group>"`` — a *stable* hash, not Python's salted ``hash()``.

    Returns
    -------
    list[SyntheticQuery]
        Sorted by ``(arrival_ns, query_id)``.
    """
    if nprobe < 1:
        raise OsdSchedError(f"nprobe must be >= 1, got {nprobe}")
    if codebook_groups < 1:
        raise OsdSchedError(
            f"codebook_groups must be >= 1, got {codebook_groups}")
    if top_k < 1:
        raise OsdSchedError(f"top_k must be >= 1, got {top_k}")
    if deadline_ns < 1:
        raise OsdSchedError(f"deadline_ns must be >= 1, got {deadline_ns}")

    try:
        arrivals = TraceArrivalSource(
            trace_path,
            deadline_ns=deadline_ns,
            split=split,
            limit=limit,
            scale_factor=scale_factor,
        ).arrivals()
    except WorkloadError as error:
        raise OsdSchedError(
            f"TraceArrivalSource failed: {error}") from error
    if not arrivals:
        raise OsdSchedError(
            "TraceArrivalSource produced no arrivals; check trace_path / split / limit")

    index = IVFPQIndex.from_manifest(
        index_manifest, project_root=project_root)
    if index.queries is None or index.queries.shape[0] == 0:
        raise OsdSchedError(
            "IVFPQ index has no query vectors; the manifest must include a queries file")

    def _slo_ns(tenant: str, base_ns: int, mode: str) -> int:
        """Per-tenant SLO.  hetero gives three tenant classes with
        different latency budgets so a deadline-aware scheduler has room to
        discriminate (50%% tight 0.5x, 30%% normal 1.0x, 20%% loose 3.0x)."""
        if mode != "hetero":
            return base_ns
        r = zlib.crc32(tenant.encode("utf-8")) % 100
        if r < 50:
            return max(1, base_ns // 2)
        if r < 80:
            return base_ns
        return base_ns * 3

    num_index_queries = index.queries.shape[0]
    queries: list[SyntheticQuery] = []
    tenant_counts: dict[str, int] = {}
    if codebook_mode == "hybrid":
        for _a in arrivals:
            tenant_counts[_a.tenant] = tenant_counts.get(_a.tenant, 0) + 1
    for i, arrival in enumerate(arrivals):
        q = index.queries[i % num_index_queries]
        clusters = tuple(sorted(int(c) for c in index.coarse_probes(q, nprobe)))
        if not clusters:
            raise OsdSchedError(
                f"coarse_probes returned empty for arrival {arrival.query_id}; "
                f"nprobe={nprobe} may exceed nlist={index.nlist}")
        if codebook_mode == "hybrid":
            # Head tenants (>= hybrid_head_min queries) get a private codebook;
            # the long tail shares one common codebook.
            if tenant_counts.get(arrival.tenant, 0) >= hybrid_head_min:
                codebook_id = f"cb-{arrival.tenant}"
            else:
                codebook_id = "cb-shared"
        elif codebook_groups <= 1:
            codebook_id = "cb-shared"
        else:
            group = zlib.crc32(arrival.tenant.encode("utf-8")) % codebook_groups
            codebook_id = f"cb-{group}"
        queries.append(SyntheticQuery(
            query_id=arrival.query_id,
            tenant=arrival.tenant,
            arrival_ns=arrival.time_ns,
            deadline_ns=arrival.deadline_ns,
            cluster_ids=clusters,
            top_k=top_k,
            codebook_id=codebook_id,
        ))
    queries.sort(key=lambda q: (q.arrival_ns, q.query_id))
    return queries


# ---------------------------------------------------------------------------
# Per-OSD scheduler facade
# ---------------------------------------------------------------------------


class OsdScheduler:
    """Facade hiding the multifactor / ablation strategy split."""

    def __init__(
        self,
        strategy: str,
        *,
        window_width: int = 128,
        weights: ScoreWeights | None = None,
        cluster_sizes: Mapping[int, int] | None = None,
        adaptive: bool = False,
    ) -> None:
        if strategy not in STRATEGIES:
            raise OsdSchedError(
                f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
        self.strategy = strategy
        self.window_width = window_width
        # ``weights`` are the OSD scheduling-window factors from the design
        # doc §2.3 (aging/codebook/ddr/deadline/cost).  They only apply to the
        # multifactor strategy; the ablation queues have their own semantics.
        self.weights = weights if weights is not None else ScoreWeights()
        self._window: SchedulingWindow | None = None
        self._queue: QueryQueue | None = None
        if strategy == "multifactor":
            self._window = SchedulingWindow(
                width=window_width, weights=self.weights,
                cluster_sizes=cluster_sizes, adaptive=adaptive)
        else:
            self._queue = QueryQueue(strategy=QueueStrategy(strategy))

    def push(self, query: QueuedQuery) -> None:
        if self._window is not None:
            self._window.push(query)
        else:
            assert self._queue is not None
            self._queue.push(query)

    def pop(self, view: OsdView) -> QueuedQuery | None:
        if self._window is not None:
            return self._window.select_next(view)
        assert self._queue is not None
        if self._queue.pending == 0:
            return None
        return self._queue.pop(view.now_ns)

    def window_queries(self) -> list[QueuedQuery]:
        """Pending window snapshot (delegates to SchedulingWindow)."""
        if self._window is not None:
            return self._window.window_queries()
        return []

    def peek(self, view: OsdView) -> QueuedQuery | None:
        """Non-mutating: the query that ``pop`` would return next."""
        if self._window is not None:
            return self._window.peek_next(view)
        assert self._queue is not None
        return self._queue.peek(view.now_ns)

    @property
    def pending(self) -> int:
        if self._window is not None:
            return self._window.pending
        assert self._queue is not None
        return self._queue.pending


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


@dataclass
class CompletedQuery:
    """One end-to-end finished query with its timing records."""

    query: SyntheticQuery
    arrival_ns: int
    start_ns: int
    finish_ns: int
    timings: tuple[QueryTiming, ...]
    assignments: tuple[ClusterAssignment, ...]

    @property
    def latency_ns(self) -> int:
        return self.finish_ns - self.arrival_ns

    @property
    def wait_ns(self) -> int:
        return self.start_ns - self.arrival_ns


def run_simulation(
    queries: Sequence[SyntheticQuery],
    *,
    num_osds: int,
    strategy: str,
    window_width: int,
    fpga_model: fpga.FpgaServiceModel,
    placement: Mapping[int, ReplicaSet] | None = None,
    hardware: OSDHardwareProfile | None = None,
    rgw_weights: RGWScoreWeights | None = None,
    osd_weights: ScoreWeights | None = None,
    heartbeat_ns: int = 0,
    state_ttl_ns: int = 300_000,
    cluster_sizes: Mapping[int, int] | None = None,
    cluster_pool: Mapping[int, int] | None = None,
    adaptive: bool = False,
    rerank_per_cluster: int = 0,
) -> tuple[list[CompletedQuery], dict[str, Any]]:
    """Deterministic event-driven closed loop.

    Modelling decisions:
    - One DesEngine shared across all OSDs; each OSD registers its own
      resources via ``add_osd_resources`` so the resource calendars do not
      collide (different names per OSD).
    - One calibrated ``FpgaServiceModel`` is shared; the OSD timing model
      uses ``codebook_resident=True`` so a BRAM-resident codebook is never
      charged twice.
    - Each OSD owns its own :class:`OsdScheduler` (multifactor or ablation)
      plus an :class:`OSDTimingModel`.
    - An OSD's "server" is modelled as a single serial processor: the next
      query cannot start until the previous one's ``end_ns``.  Pipelining
      across queries on the same OSD is NOT supported (explicit serialisation
      via ``osd_ready_ns``), matching the per-OSD pipeline being a single
      in-order processor.  Cross-OSD pipelining is implicit because each OSD
      has distinct resources.
    - ``engine.advance_to(arrival_ns)`` before routing so the RGW scheduler
      sees fresh state.  After routing, each (cluster -> chosen OSD) sub-query
      is pushed into that OSD's scheduler.
    - After each serviced query, the OSD's state manager entry is refreshed
      (queue_depth, fpga_busy_until_ns, DDR cache, heartbeat timestamp) so the
      RGW's next routing sees current data — but only when the heartbeat
      throttle (``heartbeat_ns``, §3.3) allows it.  With ``heartbeat_ns > 0``
      the refresh fires at most once per ``heartbeat_ns`` per OSD; between
      beats the RGW keeps seeing the previous snapshot, modelling the coarse
      100 ms heartbeat / 300 ms TTL split of the design doc.  With
      ``heartbeat_ns == 0`` (default) every query refreshes state, exactly as
      before.
    - Events are processed in strict time order; ties broken by (osd_id, query_id).

    ``hardware``, ``rgw_weights``, ``heartbeat_ns`` and ``state_ttl_ns`` are
    optional overrides for the OSD hardware profile, the RGW score weights,
    the heartbeat throttle period and the state TTL respectively.  When
    omitted, defaults are ``OSDHardwareProfile()``, ``RGWScoreWeights()``,
    ``heartbeat_ns=0`` (no throttle — every query updates state, reproducing
    the legacy behaviour) and ``state_ttl_ns=300_000`` (300 ms per §3.3).
    """
    if num_osds < 1:
        raise OsdSchedError(f"num_osds must be >= 1, got {num_osds}")
    if not queries:
        raise OsdSchedError("queries must be non-empty")
    if heartbeat_ns < 0:
        raise OsdSchedError(f"heartbeat_ns must be >= 0, got {heartbeat_ns}")
    if state_ttl_ns < 0:
        raise OsdSchedError(f"state_ttl_ns must be >= 0, got {state_ttl_ns}")

    import heapq

    _stats_path = os.environ.get("SCHED_STATS_PATH")
    _stats_rows = [] if _stats_path else None
    _stats_arr: dict[int, int] = {}

    engine = DesEngine(seed=0)
    # The PL DDR4 bus is one resource: on the board it reads at 5.5 GB/s, so
    # write accounting must use the same bus rate instead of the legacy
    # 19.2 GB/s calendar.
    _hw_bw = hardware if hardware is not None else OSDHardwareProfile()
    _ddr_bw = (_hw_bw.board_ddr_read_bytes_per_ns if _hw_bw.board_calibrated
               else _hw_bw.ddr_bytes_per_ns)
    for osd_id in range(num_osds):
        add_osd_resources(engine, osd_id, ddr_bytes_per_ns=int(_ddr_bw))

    state_manager = OSDStateManager(now_ns=0)
    for osd_id in range(num_osds):
        state_manager.get_osd_state(osd_id)

    # Collect all cluster ids referenced by the trace so the placement dict
    # covers every RGW lookup (the scheduler copies placement with dict()).
    all_cluster_ids: set[int] = set()
    for sq in queries:
        all_cluster_ids.update(sq.cluster_ids)
    resolved_placement = (
        placement if placement is not None
        else ceph_placement(num_osds, sorted(all_cluster_ids))
    )
    rgw = RGWScheduler(
        state_manager=state_manager,
        placement=resolved_placement,
        fpga_model=fpga_model,
        weights=rgw_weights if rgw_weights is not None else RGWScoreWeights(),
        heartbeat_ns=heartbeat_ns,
        state_ttl_ns=state_ttl_ns,
    )

    timing_models: dict[int, OSDTimingModel] = {}
    schedulers: dict[int, OsdScheduler] = {}
    osd_ready_ns: dict[int, int] = {i: 0 for i in range(num_osds)}
    # §3.3 periodic heartbeats: with heartbeat_ns > 0 every OSD emits a
    # heartbeat every heartbeat_ns regardless of whether it served a query, so
    # an *idle* OSD's state does not age into is_stale().  With heartbeat_ns == 0
    # we keep the legacy "refresh on every serviced query" behaviour instead.
    # The heap holds (next_beat_ns, osd_id).
    heartbeat_heap: list[tuple[int, int]] = []
    if heartbeat_ns > 0:
        for osd_id in range(num_osds):
            heartbeat_heap.append((0, osd_id))   # first beat at t=0
        heapq.heapify(heartbeat_heap)
    client_top_k: dict[str, int] = {}  # query_id -> client top_k
    client_slo: dict[str, int] = {}  # query_id -> SLO budget ns
    sub_query_to_client: dict[str, str] = {}  # sub_query_id -> client query_id
    # Accumulate per-query sub-query timings until all clusters finish.
    pending_finish: dict[str, dict[str, Any]] = {}

    hw_profile = hardware if hardware is not None else OSDHardwareProfile()
    for osd_id in range(num_osds):
        timing_models[osd_id] = OSDTimingModel(
            osd_id=osd_id, engine=engine, fpga=fpga_model, hardware=hw_profile,
            cluster_sizes=cluster_sizes, cluster_pool=cluster_pool)
        schedulers[osd_id] = OsdScheduler(
            strategy, window_width=window_width, weights=osd_weights,
            cluster_sizes=cluster_sizes, adaptive=adaptive)

    completed: list[CompletedQuery] = []

    def _view_for(osd_id: int, now_ns: int) -> OsdView:
        tm = timing_models[osd_id]
        return OsdView(
            now_ns=now_ns,
            bram_codebooks=tm.bram_resident(),
            ddr_cached_clusters=tm.ddr_resident(),
            max_compute_ns=max(1, int(fpga_model.predict_ns(
                n_docs=64 * 16, top_k=10, codebook_resident=True))),
        )

    def _pump_osd(osd_id: int, now_ns: int) -> bool:
        """Run one query on the OSD if it is free and has work.  Returns True
        if a query was serviced, False otherwise.
        """
        rr = max(0, int(rerank_per_cluster))
        if osd_ready_ns[osd_id] > now_ns:
            return False
        scheduler = schedulers[osd_id]
        if scheduler.pending == 0:
            return False
        _pending_before = scheduler.pending
        view = _view_for(osd_id, now_ns)
        qq = scheduler.pop(view)
        if qq is None:
            return False
        # Compute-shadow staging (window-wide): rank every cluster referenced
        # by the scheduler's window, not just the next query, and let the
        # timing model admit candidates by value density under the SSD/DDR
        # guard.  Weight = deadline urgency (0.1 .. 1.0).
        _pf_cands: list[tuple[int, float]] = []
        _seen_c: set[int] = set()
        _now_for_w = max(osd_ready_ns[osd_id], now_ns)
        try:
            _win_qs = scheduler.window_queries()
        except AttributeError:
            _win_qs = []
        if not _win_qs:
            # FIFO and the other queue strategies have no SchedulingWindow;
            # fall back to the single next query so staging still works.
            _nxt = scheduler.peek(view)
            if _nxt is not None and _nxt.query_id != qq.query_id:
                _win_qs = [_nxt]
        for _q in _win_qs:
            if _q.query_id == qq.query_id:
                continue
            _w = 1.0
            if _q.deadline_ns:
                _w = 1.0 - float(_now_for_w - _q.arrival_ns) / float(_q.deadline_ns)
                _w = max(0.1, min(1.0, _w))
            _cids = (list(_q.cluster_ids) if _q.cluster_ids
                     else [_q.cluster_id])
            for _c in _cids:
                if _c in _seen_c:
                    continue
                _seen_c.add(_c)
                _pf_cands.append((int(_c), float(_w)))
        # Start at max(osd_ready, now) — never before the OSD is free.
        start_ns = max(osd_ready_ns[osd_id], now_ns)
        cluster_ids_list = list(qq.cluster_ids) if qq.cluster_ids else [qq.cluster_id]
        # Use the client's top_k for FPGA/rerank stages.
        client_qid = sub_query_to_client.get(qq.query_id, qq.query_id)
        top_k = client_top_k.get(client_qid, max(1, qq.nprobe))
        timing = timing_models[osd_id].simulate_query(
            query_id=qq.query_id,
            cluster_ids=cluster_ids_list,
            top_k=top_k,
            codebook_id=qq.codebook_id,
            tenant=qq.tenant,
            start_ns=start_ns,
            slo_ns=client_slo.get(client_qid),
            prefetch_candidates=_pf_cands,
            rerank_per_cluster=rr,
        )
        new_ready = timing.end_ns
        osd_ready_ns[osd_id] = new_ready
        if _stats_rows is not None:
            _wlen = _blen = _dten = _mten = -1
            _win = getattr(scheduler, "_window", None)
            if _win is not None:
                _slots = _win._window
                _wlen = len(_slots)
                _blen = len(_win._backlog)
                _tc: dict[str, int] = {}
                for _q in _slots:
                    _tc[_q.tenant] = _tc.get(_q.tenant, 0) + 1
                _dten = len(_tc)
                _mten = max(_tc.values()) if _tc else 0
            _qc: dict[str, int] = {}
            if _win is not None:
                for _q in _win._window:
                    _cid = _q.query_id.split(":")[0]
                    _qc[_cid] = _qc.get(_cid, 0) + 1
            _dq = len(_qc)
            _mq = max(_qc.values()) if _qc else 0
            _depth = -1
            if _win is not None:
                _depth = sum(1 for _q in _win._window
                             if _q.arrival_ns < qq.arrival_ns)
            _se = timing.stage_end_ns
            _map_end = int(_se.get("map", start_ns))
            _ssd_end = int(_se.get("ssd", _map_end))
            _cb_end = int(_se.get("codebook", _ssd_end))
            _fpga_end = int(_se.get("fpga", _cb_end))
            _ssd_ns = max(0, _ssd_end - _map_end)
            _cbns = max(0, _cb_end - _ssd_end)
            _fpga_ns = max(0, _fpga_end - _cb_end)
            _cbload = 1 if timing.codebook_loaded else 0
            _stats_rows.append((start_ns, osd_id, _pending_before,
                                timing.end_ns - start_ns, _wlen, _blen,
                                _dten, _mten, _stats_arr.get(osd_id, 0),
                                _dq, _mq, _depth, _ssd_ns, _cbns, _fpga_ns,
                                _cbload, int(timing.ddr_transfers),
                                int(timing.raw_transfers)))

        # Legacy (heartbeat_ns == 0): refresh the RGW-visible state after every
        # serviced query.  With periodic heartbeats (heartbeat_ns > 0) the
        # heartbeat event loop owns the refresh, so we must NOT touch the state
        # manager here — the RGW only learns an OSD's state on the next beat
        # (RGW_OSD_调度策略定稿.md §3.3).
        if heartbeat_ns == 0:
            state_manager.advance_time(now_ns)
            state_manager.mark_heartbeat(
                osd_id,
                queue_depth=scheduler.pending,
                fpga_busy_until_ns=int(timing.stage_end_ns.get("fpga", new_ready)),
                ddr_cached_clusters=set(timing_models[osd_id].ddr_resident()),
            )
            state_manager.update(
                osd_id,
                queue_depth=scheduler.pending,
                fpga_busy_until_ns=int(timing.stage_end_ns.get("fpga", new_ready)),
            )

        # Accumulate the sub-query result under the client query_id.
        bucket = pending_finish.setdefault(client_qid, {
            "timings": [], "query": None, "num_subqueries": 0,
        })
        bucket["timings"].append(timing)
        return True

    def _emit_heartbeat(osd_id: int, now_ns: int) -> None:
        """Send one periodic heartbeat for ``osd_id`` (RGW_OSD_调度策略定稿.md §3.3).

        The OSD reports its *current local* state (which it knows exactly, with
        no delay) to the RGW's state manager; the RGW then keeps seeing that
        snapshot until the next beat.  Emitting on a fixed period regardless of
        query activity is what stops an idle OSD's state from ageing into
        ``is_stale()``.
        """
        state_manager.advance_time(now_ns)
        state_manager.mark_heartbeat(
            osd_id,
            queue_depth=schedulers[osd_id].pending,
            fpga_busy_until_ns=timing_models[osd_id].feedback_state().fpga_busy_until_ns,
            ddr_cached_clusters=set(timing_models[osd_id].ddr_resident()),
        )

    # Main event loop: arrivals + OSD pump events + periodic heartbeats, in
    # time order.  Termination depends only on arrivals/pumps, so heartbeats
    # cannot keep an otherwise-finished simulation alive.
    arrival_index = 0
    sorted_queries = sorted(queries, key=lambda q: (q.arrival_ns, q.query_id))
    ready_heap: list[tuple[int, int]] = []  # (ready_ns, osd_id)

    while arrival_index < len(sorted_queries) or ready_heap:
        next_arrival = (
            sorted_queries[arrival_index].arrival_ns
            if arrival_index < len(sorted_queries) else None
        )
        next_ready = ready_heap[0][0] if ready_heap else None
        next_beat = heartbeat_heap[0][0] if heartbeat_heap else None

        candidates = [t for t in (next_arrival, next_ready, next_beat) if t is not None]
        now_ns = min(candidates)

        engine.advance_to(now_ns)
        state_manager.advance_time(now_ns)

        # Fire every due heartbeat at its exact time, before routing arrivals
        # at the same instant, so the RGW's view is the freshest beat.
        while heartbeat_heap and heartbeat_heap[0][0] <= now_ns:
            _, hb_osd = heapq.heappop(heartbeat_heap)
            _emit_heartbeat(hb_osd, now_ns)
            heapq.heappush(heartbeat_heap, (now_ns + heartbeat_ns, hb_osd))

        # Process every arrival at exactly now_ns before pumping OSDs.
        while (arrival_index < len(sorted_queries)
               and sorted_queries[arrival_index].arrival_ns == now_ns):
            sq = sorted_queries[arrival_index]
            arrival_index += 1
            client_top_k[sq.query_id] = sq.top_k
            client_slo[sq.query_id] = sq.deadline_ns - sq.arrival_ns
            request = QueryRequest(
                query_id=sq.query_id,
                query_vector=[0.0],  # unused by the scheduler
                cluster_ids=list(sq.cluster_ids),
                nprobe=len(sq.cluster_ids),
                top_k=sq.top_k,
            )
            decision: SchedulingDecision = rgw.schedule(request)
            bucket = pending_finish.setdefault(sq.query_id, {
                "timings": [], "query": sq, "num_subqueries": len(decision.assignments),
            })
            bucket["query"] = sq
            bucket["num_subqueries"] = len(decision.assignments)
            for assignment in decision.assignments:
                osd_id = assignment.osd_id
                sub_qid = f"{sq.query_id}:c{assignment.cluster_id}"
                sub_query_to_client[sub_qid] = sq.query_id
                qq = QueuedQuery(
                    query_id=sub_qid,
                    tenant=sq.tenant,
                    cluster_id=assignment.cluster_id,
                    nprobe=sq.top_k,
                    arrival_ns=now_ns,
                    deadline_ns=sq.deadline_ns,
                    estimated_compute_ns=max(
                        1, int(fpga_model.predict_ns(
                            n_docs=(int(cluster_sizes.get(assignment.cluster_id, 64))
                                    if cluster_sizes else 64),
                            top_k=sq.top_k, codebook_resident=True))),
                    codebook_id=sq.codebook_id,
                    cluster_ids=(assignment.cluster_id,),
                )
                schedulers[osd_id].push(qq)
                if _stats_rows is not None:
                    _stats_arr[osd_id] = _stats_arr.get(osd_id, 0) + 1
                # If the OSD is free now, pump immediately and schedule next.
                if osd_ready_ns[osd_id] <= now_ns:
                    if _pump_osd(osd_id, now_ns):
                        heapq.heappush(ready_heap, (osd_ready_ns[osd_id], osd_id))

        # Pump every OSD whose ready time has been reached.
        while ready_heap and ready_heap[0][0] <= now_ns:
            _, osd_id = heapq.heappop(ready_heap)
            if osd_ready_ns[osd_id] > now_ns:
                continue  # stale entry
            if _pump_osd(osd_id, now_ns):
                heapq.heappush(ready_heap, (osd_ready_ns[osd_id], osd_id))

        # Collect any queries that have finished all sub-timings.
        _collect_completed(pending_finish, completed)

    # Drain any stragglers left in the pump heap.
    while ready_heap:
        now_ns, osd_id = heapq.heappop(ready_heap)
        if osd_ready_ns[osd_id] > now_ns:
            continue  # stale
        engine.advance_to(now_ns)
        state_manager.advance_time(now_ns)
        if _pump_osd(osd_id, now_ns):
            heapq.heappush(ready_heap, (osd_ready_ns[osd_id], osd_id))
        _collect_completed(pending_finish, completed)

    _collect_completed(pending_finish, completed, force=True)
    completed.sort(key=lambda c: (c.finish_ns, c.query.query_id))

    info = {
        "num_osds": num_osds,
        "strategy": strategy,
        "window_width": window_width,
        "heartbeat_ns": heartbeat_ns,
        "state_ttl_ns": state_ttl_ns,
    }
    if _stats_rows is not None:
        with open(_stats_path, "w") as _f:
            _f.write("t_ns,osd,pending,svc_ns,wlen,blen,dten,mten,narr,dq,mq,depth,ssd_ns,cb_ns,fpga_ns,cbload,ddrtr,rawtr\n")
            for _r in _stats_rows:
                _f.write(",".join(map(str, _r)) + "\n")
        info["stats_rows"] = len(_stats_rows)
    info["prefetches"] = sum(
        tm.prefetch_transfers for tm in timing_models.values())
    info["prefetch_skipped"] = sum(
        tm.prefetch_skipped for tm in timing_models.values())
    _dumps = []
    for tm in timing_models.values():
        _dumps.extend(getattr(tm, "prefetch_skip_dumps", []))
    info["prefetch_skip_dumps"] = _dumps[:6]
    return completed, info


def _collect_completed(
    pending_finish: dict[str, dict[str, Any]],
    completed: list[CompletedQuery],
    *,
    force: bool = False,
) -> None:
    """Move fully-finished queries from pending_finish into completed."""
    finished_keys: list[str] = []
    for key, bucket in pending_finish.items():
        sq = bucket.get("query")
        if sq is None:
            continue
        expected = bucket["num_subqueries"]
        timings = bucket["timings"]
        if len(timings) < expected and not force:
            continue
        if not timings:
            completed.append(CompletedQuery(
                query=sq,
                arrival_ns=sq.arrival_ns,
                start_ns=sq.arrival_ns,
                finish_ns=sq.arrival_ns,
                timings=(),
                assignments=(),
            ))
            finished_keys.append(key)
            continue
        if len(timings) < expected and force:
            pass  # forced: use whatever we have
        elif len(timings) < expected:
            continue
        finish_ns = max(t.end_ns for t in timings)
        start_ns = min(t.start_ns for t in timings)
        completed.append(CompletedQuery(
            query=sq,
            arrival_ns=sq.arrival_ns,
            start_ns=start_ns,
            finish_ns=finish_ns,
            timings=tuple(timings),
            assignments=(),
        ))
        finished_keys.append(key)
    for key in finished_keys:
        pending_finish.pop(key, None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    completed: Sequence[CompletedQuery],
    *,
    strategy: str,
    window_width: int,
    num_osds: int,
    samples: int,
    claims: Mapping[str, Any],
    queries: Sequence[SyntheticQuery] | None = None,
) -> dict[str, Any]:
    """Aggregate the standard experiment metrics.

    ``queries`` (the offered trace) enables the demand-normalised fairness
    index; without it only the raw service-share Jain is reported.
    """
    if not completed:
        raise OsdSchedError("no completed queries to aggregate")

    latencies = [c.latency_ns for c in completed]
    waits = [c.wait_ns for c in completed]
    finishes = [c.finish_ns for c in completed]
    arrivals = [c.arrival_ns for c in completed]
    span_ns = max(finishes) - min(arrivals) if len(completed) >= 2 else 1
    span_ns = max(span_ns, 1)

    violations = sum(
        1 for c in completed if c.finish_ns > c.query.deadline_ns)
    throughput = len(completed) * 1e9 / span_ns

    # Jain fairness over per-tenant service share.
    per_tenant_counts: dict[str, int] = {}
    for c in completed:
        per_tenant_counts[c.query.tenant] = per_tenant_counts.get(c.query.tenant, 0) + 1
    shares = [count / len(completed) for count in per_tenant_counts.values()]
    share_sum = sum(shares)
    share_squares = sum(s * s for s in shares)
    n_tenants = len(shares)
    jain = (share_sum * share_sum) / (n_tenants * share_squares) if share_squares and n_tenants else 1.0

    # Demand-normalised fairness.  ``jain`` above is over the *absolute* service
    # share, which penalises a scheduler that correctly serves a skewed arrival
    # distribution (a tenant that sends 29x more queries is *supposed* to get
    # 29x more service).  This variant is Jain over served_share / demand_share,
    # so it is 1.0 for perfect proportional service regardless of skew, and
    # below 1.0 only when some tenant is actually under- or over-served.
    jain_demand_normalized: float | None = None
    if queries is not None and len(queries):
        demand_counts: dict[str, int] = {}
        for q in queries:
            demand_counts[q.tenant] = demand_counts.get(q.tenant, 0) + 1
        total_demand = len(queries)
        total_served = len(completed)
        ratios = []
        for tenant, demand in demand_counts.items():
            demand_share = demand / total_demand
            served_share = per_tenant_counts.get(tenant, 0) / total_served
            ratios.append(served_share / demand_share if demand_share else 0.0)
        if ratios:
            r_sum = sum(ratios)
            r_squares = sum(r * r for r in ratios)
            jain_demand_normalized = (
                (r_sum * r_sum) / (len(ratios) * r_squares) if r_squares else 1.0)

    codebook_reloads = sum(1 for c in completed for t in c.timings if t.codebook_loaded)
    ddr_transfers = sum(t.ddr_transfers for c in completed for t in c.timings)

    per_tenant: dict[str, dict[str, Any]] = {}
    for tenant in sorted(per_tenant_counts):
        tenant_completed = [c for c in completed if c.query.tenant == tenant]
        t_waits = [c.wait_ns for c in tenant_completed]
        t_violations = sum(
            1 for c in tenant_completed if c.finish_ns > c.query.deadline_ns)
        per_tenant[tenant] = {
            "samples": len(tenant_completed),
            "share": len(tenant_completed) / len(completed),
            "wait_p50_ns": percentile(t_waits, 50),
            "wait_p95_ns": percentile(t_waits, 95),
            "wait_p99_ns": percentile(t_waits, 99),
            "slo_violations": t_violations,
            "slo_violation_rate": t_violations / len(tenant_completed) if tenant_completed else None,
        }

    return {
        "strategy": strategy,
        "window_width": window_width,
        "num_osds": num_osds,
        "samples": samples,
        "span_ns": span_ns,
        "throughput_qps": throughput,
        "latency_p50_ns": percentile(latencies, 50),
        "latency_p95_ns": percentile(latencies, 95),
        "latency_p99_ns": percentile(latencies, 99),
        "latency_max_ns": max(latencies),
        "wait_p50_ns": percentile(waits, 50),
        "wait_p95_ns": percentile(waits, 95),
        "wait_p99_ns": percentile(waits, 99),
        "slo_violations": violations,
        "slo_violation_rate": violations / len(completed),
        "jain_index": jain,
        "jain_demand_normalized": jain_demand_normalized,
        "codebook_reloads": codebook_reloads,
        "ddr_transfers": ddr_transfers,
        "per_tenant": per_tenant,
        "claims": dict(claims),
    }


# ---------------------------------------------------------------------------
# Run entry points
# ---------------------------------------------------------------------------


def run_harness(
    *,
    strategy: str,
    window_width: int,
    num_osds: int,
    num_queries: int,
    seed: int,
    num_tenants: int,
    num_clusters: int,
    span_ns: int,
    deadline_ns: int,
    skew: str,
    top_k: int,
    nprobe: int,
    project_root: Path,
    fpga_model: fpga.FpgaServiceModel | None = None,
    placement: Mapping[int, ReplicaSet] | None = None,
    hardware: OSDHardwareProfile | None = None,
    rgw_weights: RGWScoreWeights | None = None,
    osd_weights: ScoreWeights | None = None,
    cluster_skew: str = "uniform",
    heartbeat_ns: int = 0,
    state_ttl_ns: int = 300_000,
    queries: Sequence[SyntheticQuery] | None = None,
    cluster_sizes: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Run one strategy point and return the metrics payload.

    When ``queries`` is given, the harness uses that pre-built trace verbatim
    instead of calling :func:`generate_trace`.  This is the path used by the
    real RAGPulse trace integration (``generate_trace_from_real``).
    """
    if fpga_model is None:
        fpga_model = build_fpga_model(project_root)
    if queries is None:
        queries = generate_trace(
            seed=seed,
            num_queries=num_queries,
            num_tenants=num_tenants,
            num_clusters=num_clusters,
            span_ns=span_ns,
            deadline_ns=deadline_ns,
            skew=skew,
            top_k=top_k,
            nprobe=nprobe,
            codebook_per_tenant=True,
            cluster_skew=cluster_skew,
        )
    completed, info = run_simulation(
        queries,
        num_osds=num_osds,
        strategy=strategy,
        window_width=window_width,
        fpga_model=fpga_model,
        placement=placement,
        hardware=hardware,
        rgw_weights=rgw_weights,
        osd_weights=osd_weights,
        heartbeat_ns=heartbeat_ns,
        state_ttl_ns=state_ttl_ns,
        cluster_sizes=cluster_sizes,
    )
    metrics = compute_metrics(
        completed,
        strategy=strategy,
        window_width=window_width,
        num_osds=num_osds,
        samples=len(queries),
        claims=_default_claims(),
        queries=queries,
    )
    return {
        "schema": SCHEMA,
        "task": TASK,
        "status": STATUS_COMPLETE,
        **info,
        **metrics,
    }


def _default_claims() -> dict[str, Any]:
    return {
        "real_users": False,
        "real_tenants": False,
        "synthetic_trace": True,
        "deterministic_seed": True,
        "pipelining_within_osd": False,
        "osd_server_is_serial": True,
    }


def run_fixture(
    *,
    strategy: str = "multifactor",
    window_width: int = 16,
    num_osds: int = 4,
    num_queries: int = 32,
    seed: int = 7,
    heartbeat_ns: int = 0,
    state_ttl_ns: int = 300_000,
    hardware: OSDHardwareProfile | None = None,
    queries: Sequence[SyntheticQuery] | None = None,
) -> dict[str, Any]:
    """Synthetic fixture run: no model gate required.

    When ``queries`` is given, uses that pre-built trace instead of the
    synthetic generator (real RAGPulse trace path).
    """
    project_root = _PROJECT_ROOT_DEFAULT
    fpga_model = build_fpga_model(project_root)
    return run_harness(
        strategy=strategy,
        window_width=window_width,
        num_osds=num_osds,
        num_queries=num_queries,
        seed=seed,
        num_tenants=3,
        num_clusters=8,
        span_ns=1_000_000,
        deadline_ns=200_000_000,
        skew="zipf",
        top_k=10,
        nprobe=4,
        project_root=project_root,
        fpga_model=fpga_model,
        heartbeat_ns=heartbeat_ns,
        state_ttl_ns=state_ttl_ns,
        hardware=hardware,
        queries=queries,
    )


def run_formal(
    *,
    strategy: str,
    window_width: int = 128,
    num_osds: int = 8,
    num_queries: int = 256,
    seed: int = 42,
    project_root: Path | None = None,
    hardware: OSDHardwareProfile | None = None,
    queries: Sequence[SyntheticQuery] | None = None,
    rgw_weights: RGWScoreWeights | None = None,
    cluster_sizes: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Formal run: refuses with ``blocked`` if the model gate has not passed."""
    root = Path(project_root) if project_root is not None else _PROJECT_ROOT_DEFAULT
    try:
        require_model_gate(project_root=root)
    except ModelGateError as error:
        blocked: dict[str, Any] = {
            "schema": SCHEMA,
            "task": TASK,
            "status": STATUS_BLOCKED,
            "reason": "model-gate-not-passed",
            "message": str(error),
            "refused": True,
        }
        raise ModelGateNotPassedError(
            f"formal OSD scheduling run refused: {error}") from error
    return run_harness(
        strategy=strategy,
        window_width=window_width,
        num_osds=num_osds,
        num_queries=num_queries,
        seed=seed,
        num_tenants=4,
        num_clusters=32,
        span_ns=10_000_000,
        deadline_ns=500_000_000,
        skew="zipf",
        top_k=10,
        nprobe=4,
        project_root=root,
        hardware=hardware,
        queries=queries,
        rgw_weights=rgw_weights,
        cluster_sizes=cluster_sizes,
    )


# ---------------------------------------------------------------------------
# CLI + output
# ---------------------------------------------------------------------------


def _write_json_out(
    path: str | Path | None,
    payload: Mapping[str, Any],
    project_root: Path,
) -> None:
    if path is None:
        return
    target = assert_within_project(path, project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT_DEFAULT)
    parser.add_argument("--mode", choices=("dry-run", "fixture", "formal"),
                        default="fixture")
    parser.add_argument("--strategy", choices=STRATEGIES, default="multifactor")
    parser.add_argument("--window-width", type=int, default=64)
    parser.add_argument("--num-osds", type=int, default=4)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--heartbeat-ms", type=int, default=0,
                        help="RGW heartbeat throttle period in ms (0 = no "
                             "throttle, update state every query; §3.3 models "
                             "the real RGW as a ~100 ms heartbeat)")
    parser.add_argument("--state-ttl-ms", type=int, default=300,
                        help="RGW state staleness TTL in ms (default 300, "
                             "per §3.3); older heartbeats score worst-case")
    parser.add_argument("--vectors-per-cluster", type=int, default=64,
                        help="Vectors per coarse cluster = N/nlist (default 64; "
                             "Cohere 1M/1024 -> 977, MSMARCO 9M/4096 -> 2159, "
                             "1B/4096 -> 243927).  Sets the FPGA scan size.")
    parser.add_argument("--codebook-mode", choices=("groups", "hybrid"),
                        default="groups",
                        help="groups: hash tenants into codebook_groups; "
                             "hybrid: head tenants (>= --hybrid-head-min) get "
                             "private codebooks, the tail shares one")
    parser.add_argument("--slo-base-ms", type=int, default=200,
                        help="base SLO budget in ms (per-query for uniform, "
                             "per-tenant-class for hetero)")
    parser.add_argument("--deadline-mode", choices=("uniform", "hetero"),
                        default="uniform",
                        help="uniform: every query gets the same SLO budget; "
                             "hetero: three tenant classes (0.5x/1.0x/3.0x)")
    parser.add_argument("--hybrid-head-min", type=int, default=5,
                        help="min queries for a tenant to get a private "
                             "codebook under --codebook-mode hybrid")
    parser.add_argument("--affinity-weight", type=float, default=0.0,
                        help="RGW cluster-affinity weight (0 = disabled); "
                             "boosts an OSD already chosen for another cluster "
                             "of the same query, amortising the per-tenant "
                             "codebook load")
    parser.add_argument("--cache-raw", action="store_true",
                        help="Cache a cluster's raw vectors in DDR alongside its "
                             "PQ codes (default: PQ only).  Required for an 8 GiB "
                             "DDR budget to actually constrain a large working set.")
    parser.add_argument("--json-out", type=Path, default=None)
    # Real-trace integration flags.  When --trace is set the harness builds
    # queries from the real RAGPulse events file and the IVF index instead of
    # the synthetic generator.
    parser.add_argument("--trace", type=Path, default=None,
                        help="Path to a RAGPulse events.jsonl trace file.  "
                             "When set, the synthetic trace generator is "
                             "replaced by generate_trace_from_real.")
    parser.add_argument("--trace-split", type=str, default=None,
                        help="Optional trace split filter (e.g. 'test').")
    parser.add_argument("--trace-limit", type=int, default=None,
                        help="Cap the number of arrivals consumed from the trace.")
    parser.add_argument("--trace-scale", type=float, default=1.0,
                        help="Time-scaling factor for the trace replay "
                             "(> 1 = slower replay).")
    parser.add_argument("--index-manifest", type=Path, default=None,
                        help="Path to the IVFPQ index manifest JSON "
                             "(required when --trace is set).")
    parser.add_argument("--nprobe", type=int, default=4,
                        help="Number of coarse IVF clusters to probe per query "
                             "(default 4; only used with --trace).")
    parser.add_argument("--codebook-groups", type=int, default=1,
                        help="Number of codebook groups for tenant->codebook "
                             "mapping.  1 = shared (default); > 1 = stable "
                             "CRC32 hash of tenant into cb-<group>.")
    args = parser.parse_args(argv)

    root = Path(args.project_root)
    if args.heartbeat_ms < 0:
        print(f"error: --heartbeat-ms must be >= 0, got {args.heartbeat_ms}",
              file=sys.stderr)
        return EXIT_CONFIG
    if args.state_ttl_ms < 0:
        print(f"error: --state-ttl-ms must be >= 0, got {args.state_ttl_ms}",
              file=sys.stderr)
        return EXIT_CONFIG
    if args.vectors_per_cluster < 1:
        print(f"error: --vectors-per-cluster must be >= 1, got "
              f"{args.vectors_per_cluster}", file=sys.stderr)
        return EXIT_CONFIG
    heartbeat_ns = args.heartbeat_ms * 1_000_000
    state_ttl_ns = args.state_ttl_ms * 1_000_000
    hardware = OSDHardwareProfile(
        vectors_per_cluster=args.vectors_per_cluster,
        cache_raw_vectors=args.cache_raw,
    )

    # When --trace is set, build the real-trace queries up-front so they can
    # be passed through to run_harness(queries=...) via the fixture/formal
    # entry points.  Without --trace the existing synthetic path is preserved
    # exactly.
    real_trace_queries: list[SyntheticQuery] | None = None
    if args.trace is not None:
        if args.index_manifest is None:
            print("error: --index-manifest is required when --trace is set",
                  file=sys.stderr)
            return EXIT_CONFIG
        if args.nprobe < 1:
            print(f"error: --nprobe must be >= 1, got {args.nprobe}",
                  file=sys.stderr)
            return EXIT_CONFIG
        if args.codebook_groups < 1:
            print(f"error: --codebook-groups must be >= 1, got "
                  f"{args.codebook_groups}", file=sys.stderr)
            return EXIT_CONFIG
        try:
            real_trace_queries = generate_trace_from_real(
                args.trace,
                deadline_ns=args.slo_base_ms * 1_000_000,
                index_manifest=args.index_manifest,
                project_root=root,
                nprobe=args.nprobe,
                top_k=10,
                split=args.trace_split,
                limit=args.trace_limit,
                scale_factor=args.trace_scale,
                codebook_groups=args.codebook_groups,
                codebook_mode=args.codebook_mode,
                hybrid_head_min=args.hybrid_head_min,
                deadline_mode=args.deadline_mode,
            )
        except OsdSchedError as error:
            print(f"error: {error}", file=sys.stderr)
            return EXIT_CONFIG

    try:
        if args.mode == "dry-run":
            payload = {
                "schema": SCHEMA,
                "task": TASK,
                "mode": "dry-run",
                "strategy": args.strategy,
                "window_width": args.window_width,
                "num_osds": args.num_osds,
                "queries": args.queries,
                "writes": False,
            }
            print(json.dumps(payload, indent=2, sort_keys=True))
            if args.json_out is not None:
                _write_json_out(args.json_out, payload, root)
            return EXIT_OK

        if args.mode == "fixture":
            payload = run_fixture(
                strategy=args.strategy,
                window_width=args.window_width,
                num_osds=args.num_osds,
                num_queries=args.queries,
                seed=args.seed,
                heartbeat_ns=heartbeat_ns,
                state_ttl_ns=state_ttl_ns,
                hardware=hardware,
                queries=real_trace_queries,
            )
            payload["status"] = STATUS_FIXTURE
            payload["evidence_class"] = (
                "REAL_TRACE" if real_trace_queries is not None else "SYNTHETIC")
            print(json.dumps(payload, indent=2, sort_keys=True))
            if args.json_out is not None:
                _write_json_out(args.json_out, payload, root)
            return EXIT_OK

        # formal
        rgw_weights = (
            RGWScoreWeights(affinity=args.affinity_weight)
            if args.affinity_weight > 0 else None
        )
        cluster_sizes = None
        if real_trace_queries is not None and args.index_manifest is not None:
            try:
                _idx = IVFPQIndex.from_manifest(
                    args.index_manifest, project_root=root)
                _counts = np.bincount(_idx.assign, minlength=_idx.nlist)
                cluster_sizes = {int(i): int(c) for i, c in enumerate(_counts)}
            except Exception:
                cluster_sizes = None
        payload = run_formal(
            strategy=args.strategy,
            window_width=args.window_width,
            num_osds=args.num_osds,
            num_queries=args.queries,
            seed=args.seed,
            project_root=root,
            hardware=hardware,
            queries=real_trace_queries,
            rgw_weights=rgw_weights,
            cluster_sizes=cluster_sizes,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.json_out is not None:
            _write_json_out(args.json_out, payload, root)
        return EXIT_OK

    except ModelGateNotPassedError as error:
        blocked = {
            "schema": SCHEMA,
            "task": TASK,
            "status": STATUS_BLOCKED,
            "reason": "model-gate-not-passed",
            "message": str(error),
            "refused": True,
        }
        print(json.dumps(blocked, indent=2, sort_keys=True))
        if args.json_out is not None:
            try:
                _write_json_out(args.json_out, blocked, root)
            except (UnsafePathError, OSError):
                pass
        return EXIT_BLOCKED
    except (OsdSchedError, ModelGateError, UnsafePathError, ContractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_CONFIG


__all__ = [
    "CompletedQuery",
    "OsdSchedError",
    "OsdScheduler",
    "SCHEMA",
    "STATUS_BLOCKED",
    "STATUS_COMPLETE",
    "STATUS_FIXTURE",
    "STRATEGIES",
    "SyntheticQuery",
    "build_fpga_model",
    "ceph_placement",
    "compute_metrics",
    "generate_trace",
    "generate_trace_from_real",
    "main",
    "run_fixture",
    "run_formal",
    "run_harness",
    "run_simulation",
    "OSDHardwareProfile",
    "RGWScoreWeights",
]


if __name__ == "__main__":
    sys.exit(main())
