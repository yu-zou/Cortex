"""Task 14: per-OSD query ordering and multi-query batch planning.

Ordering strategies:
- FIFO: arrival order baseline;
- SPT: shortest estimated completion from DDR-hit probability, queue depth and
  estimated compute;
- EDF: earliest deadline first for SLO-bound tenants;
- DRR: deficit round robin across tenants with optional weights.

Batch planning groups queries that can share work: same tenant and codebook
(same nprobe) save FPGA codebook reloads, and repeated clusters inside a batch
merge into one DDR movement. All inputs are scheduler-visible fields supplied by
the caller; this module never reads a DES clock or future arrivals.
"""

from __future__ import annotations

import dataclasses
import math
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

from strata_sim.contracts.schema import ContractError


class QueueStrategy(str, Enum):
    FIFO = "fifo"
    SPT = "spt"
    EDF = "edf"
    DRR = "drr"
    BATCH = "batch"
    CGB = "cgb"  # Cluster-Group Batching
    LSTF = "lstf"  # Least-Slack-Time-First (physical units, no weights)


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string, got {value!r}")


def _nonneg(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class QueuedQuery:
    """One scheduler-visible query request assigned to an OSD queue."""

    query_id: str
    tenant: str
    cluster_id: int
    nprobe: int = 1
    arrival_ns: int = 0
    deadline_ns: int | None = None
    estimated_compute_ns: int = 1_000
    ddr_hit_probability: float = 0.0
    codebook_id: str | None = None
    priority: int = 0
    cluster_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _nonempty("query_id", self.query_id)
        _nonempty("tenant", self.tenant)
        _nonneg("cluster_id", self.cluster_id)
        _nonneg("arrival_ns", self.arrival_ns)
        _nonneg("estimated_compute_ns", self.estimated_compute_ns)
        if not isinstance(self.nprobe, int) or self.nprobe < 1:
            raise ContractError(f"nprobe must be a positive int, got {self.nprobe!r}")
        if self.deadline_ns is not None:
            _nonneg("deadline_ns", self.deadline_ns)
        if not isinstance(self.ddr_hit_probability, (int, float)) \
                or not 0.0 <= float(self.ddr_hit_probability) <= 1.0:
            raise ContractError(
                f"ddr_hit_probability must be in [0, 1], got {self.ddr_hit_probability!r}")
        if not isinstance(self.cluster_ids, tuple) or not all(
                isinstance(cid, int) and cid >= 0 for cid in self.cluster_ids):
            raise ContractError(
                f"cluster_ids must be a tuple of non-negative ints, got "
                f"{self.cluster_ids!r}")

    @property
    def codebook(self) -> str:
        return self.codebook_id if self.codebook_id is not None else f"nprobe={self.nprobe}"


@dataclass
class QueryQueue:
    """One OSD's pending queries with a deterministic ordering strategy."""

    strategy: QueueStrategy | str = QueueStrategy.FIFO
    base_service_ns: int = 1_000
    miss_penalty_ns: int = 10_000
    weights: Mapping[str, int] | None = None
    quantum: int = 1

    _pending: dict[str, QueuedQuery] = field(init=False, default_factory=dict)
    _insertion: list[str] = field(init=False, default_factory=list)
    _sequence: dict[str, int] = field(init=False, default_factory=dict)
    _tenant_order: list[str] = field(init=False, default_factory=list)
    _tenant_queues: dict[str, deque] = field(init=False, default_factory=dict)
    _deficit: dict[str, int] = field(init=False, default_factory=dict)
    _rr_index: int = field(init=False, default=0)
    _pushed: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if isinstance(self.strategy, str) and not isinstance(self.strategy, QueueStrategy):
            try:
                self.strategy = QueueStrategy(self.strategy)
            except ValueError as exc:
                raise ContractError(
                    f"unknown queue strategy {self.strategy!r}; "
                    f"expected one of {[s.value for s in QueueStrategy]}") from exc
        if not isinstance(self.strategy, QueueStrategy):
            raise ContractError(f"strategy must be a QueueStrategy, got {self.strategy!r}")
        if self.base_service_ns < 0 or self.miss_penalty_ns < 0:
            raise ContractError("service and miss penalty estimates must be non-negative")
        if not isinstance(self.quantum, int) or self.quantum < 1:
            raise ContractError(f"quantum must be a positive int, got {self.quantum!r}")

    @property
    def pending(self) -> int:
        return len(self._pending)

    def pending_for(self, tenant: str) -> int:
        return len(self._tenant_queues.get(tenant, ()))

    def push(self, query: QueuedQuery) -> None:
        if not isinstance(query, QueuedQuery):
            raise ContractError(f"push expects a QueuedQuery, got {type(query).__name__}")
        if query.query_id in self._pending:
            raise ContractError(f"duplicate query_id {query.query_id!r}")
        self._pending[query.query_id] = query
        self._insertion.append(query.query_id)
        self._sequence[query.query_id] = self._pushed
        self._pushed += 1
        if query.tenant not in self._tenant_queues:
            self._tenant_order.append(query.tenant)
            self._tenant_queues[query.tenant] = deque()
            self._deficit[query.tenant] = 0
        self._tenant_queues[query.tenant].append(query.query_id)

    def remove(self, query_id: str) -> QueuedQuery | None:
        query = self._pending.pop(query_id, None)
        if query is None:
            return None
        self._insertion.remove(query_id)
        self._sequence.pop(query_id, None)
        self._tenant_queues[query.tenant].remove(query_id)
        return query

    def estimated_finish_ns(self, query: QueuedQuery, now_ns: int = 0) -> int:
        """SPT estimate: queue depth plus compute plus the DDR miss penalty."""
        queue_delay = self.pending * self.base_service_ns
        miss = int(round((1.0 - float(query.ddr_hit_probability)) * self.miss_penalty_ns))
        return now_ns + queue_delay + query.estimated_compute_ns + miss

    def _slack_ns(self, query: QueuedQuery, now_ns: int = 0) -> int:
        """Slack in nanoseconds: how long this query can still wait.

        deadline_ns is None (no SLO) => treated as infinite slack, so the
        query sorts last.  estimated_compute_ns is the DES service time,
        filled from the OSD timing model (misses, codebook, tile count).
        """
        if query.deadline_ns is None:
            return math.inf
        return max(0, query.deadline_ns - now_ns - query.estimated_compute_ns)

    def peek(self, now_ns: int = 0) -> QueuedQuery | None:
        """Non-mutating preview of the ``pop`` selection."""
        if not self._pending:
            return None
        clone = QueryQueue(
            strategy=self.strategy,
            base_service_ns=self.base_service_ns,
            miss_penalty_ns=self.miss_penalty_ns,
            weights=self.weights,
            quantum=self.quantum,
        )
        for query_id in self._insertion:
            clone.push(self._pending[query_id])
        return clone.pop(now_ns)

    def pop(self, now_ns: int = 0) -> QueuedQuery:
        if not self._pending:
            raise IndexError("no queued queries")
        if self.strategy is QueueStrategy.FIFO:
            query_id = min(self._insertion,
                           key=lambda qid: (self._pending[qid].arrival_ns, qid))
        elif self.strategy is QueueStrategy.SPT:
            query_id = min(self._insertion, key=lambda qid: (
                self.estimated_finish_ns(self._pending[qid], now_ns), qid))
        elif self.strategy is QueueStrategy.BATCH:
            # Codebook-grouped FIFO: process all queries sharing the same
            # codebook consecutively (one codebook load per group), with
            # groups ordered by their earliest arrival.  A query's clusters
            # share the tenant's codebook, so this also keeps a query's
            # sub-queries together -- which matters because a query finishes
            # only when its slowest sub-query does.
            group_arrival: dict[str, int] = {}
            for qid in self._insertion:
                q = self._pending[qid]
                cb = q.codebook
                if cb not in group_arrival or q.arrival_ns < group_arrival[cb]:
                    group_arrival[cb] = q.arrival_ns
            best_cb = min(group_arrival,
                          key=lambda cb: (group_arrival[cb], cb))
            query_id = min(
                (qid for qid in self._insertion
                 if self._pending[qid].codebook == best_cb),
                key=lambda qid: (self._pending[qid].arrival_ns, qid))
        elif self.strategy is QueueStrategy.LSTF:
            query_id = min(self._insertion, key=lambda qid: (
                self._slack_ns(self._pending[qid], now_ns),
                self._pending[qid].arrival_ns,
                qid))
        elif self.strategy is QueueStrategy.EDF:
            query_id = min(self._insertion, key=lambda qid: (
                self._pending[qid].deadline_ns
                if self._pending[qid].deadline_ns is not None else math.inf,
                self._pending[qid].arrival_ns or self._sequence[qid],
                qid,
            ))
        else:
            query_id = self._select_drr()
        return self.remove(query_id)

    def order(self, now_ns: int = 0) -> tuple[str, ...]:
        """Non-mutating deterministic preview of the pop order."""
        clone = QueryQueue(
            strategy=self.strategy,
            base_service_ns=self.base_service_ns,
            miss_penalty_ns=self.miss_penalty_ns,
            weights=self.weights,
            quantum=self.quantum,
        )
        for query_id in self._insertion:
            clone.push(self._pending[query_id])
        served: list[str] = []
        while clone.pending:
            served.append(clone.pop(now_ns).query_id)
        return tuple(served)

    def _weight_for(self, tenant: str) -> int:
        if self.weights is None:
            return 1
        weight = self.weights.get(tenant, 1)
        if not isinstance(weight, int) or weight < 1:
            raise ContractError(f"tenant weight must be a positive int, got {weight!r}")
        return weight

    def _select_drr(self) -> str:
        tenant_count = len(self._tenant_order)
        misses = 0
        while misses <= tenant_count:
            tenant = self._tenant_order[self._rr_index % tenant_count]
            queue = self._tenant_queues[tenant]
            if not queue:
                self._rr_index = (self._rr_index + 1) % tenant_count
                misses += 1
                continue
            if self._deficit[tenant] < self.quantum:
                self._deficit[tenant] += self.quantum * self._weight_for(tenant)
            if self._deficit[tenant] < self.quantum:
                self._rr_index = (self._rr_index + 1) % tenant_count
                misses += 1
                continue
            self._deficit[tenant] -= self.quantum
            if self._deficit[tenant] < self.quantum:
                self._rr_index = (self._rr_index + 1) % tenant_count
            return queue[0]
        raise IndexError("no queued queries")


@dataclass(frozen=True)
class QueryBatch:
    """A group of queries sharing a tenant and codebook on one OSD."""

    tenant: str
    codebook: str
    nprobe: int
    queries: tuple[QueuedQuery, ...]

    @property
    def clusters(self) -> tuple[int, ...]:
        seen: list[int] = []
        for query in self.queries:
            if query.cluster_id not in seen:
                seen.append(query.cluster_id)
        return tuple(seen)

    @property
    def codebook_reloads(self) -> int:
        return 1

    @property
    def ddr_movements(self) -> int:
        return len(self.clusters)

    @property
    def merged_queries(self) -> int:
        return len(self.queries) - len(self.clusters)


@dataclass(frozen=True)
class BatchPlan:
    """Deterministic batching outcome with the saved-work counters."""

    batches: tuple[QueryBatch, ...]
    baseline_codebook_reloads: int
    baseline_ddr_movements: int

    @property
    def codebook_reloads(self) -> int:
        return len(self.batches)

    @property
    def saved_reloads(self) -> int:
        return self.baseline_codebook_reloads - self.codebook_reloads

    @property
    def ddr_movements(self) -> int:
        return sum(batch.ddr_movements for batch in self.batches)

    @property
    def merged_movements(self) -> int:
        return self.baseline_ddr_movements - self.ddr_movements


class BatchPlanner:
    """Groups queries by tenant/codebook and merges repeated cluster reads."""

    def __init__(self, max_batch_size: int | None = None) -> None:
        if max_batch_size is not None and (
                not isinstance(max_batch_size, int) or max_batch_size < 1):
            raise ContractError(
                f"max_batch_size must be a positive int, got {max_batch_size!r}")
        self.max_batch_size = max_batch_size

    def plan(self, queries: Iterable[QueuedQuery]) -> BatchPlan:
        ordered = sorted(queries, key=lambda query: (query.arrival_ns, query.query_id))
        for query in ordered:
            if not isinstance(query, QueuedQuery):
                raise ContractError(
                    f"plan expects QueuedQuery items, got {type(query).__name__}")
        groups: dict[tuple[str, str, int], list[QueuedQuery]] = {}
        for query in ordered:
            groups.setdefault((query.tenant, query.codebook, query.nprobe), []).append(query)
        batches: list[QueryBatch] = []
        for key in sorted(groups):
            tenant, codebook, nprobe = key
            group = groups[key]
            size = self.max_batch_size or len(group)
            for start in range(0, len(group), size):
                batches.append(QueryBatch(
                    tenant=tenant,
                    codebook=codebook,
                    nprobe=nprobe,
                    queries=tuple(group[start:start + size]),
                ))
        return BatchPlan(
            batches=tuple(batches),
            baseline_codebook_reloads=len(ordered),
            baseline_ddr_movements=len(ordered),
        )


# ---------------------------------------------------------------------------
# Sliding-window multi-factor OSD scheduler (design doc §2).
#
# The window holds up to `width` queries pulled from an unbounded backlog.
# On each select_next(), the window is (re-)filled from the backlog, every
# windowed query is scored across aging/codebook-reuse/DDR-hit/deadline/cost
# factors, a soft tenant-fairness penalty is applied, and the highest scoring
# query is removed for execution. The remaining windowed queries stay put so
# their aging keeps accumulating — this is the "sliding" part. All state is
# caller-supplied; no DES clock, no randomness, no future-arrival peeking.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreWeights:
    """Factor weights and normalisers for the multi-factor score (§2).

    The score is a weighted linear combination of five factors, each
    normalised to [0,1]. `starvation_cap_ns` caps the aging factor so a
    query that arrived before the observable universe still scores 1.0;
    `slo_window_ns` is the SLO slack beyond which deadline urgency is 0.
    `fairness_factor` and `tenant_penalty` control the soft per-tenant
    quota: when a tenant occupies more than
    ceil(width / num_tenants) * fairness_factor slots, its scores are
    multiplied by tenant_penalty (never dropped).
    """

    aging: float = 3.0
    codebook: float = 0.0
    ddr: float = 0.0
    slow_first: float = 2.0
    deadline: float = 2.0
    cost: float = 2.0
    byte: float = 0.0  # Byte-Aware Scheduling weight (0 = disabled)
    cgb: float = 0.0  # Cluster-Group Batching weight (0 = disabled)
    starvation_cap_ns: int = 200_000_000
    slo_window_ns: int = 100_000_000
    fairness_factor: float = 1.5
    tenant_penalty: float = 0.5
    deadline_feasible: bool = False

    def __post_init__(self) -> None:
        for name in ("aging", "codebook", "ddr", "deadline", "cost", "byte", "cgb",
                     "fairness_factor", "tenant_penalty"):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ContractError(
                    f"ScoreWeights.{name} must be a finite number, got {val!r}")
            if val < 0:
                raise ContractError(
                    f"ScoreWeights.{name} must be non-negative, got {val!r}")
        for name in ("starvation_cap_ns", "slo_window_ns"):
            val = getattr(self, name)
            if not isinstance(val, int) or val <= 0:
                raise ContractError(
                    f"ScoreWeights.{name} must be a positive int, got {val!r}")


@dataclass(frozen=True)
class OsdView:
    """Caller-supplied snapshot of OSD state (no DES clock access).

    `now_ns` anchors the aging and deadline-urgency factors. The two
    frozensets describe what is already resident on the OSD — BRAM
    codebooks avoid an FPGA reload, and DDR-cached clusters avoid a
    host→device transfer. `max_compute_ns` normalises the estimated
    compute cost factor to [0,1].
    """

    now_ns: int = 0
    bram_codebooks: frozenset[str] = frozenset()
    ddr_cached_clusters: frozenset[int] = frozenset()
    max_compute_ns: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.now_ns, int) or self.now_ns < 0:
            raise ContractError(
                f"OsdView.now_ns must be a non-negative int, got {self.now_ns!r}")
        if not isinstance(self.bram_codebooks, frozenset):
            raise ContractError(
                f"OsdView.bram_codebooks must be a frozenset, got "
                f"{type(self.bram_codebooks).__name__}")
        if not isinstance(self.ddr_cached_clusters, frozenset):
            raise ContractError(
                f"OsdView.ddr_cached_clusters must be a frozenset, got "
                f"{type(self.ddr_cached_clusters).__name__}")
        if not isinstance(self.max_compute_ns, int) or self.max_compute_ns < 1:
            raise ContractError(
                f"OsdView.max_compute_ns must be a positive int, got "
                f"{self.max_compute_ns!r}")


class SchedulingWindow:
    """Sliding-window, multi-factor OSD scheduler (§2 of the design doc).

    The window is a bounded buffer over an unbounded FIFO backlog. On each
    `select_next()` call the window is first topped up from the backlog up
    to `width`, then every windowed query is scored and the best one is
    removed for execution; the rest stay in the window so their aging keeps
    accumulating across calls. `order()` returns the full deterministic
    select_next sequence without mutating the window.
    """

    def __init__(
        self,
        width: int = 64,
        weights: ScoreWeights | None = None,
        cluster_sizes: dict[int, int] | None = None,
        adaptive: bool = True,
    ) -> None:
        if not isinstance(width, int) or width < 1:
            raise ContractError(
                f"SchedulingWindow width must be a positive int, got {width!r}")
        self._width = width
        self._weights = weights if weights is not None else ScoreWeights()
        # Backlog: insertion order, queries not yet in the window.
        self._backlog: deque[QueuedQuery] = deque()
        self._backlog_ids: set[str] = set()
        # Window: insertion order among windowed queries.
        self._window: deque[QueuedQuery] = deque()
        self._window_ids: set[str] = set()
        # Adaptive weight parameters for dynamic parameter tuning
        self._cluster_sizes = cluster_sizes or {}
        self._adaptive = adaptive and bool(cluster_sizes)
        # Memoised adaptive parameters (cluster_sizes is fixed per window).
        self._tail_h_cache: float | None = None
        self._adaptive_weights_cache: ScoreWeights | None = None
        # Online policy selection (SEL=1): pick one of three weight sets per pop.
        self._sel_modes = None
        if os.environ.get("SEL", "0") == "1":
            _base = self._weights
            self._sel_modes = {
                "fifo": dataclasses.replace(
                    _base, aging=0.0, codebook=0.0, ddr=0.0, deadline=0.0,
                    cost=0.0, byte=0.0, cgb=0.0, slow_first=0.0, tenant_penalty=1.0),
                "cras": dataclasses.replace(
                    _base, aging=4.0, codebook=1.0, ddr=0.0, deadline=0.5, cost=0.0),
                "ddr": dataclasses.replace(
                    _base, aging=4.0, codebook=0.0, ddr=3.0, deadline=0.5, cost=0.0),
            }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Queries not yet executed: window + backlog."""
        return len(self._window) + len(self._backlog)

    @property
    def window_size(self) -> int:
        """Queries currently held in the window."""
        return len(self._window)

    @property
    def width(self) -> int:
        return self._width

    def window_queries(self) -> list["QueuedQuery"]:
        """Snapshot of the pending scheduling window.

        Used by window-wide staging: any query in the window (not just the
        immediate next one) can supply a prefetch candidate.
        """
        return list(self._window)

    @property
    def weights(self) -> ScoreWeights:
        return self._weights

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, query: QueuedQuery) -> None:
        """Append a query to the backlog. The window fills on select_next."""
        if not isinstance(query, QueuedQuery):
            raise ContractError(
                f"SchedulingWindow.push expects a QueuedQuery, got "
                f"{type(query).__name__}")
        if query.query_id in self._backlog_ids \
                or query.query_id in self._window_ids:
            raise ContractError(
                f"duplicate query_id {query.query_id!r}")
        self._backlog.append(query)
        self._backlog_ids.add(query.query_id)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _fill_window(self) -> None:
        """Promote from backlog into the window up to `width`."""
        while len(self._window) < self._width and self._backlog:
            q = self._backlog.popleft()
            self._backlog_ids.discard(q.query_id)
            self._window.append(q)
            self._window_ids.add(q.query_id)

    def _compute_tail_heaviness(self) -> float:
        """Compute tail heaviness of cluster size distribution.

        Uses max/median ratio: higher ratio means heavier tail.
        - Cohere 1M: max/median ≈ 3.33 (relatively uniform)
        - MSMARCO 9M: max/median ≈ 4.38 (heavy-tailed)
        
        This metric has better separation than CV for distinguishing
        distributions where LPT helps vs hurts.
        """
        cached = self._tail_h_cache
        if cached is not None:
            return cached

        if not self._cluster_sizes:
            return 0.0

        sizes = list(self._cluster_sizes.values())
        if len(sizes) < 2:
            return 0.0

        sizes_sorted = sorted(sizes)
        n = len(sizes_sorted)
        median = sizes_sorted[n // 2]
        max_size = sizes_sorted[-1]
        
        if median == 0:
            return 0.0
        
        val = max_size / median
        self._tail_h_cache = val
        return val

    def _adaptive_weights(self) -> ScoreWeights:
        """Return effective weights based on cluster size distribution tail_heaviness.

        - uniform-ish (tail_heaviness < 3.5): use minimal weights (FIFO-like)
        - heavy-tailed (tail_heaviness > 4.0): aging=2.0, cost=0.75, no deadline
        - in between: interpolate
        """
        if not self._adaptive:
            return self._weights

        cached = self._adaptive_weights_cache
        if cached is not None:
            return cached

        tail_h = self._compute_tail_heaviness()
        base = self._weights

        # Thresholds: 3.5 (uniform) to 4.0 (heavy-tailed)
        if tail_h < 3.5:
            # Uniform: minimal weights (close to FIFO)
            factor = 0.0
        elif tail_h > 4.0:
            # Heavy-tailed: full SPT
            factor = 1.0
        else:
            # Interpolate
            factor = (tail_h - 3.5) / 0.5

        # Tuned on 1B mixed workload (PQ 3 GiB, rep=3, 20 OSD): aging=2.0,
        # cost=0.75, deadline=0 beats FIFO on BOTH SLO and P99 at practical
        # load points (scale 4000-5000). Deadline urgency is disabled: it
        # helps SLO at light load but blows up P99 under saturation.
        aging = 2.0 * factor
        cost = 0.75 * factor
        deadline = 0.0

        result = ScoreWeights(
            aging=aging,
            codebook=0.0,
            ddr=0.0,
            slow_first=0.0,
            deadline=deadline,
            cost=cost,
            byte=base.byte,
            cgb=base.cgb,
            starvation_cap_ns=base.starvation_cap_ns,
            slo_window_ns=base.slo_window_ns,
            fairness_factor=base.fairness_factor,
            tenant_penalty=base.tenant_penalty,
            deadline_feasible=base.deadline_feasible,
        )
        self._adaptive_weights_cache = result
        return result
    def _compute_byte_cost(self, q: QueuedQuery, view: OsdView) -> int:
        """Total bytes that must be loaded from SSD for this query's clusters."""
        if not self._cluster_sizes:
            return 0
        miss = 0
        for cid in (q.cluster_ids or (q.cluster_id,)):
            if cid not in view.ddr_cached_clusters:
                n = self._cluster_sizes.get(cid, 64)
                miss += 64 + n * 128  # header + PQ entries
        return miss

    def _select_policy(self, view: OsdView) -> ScoreWeights:
        """Pick FIFO / CRAS / DDR from the current window state.

        * no congestion (pending <= q_low)      -> FIFO (order does not matter)
        * many codebooks contending (div >= d)  -> CRAS (avoid reloads)
        * cache thrash (hit rate < h_low)       -> DDR  (protect resident clusters)
        else                                    -> FIFO
        """
        if os.environ.get("SEL_MODE", "") == "lock":
            # one-way ratchet: FIFO until the first congestion, then CRAS forever
            if getattr(self, "_sel_lock", False):
                return self._sel_modes["cras"]
            if self.pending > int(os.environ.get("SEL_Q_LOW", "3")):
                self._sel_lock = True
                return self._sel_modes["cras"]
            return self._sel_modes["fifo"]
        self._sel_n = getattr(self, "_sel_n", 0) + 1
        _every = int(os.environ.get("SEL_EVERY", "64"))
        if self._sel_n % _every != 1 and getattr(self, "_sel_last", None) is not None:
            return self._sel_last
        q_low = int(os.environ.get("SEL_Q_LOW", "3"))
        d_high = float(os.environ.get("SEL_CB_DIV", "0.5"))
        h_low = float(os.environ.get("SEL_HIT_LOW", "0.35"))
        if self.pending <= q_low:
            self._sel_last = self._sel_modes["fifo"]
            return self._sel_last
        _cbs = {q.codebook_id for q in self._window}
        div = len(_cbs) / len(self._window)
        _wanted = set()
        for q in self._window:
            _wanted.update(q.cluster_ids or (q.cluster_id,))
        hit = (len(_wanted & view.ddr_cached_clusters) / len(_wanted)) if _wanted else 1.0
        q_high = int(os.environ.get("SEL_Q_HIGH", "8"))
        if div >= d_high:
            self._sel_last = self._sel_modes["cras"]
            return self._sel_last
        if self.pending <= q_high and hit < h_low:
            mode = "ddr"
        else:
            mode = "fifo"
        self._sel_last = self._sel_modes[mode]
        return self._sel_last

    def _score_one(self, q: QueuedQuery, view: OsdView) -> float:
        """Compute the raw multi-factor score for a single query.

        Each factor is independently normalised to [0,1]; the final score
        is the weighted linear combination with the cost term subtracted.
        """
        w = getattr(self, "_cur_w", None) or self._adaptive_weights()
        # aging: fraction of starvation_cap_ns waited so far, capped at 1.
        waited = max(0, view.now_ns - q.arrival_ns)
        aging = min(1.0, waited / w.starvation_cap_ns)
        # codebook reuse: resident in FPGA BRAM ⇒ zero reload cost.
        codebook_reusable = (
            1.0 if q.codebook_id is not None
            and q.codebook_id in view.bram_codebooks
            else 0.0
        )
        # DDR hit ratio: prefer cluster_ids intersection when available.
        if q.cluster_ids:
            wanted = set(q.cluster_ids)
            hits = wanted & view.ddr_cached_clusters
            ddr_hit = len(hits) / len(wanted)
        else:
            ddr_hit = float(q.ddr_hit_probability)
        # deadline urgency: 0 if no deadline, else fraction of slo_window
        # already consumed; 1.0 means the deadline has passed or is at now.
        if q.deadline_ns is None:
            deadline_urgency = 0.0
        else:
            remaining = max(0, q.deadline_ns - view.now_ns)
            if w.deadline_feasible and remaining <= 0:
                # deadline already unrecoverable: urgency cannot rescue this
                # query and would only steal service from salvageable ones.
                deadline_urgency = 0.0
            else:
                deadline_urgency = 1.0 - min(1.0, remaining / w.slo_window_ns)
        # estimated cost: heavier queries pay more, capped at 1.
        estimated_cost = min(
            1.0, q.estimated_compute_ns / max(1, view.max_compute_ns))
        # Byte-Aware Scheduling: prefer queries with low SSD byte cost.
        byte_cost = self._compute_byte_cost(q, view)
        if w.byte > 0 and self._window:
            max_bc = max((self._compute_byte_cost(q2, view) for q2 in self._window), default=1)
            byte_eff = 1.0 - (byte_cost / max_bc if max_bc > 0 else 0)
        else:
            byte_eff = 0.0
        
        # Cluster-Group Batching: boost queries that share clusters with cached clusters
        cgb_bonus = 0.0
        if w.cgb > 0 and q.cluster_ids:
            overlap = len(set(q.cluster_ids) & view.ddr_cached_clusters)
            cgb_bonus = overlap / len(q.cluster_ids) if q.cluster_ids else 0.0
        
        return (
            w.aging * aging
            + getattr(self, "_w_cb_eff", w.codebook) * codebook_reusable
            + w.ddr * ddr_hit
            + w.deadline * deadline_urgency
            + w.slow_first * (1.0 - ddr_hit)
            - w.cost * estimated_cost
            + w.byte * byte_eff
            + w.cgb * cgb_bonus
        )

    def _ranked(self, view: OsdView) -> list[tuple[float, QueuedQuery]]:
        """Score every windowed query, apply soft tenant fairness, rank.

        Tie-break is deterministic: highest score, then earliest arrival,
        then lexicographic query_id.
        """
        w = self._weights
        self._fill_window()
        if not self._window:
            return []
        if self._sel_modes is not None:
            w = self._select_policy(view)
            self._cur_w = w
        # Adaptive codebook weight (opt-in via ADAPT_CB=1): if the window is
        # dominated by one codebook there is no switch to avoid, so the term
        # is scaled down; the more codebooks contend, the closer to w_cb.
        if w.codebook > 0 and os.environ.get("ADAPT_CB", "0") == "1":
            _cc: dict = {}
            for _q in self._window:
                _cc[_q.codebook_id] = _cc.get(_q.codebook_id, 0) + 1
            _thrash = 1.0 - (max(_cc.values()) / len(self._window))
            self._w_cb_eff = w.codebook * _thrash
        else:
            self._w_cb_eff = w.codebook
        # Per-tenant counts inside the window for the soft fairness cap.
        num_tenants = len({q.tenant for q in self._window})
        per_tenant: dict[str, int] = {}
        for q in self._window:
            per_tenant[q.tenant] = per_tenant.get(q.tenant, 0) + 1
        fair_quota = math.ceil(
            self._width / max(1, num_tenants)) * w.fairness_factor
        # Score each query, optionally applying the tenant penalty.
        scored: list[tuple[float, QueuedQuery]] = []
        if self.strategy is QueueStrategy.LSTF or os.environ.get(
                "OSD_POLICY", "window") == "lstf":
            # LSTF: least slack first.  slack = deadline - now - service.
            # All three terms are nanoseconds, so no per-workload weights.
            for q in self._window:
                scored.append((1.0 / max(1.0, float(
                    self._slack_ns(q, view.now_ns))), q))
            scored.sort(key=lambda pair: (
                -pair[0], pair[1].arrival_ns, pair[1].query_id))
            return scored
        for q in self._window:
            s = self._score_one(q, view)
            if per_tenant[q.tenant] > fair_quota:
                s *= w.tenant_penalty
            scored.append((s, q))
        scored.sort(
            key=lambda pair: (-pair[0], pair[1].arrival_ns, pair[1].query_id))
        return scored

    # ------------------------------------------------------------------
    # Public selectors
    # ------------------------------------------------------------------

    def peek_next(self, view: OsdView) -> QueuedQuery | None:
        """Non-mutating preview of the ``select_next`` winner."""
        if not isinstance(view, OsdView):
            raise ContractError(
                f"SchedulingWindow.peek_next expects an OsdView, got "
                f"{type(view).__name__}")
        ranked = self._ranked(view)
        if not ranked:
            return None
        return ranked[0][1]

    def select_next(self, view: OsdView) -> QueuedQuery | None:
        """Fill the window, score, remove the best query.

        Returns None when both window and backlog are empty. Other windowed
        queries remain so their aging keeps accumulating across calls.
        """
        if not isinstance(view, OsdView):
            raise ContractError(
                f"SchedulingWindow.select_next expects an OsdView, got "
                f"{type(view).__name__}")
        ranked = self._ranked(view)
        if not ranked:
            return None
        _, winner = ranked[0]
        self._window_ids.discard(winner.query_id)
        # Remove the winner from the window deque without reshuffling others.
        new_window: deque[QueuedQuery] = deque()
        for q in self._window:
            if q.query_id != winner.query_id:
                new_window.append(q)
        self._window = new_window
        return winner

    def order(self, view: OsdView) -> tuple[str, ...]:
        """Non-mutating deterministic preview of the full select_next sequence."""
        if not isinstance(view, OsdView):
            raise ContractError(
                f"SchedulingWindow.order expects an OsdView, got "
                f"{type(view).__name__}")
        # Snapshot the window and backlog by cloning this scheduler state
        # into a fresh SchedulingWindow and draining it.
        clone = SchedulingWindow(width=self._width, weights=self._weights, cluster_sizes=self._cluster_sizes, adaptive=self._adaptive)
        for q in self._window:
            clone._window.append(q)
            clone._window_ids.add(q.query_id)
        for q in self._backlog:
            clone._backlog.append(q)
            clone._backlog_ids.add(q.query_id)
        result: list[str] = []
        while True:
            nxt = clone.select_next(view)
            if nxt is None:
                break
            result.append(nxt.query_id)
        return tuple(result)

    def scores(self, view: OsdView) -> dict[str, float]:
        """Deterministic current score per windowed query (for inspection).

        Fills the window from the backlog first so the returned map covers
        every query that would participate in the next select_next().
        """
        if not isinstance(view, OsdView):
            raise ContractError(
                f"SchedulingWindow.scores expects an OsdView, got "
                f"{type(view).__name__}")
        w = self._weights
        self._fill_window()
        if not self._window:
            return {}
        num_tenants = len({q.tenant for q in self._window})
        per_tenant: dict[str, int] = {}
        for q in self._window:
            per_tenant[q.tenant] = per_tenant.get(q.tenant, 0) + 1
        fair_quota = math.ceil(
            self._width / max(1, num_tenants)) * w.fairness_factor
        out: dict[str, float] = {}
        for q in self._window:
            s = self._score_one(q, view)
            if per_tenant[q.tenant] > fair_quota:
                s *= w.tenant_penalty
            out[q.query_id] = s
        return out
