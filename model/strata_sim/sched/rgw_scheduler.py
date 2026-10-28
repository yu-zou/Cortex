"""RGW-side multi-factor OSD scheduler.

Implements the multi-factor scoring algorithm from
``RGW_OSD_调度策略定稿.md §3.2``:

    score(osd, cluster) = w_cache   * ddr_hit(osd, cluster)
                        - w_load    * queue_depth(osd)
                        - w_fpga    * fpga_busy_ratio(osd)
                        + w_primary * is_primary(osd, cluster)

with staleness handling per §3.3: when an OSD's heartbeat state is older than
``state_ttl_ns`` (default 300 ms), its DDR cache and FPGA busy fields are
treated as worst-case (ddr_hit=0, fpga_busy_ratio=1.0).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping

from strata_sim.contracts.schema import ContractError

if TYPE_CHECKING:
    from .osd_state_manager import OSDStateManager
    from strata_sim.des.fpga import FpgaServiceModel


# ---------------------------------------------------------------------------
# Frozen input types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RGWScoreWeights:
    """Weights for the multi-factor OSD selection score (RGW_OSD_调度策略定稿.md §3.2).

    ``queue_depth_cap`` and ``fpga_busy_window_ns`` normalise the raw OSD state
    into [0, 1] before weighting.
    """

    cache: float = 3.0
    load: float = 1.5
    fpga: float = 1.0
    primary: float = 0.5
    affinity: float = 0.0
    queue_depth_cap: int = 32
    fpga_busy_window_ns: int = 100_000

    def __post_init__(self) -> None:
        for name in ("cache", "load", "fpga", "primary", "affinity"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise ContractError(
                    f"RGWScoreWeights.{name} must be a number, got {value!r}")
            if math.isnan(value) or math.isinf(value):
                raise ContractError(
                    f"RGWScoreWeights.{name} must be finite, got {value!r}")
            if value < 0:
                raise ContractError(
                    f"RGWScoreWeights.{name} must be >= 0, got {value!r}")
        for cap_name in ("queue_depth_cap", "fpga_busy_window_ns"):
            cap = getattr(self, cap_name)
            if not isinstance(cap, int) or isinstance(cap, bool) or cap <= 0:
                raise ContractError(
                    f"RGWScoreWeights.{cap_name} must be a positive int, got {cap!r}")


@dataclass(frozen=True)
class ReplicaSet:
    """Ceph placement for one cluster: the eligible replicas and the primary.

    RGW_OSD_调度策略定稿.md §3.2: each cluster lives on 1 primary + 2 (or 3)
    replica OSDs; only eligible replicas may be chosen.
    """

    cluster_id: int
    osd_ids: tuple[int, ...]
    primary_osd_id: int

    def __post_init__(self) -> None:
        if not self.osd_ids:
            raise ContractError("ReplicaSet.osd_ids must be non-empty")
        if len(self.osd_ids) != len(set(self.osd_ids)):
            raise ContractError("ReplicaSet.osd_ids must contain unique OSD ids")
        if list(self.osd_ids) != sorted(self.osd_ids):
            raise ContractError("ReplicaSet.osd_ids must be sorted in ascending order")
        if self.primary_osd_id not in self.osd_ids:
            raise ContractError(
                f"ReplicaSet.primary_osd_id={self.primary_osd_id} "
                f"not in osd_ids={self.osd_ids}")


@dataclass(frozen=True)
class ClusterAssignment:
    """Result of selecting an OSD for one cluster."""

    cluster_id: int
    osd_id: int
    score: float
    reason: str


@dataclass(frozen=True)
class QueryRequest:
    """Query request from client."""

    query_id: str
    query_vector: list[float]
    cluster_ids: list[int]  # RGW already did coarse search
    nprobe: int
    top_k: int


@dataclass(frozen=True)
class SchedulingDecision:
    """Result of RGW scheduling decision.

    ``assignments`` has one entry per query cluster, in deterministic
    (sorted-by-cluster_id) order.
    """

    query_id: str
    assignments: tuple[ClusterAssignment, ...]
    reason: str

    @property
    def osd_ids(self) -> tuple[int, ...]:
        """Sorted unique OSD ids used across all assignments."""
        return tuple(sorted({a.osd_id for a in self.assignments}))

    @property
    def primary_osd_id(self) -> int:
        """The OSD assigned the most clusters; ties broken by smallest id."""
        counts: dict[int, int] = {}
        for a in self.assignments:
            counts[a.osd_id] = counts.get(a.osd_id, 0) + 1
        if not counts:
            raise ContractError("SchedulingDecision has no assignments")
        max_count = max(counts.values())
        return min(osd_id for osd_id, c in counts.items() if c == max_count)

    @property
    def osd_id(self) -> int:
        """Backward-compatible alias for :attr:`primary_osd_id`."""
        return self.primary_osd_id

    @property
    def estimated_finish_ns(self) -> int:
        """Max estimated finish over assigned OSDs (from state manager)."""
        return max(
            (a._estimated_finish_ns for a in self.assignments),
            default=0,
        )


# Internal extension: ClusterAssignment carries a hidden estimated finish
# so SchedulingDecision.estimated_finish_ns can compute max across OSDs
# without leaking it into the public frozen dataclass fields.
ClusterAssignment.__dataclass_fields__  # force dataclass init


def _make_cluster_assignment(
    cluster_id: int, osd_id: int, score: float, reason: str,
    *, estimated_finish_ns: int = 0,
) -> ClusterAssignment:
    """Build a ClusterAssignment with a hidden _estimated_finish_ns."""
    ca = ClusterAssignment.__new__(ClusterAssignment)
    object.__setattr__(ca, "cluster_id", cluster_id)
    object.__setattr__(ca, "osd_id", osd_id)
    object.__setattr__(ca, "score", score)
    object.__setattr__(ca, "reason", reason)
    object.__setattr__(ca, "_estimated_finish_ns", estimated_finish_ns)
    return ca


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class RGWScheduler:
    """Multi-factor OSD scheduler (RGW_OSD_调度策略定稿.md §3.2/§3.3).

    For each cluster in a query, selects the best OSD from the cluster's
    replica set (or all OSDs when no placement is supplied — the fallback
    picks the smallest OSD id as primary).

    Parameters
    ----------
    state_manager
        Global OSD state; read for queue_depth, fpga_busy_until_ns, DDR cache.
    placement
        Optional mapping cluster_id → ReplicaSet.  When ``None``, every OSD
        in the state manager is eligible and the primary is the smallest id.
    fpga_model
        Optional FPGA model for estimated finish time.  When ``None``, the
        FPGA service-time term is omitted (no fabrication).
    weights
        Score weights per §3.2.
    heartbeat_ns
        Expected heartbeat interval (informational; §3.3).
    state_ttl_ns
        State staleness threshold (default 300 ms per §3.3).
    """

    def __init__(
        self,
        state_manager: OSDStateManager,
        placement: Mapping[int, ReplicaSet] | None = None,
        fpga_model: FpgaServiceModel | None = None,
        weights: RGWScoreWeights = RGWScoreWeights(),
        heartbeat_ns: int = 100_000,
        state_ttl_ns: int = 300_000,
        ddr_transfer_ns_per_byte: float = 0.0625,
        cpu_rerank_ns_per_vector: int = 1000,
        vectors_per_cluster: int = 64,
    ) -> None:
        self.state_manager = state_manager
        self.placement = dict(placement) if placement is not None else None
        # ``fpga_model`` is optional: without one the FPGA service-time term is
        # skipped (estimate still includes DDR transfer, CPU rerank and the
        # queue-depth penalty), so a caller with no HLS profile can still run.
        self.fpga_model = fpga_model
        self.weights = weights
        self.heartbeat_ns = heartbeat_ns
        self.state_ttl_ns = state_ttl_ns
        self.ddr_transfer_ns_per_byte = ddr_transfer_ns_per_byte
        # Rolling max queue depth observed, so the load term keeps its
        # discriminating power when queues grow past queue_depth_cap
        # (e.g. nprobe=100 bursts).  Without this the normalised queue depth
        # saturates at 1.0 for every OSD and the RGW cannot balance load.
        self._max_queue_depth: int = 1
        self.cpu_rerank_ns_per_vector = cpu_rerank_ns_per_vector
        self.vectors_per_cluster = vectors_per_cluster

    # ------------------------------------------------------------------
    # Placement resolution
    # ------------------------------------------------------------------

    def _resolve_replica_set(self, cluster_id: int) -> ReplicaSet:
        """Resolve the replica set for a cluster.

        If ``placement`` was supplied at init time, look it up.  Otherwise
        fall back: every OSD in the state manager is eligible, primary is the
        smallest id.
        """
        if self.placement is not None:
            rs = self.placement.get(cluster_id)
            if rs is None:
                raise ContractError(
                    f"no placement entry for cluster_id={cluster_id}")
            return rs
        # Fallback: all known OSDs are eligible.
        all_osd_ids = tuple(sorted(self.state_manager.get_all_osd_states().keys()))
        if not all_osd_ids:
            raise ContractError(
                f"no eligible replicas for cluster_id={cluster_id}: "
                "state manager has no OSDs and no placement was supplied")
        return ReplicaSet(
            cluster_id=cluster_id,
            osd_ids=all_osd_ids,
            primary_osd_id=all_osd_ids[0],
        )

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def _now(self, now_ns: int | None) -> int:
        return self.state_manager.now_ns if now_ns is None else now_ns

    def score(
        self, osd_id: int, cluster_id: int, *, now_ns: int | None = None,
    ) -> float:
        """Compute the multi-factor score for one (osd, cluster) pair.

        RGW_OSD_调度策略定稿.md §3.2:
            score = w_cache * ddr_hit
                  - w_load  * queue_depth_norm
                  - w_fpga  * fpga_busy_ratio
                  + w_primary * is_primary

        Delegates to :meth:`_score_detail` so the formula lives in exactly one
        place (the detail variant is what ``select_osd`` uses for its reason
        string).
        """
        return self._score_detail(osd_id, cluster_id, now_ns=now_ns)[0]

    def _score_detail(
        self, osd_id: int, cluster_id: int, *, now_ns: int | None = None,
    ) -> tuple[float, bool, float, float, float, str]:
        """Return (score, stale, ddr_hit, queue_depth_norm, fpga_busy_ratio, reason_str)."""
        w = self.weights
        t = self._now(now_ns)
        state = self.state_manager.get_osd_state(osd_id)
        rs = self._resolve_replica_set(cluster_id)
        if osd_id not in rs.osd_ids:
            raise ContractError(
                f"osd_id={osd_id} not in replica set for cluster_id={cluster_id}")
        stale = state.is_stale(t, self.state_ttl_ns)
        reasons: list[str] = []

        if stale:
            ddr_hit = 0.0
            reasons.append("stale:no_ddr")
        else:
            ddr_hit = 1.0 if cluster_id in state.ddr_cached_clusters else 0.0

        self._max_queue_depth = max(self._max_queue_depth, state.queue_depth)
        cap = max(w.queue_depth_cap, self._max_queue_depth)
        queue_depth_norm = min(1.0, state.queue_depth / cap)

        if stale:
            fpga_busy_ratio = 1.0
            reasons.append("stale:busy_fpga")
        else:
            remaining = state.fpga_busy_until_ns - t
            if remaining <= 0:
                fpga_busy_ratio = 0.0
            else:
                fpga_busy_ratio = min(1.0, remaining / w.fpga_busy_window_ns)

        is_primary_val = 1.0 if osd_id == rs.primary_osd_id else 0.0

        s = (w.cache * ddr_hit
             - w.load * queue_depth_norm
             - w.fpga * fpga_busy_ratio
             + w.primary * is_primary_val)

        reason = "; ".join(reasons) if reasons else ""
        return s, stale, ddr_hit, queue_depth_norm, fpga_busy_ratio, reason

    # ------------------------------------------------------------------
    # Select
    # ------------------------------------------------------------------

    def select_osd(
        self, cluster_id: int, *, now_ns: int | None = None,
        preferred_osds: frozenset | set = frozenset(),
    ) -> ClusterAssignment:
        """Select the best OSD for one cluster.

        Ties broken by (score desc, ddr_hit desc, osd_id asc) for
        determinism (RGW_OSD_调度策略定稿.md §3.2).
        """
        rs = self._resolve_replica_set(cluster_id)
        t = self._now(now_ns)

        candidates: list[tuple[float, float, int]] = []
        details: dict[int, tuple[float, bool, float, float, float, str]] = {}
        effective: dict[int, float] = {}
        for osd_id in rs.osd_ids:
            detail = self._score_detail(osd_id, cluster_id, now_ns=now_ns)
            score_val, stale, ddr_hit, qd_norm, fpga_br, stale_reason = detail
            # Affinity bonus: prefer an OSD already chosen for another cluster
            # of the same query, amortising the per-tenant codebook load.
            aff = (self.weights.affinity
                   if (preferred_osds and osd_id in preferred_osds) else 0.0)
            eff = score_val + aff
            details[osd_id] = detail
            effective[osd_id] = eff
            # Sort key: (score desc, ddr_hit desc, osd_id asc)
            candidates.append((-eff, -ddr_hit, osd_id))

        if not candidates:
            raise ContractError(
                f"no eligible replica for cluster_id={cluster_id}")

        candidates.sort()
        _, _, chosen_osd = candidates[0]
        chosen_detail = details[chosen_osd]
        score_val = effective[chosen_osd]
        stale_reason = chosen_detail[5]

        reason_parts = [
            f"OSD{chosen_osd}",
            f"score={score_val:.3f}",
            f"ddr_hit={chosen_detail[2]:.1f}",
            f"queue_norm={chosen_detail[3]:.3f}",
            f"fpga_busy={chosen_detail[4]:.3f}",
        ]
        if chosen_osd == rs.primary_osd_id:
            reason_parts.append("primary")
        if stale_reason:
            reason_parts.append(stale_reason)
        reason_str = ", ".join(reason_parts)

        # Estimate finish time for this OSD+cluster (for estimated_finish_ns).
        est_finish = self._estimate_finish(chosen_osd, [cluster_id], now_ns=t)

        return _make_cluster_assignment(
            cluster_id=cluster_id,
            osd_id=chosen_osd,
            score=score_val,
            reason=reason_str,
            estimated_finish_ns=est_finish,
        )

    # ------------------------------------------------------------------
    # Schedule (full query)
    # ------------------------------------------------------------------

    def schedule(self, query: QueryRequest) -> SchedulingDecision:
        """Select optimal OSDs for each cluster in the query.

        Returns a :class:`SchedulingDecision` with one :class:`ClusterAssignment`
        per cluster, in deterministic (cluster_id sorted) order.
        """
        # Process clusters in sorted order for determinism.
        sorted_clusters = sorted(query.cluster_ids)
        assignments: list[ClusterAssignment] = []
        chosen_osds: set[int] = set()
        for cid in sorted_clusters:
            assignment = self.select_osd(cid, preferred_osds=chosen_osds)
            assignments.append(assignment)
            chosen_osds.add(assignment.osd_id)

        # Build reason.
        reasons = [a.reason for a in assignments]
        reason_str = f"query={query.query_id}: " + "; ".join(reasons)

        return SchedulingDecision(
            query_id=query.query_id,
            assignments=tuple(assignments),
            reason=reason_str,
        )

    # ------------------------------------------------------------------
    # Estimated finish time (for SchedulingDecision.estimated_finish_ns)
    # ------------------------------------------------------------------

    def _estimate_finish(
        self, osd_id: int, cluster_ids: list[int], *, now_ns: int,
    ) -> int:
        """Estimate when a set of clusters would finish on one OSD.

        Mirrors the old B4-style estimate: start from max(now, fpga_busy_until),
        add DDR transfers for missing clusters, FPGA processing (if model
        supplied), CPU rerank and queue-depth penalty.
        """
        state = self.state_manager.get_osd_state(osd_id)
        start_time = max(now_ns, state.fpga_busy_until_ns)
        current_time = start_time

        # DDR transfers for missing clusters.
        for cid in cluster_ids:
            if cid not in state.ddr_cached_clusters:
                transfer_bytes = self.vectors_per_cluster * 96  # 96 bytes per PQ code
                transfer_ns = int(transfer_bytes * self.ddr_transfer_ns_per_byte)
                current_time += transfer_ns

        # FPGA processing (use real model when supplied, else skip).
        total_vectors = len(cluster_ids) * self.vectors_per_cluster
        if self.fpga_model is not None:
            fpga_ns = int(self.fpga_model.predict_ns(
                n_docs=total_vectors,
                top_k=10,  # default top_k for estimation
            ))
            current_time += fpga_ns
        else:
            fpga_ns = 0

        # CPU rerank (top_k=10 for estimation).
        rerank_ns = 10 * self.cpu_rerank_ns_per_vector
        current_time += rerank_ns

        # Queue depth penalty.
        queue_penalty = state.queue_depth * fpga_ns
        current_time += queue_penalty

        return current_time


__all__ = [
    "ClusterAssignment",
    "QueryRequest",
    "RGWScheduler",
    "RGWScoreWeights",
    "ReplicaSet",
    "SchedulingDecision",
]
