"""Task X: OSD state manager for feedback-aware scheduling.

Maintains real-time state of each OSD: FPGA load, DDR cache residency,
queue depth, and estimated completion times. Provides query interface for
RGW scheduler and update interface for OSD processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from collections import defaultdict

from strata_sim.contracts.schema import ClusterKey


@dataclass
class OSDState:
    """Real-time state of one OSD.

    ``last_update_ns`` tracks the most recent time (in the DES clock) at which
    the state was refreshed via a heartbeat or update call.  Used by the RGW
    scheduler's staleness check (RGW_OSD_调度策略定稿.md §3.3): when
    ``now_ns - last_update_ns > ttl_ns`` the DDR/FPGA fields are untrusted.
    """
    osd_id: int
    queue_depth: int = 0
    estimated_finish_ns: int = 0
    fpga_busy_until_ns: int = 0
    ddr_cached_clusters: set[int] = field(default_factory=set)
    pending_queries: list[str] = field(default_factory=list)
    last_update_ns: int = 0

    def is_ddr_cached(self, cluster_id: int) -> bool:
        return cluster_id in self.ddr_cached_clusters

    def add_to_ddr_cache(self, cluster_id: int) -> None:
        self.ddr_cached_clusters.add(cluster_id)

    def remove_from_ddr_cache(self, cluster_id: int) -> None:
        self.ddr_cached_clusters.discard(cluster_id)

    def is_stale(self, now_ns: int, ttl_ns: int) -> bool:
        """RGW_OSD_调度策略定稿.md §3.3: a state older than ``ttl_ns`` is stale.

        A never-updated state (``last_update_ns == 0``) is stale whenever
        ``now_ns > 0`` because we have no evidence it is current.
        """
        age = now_ns - self.last_update_ns
        return age > ttl_ns

    def to_dict(self) -> dict:
        return {
            "osd_id": self.osd_id,
            "queue_depth": self.queue_depth,
            "estimated_finish_ns": self.estimated_finish_ns,
            "fpga_busy_until_ns": self.fpga_busy_until_ns,
            "ddr_cached_clusters": sorted(self.ddr_cached_clusters),
            "pending_queries": self.pending_queries[:10],  # limit output
            "last_update_ns": self.last_update_ns,
        }


class OSDStateManager:
    """Global state manager for all OSDs."""
    
    def __init__(self, now_ns: int = 0):
        self.now_ns = now_ns
        self._states: dict[int, OSDState] = {}
        self._query_to_osd: dict[str, int] = {}
    
    def get_osd_state(self, osd_id: int) -> OSDState:
        """Get state of one OSD."""
        if osd_id not in self._states:
            self._states[osd_id] = OSDState(osd_id=osd_id)
        return self._states[osd_id]
    
    def get_all_osd_states(self) -> Mapping[int, OSDState]:
        """Get states of all OSDs."""
        return self._states
    
    def update(self, osd_id: int, *, queue_depth: int | None = None,
               estimated_finish_ns: int | None = None,
               fpga_busy_until_ns: int | None = None) -> None:
        """Update OSD state after query processing.

        Stamps ``last_update_ns`` so the RGW scheduler's TTL check
        (RGW_OSD_调度策略定稿.md §3.3) knows when this state was last refreshed.
        """
        state = self.get_osd_state(osd_id)
        if queue_depth is not None:
            state.queue_depth = queue_depth
        if estimated_finish_ns is not None:
            state.estimated_finish_ns = estimated_finish_ns
        if fpga_busy_until_ns is not None:
            state.fpga_busy_until_ns = fpga_busy_until_ns
        state.last_update_ns = self.now_ns
    
    def query_started(self, osd_id: int, query_id: str) -> None:
        """Mark query as started on OSD."""
        state = self.get_osd_state(osd_id)
        state.queue_depth += 1
        state.pending_queries.append(query_id)
        self._query_to_osd[query_id] = osd_id
    
    def query_completed(self, osd_id: int, query_id: str) -> None:
        """Mark query as completed on OSD."""
        state = self.get_osd_state(osd_id)
        state.queue_depth = max(0, state.queue_depth - 1)
        if query_id in state.pending_queries:
            state.pending_queries.remove(query_id)
        if query_id in self._query_to_osd:
            del self._query_to_osd[query_id]
    
    def update_ddr_cache(self, osd_id: int, cluster_id: int, cached: bool) -> None:
        """Update DDR cache residency."""
        state = self.get_osd_state(osd_id)
        if cached:
            state.add_to_ddr_cache(cluster_id)
        else:
            state.remove_from_ddr_cache(cluster_id)
    
    def advance_time(self, now_ns: int) -> None:
        """Advance global clock."""
        self.now_ns = now_ns

    def mark_heartbeat(self, osd_id: int, *,
                       queue_depth: int | None = None,
                       fpga_busy_until_ns: int | None = None,
                       ddr_cached_clusters: set[int] | None = None) -> None:
        """Process an OSD heartbeat (RGW_OSD_调度策略定稿.md §3.3).

        Heartbeats arrive every ~100 ms carrying queue_depth,
        fpga_busy_until_ns and ddr_cached_clusters.  This updates the state
        and stamps ``last_update_ns`` so the TTL staleness check works.
        """
        state = self.get_osd_state(osd_id)
        if queue_depth is not None:
            state.queue_depth = queue_depth
        if fpga_busy_until_ns is not None:
            state.fpga_busy_until_ns = fpga_busy_until_ns
        if ddr_cached_clusters is not None:
            state.ddr_cached_clusters = set(ddr_cached_clusters)
        state.last_update_ns = self.now_ns


__all__ = ["OSDState", "OSDStateManager"]
