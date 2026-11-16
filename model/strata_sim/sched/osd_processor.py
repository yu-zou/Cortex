"""Task X: OSD processor with FPGA feedback integration.

Processes queries on OSD with real-time FPGA delay feedback.  The heavy-lifting
is delegated to :class:`strata_sim.sched.osd_timing.OSDTimingModel`; this module
keeps the legacy ``OSDProcessor`` facade so existing callers (state manager,
RGW scheduler demo) continue to work without a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_sim.des.fpga import FpgaServiceModel
from strata_sim.sched.osd_timing import OSDHardwareProfile, OSDTimingModel

if TYPE_CHECKING:
    from strata_sim.des.engine import DesEngine
    from .osd_state_manager import OSDStateManager


@dataclass
class QueryResult:
    """Result of processing one query on OSD."""
    query_id: str
    osd_id: int
    fpga_latency_ns: int
    total_latency_ns: int
    cluster_ids: list[int]
    top_k_results: list[tuple[int, float]]  # (doc_id, distance)


class OSDProcessor:
    """Processes queries on one OSD with FPGA feedback.

    Internally delegates to :class:`OSDTimingModel` which performs the full
    six-stage pipeline on the DES engine's resource calendars.
    """

    def __init__(
        self,
        osd_id: int,
        engine: DesEngine,
        state_manager: OSDStateManager,
        fpga_model: FpgaServiceModel,
        ddr_transfer_ns_per_byte: float = 0.0625,  # legacy; kept for API compat
        cpu_rerank_ns_per_vector: int = 1000,
    ) -> None:
        self.osd_id = osd_id
        self.engine = engine
        self.state_manager = state_manager
        self.fpga_model = fpga_model
        self.ddr_transfer_ns_per_byte = ddr_transfer_ns_per_byte
        self.cpu_rerank_ns_per_vector = cpu_rerank_ns_per_vector
        self.vectors_per_cluster = 64

        # Build the timing model.  Resources are registered lazily on the
        # engine if they don't already exist (we check by name).
        from strata_sim.sched.osd_timing import OSDResourceNames, add_osd_resources

        self._resources = OSDResourceNames.for_osd(osd_id)
        if self._resources.fpga not in engine.resource_names:
            add_osd_resources(engine, osd_id)
        self._timing = OSDTimingModel(
            osd_id=osd_id,
            engine=engine,
            fpga=fpga_model,
            hardware=OSDHardwareProfile(),
            resources=self._resources,
        )

    def process_query(
        self,
        query_id: str,
        cluster_ids: list[int],
        nprobe: int,
        top_k: int,
        now_ns: int = 0,
    ) -> QueryResult:
        """Process one query with FPGA feedback, delegating to the timing model."""
        self.state_manager.query_started(self.osd_id, query_id)

        # Advance engine clock to the caller's ``now_ns`` so acquire() does not
        # reject a start time in the past relative to the engine.
        if now_ns > self.engine.now_ns:
            self.engine.advance_to(now_ns)

        timing = self._timing.simulate_query(
            query_id=query_id,
            cluster_ids=cluster_ids,
            top_k=top_k,
        )

        fpga_latency_ns = int(timing.stage_end_ns.get("fpga", timing.end_ns)
                              - timing.stage_end_ns.get("codebook", timing.start_ns))

        # Propagate the result to the state manager
        for cid in cluster_ids:
            self.state_manager.update_ddr_cache(self.osd_id, cid, True)

        self.state_manager.update(
            osd_id=self.osd_id,
            queue_depth=len(
                self.state_manager.get_osd_state(self.osd_id).pending_queries),
            fpga_busy_until_ns=timing.stage_end_ns.get("fpga", timing.end_ns),
            estimated_finish_ns=timing.end_ns,
        )

        self.state_manager.query_completed(self.osd_id, query_id)

        total_latency_ns = timing.total_latency_ns

        return QueryResult(
            query_id=query_id,
            osd_id=self.osd_id,
            fpga_latency_ns=max(0, fpga_latency_ns),
            total_latency_ns=total_latency_ns,
            cluster_ids=list(cluster_ids),
            top_k_results=[(i, 0.1 * i) for i in range(top_k)],
        )


__all__ = ["QueryResult", "OSDProcessor"]
