"""Per-OSD query timing pipeline.

Each query on an OSD flows through six stages:

- Stage 0 mapping lookup: ~0 ns, no resource.
- Stage 1 SSD->DDR transfer for clusters not already resident in the OSD DDR:
  per missing cluster, an SSD command on ``SSD_COMMAND`` then the cluster data
  across ``PCIE``, plus a DDR write.  The cluster is marked resident afterwards.
- Stage 2 codebook load: if the codebook is not already in the OSD BRAM slot
  set, book ``ceil(codebook_load_cycles * clock_ns)`` on ``DDR`` and add it to
  the BRAM set (respecting ``bram_capacity_slots``, evicting LRU only when safe).
- Stage 3 FPGA compute: ``fpga.predict_ns(n_docs, top_k, codebook_resident=True)``
  on ``FPGA_COMPUTE``.  Stage 2 owns the codebook load, so stage 3 never
  double-counts it.
- Stage 4 CPU rerank: ``top_k * cpu_ns_per_candidate`` on ``CPU``.
- Stage 5 network return: ``ceil(response_bytes / network_bytes_per_ns) +
  network_latency_ns`` on ``NETWORK``.

Every duration derives from the hardware profile or the FPGA model; nothing
calls wall-clock time or randomness.  Two or more OSDs can share one engine by
using distinct ``OSDResourceNames``.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from strata_sim.contracts.schema import TILE_MAX_DOCS, ContractError

# Query read (QRY_READ 769) + distance-table build (DIST_TABLE_BUILD 1,613),
# measured in csynth, overlapping to a fitted 2,211 cycles.  These do not
# depend on the document batch, so a second tile of the SAME cluster can skip
# them -- but the RTL as implemented rebuilds them on every call, so reuse is
# opt-in (OSD_TABLE_REUSE=1) and off by default.
TABLE_BUILD_CYCLES = 2211.0
from strata_sim.des.engine import DesEngine
from strata_sim.des.resources import Reservation, ResourceKind


# ---------------------------------------------------------------------------
# Hardware profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OSDHardwareProfile:
    """Physical constants of one OSD's data path.

    Bandwidths are bytes per nanosecond (GB/s ~= 8 * bytes_per_ns); the
    transfer durations are ``ceil(bytes / bytes_per_ns)``.
    """

    pcie_bytes_per_ns: float = 15.75
    ddr_bytes_per_ns: float = 19.2
    ssd_command_ns: int = field(default_factory=lambda: int(__import__("os").environ.get("OSD_SSD_CMD_NS", "500")))
    cpu_ns_per_candidate: int = 1000
    network_bytes_per_ns: float = 1250.0
    network_latency_ns: int = 0
    pq_code_bytes: int = 96          # per vector
    raw_vector_bytes: int = 3072     # per vector
    codebook_bytes: int = 786432
    vectors_per_cluster: int = 64
    # --- board-calibrated overrides (S25/S27 on-board measurements) ---
    # 实测：SSD->PL DDR4 直读 735 MB/s；acc_top 读 PL DDR4 ~5.5 GB/s；
    # codebook 装载 ~143 us；大 N scan 受 DDR 喂数限制（~4.19 cyc/doc）。
    board_calibrated: bool = field(default_factory=lambda: __import__("os").environ.get("OSD_BOARD_CALIBRATED", "0") == "1")
    board_ddr_read_bytes_per_ns: float = 5.5     # PL DDR4 read (measured)
    ssd_to_ddr_bytes_per_ns: float = 0.735       # SSD->PL DDR4 direct (735 MB/s)
    board_codebook_load_ns: int = 143_000        # measured ~143 us
    board_pq_entry_bytes: int = 128              # padded PQ entry on board
    # Overlap SSD->DDR streaming with the scan (cluster/tile-level double
    # buffering).  The board data path can start scanning cluster i while
    # cluster i+1 is still arriving; without this the model serialises
    # load-then-scan and overstates the query time by the FPGA share.
    board_pipeline_scan: bool = field(default_factory=lambda: __import__('os').environ.get('OSD_PIPELINE', '0') == '1')
    response_bytes_per_result: int = 16
    bram_capacity_slots: int = 4
    # DDR cluster cache capacity.  Clusters are accounted at
    # ``cluster_footprint_bytes()`` and evicted LRU when a load would exceed
    # this; the default 8 GiB is large enough that a typical working set never
    # evicts, which preserves the pre-capacity behaviour unless the caller
    # narrows it.
    ddr_capacity_bytes: int = 8 * 1024 ** 3
    # PQ / raw SEPARATE DDR caches.  Cluster pressure (what the scheduler
    # actually manages) comes from the PQ cache; raw vectors are fetched on
    # demand for rerank candidates and cached separately.
    pq_cache_bytes: int = 6 * 1024 ** 3
    raw_cache_bytes: int = 2 * 1024 ** 3
    # Whether the OSD caches a cluster's raw vectors alongside its PQ codes.
    # The FPGA only reads PQ codes, but rerank needs the raw vectors, so a real
    # OSD keeps both in DDR.  Default False keeps the legacy PQ-only accounting
    # (``cluster_footprint_bytes() == cluster_bytes()``); the Alibaba-fairness
    # experiments turn it on so an 8 GiB DDR budget actually constrains a
    # 20M-vector working set (otherwise PQ alone fits ~15x over).
    cache_raw_vectors: bool = False

    @classmethod
    def from_model(cls, kind: str) -> "OSDHardwareProfile":
        """Build the DES profile from the analytical StrataModel.

        This is the single source of truth for the board-calibrated
        parameters: B_ssd/B_ddr/B_xdma/L_cmd/L_cpu come from
        strata_sim.model.strata_model.profile(kind).
        """
        from strata_sim.model.strata_model import profile as _profile
        m = _profile(kind)
        return cls(
            board_calibrated=True,
            board_ddr_read_bytes_per_ns=float(m.B_ddr),
            ssd_to_ddr_bytes_per_ns=float(m.B_ssd),
            board_codebook_load_ns=143_000,
            board_pq_entry_bytes=int(m.S_entry),
            ddr_bytes_per_ns=float(m.B_ddr),
            ssd_command_ns=int(m.L_cmd),
            cpu_ns_per_candidate=int(m.L_cpu),
            pcie_bytes_per_ns=float(m.B_xdma),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.ddr_capacity_bytes, int) or isinstance(
                self.ddr_capacity_bytes, bool) or self.ddr_capacity_bytes < 1:
            raise ContractError(
                f"ddr_capacity_bytes must be a positive int, got "
                f"{self.ddr_capacity_bytes!r}")
        if not isinstance(self.cache_raw_vectors, bool):
            raise ContractError(
                f"cache_raw_vectors must be a bool, got {self.cache_raw_vectors!r}")
        for name in ("pq_cache_bytes", "raw_cache_bytes"):
            val = getattr(self, name)
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                raise ContractError(
                    f"{name} must be a non-negative int, got {val!r}")

    def entry_bytes(self) -> int:
        """Bytes per PQ entry on the modelled data path.

        The board stores and fetches each entry as 2 x 512-bit beats = 128 B
        (112 B payload + 16 B pad); the RTL model uses the 96 B code size.
        One number everywhere keeps scan, transfer and cache accounting
        consistent (they differ by 33% otherwise).
        """
        return (self.board_pq_entry_bytes if self.board_calibrated
                else self.pq_code_bytes)

    def cluster_bytes(self) -> int:
        """Bytes of one cluster's PQ codes (what the FPGA reads)."""
        return self.entry_bytes() * self.vectors_per_cluster

    def cluster_footprint_bytes(self) -> int:
        """Bytes one cached cluster occupies in DDR.

        With ``cache_raw_vectors`` the OSD holds the whole cluster (PQ codes
        plus the raw vectors rerank needs), so each cached vector costs
        ``pq_code_bytes + raw_vector_bytes``.  Without it only the PQ codes are
        cached and the footprint equals :meth:`cluster_bytes`.
        """
        per_vector = self.entry_bytes() + (
            self.raw_vector_bytes if self.cache_raw_vectors else 0)
        return per_vector * self.vectors_per_cluster


# ---------------------------------------------------------------------------
# Per-OSD engine resource names (so multiple OSDs share one engine)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OSDResourceNames:
    """Per-OSD engine resource names so several OSDs can share one engine."""

    fpga: str
    ddr: str
    pcie: str
    ssd_command: str
    ssd_data: str
    cpu: str
    network: str

    @classmethod
    def for_osd(cls, osd_id: int) -> "OSDResourceNames":
        return cls(
            fpga=f"osd{osd_id}-fpga",
            ddr=f"osd{osd_id}-ddr",
            pcie=f"osd{osd_id}-pcie",
            ssd_command=f"osd{osd_id}-ssd-cmd",
            ssd_data=f"osd{osd_id}-ssd-data",
            cpu=f"osd{osd_id}-cpu",
            network=f"osd{osd_id}-net",
        )


def add_osd_resources(
    engine: DesEngine,
    osd_id: int,
    *,
    ddr_bytes_per_ns: int = 19,
    pcie_bytes_per_ns: int = 15,
    cpu_capacity: int = 4,
    resources: OSDResourceNames | None = None,
) -> OSDResourceNames:
    """Install the per-OSD resources on ``engine``.

    Returns the resource names actually registered.  ``bytes_per_ns`` must be
    positive ints; the helper uses integer truncation of the hardware profile's
    floats so the DES engine's integer-byte accounting stays exact.
    """
    names = OSDResourceNames.for_osd(osd_id) if resources is None else resources
    if not isinstance(ddr_bytes_per_ns, int) or ddr_bytes_per_ns < 1:
        raise ContractError(
            f"ddr_bytes_per_ns must be a positive int, got {ddr_bytes_per_ns!r}")
    if not isinstance(pcie_bytes_per_ns, int) or pcie_bytes_per_ns < 1:
        raise ContractError(
            f"pcie_bytes_per_ns must be a positive int, got {pcie_bytes_per_ns!r}")
    if not isinstance(cpu_capacity, int) or cpu_capacity < 1:
        raise ContractError(
            f"cpu_capacity must be a positive int, got {cpu_capacity!r}")
    engine.add_resource(names.fpga, ResourceKind.FPGA_COMPUTE, capacity=1)
    engine.add_resource(
        names.ddr, ResourceKind.DDR, capacity=1, bytes_per_ns=ddr_bytes_per_ns)
    engine.add_resource(
        names.pcie, ResourceKind.PCIE, capacity=1, bytes_per_ns=pcie_bytes_per_ns)
    engine.add_resource(names.ssd_command, ResourceKind.SSD_COMMAND, capacity=1)
    engine.add_resource(names.ssd_data, ResourceKind.SSD_DATA, capacity=1)
    engine.add_resource(names.cpu, ResourceKind.CPU, capacity=cpu_capacity)
    engine.add_resource(names.network, ResourceKind.NETWORK, capacity=1)
    return names


# ---------------------------------------------------------------------------
# Query timing output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryTiming:
    """Outcome of one simulated query on one OSD."""

    query_id: str
    osd_id: int
    start_ns: int
    end_ns: int
    stage_end_ns: Mapping[str, int]     # keys: map, ssd, codebook, fpga, rerank, net
    codebook_loaded: bool
    codebook_load_ns: int
    ddr_transfers: int
    reservations: tuple[Reservation, ...]
    ddr_evictions: int = 0
    fpga_tiles: int = 0
    raw_transfers: int = 0
    raw_evictions: int = 0

    @property
    def total_latency_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class OSDFeedbackState:
    """Snapshot of an OSD's scheduler-visible state."""

    osd_id: int
    queue_depth: int
    fpga_busy_until_ns: int
    ddr_cached_clusters: frozenset[int]
    bram_codebooks: frozenset[str]
    last_query_end_ns: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ceil_div(num: int, den: float) -> int:
    """Positive integer ceiling of ``num / den`` (den is a float bandwidth)."""
    if den <= 0:
        raise ContractError(f"bandwidth must be positive, got {den!r}")
    if num <= 0:
        return 0
    return int(math.ceil(num / den))


def _book(
    engine: DesEngine,
    resource: str,
    task_id: str,
    duration_ns: int,
    cursor: int,
    *,
    payload_bytes: int = 0,
    booked: list[Reservation],
) -> int:
    """Acquire ``duration_ns`` on ``resource`` from ``cursor``, advance cursor.

    Zero-duration stages are skipped (no reservation, cursor unchanged).  The
    engine may push the start later than ``cursor`` if the resource is busy;
    the returned cursor is the reservation's actual end.
    """
    if duration_ns <= 0:
        return cursor
    reservation = engine.acquire(
        resource,
        task_id,
        duration_ns,
        start_ns=cursor,
        payload_bytes=payload_bytes,
    )
    booked.append(reservation)
    return reservation.end_ns


# ---------------------------------------------------------------------------
# The OSD timing model
# ---------------------------------------------------------------------------


def _slo_weight(slo_ns: "int | None") -> float:
    """Urgency weight for the SLO-aware cache policy (three deadline classes)."""
    if slo_ns is None:
        return 1.0
    if slo_ns <= 37_500_000:
        return 3.0
    if slo_ns <= 100_000_000:
        return 1.5
    return 0.75


class OSDTimingModel:
    """Per-OSD query timing simulator built on a shared ``DesEngine``."""

    def __init__(
        self,
        osd_id: int,
        engine: DesEngine,
        fpga: "FpgaServiceModel",  # noqa: F821 - forward reference for typing
        hardware: OSDHardwareProfile | None = None,
        resources: OSDResourceNames | None = None,
        ddr_cached_clusters: Iterable[int] = (),
        bram_codebooks: Iterable[str] = (),
        cluster_sizes: Mapping[int, int] | None = None,
        cluster_pool: Mapping[int, int] | None = None,
    ) -> None:
        # Optional per-cluster vector counts.  When supplied, the FPGA scan
        # size (and hence the DDR footprint) follows the real cluster's size
        # instead of the uniform vectors_per_cluster; this lets the
        # simulator exercise heterogeneous workloads where the scheduler's
        # cost factor can actually discriminate.
        self._cluster_sizes: Mapping[int, int] | None = (
            dict(cluster_sizes) if cluster_sizes is not None else None)
        self.osd_id = osd_id
        self.engine = engine
        self.fpga = fpga
        self.hardware = hardware if hardware is not None else OSDHardwareProfile()
        self.resources = resources if resources is not None else OSDResourceNames.for_osd(osd_id)
        # --- PQ cache: per cluster, LRU, budget hw.pq_cache_bytes -------------
        self._pq_clusters: OrderedDict[int, int] = OrderedDict()
        for cid in ddr_cached_clusters:
            self._pq_clusters[cid] = self._pq_footprint(cid)
        self._pq_bytes: int = sum(self._pq_clusters.values())
        # --- raw cache: per cluster's rerank candidates, LRU, budget raw_cache --
        self._raw_clusters: OrderedDict[int, int] = OrderedDict()
        self._raw_bytes: int = 0
        # BRAM codebook slots (insertion order for LRU).
        self._bram_codebooks: OrderedDict[str, None] = OrderedDict()
        for cb in bram_codebooks:
            self._bram_codebooks[cb] = None
        self._queue_depth: int = 0
        self._last_query_end_ns: int = 0
        # Eviction policy (OSD_EVICT_POLICY): lru | keep_large | keep_small | lfu | freq_size
        self._evict_policy: str = os.environ.get("OSD_EVICT_POLICY", "lru")
        self._access_counts: dict[int, int] = {}
        self._urgency_score: dict[int, float] = {}
        self._last_access_ns: dict[int, int] = {}
        self._pool_split: float = float(os.environ.get("OSD_POOL_SPLIT", "0.5"))
        self._cluster_pool: dict[int, int] | None = (
            dict(cluster_pool) if cluster_pool is not None else None)
        self._prefetch_enabled: bool = (
            os.environ.get("OSD_PREFETCH", "0") == "1")
        self.prefetch_transfers: int = 0
        self.prefetch_skipped: int = 0
        self.prefetch_skip_dumps: list[str] = []
        self._prefetch_overrun_ns: int = int(
            os.environ.get("OSD_PREFETCH_OVERFLOW_NS", "1500000"))


    # ------------------------------------------------------------------
    # Public state accessors
    # ------------------------------------------------------------------

    def ddr_resident(self) -> frozenset[int]:
        """PQ-resident clusters: this is what the scheduler's ddr_hit sees."""
        return frozenset(self._pq_clusters)

    def raw_resident(self) -> frozenset[int]:
        return frozenset(self._raw_clusters)

    def cache_stats(self) -> dict[str, int]:
        return {
            "pq_clusters": len(self._pq_clusters),
            "pq_bytes": self._pq_bytes,
            "raw_clusters": len(self._raw_clusters),
            "raw_bytes": self._raw_bytes,
        }

    def bram_resident(self) -> frozenset[str]:
        return frozenset(self._bram_codebooks)

    def feedback_state(self, *, at_ns: int | None = None) -> OSDFeedbackState:
        fpga_resource = self.engine.resource(self.resources.fpga)
        return OSDFeedbackState(
            osd_id=self.osd_id,
            queue_depth=self._queue_depth,
            fpga_busy_until_ns=fpga_resource.max_end_ns(),
            ddr_cached_clusters=frozenset(self._pq_clusters),
            bram_codebooks=frozenset(self._bram_codebooks),
            last_query_end_ns=self._last_query_end_ns,
        )

    def _cluster_vectors(self, cid: int) -> int:
        """REAL vector count of a cluster when known, else the profile default."""
        hw = self.hardware
        return (int(self._cluster_sizes.get(cid, hw.vectors_per_cluster))
                if self._cluster_sizes else hw.vectors_per_cluster)

    def _pq_footprint(self, cid: int) -> int:
        """DDR bytes one cluster's PQ codes occupy (what the FPGA scans)."""
        return self.hardware.entry_bytes() * max(0, self._cluster_vectors(cid))

    def _raw_footprint(self, cid: int, rerank_per_cluster: int) -> int:
        """DDR bytes of the raw vectors needed to rerank this cluster's
        candidates: ``r`` candidates per probed cluster, ``raw_vector_bytes``
        each.  The candidates themselves are sparse inside the cluster, so this
        is far smaller than caching the whole cluster's raw vectors."""
        return self.hardware.raw_vector_bytes * max(0, int(rerank_per_cluster))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def simulate_query(
        self,
        query_id: str,
        cluster_ids: Sequence[int],
        top_k: int,
        *,
        codebook_id: str | None = None,
        tenant: str = "t0",
        start_ns: int | None = None,
        slo_ns: int | None = None,
        prefetch_cluster: int | None = None,
        prefetch_candidates: Sequence[tuple[int, float]] | None = None,
        rerank_per_cluster: int = 0,
    ) -> QueryTiming:
        """Run the six stages, booking each on the engine's resource calendars.

        Raises :class:`ContractError` on negative cluster ids, non-positive
        ``top_k``, empty ``cluster_ids`` or empty ``query_id``.  Duplicate
        cluster ids are de-duplicated silently.
        """
        # --- input validation -------------------------------------------------
        if not isinstance(query_id, str) or not query_id:
            raise ContractError(f"query_id must be a non-empty string, got {query_id!r}")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ContractError(f"top_k must be a positive int, got {top_k!r}")
        if not isinstance(cluster_ids, (list, tuple)) or len(cluster_ids) == 0:
            raise ContractError("cluster_ids must be a non-empty sequence")
        deduped: list[int] = []
        seen: set[int] = set()
        for cid in cluster_ids:
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ContractError(
                    f"cluster ids must be ints, got {type(cid).__name__}")
            if cid < 0:
                raise ContractError(
                    f"cluster ids must be non-negative, got {cid}")
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        cluster_ids_unique: list[int] = deduped

        hw = self.hardware
        res = self.resources
        task_id = f"osd{self.osd_id}:{query_id}"
        booked: list[Reservation] = []
        stage_end: dict[str, int] = {}

        cursor = self.engine.now_ns if start_ns is None else int(start_ns)
        query_start_ns = cursor
        self._queue_depth += 1

        # --- Stage 0: mapping lookup (0 ns, no resource) ----------------------
        stage_end["map"] = cursor

        # --- Stage 1: SSD -> DDR for missing clusters (PQ codes only) ---------
        # The PQ cache is the scarce resource: it is what the scheduler's
        # ddr_hit factor tracks, and what forces SSD reads under LRU pressure.
        for cid in cluster_ids_unique:
            if cid in self._pq_clusters:
                self._pq_clusters.move_to_end(cid)
        for cid in cluster_ids_unique:
            self._access_counts[cid] = self._access_counts.get(cid, 0) + 1
        if self._evict_policy == "slo_aware":
            _w = _slo_weight(slo_ns)
            for cid in cluster_ids_unique:
                self._urgency_score[cid] = self._urgency_score.get(cid, 0.0) + _w
        if self._evict_policy == "cost_aware":
            for cid in cluster_ids_unique:
                self._last_access_ns[cid] = query_start_ns
        missing = [cid for cid in cluster_ids_unique if cid not in self._pq_clusters]
        ddr_transfer_count = 0
        ddr_evictions = 0
        # A cluster this query needs is never evicted: the query's own working
        # set is protected while it loads the rest of its clusters.
        protected = set(cluster_ids_unique)
        _first_ready = None
        for cid in missing:
            pq_bytes = self._pq_footprint(cid)
            # SSD command
            cursor = _book(
                self.engine, res.ssd_command, task_id, hw.ssd_command_ns, cursor,
                booked=booked)
            # PCIe read of the PQ codes (board-calibrated: SSD->PL DDR4 direct BW)
            _xfer_bw = (hw.ssd_to_ddr_bytes_per_ns if hw.board_calibrated
                        else hw.pcie_bytes_per_ns)
            pcie_ns = _ceil_div(pq_bytes, _xfer_bw)
            cursor = _book(
                self.engine, res.pcie, task_id, pcie_ns, cursor,
                payload_bytes=pq_bytes, booked=booked)
            # NOTE: no separate DDR-write term.  ``ssd_to_ddr_bytes_per_ns``
            # is the measured end-to-end SSD->PL DDR4 rate, so C/B_ssd already
            # contains the write; charging C/B_ddr again would double count.
            self._pq_clusters[cid] = pq_bytes
            self._pq_bytes += pq_bytes
            ddr_transfer_count += 1
            if _first_ready is None:
                _first_ready = cursor
        # Eviction policy (OSD_EVICT_POLICY): lru | keep_large | keep_small | lfu
        #   | freq_size | slo_aware | cost_aware | pool_aware
        if self._evict_policy == "pool_aware" and self._cluster_pool is not None:
            _b0 = int(hw.pq_cache_bytes * self._pool_split)
            _b1 = hw.pq_cache_bytes - _b0
            while True:
                _u0 = sum(sz for c, sz in self._pq_clusters.items()
                          if self._cluster_pool.get(c, 0) == 0)
                _u1 = self._pq_bytes - _u0
                _over_g = self._pq_bytes > hw.pq_cache_bytes
                _over0 = _u0 > _b0
                _over1 = _u1 > _b1
                if not (_over_g or _over0 or _over1):
                    break
                _c0 = [c for c in self._pq_clusters
                       if c not in protected
                       and self._cluster_pool.get(c, 0) == 0]
                _c1 = [c for c in self._pq_clusters
                       if c not in protected
                       and self._cluster_pool.get(c, 0) == 1]
                if _over0 and (not _over1 or (_u0 - _b0) >= (_u1 - _b1)) and _c0:
                    victim = _c0[0]
                elif _over1 and _c1:
                    victim = _c1[0]
                elif _over_g:
                    _ev = [c for c in self._pq_clusters if c not in protected]
                    if not _ev:
                        break
                    victim = _ev[0]
                else:
                    break
                self._pq_bytes -= self._pq_clusters[victim]
                del self._pq_clusters[victim]
                ddr_evictions += 1
        else:
            while (self._pq_bytes > hw.pq_cache_bytes
                   and len(self._pq_clusters) > len(protected)):
                evictable = [c for c in self._pq_clusters if c not in protected]
                if not evictable:
                    break
                if self._evict_policy == "keep_large":
                    victim = min(evictable, key=lambda c: self._pq_clusters[c])
                elif self._evict_policy == "keep_small":
                    victim = max(evictable, key=lambda c: self._pq_clusters[c])
                elif self._evict_policy == "lfu":
                    victim = min(evictable, key=lambda c: (self._access_counts.get(c, 0), self._pq_clusters[c]))
                elif self._evict_policy == "freq_size":
                    victim = min(evictable, key=lambda c: (self._access_counts.get(c, 0) + 1) / max(1, self._pq_clusters[c]))
                elif self._evict_policy == "slo_aware":
                    victim = min(evictable, key=lambda c: (self._urgency_score.get(c, 0.0), self._pq_clusters[c]))
                elif self._evict_policy == "cost_aware":
                    victim = max(evictable, key=lambda c: (query_start_ns - self._last_access_ns.get(c, 0)) / max(1, self._pq_clusters[c]))
                else:
                    victim = evictable[0]
                self._pq_bytes -= self._pq_clusters[victim]
                del self._pq_clusters[victim]
                ddr_evictions += 1
        stage_end["ssd"] = cursor

        # --- Stage 2: codebook load (only if not in BRAM) ---------------------
        codebook_key = codebook_id if codebook_id is not None else f"{tenant}:default"
        codebook_loaded_this_query = False
        codebook_load_ns = 0
        if codebook_key not in self._bram_codebooks:
            clock_ns = float(getattr(self.fpga, "clock_ns", 1.0))
            if hw.board_calibrated:
                load_duration = int(hw.board_codebook_load_ns)
            else:
                load_cycles = float(getattr(self.fpga, "codebook_load_cycles", 0.0))
                load_duration = int(math.ceil(load_cycles * clock_ns))
            if load_duration > 0:
                cursor = _book(
                    self.engine, res.ddr, task_id, load_duration, cursor,
                    payload_bytes=hw.codebook_bytes, booked=booked)
                codebook_load_ns = load_duration
                codebook_loaded_this_query = True
            # Admit to BRAM (evict LRU if at capacity)
            if codebook_key in self._bram_codebooks:
                # refresh LRU
                self._bram_codebooks.move_to_end(codebook_key)
            else:
                if len(self._bram_codebooks) >= hw.bram_capacity_slots:
                    # evict the least-recently-used
                    self._bram_codebooks.popitem(last=False)
                self._bram_codebooks[codebook_key] = None
        else:
            # refresh LRU
            self._bram_codebooks.move_to_end(codebook_key)
        stage_end["codebook"] = cursor

        # --- Stage 3: FPGA compute (always warm: stage 2 owns the load) ------
        # The accelerator accepts at most TILE_MAX_DOCS documents per call
        # (schema.TileRef refuses a larger tile: "split first"), so a scan
        # larger than one tile is split into tiles and each tile pays its own
        # pipeline fixed base + TOPK init.  A whole cluster of a 9M/4096 index
        # is 2159 docs, so nprobe=4 already needs two tiles.
        if self._cluster_sizes:
            n_docs = sum(
                int(self._cluster_sizes.get(cid, hw.vectors_per_cluster))
                for cid in cluster_ids_unique)
        else:
            n_docs = len(cluster_ids_unique) * hw.vectors_per_cluster
        fpga_tiles = 0
        fpga_cycles = 0.0
        remaining = n_docs
        _reuse_table = os.environ.get("OSD_TABLE_REUSE", "0") == "1"
        while remaining > 0:
            tile_docs = min(remaining, TILE_MAX_DOCS)
            _compute_cyc = self.fpga.predict_cycles(
                n_docs=tile_docs, top_k=top_k, codebook_resident=True)
            if hw.board_calibrated:
                # Memory-aware bound: the fixed pipeline base is paid whether or
                # not the tile is DDR-feed-bound, so the max() applies to the
                # steady-state term only: T = base + max(proc*N, feed*N).
                _clk_ns = float(getattr(self.fpga, "clock_ns", 5.556))
                _mem_ns = (hw.board_pq_entry_bytes * tile_docs) / hw.board_ddr_read_bytes_per_ns
                _mem_cyc = _mem_ns / _clk_ns
                _proc_cyc = float(self.fpga.scan_cycles_per_doc(tile_docs))
                _compute_cyc = max(
                    _compute_cyc,
                    _compute_cyc - _proc_cyc * tile_docs + _mem_cyc)
            if _reuse_table and fpga_tiles > 0:
                # Query read + distance table are per (query, codebook), not
                # per tile: only the first tile of a cluster pays them.
                _compute_cyc = max(1.0, _compute_cyc - TABLE_BUILD_CYCLES)
            fpga_cycles += _compute_cyc
            remaining -= tile_docs
            fpga_tiles += 1
        if fpga_tiles == 0:  # empty scan still costs one pipeline setup
            fpga_cycles = self.fpga.predict_cycles(
                n_docs=0, top_k=top_k, codebook_resident=True)
            fpga_tiles = 1
        fpga_ns = int(math.ceil(
            fpga_cycles * float(getattr(self.fpga, "clock_ns", 1.0))))
        _pipelined = (hw.board_calibrated and hw.board_pipeline_scan
                      and _first_ready is not None)
        _compute_start = _first_ready if _pipelined else cursor
        _fpga_cursor = _book(
            self.engine, res.fpga, task_id, max(1, fpga_ns), _compute_start,
            booked=booked)
        if _fpga_cursor > cursor:
            cursor = _fpga_cursor
        stage_end["fpga"] = cursor
        # --- Compute-shadow staging (window-wide) -----------------------------
        # Candidates come from the scheduling window, i.e. any later query, not
        # just the immediate next one.  We greedily admit by value density
        # (caller weight per ns of SSD time) while the transfer still fits in
        # this query's compute window plus the overrun guard.
        _cands: list[tuple[int, float]] = []
        if self._prefetch_enabled:
            if prefetch_candidates:
                _cands = [(int(c), float(w)) for c, w in prefetch_candidates]
            elif prefetch_cluster is not None:
                _cands = [(int(prefetch_cluster), 1.0)]
        if _cands:
            _pf_bw = (hw.ssd_to_ddr_bytes_per_ns if hw.board_calibrated
                      else hw.pcie_bytes_per_ns)
            _scored: list[tuple[float, int, int]] = []
            for _cid, _w in _cands:
                if _cid in self._pq_clusters or _cid in protected:
                    continue
                _b = self._pq_footprint(_cid)
                _t = hw.ssd_command_ns + _ceil_div(_b, _pf_bw)
                if _t <= 0 or _w <= 0.0:
                    continue
                _scored.append((_w / float(_t), _cid, _b))
            _scored.sort(key=lambda x: (-x[0], x[1]))
            # Bound the work per query: evaluate the best few candidates and
            # admit at most a handful.  An OSD cannot usefully stage every
            # cluster of every window entry during one scan anyway.
            _scored = _scored[:int(os.environ.get("OSD_PREFETCH_MAX_EVAL", "16"))]
            _admitted = 0
            _maxpf = int(os.environ.get("OSD_PREFETCH_MAX_PER_QUERY", "4"))
            for _density, _cid, pf_bytes in _scored:
                if _admitted >= _maxpf:
                    break
                pf_cmd = hw.ssd_command_ns
                pf_pcie = _ceil_div(pf_bytes, _pf_bw)
                pf_ddr = _ceil_div(pf_bytes, hw.ddr_bytes_per_ns)
                pf_start = max(
                    _compute_start,
                    self.engine.resource(res.ssd_command).max_end_ns(),
                    self.engine.resource(res.pcie).max_end_ns(),
                    self.engine.resource(res.ddr).max_end_ns(),
                )
                if (pf_start + pf_cmd + pf_pcie + pf_ddr
                        > cursor + self._prefetch_overrun_ns):
                    self.prefetch_skipped += 1
                    continue
                pfc = pf_start
                pfc = _book(
                    self.engine, res.ssd_command, task_id + ":pf", pf_cmd,
                    pfc, booked=booked)
                pfc = _book(
                    self.engine, res.pcie, task_id + ":pf", pf_pcie, pfc,
                    payload_bytes=pf_bytes, booked=booked)
                pfc = _book(
                    self.engine, res.ddr, task_id + ":pf", pf_ddr, pfc,
                    payload_bytes=pf_bytes, booked=booked)
                # PL DDR4 is one bus: a staged write steals read bandwidth from
                # the concurrent scan one-for-one.  Charge the scan for the
                # bytes the write moves (RTL model declines this charge, where
                # writes sat on a separate 19.2 GB/s calendar).
                if hw.board_calibrated:
                    _steal = _ceil_div(
                        pf_bytes, hw.board_ddr_read_bytes_per_ns)
                    cursor = _book(
                        self.engine, res.fpga, task_id + ":pfsteal", _steal,
                        cursor, booked=booked)
                    stage_end["fpga"] = cursor
                self._pq_clusters[_cid] = pf_bytes
                self._pq_bytes += pf_bytes
                self.prefetch_transfers += 1
                _admitted += 1

        # --- Stage 3.5: raw vectors for rerank candidates (ON DEMAND) ---------
        # Only after the FPGA produces the candidate list do we know which raw
        # vectors are needed; fetch them cluster by cluster into the separate
        # raw cache.  A hit costs nothing; a miss reads the candidate bytes (r per
        # cluster) from the board SSD through the same path as stage 1.
        raw_transfers = 0
        raw_evictions = 0
        if rerank_per_cluster > 0:
            raw_bytes = self._raw_footprint(cluster_ids_unique[0], rerank_per_cluster)
            for cid in cluster_ids_unique:
                if cid in self._raw_clusters:
                    self._raw_clusters.move_to_end(cid)
                    continue
                cursor = _book(
                    self.engine, res.ssd_command, task_id, hw.ssd_command_ns,
                    cursor, booked=booked)
                # Same SSD->DDR path as stage 1: raw vectors live on the
                # board SSD, so a miss pays the SSD link, not just PCIe.
                _raw_bw = (hw.ssd_to_ddr_bytes_per_ns if hw.board_calibrated
                           else hw.pcie_bytes_per_ns)
                pcie_ns = _ceil_div(raw_bytes, _raw_bw)
                cursor = _book(
                    self.engine, res.pcie, task_id, pcie_ns, cursor,
                    payload_bytes=raw_bytes, booked=booked)
                # Same end-to-end rate as stage 1: the write is inside it.
                self._raw_clusters[cid] = raw_bytes
                self._raw_bytes += raw_bytes
                raw_transfers += 1
            while (self._raw_bytes > hw.raw_cache_bytes
                   and len(self._raw_clusters) > len(protected)):
                victim = next(
                    (c for c in self._raw_clusters if c not in protected), None)
                if victim is None:
                    break
                self._raw_bytes -= self._raw_clusters[victim]
                del self._raw_clusters[victim]
                raw_evictions += 1
        stage_end["raw"] = cursor

        # --- Stage 4: CPU rerank ----------------------------------------------
        # Rerank scores every candidate the FPGA produced (r per probed
        # cluster), so the CPU cost scales with the candidate list, not top_k.
        n_rerank = max(top_k, len(cluster_ids_unique) * max(0, int(rerank_per_cluster)))
        rerank_ns = n_rerank * hw.cpu_ns_per_candidate
        cursor = _book(
            self.engine, res.cpu, task_id, rerank_ns, cursor, booked=booked)
        stage_end["rerank"] = cursor

        # --- Stage 5: network return ------------------------------------------
        response_bytes = top_k * hw.response_bytes_per_result
        net_ns = _ceil_div(response_bytes, hw.network_bytes_per_ns)
        total_net_ns = net_ns + hw.network_latency_ns
        cursor = _book(
            self.engine, res.network, task_id, max(1, total_net_ns), cursor,
            payload_bytes=response_bytes, booked=booked)
        stage_end["net"] = cursor

        self._queue_depth = max(0, self._queue_depth - 1)
        self._last_query_end_ns = cursor

        return QueryTiming(
            query_id=query_id,
            osd_id=self.osd_id,
            start_ns=query_start_ns,
            end_ns=cursor,
            stage_end_ns=dict(stage_end),
            codebook_loaded=codebook_loaded_this_query,
            codebook_load_ns=codebook_load_ns,
            ddr_transfers=ddr_transfer_count,
            reservations=tuple(booked),
            ddr_evictions=ddr_evictions,
            fpga_tiles=fpga_tiles,
            raw_transfers=raw_transfers,
            raw_evictions=raw_evictions,
        )


__all__ = [
    "OSDFeedbackState",
    "OSDHardwareProfile",
    "OSDResourceNames",
    "OSDTimingModel",
    "QueryTiming",
    "add_osd_resources",
]
