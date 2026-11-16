"""Strata analytical model: formulas + pluggable parameters.

Everything is in nanoseconds; bandwidths are bytes/ns (1 GB/s == 1 byte/ns).

Topology (confirmed with the user):
  - one board (1 FPGA + 2 striped SSDs) = one OSD
  - 4 boards per workstation (4 independent PCIe x16 slots)
  - 5 workstations = 20 OSDs (cloud experiment / paper system level)
  - local board: 2 physical SSDs treated as one striped logical volume
  - rerank runs on the workstation host CPU, shared by the 4 local OSDs
"""
from __future__ import annotations

from dataclasses import dataclass, replace


def pcie_gen3_gbps(lanes: int) -> float:
    """One direction, encoding only; excludes TLP/DLLP overhead."""
    return 8.0 * lanes * (128.0 / 130.0) / 8.0


IDEAL_DDR_GBPS = 1600.0 * 64 / 8 / 1000  # shared DIMM data bus
IDEAL_SSD_GBPS = 2 * pcie_gen3_gbps(4)  # assumes two simultaneous reads
IDEAL_HOST_GBPS = pcie_gen3_gbps(16)


@dataclass(frozen=True)
class StrataModel:
    # ---- topology -----------------------------------------------------
    osds: int = 20
    boards_per_host: int = 4
    ssds_per_osd: int = 2
    cores_per_host: int = 8

    # ---- measured / fixed layout -------------------------------------
    S_entry: int = 128            # bytes per PQ entry (2 x 512-bit beats)
    header_bytes: int = 64        # cluster header
    raw_vector_bytes: int = 3072  # 768 x FP32
    c_proc: float = 2.0           # compute cycles/doc (RTL)
    T_build_cycles: int = 1480    # table build (RTL A/B)
    f_clk_mhz: float = 180.0
    m_candidates: int = 64        # candidates per cluster for rerank
    probes: int = 32
    top_k: int = 8

    # ---- data-path parameters (fill per profile) ---------------------
    B_ssd: float = IDEAL_SSD_GBPS  # two independent Gen3 x4 links
    B_ddr: float = IDEAL_DDR_GBPS  # two DIMMs share one data bus
    B_xdma: float = IDEAL_HOST_GBPS  # Gen3 x16, encoding-only ceiling
    L_cmd: float = 10_000.0       # NVMe command latency (ns)
    L_pcie: float = 1_000.0       # PCIe round trip (ns)
    L_cpu: float = 50.0           # ns per rerank candidate (single core)
    T_base: float = 15_000.0      # fixed scan base (fitted from N-sweep)

    # ---- derived ------------------------------------------------------
    @property
    def cores_per_osd(self) -> float:
        return max(1.0, self.cores_per_host / self.boards_per_host)

    def cluster_bytes(self, n_docs: int) -> int:
        return self.header_bytes + n_docs * self.S_entry

    def T_load(self, n_docs: int, n_miss: int = 1) -> float:
        c = self.cluster_bytes(n_docs)
        return n_miss * (self.L_cmd + c / self.B_ssd)

    def T_scan(self, n_docs: int) -> float:
        t_compute = self.c_proc * n_docs / self.f_clk_mhz * 1.0e3  # cycles -> ns
        t_memory = self.S_entry * n_docs / self.B_ddr
        return self.T_base + max(t_compute, t_memory)

    def T_raw_read(self, n_r: int) -> float:
        return n_r * self.raw_vector_bytes / self.B_xdma + self.L_pcie

    def T_rerank(self, n_r: int) -> float:
        return n_r * self.L_cpu / self.cores_per_osd

    def T_return(self) -> float:
        return 16.0 * self.top_k / self.B_xdma + self.L_pcie

    def T_subquery_serial(self, n_docs: int, n_miss: int = 1) -> float:
        n_r = self.m_candidates
        return (self.T_load(n_docs, n_miss) + self.T_scan(n_docs)
                + self.T_raw_read(n_r) + self.T_rerank(n_r) + self.T_return())

    def T_query(self, subquery_latencies_ns) -> float:
        """Client query = max over its sub-queries (scatter-gather barrier)."""
        return max(subquery_latencies_ns)


def profile(kind: str) -> StrataModel:
    base = StrataModel()
    if kind == "ideal":
        return replace(base, B_ssd=IDEAL_SSD_GBPS, B_ddr=IDEAL_DDR_GBPS,
                       B_xdma=IDEAL_HOST_GBPS,
                       L_cmd=10_000.0, L_pcie=1_000.0, L_cpu=50.0,
                       T_base=12_000.0)
    if kind == "estimated":
        return replace(base, B_ssd=6.4, B_ddr=5.5, B_xdma=20.0,
                       L_cmd=20_000.0, L_pcie=2_000.0, L_cpu=100.0,
                       T_base=15_000.0)
    if kind == "measured":
        # 0.735 GB/s IS a measurement of the SSD -> PL DDR4 path, but it was
        # taken with an unoptimized transfer design (single XDMA channel,
        # short bursts, an async CDC), so it is the *current implementation*,
        # not the verified BD's encoding-only ceiling (2x Gen3 x4).
        # Do not silently treat it as the target: define an explicit profile
        # if a "current implementation" run is intended.
        raise ValueError(
            "measured profile is not defined: 0.735 GB/s is the current "
            "(unoptimized) SSD->PL DDR4 throughput, not the link ceiling; "
            "the estimated profile is the design target")
    raise ValueError(kind)
