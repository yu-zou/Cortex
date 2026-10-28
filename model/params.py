"""Single source of truth for every numeric constant the model uses.

Two groups live here and they must not be mixed:

  Hardware constants  -- physical properties of the RTL design. They are
      immutable for a given bitstream and are ALSO needed by the C side
      (HLS kernel, host driver). This module can emit them as a C header
      so the two sides cannot drift.

  Model parameters    -- bandwidths / latencies / cache sizes the DES uses
      to compose service times. These are measurement results, design
      targets, or explicit assumptions, and every field below records
      which one it is. They never go in the C header: nothing on the
      device needs to know that we assume a 20 us NVMe command latency.

Run this module directly to regenerate the header:

    python -m model.params --emit-h > include/cortex_model_params.h

Do not hand-edit the header; edit the dataclass below and re-emit.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import Tuple

# How each field was obtained. Kept as a literal marker in the docstring of
# each dataclass so a reader can tell hard numbers from guesses at a glance.
SOURCE_MEASURED = "measured"      # taken off the board / from an RTL sim
SOURCE_FITTED = "fitted"          # regression against measured points
SOURCE_DESIGN = "design"          # a constant of the RTL / Ceph design
SOURCE_ASSUMED = "assumed"        # not yet measured; best current guess
SOURCE_TARGET = "target"          # a number we intend to reach, not at today


@dataclass(frozen=True)
class Hardware:
    """Constants that are a property of the RTL design itself.

    These are the only values that get emitted into the C header. Each one
    has a traceable origin in the HLS sources:
        M, KS, DS      hls/table-planb/include/acc_top.h
        TOPK_MAX       hls/.../acc_top.h  (= 500 systolic top-k cells)
        TILE_MAX_N     payload[8192] + ap_uint<13> idx[] in the BD kernel
                       (/home/chenming/acc_top_vivado_test/src/acc_top.cpp)
        ENTRY_BYTES    128 B on the board (96 B PQ code + 16 B metadata
                       + 16 B pad), measured as 2 x 512-bit beats
    """

    M: int = 96               # subquantizers; PQ code length == 96 B
    DIM: int = 768            # vector dimensionality of the dataset
    DS: int = 8               # dims per subquantizer == DIM / M
    KS: int = 256             # codewords per subquantizer (8-bit codes)
    TOPK_MAX: int = 500       # on-chip top-k cells, one per rank
    TILE_MAX_N: int = 8192    # docs per FPGA invocation (BRAM payload depth)
    ENTRY_BYTES: int = 128    # bytes per PQ entry as fetched from DDR
    RESULT_BYTES: int = 16    # bytes per returned (distance, id) pair
    F_CLK_MHZ: float = 180.0  # kernel clock after timing closure

    @property
    def codebook_bytes(self) -> int:
        """PQ codebook resident in BRAM: M x KS x DS x 4 B (fp32 centroids).

        96 subquantizers x 256 codewords x 8 dims x 4 B = 786,432 B. This is
        what the codebook stream phase (12,299 cycles at 180 MHz) moves into
        BRAM on a cold call, and what the 143 us board-level figure divides
        by 5.5 GB/s.
        """
        return self.M * self.KS * self.DS * 4


@dataclass(frozen=True)
class ModelParams:
    """Everything the DES needs that is NOT a property of the RTL.

    Units: bandwidths in bytes/ns (1 GB/s == 1 byte/ns), times in ns.
    """

    # ---- data layout (measured on the board) -------------------------
    S_entry: int = 128            # bytes per PQ entry (2 x 512-bit beats)
    header_bytes: int = 64        # per-cluster header prepended to the entries
    raw_vector_bytes: int = 3072  # 768 dims x FP32, read for exact rerank

    # ---- accelerator service times (RTL-calibrated) -----------------
    c_proc: float = 2.0           # cycles/doc steady state (anchored, co-sim)
    topk_init_cycles: float = 500.0  # TOPK_INIT trip count, II = 1
    codebook_load_cycles: float = 12299.0  # CB_STREAM loop, RTL co-sim delta
    pipeline_fixed_cycles: float = 2211.0  # fitted residual base, warm call
    f_clk_ns: float = 1000.0 / 180.0       # one clock period in ns

    # ---- placement / query shape -------------------------------------
    probes: int = 32             # clusters scanned per query (nprobe)
    top_k: int = 8               # results returned to the client
    m_candidates: int = 64       # candidates pulled from DDR for exact rerank
    replication: int = 3         # copies per cluster, on distinct hosts
    osds: int = 20               # 5 workstations x 4 FPGA boards
    boards_per_host: int = 4
    cores_per_host: int = 8      # host CPU cores shared by the 4 local OSDs

    # ---- bandwidths (bytes/ns) ---------------------------------------
    # 0.735 GB/s is a real measurement of the SSD -> PL DDR4 path, but of an
    # unoptimised transfer (single XDMA channel, short bursts, async CDC).
    # It is the current implementation, not the link ceiling: 2x Gen4x4 is
    # 15.76 GB/s and MIG caps the DDR at 12.8 GB/s. 6.4 is the design target
    # (about half the MIG ceiling, allowing headroom). 12.8 is the theoretical
    # bound used by the "ideal" profile. 24 is a two-channel DDR upper bound.
    B_ssd: float = 6.4            # SSD -> PL DDR4 (target)
    B_ddr: float = 5.5            # PL DDR4 read (measured, N-sweep fit)
    B_xdma: float = 20.0          # host PCIe Gen4 x16 (assumed)

    # ---- fixed latencies (ns) ----------------------------------------
    L_cmd: float = 20_000.0       # NVMe command latency (assumed)
    L_pcie: float = 2_000.0       # PCIe round trip (assumed)
    L_cpu: float = 100.0          # ns per exact-rerank candidate, 1 core

    # ---- caches (bytes per OSD) --------------------------------------
    pq_cache_bytes: int = 6 * 1024 ** 3   # 6 GiB of the 8 GB DDR4
    raw_cache_bytes: int = 2 * 1024 ** 3  # the remaining 2 GiB
    bram_slots: int = 1           # codebook slots; only 1 populated today

    # ---- SLO (ns) ----------------------------------------------------
    # Assumed three-class scheme: per-tenant CRC32 hash buckets 50/30/20
    # into 5/7/11 ms. Real tenant SLOs are not yet known.
    slo_classes_ns: Tuple[float, ...] = (5_000_000.0, 7_000_000.0, 11_000_000.0)

    @property
    def cores_per_osd(self) -> float:
        return max(1.0, self.cores_per_host / self.boards_per_host)

    @property
    def T_base_ns(self) -> float:
        """Fixed cost paid once per FPGA invocation when the codebook is warm.

        = pipeline_fixed_cycles + topk_init_cycles, converted to ns.
        2,211 + 500 = 2,711 cycles = 15.06 us at 180 MHz. The 12,299-cycle
        codebook stream is NOT in here: it is charged separately by the OSD
        timing model only on a BRAM miss, so a resident codebook is never
        billed twice.
        """
        return (self.pipeline_fixed_cycles + self.topk_init_cycles) * self.f_clk_ns

    @property
    def T_base_cold_ns(self) -> float:
        """Same, but for a cold call: the codebook stream is added back in."""
        return (self.pipeline_fixed_cycles + self.topk_init_cycles
                + self.codebook_load_cycles) * self.f_clk_ns


def emit_c_header(hw: Hardware = Hardware(), guard: str = "CORTEX_MODEL_PARAMS_H") -> str:
    """Render the hardware constants as a C header.

    Only Hardware goes in: it is the set of values shared with the HLS and
    driver sides. Model parameters (bandwidths, latencies, cache sizes) are
    simulation-only and deliberately excluded.
    """
    lines = [
        "// Generated by model/params.py -- do not edit by hand.",
        "// Regenerate with: python -m model.params --emit-h > include/cortex_model_params.h",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        "// ---- IVFPQ datapath geometry (see model/params.py:Hardware) ----",
    ]
    for f in dataclasses.fields(hw):
        value = getattr(hw, f.name)
        if isinstance(value, float):
            lines.append(f"#define CORTEX_{f.name} {value!r}")
        else:
            lines.append(f"#define CORTEX_{f.name} {value}")
    lines += [
        "",
        "// Derived: PQ codebook bytes resident in BRAM (M x KS x DS x 4).",
        f"#define CORTEX_CODEBOOK_BYTES {hw.codebook_bytes}",
        "",
        f"#endif /* {guard} */",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emit-h", action="store_true",
                    help="print the C header to stdout and exit")
    args = ap.parse_args(argv)
    if args.emit_h:
        print(emit_c_header(), end="")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
