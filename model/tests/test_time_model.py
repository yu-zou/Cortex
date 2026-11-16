"""Numeric sanity checks for the time model against RTL calibration points.

Each expected value below is an anchor from the RTL co-simulation /
board measurements. If one of these fails, either the model changed or the
calibration constants drifted -- investigate before trusting any DES output.

Run: python -m pytest tests/test_time_model.py -q   (or python this file)
"""
from __future__ import annotations

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from strata_sim.model.strata_model import profile  # noqa: E402
from params import Hardware, ModelParams  # noqa: E402


def test_hardware_codebook_bytes():
    # 96 subq x 256 codewords x DS=8 dims x 4 B fp32
    assert Hardware().codebook_bytes == 786432


def test_t_base_warm_is_2711_cycles():
    p = ModelParams()
    assert p.pipeline_fixed_cycles + p.topk_init_cycles == 2711.0
    # 2711 cycles at 180 MHz = 15.06 us
    assert math.isclose(p.T_base_ns, 15061.1, rel_tol=1e-3)


def test_t_base_cold_includes_codebook():
    p = ModelParams()
    assert math.isclose(p.T_base_cold_ns, 83388.9, rel_tol=1e-3)


def test_estimated_profile_scan_1507():
    """T_scan(1507) = 15000 + 35072 = 50.1 us (DDR-bound), matches report."""
    m = profile("estimated")
    assert math.isclose(m.T_scan(1507), 50072.0, rel_tol=1e-3)
    # memory-bound, not compute-bound
    assert m.S_entry * 1507 / m.B_ddr > m.c_proc * 1507 / m.f_clk_mhz * 1e3


def test_load_1507():
    """L_cmd 20 us + (64 + 1507*128)/6.4 B/ns = 50.1 us."""
    m = profile("estimated")
    assert math.isclose(m.T_load(1507), 50150.0, rel_tol=1e-3)


def test_measured_profile_refuses():
    try:
        profile("measured")
    except ValueError:
        return
    raise AssertionError("measured profile must refuse to be silently built")


def test_emit_header_contains_key_constants():
    from params import emit_c_header
    h = emit_c_header()
    assert "#define CORTEX_TILE_MAX_N 8192" in h
    assert "#define CORTEX_TOPK_MAX 500" in h
    assert "#define CORTEX_CODEBOOK_BYTES 786432" in h


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except AssertionError as e:
            bad += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print(f"{len(fns) - bad}/{len(fns)} passed")
    raise SystemExit(1 if bad else 0)
