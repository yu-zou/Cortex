#!/usr/bin/env python3
"""analyze_dim_variance.py — Per-dimension variance analysis for LLM embedding vectors.

Assigns bit widths based on variance distribution:
  - Top 25% highest variance dimensions → 8-bit
  - Middle 50% dimensions → 4-bit
  - Bottom 25% lowest variance dimensions → 2-bit

Outputs CSV with per-dimension analysis and prints storage comparison vs uniform 8-bit.
"""

import argparse
import csv
import os
import sys

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze per-dimension variance of embedding vectors and assign bit widths"
    )
    parser.add_argument("input", type=str,
                        help="Path to .npy file of embeddings (shape [N, DIM])")
    parser.add_argument("--output", "-o", type=str, default="dim_variance.csv",
                        help="Output CSV path (default: dim_variance.csv)")
    parser.add_argument("--dim", "-d", type=int, default=768,
                        help="Expected embedding dimension (default: 768)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── Load embeddings ────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    data = np.load(args.input)
    if data.ndim != 2:
        print(f"Error: expected 2D array, got shape {data.shape}", file=sys.stderr)
        return 1

    N, DIM = data.shape
    if DIM != args.dim:
        print(f"  [info] input dimension {DIM} != expected {args.dim}, using actual {DIM}")

    # ── Per-dimension variance ─────────────────────────────────────────────
    variances = np.var(data, axis=0, dtype=np.float64)  # shape (DIM,)
    total_var = np.sum(variances)

    # Sort by variance descending
    sorted_indices = np.argsort(variances)[::-1]
    sorted_vars = variances[sorted_indices]

    # Cumulative variance percentage
    cum_var = np.cumsum(sorted_vars)
    cum_var_pct = cum_var / total_var * 100.0

    # ── Bit-width assignment ───────────────────────────────────────────────
    # top 25% → 8-bit, mid 50% → 4-bit, bottom 25% → 2-bit
    n = DIM
    n_top = n // 4
    n_bot = n // 4

    bits_sorted = np.empty(n, dtype=np.int32)
    bits_sorted[:n_top] = 8
    bits_sorted[n_top:n - n_bot] = 4
    bits_sorted[n - n_bot:] = 2

    # Reorder bits back to original dimension index order
    bits_orig_order = np.empty(n, dtype=np.int32)
    bits_orig_order[sorted_indices] = bits_sorted

    # ── Write CSV ──────────────────────────────────────────────────────────
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["dim_index", "variance", "assigned_bits", "cum_variance_pct"])
        for rank, dim_idx in enumerate(sorted_indices):
            writer.writerow([
                int(dim_idx),
                f"{sorted_vars[rank]:.8f}",
                int(bits_sorted[rank]),
                f"{cum_var_pct[rank]:.4f}",
            ])

    # ── Summary ────────────────────────────────────────────────────────────
    db_name = os.path.basename(args.input)

    print(f"=== DIMENSION VARIANCE ANALYSIS ===")
    print(f"  Input:          {db_name} ({N} vectors × {DIM} dimensions)")
    print(f"  Total variance: {total_var:.6f}")
    print()
    print(f"  Bit-width assignment ({n} dims):")
    print(f"    8-bit (high variance): {n_top} dims (top {n_top})")
    print(f"    4-bit (mid variance):  {n - n_top - n_bot} dims")
    print(f"    2-bit (low variance):  {n_bot} dims (bottom {n_bot})")
    print()

    # Storage comparison (theoretical minimum bits for encoding DIM dims)
    # Uniform:   N * DIM * 8 bits    (8-bit per dim)
    # Non-uniform: N * sum(assigned_bits) bits
    uniform_bytes = N * DIM * 1
    nu_total_bits = int(np.sum(bits_orig_order))
    nu_bytes = N * nu_total_bits // 8
    # Account for partial byte rounding
    if (N * nu_total_bits) % 8 != 0:
        nu_bytes += 1

    print(f"  Storage comparison (PQ codes only, {N} vectors):")
    print(f"    Uniform 8-bit:   {uniform_bytes:>10} bytes  ({uniform_bytes / 1024:.1f} KB)")
    print(f"    Non-uniform:     {nu_bytes:>10} bytes  ({nu_bytes / 1024:.1f} KB)")
    print(f"    Savings:         {uniform_bytes - nu_bytes:>10} bytes  "
          f"({(1 - nu_bytes / uniform_bytes) * 100:.1f}%)")
    print()

    # Per bit-group variance contribution
    print(f"  Variance contribution by bit group:")
    for b in [8, 4, 2]:
        dims_in_b = np.where(bits_orig_order == b)[0]
        if len(dims_in_b) == 0:
            continue
        var_sum = np.sum(variances[dims_in_b])
        print(f"    {b}-bit dims ({len(dims_in_b):>3}):  "
              f"var = {var_sum:.6f}  ({var_sum / total_var * 100:.1f}% of total)")

    print(f"\n  CSV written to: {args.output}")

    # Print bit array for downstream consumption
    bits_csv = ",".join(str(b) for b in bits_orig_order)
    print(f"  Bit array (dim 0..{n - 1}):  [{bits_csv}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
