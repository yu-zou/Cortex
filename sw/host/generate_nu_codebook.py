#!/usr/bin/env python3
"""generate_nu_codebook.py — Non-uniform Product Quantization codebook generation.

Pipeline:
  1. Load embeddings + variance CSV from analyze_dim_variance.py
  2. Group dimensions into M=16 sub-quantizers (each covering DIM/M dims)
  3. Assign conservative bit width per sub-quantizer = MIN of its dimensions' bits
  4. Train k-means with K=2^bits centroids per sub-quantizer
  5. Save codebook in flat float32 binary format (matching IVFPQ fixture format)
  6. Compare recall@10: non-uniform vs uniform 8-bit PQ on held-out queries
"""

import argparse
import csv
import json
import os
import struct
import sys

import numpy as np

# ── Optional sklearn for k-means ──────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans as SKLearnKMeans

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Manual K-Means (fallback when sklearn unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

class ManualKMeans:
    """Simple k-means with random initialization and convergence check."""

    def __init__(self, n_clusters, n_iter=20, random_state=42):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        # Handle edge case: fewer samples than clusters
        k = min(self.n_clusters, n)
        if k < self.n_clusters:
            # Pad centroids with zeros
            centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=X.dtype)
            idx = rng.choice(n, k, replace=False)
            centroids[:k] = X[idx].copy()
            self.cluster_centers_ = centroids
            return self

        idx = rng.choice(n, k, replace=False)
        centroids = X[idx].copy()

        for _ in range(self.n_iter):
            # Assignment step
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # Update step
            new_centroids = centroids.copy()
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = X[mask].mean(axis=0)

            if np.allclose(centroids, new_centroids, rtol=1e-6, atol=1e-8):
                break
            centroids = new_centroids

        self.cluster_centers_ = centroids
        return self


def kmeans_fit(X, n_clusters, seed=42):
    """Wrapper: use sklearn if available, fall back to manual."""
    if HAS_SKLEARN:
        km = SKLearnKMeans(n_clusters=n_clusters, n_init=3,
                           random_state=seed, n_init_no_improvement=5)
        km.fit(X)
        return km.cluster_centers_.astype(np.float32)
    else:
        km = ManualKMeans(n_clusters=n_clusters, random_state=seed)
        km.fit(X)
        return km.cluster_centers_.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  PQ Encoding & ADC
# ═══════════════════════════════════════════════════════════════════════════════

def encode_pq(X, codebook_list, bits_per_subq):
    """Encode vectors using PQ codebook.

    Args:
        X: Input vectors, shape [N, DIM]
        codebook_list: List of M arrays, each shape [K_m, DSUB]
        bits_per_subq: List of M ints

    Returns:
        codes: List of M arrays, each shape [N] with centroid indices
    """
    N = X.shape[1]  # DIM
    M = len(codebook_list)
    codes = []
    offset = 0
    for m in range(M):
        centroids = codebook_list[m]  # [K_m, DSUB]
        DSUB = centroids.shape[1]
        sub = X[:, offset:offset + DSUB]  # [N, DSUB]
        # Squared L2 distance to all centroids
        diff = sub[:, None, :] - centroids[None, :, :]  # [N, K_m, DSUB]
        sq_dists = np.sum(diff * diff, axis=2)  # [N, K_m]
        codes.append(np.argmin(sq_dists, axis=1).astype(np.uint8))
        offset += DSUB
    return codes


def compute_adc(queries, codebook_list, codes_list):
    """Asymmetric Distance Computation for non-uniform PQ.

    Args:
        queries: Shape [NQ, DIM]
        codebook_list: List of M arrays, each [K_m, DSUB]
        codes_list: List of M arrays, each [NV] with centroid indices

    Returns:
        distances: Shape [NQ, NV]
    """
    NQ = queries.shape[0]
    NV = codes_list[0].shape[0]
    M = len(codebook_list)

    distances = np.zeros((NQ, NV), dtype=np.float32)
    offset = 0

    for m in range(M):
        centroids = codebook_list[m]  # [K_m, DSUB]
        DSUB = centroids.shape[1]
        sub_q = queries[:, offset:offset + DSUB]  # [NQ, DSUB]
        codes = codes_list[m]  # [NV]

        # Precompute distance table per query for this sub-quantizer
        # diff: [NQ, K_m, DSUB] → [NQ, K_m]
        diff = sub_q[:, None, :] - centroids[None, :, :]
        table = np.sum(diff * diff, axis=2)  # [NQ, K_m]

        # Accumulate: distances[qi, vi] += table[qi, codes[vi]]
        for qi in range(NQ):
            distances[qi, :] += table[qi, codes]

        offset += DSUB

    return distances


# ═══════════════════════════════════════════════════════════════════════════════
#  Recall Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def brute_force_topk(queries, db, k=10):
    """Exact nearest neighbor via brute-force L2."""
    dists = np.linalg.norm(queries[:, None, :] - db[None, :, :], axis=2)
    return np.argsort(dists, axis=1)[:, :k]


def recall_at_k(queries, db, codebook_list, bits_per_subq, k=10):
    """Compute recall@k for PQ search vs brute-force exact search.

    Returns:
        recall: mean recall@k across all queries
        per_query: per-query recall values
    """
    NQ = queries.shape[0]
    NV = db.shape[0]

    # Ground truth
    gt_topk = brute_force_topk(queries, db, k)

    # PQ encode
    codes = encode_pq(db, codebook_list, bits_per_subq)

    # ADC distances
    pq_dists = compute_adc(queries, codebook_list, codes)

    # PQ top-k
    pq_topk = np.argsort(pq_dists, axis=1)[:, :k]

    # Per-query recall
    per_query = np.zeros(NQ, dtype=np.float64)
    for qi in range(NQ):
        gt_set = set(gt_topk[qi])
        pq_set = set(pq_topk[qi])
        overlap = len(gt_set & pq_set)
        per_query[qi] = overlap / k

    return float(np.mean(per_query)), per_query


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate non-uniform PQ codebook with variable bit-width per sub-quantizer"
    )
    parser.add_argument("input", type=str,
                        help="Path to .npy file of embeddings (shape [N, DIM])")
    parser.add_argument("--variance-csv", "-v", type=str, default="dim_variance.csv",
                        help="Variance CSV from analyze_dim_variance.py (default: dim_variance.csv)")
    parser.add_argument("--output", "-o", type=str, default="nu_codebook.bin",
                        help="Output codebook path (default: nu_codebook.bin)")
    parser.add_argument("--bits-output", "-b", type=str, default="nu_bits.bin",
                        help="Output bits-per-subq path (default: nu_bits.bin)")
    parser.add_argument("--M", "-m", type=int, default=16,
                        help="Number of sub-quantizers (default: 16)")
    parser.add_argument("--test-split", "-t", type=float, default=0.2,
                        help="Fraction of vectors to hold out as test queries (default: 0.2)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--kmeans-iter", type=int, default=20,
                        help="Max k-means iterations (default: 20)")
    return parser.parse_args(argv)


def load_variance_csv(path):
    """Load variance CSV and return (dim_index -> assigned_bits) mapping."""
    dim_to_bits = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim_idx = int(row['dim_index'])
            bits = int(row['assigned_bits'])
            dim_to_bits[dim_idx] = bits
    return dim_to_bits


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.RandomState(args.seed)

    # ── 1. Load data ───────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.variance_csv):
        print(f"Error: variance CSV not found: {args.variance_csv}", file=sys.stderr)
        return 1

    X = np.load(args.input).astype(np.float32)
    if X.ndim != 2:
        print(f"Error: expected 2D array, got shape {X.shape}", file=sys.stderr)
        return 1

    N, DIM = X.shape
    M = args.M
    if DIM % M != 0:
        print(f"Error: DIM={DIM} not divisible by M={M}", file=sys.stderr)
        return 1
    DSUB = DIM // M

    dim_to_bits = load_variance_csv(args.variance_csv)
    if len(dim_to_bits) != DIM:
        print(f"  [info] CSV has {len(dim_to_bits)} dims, expected {DIM}")

    # ── 2. Shuffle & split train/test ──────────────────────────────────────
    indices = rng.permutation(N)
    n_test = max(1, int(N * args.test_split))
    n_train = N - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = X[train_idx]
    X_test = X[test_idx]

    print(f"=== NON-UNIFORM PQ CODEBOOK GENERATION ===")
    print(f"  Embeddings:  {N} vectors × {DIM} dims")
    print(f"  Sub-quantizers: M={M}, DSUB={DSUB}")
    print(f"  Train/Test:  {n_train}/{n_test}")
    print()

    # ── 3. Per sub-quantizer: compute min bit width ────────────────────────
    bits_per_subq = []
    for m in range(M):
        dim_start = m * DSUB
        dim_end = (m + 1) * DSUB
        sub_bits = [dim_to_bits.get(d, 8) for d in range(dim_start, dim_end)]
        min_bits = min(sub_bits)
        bits_per_subq.append(min_bits)

    print(f"  Sub-quantizer bit assignments (min of constituent dims):")
    print(f"    SubQ   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15")
    print(f"    Bits: ", end="")
    for b in bits_per_subq:
        print(f" {b:>2}", end="")
    print()

    # Distribution
    for b in [8, 4, 2]:
        cnt = bits_per_subq.count(b)
        print(f"      {b}-bit sub-quantizers: {cnt}/16")

    total_centroids = sum(2 ** b for b in bits_per_subq)
    uniform_centroids = M * 256
    print(f"    Total centroids: {total_centroids} "
          f"(uniform 8-bit: {uniform_centroids})")
    print(f"    Codebook reduction: {uniform_centroids - total_centroids} centroids "
          f"({(1 - total_centroids / uniform_centroids) * 100:.1f}%)")
    print()

    # ── 4. Train codebook ─────────────────────────────────────────────────
    codebook_list = []
    print(f"  Training codebook ({HAS_SKLEARN and 'sklearn' or 'manual'} k-means):")

    for m in range(M):
        dim_start = m * DSUB
        dim_end = (m + 1) * DSUB
        sub_vecs = X_train[:, dim_start:dim_end]  # [n_train, DSUB]

        K = 2 ** bits_per_subq[m]
        if K > n_train:
            K = n_train
            print(f"    SubQ {m:>2d}: clamping K to {K} (n_train={n_train} too small)")

        centroids = kmeans_fit(sub_vecs, K, seed=args.seed + m)
        codebook_list.append(centroids)

        if m < 8 or m >= M - 2:  # Print first few and last
            print(f"    SubQ {m:>2d}: bits={bits_per_subq[m]} K={K:>4d} "
                  f"DSUB={DSUB} centroids.shape={centroids.shape}")

    if M > 10:
        print(f"    ... ({M - 10} sub-quantizers omitted)")

    print()

    # ── 5. Save codebook ───────────────────────────────────────────────────
    # Format: flat float32 array, all centroids concatenated sub-Q by sub-Q
    # Companion bits file: M uint8 values
    cb_flat = np.concatenate([c.reshape(-1) for c in codebook_list])
    cb_flat.tofile(args.output)

    bits_arr = np.array(bits_per_subq, dtype=np.uint8)
    bits_arr.tofile(args.bits_output)

    # Also save as .npz for convenience
    npz_path = args.output.replace('.bin', '.npz') if '.bin' in args.output else args.output + '.npz'
    np.savez(npz_path,
             centroids=cb_flat,
             bits=bits_arr,
             M=np.int32(M),
             DIM=np.int32(DIM),
             DSUB=np.int32(DSUB))

    print(f"  Codebook saved to:")
    print(f"    {args.output}         ({os.path.getsize(args.output)} bytes, flat float32)")
    print(f"    {args.bits_output}       ({os.path.getsize(args.bits_output)} bytes, M×uint8)")
    print(f"    {npz_path}    (NPZ convenience archive)")
    print()

    # ── 6. Train uniform 8-bit codebook for comparison ────────────────────
    uniform_bits = [8] * M
    uniform_codebook = []
    print(f"  Training uniform 8-bit codebook (baseline):")

    for m in range(M):
        dim_start = m * DSUB
        dim_end = (m + 1) * DSUB
        sub_vecs = X_train[:, dim_start:dim_end]
        n_clusters = min(256, n_train)
        centroids = kmeans_fit(sub_vecs, n_clusters, seed=args.seed + 100 + m)
        uniform_codebook.append(centroids)
        if m < 4 or m >= M - 2:
            print(f"    SubQ {m:>2d}: K=256 centroids.shape={centroids.shape}")

    print()

    # ── 7. Recall comparison ───────────────────────────────────────────────
    k = 10
    print(f"  Evaluating recall@{k} on {n_test} test queries...")

    nu_recall, nu_per_q = recall_at_k(X_test, X_train,
                                       codebook_list, bits_per_subq, k=k)
    uni_recall, uni_per_q = recall_at_k(X_test, X_train,
                                         uniform_codebook, uniform_bits, k=k)

    print()
    print(f"  ┌─────────────────────────────────────────────────────────────┐")
    print(f"  │  RECALL@{k:<2d} COMPARISON  (mean over {n_test} queries)            │")
    print(f"  ├───────────────────────────────────┬─────────────────────────┤")
    print(f"  │  Uniform 8-bit PQ                 │  {uni_recall:.4f}                  │")
    print(f"  │  Non-uniform PQ (variable bits)   │  {nu_recall:.4f}                  │")
    print(f"  ├───────────────────────────────────┼─────────────────────────┤")
    delta = nu_recall - uni_recall
    sign = "+" if delta >= 0 else ""
    print(f"  │  Delta                             │  {sign}{delta:.4f}                  │")
    print(f"  └───────────────────────────────────┴─────────────────────────┘")
    print()

    # Storage comparison
    # Codebook storage: each centroid is float32 (4 bytes)
    uniform_cb_bytes = M * 256 * DSUB * 4
    nu_cb_bytes = int(np.sum([(2 ** b) * DSUB * 4 for b in bits_per_subq]))
    cb_savings = uniform_cb_bytes - nu_cb_bytes

    # PQ code storage (packed bits)
    # Uniform: N * M * 1 byte (8-bit = 1 byte per code)
    # Non-uniform: packed — sum(ceil(bits[m] / 8)) per vector
    nu_code_bytes_per_vec = sum((b + 7) // 8 for b in bits_per_subq)  # ceil(bits/8)
    uniform_code_storage = n_train * M * 1
    nu_code_storage = n_train * nu_code_bytes_per_vec
    code_savings = uniform_code_storage - nu_code_storage

    total_uniform = uniform_cb_bytes + uniform_code_storage
    total_nu = nu_cb_bytes + nu_code_storage

    print(f"  Storage comparison:")
    print(f"  ┌────────────────────────────────────┬──────────────────┬──────────────────┬───────────┐")
    print(f"  │  Component                         │  Uniform 8-bit   │  Non-uniform     │  Savings  │")
    print(f"  ├────────────────────────────────────┼──────────────────┼──────────────────┼───────────┤")
    print(f"  │  Codebook centroids                │  {uniform_cb_bytes:>14} B │  {nu_cb_bytes:>14} B │  {cb_savings:>7} B │")
    print(f"  │  PQ codes ({n_train} vectors)                │  {uniform_code_storage:>14} B │  {nu_code_storage:>14} B │  {code_savings:>7} B │")
    print(f"  ├────────────────────────────────────┼──────────────────┼──────────────────┼───────────┤")
    savings_pct = (1 - total_nu / total_uniform) * 100 if total_uniform > 0 else 0
    print(f"  │  TOTAL                              │  {total_uniform:>14} B │  {total_nu:>14} B │  {savings_pct:>6.1f}%  │")
    print(f"  └────────────────────────────────────┴──────────────────┴──────────────────┴───────────┘")
    print()

    # Per-query recall stats
    print(f"  Per-query recall distribution ({n_test} queries):")
    print(f"    Non-uniform:  mean={nu_recall:.4f}  "
          f"min={nu_per_q.min():.4f}  max={nu_per_q.max():.4f}  "
          f"std={nu_per_q.std():.4f}")
    print(f"    Uniform 8-bit: mean={uni_recall:.4f}  "
          f"min={uni_per_q.min():.4f}  max={uni_per_q.max():.4f}  "
          f"std={uni_per_q.std():.4f}")

    # ── Save comparison results ────────────────────────────────────────────
    result_path = args.output.replace('.bin', '_results.json') if '.bin' in args.output else 'nu_results.json'
    results = {
        "M": M,
        "DIM": DIM,
        "DSUB": DSUB,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "bits_per_subq": [int(b) for b in bits_per_subq],
        "total_centroids_nu": int(total_centroids),
        "total_centroids_uniform": int(uniform_centroids),
        "uniform_recall_at_10": float(round(uni_recall, 6)),
        "nonuniform_recall_at_10": float(round(nu_recall, 6)),
        "recall_delta": float(round(delta, 6)),
        "codebook_cb_uniform_bytes": int(uniform_cb_bytes),
        "codebook_cb_nonuniform_bytes": int(nu_cb_bytes),
        "codebook_savings_bytes": int(cb_savings),
        "pq_codes_uniform_bytes": int(uniform_code_storage),
        "pq_codes_nonuniform_bytes": int(nu_code_storage),
        "total_uniform_bytes": int(total_uniform),
        "total_nonuniform_bytes": int(total_nu),
        "total_storage_savings_pct": float(round(savings_pct, 2)),
    }
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {result_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
