#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate HNSW graph test fixtures

   DRAM layout (matches testbench_hnsw.cpp hnsw_search_engine):
     Header at offset 0: 4 × u32 LE
       {num_nodes, entry_point(=0), dim(=128), max_degree(=32)}
     Per-node at offset 16 + n * node_bytes:
       [0,  DIM*4):  DIM × float32 (vector)
       [DIM*4, +4):  u32 LE num_nbrs (actual neighbor count)
       [+4,  +MAX_DEG*4):  MAX_DEG × u32 LE neighbor IDs (pad=0xFFFFFFFF)
     node_bytes = DIM*4 + 4 + MAX_DEG*4

   Files generated in fixtures/hnsw/:
     graph_header.bin  — header (16B)
     vectors.bin       — N * DIM * 4B (FP32 LE)
     num_nbrs.bin      — N bytes (u8 actual neighbor count)
     adjacency.bin     — N * MAX_DEG * 4B (u32 LE, padded)
     queries.bin       — 5 * DIM * 4B (FP32 LE)
     golden_topk.bin   — per query, 10 × (u32 node_id + float dist) LE
     metadata.json     — parameters
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

# ── Parameters ──────────────────────────────────────────────
DIM = 128
MAX_DEG = 32
NUM_QUERIES = 5
TOP_K = 10
INVALID_ID = 0xFFFFFFFF  # padding for unused neighbor slots


def fixture_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures", "hnsw")


def as_float32(x):
    return np.asarray(x, dtype=np.float32)


def generate_connected_graph(N, max_deg, rng):
    """Build a random connected graph.

    Strategy:
      1. Ring backbone: node i connects to (i+1)%N → guaranteed connectivity.
      2. Random edges: for each node, add random distinct neighbors up to max_deg.

    Returns (num_nbrs, adj_lists):
      num_nbrs: list[N] of int, actual neighbor count per node.
      adj_lists: list[N] of list[int], neighbor IDs for each node.
    """
    adj = [set() for _ in range(N)]

    # 1. Ring backbone
    for i in range(N):
        j = (i + 1) % N
        adj[i].add(j)
        adj[j].add(i)

    # 2. Random edges (bidirectional), fill up to max_deg per node
    for i in range(N):
        # shuffle all candidate neighbors
        candidates = [j for j in range(N) if j != i and j not in adj[i]]
        rng.shuffle(candidates)
        needed = max_deg - len(adj[i])
        for j in candidates[:needed]:
            if len(adj[j]) < max_deg and j not in adj[i]:
                adj[i].add(j)
                adj[j].add(i)

    # Convert to sorted lists
    num_nbrs = [len(s) for s in adj]
    adj_lists = [sorted(s) for s in adj]
    return num_nbrs, adj_lists


def compute_golden(queries, base_vecs, top_k):
    """Brute-force L2² distances → sorted Top-K per query.

    Returns list[list[(node_id, distance)]].
    """
    results = []
    for q in queries:
        diffs = base_vecs - q[None, :]          # (N, DIM)
        dists = np.sum(diffs * diffs, axis=1)   # (N,)  L2 squared
        # Stable merge sort for deterministic tie-breaking
        order = np.argsort(dists, kind="mergesort")[:top_k]
        results.append([(int(idx), float(dists[idx])) for idx in order])
    return results


def generate(small=False):
    N = 20 if small else 100
    rng = np.random.RandomState(42)

    out_dir = fixture_dir()
    os.makedirs(out_dir, exist_ok=True)

    # ── Vectors and queries: uniform in [-5, 5) ──────────
    base_vecs = as_float32(rng.rand(N, DIM) * 10.0 - 5.0)
    queries   = as_float32(rng.rand(NUM_QUERIES, DIM) * 10.0 - 5.0)

    entry_point = 0  # always node 0

    # ── Graph ────────────────────────────────────────────
    num_nbrs, adj_lists = generate_connected_graph(N, MAX_DEG, rng)

    # ── Write binary files ───────────────────────────────

    # 1. graph_header.bin — 4 × u32 LE
    header = struct.pack("<4I", N, entry_point, DIM, MAX_DEG)
    with open(os.path.join(out_dir, "graph_header.bin"), "wb") as f:
        f.write(header)

    # 2. vectors.bin — N × DIM × float32 LE
    with open(os.path.join(out_dir, "vectors.bin"), "wb") as f:
        f.write(base_vecs.tobytes())

    # 3. num_nbrs.bin — N × uint8
    with open(os.path.join(out_dir, "num_nbrs.bin"), "wb") as f:
        f.write(struct.pack(f"{len(num_nbrs)}B", *num_nbrs))

    # 4. adjacency.bin — N × MAX_DEG × u32 LE (padded with INVALID_ID)
    adj_buf = bytearray()
    for i in range(N):
        for nbr_id in adj_lists[i]:
            adj_buf.extend(struct.pack("<I", nbr_id))
        pads = MAX_DEG - len(adj_lists[i])
        if pads > 0:
            adj_buf.extend(struct.pack(f"<{pads}I", *([INVALID_ID] * pads)))
    with open(os.path.join(out_dir, "adjacency.bin"), "wb") as f:
        f.write(adj_buf)

    # 5. queries.bin — NUM_QUERIES × DIM × float32 LE
    with open(os.path.join(out_dir, "queries.bin"), "wb") as f:
        f.write(queries.tobytes())

    # 6. golden_topk.bin — per query, TOP_K × (u32 node_id + float32 dist) LE
    golden = compute_golden(queries, base_vecs, TOP_K)
    golden_buf = bytearray()
    for q_results in golden:
        for node_id, dist in q_results:
            golden_buf.extend(struct.pack("<If", node_id, dist))
    with open(os.path.join(out_dir, "golden_topk.bin"), "wb") as f:
        f.write(golden_buf)

    # 7. metadata.json
    metadata = {
        "N": N,
        "DIM": DIM,
        "MAX_DEG": MAX_DEG,
        "num_queries": NUM_QUERIES,
        "top_k": TOP_K,
        "entry_point": entry_point,
        "fixture_type": "hnsw_graph",
        "node_bytes": DIM * 4 + 4 + MAX_DEG * 4,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return out_dir, N


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate HNSW graph test fixtures"
    )
    parser.add_argument(
        "--small", action="store_true",
        help="Use N=20 nodes instead of N=100"
    )
    args = parser.parse_args(argv)

    out_dir, N = generate(small=args.small)

    # ── Summary ──────────────────────────────────────────
    files = [
        "graph_header.bin",
        "vectors.bin",
        "num_nbrs.bin",
        "adjacency.bin",
        "queries.bin",
        "golden_topk.bin",
        "metadata.json",
    ]
    print(f"Generated HNSW fixtures in {out_dir}")
    print(f"  N={N}, DIM={DIM}, MAX_DEG={MAX_DEG}, "
          f"num_queries={NUM_QUERIES}, top_k={TOP_K}")
    for fname in files:
        fpath = os.path.join(out_dir, fname)
        if os.path.exists(fpath):
            print(f"  {fname}: {os.path.getsize(fpath)} bytes")
        else:
            print(f"  {fname}: MISSING")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
