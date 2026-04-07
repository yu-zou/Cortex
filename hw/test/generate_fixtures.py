#!/usr/bin/env python3
# pyright: reportMissingImports=false
import argparse
import faiss
import json
import os
import struct
import sys

import numpy as np


M = 4
KSUB = 16
D = 32
DSUB = D // M
NUM_VECTORS = 100
NUM_QUERIES = 5
TOP_K = 10
NBITS = 4
METRIC_TYPE = 0
MAGIC = 0x43505100
VERSION = 1


def fixture_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")


def as_float32(x):
    return np.asarray(x, dtype=np.float32)


def write_bin(path, array):
    with open(path, "wb") as f:
        f.write(np.asarray(array).tobytes())


def read_float_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.float32)


def read_uint8_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=np.uint8)


def train_pq(base_vecs):
    faiss.omp_set_num_threads(1)
    pq = faiss.ProductQuantizer(D, M, NBITS)
    pq.train(as_float32(base_vecs))
    codebook = faiss.vector_to_array(pq.centroids).reshape(M, KSUB, DSUB).copy()
    return pq, codebook.astype(np.float32)


def compute_codes(pq, base_vecs):
    raw = np.asarray(pq.compute_codes(as_float32(base_vecs)), dtype=np.uint8).reshape(base_vecs.shape[0], -1)
    codes = np.empty((base_vecs.shape[0], M), dtype=np.uint8)
    for i in range(base_vecs.shape[0]):
        byte0 = raw[i, 0]
        byte1 = raw[i, 1]
        codes[i, 0] = byte0 & 0x0F
        codes[i, 1] = byte0 >> 4
        codes[i, 2] = byte1 & 0x0F
        codes[i, 3] = byte1 >> 4
    return codes


def compute_adc_distances(queries, codebook, codes):
    distances = np.empty((queries.shape[0], codes.shape[0]), dtype=np.float32)
    for qi in range(queries.shape[0]):
        table = np.empty((M, KSUB), dtype=np.float32)
        for m in range(M):
            sub = queries[qi, m * DSUB:(m + 1) * DSUB]
            diff = codebook[m] - sub[None, :]
            table[m] = np.sum(diff * diff, axis=1)
        row = np.zeros(codes.shape[0], dtype=np.float32)
        for m in range(M):
            row += table[m][codes[:, m]]
        distances[qi] = row
    return distances


def kmeans_centroids(base_vecs):
    kmeans = faiss.Kmeans(D, 4, niter=20, nredo=1, seed=42, verbose=False)
    kmeans.train(as_float32(base_vecs))
    return as_float32(kmeans.centroids)


def write_cluster_file(path, codebook, codes):
    header = struct.pack(
        "<7I",
        MAGIC,
        VERSION,
        M,
        KSUB,
        D,
        DSUB,
        int(codes.shape[0]),
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(as_float32(codebook).tobytes())
        f.write(np.asarray(codes, dtype=np.uint8).tobytes())


def generate():
    np.random.seed(42)
    out_dir = fixture_dir()
    os.makedirs(out_dir, exist_ok=True)

    base_vecs = as_float32(np.random.randn(NUM_VECTORS, D))
    queries = as_float32(np.random.randn(NUM_QUERIES, D))

    pq, codebook = train_pq(base_vecs)
    codes = compute_codes(pq, base_vecs)
    distances = compute_adc_distances(queries, codebook, codes)

    centroids = kmeans_centroids(base_vecs)
    assign_index = faiss.IndexFlatL2(D)
    assign_index.add(centroids)
    _, assignments = assign_index.search(base_vecs, 1)
    assignments = assignments.reshape(-1)

    write_bin(os.path.join(out_dir, "codebook.bin"), codebook.reshape(-1))
    write_bin(os.path.join(out_dir, "vectors.bin"), codes.reshape(-1))
    write_bin(os.path.join(out_dir, "queries.bin"), queries.reshape(-1))
    write_bin(os.path.join(out_dir, "golden_distances.bin"), distances.reshape(-1))

    topk_payload = []
    for qi in range(NUM_QUERIES):
        order = np.argsort(distances[qi], kind="mergesort")[:TOP_K]
        for idx in order:
            topk_payload.append(struct.pack("<Qf", int(idx), float(distances[qi, idx])))
    with open(os.path.join(out_dir, "golden_topk.bin"), "wb") as f:
        f.write(b"".join(topk_payload))

    write_bin(os.path.join(out_dir, "centroids.bin"), centroids.reshape(-1))

    for cluster_id in range(4):
        cluster_indices = np.where(assignments == cluster_id)[0]
        write_cluster_file(
            os.path.join(out_dir, f"cluster_{cluster_id}.bin"),
            codebook,
            codes[cluster_indices],
        )

    metadata = {
        "M": M,
        "ksub": KSUB,
        "D": D,
        "dsub": DSUB,
        "num_vectors": NUM_VECTORS,
        "num_queries": NUM_QUERIES,
        "top_k": TOP_K,
        "metric_type": METRIC_TYPE,
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True)

    return out_dir


def verify():
    out_dir = fixture_dir()
    codebook = read_float_bin(os.path.join(out_dir, "codebook.bin")).reshape(M, KSUB, DSUB)
    codes = read_uint8_bin(os.path.join(out_dir, "vectors.bin")).reshape(NUM_VECTORS, M)
    queries = read_float_bin(os.path.join(out_dir, "queries.bin")).reshape(NUM_QUERIES, D)
    golden = read_float_bin(os.path.join(out_dir, "golden_distances.bin")).reshape(NUM_QUERIES, NUM_VECTORS)
    recomputed = compute_adc_distances(queries, codebook, codes)
    if not np.allclose(recomputed, golden, rtol=0, atol=1e-6):
        delta = np.abs(recomputed - golden)
        qi, vi = np.unravel_index(np.argmax(delta), delta.shape)
        print("VERIFY_FAIL")
        print(f"max_diff={delta[qi, vi]:.9f} at query={qi} vector={vi}")
        print(f"recomputed={recomputed[qi, vi]:.9f}")
        print(f"golden={golden[qi, vi]:.9f}")
        return 1
    print("VERIFY_OK")
    return 0


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        return verify()
    generate()
    print(f"Generated fixtures in {fixture_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
