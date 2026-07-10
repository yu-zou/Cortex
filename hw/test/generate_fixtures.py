#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Generate test fixtures in IVF binary format
   Header 64B: {M(4B), DIM(4B), N(4B), reserved[13](52B)}
   Codebook: M * KS * Ds * 4B (KS=256 hardcoded)
   PQ entry: M-byte PQ code + 8-byte doc_addr + 8-byte doc_length = M+16 bytes
   — matches acc_top.h IVFHeader + DRAM layout"""
import argparse
import faiss
import json
import os
import struct
import sys

import numpy as np

# Design parameters: KS=256 hardcoded, M=16, DIM=128
# Fast iteration params for synthesis: M=4, KS=16, DIM=32
M = 16
KS = 256       # Hardcoded, matches FPGA macro KS=256
DIM = 128
DSUB = DIM // M
NUM_VECTORS = 100
NUM_QUERIES = 5
TOP_K = 10
NBITS = 8       # PQ nbits, KS = 2^nbits = 256
METRIC_TYPE = 0  # 0=L2, 1=IP

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

def train_pq(base_vecs):
    faiss.omp_set_num_threads(1)
    pq = faiss.ProductQuantizer(DIM, M, NBITS)
    pq.train(as_float32(base_vecs))
    codebook = faiss.vector_to_array(pq.centroids).reshape(M, KS, DSUB).copy()
    return pq, codebook.astype(np.float32)

def compute_codes(pq, base_vecs):
    """PQ encoding: unpacks Faiss codes to M uint8 per vector.
    For NBITS=8: 1 byte per code → raw.shape = (N, M).
    For NBITS=4: 2 codes per byte → unpack high/low nibbles."""
    raw = np.asarray(pq.compute_codes(as_float32(base_vecs)), dtype=np.uint8)
    if NBITS < 8:
        # Unpack: each byte holds 2 codes (high nibble | low nibble)
        high = (raw >> 4).reshape(base_vecs.shape[0], -1)
        low  = (raw & 0x0F).reshape(base_vecs.shape[0], -1)
        unpacked = np.concatenate([high, low], axis=1)[:, :M]
        return unpacked.astype(np.uint8)
    return raw.reshape(base_vecs.shape[0], -1)

def compute_adc_distances(queries, codebook, codes):
    """ADC L2距离: 对每个query, 查表累加"""
    distances = np.empty((queries.shape[0], codes.shape[0]), dtype=np.float32)
    for qi in range(queries.shape[0]):
        table = np.empty((M, KS), dtype=np.float32)
        for m in range(M):
            sub = queries[qi, m * DSUB:(m + 1) * DSUB]
            diff = codebook[m] - sub[None, :]
            table[m] = np.sum(diff * diff, axis=1)
        row = np.zeros(codes.shape[0], dtype=np.float32)
        for m in range(M):
            row += table[m][codes[:, m]]
        distances[qi] = row
    return distances

def write_cluster_file(path, codebook, codes):
    """IVF DRAM layout:
       Offset 0x00: 64B Header {M(u32), DIM(u32), N(u32), reserved[13](u32)}
       Offset 0x40: Codebook: M * KS * DSUB * 4 字节
       Offset 0x40+CB: PQ Codes: N * (M + 16) 字节
          每条目: M字节PQ码 + 8B doc_addr + 8B doc_length"""
    N = codes.shape[0]
    entry_bytes = M + 16  # M PQ codes + 8 doc_addr + 8 doc_length

    # 64B Header: 16 × u32 (小端序)
    reserved = [0] * 13
    header = struct.pack("<16I", M, DIM, N, *reserved)

    # Codebook
    cb_packed = as_float32(codebook.reshape(-1)).tobytes()

    # PQ entries: 每条 M字节codes + 8B doc_addr + 8B doc_length
    pq_buf = bytearray()
    for i in range(N):
        pq_buf.extend(codes[i, :M].tobytes())  # M 字节 PQ 码
        pq_buf.extend(struct.pack("<Q", i))     # doc_addr = vector index
        pq_buf.extend(struct.pack("<Q", 0))     # doc_length = 0 (测试用)

    with open(path, "wb") as f:
        f.write(header)
        f.write(cb_packed)
        f.write(pq_buf)

def kmeans_centroids(base_vecs):
    nlist = 4
    kmeans = faiss.Kmeans(DIM, nlist, niter=20, nredo=1, seed=42, verbose=False)
    kmeans.train(as_float32(base_vecs))
    return as_float32(kmeans.centroids)

def generate():
    np.random.seed(42)
    out_dir = fixture_dir()
    os.makedirs(out_dir, exist_ok=True)

    base_vecs = as_float32(np.random.randn(NUM_VECTORS, DIM))
    queries = as_float32(np.random.randn(NUM_QUERIES, DIM))

    pq, codebook = train_pq(base_vecs)
    codes = compute_codes(pq, base_vecs)
    distances = compute_adc_distances(queries, codebook, codes)

    centroids = kmeans_centroids(base_vecs)
    assign_index = faiss.IndexFlatL2(DIM)
    assign_index.add(centroids)
    _, assignments = assign_index.search(base_vecs, 1)
    assignments = assignments.reshape(-1)

    write_bin(os.path.join(out_dir, "codebook.bin"), codebook.reshape(-1))
    write_bin(os.path.join(out_dir, "vectors.bin"), codes.reshape(-1))
    write_bin(os.path.join(out_dir, "queries.bin"), queries.reshape(-1))
    write_bin(os.path.join(out_dir, "golden_distances.bin"), distances.reshape(-1))

    # golden_topk: {uint64_t doc_addr, float dist} per entry
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
        "KS": KS,
        "D": DIM,
        "dsub": DSUB,
        "ksub": KS,
        "NBITS": NBITS,
        "num_vectors": NUM_VECTORS,
        "num_queries": NUM_QUERIES,
        "top_k": TOP_K,
        "metric_type": METRIC_TYPE,
        "header_format": "64B IVF Header: {M, DIM, N, reserved[13]}",
        "pq_entry": f"M PQ codes + 8B doc_addr + 8B doc_length = {M}+16 bytes",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    return out_dir

def read_uint8_bin(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8)

def verify():
    out_dir = fixture_dir()
    codebook = read_float_bin(os.path.join(out_dir, "codebook.bin")).reshape(M, KS, DSUB)
    codes = read_uint8_bin(os.path.join(out_dir, "vectors.bin")).reshape(NUM_VECTORS, M)
    queries = read_float_bin(os.path.join(out_dir, "queries.bin")).reshape(NUM_QUERIES, DIM)
    golden = read_float_bin(os.path.join(out_dir, "golden_distances.bin")).reshape(NUM_QUERIES, NUM_VECTORS)
    recomputed = compute_adc_distances(queries, codebook, codes)
    if not np.allclose(recomputed, golden, rtol=0, atol=1e-6):
        delta = np.abs(recomputed - golden)
        qi, vi = np.unravel_index(np.argmax(delta), delta.shape)
        print("VERIFY_FAIL")
        print(f"max_diff={delta[qi, vi]:.9f} at query={qi} vector={vi}")
        return 1
    print("VERIFY_OK")
    return 0

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--small", action="store_true",
                        help="Use smaller params for faster iteration (M=4, KS=16, DIM=32)")
    args = parser.parse_args(argv)

    if args.small:
        global M, KS, DIM, DSUB, NBITS
        M = 4; KS = 16; DIM = 32; DSUB = DIM // M; NBITS = 4

    if args.verify:
        return verify()
    generate()
    print(f"Generated fixtures in {fixture_dir()}")
    print(f"  M={M}, KS={KS}, DIM={DIM}, DSUB={DSUB}, NBITS={NBITS}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
