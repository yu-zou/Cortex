#!/usr/bin/env python3
"""Standalone, CPU-only FiQA candidate rerank evidence; no DES integration.

Selection depends only on embeddings/index. Qrels are loaded after retrieval.
Ties use ascending embedding row ID (including Flat boundary ties). Outputs
require a new directory and every file is created exclusively.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_mapping(path, key, count):
    with Path(path).open() as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != count:
        raise ValueError("mapping cardinality mismatch")
    ids = []
    for i, row in enumerate(rows):
        if type(row.get("row")) is not int or row["row"] != i:
            raise ValueError("mapping rows must be contiguous and ordered")
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError("illegal mapped ID")
        ids.append(value)
    if len(set(ids)) != count:
        raise ValueError("duplicate mapped ID")
    return ids


def validate_vectors(vectors):
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not len(vectors):
        raise ValueError("expected nonempty float32 matrix")
    for start in range(0, len(vectors), 2048):
        block = vectors[start:start + 2048]
        if not np.isfinite(block).all() or not np.allclose(
                np.linalg.norm(block, axis=1), 1., atol=1e-5, rtol=0):
            raise ValueError("vectors must be finite and L2 normalized")


def candidate_ids(ids, count, minimum=10):
    ids = np.asarray(ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("candidate IDs must be an integer vector")
    if np.any(ids < 0) or np.any(ids >= count):
        raise ValueError("candidate ID out of range (including Faiss -1 sentinel)")
    unique = np.unique(ids).astype(np.int64)
    if len(unique) < minimum:
        raise ValueError("fewer than 10 unique candidates")
    return unique


def rank(ids, scores, k=10):
    ids, scores = np.asarray(ids), np.asarray(scores)
    if len(ids) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("invalid scores")
    order = np.lexsort((ids, -scores))[:k]
    return ids[order], scores[order]


def rerank(query, documents, candidates):
    ids = candidate_ids(candidates, len(documents))
    # Full precision normalized IP, never reconstructed PQ vectors or qrels.
    scores = documents[ids] @ query
    return rank(ids, scores)


def load_qrels(path, query_ids, document_ids):
    queries, documents = set(query_ids), set(document_ids)
    result = {q: {} for q in query_ids}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            q, d, score = row["query-id"], row["corpus-id"], int(row["score"])
            if q not in queries or d not in documents or score < 0:
                raise ValueError("illegal qrel reference or relevance")
            if d in result[q]:
                raise ValueError("duplicate qrel")
            result[q][d] = score
    if any(not any(s > 0 for s in rel.values()) for rel in result.values()):
        raise ValueError("every test query requires positive qrels")
    return result


def evaluate(rows, flat, dids, qids, qrels):
    totals = np.zeros(3, dtype=np.float64)
    for qi, qid in enumerate(qids):
        ids = candidate_ids(rows[qi], len(dids))
        if len(ids) != 10 or len(rows[qi]) != 10:
            raise ValueError("evaluation requires exactly 10 unique IDs")
        rel = qrels[qid]
        relevant = {d for d, score in rel.items() if score > 0}
        retrieved = [dids[int(i)] for i in rows[qi]]
        ideal = sorted(rel.values(), reverse=True)[:10]
        idcg = sum(s / math.log2(i + 2) for i, s in enumerate(ideal))
        dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(retrieved))
        totals += (dcg / idcg, len(set(retrieved) & relevant) / len(relevant),
                   len(set(map(int, rows[qi])) & set(map(int, flat[qi]))) / 10)
    return dict(zip(("ndcg@10", "recall@10", "exact_overlap@10"),
                    (totals / len(qids)).tolist()))


def save_rows(path, qids, dids, rows, scores, candidates=None):
    with Path(path).open("x") as stream:
        for i, qid in enumerate(qids):
            row = {"query_id": qid, "query_row": i, "rows": rows[i].tolist(),
                   "document_ids": [dids[int(j)] for j in rows[i]],
                   "scores": [float(s) for s in scores[i]]}
            if candidates is not None:
                row["candidate_rows"] = candidates[i].tolist()
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def run(root, output):
    import faiss

    root, output = Path(root).resolve(), Path(output).resolve()
    # Atomic fresh directory reservation: even an empty existing directory fails.
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    tests = source.parent.parent / "tests" / "test_beir_candidate_rerank.py"
    # Staged execution can place tests alongside the script.
    if not tests.is_file():
        tests = source.with_name("test_beir_candidate_rerank.py")
    for path in (source, tests):
        with (output / path.name).open("xb") as stream:
            stream.write(path.read_bytes())
    base = root.parent
    files = [root / name for name in (
        "document_embeddings.npy", "test_query_embeddings.npy", "document_ids.jsonl",
        "test_query_ids.jsonl", "ivfpq8.faiss", "flat.faiss", "corpus.jsonl",
        "queries.jsonl", "manifest.json", "metrics.json", "flat_top100.npz", "qrels/test.tsv")]
    files += [base / name for name in ("encoder_config.json", "model_manifest.json", "summary.json")]
    model_manifest = json.loads((base / "model_manifest.json").read_text())
    encoder = json.loads((base / "encoder_config.json").read_text())
    if (model_manifest["revision"] != REVISION or encoder["revision"] != REVISION
            or not encoder["normalize_embeddings"]):
        raise ValueError("encoder provenance mismatch")
    for entry in model_manifest["files"]:
        path = (base / entry["path"]).resolve()
        if not path.is_relative_to(base) or digest(path) != entry["sha256"]:
            raise ValueError("model hash mismatch")
        files.append(path)
    pins = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)} for p in files]
    write_json(output / "input_hashes.json", {"algorithm": "sha256", "files": pins,
               "model": model_manifest["model"], "revision": REVISION})
    faiss.omp_set_num_threads(1)
    xb = np.load(root / "document_embeddings.npy", mmap_mode="r", allow_pickle=False)
    xq = np.load(root / "test_query_embeddings.npy", allow_pickle=False)
    validate_vectors(xb)
    validate_vectors(xq)
    if xb.shape != (57638, 768) or xq.shape != (648, 768):
        raise ValueError("expected complete FiQA corpus and 648 test queries")
    dids = load_mapping(root / "document_ids.jsonl", "document_id", len(xb))
    qids = load_mapping(root / "test_query_ids.jsonl", "query_id", len(xq))
    for path, ids in (("corpus.jsonl", dids), ("queries.jsonl", qids)):
        with (root / path).open() as stream:
            official = [str(json.loads(line)["_id"]) for line in stream]
        if len(set(official)) != len(official) or not set(ids) <= set(official):
            raise ValueError("mapping does not reference unique official IDs")
    index = faiss.read_index(str(root / "ivfpq8.faiss"))
    flat = faiss.read_index(str(root / "flat.faiss"))
    if not isinstance(index, faiss.IndexIVFPQ) or not isinstance(flat, faiss.IndexFlatIP):
        raise ValueError("wrong index types")
    for idx in (index, flat):
        if idx.d != 768 or idx.ntotal != len(xb) or idx.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("index dimensions/cardinality/metric mismatch")
    if index.nlist != 64 or index.pq.M != 96 or index.pq.nbits != 8:
        raise ValueError("unexpected FiQA PQ configuration")
    # Verify the loaded Flat index is exactly the pinned embedding matrix.
    for start in range(0, len(xb), 2048):
        if not np.array_equal(flat.reconstruct_n(start, min(2048, len(xb) - start)),
                              xb[start:start + 2048]):
            raise ValueError("Flat vectors differ from document embeddings")
    flat_rows, flat_scores = [], []
    flat_seconds = 0.
    for query in xq:
        t = time.perf_counter()
        scores, ids = flat.search(query[None, :], 11)
        flat_seconds += time.perf_counter() - t
        if scores[0, 9] == scores[0, 10]:
            # Resolve all boundary ties using the same ascending-row rule.
            t = time.perf_counter()
            scores, ids = flat.search(query[None, :], len(xb))
            flat_seconds += time.perf_counter() - t
        candidate_ids(ids[0], len(xb))
        rows, values = rank(ids[0], scores[0])
        flat_rows.append(rows)
        flat_scores.append(values)
    flat_rows, flat_scores = np.array(flat_rows), np.array(flat_scores)
    save_rows(output / "flat.jsonl", qids, dids, flat_rows, flat_scores)
    retrieved = []
    validation = []
    for nprobe in (8, 16, 32):
        index.nprobe = nprobe
        for top_r in (32, 64, 128):
            raw_rows, raw_scores, final_rows, final_scores, all_candidates = [], [], [], [], []
            search_times, rerank_times = [], []
            for qi, query in enumerate(xq):
                t = time.perf_counter()
                scores, ids = index.search(query[None, :], top_r)
                search_times.append(time.perf_counter() - t)
                unique = candidate_ids(ids[0], len(xb))
                if len(unique) != top_r:
                    raise ValueError("duplicate Faiss candidates in actual pilot")
                approx_rows, approx_scores = rank(ids[0], scores[0])
                t = time.perf_counter()
                rows, values = rerank(query, xb, ids[0])
                rerank_times.append(time.perf_counter() - t)
                if qi == 0:
                    # Independent scalar-per-document float64 numpy.dot oracle.
                    reference = np.array([np.dot(xb[j].astype(np.float64),
                                                query.astype(np.float64)) for j in unique])
                    oracle_rows, oracle_scores = rank(unique, reference)
                    error = float(np.max(np.abs(values.astype(np.float64) - oracle_scores)))
                    if not np.array_equal(rows, oracle_rows) or error > 1e-6:
                        raise ValueError("independent numpy dot scoring check failed")
                    validation.append({"nprobe": nprobe, "top_r": top_r, "query_id": qids[0],
                                       "max_abs_error": error, "rows_match": True,
                                       "oracle_rows": oracle_rows.tolist(),
                                       "oracle_scores": oracle_scores.tolist()})
                raw_rows.append(approx_rows)
                raw_scores.append(approx_scores)
                final_rows.append(rows)
                final_scores.append(values)
                all_candidates.append(ids[0])
            tag = "nprobe{}_R{}".format(nprobe, top_r)
            raw_rows, final_rows = np.array(raw_rows), np.array(final_rows)
            save_rows(output / (tag + "_pq.jsonl"), qids, dids, raw_rows, raw_scores)
            save_rows(output / (tag + "_rerank.jsonl"), qids, dids, final_rows,
                      final_scores, all_candidates)
            retrieved.append((tag, nprobe, top_r, raw_rows, final_rows, search_times, rerank_times))
    # Evaluation is deliberately downstream of every candidate/rank decision.
    qrels = load_qrels(root / "qrels/test.tsv", qids, dids)
    metrics = {"flat": {**evaluate(flat_rows, flat_rows, dids, qids, qrels),
                         "cpu_search_seconds": flat_seconds}}
    for tag, nprobe, top_r, raw, final, search_times, rerank_times in retrieved:
        metrics[tag] = {"nprobe": nprobe, "top_r": top_r,
                        "pq": evaluate(raw, flat_rows, dids, qids, qrels),
                        "rerank": evaluate(final, flat_rows, dids, qids, qrels),
                        "cpu_search_seconds": sum(search_times),
                        "cpu_exact_rerank_seconds": sum(rerank_times),
                        "per_query_cpu_search_seconds": search_times,
                        "per_query_cpu_exact_rerank_seconds": rerank_times}
    for pin in pins:
        if digest(pin["path"]) != pin["sha256"]:
            raise ValueError("input changed during run")
    write_json(output / "scoring_validation.json", validation)
    result = {"dataset": "fiqa", "queries": len(xq), "configurations": 9,
              "selection": "query-level Faiss IVFPQ top-R; qrels evaluation only",
              "tie_rule": "descending score then ascending embedding row ID",
              "ndcg_gain": "linear relevance (BEIR/trec)", "metrics": metrics,
              "timing_scope": "single pass, query-at-a-time CPU search and CPU rerank; not FPGA latency",
              "DES_integration": False, "hardware_execution": False,
              "environment": {"python": sys.version, "numpy": np.__version__,
                              "faiss": faiss.__version__, "platform": platform.platform(),
                              "cpu": platform.processor(), "faiss_threads": 1,
                              "thread_env": {k: os.environ.get(k) for k in
                                             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
              "input_hashes_verified_unchanged": True}
    write_json(output / "metrics.json", result)
    write_json(output / "output_hashes.json", [
        {"path": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir()) if p.is_file()])
    print(json.dumps({"status": "complete", "output": str(output), "queries": len(xq),
                      "configurations": 9}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.input, args.output)
