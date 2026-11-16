#!/usr/bin/env python3
"""Actual eight-argument ABI2 residual-IP validation; CPU offline evidence only."""
import argparse
import ctypes as C
import json
from pathlib import Path
import time

import numpy as np

from fiqa_cpucompact import cluster_ids, digest, mapping, parse_list, write_json


class Entry(C.Structure):
    _fields_ = [('distance', C.c_float), ('doc_addr', C.c_uint64),
                ('doc_length', C.c_uint32), ('pad', C.c_uint32)]


class Result(C.Structure):
    _fields_ = [('status', C.c_int32), ('count', C.c_uint32), ('entries', Entry * 500)]


class Driver:
    def __init__(self, library, mock):
        assert C.sizeof(Entry) == 24 and C.sizeof(Result) == 12008
        self.lib = C.CDLL(str(Path(library).resolve()))
        self.lib.cortex_api_version.restype = C.c_uint32
        if self.lib.cortex_api_version() != 2:
            raise ValueError('requires ABI2')
        self.lib.cortex_driver_init.argtypes = [C.c_char_p]
        self.lib.cortex_driver_init.restype = C.c_int
        self.lib.cortex_driver_shutdown.argtypes = []
        self.lib.cortex_driver_shutdown.restype = None
        self.lib.cortex_semantic_read.argtypes = [C.c_uint32, C.c_void_p, C.c_uint64,
            C.POINTER(C.c_float), C.c_uint32, C.c_uint32, C.c_uint32, C.c_uint32]
        self.lib.cortex_semantic_read.restype = Result
        self.lib.cortex_is_cluster_cached.argtypes = [C.c_uint32]
        self.lib.cortex_is_cluster_cached.restype = C.c_int
        config = json.dumps({'hw_lib_path': str(Path(mock).resolve()),
                             'max_cache_slots': 64}).encode()
        if self.lib.cortex_driver_init(config) != 0:
            raise RuntimeError('driver init failed')

    def read(self, cid, blob, query, k=10):
        buffer = C.create_string_buffer(blob)
        query = np.ascontiguousarray(query, dtype=np.float32)
        return self.lib.cortex_semantic_read(cid, buffer, len(blob),
            query.ctypes.data_as(C.POINTER(C.c_float)), len(query), k, 1, 0)

    def close(self):
        self.lib.cortex_driver_shutdown()


def validate(data, fixture, library, mock, output, count):
    import faiss
    if count < 10 or count > 648:
        raise ValueError('validate first 10 through 648 official queries')
    started = time.monotonic()
    data, fixture = Path(data), Path(fixture)
    manifest = json.loads((fixture / 'manifest.json').read_text())
    for name, record in manifest['inputs'].items():
        if digest(data / name) != record['sha256']:
            raise ValueError('input hash mismatch: ' + name)
    expected_cids = cluster_ids(manifest['identity'])
    if (len(manifest['objects']) != 64 or
            [o['list'] for o in manifest['objects']] != list(range(64)) or
            [o['cluster_id'] for o in manifest['objects']] != expected_cids):
        raise ValueError('cache namespace/list collision or mismatch')
    blobs, all_ids = [], []
    reconstructed = np.empty((manifest['ntotal'], 768), dtype=np.float32)
    list_ids = []
    for obj in manifest['objects']:
        path = fixture / obj['file']
        if digest(path) != obj['sha256']:
            raise ValueError('object byte hash mismatch')
        blob = path.read_bytes()
        cb, center, codes, ids = parse_list(blob, manifest['ntotal'])
        # Independent full-vector reconstruction, not LUT accumulation.
        reconstructed[ids] = cb[np.arange(96)[None, :], codes].reshape(-1, 768) + center
        blobs.append(blob)
        list_ids.append(ids)
        all_ids.extend(ids.tolist())
    if sorted(all_ids) != list(range(manifest['ntotal'])):
        raise ValueError('cross-list duplicate or missing global row token')
    index = faiss.read_index(str(data / 'ivfpq8.faiss'))
    if not index.by_residual or index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError('requires residual IP')
    index.nprobe = 32
    faiss.omp_set_num_threads(1)
    queries = np.load(data / 'test_query_embeddings.npy', allow_pickle=False)[:count]
    qids = mapping(data / 'test_query_ids.jsonl', 'query_id', 648)
    coarse_scores, probes = index.quantizer.search(queries, 32)
    fs, fi = index.search_preassigned(queries, 10, probes, coarse_scores)
    ordinary_scores, ordinary_ids = index.search(queries, 10)
    if not np.array_equal(fi, ordinary_ids) or not np.allclose(fs, ordinary_scores, atol=1e-5, rtol=1e-5):
        raise AssertionError('Faiss preassigned and ordinary search disagree')
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    driver = Driver(library, mock)
    rows, max_faiss, max_recon, scanned, calls, cold = [], 0., 0., 0, 0, 0
    try:
        for qi, query in enumerate(queries):
            hits, candidates = [], []
            for lid in probes[qi]:
                lid = int(lid)
                cid = expected_cids[lid]
                if not driver.lib.cortex_is_cluster_cached(cid):
                    cold += 1
                result = driver.read(cid, blobs[lid], query)
                calls += 1
                if result.status or result.count != min(10, len(list_ids[lid])):
                    raise AssertionError('C ABI status/count mismatch: %s/%s' % (result.status, result.count))
                for entry in result.entries[:result.count]:
                    if (entry.doc_addr not in list_ids[lid] or entry.doc_length or entry.pad or
                            not np.isfinite(entry.distance)):
                        raise AssertionError('invalid C ABI row token/metadata/score')
                    hits.append((float(entry.distance), int(entry.doc_addr)))
                candidates.append(list_ids[lid])
            ids = np.concatenate(candidates)
            scanned += len(ids)
            negative_ip = -(reconstructed[ids].astype(np.float64) @ query.astype(np.float64))
            order = np.lexsort((ids, negative_ip))
            expected = ids[order[:10]]
            merged = sorted(hits)[:10]
            got = np.array([x[1] for x in merged])
            distances = np.array([x[0] for x in merged])
            oracle = negative_ip[order[:10]]
            # Boundary ties are recorded; a differing ID order is still a failure.
            boundary_gap = float(negative_ip[order[10]] - oracle[-1])
            if not np.array_equal(got, fi[qi]) or not np.array_equal(got, expected):
                raise AssertionError('top10 ID mismatch query %d; boundary gap %g' % (qi, boundary_gap))
            if not np.allclose(distances, -fs[qi], atol=1e-5, rtol=1e-5):
                raise AssertionError('Faiss score tolerance exceeded')
            if not np.allclose(distances, oracle, atol=1e-5, rtol=1e-5):
                raise AssertionError('independent reconstruction score tolerance exceeded')
            # Check every returned list-local score too, not only global winners.
            hit_ids = np.array([x[1] for x in hits])
            hit_scores = np.array([x[0] for x in hits])
            hit_oracle = -(reconstructed[hit_ids].astype(np.float64) @ query.astype(np.float64))
            if not np.allclose(hit_scores, hit_oracle, atol=1e-5, rtol=1e-5):
                raise AssertionError('list-local reconstruction score mismatch')
            max_faiss = max(max_faiss, float(np.max(np.abs(distances + fs[qi]))))
            max_recon = max(max_recon, float(np.max(np.abs(hit_scores - hit_oracle))))
            rows.append({'row': qi, 'query_id': qids[qi], 'probes': probes[qi].tolist(),
                         'candidate_count': len(ids), 'ids': got.tolist(),
                         'negative_ip': distances.tolist(), 'faiss_ip': fs[qi].tolist(),
                         'reconstruction_negative_ip': oracle.tolist(),
                         'boundary_gap': boundary_gap})
    finally:
        driver.close()
    # Immutable fixture checks after all warm cache reads.
    for obj in manifest['objects']:
        if digest(fixture / obj['file']) != obj['sha256']:
            raise ValueError('object changed during cached run')
    for name, record in manifest['inputs'].items():
        if digest(data / name) != record['sha256']:
            raise ValueError('input changed during validation: ' + name)
    summary = {'passed': True, 'scope': 'offline CPU mock actual ABI2; no Ceph/hardware/cloud',
               'queries': count, 'first10_verified': True, 'nprobe': 32, 'k': 10,
               'abi_calls': calls, 'cold_loads': cold, 'warm_reads': calls - cold,
               'candidate_scores_reconstructed': scanned,
               'exact_top10_id_matches_faiss': count,
               'exact_top10_id_matches_reconstruction': count,
               'max_abs_error_vs_faiss': max_faiss,
               'max_abs_error_all_list_hits_vs_reconstruction': max_recon,
               'tolerance': 'abs_err <= 1e-5 + 1e-5*abs(reference)',
               'score_direction': 'Cortex min negative IP; Faiss max positive IP',
               'cache_contract': 'fresh driver; immutable fixture; all 64 unique hash-namespaced IDs; hits ignore changed object bytes',
               'fixture_manifest_sha256': digest(fixture / 'manifest.json'),
               'driver_sha256': digest(library), 'mock_sha256': digest(mock),
               'faiss_version': faiss.__version__, 'elapsed_seconds': time.monotonic() - started}
    write_json(output / 'queries.json', rows)
    summary['queries_sha256'] = digest(output / 'queries.json')
    write_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['data', 'fixture', 'library', 'mock', 'output']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--queries', type=int, default=10)
    args = parser.parse_args()
    validate(args.data, args.fixture, args.library, args.mock, args.output, args.queries)
