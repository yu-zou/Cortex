"""Real IVF-PQ export and correctness-gated resident CPU comparison.

CLI commands perform file/software operations only. Hardware orchestration is
an API accepting an owner-created transport, with explicit historical ABI.
"""
import argparse
import hashlib
import json
import struct
import time
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'host_benchmark'))
import host
import layout

M, DIM, KS, DS = 96, 768, 256, 8
RTOL, ATOL = 1e-5, 1e-3


def extract(index):
    if type(index) is not faiss.IndexIVFPQ:
        raise ValueError('plain IndexIVFPQ required (no transforms/refinement)')
    if (index.d, index.pq.M, index.pq.nbits, index.metric_type) != (DIM, M, 8, faiss.METRIC_L2):
        raise ValueError('only D=768 M=96 nbits=8 squared L2 supported')
    if type(faiss.downcast_index(index.quantizer)) is not faiss.IndexFlatL2:
        raise ValueError('plain L2 coarse quantizer required')
    if not index.is_trained or index.max_codes or index.polysemous_ht:
        raise ValueError('trained full scan required; max_codes/polysemous filtering unsupported')
    cb = faiss.vector_to_array(index.pq.centroids).reshape(M, KS, DS).copy()
    centroids = np.stack([index.quantizer.reconstruct(i) for i in range(index.nlist)])
    codes, ids, offsets = [], [], [0]
    for lid in range(index.nlist):
        n = index.invlists.list_size(lid)
        if n > 8192:
            raise ValueError('list exceeds hardware limit of 8192 candidates')
        if n:
            cp, ip = index.invlists.get_codes(lid), index.invlists.get_ids(lid)
            try:
                codes.append(faiss.rev_swig_ptr(cp, n*M).reshape(n, M).copy())
                ids.append(faiss.rev_swig_ptr(ip, n).copy())
            finally:
                index.invlists.release_codes(lid, cp)
                index.invlists.release_ids(lid, ip)
        offsets.append(offsets[-1]+n)
    if not ids:
        raise ValueError('empty index')
    codes, ids = np.concatenate(codes), np.concatenate(ids)
    if len(ids) != index.ntotal or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
        raise ValueError('unique nonnegative IDs and complete lists required')
    if not np.isfinite(cb).all() or not np.isfinite(centroids).all():
        raise ValueError('non-finite codebook/centroid')
    return cb, centroids, codes, ids, np.asarray(offsets, dtype=np.int64)


def pack_cluster(codes, ids, cb=None):
    """Pack the actual bit slices consumed by adc_engine, not stale comments."""
    records = np.zeros((len(ids), 128), dtype=np.uint8)
    records[:, :64] = codes[:, :64][:, ::-1]
    records[:, 96:] = codes[:, 64:][:, ::-1]
    records[:, 88:96] = np.asarray(ids, dtype='<u8').view(np.uint8).reshape(-1, 8)
    # doc_len is metadata only; zero is used for exported Faiss IDs.
    header = struct.pack('<III', M, DIM, len(ids))+bytes(52)
    return header+(b'' if cb is None else cb.astype('<f4').tobytes())+records.tobytes()


def prepare(index, queries, list_ids, k):
    cb, centroids, codes, ids, offsets = extract(index)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if not np.issubdtype(np.asarray(list_ids).dtype, np.integer):
        raise ValueError('list IDs must be integers')
    if np.asarray(list_ids).dtype.kind not in 'iu':
        raise ValueError('integer list IDs required')
    list_ids = np.ascontiguousarray(list_ids, dtype=np.int64)
    if queries.ndim != 2 or queries.shape[1] != DIM or not len(queries) or not np.isfinite(queries).all():
        raise ValueError('finite nonempty Nx768 queries required')
    if list_ids.ndim != 2 or list_ids.shape[0] != len(queries) or not 1 <= list_ids.shape[1] <= index.nlist:
        raise ValueError('NxNprobe list IDs required')
    if np.any(list_ids < 0) or np.any(list_ids >= index.nlist) or any(len(np.unique(row)) != len(row) for row in list_ids):
        raise ValueError('list IDs must be valid and unique per query')
    if not 1 <= k <= 500 or any(sum(offsets[l+1]-offsets[l] for l in row) < k for row in list_ids):
        raise ValueError('invalid k or too few preassigned candidates')
    coarse = np.sum((queries[:, None, :]-centroids[list_ids])**2, axis=2, dtype=np.float32)
    residual_queries = queries[:, None, :]-centroids[list_ids] if index.by_residual else np.broadcast_to(queries[:, None, :], (*list_ids.shape, DIM)).copy()
    return dict(index=faiss.serialize_index(index), queries=queries, list_ids=list_ids,
                coarse=np.ascontiguousarray(coarse), scan_queries=np.ascontiguousarray(residual_queries),
                codebook=cb, centroids=centroids, codes=codes, ids=ids, offsets=offsets,
                k=np.asarray(k), by_residual=np.asarray(bool(index.by_residual)))


def candidates(data, qi, probe):
    lid = data['list_ids'][qi, probe]
    a, b = data['offsets'][lid:lid+2]
    codes, ids = data['codes'][a:b], data['ids'][a:b]
    delta = data['scan_queries'][qi, probe].astype(np.float64).reshape(M, 1, DS)-data['codebook'].astype(np.float64)
    table = np.sum(delta*delta, axis=2)
    scores = table[np.arange(M)[None, :], codes].sum(axis=1)
    return ids, scores


def check_top(ids, distances, ref_ids, ref_distances, k):
    order = np.lexsort((ref_ids, ref_distances))
    if k < len(order) and np.isclose(ref_distances[order[k-1]], ref_distances[order[k]], rtol=RTOL, atol=ATOL):
        raise ValueError('top-k boundary numerically ambiguous')
    expected = {int(ref_ids[i]): float(ref_distances[i]) for i in order[:k]}
    if len(ids) != k or len(set(map(int, ids))) != k or set(map(int, ids)) != set(expected):
        raise ValueError('top-k IDs disagree with independent ADC')
    if not np.isfinite(distances).all() or not np.allclose(distances, [expected[int(i)] for i in ids], rtol=RTOL, atol=ATOL):
        raise ValueError('distances disagree with independent ADC')
    if np.any(np.diff(distances) < 0):
        raise ValueError('results not sorted')


def load_bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    index = faiss.deserialize_index(data['index'])
    # Re-derive everything from the serialized index, including residual policy.
    expected = prepare(index, data['queries'], data['list_ids'], int(data['k']))
    if set(data) != set(expected) or any(not np.array_equal(data[key], value) for key, value in expected.items()):
        raise ValueError('bundle/index mismatch')
    return index, data


def search(index, data):
    index.nprobe = data['list_ids'].shape[1]
    return index.search_preassigned(data['queries'], int(data['k']), data['list_ids'], data['coarse'])


def validate(index, data):
    distances, ids = search(index, data)
    for qi in range(len(ids)):
        parts = [candidates(data, qi, p) for p in range(data['list_ids'].shape[1])]
        ri, rd = (np.concatenate(values) for values in zip(*parts))
        check_top(ids[qi], distances[qi], ri, rd, int(data['k']))
    return distances, ids


def stage(transport, data, *, abi=None):
    """Stage all selected lists once, before timed resident searches.

    Calls must be serialized: sessions share query/result buffers and codebook.
    """
    if abi != host.ABI:
        raise PermissionError('hardware ABI unverified')
    index = faiss.deserialize_index(data['index'])
    expected = prepare(index, data['queries'], data['list_ids'], int(data['k']))
    if set(data) != set(expected) or any(not np.array_equal(data[key], value) for key, value in expected.items()):
        raise ValueError('bundle/index mismatch')
    validate(index, data)
    plans, address = [], host.CLUSTER
    for lid in np.unique(data['list_ids']):
        a, b = data['offsets'][lid:lid+2]
        if a == b:
            continue
        raw = pack_cluster(data['codes'][a:b], data['ids'][a:b])
        plans.append((int(lid), address, raw))
        address += len(raw)
    if address > host.CODEBOOK:
        raise ValueError('selected lists exceed reserved FPGA DDR region')
    sessions = {lid: host.Session(transport, raw, data['codebook'].astype('<f4').tobytes(),
                                 abi=abi, cluster_address=address) for lid, address, raw in plans}
    return sessions


def matched_fpga_search(sessions, data):
    """Scan identical lists, merge local top-k and validate before returning.

    There is no speedup output here: observed polling time is not kernel time.
    """
    begin = time.perf_counter_ns()
    output, timings = [], []
    k = int(data['k'])
    for qi, row in enumerate(data['list_ids']):
        merged = []
        for probe, lid in enumerate(row):
            a, b = data['offsets'][lid:lid+2]
            if a == b:
                continue
            results, timing = sessions[int(lid)].search(data['scan_queries'][qi, probe].astype('<f4').tobytes(), min(k, b-a))
            ri, rd = candidates(data, qi, probe)
            check_top([r.address for r in results], [r.distance for r in results], ri, rd, min(k, b-a))
            if any(r.length != 0 for r in results):
                raise ValueError('unexpected result metadata')
            merged.extend(results)
            timings.append(timing)
        merged.sort(key=lambda r: (r.distance, r.address))
        parts = [candidates(data, qi, p) for p in range(len(row))]
        ri, rd = (np.concatenate(values) for values in zip(*parts))
        check_top([r.address for r in merged[:k]], [r.distance for r in merged[:k]], ri, rd, k)
        output.append(merged[:k])
    return output, {'transport_kind': next(iter(sessions.values())).transport.kind,
                    'per_list': timings, 'validated_batch_end_to_end_ms': (time.perf_counter_ns()-begin)/1e6,
                    'kernel_ms': None, 'speedup': None}


def export(index_path, query_path, output, k, nprobe, list_path=None):
    index = faiss.read_index(str(index_path))
    queries = np.load(query_path, allow_pickle=False)
    if list_path is None:
        if not 1 <= nprobe <= index.nlist:
            raise ValueError('invalid nprobe')
        _, list_ids = index.quantizer.search(np.ascontiguousarray(queries, dtype=np.float32), nprobe)
    else:
        list_ids = np.load(list_path, allow_pickle=False)
    data = prepare(index, queries, list_ids, k)
    validate(index, data)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output/'matched.npz', **data)
    (output/'codebook.bin').write_bytes(data['codebook'].astype('<f4').tobytes())
    for qi, row in enumerate(data['list_ids']):
        (output/f'query_{qi}.bin').write_bytes(data['queries'][qi].astype('<f4').tobytes())
        for probe, lid in enumerate(row):
            (output/f'query_{qi}.list_{lid}.bin').write_bytes(data['scan_queries'][qi, probe].astype('<f4').tobytes())
    for lid in np.unique(list_ids):
        a, b = data['offsets'][lid:lid+2]
        for name, cb in (('historical', None), ('source-inline', data['codebook'])):
            (output/f'list_{lid}.{name}.bin').write_bytes(pack_cluster(data['codes'][a:b], data['ids'][a:b], cb))
        if a != b:
            (output/f'list_{lid}.cxivfpq1').write_bytes(layout.encode(int(lid),
                pack_cluster(data['codes'][a:b], data['ids'][a:b]),
                data['codebook'].astype('<f4').tobytes(), data['centroids'][lid].astype('<f4').tobytes(),
                residual=bool(index.by_residual)))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    (output/'manifest.json').write_text(json.dumps({'schema': 1, 'metric': 'squared L2',
        'by_residual': bool(index.by_residual), 'k': k, 'list_ids': list_ids.tolist(),
        'faiss_version': faiss.__version__, 'files_sha256': files,
        'list_envelope': 'CXIVFPQ1', 'id_semantics': 'Faiss vector IDs; not document addresses',
        'historical_abi': host.ABI, 'source_inline_control': 'BLOCKED: generated register map required'}, indent=2)+'\n')


def benchmark(path, threads=1, warmup=3, repetitions=20):
    if threads < 1 or warmup < 0 or repetitions < 1:
        raise ValueError('invalid timing parameters')
    faiss.omp_set_num_threads(threads)
    start = time.perf_counter_ns()
    index, data = load_bundle(path)
    setup = (time.perf_counter_ns()-start)/1e6
    start = time.perf_counter_ns()
    distances, ids = validate(index, data)
    correctness = (time.perf_counter_ns()-start)/1e6
    for _ in range(warmup):
        search(index, data)
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        search(index, data)
        timings.append((time.perf_counter_ns()-start)/1e6)
    return {'scope': 'resident CPU IVF-PQ search_preassigned; identical exported candidates',
            'bundle_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'faiss_version': faiss.__version__, 'threads': threads, 'queries': len(data['queries']),
            'k': int(data['k']), 'nprobe': data['list_ids'].shape[1],
            'metric': 'squared L2', 'by_residual': bool(data['by_residual']),
            'correctness': 'passed independent float64 ADC and top-k', 'setup_ms': setup,
            'correctness_ms': correctness, 'transfer_ms': 0,
            'resident_batch_search_ms': {'min': min(timings), 'median': float(np.median(timings)),
                'p95': float(np.percentile(timings, 95)), 'repetitions': repetitions, 'warmup': warmup},
            'timing_scope': 'Python call through return including output allocation; coarse selection excluded',
            'fpga_kernel_ms': None, 'fpga_end_to_end_ms': None, 'speedup': None,
            'ids': ids.tolist(), 'distances': distances.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('--index', type=Path, required=True)
    exp.add_argument('--queries', type=Path, required=True)
    exp.add_argument('--list-ids', type=Path)
    exp.add_argument('--output', type=Path, required=True)
    exp.add_argument('--k', type=int, default=10)
    exp.add_argument('--nprobe', type=int, default=1)
    bench = sub.add_parser('benchmark')
    bench.add_argument('--bundle', type=Path, required=True)
    bench.add_argument('--threads', type=int, default=1)
    bench.add_argument('--warmup', type=int, default=3)
    bench.add_argument('--repetitions', type=int, default=20)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.index, args.queries, args.output, args.k, args.nprobe, args.list_ids)
    else:
        print(json.dumps(benchmark(args.bundle, args.threads, args.warmup, args.repetitions), indent=2))


if __name__ == '__main__':
    main()
