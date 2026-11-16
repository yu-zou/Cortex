#!/usr/bin/env python3
"""Pack an existing residual-IP FiQA IVFPQ index into CPU-only CPQ1v2.

doc_addr is explicitly an embedding row token; doc_length=0. These objects
cannot be used as text byte-address objects or as 128-byte FPGA records.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def mapping(path, key, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(rows) != count or any(type(r.get('row')) is not int or r['row'] != n
                                 for n, r in enumerate(rows)):
        raise ValueError('mapping cardinality/order mismatch')
    ids = [r.get(key) for r in rows]
    if any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != count:
        raise ValueError('invalid or duplicate mapping IDs')
    return ids


def pack_list(codebook, center, codes, ids, total):
    cb = np.asarray(codebook, dtype='<f4')
    center = np.asarray(center, dtype='<f4')
    codes, ids = np.asarray(codes), np.asarray(ids)
    if cb.shape != (96, 256, 8) or center.shape != (768,):
        raise ValueError('expected M96/KS256/D768')
    if not np.isfinite(cb).all() or not np.isfinite(center).all():
        raise ValueError('nonfinite centroid')
    if codes.dtype != np.uint8 or codes.shape != (len(ids), 96):
        raise ValueError('invalid residual codes')
    if ids.dtype.kind not in 'iu' or np.any(ids < 0) or np.any(ids >= total):
        raise ValueError('global row ID out of range')
    if len(np.unique(ids)) != len(ids):
        raise ValueError('duplicate global row ID')
    header = struct.pack('<7I', 96, 768, len(ids), 0x31515043, 2, 256, 1) + bytes(36)
    records = np.zeros(len(ids), dtype=[('codes', 'u1', (96,)),
                                       ('row_token', '<u8'), ('length', '<u8')])
    assert records.dtype.itemsize == 112
    records['codes'], records['row_token'] = codes, ids
    return header + cb.tobytes() + center.tobytes() + records.tobytes()


def parse_list(blob, total):
    if len(blob) < 64:
        raise ValueError('truncated header')
    m, d, n, magic, version, ks, flags = struct.unpack_from('<7I', blob)
    if (m, d, magic, version, ks, flags) != (96, 768, 0x31515043, 2, 256, 1):
        raise ValueError('unsupported CPU compact format/version')
    if any(blob[28:64]) or len(blob) != 789568 + n * 112:
        raise ValueError('truncated/trailing data or reserved header bytes')
    cb = np.frombuffer(blob, '<f4', 96 * 256 * 8, 64).reshape(96, 256, 8)
    center = np.frombuffer(blob, '<f4', 768, 786496)
    records = np.frombuffer(blob, dtype=[('codes', 'u1', (96,)),
                                        ('row_token', '<u8'), ('length', '<u8')],
                            count=n, offset=789568)
    ids = records['row_token']
    if (np.any(ids >= total) or len(np.unique(ids)) != n or
            np.any(records['length'] != 0)):
        raise ValueError('invalid row token or nonzero fixture text length')
    if not np.isfinite(cb).all() or not np.isfinite(center).all():
        raise ValueError('nonfinite centroid')
    return cb, center, records['codes'], ids.astype(np.int64)


def cluster_ids(identity):
    # Full SHA256 is authoritative; 32-bit C ABI keys are collision checked.
    result = [int.from_bytes(hashlib.sha256(
        ('fiqa-cpq1v2-row-token:' + identity + ':' + str(i)).encode()).digest()[:4],
        'little') for i in range(64)]
    if len(set(result)) != 64:
        raise ValueError('32-bit cache namespace collision')
    return result


def pack(data, output):
    import faiss
    data, output = Path(data), Path(output)
    names = ['ivfpq8.faiss', 'test_query_embeddings.npy', 'document_ids.jsonl',
             'test_query_ids.jsonl', 'qrels/test.tsv', 'manifest.json']
    inputs = {n: {'sha256': digest(data / n), 'bytes': (data / n).stat().st_size}
              for n in names}
    index = faiss.read_index(str(data / names[0]))
    if (type(index).__name__ != 'IndexIVFPQ' or not index.is_trained or
            (index.d, index.nlist, index.pq.M, index.pq.nbits, index.pq.ksub,
             index.pq.dsub, index.code_size, index.metric_type) !=
            (768, 64, 96, 8, 256, 8, 96, faiss.METRIC_INNER_PRODUCT) or
            not index.by_residual or index.ntotal != 57638):
        raise ValueError('not the expected actual FiQA residual-IP index')
    queries = np.load(data / names[1], allow_pickle=False)
    if queries.dtype != np.float32 or queries.shape != (648, 768):
        raise ValueError('unexpected official test query matrix')
    norms = np.linalg.norm(queries, axis=1)
    if not np.isfinite(queries).all() or not np.allclose(norms, 1, atol=1e-5, rtol=0):
        raise ValueError('queries not finite/unit normalized')
    mapping(data / names[2], 'document_id', index.ntotal)
    qids = mapping(data / names[3], 'query_id', len(queries))
    official = sorted(set(line.split('\t')[0] for line in
                          (data / 'qrels/test.tsv').read_text().splitlines()[1:]))
    if qids != official:
        raise ValueError('query rows are not sorted official test IDs')
    cb = faiss.vector_to_array(index.pq.centroids).reshape(96, 256, 8)
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    cache_ids = cluster_ids(identity)
    # Refuse reuse/overwrite of any output directory.
    output.mkdir(parents=False, exist_ok=False)
    objects, seen = [], []
    for lid in range(64):
        inv = index.invlists
        n = inv.list_size(lid)
        ip, cp = inv.get_ids(lid), inv.get_codes(lid)
        try:
            ids = faiss.rev_swig_ptr(ip, n).copy()
            codes = faiss.rev_swig_ptr(cp, n * 96).copy().reshape(n, 96)
        finally:
            inv.release_ids(lid, ip)
            inv.release_codes(lid, cp)
        blob = pack_list(cb, index.quantizer.reconstruct(lid), codes, ids, index.ntotal)
        parse_list(blob, index.ntotal)
        path = output / ('list-%02d.cpq' % lid)
        with path.open('xb') as stream:
            stream.write(blob)
        seen.extend(ids.tolist())
        objects.append({'list': lid, 'cluster_id': cache_ids[lid], 'count': n,
                        'file': path.name, 'bytes': len(blob), 'sha256': digest(path)})
    if sorted(seen) != list(range(index.ntotal)):
        raise ValueError('global IDs must cover each document row exactly once')
    # Verify inputs did not change while packing.
    if any(digest(data / n) != v['sha256'] for n, v in inputs.items()):
        raise ValueError('input changed while packing')
    manifest = {'format': 'CPQ1v2-CPUcompact-residual-IP', 'inputs': inputs,
                'identity': identity, 'faiss_version': faiss.__version__,
                'dimension': 768, 'M': 96, 'KS': 256, 'nlist': 64,
                'by_residual': bool(index.by_residual), 'faiss_metric': int(index.metric_type),
                'cortex_metric': 1, 'ntotal': index.ntotal, 'record_bytes': 112,
                'records_offset': 789568, 'objects': objects,
                'doc_addr_semantics': 'global embedding row token, NOT a text byte address',
                'doc_length': 0, 'mapping': 'document_ids.jsonl row -> document_id',
                'normalization': 'queries verified unit L2; no renormalization applied; PQ reconstructions not normalized',
                'query_norm_range': [float(norms.min()), float(norms.max())],
                'first10_query_ids': qids[:10]}
    write_json(output / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pack(args.data, args.output)
