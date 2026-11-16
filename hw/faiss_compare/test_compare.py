import tempfile
import unittest
from pathlib import Path

import faiss
import numpy as np

import compare


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1)
        rng = np.random.default_rng(41)
        quantizer = faiss.IndexFlatL2(768)
        centers = rng.normal(size=(4, 768)).astype('float32')
        quantizer.add(centers)
        cls.index = faiss.IndexIVFPQ(quantizer, 768, 4, 96, 8)
        cb = rng.normal(size=96*256*8).astype('float32')
        faiss.copy_array_to_vector(cb, cls.index.pq.centroids)
        cls.index.is_trained = True
        cls.index.by_residual = True
        for lid, count in enumerate((12, 0, 15, 9)):
            if not count:
                continue
            ids = np.arange(count, dtype='int64')+1000*(lid+1)
            codes = rng.integers(0, 256, (count, 96), dtype='uint8')
            cls.index.invlists.add_entries(lid, count, faiss.swig_ptr(ids), faiss.swig_ptr(codes))
            cls.index.ntotal += count
        cls.queries = rng.normal(size=(2, 768)).astype('float32')
        cls.lists = np.array([[0, 1, 2], [2, 3, 1]], dtype='int64')

    def test_residual_multi_list_and_pack(self):
        data = compare.prepare(self.index, self.queries, self.lists, 3)
        distances, ids = compare.validate(self.index, data)
        self.assertEqual(ids.shape, (2, 3))
        np.testing.assert_array_equal(data['list_ids'], self.lists)
        raw = compare.pack_cluster(data['codes'], data['ids'])
        records = np.frombuffer(raw[64:], dtype='uint8').reshape(-1, 128)
        np.testing.assert_array_equal(records[:, :64][:, ::-1], data['codes'][:, :64])
        np.testing.assert_array_equal(records[:, 96:][:, ::-1], data['codes'][:, 64:])
        mock = compare.host.MockTransport(statuses=(4, 4, 4))
        sessions = compare.stage(mock, data, abi=compare.host.ABI)
        def search(query, k, **kwargs):
            qi, probe = next(search.calls)
            ri, rd = compare.candidates(data, qi, probe)
            order = np.argsort(rd)[:k]
            return [compare.host.Result(float(rd[i]), int(ri[i]), 0) for i in order], {}
        search.calls = iter((qi, p) for qi, row in enumerate(self.lists) for p, lid in enumerate(row)
                            if data['offsets'][lid] != data['offsets'][lid+1])
        for session in sessions.values():
            session.search = search
        output, timing = compare.matched_fpga_search(sessions, data)
        self.assertEqual(timing['transport_kind'], 'mock')
        self.assertIsNone(timing['speedup'])
        for row, expected in zip(output, ids):
            self.assertEqual({r.address for r in row}, set(expected))

    def test_bundle_tamper(self):
        data = compare.prepare(self.index, self.queries, self.lists, 3)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'matched.npz'
            np.savez(path, **data)
            compare.load_bundle(path)
            data['scan_queries'][0, 0, 0] += 1
            np.savez(path, **data)
            with self.assertRaises(ValueError):
                compare.load_bundle(path)


if __name__ == '__main__':
    unittest.main()
