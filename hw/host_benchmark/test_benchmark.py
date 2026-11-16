import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

import compare
import host


def fixture(residual=True):
    rng = np.random.default_rng(13)
    quantizer = faiss.IndexFlatL2(compare.DIM)
    centroids = rng.normal(size=(3, compare.DIM)).astype('float32')
    quantizer.add(centroids)
    index = faiss.IndexIVFPQ(quantizer, compare.DIM, 3, compare.M, 8)
    cb = rng.normal(size=(compare.M, 256, 8)).astype('float32')
    faiss.copy_array_to_vector(cb.ravel(), index.pq.centroids)
    index.is_trained = True
    index.by_residual = residual
    for lid, n in enumerate((7, 9, 0)):
        if n:
            codes = rng.integers(0, 256, (n, compare.M), dtype=np.uint8)
            ids = np.arange(n, dtype=np.int64)+100*lid
            index.invlists.add_entries(lid, n, faiss.swig_ptr(ids), faiss.swig_ptr(codes))
            index.ntotal += n
    queries = rng.normal(size=(2, compare.DIM)).astype('float32')
    lists = np.asarray([[0, 1, 2], [1, 0, 2]], dtype=np.int64)
    return index, compare.prepare(index, queries, lists, 3)


def payload(ids, scores):
    return b''.join(struct.pack('<fQI', d, int(i), 0)+bytes(48) for i, d in zip(ids, scores))


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1)

    def test_residual_and_nonresidual_preassigned(self):
        for residual in (True, False):
            index, data = fixture(residual)
            compare.validate(index, data)
            expected = data['queries'][0]-data['centroids'][0] if residual else data['queries'][0]
            np.testing.assert_array_equal(data['scan_queries'][0, 0], expected)
            raw = compare.pack_cluster(data['codes'][:7], data['ids'][:7])
            records = np.frombuffer(raw, np.uint8, offset=64).reshape(7, 128)
            decoded = np.concatenate((records[:, :64][:, ::-1], records[:, 96:][:, ::-1]), axis=1)
            np.testing.assert_array_equal(decoded, data['codes'][:7])
            np.testing.assert_array_equal(np.frombuffer(records[:, 88:96].copy(), '<u8'), data['ids'][:7])
            inline = compare.pack_cluster(data['codes'][:7], data['ids'][:7], data['codebook'])
            self.assertEqual(inline[64:64+786432], data['codebook'].astype('<f4').tobytes())

    def test_export_roundtrip_cpu_timing_and_corruption(self):
        index, data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            faiss.write_index(index, str(root/'index.faiss'))
            np.save(root/'queries.npy', data['queries'])
            np.save(root/'lists.npy', data['list_ids'])
            compare.export(root/'index.faiss', root/'queries.npy', root/'export', 3, 3, root/'lists.npy')
            report = compare.benchmark(root/'export/matched.npz', warmup=1, repetitions=2)
            self.assertIsNone(report['speedup'])
            self.assertGreater(report['resident_batch_search_ms']['median'], 0)
            manifest = json.loads((root/'export/manifest.json').read_text())
            self.assertEqual(len(manifest['files_sha256']), 16)
            np.testing.assert_array_equal(np.frombuffer((root/'export/query_0.list_0.bin').read_bytes(), '<f4'), data['scan_queries'][0, 0])
            data['codes'][0, 0] ^= 255
            np.savez(root/'corrupt.npz', **data)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                compare.load_bundle(root/'corrupt.npz')

    def test_matched_multilist_mock_and_corruption(self):
        _, data = fixture()
        class SoftwareSession:
            def __init__(self, lid):
                self.lid, self.calls = lid, 0
            def search(self, query, k):
                qi = self.calls
                self.calls += 1
                probe = list(data['list_ids'][qi]).index(self.lid)
                np.testing.assert_array_equal(np.frombuffer(query, '<f4'), data['scan_queries'][qi, probe])
                ids, distances = compare.candidates(data, qi, probe)
                order = np.argsort(distances)[:k]
                return [host.Result(float(distances[i]), int(ids[i]), 0) for i in order], {'kernel_ms': None}
        sessions = {lid: SoftwareSession(lid) for lid in (0, 1)}
        got, report = compare.matched_fpga_search(sessions, data)
        self.assertEqual(len(got), 2)
        self.assertEqual(len(report['per_list']), 4)
        self.assertIsNone(report['speedup'])
        ri, rd = compare.candidates(data, 0, 0)
        with self.assertRaises(ValueError):
            compare.check_top([999]*3, [1, 2, 3], ri, rd, 3)

    def test_bad_lists_filtering_and_boundary(self):
        index, data = fixture()
        for lists in (np.asarray([[0, 0], [1, 0]]), np.asarray([[0, 3], [1, 0]]),
                      np.asarray([[0., 1.], [1., 0.]])):
            with self.assertRaises(ValueError):
                compare.prepare(index, data['queries'], lists, 3)
        index.max_codes = 1
        with self.assertRaises(ValueError):
            compare.extract(index)
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            compare.check_top([1], [1.], np.asarray([1, 2]), np.asarray([1., 1.]), 1)

    def test_stage_disjoint_and_no_unsafe_open(self):
        _, data = fixture()
        mock = host.MockTransport(statuses=(4, 4))
        sessions = compare.stage(mock, data, abi=host.ABI)
        self.assertEqual(set(sessions), {0, 1})
        self.assertLess(sessions[0].cluster_address, sessions[1].cluster_address)
        with patch('os.open', side_effect=AssertionError('must not open')):
            with self.assertRaises(PermissionError):
                host.VfioTransport(11, '0000:8e:00.0', ownership_confirmed=True)
            with self.assertRaises(PermissionError):
                host.OwnedXdmaTransport(1, 2)

    def test_control_sequence_failures_timeout_and_decode(self):
        _, data = fixture()
        raw = compare.pack_cluster(data['codes'][:7], data['ids'][:7])
        cb, query = data['codebook'].astype('<f4').tobytes(), data['scan_queries'][0, 0].astype('<f4').tobytes()
        result = payload([1], [1.5])
        mock = host.MockTransport(result)
        session = host.Session(mock, raw, cb, abi=host.ABI)
        got, timing = session.search(query, 1, sleep=lambda _: None)
        self.assertEqual(got, [host.Result(1.5, 1, 0)])
        self.assertIsNone(timing['kernel_ms'])
        self.assertEqual(mock.operations[-1], ('read', host.RESULT, 64))
        self.assertIn(('write32', host.ACC_CTL+0x14, 0x10), mock.operations)
        for step in range(1, len(mock.operations)+1):
            failing = host.MockTransport(result, fail_at=step)
            with self.subTest(step=step), self.assertRaises(OSError):
                s = host.Session(failing, raw, cb, abi=host.ABI)
                s.search(query, 1, sleep=lambda _: None)
            self.assertEqual(len(failing.operations), step)
        timeout = host.MockTransport(statuses=(4, 4, 0))
        s = host.Session(timeout, raw, cb, abi=host.ABI)
        ticks = iter((0, 2))
        with self.assertRaises(TimeoutError):
            s.search(query, 1, timeout=1, clock=lambda: next(ticks), sleep=lambda _: None)
        count = len(timeout.operations)
        with self.assertRaises(RuntimeError):
            s.search(query, 1)
        self.assertEqual(len(timeout.operations), count)
        self.assertFalse(any(op[0] == 'read' for op in timeout.operations))
        for statuses in ((0xffffffff,), (0,), (0x84,), (4, 4, 0xffffffff)):
            unsafe = host.MockTransport(result, statuses=statuses)
            with self.assertRaises(RuntimeError):
                s = host.Session(unsafe, raw, cb, abi=host.ABI)
                s.search(query, 1, sleep=lambda _: None)
            self.assertFalse(any(op[0] == 'read' for op in unsafe.operations))
        invalid = host.MockTransport()
        with self.assertRaises(ValueError):
            host.Session(invalid, raw[:-1], cb, abi=host.ABI)
        self.assertEqual(invalid.operations, [])
        for bad in (b'', payload([1], [float('nan')]), payload([1, 1], [1, 2])):
            with self.assertRaises(ValueError):
                host.decode_results(bad, 1 if len(bad) != 128 else 2)

    def test_owned_transport_short_io_poison_and_allowlist(self):
        fake_stat = type('Stat', (), {'st_mode': 0o020600})()
        with patch('os.fstat', return_value=fake_stat), patch('os.pwrite', return_value=3) as write:
            t = host.OwnedXdmaTransport(3, 4, ownership_confirmed=True, driver_verified=True, abi=host.ABI)
            with self.assertRaises(ValueError):
                t.write(0, b'1234')
            write.assert_not_called()
            with self.assertRaises(OSError):
                t.write32(host.ACC_CTL, 1)
            with self.assertRaises(OSError):
                t.write32(host.ACC_CTL, 1)
            self.assertEqual(write.call_count, 1)


if __name__ == '__main__':
    unittest.main()
