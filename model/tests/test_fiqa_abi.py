"""CPUcompact guards plus actual ABI malformed-input/cache checks when requested."""
import os
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fiqa_cpucompact import cluster_ids, pack_list, parse_list
from validate_fiqa_abi import Driver


class FormatTests(unittest.TestCase):
    def setUp(self):
        self.cb = np.zeros((96, 256, 8), np.float32)
        self.center = np.zeros(768, np.float32)
        self.codes = np.zeros((1, 96), np.uint8)
        self.blob = pack_list(self.cb, self.center, self.codes, np.array([0]), 1)

    def test_exact_layout(self):
        self.assertEqual(len(self.blob), 789680)
        cb, center, codes, ids = parse_list(self.blob, 1)
        self.assertEqual(ids.tolist(), [0])
        self.assertEqual(codes.shape, (1, 96))

    def test_truncated(self):
        for size in [0, 63, 64, 789567, len(self.blob) - 1]:
            with self.assertRaises(ValueError):
                parse_list(self.blob[:size], 1)

    def test_wrong_version(self):
        blob = bytearray(self.blob)
        struct.pack_into('<I', blob, 16, 3)
        with self.assertRaises(ValueError):
            parse_list(blob, 1)

    def test_out_of_range_pack(self):
        for row in [-1, 1]:
            with self.assertRaises(ValueError):
                pack_list(self.cb, self.center, self.codes, np.array([row]), 1)

    def test_out_of_range_parse(self):
        blob = bytearray(self.blob)
        struct.pack_into('<Q', blob, 789568 + 96, 1)
        with self.assertRaises(ValueError):
            parse_list(blob, 1)

    def test_namespace(self):
        self.assertEqual(cluster_ids('a'), cluster_ids('a'))
        self.assertEqual(len(set(cluster_ids('a') + cluster_ids('b'))), 128)

    @unittest.skipUnless(os.environ.get('FIQA_ABI_DRIVER'), 'set actual shared library paths')
    def test_actual_abi_rejection_and_cache(self):
        driver = Driver(os.environ['FIQA_ABI_DRIVER'], os.environ['FIQA_ABI_MOCK'])
        query = np.ones(768, np.float32)
        try:
            self.assertEqual(driver.read(100, self.blob[:-1], query).status, -22)
            bad = bytearray(self.blob)
            struct.pack_into('<I', bad, 16, 99)
            self.assertEqual(driver.read(101, bytes(bad), query).status, -22)
            good = driver.read(102, self.blob, query)
            self.assertEqual((good.status, good.count, good.entries[0].doc_addr), (0, 1, 0))
            # A cache hit really ignores even malformed replacement bytes.
            cached = driver.read(102, bytes(bad), query)
            self.assertEqual((cached.status, cached.count), (0, 1))
            changed = bytearray(self.blob)
            struct.pack_into('<f', changed, 786496, 1.)
            self.assertEqual(driver.read(102, bytes(changed), query).entries[0].distance, 0.)
            self.assertEqual(driver.read(103, bytes(changed), query).entries[0].distance, -1.)
        finally:
            driver.close()


if __name__ == '__main__':
    unittest.main()
