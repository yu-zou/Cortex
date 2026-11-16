import socket
import struct
import threading
import unittest
from concurrent.futures import CancelledError

import host
import layout


def inputs():
    return struct.pack('<III', 96, 768, 2)+bytes(52+256), bytes(786432)


def result():
    return struct.pack('<fQI', 1.5, 42, 0)+bytes(48)


class Tests(unittest.TestCase):
    def test_success_and_fail_every_operation(self):
        baseline = host.MockTransport(result())
        session = host.Session(baseline, *inputs(), abi=host.ABI)
        got, timing = session.search(bytes(3072), 1, sleep=lambda _: None)
        self.assertEqual(got, [host.Result(1.5, 42, 0)])
        self.assertIsNone(timing['kernel_ms'])
        for step in range(1, len(baseline.operations)+1):
            mock = host.MockTransport(result(), fail_at=step)
            with self.subTest(step=step), self.assertRaises(OSError):
                host.Session(mock, *inputs(), abi=host.ABI).search(bytes(3072), 1, sleep=lambda _: None)
            self.assertEqual(len(mock.operations), step)
            self.assertTrue(mock.poisoned)

    def test_timeout_cancel_and_shared_quarantine(self):
        for cancel in (False, True):
            mock = host.MockTransport(statuses=(4, 4, 0))
            session = host.Session(mock, *inputs(), abi=host.ABI)
            ticks = iter((0, 0, 2))
            with self.assertRaises(CancelledError if cancel else TimeoutError):
                session.search(bytes(3072), 1, timeout=1, clock=lambda: next(ticks),
                               sleep=lambda _: None, cancelled=lambda: cancel)
            before = list(mock.operations)
            with self.assertRaises(RuntimeError):
                session.search(bytes(3072), 1)
            self.assertEqual(mock.operations, before)
            self.assertFalse(any(op[0] == 'read' for op in before))

    def test_envelope(self):
        cluster, cb = inputs()
        cluster = bytearray(cluster)
        struct.pack_into('<Q', cluster, 64+128+88, 1)
        blob = layout.encode(7, bytes(cluster), cb, bytes(3072), residual=True)
        decoded = layout.decode(blob)
        self.assertEqual(decoded['list_id'], 7)
        self.assertTrue(decoded['residual'])
        self.assertEqual(decoded['cluster'], cluster)
        with self.assertRaises(ValueError):
            layout.decode(blob, require_document_addresses=True)
        for bad in (blob[:-1], blob[:128]+b'X'+blob[129:], b''):
            with self.assertRaises(ValueError):
                layout.decode(bad)

    def test_hardware_paths_closed(self):
        with self.assertRaises(PermissionError):
            host.VfioTransport()
        with self.assertRaises(PermissionError):
            host.OwnedXdmaTransport(0, 1)

    def test_owner_rpc_and_deadline(self):
        client, server = socket.socketpair()
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        transport = host.OwnerTransport(client, abi=host.ABI, timeout=0.03)
        def owner():
            request = bytearray()
            while len(request) < transport.HEADER.size:
                request.extend(server.recv(transport.HEADER.size-len(request)))
            server.sendall(request+struct.pack('<I', 4))
        thread = threading.Thread(target=owner)
        thread.start()
        transport.begin(1)
        self.assertEqual(transport.read32(host.ACC_CTL), 4)
        thread.join(1)
        with self.assertRaises(BlockingIOError):
            transport.begin(1)
        with self.assertRaises(TimeoutError):
            transport.read32(host.ACC_CTL)
        self.assertTrue(transport.poisoned)
        transport.end(True)


if __name__ == '__main__':
    unittest.main()
