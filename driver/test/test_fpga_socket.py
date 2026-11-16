"""Isolated software broker fixture. Never opens accelerator or SSD devices."""
import ctypes as C
import errno
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest

HEADER = struct.Struct('<IIIIQQiI')
MAGIC = 0x31465043


class Entry(C.Structure):
    _fields_ = [('distance', C.c_float), ('token', C.c_uint64),
                ('length', C.c_uint32), ('pad', C.c_uint32)]


class Result(C.Structure):
    _fields_ = [('status', C.c_int32), ('count', C.c_uint32), ('entries', Entry*500)]


def receive(conn, size):
    result = b''
    while len(result) < size:
        part = conn.recv(size-len(result))
        if not part:
            raise EOFError
        result += part
    return result


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(sys.argv[1])
        cls.lib.cortex_driver_init.argtypes = [C.c_char_p]
        args = [C.c_uint32, C.c_void_p, C.c_uint64, C.POINTER(C.c_float)] + [C.c_uint32]*4
        cls.lib.cortex_semantic_read.argtypes = args
        cls.lib.cortex_semantic_read.restype = Result
        cls.cancel_type = C.CFUNCTYPE(C.c_int, C.c_void_p)
        cls.lib.cortex_fpga_read_cancellable.argtypes = args + [cls.cancel_type, C.c_void_p]
        cls.lib.cortex_fpga_read_cancellable.restype = Result

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cortex-fpga-')
        self.path = os.path.join(self.temp.name, 'broker.sock')
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(self.path)
        self.listener.listen()
        self.listener.settimeout(.05)
        self.stop = threading.Event()
        self.entered = threading.Event()
        self.eof = threading.Event()
        self.behavior = 'ok'
        self.errors = []
        self.thread = threading.Thread(target=self.serve)
        self.thread.start()
        self.object = bytearray(64+786432+3072+112)
        struct.pack_into('<7I', self.object, 0, 96, 768, 1, 0x31515043, 2, 256, 1)
        struct.pack_into('<QQ', self.object, len(self.object)-16, 123, 17)
        self.buffer = (C.c_ubyte*len(self.object)).from_buffer(self.object)
        self.query = (C.c_float*768)()

    def tearDown(self):
        self.lib.cortex_driver_shutdown()
        self.stop.set()
        self.thread.join(2)
        self.listener.close()
        self.temp.cleanup()
        self.assertFalse(self.thread.is_alive())
        self.assertEqual(self.errors, [])

    def config(self, timeout=100):
        return ('{"backend":"fpga","socket":"%s","profile":"cpq1v2-residual-ip",'
                '"timeout_ms":%d}' % (self.path, timeout)).encode()

    def init(self, timeout=100):
        self.assertEqual(self.lib.cortex_driver_init(self.config(timeout)), 0)

    def read(self, cancel=None):
        args = (1, self.buffer, len(self.object), self.query, 768, 1, 1, 0)
        if cancel:
            return self.lib.cortex_fpga_read_cancellable(*args, cancel, None)
        return self.lib.cortex_semantic_read(*args)

    def serve(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            try:
                with conn:
                    conn.settimeout(1)
                    while not self.stop.is_set():
                        h = HEADER.unpack(receive(conn, 40))
                        magic, version, op, profile, ident, size, status, reserved = h
                        assert (magic, version, status, reserved) == (MAGIC, 1, 0, 0)
                        body = receive(conn, size)
                        error = 0
                        if op == 1:
                            body = struct.pack('<II', 0 if self.behavior == 'mock' else 1, 2)
                        else:
                            assert op == 2 and len(body) == 32+3072+len(self.object)
                            assert body[32+3072:] == self.object
                            self.entered.set()
                            if self.behavior == 'wait':
                                while not self.stop.is_set():
                                    try:
                                        if not conn.recv(1):
                                            self.eof.set()
                                            break
                                    except socket.timeout:
                                        continue
                                break
                            if self.behavior == 'disconnect':
                                break
                            body = struct.pack('<IIfQI', 1, 0, 2.5, 999 if self.behavior == 'token' else 123, 17)
                            if self.behavior == 'error':
                                error, body = -errno.EIO, b''
                            if self.behavior == 'oversize':
                                size = 2**40
                            else:
                                size = len(body)
                            if self.behavior == 'stale':
                                ident += 1
                        if op == 1:
                            size = len(body)
                        frame = HEADER.pack(MAGIC, 1, op | 0x80000000, profile, ident, size, error, 0)+body
                        # Deliberate fragmentation checks stream framing.
                        conn.sendall(frame[:13])
                        conn.sendall(frame[13:])
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.errors.append(repr(exc))

    def test_success_and_abi(self):
        self.assertEqual(self.lib.cortex_api_version(), 2)
        self.init()
        result = self.read()
        self.assertEqual((result.status, result.count), (0, 1))
        e = result.entries[0]
        self.assertEqual((e.distance, e.token, e.length, e.pad), (2.5, 123, 17, 0))
        self.assertEqual(self.lib.cortex_is_cluster_cached(1), 0)

    def test_explicit_configuration(self):
        for config in (b'{}', b'{"backend":"mock"}', self.config()+b' ', self.config(0),
                       self.config().replace(b':100}', b':-1}'),
                       self.config().replace(b':100}', b':4294967396}')):
            self.assertEqual(self.lib.cortex_driver_init(config), -errno.EINVAL)
        self.behavior = 'mock'
        self.assertEqual(self.lib.cortex_driver_init(self.config()), -errno.ENOTSUP)

    def test_missing_broker(self):
        self.assertEqual(self.lib.cortex_driver_init(self.config().replace(b'broker.sock', b'absent.sock')), -errno.ENOENT)
        self.assertEqual(self.read().status, -errno.ESHUTDOWN)

    def test_bad_completion_and_remote_errors(self):
        self.init()
        for behavior, status in [('stale', -errno.EPROTO), ('token', -errno.EPROTO),
                                 ('oversize', -errno.EPROTO), ('error', -errno.EIO),
                                 ('disconnect', -errno.ECONNRESET)]:
            self.behavior = behavior
            result = self.read()
            self.assertEqual((result.status, result.count), (status, 0), behavior)
            self.assertTrue(all(e.token == 0 for e in result.entries))

    def test_timeout(self):
        self.init(70)
        self.behavior = 'wait'
        start = time.monotonic()
        self.assertEqual(self.read().status, -errno.ETIMEDOUT)
        self.assertLess(time.monotonic()-start, .5)
        self.assertTrue(self.eof.wait(.5))

    def test_cancel_and_busy(self):
        self.init(1000)
        self.behavior = 'wait'
        canceled = threading.Event()
        callback = self.cancel_type(lambda _: canceled.is_set())
        results = []
        worker = threading.Thread(target=lambda: results.append(self.read(callback)))
        worker.start()
        self.assertTrue(self.entered.wait(.5))
        self.assertEqual(self.read().status, -errno.EBUSY)
        canceled.set()
        worker.join(.5)
        self.assertFalse(worker.is_alive())
        self.assertEqual((results[0].status, results[0].count), (-errno.ECANCELED, 0))
        self.assertTrue(self.eof.wait(.5))

    def test_shutdown(self):
        self.init(1000)
        self.behavior = 'wait'
        results = []
        worker = threading.Thread(target=lambda: results.append(self.read()))
        worker.start()
        self.assertTrue(self.entered.wait(.5))
        self.lib.cortex_driver_shutdown()
        worker.join(.5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0].status, -errno.ESHUTDOWN)

    def test_invalid_object_and_pre_cancel(self):
        self.init()
        callback = self.cancel_type(lambda _: 1)
        self.assertEqual(self.read(callback).status, -errno.ECANCELED)
        self.object[16] = 1
        self.assertEqual(self.read().status, -errno.EINVAL)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
