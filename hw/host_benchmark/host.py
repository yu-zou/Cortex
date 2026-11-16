"""Historical A6 protocol, using only already-owned XDMA descriptors or mocks.

There is deliberately no device-opening CLI. The owner supplies descriptors and
an independently verified ABI; neither PCI configuration nor VFIO is touched.
"""
import math
import struct
import time
import socket
import threading
from concurrent.futures import CancelledError
from dataclasses import dataclass

ACC_CTL = 0xA8010000
DDR = 0x1000000000
CLUSTER, CODEBOOK = DDR, DDR + 0x08000000
QUERY, RESULT = DDR + 0x10000000, DDR + 0x10010000
ABI = 'historical-a6-separate-codebook-results64'


@dataclass(frozen=True)
class Result:
    distance: float
    address: int
    length: int


def validate_inputs(cluster, codebook, query, k):
    if len(cluster) < 64:
        raise ValueError('short cluster header')
    m, dim, n = struct.unpack_from('<III', cluster)
    if (m, dim) != (96, 768) or not 1 <= n <= 8192:
        raise ValueError('unsupported cluster dimensions/count')
    if len(cluster) != 64 + 128*n or len(codebook) != 786432 or len(query) != 3072:
        raise ValueError('invalid input length')
    if not 1 <= k <= min(n, 500):
        raise ValueError('invalid top-k')
    for data in (codebook, query):
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', data)):
            raise ValueError('non-finite input')
    return n


def decode_results(data, count):
    if not 1 <= count <= 500 or len(data) != count*64:
        raise ValueError('short or oversized result')
    results = [Result(*struct.unpack_from('<fQI', data, i*64)) for i in range(count)]
    if any(not math.isfinite(r.distance) or r.distance < 0 for r in results):
        raise ValueError('invalid L2 distance')
    if len({r.address for r in results}) != count:
        raise ValueError('duplicate result ID')
    if any(a.distance > b.distance for a, b in zip(results, results[1:])):
        raise ValueError('unsorted results')
    return results


class VfioTransport:
    def __init__(self, *args, **kwargs):
        raise PermissionError('VFIO blocked: device release/close may reset the device or bus')


class OwnedXdmaTransport:
    """Borrow, never open/duplicate/close, owner-provided synchronous MM FDs.

    Caller must hold exclusive ownership for the entire lifetime and verify that
    the driver implements synchronous positional MM DMA without reset on use.
    Control uses AXI addresses over DMA, not the XDMA engine's BAR registers.
    """
    def __init__(self, h2c_fd, c2h_fd, *, ownership_confirmed=False,
                 driver_verified=False, abi=None):
        raise PermissionError('unbounded positional XDMA I/O disabled; use bounded owner RPC')
        if not ownership_confirmed or not driver_verified or abi != ABI:
            raise PermissionError('exclusive ownership, safe MM driver and historical ABI required')
        if h2c_fd == c2h_fd:
            raise ValueError('separate H2C/C2H descriptors required')
        for fd in (h2c_fd, c2h_fd):
            if not stat.S_ISCHR(os.fstat(fd).st_mode):
                raise ValueError('XDMA character descriptors required')
        self.h2c_fd, self.c2h_fd = h2c_fd, c2h_fd
        self.poisoned = False

    @staticmethod
    def _range(address, size, write):
        regions = [(CLUSTER, CODEBOOK-CLUSTER), (CODEBOOK, 786432), (QUERY, 3072)] if write else [(RESULT, 500*64)]
        registers = {0, 0x10, 0x14, 0x1c, 0x20, 0x28, 0x2c, 0x34, 0x38,
                     0x40, 0x48, 0x50, 0x58, 0x70} if write else {0}
        if size == 4 and address-ACC_CTL in registers:
            return
        if size > 0 and any(base <= address and address+size <= base+length for base, length in regions):
            return
        raise ValueError('transfer outside accelerator allowlist')

    def _transfer(self, address, payload, write):
        size = len(payload) if write else payload
        self._range(address, size, write)
        if self.poisoned:
            raise OSError('transport failed previously; owner intervention required')
        try:
            if write:
                if os.pwrite(self.h2c_fd, payload, address) != size:
                    raise OSError('short XDMA write')
                return None
            data = os.pread(self.c2h_fd, size, address)
            if len(data) != size:
                raise OSError('short XDMA read')
            return data
        except OSError:
            self.poisoned = True
            raise

    def write(self, address, data):
        self._transfer(address, data, True)

    def read(self, address, size):
        return self._transfer(address, size, False)

    def write32(self, address, value):
        self.write(address, struct.pack('<I', value))

    def read32(self, address):
        return struct.unpack('<I', self.read(address, 4))[0]


class Session:
    """Stage once into FPGA DDR; then issue resident single-list L2 scans."""
    def __init__(self, transport, cluster, codebook, *, abi=None, cluster_address=CLUSTER):
        if abi != ABI:
            raise PermissionError('unverified register map; current source ABI is different')
        self.n = validate_inputs(cluster, codebook, bytes(3072), 1)
        if cluster_address % 64 or not CLUSTER <= cluster_address < CODEBOOK or cluster_address+len(cluster) > CODEBOOK:
            raise ValueError('cluster outside reserved DDR region')
        self.cluster_address = cluster_address
        self.transport = transport
        self.failed = False
        start = time.perf_counter_ns()
        transport.begin(5.0)
        try:
            # ap_idle required; reading ap_done can clear stale completion.
            status = transport.read32(ACC_CTL)
            if status == 0xffffffff or not status & 4 or status & 0x81:
                raise RuntimeError('accelerator not idle or auto-restart enabled')
            transport.write(cluster_address, cluster)
            transport.write(CODEBOOK, codebook)
        except Exception:
            self.failed = True
            raise
        finally:
            transport.end(self.failed)
        self.setup_transfer_ms = (time.perf_counter_ns()-start)/1e6

    def search(self, query, k, *, timeout=2.0, clock=time.monotonic, sleep=time.sleep,
               cancelled=lambda: False):
        if self.failed or self.transport.poisoned:
            raise RuntimeError('session failed; no further device operations permitted')
        if len(query) != 3072 or not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', query)):
            raise ValueError('invalid query')
        if not 1 <= k <= min(self.n, 500) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('invalid k or timeout')
        t = self.transport
        begin = time.perf_counter_ns()
        t.begin(timeout)
        try:
            deadline = clock()+timeout
            if cancelled():
                raise CancelledError('cancelled before launch')
            status = t.read32(ACC_CTL)
            if status == 0xffffffff or not status & 4 or status & 0x81:
                raise RuntimeError('accelerator not idle')
            t.write(QUERY, query)
            transferred = time.perf_counter_ns()
            for off, address in ((0x10, self.cluster_address), (0x1c, CODEBOOK), (0x28, QUERY), (0x34, RESULT)):
                t.write32(ACC_CTL+off, address & 0xffffffff)
                t.write32(ACC_CTL+off+4, address >> 32)
            # Reload from resident DDR each invocation; BRAM reuse is unverified.
            for off, value in ((0x40, k), (0x48, 0), (0x50, 1), (0x58, 0), (0x70, 1)):
                t.write32(ACC_CTL+off, value)
            sleep(0.0002)
            launched = time.perf_counter_ns()
            t.write32(ACC_CTL, 1)
            while True:
                if cancelled():
                    raise CancelledError('cancelled; owner retains device and buffers')
                if clock() >= deadline:
                    raise TimeoutError('completion timed out; no reset or result read attempted')
                status = t.read32(ACC_CTL)
                if status == 0xffffffff or status & 0x80:
                    raise RuntimeError('invalid accelerator status')
                if status & 2:
                    break
                if clock() >= deadline:
                    raise TimeoutError('completion timed out; no reset or result read attempted')
                sleep(0.001)
            completed = time.perf_counter_ns()
            raw = t.read(RESULT, k*64)
            received = time.perf_counter_ns()
            results = decode_results(raw, k)
            if cancelled():
                raise CancelledError('cancelled before result publication')
        except Exception:
            self.failed = True
            t.poisoned = True
            raise
        finally:
            t.end(self.failed)
        return results, {'query_transfer_ms': (transferred-begin)/1e6,
                         'control_ms': (launched-transferred)/1e6,
                         'launch_to_done_observed_ms': (completed-launched)/1e6,
                         'result_transfer_ms': (received-completed)/1e6,
                         'end_to_end_ms': (time.perf_counter_ns()-begin)/1e6,
                         'kernel_ms': None}


class MockTransport:
    kind = 'mock'
    def __init__(self, result=b'', statuses=(4, 4, 2), fail_at=None):
        self.result, self.statuses = result, iter(statuses)
        self.operations, self.fail_at = [], fail_at
        self.poisoned = False
        self.lock = threading.Lock()

    def begin(self, timeout):
        if self.poisoned:
            raise OSError('mock quarantined')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('mock already owned')

    def end(self, failed=False):
        self.poisoned |= failed
        self.lock.release()

    def _record(self, op, address, value):
        self.operations.append((op, address, value))
        if len(self.operations) == self.fail_at:
            raise OSError('injected failure')

    def write(self, address, data):
        self._record('write', address, len(data))

    def write32(self, address, value):
        self._record('write32', address, value)

    def read32(self, address):
        self._record('read32', address, 4)
        return next(self.statuses, 0)

    def read(self, address, size):
        self._record('read', address, size)
        return self.result


class OwnerTransport:
    """Bounded RPC over a borrowed, connected AF_UNIX SOCK_STREAM socket.

    The persistent owner alone holds VFIO/device/DMA resources. Disconnect never
    means cancel, unmap, close VFIO, or reset. See OWNER_PROTOCOL.md. No device
    files are opened by this client. There is no CPU fallback.
    """
    kind = 'owner-rpc-unverified'
    HEADER = struct.Struct('<8sIIQQQiI')
    MAGIC = b'CXOWN001'

    def __init__(self, connection, *, abi, timeout=1.0):
        if abi != ABI or not math.isfinite(timeout) or not 0 < timeout <= 60:
            raise ValueError('verified historical ABI and bounded timeout required')
        if connection.family != socket.AF_UNIX or connection.type != socket.SOCK_STREAM:
            raise ValueError('connected Unix stream required')
        self.connection, self.timeout = connection, timeout
        self.poisoned, self.sequence = False, 0
        self.lock = threading.Lock()
        self.deadline = None

    def begin(self, timeout):
        if self.poisoned:
            raise OSError('owner connection quarantined')
        if not self.lock.acquire(blocking=False):
            raise BlockingIOError('accelerator buffers already owned by another request')
        self.deadline = time.monotonic() + timeout

    def end(self, failed=False):
        self.poisoned |= failed
        self.deadline = None
        self.lock.release()

    def _transfer(self, address, value, write):
        size = len(value) if write else value
        OwnedXdmaTransport._range(address, size, write)
        if self.poisoned or self.deadline is None:
            raise OSError('active exclusive transaction required')
        if size > 2**20:
            raise ValueError('owner frame exceeds 1 MiB')
        deadline = min(self.deadline, time.monotonic()+self.timeout)
        def bounded():
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('owner RPC deadline expired')
            self.connection.settimeout(remaining)
        def receive(n):
            result = bytearray()
            while len(result) < n:
                bounded()
                part = self.connection.recv(n-len(result))
                if not part:
                    raise ConnectionError('owner disconnected; resources remain owner responsibility')
                result.extend(part)
            return bytes(result)
        try:
            self.sequence += 1
            op = 1 if write else 2
            request = self.HEADER.pack(self.MAGIC, 1, op, self.sequence, address, size, 0, 0)
            bounded()
            self.connection.sendall(request+(value if write else b''))
            header = self.HEADER.unpack(receive(self.HEADER.size))
            if header != (self.MAGIC, 1, op, self.sequence, address, size, 0, 0):
                raise OSError('owner response mismatch or negative completion')
            return None if write else receive(size)
        except Exception:
            self.poisoned = True
            raise

    def write(self, address, data):
        self._transfer(address, data, True)

    def read(self, address, size):
        return self._transfer(address, size, False)

    def write32(self, address, value):
        self.write(address, struct.pack('<I', value))

    def read32(self, address):
        return struct.unpack('<I', self.read(address, 4))[0]
