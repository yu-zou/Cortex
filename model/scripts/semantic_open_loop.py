"""Bounded, pre-encoded HTTP open-loop preparation client (Python 3.8 stdlib).

Input JSON: {"queries": [{"request_id": "q1", "offset_ns": 0,
"query_vector": [768 finite numbers]}]}. Offsets are absolute relative to one
monotonic run epoch; equal offsets are allowed. No POST retry or connection pool.
Optional oracle entries bind request_id AND vector_sha256 to ordered results;
this is a numerical golden oracle, never qrels or a relevance-quality metric.
"""
import argparse
import base64
import hashlib
import http.client
import io
import ipaddress
import json
import math
import re
import socket
import threading
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit

from measurement_contract import Measurement, RequestRecord, RunMetadata


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


@dataclass(frozen=True)
class Config:
    endpoint: str
    tenant: str
    namespace: str
    metadata: RunMetadata
    send_lag_limit_ns: int
    max_inflight: int = 8
    max_inflight_bytes: int = 8 * 1024 * 1024
    max_response_bytes: int = 1024 * 1024
    max_queries: int = 10000
    max_input_bytes: int = 16 * 1024 * 1024
    timeout_s: float = 5.0
    top_k: int = 10
    nprobe: int = 32
    metric: str = "IP"

    def __post_init__(self):
        u = urlsplit(self.endpoint)
        if (u.scheme not in ("http", "https") or not u.hostname or
                u.username or u.password or u.fragment or u.query or
                u.path != "/v1/semantic_search"):
            raise ValueError("endpoint must be http(s)://host[:port]/v1/semantic_search")
        try:
            u.port
            ipaddress.ip_address(u.hostname)
        except ValueError:
            raise ValueError("strict transport requires a numeric IP endpoint and valid port")
        if not isinstance(self.metadata, RunMetadata):
            raise ValueError("explicit backend metadata required")
        for value in (self.tenant, self.namespace, self.metadata.index,
                      self.metadata.version):
            if not value or any(ord(c) < 32 or ord(c) > 126 for c in value):
                raise ValueError("header/namespace must be nonempty printable ASCII")
        for name in ("max_inflight", "max_inflight_bytes", "max_response_bytes",
                     "max_queries", "max_input_bytes", "top_k", "nprobe"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + " must be a positive integer")
        if type(self.send_lag_limit_ns) is not int or self.send_lag_limit_ns < 0:
            raise ValueError("send_lag_limit_ns must be predeclared and nonnegative")
        if (type(self.timeout_s) not in (int, float) or
                not math.isfinite(self.timeout_s) or self.timeout_s <= 0):
            raise ValueError("timeout must be finite and positive")
        if self.metric not in ("IP", "L2") or self.nprobe > 64 or self.top_k > 500:
            raise ValueError("invalid metric/nprobe/top_k")


class Clock:
    now = staticmethod(time.monotonic_ns)

    def wait_until(self, deadline):
        while True:
            remaining = deadline - self.now()
            if remaining <= 0:
                return
            time.sleep(remaining / 1e9)


def join_workers(threads, timeout_s):
    """Fail loudly on a stuck non-daemon worker; never report partial success.

    This detects but cannot cancel OS DNS resolution or arbitrary transports.
    The caller must release such work; numeric endpoints avoid resolver stalls.
    """
    deadline = time.monotonic() + timeout_s
    for thread in threads:
        thread.join(max(0, deadline - time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    if alive:
        raise RuntimeError("workers still alive after bounded join: " + ", ".join(alive))


def prepare(queries, config, oracle=None):
    if not isinstance(queries, list) or not 0 < len(queries) <= config.max_queries:
        raise ValueError("query count exceeds bounded manifest or is empty")
    if len(encoded(queries)) > config.max_input_bytes:
        raise ValueError("input manifest exceeds byte cap")
    prepared, ids, previous = [], set(), -1
    for q in queries:
        qid, offset, vector = q["request_id"], q["offset_ns"], q["query_vector"]
        if (not isinstance(qid, str) or not qid.strip() or qid != qid.strip() or qid in ids
                or any(ord(c) < 32 or ord(c) > 126 for c in qid)):
            raise ValueError("request IDs must be unique nonempty unpadded strings")
        if type(offset) is not int or offset < 0 or offset < previous:
            raise ValueError("offsets must be nonnegative sorted integer nanoseconds")
        if (not isinstance(vector, list) or len(vector) != 768 or
                any(type(v) not in (int, float) or not math.isfinite(v) or
                    abs(v) > 3.402823466e38 for v in vector)):
            raise ValueError("query vector must contain 768 finite float32-range numbers")
        body = encoded(dict(query_vector=vector, top_k=config.top_k,
                            nprobe=config.nprobe, metric=config.metric))
        vh = digest(vector)
        expected = None
        if oracle is not None:
            entry = oracle.get(qid)
            if not entry or entry.get("vector_sha256") != vh:
                raise ValueError("oracle missing ID or vector hash mismatch: " + qid)
            if encoded(entry.get("scope")) != encoded(oracle_scope(config)):
                raise ValueError("oracle request scope mismatch: " + qid)
            expected = entry["results"]
            validate_results({"results": expected}, config.top_k)
        prepared.append((qid, offset, body, vh, expected))
        ids.add(qid)
        previous = offset
    return prepared


def oracle_scope(config):
    return dict(tenant=config.tenant, namespace=config.namespace,
                index=config.metadata.index, version=config.metadata.version,
                metric=config.metric, nprobe=config.nprobe, top_k=config.top_k)


class DeadlineReader(io.RawIOBase):
    def __init__(self, transport):
        self.transport = transport

    def readable(self):
        return True

    def readinto(self, buffer):
        self.transport.remaining()
        return self.transport.sock.recv_into(buffer)

    def close(self):
        if not self.closed:
            self.transport.reader_open = False
            if self.transport.owner_closed:
                self.transport.sock.close()
        super().close()


class DeadlineSocket:
    """Deadline checks below HTTP's buffered headers, body and chunk parser.

    Counts plaintext bytes accepted by socket.send, not wire bytes or receipt.
    A failed send has an unknown additional contribution; counts are lower bounds.
    """
    def __init__(self, sock, deadline, row, clock, on_send):
        self.sock, self.deadline, self.row = sock, deadline, row
        self.clock, self.on_send = clock, on_send
        self.body_phase = False
        self.reader_open, self.owner_closed = False, False

    def remaining(self):
        seconds = (self.deadline - self.clock.now()) / 1e9
        if seconds <= 0:
            raise socket.timeout()
        self.sock.settimeout(seconds)

    def sendall(self, data):
        view = memoryview(data)
        while view:
            self.remaining()
            if self.row["send_ns"] is None:
                self.row["send_ns"] = self.clock.now()
                self.on_send(self.row["request_id"])
            try:
                count = self.sock.send(view)
                if not count:
                    raise ConnectionError("zero-byte socket send")
            except OSError:
                self.row["request_write_bytes_exact"] = False
                raise
            self.row["request_application_bytes_confirmed"] += count
            if self.body_phase:
                self.row["request_body_bytes"] += count
            view = view[count:]

    def makefile(self, mode):
        if mode != "rb":
            raise ValueError("read-only transport stream")
        self.reader_open = True
        return io.BufferedReader(DeadlineReader(self))

    def close(self):
        self.owner_closed = True
        if not self.reader_open:
            self.sock.close()


def validate_results(value, top_k):
    results = value.get("results") if isinstance(value, dict) else None
    if not isinstance(results, list) or len(results) != top_k:
        raise ValueError("expected exactly top_k results")
    seen, previous = set(), None
    for r in results:
        addr, distance, length = r["doc_addr"], r["distance"], r["doc_length"]
        if (type(addr) is not int or not 0 <= addr < 2 ** 64 or addr in seen or
                type(length) is not int or not 0 < length < 2 ** 32 or
                type(distance) not in (int, float) or not math.isfinite(distance)):
            raise ValueError("invalid doc_addr/doc_length/distance")
        key = (distance, addr)
        if previous is not None and key < previous:
            raise ValueError("results must be ordered by distance then doc_addr")
        previous = key
        seen.add(addr)
    return results


class StrictChunkResponse(http.client.HTTPResponse):
    # RFC chunk-size = 1*HEXDIG; extensions use token / quoted-string values.
    # Preserve stdlib line bound and int conversion, but never pass a signed
    # length to Python3.8's read1 chunk implementation (negative means read-all).
    _token = rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+"
    _quoted = rb'"(?:[\x09\x20\x21\x23-\x5b\x5d-\x7e\x80-\xff]|\\[\x09\x20-\x7e\x80-\xff])*"'
    _chunk_line = re.compile(rb"[0-9A-Fa-f]+(?:[ \t]*;[ \t]*" + _token +
                             rb"(?:[ \t]*=[ \t]*(?:" + _token + rb"|" + _quoted +
                             rb"))?)*\r\n")

    def _read_next_chunk_size(self):
        line = self.fp.readline(http.client._MAXLINE + 1)
        if len(line) > http.client._MAXLINE:
            raise http.client.LineTooLong("chunk size")
        if self._chunk_line.fullmatch(line) is None:
            self._close_conn()
            raise ValueError("invalid chunk size/extension")
        return int(line.split(b";", 1)[0], 16)


def http_exchange(config, body, headers, row, clock, on_send):
    """One attempt; terminal timestamp is after consumption or read failure.

    Body byte counts exclude HTTP/TLS headers/framing. Oversize bodies retain at
    most cap+1 bytes and are explicitly incomplete, never a successful result.
    Authorization is unsupported; no arbitrary request/response headers logged.
    Numeric addresses avoid DNS. Deadline starts before connect and applies to
    every application send/recv, including header/chunk parsing, after connect.
    """
    u = urlsplit(config.endpoint)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    class Connection(cls):
        response_class = StrictChunkResponse

        def connect(self):
            # Direct numeric sockaddr: no resolver call and no retry/address loop.
            address = ipaddress.ip_address(u.hostname)
            raw = socket.socket(socket.AF_INET6 if address.version == 6 else socket.AF_INET,
                                socket.SOCK_STREAM)
            self.sock = raw
            raw.settimeout(max(1e-9, (deadline - clock.now()) / 1e9))
            raw.connect((u.hostname, u.port or (443 if u.scheme == "https" else 80)))
            if u.scheme == "https":
                remaining = (deadline - clock.now()) / 1e9
                if remaining <= 0:
                    raise socket.timeout()
                raw.settimeout(remaining)
                self.sock = self._context.wrap_socket(raw, server_hostname=u.hostname)

        def send(self, data):
            self.sock.body_phase = data is body
            super().send(data)
    conn = Connection(u.hostname, u.port, timeout=config.timeout_s)
    response = None
    chunks = bytearray()
    try:
        row["connect_start_ns"] = clock.now()
        deadline = clock.now() + int(config.timeout_s * 1e9)
        conn.connect()
        row["connect_complete_ns"] = clock.now()
        transport = DeadlineSocket(conn.sock, deadline, row, clock, on_send)
        conn.sock = transport
        transport.remaining()
        conn.request("POST", u.path, body=body, headers=headers)
        row["request_transmitted_complete"] = True
        transport.remaining()
        response = conn.getresponse()
        row["http_status"] = response.status
        while len(chunks) <= config.max_response_bytes:
            transport.remaining()
            chunk = response.read1(min(65536, config.max_response_bytes + 1 - len(chunks)))
            if not chunk:
                row["response_complete"] = response.length in (None, 0)
                break
            chunks.extend(chunk)
        row["terminal_ns"] = clock.now()
        if len(chunks) > config.max_response_bytes:
            raise ValueError("response_oversize")
        if response.length not in (None, 0):
            raise ValueError("response_truncated")
        if response.status != 200:
            raise ValueError("http_status")
    finally:
        row["terminal_ns"] = row.get("terminal_ns") or clock.now()
        row["response_body_b64"] = base64.b64encode(chunks).decode("ascii")
        row["response_body_bytes"] = len(chunks)
        if response is not None:
            response.close()
        conn.close()


def run(queries, config, oracle=None, clock=None, exchange=http_exchange):
    prepared = prepare(queries, config, oracle)
    clock = clock or Clock()
    epoch = clock.now()
    rows, sent_ids, threads = [], [], []
    lock = threading.Lock()
    active_count, active_bytes = 0, 0
    headers = {"Content-Type": "application/json", "Connection": "close",
               "X-Tenant-ID": config.tenant, "X-Knowledge-Base": config.metadata.index,
               "X-Cortex-Index-Version": config.metadata.version}

    def worker(row, body, expected, reservation):
        nonlocal active_count, active_bytes
        try:
            def on_send(qid):
                with lock:
                    sent_ids.append(qid)
            exchange(config, body, dict(headers, **{"X-Request-ID": row["request_id"]}),
                     row, clock, on_send)
            results = validate_results(json.loads(base64.b64decode(row["response_body_b64"])),
                                       config.top_k)
            if expected is not None:
                for got, want in zip(results, expected):
                    if (got["doc_addr"] != want["doc_addr"] or
                            got["doc_length"] != want["doc_length"] or
                            not math.isclose(got["distance"], want["distance"],
                                             rel_tol=1e-5, abs_tol=1e-5)):
                        raise ValueError("oracle_mismatch")
            row["status"] = "ok"
            row["good_result"] = True
        except (socket.timeout, TimeoutError):
            row.update(status="timeout", error="timeout")
        except Exception as exc:
            # Exception text may contain credentials returned by a server.
            row.update(status="error", error=type(exc).__name__)
        finally:
            if row["send_ns"] is not None:
                row["terminal_ns"] = row["terminal_ns"] or clock.now()
                row["send_lag_ns"] = row["send_ns"] - row["planned_ns"]
            else:
                row.update(status="censored", terminal_ns=None,
                           scheduling_failure="worker_before_send")
            with lock:
                active_count -= 1
                active_bytes -= reservation

    for qid, offset, body, vh, expected in prepared:
        planned = epoch + offset
        clock.wait_until(planned)
        row = dict(request_id=qid, planned_ns=planned, send_ns=None, terminal_ns=None,
                   connect_start_ns=None, connect_complete_ns=None,
                   request_application_bytes_confirmed=0, request_write_bytes_exact=True,
                   request_transmitted_complete=False, planned_request_body_bytes=len(body),
                   send_lag_ns=None, status="censored", good_result=False,
                   scheduling_failure=None, namespace=config.namespace,
                   vector_sha256=vh, request_body_b64=base64.b64encode(body).decode("ascii"),
                   request_body_bytes=0, response_body_bytes=0,
                   response_body_b64="", response_complete=False, http_status=None)
        rows.append(row)
        reservation = len(body) + config.max_response_bytes + 1
        with lock:
            accepted = (active_count < config.max_inflight and
                        active_bytes + reservation <= config.max_inflight_bytes)
            if accepted:
                active_count += 1
                active_bytes += reservation
        if not accepted:
            row["scheduling_failure"] = "not_sent_capacity"
            continue
        threads = [t for t in threads if t.is_alive()]
        t = threading.Thread(target=worker, args=(row, body, expected, reservation))
        try:
            t.start()
            threads.append(t)
        except RuntimeError:
            with lock:
                active_count -= 1
                active_bytes -= reservation
            row["scheduling_failure"] = "not_sent_worker_start"
    join_workers(threads, config.timeout_s + 1.0)
    horizon = max(clock.now(), max(r["terminal_ns"] or r["planned_ns"] for r in rows)) + 1
    records = tuple(RequestRecord(r["request_id"], "client_monotonic", r["planned_ns"],
                                  r["send_ns"], r["terminal_ns"], r["status"],
                                  r["request_body_bytes"] if r["send_ns"] is not None else 0,
                                  r["response_body_bytes"], config.metadata) for r in rows)
    measurement = Measurement(records, tuple(sent_ids), "client_monotonic", horizon)
    failures = sum(r["scheduling_failure"] is not None for r in rows)
    late = sum(r["send_lag_ns"] is not None and
               r["send_lag_ns"] > config.send_lag_limit_ns for r in rows)
    valid = not failures and not late
    window = measurement.window(epoch, horizon)
    return dict(schema="semantic-open-loop-v2", config=asdict(config),
                arrival_semantics="client_first_write_attempt_after_connect_not_server_admission",
                byte_semantics="confirmed_socket_send_plaintext_lower_bounds_if_write_failed",
                deadline_scope="numeric_IP_connect_then_per_send_recv_absolute_deadline",
                resource_scope="concurrent_exchanges_not_strict_live_thread_count",
                request_id_scope="X-Request-ID_sent_no_RGW_ack_or_cross_service_correlation_claim",
                input_manifest_sha256=digest(queries), input_manifest=queries,
                oracle_kind="hash-bound-numerical-non-qrels" if oracle is not None else "none",
                oracle_sha256=digest(oracle) if oracle is not None else None,
                encoding_scope="preencoded_query_vectors_and_JSON_before_epoch",
                byte_scope="measurement bytes_in=confirmed body sends (lower bound on write failure); bytes_out=consumed entity bytes; excludes headers/framing/TLS",
                epoch_ns=epoch, requests=rows, arrived_request_ids=sent_ids,
                measurement_records=[asdict(r) for r in records], measurement=window,
                scheduled_count=len(rows), scheduling_failure_count=failures,
                send_lag_exceeded_count=late, load_generator_valid=valid,
                reportable_complete_p99_ms=(window["cohort_complete_p99_ms"] if valid else None),
                integration_status="pending_real_Ceph_device_calibration")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("endpoint", "tenant", "namespace", "backend-label", "profile", "index", "version",
                 "input", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--send-lag-limit-ns", required=True, type=int)
    parser.add_argument("--oracle", help="optional hash-bound numerical oracle JSON, not qrels")
    for name, default in (("max-inflight", 8), ("max-inflight-bytes", 8388608),
                          ("max-response-bytes", 1048576), ("max-queries", 10000),
                          ("max-input-bytes", 16777216), ("top-k", 10), ("nprobe", 32)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--metric", choices=("IP", "L2"), default="IP")
    args = vars(parser.parse_args(argv))
    input_path, output_path, oracle_path = args.pop("input"), args.pop("output"), args.pop("oracle")
    args["metadata"] = RunMetadata(*(args.pop(k) for k in ("backend_label", "profile", "index", "version")))
    try:
        config = Config(**args)
        def load(path):
            with open(path, "rb") as stream:
                data = stream.read(config.max_input_bytes + 1)
            if len(data) > config.max_input_bytes:
                raise ValueError("input file byte cap exceeded")
            return json.loads(data)
        result = run(load(input_path)["queries"], config,
                     load(oracle_path) if oracle_path else None)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, allow_nan=False, indent=2)
        return 0 if result["load_generator_valid"] and all(
            r["status"] == "ok" for r in result["requests"]) else 1
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error("invalid configuration/input: " + type(exc).__name__)


if __name__ == "__main__":
    raise SystemExit(main())
