"""Local HTTP and deterministic accounting tests; no Ceph/backend claim."""
import base64
import faulthandler
import io
import json
import pathlib
import subprocess
import sys
import threading
import tempfile
import socket
import time
from unittest import mock
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import semantic_open_loop as client
from measurement_contract import RunMetadata


RESULTS = [{"doc_addr": 7, "doc_length": 16, "distance": -0.5}]
BODY = client.encoded({"results": RESULTS})


def queries(n=2):
    return [dict(request_id="q%d" % i, offset_ns=0, query_vector=[float(i)] * 768)
            for i in range(n)]


def config(**changes):
    values = dict(endpoint="http://127.0.0.1:1/v1/semantic_search", tenant="fixture-tenant",
                  namespace="fixture-ns", metadata=RunMetadata("localfixture", "fixture-v1",
                                                               "fixture-index", "abc123"),
                  send_lag_limit_ns=1_000_000_000, top_k=1,
                  timeout_s=2.0, max_response_bytes=4096)
    values.update(changes)
    return client.Config(**values)


class FakeClock:
    def __init__(self):
        self.value = 100

    def now(self):
        return self.value

    def wait_until(self, deadline):
        self.value = max(self.value, deadline)


def success(config, body, headers, row, clock, on_send):
    row.update(send_ns=clock.now() + 20, terminal_ns=clock.now() + 50,
               response_body_b64=base64.b64encode(BODY).decode(),
               response_body_bytes=len(BODY), response_complete=True, http_status=200)
    row.update(request_body_bytes=len(body), request_transmitted_complete=True)
    on_send(row["request_id"])


class LocalFixture:
    def __init__(self, mode="ok", reverse_arrivals=False):
        self.mode = mode
        self.seen = []
        self.second = threading.Event()
        self.release = threading.Event()
        self.arrival_second = threading.Event()
        self.stop = threading.Event()
        self.errors = []
        self.handlers = []
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            timeout = 2

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if reverse_arrivals:
                    if json.loads(body)["query_vector"][0] == 0:
                        if not fixture.arrival_second.wait(2):
                            raise AssertionError("second arrival not observed")
                with fixture.server.lock:
                    fixture.seen.append((self.path, dict(self.headers), body))
                    ordinal = len(fixture.seen)
                if reverse_arrivals and ordinal == 1:
                    fixture.arrival_second.set()
                payload = BODY
                if fixture.mode == "negative_chunk":
                    try:
                        self.connection.sendall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                                                b"-1\r\n" + b"x" * 16384)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if fixture.mode.startswith("drip_"):
                    if fixture.mode == "drip_headers":
                        prefix, drip = b"HTTP/1.1 200 OK\r\nX-Drip: ", b"x" * 100
                    elif fixture.mode == "drip_chunk":
                        prefix = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                        drip = b"0" * 100
                    else:
                        prefix = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n"
                        drip = b"x" * 100
                    try:
                        self.connection.sendall(prefix)
                        for byte in drip:
                            if fixture.release.wait(0.02):
                                break
                            self.connection.sendall(bytes([byte]))
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                if fixture.mode == "slowfirst":
                    if ordinal == 1:
                        fixture.second.wait(2)
                    else:
                        fixture.second.set()
                if fixture.mode == "timeout":
                    fixture.release.wait(2)
                if fixture.mode == "malformed":
                    payload = b'{"results":['
                if fixture.mode == "invalidresult":
                    payload = b'{"results":[{"doc_addr":true,"doc_length":1,"distance":0}]}'
                if fixture.mode == "oversize":
                    payload = b"x" * 5000
                self.send_response(503 if fixture.mode == "http_error" else 200)
                self.send_header("Content-Length", str(len(payload) +
                                                        (10 if fixture.mode == "truncated" else 0)))
                self.send_header("Authorization", "never-record-this-secret")
                self.end_headers()
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        class Server(ThreadingHTTPServer):
            # Explicit bounded joins below replace socketserver's unbounded join.
            block_on_close = False

            def process_request(self, request, address):
                thread = threading.Thread(target=self.process_request_thread,
                                          args=(request, address), daemon=False)
                fixture.handlers.append(thread)
                thread.start()

            def handle_error(self, request, address):
                fixture.errors.append(sys.exc_info()[1])

        self.server = Server(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.server.lock = threading.Lock()
        self.server.timeout = 0.01
        def serve():
            while not self.stop.is_set():
                self.server.handle_request()
        self.thread = threading.Thread(target=serve, daemon=False)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.release.set()
        self.second.set()
        self.arrival_second.set()
        self.stop.set()
        try:
            client.join_workers([self.thread], 3)
        finally:
            self.server.server_close()
            client.join_workers(self.handlers, 3)
        if self.errors:
            raise AssertionError("fixture handler failed: " + repr(self.errors))

    def config(self, **changes):
        return config(endpoint="http://127.0.0.1:%d/v1/semantic_search" %
                      self.server.server_address[1], **changes)


class OpenLoopTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = set(threading.enumerate())

    def tearDown(self):
        remaining = [t for t in threading.enumerate() if t not in self.previous_threads]
        client.join_workers(remaining, 3)

    def test_refused_connect_is_unsent_not_arrived(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            endpoint = "http://127.0.0.1:%d/v1/semantic_search" % reserved.getsockname()[1]
            result = client.run(queries(1), config(endpoint=endpoint))
        row = result["requests"][0]
        self.assertIsNone(row["send_ns"])
        self.assertEqual(result["arrived_request_ids"], [])
        self.assertEqual(result["measurement"]["arrival_count"], 0)
        self.assertEqual(result["scheduling_failure_count"], 1)

    def test_oracle_scope_mismatch_rejected_before_transport(self):
        qs = queries(1)
        scope = dict(tenant="fixture-tenant", namespace="fixture-ns", index="fixture-index",
                     version="abc123", metric="IP", nprobe=32, top_k=1)
        oracle = {"q0": dict(vector_sha256=client.digest(qs[0]["query_vector"]),
                             scope=scope, results=RESULTS)}
        for key in scope:
            bad = {"q0": dict(oracle["q0"], scope=dict(scope, **{key: "wrong"}))}
            with self.subTest(key=key), self.assertRaises(ValueError):
                client.prepare(qs, config(), bad)

    def test_local_two_posts_slow_first_does_not_gate_second(self):
        self.check_two_posts()

    def test_reversed_server_arrival_is_not_submission_order(self):
        self.check_two_posts(reverse_arrivals=True)

    def check_two_posts(self, reverse_arrivals=False):
        with LocalFixture("slowfirst", reverse_arrivals) as fixture:
            result = client.run(queries(), fixture.config())
            self.assertTrue(fixture.second.is_set())
            first, second = result["requests"]
            self.assertLess(second["send_ns"], first["terminal_ns"])
            self.assertEqual(len(fixture.seen), 2)
            self.assertEqual(result["measurement"]["arrival_count"], 2)
            self.assertEqual(result["measurement"]["conservation_residual"], 0)
            self.assertTrue(all(r["good_result"] for r in result["requests"]))
            # Server arrival order is independent of scheduler submission order.
            by_payload = {client.digest(json.loads(item[2])["query_vector"]): item
                          for item in fixture.seen}
            path, headers, body = by_payload[first["vector_sha256"]]
            self.assertEqual(first["request_body_bytes"], len(body))
            self.assertEqual(first["response_body_bytes"], len(BODY))
            self.assertEqual(base64.b64decode(first["response_body_b64"]), BODY)
            self.assertEqual(path, "/v1/semantic_search")
            self.assertEqual(headers["X-Tenant-ID"], "fixture-tenant")
            self.assertEqual(headers["X-Knowledge-Base"], "fixture-index")
            self.assertEqual(headers["X-Cortex-Index-Version"], "abc123")
            self.assertEqual(headers["X-Request-ID"], first["request_id"])
            self.assertTrue(first["request_transmitted_complete"])
            self.assertGreater(first["request_application_bytes_confirmed"], len(body))
            self.assertLessEqual(first["connect_complete_ns"], first["send_ns"])
            self.assertEqual(client.digest(json.loads(body)["query_vector"]),
                             first["vector_sha256"])
            self.assertNotIn("Authorization", json.dumps(result))
            self.assertNotIn("never-record-this-secret", json.dumps(result))

    def test_byte_cap_rejects_all_without_system_arrivals(self):
        def forbidden(*args):
            self.fail("unsent request reached transport")
        result = client.run(queries(), config(max_inflight_bytes=1),
                            clock=FakeClock(), exchange=forbidden)
        self.assertEqual(result["scheduled_count"], 2)
        self.assertEqual(result["scheduling_failure_count"], 2)
        self.assertEqual(result["arrived_request_ids"], [])
        self.assertEqual(result["measurement"]["arrival_count"], 0)
        self.assertEqual(result["measurement"]["cohort_censored_count"], 0)
        self.assertTrue(all(r["send_ns"] is None for r in result["requests"]))
        self.assertFalse(result["load_generator_valid"])
        self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_worker_count_cap_no_queue(self):
        entered, release = threading.Event(), threading.Event()
        class Clock(FakeClock):
            def wait_until(self, deadline):
                if deadline > 100:
                    if not entered.wait(2):
                        raise AssertionError("first worker did not enter")
                super().wait_until(deadline)
        def blocked(c, b, h, r, clock, on_send):
            entered.set()
            if not release.wait(2):
                raise AssertionError("not released")
            success(c, b, h, r, clock, on_send)
        # Releasing on the third schedule makes the second rejection deterministic.
        class ReleaseClock(Clock):
            def wait_until(self, deadline):
                super().wait_until(deadline)
                if deadline == 102:
                    release.set()
        qs = queries(3)
        qs[1]["offset_ns"], qs[2]["offset_ns"] = 1, 2
        result = client.run(qs, config(max_inflight=1), clock=ReleaseClock(), exchange=blocked)
        self.assertEqual(result["requests"][1]["scheduling_failure"], "not_sent_capacity")
        self.assertNotIn("q1", result["arrived_request_ids"])
        self.assertEqual(result["measurement"]["conservation_residual"], 0)

    def test_send_lag_and_scheduled_response_separate(self):
        result = client.run(queries(1), config(), clock=FakeClock(), exchange=success)
        window = result["measurement"]
        self.assertEqual(window["cohort_complete_p99_ms"], 30 / 1e6)
        self.assertEqual(window["send_delay_p99_ms"], 20 / 1e6)
        self.assertEqual(window["successful_terminal_minus_planned_p99_ms"], 50 / 1e6)

    def test_predeclared_lag_limit_invalidates_without_correction(self):
        result = client.run(queries(1), config(send_lag_limit_ns=19),
                            clock=FakeClock(), exchange=success)
        self.assertFalse(result["load_generator_valid"])
        self.assertEqual(result["send_lag_exceeded_count"], 1)
        self.assertIsNone(result["reportable_complete_p99_ms"])
        self.assertEqual(result["requests"][0]["send_lag_ns"], 20)

    def test_local_errors_timeout_and_no_retry(self):
        for mode in ("malformed", "invalidresult", "oversize", "timeout", "http_error"):
            with self.subTest(mode=mode), LocalFixture(mode) as fixture:
                result = client.run(queries(1), fixture.config(timeout_s=0.15))
                self.assertEqual(len(fixture.seen), 1)
                self.assertEqual(result["measurement"]["terminal_failure_count"], 1)
                self.assertEqual(result["measurement"]["arrival_count"], 1)
                self.assertIsNone(result["measurement"]["cohort_complete_p99_ms"])
                self.assertIsNone(result["reportable_complete_p99_ms"])
                row = result["requests"][0]
                self.assertFalse(row["good_result"])
                self.assertEqual(row["status"], "timeout" if mode == "timeout" else "error")
                if mode == "oversize":
                    self.assertEqual(row["response_body_bytes"], 4097)
                    self.assertFalse(row["response_complete"])

    def test_oracle_bound_to_id_and_vector_and_non_qrels(self):
        qs = queries(1)
        oracle = {"q0": dict(vector_sha256=client.digest(qs[0]["query_vector"]),
                             scope=client.oracle_scope(config()), results=RESULTS)}
        result = client.run(qs, config(), oracle, FakeClock(), success)
        self.assertEqual(result["oracle_kind"], "hash-bound-numerical-non-qrels")
        self.assertTrue(result["requests"][0]["good_result"])
        for bad in ({}, {"q0": dict(vector_sha256="wrong", results=RESULTS)}):
            with self.assertRaises(ValueError):
                client.run(qs, config(), bad, FakeClock(), success)

    def test_oracle_wrong_result_is_failure(self):
        qs = queries(1)
        oracle = {"q0": dict(vector_sha256=client.digest(qs[0]["query_vector"]),
                             scope=client.oracle_scope(config()),
                             results=[dict(RESULTS[0], doc_addr=9)])}
        result = client.run(qs, config(), oracle, FakeClock(), success)
        self.assertEqual(result["requests"][0]["status"], "error")

    def test_invalid_config(self):
        for changes in (dict(max_inflight=0), dict(timeout_s=float("nan")),
                        dict(send_lag_limit_ns=-1), dict(max_response_bytes=0),
                        dict(tenant="bad\r\nAuthorization: secret"),
                        dict(endpoint="http://user:secret@localhost/v1/semantic_search"),
                        dict(endpoint="http://localhost/wrong"), dict(nprobe=65)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config(**changes)

    def test_invalid_manifests(self):
        for qs in ([], queries(2) + queries(1),
                   [dict(queries(1)[0], offset_ns=-1)],
                   [dict(queries(1)[0], query_vector=[1.0])],
                   [dict(queries(1)[0], query_vector=[float("inf")] * 768)]):
            with self.assertRaises(ValueError):
                client.prepare(qs, config())
        with self.assertRaises(ValueError):
            client.prepare(queries(2), config(max_queries=1))
        with self.assertRaises(ValueError):
            client.prepare(queries(1), config(max_input_bytes=1))

    def test_response_validation(self):
        for rows in ([], RESULTS * 2, [dict(RESULTS[0], distance=float("nan"))],
                     [dict(RESULTS[0], doc_addr=-1)], [dict(RESULTS[0], doc_length=0)]):
            with self.assertRaises(ValueError):
                client.validate_results({"results": rows}, 1)

    def test_cli_help_and_invalid_config(self):
        script = pathlib.Path(client.__file__)
        help_result = subprocess.run([sys.executable, str(script), "--help"],
                                     capture_output=True, text=True, timeout=5)
        self.assertEqual(help_result.returncode, 0)
        args = [sys.executable, str(script)]
        for key, value in dict(endpoint="http://127.0.0.1:1/v1/semantic_search", tenant="x",
                               namespace="x", backend_label="localfixture", profile="fixture",
                               index="x", version="x", input="absent.json", output="absent-out.json",
                               send_lag_limit_ns="1", max_inflight="0").items():
            args.extend(["--" + key.replace("_", "-"), value])
        invalid = subprocess.run(args, capture_output=True, text=True, timeout=5)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid configuration", invalid.stderr)

    def test_bounded_join_rejects_stuck_worker_without_success(self):
        release = threading.Event()
        worker = threading.Thread(target=release.wait, daemon=False)
        worker.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "workers still alive"):
                client.join_workers([worker], 0.01)
            self.assertTrue(worker.is_alive())
        finally:
            release.set()
            client.join_workers([worker], 1)

    def test_drip_headers_chunk_framing_and_body_deadline(self):
        for mode in ("drip_headers", "drip_chunk", "drip_body"):
            with self.subTest(mode=mode), LocalFixture(mode) as fixture:
                start = time.monotonic()
                result = client.run(queries(1), fixture.config(timeout_s=0.12))
                self.assertLess(time.monotonic() - start, 0.8)
                self.assertEqual(result["requests"][0]["status"], "timeout")
                self.assertEqual(result["measurement"]["arrival_count"], 1)
                self.assertEqual(len(fixture.seen), 1)
                self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_slow_drips_bounded_in_subprocess(self):
        done = subprocess.run([sys.executable, __file__,
                               "OpenLoopTests.test_drip_headers_chunk_framing_and_body_deadline", "-v"],
                              capture_output=True, text=True, timeout=4)
        self.assertEqual(done.returncode, 0, done.stderr)

    def test_exact_response_cap_and_truncated_content_length(self):
        with LocalFixture() as fixture:
            result = client.run(queries(1), fixture.config(max_response_bytes=len(BODY)))
            self.assertEqual(result["requests"][0]["status"], "ok")
            self.assertTrue(result["requests"][0]["response_complete"])
        with LocalFixture("truncated") as fixture:
            result = client.run(queries(1), fixture.config())
            self.assertEqual(result["requests"][0]["status"], "error")
            self.assertFalse(result["requests"][0]["response_complete"])
            self.assertIsNone(result["reportable_complete_p99_ms"])

    def chunk_response(self, wire, cap=32):
        class FakeSocket:
            def __init__(self):
                self.stream = io.BytesIO(wire)
            def settimeout(self, value):
                pass
            def connect(self, address):
                pass
            def send(self, data):
                return len(data)
            def recv_into(self, buffer):
                return self.stream.readinto(buffer)
            def close(self):
                self.stream.close()
        sock = FakeSocket()
        with mock.patch.object(socket, "socket", return_value=sock):
            result = client.run(queries(1), config(max_response_bytes=cap))
        self.assertTrue(sock.stream.closed)
        return result

    def test_negative_chunk_rejected_before_body_read_actual_transport(self):
        wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n-1\r\n" + b"x" * 16384
        result = self.chunk_response(wire)
        row = result["requests"][0]
        self.assertLessEqual(row["response_body_bytes"], 33)
        self.assertEqual(row["response_body_bytes"], 0)
        self.assertEqual(row["status"], "error")
        self.assertFalse(row["response_complete"])
        self.assertIsNone(result["reportable_complete_p99_ms"])

        partial = self.chunk_response(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                                      b"3\r\nabc\r\n-1\r\n" + b"x" * 16384)
        self.assertEqual(partial["requests"][0]["response_body_bytes"], 3)
        self.assertEqual(base64.b64decode(partial["requests"][0]["response_body_b64"]), b"abc")
        self.assertEqual(partial["measurement_records"][0]["bytes_out"], 3)
        self.assertEqual(partial["requests"][0]["status"], "error")
        self.assertFalse(partial["requests"][0]["response_complete"])

    def test_negative_chunk_loopback(self):
        with LocalFixture("negative_chunk") as fixture:
            result = client.run(queries(1), fixture.config(max_response_bytes=32))
            row = result["requests"][0]
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["response_body_bytes"], 0)
            self.assertFalse(row["response_complete"])
            self.assertEqual(len(fixture.seen), 1)
            self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_chunk_hex_extensions_zero_and_huge_lengths(self):
        prefix = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
        size = ("%X" % len(BODY)).encode()
        result = self.chunk_response(prefix + size + b';name=token;quoted="a\\\"b"\r\n' +
                                     BODY + b"\r\n0;end=yes\r\n\r\n", cap=len(BODY))
        self.assertEqual(result["requests"][0]["status"], "ok")
        self.assertTrue(result["requests"][0]["response_complete"])
        for length in (b"+1", b"0x10", b"1_0", b"g", b"", b"1;bad=", b"1;q=\"unterminated",
                       b"f" * 65537, b"f" * 1024):
            with self.subTest(length_prefix=length[:20]):
                result = self.chunk_response(prefix + length + b"\r\n" + b"x" * 16384)
                row = result["requests"][0]
                self.assertEqual(row["status"], "error")
                self.assertLessEqual(row["response_body_bytes"], 33)
                self.assertFalse(row["response_complete"])
                self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_partial_write_counts_confirmed_lower_bound_not_planned_body(self):
        class FailingSocket:
            calls = 0
            def settimeout(self, value):
                pass
            def send(self, data):
                self.calls += 1
                if self.calls == 1:
                    return 3
                raise BrokenPipeError()
        def partial(c, body, headers, row, clock, on_send):
            transport = client.DeadlineSocket(FailingSocket(), clock.now() + 100, row, clock, on_send)
            transport.body_phase = True
            transport.sendall(body)
        result = client.run(queries(1), config(), clock=FakeClock(), exchange=partial)
        row = result["requests"][0]
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["request_body_bytes"], 3)
        self.assertFalse(row["request_write_bytes_exact"])
        self.assertFalse(row["request_transmitted_complete"])
        self.assertEqual(result["measurement_records"][0]["bytes_in"], 3)
        self.assertEqual(result["measurement"]["arrival_count"], 1)
        self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_thread_start_failure_releases_cap(self):
        original = threading.Thread.start
        calls = []
        def start(thread):
            calls.append(thread)
            if len(calls) == 1:
                raise RuntimeError("injected start failure")
            return original(thread)
        with mock.patch.object(threading.Thread, "start", start):
            result = client.run(queries(), config(max_inflight=1), clock=FakeClock(), exchange=success)
        self.assertEqual(result["requests"][0]["scheduling_failure"], "not_sent_worker_start")
        self.assertIsNone(result["requests"][0]["send_ns"])
        self.assertTrue(result["requests"][1]["good_result"])
        self.assertFalse(result["load_generator_valid"])

    def test_numeric_endpoint_required_before_network(self):
        with self.assertRaises(ValueError):
            config(endpoint="http://localhost/v1/semantic_search")

    def test_numeric_transport_never_calls_resolver(self):
        with LocalFixture() as fixture:
            with mock.patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")):
                result = client.run(queries(1), fixture.config())
            self.assertTrue(result["requests"][0]["good_result"])

    def test_delayed_worker_start_invalidates_lag(self):
        original = threading.Thread.start
        clock = FakeClock()
        def delayed(thread):
            clock.value += 100
            return original(thread)
        with mock.patch.object(threading.Thread, "start", delayed):
            result = client.run(queries(1), config(send_lag_limit_ns=50),
                                clock=clock, exchange=success)
        self.assertEqual(result["requests"][0]["send_lag_ns"], 120)
        self.assertFalse(result["load_generator_valid"])
        self.assertIsNone(result["reportable_complete_p99_ms"])

    def test_cli_local_http_two_requests(self):
        with LocalFixture() as fixture, tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "input.json"
            output = pathlib.Path(directory) / "output.json"
            source.write_text(json.dumps({"queries": queries()}))
            args = [sys.executable, client.__file__]
            for key, value in dict(endpoint=fixture.config().endpoint, tenant="t", namespace="n",
                                   backend_label="localfixture", profile="fixture", index="i",
                                   version="v", input=str(source), output=str(output), top_k="1",
                                   send_lag_limit_ns="1000000000").items():
                args.extend(["--" + key.replace("_", "-"), value])
            done = subprocess.run(args, capture_output=True, text=True, timeout=5)
            self.assertEqual(done.returncode, 0, done.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["measurement"]["arrival_count"], 2)
            self.assertEqual(len(fixture.seen), 2)
            self.assertEqual(result["config"]["metadata"]["backend_label"], "localfixture")


if __name__ == "__main__":
    faulthandler.dump_traceback_later(10, repeat=True)
    try:
        unittest.main()
    finally:
        faulthandler.cancel_dump_traceback_later()
