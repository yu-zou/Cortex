"""Offline request fixtures exercising the strict measurement contract."""
import dataclasses
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from measurement_contract import CostRecord, Measurement, RequestRecord, RunMetadata


META = RunMetadata(backend_label="cpu-reference", profile="measured-v1",
                   index="scifact/ivfflat", version="sha256:pinned")


def request(qid, planned, terminal, status="ok", **changes):
    values = dict(request_id=qid, time_domain="client_monotonic",
                  planned_ns=planned, send_ns=planned, terminal_ns=terminal,
                  status=status, bytes_in=128, bytes_out=64, metadata=META)
    values.update(changes)
    return RequestRecord(**values)


def fixture():
    return (request("prior", 0, 10), request("at-start", 10, 20),
            request("error", 11, 15, "error"),
            request("timeout", 12, 18, "timeout"),
            request("censored", 19, None, "censored"),
            request("at-end", 20, 25))


def measurement(rows=None, **changes):
    rows = fixture() if rows is None else tuple(rows)
    values = dict(requests=rows, arrived_request_ids=tuple(r.request_id for r in rows
                                                        if r.send_ns is not None),
                  time_domain="client_monotonic", observation_horizon_ns=30)
    values.update(changes)
    return Measurement(**values)


class MeasurementContractTests(unittest.TestCase):
    def test_actual_e2e_excludes_send_delay(self):
        row = request("delayed", 0, 100_000_000, send_ns=90_000_000)
        result = measurement([row], observation_horizon_ns=100_000_000).window(0, 100_000_000)
        self.assertEqual(result["successful_latency_p99_ms"], 10)
        self.assertEqual(result["cohort_complete_p99_ms"], 10)
        self.assertEqual(result["send_delay_p50_ms"], 90)
        self.assertEqual(result["send_delay_p99_ms"], 90)
        self.assertEqual(result["successful_terminal_minus_planned_p99_ms"], 100)

    def test_later_send_and_never_send_are_not_system_arrivals(self):
        rows = (request("later", 0, 100_000_000, send_ns=90_000_000),
                request("never", 0, None, "censored", send_ns=None))
        result = measurement(rows, arrived_request_ids=("later",),
                             observation_horizon_ns=100_000_000).window(0, 50_000_000)
        self.assertEqual(result["arrival_count"], 0)
        self.assertEqual(result["cohort_count"], 0)
        self.assertEqual((result["outstanding_start"], result["outstanding_end"]), (0, 0))
        self.assertTrue(all(count == 0 for _, count in result["backlog"]))
        self.assertIsNone(result["send_delay_p99_ms"])
        self.assertEqual(result["conservation_residual"], 0)

    def test_mixed_cpu_emu_same_clock_rejected(self):
        rows = (request("cpu", 0, 1), request("emu", 0, 1,
                metadata=dataclasses.replace(META, backend_label="emu")))
        with self.assertRaises(ValueError):
            measurement(rows)

    def test_metadata_output_and_all_mixed_identities_rejected(self):
        result = measurement().window(10, 20)
        self.assertEqual(result["metadata"], dataclasses.asdict(META))
        result["metadata"]["profile"] = "changed"
        self.assertEqual(measurement().window(10, 20)["metadata"]["profile"], META.profile)
        self.assertIsNone(measurement([]).window(0, 30)["metadata"])
        for field in ("profile", "index", "version"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                measurement([request("a", 0, 1), request("b", 0, 1,
                            metadata=dataclasses.replace(META, **{field: "other"}))])

    def test_unsent_censor_excluded_from_manifest_and_completed_cohort(self):
        rows = (request("ok", 0, 1), request("unsent", 0, None, "censored", send_ns=None))
        result = measurement(rows).window(0, 30)
        self.assertEqual(result["arrival_count"], 1)
        self.assertEqual(result["cohort_complete_p99_ms"], 1 / 1e6)
        self.assertEqual(result["cohort_censored_count"], 0)
        self.assertEqual(result["outstanding_end"], 0)
        with self.assertRaises(ValueError):
            measurement(rows, arrived_request_ids=("ok", "unsent"))

    def test_half_open_conservation_and_full_series(self):
        m = measurement().window(10, 20, boundaries_ns=(0, 10, 15, 18, 19, 20, 25, 30))
        self.assertEqual(m["arrival_count"], 4)
        self.assertEqual(m["ok_completion_count"], 1)
        self.assertEqual(m["terminal_failure_count"], 2)
        self.assertEqual(m["error_count"], 1)
        self.assertEqual(m["timeout_count"], 1)
        self.assertEqual(m["completion_qps"], 1e8)
        self.assertEqual((m["outstanding_start"], m["outstanding_end"]), (1, 2))
        self.assertEqual(m["outstanding_growth"], 4 - 1 - 2)
        self.assertEqual(m["conservation_residual"], 0)
        self.assertEqual(m["backlog"], ((0, 0), (10, 1), (15, 3), (18, 2),
                                       (19, 1), (20, 2), (25, 2), (30, 1)))

    def test_failure_and_censor_never_disappear_from_p99(self):
        m = measurement().window(10, 20)
        self.assertEqual(m["cohort_count"], 4)
        self.assertEqual(m["cohort_ok_count"], 1)
        self.assertEqual(m["cohort_failure_count"], 2)
        self.assertEqual(m["cohort_censored_count"], 1)
        self.assertEqual(m["successful_latency_p99_ms"], 10 / 1e6)
        self.assertIsNone(m["cohort_complete_p50_ms"])
        self.assertIsNone(m["cohort_complete_p99_ms"])
        self.assertIsNone(measurement().requests[4].terminal_ns)

    def test_complete_cohort_linear_percentiles_follow_through_drain(self):
        rows = (request("a", 10, 20), request("b", 19, 30))
        m = measurement(rows).window(10, 20)
        self.assertEqual(m["ok_completion_count"], 0)
        self.assertAlmostEqual(m["cohort_complete_p50_ms"], 10.5 / 1e6)
        self.assertAlmostEqual(m["cohort_complete_p99_ms"], 10.99 / 1e6)
        self.assertIsNone(measurement(rows).window(0, 10)["cohort_complete_p99_ms"])

    def test_frozen_records_and_snapshot(self):
        row = request("a", 0, 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            row.status = "error"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            META.profile = "other"
        rows = [row]
        m = measurement(rows, requests=rows, arrived_request_ids=["a"])
        rows.clear()
        self.assertEqual(m.requests, (row,))
        self.assertEqual(m.arrived_request_ids, ("a",))

    def test_invalid_requests(self):
        cases = ({"planned_ns": -1}, {"send_ns": 9}, {"terminal_ns": 9},
                 {"terminal_ns": None}, {"send_ns": None},
                 {"status": "unknown"}, {"status": "censored"},
                 {"bytes_in": -1}, {"bytes_out": True}, {"planned_ns": 1.5},
                 {"terminal_ns": float("nan")}, {"request_id": " "},
                 {"time_domain": "ideal"}, {"metadata": {}})
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                request("a", 10, 20, **change)

    def test_duplicates_missing_arrivals_and_invalid_horizon(self):
        row = request("a", 10, 20)
        cases = (([row, row], {}), ([row], {"arrived_request_ids": ("a", "lost")}),
                 ([row], {"arrived_request_ids": ()}),
                 ([row], {"arrived_request_ids": ("a", "a")}),
                 ([row], {"observation_horizon_ns": 19}),
                 ([row], {"observation_horizon_ns": -1}),
                 ([request("c", 31, None, "censored")], {}))
        for rows, changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                measurement(rows, **changes)

    def test_clock_domains_explicit_not_model_tag_inferred(self):
        modeled = request("des", 10, 20, time_domain="des_logical",
                          metadata=dataclasses.replace(META, profile="ideal"))
        with self.assertRaises(ValueError):
            measurement([request("client", 0, 1), modeled])
        with self.assertRaises(ValueError):
            measurement([modeled])
        with self.assertRaises(ValueError):
            measurement([modeled], time_domain="ideal")
        m = measurement([modeled], time_domain="des_logical").window(0, 30)
        self.assertEqual(m["time_domain"], "des_logical")
        self.assertEqual(m["completion_qps"], 1e9 / 30)
        # A profile/model label is metadata, never evidence of a measured clock.
        self.assertEqual(modeled.metadata.profile, "ideal")

    def test_window_and_boundary_validation(self):
        for start, end in ((10, 10), (20, 10), (-1, 10), (0, 31), (True, 20)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                measurement().window(start, end)
        for boundaries in ((-1,), (31,), (20, 10), (10, 10), (1.5,)):
            with self.subTest(boundaries=boundaries), self.assertRaises(ValueError):
                measurement().window(10, 20, boundaries_ns=boundaries)

    def test_empty_and_zero_duration_requests(self):
        self.assertEqual(measurement([]).window(0, 30)["completion_qps"], 0)
        m = measurement([request("instant", 10, 10)]).window(10, 20)
        self.assertEqual(m["conservation_residual"], 0)
        self.assertEqual(m["cohort_complete_p99_ms"], 0)

    def test_every_integer_window_conserves_including_ties(self):
        rows = fixture() + (request("instant", 15, 15),
                            request("delayed-send", 10, 29, send_ns=28),
                            request("unsent", 0, None, "censored", send_ns=None))
        sample = measurement(rows)
        for start in range(30):
            for end in range(start + 1, 31):
                with self.subTest(start=start, end=end):
                    self.assertEqual(sample.window(start, end)["conservation_residual"], 0)

    def test_each_non_success_blocks_complete_percentiles(self):
        for status in ("error", "timeout", "censored"):
            row = request("bad", 10, None if status == "censored" else 20, status)
            result = measurement([request("good", 10, 11), row]).window(10, 20)
            self.assertIsNone(result["cohort_complete_p99_ms"])
            self.assertEqual(result["successful_latency_p99_ms"], 1 / 1e6)

    def test_send_and_terminal_must_be_observed_and_ordered(self):
        with self.assertRaises(ValueError):
            request("a", 10, 20, send_ns=21)
        with self.assertRaises(ValueError):
            measurement([request("a", 10, None, "censored", send_ns=31)])
        with self.assertRaises(ValueError):
            request("a", 10, None, "censored", send_ns=True)

    def test_cost_and_metadata_validation(self):
        cost = CostRecord(build=1.25, run=0, storage=2, currency="USD")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cost.run = 5
        for field in ("build", "run", "storage"):
            for value in (-1, float("nan"), float("inf"), True, "1"):
                values = dict(build=0, run=0, storage=0, currency="USD")
                values[field] = value
                with self.subTest(values=values), self.assertRaises(ValueError):
                    CostRecord(**values)
        for currency in ("", "usd", " USD", None):
            with self.assertRaises(ValueError):
                CostRecord(0, 0, 0, currency)
        for field in ("backend_label", "profile", "index", "version"):
            with self.assertRaises(ValueError):
                dataclasses.replace(META, **{field: ""})


if __name__ == "__main__":
    unittest.main()
