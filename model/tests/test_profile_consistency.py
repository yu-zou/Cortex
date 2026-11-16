"""Regression checks for ideal transfer costs and live scheduler estimates."""
import math
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from strata_sim.sched.osd_timing import OSDHardwareProfile
from strata_sim.sched.query_queue import OsdView, QueryQueue, QueuedQuery
from strata_sim.des.engine import DesEngine
from strata_sim.sched.osd_timing import OSDTimingModel, add_osd_resources


class FakeFpga:
    clock_ns = 1000 / 180
    codebook_load_cycles = 12299

    def predict_cycles(self, n_docs, top_k, codebook_resident=True):
        return 500 + 200 + 2 * n_docs

    def scan_cycles_per_doc(self, n_docs):
        return 2


class ProfileConsistency(unittest.TestCase):
    def test_prediction_matches_serial_execution_and_does_not_mutate(self):
        engine = DesEngine()
        add_osd_resources(engine, 0)
        model = OSDTimingModel(0, engine, FakeFpga(),
                               OSDHardwareProfile.from_model("ideal"),
                               cluster_sizes={0: 9000, 1: 1000})
        before = model.cache_stats()
        predicted = model.estimate_service_ns([0, 1], 10, 'cb-user-1', 64)
        self.assertEqual(before, model.cache_stats())
        actual = model.simulate_query('a', [0, 1], 10,
                                      codebook_id='cb-user-1', start_ns=0,
                                      rerank_per_cluster=64)
        self.assertEqual(predicted, actual.end_ns - actual.start_ns)
        warm = model.estimate_service_ns([0, 1], 10, 'cb-user-1', 64)
        self.assertEqual(predicted - warm,
                         model.hardware.codebook_duration_ns()
                         + model.hardware.pq_load_ns(9000)
                         + model.hardware.pq_load_ns(1000))

    def test_bd_ideal_codebook_formula(self):
        hw = OSDHardwareProfile.from_model("ideal")
        self.assertEqual(hw.board_ddr_read_bytes_per_ns, 12.8)
        self.assertEqual(hw.codebook_duration_ns(), math.ceil(12288 * 1000 / 180))
        self.assertEqual(hw.board_codebook_load_ns, hw.codebook_duration_ns())

    def test_gen3_bandwidths(self):
        hw = OSDHardwareProfile.from_model("ideal")
        self.assertAlmostEqual(hw.ssd_to_ddr_bytes_per_ns, 7.876923076923077)
        self.assertAlmostEqual(hw.pcie_bytes_per_ns, 15.753846153846155)

    def test_ingress_clock_limits_even_fast_memory(self):
        from dataclasses import replace
        hw = replace(OSDHardwareProfile.from_model("ideal"),
                     board_ddr_read_bytes_per_ns=100, accelerator_clock_mhz=120)
        self.assertEqual(hw.codebook_duration_ns(), 102400)

    def test_load_charged_once(self):
        hw = OSDHardwareProfile.from_model("ideal")
        self.assertEqual(hw.pq_load_ns(1000),
                         10000 + math.ceil((64 + 128000) / hw.ssd_to_ddr_bytes_per_ns))

    def test_live_cost_changes_selection(self):
        def make():
            q = QueryQueue(strategy="cache")
            for name in ("a", "b"):
                q.push(QueuedQuery(name, "t", 0, deadline_ns=1000000))
            return q
        view = OsdView(service_estimates_ns={"a": 100, "b": 10})
        q = make()
        self.assertEqual(q.peek(0, view).query_id, "b")
        self.assertEqual(q.pending, 2)
        self.assertEqual(q.pop(0, view).query_id, "b")
        view = OsdView(service_estimates_ns={"a": 1, "b": 10})
        self.assertEqual(make().pop(0, view).query_id, "a")


if __name__ == "__main__":
    unittest.main()
